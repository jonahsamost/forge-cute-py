import torch
import inspect
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack
from cutlass import const_expr
from cute_viz import render_layout_svg, display_layout


class ReduceSum:
    def __init__(self, shape, dtype, dim=-1):
        self.shape = shape
        self.dtype = dtype
        self.num_warps = 4
        self.warp_size = 32
        self.threads_per_block = self.num_warps * self.warp_size
        self.NEG_INF = Float32(float('-inf'))
        self.dim = dim

        strides = [1]
        mult = 1
        for d in self.shape[1:][::-1]:
            mult *= d
            strides.insert(0, mult)
        self.strides = strides

        self.before_shape = shape[:dim]
        self.before_stride = strides[:dim]

        self.after_shape = shape[dim + 1:]
        self.after_stride = strides[dim + 1:]
        
        self.reduce_size = shape[dim]
        self.reduce_stride = strides[dim]

        self.output_shape = (*self.before_shape, *self.after_shape)
        self.output_stride = (*self.before_stride, *self.after_stride)

    @cute.jit
    def __call__(self, gInput: cute.Tensor, gOutput: cute.Tensor, stream: cuda.CUstream = None):
        blocks_over_reduce_dim = cute.ceil_div(self.reduce_size, self.warp_size)

        val_layout = cute.make_layout((1, 1))
        thr_layout = cute.make_layout(
            (self.output_shape, self.reduce_size),
            stride=(self.output_stride, self.reduce_stride)
        ) 
        tiled_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
        gX = cute.zipped_divide(gInput, tiled_mn)
        print(f"gx: {gX}")

        # tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
        # # render_layout_svg(thr_layout, 'output.svg')

        # blocks = cute.ceil_div(self.blocks, self.num_warps)
        # self.kernel(
        #     gInput, gOutput, tiler_mn, tiled_copy
        # ).launch(
        #     grid=(blocks, 1, 1),
        #     block=(self.threads_per_block, 1, 1),
        #     stream=stream
        # )
    
    @cute.kernel
    def kernel(self, gInput: cute.Tensor, gOutput: cute.Tensor, tiler_mn: cute.Shape, tiled_copy: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // self.warp_size
        lane_idx = tidx % self.warp_size

        gX = cute.local_tile(gInput, tiler_mn, (bidx, 0) if self.dim == -1 else (0, bidx))
        tidxSlice = tiled_copy.get_slice(tidx)  
        tidxIndices = tidxSlice.partition_S(gX)
        tidxRegs = cute.make_rmem_tensor_like(tidxIndices)
        cute.autovec_copy(tidxIndices, tidxRegs)
        tidxValues = tidxRegs.load()
        tidLocalSum = tidxValues.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
        rowSum = cute.arch.warp_reduction_sum(tidLocalSum)
        if lane_idx == 0:
            gOutput[bidx * self.num_warps + warp_idx] = rowSum


def benchmark(dim=1):
    print(f"Testing dim: {dim}")
    import time

    a, b, c = 64, 128, 256
    dtype = torch.float32
    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }
    cute_dtype = dtype_map[dtype]
    
    x = torch.randn(a, b, c, device='cuda', dtype=dtype)
    output_shape = list(x.shape)
    output_shape.pop(dim)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    #### 

    input_shape = [cute.sym_int() for _ in range(len(x.shape))]
    input_shape[dim] = x.shape[dim]
    input_cute = cute.runtime.make_fake_compact_tensor(
        cute_dtype,
        tuple(input_shape),
        stride_order=tuple(range(len(x.shape)))[::-1]
    )

    output_shape = [cute.sym_int() for _ in range(len(x.shape))]
    output_shape.pop(dim)
    output_cute = cute.runtime.make_fake_compact_tensor(
        cute_dtype,
        tuple(output_shape),
        stride_order=tuple(range(len(x.shape) - 1))[::-1]
    ) 
    softmax = ReduceSum(x.shape, dtype_map[dtype], dim=dim)
    fn = cute.compile(
        softmax, input_cute, output_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi", 
    )
    fn(x, output)
    
    print("Correctness check:")
    expected = x.sum(dim=dim)
    is_close = torch.allclose(output, expected, rtol=1e-3, atol=1e-3)
    print(f"  dim={dim}: {'✓ PASS' if is_close else '✗ FAIL'}")
    if not is_close:
        max_diff = (output - expected).abs().max().item()
        print(f"         max diff: {max_diff}")
    
    print("\nBenchmarks:")
    
    # Warmup
    for _ in range(10):
        fn(x, output)
    torch.cuda.synchronize()
    
    # Benchmark our softmax
    start = time.perf_counter()
    for _ in range(100):
        fn(x, output)
    torch.cuda.synchronize()
    print(f"  reduce sum cute dim=-1: {(time.perf_counter() - start) * 10:.3f} ms")
    
    # Compare to PyTorch
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.nn.functional.softmax(x, dim=-1)
    torch.cuda.synchronize()
    print(f"  torch reduce sum dim=-1:  {(time.perf_counter() - start) * 10:.3f} ms")

benchmark(dim=1)