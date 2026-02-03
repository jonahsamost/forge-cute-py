import argparse
import time
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cute_viz import render_tv_layout_svg, display_tv_layout


custom_cache = {}


@cute.kernel
def _reduce_sum(input: cute.Tensor, output: cute.Tensor, tv_layout, num_warps: int):
    smem_alloc = cutlass.utils.SmemAllocator()
    shmem = smem_alloc.allocate_tensor(cute.Float32, cute.make_layout((32,)))

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    acc = cute.Float32(0.0)

    _, mode1 = input.shape
    ntiles = mode1[1]

    for tile_idx in range(ntiles):
        blk_coord = ((0, None), (bidx, tile_idx)) # all values in this [bidx, tile_idx] tile 
        cta_tile = input[blk_coord]
        thr_frag = cute.composition(cta_tile, tv_layout)
        thr_src  = thr_frag[(tidx, None)]
        acc += thr_src[0]

    acc = cute.arch.warp_reduction_sum(acc)
    if lane == 0:
        shmem[warp] = acc
    cute.arch.sync_threads()

    if warp == 0:
        acc2 = shmem[lane] if lane < num_warps else 0.0
        acc2 = cute.arch.warp_reduction_sum(acc2)
        if lane == 0:
            output[bidx] = acc2


@cute.jit
def reduce_sum(x, output):
    num_warps = 16
    threads_per_block = 512
    M, N = x.shape

    tiler_mn = (1, 2048)
    gX = cute.zipped_divide(x, tiler_mn)

    thr_layout = cute.make_layout((threads_per_block,), stride=(1,))
    val_layout = cute.make_layout((4,), stride=(1,))
    _, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    
    _reduce_sum(gX, output, tv_layout, num_warps).launch(
        grid=(M, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def _invoke_reduce_sum(gInput, dim=-1):
    a_ = from_dlpack(gInput)

    sz = gInput.size(0) if dim == -1 else gInput.size(1)
    gOutput = torch.empty((sz,), dtype=gInput.dtype, device=gInput.device)
    b_ = from_dlpack(gOutput)

    kernel_fn = cute.compile(reduce_sum, a_, b_,)
    kernel_fn(a_, b_)
    return gOutput


def simple_launch(dim=-1):
    M, N = 1234, 4321

    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    output = _invoke_reduce_sum(a, dim=-1)

    torch_sum = a.sum(dim=dim)
    assert torch.allclose(torch_sum, output, rtol=1e-4, atol=1e-4)
    print("Success!")


@cute.jit
def print_layout(mA):
    threads_per_block = 32
    thr_layout = cute.make_layout((threads_per_block,), stride=(1,))
    val_layout = cute.make_layout((1,), stride=(1,))
    tiler_mn, layout_tv = cute.make_layout_tv(thr_layout, val_layout)
    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print(f"layout: {layout_tv}")
    print(f"tiler: {tiler_mn}")
    print(f"gA: {gA}")
    tiler_2d = (1, tiler_mn[0])
    gX = cute.zipped_divide(mA, tiler_2d)
    print(f"gx: {gX}")
    size = cute.size(gX, mode=[1])
    print(f"size: {size}")
    # render_tv_layout_svg(layout_tv, tiler_2d, "tv_layout.svg")

M, N = 64, 64
a = torch.randn(M, N, device="cuda", dtype=torch.float32)
print_layout(a)
