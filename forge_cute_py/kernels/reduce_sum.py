import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUTLASS_CUDA_ARCH'] = '86'

import math
import torch
from cutlass.cute.runtime import from_dlpack

import cutlass
import cutlass.cute as cute

from cutlass import dsl_user_op
from cutlass.cute.arch import nvvm
from cutlass._mlir.dialects.nvvm import AtomicOpKind, MemOrderKind, MemScopeKind
from cutlass.base_dsl.typing import T


@cute.kernel
def reduce_sum_kernel_one(input: cute.Tensor, output: cute.Tensor, N:cute.Int32, num_warps: int, max_iters: int):
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((32,))
    shmem = smem_alloc.allocate_tensor(cute.Float32, smem_layout)
    
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    idx = bdim * bidx + tidx

    acc = cute.Float32(0)
    for i in range(max_iters):
        idx = idx + i * bdim
        if idx < N:
            acc = acc + input[idx]
    acc = cute.arch.warp_reduction_sum(acc) 
    if lane_idx == 0:
        shmem[warp_idx] = acc
    cute.arch.sync_threads()
    if warp_idx == 0:
        acc = shmem[lane_idx] if lane_idx < num_warps else 0.0
        acc = cute.arch.warp_reduction_sum(acc) 
        if lane_idx == 0:
            output[bidx] = acc


@dsl_user_op
def atomicAddF32(dst_ptr: cute.Pointer, val: cute.Float32, loc=None, ip=None) -> cute.Float32:
    return nvvm.atomicrmw(
        T.f32(),
        AtomicOpKind.FADD,
        dst_ptr.llvm_ptr,
        val.ir_value(loc=loc, ip=ip),
        mem_order=MemOrderKind.RELAXED,
        syncscope=MemScopeKind.SYS,
        loc=loc,
        ip=ip,
    )

@cute.kernel
def reduce_sum_kernel_atomic(input: cute.Tensor, output: cute.Tensor, N:cute.Int32, num_warps: int, coarsen: int):
    smem_alloc = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout((32,))
    shmem = smem_alloc.allocate_tensor(cute.Float32, smem_layout)
    
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    base_idx = coarsen * bdim * bidx + tidx

    acc = cute.Float32(0)
    for i in range(coarsen):
        idx = base_idx + i * bdim
        if idx < N:
            acc = acc + input[idx]
    acc = cute.arch.warp_reduction_sum(acc) 
    if lane_idx == 0:
        shmem[warp_idx] = acc
    cute.arch.sync_threads()
    if warp_idx == 0:
        acc = shmem[lane_idx] if lane_idx < num_warps else 0.0
        acc = cute.arch.warp_reduction_sum(acc) 
        if lane_idx == 0:
            atomicAddF32(output.iterator, acc)


@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):
    # if N <= 128:
    #     reduce_sum_kernel_one( input, output, N, 4, 1
    #     ).launch( grid=(1, 1, 1), block=(128, 1, 1))
    # elif N <= 10240:
    #     reduce_sum_kernel_one( input, output, N, 32, 10
    #     ).launch( grid=(1, 1, 1), block=(1024, 1, 1))
    # else:
    num_warps = 8
    threads_per_block = 32 * num_warps
    coarsen = 8
    blocks = cute.ceil_div(N, threads_per_block * coarsen)
    reduce_sum_kernel_atomic(input, output, N, num_warps, coarsen
    ).launch( grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))


N = 100000
a = torch.randn((N,), device="cuda", dtype=torch.float32)
b = torch.zeros((1,), device='cuda', dtype=torch.float32)
vadd_compiled = cute.compile(solve, from_dlpack(a), from_dlpack(b), N)
vadd_compiled(from_dlpack(a), from_dlpack(b), N)
print(f'b=={b}')