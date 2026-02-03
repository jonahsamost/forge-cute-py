import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import BFloat16, Float16, Float32
from cutlass.cute.runtime import from_dlpack
from cutlass import const_expr
from cute_viz import render_tiled_copy_svg
import inspect


def _get_tiled_copy(vecsize, dtype, N):
    """
    Adapted from quack's tiles_copy_2d()
    Reference: https://github.com/Dao-AILab/quack/blob/2e62faaeb6271a780a1360e6c96a003492e47eed/quack/copy_utils.py#L98
    """
    threads_per_row = 32
    num_threads = 128
    # thread groups (of size 32 each) needed to cover N // vecsize
    num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)

    # each tile covers [4, ~N]
    tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)

    num_copy_bits = vecsize * dtype.width
    copy_op = cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, vecsize))
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    render_tiled_copy_svg(tiled_copy, tiler_mn, "my_copy_layout.svg")
    print(f"tild copy: {tiled_copy}")
    return tiler_mn, tiled_copy, threads_per_row


@cute.jit
def test_jit(mX, mY):
    dtype=Float32
    vecsize = 128 // dtype.width
    tiler_mn, tiled_copy, threads_per_row = _get_tiled_copy(vecsize=vecsize, dtype=dtype, N=mX.shape[1])
    num_threads = tiled_copy.size

    kernel(mX, mY, tiler_mn, tiled_copy, threads_per_row).launch(
        grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
        block=[num_threads, 1, 1]
    )

@cute.kernel
def kernel(
    mX: cute.Tensor,
    mO: cute.Tensor,
    tiler_mn: cute.Shape,
    tiled_copy: cute.TiledCopy,
    threads_per_row: cutlass.Constexpr[int],
):
    # tv_layout = (thread_layout, value_layout) = ((threads_per_row, num_rows), vec_size)
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    gX = cute.local_tile(mX, tiler_mn, (bidx, 0))  # (tileM, tileN)
    # TODO: vectorized store
    # gO = cute.local_tile(mO, cute.select(tiler_mn, mode=[0]), (bidx,))  # (tileM,)

    thr_copy_X = tiled_copy.get_slice(tidx)
    # gmem -> rmem
    tXgX = thr_copy_X.partition_S(gX)

    tXrX = cute.make_rmem_tensor_like(tXgX)
    cute.autovec_copy(tXgX, tXrX)

    # reduce with higher precision for numerical stability
    x = tXrX.load()
    val = x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)

    val = cute.arch.warp_reduction_sum(val)
    if tidx == 64 and bidx == 0:
        print(f"slice:\n{thr_copy_X}")
        print(f"partition: {tXgX}")
        print(f"x tyep: {type(x)}")
        print(f"val: {val}")
        print(f"gx: {gX}")

    lane_id = cute.arch.lane_idx()
    warp_id = cute.arch.warp_idx()

    warps_per_row = threads_per_row // cute.arch.WARP_SIZE

    row_idx = warp_id // warps_per_row
    col_idx = warp_id % warps_per_row

    # TODO: vetorized store
    if lane_id == 0 and col_idx == 0:
        mO[row_idx + tiler_mn[0] * bidx] = val


def test():
    M, N = 512, 1024
    X = torch.randn(M, N, dtype=torch.float32, device='cuda')
    Y = torch.empty((N,), dtype=torch.float32, device='cuda')

    dX = from_dlpack(X)
    dY = from_dlpack(Y)
    fn = cute.compile(
        test_jit,
        dX,
        dY
    )
    fn(dX, dY)


from cute_viz import render_layout_svg

@cute.jit
def visual():
    U = ((2,2),4,(9,(3,3)))
    layout = cute.make_layout(U)
    render_layout_svg(layout, "test.svg")



visual()