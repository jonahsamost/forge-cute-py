"""
Tiled copy/transpose kernel using CuTe DSL.
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import Float16, Float32, BFloat16, const_expr
from cutlass.cute.runtime import from_dlpack


class CopyTranspose:
    """Tiled copy/transpose operation using CuTe DSL."""

    def __init__(self, dtype: type, tile_size: int = 16):
        """
        Initialize copy/transpose kernel.

        Args:
            dtype: CUTLASS numeric type (Float16, Float32, BFloat16)
            tile_size: Size of square tile (default: 16)
        """
        self.dtype = dtype
        self.tile_size = tile_size
        # Use 16x16 tiles with 256 threads (16x16 = 256)
        self.num_threads = tile_size * tile_size

    @cute.jit
    def __call__(
        self,
        input: cute.Tensor,
        output: cute.Tensor,
        stream=None,
    ):
        """
        Execute tiled transpose.

        Args:
            input: Input tensor of shape (M, N)
            output: Output tensor of shape (N, M)
            stream: CUDA stream
        """
        M, N = input.shape
        tile_m = const_expr(self.tile_size)
        tile_n = const_expr(self.tile_size)

        # Calculate grid dimensions
        grid_m = cute.ceil_div(M, tile_m)
        grid_n = cute.ceil_div(N, tile_n)

        # Launch kernel
        self.kernel(input, output).launch(
            grid=[grid_m, grid_n, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input: cute.Tensor,
        output: cute.Tensor,
    ):
        """
        Transpose kernel implementation.

        Strategy:
        1. Each thread block handles a tile_m x tile_n tile
        2. Load from input to shared memory with coalesced reads
        3. Transpose in shared memory by swapping indices
        4. Write to output with coalesced writes
        """
        # Get thread and block indices
        tidx = cute.arch.thread_idx()[0]
        bidx, bidy = cute.arch.block_idx()[0], cute.arch.block_idx()[1]

        M, N = input.shape

        tile_m = const_expr(self.tile_size)
        tile_n = const_expr(self.tile_size)

        # Calculate tile coordinates in input
        tile_row = bidx * tile_m
        tile_col = bidy * tile_n

        # Thread layout: organize threads as tile_m x tile_n grid
        tx = tidx % tile_n  # Thread column within tile
        ty = tidx // tile_n  # Thread row within tile

        # Allocate shared memory for the tile (with padding to avoid bank conflicts)
        # Use tile_n + 1 stride to avoid bank conflicts
        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_layout(
            (tile_m, tile_n),
            stride=(tile_n + 1, 1),
        )
        tile_smem = smem.allocate_tensor(self.dtype, smem_layout, byte_alignment=16)

        # Load from global memory to shared memory
        # Each thread loads one element
        global_row = tile_row + ty
        global_col = tile_col + tx

        # Bounds check and load
        if global_row < M and global_col < N:
            val = input[global_row, global_col]
            tile_smem[ty, tx] = val

        # Synchronize to ensure all loads are complete
        cute.arch.sync_threads()

        # Write transposed tile to global memory
        # Note: we swap the indices when reading from shared memory
        # Thread (tx, ty) writes element from position (tx, ty) in smem
        # to position (tile_col + tx, tile_row + ty) in output
        out_row = tile_col + tx  # Swapped
        out_col = tile_row + ty  # Swapped

        # Bounds check and store
        # Output shape is (N, M) - transposed
        if out_row < N and out_col < M:
            # Read the element loaded by this thread
            val = tile_smem[ty, tx]
            output[out_row, out_col] = val


def copy_transpose_cute(x: torch.Tensor, tile: int = 16) -> torch.Tensor:
    """
    Perform tiled transpose using CuTe DSL.

    Args:
        x: Input tensor of shape (M, N)
        tile: Tile size (default: 16)

    Returns:
        Transposed tensor of shape (N, M)
    """
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor")

    if x.ndim != 2:
        raise ValueError("Input must be a 2D tensor")

    M, N = x.shape

    # Create output tensor (transposed shape)
    y = torch.empty((N, M), dtype=x.dtype, device=x.device)

    # Map PyTorch dtype to CUTLASS dtype
    dtype_map = {
        torch.float16: Float16,
        torch.float32: Float32,
        torch.bfloat16: BFloat16,
    }

    if x.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    cute_dtype = dtype_map[x.dtype]

    # Create and run kernel
    kernel = CopyTranspose(cute_dtype, tile_size=tile)
    input_cute = from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    output_cute = from_dlpack(y, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    kernel(input_cute, output_cute, stream=cutlass_torch.current_stream())

    return y
