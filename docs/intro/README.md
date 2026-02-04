### CuTe DSL Intro

The stated goal of CUTLASS is to bridge the gap between productivity and performance for CUDA kernel development. The goal of CuTe DSL is to enable rapid prototyping and iteration on top of CUTLASS.
The goals of this blogpost are twofold. One is to try to rapidly get _you_ sped up such that you can accomplish the stated goals of CuTe.


The first part of this blogpost will go into the how and why CuTe wants us to program they way it does and the second part will attempt to get across the mental model CuTe wants us to have while programming in CuTe.


Let's start with a simple question, how _should_ I be running my code?
A lot of this example inspiration comes from Nvidia's cutlass example [here](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb).

But instead of doing an add, we'll be using `torch.sum` as our running example.

So, let's create just a simple kernel, for now only focusing on the last dimension and assume a contiguous, 2d tensor.
```python
@cute.kernel
def _reduce_sum_kernel(gA: cute.Tensor, output: cute.Tensor):
    smem_alloc = cutlass.utils.SmemAllocator()
    shmem = smem_alloc.allocate_tensor(cute.Float32, cute.make_layout((32,)))  # warp size

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    M, N = gA.shape
    ntiles = cute.ceil_div(N, bdimx)
    acc = 0.0
    for tile_idx in range(ntiles):
        idx = tile_idx * bdimx + tidx
        if idx < N:
            acc += gA[bidx, idx]

    acc = cute.arch.warp_reduction_sum(acc)
    if lane == 0:
        shmem[warp] = acc
    cute.arch.sync_threads()

    if warp == 0:
        block_warps = bdimx // 32
        acc2 = shmem[lane] if lane < block_warps else 0.0
        acc2 = cute.arch.warp_reduction_sum(acc2)
        if lane == 0:
            output[bidx] = acc2

@cute.jit
def reduce_sum(x: cute.Tensor, output: cute.Tensor, dim=-1):
    num_warps = 4
    threads_per_block = num_warps * 32
    M, N = x.shape

    _reduce_sum_kernel(x, output).launch(
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
```

The code here is pretty self explanatory and reads very much like regular Cuda code.
There are a few things we will be going over later like `from_dlpack`,
`cute.compile`, and actually launching the kernel, but for now let's take that for granted.

When you run `uv run python part1.py simple_launch`, we see `Success!`, but how fast is it? Let's compare versus torch.

Let's run `uv run python part1.py compare_torch_initial`. 
```bash
(forge-cute-py) root@33835e6c1ad1:~/forge-cute-py/docs/intro# python part1.py compare_torch_initial
  cute dsl reduce sum: 135.723 ms
  torch kernel add: 0.127 ms
```
Ok, not great! We're clearly doing something wrong.

[Reading the docs](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html#custom-caching-with-cute-compile) you quickly see that `cute.compile` bypasses caching in CuTe DSL and _always_
performs compilation. There are also a [couple parameters](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html?#jit) we can control as it relates to JIT compilation.

Ok, so on that [custom caching](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html?#jit) page in the docs, seemingly we only need a regular 
dictionary and a key that maps us back to our compiled function.

We can assume that in some future state of this function, the key for that custom cache might be some tuple of
shape, dtype, etc.

Let's try that.

```python
def _invoke_reduce_sum_with_cache(gInput, dim=-1):
    a_ = from_dlpack(gInput)

    sz = gInput.size(0) if dim == -1 else gInput.size(1)
    gOutput = torch.empty((sz,), dtype=gInput.dtype, device=gInput.device)
    b_ = from_dlpack(gOutput)

    key = f'reduce_sum_{gInput.dtype}'
    if key not in custom_cache:
        custom_cache[key] = cute.compile(reduce_sum, a_, b_)
    fn = custom_cache[key]
    fn(a_, b_)
    return gOutput
```

```bash
(forge-cute-py) root@33835e6c1ad1:~/forge-cute-py/docs/intro# python part1.py compare_torch_initial_cached
  cute dsl reduce sum: 0.115 ms
  torch kernel sum: 0.127 ms
```

Ok, caching really helped!




Though our code looks more like Cuda written in python, rather than CuTe.
But we haven't actually used any core abstractions that CuTe supports, namely [Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#).
A Layout, as described in those docs, "present a common interface to multidimensional array access that abstracts away the details of how the array’s elements are organized in memory".

Ok, so let's make use of a simple [layout](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/core.py#L2808).
As you can see from those docs, a layout is defined mainly by a shape and an optional stride. A layout is not data itself, it is the shape
and indexing rule that allows you to know where the data is stored and how to traverse it. What is also interesting about layouts is that
they are both [hierarchical](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#hierarchical-access-functions) and compositional, which is to say, you can define a layout of layouts.

Ok, so given this, in order to layout-ify our reduce_sum kernel, we know we'd want _some_ sort of hierarchy. If we map our current kernel
to this layout frame of mind, we know we want each thread to access a single value for each iteration in the main loop.
So, we can think of this as a layout of shape (1,). We know these N warps will be their own layout, so we can think of them as a layout
of shape (32 * N,). Finally, we know that in a matrix of shape (M, N) that we will want a layout of (M, warp_layout).

So, conceptually, something similar to:
```bash
Layout<
    (M,),
    Layout<
        (ceil_div(N, T),),
        (T,)
    >
>
```

But how do we do this? [This](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/core.py#L2808) layout function
looks like we can use it for ours warps and single thread value. 
Looking into the cutlass [notebook](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb)
we see that in order to combine them we must make use of `cute.make_layout_tv` from [here](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/core.py#L3956).
So, we will need to do something like this:
```python
@cute.jit
def print_layout():
    threads_per_block = 32
    thr_layout = cute.make_layout((threads_per_block,), stride=(1,))
    val_layout = cute.make_layout((1,), stride=(1,))
    tiler_mn, layout_tv = cute.make_layout_tv(thr_layout, val_layout)
    print(f"layout: {layout_tv}")
    print(f"tiler: {tiler_mn}")
    tiler_2d = (1, tiler_mn[0]) # cute-viz expects 2d
    render_tv_layout_svg(layout_tv, tiler_2d, "tv_layout.svg")

print_layout()
...
layout: (32,1):(1,0)
tiler: (32,)
```

Visualizing that with cute-viz's `render_tv_layout_svg` shows that each thread in the warp will work on a
single value at a time. And we can just iterate over all the columns of whatever row we are to retrieve all the values.
![TV Layout Visualization](tv_layout_32_1.svg)

Looking back to the cutlass notebook, we see that `cute.zipped_divide` is the means through which we will then tile our
layout over our whole MxN tensor. Extending our print_layout() function to call `cute.zipped_divide` 
```python
    ...
    gX = cute.zipped_divide(mA, tiler_2d)
    print(f"gx: {gX}")
```
We see: `gx: tensor<ptr<f32, gmem> o ((1,32),(?,?)):((0,1),(?,32))>`, which is CuTe telling us that some values are
dynamic and not statically known at compile time. But how should we read this? CuTe will print tensors as:
`tensor<ptr<T, space> o SHAPE : STRIDE >`. Looking back to the [cutlass](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#zipped-tiled-flat-divides) docs sheds some light.

```bash
# We see from the docs that the shape returned from zipped_divide is ((TileM,TileN), (RestM,RestN,L,...))
# This shape corresponds to how we called zipped_divide
shape = (1,32),(?,?)  
# Since we called
zipped_divide(mA, tiler=(1,32))
# Because tiler is (1,32)
# we know that TileM=1, TileN=32
# and RestM = M / 1 = M and RestN = N / 32 = ceil(N, 32)
```

And putting it altogether, we get something like this:
```python
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
        blk = input[blk_coord]
        thr_frag = cute.composition(blk, tv_layout)
        thr_val  = thr_frag[tidx, 0]
        acc += thr_val

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

    thr_layout = cute.make_layout((threads_per_block,), stride=(1,))
    val_layout = cute.make_layout((1,), stride=(1,))
    tiled_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gX = cute.zipped_divide(x, (1, tiled_mn[0]))
    
    _reduce_sum(gX, output, tv_layout, num_warps).launch(
        grid=(M, 1, 1),
        block=(threads_per_block, 1, 1),
    )
```





But is caching really that simple? Aren't we incuring overhead from that `from_dlpack` call?
What does that call even do? Looking at the [docs](https://github.com/NVIDIA/cutlass/blob/a4eb0e05f6dd0403f94087b495393bdca75bf0ad/python/CuTeDSL/cutlass/cute/runtime.py#L746), this function is seemingly only creating for us a `cutlass.cute.Tensor`.
How much overhead could that be? 

Let's run
```
nsys profile --trace=cuda,nvtx,osrt --force-overwrite true -o launch_profile uv run python -m compare_torch_initial_cached
```
