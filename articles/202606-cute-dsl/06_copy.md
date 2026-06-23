# 动手学 CuTeDSL 06：从 Tensor 赋值到 Copy Atom：`copy`、`cp.async` 与 `ldmatrix`

前面几篇已经分别介绍了 `Layout`、layout 代数、TV layout 和 `TiledMMA`。要读懂单精度矩阵乘法的实现 `sgemm.py`[8]，还缺一条主线：数据是怎样从 global memory 搬到 shared memory，再进入每个线程的寄存器，最后写回 global memory 的。

在 CuTe DSL 里，“copy”可以有好几层含义：

- 最直接的是把一个已经求值的 tensor 元素写到另一个 tensor；
- 再往上一层，是把一个 tensor slice 先变成线程私有寄存器值，再回写；
- 再往下贴近硬件，则是用 `CopyUniversalOp`、`cp.async.CopyG2SOp`、`LdMatrix8x8x16bOp` 这类 copy atom 明确描述数据如何在 global memory、shared memory 和寄存器之间流动。

本文用一个 `16x16` 的 `Float16` 矩阵，顺着这个层次走一遍。主线对应 `sgemm.py` 里的几段关键代码：

1. 先看为什么“标量赋值”能工作，而“slice 赋值”会报错；
2. 再看 `TensorSSA`、`autovec_copy` 与 `CopyUniversalOp` 这几种等价但抽象层次不同的写法；
3. 重点理解 `cp.async`、`TiledCopy`、`partition_S/D` 如何完成 global-to-shared staging；
4. 最后把 `ldmatrix` 作为 Ampere Tensor Core 路线的补充知识。`sgemm.py` 是 FP32 SIMT GEMM，shared-to-register 走的是 `autovec_copy`，不是 `ldmatrix`。

为了让输出一眼可读，下面把输入矩阵 `A` 设成递增序列：

```python
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

M, N = 16, 16

A = torch.arange(M * N, dtype=torch.float16, device="cuda").reshape(M, N)
B = torch.randn(M, N, dtype=torch.float16, device="cuda")

mA = from_dlpack(A, assumed_align=16)
mB = from_dlpack(B, assumed_align=16)
```

---

## 先看最简单的标量 copy

如果索引的是单个元素，那么右侧表达式会先被求值成线程私有的标量值，因此可以直接赋给目标 tensor：

```python
B[:, :] = 0.

@cute.jit
def host_func(mA: cute.Tensor, mB: cute.Tensor):
    kernel_func(mA, mB).launch(grid=[1, 1, 1], block=[32, 1, 1])


@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    mB[tidx] = mA[tidx]

    coord = cute.make_identity_tensor(mA.layout.shape)[tidx]
    cute.printf("thread {} copies mB[{}] = mB[{}]", tidx, tidx, coord)


host_func(mA, mB)
print(f"B: {B}")
```

这个例子最值得注意的不是“能复制”，而是它复制了哪一部分。

打印结果显示：

- `tidx=0..15` 对应 `mB[(0,0)]` 到 `mB[(15,0)]`；
- `tidx=16..31` 对应 `mB[(0,1)]` 到 `mB[(15,1)]`。

这符合前面讲到的列主序的坐标映射逻辑。对这个 `16x16` 例子，32 个线程只覆盖了前 32 个逻辑元素，因此 `B` 里最终被写好的其实是前两列，而不是前两行。

这正是“full evaluation”的意思：右侧已经是单个值了，赋值本身没有问题；但如果希望按“整行”或“整块”复制，就需要显式切到更高一级的 copy 抽象。

---

## 为什么 tensor slice 不能直接赋值

如果把右侧改成一个 slice：

```python
@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < M:
        mB[(tidx, None)] = mA[(tidx, None)]
```

会直接报错：

```text
ValueError: Expected TensorSSA, but got tensor<ptr<f16, gmem, align<16>> o (16):(1)>
```

原因是 `mA[(tidx, None)]` 仍然是一个 **tensor view**，它描述的是 global memory 上一整行的视图，而不是已经进入寄存器、带 value semantics 的线程私有值。CuTe 对 slice 赋值要求右侧是 `TensorSSA` [1][2]。

换句话说：

- `mA[tidx]` 是“这个线程现在就拿到的一个标量”；
- `mA[(tidx, None)]` 是“这个线程看到的一段 tensor 视图”；
- 后者如果要写回目标 tensor，必须先显式地做 load，得到寄存器里的 `TensorSSA`。

---

## 用 `TensorSSA` 做 slice copy

最直接的修正方式，就是先 `.load()`，再 `.store()`：

```python
B[:, :] = 0.

@cute.jit
def host_func(mA: cute.Tensor, mB: cute.Tensor):
    kernel_func(mA, mB).launch(grid=[1, 1, 1], block=[32, 1, 1])


@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < M:
        rA = mA[(tidx, None)].load()
        mB[(tidx, None)].store(rA)


host_func(mA, mB)
print(f"B: {B}")
```

这里的关键变化只有一处：

- `mA[(tidx, None)].load()` 把 global-memory slice 变成寄存器里的 `TensorSSA`；
- `mB[(tidx, None)].store(rA)` 再把这个寄存器值写回目标 tensor。

输出结果里，线程 `0-15` 各自处理一整行，线程 `16-31` 跳过，因此 `B` 会变成和 `A` 完全一致的 `16x16` 矩阵。

这也说明了一个很实用的判断标准：只要右侧还是 memory-backed tensor view，而不是 value tensor，就不要期待它能像普通 Python 值一样直接赋值。

---

## `autovec_copy` 与 `CopyUniversalOp`

在“源和目标都是静态 shape 的 slice”这个场景下，上面的 `load()/store()` 还可以换成两个更贴近 CuTe copy 抽象的写法。

### `autovec_copy`

`autovec_copy` 会自动挑选当前场景下安全的最大向量宽度 [3]：

```python
@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < M:
        cute.autovec_copy(mA[(tidx, None)], mB[(tidx, None)])
```

对这一节的 `16` 个 `Float16` 行切片来说，它和前一节的 `load()/store()` 有相同效果：每个活跃线程复制自己负责的一整行。

### `CopyUniversalOp`

如果希望显式地走 copy atom 的接口，可以写成：

```python
@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mA.element_type)

    if tidx < M:
        cute.copy(copy_atom, mA[(tidx, None)], mB[(tidx, None)])
```

`CopyUniversalOp` 可以把这类“普通的 SIMT copy”也纳入统一的 atom / tiled-copy 框架 [4]。它很适合作为 `TensorSSA` 级写法和后面硬件专用 atom 之间的过渡层去学习。

---

## `cp.async`：先把数据搬到 shared memory

到了 `cp.async.CopyG2SOp`，copy 的语义就更贴近硬件了：它描述的是 **global memory 到 shared memory** 的异步 copy atom [5][6]。

先看最直接的写法：

```python
B[:, :] = 0.

@cute.jit
def host_func(mA: cute.Tensor, mB: cute.Tensor):
    kernel_func(mA, mB).launch(grid=[1, 1, 1], block=[32, 1, 1])


@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()

    allocator = cutlass.utils.SmemAllocator()
    smem_tensor = allocator.allocate_tensor(
        element_type=mA.element_type,
        layout=mA.layout,
        byte_alignment=128,
        swizzle=None,
    )

    async_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        mA.element_type,
        num_bits_per_copy=128,
    )

    if tidx < M:
        cute.copy(async_copy_atom, mA[(tidx, None)], smem_tensor[(tidx, None)])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(n=0)

        rmem_tensor = smem_tensor[(tidx, None)].load()
        mB[(tidx, None)] = rmem_tensor

    cute.arch.sync_threads()
```

这个例子有三个重要观察点。

第一，`cp.async` 的核心价值不是“换一种写法做 load/store”，而是把 **GMem -> SMem** 这段搬运变成可排队的异步操作。真实 GEMM kernel 不会像这个小例子一样 copy 后立刻等待，而是会先发起若干组 `cp.async`，一边消费已经到达 shared memory 的 tile 做计算，一边让后续 tile 的 global memory 访问在后台推进。`sgemm.py` 里的 prologue、mainloop、`cp_async_commit_group()`、`cp_async_wait_group(k_pipe_max - 2)` 就是在组织这种 copy 和 compute 的重叠。

第二，`cp.async` 的 atom 粒度是 `128 bit = 16 B`。对 `Float16` 来说，这正好是 `8` 个元素，而不是一整行 `16` 个元素。

第三，从这段输出可以推断，直接把 `(16)` 的整行 slice 喂给一个 `128-bit` 的 atom，并不能自然表达“整行搬运”这件事：共享内存打印结果里，首行只有前半段是确定写好的，剩余位置没有被这一条 atom 完整覆盖。

因此，这一节更适合被看成一个 API 形态示例：它说明了 `cp.async` 需要 `commit_group`、`wait_group`，并且在消费 shared memory 之前还要有 block 级同步；但如果要稳定地覆盖更大的 tile，最好还是把 tensor 显式切成与 atom 粒度匹配的小块。

---

## 用 `TiledCopy` 组织 `cp.async`

对这个 `16x16`、`Float16`、`128-bit` 的例子，最自然的分块方式就是：

- 每次 copy `8` 个 `Float16`；
- 每行分成 `2` 个列块；
- `32` 个线程排成 `(16, 2)` 的线程布局；
- 每个线程的 value 布局是 `(1, 8)`。

代码如下：

```python
from cute_viz import display_tiled_copy, display_tv_layout

num_threads = 32

@cute.jit
def vis_tiled_copy(mA: cute.Tensor, mB: cute.Tensor):
    async_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        mA.element_type,
        num_bits_per_copy=128,
    )

    copy_elems = 128 // 16
    col_blocks = N // copy_elems
    thread_layout = cute.make_layout(
        (num_threads // col_blocks, col_blocks),
        stride=(col_blocks, 1),
    )  # (16, 2)
    value_layout = cute.make_layout((1, copy_elems))  # (1, 8)

    tiled_async_copy = cute.make_tiled_copy_tv(
        async_copy_atom,
        thread_layout,
        value_layout,
    )

    display_tiled_copy(tiled_async_copy, (M, N))
    print(f"tiled_async_copy.layout_src_tv_tiled: {tiled_async_copy.layout_src_tv_tiled}")
    display_tv_layout(tiled_async_copy.layout_src_tv_tiled, (M, N))
```

这个 `TiledCopy` 的源端 TV layout 会打印成：

$$
\text{layout\_src\_tv\_tiled} = ((2,16),(8,1)):((128,1),(16,0))

$$

这张图展示了 `cp.async` 的完整 `16x16` tiled copy：32 个线程被组织成两列 thread block，每个线程处理一个连续的 `8` 元素向量。

![The `cp.async` tiled copy on `16x16` organizes 32 threads into two column blocks, each carrying one contiguous 8-element vector.](img/06_copy_cpasync_tiled_16x16.svg)

把源端 TV layout 单独拿出来看，会更容易理解“谁负责哪一个 `16B` 向量”。

![The source TV layout of the `cp.async` tiled copy highlights how the `16x16` tile is partitioned into thread-owned 8-element vectors.](img/06_copy_cpasync_src_tv_16x16.svg)

真正执行时，每个线程先取自己的 slice：

```python
@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()

    allocator = cutlass.utils.SmemAllocator()
    smem_tensor = allocator.allocate_tensor(
        element_type=mA.element_type,
        layout=mA.layout,
        byte_alignment=128,
        swizzle=None,
    )

    async_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        mA.element_type,
        num_bits_per_copy=128,
    )

    copy_elems = 128 // 16
    col_blocks = N // copy_elems
    thread_layout = cute.make_layout((16, 2), stride=(2, 1))
    value_layout = cute.make_layout((1, copy_elems))

    tiled_async_copy = cute.make_tiled_copy_tv(
        async_copy_atom,
        thread_layout,
        value_layout,
    )
    thr_async_copy = tiled_async_copy.get_slice(tidx)

    mA_tile = thr_async_copy.partition_S(mA)
    smem_tile = thr_async_copy.partition_D(smem_tensor)

    cute.copy(thr_async_copy, mA_tile, smem_tile)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(n=0)
    cute.arch.sync_threads()

    if tidx < M:
        rmem_tensor = smem_tensor[(tidx, None)].load()
        mB[(tidx, None)] = rmem_tensor
```

输出里，线程 `3` 看到的 `mA_tile` 恰好是 `24..31` 这一段，说明它负责的是第 `1` 行后半个 `16B` 向量。最终 `B` 与 `A` 完全一致，说明这一次 partition 和 atom 粒度终于对齐了。

这正是 `sgemm.py` 里 global-to-shared copy 的缩小版：

- `__call__` 中先根据 A/B 的 layout 构造 `atom_async_copy_A/B`、`tA/tB`、`vA/vB`，再得到 `tiled_copy_A/B`；
- `kernel` 中每个线程调用 `tiled_copy_A.get_slice(tidx)` 得到 `thr_copy_A`；
- `thr_copy_A.partition_S(gA)` 把 global memory tile 切成当前线程要读的片段；
- `thr_copy_A.partition_D(sA)` 把 shared memory tile 切成当前线程要写的片段；
- 最后 `cute.copy(tiled_copy_A, tAgA[...], tAsA[...], pred=...)` 发出真正的 copy atom。

真实 SGEMM 里多出来的维度主要是 K tile 和 pipeline stage：`gA`/`gB` 形如 `(BLK_M, BLK_K, k)`，`sA`/`sB` 形如 `(BLK_M, BLK_K, PIPE)`。但核心动作仍然是同一个：先定义线程到数据的 TV layout，再用 `partition_S/D` 把大 tile 切成每个线程的 copy 任务。

---

## 不借助 `TiledCopy`，也可以手工把 `cp.async` 分块

如果不想用 `make_tiled_copy_tv`，也可以把同样的分块逻辑手工写出来。核心是先把张量按 `(1, 8)` 的 atom 形状切开：

```python
tiler = (cute.make_layout(1), cute.make_layout(8))
tiled_mA = cute.zipped_divide(mA, tiler)
tiled_smem = cute.zipped_divide(smem_tensor, tiler)
```

对这个 `16x16` 例子，打印结果是：

```text
tiled_mA:   tensor<... o ((1,8),(16,2)):((0,1),(16,8))>
tiled_smem: tensor<... o ((1,8),(16,2)):((0,1),(16,8))>
```

这说明原始矩阵已经被改写成：

- atom 内部形状是 `(1, 8)`；
- 外层 tile 网格是 `(16, 2)`，也就是 16 行、每行 2 个向量块。

因此每个线程只要做：

```python
y = tidx // 2
x = tidx % 2
mA_tile = tiled_mA[((0, None), (y, x))]
smem_tile = tiled_smem[((0, None), (y, x))]
```

就能拿到和上一节 `TiledCopy` 版本等价的 `8` 元素向量 slice。

完整代码如下：

```python
@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()

    allocator = cutlass.utils.SmemAllocator()
    smem_tensor = allocator.allocate_tensor(
        element_type=mA.element_type,
        layout=mA.layout,
        swizzle=None,
    )

    async_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        mA.element_type,
        num_bits_per_copy=128,
    )

    tiler = (cute.make_layout(1), cute.make_layout(8))
    tiled_mA = cute.zipped_divide(mA, tiler)
    tiled_smem = cute.zipped_divide(smem_tensor, tiler)

    y = tidx // 2
    x = tidx % 2
    mA_tile = tiled_mA[((0, None), (y, x))]
    smem_tile = tiled_smem[((0, None), (y, x))]

    cute.copy(async_copy_atom, mA_tile, smem_tile)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(n=0)
    cute.arch.sync_threads()

    if tidx < M:
        rmem_tensor = smem_tensor[(tidx, None)].load()
        mB[(tidx, None)] = rmem_tensor
```

这里还有一个很实用的细节：如果把索引写成 `tiled_mA[(None, (y, x))]`，原例里会触发 `CUDA error: misaligned address`。`((0, None), (y, x))` 这种写法得到的是对齐好的 `8` 元素向量视图，正好匹配 `128-bit` 的 `cp.async` atom。

因此，这一节和上一节的关系可以概括成一句话：`TiledCopy` 本质上就是把“手工切块 + 每线程取 slice + 做 atom copy”这套模式封装成了一个更高层、更容易检查 TV layout 的接口。

---

## `ldmatrix`：把 shared memory tile 装入寄存器

`cp.async` 负责的是 **GMem -> SMem**。如果接下来要喂给 Tensor Core，通常还要再做一步 **SMem -> Register**，在 Ampere 架构下配合 Tensor Core 使用的就是 `ldmatrix` [7]。

如果当前目标只是读懂 `ampere/sgemm.py`，这一节可以先跳过。`sgemm.py` 是 FP32 SIMT GEMM，它在 shared-to-register 阶段使用的是：

```python
cute.autovec_copy(tCsA_p[None, None, k_block_next], tCrA[None, None, k_block_next])
cute.autovec_copy(tCsB_p[None, None, k_block_next], tCrB[None, None, k_block_next])
```

也就是普通的 vectorized load/store。`ldmatrix` 更常见于 Ampere 上的 FP16/BF16 Tensor Core 路线；到了 Hopper 以后，还会出现 TMA、WGMMA 等新的数据搬运和计算模型。因此，下面两小节更像是理解 Ampere Tensor Core 数据路径的补充。

`ldmatrix` 的基本流程可以压缩成五步：

1. 先用 `cp.async` 或普通 copy 把一个 tile 放进 shared memory；
2. 在所有线程都会经过的位置等待 copy 完成，并做 block 级同步；
3. 用 `LdMatrix8x8x16bOp` 创建 shared-to-register 的 copy atom；
4. 用 `make_tiled_copy_tv` 或 `make_tiled_copy` 描述 warp 内线程如何从 shared memory 读出 fragment；
5. 对当前线程调用 `get_slice(tidx)`，再通过 `partition_S` 和 `retile` 得到源端 shared-memory 视图和目标端 register 视图，最后 `cute.copy(...)`。

### 标准 `8x8`：每线程拿 2 个 `Float16`

`8x8` 的 shared-to-register copy 可以写成：

```python
M, N = 8, 8

@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()

    allocator = cutlass.utils.SmemAllocator()
    smem_tensor = allocator.allocate_tensor(
        element_type=mA.element_type,
        layout=mA.layout,
        byte_alignment=128,
        swizzle=None,
    )

    async_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        mA.element_type,
        num_bits_per_copy=128,
    )

    if tidx < M:
        cute.copy(async_copy_atom, mA[(tidx, None)], smem_tensor[(tidx, None)])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(n=0)

    cute.arch.sync_threads()

    ldmatrix_op = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False)
    copy_atom = cute.make_copy_atom(ldmatrix_op, mA.element_type)
    thr_layout = cute.make_layout((8, 4), stride=(4, 1))
    val_layout = cute.make_layout((1, 2), stride=(2, 1))
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy_ldmatrix = tiled_copy.get_slice(tidx)

    tiled_val_layout = cute.make_layout((2, 1, 1), stride=(1, 0, 0))
    rmem_tensor = cute.make_rmem_tensor_like(tiled_val_layout, dtype=mA.element_type)

    smem_tensor_view = thr_copy_ldmatrix.partition_S(smem_tensor)
    rmem_tensor_view = thr_copy_ldmatrix.retile(rmem_tensor)

    cute.copy(
        thr_copy_ldmatrix,
        smem_tensor_view[None, None, None],
        rmem_tensor_view[None, None, None],
    )

    y = tidx // 4
    x1 = (tidx % 4) * 2
    x2 = x1 + 1
    mB[(y, x1)] = rmem_tensor[0]
    mB[(y, x2)] = rmem_tensor[1]
```

打印出来的 `thr_copy_ldmatrix` 是：

```text
Tiled Copy
  Tiler MN:        (8:1,8:1)
  TV Layout tiled: ((4,8),2):((16,1),8)
Copy Atom
  ThrID:           32:1
  TV Layout Src:   ((8,4),8):((8,0),1)
  TV Layout Dst:   (32,2):(2,1)
```

这个 layout 和 `ldmatrix` 的经典语义是对上的：

- shared memory 源端是 `8x8` tile；
- 线程 `0-7` 负责 8 行地址；
- 32 个线程在寄存器目标端各拿 `2` 个 `Float16`。

这张图展示了标准 `8x8` `ldmatrix` copy 的 thread-value 分工：左边是 shared memory 源布局，右边是寄存器目标布局。

这张图可以由同目录下的 `06_copy_figures.py` 生成，核心就是对 `LdMatrix8x8x16bOp` 创建 `TiledCopy` 后调用 `render_tiled_copy_svg`。

![The standard `8x8` `ldmatrix` copy maps shared-memory rows to a register fragment where each thread owns two values.](img/06_copy_ldmatrix_8x8.svg)

回写到 `mB` 时，使用

- `y = tidx // 4`
- `x1 = (tidx % 4) * 2`
- `x2 = x1 + 1`

就能把每个线程手里的两个寄存器值重新散回 `8x8` 的正确位置，最终 `B` 与 `A` 完全一致。

---

## `.num_matrices=4`：让一个 warp 一次装入 `16x16`

如果 shared memory 里已经有一个 `16x16` tile，还可以让同一个 warp 一次执行 `.x4` 风格的 `ldmatrix`。在 CuTe DSL 里，对应的是：

```python
ldmatrix_op = cute.nvgpu.warp.LdMatrix8x8x16bOp(
    transpose=False,
    num_matrices=4,
)
```

这时要配一个能覆盖 `16x16` 的目标 TV layout：

```python
tv_layout = cute.make_layout(
    ((4, 8), (2, 2, 2)),
    stride=((32, 1), (16, 8, 128)),
)
```

完整的 S2R 部分如下：

```python
ldmatrix_op = cute.nvgpu.warp.LdMatrix8x8x16bOp(
    transpose=False,
    num_matrices=4,
)
copy_atom = cute.make_copy_atom(ldmatrix_op, mA.element_type)

tv_layout = cute.make_layout(
    ((4, 8), (2, 2, 2)),
    stride=((32, 1), (16, 8, 128)),
)
tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (16, 16))
thr_copy_ldmatrix = tiled_copy.get_slice(tidx)

tiled_val_layout = cute.make_layout((8, 1, 1), stride=(1, 0, 0))
rmem_tensor = cute.make_rmem_tensor_like(tiled_val_layout, dtype=mA.element_type)

smem_tensor_view = thr_copy_ldmatrix.partition_S(smem_tensor)
rmem_tensor_view = thr_copy_ldmatrix.retile(rmem_tensor)

cute.copy(
    thr_copy_ldmatrix,
    smem_tensor_view[None, None, None],
    rmem_tensor_view[None, None, None],
)
```

这时的打印结果是：

```text
Tiled Copy
  Tiler MN:        (16:1,16:1)
  TV Layout tiled: ((4,8),(2,2,2)):((32,1),(16,8,128))
Copy Atom
  ThrID:           32:1
  TV Layout Src:   (32,8):(8,1)
  TV Layout Dst:   (32,(2,4)):(2,(1,64))
```

和标准 `8x8` 版本相比，这里最关键的变化是：

- 源端不再是“8 行地址 + 每线程 2 个目标值”的单矩阵模式；
- 32 个线程现在各自带着 `8` 个 `Float16` 寄存器值；
- 这 `8` 个值对应 `16x16` tile 中四个 `8x8` 子矩阵的 fragment。

这张图展示了 `.num_matrices=4` 的 `16x16` `ldmatrix` copy 布局：同一个 warp 仍然只有 32 个线程，但每个线程携带的寄存器 fragment 已经明显变大。

![The `.num_matrices=4` `ldmatrix` copy covers a `16x16` tile with one warp, giving each thread a larger 8-value register fragment.](img/06_copy_ldmatrix_num4_16x16.svg)

最后把 `rmem_tensor[0:8]` 依次散回四个象限：

```python
y = tidx // 4
x1 = (tidx % 4) * 2
x2 = x1 + 1

mB[(y, x1)] = rmem_tensor[0]
mB[(y, x2)] = rmem_tensor[1]
mB[(y + 8, x1)] = rmem_tensor[2]
mB[(y + 8, x2)] = rmem_tensor[3]
mB[(y, x1 + 8)] = rmem_tensor[4]
mB[(y, x2 + 8)] = rmem_tensor[5]
mB[(y + 8, x1 + 8)] = rmem_tensor[6]
mB[(y + 8, x2 + 8)] = rmem_tensor[7]
```

这样就能把一个 warp 装回来的四组 fragment 重新拼成完整的 `16x16` 输出 tile。

如果希望把这一节单独跑通，下面是一份完整代码。它包含：

- `16x16` 的 global-memory 输入；
- 用 `cp.async` 把数据按 `(1, 8)` 向量切块搬到 shared memory；
- 用 `.num_matrices=4` 的 `ldmatrix` 一次装入整个 `16x16` tile；
- 最后把每线程的 `8` 个寄存器值重新散回 `mB`。

附完整代码。

```python
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

M, N = 16, 16

A = torch.arange(M * N, dtype=torch.float16, device="cuda").reshape(M, N)
B = torch.randn(M, N, dtype=torch.float16, device="cuda")

mA = from_dlpack(A, assumed_align=16)
mB = from_dlpack(B, assumed_align=16)

B[:, :] = 0.


@cute.jit
def host_func(mA: cute.Tensor, mB: cute.Tensor):
    kernel_func(mA, mB).launch(grid=[1, 1, 1], block=[32, 1, 1])


@cute.kernel
def kernel_func(mA: cute.Tensor, mB: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()

    # GMem -> SMem: use 128-bit cp.async on (1, 8) Float16 vectors
    allocator = cutlass.utils.SmemAllocator()
    smem_tensor = allocator.allocate_tensor(
        element_type=mA.element_type,
        layout=mA.layout,
        byte_alignment=128,
        swizzle=None,
    )

    async_copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        mA.element_type,
        num_bits_per_copy=128,
    )

    tiler = (cute.make_layout(1), cute.make_layout(8))
    tiled_mA = cute.zipped_divide(mA, tiler)
    tiled_smem = cute.zipped_divide(smem_tensor, tiler)

    y = tidx // 2
    x = tidx % 2
    mA_tile = tiled_mA[((0, None), (y, x))]
    smem_tile = tiled_smem[((0, None), (y, x))]

    cute.copy(async_copy_atom, mA_tile, smem_tile)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(n=0)
    cute.arch.sync_threads()

    # SMem -> Register: one warp loads the whole 16x16 tile via ldmatrix.x4
    ldmatrix_op = cute.nvgpu.warp.LdMatrix8x8x16bOp(
        transpose=False,
        num_matrices=4,
    )
    copy_atom = cute.make_copy_atom(ldmatrix_op, mA.element_type)
    tv_layout = cute.make_layout(
        ((4, 8), (2, 2, 2)),
        stride=((32, 1), (16, 8, 128)),
    )
    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (16, 16))
    thr_copy_ldmatrix = tiled_copy.get_slice(tidx)

    tiled_val_layout = cute.make_layout((8, 1, 1), stride=(1, 0, 0))
    rmem_tensor = cute.make_rmem_tensor_like(tiled_val_layout, dtype=mA.element_type)

    smem_tensor_view = thr_copy_ldmatrix.partition_S(smem_tensor)
    rmem_tensor_view = thr_copy_ldmatrix.retile(rmem_tensor)

    cute.copy(
        thr_copy_ldmatrix,
        smem_tensor_view[None, None, None],
        rmem_tensor_view[None, None, None],
    )

    # Register -> GMem: scatter the 8 values back to the four 8x8 quadrants
    y = tidx // 4
    x1 = (tidx % 4) * 2
    x2 = x1 + 1

    mB[(y, x1)] = rmem_tensor[0]
    mB[(y, x2)] = rmem_tensor[1]
    mB[(y + 8, x1)] = rmem_tensor[2]
    mB[(y + 8, x2)] = rmem_tensor[3]
    mB[(y, x1 + 8)] = rmem_tensor[4]
    mB[(y, x2 + 8)] = rmem_tensor[5]
    mB[(y + 8, x1 + 8)] = rmem_tensor[6]
    mB[(y + 8, x2 + 8)] = rmem_tensor[7]


compiled_func = cute.compile(host_func, mA, mB)
compiled_func(mA, mB)
print(f"A: {A}")
print(f"B: {B}")
```

---

## 附录：一个容易踩中的同步错误

先说结论：`cute.arch.sync_threads()` 不能放在依赖 `tidx < M` 这类条件的不同控制流里，否则非常容易挂住。

这不是 `ldmatrix` 特有的问题，而是 block 级屏障的基本要求：同一个 thread block 里的所有线程，必须在一致的结构位置执行同步。把 barrier 分别塞到 `if` 和 `else` 分支中，哪怕两个分支里都“写了一个 barrier”，也不等价于“整个 block 在同一个点同步”。

这一点在把 `8x8` tile 先搬入 shared memory、再做 `ldmatrix` 时尤其容易出错，因为只有线程 `0-7` 负责提供 8 行的起始地址，其余线程在 G2S 阶段往往只是占位线程。

下面这段就是一个会挂住的错误结构。问题不在 `cp.async` 或 `ldmatrix` 本身，而在于 `sync_threads()` 被放进了两个分支里：

```python
if tidx < M:
    cute.copy(async_copy_atom, mA[(tidx, None)], smem_tensor[(tidx, None)])
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(n=0)

    cute.arch.sync_threads()  # 错误：barrier 在 if 分支里
else:
    cute.arch.sync_threads()  # 错误：这里不是同一个同步点
```

这种写法的问题可以直接概括成一句话：**barrier 必须出现在所有线程都会以完全一致的控制流经过的位置**。只要把它放进条件分支，后面就很容易出现“有的线程已经继续往下执行，有的线程仍然卡在同步点”的情况。

这背后是 GPU 硬件执行模型的约束。同一个 warp 内的线程会按同一条指令流推进；即使某些线程不满足 `if tidx < M`，它们也不是“完全不参与这段控制流”，而是会跟着 warp 一起经历分支发散与 reconvergence，只是在自己不活跃的那一支上不真正执行对应的访存或算术操作。

因此，不能把它想成 CPU 上“不同线程各跑各的 if/else”。对 warp 来说，分支两侧通常是按硬件规则串行化处理的；而 `cute.arch.sync_threads()` 对应的是 block 级 barrier，它要求所有未退出线程都从**同一个程序点**到达同步位置。

---

## 小结

围绕这组 copy 例子，可以把 CuTe DSL 里的层次关系压缩成五句话：

- 标量索引可以直接赋值，因为右侧已经是线程私有值；slice 索引不行，因为它还是 memory-backed tensor view。
- 要做 slice copy，最基础的方法是 `load()` 成 `TensorSSA` 再 `store()`；`autovec_copy` 和 `CopyUniversalOp` 则是在这个基础上进一步抽象。
- `cp.async` 的 atom 粒度必须和 tile 切分对齐；对 `128-bit`、`Float16` 来说，最自然的基本块就是 `8` 个元素。
- `TiledCopy` 把“每个线程负责哪段源 tensor、写到哪段目标 tensor”封装成 `get_slice + partition_S/D`，这是 `sgemm.py` 里 GMem -> SMem copy 的核心。
- `ldmatrix` 是 Ampere Tensor Core 路线的 shared-to-register 工具；读 `sgemm.py` 时更重要的是 `autovec_copy`，因为这个 SGEMM 示例走的是 FP32 SIMT FMA。

如果继续往下看 `sgemm.py` 的 mainloop，可以把这一篇里的 copy 路径直接套上去：先用 `TiledCopy` 把 A/B 的 K tile 异步搬进 shared memory，再用 `autovec_copy` 把当前 K block 搬进寄存器，最后交给前一篇讲过的 `TiledMMA` 做计算。

---

## 参考链接

[1] [NVIDIA CUTLASS Python DSL API: `cutlass.cute.TensorSSA`](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.TensorSSA)

[2] [CuTe DSL TensorSSA tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/tensorssa.ipynb)

[3] [NVIDIA CUTLASS Python DSL API: `cutlass.cute.autovec_copy`](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.autovec_copy)

[4] [NVIDIA CUTLASS Python DSL API: `cutlass.cute.nvgpu.CopyUniversalOp`](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_common.html#cutlass.cute.nvgpu.CopyUniversalOp)

[5] [NVIDIA CUTLASS Python DSL API: `cutlass.cute.nvgpu.cpasync.CopyG2SOp`](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyG2SOp)

[6] [PTX: asynchronous non-bulk copy instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy)

[7] [PTX: warp-level matrix instruction `ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)

[8] [sgemm的CuTe DSL实现](https://github.com/nvidia/cutlass/blob/main/examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/sgemm.py)