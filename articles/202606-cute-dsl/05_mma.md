# 动手学 CuTeDSL 05：MMA Atom、`TiledMMA`、`atom_layout_mnk` 与 `permutation_mnk`

TODO：gemm 有多种实现， cuda core， tensor core的， tensor core也有迭代不同版本的实现，warp level的， warp group level的，tensorcore 5th generation等等。本文重点介绍 warp level mma 和 cuda core。

CuTe 里和 Tensor Core GEMM 最相关的一组对象，是 **MMA Atom** 与它向上组成的 **`TiledMMA`**。

- **MMA Atom** 对应一条底层 warp-level MMA 指令的寄存器分工；
- **`TiledMMA`** 则是在这个 atom 之上，继续沿 $M/N/K$ 方向做复制、重排与扩展，得到一个更大的逻辑计算单元。

本文围绕 `SM80` 上常见的 `16x8x8` MMA 例子，说明：

1. `display_tiled_mma` 展示的到底是什么；
2. `thread_mma.get_slice()` 如何把一个 tile 切成线程级任务；
3. `atom_layout_mnk` 如何在 $M/N/K$ 三个方向复制 MMA atom；
4. `permutation_mnk` 如何在不改变单次 MMA 语义的前提下，重排更大 tile 内的逻辑坐标。

参考资料：

- [CuTe MMA atoms](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html)
- [PTX warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma)
- [cute-viz MMA example](https://github.com/NTT123/cute-viz/blob/main/examples/mma_atom_example.py)

本文示例默认使用：

```python
import cutlass
from cutlass import cute, Float16, Float32
from cute_viz import display_tiled_mma, display_tv_layout

from cutlass.cute.runtime import from_dlpack
import torch
```

---

## 从一个 `16x8x8` MMA Atom 开始

先看最基本的 warp-level MMA：输入矩阵 $A$ 的形状是 $(M,K)$，输入矩阵 $B$ 的形状是 $(N,K)$，输出/累加矩阵 $C$ 的形状是 $(M,N)$。对于 `SM80` 上的 `F16/BF16 -> F32` MMA atom，一个常见形状是：

$$
(M,N,K) = (16,8,8)

$$

也就是说，一次 atom 逻辑上完成一个 $16 \times 8$ 输出块的累加，归约维度长度为 $8$。

```python
@cute.jit
def mma_atom_demo():
    tile_mnk = (16, 8, 8)
    m, n, k = tile_mnk

    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, tile_mnk)
    tiled_mma = cute.make_tiled_mma(mma_atom)

    print(f"tile_mnk = {tile_mnk}")
    print(f"tiled_mma = {tiled_mma}")

    display_tiled_mma(tiled_mma, tile_mnk)

    print("A TV layout:")
    display_tv_layout(tiled_mma.tv_layout_A, (m, k))
    print("B TV layout:")
    display_tv_layout(tiled_mma.tv_layout_B, (n, k))
    print("C TV layout:")
    display_tv_layout(tiled_mma.tv_layout_C, (m, n))
```

这里有一个非常实用的理解方式：

- `display_tiled_mma(tiled_mma, tile_mnk)` 展示的是整个 MMA 的输入输出分工图；
- `tiled_mma.tv_layout_A`、`tv_layout_B`、`tv_layout_C` 则分别给出 A/B/C 三个矩阵各自的 **thread-value layout**。

因此，可以把 `display_tiled_mma` 看成是把三张 TV layout 图放到同一个 MMA 语义框架里一起看：

- 左边是 A 的 $(M,K)$ 分工；
- 上边是 B 的 $(N,K)$ 分工；
- 右下是 C 的 $(M,N)$ 分工。

对于这个 `16x8x8` atom，warp 中 32 个线程并不是“每个线程负责一整行或一整列”，而是每个线程持有若干寄存器槽位。可视化图里常见的 `T0` 到 `T31` 表示线程编号，`V0`、`V1`、`V2`、`V3` 表示该线程内部不同的 value/register 槽位。这是硬件上 warp-level MMA 的要求，可以看官方文档中 [Matrix Fragments for mma.m16n8k8](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688) 一节的图示说明来和下图对应。

这张基础 `16x8x8` MMA atom 图把 A、B、C 三部分的 thread-value 分工放在同一个视图里展示。

![The base `16x8x8` MMA atom visualization shows the thread-value ownership of A, B, and C in one unified view.](img/05_mma_atom_16x8x8.svg)

---

## `thread_mma.get_slice()`：把 tile 切成线程级工作

`TiledMMA` 在实际使用时，通常不是直接“整体访问 A/B/C”，而是先取出某个线程对应的切片，再让这个线程只访问自己负责的那部分 fragment。

下面用 A 矩阵演示这个过程。为了让线程切片更明显，这里构造一个比单次 MMA 更大的 A tile：

$$
A \in \mathbb{R}^{(16 \cdot 2) \times (8 \cdot 3)} = \mathbb{R}^{32 \times 24}

$$

```python
tile_mnk = (16, 8, 8)
m, n, k = tile_mnk
mma_m = 2
mma_k = 3

@cute.jit
def thread_mma_demo(A: cute.Tensor):
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, tile_mnk)
    tiled_mma = cute.make_tiled_mma(mma_atom)

    thread_idx = 0
    thread_mma = tiled_mma.get_slice(thread_idx)

    cute.printf("A = ")
    cute.print_tensor(A)

    # (BLK_M, BLK_K) -> (MMA, MMA_M, MMA_K)
    tAgA = thread_mma.partition_A(A)
    print(f"tAgA: {tAgA}")

    cute.printf("tAgA[None, 0, 0] = ")
    cute.print_tensor(tAgA[None, 0, 0])

    cute.printf("tAgA = ")
    cute.print_tensor(tAgA)


A = torch.arange(m * mma_m * k * mma_k).reshape(m * mma_m, k * mma_k)
thread_mma_demo(from_dlpack(A))
```

这段代码里最关键的是：

```python
thread_mma = tiled_mma.get_slice(thread_idx)
tAgA = thread_mma.partition_A(A)
```

`partition_A` 的结果可理解为把原始的 $(\text{BLK\_M}, \text{BLK\_K})$ tile，改写成：

$$
(\text{MMA}, \text{MMA\_M}, \text{MMA\_K})

$$

其中：

- `MMA` 表示单个线程在“一次 MMA atom”内持有的寄存器片段；
- `MMA_M` 表示这个更大 tile 在 $M$ 方向上需要分几块；
- `MMA_K` 表示这个更大 tile 在 $K$ 方向上需要分几块。

也就是说，`thread_mma` 的职责就是把“一个大 tile 的线程工作”拆成“这个线程需要参与哪些 MMA、小块内拿哪些元素”。这个操作就是前面提到的layout除法的封装，具体来说就是 `tiled_divide`。

还有一点需要提醒，上面这段代码是用 `@cute.jit`装饰，而不是用 `@cute.kernel`，所以它运行在CPU上，而不是GPU上。这里跟实际使用的情况不同，只是在CPU上用 `thread_idx=0`来对Tensor进行`cute`运算，展示Tensor的变换。

---

## `atom_layout_mnk`：沿 $M/N/K$ 复制 MMA atom

单个 MMA atom 只覆盖固定大小的 $(16,8,8)$。如果希望一个更大的逻辑 tile 由多个 atom 共同组成，可以通过 `atom_layout_mnk` 指定在 $M/N/K$ 三个方向各复制多少次。

例如下面这个例子，把 atom 在 $M$ 和 $N$ 上各复制 2 次：

$$
\text{atom\_layout\_mnk}=(2,2,1)

$$

于是总 tile 变成：

$$
(M,N,K) = (16 \cdot 2,\ 8 \cdot 2,\ 8 \cdot 1) = (32,16,8)

$$

```python
mma_mnk = (16, 8, 8)
mma_m, mma_n, mma_k = mma_mnk
atom_m = 2
atom_n = 2
atom_k = 1

m = mma_m * atom_m
n = mma_n * atom_n
k = mma_k * atom_k
tile_mnk = (m, n, k)

@cute.jit
def atom_layout_demo():
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, mma_mnk)
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(atom_m, atom_n, atom_k),
    )

    display_tiled_mma(tiled_mma, tile_mnk)
```

直观上可以把它理解为：原来只有一个 `16x8x8` 的 MMA atom，现在变成了一个由多个 atom 拼出来的更大逻辑 MMA。

这张 `atom_layout_mnk=(2,2,1)` 图展示了 4 个 MMA atom 如何拼成一个更大的 `32x16x8` 逻辑 MMA。

![The `atom_layout_mnk=(2,2,1)` visualization shows four MMA atoms tiled into a larger `32x16x8` logical MMA.](img/05_atom_layout_2x2x1.svg)

### 为什么通常不建议把 `atom_k` 设置大于1？

如果把 `atom_k` 也扩成 2，例如：

$$
\text{atom\_layout\_mnk}=(2,2,2)

$$

那么多个 atom 会在同一个输出 C tile 上，沿着同一组 $(M,N)$ 坐标同时做不同 K 分块的累加。逻辑上这当然仍然是在做 GEMM 的 K 维归约，但它也意味着：

- 多个并行 MMA 会共同更新同一片 C 结果；
- 需要更仔细地安排寄存器累加与后续同步；
- 设计上通常不如只在 $M/N$ 上扩展那样直接。

因此，实际模板里更常见的是：

- 在 $M$ 上扩 atom；
- 在 $N$ 上扩 atom；
- `atom_k = 1` 保持不变，让atom迭代多次来完成$K$维度上累加 ~~，这个迭代是通过后面的`permutation_mnk`来实现的~~

下面这个例子能直观看到 $K$ 方向也扩张后的逻辑形状：

```python
mma_mnk = (16, 8, 8)
atom_m = 2
atom_n = 2
atom_k = 2

tile_mnk = (
    mma_mnk[0] * atom_m,
    mma_mnk[1] * atom_n,
    mma_mnk[2] * atom_k,
)

@cute.jit
def atom_layout_k_demo():
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, mma_mnk)
    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(atom_m, atom_n, atom_k),
    )

    display_tiled_mma(tiled_mma, tile_mnk)
```

这张 `atom_layout_mnk=(2,2,2)` 图把沿 K 方向继续扩展之后的 MMA 组织方式直观画了出来，可以看到 A 矩阵和 B 矩阵在 K 方向上有了两倍的线程参与（也就是会使用 2倍的mma计算单元）。但是实际很少这样使用。

![The `atom_layout_mnk=(2,2,2)` visualization makes the extra K-direction expansion visible inside the larger MMA tile.](img/05_atom_layout_2x2x2.svg)

---

## `permutation_mnk`：重排更大 tile 的逻辑坐标

仅靠 `atom_layout_mnk`，我们得到的是“把多个 atom 直接并排摆起来”的大 tile。但有时这还不够，因为我们还希望：

- 不增加线程数，而是通过增加每个线程处理的元素个数来扩展更大的tile；
- 同一线程在寄存器里的 value 排列更规整；
- 某个维度上的访问尽量连续；
- 更方便和 shared memory / global memory 的布局配合。

这时就会用到 `permutation_mnk`。

一个非常有用的理解是：`permutation_mnk` 可以看成分别作用在 $M/N/K$ 三个 mode 上的 **tiler/layout**。它先重排这些逻辑坐标，再把 TV layout 映射应用上去。

最简单的情况，是把 `permutation_mnk` 直接写成 tile 本身：

```python
mma_mnk = (16, 8, 8)
tile_mnk = (16, 16, 8)

@cute.jit
def mma_permutation_identity_demo():
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, mma_mnk)

    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(1, 1, 1),
        permutation_mnk=tile_mnk,
    )

    display_tiled_mma(tiled_mma, tile_mnk)
```

当 $N$ 从 8 扩成 16 时，线程数不会像前面 `atom_layout_mnk` 的例子中变多，而是每个线程会持有更多 value，但如果只做“直接扩展”，这些 value 在逻辑坐标上通常只是简单重复原来的 pattern，不一定连续，也不一定是最适合后续访存的顺序。

这张直接扩展到 `16x16x8` 的图说明：虽然整体 tile 变大了，但每个线程拿到的 value 仍然保持未经 permutation 的原始排布方式。

![The direct `16x16x8` expansion keeps the larger MMA shape simple, but the per-thread values are still arranged in the unpermuted pattern.](img/05_permutation_identity_16x16x8.svg)

---

## 在 $N$ 方向做规则化重排

`permutation_mnk` 更典型的用途，是只对某一个 mode 做自定义重排。例如下面只重排 $N$ mode：

```python
mma_mnk = (16, 8, 8)
m, n, k = 16, 16, 8

@cute.jit
def mma_permutation_n_demo():
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, mma_mnk)

    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(1, 1, 1),
        permutation_mnk=(
            m,
            cute.make_layout((2, 4, 2), stride=(1, 4, 2)),
            k,
        ),
    )

    display_tiled_mma(tiled_mma, (m, n, k))
```

这个 `cute.make_layout((2,4,2), stride=(1,4,2))` 只作用在 $N$ 维上。它不是改变单次 `16x8x8` MMA atom 的内部寄存器规则，而是把“多次 MMA 组合起来之后”的逻辑 $N$ 坐标重新排列。

这样做的结果通常是：

- 相同线程在 $N$ 方向负责的元素更连续；
- B/C 的逻辑布局更容易和共享内存或寄存器布局对齐；
- 多次 MMA 的结果在更大的 tile 中呈现“交错但规整”的排布。

这也是很多 GEMM kernel 里需要的效果，因为访存连续性和寄存器布局是否顺手，都会直接影响后续 copy / store 的设计。

这张 N 方向 scatter permutation 图展示了两个 `16x8x8` MMA 子块如何在 N 维上交错重排，从而让每个线程看到更规整的布局。

![The N-mode scatter permutation interleaves the two `16x8x8` MMA images so that each thread sees a more regular layout along N.](img/05_permutation_n_scatter_16x16x8.svg)

---

## `atom_layout_mnk` 与 `permutation_mnk` 组合使用

真实 kernel 里，常常会先用 `atom_layout_mnk` 扩大参与计算的线程数，再用 `permutation_mnk` 调整输出 tile 内部的逻辑排布，以及让每个线程处理更多的数据。

例如先用 `atom_layout_mnk` 在 $M/N$ 上各扩 2 倍：

$$
(16,8,8) \rightarrow (32,16,8)

$$

然后再用 `permutation_mnk` 把数据逻辑上扩到更大的 $N$ 维 tile：

$$
(32,16,8) \rightarrow (32,32,8)

$$

```python
mma_mnk = (16, 8, 8)
atom_m = 2
atom_n = 2
atom_k = 1

m = 16 * atom_m
n = 8 * atom_n * 2
k = 8 * atom_k

@cute.jit
def mma_atom_and_permutation_demo():
    mma_atom = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, mma_mnk)

    tiled_mma = cute.make_tiled_mma(
        mma_atom,
        atom_layout_mnk=(atom_m, atom_n, atom_k),
        permutation_mnk=(
            m,
            cute.make_layout((2, 4, 4), stride=(1, 8, 2)),
            k,
        ),
    )

    print(f"tiled_mma = {tiled_mma}")
    display_tiled_mma(tiled_mma, (m, n, k))
```

这个模式的价值在于：

- `atom_layout_mnk` 负责“需要多少个 MMA atom 一起工作”；
- `permutation_mnk` 负责“这些 atom 的结果在更大 tile 中怎么排更合适”。

两者职责不同，但配合起来正好构成了从“硬件原子指令”到“工程上可用的大 tile MMA”之间的桥梁。

这张 `atom_layout_mnk` 与 permutation 组合图展示了一个更大的 `32x32x8` MMA tile 如何同时完成线程扩展和逻辑重排。

![The combined `atom_layout_mnk` plus permutation visualization shows how a larger `32x32x8` MMA tile can be both expanded and reordered at the same time.](img/05_atom_permutation_32x32x8.svg)

---

## 小结

- `MmaF16BF16Op(Float16, Float32, (16,8,8))` 描述的是一条底层 warp-level MMA atom。
- `display_tiled_mma` 本质上是在同一张图里同时展示 A/B/C 三个 TV layout。
- `thread_mma.get_slice(thread_idx)` 可以把一个大 tile 切成线程级 fragment，再用 `partition_A/B/C` 取出当前线程负责的数据。
- `atom_layout_mnk` 用来沿 $M/N/K$ 复制 MMA atom，实际中最常见的是扩 $M/N$，而不是扩 $K$。
- `permutation_mnk` 可以让每个线程处理更多的数据，从而用来重排更大 tile 内的逻辑坐标，尤其适合把某个 mode 的访问组织得更连续、更利于后续访存和寄存器布局设计。

如果只记一个心智模型，可以记成：

$$
\text{MMA Atom} \xrightarrow{\text{atom\_layout\_mnk}} \text{更大的 MMA 线程组织}
\xrightarrow{\text{permutation\_mnk}} \text{更适合工程实现的逻辑排布}

$$

## 附录： `permutation_mnk` 会不会和 MMA 对寄存器的要求冲突？

不会。这一点最容易让人困惑，因为 `permutation_mnk` 看上去像是在“重排寄存器”，而 MMA 指令本身又确实要求 fragment 的寄存器分布满足固定格式。关键在于，要区分两种不同层次的排布：

- **单次 MMA atom 内部的寄存器排布**；
- **多次 MMA atom 组合成更大 tile 之后的逻辑坐标排布**。

MMA 指令真正要求固定的是前者，而 `permutation_mnk` 影响的主要是后者。

### 先看单次 `16x8x8` MMA 对 C fragment 的要求

以 `C` 为例，单次 `16x8x8` MMA 的输出块大小是：

$$
16 \times 8 = 128

$$

warp 一共有 32 个线程，因此平均到每个线程上，就是：

$$
128 / 32 = 4

$$

也就是说，在一次 `16x8x8` MMA 中，每个线程都会写入 `fragC` 中固定的 4 个元素。在可视化图里，它们通常就是这个线程对应的 `V0`、`V1`、`V2`、`V3`。

这 4 个 value 的相对位置，是由底层 MMA atom 决定的。只要我们仍然是在执行这条 `16x8x8` 的 MMA 指令，那么这 4 个 value 在“这一次 MMA 对应的 fragment”里的内部关系就不能乱。

### 再看把 tile 从 `16x8x8` 扩到 `16x16x8`

当我们把输出 tile 从：

$$
(M,N,K) = (16,8,8)

$$

扩成：

$$
(M,N,K) = (16,16,8)

$$

本质上并不是把“单次 MMA 指令本身”改成了另一种寄存器规则，而是让同一个 warp 逻辑上要完成两次 `16x8x8` 子块计算，合起来覆盖一个更大的 `16x16` 输出块。

于是可以把这个 warp 的工作理解成：

- 第一次 MMA：每个线程产出自己的 `V0`、`V1`、`V2`、`V3`；
- 第二次 MMA：每个线程再产出自己的 `V4`、`V5`、`V6`、`V7`。

这里 `V4` 到 `V7` 可以理解为“第二次 MMA 贡献的那一组 value”。它们并不是把第一次 MMA 的 4 个寄存器槽位内部打乱了，而只是让 warp 在更大的逻辑 tile 中又多写了一组结果。

### `permutation_mnk` 改变的是绝对坐标，不是单次 MMA 的内部关系

如果不做特殊 permutation，而只是把 `N` 方向直接扩展到 16，那么从图上会看到：

- 同一个线程确实拿到了更多 value；
- 这些新增 value 往往只是原始 pattern 的重复；
- 它们在更大 tile 里的坐标未必连续。

如果加入前一节的 `permutation_mnk`，那么变化发生在“这些 value 被放到更大 C tile 的什么位置”这件事上。

换句话说：

- `V0`、`V1`、`V2`、`V3` 这一组内部的相对关系没有变；
- `V4`、`V5`、`V6`、`V7` 这一组内部的相对关系也没有变；
- 变化的是这两组小块在整个 `16x16` 逻辑图中的**拼接顺序**与**绝对坐标**。

因此，`permutation_mnk` 更像是在说：

> 先让 warp 分别完成两次合法的 `16x8x8` MMA，再把这两次结果映射到更大的逻辑输出 tile 中更合适的位置。

这就解释了为什么它不会破坏 MMA 指令本身对 fragment 的要求。

### 一个更直观的理解

可以把单次 MMA 看成一个“不能拆散内部结构的小积木块”：

- 每个线程在这个小积木块里该拿哪 4 个元素，是硬件/atom 决定的；
- `permutation_mnk` 不能改变这个小积木块的内部连接关系；
- 它能做的是把多个小积木块在更大的平面上重新摆放。

所以，`permutation_mnk` 并不是在说“把一次 MMA 的寄存器随意洗牌”，而是在说“保持每次 MMA 自身合法，再把多次 MMA 的结果交错排布”。

### 为什么这种重排反而有价值？

因为从 kernel 设计角度看，我们最终关心的往往不是“单次 MMA 结束后寄存器长什么样”本身，而是：

- 后续要不要把这些结果写回 shared memory；
- 写回 global memory 时是否连续；
- 同一个线程负责的元素是否在某个维度上更规整。

这正是 `permutation_mnk` 的价值所在。它不去碰单次 MMA atom 的底层约束，而是在更高一层重新组织多个 MMA 子块的逻辑位置，从而把寄存器结果排成一个对后续 copy / store 更友好的大 tile。

---
