# 动手学 CuTeDSL 05：MMA Atom、`TiledMMA`、`atom_layout_mnk` 与 `permutation_mnk`

CuTe 里和 GEMM 计算最相关的一组对象，是 **MMA Atom** 与它向上组成的 **`TiledMMA`**。这里的 MMA 不只指 Tensor Core 指令，也可以表示 CUDA core 上的标量 FMA。CuTe 文档把 MMA Atom 放在多个硬件层级上理解：


| 层级          | 典型硬件/计算能力                     | 典型指令或 CuTe 接口                              | 直观含义                                                 |
| --------------- | --------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| single thread | CUDA core                             | PTX`fma.f32`，CuTe `MmaUniversalOp(Float32)`      | 一个线程独立做标量乘加                                   |
| quadpair      | Volta V100，SM70 / CC 7.0             | Volta Tensor Core HMMA 的 quadpair 建模           | warp 内少量线程协作完成一个 atom                         |
| warp          | Ampere A100，SM80 / CC 8.0            | PTX`mma.sync`，CuTe `MmaF16BF16Op(..., (16,8,8))` | 一个 warp 的 32 个线程共同完成一个矩阵乘累加             |
| warpgroup     | Hopper H100，SM90 / CC 9.0            | PTX`wgmma.mma_async`                              | 多个 warp 组成 warpgroup，共同驱动更大的 Tensor Core MMA |
| tcgen05       | Blackwell GB200/B200，SM100 / CC 10.0 | PTX TensorCore 5th Generation family instructions | 第五代 Tensor Core 的异步 MMA 执行模型                   |

这些层级的编程接口和同步模型不同，但在 CuTe 里都可以放进“某个硬件层级上的 MMA Atom，再向上组合成 `TiledMMA`”这条主线里理解。

本文重点介绍两类最容易建立编程模型的情况，并且先讲 Tensor Core，再讲 CUDA core：

- **warp-level Tensor Core MMA**：它更接近通用的 MMA Atom 抽象，一个 atom 对应一条矩阵乘累加指令，一个 warp 按固定 fragment 规则协作执行；
- **CUDA core MMA**：它是大小接近 `1x1x1` 的 single-thread MMA 特例，每个线程用 FP32 FMA 完成自己负责的累加。

贯穿这两类实现的核心对象是：

- **MMA Atom** 对应某个硬件层级上的最小乘加计算单元，以及它要求的线程/寄存器分工；
- **`TiledMMA`** 则是在这个 atom 之上，继续沿 $M/N/K$ 方向做复制、重排与扩展，得到一个更大的逻辑计算单元。

前半部分先围绕 `SM80` 上常见的 `16x8x8` warp-level MMA 例子，说明：

1. `display_tiled_mma` 展示的到底是什么；
2. `thread_mma.get_slice()` 如何把一个 tile 切成线程级任务；
3. `atom_layout_mnk` 如何在 $M/N/K$ 三个方向复制 MMA atom；
4. `permutation_mnk` 如何在不改变单次 MMA 语义的前提下，重排更大 tile 内的逻辑坐标。

最后再回到 CUDA core SGEMM，说明同一套 `TiledMMA` 抽象如何落到每个线程自己的 FP32 FMA 上。

参考资料：

- [CuTe MMA atoms](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html)
- [PTX warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma)
- [PTX warpgroup-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions)
- [PTX TensorCore 5th Generation family instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-family-instructions)
- [PTX floating point fma instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma)
- [CUTLASS Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
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

这里 `cute.make_tiled_mma(mma_atom)` 没有额外指定 `atom_layout_mnk` 或 `permutation_mnk`，所以 `TiledMMA` 只包含一个原始 MMA atom，逻辑 tile 尺寸仍然是 `16x8x8`。后面几节会再看如何通过 `atom_layout_mnk` 和 `permutation_mnk` 把这个基础 atom 扩成更大的计算图样。

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

## 把 tile 切成线程级工作

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

- `MMA` 表示单个线程在“一次 Tiled MMA”内持有的寄存器片段；
- `MMA_M` 表示这个更大 tile 在 $M$ 方向上需要分几块；
- `MMA_K` 表示这个更大 tile 在 $K$ 方向上需要分几块。

也就是说，`thread_mma` 的职责就是把“一个大 tile 的线程工作”拆成“这个线程需要参与哪些 MMA、小块内拿哪些元素”。这个操作就是前面提到的layout除法的封装，具体来说就是 `tiled_divide`。

还有一点需要提醒，上面这段代码是用 `@cute.jit`装饰，而不是用 `@cute.kernel`，所以它运行在CPU上，而不是GPU上。这里跟实际使用的情况不同，它只是在CPU上用 `thread_idx=0`来对Tensor进行`cute`运算，展示Tensor的变换。


把同一个例子扩展到 A/B/C 三个张量，可以得到下面的形状关系。这里使用设置：

```text
MMA atom = 16x8x8
A = (32,32)
B = (24,32)
C = (32,24)
```

实际partition 结果为：


| partition        | 输入形状                   | 输出形状      | 抽象含义              |
| ------------------ | ---------------------------- | --------------- | ----------------------- |
| `partition_A(A)` | `(BLK_M, BLK_K) = (32,32)` | `((2,2),2,4)` | `(MMA, MMA_M, MMA_K)` |
| `partition_B(B)` | `(BLK_N, BLK_K) = (24,32)` | `(2,3,4)`     | `(MMA, MMA_N, MMA_K)` |
| `partition_C(C)` | `(BLK_M, BLK_N) = (32,24)` | `((2,2),2,3)` | `(MMA, MMA_M, MMA_N)` |

这里的 `MMA` 维度来自单次 `16x8x8` warp-level MMA atom 内部的线程 fragment：A/C 中当前线程持有 `((2,2))`，B 中当前线程持有 `2`。外层的 `2/3/4` 才是大 tile 相对单次 atom 在 $M/N/K$ 方向上的重复次数。

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

## 组合 `TiledMMA` 的 `partition_A/B/C`

上面的组合示例中，`TiledMMA` 的基础计算图样是：

```text
tiled_mma tile_mnk = (32,32,8)
```

为了看清楚外层重复维度，使用了比基础图样在 $M/N/K$ 上都大一倍的 tile：

```text
A = (64,16)
B = (64,16)
C = (64,64)
```

实际 某个 thread 的 partition 结果为：


| partition        | 输入形状                   | 输出形状          | 抽象含义              |
| ------------------ | ---------------------------- | ------------------- | ----------------------- |
| `partition_A(A)` | `(BLK_M, BLK_K) = (64,16)` | `((2,2),2,2)`     | `(MMA, MMA_M, MMA_K)` |
| `partition_B(B)` | `(BLK_N, BLK_K) = (64,16)` | `(2,(2,2),2)`     | `(MMA, MMA_N, MMA_K)` |
| `partition_C(C)` | `(BLK_M, BLK_N) = (64,64)` | `((2,2),2,(2,2))` | `(MMA, MMA_M, MMA_N)` |

这个例子里需要区分三层：

- `MMA`：来自单次 `16x8x8` warp-level atom 的线程 fragment，例如 A/C 中当前线程持有 `((2,2))`；
- `atom_layout_mnk=(2,2,1)`：把基础 atom 在 $M/N$ 上扩成多个 warp-level atom，形成 `32x16x8` 的中间图样；
- `permutation_mnk`：继续把 $N$ 方向组织成 `32`，形成最终的 `32x32x8` 基础 `TiledMMA` 图样。

当输入 tile 扩到 `64x64x16` 时，`partition_A/B/C` 又会在这个基础图样外面引入额外重复：$M$ 方向重复 2 次，$K$ 方向重复 2 次，$N$ 方向也重复 2 次。因此可以看到 A 的输出是 `((2,2),2,2)`，C 的输出是 `((2,2),2,(2,2))`。

---

## CUDA Core MMA：用 FMA 组织 SIMT GEMM

前面几节讲的是 Tensor Core 路线：一个 MMA atom 对应类似 `mma.sync` 这样的 warp-level 指令，32 个线程共同完成一个固定形状的矩阵乘累加。

在Tensor Core硬件之前，是用CUDA core 来完成GEMM的。它的编程模型明显不同：底层不是 warp 协作的一条矩阵指令，而是每个线程在自己的寄存器里反复执行标量 FMA（乘加运算）。

PTX 里的 `fma.f32` 语义可以写成：

$$
d = a \times b + c

$$

它会把乘法和加法作为 fused multiply-add 执行，中间乘积和加法不会在二者之间先舍入一次。也就是说，对 FP32 SGEMM 来说，最底层的计算单元可以理解为：

```python
acc = fma(a, b, acc)
```

这仍然可以被 CuTe 表达成 MMA，只是这个 atom 的硬件层级是 **single thread**，而不是一个 warp。

### `MmaUniversalOp(Float32)`：单线程 FMA atom

典型的单精度矩阵乘法`sgemm` 中使用的 CUDA core MMA 的核心是：

```python
op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)

tiled_mma = cute.make_tiled_mma(
    op,
    atom_layout_mnk,
    permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
)
```

这里的 `MmaUniversalOp(Float32)` 可以理解为一个通用的 FP32 FMA atom。它不像 `MmaF16BF16Op(Float16, Float32, (16, 8, 8))` 那样描述一条 `16x8x8` Tensor Core 指令，而是描述“每个线程自己做一个标量累加”。

因此，Tensor Core MMA 和 CUDA core MMA 的差别可以这样对比：


| 路线        | 底层 atom                     | 线程协作方式   | 一次 atom 的直观含义              |
| ------------- | ------------------------------- | ---------------- | ----------------------------------- |
| Tensor Core | `MmaF16BF16Op(..., (16,8,8))` | 一个 warp 协作 | 32 个线程共同完成一个`16x8x8` MMA |
| CUDA core   | `MmaUniversalOp(Float32)`     | 每个线程独立   | 一个线程执行若干 FP32 FMA         |

### `atom_layout_mnk`：把线程排成一个 C tile

既然单个 FMA atom 只属于一个线程，那么更大的输出 tile 就需要靠很多线程并排组成。这里继续沿用前文的命名，把这层线程组织称为 `atom_layout_mnk`：

```python
cta_tiler = (128, 128, 8)
num_threads = 256

atom_layout_mnk = cute.make_layout(
    (num_threads // 16, 16, 1), stride=(16, 1, 0)
)
```

当 `num_threads = 256` 时，`atom_layout_mnk.shape = (16, 16, 1)`，也就是把 256 个线程组织成一个 $16 \times 16$ 的线程网格。因为 `MmaUniversalOp` 的 atom 本身可以近似看成 `1x1x1`，所以这里的 `atom_layout_mnk` 同时也是 C tile 上的线程分布方式。

如果 C 是列主序，脚本会把线程网格换一个方向：

```python
atom_layout_mnk = cute.make_layout(
    (16, num_threads // 16, 1), stride=(1, 16, 0)
)
```

这体现了 CUDA core MMA 的一个关键点：既然没有固定的 Tensor Core fragment 规则，线程和输出元素的对应关系就主要由 layout 决定。C 的主序不同，`partition_C(gC)` 生成的线程私有 C fragment 也不同；如果线程布局顺着 C 的连续维度展开，那么相邻线程写回 global memory 时更容易形成连续地址段。对行主序 C 来说，连续维度是 $N$；对列主序 C 来说，连续维度是 $M$。这里切换 `atom_layout_mnk` 的方向，本质上是在让 epilogue 的 store 尽量沿 C 的连续维度组织，减少跨步很大的分散写。

### `permutation_mnk`：让每个线程拿连续的 4 个元素

这里的 `4` 只描述一次 `TiledMMA` 的基础图样：把 $16 \times 16$ 的线程网格在 M/N 方向各扩成 $16 \times 4 = 64$，也就是形成一个 $64 \times 64$ 的线程-元素映射图样。

在这个基础图样中，每个线程负责：

$$
4 \times 4 = 16

$$

个 C 累加元素。也就是说，`permutation_mnk` 在这里的作用不是决定最终 CTA tile 的完整覆盖范围，而是决定这个 $64 \times 64$ 基础图样里“连续数据由哪个线程持有”。如果只把线程平铺到 C tile 上，每个线程负责的元素可能在逻辑上不够连续，不利于从 shared memory 拿 A/B，也不利于寄存器复用。所以要用 `permutation_mnk` 做这个重排：

```python
permutation_tiler_M = cute.make_layout(
    (atom_layout_mnk.shape[0], 4), stride=(4, 1)
)
permutation_tiler_N = cute.make_layout(
    (atom_layout_mnk.shape[1], 4), stride=(4, 1)
)

tiled_mma = cute.make_tiled_mma(
    op,
    atom_layout_mnk,
    permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
)
```

这里的 `4` 可以理解为“让同一个线程在对应方向上拿到一组连续元素”。给一个直观的对比：按 tensor 中连续下标看，未重排时线程编号类似：

```text
0 1 2 ... 15 0 1 2 ... 15 0 1 2 ... 15 ...
```

重排后变成：

```text
0 0 0 0 1 1 1 1 2 2 2 2 ... 15 15 15 15 ...
```

也就是说，`permutation_mnk` 不是改变 FMA 的数学语义，而是改变“连续数据由哪个线程持有”。这样每个线程从 shared memory 到 register 的搬运更容易向量化，寄存器里的数据也更适合做连续的 FMA。

### `partition_A/B/C`：把 CTA tile 切成线程私有 fragment

进入 kernel 后，每个线程先通过 `get_slice(tidx)` 拿到自己的 MMA 视角，再对 A/B/C tile 做分区：

```python
tidx, tidy, tidz = cute.arch.thread_idx()
thr_mma = tiled_mma.get_slice(tidx)

tCsA = thr_mma.partition_A(sA)
tCsB = thr_mma.partition_B(sB)
tCgC = thr_mma.partition_C(gC)
```

这几行和前面 warp-level MMA 的 `thread_mma.get_slice()` 是同一个抽象：先从全局 tile 得到当前线程的视角，再构造寄存器 fragment。区别在于 CUDA core 的 `MMA` 维度是平凡的 `1`，因为 `MmaUniversalOp(Float32)` 是 single-thread `1x1x1` FMA atom。

使用设置：

```text
sA = (128,8,3)
sB = (128,8,3)
gC = (128,128)
```

实际某个 thread 的 partition 结果为：


| partition         | 输入形状                           | 输出形状          | 抽象含义                    |
| ------------------- | ------------------------------------ | ------------------- | ----------------------------- |
| `partition_A(sA)` | `(BLK_M, BLK_K, PIPE) = (128,8,3)` | `(1,(4,2),8,3)`   | `(MMA, MMA_M, MMA_K, PIPE)` |
| `partition_B(sB)` | `(BLK_N, BLK_K, PIPE) = (128,8,3)` | `(1,(4,2),8,3)`   | `(MMA, MMA_N, MMA_K, PIPE)` |
| `partition_C(gC)` | `(BLK_M, BLK_N) = (128,128)`       | `(1,(4,2),(4,2))` | `(MMA, MMA_M, MMA_N)`       |

这里 `(4,2)` 的两层含义是：

- `4`：单个 $64 \times 64$ 基础图样内，同一个线程在该方向上连续持有的 4 个元素；
- `2`：完整 $128 \times 128$ CTA tile 相对 $64 \times 64$ 基础图样，在该方向上还要重复 2 次。

因此每个线程在 C 上总共覆盖：

$$
(4 \times 4) \times (2 \times 2) = 64

$$

个元素，全 CTA 覆盖量为：

$$
256 \text{ threads} \times 64 = 16384 = 128 \times 128

$$

`permutation_mnk` 的 $4 \times 4$ 和 `partition_C` 的 $2 \times 2$ 最终都会变成多次 FMA，但层级不同：前者决定 `TiledMMA` 基础图样内每个线程的寄存器 fragment 和局部展开顺序，后者是 CTA tile 相对基础图样的外层重复 mode。

### 主循环：shared memory pipeline + register pipeline + FMA

`sgemm` 的典型 CTA tile 是：

$$
(M,N,K)=(128,128,8)

$$

线程块每次处理一个 $128 \times 128$ 的 C 子块，并沿 K 方向以 8 为单位推进。主循环里真正做计算的是：

```python
cute.gemm(
    tiled_mma,
    tCrC,
    tCrA[None, None, k_block],
    tCrB[None, None, k_block],
    tCrC,
)
```

对 CUDA core MMA 来说，这里的 `cute.gemm` 会展开成当前线程寄存器 fragment 上的一串 FP32 FMA。一个直观的伪代码是：

```python
for k in k_fragment:
    for m_value in thread_m_values:
        for n_value in thread_n_values:
            acc[m_value, n_value] = fma(a[m_value, k], b[n_value, k], acc[m_value, n_value])
```

为了让这串 FMA 尽量不断粮，脚本同时做了两层 pipeline：

- **shared memory pipeline**：用 `cp.async` 把后续 K tile 从 global memory 提前搬到 shared memory，默认 `num_stages = 3`；
- **register pipeline**：用 `cute.autovec_copy` 把下一个 `k_block` 的 A/B 从 shared memory 提前搬到寄存器，和当前 `k_block` 的 FMA 交错起来。

因此，这个 CUDA core SGEMM 的核心并不是“写一个三重 for 循环”，而是：

1. 先用 `cta_tiler` 固定 CTA 级别的工作块；
2. 用 `atom_layout_mnk` 把线程组织成输出 tile 上的计算网格；
3. 用 `permutation_mnk` 让每个线程拿到更连续、更适合寄存器复用的数据；
4. 用 `partition_A/B/C` 生成线程私有 fragment；
5. 在主循环里用 shared memory pipeline 和 register pipeline 喂满 CUDA core FMA。

这和前面 Tensor Core MMA 的层次是对应的：`TiledMMA` 仍然负责“把大 tile 分给线程并组织寄存器 fragment”，只是最终落到底层时，Tensor Core 路线发出的是 warp-level MMA 指令，CUDA core 路线发出的是每个线程自己的 FMA 指令。

---

## 小结

- `MmaF16BF16Op(Float16, Float32, (16,8,8))` 描述的是一条底层 warp-level MMA atom。
- `display_tiled_mma` 本质上是在同一张图里同时展示 A/B/C 三个 TV layout。
- `thread_mma.get_slice(thread_idx)` 可以把一个大 tile 切成线程级 fragment，再用 `partition_A/B/C` 取出当前线程负责的数据。
- `atom_layout_mnk` 用来沿 $M/N/K$ 复制 MMA atom，实际中最常见的是扩 $M/N$，而不是扩 $K$。
- `permutation_mnk` 可以让每个线程处理更多的数据，从而用来重排更大 tile 内的逻辑坐标，尤其适合把某个 mode 的访问组织得更连续、更利于后续访存和寄存器布局设计。
- `MmaUniversalOp(Float32)` 描述的是 CUDA core / SIMT 路线的 FP32 FMA atom，配合 `atom_layout_mnk`、`permutation_mnk` 和 pipeline 可以组织出完整的 SGEMM。

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
