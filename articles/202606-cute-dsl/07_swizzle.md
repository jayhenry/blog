# CuTe：`Swizzle` 如何重排 Layout 以缓解 Shared Memory Bank Conflict

`Swizzle` 是 CuTe 里非常常见的一类 layout 变换。它不改变元素总数，而是通过对地址位做重排，把原本集中落到同一批 bank 的访问分散开，从而改善 shared memory 的访问冲突。

本文用两个例子来理解它：

1. 先看一个 `8x8` 的基础例子，直观感受 `Swizzle<b, m, s>` 对 layout 的影响；
2. 再看一个更接近实际使用的 shared memory layout，说明 `swizzle_bits` 是怎么根据一次 copy 的位宽推出来的。

本文示例默认使用：

```python
import math
import cutlass
from cutlass import cute
from cute_viz import display_layout, display_swizzle_layout
```

---

## 从一个 `8x8` layout 开始

先构造一个最基础的 `8x8` row-major layout：

```python
@cute.jit
def main():
    base_layout = cute.make_layout((8, 8), stride=(8, 1))

    print("Base layout:")
    print(base_layout)
    display_layout(base_layout)
```

它的打印结果是：

$$
(8,8):(8,1)

$$

这表示逻辑形状是 `(8, 8)`，步长是 `(8, 1)`，也就是最常见的按行连续存储。下面这张图展示了这个基础 layout 的编号方式。

![The base `8x8` layout is a plain row-major mapping before any swizzle is applied.](img/07_swizzle_base_8x8.svg)

---

## `Swizzle<3, 0, 3>`：把列访问打散

在 CuTe 里，swizzle 可以通过 `cute.make_swizzle(b, m, s)` 构造。按照 [2] 的理解，swizzle 的本质也是一个函数，只不过它作用的对象不是逻辑坐标，而是 layout 已经算出来的一维 offset。

如果把普通 layout 写成：

$$
\text{layout}: \text{coord} \mapsto \text{offset}

$$

那么 swizzle 可以写成：

$$
\text{swizzle}: \text{offset} \mapsto \text{offset}'

$$

两者组合之后，得到的就是复合函数：

$$
(\text{swizzle} \circ \text{layout})(\text{coord})
= \text{swizzle}(\text{layout}(\text{coord}))

$$

也就是说，layout 负责把二维或更高维坐标映射成线性地址，而 swizzle 负责把这个线性地址再映射成一个更适合 shared memory 访问的、尽量 bank-conflict-free 的地址 [2]。

这个例子使用：

```python
swizzle = cute.make_swizzle(b=3, m=0, s=3)
```

也就是：

- `b=3`：参与 swizzle 的 bit 组宽度。直观上，可以把它理解成二维重排里“有多少列参与异或重排”；这里对应 `2^b = 8` 个列位置 [2]；
- `m=0`：最底层基础元素的大小。按照 [2] 的说法，可以把一维地址里连续的 `2^m` 个元素看成二维空间里的一个最小单元；这里 `m=0`，因此最小单元大小是 `2^0 = 1`；
- `s=3`：另一组 bit 与当前 bit 组之间的位移距离。直观上，它决定了“行坐标”来自更高多少位；这里可以把它理解成有 `2^s = 8` 个行位置参与重排 [2]。

如果沿着 [2] 中“把一维 offset 重新解释为二维空间”的思路来看，那么 `B`、`M`、`S` 共同描述的是三个层次：

- 连续的 `2^m` 个元素先组成一个最基本的 cell；
- 这些 cell 再沿着列方向排成 `2^b` 列；
- 同时再沿着行方向展开 `2^s` 行；
- swizzle 做的事情，就是在这个二维视角下对行、列相关的 bit 进行异或重排，再映射回线性地址。

把它组合到原 layout 上：

```python
@cute.jit
def main():
    base_layout = cute.make_layout((8, 8), stride=(8, 1))
    swizzle = cute.make_swizzle(b=3, m=0, s=3)
    swizzled_layout = cute.make_composed_layout(swizzle, 0, base_layout)

    print("\nSwizzle:")
    print(swizzle)
    print("\nSwizzled layout:")
    print(swizzled_layout)

    display_swizzle_layout(swizzled_layout)
    display_layout(swizzled_layout)
```

这时打印结果变成：

$$
\text{Swizzle} = S<3,0,3>

$$

$$
\text{Swizzled layout} = S<3,0,3> \circ 0 \circ (8,8):(8,1)

$$

第一张图展示了 swizzle 之后的重排关系。可以把它理解成：原来连续落在同一列上的元素，被重新分散到不同列中。

![The `Swizzle<3,0,3>` view shows how a regular `8x8` layout is remapped across columns.](img/07_swizzle_swizzled_8x8.svg)

从 shared memory 的角度看，这个例子最重要的直觉是：

- 如果把一列近似看成一组 bank 访问目标，那么原本同列聚集的数据更容易产生冲突；
- 经过 swizzle 之后，同一批逻辑相邻元素会被打散到不同列；
- 因此同一次访问更有机会覆盖不同 bank，从而缓解 bank conflict。

---

## 实际使用：根据 copy 位宽选择 `swizzle_bits`

实际写 shared memory layout 时，swizzle 往往不是手工拍脑袋选出来的，而是和一次 copy 的位宽、数据类型、bank 组织方式一起决定。

对 `SM90` 而言，这里先区分两个容易混在一起的“位宽”概念：

- `copy_bits` 表示下游消费者一次从 shared memory 连续取走多少 bit 的数据。这里写成 `128`，表示一次 copy 宽度是 `128 bit = 16 B`。在 CUTLASS / CuTe 的 copy atom 语境里，这个量对应“每次 copy 的向量宽度”；对于 TMA 相关 API，官方文档也把它描述成一次 TMA copy 的 vector length [3]。
- 另外，Hopper 的硬件(`SM90` )的 TMA shared-memory swizzle 提供 `32B`、`64B`、`128B` 三种 swizzle width，而且 swizzle 的基本粒度固定为 `16B` [4]。TMA 有单独的 swizzle 接口，不是本文的 API。

下面这个例子考虑 `Float16` 和 `128-bit` copy：

```python
dtype = cutlass.Float16
copy_bits = 128

@cute.jit
def example():
    m, k = 128, 32
    smem_tiler = (m, k)
    k = 64 if k >= 64 else k  # 32bank * 4Byte / 2Byte = 64 Float16
    swizzle_bits = int(math.log2(k * dtype.width // copy_bits))
    print(f"swizzle_bits before min: {swizzle_bits}")
    swizzle_bits = min(swizzle_bits, 3)
```

这个例子里：

- `dtype.width = 16`，也就是每个 `Float16` 元素占 16 bit；
- `copy_bits = 128`，也就是每次按 `16B` 的向量宽度来分析 shared memory 访问；
- `k = 64 if k >= 64 else k` 的含义是把参与 swizzle 分析的内层宽度限制在 `64` 个 `Float16` 元素以内，也就是

$$
k_{\text{eff}} = \min(k, 64)

$$

之所以是 `64`，是因为 `SM90` 的 shared memory 仍然可以按 `32` 个 bank、每个 bank `4B` 来理解，一整条 bank line 一共是

$$
32 \times 4\text{B} = 128\text{B}

$$

对于 `Float16`，一个元素是 `2B`，所以一整条 `128B` bank line 最多容纳

$$
128\text{B} / 2\text{B} = 64

$$

个元素。也就是说，当内层宽度超过 `64` 个 `Float16` 时，后面的部分已经进入下一段重复的 bank pattern 了，swizzle 的选择只需要看一个 `128B` 周期内的情况即可。

按这个思路，这一节里的位宽选择逻辑可以拆成 4 步：

1. 先确定一个 bank 周期内需要考虑多少个元素。对于 `Float16`，就是 `k_eff = min(k, 64)`。
2. 把这个范围换算成 bit 数：

$$
\text{row\_bits} = k_{\text{eff}} \times \text{dtype.width}

$$

3. 再用 `copy_bits` 去看“这一行里会出现多少个独立的 copy 向量”：

$$
\text{num\_vec} = \frac{\text{row\_bits}}{\text{copy\_bits}}

$$

4. 最后取对数，得到需要多少个 swizzle bit 才能把这些向量打散开：

$$
\text{swizzle\_bits} = \log_2(\text{num\_vec})

$$

代入这段代码的具体数值：

- 原始 `k = 32`，因此 `k_eff = min(32, 64) = 32`；
- `row_bits = 32 * 16 = 512 bit = 64B`；
- `num_vec = 512 / 128 = 4`，也就是这一行里有 4 个 `16B` copy 向量；
- `swizzle_bits = log2(4) = 2`。

因此在当前例子里，代码会打印：

- `swizzle_bits before min: 2`

最后还有一句：

```python
swizzle_bits = min(swizzle_bits, 3)
```

这一句是在做上界裁剪。对 `SM90` 来说，Hopper 文档里的最大 swizzle width 是 `128B`，而 swizzle 的基本粒度是 `16B` （即`copy_bits=128`）[4]，因此最多只需要区分

$$
128\text{B} / 16\text{B} = 8 = 2^3

$$

个粒度块，所以 `swizzle_bits` 的有效上限就是 `3`。如果前面的计算结果更大，也会被截断到 `3`。

因此这里最终得到的是：

$$
\text{swizzle\_bits before min} = 2

$$

---

## 先看未做 swizzle 的 layout atom

接着先构造外层的基础 layout：

```python
@cute.jit
def example():
    ...
    layout_atom_outer = cute.make_layout((8, k), stride=(k, 1))
    print(f"layout_atom_outer: {layout_atom_outer}")
    display_layout(layout_atom_outer)
```

打印结果是：

$$
\text{layout\_atom\_outer} = (8,32):(32,1)

$$

这张图展示了尚未加入 swizzle 时的 `8x32` layout atom。

![The outer `8x32` layout atom is the unswizzled shared-memory tile.](img/07_swizzle_layout_atom_outer_8x32.svg)

---

## 把 `Swizzle<2, 3, 3>` 组合进 layout atom

有了 `swizzle_bits` 之后，就可以把 swizzle 组合到 layout atom 上：

```python
@cute.jit
def example():
    ...
    layout_atom = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 3, 3),
        0,
        layout_atom_outer,
    )
    print(f"After swizzle, layout_atom: {layout_atom}")
    display_layout(layout_atom)
```

这里最值得注意的是，`make_composed_layout` 不是“重新定义一个新的 shape”，而是把前一节得到的 swizzle 函数，复合到已有的 row-major layout 上 [2][3]。也就是说：

$$
\text{layout\_atom} = \text{swizzle} \circ \text{layout\_atom\_outer}
$$

对于逻辑坐标 `(r, c)`，原始的 `layout_atom_outer = (8,32):(32,1)` 先给出线性地址：

$$
\text{offset}_{\text{outer}} = 32r + c
$$

然后 `S<2,3,3>` 再把这个 offset 映射成新的 shared-memory 地址。逻辑上的 `(8,32)` 形状完全没有变，变化的是“同一个逻辑坐标最终落到哪一个物理 offset 上”。

这里最终使用的是：

$$
S<2,3,3>

$$

这个参数组合正好可以和前两节连起来看：

- 前一节已经算出 `swizzle_bits = 2`，所以这里的第一个参数 `B=2`，表示需要重排的是 `2` 个 bit，也就是 `4` 个 `16B` copy 向量之间的选择位；
- `copy_bits = 128 bit = 16B`，而一个 `Float16` 元素是 `2B`，因此单个 copy 向量正好覆盖 `8` 个连续元素，所以要保留向量内部的位置不变，需要 `M=3`，因为 `2^3 = 8`；
- 第三个参数 `S=3` 则表示：把“向量编号”相关的 bit，跨过这 `3` 个最低位以后，再和更高位做异或混合 [2][3]。直观上，可以把它理解成“保持每个 `16B` 向量内部连续，但让不同 row 的向量编号发生变化”。

如果只抓住这一节最关键的两个问题，那么其实就是：为什么 `m=3`，以及为什么 `s=3`。

### 为什么 `m=3`

`MBase` 的定义是“最低多少个 bit 保持不变” [2][3]。在这个例子里，一次 copy 的粒度是：

$$
128\text{ bit} = 16\text{B}
$$

而每个 `Float16` 元素占 `2B`，所以一个 copy 向量里有：

$$
16\text{B} / 2\text{B} = 8
$$

个连续元素，也就是正好需要 `3` 个 bit 来标识：

$$
\log_2 8 = 3
$$

因此这里选 `m=3` 的本质是：把一个 `16B` copy 向量内部的元素位置完整保留下来，不让 swizzle 去打乱它们。

如果把线性地址写成元素下标，那么最低 `3` 个 bit 恰好表示“当前元素在这个 `16B` 向量里的第几个位置”。因此：

- `m < 3` 不合适，因为这会让 swizzle 直接动到向量内部的位置位，破坏 `16B` 连续访问；
- `m = 3` 刚好合适，因为它精确保留了一个 `16B` 向量内部的顺序；
- `m > 3` 也可以构造出别的合法 swizzle，但那意味着保留的连续区域超过了一个 `16B` 向量，会减少可用于打散 bank 的自由度；对于这个例子来说就偏保守了。

### 为什么 `s=3`

`SShift` 的作用是把高位那组参与异或的 bit，平移到低位来和当前 bit 组做 xor [2][3]。这里取 `s=3`，核心原因是：我们希望 swizzle 的作用单位就是“一个 `16B` 向量”，而不是更细或更粗的粒度。

前面已经知道：

- 最低 `3` 个 bit 是向量内部位置；
- 再往上的 `2` 个 bit 是当前 row 里“第几个 `16B` 向量”；
- 更高的 bit 则来自 row 编号或者更高层的 tile 结构。

因此把 `s` 设成 `3`，等价于说：先跨过这 `3` 个“向量内部位置位”，再去拿更高位的信息来改写“向量编号位”。这样做的效果是：

- 向量内部顺序不变；
- 被改写的是“当前元素属于 row 内第几个 `16B` 向量”；
- 参与改写的信息来自更高位，也就是 row / tile 相关的位，因此不同 row 会得到不同的向量编号映射。

直观上，可以把它近似理解成：

$$
\text{new\_vec} = \text{vec} \oplus f(\text{row})
$$

而不是去改写 `intra`。也就是说，swizzle 改变的是“这个 `16B` 块落到哪一组 bank”，而不是“块内部 8 个元素的相对顺序”。这里的 `f(row)` 是对更高位的简写，表示它由 row 或更高层级的地址 bit 决定；这是对文档与代码行为的直观概括。

从反面看，这个选择也很自然：

- 如果 `s < 3`，高位 mask 会侵入向量内部那 `3` 个位置位，容易破坏 `16B` copy 的局部连续性；
- 如果 `s > 3`，swizzle 就会跳过离当前向量最近的高位结构，只和更远的位做混合。那样通常仍可能构造出合法布局，但对这个 `8x32` atom 来说，就不再是“以一个 `16B` 向量为基本单位来做局部 bank 打散”了。

因此，`m=3` 和 `s=3` 其实是一组互相配合的选择：

- `m=3` 先定义“一个不可拆的局部单位就是 8 个 `Float16`，也就是 `16B`”；
- `s=3` 再保证 swizzle 的 xor 混合是在这个单位之上发生，而不是跑到单位内部去。

如果把列坐标 `c` 按 `16B` 向量拆开来看：

$$
c = (\text{vec} \times 8) + \text{intra}, \qquad \text{intra} \in [0, 7]
$$

那么：

- `intra = c \bmod 8` 对应最低 `3` 个 bit，它们由 `M=3` 保持不变；
- `vec = \lfloor c / 8 \rfloor` 对应接下来的 `2` 个 bit，它们正是 `B=2` 负责重排的对象；
- swizzle 之后，变化的主要是 `vec` 这一层，而不是 `intra` 这一层。

因此，`S<2,3,3>` 的效果可以近似理解为：

- 同一个 `16B` 向量内部，元素依然保持连续；
- 但不同 row 上，“这是第几个 `16B` 向量”这个编号会被 row 相关的更高位异或打散；
- 于是原本每一行都落到同一组 bank 的访问模式，会在不同行之间分散到不同 bank 组里。

这也是为什么这里的 swizzle 参数和前面的 `Swizzle<3,0,3>` 教学例子不同。`Swizzle<3,0,3>` 更适合直观看“整列被打散”；而这里的 `S<2,3,3>` 是围绕 `SM90` 上的 `16B` copy 向量来设计的，它更关心的是：

- 向量内部不要被打乱；
- 向量之间要能被 row 相关位打散；
- 最终结果仍然要适合后续的 warp-level load/store 或 TMA 访问。

需要补一句的是：`SM90` 官方的 `warpgroup.make_smem_layout_atom` 也提供了一组“紧凑且合法”的 layout atom，并明确说明它们是构造 TMA / UMMA 合法 SMEM layout 的最小单元，但同时也指出“还有其他合法布局的构造方式” [5]。因此，这里的 `S<2,3,3>` 更适合理解成一个教学上很直观的、围绕 `16B` 向量构造出来的 swizzled atom；它抓住了核心机制，但不必把它当成 `SM90` 官方 helper 所枚举 layout atom 的唯一写法。

对应的打印结果是：

$$
\text{After swizzle, layout\_atom} = S<2,3,3> \circ 0 \circ (8,32):(32,1)

$$

这里的 `0` 是组合时使用的基地址偏移；在这个例子里它不做额外平移，所以重点完全落在 `S<2,3,3>` 如何改写 offset 上。

从图上看，应用 swizzle 之后，原来按 row-major 连续排列的 `8x32` 元素，不再是“每一行都以完全相同的 bank 模式”落到 shared memory 中，而是形成了 row-dependent 的交错分布。这种分布的核心收益是：

- 对单个线程或单次向量 copy 来说，`16B` 局部连续性还在；
- 对一组线程协同访问多行数据来说，不同行更不容易击中同一批 bank；
- 因而它更适合作为后续 `ldmatrix` / `wgmma` / TMA staging 的 shared-memory 基础布局 [4][5]。

把它叫作 `layout atom` 也很贴切。按照 CUTLASS 对 `SM90` 工具函数的说明，实际工程里常见的流程就是先选一个“紧凑的 SMEM layout atom”，再把它 tile 到更大的 MMA tile 或 pipeline stage 上 [5]。这一节里的 `S<2,3,3> o 0 o (8,32):(32,1)`，正是在手工演示这个 atom 长什么样。

![The `8x32` layout atom after swizzle reflects a bank-conflict-aware shared-memory arrangement.](img/07_swizzle_layout_atom_8x32.svg)

---

## `tile_to_shape`：把一个 swizzled atom 平铺成更大的 shared memory layout

最后，利用 `tile_to_shape` 把单个 swizzled atom 扩展到完整的 shared memory tile：

```python
@cute.jit
def example():
    ...
    layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))
    print(f"after tile_to_shape(将tile扩展成shape), layout: {layout}")
    display_layout(layout)
```

打印结果是：

$$
\text{after tile\_to\_shape, layout}
= S<2,3,3> \circ 0 \circ ((8,16),(32,1)):((32,256),(1,0))

$$

可以把它理解成两步：

- 先定义一个已经带 swizzle 的局部 atom；
- 再把这个 atom 按目标 shape 平铺成完整的 shared memory layout。

这张图展示了把 swizzled atom 扩展到 `128x32` 之后的整体布局。

![The tiled `128x32` shared-memory layout is formed by repeating the swizzled atom over the target shape.](img/07_swizzle_layout_tiled_128x32.svg)

---

## 小结

围绕本文的两个例子，可以把 swizzle 的作用压缩成三句话：

- `Swizzle` 的核心不是改变逻辑 shape，而是改变地址位的映射方式；
- 教学用的 `8x8` 例子适合直观看“数据被打散到不同列”；
- 实际写 shared memory layout 时，更常见的做法是先根据数据类型和 copy 位宽选出 `swizzle_bits`，再把 swizzle 组合进 layout atom，最后扩展到更大的 tile。

如果继续往下看 `LDMATRIX`、`TiledCopy` 或 MMA 的 shared memory staging，这类 swizzled layout 就是连接“shared memory 中的存储形状”和“后续 warp 级装载模式”的重要桥梁。

---

## 参考链接

[1] [cute-viz swizzle example](https://github.com/NTT123/cute-viz/blob/main/examples/swizzle_layout_example.py)

[2] [CuTe 之 Swizzle](https://zhuanlan.zhihu.com/p/671419093)

[3] [NVIDIA CUTLASS Python DSL API: `cutlass.cute.Atom`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html)

[4] [CUDA C++ Programming Guide: Asynchronous Data Copies / TMA Swizzle for Compute Capability 9](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/async-copies.html)

[5] [NVIDIA CUTLASS Python DSL API: `cutlass.cute.nvgpu.warpgroup`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warpgroup.html)
