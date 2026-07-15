# Ampere Tensor Core GEMM 如何避免 Shared Memory Bank Conflict

结论：`ampere/tensorop_gemm.py` 的 A/B shared memory 不靠 padding，而是靠 `ComposedLayout(Swizzle, layout_atom)`。这个 swizzle 是 `sA/sB` 的地址映射属性，因此 **global-to-shared 写入** 和 **shared-to-register 读取** 都会经过同一套 swizzled shared layout。

![](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/07_ampere_tensorcore_avoid_bank_conflict/flowchartLRAglobalmemorybrAB128--907c641412ea5862.jpg)

## 共同基础：sA/sB 是 swizzled layout

代码先为 A/B 构造 shared memory layout：

```python
ab_copy_bits = 128
sA_layout = self._make_smem_layout_AB(
    mA.element_type,
    self.a_major_mode,
    ab_copy_bits,
    (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
)
sB_layout = self._make_smem_layout_AB(
    mB.element_type,
    self.b_major_mode,
    ab_copy_bits,
    (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
)
```

核心 helper 是：

```python
def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
    major_mode_size = (
        smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
    )
    major_mode_size = 64 if major_mode_size >= 64 else major_mode_size

    swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
    swizzle_bits = min(swizzle_bits, 3)

    layout_atom_outer = (
        cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
        if major_mode == utils.LayoutEnum.ROW_MAJOR
        else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
    )
    layout_atom = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 3, 3),
        0,
        layout_atom_outer,
    )
    layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))
    return layout
```

这里 `copy_bits = 128`，A/B 是 `Float16`，所以一次 copy 的基本连续粒度是：

```text
128 bit / 16 bit = 8 FP16 = 16B
```

`cute.make_swizzle(swizzle_bits, 3, 3)` 里的 `MBase = 3` 会保留低 3 bit，也就是保留 8 个 FP16 的局部连续性。swizzle 只重排不同 `16B` 块之间的物理位置。

![](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/07_ampere_tensorcore_avoid_bank_conflict/flowchartTDA8FP16br16Bcopyatom---3fea6cd4d7a31135.jpg)

## 为什么 B/M/S 这样取

CuTe 打印的 `S<B,M,S>` 对应代码里的：

```python
cute.make_swizzle(B, M, S)
```

在 `tensorop_gemm.py` 的 A/B layout 中，固定写成：

```python
cute.make_swizzle(swizzle_bits, 3, 3)
```

也就是：

```text
B = swizzle_bits
M = 3
S = 3
```

这三个值分别解决三个问题：

![](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/07_ampere_tensorcore_avoid_bank_conflict/flowchartTDAM=3--B8FP16br16BCB=s-fc5c619b411f32aa.jpg)

### 为什么 
`M = 3`

A/B 是 `Float16`，一次 copy 和一次 `ldmatrix` 设计上都围绕 `128 bit = 16B` 的连续块：

```text
16B / 2B = 8 FP16 = 2^3 FP16
```

因此最低 `3` 个元素 offset bit 表示“当前元素在这个 16B 向量里的第几个位置”。`M = 3` 的含义就是保留这 3 个 bit，不让 swizzle 打乱 16B 内部顺序。

```text
element_offset = ... vec_bits ... intra_bits
                                      ^^^^^^^
                                      低 3 bit，保留
```

所以 `M = 3` 同时服务两条路径：

-   `global -> shared`：每个 `cp.async` 写入的 8 个 FP16 仍然连续。
-   `shared -> register`：`ldmatrix` 看到的每个 16B row fragment 内部仍然连续。

### 为什么 
`B = swizzle_bits`

`B` 表示参与 xor 的 bit 数，也就是要打散多少个 `16B` 向量块。代码计算方式是：

```python
swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
swizzle_bits = min(swizzle_bits, 3)
```

可以理解为：

```text
参与分析的连续元素数 = major_mode_size
这一段有多少 bit     = major_mode_size * dtype.width
每个向量块多少 bit   = copy_bits = 128

16B 向量块数量 = major_mode_size * dtype.width / copy_bits
B              = log2(16B 向量块数量)
```

`min(..., 3)` 来自 shared memory 的 bank 周期：Ampere 可以按 `32` 个 bank、每个 bank `4B` 理解，一个完整 bank 周期是：

```text
32 banks * 4B = 128B
```

对 FP16 来说：

```text
128B / 2B = 64 FP16
128B / 16B = 8 个 16B 向量块 = 2^3
```

所以在一个 `128B` bank 周期内，最多只需要区分 `8` 个 16B 块，`B` 的有效上限就是 `3`。

### Row-major A/B：为什么是 
`S<2,3,3>`

当 A/B 是 row-major，也就是 K-major：

```bash
--a_major k --b_major k
```

`major_mode_size` 取 K tile 大小：

```text
major_mode_size = bK = 32
```

因此：

```text
16B 向量块数量 = 32 * 16 / 128 = 4 = 2^2
B = 2
```

基础 layout atom 是：

```text
(8,32):(32,1)
```

把坐标写成 `(r, k)`，线性元素 offset 是：

```text
offset = 32 * r + k
```

再把 K 拆成 16B 向量块编号和向量内部编号：

```text
k     = vec * 8 + intra
intra = k[0..2]
vec   = k[3..4]
```

`S<2,3,3>` 的效果是：

```text
保留 intra，也就是 16B 内部 8 个 FP16 不变；
把 offset bit 6..7 xor 到 offset bit 3..4；
也就是打散 vec 这个 16B 向量块编号。
```

直观公式可以写成：

```text
new_vec = vec xor (r >> 1)
```

注意 offset bit 5 不参与这次 xor。它对应的是 64B 半个 bank 周期；`S = 3` 让参与 xor 的高位从 element offset bit 6 开始，也就是从 `128B` bank 周期边界开始。

### M/N-major A/B：为什么是 
`S<3,3,3>`

当 A 是 M-major、B 是 N-major：

```bash
--a_major m --b_major n
```

连续的 M/N 方向 tile 是 `128`，但分析 bank pattern 时只需要看一个 `128B` 周期，也就是 `64` 个 FP16：

```text
major_mode_size = min(128, 64) = 64
```

因此：

```text
16B 向量块数量 = 64 * 16 / 128 = 8 = 2^3
B = 3
```

基础 layout atom 是：

```text
(64,8):(1,64)
```

把坐标写成 `(major, k8)`，线性元素 offset 是：

```text
offset = major + 64 * k8
```

此时：

```text
major = vec * 8 + intra
intra = major[0..2]
vec   = major[3..5]
k8    = offset[6..8]
```

`S<3,3,3>` 的效果是：

```text
保留 intra，也就是 16B 内部 8 个 FP16 不变；
把 offset bit 6..8 xor 到 offset bit 3..5；
也就是把 k8 相关 bit xor 到 M/N 方向的 16B 向量块编号上。
```

直观公式可以写成：

```text
new_vec = vec xor k8
```

### 为什么 
`S = 3`

`SShift = 3` 的核心原因是：当前 layout 以 `16B` 为最小连续向量块，而 shared memory 的 bank pattern 周期是 `128B`。

用 FP16 元素 offset 表示：

```text
16B  = 8 FP16  = 2^3 elements  -> 16B 向量块编号从 bit 3 开始
128B = 64 FP16 = 2^6 elements  -> bank 周期相关高位从 bit 6 开始
```

两者相差：

```text
6 - 3 = 3
```

所以 `S = 3`。它把 `128B` 周期相关的高位 xor 到 `16B` 向量块编号上，同时不触碰最低 3 bit 的向量内部顺序。

## 1. Global 写入 Shared

global-to-shared 路径使用 `cp.async`：

```python
atom_async_copy = cute.make_copy_atom(
    cute.nvgpu.cpasync.CopyG2SOp(
        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
    ),
    mA.element_type,
    num_bits_per_copy=ab_copy_bits,
)
```

每个 copy atom 从 global memory 读取 `128 bit = 16B`。对 A/B 的 `Float16` 来说，就是连续 8 个元素。

线程和值的 copy layout 由 `_make_gmem_tiled_copy_AB` 决定：

```python
copy_elems = copy_bits // dtype.width

value_layout = (
    cute.make_layout((1, copy_elems))
    if major_mode == utils.LayoutEnum.ROW_MAJOR
    else cute.make_layout((copy_elems, 1))
)
```

所以 global 侧是按矩阵主序做连续 128-bit 读取：

-   `ROW_MAJOR`，也就是 A/B 的 K-major：每个线程沿 K 方向读连续 8 个 FP16。
-   非 `ROW_MAJOR`，也就是 A 的 M-major / B 的 N-major：每个线程沿 M/N 方向读连续 8 个 FP16。

但 shared 侧不是简单线性连续写入。目标 tensor 是 `sA/sB`：

```python
tAsA = thr_copy_A.partition_D(sA)
tBsB = thr_copy_B.partition_D(sB)
```

真正写入时：

```python
cute.copy(
    tiled_copy_A,
    tAgA[None, None, None, k_tile_index],
    tAsA[None, None, None, smem_pipe_write],
    pred=tApA,
)
```

这里 `tAsA/tBsB` 是对 swizzled `sA/sB` 做 `partition_D` 得到的 destination view。因此：

```text
global 读：每个 copy atom 连续 16B
shared 写：16B 块内部连续，块与块之间按 swizzle 后的 shared offset 分布
```

这点很关键：swizzle 不是只在读取 shared 时才生效。只要目标是 `sA/sB`，写 shared 的物理地址就已经经过 swizzle。

### K-major 时的写入形状

如果运行：

```bash
--a_major k --b_major k
```

A `[M,K]` 和 B `[N,K]` 都是 K-major。此时：

```text
major_mode_size = bK = 32
swizzle_bits = log2(32 * 16 / 128) = 2
layout_atom_outer = (8,32):(32,1)
sA/sB layout atom = S<2,3,3> o (8,32):(32,1)
```

也就是保留每个 16B 向量内部的连续性，再把一个 layout atom 内的多个 16B 向量块交错打散。

### M/N-major 时的写入形状

如果运行默认示例：

```bash
--a_major m --b_major n
```

A 沿 M 连续，B 沿 N 连续。此时：

```text
major_mode_size = min(128, 64) = 64
swizzle_bits = log2(64 * 16 / 128) = 3
layout_atom_outer = (64,8):(1,64)
sA/sB layout atom = S<3,3,3> o (64,8):(1,64)
```

这里一个 shared layout atom 仍然按 `16B` 连续块作为基本单位，只是参与打散的块数从 `2^2` 变成 `2^3`。

## 2. Shared 读取 Register

shared-to-register 路径不是普通 `autovec_copy`，而是 `ldmatrix`：

```python
atom_copy_s2r_A = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(
        self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
    ),
    mA.element_type,
)
atom_copy_s2r_B = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(
        self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
    ),
    mB.element_type,
)
```

`4` 表示 `ldmatrix.x4`。这个 copy atom 会被 retile 成 MMA 需要的 thread-value layout：

```python
tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
```

实际读取发生在 mainloop：

```python
cute.copy(
    tiled_copy_s2r_A,
    tCsA_p[None, None, k_block_next],
    tCrA_copy_view[None, None, k_block_next],
)
cute.copy(
    tiled_copy_s2r_B,
    tCsB_p[None, None, k_block_next],
    tCrB_copy_view[None, None, k_block_next],
)
```

这里 `tCsA_copy_view/tCsB_copy_view` 是对同一个 swizzled `sA/sB` 做 `partition_S` 得到的 source view。也就是说：

```text
shared 读：从 swizzled shared offset 读取
register 写：写入 MMA fragment，不再涉及 shared bank
```

这也是 `_make_smem_layout_AB` 上方注释说 swizzle 目标是 shared-to-register copy 的原因：`ldmatrix` 的 warp 级读取模式很固定，如果 shared layout 不配合，多个 lane 很容易在同一条 shared load 指令里访问同一 bank 的不同地址。swizzle 把不同 16B 块打散，让 `ldmatrix.x4` 的读取模式更接近 bank-conflict-free。

## 写入和读取的关系

两条路径不是各自独立选择地址，而是通过同一个 `sA/sB` layout 闭环：

![](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/07_ampere_tensorcore_avoid_bank_conflict/flowchartTDAbrmknk--BsAsBlayoutB-3c4b0e3a705c8b8e.jpg)

因此可以这样理解：

```text
swizzle 生效的位置：shared memory 的逻辑坐标到物理 offset 映射。
global-to-shared：global 读连续 16B；shared 写使用 swizzled offset。
shared-to-register：shared 读使用 swizzled offset；register fragment 按 MMA layout 接收。
```

这和 `sgemm.py` 的 padding 方案不同。`sgemm.py` 是用 `stride_k += 4` 改变 K 方向步长；`tensorop_gemm.py` 则用 `Swizzle` 改变地址位，既不增加 shared memory shape，也能适配 `cp.async + ldmatrix` 的 Tensor Core 数据通路。

## 小结

`tensorop_gemm.py` 的 bank-conflict 处理可以压缩成三句话：

-   `global` 侧尽量做连续、合并的 `128-bit cp.async` 读取；
-   `shared` 侧的 `sA/sB` 本身是 swizzled layout，所以写入和读取都会使用 swizzled 物理地址；
-   `MBase=3` 保留每个 `16B = 8 FP16` copy/ldmatrix 粒度内部连续，`swizzle_bits` 负责打散多个 16B 块，从而适配 `ldmatrix.x4` 的 warp 级 shared-memory 读取。



Reference:

