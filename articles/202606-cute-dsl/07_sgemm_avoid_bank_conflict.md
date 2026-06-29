# SGEMM 如何避免 Shared Memory Bank Conflict

结论：`ampere/sgemm.py`、`ampere/sgemm_swizzle.py` 和 `ampere/sgemm_swizzle_hierarchy.py` 展示了三种解决同一个 bank conflict 的方案。
这个 conflict 主要来自 A/B row-major 时的 global-to-shared 写入路径；shared-to-register 读取路径是另一件事，代码把它设计成 bank-friendly，但不能只看 Python 源码就宣称每条 shared load 都严格保证 0 conflict。

- `sgemm.py`：用 `+4 float` padding，把 K 维 stride 从 `128` 改成 `132`。
- `sgemm_swizzle.py`：不用 padding，在 flat shared layout 上用 `S<3,2,5>` 改写 shared-memory offset。
- `sgemm_swizzle_hierarchy.py`：先构造 `32x8` layout atom，再用 `S<3,2,3>` 和 `tile_to_shape` 扩展到完整 shared tile。

```mermaid
flowchart LR
    A["A[M,K] row-major<br/>B[N,K] row-major"] --> B["K 维连续<br/>LayoutEnum.ROW_MAJOR"]
    B --> C["问题<br/>offset = major + 128*k"]
    C --> D["方案 1: padding<br/>stride_k = 132"]
    C --> E["方案 2: flat swizzle<br/>S<3,2,5> o layout"]
    C --> H["方案 3: hierarchy swizzle<br/>S<3,2,3> o 32x8 atom"]
    D --> F["bank = major + 4*k mod 32"]
    E --> G["bank-select bits 被 K bits xor 打散"]
    H --> I["tile_to_shape 扩展到 128x8xPIPE"]
```

## Bank conflict 基础

Shared memory 可以按 32 个 bank 来理解。对 `float32` 这类 32-bit 访问，地址到 bank 的映射是：

```text
bank_id = (byte_address / 4) % 32
```

也可以写成按 `float32` 元素下标取模：

```text
bank_id = element_offset % 32
```

一个 bank 不是只有 4 字节容量；更准确地说，每个 bank 的基本访问宽度是 4 字节，也就是一个 32-bit word。shared memory 会把连续的 32-bit word 轮流分配给 32 个 bank：

```text
word 0  -> bank 0
word 1  -> bank 1
...
word 31 -> bank 31
word 32 -> bank 0
```

bank conflict 发生在同一个 warp 的一条 shared-memory 指令里：多个线程访问了同一个 bank 的不同地址，这些访问需要被拆成多轮服务。反过来，如果 32 个线程分别落到 32 个不同 bank，就没有 bank conflict。

```text
thread i 读取 element[i]      -> bank = i % 32      -> 无冲突
thread i 读取 element[i * 32] -> bank = 0           -> 32-way conflict
```

如果多个线程读的是完全相同的 shared memory 地址，硬件可以做 broadcast，这不是本文讨论的典型 bank conflict。

还需要区分三个容易混在一起的粒度：


| 概念                                | 对应路径                      |               粒度 | 说明                                                                |
| ------------------------------------- | ------------------------------- | -------------------: | --------------------------------------------------------------------- |
| shared memory bank                  | shared 读/写                  |               `4B` | 32 个 bank，连续 32-bit word 映射到连续 bank。                      |
| `cp.async` / LDGSTS per-thread copy | global -> shared              |    `4B / 8B / 16B` | 官方文档支持这三种尺寸；`16B = 128bit` 是常用最大 per-thread 粒度。 |
| global memory transaction           | global -> shared 的 global 读 | `32B / 64B / 128B` | warp 的 global memory 访问会合并成一个或多个自然对齐 transaction。  |

因此，本文后面用 `128B` 分析 shared memory bank pattern，指的是：

```text
32 banks * 4B/bank = 128B
```

也就是一个完整 bank 周期。它不是说 `cp.async` 的最小 copy 粒度是 `128B`。在 `sgemm.py` 的 row-major A/B 分支里，global-to-shared 的 copy atom 反而是每线程 `32-bit = 4B`，因为代码为了沿 K 方向搬运而关闭了向量化。

当一个 warp 的单条 shared-memory 指令访问量超过一个 `128B` bank 周期时，分析 bank conflict 应按事务分片分别看，而不是把所有字节混成一个集合。本文的表格都按一个 `128B` bank 周期来分析。

## 问题来源

当前脚本都使用 A/B row-major：

```bash
--a_major k --b_major k --c_major n
```

A 的逻辑形状是 `[M,K]`，B 的逻辑形状是 `[N,K]`。二者都是 row-major 时，K 是连续维度；在 CuTe/CUTLASS 的 `LayoutEnum` 里对应 `ROW_MAJOR`，也就是 K-major。

但 shared memory 中的 `sA/sB` 被组织成 M/N-major：

```text
sA: (M, K, PIPE), stride=(1, stride_k, ...)
sB: (N, K, PIPE), stride=(1, stride_k, ...)
```

默认 tile 是：

```text
bM = 128
bN = 128
bK = 8
```

如果不用 padding 或 swizzle，那么：

```text
offset(major, k) = major + 128 * k
bank(major, k)   = (major + 128 * k) % 32
                 = major % 32
```

也就是说，对固定的 `major`，不同 `k` 切片会反复落到同一个 bank。`sA` 里 `major = m`，`sB` 里 `major = n`。

为什么 row-major 需要处理，col-major 不需要？关键要分开看两条 shared memory 路径：


| 路径               | Row-major A/B                        | Col-major A / B                          | shared 侧 layout         | 结论                                                            |
| -------------------- | -------------------------------------- | ------------------------------------------ | -------------------------- | ----------------------------------------------------------------- |
| global -> shared   | global 沿 K 连续，copy 线程沿 K 展开 | global 沿 M/N 连续，copy 线程沿 M/N 展开 | `sA/sB` 始终是 M/N-major | row-major 写入会跨`stride_k=128`，需要处理。                    |
| shared -> register | 固定 K，沿 M/N 取 fragment           | 固定 K，沿 M/N 取 fragment               | `sA/sB` 始终是 M/N-major | 两者都服务于 M/N 连续读取，不是 row-major conflict 的主要来源。 |

也就是说，`sgemm.py` 的真正差异是 global memory 的主序：row-major A/B 是 K-major，global-to-shared copy 的 lane 会沿 K 方向展开；但 shared memory 里的 `sA/sB` 为了后续 shared-to-register 读取，始终被组织成 M/N-major。

```mermaid
flowchart TD
    A["ROW_MAJOR<br/>A/B 是 K-major"] --> B["copy 线程沿 K 展开"]
    B --> C["写入 M/N-major smem 时<br/>lane 间跨 stride_k"]
    C --> D["stride_k = 128<br/>正好是 32 banks 的整数轮"]
    D --> E["需要 padding 或 swizzle"]

    F["COL_MAJOR<br/>A/B 是 M/N-major"] --> G["copy 线程沿 M/N 连续展开"]
    G --> H["写入 M/N-major smem 时<br/>lane 访问连续 word"]
    H --> I["自然分散到 bank"]
```

对 row-major，也就是 K-major 的 A/B，代码使用默认 copy layout：

```python
tA = cute.make_layout((self._num_threads // self._bK, self._bK), stride=(self._bK, 1))
tB = cute.make_layout((self._num_threads // self._bK, self._bK), stride=(self._bK, 1))
```

在默认 `bK = 8` 时，一个 warp 的 lane 可以粗略理解成：

```text
lane  = outer * 8 + k
outer = 0..3
k     = 0..7
```

这里 `lane` 是 warp 内线程编号 `0..31`。`outer` 表示除 K 以外的连续维：对 `sA` 是 M 维，对 `sB` 是 N 维。

不做处理时，同一个 `outer` 的 8 个 `k` 会写到：

```text
offset = outer + k * 128
bank   = (outer + k * 128) % 32
       = outer
```

这就是 row-major 需要处理的根因：**copy 线程沿 K 变，目标 shared layout 的 K stride 又刚好是 32 个 bank 的整数倍**。

对 col-major，也就是 A 的 M-major / B 的 N-major，代码会改成沿 M/N 连续维做 vectorized copy。源主序和目标 shared layout 主序一致，warp 优先覆盖连续的 M/N 段，每个线程再通过 `vA/vB = (4,1)` 写连续 4 个 `float`，自然按 bank 轮转分布。

所以后面的 padding、flat swizzle、hierarchy swizzle 主要都是在处理 **row-major A/B 的 global-to-shared 写入**。

## 方案一：Padding

`sgemm.py` 的实现是给 K-major 情况加 4 个 `float` padding：

```python
padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0

sA_layout = cute.make_layout(
    (self._bM, self._bK, self._num_stages),
    stride=(1, (self._bM + padding_a), self._bK * (self._bM + padding_a)),
)
sB_layout = cute.make_layout(
    (self._bN, self._bK, self._num_stages),
    stride=(1, (self._bN + padding_b), self._bK * (self._bN + padding_b)),
)
```

加 padding 后，K 维 stride 从 `128` 变成 `132`：

```text
offset(major, k) = major + 132 * k
bank(major, k)   = (major + 132 * k) % 32
                 = (major + 4 * k) % 32
```

所以每推进一个 `k`，bank 会错开 4 个位置：

```text
k = 0: bank = major
k = 1: bank = major + 4
k = 2: bank = major + 8
...
```

用默认尺寸看一个完整 warp 的 row-major 写入。此时每个 lane 写一个 `float32 = 4B`，一个 warp 一次写：

```text
32 lanes * 4B = 128B
```

这正好覆盖一个完整 bank 周期。默认 copy layout 可以把一个 warp 粗略写成：

```text
lane  = outer * 8 + k
outer = 0..3
k     = 0..7
```

无 padding 时：

```text
offset = outer + 128 * k
bank   = outer
```


| lane 分组                | 写入坐标`(major,k)` | bank 分布         | 结果                                   |
| -------------------------- | --------------------- | ------------------- | ---------------------------------------- |
| `outer=0`, lane `0..7`   | `(0,0)..(0,7)`      | `0,0,0,0,0,0,0,0` | bank 0 上 8 个不同地址，8-way conflict |
| `outer=1`, lane `8..15`  | `(1,0)..(1,7)`      | `1,1,1,1,1,1,1,1` | bank 1 上 8 个不同地址，8-way conflict |
| `outer=2`, lane `16..23` | `(2,0)..(2,7)`      | `2,2,2,2,2,2,2,2` | bank 2 上 8 个不同地址，8-way conflict |
| `outer=3`, lane `24..31` | `(3,0)..(3,7)`      | `3,3,3,3,3,3,3,3` | bank 3 上 8 个不同地址，8-way conflict |

加 4 个 `float` padding 后：

```text
offset = outer + 132 * k
bank   = outer + 4 * k mod 32
```


| lane 分组                | 写入坐标`(major,k)` | bank 分布               | 结果             |
| -------------------------- | --------------------- | ------------------------- | ------------------ |
| `outer=0`, lane `0..7`   | `(0,0)..(0,7)`      | `0,4,8,12,16,20,24,28`  | 分散到 8 个 bank |
| `outer=1`, lane `8..15`  | `(1,0)..(1,7)`      | `1,5,9,13,17,21,25,29`  | 分散到 8 个 bank |
| `outer=2`, lane `16..23` | `(2,0)..(2,7)`      | `2,6,10,14,18,22,26,30` | 分散到 8 个 bank |
| `outer=3`, lane `24..31` | `(3,0)..(3,7)`      | `3,7,11,15,19,23,27,31` | 分散到 8 个 bank |

合起来看，这 32 个 lane 正好落到 32 个不同 bank，所以这一批 `128B` 写入没有 bank conflict。若某条 shared-memory 指令的访问量超过 `128B`，仍应按每个 `128B` 事务分片重复这个分析。

优点是简单、稳定、地址计算直观。代价是多占一点 shared memory。默认配置下，每个 operand 多：

```text
4 padding elements * 8 K * 3 stages = 96 float
```

对 A/B 合计也只有 `192 float = 768B`。

## Swizzle 原理

CuTe `Swizzle` 是一个作用在线性 offset 上的地址变换。普通 layout 先把逻辑坐标映射成 offset：

```text
layout: coord -> offset
```

swizzle 再把这个 offset 改写成新的 shared-memory offset：

```text
swizzle: offset -> offset'
```

组合起来就是：

```text
coord -> layout(coord) -> swizzle(layout(coord))
```

CuTe 的 `S<B,M,S>` 可以按下面的 bit 图理解：

```text
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase (least-sig bits kept constant)
                 ^-^       ^-^     BBits (number of bits in mask)
                   ^---------^     SShift (distance to shift YYY)
                                      (positive: right, negative: left)

Given:    0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
Result:   0bxxxxxxxxxxxxxxxYYYxxxxxxxAAAxxxx
          where AAA = ZZZ xor YYY
```

这里使用 xor，是因为固定 `YYY` 时，`ZZZ -> ZZZ xor YYY` 是双射：不同的 `ZZZ` 不会映射到同一个结果，并且再 xor 一次同一个 `YYY` 就能还原。swizzle 因此能重排 bank 选择位，同时不改变元素总数，也不会制造地址重叠。

三个参数的含义是：

- `M` / `MBase`：最低多少个 bit 保持不变，通常对应一个局部连续向量内部的位置。
- `B` / `BBits`：参与 xor 的 bit 数，也就是要打散多少个向量块编号 bit。
- `S` / `SShift`：高位 mask 与低位 mask 的距离，决定从哪里取高位信息 xor 到低位块编号。

也可以更形象地理解：最低 `2^M` 个元素组成一个 cell；接下来的 `B` 个 bit 表示 cell 的列编号(即 ZZZ)；相隔 `S` 的高位表示行信息（即YYY）。swizzle 用行信息 xor 列编号（即 `AAA = ZZZ xor YYY`），再映射回线性地址。具体可以看 [7] 中的示意图和解释。

![](https://picx.zhimg.com/v2-21a666cbc0e7246fcc30c6ac333442fd_1440w.jpg)

因此，选择 swizzle 参数时要先回答三件事：

```text
1. 哪一段低位必须保持连续？即M如何设置
2. 要打散几个 bit？即B如何设置
3. 冲突来源的高位从 offset 的哪一位开始？即S如何设置
```

## 方案二：Flat Swizzle

`sgemm_swizzle.py` 不加 padding，而是构造无 padding 的基础 layout：

```python
base_layout = cute.make_layout(
    (major_extent, k_extent, num_stages),
    stride=(1, major_extent, k_extent * major_extent),
)
```

然后在 K-major 情况下组合 swizzle：

```python
return cute.make_composed_layout(
    cute.make_swizzle(3, 2, 5), 0, base_layout
)
```

完整 helper 是：

```python
@cute.jit
def _make_smem_layout(
    self,
    major_extent: cutlass.Constexpr,
    k_extent: cutlass.Constexpr,
    num_stages: cutlass.Constexpr,
    major_mode: cutlass.Constexpr,
):
    base_layout = cute.make_layout(
        (major_extent, k_extent, num_stages),
        stride=(1, major_extent, k_extent * major_extent),
    )
    if cutlass.const_expr(major_mode == utils.LayoutEnum.ROW_MAJOR):
        return cute.make_composed_layout(
            cute.make_swizzle(3, 2, 5), 0, base_layout
        )
    return cute.make_composed_layout(
        cute.make_swizzle(0, 2, 5), 0, base_layout
    )
```

这里用 `S<3,2,5>`，也就是 `cute.make_swizzle(3, 2, 5)`。它是针对当前默认 FP32 `128x128x8` tile 推出来的：

```text
offset = major + 128 * k
128    = 2^7
bK     = 8 = 2^3
```

### 为什么是 `MBase = 2`

`float32` 一个元素 4B。这里保留连续 4 个 value，是为了保留 `16B = 128bit` 的局部向量粒度；它对应常见最大 per-thread copy / vector load 粒度。对 FP32 来说：

```text
4 float = 16B
```

保留最低 2 个元素 bit，就能保留 4 个 `float32` 的局部连续性：

```text
2^2 = 4 elements = 16B
```

所以 `MBase = 2`。

### 为什么是 `BBits = 3`

当前 `bK = 8`，K tile 内有 8 个 K 位置：

```text
8 = 2^3
```

我们希望把这 3 个 K bit 打散到 bank-select bits 上，所以 `BBits = 3`。

### 为什么是 `SShift = 5`

无 padding 时：

```text
offset = major + 128 * k = major + 2^7 * k
```

K bit 从 offset 的 bit 7 开始。我们保留最低 2 bit 后，希望改写 bit 2..4 这三个 bank-select bit。因此位移距离是：

```text
7 - 2 = 5
```

所以 `SShift = 5`。

直观上，swizzle 后 bank 不再只是：

```text
bank = major % 32
```

而是把 `k` 相关 bit xor 到 `major` 的 bank-select bits 中。可以近似理解为：

```text
bank = (major & 3) + 4 * (((major >> 2) & 7) xor k)
```

这样固定 `major`、不同 `k` 会落到不同 bank 组，达到和 padding 类似的 bank skew 效果，但不增加 shared memory 占用。

注意，不能在这个 flat layout 上直接把 `SShift` 改成 `3`。因为 flat layout 的 K stride 是 `128 = 2^7`，K bit 从 offset bit 7 开始；`S<3,2,3>` 只会把 offset bit 5..7 xor 到 bit 2..4，其中只有 bit 7 是 K 的最低位，不能完整打散 `k = 0..7`。

![Flat swizzle applies `S<3,2,5>` directly on the full `128x8` shared tile.](img/07_sgemm_flat_swizzle_128x8.svg)

## 方案三：层次化 Swizzle

`sgemm_swizzle_hierarchy.py` 使用另一种写法：先构造一个更小的 `32x8` layout atom，再用 `tile_to_shape` 平铺成完整的 `(128, 8, PIPE)` shared tile。

核心 helper 是：

```python
@cute.jit
def _make_smem_layout(
    self,
    major_extent: cutlass.Constexpr,
    k_extent: cutlass.Constexpr,
    num_stages: cutlass.Constexpr,
    major_mode: cutlass.Constexpr,
):
    base_layout = cute.make_layout(
        (major_extent, k_extent, num_stages),
        stride=(1, major_extent, k_extent * major_extent),
    )
    if cutlass.const_expr(major_mode == utils.LayoutEnum.ROW_MAJOR):
        layout_atom_outer = cute.make_layout(
            (32, k_extent), stride=(1, 32)
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(
            layout_atom,
            (major_extent, k_extent, num_stages),
            (0, 1, 2),
        )
    return cute.make_composed_layout(
        cute.make_swizzle(0, 2, 5), 0, base_layout
    )
```

这里的关键不是“把 `SShift` 从 5 改成 3”，而是先改变 swizzle 作用的基础 layout atom。

### 为什么是 `MBase = 2`

层次化版本仍然是 FP32 SIMT SGEMM，shared-to-register 读取仍然希望每个线程拿连续 4 个 `float32`：

```text
4 float32 = 16B = 2^2 elements
```

因此最低 2 个元素 bit 必须保持不变：

```text
MBase = 2
```

### 为什么是 `BBits = 3`

默认 K tile 大小仍然是：

```text
bK = 8 = 2^3
```

row-major A/B 的冲突来自同一个 `major_inner` 下 8 个不同 K 位置落到同一批 bank。要把 `k=0..7` 全部打散，就需要 3 个 K bit 参与 xor：

```text
BBits = 3
```

### 为什么是 `SShift = 3`

在 `32x8` atom 内：

```text
offset = major_inner + 32 * k
       = major_inner + 2^5 * k
```

FP32 下仍然希望保留 4 个连续元素：

```text
4 float = 16B = 2^2 elements
```

因此：

```text
MBase = 2
BBits = 3
SShift = 5 - 2 = 3
```

也就是：

```text
S<3,2,3>
```

直观上，这相当于把原来的 `128x8` shared tile 切成 4 个 `32x8` atom：

```mermaid
flowchart TD
    A["flat layout<br/>128x8"] --> B["K stride = 128<br/>K bits 从 bit 7 开始"]
    B --> C["需要 S<3,2,5>"]

    D["hierarchy layout<br/>32x8 atom"] --> E["atom 内 K stride = 32<br/>K bits 从 bit 5 开始"]
    E --> F["需要 S<3,2,3>"]
    F --> G["tile_to_shape<br/>平铺到 128x8xPIPE"]
```

在单个 `32x8` atom 内，可以近似理解为：

```text
bank = (major_inner & 3) + 4 * (((major_inner >> 2) & 7) xor k)
```

它和 flat `S<3,2,5>` 的目标相同：保留 4 个 FP32 的局部连续性，同时把 `k` 的 3 个 bit xor 到 bank-select bits 上。区别是物理 layout 不同：

- flat swizzle：在完整 `(128,8,PIPE)` layout 上直接改写地址位；
- hierarchy swizzle：先在 `32x8` layout atom 内改写地址位，再平铺到完整 shared tile。

所以结论是：

```text
flat layout + S<3,2,3>：不对，K bits 没有完整参与 xor。
32x8 atom + S<3,2,3> + tile_to_shape：可以，atom 内 K bits 正好位于 bit 5..7。
```

![Hierarchy swizzle first applies `S<3,2,3>` on a `32x8` atom.](img/07_sgemm_hierarchy_swizzle_atom_32x8.svg)

## Shared-to-register 读取是否保证无冲突

先给结论：`sgemm.py` 的 shared-to-register 读取是 bank-friendly 的，但不是源码层面的“强保证无 bank conflict”。

需要把两条 shared memory 路径分开：

```mermaid
flowchart LR
    A["global-to-shared<br/>cp.async"] --> B["写 sA/sB"]
    B --> C["padding / swizzle<br/>处理 K-major 写入冲突"]
    B --> D["shared-to-register<br/>autovec_copy"]
    D --> E["M/N-major 连续读取<br/>permutation_tiler_M/N"]
```

padding 和 swizzle 解决的是前一条路径：A/B 是 row-major，copy 线程沿 K 方向展开，而目标 `sA/sB` 的 K stride 原本是 `128`，正好是 `32` 个 bank 的整数倍。

读取路径不再按这个模式访问。这些方案都保留了同一个逻辑形状：

```text
sA: (M, K, PIPE)
sB: (N, K, PIPE)
```

区别只是逻辑坐标到 shared-memory offset 的映射方式不同。

计算阶段，代码通过这个 shared layout 做 partition：

```python
tCsA = thr_mma.partition_A(sA)
tCsB = thr_mma.partition_B(sB)
```

再用同步的 `autovec_copy` 从 shared memory 读到寄存器：

```python
cute.autovec_copy(tCsA_p[None, None, k_block_next], tCrA[None, None, k_block_next])
cute.autovec_copy(tCsB_p[None, None, k_block_next], tCrB[None, None, k_block_next])
```

这里每次 `autovec_copy` 固定一个 `k_block_next`，主要沿 M/N 方向取数，不是让 warp 内线程沿 K stride 读取。因此它不会复现前面 `stride_k = 128` 导致的典型 conflict。

用 `inspect_cuda_core_partitions.py` 看当前 CUDA core MMA 的坐标划分，thread 0 的 A/B 读取形状是：

```text
tCsA = tensor<(0,0,0) o (1,(4,2),8,3):(0,(1@0,64@0),1@1,1@2)>
tCsB = tensor<(0,0,0) o (1,(4,2),8,3):(0,(1@0,64@0),1@1,1@2)>
```

对固定的 `k_block` 和 pipeline stage，可以读成：

```text
每个线程读取两组 M/N 方向的数据；
每组 4 个连续 float32；
两组之间在 M/N 方向相隔 64 个元素。
```

这和代码里的 permutation 对应：

`sgemm.py` 还通过 `permutation_tiler_M/N` 让每个线程在 M/N 方向各拿连续的 4 个 value：

```python
permutation_tiler_M = cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1))
permutation_tiler_N = cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1))
```

连续 4 个 `float32` 正好是 `16B`，会落到 4 个相邻 bank 上。对 A 来说，很多 lane 只是 N 方向不同、M/K 相同，读到的 A 地址可能相同，硬件可以 broadcast；对 B 来说，`permutation_tiler_N` 让 N 方向访问尽量连续。这个布局目标是让读取阶段也顺，但它和 padding/swizzle 是配合关系，不是替代关系。

严格说，bank conflict 是“warp 的某条 shared-memory 指令”上的属性，最后取决于 CuTe lowering、实际生成的 shared load 指令、向量化宽度、transaction 拆分和 lane 到地址的对应关系。因此：

```text
可以说：当前 sgemm 的 shared-to-register 读取布局是 bank-friendly。
不应说：只凭 sgemm.py 源码即可保证所有 shared-memory 读取都 0 bank conflict。
```

如果要确认硬件层面的结果，需要看 SASS 或用 NCU 采集 shared memory bank conflict 相关指标。本文的 padding/swizzle 推导，严格对应的是 row-major A/B 写入 `sA/sB` 时的 bank 分布。

## 性能对比

测试脚本：

```bash
bash articles/202606-cute-dsl/ampere/sgemm.sh
bash articles/202606-cute-dsl/ampere/sgemm_swizzle.sh
python articles/202606-cute-dsl/ampere/sgemm_swizzle_hierarchy.py \
  --mnk 8192,8192,8192 --a_major k --b_major k --c_major n \
  --warmup_iterations 2 --iterations 100 --skip_ref_check
```

共同参数：

```text
mnk = 8192,8192,8192
a_major = k
b_major = k
c_major = n
warmup_iterations = 2
iterations = 100
```

三个版本都做过正确性验证。实测结果：


| 版本                                          |        Run 1 |        Run 2 |         平均 |
| ----------------------------------------------- | -------------: | -------------: | -------------: |
| padding`sgemm.py`                             | `66.0516 ms` | `66.0295 ms` | `66.0406 ms` |
| flat swizzle`sgemm_swizzle.py`                | `65.6068 ms` | `65.6027 ms` | `65.6048 ms` |
| hierarchy swizzle`sgemm_swizzle_hierarchy.py` | `65.8150 ms` | `65.8117 ms` | `65.8134 ms` |

两个 swizzle 版本都略快于 padding。flat swizzle 在这组测试里更快一些，但差距很小；是否真正减少 bank conflict，需要再用 NCU 看 shared memory bank conflict 指标。

## 取舍

```mermaid
flowchart TD
    A["避免 K-major smem bank conflict"] --> B["Padding"]
    A --> C["Flat Swizzle"]
    A --> F["Hierarchy Swizzle"]
    B --> D["实现简单<br/>地址计算直观<br/>多占少量 smem"]
    C --> E["不浪费 smem<br/>layout 更复杂<br/>参数依赖 tile / dtype / copy 形状"]
    F --> G["结构接近 layout atom<br/>可用 SShift=3<br/>nested layout 更复杂"]
```

padding 更适合作为教学和基线实现：代码短、容易推导、对当前 FP32 SIMT SGEMM 足够有效。

swizzle 更接近高性能 GEMM 模板里的常见方向：它不改变逻辑 shape，而是改写 shared-memory 地址位。flat 版本推导直接，hierarchy 版本更接近 layout atom 的写法。代价是参数和访问模式强绑定，必须验证 `partition_D`、`partition_A/B`、`autovec_copy` 这些路径都按同一个 `ComposedLayout` 工作。

shared-to-register 读取这条路径还要单独看：`permutation_tiler_M/N` 让每个线程拿连续 4 个元素，整体是 bank-friendly 的读取形状；但是否完全 0 conflict，应以生成代码和 NCU 指标为准。

## 总结

这些版本解决的是同一个问题：

```text
row-major A/B => K-major copy => 写入 M/N-major smem 时 stride_k = 128
=> 不处理会让不同 k 落到同一 bank
```

三种修复方式是：

```text
padding: stride_k = 128 + 4 = 132
flat swizzle: offset' = S<3,2,5>(major + 128*k)
hierarchy swizzle: offset' = tile_to_shape(S<3,2,3> o (32,8):(1,32))
```

padding 用空间换简单；swizzle 用更复杂的地址映射换更紧凑的 shared memory。当前脚本实测三者性能非常接近，两个 swizzle 版本都略快但差距不到 1%。

shared-to-register 读取阶段不是 padding/swizzle 主要解决的矛盾。它靠 M/N-major 的 shared layout 和 `permutation_tiler_M/N` 获得连续读取形状；这能降低 bank conflict 风险，但不能替代 SASS/NCU 级别的验证。

## 参考

1. [NVIDIA CUDA C++ Programming Guide: Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x)：32 个 bank、连续 32-bit word 映射到连续 bank。
2. [NVIDIA CUDA C++ Programming Guide: Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)：LDGSTS / `cp.async` 支持 4、8、16B，16B 时可走 L1 bypass，128B 是推荐对齐粒度之一。
3. [NVIDIA PTX ISA: `cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：`cp.async` 的 `cp-size` 只能是 4、8、16 bytes。
4. [NVIDIA CUDA C++ Programming Guide: Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)：global memory transaction 可以是 32、64、128B。
5. [NVIDIA CUTLASS Python DSL Types: Swizzle](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/types.html#swizzle)：`Swizzle` 参数和 `ComposedLayout` 语义。
6. [NVIDIA CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)：GEMM 的 warp-level 阶段会从 shared memory 取数，shared memory access 应尽量 bank-conflict free。
7. [CuTe 之 Swizzle](https://zhuanlan.zhihu.com/p/671419093)
8. [07_swizzle.md](07_swizzle.md)：本项目内对 CuTe `Swizzle` 的详细解释。
9. [ampere/sgemm.py](ampere/sgemm.py)：padding 版本。
10. [ampere/sgemm_swizzle.py](ampere/sgemm_swizzle.py)：swizzle 版本。
11. [ampere/sgemm_swizzle_hierarchy.py](ampere/sgemm_swizzle_hierarchy.py)：层次化 swizzle 版本。
