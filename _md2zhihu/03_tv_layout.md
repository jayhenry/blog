# 动手学CuTeDSL 03：Thread–Value（TV）Layout 与可视化

**Thread–value partitioning** 描述「每个线程持有哪些槽位」：把线程下标与线程内的 value 下标一起，映射到某个数据块（tile）上。它的作用是方便用线程下标和线程内value下标，来得到某个元素在这个数据块上的一维线性地址。

它通常再与张量自身的内存 layout 做复合运算，得到实际访存模式。概念与记号以官方说明为准：[CuTe — Tensor：Thread–value partitioning](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html#thread-value-partitioning)。

在 CuTe DSL 中，常用 `cute.make_layout_tv(thr_layout, val_layout)` 由**线程布局**与**每线程 value 布局**生成：

-   `tiler_mn`：这一组线程覆盖的全体数据形成的二维 tile 形状，通常记为 <img src="https://www.zhihu.com/equation?tex=%28M%2CN%29" alt="(M,N)" class="ee_img tr_noresize" eeimg="1">
-   **TV layout**：把线程下标和线程内value下标 <img src="https://www.zhihu.com/equation?tex=%28t%2Cv%29" alt="(t,v)" class="ee_img tr_noresize" eeimg="1"> 映到 tile 内一维线性索引

```python
from cutlass import cute
import cutlass
from cute_viz import render_tv_layout_svg, display_tv_layout
```

---

## 列主序风格的线程网格：
`make_layout_tv`

看一个具体的例子：

-   取 <img src="https://www.zhihu.com/equation?tex=4%5Ctimes%205" alt="4\times 5" class="ee_img tr_noresize" eeimg="1"> 的线程网格，步长 <img src="https://www.zhihu.com/equation?tex=%281%2C4%29" alt="(1,4)" class="ee_img tr_noresize" eeimg="1">，即二维线程编号按**列主序**展开；
-   每个线程再持有 <img src="https://www.zhihu.com/equation?tex=2%5Ctimes%203" alt="2\times 3" class="ee_img tr_noresize" eeimg="1"> 个 value，value 子布局步长 <img src="https://www.zhihu.com/equation?tex=%281%2C2%29" alt="(1,2)" class="ee_img tr_noresize" eeimg="1">。

所以，这20个线程一共持有 <img src="https://www.zhihu.com/equation?tex=8%5Ctimes15" alt="8\times15" class="ee_img tr_noresize" eeimg="1"> 个value 的块（tile）。

二者通过 `make_layout_tv` 合成后，典型打印为：

-   tile <img src="https://www.zhihu.com/equation?tex=%288%2C15%29" alt="(8,15)" class="ee_img tr_noresize" eeimg="1">
-   TV layout 形状 <img src="https://www.zhihu.com/equation?tex=%28%284%2C5%29%2C%282%2C3%29%29" alt="((4,5),(2,3))" class="ee_img tr_noresize" eeimg="1">，步长 <img src="https://www.zhihu.com/equation?tex=%28%282%2C24%29%2C%281%2C8%29%29" alt="((2,24),(1,8))" class="ee_img tr_noresize" eeimg="1">

下例打印 <img src="https://www.zhihu.com/equation?tex=%28T_i%2CV_j%29" alt="(T_i,V_j)" class="ee_img tr_noresize" eeimg="1"> 到 tile 坐标的对应关系，并输出可视化关系图。

```python
@cute.jit
def example_col_major():
    thr_layout = cute.make_layout((4, 5), stride=(1, 4))
    val_layout = cute.make_layout((2, 3), stride=(1, 2))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    thread_size = cute.size(thr_layout)
    value_size = cute.size(val_layout)
    m, n = tiler_mn
    for i in cutlass.range_constexpr(thread_size):
        for j in cutlass.range_constexpr(value_size):
            idx = tv_layout((i, j))
            tiler_i = idx % m
            tiler_j = idx // m
            print(f"(T{i}, V{j})->({tiler_i},{tiler_j})={idx}", end=" ")
        print()

    display_tv_layout(tv_layout, tiler_mn)
```

图中横纵轴为 tile 上的坐标网格，每个格内标注该位置对应的线程号与 value 号（颜色区分线程）。

比如线程 T1 中 V1 元素 (T1,V1) 在 tile 中的二维下标为 (3,0)，由于是列主序，其一维索引为 3。

![列主序线程网格对应的 TV 布局（$8\times 15$ tile）](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/03_tv_layout/ae780057e5fd58d8-03_col_major_tv.jpg)

---

## 与 
`make_layout_tv`
 等价的显式 TV layout

同一 TV 关系也可以直接写出分层 shape / stride，无需先构造 `thr_layout`与 `val_layout`：

<img src="https://www.zhihu.com/equation?tex=%5Ctext%7Bshape%7D%3D%28%284%2C5%29%2C%282%2C3%29%29%2C%5Cquad%20%5Ctext%7Bstride%7D%3D%28%282%2C24%29%2C%281%2C8%29%29%5C%5C" alt="\text{shape}=((4,5),(2,3)),\quad \text{stride}=((2,24),(1,8))\\" class="ee_img tr_noresize" eeimg="1">

```python
@cute.jit
def example_col_major_manual():
    tiler_mn = (8, 15)
    tv_layout = cute.make_layout(((4, 5), (2, 3)), stride=((2, 24), (1, 8)))
    thread_size = cute.size(tv_layout[0])
    value_size = cute.size(tv_layout[1])
    m, n = tiler_mn
    for i in cutlass.range_constexpr(thread_size):
        for j in cutlass.range_constexpr(value_size):
            idx = tv_layout((i, j))
            tiler_i = idx % m
            tiler_j = idx // m
            print(f"(T{i}, V{j})->({tiler_i},{tiler_j})={idx}", end=" ")
        print()
    display_tv_layout(tv_layout, tiler_mn)
```

与前一节语义相同（仅构造路径不同），`display_tv_layout` 得到的图与上一节一致。

---

## tv layout的实际应用模式以及layout复合函数

参考：https://docs.nvidia.com/cutlass/media/docs/cpp/cute/03_tensor.html#thread-value-partitioning

实际使用场景：需要访问一个矩阵A，不同thread访问矩阵A的不同部分。

```code
foreach thread_idx:
  foreach value_idx_in_cur_thread:
    idx = tv_compute_idx(thread_idx, value_idx)
    A(idx)
```

其中，`A(idx)` 从 `idx` 到实际内存地址的映射，由 Tensor A 自身的 layout（行主序、列主序，或者 A 是某个更大矩阵的分片等）负责。

在 CuTe 中，一个 Tensor 可以看作两个部分的复合：

1.  **Engine**（记作 <img src="https://www.zhihu.com/equation?tex=E" alt="E" class="ee_img tr_noresize" eeimg="1">）：类似可随机访问指针的对象，支持两个基本操作：
    -   偏移操作：`e + d -> e`，按照 layout 值域中的元素偏移 engine；
    -   解引用操作：`*e -> v`，读取 engine 当前指向位置的值。

1.  **Layout**（记作 <img src="https://www.zhihu.com/equation?tex=L" alt="L" class="ee_img tr_noresize" eeimg="1">）：定义从逻辑坐标到偏移量的映射。

形式上，Tensor 是 Engine 和 Layout 的复合，可以写成 <img src="https://www.zhihu.com/equation?tex=T%20%3D%20E%20%5Ccirc%20L" alt="T = E \circ L" class="ee_img tr_noresize" eeimg="1">。当用坐标 <img src="https://www.zhihu.com/equation?tex=c" alt="c" class="ee_img tr_noresize" eeimg="1"> 访问 Tensor 时，会先用 layout 把 <img src="https://www.zhihu.com/equation?tex=c" alt="c" class="ee_img tr_noresize" eeimg="1"> 映射到偏移量，再把 engine 偏移到对应位置，最后解引用得到值：

<img src="https://www.zhihu.com/equation?tex=T%28c%29%20%3D%20%28E%20%5Ccirc%20L%29%28c%29%20%3D%20%2A%28E%20%2B%20L%28c%29%29%5C%5C" alt="T(c) = (E \circ L)(c) = *(E + L(c))\\" class="ee_img tr_noresize" eeimg="1">

所以，上面的 `A(idx)` 不是单纯的数组下标访问，而是一次 Tensor 访问：`idx` 先经过 Tensor A 自身的 layout 变成 offset，再由 A 的 engine 找到并读取真实值。

而 `tv_compute_idx(thread_idx, value_idx)` 则由 TV layout 负责。
因为先应用 `tv_compute_idx()`，再应用 `A()`，所以也可以先构造这两个映射的复合函数，然后直接用这个复合函数来寻址：

```code
TV_A(thread_idx, value_idx) = A(tv_compute_idx(thread_idx, value_idx))
```

下面两个例子先后展示了两种做法，使用的也是参考链接中的数据。

```python
from cutlass.cute.runtime import from_dlpack
import torch

@cute.jit
def tensor_tv_layout(a: cute.Tensor):
    # Tensor: (M4,N8)
    cute.printf("a = ")
    cute.print_tensor(a)
    # Construct a TV-layout that maps 8 thread indices and 4 value indices
    # to 1D coordinates within a 4x8 tensor
    # tv_layout: (T8,V4) 
    tv_layout = cute.make_layout(((2,4), (2,2)), stride=((8,1), (4,16)))
    print(f"tv_layout = {tv_layout}")
    # tv_a = cute.composition(a, tv_layout)
    # cute.printf("tv_a = ")
    # cute.print_tensor(tv_a)

    for thread_idx in range(8):
        #                           value_idx
        v0 = a[tv_layout((thread_idx, 0))]
        v1 = a[tv_layout((thread_idx, 1))]
        v2 = a[tv_layout((thread_idx, 2))]
        v3 = a[tv_layout((thread_idx, 3))]
        cute.printf("thread_idx = {}, tv_a slice = {}, {}, {}, {}", thread_idx, v0, v1, v2, v3)

def tv_layout_app():
    a = torch.arange(32).reshape(8,4).transpose(0,1)
    print(f"torch tensor = \n{a}")
    tensor_tv_layout(from_dlpack(a))
```

```python
tv_layout_app()
```

第二个例子使用复合函数 `cute.compose` 来做同样的事情。

```python
from cutlass.cute.runtime import from_dlpack
import torch

@cute.jit
def tensor_tv_layout(a: cute.Tensor):
    # Tensor: (M4,N8)
    cute.printf("a = ")
    cute.print_tensor(a)
    # Construct a TV-layout that maps 8 thread indices and 4 value indices
    # to 1D coordinates within a 4x8 tensor
    # tv_layout: (T8,V4) 
    tv_layout = cute.make_layout(((2,4), (2,2)), stride=((8,1), (4,16)))
    print(f"tv_layout = {tv_layout}")
    tv_a = cute.composition(a, tv_layout)
    cute.printf("tv_a = ")
    cute.print_tensor(tv_a)

    for thread_idx in range(8):
        cute.printf("thread_idx = {}, tv_a slice = {}", thread_idx, tv_a[(thread_idx, None)])

def tv_layout_app2():
    a = torch.arange(32).reshape(8,4).transpose(0,1)
    print(f"torch tensor = \n{a}")
    tensor_tv_layout(from_dlpack(a))
```

```python
tv_layout_app2()
```

---

## 行主序风格的线程网格

将线程布局改为行主序 <img src="https://www.zhihu.com/equation?tex=%284%2C5%29%3A%285%2C1%29" alt="(4,5):(5,1)" class="ee_img tr_noresize" eeimg="1">，value 布局 <img src="https://www.zhihu.com/equation?tex=%282%2C3%29%3A%283%2C1%29" alt="(2,3):(3,1)" class="ee_img tr_noresize" eeimg="1">，则 `make_layout_tv` 给出的 TV layout 形如 <img src="https://www.zhihu.com/equation?tex=%28%285%2C4%29%2C%283%2C2%29%29%3A%28%2824%2C2%29%2C%288%2C1%29%29" alt="((5,4),(3,2)):((24,2),(8,1))" class="ee_img tr_noresize" eeimg="1">，tile 仍为 <img src="https://www.zhihu.com/equation?tex=%288%2C15%29" alt="(8,15)" class="ee_img tr_noresize" eeimg="1">。画图与打印方式与列主序例子相同。

```python
@cute.jit
def example_row_major():
    thr_layout = cute.make_layout((4, 5), stride=(5, 1))
    val_layout = cute.make_layout((2, 3), stride=(3, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"TV Layout: {tv_layout}")  # ((5,4),(3,2)):((24,2),(8,1))

    thread_size = cute.size(thr_layout)
    value_size = cute.size(val_layout)
    m, n = tiler_mn
    for i in cutlass.range_constexpr(thread_size):
        for j in cutlass.range_constexpr(value_size):
            idx = tv_layout((i, j))
            tiler_i = idx % m
            tiler_j = idx // m
            print(f"(T{i}, V{j})->({tiler_i},{tiler_j})={idx}", end=" ")
        print()

    display_tv_layout(tv_layout, tiler_mn)
```

![行主序线程网格对应的 TV 布局](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/03_tv_layout/22928707ba6dbf02-03_row_major_tv.jpg)

---

## 为何 TV layout 里线程域的「形状」和图上排列不一致？

行主序一例中，打印出的 TV layout `((5,4),(3,2)):((24,2),(8,1))` 线程半区形状是 <img src="https://www.zhihu.com/equation?tex=%285%2C4%29" alt="(5,4)" class="ee_img tr_noresize" eeimg="1">，而 `display_tv_layout`仍按原始`thr_layout` 的 <img src="https://www.zhihu.com/equation?tex=%284%2C5%29" alt="(4,5)" class="ee_img tr_noresize" eeimg="1"> 网格直觉来排布线程块。类似地，TV layout中value形状是 <img src="https://www.zhihu.com/equation?tex=%283%2C2%29" alt="(3,2)" class="ee_img tr_noresize" eeimg="1">，而展示出来也仍按原始 `val_layout`的 <img src="https://www.zhihu.com/equation?tex=%282%2C3%29" alt="(2,3)" class="ee_img tr_noresize" eeimg="1">来排布。这是为什么？

CuTeDSL 官方Notebook示例：[elementwise_add.ipynb](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/notebooks/elementwise_add.ipynb) 中 “***Why modes of thread domain of TV Layout looks swapped especially when tensor is row major?***” 一节的讨论做了下面的回答。

> It's important to keep in mind that *TV Layout* maps `(thread_index, value_index)` to `(row_index, column_index)` of logical domain `(TileM, TileN)`. However, visualization shows **inverse** mapping of logical domain `(TileM, TileN)` to `(thread_domain, value_domain)`, because this is more intuitive for human developer.
> 
> That's why the shape of domain of *TV Layout* doesn't necessarily match logical view.


原因在于**TV layout 定义的是** $(\text{thread_index},\text{value_index})\to(\text{tile 内坐标})$**的正向映射**；**可视化则采用从 tile 逻辑域到「线程域 × value 域」的逆映射**，便于人眼对照「这一格是谁的哪个 value」。

因此，layout 元组里 thread 部分的 shape 不必与图上线程块的二维排布数字顺序一致。

上面官方的解释不容易理解，这里再做更通俗的解释。

前两篇文章，我们看到了很多普通 layout 的可视化，但是注意前面可视化图中横纵坐标就是layout的坐标，而网格中数值为layout的一维索引。关于坐标和索引的概念，见上一篇文章“坐标映射和索引映射”一节，这里也可以简单理解为坐标是layout的输入，一维索引是layout的输出。

而 TV Layout 可视化的特别之处在于，图中横纵坐标不再是layout的坐标，因此它不必与layout的坐标形状一致。实际上，图中的横纵坐标是layout输出的索引，并且是将一维索引按列主序展开成二维索引。

为什么是按“列主序”展开成二维索引，而不是按“行主序”展开呢？

在前面实际应用模式中，我们看到 TV layout 输出的索引，本身是要传递给Tensor A自身的layout，作为Tensor A layout的输入坐标。假设A layout是二维矩阵，从A的视角看它的输入坐标从二维到一维的转换关系是“列主序”的。这再次使用到了上一篇文章“坐标映射和索引映射”的基础知识。

---

## 高维嵌套的 thread / value：
<img src="https://www.zhihu.com/equation?tex=8%5Ctimes%208" alt="8\times 8" class="ee_img tr_noresize" eeimg="1">
 示例

下面与 [cute-viz 的 tv_layout 示例](https://github.com/NTT123/cute-viz/blob/main/examples/tv_layout_example.py) 类似：thread 与 value 各自为 <img src="https://www.zhihu.com/equation?tex=%282%2C2%2C2%29" alt="(2,2,2)" class="ee_img tr_noresize" eeimg="1"> 的三维嵌套，通过给定 stride 定义到 <img src="https://www.zhihu.com/equation?tex=8%5Ctimes%208" alt="8\times 8" class="ee_img tr_noresize" eeimg="1"> tile 的映射。可视化图如下。这不再是简单的连续排布，而是线程和value互相交错的一种排布，后面可以看到一些tensor core的mma算子会要求类似地排布。

```python
@cute.jit
def example_tv_8x8():
    tile_mn = (8, 8)
    tv_layout = cute.make_layout(
        shape=((2, 2, 2), (2, 2, 2)),
        stride=((1, 16, 4), (8, 2, 32)),
    )
    thread_size = cute.size(tv_layout[0])
    value_size = cute.size(tv_layout[1])
    m, n = tile_mn
    for i in cutlass.range_constexpr(thread_size):
        for j in cutlass.range_constexpr(value_size):
            idx = tv_layout((i, j))
            tiler_i = idx % m
            tiler_j = idx // m
            print(f"(T{i}, V{j})->({tiler_i},{tiler_j})={idx}", end=" ")
        print()
    display_tv_layout(tv_layout, tile_mn)
```

![$8\times 8$ tile上三维 thread × 三维 value 的 TV 布局](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/03_tv_layout/4c4cd4e194a9a08e-03_tv_8x8_nested_dims.jpg)

---

## 小结

这一篇把 `Layout` 用到线程和值的分配上：TV layout 描述的是每个线程、每个线程内 value，分别落到 tile 的哪个位置。

-   `cute.make_layout_tv(thr_layout, val_layout)` 会把线程布局和每线程 value 布局合成为一个 TV layout，形式上是 $(\text{thread_idx},\text{value_idx}) \to \text{tile index}$。
-   实际访存时，TV layout 的输出还要交给 Tensor 自身的 layout；也就是先决定「访问 tile 中哪个元素」，再由 Tensor 决定「这个元素在真实内存哪里」。
-   `cute.composition(a, tv_layout)` 可以把 TV layout 和 Tensor A 的访问规则复合起来，直接得到按 thread / value 视角访问的 `tv_a`。
-   TV layout 的可视化画的是从 tile 网格反查到 thread / value 的关系，因此图上的二维排布不一定等于 TV layout 元组里 thread 域、value 域的 shape。
-   高维嵌套的 thread / value 布局可以表达更复杂的交错访问模式，是后续理解 MMA、ldmatrix 和高效 GEMM 数据排布的基础。



Reference:

