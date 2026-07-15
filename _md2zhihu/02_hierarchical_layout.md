# 动手学CuTeDSL 02：分层（嵌套）
`Layout`

普通二维布局把 `shape` 写成 <img src="https://www.zhihu.com/equation?tex=%28M%2C%20N%29" alt="(M, N)" class="ee_img tr_noresize" eeimg="1">、`stride` 写成 <img src="https://www.zhihu.com/equation?tex=%28s_0%2C%20s_1%29" alt="(s_0, s_1)" class="ee_img tr_noresize" eeimg="1"> 即可。CuTe 还允许 **`shape` 与 `stride` 在某一维上再嵌套元组**，用来表达「块 / tile / 子布局」的层次结构：外层下标选大块，内层下标在块内再细分。概念与记号仍以官方说明为准：[CuTe — Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)。

API 上仍用 `cute.make_layout(shape, stride=...)`；嵌套时，`cute.rank` 仍是**最外层维数**，`cute.size` 为元素总数，`cute.depth(layout[k])` 可查看第 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 维是否还带嵌套（深度大于 0 表示该 mode 内部还有子结构）。

下文示例均需在 `@cute.jit` 内执行；先引入：

```python
from cutlass import cute
from cute_viz import display_layout, render_layout_svg
```

---

## 平面二维对比：
<img src="https://www.zhihu.com/equation?tex=%282%2C4%29" alt="(2,4)" class="ee_img tr_noresize" eeimg="1">
 行主序

先固定一个**无嵌套**的 <img src="https://www.zhihu.com/equation?tex=2%5Ctimes%204" alt="2\times 4" class="ee_img tr_noresize" eeimg="1"> 行主序布局，便于与后面分层例子对照：

<img src="https://www.zhihu.com/equation?tex=%5Ctext%7Bshape%7D%3D%282%2C4%29%2C%5Cquad%20%5Ctext%7Bstride%7D%3D%284%2C1%29%2C%5Cquad%20%5Ctext%7Bindex%7D%28i%2Cj%29%3D4i%2Bj%5C%5C" alt="\text{shape}=(2,4),\quad \text{stride}=(4,1),\quad \text{index}(i,j)=4i+j\\" class="ee_img tr_noresize" eeimg="1">

```python
@cute.jit
def example_row_major_2x4():
    layout = cute.make_layout((2, 4), stride=(4, 1))
    display_layout(layout)
```

格内数字为映射到的一维线性索引。

![(2,4) 行主序](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/02_hierarchical_layout/5c820fa40d08e245-02_row_major_2x4.jpg)

---

## 分层形状：
<img src="https://www.zhihu.com/equation?tex=%282%2C%282%2C2%29%29" alt="(2,(2,2))" class="ee_img tr_noresize" eeimg="1">
，步长 
<img src="https://www.zhihu.com/equation?tex=%284%2C%282%2C1%29%29" alt="(4,(2,1))" class="ee_img tr_noresize" eeimg="1">


在上一节**平面** <img src="https://www.zhihu.com/equation?tex=%282%2C4%29" alt="(2,4)" class="ee_img tr_noresize" eeimg="1"> 的基础上，将第二维从「长度为 <img src="https://www.zhihu.com/equation?tex=4" alt="4" class="ee_img tr_noresize" eeimg="1"> 的单层下标」改写成**嵌套的** <img src="https://www.zhihu.com/equation?tex=%282%2C2%29" alt="(2,2)" class="ee_img tr_noresize" eeimg="1">：同一维在逻辑上仍是 <img src="https://www.zhihu.com/equation?tex=2%5Ctimes%202%3D4" alt="2\times 2=4" class="ee_img tr_noresize" eeimg="1"> 个位置，但坐标写成 <img src="https://www.zhihu.com/equation?tex=%28j%2Ck%29" alt="(j,k)" class="ee_img tr_noresize" eeimg="1"> 以强调块内 <img src="https://www.zhihu.com/equation?tex=2%5Ctimes%202" alt="2\times 2" class="ee_img tr_noresize" eeimg="1"> 结构；第一维仍为 <img src="https://www.zhihu.com/equation?tex=i%5Cin%5C%7B0%2C1%5C%7D" alt="i\in\{0,1\}" class="ee_img tr_noresize" eeimg="1">。配合步长 <img src="https://www.zhihu.com/equation?tex=%284%2C%282%2C1%29%29" alt="(4,(2,1))" class="ee_img tr_noresize" eeimg="1">，块与块之间相隔 <img src="https://www.zhihu.com/equation?tex=4" alt="4" class="ee_img tr_noresize" eeimg="1">，块内沿 <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1"> 步长 <img src="https://www.zhihu.com/equation?tex=2" alt="2" class="ee_img tr_noresize" eeimg="1">、沿 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 步长 <img src="https://www.zhihu.com/equation?tex=1" alt="1" class="ee_img tr_noresize" eeimg="1">，共 <img src="https://www.zhihu.com/equation?tex=8" alt="8" class="ee_img tr_noresize" eeimg="1"> 个元素，索引连续铺满 <img src="https://www.zhihu.com/equation?tex=0%5Csim%207" alt="0\sim 7" class="ee_img tr_noresize" eeimg="1">：

<img src="https://www.zhihu.com/equation?tex=%5Ctext%7Bshape%7D%3D%282%2C%282%2C2%29%29%2C%5Cquad%20%5Ctext%7Bstride%7D%3D%284%2C%282%2C1%29%29%2C%5Cquad%20%5Ctext%7Bindex%7D%28i%2C%28j%2Ck%29%29%3D4i%2B2j%2Bk%5C%5C" alt="\text{shape}=(2,(2,2)),\quad \text{stride}=(4,(2,1)),\quad \text{index}(i,(j,k))=4i+2j+k\\" class="ee_img tr_noresize" eeimg="1">

关于嵌套layout有多种不同的解读方式，比如这个例子。

第一种解读方式：块下标是最左下标 <img src="https://www.zhihu.com/equation?tex=i%5Cin%5C%7B0%2C1%5C%7D" alt="i\in\{0,1\}" class="ee_img tr_noresize" eeimg="1">， 选**上下两大行块**；每个块内再用最右下标 <img src="https://www.zhihu.com/equation?tex=%28j%2Ck%29" alt="(j,k)" class="ee_img tr_noresize" eeimg="1"> 表示 <img src="https://www.zhihu.com/equation?tex=2%5Ctimes%202" alt="2\times 2" class="ee_img tr_noresize" eeimg="1"> 子网格的块内元素下标。

第二种解读方式：块下标是最右下标 <img src="https://www.zhihu.com/equation?tex=%28j%2Ck%29" alt="(j,k)" class="ee_img tr_noresize" eeimg="1"> ，选 <img src="https://www.zhihu.com/equation?tex=2%5Ctimes2" alt="2\times2" class="ee_img tr_noresize" eeimg="1"> 4个子块；每个块内再用最左下标 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 表示块内元素。

```python
@cute.jit
def example_shape_2_2x2():
    layout = cute.make_layout((2, (2, 2)), stride=(4, (2, 1)))
    # cute.rank(layout) == 2，cute.depth(layout[1]) 为 1

    display_layout(layout, flatten_hierarchical=False)
    display_layout(layout, flatten_hierarchical=True)
```

注意 `cute-viz` 总是将最左下标当做块内下标，这对应了前面的第一种解读方式。

**cute-viz** 对分层布局有两种画法（均通过 `display_layout` / `render_layout_svg` 的 `flatten_hierarchical` 控制）：

-   **`flatten_hierarchical=False`**：画出 **tile块 边界**（粗蓝线），外层块下标（蓝色）与块内层元素下标（黑色）用不同颜色区分，突出层次。

![(2,(2,2)) 嵌套边界](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/02_hierarchical_layout/c91d4cddbdceea4d-02_hierarchical_2_2x2_nested.jpg)

-   **`flatten_hierarchical=True`**：压成一张平面网格。左侧行号是最左下标 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">；顶部列号不是原始的二维坐标 <img src="https://www.zhihu.com/equation?tex=%28j%2Ck%29" alt="(j,k)" class="ee_img tr_noresize" eeimg="1">，而是 <img src="https://www.zhihu.com/equation?tex=%28j%2Ck%29" alt="(j,k)" class="ee_img tr_noresize" eeimg="1"> 被压平后的一维坐标。二维坐标和一维坐标的转换关系见下一节。

![(2,(2,2)) 扁平网格](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/02_hierarchical_layout/32c820983d48b2d7-02_hierarchical_2_2x2_flat.jpg)

## Layout维护两种映射：坐标映射和索引映射

前面已经提到：**Layout** 描述的是「逻辑多维坐标」到「一维地址（或索引）」的映射。

但在嵌套 Layout 里，坐标本身也可能有多种写法。以上一节的布局为例：

<img src="https://www.zhihu.com/equation?tex=%5Ctext%7Bshape%7D%3D%282%2C%282%2C2%29%29%2C%5Cquad%20%5Ctext%7Bstride%7D%3D%284%2C%282%2C1%29%29%2C%5Cquad%20%5Ctext%7Bindex%7D%28i%2C%28j%2Ck%29%29%3D4i%2B2j%2Bk%5C%5C" alt="\text{shape}=(2,(2,2)),\quad \text{stride}=(4,(2,1)),\quad \text{index}(i,(j,k))=4i+2j+k\\" class="ee_img tr_noresize" eeimg="1">

Layout同时维护两类映射：

-   **坐标映射**：把合法但不完全展开的坐标（例如 <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 或 <img src="https://www.zhihu.com/equation?tex=%28i%2Cq%29" alt="(i,q)" class="ee_img tr_noresize" eeimg="1">），转换成和 `shape` 对齐的自然坐标。这里的自然坐标是 <img src="https://www.zhihu.com/equation?tex=%28i%2C%28j%2Ck%29%29" alt="(i,(j,k))" class="ee_img tr_noresize" eeimg="1">。转换规则是反字典序，也就是 LayoutLeft 列主序：最左侧叶子下标变化最快。
-   **索引映射**：把自然坐标转换成线性的 index。做法是将自然坐标和 `stride` 逐项相乘再求和；在这个例子里就是 <img src="https://www.zhihu.com/equation?tex=4i%2B2j%2Bk" alt="4i+2j+k" class="ee_img tr_noresize" eeimg="1">。

因此，`Layout` 的完整过程可以理解为：非自然坐标（例如 <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 或 <img src="https://www.zhihu.com/equation?tex=%28i%2Cq%29" alt="(i,q)" class="ee_img tr_noresize" eeimg="1">）先通过坐标映射还原为自然坐标 <img src="https://www.zhihu.com/equation?tex=%28i%2C%28j%2Ck%29%29" alt="(i,(j,k))" class="ee_img tr_noresize" eeimg="1">，再通过索引映射计算 <img src="https://www.zhihu.com/equation?tex=4i%2B2j%2Bk" alt="4i+2j+k" class="ee_img tr_noresize" eeimg="1">，最终得到线性 index。

第一个映射先解决「坐标怎么写」的问题。对 `shape=(2,(2,2))` 来说，1-D 坐标 <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 和 2-D 坐标 <img src="https://www.zhihu.com/equation?tex=%28i%2Cq%29" alt="(i,q)" class="ee_img tr_noresize" eeimg="1"> 都是合法写法，但它们不是最终参与 `stride` 计算的自然坐标。它们会先按 LayoutLeft 还原：

<table>
<tr class="header">
<th>1-D坐标<span class="math display"><em>n</em></span></th>
<th>2-D坐标<span class="math display">(<em>i</em>,<em>q</em>)</span></th>
<th>自然坐标<span class="math display">(<em>i</em>,(<em>j</em>,<em>k</em>))</span></th>
<th>index</th>
</tr>
<tr class="odd">
<td><code>0</code></td>
<td><code>(0,0)</code></td>
<td><code>(0,(0,0))</code></td>
<td><code>0</code></td>
</tr>
<tr class="even">
<td><code>1</code></td>
<td><code>(1,0)</code></td>
<td><code>(1,(0,0))</code></td>
<td><code>4</code></td>
</tr>
<tr class="odd">
<td><code>2</code></td>
<td><code>(0,1)</code></td>
<td><code>(0,(1,0))</code></td>
<td><code>2</code></td>
</tr>
<tr class="even">
<td><code>3</code></td>
<td><code>(1,1)</code></td>
<td><code>(1,(1,0))</code></td>
<td><code>6</code></td>
</tr>
<tr class="odd">
<td><code>4</code></td>
<td><code>(0,2)</code></td>
<td><code>(0,(0,1))</code></td>
<td><code>1</code></td>
</tr>
<tr class="even">
<td><code>5</code></td>
<td><code>(1,2)</code></td>
<td><code>(1,(0,1))</code></td>
<td><code>5</code></td>
</tr>
<tr class="odd">
<td><code>6</code></td>
<td><code>(0,3)</code></td>
<td><code>(0,(1,1))</code></td>
<td><code>3</code></td>
</tr>
<tr class="even">
<td><code>7</code></td>
<td><code>(1,3)</code></td>
<td><code>(1,(1,1))</code></td>
<td><code>7</code></td>
</tr>
</table>

这张表也解释了上一节 `flatten_hierarchical=True` 的可视化：

-   图被压成 <img src="https://www.zhihu.com/equation?tex=2%5Ctimes4" alt="2\times4" class="ee_img tr_noresize" eeimg="1"> 的平面网格。
-   左侧行号 `0,1` 对应最左下标 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">。
-   顶部列号 `0,1,2,3` 对应被压平的块坐标 <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">，其中 <img src="https://www.zhihu.com/equation?tex=q%3Dj%2B2k" alt="q=j+2k" class="ee_img tr_noresize" eeimg="1">。所以列 `0,1,2,3` 依次对应 <img src="https://www.zhihu.com/equation?tex=%28j%2Ck%29%3D%280%2C0%29%2C%281%2C0%29%2C%280%2C1%29%2C%281%2C1%29" alt="(j,k)=(0,0),(1,0),(0,1),(1,1)" class="ee_img tr_noresize" eeimg="1">。
-   格子里的数字不是坐标，而是第二步索引映射的结果：<img src="https://www.zhihu.com/equation?tex=4i%2B2j%2Bk" alt="4i+2j+k" class="ee_img tr_noresize" eeimg="1">。所以第一行是 `0,2,1,3`，第二行是 `4,6,5,7`。

所以，`flatten_hierarchical=True` 并没有改变 Layout 的索引规则；它只是先把嵌套坐标 <img src="https://www.zhihu.com/equation?tex=%28j%2Ck%29" alt="(j,k)" class="ee_img tr_noresize" eeimg="1"> 按坐标映射压成一维 <img src="https://www.zhihu.com/equation?tex=q" alt="q" class="ee_img tr_noresize" eeimg="1">，再把每个自然坐标对应的 index 填到平面网格里。

## 更多嵌套layout的例子

---

### 分层形状：
<img src="https://www.zhihu.com/equation?tex=%28%282%2C2%29%2C2%29" alt="((2,2),2)" class="ee_img tr_noresize" eeimg="1">


Shape 为 <img src="https://www.zhihu.com/equation?tex=%28%282%2C2%29%2C2%29" alt="((2,2),2)" class="ee_img tr_noresize" eeimg="1">，逻辑坐标写作 <img src="https://www.zhihu.com/equation?tex=%28%28i%2Cj%29%2Ck%29" alt="((i,j),k)" class="ee_img tr_noresize" eeimg="1">：第一维是嵌套的 <img src="https://www.zhihu.com/equation?tex=%282%2C2%29" alt="(2,2)" class="ee_img tr_noresize" eeimg="1">，第二维是 <img src="https://www.zhihu.com/equation?tex=k%5Cin%5C%7B0%2C1%5C%7D" alt="k\in\{0,1\}" class="ee_img tr_noresize" eeimg="1">。默认（列主序）紧凑布局下常见打印为 <img src="https://www.zhihu.com/equation?tex=%28%282%2C2%29%2C2%29%3A%28%281%2C2%29%2C4%29" alt="((2,2),2):((1,2),4)" class="ee_img tr_noresize" eeimg="1">。

与 <img src="https://www.zhihu.com/equation?tex=%282%2C%282%2C2%29%29" alt="(2,(2,2))" class="ee_img tr_noresize" eeimg="1"> 不同，**嵌套写在 shape 的左侧**（第一维），而非右侧。同样可以有两种叙述角度：

其一，先按 <img src="https://www.zhihu.com/equation?tex=%28i%2Cj%29" alt="(i,j)" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=2%5Ctimes2" alt="2\times2" class="ee_img tr_noresize" eeimg="1"> 上分区，然后在每个分区内用 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 排布；

其二，先按 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> 把整体分成两半，再在每半内用 <img src="https://www.zhihu.com/equation?tex=%28i%2Cj%29" alt="(i,j)" class="ee_img tr_noresize" eeimg="1"> 排布。

**cute-viz** 嵌套视图可视化仍与上一节一致：把**坐标中最左的下标**按块内层处理，并且**图中蓝字为块坐标，黑字为块内坐标**。

```python
@cute.jit
def example_shape_2x2_2():
    layout = cute.make_layout(((2, 2), 2))

    display_layout(layout, flatten_hierarchical=False)
    display_layout(layout, flatten_hierarchical=True)
```

![((2,2),2) 嵌套边界](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/02_hierarchical_layout/cf7e833e0a6266d3-02_hierarchical_2x2_2_nested.jpg)

![((2,2),2) 扁平网格](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/02_hierarchical_layout/b2356a4954798da3-02_hierarchical_2x2_2_flat.jpg)

---

### 更大分层块网格与非线性步长：
<img src="https://www.zhihu.com/equation?tex=%28%282%2C2%29%2C%283%2C4%29%29" alt="((2,2),(3,4))" class="ee_img tr_noresize" eeimg="1">


令

<img src="https://www.zhihu.com/equation?tex=%5Ctext%7Bshape%7D%3D%28%282%2C2%29%2C%283%2C4%29%29%2C%5Cquad%20%5Ctext%7Bstride%7D%3D%28%281%2C6%29%2C%282%2C12%29%29%5C%5C" alt="\text{shape}=((2,2),(3,4)),\quad \text{stride}=((1,6),(2,12))\\" class="ee_img tr_noresize" eeimg="1">

逻辑坐标写作 <img src="https://www.zhihu.com/equation?tex=%28%28i%2Cj%29%2C%28m%2Cn%29%29" alt="((i,j),(m,n))" class="ee_img tr_noresize" eeimg="1">：两个 mode 都是嵌套维，步长 <img src="https://www.zhihu.com/equation?tex=%28%281%2C6%29%2C%282%2C12%29%29" alt="((1,6),(2,12))" class="ee_img tr_noresize" eeimg="1"> 。

口头描述既可以强调「先有一块 <img src="https://www.zhihu.com/equation?tex=3%5Ctimes4" alt="3\times4" class="ee_img tr_noresize" eeimg="1"> 的块上网格，再在每个块里细分 <img src="https://www.zhihu.com/equation?tex=2%5Ctimes2" alt="2\times2" class="ee_img tr_noresize" eeimg="1">」，也可以调换叙述顺序。

cute-viz可视化工具规则仍与前面相同：**最左下标**在可视化里按块内层处理，图中粗蓝线包围。**读图时以蓝字（块坐标）与黑字（块内坐标）为准**。

```python
@cute.jit
def example_shape_2x2_3x4():
    layout = cute.make_layout(((2, 2), (3, 4)), stride=((1, 6), (2, 12)))

    display_layout(layout, flatten_hierarchical=False)
    display_layout(layout, flatten_hierarchical=True)
```

![((2,2),(3,4)) 嵌套边界](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/02_hierarchical_layout/4b09028fb8041bdb-02_hierarchical_2x2_3x4_nested.jpg)

![((2,2),(3,4)) 扁平网格](https://gitee.com/henryzhao28/blog/raw/main-md2zhihu-asset/02_hierarchical_layout/1db0bf77ed2a9b23-02_hierarchical_2x2_3x4_flat.jpg)

---

## 小结

这一篇把普通 `Layout` 扩展到嵌套 `shape` / `stride`：嵌套不是新的机制，而是用同一套坐标到索引的规则，表达更有层次感的块结构。

-   嵌套 `shape` / `stride` 仍然用 `cute.make_layout` 构造；`rank` 看最外层维数，`size` 看元素总数，`depth` 看某个 mode 内部是否还有子结构。
-   嵌套 Layout 同时维护两类映射：先把 1-D、2-D 等非自然坐标按 LayoutLeft 规则还原为自然坐标，再用自然坐标和 `stride` 计算线性 index。
-   `flatten_hierarchical=False` 保留层次边界，图中蓝字表示块坐标，黑字表示块内坐标；这适合观察 tile 的分层结构。
-   `flatten_hierarchical=True` 把嵌套坐标压成平面坐标，但不会改变 Layout 的索引规则；它只是把自然坐标对应的 index 填到一个更容易对照的平面网格里。
-   同一个嵌套布局可以有多种叙述角度。读图时以坐标映射和索引映射为准，不必把 cute-viz 的展示顺序当成唯一解释。



Reference:

