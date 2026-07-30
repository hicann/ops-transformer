# FA

## 核心交付件
1. `csrc/flash_attn_minimal/op_kernel` 算子kernel实现
2. `csrc/flash_attn_minimal/torch_interface.cpp` 算子pytorch层和C++实现层接口
3. `csrc/flash_attn_minimal/CMakeLists.txt` 算子cmake配置

## 计算公式

self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

$$
Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
$$

其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

**说明**：
<blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
<br>Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>)的计算结果。</blockquote>


## 接口

```python
torch.ops.ascend_ops.flash_attn_minimal(query, key, value, softmaxScale=0) -> Tensor
```

计算稠密（全量）自注意力，输出与 query 同形状。

## 接口参数说明

- <a id="query"></a>**query/key/value**

    输入布局为 4 维 `BSND`（Batch, Seq-Length, Head-Num, Head-Dim），当前算子对形状与数据类型的约束如下：

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 500px">
        <col style="width: 342px">
        </colgroup>
        <thead>
            <tr>
                <th>属性</th>
                <th>含义</th>
                <th>约束</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>Batch(B)</td>
            <td>输入样本批量大小</td>
            <td>Q、K、V 的 batch 维度一致</td>
        </tr>
        <tr>
            <td>Head-Num(N)</td>
            <td>多头数</td>
            <td>Q_N ≥ KV_N，且 Q_N 被 KV_N 整除（G = Q_N / KV_N 为整数，支持 GQA）</td>
        </tr>
        <tr>
            <td>Seq-Length(S)</td>
            <td>输入样本序列长度</td>
            <td>Q_S = KV_S，且被 128 整除</td>
        </tr>
        <tr>
            <td>Head-Dim(D)</td>
            <td>隐藏层最小的单元尺寸</td>
            <td>固定为 128</td>
        </tr>
        <tr>
            <td>数据类型</td>
            <td>Q、K、V 矩阵中的数据类型</td>
            <td>只支持 bfloat16</td>
        </tr>
        </tbody>
    </table>

    典型用例：`B=1, N1=32, N2=1, S1=8192, S2=8192, D=128`（GQA, G=32）。**不支持空 Tensor 传入。**

- <a id="softmaxScale"></a>**softmaxScale**

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 500px">
        <col style="width: 342px">
        </colgroup>
        <thead>
            <tr>
                <th>属性</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>float</td>
            <td>Q 与 K 矩阵相乘之后的缩放系数，等于 1 表示不缩放，默认为 0</td>
            <td>常取 1/√D</td>
        </tr>
        </tbody>
    </table>

## 代码架构
算子执行在 Device 侧（NPU，执行 Kernel 函数），输入由 Host 侧（CPU）准备并拉起 Kernel。代码主要分为如下两部分：
 - **csrc/flash_attn_minimal/torch_interface.cpp** 包含算子执行的整体流程：上层 python 入参传递到该文件，校验输入合法性后直接拉起算子的 Kernel 函数执行，无独立的 Host Tiling 阶段。
 - **csrc/flash_attn_minimal/op_kernel** 算子的 Kernel 函数所有文件都在该目录下。

## 内部实现文档

欲了解算子的内存层级分配、多核切分策略、跨核同步及典型 Case 走读，请参阅 [FA_IMPL.md](./FA_IMPL.md)。
