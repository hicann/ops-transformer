# FA

English | [简体中文](./README.md)

## Core Deliverables
1. `csrc/flash_attn_minimal/op_kernel`: operator kernel implementation
2. `csrc/flash_attn_minimal/torch_interface.cpp`: interface between the PyTorch layer and the C++ implementation layer of the operator
3. `csrc/flash_attn_minimal/CMakeLists.txt`: operator CMake configuration

## Computation Formula

self-attention builds an attention model using the relationships within the input sample itself. The principle is as follows: assume an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. The sequence is transformed by 3 weight matrices to obtain 3 matrices of dimension $n*d$.

The computation formula of self-attention is generally defined as follows, where $Q$, $K$, and $V$ are important attribute elements of the input sample. They are obtained by spatial transformation of the input sample and can be unified into one feature space. "Attention" in the formula and the operator name is the abbreviation of "self-attention".

$$
Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
$$

The product of $Q$ and $K^T$ represents the attention of the input $x$. To prevent this value from becoming too large, it is usually scaled by dividing by the square root of $d$, and softmax normalization is applied to each row. Multiplying the result by $V$ yields an $n*d$ matrix.

**Note**:
<blockquote>The data layout formats of query, key, and value can be interpreted from multiple dimensions, where B (Batch) indicates the batch size of the input sample, S (Seq-Length) indicates the sequence length of the input sample, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the size of the smallest unit of the hidden layer, with D=H/N.
<br>Q_S denotes the S in the query shape, KV_S denotes the S in the key and value shapes, Q_N denotes num_query_heads, and KV_N denotes num_key_value_heads. P denotes the computation result of Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>).</blockquote>


## Interface

```python
torch.ops.ascend_ops.flash_attn_minimal(query, key, value, softmaxScale=0) -> Tensor
```

It computes dense (full) self-attention, and the output has the same shape as query.

## Interface Parameter Description

- <a id="query"></a>**query/key/value**

    The input layout is 4-dimensional `BSND` (Batch, Seq-Length, Head-Num, Head-Dim). The current operator imposes the following constraints on shapes and data types:

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 500px">
        <col style="width: 342px">
        </colgroup>
        <thead>
            <tr>
                <th>Attribute</th>
                <th>Meaning</th>
                <th>Constraint</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>Batch(B)</td>
            <td>Batch size of the input sample</td>
            <td>The batch dimensions of Q, K, and V are consistent</td>
        </tr>
        <tr>
            <td>Head-Num(N)</td>
            <td>Number of heads</td>
            <td>Q_N ≥ KV_N, and Q_N is divisible by KV_N (G = Q_N / KV_N is an integer; GQA is supported)</td>
        </tr>
        <tr>
            <td>Seq-Length(S)</td>
            <td>Sequence length of the input sample</td>
            <td>Q_S = KV_S, and it is divisible by 128</td>
        </tr>
        <tr>
            <td>Head-Dim(D)</td>
            <td>Size of the smallest unit of the hidden layer</td>
            <td>Fixed to 128</td>
        </tr>
        <tr>
            <td>Data type</td>
            <td>Data type in the Q, K, and V matrices</td>
            <td>Only bfloat16 is supported</td>
        </tr>
        </tbody>
    </table>

    Typical case: `B=1, N1=32, N2=1, S1=8192, S2=8192, D=128` (GQA, G=32). **Passing an empty Tensor is not supported.**

- <a id="softmaxScale"></a>**softmaxScale**

    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 500px">
        <col style="width: 342px">
        </colgroup>
        <thead>
            <tr>
                <th>Attribute</th>
                <th>Meaning</th>
                <th>Remark</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>float</td>
            <td>Scaling factor applied after the Q and K matrices are multiplied. A value of 1 indicates no scaling, and the default value is 0</td>
            <td>Usually set to 1/√D</td>
        </tr>
        </tbody>
    </table>

## Code Architecture
The operator executes on the Device side (NPU, running the kernel function), and the input is prepared and the kernel is launched on the Host side (CPU). The code is mainly divided into the following two parts:
 - **csrc/flash_attn_minimal/torch_interface.cpp** contains the overall flow of operator execution: the Python input parameters from the upper layer are passed to this file, the input validity is checked, and then the kernel function of the operator is launched directly. There is no separate Host Tiling stage.
 - **csrc/flash_attn_minimal/op_kernel** contains all files of the kernel function of the operator.

## Internal Implementation Documentation

For the memory hierarchy partitioning, multi-core splitting strategy, cross-core synchronization, and a walkthrough of a typical case of the operator, see [FA_IMPL_en.md](./FA_IMPL_en.md).
