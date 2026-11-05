# Compressor

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: Compressor is a preprocessing operator for SMLA and QLI in inference scenarios. It compresses the KV cache of every 4 or 128 tokens into one, and then each token performs DSA computation with these compressed KV caches. In long-sequence cases, Compressor effectively reduces the computation overhead. The main computation process is:
    1. Performs matmul on the input $X$ and $W^{KV}$ to obtain $kv\_state$, and performs matmul on the input $X$ and $W^{Gate}$ followed by an Add operation with $Ape$ to obtain $score\_state$. $kv\_state$ and $score\_state$ are updated based on the input start_pos and cu_seqlens.
    2. When coff is 2, rearranges the data of $kv\_state$ and $score\_state$.
    3. Performs softmax on $score\_state$, multiplies the softmax result with $kv\_state$ through a Mul operation, and then performs a ReduceSum operation.

- Computation formula:

    1. Computes the matmul:

        $$
        C4A:\left[kv\_state^a, score\_state^a\right] = X @ \left[W^{aKV}, W^{aGate}\right], \left[kv\_state^b, score\_state^b\right] = X @ \left[W^{bKV}, W^{bGate}\right];
        $$

        $$
        C128A:\left[kv\_state, score\_state\right] = X @ \left[W^{KV}, W^{Gate}\right]
        $$

    2. Computes the grouped addition:

        $$
        C4A:score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A:score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};
        $$

    3. Computes the grouped softmax:

        $$
        C4A:S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A:S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};
        $$

    4. Computes the Hadamard product:

        $$
        C4A:(S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ;kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A:S_H = S_i^\prime \odot kv\_state;
        $$

    5. Computes the grouped sum along the compression axis:

        $$
        C4A:C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A:C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};
        $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 330px">
  <col style="width: 250px">
  <col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td><em>X</em> in the formula, indicating the original data without compression.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wkv</td>
    <td>Input</td>
    <td><em>W<sup>KV</sup></em> in the formula, indicating the kv compression weight.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wgate</td>
    <td>Input</td>
    <td><em>W<sup>Gate</sup></em> in the formula, indicating the gate compression weight.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>state_cache</td>
    <td>Input/Output</td>
    <td><em>[kv_state, score_state]</em> in the formula, indicating the historical data of kv_state and score_state.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ape</td>
    <td>Input</td>
    <td><em>Ape</em> in the formula, indicating positional biases.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cmp_ratio</td>
    <td>Attribute</td>
    <td>Used for sparse computation. It indicates the data compression ratio, and the value is an integer in the range [2, 128].</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>state_block_table</td>
    <td>Optional input</td>
    <td>Indicates the block mapping table used for state_cache storage.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cu_seqlens</td>
    <td>Optional input</td>
    <td>Indicates the number of valid tokens in different batches. This parameter is a prefix-sum array. The value of each later element must be greater than or equal to the value of the previous element, and the first element must be 0.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>seqused</td>
    <td>Optional input</td>
    <td>Indicates the number of tokens that actually participate in compression in different batches. When set to None, it is the same as the Sequence Length of each batch.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>start_pos</td>
    <td>Optional input</td>
    <td>Indicates the computation start position. When set to None, computation starts from 0.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>coff</td>
    <td>Optional attribute</td>
    <td>Indicates whether overlap data rearrangement is performed. coff=1: no overlap data rearrangement is required; coff=2: overlap data rearrangement is required. The default value is 1.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cache_mode</td>
    <td>Optional attribute</td>
    <td>Indicates the storage mode of state_cache. cache_mode=1: continuous buffer; cache_mode=2: circular buffer. The default value is 1.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>state_cache_stride_dim0</td>
    <td>Optional attribute</td>
    <td>Indicates the stride of axis 0 of state_cache. The default value is 0.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>grad_enabled</td>
    <td>Optional attribute</td>
    <td>Whether to export the softmax_score/kv intermediate results (for the backward CompressorGrad). They are exported when true and not exported when false. The default value is false.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cmp_kv</td>
    <td>Output</td>
    <td>Indicates the compressed data.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>softmax_score</td>
    <td>Output</td>
    <td>Intermediate result of the softmax weight of compressed blocks. The output is valid only when grad_enabled is true.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>kv</td>
    <td>Output</td>
    <td>Intermediate result of the raw KV of compressed blocks. The output is valid only when grad_enabled is true.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - cmp_ratio supports only 2/4/8/16/32/64/128.

## Constraints

- Dimension meanings of x: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, D (Head Dim) indicates the size of the minimum unit of the hidden layer, and T indicates the accumulated sequence length of input samples across all batches.
- Input shape restrictions:
    - state_cache supports the input shape [block_num, block_size, 2\*coff\*D] and requires block_num>0. When cache_mode=2, block_size >= coff * cmp_ratio + S - 1 must be satisfied.
    - If the BS dimensions of x are merged, that is, the input shape of x is [T,H]:
        - The input shape of cu_seqlens must be [B+1,]. The value of each element in this parameter indicates the total number of tokens of the current batch and all preceding batches, that is, the prefix sum. Therefore, the value of each later element must be greater than or equal to the value of the previous element, the first element must be 0, and the last element must be T.
        - seqused supports the input shape [B,] and requires that the number of valid tokens of each batch be less than or equal to the corresponding Sequence Length, that is, seqused[n] <= cu_seqlens[n+1] - cu_seqlens[n], and not less than 0.
        - When cache_mode=1, state_block_table supports the input shape [B, ceil(Smax/block_size)]. Smax is the maximum Sequence Length across batches, that is, Smax=max(start_pos)+max(cu_seqlens[n+1] - cu_seqlens[n]). When cache_mode=2, state_block_table supports the input shape [B].
        - cmp_kv has the output shape [min(T, T/cmp_ratio+B), D]: compressed_tokens + compressed_tokens + ... + compressed_tokens + pad.
    - If the BS dimensions of x are not merged, that is, the input shape of x is [B,S,H]:
        - cu_seqlens must be empty.
        - seqused supports the input shape [B,] and requires that the number of valid tokens of each batch be less than or equal to the corresponding Sequence Length, that is, seqused[n] <= S, and not less than 0.
        - When cache_mode=1, state_block_table supports the input shape [B, ceil(Smax/block_size)]. Smax is the maximum Sequence Length across batches, that is, Smax=max(start_pos)+S. When cache_mode=2, state_block_table supports the input shape [B].
        - cmp_kv has the output shape [B, ceil(S/cmp_ratio), D]: (compressed_tokens+pad0) + (compressed_tokens+pad1) + ... + (compressed_tokens+padN).
- Input value range restrictions:
  - This API supports generalization over B and S, with the following scenario restrictions:
      - In some long-sequence scenarios, an excessive computation amount may cause an error that the memory usage exceeds NPU memory. Note: the computation amount is affected by the input shape of x; the larger the values, the greater the computation amount. Typical long-sequence scenarios (that is, a large product of B and S, or a large T) include the following cases:

      <table style="undefined;table-layout: fixed; width: 400px"><colgroup>
        <col style="width: 100px">
        <col style="width: 100px">
        <col style="width: 100px">
      </colgroup>
      <thead>
        <tr>
          <th>B</th>
          <th>S</th>
          <th>H</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>100</td><td>65525</td><td>4096</td></tr>
        <tr><td>25</td><td>261120</td><td>4096</td></tr>
        <tr><td>100</td><td>131072</td><td>4096</td></tr>
        <tr><td>100</td><td>261120</td><td>4096</td></tr>
      </tbody>
      </table>
  - This API supports B, S, and T being 0. That is, input parameters whose shapes depend on the values of B, S, and T allow empty tensors, while other input parameters do not support empty tensors. In this scenario, state_cache is not updated, and the output cmp_kv is an empty tensor.
  - The value range of state_block_table elements is [0, block_num), where block_num is the size of dimension 0 of state_cache. Element values are directly used as block indices of state_cache, and out-of-range values cause invalid memory access. When an element value is 0: in cache_mode=1 (continuous buffer), the write state_cache operation skips this position; in cache_mode=2 (circular buffer), neither read nor write operations skip it. The operator does not perform duplicate validation; the caller must ensure the uniqueness of element values: in cache_mode=1, the element value 0 is the "unallocated" sentinel and may appear repeatedly, while non-zero values must be globally unique; in cache_mode=2, 0 is a valid physical block number, and all element values must be globally unique. Duplicates cause multiple logical blocks/batches to write the same physical block region, resulting in state_cache data being overwritten.
- Input attribute restrictions:
  - D of 128/512 is supported.
  - H of 1K to 10K, aligned to 512, is supported.
  - block_size of 1 to 1024 is supported.
  - The following three typical combination scenarios are supported:
      - C4A: D=512, coff=2, and cmp_ratio=4.
      - C4Li: D=128, coff=2, and cmp_ratio=4.
      - C128A: D=512, coff=1, and cmp_ratio=128.
- Deterministic computation and batch consistency:
  - A deterministic implementation is used by default, and multiple calls with the same input produce consistent results.
<!-- npu="950" id1 -->
  - <term>Ascend 950PR&950DT products</term>: batch consistency: set ACL_OPT_DETERMINISTIC to 3 through aclrtSetSysParamOpt() to enable batch consistency. After it is enabled, computation results are independent of the batch size and position.
<!-- end id1 -->
<!-- npu="A3,910b" id2 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: batch consistency: set ACL_OPT_DETERMINISTIC to 3 through aclrtSetSysParamOpt() to enable batch consistency. After it is enabled, computation results are independent of the batch size and position.
<!-- end id2 -->

## Invocation Description

- <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, and <term>Atlas A2 products</term>:

  | Invocation Mode | Sample Code | Description |
  | ---------- | -------- | --------------------------------------------- |
  | aclnn API | [test_aclnn_compressor](./examples/arch35/test_aclnn_compressor.cpp) | Calls the Compressor operator through the [aclnnCompressor](docs/aclnnCompressor_en.md) API. |
  | PyTorch API | - | Calls the Compressor operator through the [compressor](docs/torchapi_compressor_en.md) API. |
