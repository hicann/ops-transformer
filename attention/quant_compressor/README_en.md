# QuantCompressor

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×    |

## Description

- Operator function: QuantCompressor is a preprocessing operator of SMLA and QLI in the inference scenario. It is the quantized version of [Compressor](../compressor/README_en.md). It compresses every 4 or 128 tokens of the KV cache into one, and then each token performs DSA computation with these compressed KV caches. In the case of long sequences, QuantCompressor can effectively reduce the computation overhead. The difference from Compressor is that the inputs $x$, $W^{KV}$, and $W^{Gate}$ are HIFLOAT8 quantized data and directly participate in Matmul computation as HIFLOAT8 (natively supported by hardware). Then the FLOAT32 Matmul output is multiplied by the merged scale factor for dequantization, thereby reducing memory usage and movement overhead. The main computation process is:
    1. Matmul and dequantization: Perform Matmul on the HIFLOAT8 quantized input $X$ and $W^{KV}$ to obtain a FLOAT32 result, then multiply it by the merged scale factor $x\_descale \cdot wkv\_descale$ to complete dequantization and obtain $kv\_state$. Perform Matmul on $X$ and $W^{Gate}$ to obtain a FLOAT32 result, then multiply it by the merged scale factor $x\_descale \cdot wgate\_descale$ to complete dequantization and obtain $score\_state$. x_descale is a per-tensor scale (a single scalar), and wkv_descale and wgate_descale are per-channel scales (the number of channels is coff\*D). After merging, the scale is still per-channel. $kv\_state$ and $score\_state$ are updated according to the input start_pos and cu_seqlens.
    2. When coff is 2, rearrange the data of $kv\_state$ and $score\_state$.
    3. Group $score\_state$ by the compression ratio and add it to $Ape$, then perform the softmax computation. Compute Mul on the softmax result and $kv\_state$, and then perform the ReduceSum computation.

- Computation formula:

    1. Compute matrix multiplication and dequantization (quant_mode=1, A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL):

    Perform Matmul with HIFLOAT8 input. After the FLOAT32 result is output, multiply it by the merged scale factor (the product of x_descale and wkv/wgate_descale) to complete dequantization:

    $$
    C4A:\left[kv\_state^a, score\_state^a\right] = (X_{hif8} @ \left[W^{aKV}_{hif8}, W^{aGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^a, wgate\_descale^a]), \left[kv\_state^b, score\_state^b\right] = (X_{hif8} @ \left[W^{bKV}_{hif8}, W^{bGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^b, wgate\_descale^b]);
    $$

    $$
    C128A:\left[kv\_state, score\_state\right] = (X_{hif8} @ \left[W^{KV}_{hif8}, W^{Gate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale, wgate\_descale])
    $$

    x_descale is a per-tensor scalar (shape [1,]), and wkv_descale and wgate_descale are per-channel vectors (shape [coff\*D,], scaled along the weight output channel dimension). After the two are multiplied, they merge into a per-channel scale factor.

    2. Compute grouped addition:

    $$
    C4A:score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A:score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};
    $$

    3. Compute grouped Softmax:

    $$
    C4A:S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A:S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};
    $$

    4. Compute the Hadamard product:

    $$
    C4A:(S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ;kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A:S_H = S_i^\prime \odot kv\_state;
    $$

    5. Compute grouped summation along the compression axis:

    $$
    C4A:C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};
    $$

    $$
    C128A:C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};
    $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 220px">
  <col style="width: 170px">
  <col style="width: 700px">
  <col style="width: 286px">
  <col style="width: 200px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
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
      <td><span class="math-inline">X</span> in the formula, indicating the original data without compression. It is the HIFLOAT8 quantized input.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wkv</td>
      <td>Input</td>
      <td><span class="math-inline">W<sup>KV</sup></span> in the formula, indicating the KV compression weight. It is the HIFLOAT8 quantized input.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wgate</td>
      <td>Input</td>
      <td><span class="math-inline">W<sup>Gate</sup></span> in the formula, indicating the gate compression weight. It is the HIFLOAT8 quantized input.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>state_cache</td>
      <td>Input/Output</td>
      <td><span class="math-inline">[kv_state, score_state]</span> in the formula, indicating the historical data of kv_state and score_state.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ape</td>
      <td>Input</td>
      <td><span class="math-inline">Ape</span> in the formula, indicating positional biases.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x_descale</td>
      <td>Optional input</td>
      <td>Dequantization scale factor of x, with per-tensor scale. Required when quant_mode=1.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wkv_descale</td>
      <td>Optional input</td>
      <td>Dequantization scale factor of wkv, with per-channel scale and coff*D channels. Required when quant_mode=1.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wgate_descale</td>
      <td>Optional input</td>
      <td>Dequantization scale factor of wgate, with per-channel scale and coff*D channels. Required when quant_mode=1.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>state_block_table</td>
      <td>Optional input</td>
      <td>Indicates the block mapping table used for state_cache storage. In the cachemode=1/2 scenarios supported by the current operator, this parameter is mandatory.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens</td>
      <td>Optional input</td>
      <td>Indicates the number of valid tokens in each batch.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused</td>
      <td>Optional input</td>
      <td>Indicates the number of tokens that actually participate in compression in each batch.<br>If None is specified, it indicates the same as the Sequence Length of each batch.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>start_pos</td>
      <td>Optional input</td>
      <td>Indicates the start position of computation.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Quantization mode. The value range is [1]. 1 indicates A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL (HIFLOAT8 input; x is dequantized with per-tensor scale, and wkv/wgate with per-channel scale).</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>Attribute</td>
      <td>Used for sparse computation. It indicates the data compression ratio, which is an integer in the range [2, 128].</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>coff</td>
      <td>Optional attribute</td>
      <td>Indicates whether to perform overlap data rearrangement.<br>coff=1: no overlap data rearrangement is required; coff=2: overlap data rearrangement is required.<br>The default value is 1.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cache_mode</td>
      <td>Optional attribute</td>
      <td>Indicates the storage mode of state_cache.<br>cache_mode=1: continuous buffer; cache_mode=2: circular buffer.<br>The default value is 1.</td>
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
      <td>cmp_kv</td>
      <td>Output</td>
      <td>Indicates the compressed data. The content of the pad rows at the tail that do not form a complete compression group is undefined (the operator does not write them, and the memory content before the call is retained). The caller should consume only the valid compressed rows.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- Dimension meanings of the x parameter: B (Batch Size) indicates the batch size of the input sample, S (Sequence Length) indicates the sequence length of the input sample, H (Head Size) indicates the size of the hidden layer, D (Head Dim) indicates the smallest unit size of the hidden layer, and T indicates the cumulative sum of the sequence lengths of input samples across all batches.
- Input shape restrictions:
    - state\_cache supports the input shape [block_num,block_size,2\*coff\*D], with block_num>0 required. When cache_mode=2, block_size >= coff * cmp_ratio + S - 1 must be satisfied.
    - If the dimensions of x adopt BS axis merging, that is, the input shape of x is [T,H]:
        - The input shape of cu\_seqlens must be [B+1,]. Each element in this parameter indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element, the first element must be 0, and the last element must be T.
        - seqused supports the input shape [B,]. The number of valid tokens in each batch must be less than or equal to the corresponding Sequence Length, that is, seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n], and not less than 0.
        - When cache_mode=1, state\_block\_table supports the input shape [B,ceil(Smax/block_size)]. Smax is the largest Sequence Length in each batch, that is, Smax=max(start\_pos)+max(cu\_seqlens[n+1] - cu\_seqlens[n]). When cache_mode=2, state\_block\_table supports the input shape [B].
        - cmp\_kv has the output shape [min(T,T//cmp_ratio+B),D]: compressed_tokens + compressed_tokens + ... + compressed_tokens + pad.
    - If the dimensions of x do not adopt BS axis merging, that is, the input shape of x is [B,S,H]:
        - cu\_seqlens must be empty.
        - seqused supports the input shape [B,]. The number of valid tokens in each batch must be less than or equal to the corresponding Sequence Length, that is, seqused[n] <= S, and not less than 0.
        - When cache_mode=1, state\_block\_table supports the input shape [B,ceil(Smax/block_size)]. Smax is the largest Sequence Length in each batch, that is, Smax=max(start\_pos)+S. When cache_mode=2, state\_block\_table supports the input shape [B].
        - cmp\_kv has the output shape [B,ceil(S/cmp_ratio),D]: (compressed_tokens+pad0) + (compressed_tokens+pad1) + ...  + (compressed_tokens+padN).
- Input value range restrictions:
  - This interface supports generalization of B and S, with the following scenario restrictions:
      - In some long-sequence scenarios, an excessively large computation amount may cause an error that exceeds NPU memory. Note: the computation amount is affected by the input shape of x; the larger the values, the larger the computation amount. Typical long-sequence scenarios (that is, a large product of B and S, or a large T) include but are not limited to the following:
      <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 400px"><colgroup>
      <col style="width: 100px">
      <col style="width: 100px">
      <col style="width: 100px">
      </colgroup><thead>
      <tr>
      <th>B</th>
      <th>S</th>
      <th>H</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>100</td>
      <td>65525</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>25</td>
      <td>261120</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>100</td>
      <td>131072</td>
      <td>4096</td>
      </tr>
      <tr>
      <td>100</td>
      <td>261120</td>
      <td>4096</td>
      </tr>
      </tbody>
      </table>
      </div>
- This interface supports B, S, and T being 0. That is, input parameters whose shapes are related to the B, S, and T values allow empty tensors, and the other input parameters do not support empty tensors. In this scenario, state_cache is not updated, and the output cmp_kv is an empty tensor.
- The value of each element in state_block_table ranges from [0, block_num), where block_num is the size of dimension 0 of state_cache. Element values are directly used as the block index of state_cache, and out-of-range values cause illegal memory access. When an element value is 0: under cache_mode=1 (continuous buffer), the write operation on state_cache skips this position; under cache_mode=2 (circular buffer), neither the read nor the write operation skips it. The operator does not check for duplicates, and the caller must ensure the uniqueness of element values: under cache_mode=1, the element value 0 is an "unallocated" sentinel value and may appear repeatedly, while non-zero values must be globally unique; under cache_mode=2, 0 is a valid physical block number, and all element values must be globally unique. Duplicates cause multiple logical blocks/batches to write to the same physical block area, resulting in data overwrite of state_cache.
- Input attribute restrictions:
  - quant_mode is 1 (A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL). In this case, x_descale, wkv_descale, and wgate_descale are required inputs.
  - D of 128/512 is supported.
  - H from 1K to 10K, aligned to 512, is supported.
  - block_size from 1 to 1024 is supported.
  - The following three typical combination scenarios are supported:
      - C4A: D=512, coff=2, cmp_ratio=4;
      - C4Li: D=128, coff=2, cmp_ratio=4;
      - C128A: D=512, coff=1, cmp_ratio=128.
- Deterministic computation and batch consistency:
  - Deterministic implementation by default. Multiple calls with the same input produce the same result.
  - Set ACL_OPT_DETERMINISTIC to 3 through aclrtSetSysParamOpt() to enable batch consistency. After it is enabled, the computation result is independent of the batch size and position.

## Calling Instructions

| Calling Method | Sample Code | Description |
| ------- | ------- | ---- |
| aclnn API | [test_aclnn_quant_compressor](./examples/arch35/test_aclnn_quant_compressor.cpp) | Calls the QuantCompressor operator through the [aclnnQuantCompressor](docs/aclnnQuantCompressor_en.md) interface. |
| PyTorch API | - | Calls the QuantCompressor operator through the [quant_compressor](docs/torchapi_quant_compressor_en.md) interface. |
