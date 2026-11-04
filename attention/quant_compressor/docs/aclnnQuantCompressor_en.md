# aclnnQuantCompressor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:23:57.846Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: QuantCompressor is a preprocessing operator for SMLA and QLI in inference use cases. It is the quantized version of [Compressor](../../compressor/docs/aclnnCompressor_en.md). It compresses the KV cache of every 4 or 128 tokens into one, and each token then performs DSA computation with the compressed KV caches. For long sequences, QuantCompressor can effectively reduce the computation overhead. It differs from Compressor as follows: the inputs $x$, $W^{KV}$, and $W^{Gate}$ are HIFLOAT8 quantized data. The data directly participates in the Matmul operation in HIFLOAT8 format (natively supported by hardware). Then, the FLOAT32 Matmul output is multiplied by the merged scale factor for dequantization. This reduces memory usage and data movement overhead. The main computation process is as follows:
    1. Matmul and dequantization: Perform the Matmul operation on the HIFLOAT8-quantized input $X$ and $W^{KV}$ to obtain a FLOAT32 result. Multiply the result by the merged scale factor $x\_descale \cdot wkv\_descale$ to complete dequantization and obtain $kv\_state$. Perform the Matmul operation on $X$ and $W^{Gate}$ to obtain a FLOAT32 result. Multiply the result by the merged scale factor $x\_descale \cdot wgate\_descale$ to complete dequantization and obtain $score\_state$. Here, x_descale applies per-tensor scaling (a single scalar), and wkv_descale and wgate_descale apply per-channel scaling (the number of channels is coff\*D). After merging, the scaling remains per-channel. $kv\_state$ and $score\_state$ are updated based on the input start_pos and cu_seqlens.
    2. When coff is 2, perform data rearrangement on $kv\_state$ and $score\_state$.
    3. Group $score\_state$ by compression ratio and add $Ape$, then perform softmax. Compute Mul between the softmax result and $kv\_state$, and then perform ReduceSum.

- Formula:

    1. Compute matrix multiplication and dequantization (quant_mode=1, A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL):

        Perform Matmul with the HIFLOAT8 input. Multiply the FLOAT32 output by the merged scale factor (product of x_descale and wkv/wgate_descale) to complete dequantization:

        $$
        C4A: \left[kv\_state^a, score\_state^a\right] = (X_{hif8} @ \left[W^{aKV}_{hif8}, W^{aGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^a, wgate\_descale^a]), \left[kv\_state^b, score\_state^b\right] = (X_{hif8} @ \left[W^{bKV}_{hif8}, W^{bGate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale^b, wgate\_descale^b]);
        $$

        $$
        C128A: \left[kv\_state, score\_state\right] = (X_{hif8} @ \left[W^{KV}_{hif8}, W^{Gate}_{hif8}\right]) \cdot (x\_descale \cdot [wkv\_descale, wgate\_descale])
        $$

        x_descale is a per-tensor scalar (shape [1,]). wkv_descale and wgate_descale are per-channel vectors (shape [coff\*D,], scaled along the weight output channel dimension). The product of the two is merged into a per-channel scale factor.

    2. Compute grouped addition:

        $$
        C4A: score\_state_i^\prime = \left[score\_state_{\left[4(i-1)+1:4i,:\right]}^a; score\_state_{\left[4i+1:4(i+1),:\right]}^b\right] + Ape,~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A: score\_state_i^\prime = score\_state_{\left[128(i-1)+1:128i,:\right]} + Ape,~i=1,2,\cdots, \frac{s}{128};
        $$

    3. Compute the grouped Softmax:

        $$
        C4A: S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A: S_i^\prime = softmax(score\_state_i^\prime),~i=1,2,\cdots, \frac{s}{128};
        $$

    4. Compute the Hadamard product:

        $$
        C4A: (S_H)_i = S_i^\prime \odot \left[kv\_state^a_{\left[4(i-1)+1:4i,:\right]} ;kv\_state^b_{\left[4i+1:4(i+1),:\right]}\right],~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A: S_H = S_i^\prime \odot kv\_state;
        $$

    5. Perform the grouped summation along the compression axis:

        $$
        C4A: C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A: C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};
        $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnQuantCompressorGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the process. Then call "aclnnQuantCompressor" to execute the computation.

```cpp
aclnnStatus aclnnQuantCompressorGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *wkv,
    const aclTensor *wgate,
    aclTensor       *stateCacheRef,
    const aclTensor *ape,
    const aclTensor *xDescaleOptional,
    const aclTensor *wkvDescaleOptional,
    const aclTensor *wgateDescaleOptional,
    const aclTensor *stateBlockTableOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional,
    int64_t          quantMode,
    int64_t          cmpRatio,
    int64_t          coff,
    int64_t          cacheMode,
    int64_t          stateCacheStrideDim0,
    const aclTensor *cmpKvOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

``` cpp
aclnnStatus aclnnQuantCompressor(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantCompressorGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 220px">
  <col style="width: 110px">
  <col style="width: 300px">
  <col style="width: 300px">
  <col style="width: 240px">
  <col style="width: 100px">
  <col style="width: 160px">
  <col style="width: 110px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (const aclTensor*)</td>
      <td>Input</td>
      <td><span class="math-inline">X</span> in the formula. Indicates the original data without compression. HIFLOAT8 quantized input.</td>
      <td>Supports empty tensors with B=0, S=0, T=0.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>BS axis combination: [T,H]; BS without axis combination: [B,S,H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>wkv (const aclTensor*)</td>
      <td>Input</td>
      <td><span class="math-inline">W<sup>KV</sup></span> in the formula. Indicates the KV compression weight. HIFLOAT8 quantized input.</td>
      <td>Empty Tensor is not supported.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>[coff* D,H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>wgate (const aclTensor*)</td>
      <td>Input</td>
      <td><span class="math-inline">W<sup>Gate</sup></span> in the formula, indicating the gate compression weight, HIFLOAT8 quantized input.</td>
      <td>Empty Tensor is not supported.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>[coff* D,H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>stateCacheRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>In the formula, <span class="math-inline">[kv_state, score_state]</span> indicates the historical data of kv_state and score_state. Updated in place.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[block_num,block_size,2* coff* D]</td>
      <td>Supports non-contiguous tensor on axis 0.</td>
    </tr>
    <tr>
      <td>ape (const aclTensor*)</td>
      <td>Input</td>
      <td><span class="math-inline">Ape</span> in the formula. Indicates positional biases.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[cmp_ratio,coff* D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>xDescaleOptional (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Dequantization scale factor of x, per-tensor scaling. Mandatory when quant_mode=1.</td>
      <td>Empty Tensor is not supported.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[1,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>wkvDescaleOptional (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Dequantization scale factor of wkv, per-channel scaling, number of channels is coff*D. Mandatory when quant_mode=1.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[coff* D,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>wgateDescaleOptional(const aclTensor*)</td>
      <td>Optional input</td>
      <td>Dequantization scale factor of wgate, for per-channel scaling. The number of channels is coff*D. Mandatory when quant_mode=1.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[coff* D,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>stateBlockTableOptional (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the block mapping table used for state_cache storage.</td>
      <td><ul><li>When cacheMode=1, the shape is [B, ceil(Smax/block_size)], where Smax is the maximum Sequence Length for each Batch. If the shape of x is [B,S,H], Smax=max(startPosOptional)+S. If the shape of x is [T,H], Smax=max(startPosOptional)+max(cuSeqlensOptional[n+1] - cuSeqlensOptional[n]).</li><li>When cacheMode=2, the shape is [B].</li><li>In the cacheMode=1/2 scenarios supported by the current operator, this parameter is mandatory.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>When cacheMode=1: [B, ceil(Smax/block_size)]<br>When cacheMode=2: [B]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqlensOptional (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the number of valid tokens in different batches.</td>
      <td>Empty tensors with B=0, S=0, and T=0 are supported. This parameter must be empty when the shape of x is [B,S,H].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>Input shape is [B+1,] when the shape of x is [T,H].</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedOptional (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the number of tokens that actually participate in compression in different batches.</td>
      <td>If None, the value is the same as the Sequence Length of each batch; empty tensor with B=0 is supported. The number of valid tokens in each batch of this input must be less than or equal to the corresponding Sequence Length. When the shape of x is [B,S,H], sequsedOptional[n] must be <= S and not less than 0; when the shape of x is [T,H], sequsedOptional[n] must be <= cuSeqlensOptional[n+1] - cuSeqlensOptional[n] and not less than 0.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>startPosOptional (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the start position of the computation.</td>
      <td>Supports empty tensors with B=0, T=0; when the input is None, computation starts from 0.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Quantization mode.</td>
      <td>The value range is [1]. 1 indicates A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL (HIFLOAT8 input; x is dequantized with per-tensor scaling, and wkv/wgate with per-channel scaling).</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Used for sparse computation. Indicates the data compression ratio.</td>
      <td>Value range: an integer within [2, 128].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>coff (int64_t)</td>
      <td>Input</td>
      <td>Indicates whether to perform overlap data rearrangement.</td>
      <td>Value range: {1, 2}. If coff is 1, no overlap data rearrangement is required. If coff is 2, overlap data rearrangement is required.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the storage mode of state_cache.</td>
      <td>Value range: [1, 2]; 1 indicates a contiguous buffer, and 2 indicates a circular buffer.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stateCacheStrideDim0 (int64_t)</td>
      <td>Input</td>
      <td>Indicates the stride of state_cache on dim 0.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpKvOut(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the compressed data.</td>
      <td><ul><li>Supports empty tensors with B=0, S=0, T=0.</li><li>The content of the pad rows at the tail that do not form a complete compression group is undefined (the operator does not write them, and the memory content before the call is retained). The caller should consume only the valid compressed rows.</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>BS axis combination: [min(T,T//cmp_ratio+B),D]; BS non-axis combination: [B,ceil(S/cmp_ratio),D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Indicates the workspace size to be requested by the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

    aclnnStatus: return status code, for details see [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

    When the first-phase API performs parameter validation and tiling, the following errors may be returned:

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
      <thead>
        <tr>
          <th>Return Value</th>
          <th>Error Code</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>A mandatory parameter (such as the input/output parameter on which the API core depends) contains a null pointer.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>The dtype (data type) of the input parameter is not supported by the API.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>An exception occurs when the API calls the NPU Runtime API for memory operations (for example, the Runtime service is not started or memory allocation fails).</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_TILING_ERROR</td>
          <td>561002</td>
          <td>An exception occurs during tiling: the shape (dimension/size), attribute value, or the presence/absence of an optional input does not match the current configuration (for example, cuSeqlensOptional is passed with the BSH layout, or the descale input is missing when quantMode=1).</td>
        </tr>
      </tbody>
    </table>

## aclnnQuantCompressor

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device, obtained through the first-phase API aclnnQuantCompressorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>The op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnQuantCompressor defaults to a deterministic implementation.
  - batch consistency: Enable batch consistency by setting ACL_OPT_DETERMINISTIC to 3 through aclrtSetSysParamOpt(). Then computation results are independent of the batch size and position.
- Meanings of x dimensions: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, D (Head Dim) indicates the size of the minimum unit of the hidden layer, and T indicates the cumulative sum of sequence lengths of input samples across all batches.
- Input shape constraints:
    - stateCache supports input shape[block_num,block_size,2\*coff\*D]. Requirement: blockNum>0. When cacheMode=2, blockSize >= coff * cmp_ratio + S - 1 must be met.
    - If the dimensions of x use BS axis combination, that is, the input shape of x is [T,H]
        - cuSeqlensOptional: the input shape must be [B+1,]. Each element indicates the total number of tokens in the current batch and all preceding batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the preceding element, the first element must be 0, and the last element must be T.
        - sequsedOptional: supports input shape[B,]. Requirement: the number of valid tokens in each batch must be less than or equal to the corresponding Sequence Length, that is, sequsedOptional[n] <= cuSeqlensOptional[n+1] - cuSeqlensOptional[n], and not less than 0.
        - When cacheMode=1, stateBlockTableOptional supports input shape[B,ceil(Smax/block_size)]. Smax is the maximum Sequence Length among all batches, that is, Smax=max(startPosOptional)+max(cuSeqlensOptional[n+1] - cuSeqlensOptional[n]). When cacheMode=2, stateBlockTableOptional supports input shape[B].
        - cmpKv: the output shape is [min(T,T//cmp_ratio+B),D]: compressed_tokens + compressed_tokens + ... + compressed_tokens + pad.
    - If the dimensions of x do not use BS axis combination, that is, the input shape of x is [B,S,H]
        - cuSeqlensOptional: this parameter must be empty.
        - sequsedOptional: supports input shape[B,]. Requirement: the number of valid tokens in each batch must be less than or equal to the corresponding Sequence Length, that is, sequsedOptional[n] <= S, and not less than 0.
        - When cacheMode=1, stateBlockTable supports input shape [B,ceil(Smax/block_size)]. Smax indicates the maximum Sequence Length in each batch, that is, Smax=max(startPosOptional)+S. When cacheMode=2, stateBlockTable supports input shape [B].
        - cmpKv, output shape is [B,ceil(S/cmp_ratio),D]: (compressed_tokens+pad0) + (compressed_tokens+pad1) + ...  + (compressed_tokens+padN).
- Input value range restrictions:
  - This API supports generalization of B and S and has the following use case constraints:
      - In some long-sequence use cases, an excessively large computation amount may cause an NPU out-of-memory error. Note: the computation amount is affected by the x input shape. A larger value indicates a larger computation amount. Typical long-sequence use cases (that is, a large product of B and S or a large T) include but are not limited to:
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
- This API supports B, S, and T values of 0. That is, input parameters whose shapes depend on the B, S, and T values can be empty tensors, while other input parameters do not support empty tensors. In this case, stateCache is not updated, and the output cmpKv is an empty tensor.
- The value range of stateBlockTableOptional elements is [0, block_num), where block_num is the size of dimension 0 of stateCacheRef. Element values are directly used as block indexes of state_cache, and out-of-bounds values cause invalid memory access. When the element value is 0: for cacheMode=1 (continuous buffer), the state_cache write operation skips the position; for cacheMode=2 (circular buffer), neither read nor write operations skip it. The operator does not perform duplicate checks, so the caller must ensure element value uniqueness: for cacheMode=1, the element value 0 is a sentinel value indicating "unallocated" and can appear repeatedly, while non-zero values must be globally unique; for cacheMode=2, 0 is a valid physical block ID and all element values must be globally unique. Duplicates cause multiple logical blocks/batches to write to the same physical block area, overwriting data in state_cache.
- Input attribute constraints:
  - quantMode is 1 (A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL). In this case, xDescale, wkvDescale, and wgateDescale are mandatory inputs.
  - D of 128/512 is supported.
  - H of 1K-10K is supported and must be 512-aligned.
  - blockSize of 1-1024 is supported.
  - The following three typical combination use cases are supported:
      - C4A: D=512, coff=2, cmp_ratio=4;
      - C4Li: D=128, coff=2, cmp_ratio=4;
      - C128A: D=512, coff=1, cmp_ratio=128.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_compressor.cpp
 * \brief QuantCompressor operator aclnn call example (A5 / ascend950)
 *        Use case: C4A (D=512, coff=2, cmp_ratio=4, cache_mode=1, BSH layout, HIFLOAT8)
 *        QuantCompressor supports ascend950 only.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_compressor.h"
#include "opdev/bfloat16.h"
#include "opdev/hifloat8.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

using op::bfloat16;
using op::HiFloat8;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    } else {
        *deviceAddr = nullptr;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void PrintBf16Result(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<bfloat16> resultData(size, bfloat16(0.0f));
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) { // 10: max print
        LOG_PRINT("cmp_kv[%ld] is: %f\n", i, static_cast<float>(resultData[i]));
    }
}

} // namespace

int main()
{
    // 1. device/stream initialization
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Use case parameters: C4A (D=512, coff=2, cmp_ratio=4, cache_mode=1, BSH layout)
    int64_t B = 1;
    int64_t S = 4;
    int64_t hiddenSize = 4096;
    int64_t headDim = 512;
    int64_t coff = 2;
    int64_t cmpRatio = 4;
    int64_t cacheMode = 1; // 1: contiguous buffer (LINEAR_BUFFER)
    int64_t quantMode = 1; // 1: A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL
    int64_t blockSize = 128;

    int64_t Smax = S;
    int64_t maxBlockNumPerBatch = (Smax + blockSize - 1) / blockSize;
    int64_t blockNum = B * maxBlockNumPerBatch;
    int64_t coffD = coff * headDim;
    int64_t stateCacheStrideDim0 = blockSize * 2 * coffD; // state_cache dim 0 stride = dim1 * dim2

    // 2. Construct input and output shapes.
    std::vector<int64_t> xShape = {B, S, hiddenSize};
    std::vector<int64_t> wkvShape = {coffD, hiddenSize};
    std::vector<int64_t> wgateShape = {coffD, hiddenSize};
    std::vector<int64_t> stateCacheShape = {blockNum, blockSize, 2 * coffD};
    std::vector<int64_t> apeShape = {cmpRatio, coffD};
    std::vector<int64_t> xDescaleShape = {1};
    std::vector<int64_t> wkvDescaleShape = {coffD};
    std::vector<int64_t> wgateDescaleShape = {coffD};
    std::vector<int64_t> stateBlockTableShape = {B, maxBlockNumPerBatch};
    std::vector<int64_t> startPosShape = {B};
    int64_t Sr = (S + cmpRatio - 1) / cmpRatio;
    std::vector<int64_t> cmpKvShape = {B, Sr, headDim};

    // 3. Construct host data.
    int64_t xSize = GetShapeSize(xShape);
    int64_t wkvSize = GetShapeSize(wkvShape);
    int64_t wgateSize = GetShapeSize(wgateShape);
    int64_t stateCacheSize = GetShapeSize(stateCacheShape);
    int64_t apeSize = GetShapeSize(apeShape);
    int64_t cmpKvSize = GetShapeSize(cmpKvShape);

    std::vector<HiFloat8> xHostData(xSize, HiFloat8(0.1f));
    std::vector<HiFloat8> wkvHostData(wkvSize, HiFloat8(0.1f));
    std::vector<HiFloat8> wgateHostData(wgateSize, HiFloat8(0.1f));
    std::vector<float_t> stateCacheHostData(stateCacheSize, 0.1f);
    std::vector<float_t> apeHostData(apeSize, 0.1f);
    std::vector<float_t> xDescaleHostData = {0.5f};
    std::vector<float_t> wkvDescaleHostData(coffD, 0.1f);
    std::vector<float_t> wgateDescaleHostData(coffD, 0.1f);
    std::vector<int32_t> stateBlockTableHostData;
    for (int64_t i = 0; i < B * maxBlockNumPerBatch; i++) {
        stateBlockTableHostData.push_back(static_cast<int32_t>(i + 1));
    }
    std::vector<int32_t> startPosHostData(B, 0);
    std::vector<bfloat16> cmpKvHostData(cmpKvSize, bfloat16(0.0f));

    // 4. Create aclTensor.
    void *xDeviceAddr = nullptr;
    void *wkvDeviceAddr = nullptr;
    void *wgateDeviceAddr = nullptr;
    void *stateCacheDeviceAddr = nullptr;
    void *apeDeviceAddr = nullptr;
    void *xDescaleDeviceAddr = nullptr;
    void *wkvDescaleDeviceAddr = nullptr;
    void *wgateDescaleDeviceAddr = nullptr;
    void *stateBlockTableDeviceAddr = nullptr;
    void *startPosDeviceAddr = nullptr;
    void *cmpKvDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *wkv = nullptr;
    aclTensor *wgate = nullptr;
    aclTensor *stateCacheRef = nullptr;
    aclTensor *ape = nullptr;
    aclTensor *xDescale = nullptr;
    aclTensor *wkvDescale = nullptr;
    aclTensor *wgateDescale = nullptr;
    aclTensor *stateBlockTable = nullptr;
    aclTensor *startPos = nullptr;
    aclTensor *cmpKvOut = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_HIFLOAT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvHostData, wkvShape, &wkvDeviceAddr, aclDataType::ACL_HIFLOAT8, &wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateHostData, wgateShape, &wgateDeviceAddr, aclDataType::ACL_HIFLOAT8, &wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateCacheHostData, stateCacheShape, &stateCacheDeviceAddr, aclDataType::ACL_FLOAT,
                          &stateCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(apeHostData, apeShape, &apeDeviceAddr, aclDataType::ACL_FLOAT, &ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xDescaleHostData, xDescaleShape, &xDescaleDeviceAddr, aclDataType::ACL_FLOAT, &xDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvDescaleHostData, wkvDescaleShape, &wkvDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &wkvDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateDescaleHostData, wgateDescaleShape, &wgateDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &wgateDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateBlockTableHostData, stateBlockTableShape, &stateBlockTableDeviceAddr,
                          aclDataType::ACL_INT32, &stateBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHostData, startPosShape, &startPosDeviceAddr, aclDataType::ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_BF16, &cmpKvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. Call aclnnQuantCompressorGetWorkspaceSize.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnQuantCompressorGetWorkspaceSize(x, wkv, wgate, stateCacheRef, ape, xDescale, wkvDescale, wgateDescale,
                                               stateBlockTable, nullptr, nullptr, startPos, quantMode, cmpRatio, coff,
                                               cacheMode, stateCacheStrideDim0, cmpKvOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantCompressorGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 6. Allocate workspace.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 7. Call aclnnQuantCompressor.
    ret = aclnnQuantCompressor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantCompressor failed. ERROR: %d\n", ret); return ret);

    // 8. Wait for synchronization.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 9. Obtain the output.
    LOG_PRINT("QuantCompressor execution succeeded.\n");
    PrintBf16Result(cmpKvShape, &cmpKvDeviceAddr);

    // 10. Release resources.
    aclDestroyTensor(x);
    aclDestroyTensor(wkv);
    aclDestroyTensor(wgate);
    aclDestroyTensor(stateCacheRef);
    aclDestroyTensor(ape);
    aclDestroyTensor(xDescale);
    aclDestroyTensor(wkvDescale);
    aclDestroyTensor(wgateDescale);
    aclDestroyTensor(stateBlockTable);
    aclDestroyTensor(startPos);
    aclDestroyTensor(cmpKvOut);

    aclrtFree(xDeviceAddr);
    aclrtFree(wkvDeviceAddr);
    aclrtFree(wgateDeviceAddr);
    aclrtFree(stateCacheDeviceAddr);
    aclrtFree(apeDeviceAddr);
    aclrtFree(xDescaleDeviceAddr);
    aclrtFree(wkvDescaleDeviceAddr);
    aclrtFree(wgateDescaleDeviceAddr);
    aclrtFree(stateBlockTableDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(cmpKvDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}

```
