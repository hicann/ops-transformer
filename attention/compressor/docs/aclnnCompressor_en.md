# aclnnCompressor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:26:37.708Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

- Description: Compressor is a preprocessing operator for SMLA and QLI in inference use cases. It compresses the KV cache of every 4 or 128 tokens into one, and then each token performs DSA computation with these compressed KV caches. For long sequences, Compressor effectively reduces the computation overhead. The main computation process is as follows:
    1. Perform Matmul on the input $X$ and $W^{KV}$ to obtain $kv\_state$, perform Matmul on the input $X$ and $W^{Gate}$ and then perform Add with $Ape$ to obtain $score\_state$. $kv\_state$ and $score\_state$ are updated based on the input start_pos and cu_seqlens.
    2. When coff is 2, perform data rearrangement on $kv\_state$ and $score\_state$.
    3. Perform softmax on $score\_state$, multiply the softmax result with $kv\_state$, and then perform ReduceSum.

- Formula:

    1. Compute the matrix multiplication:

        $$
        C4A: \left[kv\_state^a, score\_state^a\right] = X @ \left[W^{aKV}, W^{aGate}\right]^{T}, \left[kv\_state^b, score\_state^b\right] = X @ \left[W^{bKV}, W^{bGate}\right]^{T};
        $$

        $$
        C128A: \left[kv\_state, score\_state\right] = X @ \left[W^{KV}, W^{Gate}\right]^{T}
        $$

    2. Compute the grouped addition:

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

    5. Compute the grouped sum along the compression axis:

        $$
        C4A: C_{i}^{\text{Comp}} = \left[1\right]_{1\times8} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{4};
        $$

        $$
        C128A: C_{i}^{\text{Comp}} = \left[1\right]_{1\times128} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{128};
        $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnCompressorGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnCompressor" API to perform the computation.

```cpp
aclnnStatus aclnnCompressorGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *wkv,
    const aclTensor *wgate,
    aclTensor       *stateCacheRef,
    const aclTensor *ape,
    const aclTensor *stateBlockTableOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional,
    int64_t          cmpRatio,
    int64_t          coff,
    int64_t          cacheMode,
    int64_t          stateCacheStrideDim0,
    bool             gradEnabled,
    const aclTensor *cmpKvOut,
    const aclTensor *softmaxScoreOut,
    const aclTensor *kvOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnCompressor(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnCompressorGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 230px">
    <col style="width: 360px">
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 220px">
    <col style="width: 140px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x (const aclTensor*)</td>
      <td>Input</td>
      <td><span class="math-inline">X</span> in the formula indicates the original data without compression.</td>
      <td><ul><li>Supports empty tensors with B=0, S=0, and T=0.</li><li>B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, and T indicates the accumulated sum of sequence lengths of input samples in all batches.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>BS coaxial: [T,H]<br>BS non-coaxial: [B,S,H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>wkv (const aclTensor*)</td>
      <td>Input</td>
      <td><span class="math-inline">W<sup>KV</sup></span> in the formula indicates the KV compression weight.</td>
      <td>Does not support empty tensor.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[coff* D,H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>wgate (const aclTensor*)</td>
      <td>Input</td>
      <td><span class="math-inline">W<sup>Gate</sup></span> in the formula indicates the gate compression weight.</td>
      <td>Does not support empty tensor.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[coff* D,H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>stateCacheRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td><span class="math-inline">[kv_state, score_state]</span> in the formula indicates the historical data of kv_state and score_state. After computation, kv_state and score_state are updated in place in this Tensor</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[block_num,block_size,2* coff* D], requiring block_num>0; when cacheMode=2, block_size >= coff* cmp_ratio + S - 1 must be satisfied.</td>
      <td>Supports non-contiguous input on axis 0.</td>
    </tr>
    <tr>
      <td>ape (const aclTensor*)</td>
      <td>Input</td>
      <td><span class="math-inline">Ape</span> in the formula, which indicates positional biases.</td>
      <td>Empty tensor is not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[cmp_ratio,coff* D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>stateBlockTableOptional (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the block mapping table used for state_cache storage.</td>
      <td><ul><li>When cacheMode=1, the shape is [B, ceil(Smax/block_size)], where Smax is the maximum Sequence Length in each batch. When the shape of x is [B,S,H], Smax=max(start_pos)+S; when the shape of x is [T,H], Smax=max(start_pos)+max(cuSeqlensOptional[n+1] - cuSeqlensOptional[n]).</li><li>When cacheMode=2, the shape is [B].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>When cacheMode=1: [B, ceil(Smax/block_size)]<br>When cacheMode=2: [B]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqlensOptional (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the number of valid tokens in different batches. The value of each element in this parameter indicates the total number of tokens in the current batch and all previous batches, that is, the prefix sum. Therefore, the value of each element must be greater than or equal to that of the previous element, and the first element must be 0.</td>
      <td><ul><li>Supports empty tensors with B=0, S=0, and T=0.</li><li>If the shape of x is [B,S,H], a null pointer must be passed for this parameter.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>If the shape of x is [T,H], the input shape is [B+1,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedOptional (const aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the number of tokens actually involved in compression in each batch. If a null pointer is passed, the tokens are the same as the sequence length of each batch.</td>
      <td><ul><li>Supports empty tensors with B=0.</li><li>If the shape of x is [B,S,H], 0 ≤ sequsedOptional[n] ≤ S is required.</li><li>If the shape of x is [T,H], 0 ≤ sequsedOptional[n] ≤ cuSeqlensOptional[n+1] - cuSeqlensOptional[n] is required.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>startPosOptional(const aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the start position of the computation.</td>
      <td><ul><li>Supports empty tensors with B=0 and T=0.</li><li>If a null pointer is passed in, the computation starts from 0.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Used to perform sparse computation and indicates the data compression ratio.</td>
      <td>The value range is integers in [2, 128].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>coff (int64_t)</td>
      <td>Input</td>
      <td>Indicates whether to perform overlap data rearrangement.</td>
      <td>Value range: {1, 2}. When coff=1, no overlap data rearrangement is required; when coff=2, overlap data rearrangement is required.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the storage mode of state_cache.</td>
      <td>Value range: {1, 2}. 1 indicates a continuous buffer, and 2 indicates a circular buffer.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stateCacheStrideDim0 (int64_t)</td>
      <td>Input</td>
      <td>Indicates the stride of dim 0 of state_cache.</td>
      <td>Passed in by the framework based on the actual stride of state_cache.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradEnabled (bool)</td>
      <td>Input</td>
      <td>Indicates whether to export the softmax_score/kv intermediate results.</td>
      <td>Value range: true/false, default false. If the value is true, the operator outputs the softmax_score and kv intermediate results (for backward propagation); if the value is false, the softmax_score and kv outputs are invalid.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpKvOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the compressed data, corresponding to <span class="math-inline">C<sub>i</sub><sup>Comp</sup></span> in the formula.</td>
      <td><ul><li>Supports empty tensors with B=0, S=0, and T=0.</li><li>The output shape is [min(T, T/cmp_ratio+B), D] when BS is coaxial.</li><li>The output shape is [B, ceil(S/cmp_ratio), D] when BS is non-coaxial.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>BS coaxial: [min(T, T/cmp_ratio+B), D]<br>BS non-coaxial: [B, ceil(S/cmp_ratio), D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>softmaxScoreOut (aclTensor*)</td>
      <td>Output</td>
      <td><span class="math-inline">S<sup>′</sup></span> in the formula, which indicates the grouped softmax result.</td>
      <td>Valid only when gradEnabled is true; supports empty tensors with B=0, S=0, or T=0.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>BS coaxial: [min(T, T/cmp_ratio+B), coff*cmp_ratio, D]<br>BS non-coaxial: [B, ceil(S/cmp_ratio), coff*cmp_ratio, D]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>kvOut (aclTensor*)</td>
      <td>Output</td>
      <td><span class="math-inline">(S_H)<sub>i</sub></span> in the formula, which indicates the Hadamard product of the softmax result and kv_state.</td>
      <td>Output is valid only when gradEnabled is true; supports empty tensors with B=0, S=0, and T=0.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>Same as softmaxScoreOut</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - D (Head Dim) indicates the minimum unit size of the hidden layer. The value of D is obtained by dividing the size of the first dimension of wkv by coff.

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
    - cmpRatio supports only 2/4/8/16/32/64/128.
    - Setting gradEnabled to true is not supported.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 650px">
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
      <td>A null pointer exists in required parameters such as x, wkv, wgate, stateCacheRef, ape, and cmpKvOut.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of x, wkv, wgate, stateCacheRef, ape, and cmpKvOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape dimensions of x, wkv, wgate, stateCacheRef, and ape are not within the supported range.</td>
    </tr>
    <tr>
      <td>The values of cmpRatio, coff, and cacheMode are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape combination of cuSeqlensOptional, sequsedOptional, startPosOptional, and x does not meet the constraints.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>An exception occurs during tiling, or the dtype or shape of an input parameter is incorrect.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurs when the API calls an NPU Runtime interface to allocate memory (for example, the Runtime service is not started or memory application fails).</td>
    </tr>
  </tbody>
  </table>

## aclnnCompressor

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address applied by the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied by the device side, obtained by the first-phase API aclnnCompressorGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, including the computation process of the operator.</td>
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
  - aclnnCompressor defaults to a deterministic implementation.
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: batch consistency: Set ACL_OPT_DETERMINISTIC to 3 via aclrtSetSysParamOpt() to enable batch consistency. After batch consistency is enabled, computation results are independent of the batch size and position.
  <!-- end id8 -->
- Dimensions of x: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, D (Head Dim) indicates the size of the minimum unit of the hidden layer, and T indicates the accumulated sequence length of input samples across all batches.
- Input shape constraints:
  - stateCacheRef supports input shape [block_num, block_size, 2\*coff\*D], and block_num must be greater than 0; when cacheMode=2, block_size >= coff * cmp_ratio + S - 1 must be satisfied.
  - When BS coaxial is adopted for x, that is, the input shape of x is [T,H]:
    - The input shape of cuSeqlensOptional must be [B+1,]. This parameter is a prefix sum array. The value of each element must be greater than or equal to that of the previous element, the first element must be 0, and the last element must be T.
    - sequsedOptional supports input shape [B,], and 0 ≤ sequsedOptional[n] ≤ cuSeqlensOptional[n+1] - cuSeqlensOptional[n] is required.
    - When cacheMode=1, stateBlockTableOptional supports input shape [B, ceil(Smax/block_size)], where Smax=max(start_pos)+max(cuSeqlensOptional[n+1] - cuSeqlensOptional[n]); when cacheMode=2, the supported input shape is [B].
    - The output shape of cmpKvOut is [min(T, T/cmp_ratio+B), D]: compressed_tokens + compressed_tokens + ... + compressed_tokens + pad.
  - When BS coaxial is not adopted for x, that is, the input shape of x is [B,S,H]:
    - A null pointer must be passed to cuSeqlensOptional.
    - sequsedOptional supports input shape [B,], and 0 ≤ sequsedOptional[n] ≤ S is required.
    - When cacheMode=1, stateBlockTableOptional supports input shape of [B, ceil(Smax/block_size)], where Smax=max(start_pos)+S; when cacheMode=2, the supported input shape is [B].
    - The output shape of cmpKvOut is [B, ceil(S/cmp_ratio), D]: (compressed_tokens+pad0) + (compressed_tokens+pad1) + ... + (compressed_tokens+padN).
- Valid output range: compressed_tokens[n]=(start_pos[n]+Seqlen[n])/cmp_ratio-start_pos[n]/cmp_ratio, where Seqlen indicates the number of valid tokens in each batch. For the non-BS-merged layout, the valid results of each batch occupy the first compressed_tokens[n] rows; for the BS-merged layout, valid results are concatenated in batch order and the total valid range is the first sum(compressed_tokens[n]) rows. The remaining tail rows in the allocated capacity are not written by the operator.
- Input value range restrictions:
  - This API supports generalization of B and S, with the following use case restrictions:
    - In some long-sequence use cases, an excessively large computation amount may cause an error that NPU memory is exceeded. Note: The computation amount here is affected by the input shape of x. The larger the value, the greater the computation amount. Typical long-sequence use cases (that is, the product of B and S is large or T is large) include but are not limited to:

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
  - This API supports B, S, and T values of 0, which means input parameters whose shapes are related to the B, S, and T values allow empty tensors, while other input parameters do not support empty tensors. In this case, stateCacheRef is not updated, and the output cmpKvOut is an empty tensor.
  - The value range of stateBlockTableOptional elements is [0, block_num), where block_num indicates the size of dimension 0 of stateCacheRef. Element values are directly used as block indexes of state_cache, and out-of-range values cause illegal memory access. When the element value is 0: if cacheMode=1 (continuous buffer), the write operation to state_cache skips this position; if cacheMode=2 (circular buffer), neither read nor write operations skip it. The operator does not perform duplicate checks, and the caller must ensure the uniqueness of element values: if cacheMode=1, the element value 0 is an "unallocated" sentinel value and can appear repeatedly, while non-zero values must be globally unique; if cacheMode=2, 0 is a valid physical block number, and all element values must be globally unique. Duplicates cause multiple logical blocks/batches to write to the same physical block area, overwriting the data in state_cache.
- Input attribute restrictions:
  - Supports D of 128/512.
  - Supports H of 1K-10K, aligned to 512.
  - Supports blockSize of 1-1024.
  - Supports the following three typical combination use cases:
    - C4A: D=512, coff=2, cmp_ratio=4.
    - C4Li: D=128, coff=2, cmp_ratio=4.
    - C128A: D=512, coff=1, cmp_ratio=128.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_compressor.cpp
 * \brief Compressor operator aclnn call example (A5 / ascend950)
 *        Use case: C4A (D=512, coff=2, cmp_ratio=4, cache_mode=1, BSH layout, BF16)
 *        For A5, when B=1, S=4, tokenSize=4 <= 256, the FULL_LOAD high-performance template is triggered.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_compressor.h"
#include "opdev/bfloat16.h"

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
    // 1. Initialize the device/stream.
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Use case parameters: C4A (D=512, coff=2, cmp_ratio=4, cache_mode=1, BSH layout, BF16)
    int64_t B = 1;
    int64_t S = 4;
    int64_t hiddenSize = 4096;
    int64_t headDim = 512;
    int64_t coff = 2;
    int64_t cmpRatio = 4;
    int64_t cacheMode = 1; // 1: continuous buffer (LINEAR_BUFFER)
    int64_t blockSize = 128;

    int64_t Smax = S;
    int64_t maxBlockNumPerBatch = (Smax + blockSize - 1) / blockSize;
    int64_t blockNum = B * maxBlockNumPerBatch + 1;
    int64_t coffD = coff * headDim;
    int64_t stateCacheStrideDim0 = blockSize * 2 * coffD; // state_cache dim0 stride = dim1 * dim2

    // 2. Construct input and output shapes.
    std::vector<int64_t> xShape = {B, S, hiddenSize};
    std::vector<int64_t> wkvShape = {coffD, hiddenSize};
    std::vector<int64_t> wgateShape = {coffD, hiddenSize};
    std::vector<int64_t> stateCacheShape = {blockNum, blockSize, 2 * coffD};
    std::vector<int64_t> apeShape = {cmpRatio, coffD};
    std::vector<int64_t> stateBlockTableShape = {B, maxBlockNumPerBatch};
    std::vector<int64_t> startPosShape = {B};
    int64_t Sr = (S + cmpRatio - 1) / cmpRatio;
    std::vector<int64_t> cmpKvShape = {B, Sr, headDim};
    std::vector<int64_t> softmaxScoreShape = {B, Sr, coff * cmpRatio, headDim};
    std::vector<int64_t> kvShape = {B, Sr, coff * cmpRatio, headDim};

    // 3. Construct host data.
    int64_t xSize = GetShapeSize(xShape);
    int64_t wkvSize = GetShapeSize(wkvShape);
    int64_t wgateSize = GetShapeSize(wgateShape);
    int64_t stateCacheSize = GetShapeSize(stateCacheShape);
    int64_t apeSize = GetShapeSize(apeShape);
    int64_t cmpKvSize = GetShapeSize(cmpKvShape);

    std::vector<bfloat16> xHostData(xSize, bfloat16(0.1f));
    std::vector<bfloat16> wkvHostData(wkvSize, bfloat16(0.1f));
    std::vector<bfloat16> wgateHostData(wgateSize, bfloat16(0.1f));
    std::vector<float_t> stateCacheHostData(stateCacheSize, 0.1f);
    std::vector<float_t> apeHostData(apeSize, 0.1f);
    std::vector<int32_t> stateBlockTableHostData;
    for (int64_t i = 0; i < B * maxBlockNumPerBatch; i++) {
        stateBlockTableHostData.push_back(static_cast<int32_t>(i + 1));
    }
    std::vector<int32_t> startPosHostData(B, 0);
    std::vector<bfloat16> cmpKvHostData(cmpKvSize, bfloat16(0.0f));
    int64_t softmaxScoreSize = GetShapeSize(softmaxScoreShape);
    int64_t kvSize = GetShapeSize(kvShape);
    std::vector<float_t> softmaxScoreHostData(softmaxScoreSize, 0.0f);
    std::vector<float_t> kvHostData(kvSize, 0.0f);

    // 4. Create aclTensor.
    void *xDeviceAddr = nullptr;
    void *wkvDeviceAddr = nullptr;
    void *wgateDeviceAddr = nullptr;
    void *stateCacheDeviceAddr = nullptr;
    void *apeDeviceAddr = nullptr;
    void *stateBlockTableDeviceAddr = nullptr;
    void *startPosDeviceAddr = nullptr;
    void *cmpKvDeviceAddr = nullptr;
    void *softmaxScoreDeviceAddr = nullptr;
    void *kvDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *wkv = nullptr;
    aclTensor *wgate = nullptr;
    aclTensor *stateCacheRef = nullptr;
    aclTensor *ape = nullptr;
    aclTensor *stateBlockTable = nullptr;
    aclTensor *startPos = nullptr;
    aclTensor *cmpKvOut = nullptr;
    aclTensor *softmaxScoreOut = nullptr;
    aclTensor *kvOut = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvHostData, wkvShape, &wkvDeviceAddr, aclDataType::ACL_BF16, &wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateHostData, wgateShape, &wgateDeviceAddr, aclDataType::ACL_BF16, &wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateCacheHostData, stateCacheShape, &stateCacheDeviceAddr, aclDataType::ACL_FLOAT,
                          &stateCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(apeHostData, apeShape, &apeDeviceAddr, aclDataType::ACL_FLOAT, &ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateBlockTableHostData, stateBlockTableShape, &stateBlockTableDeviceAddr,
                          aclDataType::ACL_INT32, &stateBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHostData, startPosShape, &startPosDeviceAddr, aclDataType::ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_BF16, &cmpKvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxScoreHostData, softmaxScoreShape, &softmaxScoreDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxScoreOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kvHostData, kvShape, &kvDeviceAddr, aclDataType::ACL_FLOAT, &kvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. Call aclnnCompressorGetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    bool gradEnabled = false;

    ret = aclnnCompressorGetWorkspaceSize(x, wkv, wgate, stateCacheRef, ape, stateBlockTable, nullptr, nullptr,
                                          startPos, cmpRatio, coff, cacheMode, stateCacheStrideDim0, gradEnabled,
                                          cmpKvOut, softmaxScoreOut, kvOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCompressorGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 6. Allocate workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 7. Call aclnnCompressor
    ret = aclnnCompressor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCompressor failed. ERROR: %d\n", ret); return ret);

    // 8. Wait for synchronization
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 9. Obtain the output
    LOG_PRINT("Compressor execution succeeded.\n");
    PrintBf16Result(cmpKvShape, &cmpKvDeviceAddr);

    // 10. Release resources
    aclDestroyTensor(x);
    aclDestroyTensor(wkv);
    aclDestroyTensor(wgate);
    aclDestroyTensor(stateCacheRef);
    aclDestroyTensor(ape);
    aclDestroyTensor(stateBlockTable);
    aclDestroyTensor(startPos);
    aclDestroyTensor(cmpKvOut);
    aclDestroyTensor(softmaxScoreOut);
    aclDestroyTensor(kvOut);

    aclrtFree(xDeviceAddr);
    aclrtFree(wkvDeviceAddr);
    aclrtFree(wgateDeviceAddr);
    aclrtFree(stateCacheDeviceAddr);
    aclrtFree(apeDeviceAddr);
    aclrtFree(stateBlockTableDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(cmpKvDeviceAddr);
    aclrtFree(softmaxScoreDeviceAddr);
    aclrtFree(kvDeviceAddr);
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
