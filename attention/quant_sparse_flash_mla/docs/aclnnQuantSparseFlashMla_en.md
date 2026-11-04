# aclnnQuantSparseFlashMla

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:55:25.981Z -->

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

- Description:

  The `aclnnQuantSparseFlashMla` operator performs MLA (Multi-head Latent Attention) attention computation in full-quantization and sparse scenarios. This API supports the following three computation modes:
  - **SWA (Sliding Window Attention)**: Only `oriKvOptional` is passed in to perform sliding window attention on the original KV.
  - **CSA (Compressed Sparse Attention)**: `oriKvOptional`, `cmpKvOptional`, and `cmpSparseIndicesOptional` are passed in together to perform attention on both the original KV window and the compressed KV selected by topK.
  - **HCA (Heavily Compressed Attention)**: Pass in both `oriKvOptional` and `cmpKvOptional`. Compute attention jointly over the original KV window and the continuous compressed KV segments.

  `aclnnQuantSparseFlashMlaMetadata` provides the core partition information of `aclnnQuantSparseFlashMla` and is generated before the main operator is executed. In the current version, this metadata must be passed in to the main operator. The typical call flow is as follows:

  **This operator is not recommended for standalone use. Use it together with the aclnnQuantSparseFlashMlaMetadata operator to form a complete workflow.**

  1. Prepare the corresponding inputs based on the use case.
  2. Call `aclnnQuantSparseFlashMlaMetadata` to generate `metadata`, and pass it as an input parameter of `aclnnQuantSparseFlashMla`.
  3. Call `aclnnQuantSparseFlashMla` to generate the computation result.

- Formula:

  $$
  O = \text{softmax}(Q \cdot \tilde{K}^T \cdot \text{softmaxScale}) \cdot \tilde{V}
  $$

  where $\tilde{K} = \tilde{V}$ (shared KV), and $\tilde{K}$ is obtained by concatenating the original KV in the sliding window and the compressed KV within the causal boundary. The specific KV range participating in the calculation is determined by the template mode and mask parameters:

  - Sliding window part (oriKv): For the $i_{S1}$-th Query token, its causal diagonal position is $\text{oriThreshold} = S2_{act} - S1_{act} + i_{S1} + 1$, and the window range is $[\max(\text{oriThreshold} - \text{oriWinLeft} - 1, 0), \text{ori\_threshold} + \text{ori\_win\_right})$.

  - Compressed KV part (cmpKv): The causal boundary threshold is $\text{cmpThreshold} = \lfloor \frac{\text{oriThreshold}}{\text{cmpRatio}} \rfloor$. In the HCA use case, the continuous compressed KV within $[0, \text{cmpThreshold})$ is taken; in the CSA use case, blocks are collected from the compressed KV on demand via TopK indices, and only blocks with $\text{beginIdx} < \text{cmpThreshold}$ are retained.

  The attention computation uses Online Softmax (Flash Attention V2). Iterations are performed in blocks along the S2 dimension, and sinks is used as the initial maximum value of the softmax for each row:

  $$
  \text{rowMax}^{(0)} = \text{sinks}[g], \quad \text{rowSum}^{(0)} = 1.0, \quad O^{(0)} = 0
  $$

  $$
  S^{(t)} = Q \cdot K_{tile}^{(t)T} \cdot \text{softmaxScale}
  $$

  $$
  \text{rowMax}^{(t+1)} = \max(\text{rowMax}^{(t)}, \max(S^{(t)}, \text{dim}=-1))
  $$

  $$
  \text{rowSum}^{(t+1)} = \exp(\text{rowMax}^{(t)} - \text{rowMax}^{(t+1)}) \cdot \text{rowSum}^{(t)} + \sum \exp(S^{(t)} - \text{rowMax}^{(t+1)})
  $$

  $$
  O^{(t+1)} = \exp(\text{rowMax}^{(t)} - \text{rowMax}^{(t+1)}) \cdot O^{(t)} + \exp(S^{(t)} - \text{rowMax}^{(t+1)}) \cdot V_{tile}^{(t)}
  $$

  $$
  O_{final} = O^{(T_{s2})} / \text{rowSum}^{(T_{s2})}
  $$

- Symbol Description

  | Symbol              | Meaning                                               |
  | ------------------- | --------------------------------------------------------- |
  | $Q$                 | Query input with the [G, D] shape (single row)             |
  | $K_{tile}^{t}$      | KV data of the t-th S2 block, where K=V (shared KV)         |
  | $S_t$               | Scaled QK attention scores of the t-th block                |
  | $P_t$               | Softmax probabilities of the t-th block                     |
  | $O_t$               | Accumulated output after the t-th block                     |
  | $softmaxScale$      | Scaling coefficient, usually the reciprocal of the square root of the dimension of each attention head |
  | $B$                 | Batch size                                                  |
  | $S1$/$S1_{act}$     | Query sequence length/actual valid length                   |
  | $S2$/$S2_{act}$     | Original KV sequence length/actual valid length             |
  | $N1$                | Number of Query heads                                       |
  | $N2$                | Number of KV heads                                          |
  | $G$                 | GQA grouping ratio, $G=N1/N2$                               |
  | $D$                 | Dimension of each attention head                            |
  | $sinks$             | Attention sinks with the [N1] shape                         |
  | $cmpRatio$          | Compression ratio of cmpKv, used to convert the pre-compression KV length of the cmp-side mask |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnQuantSparseFlashMlaGetWorkspaceSize` API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the `aclnnQuantSparseFlashMla` API to perform the actual computation.

```Cpp
aclnnStatus aclnnQuantSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
    const aclTensor *qDescaleOptional,
    const aclTensor *oriKvDescaleOptional,
    const aclTensor *cmpKvDescaleOptional,
    const aclTensor *oriSparseIndicesOptional,
    const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional,
    const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    int64_t          quantMode,
    double           softmaxScale,
    int64_t          cmpRatio,
    int64_t          oriMaskMode,
    int64_t          cmpMaskMode,
    int64_t          oriWinLeft,
    int64_t          oriWinRight,
    char            *layoutQOptional,
    char            *layoutKvOptional,
    int64_t          topkValueMode,
    bool             returnSoftmaxLse,
    const aclTensor *attnOutOut,
    const aclTensor *softmaxLseOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantSparseFlashMla(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantSparseFlashMlaGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 300px">
    <col style="width: 300px">
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 280px">
    <col style="width: 100px">
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
      <td>q (aclTensor*)</td>
      <td>Input</td>
      <td>Query input tensor.</td>
      <td>Empty tensors are not supported. qN supports only 1 to 128; D supports only 512.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>        <ul>
          <li>If layoutQ is BSND: (b, qS, qN, qD)</li>
          <li>If layoutQ is TND: (qT, qN, qD)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Original KV input tensor. Key and Value share the same data.</td>
      <td>Required in the current version. Corresponds to part of K and V in the formula, indicating the original quantized KV without compression.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutKv is PA_BBND: (oriKvBlockNums, oriKvBlockSize, kvN, kvD), oriKvBlockSize supports 1 to 1024</li>
          <li>When layoutKv is BSND: (b, oriKvS, kvN, kvD)</li>
          <li>When layoutKv is TND: (oriKvT, kvN, kvD)</li>
        </ul>
        kvN supports only 1, and kvD supports only 512.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Compressed KV input tensor. Key and Value share the same data.</td>
      <td>Optional input. Corresponds to part of K and V in the formula, indicating the compressed quantized KV.</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>        <ul>
          <li>If layoutKv is PA_BBND: (cmpKvBlockNums, cmpKvBlockSize, kvN, kvD). cmpKvBlockSize supports 1 to 1024</li>
          <li>If layoutKv is BSND: (b, cmpKvS, kvN, kvD)</li>
          <li>If layoutKv is TND: (cmpKvT, kvN, kvD)</li>
        </ul>
        kvN supports only 1, and kvD supports only 512.
      </td>
      <td>√</td>
    </tr>
    <tr>
    <td>qDescaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization parameter of q.</td>
      <td>Currently required.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>        The current shape is fixed to [1], corresponding to per-tensor quantization.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriKvDescaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization parameter corresponding to oriKvOptional.</td>
      <td>Must be passed currently.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>        The shape is currently fixed to [1], corresponding to per-tensor quantization.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpKvDescaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization parameter corresponding to cmpKvOptional.</td>
      <td>Optional input.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>        The shape is currently fixed to [1], corresponding to per-tensor quantization.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriSparseIndicesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the index for discrete retrieval from oriKvCache.</td>
      <td>Optional input. Invalid positions are filled with -1, and the rest are non-negative integers.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutQ is BSND: (b, qS, kvN, oriKvK)</li>
          <li>When layoutQ is TND: (qT, kvN, oriKvK)</li>
        </ul>
        Here oriKvK is the TopK sparse selection count for oriKvOptional, and the supported range is greater than 0.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpSparseIndicesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the TopK indices for discretely fetching cmpKvCache.</td>
      <td>Optional input. Invalid positions are filled with -1, and the rest are non-negative integers.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutQ is BSND: (b, qS, kvN, cmpKvK)</li>
          <li>When layoutQ is TND: (qT, kvN, cmpKvK)</li>
        </ul>
        cmpKvK is the number of TopK sparse selections from cmpKvOptional. Values greater than 0 are supported.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriBlockTableOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Block mapping table used for oriKvCache storage in PageAttention.</td>
      <td>Must be passed in when layoutKv is PA_BBND. The length of the second dimension must not be less than the number of blocks corresponding to the maximum S2 among all batches.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b, Ceil(oriKvSMax/oriKvBlockSize))</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpBlockTableOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Block mapping table used for cmpKvCache storage in PageAttention.</td>
      <td>Must be passed in when cmpKvOptional is passed and layoutKv is PA_BBND.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b, Ceil(cmpKvSMax/cmpKvBlockSize))</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of valid tokens of q in different batches (prefix sum form).</td>
      <td>Must be passed in when layoutQOptional is TND. Each element indicates the total number of tokens of the current batch and all previous batches.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of valid tokens of oriKv in different batches (prefix sum form).</td>
      <td>Must be passed when layoutKvOptional is TND.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of valid tokens of cmpKv in different batches (prefix sum form).</td>
      <td>Must be passed in when layoutKvOptional is TND and cmpKvOptional exists.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of tokens of q that actually participate in the calculation in different batches.</td>
      <td>Currently, specifying this parameter is not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of tokens in oriKv that actually participate in the computation in different batches.</td>
      <td>Must be passed in when layoutKvOptional is PA_BBND; optionally passed when layoutKvOptional is BSND to specify the valid length of oriKv of each batch; when layoutKvOptional is TND, cuSeqlensOriKvOptional is used to express sequence boundaries.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of tokens in cmpKv actually participating in the calculation for different batches.</td>
      <td>Optional input. When passed, the shape must be (B,), serving as the logical valid length of cmp for each batch. It takes precedence over inference from the cmpKvOptional shape, cuSeqlensCmpKvOptional, or the PA block table.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Residual compressed KV, used to restore the KV length before compression used by the mask on the cmp side.</td>
      <td>Optional input. If passed, the shape must be (B,). The b-th batch restores the KV length before compression using cmp_len * cmpRatio + cmpResidualKvOptional[b]. Must be passed when cmpMaskMode is 3 and cmpRatio != 1; it cannot be passed when cmpMaskMode is 0 or cmpRatio equals 1.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Used to identify the length of oriSparseIndicesOptional actually participating in the calculation.</td>
      <td>Must be passed when oriMaskMode=0 and oriSparseIndicesOptional is passed in; other scenarios do not support passing it.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutQ is BSND: (b, qS, kvN)</li>
          <li>When layoutQ is TND: (qT, kvN)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Identifies the length of calculation actually participating in cmpSparseIndicesOptional.</td>
      <td>Must be passed when cmpMaskMode=0 and cmpSparseIndicesOptional is passed in; other scenarios do not support passing it in.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutQ is BSND: (b, qS, kvN)</li>
          <li>When layoutQ is TND: (qT, kvN)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>sinksOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents an independent learnable virtual bias term set for each attention head, used to maintain stability in long-text inference.</td>
      <td>Must be passed in.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(qN,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>metadataOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Core partition result of the AICPU operator aclnnQuantSparseFlashMlaMetadata.</td>
      <td>Must be passed in. Generated by the aclnnQuantSparseFlashMlaMetadata operator.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the quantization mode.</td>
      <td>Quantization mode 1 indicates that Q, K, and V use per-tensor quantization and the data type is HIFLOAT8.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxScale (double)</td>
      <td>Input</td>
      <td>Scaling factor, corresponding to softmaxScale in the formula.</td>
      <td>The recommended value is 1/√D, where D is the dimension of each attention head.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Compression ratio of cmpKv relative to the pre-compression KV length. Used to restore the pre-compression KV length used by the mask on the cmp side.</td>
      <td>cmpRatio supports values from 1 to 128. When cmpKv is not passed, only the default value 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Mask mode for computing q and oriKv.</td>
      <td>Supported values:<br>0: No Mask.<br>3: RightDownCausal mode.<br>4: Band mode.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode for the computation of q and cmpKv.</td>
      <td>Supported values:<br>0: No Mask.<br>3: RightDownCausal mode.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of past tokens that q attends to in the computation of q and oriKvOptional.</td>
      <td>Supports -1 or a non-negative number.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of future tokens that q attends to in the calculation of q and oriKvOptional.</td>
      <td>Supports -1 or a non-negative number.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional (char*)</td>
      <td>Input</td>
      <td>Identifies the data layout format of input q.</td>
      <td>Supports "BSND" and "TND".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional (char*)</td>
      <td>Input</td>
      <td>Identifies the data layout format of the input oriKvOptional and cmpKvOptional.</td>
      <td>Supports "PA_BBND", "BSND", and "TND".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkValueMode (int64_t)</td>
      <td>Input</td>
      <td>Mode of topk index values.</td>
      <td>Currently, 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse (bool)</td>
      <td>Input</td>
      <td>Whether to return softmaxLse.</td>
      <td>Supports true or false.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOutOut (aclTensor*)</td>
      <td>Output</td>
      <td>Attention score computation output.</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>Same shape as q</td>
      <td>×</td>
    </tr>
    <tr>
      <td>softmaxLseOutOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Log-sum-exp result of softmax.</td>
      <td>Returns a placeholder tensor when returnSoftmaxLse is false; returns the log-sum-exp result of softmax when returnSoftmaxLse is true.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutQ is BSND: (b, kvN, qS, qN/kvN)</li>
          <li>When layoutQ is TND: (kvN, qS, qN/kvN)</li>
          <li>When returnSoftmaxLse is false: placeholder tensor</li>
        </ul>
      </td>
      <td></td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Common Field Descriptions**

|    Name    |                            Meaning                            |
| :---------: | :---------------------------------------------------------: |
|      b      |      Batch size of input samples.                |
|     qS     |      Sequence length of input q.      |
|    oriKvS    |  Sequence length of input oriKvOptional.  |
|    cmpKvS    |  Sequence length of input cmpKvOptional.  |
|     qN     |        Number of heads of input q.        |
|    kvN    |    Number of heads of input oriKvOptional/cmpKvOptional.    |
|      qD      |          Dimension of the attention head of input q.         |
|      kvD      |          Dimension of the attention head of input oriKvOptional/cmpKvOptional.         |
|     qT     |          Accumulated sum of sequence lengths of all batches in input q.          |
|     oriKvT    |          Accumulated sum of sequence lengths of all batches in input oriKvOptional.          |
|     cmpKvT    |          Accumulated sum of sequence lengths of all batches in input cmpKvOptional.          |
|      oriKvK      |           Number of tokens selected by topK in input oriSparseIndicesOptional.         |
|      cmpKvK      |           Number of tokens selected by topK in input cmpSparseIndicesOptional.         |
|      oriKvSMax      |           Maximum sequence length of input oriKvOptional.         |
|      cmpKvSMax      |           Maximum sequence length of input cmpKvOptional.         |
|      oriKvBlockSize      |           Block size of input oriKvOptional in the PagedAttention use case.         |
|      cmpKvBlockSize      |           Block size of input cmpKvOptional in the PagedAttention use case.         |
|      oriKvBlockNums      |           Number of blocks of input oriKvOptional in the PagedAttention use case.         |
|      cmpKvBlockNums      |           Number of blocks of input cmpKvOptional in the PagedAttention use case.         |

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1200px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 817px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed-in parameter is a required input, output, or required attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The data type, data format, and attribute value of the input variable are out of the supported range.</td>
    </tr>

  </tbody>
  </table>

## aclnnQuantSparseFlashMla

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnQuantSparseFlashMlaGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>The op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>The stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation

  - aclnnQuantSparseFlashMla defaults to a deterministic implementation. Multiple calls with the same input produce the same result.

- Usage constraints

  - Inputs such as cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriBlockTableOptional, and cmpBlockTableOptional are tensors. The operator cannot obtain the actual values of tensors during the Tiling phase, so tiling does not validate the values. You must ensure correctness. If invalid values are passed to these parameters, undefined behavior occurs (such as precision issues and program crashes caused by illegal memory access).
  - The input parameters of aclnnQuantSparseFlashMlaMetadata and aclnnQuantSparseFlashMla should be consistent across calls. The operator is split into two APIs called separately and cannot perform validation itself. You must ensure correctness. If the parameters passed to the APIs are inconsistent, undefined behavior occurs (such as precision issues and program crashes caused by illegal memory access).
  - oriTopkLengthOptional and cmpTopkLengthOptional indicate the lengths of ori/cmp sparseIndices actually participating in calculation. The values must not be greater than the size of the last dimension of sparseIndicesOptional. When sequsedQOptional is passed in, the values in the valid part of topkLength must be greater than or equal to 0.
  - When oriMaskMode/cmpMaskMode is 0, oriKvK/cmpKvK must be greater than or equal to the maximum value of oriTopkLengthOptional/cmpTopkLengthOptional.
  - cmpResidualKvOptional is used together with cmpRatio to restore the KV length before compression. The value of each batch must be less than cmpRatio. It can be passed in only when cmpMaskMode is 3 and cmpRatio != 1; it must not be passed in when cmpMaskMode is 0 or cmpRatio equals 1.
  - attnOutOut: tensor type, output in the formula. Supported data type: BFLOAT16. Supported data format: ND. Constraint: the shape of this output parameter must be the same as the shape of input q.
  - When returnSoftmaxLse=False, a tensor with shape [1] and value 0 is returned; when returnSoftmaxLse=True, the FLOAT32 log-sum-exp result is returned.
  - cuSeqlensQOptional, cuSeqlensOriKvOptional, and cuSeqlensCmpKvOptional must satisfy the following requirements: the first element is 0 and the sequence is non-decreasing as a whole, that is, each element is not less than its previous element.
  - When layoutKv is PA_BBND, oriKvOptional and cmpKvOptional support non-contiguous axis 0.
  - For dimensions denoted by the same symbol in the shapes of parameters, the actual values of the corresponding axes must be consistent.

### Feature Parameter Group

|      Feature Parameter Group      |     Parameter Field Name     |
| :-------------------: | :-------------------: |
|      Common parameter group      | q, quantMode, oriKvOptional, cmpKvOptional, qDescaleOptional, oriKvDescaleOptional, cmpKvDescaleOptional, metadataOptional, softmaxScale, layoutQ, layoutKv, attnOutOut |
|      Mask parameter group      | oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight |
|   SeqLens parameter group   | cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional |
|   Sparse compression parameter group    | cmpRatio, cmpResidualKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional, oriTopkLengthOptional, cmpTopkLengthOptional, topkValueMode |
| Paged Attention parameter group | oriBlockTableOptional, cmpBlockTableOptional |
|   Sinks parameter group   | sinksOptional |
|   SoftmaxLse parameter group   | returnSoftmaxLse, softmaxLseOutOptional |

### Computation Mode Description

|    Name    |    Parameters Required for Typical Use Case    |    Parameters Required for Fully Sparse Use Case    |
| :---------: | :--------------------------------: | :---------------------------: |
|      SWA      | oriKvOptional | oriKvOptional, oriKvSparseIndicesOptional |
|      HCA      | oriKvOptional, cmpKvOptional|-|
|      CSA      | oriKvOptional, cmpKvOptional, cmpKvSparseIndicesOptional| oriKvOptional, oriKvSparseIndicesOptional, cmpKvOptional, cmpKvSparseIndicesOptional|

### Parameter Group Constraints

#### Common Parameter Group

- Handling of empty input parameters:
  - An empty tensor means that the shape size of a required input, or of an input or output that must be passed in a specific use case, is 0, that is, any axis is 0.
  - All use cases that trigger an empty tensor are intercepted with an error.

- Validation of q, oriKvOptional, cmpKvOptional, qDescaleOptional, oriKvDescaleOptional, cmpKvDescaleOptional, and attnOutOut

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 232px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Single-Parameter Validation</th>
    <th>Existence Validation</th>
    <th>Consistency Validation</th>
    <th>Feature Cross-Validation</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>            <ul>
                <li>dtype supports HIFLOAT8</li>
                <li>When layoutQ is BSND, the shape of q is (b, qS, qN, qD)</li>
                <li>When layoutQ is TND, the shape of q is (qT, qN, qD)</li>
            </ul>
        </td>
        <td>            Must be passed in.
        </td>
        <td rowspan="4">            <ul>
                <li>The shapes of q and attnOutOut must be the same</li>
                <li>If cmpKvOptional is passed in, the dtypes of oriKvOptional and cmpKvOptional must be consistent</li>
                <li>When layoutKv is not PA_BBND, layoutQ and layoutKv must be consistent</li>
                <li>When layoutKv is PA_BBND, layoutQ can be BSND or TND</li>
                <li>When quantMode=1, the dtypes of q, oriKvOptional, and cmpKvOptional are HIFLOAT8</li>
            </ul>
        </td>
        <td rowspan="4">            Dimension validation:
            <ul>
                <li>b > 0</li>
                <li>qS > 0</li>
                <li>0 < qN <= 128</li>
                <li>qD = 512</li>
                <li>qT > 0</li>
                <li>oriKvS > 0</li>
                <li>cmpKvS > 0</li>
                <li>kvN = 1</li>
                <li>kvD = 512</li>
                <li>oriKvT > 0</li>
                <li>cmpKvT > 0</li>
                <li>oriKvBlockNums > 0</li>
                <li>cmpKvBlockNums > 0</li>
                <li>1 <= oriKvBlockSize <= 1024</li>
                <li>1 <= cmpKvBlockSize <= 1024</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>oriKvOptional</td>
        <td>            <ul>
                <li>dtype supports HIFLOAT8</li>
                <li>When layoutKv is BSND, the shape of oriKv is (b, oriKvS, kvN, kvD)</li>
                <li>When layoutKv is TND, the shape of oriKv is (oriKvT, kvN, kvD)</li>
                <li>When layoutKv is PA_BBND, the shape of oriKv is (oriKvBlockNums, oriKvBlockSize, kvN, kvD)</li>
            </ul>
        </td>
        <td>            Must be passed in the current version.
        </td>
    </tr>
    <tr>
        <td>attnOutOut</td>
        <td>            <ul>
                <li>dtype supports BFLOAT16</li>
                <li>When layoutQ is BSND, the shape of attnOutOut is (b, qS, qN, qD)</li>
                <li>When layoutQ is TND, the shape of attnOutOut is (qT, qN, qD)</li>
            </ul>
        </td>
        <td>            Must be passed in.
        </td>
    </tr>
    <tr>
        <td>cmpKvOptional</td>
        <td>            <ul>
                <li>dtype supports HIFLOAT8.</li>
                <li>If layoutKv is BSND, the shape of cmpKvOptional is (b, cmpKvS, kvN, kvD).</li>
                <li>If layoutKv is TND, the shape of cmpKvOptional is (cmpKvT, kvN, kvD).</li>
                <li>If layoutKv is PA_BBND, the shape of cmpKvOptional is (cmpKvBlockNums, cmpKvBlockSize, kvN, kvD).</li>
            </ul>
        </td>
        <td>            Optional input
        </td>
    </tr>
    <tr>
        <td>qDescaleOptional</td>
        <td>            dtype supports FLOAT32.
        </td>
        <td rowspan="2">            Required in the current version
        </td>
        <td rowspan="3">            None
        </td>
        <td rowspan="3">            If quantMode is 1, the shape is [1].
        </td>
    </tr>
    <tr>
        <td>oriKvDescaleOptional</td>
        <td>            dtype supports FLOAT32.
        </td>
    </tr>
    <tr>
        <td>cmpKvDescaleOptional</td>
        <td>            dtype: FLOAT32
        </td>
        <td>            <li>Must be passed in when cmpKvOptional is passed.</li>
            <li>Does not support passing when cmpKvOptional is empty.</li>
        </td>
    </tr>
</tbody>
</table>
Layout matching relationship table:
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 400px">
<col style="width: 400px">
<col style="width: 400px">
<col style="width: 400px">
</colgroup>
<thead>
<tr>
    <th>layoutQ</th>
    <th>layoutKv</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>BSND</td>
        <td>          <li>BSND</li>
          <li>PA_BBND</li>
        </td>
    </tr>
    <tr>
        <td>TND</td>
        <td>          <li>TND</li>
          <li>PA_BBND</li>
        </td>
    </tr>
</tbody>
</table>

metadataOptional Validation
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>metadataOptional</td>
            <td>                <ul>
                    <li>dtype only supports INT32.</li>
                    <li>shape is dynamically calculated by aclnnQuantSparseFlashMlaMetadata.</li>
                </ul>
            </td>
            <td>Must be passed in the current version</td>
            <td>None</td>
            <td>When passed, it must be consistent with the result generated by aclnnQuantSparseFlashMlaMetadata.</td>
        </tr>
    </tbody>
</table>

#### Mask parameter group

<ul>
    <li>oriMaskMode/cmpMaskMode=0, full computation mode (default value)</li>
    <li>oriMaskMode/cmpMaskMode=3, Causal mode</li>
    <li>oriMaskMode=4, SlidingWindow mode</li>
</ul>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 200px">
        <col style="width: 100px">
        <col style="width: 200px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>oriMaskMode</td>
            <td>                <ul>
                    <li>The dtype is INT32.</li>
                    <li>The supported input range is only 0, 3, and 4. The default value is 0.</li>
                </ul>
            </td>
            <td>                Optional. If not passed, the default value is 0.
            </td>
            <td>                None
            </td>
            <td>                <ul>
                     <li>oriMaskMode supports 0, 3, and 4.</li>
                 </ul>
             </td>
         </tr>
         <tr>
             <td>cmpMaskMode</td>
             <td>                 <ul>
                     <li>dtype supports INT32.</li>
                     <li>Supports an input range of only 0 and 3. The default value is 0.</li>
                 </ul>
             </td>
             <td>                 Optional. If not passed, the default value is 0.
             </td>
             <td>                 None
             </td>
             <td>                <ul>
                    <li>When cmpKv is not passed, cmpMaskMode must be 0.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>oriWinLeft/oriWinRight</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>Supports an input range of only -1 and >= 0.</li>
                </ul>
            </td>
            <td>                Optional. If not passed, the default value is -1.
            </td>
            <td>                None
            </td>
            <td>                <ul>
                    <li>oriWinLeft/oriWinRight can be >=0 only when oriMaskMode is 4.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### SeqLens Parameter Group

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>sequsedQOptional</td>
            <td>                <ul>
                    <li>The dtype is INT32.</li>
                    <li>The shape is (b,).</li>
                    <li>The value only supports non-negative integers.</li>
                    <li>Values in sequsedQOptional must be less than or equal to qS.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>sequsedOriKvOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>shape is (b,).</li>
                    <li>The value only supports non-negative integers.</li>
                    <li>Values in sequsedOriKvOptional must be less than or equal to kvS.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional when layoutKv is BSND.</li>
                    <li>Must be passed in when layoutKv is PA_BBND.</li>
                    <li>Optional when oriTopkLengthOptional is passed.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>sequsedCmpKvOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>shape is (b,).</li>
                    <li>The value only supports non-negative integers.</li>
                    <li>Values in sequsedCmpKvOptional must be less than or equal to kvS.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional when layoutKv is BSND.</li>
                    <li>Must be passed in when layoutKv is PA_BBND and cmpKvOptional is passed.</li>
                    <li>Optional when cmpTopkLengthOptional is passed.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cuSeqlensQOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>The shape is (b+1,).</li>
                    <li>The value only supports non-negative integers.</li>
                    <li>Its value should be non-decreasing (greater than or equal to the previous value); the first element is 0 and the last element equals qT.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when layoutQ is TND.</li>
                    <li>Passing is not supported when layoutQ is not TND.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cuSeqlensOriKvOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>The shape is (b+1,).</li>
                    <li>The value only supports non-negative integers.</li>
                    <li>Its value should be non-decreasing (greater than or equal to the previous value); the first element is 0 and the last element equals oriKvT.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when layoutKv is TND.</li>
                    <li>Passing is not supported when layoutKv is not TND.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cuSeqlensCmpKvOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>shape: (b+1,).</li>
                    <li>The value only supports non-negative integers.</li>
                    <li>Its values should be non-decreasing (greater than or equal to the previous value); the first element is 0 and the last element equals cmpKvT.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when layoutKv is TND.</li>
                    <li>Passing is not supported when layoutKv is not TND.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
    </tbody>
</table>

#### Sparse Compression Parameter Group

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>cmpRatio</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>Indicates the compression ratio of cmpKvOptional relative to the KV length before compression. The value must be greater than 0.</li>
                    <li>Default value is 1.</li>
                </ul>
            </td>
            <td>                Optional, default value is 1
            </td>
            <td>None</td>
            <td>                When cmpKv is not passed, only the default value 1 is supported.
            </td>
        </tr>
        <tr>
            <td>cmpResidualKvOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>shape is (b,).</li>
                    <li>value only supports non-negative integers.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Only validated when cmpKvOptional is passed.</li>
                    <li>Optional</li>
                    <li>Mandatory when cmpMaskMode=3 and cmpRatio!=1.</li>
                </ul>
            </td>
            <td>                None
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>oriSparseIndicesOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>shape is (qT, kvN, oriKvK) or (b, qS, kvN, oriKvK).</li>
                    <li>Invalid positions are filled with -1, and the rest are non-negative integers.</li>
                </ul>
            </td>
            <td>                Optional
            </td>
            <td>                <ul>
                    <li>When layoutQ is TND, the shape is (qT, kvN, oriKvK).</li>
                    <li>When layoutQ is BSND, the shape is (b, qS, kvN, oriKvK).</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>cmpSparseIndicesOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>The shape is (qT, kvN, cmpKvK) or (b, qS, kvN, cmpKvK).</li>
                    <li>Invalid positions are filled with -1, and the rest are non-negative integers.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Validated only when cmpSparseIndicesOptional is passed.</li>
                    <li>Optional.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layoutQ is TND, the shape is (qT, kvN, cmpKvK).</li>
                    <li>When layoutQ is BSND, the shape is (b, qS, kvN, cmpKvK).</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>oriTopkLengthOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>The shape is (b, qS, kvN) or (qT, kvN).</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when oriMaskMode=0 and oriSparseIndicesOptional is not empty.</li>
                    <li>Passing is not supported in other scenarios.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layoutQ is TND, the shape is (qT, kvN).</li>
                    <li>When layoutQ is BSND, the shape is (b, qS, kvN).</li>
                </ul>
            </td>
            <td>                passing is not supported when oriMaskMode is not 0.
            </td>
        </tr>
        <tr>
            <td>cmpTopkLengthOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>shape is (b, qS, kvN) or (qT, kvN).</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Only validated when cmpKvOptional is passed.</li>
                    <li>Must be passed in when cmpMaskMode=0 and cmpSparseIndicesOptional is not empty.</li>
                    <li>passing is not supported in other scenarios.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layoutQ is TND, the shape is (qT, kvN).</li>
                    <li>When layoutQ is BSND, the shape is (b, qS, kvN).</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>topkValueMode</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>topK index value mode, default value is 1.</li>
                </ul>
            </td>
            <td>Optional attribute, default value is 1</td>
            <td>None</td>
            <td>None</td>
        </tr>
    </tbody>
</table>

#### Paged Attention Parameter Group

Paged Attention is enabled when layoutKv is PA_BBND.
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>oriBlockTableOptional</td>
            <td>                <ul>
                    <li>dtype supports only INT32.</li>
                    <li>The shape is (b, Ceil(oriKvSMax/oriKvBlockSize)).</li>
                    <li>The value must be a positive integer.</li>
                </ul>
            </td>
            <td>Optional</td>
            <td>None</td>
            <td>                <ul>
                    <li>When oriBlockTableOptional exists, sequsedOriKvOptional must be passed in.</li>
                    <li>When PagedAttention is enabled, blockTable must not be empty.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmpBlockTable</td>
            <td>                <ul>
                    <li>Only INT32 is supported for dtype.</li>
                    <li>The shape is (b, cmpKvSMax/cmpKvBlockSize).</li>
                    <li>Values must be positive integers.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Validated only when cmpKvOptional is passed.</li>
                    <li>Optional.</li>
                </ul>
            </td>
            <td>None</td>
            <td>                <ul>
                    <li>When cmpBlockTable exists, sequsedCmpKvOptional must be passed in.</li>
                    <li>When PagedAttention is enabled, blockTable must not be empty.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Sinks parameter group

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>sinksOptional</td>
            <td>                <ul>
                    <li>dtype supports FLOAT32.</li>
                    <li>shape is (qN, ).</li>
                </ul>
            </td>
            <td> Must be passed in the current version </td>
            <td> None </td>
            <td> None </td>
        </tr>
    </tbody>
</table>

#### SoftmaxLse Parameter Group

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Single-Parameter Validation</th>
            <th>Existence Validation</th>
            <th>Consistency Validation</th>
            <th>Feature Cross-Validation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>returnSoftmaxLse</td>
            <td>                <ul>
                    <li>dtype supports only BOOL.</li>
                    <li>true indicates softmaxLse is enabled, false indicates softmaxLse is disabled.</li>
                </ul>
            </td>
            <td>Optional, default value is false.</td>
            <td rowspan="2">                <ul>
                     <li>When returnSoftmaxLse is false, a tensor with shape [1] and value 0 is output.</li>
                    <li>When returnSoftmaxLse is true, the relationship between the shape of softmaxLse and layoutQ is as follows:<ul><li>When layoutQ is BSND, the shape of softmaxLse is (b, kvN, qS, qN/kvN).</li><li>When layoutQ is TND, the shape of softmaxLse is (kvN, qT, qN/kvN).</li></ul></li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>softmaxLseOutOptional</td>
            <td>                Only FLOAT32 is supported for dtype.
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
    </tbody>
</table>

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_sparse_flash_mla.h"
#include "aclnnop/aclnn_mixed_quant_sparse_flash_mla_metadata.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

namespace {

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

uint16_t FloatToBf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t roundingBias = 0x7fffu + lsb;
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void PrintOutResult(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, Bf16ToFloat(resultData[i]));
    }
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

std::vector<uint16_t> MakeBf16Data(int64_t size, float value)
{
    std::vector<uint16_t> data(static_cast<size_t>(size), FloatToBf16(value));
    return data;
}

std::vector<uint8_t> MakeHif8Data(int64_t size)
{
    std::vector<uint8_t> data(static_cast<size_t>(size), 0x08);
    return data;
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t B = 1;
    int64_t S1 = 1;
    int64_t S2 = 1024;
    int64_t N1 = 64;
    int64_t N2 = 1;
    int64_t D = 512;
    int64_t K = 512;
    int64_t oriBlockSize = 128;
    int64_t cmpBlockSize = 128;
    int64_t s2Act = 1024;
    int64_t cmpRatio = 4;
    int64_t oriWinLeft = 127;
    int64_t oriWinRight = 0;
    int64_t oriMaskMode = 4;
    int64_t cmpMaskMode = 3;
    int64_t quantMode = 1;
    int64_t ropeHeadDim = 64;
    double softmaxScale = 1.0 / sqrt(static_cast<double>(D));

    int64_t kvD = D;

    int64_t T1 = B * S1;
    int64_t cmpKvLen = s2Act / cmpRatio;
    int64_t oriBlockNum = ((s2Act + oriBlockSize - 1) / oriBlockSize) * B;
    int64_t cmpBlockNum = ((cmpKvLen + cmpBlockSize - 1) / cmpBlockSize) * B;

    std::vector<int64_t> qShape = {T1, N1, D};
    std::vector<int64_t> oriKvShape = {oriBlockNum, oriBlockSize, N2, kvD};
    std::vector<int64_t> cmpKvShape = {cmpBlockNum, cmpBlockSize, N2, kvD};
    std::vector<int64_t> cmpSparseIndicesShape = {T1, N2, K};
    std::vector<int64_t> oriBlockTableShape = {B, (s2Act + oriBlockSize - 1) / oriBlockSize};
    std::vector<int64_t> cmpBlockTableShape = {B, (cmpKvLen + cmpBlockSize - 1) / cmpBlockSize};
    std::vector<int64_t> cuSeqLensQShape = {B + 1};
    std::vector<int64_t> seqUsedOriKvShape = {B};
    std::vector<int64_t> seqUsedCmpKvShape = {B};
    std::vector<int64_t> cmpResidualKvShape = {B};
    std::vector<int64_t> sinksShape = {N1};
    std::vector<int64_t> metadataShape = {1024};
    std::vector<int64_t> descaleShape = {1};
    std::vector<int64_t> attnOutShape = {T1, N1, D};
    std::vector<int64_t> softmaxLseShape = {T1, N1, 1};
    std::vector<int64_t> emptyShape = {0};

    void *qDeviceAddr = nullptr;
    void *oriKvDeviceAddr = nullptr;
    void *cmpKvDeviceAddr = nullptr;
    void *qDescaleDeviceAddr = nullptr;
    void *oriKvDescaleDeviceAddr = nullptr;
    void *cmpKvDescaleDeviceAddr = nullptr;
    void *cmpSparseIndicesDeviceAddr = nullptr;
    void *oriBlockTableDeviceAddr = nullptr;
    void *cmpBlockTableDeviceAddr = nullptr;
    void *cuSeqLensQDeviceAddr = nullptr;
    void *cuSeqLensOriKvDeviceAddr = nullptr;
    void *cuSeqLensCmpKvDeviceAddr = nullptr;
    void *seqUsedQDeviceAddr = nullptr;
    void *seqUsedOriKvDeviceAddr = nullptr;
    void *seqUsedCmpKvDeviceAddr = nullptr;
    void *cmpResidualKvDeviceAddr = nullptr;
    void *sinksDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;
    void *softmaxLseDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *oriKv = nullptr;
    aclTensor *cmpKv = nullptr;
    aclTensor *qDescale = nullptr;
    aclTensor *oriKvDescale = nullptr;
    aclTensor *cmpKvDescale = nullptr;
    aclTensor *cmpSparseIndices = nullptr;
    aclTensor *oriBlockTable = nullptr;
    aclTensor *cmpBlockTable = nullptr;
    aclTensor *cuSeqLensQ = nullptr;
    aclTensor *cuSeqLensOriKv = nullptr;
    aclTensor *cuSeqLensCmpKv = nullptr;
    aclTensor *seqUsedQ = nullptr;
    aclTensor *seqUsedOriKv = nullptr;
    aclTensor *seqUsedCmpKv = nullptr;
    aclTensor *cmpResidualKv = nullptr;
    aclTensor *sinks = nullptr;
    aclTensor *metadata = nullptr;
    aclTensor *attnOut = nullptr;
    aclTensor *softmaxLse = nullptr;

    int64_t qSize = GetShapeSize(qShape);
    int64_t oriKvSize = GetShapeSize(oriKvShape);
    int64_t cmpKvSize = GetShapeSize(cmpKvShape);
    int64_t cmpSparseIndicesSize = GetShapeSize(cmpSparseIndicesShape);
    int64_t oriBlockTableSize = GetShapeSize(oriBlockTableShape);
    int64_t cmpBlockTableSize = GetShapeSize(cmpBlockTableShape);
    int64_t attnOutSize = GetShapeSize(attnOutShape);
    int64_t softmaxLseSize = GetShapeSize(softmaxLseShape);

    std::vector<uint8_t> qHostData = MakeHif8Data(qSize);
    std::vector<uint8_t> oriKvHostData = MakeHif8Data(oriKvSize);
    std::vector<uint8_t> cmpKvHostData = MakeHif8Data(cmpKvSize);
    std::vector<float> qDescaleHostData = {1.0f};
    std::vector<float> oriKvDescaleHostData = {1.0f};
    std::vector<float> cmpKvDescaleHostData = {1.0f};
    std::vector<int32_t> cmpSparseIndicesHostData(cmpSparseIndicesSize);
    std::vector<int32_t> oriBlockTableHostData(oriBlockTableSize);
    std::iota(oriBlockTableHostData.begin(), oriBlockTableHostData.end(), 0);
    std::vector<int32_t> cmpBlockTableHostData(cmpBlockTableSize);
    std::iota(cmpBlockTableHostData.begin(), cmpBlockTableHostData.end(), 0);
    std::vector<int32_t> cuSeqLensQHostData(B + 1);
    for (int64_t i = 0; i <= B; i++) {
        cuSeqLensQHostData[i] = static_cast<int32_t>(i * S1);
    }
    std::vector<int32_t> emptyHostData;
    std::vector<int32_t> seqUsedOriKvHostData(B, static_cast<int32_t>(s2Act));
    std::vector<int32_t> seqUsedCmpKvHostData(B, static_cast<int32_t>(cmpKvLen));
    std::vector<int32_t> cmpResidualKvHostData(B);
    for (int64_t i = 0; i < B; i++) {
        cmpResidualKvHostData[i] = seqUsedOriKvHostData[i] % static_cast<int32_t>(cmpRatio);
    }
    std::vector<float> sinksHostData(N1, 1.0f);
    std::vector<int32_t> metadataHostData(1024, 0);
    std::vector<uint16_t> attnOutHostData = MakeBf16Data(attnOutSize, 0.0f);
    std::vector<float> softmaxLseHostData(softmaxLseSize, 0.0f);

    std::mt19937 gen(42);
    for (int64_t t = 0; t < T1; t++) {
        for (int64_t n = 0; n < N2; n++) {
            for (int64_t k = 0; k < K; k++) {
                cmpSparseIndicesHostData[t * N2 * K + n * K + k] = static_cast<int32_t>(gen() % cmpKvLen);
            }
        }
    }

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_HIFLOAT8, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_HIFLOAT8, &oriKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_HIFLOAT8, &cmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qDescaleHostData, descaleShape, &qDescaleDeviceAddr, aclDataType::ACL_FLOAT, &qDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(oriKvDescaleHostData, descaleShape, &oriKvDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &oriKvDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvDescaleHostData, descaleShape, &cmpKvDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &cmpKvDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpSparseIndicesHostData, cmpSparseIndicesShape, &cmpSparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &cmpSparseIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(oriBlockTableHostData, oriBlockTableShape, &oriBlockTableDeviceAddr, aclDataType::ACL_INT32,
                          &oriBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpBlockTableHostData, cmpBlockTableShape, &cmpBlockTableDeviceAddr, aclDataType::ACL_INT32,
                          &cmpBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensQShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32,
                          &cuSeqLensQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensOriKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensOriKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensCmpKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensCmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(emptyHostData, emptyShape, &seqUsedQDeviceAddr, aclDataType::ACL_INT32, &seqUsedQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqUsedOriKvHostData, seqUsedOriKvShape, &seqUsedOriKvDeviceAddr, aclDataType::ACL_INT32,
                          &seqUsedOriKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqUsedCmpKvHostData, seqUsedCmpKvShape, &seqUsedCmpKvDeviceAddr, aclDataType::ACL_INT32,
                          &seqUsedCmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpResidualKvHostData, cmpResidualKvShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32,
                          &cmpResidualKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxLse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BBND";

    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;

    ret = aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize(
        cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv, seqUsedQ, seqUsedOriKv, seqUsedCmpKv, cmpResidualKv, nullptr,
        nullptr, N1, N2, D, quantMode, B, S1, S2, cmpKvLen, 0, K, ropeHeadDim, cmpRatio, oriMaskMode, cmpMaskMode,
        oriWinLeft, oriWinRight, layoutQ, layoutKv, true, true, metadata, &metadataWorkspaceSize, &metadataExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *metadataWorkspaceAddr = nullptr;
    if (metadataWorkspaceSize > 0) {
        ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnMixedQuantSparseFlashMlaMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadata failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret);
              return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnQuantSparseFlashMlaGetWorkspaceSize(
        q, oriKv, cmpKv, qDescale, oriKvDescale, cmpKvDescale, nullptr, cmpSparseIndices, oriBlockTable, cmpBlockTable,
        cuSeqLensQ, nullptr, nullptr, nullptr, seqUsedOriKv, seqUsedCmpKv, cmpResidualKv, nullptr, nullptr, sinks,
        metadata, quantMode, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQ,
        layoutKv, 1, false, attnOut, softmaxLse, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantSparseFlashMlaGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnQuantSparseFlashMla(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantSparseFlashMla failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResult(attnOutShape, &attnOutDeviceAddr);

    aclDestroyTensor(q);
    aclDestroyTensor(oriKv);
    aclDestroyTensor(cmpKv);
    aclDestroyTensor(qDescale);
    aclDestroyTensor(oriKvDescale);
    aclDestroyTensor(cmpKvDescale);
    aclDestroyTensor(cmpSparseIndices);
    aclDestroyTensor(oriBlockTable);
    aclDestroyTensor(cmpBlockTable);
    aclDestroyTensor(cuSeqLensQ);
    aclDestroyTensor(cuSeqLensOriKv);
    aclDestroyTensor(cuSeqLensCmpKv);
    aclDestroyTensor(seqUsedQ);
    aclDestroyTensor(seqUsedOriKv);
    aclDestroyTensor(seqUsedCmpKv);
    aclDestroyTensor(cmpResidualKv);
    aclDestroyTensor(sinks);
    aclDestroyTensor(metadata);
    aclDestroyTensor(attnOut);
    aclDestroyTensor(softmaxLse);

    aclrtFree(qDeviceAddr);
    aclrtFree(oriKvDeviceAddr);
    aclrtFree(cmpKvDeviceAddr);
    aclrtFree(qDescaleDeviceAddr);
    aclrtFree(oriKvDescaleDeviceAddr);
    aclrtFree(cmpKvDescaleDeviceAddr);
    aclrtFree(cmpSparseIndicesDeviceAddr);
    aclrtFree(oriBlockTableDeviceAddr);
    aclrtFree(cmpBlockTableDeviceAddr);
    if (cuSeqLensQDeviceAddr != nullptr) {
        aclrtFree(cuSeqLensQDeviceAddr);
    }
    if (seqUsedOriKvDeviceAddr != nullptr) {
        aclrtFree(seqUsedOriKvDeviceAddr);
    }
    if (seqUsedCmpKvDeviceAddr != nullptr) {
        aclrtFree(seqUsedCmpKvDeviceAddr);
    }
    if (cmpResidualKvDeviceAddr != nullptr) {
        aclrtFree(cmpResidualKvDeviceAddr);
    }
    aclrtFree(sinksDeviceAddr);
    aclrtFree(metadataDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
    aclrtFree(softmaxLseDeviceAddr);
    if (metadataWorkspaceSize > 0) {
        aclrtFree(metadataWorkspaceAddr);
    }
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
