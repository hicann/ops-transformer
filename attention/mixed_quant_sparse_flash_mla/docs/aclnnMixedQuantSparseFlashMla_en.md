# aclnnMixedQuantSparseFlashMla

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:25:31.569Z -->

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

  The `aclnnMixedQuantSparseFlashMla` operator performs MLA (Multi-head Latent Attention) attention computation in quantization and sparsity use cases. This API supports the following three computation modes:
  - **SWA (Sliding Window Attention)**: Pass in only `oriKvOptional` to perform sliding window attention on the original KV.
  - **CSA (Compressed Sparse Attention)**: Pass in `oriKvOptional`, `cmpKvOptional`, and `cmpSparseIndicesOptional` simultaneously to perform attention on the original KV window and the compressed KV selected by topK.
  - **HCA (Heavily Compressed Attention)**: Pass in `oriKvOptional` and `cmpKvOptional` to perform attention on both the original KV window and the contiguous compressed KV segments.

  `aclnnMixedQuantSparseFlashMlaMetadata` is the core split information of `aclnnMixedQuantSparseFlashMla` and is generated before the main operator is executed. In the current version, this metadata must be passed in for the main operator. The typical call procedure is as follows:

  **This API is not recommended for standalone use. Use it together with the aclnnMixedQuantSparseFlashMlaMetadata API to form a complete workflow.**

  1. Prepare the corresponding input based on the use case.
  2. Call `aclnnMixedQuantSparseFlashMlaMetadata` to generate `metadata` as an input parameter of `aclnnMixedQuantSparseFlashMla`.
  3. Call `aclnnMixedQuantSparseFlashMla` to generate the computation result.

- Formula:

  $$
  O = \text{softmax}(Q \cdot \tilde{K}^T \cdot \text{softmax\_scale}) \cdot \tilde{V}
  $$

  where $\tilde{K} = \tilde{V}$ (shared KV), $\tilde{K}$ is concatenated from the original KV within the sliding window and the compressed KV within the causal boundary, and the specific range of KV participating in the calculation is determined by the template mode and mask parameter:

  - Sliding window part (oriKv): For the $i_{S1}$-th Query token, its causal diagonal position is $\text{oriThreshold} = S2_{act} - S1_{act} + i_{S1} + 1$, and the window range is $[\max(\text{oriThreshold} - \text{oriWinLeft} - 1, 0), \text{oriThreshold} + \text{oriWinRight})$.

  - Compressed KV part (cmpKv): The causal boundary threshold is $\text{cmpThreshold} = \lfloor \frac{\text{oriThreshold}}{\text{cmpRatio}} \rfloor$. In HCA use cases, the contiguous compressed KV within $[0, \text{cmpThreshold})$ is used; in CSA use cases, the compressed KV is collected on demand using TopK indices, and only the blocks where $\text{beginIdx} < \text{cmpThreshold}$ are retained.

  The attention computation uses Online Softmax (Flash Attention V2). The S2 dimension is processed in a loop with a tile size of 512, and sinks is used as the initial maximum value of the softmax for each row:

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
  | $Q$                 | Query input, with shape [G, D] (single row).              |
  | $K_{tile}^{t}$      | KV data of the t-th S2 tile, where K=V (shared KV).       |
  | $S_t$               | Scaled QK attention score of the t-th tile.               |
  | $P_t$               | Softmax probability of the t-th tile.                     |
  | $O_t$               | Accumulated output after the t-th tile.                   |
  | $softmaxScale$      | Scaling factor, usually the reciprocal of the square root of each attention head dimension. |
  | $B$                 | Batch size.                                               |
  | $S1$/$S1_{act}$     | Query sequence length/actual valid length.                |
  | $S2$/$S2_{act}$     | Original KV sequence length/actual valid length.          |
  | $N1$                | Number of Query heads.                                    |
  | $N2$                | Number of KV heads.                                       |
  | $G$                 | GQA grouping ratio, $G=N1/N2$.                            |
  | $D$                 | Dimension of each attention head.                         |
  | $sinks$             | Attention sink, with shape [N1].                          |
  | $cmpRatio$          | Compression ratio of cmpKv, used to calculate the pre-compression KV length of the cmp-side mask. |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call `aclnnMixedQuantSparseFlashMlaGetWorkspaceSize` to obtain the workspace size required for computation and the executor that includes the operator computation process. Then, call `aclnnMixedQuantSparseFlashMla` to perform the actual computation.

```Cpp
aclnnStatus aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
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
    int64_t          ropeHeadDim,
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
aclnnStatus aclnnMixedQuantSparseFlashMla(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMixedQuantSparseFlashMlaGetWorkspaceSize

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
      <td>Empty tensors are not supported. qN supports 1-128; D only supports 512.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutQ is BSND: (b, qS, qN, qD)</li>
          <li>When layoutQ is TND: (qT, qN, qD)</li>
        </ul>
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Original KV input tensor. Key and Value share the same data.</td>
      <td>Must be passed in for SWA/CSA/HCA scenarios. The quantized KV layout is determined by quantMode: when quant_mode is 1, the quantized KV is concatenated in the order of rope (64, bfloat16), nope (448, FLOAT8_E4M3FN), scale (7, bfloat16), and pad (18B); when quant_mode is 2, the quantized KV is concatenated in the order of nope (448, FLOAT8_E4M3FN), rope (64, bfloat16), scale (7, FLOAT8_E8M0), and pad (1B). Currently, only 1 and 2 are supported, and quantization mode 2 supports only layout_kv set to PA_BBND. Each quantization mode supports UINT8 and FLOAT8_E4M3FN as single-byte storage views, with the underlying byte content unchanged.</td>
      <td>See quantMode for details.</td>
      <td>ND</td>
      <td>        <ul>
          <li>If layoutKv is PA_BBND: (oriKvBlockNums, oriKvBlockSize, kvN, kvD), where oriKvBlockSize supports 1 to 1024</li>
          <li>If layoutKv is BSND: (b, oriKvS, kvN, kvD)</li>
          <li>If layoutKv is TND: (oriKvT, kvN, kvD)</li>
        </ul>
        kvN supports only 1, and kvD is determined by quantMode.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Compressed KV input tensor. Key and Value share the same data.</td>
      <td>Must be passed in for CSA/HCA scenarios and not passed in for SWA scenarios. The quantized KV layout is determined by quantMode, same as oriKvOptional.</td>
      <td>See quantMode for details.</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutKv is PA_BBND: (cmpKvBlockNums, cmpKvBlockSize, kvN, kvD), where cmpKvBlockSize supports values from 1 to 1024</li>
          <li>When layoutKv is BSND: (b, cmpKvS, kvN, kvD)</li>
          <li>When layoutKv is TND: (cmpKvT, kvN, kvD)</li>
        </ul>
        kvN supports only 1, and kvD is determined by quantMode.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriSparseIndicesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the index for discretely selecting from oriKvCache.</td>
      <td>Optional input. Invalid positions are filled with -1, and the rest are non-negative integers.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutQ is BSND: (b, qS, kvN, oriKvK)</li>
          <li>When layoutQ is TND: (qT, kvN, oriKvK)</li>
        </ul>
        oriKvK is the number of TopK sparse selections from oriKvOptional, and values greater than 0 are supported.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpSparseIndicesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the TopK indices for discretely selecting cmpKvCache.</td>
      <td>Must be passed in for the cmpKv sparse scenario and not passed in for other scenarios. Fill invalid positions with -1, and the rest are non-negative integers.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>        <ul>
          <li>When layoutQ is BSND: (b, qS, kvN, cmpKvK)</li>
          <li>When layoutQ is TND: (qT, kvN, cmpKvK)</li>
        </ul>
        cmpKvK indicates the TopK sparse selection count of cmpKvOptional, and the supported range is greater than 0.
      </td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriBlockTableOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Block mapping table used for oriKvCache storage in PagedAttention.</td>
      <td>Must be passed in when layoutKv is PA_BBND. The length of the second dimension must be greater than or equal to the number of blocks corresponding to the maximum S2 among all batches.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b, Ceil(oriKvSMax/oriKvBlockSize))</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpBlockTableOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Block mapping table used for cmpKvCache storage in PagedAttention.</td>
      <td>Must be passed in when cmpKv is passed and layoutKv is PA_BBND.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b, Ceil(cmpKvSMax/cmpKvBlockSize))</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensQOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of valid q tokens in different batches (prefix sum form).</td>
      <td>Must be passed in when layoutQOptional is TND. Each element indicates the total number of tokens in the current batch and all previous batches.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of valid tokens of oriKv in different batches (prefix sum form).</td>
      <td>Must be passed in when layoutKvOptional is TND.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of valid tokens of cmpKv in different batches (prefix sum form).</td>
      <td>Must be passed when layoutKvOptional is TND and cmpKvOptional exists.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b+1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of tokens in q actually participating in the computation in different batches.</td>
      <td>Currently, specifying this parameter is not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of tokens in oriKv actually participating in the computation in different batches.</td>
      <td>Must be passed in when layoutKvOptional is PA_BBND; optional to pass in when layoutKvOptional is BSND, to specify the valid length of oriKv for each batch; when layoutKvOptional is TND, cuSeqlensOriKvOptional is used to express the sequence boundary.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of tokens of cmpKv actually participating in the computation in different batches.</td>
      <td>Optional input. When passed in, the shape must be (B,). It serves as the logical valid length of cmp for each batch and takes precedence over the derivation from cmpKvOptional shape, cuSeqlensCmpKvOptional, or the PA block table.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Compressed KV remainder, used to restore the pre-compression KV length used by the cmp-side mask.</td>
      <td>Optional input. When passed in, the shape must be (B,). The pre-compression KV length of the b-th batch is restored as cmp_len * cmpRatio + cmpResidualKvOptional[b]. This input must be passed in when cmpMaskMode is 3 and cmpRatio is not 1, and must not be passed in when cmpMaskMode is 0 or cmpRatio is 1.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(b,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Used to identify the length of oriSparseIndicesOptional that actually participates in the calculation.</td>
      <td>Must be passed in when oriMaskMode=0 and oriSparseIndicesOptional is passed in; not supported to pass in other scenarios.</td>
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
      <td>Used to identify the length of cmpSparseIndicesOptional actually participating in the calculation.</td>
      <td>Must be passed in when cmpMaskMode=0 and cmpSparseIndicesOptional is passed in; not supported to pass in other scenarios.</td>
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
      <td>Indicates an independent learnable virtual offset item set for each attention head, used to maintain stability during long-text inference.</td>
      <td>Must be passed in.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(qN,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>metadataOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Core-splitting information of the AICPU operator aclnnMixedQuantSparseFlashMlaMetadata.</td>
      <td>Must be passed in. Generated by the aclnnMixedQuantSparseFlashMlaMetadata operator.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the quantization mode.</td>
      <td>Indicates the quantization mode. Quantization mode 1 indicates that K and V nope use per-token-group quantization, and the scale type is bfloat16. Quantization mode 2 indicates that K and V nope use per-token-group quantization, and the scale type is FLOAT8_E8M0. Currently, only 1 and 2 are supported, and for quantization mode 2, only PA_BBND is supported for layout_kv.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ropeHeadDim (int64_t)</td>
      <td>Input</td>
      <td>Indicates the rope head dimension.</td>
      <td>Only 64 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxScale (double)</td>
      <td>Input</td>
      <td>Scaling factor, corresponding to softmaxScale in the formula.</td>
      <td>The recommended value is 1/√D, where D is the head dimension of each attention head.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Compression ratio of cmpKv relative to the KV length before compression, used to restore the pre-compression KV length used by the cmp-side mask.</td>
      <td>cmpRatio supports 1 to 128. When cmpKv is not passed, only the default value 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Mask mode for computing q and oriKv.</td>
      <td>Supported:<br>0: No mask.<br>3: rightDownCausal mode.<br>4: sliding window mode.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Mask mode for the calculation of q and cmpKv.</td>
      <td>Supported values:<br>0: No mask.<br>3: rightDownCausal mode. Only the default value 0 is supported when cmpKv is not passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of historical tokens that q attends to in the calculation with oriKvOptional.</td>
      <td>Supports -1 or a non-negative value. -1 indicates infinity, that is, all tokens participate in the computation.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of future tokens that q computes in the calculation of q and oriKvOptional.</td>
      <td>Supports -1 or non-negative numbers. -1 indicates infinity, that is, all tokens participate in the computation.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional (char*)</td>
      <td>Input</td>
      <td>Specifies the data layout format of input q.</td>
      <td>Supports "BSND" and "TND".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional (char*)</td>
      <td>Input</td>
      <td>Identifies the data layout format of inputs oriKvOptional and cmpKvOptional.</td>
      <td>Supports "PA_BBND", "BSND", and "TND".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkValueMode (int64_t)</td>
      <td>Input</td>
      <td>Mode of obtaining topk index values.</td>
      <td>Currently, only 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxLse (bool)</td>
      <td>Input</td>
      <td>Whether to return softmaxLse.</td>
      <td>true or false</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOutOut (aclTensor*)</td>
      <td>Output</td>
      <td>Attention computation output.</td>
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
          <li>When layoutQ is TND: (kvN, qT, qN/kvN)</li>
          <li>When returnSoftmaxLse is false: placeholder tensor</li>
        </ul>
      </td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be side-applied on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the calculation process of the operator.</td>
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
|      qD      |          Attention head dimension of input q.         |
|      kvD      |          Attention head dimension of input oriKvOptional/cmpKvOptional.         |
|     qT     |          Cumulative sum of sequence lengths of all batches of input q.          |
|     oriKvT    |          Cumulative sum of sequence lengths of all batches of input oriKvOptional.          |
|     cmpKvT    |          Cumulative sum of sequence lengths of all batches of input cmpKvOptional.          |
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
      <th>Return Value</th>
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
      <td>The data type, data format, or attribute value of the input variable is out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnMixedQuantSparseFlashMla

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
      <td>Device-side applied workspace memory address.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Device-side applied workspace size, obtained by the first-phase API aclnnMixedQuantSparseFlashMlaGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which includes the operator calculation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation

  - aclnnMixedQuantSparseFlashMla defaults to a deterministic implementation. The same input yields the same result across multiple calls.

- Usage constraints

  - Inputs such as cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriBlockTableOptional, and cmpBlockTableOptional are tensors. Since the operator cannot obtain the actual tensor values during the Tiling phase, the tiling side does not validate the values, and correctness must be ensured by users. If invalid values are passed in for the preceding parameters, undefined behavior is triggered (such as accuracy issues or program crashes caused by illegal memory access).
  - The input parameters of aclnnMixedQuantSparseFlashMlaMetadata and aclnnMixedQuantSparseFlashMla must be consistent when called. Since the operator is called in two phases through two APIs, the operator cannot perform this check itself, and correctness must be ensured by users. If the parameters passed to the APIs are inconsistent, undefined behavior occurs (such as accuracy issues or program crashes caused by illegal memory access).
  - oriTopkLengthOptional and cmpTopkLengthOptional indicate the length of ori/cmp sparseIndices actually participating in computation. The values cannot be greater than the size of the last dimension of sparseIndicesOptional. When sequsedQOptional is passed in, the values of the valid part corresponding to topkLength must be greater than or equal to 0.
  - When oriMaskMode/cmpMaskMode is 0, oriKvK/cmpKvK must be greater than or equal to the maximum value of oriTopkLengthOptional/cmpTopkLengthOptional.
  - cmpResidualKvOptional is used with cmpRatio to restore the KV length before compression. The value of each batch must be less than cmpRatio. This parameter can be passed in only when cmpMaskMode is 3 and cmpRatio is not 1; it cannot be passed in when cmpMaskMode is 0 or cmpRatio is 1.
  - attnOutOut: tensor type, the output in the formula. The supported data type is BFLOAT16. The supported data format is ND. Constraint: The shape of this output parameter must be the same as that of input parameter q, and the dtype must be the same as q.
  - When returnSoftmaxLse=False, a tensor with shape [1] and value 0 is returned; when returnSoftmaxLse=True, a FLOAT32 log-sum-exp result is returned.
  - cuSeqlensQOptional, cuSeqlensOriKvOptional, and cuSeqlensCmpKvOptional must satisfy that the first element is 0 and the sequence is non-decreasing overall, that is, each element is not less than its previous value.
  - When layoutKv is PA_BBND, oriKvOptional and cmpKvOptional support non-contiguous storage along axis 0.
  - For dimensions denoted by the same symbol in the shapes of the parameters, the actual values of the corresponding axes must be consistent.

### Feature Parameter Group

|      Feature Parameter Group      |     Parameter Field Name     |
| :-------------------: | :-------------------: |
|      Common Parameter Group      | q, quantMode, oriKvOptional, cmpKvOptional, metadataOptional, ropeHeadDim, softmaxScale, layoutQ, layoutKv, attnOutOut |
|      Mask Parameter Group      | oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight |
|   SeqLens Parameter Group   | cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional |
|   Sparse Compression Parameter Group    | cmpRatio, cmpResidualKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional, oriTopkLengthOptional, cmpTopkLengthOptional, topkValueMode |
| Paged Attention Parameter Group | oriBlockTableOptional, cmpBlockTableOptional |
|   Sinks Parameter Group   | sinksOptional |
|   SoftmaxLse Parameter Group   | returnSoftmaxLse, softmaxLseOutOptional |

### Computation Mode Description

|    Name    |    Parameters Required in Typical Use Cases    |    Parameters Required in Fully Sparse Use Cases    |
| :---------: | :--------------------------------: | :---------------------------: |
|      SWA      | oriKvOptional | oriKvOptional, oriSparseIndicesOptional |
|      HCA      | oriKvOptional, cmpKvOptional|-|
|      CSA      | oriKvOptional, cmpKvOptional, cmpSparseIndicesOptional| oriKvOptional, oriSparseIndicesOptional, cmpKvOptional, cmpSparseIndicesOptional|

### Parameter Group Constraints

#### Common Parameter Group

- Handling of empty input parameters:
    - An empty tensor means a required input, an input that must be passed in under a specific use case, or an output has a shape size of 0, that is, any axis is 0.
    - All use cases that trigger empty tensors are intercepted with errors.

- Verification of q, oriKvOptional, cmpKvOptional, and attnOutOut

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
    <th>Single-Parameter Check</th>
    <th>Existence Check</th>
    <th>Consistency Check</th>
    <th>Feature Cross-Check</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>            <ul>
                <li>dtype supports BFLOAT16.</li>
                <li>When layoutQ is BSND, the shape of q is (b, qS, qN, qD).</li>
                <li>When layoutQ is TND, the shape of q is (qT, qN, qD).</li>
            </ul>
        </td>
        <td>            Must be passed in
        </td>
        <td rowspan="4">            <ul>
                <li>The dtype and shape of q and attnOutOut must be the same.</li>
                <li>If cmpKvOptional is passed in, the dtype of oriKvOptional must be the same as that of cmpKvOptional.</li>
                <li>If layoutKv is not PA_BBND, layoutQ and layoutKv must be the same.</li>
                <li>If layoutKv is PA_BBND, layoutQ can be BSND or TND.</li>
            </ul>
        </td>
        <td rowspan="4">            Axis check:
            <ul>
                <li>b > 0</li>
                <li>qS > 0</li>
                <li>0 < qN <= 128</li>
                <li>qD = 512</li>
                <li>qT > 0</li>
                <li>oriKvS > 0</li>
                <li>cmpKvS > 0</li>
                <li>kvN = 1</li>
                <li>kvD=608 when quantMode=1, kvD=584 when quantMode=2</li>
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
                <li>dtype supports FLOAT8_E4M3FN.</li>
                <li>When layoutKv is BSND, the shape of oriKvOptional is (b, oriKvS, kvN, kvD).</li>
                <li>When layoutKv is TND, the shape of oriKvOptional is (oriKvT, kvN, kvD).</li>
                <li>When layoutKv is PA_BBND, the shape of oriKvOptional is (oriKvBlockNums, oriKvBlockSize, kvN, kvD).</li>
            </ul>
        </td>
        <td>            Must be passed in the current version
        </td>
    </tr>
    <tr>
        <td>attnOutOut</td>
        <td>            <ul>
                <li>dtype supports BFLOAT16.</li>
                <li>When layoutQ is BSND, the shape of attnOutOut is (b, qS, qN, qD).</li>
                <li>When layoutQ is TND, the shape of attnOutOut is (qT, qN, qD).</li>
            </ul>
        </td>
        <td>            Must be passed in
        </td>
    </tr>
    <tr>
        <td>cmpKvOptional</td>
        <td>            <ul>
                <li>dtype supports FLOAT8_E4M3FN</li>
                <li>When layoutKv is BSND, the shape of cmpKvOptional is (b, cmpKvS, kvN, kvD)</li>
                <li>When layoutKv is TND, the shape of cmpKvOptional is (cmpKvT, kvN, kvD)</li>
                <li>When layoutKv is PA_BBND, the shape of cmpKvOptional is (cmpKvBlockNums, cmpKvBlockSize, kvN, kvD)</li>
            </ul>
        </td>
        <td>            optional input
        </td>
    </tr>
</tbody>
</table>
layout matching relationship table:
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

metadataOptional Checks
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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>metadataOptional</td>
            <td>                <ul>
                    <li>dtype only supports INT32.</li>
                    <li>shape is dynamically calculated by aclnnMixedQuantSparseFlashMlaMetadata.</li>
                </ul>
            </td>
            <td>Required in the current version.</td>
            <td>None</td>
            <td>When passed in, must be consistent with the result generated by aclnnMixedQuantSparseFlashMlaMetadata.</td>
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
            <th>Single-parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>oriMaskMode</td>
            <td>                <ul>
                    <li>The dtype supports INT32.</li>
                    <li>The input range only supports 0, 3, and 4. The default value is 0.</li>
                </ul>
            </td>
            <td>                Optional. If not passed, the default value is 0.
            </td>
            <td>                <ul>
                    <li>None</li>
                </ul>
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
                     <li>The supported input range is only 0 and 3; the default value is 0.</li>
                 </ul>
             </td>
             <td>                 Optional. If not passed, the default value is 0.
             </td>
             <td>                 <ul>
                     <li>None</li>
                 </ul>
             </td>
             <td>                    <li>When cmpKvOptional is not passed, cmpMaskMode must be 0 </li>
            </td>
        </tr>
        <tr>
            <td>oriWinLeft/oriWinRight</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>The supported input range is only -1 or values greater than or equal to 0.</li>
                </ul>
            </td>
            <td>                Optional. If not passed, the default value is -1.
            </td>
            <td>                <ul>
                    None
                </ul>
            </td>
            <td>                    <li>oriWinLeft/oriWinRight can be >=0 only when oriMaskMode is 4.</li>
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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>sequsedQOptional</td>
            <td>                <ul>
                    <li>dtype: INT32.</li>
                    <li>shape: (b,).</li>
                    <li>value only supports non-negative integers.</li>
                    <li>value in sequsedQOptional must be less than or equal to qS.</li>
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
                    <li>The value in sequsedOriKvOptional must be less than or equal to kvS.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional when layoutKv is BSND.</li>
                    <li>Must be passed in when layoutKv is PA_BBND.</li>
                    <li>Can be omitted when oriTopkLengthOptional is passed.</li>
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
                    <li>The value in sequsedCmpKvOptional must be less than or equal to kvS.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional when layoutKv is BSND.</li>
                    <li>Must be passed in when layoutKv is PA_BBND and cmpKvOptional is passed.</li>
                    <li>Can be omitted when cmpTopkLengthOptional is passed.</li>
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
                    <li>Its value should be non-decreasing (greater than or equal to the previous value), with the first element being 0 and the last element equal to qT.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when layoutQ is TND.</li>
                    <li>Not supported to pass when layoutQ is not TND.</li>
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
                    <li>Its value should be non-decreasing (greater than or equal to the previous value), with the first element being 0 and the last element equal to oriKvT.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when layoutKv is TND.</li>
                    <li>Not supported to pass when layoutKv is not TND.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
        <tr>
            <td>cuSeqlensCmpKvOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>shape is (b+1,).</li>
                    <li>The value only supports non-negative integers.</li>
                    <li>Its value should be non-decreasing (greater than or equal to the previous value). The first element is 0, and the last element equals cmpKvT.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when layoutKv is TND.</li>
                    <li>Not supported to pass in when layoutKv is not TND.</li>
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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>cmpRatio</td>
            <td>                <ul>
                    <li>data_type supports INT32.</li>
                    <li>Indicates the compression ratio of cmpKvOptional relative to the KV length before compression. The value must be greater than 0.</li>
                    <li>The default value is 1.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional. The default value is 1.</li>
                </ul>
            </td>
            <td>None</td>
            <td>                <ul>
                    <li>When cmpKv is not passed, only the default value 1 is supported.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmpResidualKvOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32</li>
                    <li>shape is (b,)</li>
                    <li>The value only supports non-negative integers</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Checked only when cmpKvOptional is passed in</li>
                    <li>Optional</li>
                    <li>Must be passed in when cmpMaskMode=3 and cmpRatio!=1</li>
                </ul>
            </td>
            <td>                <ul>None</ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>oriSparseIndicesOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32</li>
                    <li>shape is (qT, kvN, oriKvK) or (b, qS, kvN, oriKvK)</li>
                    <li>Invalid positions are filled with -1, and the rest are non-negative integers</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Optional</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>When layoutQ is TND, the shape is (qT, kvN, oriKvK)</li>
                    <li>When layoutQ is BSND, the shape is (b, qS, kvN, oriKvK)</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>cmpSparseIndicesOptional</td>
            <td>                <ul>
                    <li>dtype: INT32.</li>
                    <li>shape: (qT, kvN, cmpKvK) or (b, qS, kvN, cmpKvK).</li>
                    <li>Invalid positions are filled with -1, and the rest are non-negative integers.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Checked only when cmpKvOptional is passed in.</li>
                    <li>Optional.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>If layoutQ is TND, the shape is (qT, kvN, cmpKvK).</li>
                    <li>If layoutQ is BSND, the shape is (b, qS, kvN, cmpKvK).</li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>oriTopkLengthOptional</td>
            <td>                <ul>
                    <li>dtype: INT32.</li>
                    <li>shape: (b, qS, kvN) or (qT, kvN).</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Must be passed in when oriMaskMode=0 and oriSparseIndicesOptional is not empty.</li>
                    <li>Not supported to pass in other scenarios.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>If layoutQ is TND, the shape is (qT, kvN).</li>
                    <li>If layoutQ is BSND, the shape is (b, qS, kvN).</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Not supported to pass when oriMaskMode is not 0.</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmpTopkLengthOptional</td>
            <td>                <ul>
                    <li>dtype supports INT32.</li>
                    <li>The shape is (b, qS, kvN) or (qT, kvN).</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Validated only when cmpKvOptional is passed in.</li>
                    <li>Must be passed in when cmpMaskMode=0 and cmpSparseIndicesOptional is not empty.</li>
                    <li>Not supported to pass in other scenarios.</li>
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
                    <li>data_type supports INT32.</li>
                    <li>topK index value mode. The default value is 1.</li>
                </ul>
            </td>
            <td>Optional attribute. The default value is 1.</td>
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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>oriBlockTableOptional</td>
            <td>                <ul>
                    <li>dtype only supports INT32.</li>
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
            <td>cmpBlockTableOptional</td>
            <td>                <ul>
                    <li>dtype only supports INT32.</li>
                    <li>shape is (b, Ceil(cmpKvSMax/cmpKvBlockSize)).</li>
                    <li>The value can only be a positive integer.</li>
                </ul>
            </td>
            <td>                <ul>
                    <li>Checked only when cmpKvOptional is passed in.</li>
                    <li>Optional.</li>
                </ul>
            </td>
            <td>None</td>
            <td>                <ul>
                    <li>When cmpBlockTableOptional exists, sequsedCmpKvOptional must be passed in.</li>
                    <li>When PagedAttention is enabled, blockTable must not be empty.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Sinks Parameter Group

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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
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
            <th>Single-Parameter Check</th>
            <th>Existence Check</th>
            <th>Consistency Check</th>
            <th>Feature Cross-Check</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>returnSoftmaxLse</td>
            <td>                <ul>
                    <li>data_type only supports bool.</li>
                    <li>true indicates that softmaxLse is enabled, and false indicates that softmaxLse is disabled.</li>
                </ul>
            </td>
            <td>Optional. The default value is false.</td>
            <td rowspan="2">                <ul>
                     <li>When returnSoftmaxLse is false, the output is a tensor with shape [1] and value 0.</li>
                    <li>When returnSoftmaxLse is true, the relationship between the shape of softmaxLseOutOptional and layoutQ is as follows:<ul><li>When layoutQ is BSND, the shape of softmaxLseOutOptional is (b, kvN, qS, qN/kvN).</li><li>When layoutQ is TND, the shape of softmaxLseOutOptional is (kvN, qT, qN/kvN).</li></ul></li>
                </ul>
            </td>
            <td>None</td>
        </tr>
        <tr>
            <td>softmaxLseOutOptional</td>
            <td>                <ul>
                    <li>data_type only supports FLOAT32.</li>
                </ul>
            </td>
            <td>None</td>
            <td>None</td>
        </tr>
    </tbody>
</table>

## Example

The sample code is as follows, for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_mixed_quant_sparse_flash_mla.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mixed_quant_sparse_flash_mla.h"
#include "aclnnop/aclnn_mixed_quant_sparse_flash_mla_metadata.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
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

void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
  auto size = GetShapeSize(shape);
  std::vector<uint16_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size && i < 10; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Bf16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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

std::vector<uint8_t> MakeFp8Data(int64_t size, uint8_t value)
{
  std::vector<uint8_t> data(static_cast<size_t>(size), value);
  return data;
}

}  // namespace

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
  int64_t tileSize = 64;
  double softmaxScale = 1.0 / sqrt(static_cast<double>(D));

  int64_t nopeHeadDim = D - ropeHeadDim;
  int64_t quantScaleHeadDim = (nopeHeadDim + tileSize - 1) / tileSize;
  int64_t kvD = nopeHeadDim + ropeHeadDim * 2 + quantScaleHeadDim * 2 + 18;

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
  std::vector<int64_t> attnOutShape = {T1, N1, D};
  std::vector<int64_t> softmaxLseShape = {T1, N1, 1};
  std::vector<int64_t> emptyShape = {0};

  void* qDeviceAddr = nullptr;
  void* oriKvDeviceAddr = nullptr;
  void* cmpKvDeviceAddr = nullptr;
  void* cmpSparseIndicesDeviceAddr = nullptr;
  void* oriBlockTableDeviceAddr = nullptr;
  void* cmpBlockTableDeviceAddr = nullptr;
  void* cuSeqLensQDeviceAddr = nullptr;
  void* cuSeqLensOriKvDeviceAddr = nullptr;
  void* cuSeqLensCmpKvDeviceAddr = nullptr;
  void* seqUsedQDeviceAddr = nullptr;
  void* seqUsedOriKvDeviceAddr = nullptr;
  void* seqUsedCmpKvDeviceAddr = nullptr;
  void* cmpResidualKvDeviceAddr = nullptr;
  void* sinksDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* attnOutDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* oriKv = nullptr;
  aclTensor* cmpKv = nullptr;
  aclTensor* cmpSparseIndices = nullptr;
  aclTensor* oriBlockTable = nullptr;
  aclTensor* cmpBlockTable = nullptr;
  aclTensor* cuSeqLensQ = nullptr;
  aclTensor* cuSeqLensOriKv = nullptr;
  aclTensor* cuSeqLensCmpKv = nullptr;
  aclTensor* seqUsedQ = nullptr;
  aclTensor* seqUsedOriKv = nullptr;
  aclTensor* seqUsedCmpKv = nullptr;
  aclTensor* cmpResidualKv = nullptr;
  aclTensor* sinks = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* attnOut = nullptr;
  aclTensor* softmaxLse = nullptr;

  int64_t qSize = GetShapeSize(qShape);
  int64_t oriKvSize = GetShapeSize(oriKvShape);
  int64_t cmpKvSize = GetShapeSize(cmpKvShape);
  int64_t cmpSparseIndicesSize = GetShapeSize(cmpSparseIndicesShape);
  int64_t oriBlockTableSize = GetShapeSize(oriBlockTableShape);
  int64_t cmpBlockTableSize = GetShapeSize(cmpBlockTableShape);
  int64_t attnOutSize = GetShapeSize(attnOutShape);
  int64_t softmaxLseSize = GetShapeSize(softmaxLseShape);

  std::vector<uint16_t> qHostData = MakeBf16Data(qSize, 1.0f);
  std::vector<uint8_t> oriKvHostData = MakeFp8Data(oriKvSize, 0x38);
  std::vector<uint8_t> cmpKvHostData = MakeFp8Data(cmpKvSize, 0x38);
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

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_BF16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &oriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &cmpKv);
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
  ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensQShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensOriKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensCmpKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &seqUsedQDeviceAddr, aclDataType::ACL_INT32, &seqUsedQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqUsedOriKvHostData, seqUsedOriKvShape, &seqUsedOriKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqUsedCmpKvHostData, seqUsedCmpKvShape, &seqUsedCmpKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpResidualKvHostData, cmpResidualKvShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32, &cmpResidualKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  char layoutQ[] = "TND";
  char layoutKv[] = "PA_BBND";

  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;

  ret = aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv,
      seqUsedQ, seqUsedOriKv, seqUsedCmpKv,
      cmpResidualKv, nullptr, nullptr,
      N1, N2, D, quantMode, B,
      S1, S2, cmpKvLen,
      0, K, ropeHeadDim,
      cmpRatio, oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      true, true,
      metadata,
      &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMixedQuantSparseFlashMlaMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadata failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
      q, oriKv, cmpKv,
      nullptr, cmpSparseIndices,
      oriBlockTable, cmpBlockTable,
      cuSeqLensQ, nullptr, nullptr,
      nullptr, seqUsedOriKv, seqUsedCmpKv,
      cmpResidualKv, nullptr, nullptr,
      sinks, metadata,
      quantMode, ropeHeadDim,
      softmaxScale, cmpRatio,
      oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      1, false,
      attnOut, softmaxLse,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMlaGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMixedQuantSparseFlashMla(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMla failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  PrintOutResult(attnOutShape, &attnOutDeviceAddr);

  aclDestroyTensor(q);
  aclDestroyTensor(oriKv);
  aclDestroyTensor(cmpKv);
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
