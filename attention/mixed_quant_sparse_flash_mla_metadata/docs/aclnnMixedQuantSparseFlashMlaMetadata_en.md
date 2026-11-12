# aclnnMixedQuantSparseFlashMlaMetadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:20:21.440Z -->

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

- Function: `aclnnMixedQuantSparseFlashMlaMetadata` is the preceding operator of the `aclnnMixedQuantSparseFlashMla` operator. It generates a load-balanced task partitioning scheme for the subsequent Attention computation. This operator does not perform the actual Attention computation. Instead, it computes on the AI CPU the start and end ranges of the Attention computation to be processed by each AI Core based on the input parameters, maximizing the utilization of computing resources and avoiding load imbalance across Cores.

  **This operator is not recommended for standalone use. It is recommended to use it together with the aclnnMixedQuantSparseFlashMla operator to form a complete workflow.**
    1. Accepts the input data shape information of the aclnnMixedQuantSparseFlashMla operator API, including batchSize, qSeqlen, kSeqlen, and mask. The input is divided into chunks, and the computing time is simulated to evenly allocate the chunks to available cores. This reduces the overall computing time of the aclnnMixedQuantSparseFlashMla operator and improves hardware utilization.
    2. After the allocation result is output, it is used as the input of the aclnnMixedQuantSparseFlashMla operator.
    3. The allocation result includes the start and end points of the basic block of each AIC core, and the FD task information of each AIV core. For details, see [Example](#example).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call "aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize" first to obtain the workspace size, and then call "aclnnMixedQuantSparseFlashMlaMetadata" to perform computation.

``` cpp
aclnnStatus aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensOriKvOptional,
    const aclTensor   *cuSeqlensCmpKvOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedOriKvOptional,
    const aclTensor   *sequsedCmpKvOptional,
    const aclTensor   *cmpResidualKvOptional,
    const aclTensor   *oriTopkLengthOptional,
    const aclTensor   *cmpTopkLengthOptional,
    int64_t            numHeadsQ,
    int64_t            numHeadsKv,
    int64_t            headDim,
    int64_t            quantMode,
    int64_t            batchSize,
    int64_t            maxSeqlenQ,
    int64_t            maxSeqlenOriKv,
    int64_t            maxSeqlenCmpKv,
    int64_t            oriTopk,
    int64_t            cmpTopk,
    int64_t            ropeHeadDim,
    int64_t            cmpRatio,
    int64_t            oriMaskMode,
    int64_t            cmpMaskMode,
    int64_t            oriWinLeft,
    int64_t            oriWinRight,
    const char        *layoutQOptional,
    const char        *layoutKvOptional,
    bool               hasOriKv,
    bool               hasCmpKv,
    const aclTensor   *metadata,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

``` cpp
aclnnStatus aclnnMixedQuantSparseFlashMlaMetadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 240px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 160px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 170px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>cuSeqlensQOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates different valid Sequence Length of q in each Batch.</td>
      <td><ul><li>Supports empty tensor.</li><li>The shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates different valid Sequence Length of oriKvOptional in each Batch.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates different valid Sequence Length of cmpKvOptional in each batch.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the Sequence Length of q that actually participates in the operation in different Batches.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedOriKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the Sequence Length of oriKvOptional that actually participates in the operation of the operation in different Batches.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the Sequence Length of cmpKvOptional that actually participates in the operation across different Batches.</td>
      <td><ul><li>Supports empty Tensor.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the remainder of the compressed Sequence Length of cmpKvOptional across different Batches. Used together with cmpRatio to compute the mask and load of the cmpKvOptional part.</td>
      <td><ul><li>Supports empty tensor.</li><li>Must be passed in when cmpRatio is not 1 and mask is 3.</li><li>Shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens in the oriKvOptional part corresponding to different q tokens.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape is (B, S1, N2) or (T1, N2).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2D, 3D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens in the cmpKvOptional part corresponding to different q tokens.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is (B, S1, N2) or (T1, N2).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2D, 3D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of q heads.</td>
      <td>Currently supports [1, 128].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv (int64_t)</td>
      <td>Input</td>
      <td>Number of multi-heads corresponding to oriKvOptional and cmpKvOptional.</td>
      <td>Currently only supports 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim (int64_t)</td>
      <td>Input</td>
      <td>Dimension of the attention head.</td>
      <td>Currently only supports 512.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the quantization mode.</td>
      <td><ul><li>1: Q: nope+rope: non-quantized; KV: nope: per-token-group FP8_e4m3. group_size=64; rope: non-quantized and consistent with q; scale: bf16; kv_cache_layout: block_size*(rope[64*2]+nope[448]+scale[448/64*2]+pad[18])</li><li>2: Q: nope+rope: non-quantized; KV: nope: per-token-group FP8_e4m3. group_size=64; rope: non-quantized and consistent with q; scale: e8m0; kv_cache_layout: block_size*(nope+rope)+block_size*(scale+pad[1])</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of batches.</td>
      <td><ul><li>Supports non-negative values.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of q.</td>
      <td><ul><li>Supports non-negative values.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenOriKv (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of oriKvOptional.</td>
      <td><ul><li>Supports non-negative values.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenCmpKv (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of cmpKvOptional.</td>
      <td><ul><li>Supports non-negative numbers.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriTopk (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens filtered in oriKvOptional. 0 indicates a non-sparse scenario.</td>
      <td>Recommended value is 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpTopk (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens filtered out from cmpKvOptional. 0 indicates a non-sparse scenario.</td>
      <td>Recommended value is 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rope_head_dim (int64_t)</td>
      <td>Input</td>
      <td>Dimension of the rope head.</td>
      <td><ul><li>Currently only supports 64.</li><li>Recommended value is 64.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Indicates the compression ratio of cmpKvOptional.</td>
      <td><ul><li>Currently supports [1, 128].</li><li>Recommended value is 1, indicating no compression.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode for the computation of q and oriKvOptional.</td>
      <td><ul><li>0: No Mask.</li><li>3: RightDownCausal mode.</li><li>4: sliding window mode.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode of the computation between q and cmpKvOptional.</td>
      <td><ul><li>0: No Mask.</li><li>3: RightDownCausal mode.</li><li>Recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of past tokens that q attends to during the computation of q and oriKvOptional.</td>
      <td><ul><li>Value range: ≥-1, where -1 indicates infinity.</li><li>Recommended value is -1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of future tokens computed by q in the computation of q and oriKvOptional.</td>
      <td><ul><li>Value range: ≥-1, where -1 indicates infinity.</li><li>Recommended value is -1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format of q.</td>
      <td><ul><li>Supports BSND and TND.</li><li>Recommended value is BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format of oriKvOptional and cmpKvOptional.</td>
      <td><ul><li>Supports BSND, TND, and PA_BBND.</li><li>Recommended value is BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasOriKv (bool)</td>
      <td>Input</td>
      <td>Indicates whether oriKvOptional is included.</td>
      <td><ul><li>true: oriKvOptional is included.</li><li>false: oriKvOptional is not included.</li><li>Recommended value is true.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasCmpKv (bool)</td>
      <td>Input</td>
      <td>Identifies whether cmpKvOptional exists.</td>
      <td><ul><li>true: cmpKvOptional exists.</li><li>false: cmpKvOptional does not exist.</li><li>Recommended value is true.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output of the load balancing result.</td>
      <td>The shape is fixed to (1024, ).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1 dimension</td>
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
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>Failed to create aclOpExecutor.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>Parameter workspaceSize or executor is a null pointer, or parameter cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, or cmpTopkLengthOptional is a null pointer after Contiguous processing.</td>
      </tr>
      <tr>
        <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="1">161002</td>
        <td>The specifications of parameters cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional, numHeadsQ, numHeadsKv, headDim, quantMode, batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, oriTopk, cmpTopk, ropeHeadDim, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv, or hasCmpKv are not within the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnMixedQuantSparseFlashMlaMetadata

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
            <td>Workspace memory address applied by the Device side.</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>Input</td>
            <td>Size of the workspace applied by the Device side, obtained through the first-phase API aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize.</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>Input</td>
            <td>op executor, which contains the computation flow of the operator.</td>
        </tr>
        <tr>
            <td>stream</td>
            <td>Input</td>
            <td>Stream on which the task is executed.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - aclnnMixedQuantSparseFlashMlaMetadata defaults to a deterministic implementation.
  - B (Batch) indicates the batch size of input samples. q, oriKvOptional, and cmpKvOptional are input parameters of the corresponding aclnnMixedQuantSparseFlashMla operator. S1 indicates the size of the S axis in the q shape when layoutQOptional=BSND. T1 indicates the size of the T axis in the q shape when layoutQOptional=TND. S2 indicates the size of the S axis in the oriKvOptional shape when layoutKvOptional=BSND. S3 indicates the size of the S axis in the cmpKvOptional shape when layoutKvOptional=BSND. N2 indicates the size of the N axis in the oriKvOptional and cmpKvOptional shapes.
  - Parameters cuSeqlensQOptional, cuSeqlensOriKvOptional, and cuSeqlensCmpKvOptional must be the cumulative number of valid tokens in the current batch and preceding batches. The first element is fixed to 0. Each subsequent value must be greater than or equal to the previous value.
  - Parameters sequsedQOptional, sequsedOriKvOptional, and sequsedCmpKvOptional require their values to indicate the number of valid tokens in each batch.
  - In non-PA use cases, layoutQOptional and layoutKvOptional must be the same.
  - Parameter cmpResidualKvOptional must satisfy cmpResidualKvOptional[i] < cmpRatio.
  - numHeadsQ must be divisible by numHeadsKv.
  - If hasOriKv is true, oriTopk greater than 0 indicates that the oriKvOptional part is sparse, and oriTopk 0 indicates that the oriKvOptional part is non-sparse.
  - If hasCmpKv is true, cmpTopk greater than 0 indicates that the cmpKvOptional part is sparse, and cmpTopk 0 indicates that the cmpKvOptional part is non-sparse.
  - If hasOriKv is true, oriTopk is not 0, and oriMaskMode is 0, oriTopkLengthOptional must be passed in. In this case, the minimum value of the oriMaskMode rule and the elements in oriTopkLengthOptional is used as the valid seqlen of oriKvOptional corresponding to the current q token. In other oriKvOptional sparse use cases, the minimum value of the oriMaskMode rule and oriTopk is used as the valid seqlen of oriKvOptional corresponding to the current q token.
  - If hasCmpKv is true, cmpTopk is not 0, and cmpMaskMode is 0, cmpTopkLengthOptional must be passed in. In this case, the minimum value of the cmpMaskMode rule and the elements in cmpTopkLengthOptional is used as the valid seqlen of cmpKvOptional corresponding to the current q token. In other cmpKvOptional sparse use cases, the minimum value of the cmpMaskMode rule and cmpTopk is used as the valid seqlen of cmpKvOptional corresponding to the current q token.
  - In the layoutQOptional=BSND use case
    - maxSeqlenQ must be passed in with the value of S1.
  - layoutKvOptional=BSND scenario
    - If hasOriKv is true, maxSeqlenOriKv must be passed in with the value of S2.
    - If hasCmpKv is true, maxSeqlenCmpKv must be passed in with the value of S3.
  - layoutQOptional=TND scenario
    - cuSeqlensQOptional must be passed in.
  - layoutKvOptional=TND scenario
    - If hasOriKv is true, cuSeqlensOriKvOptional must be passed in.
    - If hasCmpKv is true, cuSeqlensCmpKvOptional must be passed in.
  - layoutKvOptional=PA_BBND scenario
    - If hasOriKv is true, oriTopk is not 0, and oriMaskMode is 0 (scenario where oriTopkLengthOptional must be passed in), sequsedOriKvOptional is optional; in other scenarios, sequsedOriKvOptional must be passed in.
    - When hasCmpKv is true, cmpTopk is not 0, and cmpMaskMode is 0 (scenario where cmpTopkLengthOptional must be passed in), sequsedCmpKvOptional is optional. In other scenarios, sequsedCmpKvOptional must be passed in.
  - Batch value rule
    - When layoutQOptional is BSND, prefer to use the shape of sequsedQOptional to derive the batch. If sequsedQOptional is not passed in, get the number of batches through batch_size.
    - When layoutQOptional is TND, prefer to use the shape of sequsedQOptional to derive the batch. If sequsedQOptional is not passed in, use the shape of cuSeqlensQOptional to derive the batch.
  - q Seqlen value rule
    - When layoutQOptional is BSND, prefer to use elements in sequsedQOptional to get seqlen. If sequsedQOptional is not passed in, get seqlen through maxSeqlenQ.
    - When layoutQOptional is TND, prefer to use elements in sequsedQOptional to get seqlen. If sequsedQOptional is not passed in, get seqlen through elements in cuSeqlensQOptional.
  - oriKvOptional Seqlen value rule
    - When layoutKvOptional is BSND, prefer to use elements in sequsedOriKvOptional to get seqlen. If sequsedOriKvOptional is not passed in, get seqlen through maxSeqlenOriKv.
    - When layoutKvOptional is TND, prefer to use elements in sequsedOriKvOptional to get seqlen. If sequsedOriKvOptional is not passed in, get seqlen through elements in cuSeqlensOriKvOptional.
    - When layoutKvOptional is PA_BBND, prefer to use elements in sequsedOriKvOptional to get seqlen. If sequsedOriKvOptional is not passed in, get seqlen through oriTopkLengthOptional.
  - cmpKvOptional seqlen value rule
    - When layoutKvOptional is BSND, prefer to use elements in sequsedCmpKvOptional to get seqlen. If sequsedCmpKvOptional is not passed in, get seqlen through maxSeqlenCmpKv.
    - When layoutKvOptional is TND, prefer to use elements in sequsedCmpKvOptional to get seqlen. If sequsedCmpKvOptional is not passed in, get seqlen through elements in cuSeqlensCmpKvOptional.
    - When layoutKvOptional is PA_BBND, prefer to use elements in sequsedCmpKvOptional to get seqlen. If sequsedCmpKvOptional is not passed in, get seqlen through cmpTopkLengthOptional.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

``` cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_mixed_quant_sparse_flash_mla_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

// Refer to mixed_quant_sparse_flash_mla_metadata.h
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t MQSMLA_METADATA_TOTAL_SIZE = 1024;
constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_S2_MAX_NUM = 8;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

struct MqsmlaMetadata {
    uint32_t faMetadata[AIC_CORE_MAX_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_MAX_NUM][FD_METADATA_SIZE];
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    // Copying is prohibited.
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr { nullptr };
    void *deviceAddr { nullptr };
    aclTensor *data { nullptr };
};

struct ArgScenario {
    bool hasCuSeq { false };
    bool hasSeqused { false };
};

struct ArgContext {
    // required input
    int64_t numHeadsQ { 0 };
    int64_t numHeadsKv { 0 };
    int64_t headDim { 0 };
    int64_t quantMode { 1 };
    // optional input
    Tensor cuSeqlensQOptional {};
    Tensor cuSeqlensOriKvOptional {};
    Tensor cuSeqlensCmpKvOptional {};
    Tensor sequsedQOptional {};
    Tensor sequsedOriKvOptional {};
    Tensor sequsedCmpKvOptional {};
    Tensor cmpResidualKvOptional {};
    Tensor oriTopkLengthOptional {};
    Tensor cmpTopkLengthOptional {};
    int64_t batchSize { 0 };
    int64_t maxSeqlenQ { 0 };
    int64_t maxSeqlenOriKv { 0 };
    int64_t maxSeqlenCmpKv { 0 };
    int64_t oriTopk { 0 };
    int64_t cmpTopk { 0 };
    int64_t ropeHeadDim { 64 };
    int64_t cmpRatio { 0 };
    int64_t oriMaskMode { 0 };
    int64_t cmpMaskMode { 0 };
    int64_t oriWinLeft { -1 };
    int64_t oriWinRight { -1 };
    char *layoutQOptional { nullptr };
    char *layoutKvOptional { nullptr };
    bool hasOriKv { true };
    bool hasCmpKv { true };
    // output
    Tensor metadata {};
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate, initialization
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // Call aclrtMallocHost to allocate host-side memory.
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // Call aclrtMalloc to allocate device memory.
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // Call the aclCreateTensor API to create an aclTensor.
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);
    CHECK_LOG_RET(tensor.data != nullptr, ACL_ERROR_FAILURE, "aclCreateTensor failed");
    // Call aclrtMemcpy to copy host-side data to device memory.
    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqlensQOptional);
    DestroyTensor(context.cuSeqlensOriKvOptional);
    DestroyTensor(context.cuSeqlensCmpKvOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedOriKvOptional);
    DestroyTensor(context.sequsedCmpKvOptional);
    DestroyTensor(context.cmpResidualKvOptional);
    DestroyTensor(context.oriTopkLengthOptional);
    DestroyTensor(context.cmpTopkLengthOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKvOptional != nullptr) {
        free(context.layoutKvOptional);
        context.layoutKvOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    context.numHeadsQ = 64;
    context.numHeadsKv = 1;
    context.headDim = 512;
    context.quantMode = 1;
    ret = CreateTensor(aclDataType::ACL_INT32, { MQSMLA_METADATA_TOTAL_SIZE }, context.metadata);     // 1024: Fix size
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);
    context.oriTopk = 0;
    context.cmpTopk = 0;
    context.ropeHeadDim = 64;
    context.cmpRatio = 128;
    context.oriMaskMode = 4;
    context.cmpMaskMode = 3;
    context.oriWinLeft = 127;
    context.oriWinRight = 0;
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKvOptional = (char *)malloc(sizeof(char) * 16);
    CHECK_LOG_RET(context.layoutQOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutQOptional failed");
    CHECK_LOG_RET(context.layoutKvOptional != nullptr, ACL_ERROR_FAILURE, "Create layoutKvOptional failed");
    strcpy(context.layoutQOptional, "BSND");                // BSND,TND
    strcpy(context.layoutKvOptional, "BSND");               // BSND,TND,PA_BBND
    context.hasOriKv = true;
    context.hasCmpKv = true;

    context.batchSize = 4;
    context.maxSeqlenOriKv = 1024;
    context.maxSeqlenCmpKv = 1024;
    context.maxSeqlenQ = 1024;

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize + 1 }, context.cuSeqlensCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensCmpKvOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.sequsedCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedCmpKvOptional failed. Error: %d", ret);
    }

    if (context.hasCmpKv && context.cmpRatio != 1 && context.cmpMaskMode == 3) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { context.batchSize }, context.cmpResidualKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKvOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main() {
    // 1. (Boilerplate) Initialize the device/stream. See the public API list.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. Construct inputs and outputs based on the API definition.
    ArgScenario scenario {};
    scenario.hasCuSeq = false;
    scenario.hasSeqused = false;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. Call the CANN operator library API. Replace it with the specific API.
    // Call the first-phase API of aclnnMixedQuantSparseFlashMlaMetadata.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensOriKvOptional.data, context.cuSeqlensCmpKvOptional.data,
        context.sequsedQOptional.data, context.sequsedOriKvOptional.data, context.sequsedCmpKvOptional.data,
        context.cmpResidualKvOptional.data, context.oriTopkLengthOptional.data, context.cmpTopkLengthOptional.data,
        context.numHeadsQ, context.numHeadsKv, context.headDim, context.quantMode, context.batchSize,
        context.maxSeqlenQ, context.maxSeqlenOriKv, context.maxSeqlenCmpKv, context.oriTopk, context.cmpTopk,
        context.ropeHeadDim, context.cmpRatio, context.oriMaskMode, context.cmpMaskMode, context.oriWinLeft,
        context.oriWinRight, context.layoutQOptional, context.layoutKvOptional, context.hasOriKv, context.hasCmpKv,
        context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d\n", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    // Call the second-phase API of aclnnMixedQuantSparseFlashMlaMetadata.
    ret = aclnnMixedQuantSparseFlashMlaMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnMixedQuantSparseFlashMlaMetadata failed. ERROR: %d\n", ret);

    // 4. (Boilerplate) Wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. Print output.
    MqsmlaMetadata result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < AIC_CORE_MAX_NUM; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Core Enable : %u\n", result.faMetadata[i][FA_CORE_ENABLE_INDEX]);
        printf("    Start BN2   : %u\n", result.faMetadata[i][FA_BN2_START_INDEX]);
        printf("    Start M     : %u\n", result.faMetadata[i][FA_M_START_INDEX]);
        printf("    Start S2    : %u\n", result.faMetadata[i][FA_S2_START_INDEX]);
        printf("    End BN2     : %u\n", result.faMetadata[i][FA_BN2_END_INDEX]);
        printf("    End M       : %u\n", result.faMetadata[i][FA_M_END_INDEX]);
        printf("    End S2      : %u\n", result.faMetadata[i][FA_S2_END_INDEX]);
        printf("    First Workspace Index : %u\n", result.faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX]);
        printf("    Max S2 Block Num : %u\n", result.faMetadata[i][FA_S2_MAX_NUM]);
    }
    for (uint32_t i = 0; i < AIV_CORE_MAX_NUM; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result.fdMetadata[i][FD_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result.fdMetadata[i][FD_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result.fdMetadata[i][FD_M_IDX_INDEX]);
        printf("    FD Task Workspace Idx   : %u\n", result.fdMetadata[i][FD_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdMetadata[i][FD_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdMetadata[i][FD_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdMetadata[i][FD_M_NUM_INDEX]);
    }

    return 0;
}
```
