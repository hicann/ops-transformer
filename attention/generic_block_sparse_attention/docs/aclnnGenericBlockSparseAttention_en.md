# aclnnGenericBlockSparseAttention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:06:47.236Z -->

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

- **Description**: GenericBlockSparseAttention performs sparse attention computation. It supports block sparse mode with arbitrary granularity along the S axis. sparseBlockIdx specifies each Q block's selected KV blocks, and sparseBlockCount specifies the number of KV blocks retained for each Q block, enabling efficient sparse attention computation. Call aclnnGenericBlockSparseAttentionMetadata to generate metadata first.

- **Formula**: Sparse block size: $blockShapeX \times blockShapeY$

  $$
  attentionOut = Softmax(scaleValue \cdot query \cdot key_{sparse}^{T} + atten\_mask) \cdot value_{sparse}
  $$

  The input layout is specified by layoutQ and layoutKv. Supported combinations are listed under <a href="#other-constraints">Other Constraints</a> (TND/BSND/BNSD with dense or PA_BBND/PA_BNBD on Ascend 950; see that section for product scope).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnGenericBlockSparseAttentionGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnGenericBlockSparseAttention" API to execute the computation.

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *sparseBlockIdx,
  const aclTensor   *sparseBlockCount,
  const aclTensor   *metadataOptional,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *qDequantScaleOptional,
  const aclTensor   *kDequantScaleOptional,
  const aclTensor   *vDequantScaleOptional,
  const aclTensor   *pQuantScaleOptional,
  const aclTensor   *cuSeqLengthsQOptional,
  const aclTensor   *cuSeqLengthsKvOptional,
  const aclTensor   *sequsedQOptional,
  const aclTensor   *sequsedKvOptional,
  const aclTensor   *blockTableOptional,
  const aclIntArray *blockShape,
  char              *layoutQ,
  char              *layoutKv,
  int64_t            layoutSparsePattern,
  double             scaleValue,
  int64_t            maskType,
  int64_t            quantType,
  double             dstTypeMax,
  int64_t            softmaxPrecision,
  int64_t            winLeft,
  int64_t            winRight,
  int64_t            returnSoftmaxlse,
  int64_t            residualBlockMode,
  bool               isConsistentTopK,
  aclTensor         *attentionOut,
  aclTensor         *softmaxLseOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnGenericBlockSparseAttentionGetWorkspaceSize

- **Parameters**

  For the meaning of the shape variables in the table, see <a href="#baseline-description">Baseline Description</a>.

  <table style="table-layout: fixed; width: 100%; word-break: break-word; overflow-wrap: anywhere;">
  <colgroup>
  <col style="width: 12%">
  <col style="width: 6%">
  <col style="width: 15%">
  <col style="width: 25%">
  <col style="width: 11%">
  <col style="width: 5%">
  <col style="width: 20%">
  <col style="width: 6%">
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
      <td>query</td>
      <td>Input</td>
      <td>The query in the formula.</td>
      <td>Supports layoutQ="TND"/"BSND"/"BNSD". For shapes and product scope, see <a href="#layout-correspondence-description">Layout Correspondence Description</a> and <a href="#other-constraints">Other Constraints</a>.</td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT4_E2M1FN, HIFLOAT8</td>
      <td>ND</td>
      <td>[totalQTokens, headNum, headDim]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>key in the formula.</td>
      <td>Can be used as the original Key or Paged KV Cache. Supports layoutKv="TND"/"BSND"/"BNSD"/"PA_BBND"/"PA_BNBD" within product scope, including matching dense Q/KV layouts or PA layouts subject to product and quantization constraints below. For shapes, see <a href="#layout-correspondence-description">Layout Correspondence Description</a> and <a href="#paged-attention-related-description">Paged Attention Related Description</a>; for non-contiguous dim0, see <a href="#other-constraints">Other Constraints</a>.</td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT4_E2M1FN, HIFLOAT8</td>
      <td>ND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads, headDim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>value in the formula.</td>
      <td>Can be used as the original Value input or Paged KV Cache input, and its shape is the same as that of key. Supported layouts match key, including PA_BBND/PA_BNBD KV or matching dense Q/KV layouts, subject to the product and quantization constraints below. See <a href="#other-constraints">Other Constraints</a>.</td>
      <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT4_E2M1FN, HIFLOAT8</td>
      <td>ND</td>
      <td>Same as key</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sparseBlockIdx</td>
      <td>Input</td>
      <td>Sparse block index array. Specifies the KV block indices selected by each Q block.</td>
      <td>The value must be a valid KV block index (blocked based on the storage length of cu); invalid positions can be filled with -1, and valid values must be within the first sparseBlockCount positions. Detailed Reference: <a href="#layout-correspondence-description">layout Correspondence Description</a>.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks, maxKvBlockCount]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sparseBlockCount</td>
      <td>Input</td>
      <td>Number of KV blocks actually selected for each Q block.</td>
      <td>Detailed Reference: <a href="#layout-correspondence-description">layout Correspondence Description</a>.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[numKeyValueHeads, totalQBlocks]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>Input</td>
      <td>Core split result of the AICPU operator aclnnGenericBlockSparseAttentionMetadata.</td>
      <td>        Must be passed in. Generated by aclnnGenericBlockSparseAttentionMetadata and must be regenerated for each call.
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td>atten_mask in the formula.</td>
      <td>
        See <a href="#mask-description">Mask Description</a>.<br>
        maskType=0/1: must be nullptr.<br>
        maskType=3 (Union): must be INT32 exactMask with shape [kvHeads, G, U×blockShapeY];
        bit <code>i</code> of each int32 (from the low bit, i&lt;unionSize) marks whether the i-th Q in the group can see that KV column.
      </td>
      <td>BOOL, INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qDequantScaleOptional</td>
      <td>Input</td>
      <td>Dequantization scaling factor of query.</td>
      <td>See <a href="#quantization-related-description">Quantization Related Description</a> for details. Currently, nullptr can be passed in only in scenarios where quantType=0/5; quantType=1-4 is currently not supported, and nullptr must be passed in.
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>kDequantScaleOptional</td>
      <td>Input</td>
      <td>Dequantization scaling factor of key.</td>
      <td>See <a href="#quantization-related-description">Quantization Related Description</a>. Currently, nullptr can be passed in only quantType=0/5 use cases; quantType=1-4 is currently not supported, and nullptr must be passed in.
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>vDequantScaleOptional</td>
      <td>Input</td>
      <td>Dequantization scaling factor of value.</td>
      <td>See Details in <a href="#quantization-related-description">Quantization Related Description</a>. Currently, nullptr can be passed in only quantType=0/5 scenarios; quantType=1-4 is currently not supported, and nullptr must be passed.
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>x</td>
      <td>×</td>
    </tr>
    <tr>
      <td>pQuantScaleOptional</td>
      <td>Input</td>
      <td>Quantization coefficients required for the P matrix of the online-softmax result in non-mx quantization mode.</td>
      <td>See Details in <a href="#quantization-related-description">Quantization Related Description</a>. Currently, nullptr must be passed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>x</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqLengthsQOptional</td>
      <td>Input</td>
      <td>Describes the query sequence length of each batch, stored as prefix sums.</td>
      <td>        Optional input, used for variable-length sequence use cases.
        <ul>
          <li>Must be passed in when layoutQ is TND.</li>
          <li>When layoutQ is BNSD/BSND: if passed in, the operator processes data based on the actual sequence length specified by this input; if nullptr is passed in, S in the query shape is used (currently not supported).</li>
          <li>Elements are prefix sums: the 0th element is 0, the last element equals totalQTokens, and each subsequent element must be ≥ the previous element; the <strong>storage length</strong> qStorageLen_i of each batch is obtained by differencing adjacent elements, and Σ qStorageLen_i = totalQTokens.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqLengthsKvOptional</td>
      <td>Input</td>
      <td>Describes the key/value sequence length of each batch, stored as prefix sums.</td>
      <td>        Optional input, used for variable-length sequence use cases.
        <ul>
          <li>Must be passed in when layoutKv is TND.</li>
          <li>Not passed in when layoutKv is not TND (must be nullptr). For PA_BBND/PA_BNBD, pass sequsedKvOptional instead.</li>
          <li>When layoutKv is BNSD/BSND: if passed in, the operator processes data based on the actual sequence length specified by this input; if nullptr is passed in, S in the key/value shape is used (currently not supported).</li>
          <li>Elements are prefix sums: the 0th element is 0, the last element equals the sum of KV storage lengths of all batches, and each subsequent element must be ≥ the previous element.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>Input</td>
      <td>Actual sequence length of the query in each batch.</td>
      <td>        <ul>
          <li>If the actual length is not specified, nullptr can be passed in, indicating that the storage length is the same as that obtained by difference from cuSeqLengthsQOptional.</li>
          <li>If passed, the shape is (B,), and each element must be ≥0 and ≤ the storage length of cu in the corresponding batch. See <a href="#other-constraints">Other Constraints</a> for the dual-length semantics when coexisting with cu.</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>Input</td>
      <td>Actual sequence length of KV in each batch.</td>
      <td>        <ul>
          <li>Must be passed in when layoutKv is PA_BBND/PA_BNBD, with shape (B,); each element must be ≥ 0.</li>
          <li>When layoutKv is TND: if the actual length is not specified, nullptr can be passed in, indicating that the storage length is the same as that obtained by difference from cuSeqLengthsKvOptional; if passed in, shape is (B,) and each element must be ≥ 0 and ≤ the storage length of cu in the corresponding batch. For the dual-length semantics when coexisting with cu, see <a href="#other-constraints">Other Constraints</a>.</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>Input</td>
      <td>The block table is used for PagedAttention.</td>
      <td>Must be passed in when layoutKv is PA_BBND/PA_BNBD; must be nullptr for non-PA layouts. See <a href="#paged-attention-related-description">Paged Attention Related Description</a> and <a href="#other-constraints">Other Constraints</a>.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, maxNumBlocksPerBatch)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>Input</td>
      <td>Represents the sparse block shape array.</td>
      <td>Contains two elements [blockShapeX, blockShapeY]. Non-union uses [1,Y]; union (maskType=3) uses [8,Y] (unionSize is read from blockShapeX; currently only 8). See product constraints in <a href="#other-constraints">Other Constraints</a> and the mask description.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>Input</td>
      <td>Represents the data layout format of the input query.</td>
      <td>Supported targets: "TND", "BNSD", and "BSND". For details and product scope, see <a href="#layout-correspondence-description">Layout Correspondence Description</a> and <a href="#other-constraints">Other Constraints</a>.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>Input</td>
      <td>Represents the data layout of the input key and value.</td>
      <td>The target supports "TND", "BNSD", "BSND", "PA_BBND", and "PA_BNBD". See <a href="#layout-correspondence-description">Layout Correspondence Description</a> and <a href="#other-constraints">Other Constraints</a> for product scope.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutSparsePattern</td>
      <td>Input</td>
      <td>Represents the data layout format of input sparseBlockIdx and sparseBlockCount.</td>
      <td>Currently only supports the value "4". Detailed reference: <a href="#layout-correspondence-description">layout correspondence description</a>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>The scale in the formula. Represents the scaling factor.</td>
      <td>Generally set to D^-0.5. If 0 is passed, the operator processes it as 1/√D internally.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>Input</td>
      <td>Represents the mask type in the attention computation.</td>
      <td>        Value range: 0-5. For details, see <a href="#mask-description">Mask Description</a>.<br>
        Currently supports 0 (non-causal) and 1 (causal); Ascend 950PR&amp;950DT products and non-quantized Atlas A2/A3 products also support 3 (Union exactMask).
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantType</td>
      <td>Input</td>
      <td>Represents the quantization method used.</td>
      <td>        Value range: 0-6. See <a href="#quantization-related-description">Quantization Related Description</a> for the complete configuration:<br>
        Currently available: 0, 5, and 6 on Ascend 950PR&amp;950DT products. If 1-4 is passed, the check fails.
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax</td>
      <td>Input</td>
      <td>Custom quantization range passed in for MXFP4 CX quantization.</td>
      <td>        <ul>
          <li>Custom ranges are currently not supported. Must be passed in as 0.0.</li>
        </ul>
      </td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxPrecision</td>
      <td>Input</td>
      <td>Precision level used for Softmax computation.</td>
      <td>        Controls the data type used for operations in the online softmax and rescale phases. The value is 0 or 1:
        <ul>
          <li>0: fp32 is used for both online softmax and rescale. Suitable for use cases that require high computation precision.</li>
          <li>1: Mixed precision; fp16/bf16 (same as attentionOut) is used for online softmax, and fp32 is used for rescale. Numeric overflow may occur in the online softmax phase.</li>
        </ul>
        On Ascend 950PR&950DT: in non-Union scenarios (maskType=0/1), non-quantized FLOAT16/BFLOAT16 supports 0 or 1, quantType=5 FP8 supports only 1, and MXFP8 (quantType=6) uses FP32 scores and softmax for both values (0 is recommended, 1 is a compatibility alias). In Union scenarios (maskType=3), both non-quantized and FP8 quantization support only 0. On Atlas A2/A3, FLOAT16 supports 0 or 1 and BFLOAT16 supports only 0.
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>Input</td>
      <td>In sliding window attention use cases, the number of preceding tokens that the sliding window needs to include.</td>
      <td>Used for sliding window attention; must be -1 when not enabled and must be used together with maskType. See <a href="#mask-description">Mask Description</a>. Currently, only -1 can be passed in.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>Input</td>
      <td>In sliding window attention use cases, the number of backward tokens that the sliding window needs to include.</td>
      <td>Used for sliding window attention; must be -1 when not enabled and must be used together with maskType. See <a href="#mask-description">Mask Description</a>. Currently only supports passing in -1.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnSoftmaxlse</td>
      <td>Input</td>
      <td>Flag indicating whether to enable softmaxLse output.</td>
      <td>    Non-quant supports 0 or 1; FP8 union (D=128) supports 0 or 1; MXFP8 and other FP8 paths with quantType=5 support only 0.
    <ul>
          <li>0: indicates that softmaxLse is not output, and softmaxLseOptional must be passed as nullptr</li>
          <li>1: indicates that softmaxLse is output, and softmaxLseOptional is required</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residualBlockMode</td>
      <td>Input</td>
      <td>Represents the state of the trailing incomplete block after the KV sequence is sparsified in units of blockShapeY.</td>
      <td>Supports 0 or 1 (union supports only 0). 0 indicates that whether the trailing incomplete block participates in the computation is determined by the value passed in for sparseBlockIdx. 1 indicates that the trailing incomplete block must participate in the computation but is definitely not included in sparseBlockIdx.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isConsistentTopK</td>
      <td>Input</td>
      <td>Whether the maximum number of KV blocks selected by each Q block is the same within the same batch and head when the preceding operator performs sparse block selection.</td>
      <td>Only supports 0 or 1. 0 indicates inconsistency, and 1 indicates consistency.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>Output</td>
      <td>attentionOut in the formula.</td>
      <td>Data type and shape are the same as query. layoutQ supports TND, BSND, or BNSD; for product scope, see <a href="#other-constraints">Other Constraints</a>.</td>
      <td>FLOAT16, BFLOAT16 (specified by this tensor for FP8 input)</td>
      <td>ND</td>
      <td>Same as query</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>Output</td>
      <td>Intermediate log-sum-exp result of the Softmax computation.</td>
      <td>When returnSoftmaxlse=1 (non-quant, or FP8 union with D=128), outputs FLOAT32; TND is [Tq,Nq,1], BSND/BNSD is [B,Nq,Sq,1]. LSE output is not supported for regular FP8 (non-Union quantType=5) or MXFP8 (quantType=6). When disabled, pass nullptr.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 280px">
  <col style="width: 90px">
  <col style="width: 785px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Required pointers such as query/key/value/sparseBlockIdx/sparseBlockCount/attentionOut are null.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>Parameters such as layout, maskType, blockShape, softmaxPrecision, quantType, returnSoftmaxlse, layoutSparsePattern, residualBlockMode, or isConsistentTopK do not match the constraints.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>metadata is null, or Contiguous/InferShape fails (for example, layout is not supported, or blockTable is missing).</td>
    </tr>
  </tbody>
  </table>

## aclnnGenericBlockSparseAttention

- **Parameters**

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
      <td>Size of the workspace applied on the Device side, obtained by calling the first-phase API aclnnGenericBlockSparseAttentionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

### Baseline Description

The common fields in the constraints are defined as follows:

| Naming | Meaning |
| :---: | :--- |
| B | Batch Size, batch size of input samples. |
| S | Sequence Length, sequence length. |
| H | Head Size, size of the hidden layer. |
| N | Head Num, number of heads. |
| D | Head Dim, head dimension, where D=H/N. |
| T | Sum of sequence lengths of all batches. |
| totalQTokens | T of the query. |
| totalKTokens | T of the key. |
| totalVTokens | T of the value. |
| headNum | N of the query. |
| numKeyValueHeads | N of key/value. |
| headDim | D |
| batch | B |
| maxQSeqLength | S of the query in BNSD/BSND format. |
| maxKvSeqLength | S of key/value in BNSD/BSND format. |
| numBlocks | Number of physical pages of the Paged KV Cache. |
| blockSize | Number of tokens each page holds. |
| maxNumBlocksPerBatch | Second dimension of blockTable, which must be ≥ ceilDiv(maxKvSeqLength, blockSize). |
| qStorageLen_i | Storage length of each batch obtained by difference of the prefix sum of cuSeqLengthsQ. |
| totalQBlocks | Total number of Q blocks after chunking by storage length, i.e., $\sum_i \mathrm{ceilDiv}(qStorageLen_i, blockShapeX)$ |
| totalKBlocks | Total number of KV blocks after chunking by storage length, i.e., $\sum_i \mathrm{ceilDiv}(kvStorageLen_i, blockShapeY)$ |
| maxKvBlockCount | Last dimension of sparseBlockIdx, which must be no less than the maximum of all elements in sparseBlockCount. Capacity upper limit is INT32_MAX. |
| blockShapeX / blockShapeY | Size of sparse blocks in the Q and KV directions. |

### Layout Correspondence Description

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 160px">
  <col style="width: 180px">
  <col style="width: 815px">
  </colgroup>
  <thead>
    <tr>
      <th>layoutQ</th>
      <th>layoutKv</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">TND</td>
      <td>TND</td>
      <td>Used for the original KV input, in use cases where the seqlen of each batch is concatenated.</td>
    </tr>
    <tr>
      <td>PA_BBND</td>
      <td>Used for the paged KV cache input, with data arranged as [numBlocks, blockSize, numKeyValueHeads, headDim].</td>
    </tr>
    <tr>
      <td>PA_BNBD</td>
      <td>Used for paged kv cache input, data by [numBlocks, numKeyValueHeads, blockSize, headDim] layout.</td>
    </tr>
    <tr>
      <td rowspan="3">BSND</td>
      <td>BSND</td>
      <td>Used for original KV input.</td>
    </tr>
    <tr>
      <td>PA_BBND</td>
      <td>Used for paged kv cache input, data by [numBlocks, blockSize, numKeyValueHeads, headDim] layout.</td>
    </tr>
    <tr>
      <td>PA_BNBD</td>
      <td>Used for paged kv cache input, data by [numBlocks, numKeyValueHeads, blockSize, headDim] layout.</td>
    </tr>
    <tr>
      <td rowspan="3">BNSD</td>
      <td>BNSD</td>
      <td>Used for original KV input.</td>
    </tr>
    <tr>
      <td>PA_BBND</td>
      <td>Used for paged kv cache input, with data laid out in [numBlocks, blockSize, numKeyValueHeads, headDim].</td>
    </tr>
    <tr>
      <td>PA_BNBD</td>
      <td>Used for paged kv cache input, with data laid out in [numBlocks, numKeyValueHeads, blockSize, headDim].</td>
    </tr>
  </tbody>
  </table>

Supported combinations and product scope are listed under <a href="#other-constraints">Other Constraints</a>; other layoutQ/layoutKv combinations are not supported.

The shapes of query and attentionOut are determined by layoutQ:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 160px">
  <col style="width: 815px">
  <col style="width: 180px">
  </colgroup>
  <thead>
    <tr>
      <th>layoutQ</th>
      <th>shape</th>
      <th>Currently Supported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TND</td>
      <td>[totalQTokens, headNum, headDim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>BNSD</td>
      <td>[batch, headNum, maxQSeqLength, headDim]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>BSND</td>
      <td>[batch, maxQSeqLength, headNum, headDim]</td>
      <td>×</td>
    </tr>
  </tbody>
  </table>

The shapes of sparseBlockIdx and sparseBlockCount are determined by layoutSparsePattern. Currently, layoutSparsePattern only supports the value 4.

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 140px">
  <col style="width: 380px">
  <col style="width: 260px">
  <col style="width: 375px">
  </colgroup>
  <thead>
    <tr>
      <th>layoutSparsePattern</th>
      <th>sparseBlockIdx</th>
      <th>sparseBlockCount</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>[batch, numKeyValueHeads, maxQBlockCount, maxKvBlockCount]</td>
      <td>[batch, numKeyValueHeads, maxQBlockCount]</td>
      <td><ul><li>qHeads in the same Group share sparsePattern.</li><li>Indicates which KV blocks are selected by each Q block.</li></ul></td>
    </tr>
    <tr>
      <td>1</td>
      <td>[batch, numKeyValueHeads, maxKvBlockCount, maxQBlockCount]</td>
      <td>[batch, numKeyValueHeads, maxKvBlockCount]</td>
      <td><ul><li>qHeads in the same Group share sparsePattern.</li><li>Indicates the Q blocks selected by each KV block.</li></ul></td>
    </tr>
    <tr>
      <td>2</td>
      <td>[batch, headNum, maxQBlockCount, maxKvBlockCount]</td>
      <td>[batch, headNum, maxQBlockCount]</td>
      <td><ul><li>qHeads in the same Group have independent sparsePattern.</li><li>Indicates the KV blocks selected by each Q block.</li></ul></td>
    </tr>
    <tr>
      <td>3</td>
      <td>[batch, headNum, maxKvBlockCount, maxQBlockCount]</td>
      <td>[batch, headNum, maxKvBlockCount]</td>
      <td><ul><li>qHeads in the same Group have independent sparsePattern.</li><li>Indicates the Q blocks selected by each KV block.</li></ul></td>
    </tr>
    <tr>
      <td>4</td>
      <td>[numKeyValueHeads, totalQBlocks, maxKvBlockCount]</td>
      <td>[numKeyValueHeads, totalQBlocks]</td>
      <td><ul><li>qHeads in the same Group share sparsePattern.</li><li>Indicates which KV blocks are selected by each Q block.</li></ul></td>
    </tr>
    <tr>
      <td>5</td>
      <td>[numKeyValueHeads, totalKBlocks, maxQBlockCount]</td>
      <td>[numKeyValueHeads, totalKBlocks]</td>
      <td><ul><li>qHeads in the same Group share sparsePattern.</li><li>Indicates which Q blocks are selected by each KV block.</li></ul></td>
    </tr>
    <tr>
      <td>6</td>
      <td>[headNum, totalQBlocks, maxKvBlockCount]</td>
      <td>[headNum, totalQBlocks]</td>
      <td><ul><li>qHeads in the same Group have independent sparsePattern.</li><li>Indicates the KV blocks selected by each Q block.</li></ul></td>
    </tr>
    <tr>
      <td>7</td>
      <td>[headNum, totalKBlocks, maxQBlockCount]</td>
      <td>[headNum, totalKBlocks]</td>
      <td><ul><li>qHeads in the same Group have independent sparsePattern.</li><li>Indicates the Q blocks selected by each KV block.</li></ul></td>
    </tr>
  </tbody>
  </table>

The shapes of key and value are determined by layoutKv and whether Paged Cache is enabled. See <a href="#paged-attention-related-description">Paged Attention Related Description</a>.

### Paged Attention Related Description

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 420px">
  <col style="width: 160px">
  <col style="width: 575px">
  </colgroup>
  <thead>
    <tr>
        <th>blockTable</th>
        <th>kvLayout</th>
        <th>Key/Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Non-empty, with shape [batch, maxNumBlocksPerBatch], indicating that paged cache is enabled.</td>
      <td>PA_BBND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads, headDim]</td>
    </tr>
    <tr>
      <td>PA_BNBD</td>
      <td>[numBlocks, numKeyValueHeads, blockSize, headDim]</td>
    </tr>
    <tr>
      <td rowspan="3">Empty, indicating that paged cache is not enabled. The operator receives the original KV input.</td>
      <td>TND</td>
      <td>[totalKTokens, numKeyValueHeads, headDim]</td>
    </tr>
    <tr>
      <td>BSND</td>
      <td>[batch, maxKvSeqLength, numKeyValueHeads, headDim]</td>
    </tr>
    <tr>
      <td>BNSD</td>
      <td>[batch, numKeyValueHeads, maxKvSeqLength, headDim]</td>
    </tr>
  </tbody>
  </table>

PA_BBND/PA_BNBD require a non-empty blockTable. TND/BSND/BNSD require nullptr. For supported layout combinations and product scope, see <a href="#other-constraints">Other Constraints</a>.

### Quantization Related Description

The following quantization configurations are described by the Ascend 950PR&950DT products (A5) generation; quantType=0 is Currently Available, and quantType=5 is Optional on Ascend 950PR&950DT products.

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 120px">
  <col style="width: 240px">
  <col style="width: 130px">
  <col style="width: 240px">
  <col style="width: 260px">
  <col style="width: 280px">
  <col style="width: 280px">
  </colgroup>
  <thead>
    <tr>
        <th>quantType</th>
        <th>QKV Data Type</th>
        <th>Symmetric/Asymmetric</th>
        <th>P Quantization Dynamic/Static</th>
        <th>Quantization Granularity</th>
        <th>Quantization Parameter shape</th>
        <th>Quantization Parameter dType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Non-quantized. QKV is used directly as input for computation.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>        qDequantScaleOptional(not passed in)<br>
        kDequantScaleOptional(not passed in)<br>
        vDequantScaleOptional(not passed in)<br>
        pQuantScaleOptional(not passed in)
      </td>
      <td>-</td>
    </tr>
    <tr>
      <td>1</td>
      <td rowspan="2">FLOAT8_E4M3</td>
      <td rowspan="4">Symmetric</td>
      <td>Static</td>
      <td>perGroup, QKV are grouped along the S dimension, and the group size must be the same as the sparse block size.<br>Specifically, when KV is a paged cache, blockSize must be an integer multiple of blockShapeY.</td>
      <td>        qDequantScaleOptional(required):
        <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeX), headNum, 1].</li>
          <li>BNSD: [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), 1].</li>
          <li>BSND: [batch, ceilDiv(maxQSeqLength, blockShapeX), headNum, 1].</li>
        </ul>
        kDequantScaleOptional(required):
        <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeY), kvHeadNum, 1].</li>
          <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, blockShapeY), 1].</li>
          <li>BSND: [batch, ceilDiv(maxKvSeqLength, blockShapeY), kvHeadNum, 1].</li>
          <li>PA_BBND: [batch, ceilDiv(blockSize, blockShapeY), kvHeadNum, 1].</li>
          <li>PA_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, blockShapeY), 1].</li>
        </ul>
        vDequantScaleOptional(required):
        <ul>
          <li>TND: [batch*ceilDiv(qSeqLength, blockShapeY), kvHeadNum, 1].</li>
          <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, blockShapeY), 1].</li>
          <li>BSND: [batch, ceilDiv(maxKvSeqLength, blockShapeY), kvHeadNum, 1].</li>
          <li>PA_BBND: [batch, ceilDiv(blockSize, blockShapeY), kvHeadNum, 1].</li>
          <li>PA_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, blockShapeY), 1].</li>
        </ul>
        pQuantScaleOptional(optional):
        <ul>
          <li>When passed in, it contains only a single element, which is used by the user to control the static quantization factor of P: [1].</li>
          <li>nullptr: The operator defaults the static quantization factor of P to 448.0.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
    </tr>
    <tr>
      <td>2</td>
      <td>dynamic</td>
      <td rowspan="3">micro scaling, QKV is grouped along the matrix multiplication accumulation axis in a fixed group size of 32.<br>Specifically, when KV is a paged cache, blockSize must be an integer multiple of 64.</td>
      <td rowspan="3">        qDequantScaleOptional(required):
        <ul>
          <li>TND: [totalQTokens, headNum, ceilDiv(headDim, 64), 2].</li>
          <li>BNSD: [batch, headNum, maxQSeqLength, ceilDiv(headDim, 64), 2].</li>
          <li>BSND: [batch, maxQSeqLength, headNum, ceilDiv(headDim, 64), 2].</li>
        </ul>
        kDequantScaleOptional(required):
        <ul>
          <li>TND: [totalKTokens, kvHeadNum, ceilDiv(headDim, 64), 2].</li>
          <li>BNSD: [batch, kvHeadNum, maxKvSeqLength, ceilDiv(headDim, 64), 2].</li>
          <li>BSND: [batch, maxKvSeqLength, kvHeadNum, ceilDiv(headDim, 64), 2].</li>
          <li>PA_BBND: [batch, blockSize, kvHeadNum, ceilDiv(headDim, 64), 2].</li>
          <li>PA_BNBD: [batch, kvHeadNum, blockSize, ceilDiv(headDim, 64), 2].</li>
        </ul>
        vDequantScaleOptional(required):
        <ul>
          <li>TND: [batch*ceilDiv(kvSeqLength, 64), kvHeadNum, headDim, 2].</li>
          <li>BNSD: [batch, kvHeadNum, ceilDiv(maxKvSeqLength, 64), headDim, 2].</li>
          <li>BSND: [batch, ceilDiv(maxKvSeqLength, 64), kvHeadNum, headDim, 2].</li>
          <li>PA_BBND: [batch, ceilDiv(blockSize, 64), kvHeadNum, headDim, 2].</li>
          <li>PA_BNBD: [batch, kvHeadNum, ceilDiv(blockSize, 64), headDim, 2].</li>
        </ul>
      </td>
      <td rowspan="3">FLOAT8_E4M3</td>
    </tr>
    <tr>
      <td>3</td>
      <td rowspan="2">FLOAT4_E2M1</td>
      <td>Dynamic OCP</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Dynamic CX</td>
    </tr>
    <tr>
      <td>5</td>
      <td>FLOAT8_E4M3</td>
      <td>Symmetric</td>
      <td>Static</td>
      <td>No quantization coefficients are passed in; instead, P is directly cast to fp8 in the operator.</td>
      <td>        qDequantScaleOptional(not passed)<br>
        kDequantScaleOptional(not passed)<br>
        vDequantScaleOptional(not passed)<br>
        pQuantScaleOptional(not passed)
      </td>
      <td>FLOAT32</td>
    </tr>
  </tbody>
  </table>

Currently, quantType=0 is available; quantType=5 is optional on Ascend 950PR&950DT products. quantType=1-4 is currently not supported.

### Mask Description

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 160px">
  <col style="width: 140px">
  <col style="width: 655px">
  <col style="width: 200px">
  </colgroup>
  <thead>
    <tr>
        <th>maskType</th>
        <th>Meaning</th>
        <th>attentionMaskOptional</th>
        <th>winLeft/winRight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>No mask</td>
      <td>Not passed</td>
      <td>-1/-1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>causal mask</td>
      <td>attenMaskOptional not passed in (built-in causal); planned BOOL [2048,2048] lower triangle.</td>
      <td>-1/-1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>window mask</td>
      <td>Plan: BOOL [2048,2048] lower triangle, used together with winLeft/winRight.</td>
      <td>Actual number of tokens to look forward/backward included in the window.</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Union exactMask</td>
      <td>Pass INT32 exactMask, shape [kvHeads, G, U×blockShapeY]; bit i marks whether the i-th Q in the group can see that column.</td>
      <td>-1/-1</td>
    </tr>
    <tr>
      <td>4-5</td>
      <td>Various specialized masks</td>
      <td>Mask descriptions to be added later.</td>
      <td>-1/-1</td>
    </tr>
  </tbody>
  </table>

Supports maskType=0 (non-causal) and 1 (built-in causal); both require attenMaskOptional=nullptr and winLeft/winRight=-1. Ascend 950PR&amp;950DT products also support maskType=3 (Union), which requires INT32 exactMask (see the table above) with winLeft/winRight=-1; non-quantized Atlas A2/A3 products support Union as well. maskType 2/4/5 is not supported.

### Other Constraints

- Deterministic computation: aclnnGenericBlockSparseAttention defaults to a deterministic implementation.
- Before calling this API, run aclnnGenericBlockSparseAttentionMetadata to generate metadataOptional; the metadata must match the current inputs/attributes and must be regenerated for each call.
- Non-quantized FLOAT16/BFLOAT16 on Atlas A2/A3 and Ascend 950PR&amp;950DT supports any headDim from 1 to 256; `MXFP8` supports only D=192/256 under the grouped conditions; regular FP8 (quantType=5) on Ascend 950PR&amp;950DT supports only 128. The PA KV page blockSize is fixed at 128 and is independent of sparse blockShapeY.
- Ascend 950PR&amp;950DT non-quant FLOAT16/BFLOAT16: `maskType` supports 0/1/3; non-union uses `blockShape=[1,Y]` (`1≤Y≤128`), union uses `blockShape=[8,Y]` (currently only 8); `residualBlockMode` supports 0 or 1 (union supports only 0). Layout combinations match the Chinese product constraints (TND/BSND/BNSD with dense or PA_BBND/PA_BNBD).
- Atlas A2/A3 non-quant FLOAT16/BFLOAT16: `maskType` supports 0/1/3; non-union uses `blockShape=[1,Y]` (Y in {1,2,4,8,16,32,64,128}), union uses `blockShape=[8,Y]`; `residualBlockMode` supports 0 or 1 (union supports only 0). Union supports dense KV with the same layout (TND/BSND/BNSD) and paged KV (PA_BBND/PA_BNBD) with a page size of 128, requires INT32 exactMask, and requires the product of the GQA group size and blockShapeX to be no greater than 128.
- Regular FP8 on Ascend 950PR&amp;950DT: `headDim=128`, `layoutQ/layoutKv=TND/PA_BBND`, `maskType=1`, `residualBlockMode=0`, `blockShape=[1,Y]` (`1≤Y≤128`). Union FP8 additionally allows the union layouts/`blockShape=[8,Y]` (`1≤Y≤128`)/`residualBlockMode=0` and requires INT32 exactMask; for paged KV, `Y` must also divide the PA page size 128 (`Y∈{1,2,4,8,16,32,64,128}`).
- `MXFP8` (`quantType=6`) on Ascend 950PR&amp;950DT uses only the grouped path (D=192/256, `blockShape` `[1,4]` or `[1,8]`, no LSE). See the [MXFP8 quantization contract](./aclnnGenericBlockSparseAttention.md#mxfp8quanttype6) (Chinese) for full conditions.
- `softmaxPrecision`: On Ascend 950PR&amp;950DT, non-Union scenarios (maskType=0/1) allow non-quantized FLOAT16/BFLOAT16 with 0 or 1, regular FP8 (quantType=5) with only 1, and `MXFP8` with 0 or 1 (both use FP32 scores/Softmax; 0 is recommended and 1 is a compatibility alias); Union scenarios (maskType=3) support only 0 for both non-quantized and FP8. On Atlas A2/A3, non-Union scenarios allow FLOAT16 with 0 or 1 and BFLOAT16 with only 0, and Union scenarios (maskType=3) support only 0.
- Sparse index last-dimension capacity is 1 to `INT32_MAX` for non-quantized Atlas A2/A3, non-quantized FLOAT16/BFLOAT16 and `MXFP8` on Ascend 950PR&amp;950DT, and regular FP8 including FP8 Union.
- For TND: totalQBlocks is blocked by the storage length obtained by differencing cuSeqLengthsQ, i.e., $\sum_i \mathrm{ceilDiv}(qStorageLen_i, blockShapeX)$; both sparse blocking and QKV addressing use this storage length and are not re-split by seqused.
- When layoutKv is PA_BBND/PA_BNBD, sequsedKvOptional must be passed in, and cuSeqLengthsKvOptional must not be passed.
- When seqused and the corresponding cu prefix sum are passed in together (on the Q side, and on the KV side when layoutKv is TND): the core splitting/task space is accumulated based on the actual valid length (seqused) of each batch; the seqused element of each batch must be ≤ the corresponding cu storage length and must be exactly the same as that on the Metadata side.
- The data types of the input query, key, and value must be consistent.
- Under `PA_BBND`/`PA_BNBD`, only dim0 (physical page axis) of key/value can be non-contiguous; the elements within each page must be contiguous; stride0 must be ≥ blockSize×numKeyValueHeads×headDim and aligned to numKeyValueHeads×headDim. For `MXFP8`, ACLNN normalizes K/V to contiguous pages; direct GE calls must provide contiguous page strides.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_metadata.h"

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

uint16_t FloatToFp16(float f)
{
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 31) & 0x1u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = (bits >> 13) & 0x3ffu;
  if (exp <= 0) {
    return static_cast<uint16_t>(sign << 15);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>((sign << 15) | 0x7c00u);
  }
  return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | mant);
}

float Fp16ToFloat(uint16_t h)
{
  uint32_t sign = (h >> 15) & 0x1u;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t f;
  if (exp == 0) {
    f = (sign << 31) | (mant << 13);
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
  }
  float result;
  std::memcpy(&result, &f, sizeof(result));
  return result;
}

void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
  auto size = GetShapeSize(shape);
  std::vector<uint16_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size && i < 10; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Fp16ToFloat(resultData[i]));
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

std::vector<uint16_t> MakeFp16Data(int64_t size, float value)
{
  std::vector<uint16_t> data(static_cast<size_t>(size), FloatToFp16(value));
  return data;
}

}  // namespace

int main()
{
  // 1. (boilerplate) Device/stream initialization. Refer to the acl API manual.
  // Fill in deviceId based on your actual device.
  int32_t deviceId = 0;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Regular path smoke case: TND query + PA_BBND KV
  int64_t B = 1;
  int64_t S1 = 4;
  int64_t S2 = 256;
  int64_t N1 = 4;
  int64_t N2 = 1;
  int64_t D = 128;
  int64_t topK = 2;
  int64_t blockSize = 128;
  int64_t blockShapeX = 1;
  double scaleValue = 1.0 / sqrt(static_cast<double>(D));

  int64_t T = B * S1;
  int64_t maxBlocks = (S2 + blockSize - 1) / blockSize;
  int64_t totalQBlocks = T;  // blockShapeX == 1

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> qShape = {T, N1, D};
  std::vector<int64_t> kvShape = {maxBlocks, blockSize, N2, D};
  std::vector<int64_t> sparseIdxShape = {N2, totalQBlocks, topK};
  std::vector<int64_t> sparseCountShape = {N2, totalQBlocks};
  std::vector<int64_t> blockTableShape = {B, maxBlocks};
  std::vector<int64_t> cuSeqQShape = {B + 1};
  std::vector<int64_t> sequsedKvShape = {B};
  std::vector<int64_t> metadataShape = {1024};
  std::vector<int64_t> attnOutShape = {T, N1, D};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* sparseIdxDeviceAddr = nullptr;
  void* sparseCountDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* cuSeqQDeviceAddr = nullptr;
  void* sequsedKvDeviceAddr = nullptr;
  void* blockTableDeviceAddr = nullptr;
  void* attnOutDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* sparseIdx = nullptr;
  aclTensor* sparseCount = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* cuSeqQ = nullptr;
  aclTensor* sequsedKv = nullptr;
  aclTensor* blockTable = nullptr;
  aclTensor* attnOut = nullptr;

  int64_t qSize = GetShapeSize(qShape);
  int64_t kvSize = GetShapeSize(kvShape);
  int64_t sparseIdxSize = GetShapeSize(sparseIdxShape);
  int64_t sparseCountSize = GetShapeSize(sparseCountShape);
  int64_t blockTableSize = GetShapeSize(blockTableShape);
  int64_t attnOutSize = GetShapeSize(attnOutShape);

  std::vector<uint16_t> qHostData = MakeFp16Data(qSize, 1.0f);
  std::vector<uint16_t> kHostData = MakeFp16Data(kvSize, 1.0f);
  std::vector<uint16_t> vHostData = MakeFp16Data(kvSize, 1.0f);
  std::vector<int32_t> sparseIdxHostData(sparseIdxSize, -1);
  std::vector<int32_t> sparseCountHostData(sparseCountSize, 0);
  std::vector<int32_t> blockTableHostData(blockTableSize);
  std::iota(blockTableHostData.begin(), blockTableHostData.end(), 0);
  std::vector<int64_t> cuSeqQHostData = {0, S1};
  std::vector<int32_t> sequsedKvHostData = {static_cast<int32_t>(S2)};
  std::vector<int32_t> metadataHostData(1024, 0);
  std::vector<uint16_t> attnOutHostData = MakeFp16Data(attnOutSize, 0.0f);

  // Causal-like sparse window: keep up to topK trailing KV blocks per Q token.
  int64_t history = S2 - S1;
  for (int64_t qBlock = 0; qBlock < totalQBlocks; qBlock++) {
    int64_t visible = history + qBlock + 1;
    int64_t lastKvBlock = std::min(maxBlocks - 1, (visible - 1) / blockSize);
    int64_t count = std::min(topK, lastKvBlock + 1);
    int64_t start = lastKvBlock - count + 1;
    for (int64_t i = 0; i < count; i++) {
      sparseIdxHostData[qBlock * topK + i] = static_cast<int32_t>(start + i);
    }
    sparseCountHostData[qBlock] = static_cast<int32_t>(count);
  }

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, kvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseIdxHostData, sparseIdxShape, &sparseIdxDeviceAddr, aclDataType::ACL_INT32, &sparseIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sparseCountHostData, sparseCountShape, &sparseCountDeviceAddr, aclDataType::ACL_INT32,
                        &sparseCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqQHostData, cuSeqQShape, &cuSeqQDeviceAddr, aclDataType::ACL_INT64, &cuSeqQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sequsedKvHostData, sequsedKvShape, &sequsedKvDeviceAddr, aclDataType::ACL_INT32, &sequsedKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(blockTableHostData, blockTableShape, &blockTableDeviceAddr, aclDataType::ACL_INT32,
                        &blockTable);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_FLOAT16, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const int64_t blockShapeData[] = {blockShapeX, blockSize};
  aclIntArray* blockShape = aclCreateIntArray(blockShapeData, 2);
  CHECK_RET(blockShape != nullptr, LOG_PRINT("aclCreateIntArray failed\n"); return -1);

  char layoutQ[] = "TND";
  char layoutKv[] = "PA_BBND";

  // 3. Call Metadata first, and then call the main operator.
  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;
  ret = aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
      sparseIdx, sparseCount, cuSeqQ, nullptr, nullptr, sequsedKv, S1, S2, N1, N2, D, blockShape, layoutQ, layoutKv, 4, 1,
      0, 1, -1, -1, 0, 0, metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnGenericBlockSparseAttentionMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor,
                                                 stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttentionMetadata failed. ERROR: %d\n", ret);
            return ret);

  // 4. (boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret);
            return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnGenericBlockSparseAttentionGetWorkspaceSize(
      q, k, v, sparseIdx, sparseCount, metadata, nullptr, nullptr, nullptr, nullptr, nullptr, cuSeqQ, nullptr, nullptr,
      sequsedKv, blockTable, blockShape, layoutQ, layoutKv, 4, scaleValue, 1, 0, 0.0, 1, -1, -1, 0, 0, 0, attnOut, nullptr,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnGenericBlockSparseAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnGenericBlockSparseAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGenericBlockSparseAttention failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from device memory to host memory. Modify the code based on the API definition.
  PrintOutResult(attnOutShape, &attnOutDeviceAddr);

  // 6. Release aclTensor. Modify the code based on the API definition.
  aclDestroyIntArray(blockShape);
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(sparseIdx);
  aclDestroyTensor(sparseCount);
  aclDestroyTensor(metadata);
  aclDestroyTensor(cuSeqQ);
  aclDestroyTensor(sequsedKv);
  aclDestroyTensor(blockTable);
  aclDestroyTensor(attnOut);

  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(sparseIdxDeviceAddr);
  aclrtFree(sparseCountDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(cuSeqQDeviceAddr);
  aclrtFree(sequsedKvDeviceAddr);
  aclrtFree(blockTableDeviceAddr);
  aclrtFree(attnOutDeviceAddr);
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
