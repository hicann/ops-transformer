# aclnnFlashAttentionScoreGradV4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:12:25.934Z -->

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

- Description: Computes the backward output of attention in the training task, that is, the backward computation of [FlashAttentionScoreV4](../../flash_attention_score/docs/aclnnFlashAttentionScoreV4_en.md). **The query, key, and value parameters of this API support multiple sequences of equal or unequal lengths**
  - **This API combines the [FlashAttentionScoreGradV2](./aclnnFlashAttentionScoreGradV2_en.md) API and the [FlashAttentionUnpaddingScoreGradV2](./aclnnFlashAttentionUnpaddingScoreGradV2_en.md) API, and adjusts the Dropout functionality**:

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>: When keepProb is less than 1.0, if no external DropoutMask is passed in, the newly added parameter is used to generate the DropoutMask; if an external DropoutMask is passed in, the external DropoutMask is used.
    <!-- end id7 -->

- Formula:

  - When pseType is 1, the formula is the same as that of [FlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad_en.md).
  - For other values of pseType, the formula is as follows:

  $$
  Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
  $$

    For ease of expression, variables $S$ and $P$ are used to represent the formulas:

  $$
  S=Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask
  $$

  $$
  P=Dropout(Softmax(S),keep\_prob)
  $$

  $$
  Y=PV
  $$

    The backward propagation formula for attention is:

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

  The computation logic after adding **sink** is as follows, mainly modifying the relevant softmax_max and softmax_sum logic computation parts.

  $$
  m = max(sink, max(S))
  $$

  $$
  Attention = \frac{e^{S - m} @ V}{\sum e^{S-m} + S^{sink - m}}
  $$

  $$
  dSink = reduce(-Softmax(S) * dP * SimpleSoftmax(sink, x\_max, x\_sum))
  $$

    **NOTE:**
    The data layout formats of query, keyIn, and value can be interpreted in multiple dimensions. T (Total S Length) indicates the total length of S of all batches. B (Batch) indicates the input batch size. S (Seq-Length) indicates the input sequence length. H (Head-Size) indicates the hidden layer size. N (Head-Num) indicates the number of heads. d (Head-Dim) indicates the smallest unit size of the hidden layer, where d=H/N.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnFlashAttentionScoreGradV4GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnFlashAttentionScoreGradV4" API to perform the computation.

```c++
aclnnStatus aclnnFlashAttentionScoreGradV4GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *keyIn,
  const aclTensor   *value,
  const aclTensor   *dy,
  const aclTensor   *pseShiftOptional,
  const aclTensor   *dropMaskOptional,
  const aclTensor   *paddingMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *softmaxMaxOptional,
  const aclTensor   *softmaxSumOptional,
  const aclTensor   *softmaxInOptional,
  const aclTensor   *attentionInOptional,
  const aclTensor   *sinkInOptional,
  const aclTensor   *queryRopeOptional,
  const aclTensor   *keyRopeOptional,
  const aclTensor   *dScaleQOptional,
  const aclTensor   *dScaleKOptional,
  const aclTensor   *dScaleVOptional,
  const aclTensor   *dScaleDyOptional,
  const aclTensor   *dScaleOOptional,
  const aclIntArray *prefixOptional,
  const aclIntArray *actualSeqQLenOptional,
  const aclIntArray *actualSeqKvLenOptional,
  const aclIntArray *qStartIdxOptional,
  const aclIntArray *kvStartIdxOptional,
  double             scaleValue,
  double             keepProb,
  int64_t            preTokens,
  int64_t            nextTokens,
  int64_t            headNum,
  char              *inputLayout,
  char              *softmaxInLayout,
  int64_t            innerPrecise,
  int64_t            sparseMode,
  int64_t            pseType,
  int64_t            seed,
  int64_t            offset,
  int64_t            outDtype,
  const aclTensor   *dqOut,
  const aclTensor   *dkOut,
  const aclTensor   *dvOut,
  const aclTensor   *dqRopeOut,
  const aclTensor   *dkRopeOut,
  const aclTensor   *dpseOut,
  const aclTensor   *dsinkOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)`
```

```c++
aclnnStatus aclnnFlashAttentionScoreGradV4(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnFlashAttentionScoreGradV4GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1565px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
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
      <td>Q in the formula.</td>
      <td>The data type is consistent with keyIn/value.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>queryRopeOptional</td>
      <td>Input</td>
      <td>The rope part of input Q in the formula.</td>
      <td>Data type consistent with query.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyIn</td>
      <td>Input</td>
      <td>K in the formula.</td>
      <td>Data type consistent with query/value.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyRopeOptional</td>
      <td>Input</td>
      <td>Rope part of the input K in the formula.</td>
      <td>Data type is consistent with keyIn.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>V in the formula.</td>
      <td>The data type is consistent with query/keyIn.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>dY in the formula.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pseShiftOptional</td>
      <td>Optional input</td>
      <td>pse in the formula, indicating positional encoding.</td>
      <td>Data type is consistent with query/keyIn/value.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv], [B,N,1,Skv], [1,N,Sq,Skv], [B,N,1024,Skv], [1,N,1024,Skv]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dropMaskOptional</td>
      <td>Optional input</td>
      <td>Dropout in the formula.</td>
      <td>Supported: [B,N,S,S]</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddingMaskOptional</td>
      <td>Optional input</td>
      <td>Reserved parameter, not used yet.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qStartIdxOptional</td>
      <td>Optional input</td>
      <td>Represents the long-sequence splitting use case. It is the start index of the current Q chunk's sequence in the global sequence.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kvStartIdxOptional</td>
      <td>Optional input</td>
      <td>Start index of the current chunked KV sequence in the global sequence, indicating the external splitting use case.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Optional input</td>
      <td>atten_mask in the formula.</td>
      <td>        <ul>
          <li>The value 1 indicates that the bit is not involved in calculation, and 0 indicates that the bit is involved in calculation.</li>
        </ul>
      </td>
      <td>BOOL, UINT8</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxMaxOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of the attention direction calculation.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[B,N,Sq,8],[N,T,8],[T,N,8]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSumOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of the attention forward calculation.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[B,N,Sq,8],[N,T,8],[T,N,8]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxInOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of the attention forward direction calculation. Reserved parameter, not used yet.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[B,N,Sq,8]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attentionInOptional</td>
      <td>Optional input</td>
      <td>Final output of the attention forward computation.</td>
      <td>Data type and shape are consistent with dy.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sinkInOptional</td>
      <td>Optional input</td>
      <td>The sink in the formula.</td>
      <td>The length is headNum.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[headNum]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleQOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the query input.</td>
      <td>Not used for now</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleKOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the key input.</td>
      <td>Not used for now.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleVOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the value input.</td>
      <td>Not used for now.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleDyOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the dy input.</td>
      <td>not used for now</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dScaleOOptional</td>
      <td>Optional input</td>
      <td>Dequantization parameter of the attentionInOptional input.</td>
      <td>not used for now</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>prefixOptional</td>
      <td>Optional input</td>
      <td>N per Batch in the prefix sparse use case.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqQLenOptional</td>
      <td>Input</td>
      <td>Indicates the query sequence length of each batch.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqKvLenOptional</td>
      <td>Input</td>
      <td>Indicates the kv sequence length of each batch.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>The scaling factor scale in the formula.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepProb</td>
      <td>Input</td>
      <td>The ratio of 1s in dropMask.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>Input</td>
      <td>Left boundary of the sparse computation window.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>Input</td>
      <td>Right boundary of the sparse computation window.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNum</td>
      <td>Input</td>
      <td>Number of heads per device, corresponding to the N dimension of query.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputLayout</td>
      <td>Input</td>
      <td>Data layout of query/key/value.</td>
      <td>Supports BSH, SBH, BSND, BNSD, and TND.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxInLayout</td>
      <td>Input</td>
      <td>Controls the actual data layout of softmaxMax and softmaxSum.</td>
      <td>Pass "same_as_input" if the shape is TND but the actual data layout needs to be NTD. Pass "" for TND.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise</td>
      <td>Input</td>
      <td>Reserved parameter, not used yet.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Sparse mode.</td>
      <td>Supported values: 0-8.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pseType</td>
      <td>Input</td>
      <td>Integer on the host.</td>
      <td>Supported values: 0-3.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Input</td>
      <td>If keepProbOptional is less than 1.0, DropoutMask is generated based on seedOptional and offsetOptional.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Integer on the host.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outDtype</td>
      <td>Input</td>
      <td>A value of 0 indicates that outputs such as dqOut are FLOAT16, and a value of 1 indicates BFLOAT16.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>Output</td>
      <td>dQ in the formula. Gradient of query.</td>
      <td>Data type is consistent with that of query.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dqRopeOut</td>
      <td>Output</td>
      <td>Indicates the gradient of queryRope.</td>
      <td>Consistent with dqOut</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dkOut</td>
      <td>Output</td>
      <td>dK in the formula, the gradient of keyIn.</td>
      <td>Data type: consistent with keyIn.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dkRopeOut</td>
      <td>Output</td>
      <td>dkRope in the formula, indicating the gradient of keyInRope.</td>
      <td>Consistent with dkOut</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dvOut</td>
      <td>Output</td>
      <td>dV in the formula, the gradient of value.</td>
      <td>Consistent with value</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH], [TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dpseOut</td>
      <td>Output</td>
      <td>Gradient of d(pse).</td>
      <td>Not used yet. The input shape must be consistent with pse_shape.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>0, 4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dsinkOut</td>
      <td>Output</td>
      <td>Gradient of dSink and d(sinkInOptional) in the formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the operator executor, which contains the computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
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
      <td>The passed parameter, which is a required input, output, or required attribute, is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The data type of query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, dqOut, dkOut, dvOut, or softmaxInLayout is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionScoreGradV4

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Address of the workspace memory requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory requested on the device. The value is obtained by the first-phase API aclnnFlashAttentionScoreGradV4GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator calculation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints<a name="1"></a>

- deterministic computation:
  - aclnnFlashAttentionScoreGradV4 defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.
- Constraints on the inputs query, key, value, and dy are as follows:
  - B: The batchsize must be equal.
  - The inputLayout must be consistent.
  - D: For Head-Dim, the D of query and key must be equal, the D of value and dy must be equal, and the D of query and key must be greater than or equal to the D of value and dy.
- The N of query and dy and the N of key and value can be different, but they must be in a proportional relationship, that is, Nq/Nkv must be a non-zero integer, with Nq in the value range of 1 to 256.
- Constraints on data shape, taking TND of inputLayout as an example, where:

    - B: value range is 1 to 2K. When prefixOptional is present, B supports a maximum of 1K.
    - N: ranges from 1 to 256.
    - S: ranges from 1 to 1M.
    - D: ranges from 1 to 768.
    - KeepProb: value range is (0, 1].
- In some scenarios, if the computation workload is too large, the operator execution may time out (aicore error with errorStr: timeout or trap error). In this case, axis splitting is recommended. Note: The computation workload here is affected by parameters such as B, S, N, and D. Larger values result in a heavier workload.
- The prefixOptional sparse computation supports only the compression scenario, sparseModeOptional=6. When Sq > Skv, the N value range of prefix is [0, Skv]; when Sq <= Skv, the N value range of prefix is [Skv-Sq, Skv]. When sparseModeOptional=5, prefix N > Skv, or prefixOptional is not passed, full computation is performed. sparseModeOptional=6 requires that prefixOptional must be passed.
- When sparseMode=7, the optional input pseShiftOptional is not supported.
- When sparseMode=8, the optional input pseShiftOptional is supported when q and kv of each sequence have equal lengths, and pse is generated globally. Outer splitting in the q direction is supported, which requires that q and kv of each sequence have equal lengths before outer splitting, and after outer splitting, actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0 (this feature is experimental).
- The actualSeqQLenOptional input supports a case where the S length of a batch is 0, but the optional input pseShiftOptional is not supported in this case.
- Constraints on the softmaxMax and softmaxSum parameters: The input format is fixed as [B, N, S, 8], except for the TND input format, which is [N, T, 8]. Note: T = B * S.
- The value of headNum must be consistent with the N value in the input query.

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:

    - When queryRopeOptional and keyRopeOptional are not empty, the per-head D of query and keyIn must be 128, the per-head D of value and dy must be 128, and the per-head D of queryRopeOptional and keyRopeOptional must be 64.
    - seedOptional and offsetOptional take effect only when keepProbOptional is less than 1.0; otherwise, they do not take effect.
    - When keepProbOptional is less than 1.0, if dropMaskOptional is not nullptr, the input dropMask is used; otherwise, the dropMask generated from seed and offset is used.
<!-- end id8 -->
- In TND format, trailing batches that do not participate in computation are supported. In this case, pass the corresponding number of zeros at the end of actual_seq_q_len and actual_seq_kv_len. Assume the actual S length is [2, 3, 4, 5, 6]. If you want the last two batches to not participate in computation, pass actual_seq_q_len as [2, 3, 4, 0, 0]. In this case, if prefixOptional needs to be passed, its trailing part must also contain the same number of zeros, for example, [1, 1, 1, 0, 0].
- When sinkInOptional is not None, input tensors such as query and key only support two data types: float16 and bfloat16. The dimension of sinkInOptional is 1, and its length must be the same as the headnum of query.
- When the data layout of softmaxSum and softmaxMax is TND, softmaxInLayout must be "same_as_input".

## Example

The sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate. Initialize AscendCL.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the list of external AscendCL APIs.
  // Set deviceId based on your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction according to the API.
  int64_t B = 1;
  int64_t N1 = 1;
  int64_t N2 = 1;
  int64_t S1 = 128;
  int64_t S2 = 128;
  int64_t D = 128;

  int64_t H1 = N1 * D;
  int64_t H2 = N2 * D;

  int64_t q_size = S1 * B * H1;
  int64_t kv_size = S2 * B * H2;
  int64_t atten_mask_size = S1 * S2;
  int64_t softmax_size = B * N1 * S1 * 8;

  std::vector<int64_t> qShape = {S1, B, H1};
  std::vector<int64_t> kShape = {S2, B, H2};
  std::vector<int64_t> vShape = {S2, B, H2};
  std::vector<int64_t> dxShape = {S1, B, H1};
  std::vector<int64_t> attenmaskShape = {S1, S2};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 8};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 8};
  std::vector<int64_t> attentionInShape = {S1, B, H1};
  std::vector<int64_t> sinkInOptionalShape = {N1};
  std::vector<int64_t> dqShape = {S1, B, H1};
  std::vector<int64_t> dkShape = {S2, B, H2};
  std::vector<int64_t> dvShape = {S2, B, H2};
  std::vector<int64_t> dsinkShape = {N1};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* sinkInOptionalDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;
  void* dsinkDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* queryRope = nullptr;
  aclTensor* keyRope = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* dScaleDy = nullptr;
  aclTensor* dScaleO = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxIn = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* sinkInOptional = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;
  aclTensor* dqRope = nullptr;
  aclTensor* dkRope = nullptr;
  aclTensor* dsink = nullptr;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 1.0f); // Normal distribution with a mean of 0 and a standard deviation of 1

  std::vector<float> qHostData(q_size);
  for (auto& val : qHostData) {
      val = dist(gen);
  }

  std::vector<float> kHostData(kv_size);
  for (auto& val : kHostData) {
      val = dist(gen);
  }

  std::vector<float> vHostData(kv_size);
  for (auto& val : vHostData) {
      val = dist(gen);
  }

  std::vector<float> dxHostData(q_size);
  for (auto& val : dxHostData) {
      val = dist(gen);
  }

  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);
  std::vector<float> attentionInHostData(q_size, 255.0);
  std::vector<float> sinkInOptionalHostData(N1, 0);
  std::vector<float> dqHostData(q_size, 2.0);
  std::vector<float> dkHostData(kv_size, 2.0);
  std::vector<float> dvHostData(kv_size, 3.0);
  std::vector<float> dsinkHostData(N1, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT, &attentionIn);
  ret = CreateAclTensor(sinkInOptionalHostData, sinkInOptionalShape, &sinkInOptionalDeviceAddr, aclDataType::ACL_FLOAT, &sinkInOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dsinkHostData, dsinkShape, &dsinkDeviceAddr, aclDataType::ACL_FLOAT, &dsink);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double scaleValue = 1.0/sqrt(128);
  double keepProb = 1.0;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  int64_t outDtype = 1;
  int64_t seed = 0;
  int64_t offset = 0;
  char inputLayOut[5] = {'S', 'B', 'H', 0};
  char softmaxInLayoutArr[] = "";

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionScoreGradV4
  ret = aclnnFlashAttentionScoreGradV4GetWorkspaceSize(q, k, v, dx, pse, dropMask, padding,
            attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, nullptr, queryRope, keyRope, dScaleQ, dScaleK, dScaleV,
            dScaleDy, dScaleO, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue, keepProb,
            preTokens, nextTokens, headNum, inputLayOut, softmaxInLayoutArr, innerPrecise, sparseMode, pseType, seed, offset, outDtype,
            dq,dk,dv,dqRope,dkRope,dpse, nullptr, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionScoreGradV4
  ret = aclnnFlashAttentionScoreGradV4(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV4 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to the host. Modify it according to the definition of the specific API.
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify it according to the definition of the specific API.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(sinkInOptional);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);
  aclDestroyTensor(dsink);

  // 7. Release device resources
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(sinkInOptionalDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  aclrtFree(dsinkDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
