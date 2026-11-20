# aclnnFlashAttentionUnpaddingScoreGradV5

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:17:52.289Z -->

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

- Description: Implements self-attention computation using the FlashAttention algorithm in training scenarios. Adds the `sinkInOptional` optional input.

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The sinkInOptional parameter is not currently supported.

  <!-- end id7 -->

- Formula:

  The forward attention formula is:

  $$
  =Dropout(Softmax(Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask),keep\_prob)V
  $$

  For ease of expression, variables $S$ and $P$ are used to represent the formulas:

  $$
  =Mask(\frac{QK^T+pse}{\sqrt{d}}),atten\_mask
  $$

  $$
  =Dropout(Softmax(S),keep\_prob)
  $$

  $$
  =PV
  $$

  The backward propagation formula for attention is:

  $$
  V=P^TdY
  $$

  $$
  Q=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  K=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

The computation logic after adding **sink** is as follows, mainly modifying the relevant softmax_max and softmax_sum logic computation parts.

$$
S = Q @ K^{T}
$$

$$
m = max(sink, max(S))
$$

$$
Attention = \frac{e^{S - m} @ V}{\sum e^{S-m} + e^{sink - m}}
$$

$$
dSink = reduce(-P * dP * SimpleSoftmax(sink, x\_max, x\_sum))
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call the "aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize" API or the "aclnnFlashAttentionUnpaddingScoreGradV5GetMaxWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnFlashAttentionUnpaddingScoreGradV5" API to perform the computation.

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(
 const aclTensor   *query,
 const aclTensor   *queryRope,
 const aclTensor   *keyIn,
 const aclTensor   *keyInRope,
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
 int64_t            innerPrecise,
 int64_t            sparseMode,
 int64_t            pseType,
 char              *softmaxInLayout,
 const aclTensor   *dqOut,
 const aclTensor   *dqRopeOut,
 const aclTensor   *dkOut,
 const aclTensor   *dkRopeOut,
 const aclTensor   *dvOut,
 const aclTensor   *dpseOut,
 const aclTensor   *dsinkOut,
 uint64_t          *workspaceSize,
 aclOpExecutor    **executor);
```

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5GetMaxWorkspaceSize(
    const aclTensor *query,
    const aclTensor *queryRope,
    const aclTensor *keyIn,
    const aclTensor *keyInRope,
    const aclTensor *value,
    const aclTensor *dy,
    const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional,
    const aclTensor *sinkInOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double scaleValue,
    double keepProb,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,
    int64_t pseType,
    char *softmaxInLayout,
    const aclTensor *dqOut,
    const aclTensor *dqRopeOut,
    const aclTensor *dkOut,
    const aclTensor *dkRopeOut,
    const aclTensor *dvOut,
    const aclTensor *dpseOut,
    const aclTensor *dsinkOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1529px"><colgroup>
    <col style="width: 198px">
    <col style="width: 120px">
    <col style="width: 289px">
    <col style="width: 302px">
    <col style="width: 238px">
    <col style="width: 106px">
    <col style="width: 130px">
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
      </tr></thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>Input</td>
        <td>Q in the formula.</td>
        <td>Data type is consistent with keyIn/value.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
    <tr>
        <td>queryRope</td>
        <td>Optional input</td>
        <td>The rope part (rotary position embedding) of Q.</td>
        <td>The data type must be consistent with query.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyIn</td>
        <td>Input</td>
        <td>K in the formula.</td>
        <td>The data type must be consistent with query/value.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyInRope</td>
        <td>Optional input</td>
        <td>The rope part of K.</td>
        <td>The data type is consistent with that of keyIn.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>V in the formula.</td>
        <td>Data type is consistent with query/keyIn.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dy</td>
        <td>Input</td>
        <td>dY in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>pseShiftOptional</td>
        <td>Optional input</td>
        <td>pse in the formula.</td>
        <td>The data type must be consistent with the data type of query. This parameter must be used together with pseType.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[B,N,1024,Skv], [1,N,1024,Skv], [B,N], [N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>Optional input</td>
        <td>Dropout mask.</td>
        <td>Incompatible with rope. Pass nullptr if not used.</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>paddingMaskOptional</td>
        <td>Optional input.</td>
        <td>Reserved parameter, not used yet.</td>
        <td>Pass an empty tensor when calling.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>Optional input</td>
        <td>Represents the outer truncation use case: the start index of the query sequence of the current block in the global sequence.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>Optional input</td>
        <td>Represents the start index of the key and value sequence of the current block in the global sequence, in the unpadding scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Optional input</td>
        <td>The atten_mask in the formula.</td>
        <td>A value of 1 indicates that the bit is not involved in calculation, and 0 indicates that the bit is involved in calculation.</td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv] </td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxMaxOptional</td>
        <td>Optional input</td>
        <td>Intermediate output of the forward softmax.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[TN8], [NT8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOptional</td>
        <td>Optional input</td>
        <td>Intermediate output of forward softmax.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[TN8], [NT8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxInOptional</td>
        <td>Optional input</td>
        <td>Intermediate output of forward softmax.</td>
        <td>Reserved parameter, not used yet.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionInOptional</td>
        <td>Optional input</td>
        <td>Forward attention output.</td>
        <td>Consistent with the data type and shape of query.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
    <tr>
        <td>sinkInOptional</td>
        <td>Optional input</td>
        <td>sink in the formula.</td>
        <td>The length is headNum.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[headNum]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>Optional input</td>
        <td>N of each Batch in the prefix sparse use case.</td>
        <td>Pass nullptr if not used.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>Optional input</td>
        <td>Actual sequence length of Query.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>Optional input</td>
        <td>Actual Key/Value sequence length.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Optional input</td>
        <td>Scaling factor for scale.</td>
        <td>Generally set to D^-0.5.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>Optional input</td>
        <td>Ratio of 1s in dropMaskOptional.</td>
        <td>Generally set to 1.0.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Optional input</td>
        <td>Left boundary of the sliding window in sparse computation.</td>
        <td>If not specified, 2147483647 is recommended.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Optional input</td>
        <td>Right boundary of the sliding window for sparse computation.</td>
        <td>If not specified, 2147483647 is recommended.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Number of heads on a single device, that is, the N-axis length of Query.</td>
        <td>For overall constraints, see Constraints.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Data layout of input Q/K/V.</td>
        <td>TND is supported.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Optional input</td>
        <td>Controls internal calculation precision.</td>
        <td>Reserved parameter, not used yet.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Optional input</td>
        <td>Sparse mode.</td>
        <td>Supports values 0-8, excluding 5. 6 is not supported when rope is passed in.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>Optional input</td>
        <td>PSE type.</td>
        <td>Supports configuration values 0-3.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxInLayout</td>
        <td>Optional input</td>
        <td>Controls the actual data layout of softmaxMax and softmaxSum.</td>
        <td>Pass "same_as_input" when the shape is TND and the actual data layout needs to be NTD; pass "" when the actual data layout is TND.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dqOut</td>
        <td>Output</td>
        <td>dQ in the formula, the Query gradient.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dkOut</td>
        <td>Output</td>
        <td>dK in the formula, the Key gradient.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dvOut</td>
        <td>Output</td>
        <td>dV in the formula, the gradient of Value.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dpseOut</td>
        <td>Output</td>
        <td>d(pse) gradient.</td>
        <td>Reserved parameter, not used yet.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dsinkOut</td>
        <td>Output</td>
        <td>Gradient of dSink and d(sinkInOptional) in the formula.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[headNum]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be applied by the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator calculation process.</td>
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
  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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
      <td>The passed-in parameter is a required input, output, or required attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, sinkInOptional, dqOut, dkOut, dvOut, dsinkOut, and softmaxInLayout are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data formats of query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, sinkInOptional, dqOut, dkOut, dvOut, dsinkOut, and softmaxInLayout are not in the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionUnpaddingScoreGradV5

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
      <td>Workspace memory address applied by the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied by the Device, obtained by the first-phase API aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
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
  - aclnnFlashAttentionUnpaddingScoreGradV5 defaults to a non-deterministic implementation. Deterministic computation can be enabled by calling aclrtSetSysParamOpt.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- The B (batchsize) dimension of the inputs query, queryRope, key, keyRope, value, and dy must be equal.
- The inputLayout of query, key, value, and dy must be consistent.
- The inputLayout of query, queryRope, key, keyRope, value, and dy must be TND.
- The data type of query, key, value, and pseShiftOptional must be consistent.
- The shape of key and value must be consistent. When the d dimensions of query, key, and value are the same, the shape of query and dy must be consistent. Note: The current version only supports the case where the d dimensions of query, key, and value are the same.
- The N of query and the N of key/value may differ, but they must be proportional, that is, Nq/Nkv must be a non-zero integer, with Nq in the value range of 1 to 256.
- Constraints on data shape, taking TND of inputLayout as an example, where:

    - T(B*S): ranges from 1 to 1M.
    - B: ranges from 1 to 20000. When prefixOptional is used, B supports a maximum of 1K.
    - N: ranges from 1 to 256.
    - S: ranges from 1 to 1M.
    - D: ranges from 1 to 768.
    - KeepProb: value range: (0, 1].
- The data layout format of query, key, and value only supports TND, where T is the tightly packed data of the B and S axes (SeqLenQ and SeqLenKV of each batch). B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden size, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the smallest unit dimension of the hidden layer, satisfying D = H/N.
- In TND format, some trailing batches are supported to be not involved in calculation. In this case, pass the corresponding number of 0s at the end of actual_seq_qlen and actual_seq_kv_len. For example, if the actual S length is [2, 3, 4, 5, 6] and the last two batches are not involved in calculation, the passed actual_seq_qlen is [2, 5, 9, 0, 0].
- In TND use cases, each batch segment is still generated as [N, Sq_i, Skv_i] internally, but is uniformly flattened for storage and parameter passing. If the actual query length of the i-th batch segment is Sq_i and the actual key/value length is Skv_i, the number of PSE elements in this segment is N *Sq_i* Skv_i, and the total length pseTotalLen of the entire PSE is sum_i(N *Sq_i* Skv_i).
- pseShiftOptional: If Sq is greater than 1024, the Sq and Skv of each batch are of the same length, and sparseMode is 0, 2, or 3 in the lower triangular mask use case, alibi position encoding compression can be enabled. In this case, only the last 1024 rows of the original PSE need to be input, that is, alibi_compress = ori_pse[:, :, -1024:, :], to achieve memory optimization. Details are as follows:
  - When the parameter differs for each batch, the shape is BNHSkv (H = 1024).
  - When the parameter is the same for each batch, the shape is 1NHSkv (H = 1024).
  - If pseType is 2 or 3, the data type must be FLOAT32, and the supported shapes are [B,N] or [N].
  - If this parameter is not enabled, pass nullptr for pseShiftOptional and 1 for pseType.
- Meaning of each pseType value

  | pseType | Meaning | Note |
  | ----------- | --------------------------------- | ----------|
  | 0 | Externally passed pse: mul first, then add | - |
  | 1 | Externally passed pse: add first, then mul | Consistent with the implementation of [FlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad_en.md). |
  | 2 | Internally generated pse: mul first, then add | - |
  | 3 | Internally generated pse: mul first, then add, then sqrt | - |

- Constraints on sparseMode are as follows:
  - When the shapes of all attenMaskOptional are smaller than 2048 and identical, it is recommended to use the default mode to reduce memory usage.
  - When configured as 1, 2, or 3, the user-configured preTokens and nextTokens do not take effect.
  - When sparseMode is configured as 0 or 4, ensure that attenMaskOptional is consistent with the ranges of preTokens and nextTokens.
  - If not explicitly specified, it is recommended to pass 0.
  - For details about different sparse modes, see [sparseMode Introduction](../../../docs/en/context/sparse_mode_introduction.md).
  - When configured as 7, the optional input pseShiftOptional is not supported.
  - When configured as 8, the optional input pseShiftOptional is supported when q and kv of each sequence have equal lengths, and pse is generated globally. Outward slicing in the q direction is supported, which requires that q and kv of each sequence have equal lengths before slicing, and after slicing, actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0 (this is an experimental feature).
- For details about different data formats, see [Data Formats](../../../docs/en/context/data_format.md).
- In some scenarios, if the computation workload is too large, the operator execution may time out (aicore error with errorStr: timeout or trap error). In this case, axis splitting is recommended. Note: The computation workload here is affected by parameters such as B, S, N, and D. Larger values result in a heavier workload.
- The prefixOptional sparse computation only supports the compressed scenario, sparseMode=6. When Sq > Skv, the N value range of prefix is [0, Skv]; when Sq <= Skv, the N value range of prefix is [Skv-Sq, Skv].
- The actualSeqQLenOptional input supports a case where the S length of a batch is 0, but the optional input pseShiftOptional is not supported in this case.
- Constraints on the softmaxMax and softmaxSum parameters: The input format is fixed as [B, N, S, 8], except for the TND input format, which is [T, N, 8]. Note: T=B*S.
- The value of headNum must be consistent with the N value in the input query.
- When the data layout of softmaxSum and softmaxMax is NTD, softmaxInLayout must be "same_as_input".
- The dimension of sinkInOptional is 1, and its length must be the same as the headNum of query.

## Example

The sample code for the API call is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
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

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // Boilerplate for resource initialization.
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
                      aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct input and output. Customize the construction based on the API.
  std::vector<int64_t> qShape = {256, 1, 128};
  std::vector<int64_t> kShape = {256, 1, 128};
  std::vector<int64_t> vShape = {256, 1, 128};
  std::vector<int64_t> dxShape = {256, 1, 128};
  std::vector<int64_t> attenmaskShape = {256, 256};
  std::vector<int64_t> softmaxMaxShape = {256, 1, 8};
  std::vector<int64_t> softmaxSumShape = {256, 1, 8};
  std::vector<int64_t> attentionInShape = {256, 1, 128};
  std::vector<int64_t> sinkInOptionalShape = {1};
  std::vector<int64_t> dqShape = {256, 1, 128};
  std::vector<int64_t> dkShape = {256, 1, 128};
  std::vector<int64_t> dvShape = {256, 1, 128};
  std::vector<int64_t> dsinkShape = {1};

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
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxIn = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* sinkInOptional = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;
  aclTensor* dsink = nullptr;

  std::vector<float> qHostData(32768, 1);
  std::vector<float> kHostData(32768, 1);
  std::vector<float> vHostData(32768, 1);
  std::vector<float> dxHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);
  std::vector<float> attentionInHostData(32768, 1);
  std::vector<float> sinkInOptionalHostData(1, 0);
  std::vector<float> dqHostData(32768, 0);
  std::vector<float> dkHostData(32768, 0);
  std::vector<float> dvHostData(32768, 0);
  std::vector<float> dsinkHostData(1, 0);
  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT16, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinkInOptionalHostData, sinkInOptionalShape, &sinkInOptionalDeviceAddr, aclDataType::ACL_FLOAT, &sinkInOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dsinkHostData, dsinkShape, &dsinkDeviceAddr, aclDataType::ACL_FLOAT, &dsink);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::vector<int64_t> prefixOp = {0};
  aclIntArray* prefix = aclCreateIntArray(prefixOp.data(), 1);
  std::vector<int64_t>  acSeqQLenOp = {256};
  std::vector<int64_t>  acSeqKvLenOp = {256};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  char softmaxInLayoutArr[] = "same_as_input";
  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionUnpaddingScoreGradV5.
  ret = aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(q, nullptr, k, nullptr, v, dx, pse, dropMask, padding,
              attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, sinkInOptional, prefix, acSeqQLen, acSeqKvLen,
              qStartIdx, kvStartIdx, scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise, sparseMode,
              pseType, softmaxInLayoutArr, dq, nullptr, dk, nullptr, dv, dpse, dsink, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory using the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionUnpaddingScoreGradV5.
  ret = aclnnFlashAttentionUnpaddingScoreGradV5(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV5 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait for the task execution to complete synchronously.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code according to the specific API definition.
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);
  PrintOutResult(dsinkShape, &dsinkDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
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
  // 7. Release device resources.
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
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
