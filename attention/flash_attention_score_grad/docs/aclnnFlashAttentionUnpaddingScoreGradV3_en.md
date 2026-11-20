# aclnnFlashAttentionUnpaddingScoreGradV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:13:38.351Z -->

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

- Description: Computes the backward output of attention in training use cases, that is, the backward computation of [aclnnFlashAttentionVarLenScoreV3](../../flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV3_en.md). Compared with [aclnnFlashAttentionUnpaddingScoreGradV2](./aclnnFlashAttentionUnpaddingScoreGradV2_en.md), this API adds the queryRope, keyRope, dqRope, and dkRope parameters.

- Formula:

  The forward formula of attention is as follows:

  $$
  Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
  $$

  Where:

  $$
  Q=[query, queryRope], K=[key, keyRope]
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
  dqRope=\frac{((dS)*kRope)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

  $$
  dkRope=\frac{((dS)^T*qRope)}{\sqrt{d}}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnFlashAttentionUnpaddingScoreGradV3" API to execute the computation.

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize(
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
  const aclTensor   *dqOut,
  const aclTensor   *dqRopeOut,
  const aclTensor   *dkOut,
  const aclTensor   *dkRopeOut,
  const aclTensor   *dvOut,
  const aclTensor   *dpseOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize

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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>Input</td>
        <td>Q in the formula.</td>
        <td>The data type is the same as keyIn/value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryRope</td>
        <td>Optional input</td>
        <td>The rope part (rotary position embedding) of Q.</td>
        <td>The data type is the same as that of query.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyIn</td>
        <td>Input</td>
        <td>K in the formula.</td>
        <td>The data type is the same as that of query/value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyInRope</td>
        <td>Optional input</td>
        <td>Rope part of K.</td>
        <td>The data type is the same as that of keyIn.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>V in the formula.</td>
        <td>Data type: same as query/keyIn.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dy</td>
        <td>Input</td>
        <td>dY in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>pseShiftOptional</td>
        <td>Optional input</td>
        <td>pse in the formula, position encoding.</td>
        <td>Must be nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>Optional input</td>
        <td>Dropout mask.</td>
        <td>Must be nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>paddingMaskOptional</td>
        <td>Optional input</td>
        <td>Reserved parameter, not used yet.</td>
        <td>Pass an empty value in the call.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>Optional input</td>
        <td>In the external split scenario, start index of the current Query chunk in the global sequence.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>Optional input</td>
        <td>In the non-aligned scenario, the start index of the current chunk of Key/Value in the global sequence.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Optional input</td>
        <td>atten_mask in the formula.</td>
        <td>A value of 1 indicates that the position does not participate in computation, and 0 indicates that it does.</td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv] </td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxMaxOptional</td>
        <td>Optional input</td>
        <td>Forward softmax intermediate output.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[N,T,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOptional</td>
        <td>Optional input</td>
        <td>Forward softmax intermediate output.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[N,T,8]</td>
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
        <td>Same data type and shape as query.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>Optional input</td>
        <td>N value of each batch in prefix sparse use cases.</td>
        <td>Can be nullptr.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>Optional input</td>
        <td>Actual Query sequence length.</td>
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
        <td>Scaling factor of scale.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>Optional input</td>
        <td>Ratio of 1s in dropMaskOptional.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Optional input</td>
        <td>Left boundary of the sliding window in sparse computation.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Optional input</td>
        <td>Right boundary of the sliding window in sparse computation.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Number of heads per device, that is, the length of the N dimension of Query.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Layout of the input Q/K/V data.</td>
        <td>Supports TND.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Optional input</td>
        <td>Controls the internal computation precision.</td>
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
        <td>Supported values: 0-8. Values 5 and 6 are not supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>Optional input</td>
        <td>Data type: INT64.</td>
        <td>Only the value 1 is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dqOut</td>
        <td>Output</td>
        <td>dQ in the formula. Query gradient.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dqRopeOut</td>
        <td>Output</td>
        <td>Rope gradient of Q.</td>
        <td>-</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dkOut</td>
        <td>Output</td>
        <td>dK in the formula, the gradient of Key.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dkRopeOut</td>
        <td>Output</td>
        <td>Rope gradient of K.</td>
        <td>-</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dvOut</td>
        <td>Output</td>
        <td>dV in the formula, the gradient of Value.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
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
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size required on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation flow.</td>
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
      <th>Error Codes</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The parameter passed in is a required input, output, or required attribute, and it is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of query, queryRope, keyIn, keyInRope, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, dqOut, dkOut, or dvOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data format of query, queryRope, keyIn, keyInRope, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, dqOut, dkOut, or dvOut is not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionUnpaddingScoreGradV3

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
        <td>Workspace memory address allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace memory size allocated on the device. Obtained by the first-phase API aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, containing the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
      </tr>
    </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnFlashAttentionUnpaddingScoreGradV3 defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.
- When this API is used with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages match.
- The B (batchsize) dimension of the inputs query, queryRope, key, keyRope, value, and dy must be equal.
- The inputLayout of query, queryRope, key, keyRope, value, and dy must be TND.
- When the d dimensions of query, key, and value are the same, the shapes of query and dy must be consistent.
- The d dimensions of query and key must be the same, and d must be an integer multiple of 8.
- The d dimensions of queryRope and keyRope must be the same, and d must be an integer multiple of 8 and must be less than or equal to the d of query and key.
- The N of query and dy and the N of key and value can be different, but they must be in a proportional relationship, that is, Nq/Nkv must be a non-zero integer, with Nq in the value range of 1 to 256.
- Constraints on data shape, taking TND of inputLayout as an example, where:
    - T: The value range is 1 to 1M.
    - N: ranges from 1 to 256.
    - D: ranges from 1 to 768.
    - KeepProb: value range is 1.
- The data layout format of query, key, and value only supports TND, where T is the tightly packed data of the B and S axes (SeqLenQ and SeqLenKV of each batch). B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden size, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the smallest unit dimension of the hidden layer, satisfying D = H/N.
- In TND format, some trailing batches can be excluded from computation. In this case, pass the corresponding number of 0s at the end of actual_seq_qlen and actual_seq_kv_len. For example, if the actual S length is [2, 3, 4, 5, 6] and the last two batches are excluded from computation, the passed actual_seq_qlen is [2, 5, 9, 0, 0].
- In TND use cases, each batch segment is still generated as [N, Sq_i, Skv_i] internally, but is uniformly flattened for storage and parameter passing. If the actual query length of the i-th batch segment is Sq_i and the actual key/value length is Skv_i, the number of PSE elements in this segment is N *Sq_i* Skv_i, and the total length pseTotalLen of the entire PSE is sum_i(N *Sq_i* Skv_i).
- Constraints on sparseMode are as follows:
  - When the shapes of all attenMaskOptional are smaller than 2048 and identical, it is recommended to use the default mode to reduce memory usage.
  - When configured as 1, 2, or 3, the user-configured preTokens and nextTokens do not take effect.
  - When configured as 0 or 4, ensure that attenMaskOptional is consistent with the ranges of preTokens and nextTokens.
  - If not explicitly specified, it is recommended to pass 0.
  - For details about different sparse modes, see [sparseMode Introduction](../../../docs/en/context/sparse_mode_introduction.md).
  - When configured as 7, the optional input pseShiftOptional is not supported.
  - When configured as 8, the optional input pseShiftOptional is supported when q and kv of each sequence have equal lengths, and pse is generated globally. Outward slicing in the q direction is supported, which requires that q and kv of each sequence have equal lengths before slicing, and after slicing, actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0 (this is an experimental feature).
- In some scenarios, if the computation workload is too large, the operator execution may time out (aicore error with errorStr: timeout or trap error). In this case, axis splitting is recommended. Note: The computation workload here is affected by parameters such as B, S, N, and D. Larger values result in a heavier workload.
- actualSeqQLenOptional input supports an S length of 0 in a batch, but the optional input pseShiftOptional is not supported in this case. The length of actualSeqQLenOptional ranges from 1 to 2K. When the prefixOptional input is present, its maximum length is 1K.
- Constraints on the softmaxMax and softmaxSum parameters: The input format is fixed as [T, N, 8], where T = B * S.
- The value of headNum must be consistent with the N value in the input query.
- pseType must be 1.
- pseShiftOptional must be empty.
- dropMaskOptional must be empty.
- attenMaskOptional cannot be empty.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```C++
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

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate for resource initialization
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
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> qShape = {256, 1, 128};
  std::vector<int64_t> qRopeShape = {256, 1, 64};
  std::vector<int64_t> kShape = {256, 1, 128};
  std::vector<int64_t> kRopeShape = {256, 1, 64};
  std::vector<int64_t> vShape = {256, 1, 128};
  std::vector<int64_t> dxShape = {256, 1, 128};
  std::vector<int64_t> attenmaskShape = {256, 256};
  std::vector<int64_t> softmaxMaxShape = {256, 1, 8};
  std::vector<int64_t> softmaxSumShape = {256, 1, 8};
  std::vector<int64_t> attentionInShape = {256, 1, 128};

  std::vector<int64_t> dqShape = {256, 1, 128};
  std::vector<int64_t> dqRopeShape = {256, 1, 64};
  std::vector<int64_t> dkShape = {256, 1, 128};
  std::vector<int64_t> dkRopeShape = {256, 1, 64};
  std::vector<int64_t> dvShape = {256, 1, 128};

  void* qDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dqRopeDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dkRopeDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* k = nullptr;
  aclTensor* kRope = nullptr;
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
  aclTensor* dq = nullptr;
  aclTensor* dqRope = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dkRope = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;

  std::vector<float> qHostData(32768, 1);
  std::vector<float> qRopeHostData(16384, 1);
  std::vector<float> kHostData(32768, 1);
  std::vector<float> kRopeHostData(16384, 1);
  std::vector<float> vHostData(32768, 1);
  std::vector<float> dxHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);
  std::vector<float> attentionInHostData(32768, 1);
  std::vector<float> dqHostData(32768, 0);
  std::vector<float> dqRopeHostData(16384, 0);
  std::vector<float> dkHostData(32768, 0);
  std::vector<float> dkRopeHostData(16384, 0);
  std::vector<float> dvHostData(32768, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_BF16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_BF16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_BF16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_BF16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_BF16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_BF16, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_BF16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_BF16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqRopeHostData, dqRopeShape, &dqRopeDeviceAddr, aclDataType::ACL_BF16, &dqRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_BF16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkRopeHostData, dkRopeShape, &dkRopeDeviceAddr, aclDataType::ACL_BF16, &dkRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_BF16, &dv);
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
  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. Call the CANN operator library API. Replace the name with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionUnpaddingScoreGradV3
  ret = aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize(q, qRope, k, kRope, v, dx, pse, dropMask, padding,
            attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
            scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise, sparseMode, pseType,
            dq, dqRope, dk, dkRope, dv, dpse, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory according to the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionUnpaddingScoreGradV3
  ret = aclnnFlashAttentionUnpaddingScoreGradV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV3 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, and copy the results on the device memory to the host. Modify this part based on the API definition.
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dqRopeShape, &dqRopeDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dkRopeShape, &dkRopeDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify this part based on the API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(qRope);
  aclDestroyTensor(k);
  aclDestroyTensor(kRope);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dq);
  aclDestroyTensor(dqRope);
  aclDestroyTensor(dk);
  aclDestroyTensor(dkRope);
  aclDestroyTensor(dv);

  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dqRopeDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dkRopeDeviceAddr);
  aclrtFree(dvDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
