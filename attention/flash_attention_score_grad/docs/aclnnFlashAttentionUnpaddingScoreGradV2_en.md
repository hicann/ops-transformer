# aclnnFlashAttentionUnpaddingScoreGradV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:10:18.481Z -->

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

- Description: Computes the backward output of attention during training, that is, the backward computation of [aclnnFlashAttentionVarLenScoreV2](../../flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV2_en.md). **Compared with [aclnnFlashAttentionUnpaddingScoreGrad](./aclnnFlashAttentionUnpaddingScoreGrad_en.md), this API adds the pseType parameter**:

  - When pseType=1, the implementation is the same as [aclnnFlashAttentionUnpaddingScoreGrad](./aclnnFlashAttentionUnpaddingScoreGrad_en.md).
  - When pseType is set to other values, mul is performed first, followed by add.

- Formula:

  Given the forward computation formula of the attention mechanism (when pseType=1, the formula is the same as that of [aclnnFlashAttentionUnpaddingScoreGrad](./aclnnFlashAttentionUnpaddingScoreGrad_en.md); when pseType takes other values, the formula is as follows):

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

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the computation process, and then call the "aclnnFlashAttentionUnpaddingScoreGradV2" API to perform the computation.

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(
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
  const aclTensor   *dkOut,
  const aclTensor   *dvOut,
  const aclTensor   *dpseOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize

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
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td>Q in the formula.</td>
      <td>The data type must be consistent with keyIn/value.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyIn</td>
      <td>Input</td>
      <td>K in the formula.</td>
      <td>Data type is consistent with query/value.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>V in the formula.</td>
      <td>Data type is consistent with query/keyIn.</td>
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
      <td>pse in the formula.</td>
      <td>The data type is consistent with the data type of query. This parameter must be used together with pseType.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[B,N,1024,Skv], [1,N,1024,Skv], [B,N], [N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dropMaskOptional</td>
      <td>Optional input</td>
      <td>Dropout mask.</td>
      <td>-</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>paddingMaskOptional</td>
      <td>Optional input</td>
      <td>Reserved parameter, not used currently.</td>
      <td>Pass an empty value during the call.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qStartIdxOptional</td>
      <td>Optional input</td>
      <td>Represents the start index of the query sequence of the current chunk in the global sequence in the extrapolation use case.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>0, 1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvStartIdxOptional</td>
      <td>Optional input</td>
      <td>Represents the start index of the key and value sequences of the current chunk in the global sequence in the unpadding scenario.</td>
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
      <td>The value 1 indicates that the bit is not involved in calculation, and the value 0 indicates that the bit is involved in calculation.</td>
      <td>BOOL, UINT8</td>
      <td>ND</td>
      <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv] </td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxMaxOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of forward softmax.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[N,T,8]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSumOptional</td>
      <td>Optional input</td>
      <td>Intermediate output of forward softmax.</td>
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
      <td>Reserved parameter, not used currently.</td>
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
      <td>prefixOptional</td>
      <td>Optional input</td>
      <td>N of each batch in the prefix sparse scenario.</td>
      <td>-</td>
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
      <td>Proportion of 1 in dropMaskOptional.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>Optional input</td>
      <td>Left boundary of the sliding window during sparse computation.</td>
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
      <td>Number of heads on a single device, that is, the length of the N axis of Query.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputLayout</td>
      <td>Input</td>
      <td>Layout of input Q/K/V data.</td>
      <td>Supports TND.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise</td>
      <td>Optional input</td>
      <td>Controls internal computation precision.</td>
      <td>Reserved parameter, not used currently.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Optional input</td>
      <td>Sparse mode.</td>
      <td>The value range is 0-8. 5 is not supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pseType</td>
      <td>Optional input</td>
      <td>The data type must be INT64.</td>
      <td>The supported values are 0, 1, 2, and 3.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>Output</td>
      <td>dQ in the formula, the gradient of Query.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[TND]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dkOut</td>
      <td>Output</td>
      <td>dK in the Formula, the Key gradient.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
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
      <td>Reserved parameter, not used currently.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>The passed-in parameter is a required input, output, or required attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, dqOut, dkOut, or dvOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data format of query, keyIn, value, dy, pseShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOptional, softmaxSumOptional, softmaxInOptional, attentionInOptional, dqOut, dkOut, or dvOut is not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionUnpaddingScoreGradV2

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
        <td>Pointer to the workspace memory allocated on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream used to execute the task.</td>
      </tr>
    </tbody>
    </table>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFlashAttentionUnpaddingScoreGradV2 defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- For the B dimension of the input query, key, value, and dy: the batchsize must be equal, and the inputLayout must be consistent.
- The D (Head Dim) of the input query, key, and value must satisfy (qD == kD && kD >= vD).
- The data type of query, key, value, and pseShiftOptional must be consistent.
- The shapes of the input key and value must be consistent except for D. When the D dimensions of query, key, and value are the same, the shapes of query and dy must be consistent.
- The N of query and the N of key/value may differ, but they must be proportional, that is, Nq/Nkv must be a non-zero integer, with Nq in the value range of 1 to 256.
- Constraints on data shape, taking TND of inputLayout as an example, where:
    - T: The value range is 1 to 1M.
    - N: ranges from 1 to 256.
    - D: ranges from 1 to 768.
    - KeepProb: value range: (0, 1].
- The data layout format of query, key, and value only supports TND, where T is the tightly packed data of the B and S axes (SeqLenQ and SeqLenKV of each batch). B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden size, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the smallest unit dimension of the hidden layer, satisfying D = H/N.
- In TND format, some trailing batches can be not involved in calculation. In this case, pass the corresponding number of zeros at the end of actual_seq_qlen and actual_seq_kv_len. Suppose the actual length of S is [2, 3, 4, 5, 6] and the last two batches are not involved in calculation, the passed-in actual_seq_qlen is [2, 5, 9, 0, 0].
- In TND use cases, each batch segment is still generated as [N, Sq_i, Skv_i] internally, but is uniformly flattened for storage and parameter passing. If the actual query length of the i-th batch segment is Sq_i and the actual key/value length is Skv_i, the number of PSE elements in this segment is N *Sq_i* Skv_i, and the total length pseTotalLen of the entire PSE is sum_i(N *Sq_i* Skv_i).
- pseShiftOptional: If Sq is greater than 1024, Sq of each batch has the same length as Skv, and the mask is a lower triangular mask with sparseMode set to 0, 2, or 3, alibi position encoding compression can be enabled. In this case, only the last 1024 rows of the original PSE need to be input, that is, alibi_compress = ori_pse[:, :, -1024:, :], to achieve memory optimization. Details are as follows:
  - When the parameter differs for each batch, the shape is BNHSkv (H = 1024).
  - When the parameter is the same for each batch, the shape is 1NHSkv (H = 1024).
  - If pseType is 2 or 3, the data type must be FLOAT32. Supported shapes are [B,N] or [N].
  - If this parameter is not enabled, set pseShiftOptional to nullptr and pseType to 1.
- Meaning of each pseType value

    | pseType     | Meaning                       |      Note   |
    | ----------- | --------------------------------- | ----------|
    | 0           | Externally input pse is multiplied first and then added | - |
    | 1           | Externally input pse is added first and then multiplied | Consistent with the implementation of [FlashAttentionUnpaddingScoreGrad](./aclnnFlashAttentionUnpaddingScoreGrad_en.md). |
    | 2           | Internally generated pse is multiplied first and then added | - |
    | 3           | Internally generated pse is multiplied first, then added, and then square-rooted | - |

- Constraints on sparseMode are as follows:
  - When the shapes of all attenMaskOptional are smaller than 2048 and identical, it is recommended to use the default mode to reduce memory usage.
  - When configured as 1, 2, or 3, the user-configured preTokens and nextTokens do not take effect.
  - When configured as 0 or 4, ensure that attenMaskOptional is consistent with the ranges of preTokens and nextTokens.
  - If not explicitly specified, it is recommended to pass 0.
  - For details about different sparse modes, see [sparseMode Introduction](../../../docs/en/context/sparse_mode_introduction.md).
  - When configured as 7, the optional input pseShiftOptional is not supported.
  - When configured as 8, the optional input pseShiftOptional is supported when q and kv of each sequence have equal lengths, and pse is generated globally. Outward slicing in the q direction is supported, which requires that q and kv of each sequence have equal lengths before slicing, and after slicing, actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0 (this is an experimental feature).
- In some scenarios, if the computation workload is too large, the operator execution may time out (aicore error with errorStr: timeout or trap error). In this case, axis splitting is recommended. Note: The computation workload here is affected by parameters such as B, S, N, and D. Larger values result in a heavier workload.
- The prefixOptional sparse computation only supports the compressed scenario, sparseMode=6. When Sq > Skv, the N value range of prefix is [0, Skv]; when Sq <= Skv, the N value range of prefix is [Skv-Sq, Skv].
- actualSeqQLenOptional input supports an S length of 0 in a batch, but the optional input pseShiftOptional is not supported in this case. The length of actualSeqQLenOptional ranges from 1 to 2K. When the prefixOptional input is present, its maximum length is 1K.
- Constraints on the softmaxMax and softmaxSum parameters: The input format is fixed as [B, N, S, 8], except for the TND input format, which is [T, N, 8]. Note: T=B*S.
- The value of headNum must be consistent with the N value in the input query.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate for resource initialization.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> qShape = {256, 1, 128};
  std::vector<int64_t> kShape = {256, 1, 128};
  std::vector<int64_t> vShape = {256, 1, 128};
  std::vector<int64_t> dxShape = {256, 1, 128};
  std::vector<int64_t> attenmaskShape = {256, 256};
  std::vector<int64_t> softmaxMaxShape = {256, 1, 8};
  std::vector<int64_t> softmaxSumShape = {256, 1, 8};
  std::vector<int64_t> attentionInShape = {256, 1, 128};

  std::vector<int64_t> dqShape = {256, 1, 128};
  std::vector<int64_t> dkShape = {256, 1, 128};
  std::vector<int64_t> dvShape = {256, 1, 128};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;

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
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;

  std::vector<float> qHostData(32768, 1);
  std::vector<float> kHostData(32768, 1);
  std::vector<float> vHostData(32768, 1);
  std::vector<float> dxHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);
  std::vector<float> attentionInHostData(32768, 1);
  std::vector<float> dqHostData(32768, 0);
  std::vector<float> dkHostData(32768, 0);
  std::vector<float> dvHostData(32768, 0);

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
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT, &dv);
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

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionUnpaddingScoreGradV2.
  ret = aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(q, k, v, dx, pse, dropMask, padding,
            attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
            scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise, sparseMode, pseType,
            dq, dk, dv, dpse, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionUnpaddingScoreGradV2.
  ret = aclnnFlashAttentionUnpaddingScoreGradV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionUnpaddingScoreGradV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify it based on the specific API definition.
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify this step based on the specific API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);

  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
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
