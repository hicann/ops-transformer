# aclnnFlashAttentionVarLenScoreV5

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:00:42.362Z -->

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

- Description: In training scenarios, the FlashAttention algorithm is used to implement self-attention computation. It supports the sink function for gpt-oss models. **The difference from the [aclnnFlashAttentionVarLenScoreV3](./aclnnFlashAttentionVarLenScoreV3_en.md) API is: the `sinkOptional` optional input is added, while the `softmaxOutLayout` optional input from [aclnnFlashAttentionVarLenScoreV4](./aclnnFlashAttentionVarLenScoreV4_en.md) is retained.**

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The sinkOptional and softmaxOutLayout parameters are not supported.

  <!-- end id7 -->

- Formula:
  The forward calculation formula of attention is as follows:

  $$
  Attention=Dropout(Softmax(Mask(scale*(query*key^T + queryRope*keyRope^T) + pse),atten\_mask),keep\_prob)*value
  $$

  The computation logic after adding **sink** is as follows, mainly modifying the relevant softmax_max and softmax_sum logic computation parts.

  $$
  S = Q * K^{T}
  $$

  $$
  m = max(sink, max(S))
  $$

  $$
  Attention = \frac{e^{S - m} * V}{\sum e^{S-m} + e^{sink - m}}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that includes the computation flow of the operator, and then call the "aclnnFlashAttentionVarLenScoreV5" API to perform the computation.

```c++
aclnnStatus aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *queryRope,
    const aclTensor   *key,
    const aclTensor   *keyRope,
    const aclTensor   *value,
    const aclTensor   *realShiftOptional,
    const aclTensor   *dropMaskOptional,
    const aclTensor   *paddingMaskOptional,
    const aclTensor   *attenMaskOptional,
    const aclTensor   *sinkOptional,
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
    char              *softmaxOutLayout,
    const aclTensor   *softmaxMaxOut,
    const aclTensor   *softmaxSumOut,
    const aclTensor   *softmaxOutOut,
    const aclTensor   *attentionOutOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```c++
aclnnStatus aclnnFlashAttentionVarLenScoreV5(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```

## aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1452px"><colgroup>
    <col style="width: 174px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 213px">
    <col style="width: 115px">
    <col style="width: 169px">
    <col style="width: 145px">
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
        <td>The query in the formula.</td>
        <td>Data type is consistent with the data type of key/value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryRope</td>
        <td>Optional input</td>
        <td>The queryRope in the formula.</td>
        <td>Data type must be consistent with the data type of key/value.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>key in the formula.</td>
        <td>Data type must be consistent with the data type of query/value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyRope</td>
        <td>Optional input</td>
        <td>keyRope in the formula.</td>
        <td>Data type must be consistent with the data type of query/value.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Value in the formula.</td>
        <td>The data type is the same as the data type of query/key.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>realShiftOptional</td>
        <td>Optional input</td>
        <td>pse in the formula.</td>
        <td>Incompatible with queryRope and keyRope. When rope is not used, the data type is the same as that of query, and this parameter must be used together with pseType.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[B,N,1024,Skv], [1,N,1024,Skv], [pseTotalLen], [B,N], [N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>Optional input</td>
        <td>Dropout in the formula.</td>
        <td>Incompatible with queryRope and keyRope.</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>paddingMaskOptional</td>
        <td>Optional input</td>
        <td>Reserved parameter, not used for now.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Optional input</td>
        <td>atten_mask in the formula.</td>
        <td>The value 1 indicates that the bit does not participate in the computation, and 0 indicates that the bit participates in the computation.</td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv] </td>
        <td>√</td>
      </tr>
      <tr>
        <td>sinkOptional</td>
        <td>Optional input</td>
        <td>The sink in the formula.</td>
        <td>Provides the sink function.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[headNum]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>Optional input</td>
        <td>Indicates the N value of each batch in the prefix sparse computation scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0 or 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>Optional input</td>
        <td>Describes the sequence length of the query in each batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>Optional input</td>
        <td>Describes the sequence length of key/value corresponding to each Batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>Optional input</td>
        <td>Indicates the start index of the query sequence of the current chunk in the global sequence in the outer-ring scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>Optional input</td>
        <td>Indicates the start indices of the key and value sequences of the current block in the global sequence in the external splitting scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Optional input</td>
        <td>scale in the formula, which indicates the scaling factor.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>Optional input</td>
        <td>Indicates the proportion of 1 in dropMaskOptional.</td>
        <td>The value range is (0, 1].</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Optional input</td>
        <td>Used for sparse computation, indicating the left boundary of the sliding window.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Optional input</td>
        <td>Used for sparse computation. Indicates the right boundary of the sliding window.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Indicates the number of heads on a single device, that is, the length of the N axis of the input query.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Represents the data layout format of the input query, key, and value.</td>
        <td>Supports TND.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Optional input</td>
        <td>Used to improve precision.</td>
        <td>The default value 0 is recommended.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Optional input</td>
        <td>Indicates the sparse mode.</td>
        <td>Value range: 0-8. Value 5 is not supported. Value 6 is not supported when rope is passed in.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>Optional input</td>
        <td>Controls the computation order of mul and add.</td>
        <td>Configured as 0-3 when rope is not used; only 1 is supported when rope is used.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
          <td>softmaxOutLayout</td>
          <td>Optional input</td>
          <td>Controls the softmax output in the TND use case.</td>
          <td>If "same_as_input" is passed in, the softmax output layout is consistent with the input, that is, the TND layout. If an empty string is passed in, the original logic is retained and the softmax output layout is NTD.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>Output</td>
        <td>Intermediate Max result of the Softmax computation, used for backward computation.</td>
        <td>The output shape is determined by softmaxOutLayout.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T,N,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>Output</td>
        <td>Intermediate Sum result of the Softmax computation, used for backward computation.</td>
        <td>The output shape is determined by softmaxOutLayout.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T,N,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxOutOut</td>
        <td>Output</td>
        <td>Reserved parameter, not used for now.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>Output</td>
        <td>The final output in the formula.</td>
        <td>Data type and shape type are consistent with query.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[T,N,D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which includes computation of the operator.</td>
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
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, sinkOptional, softmaxMaxOut, softmaxSumOut, softmaxOutOut, and attentionOutOut are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data formats of query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, sinkOptional, softmaxMaxOut, softmaxSumOut, softmaxOutOut, and attentionOutOut are not in the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>Internal API verification error, usually caused by the input shape or attribute specifications being out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionVarLenScoreV5

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
      <td>Workspace size allocated on the device, obtained through the first-phase API aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. Includes computation flow of the operator.</td>
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

- Deterministic computation:
  - aclnnFlashAttentionVarLenScoreV5 defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- The B (batch size) dimension of the input query, key, and value must be equal.
- The D (Head-Dim) of the input query, key, and value must satisfy (qD == kD && kD >= vD).
- The input_layout of the input query, key, and value must be consistent.
- The data types of the input query, key, value, and realShiftOptional must be consistent.
- The N of the input query and the N of key/value may differ, but they must be in a proportional relationship, i.e., Nq/Nkv must be a non-zero integer, with Nq ranging from 1 to 256. When Nq/Nkv > 1, it is GQA (grouped-query attention); when Nkv = 1, it is MQA (multi-query attention). Unless otherwise specified in this document, N refers to Nq.
- Constraints on data shape:
  - T(B*S): ranges from 1 to 1M.
  - B: ranges from 1 to 20000. When prefixOptional is used, B supports a maximum of 1K.
  - N: ranges from 1 to 256.
  - S: ranges from 1 to 1M.
  - D: ranges from 1 to 768.
- The data layout format of query, key, and value only supports TND, where T is the tightly packed data of the B and S axes (SeqLenQ and SeqLenKV of each batch). B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden size, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the smallest unit dimension of the hidden layer, satisfying D = H/N.
- Description of the empty tensor scenario:
    - When either T(B*S) or N1(headNum) is 0, the operator performs no computation and directly returns success, with workspaceSize of 0.
    - When either dimension of N2 (number of key/value heads) or D (Head-Dim) is 0, query/key/value are empty tensors (shapeSize is 0), but the output shapes of softmaxMaxOut and softmaxSumOut may be non-empty (their shapes do not depend on the D dimension). In this case, the operator executes the empty input processing flow to initialize the non-empty outputs.
    - In the preceding empty tensor use cases, the operator returns normally and no error is reported.
- realShiftOptional: If Sq is greater than 1024, the Sq of each batch is equal in length to Skv, and the use case is a lower-triangular mask with sparseMode set to 0, 2, or 3, alibi position encoding compression can be enabled. In this case, only the last 1024 rows of the original PSE need to be input to optimize memory usage, that is, alibi_compress = ori_pse[:, :, -1024:, :]. The details are as follows:
  - When the parameter differs for each batch, the shape is BNHSkv (H = 1024).
  - When the parameter is the same for each batch, the shape is 1NHSkv (H = 1024).
  - In the TND use case, each batch segment is still generated as [N, Sq_i, Skv_i], but is uniformly flattened for storage and parameter passing. If the actual query length of the i-th batch segment is Sq_i and the actual key/value length is Skv_i, the number of PSE elements in this segment is N *Sq_i* Skv_i, and the total length of the entire PSE, pseTotalLen, is sum_i(N *Sq_i* Skv_i).
  - If pseType is 2 or 3, the data type must be FLOAT32, and the supported shape range is [B,N] or [N].
  - If this parameter is not enabled, realShiftOptional must be passed as nullptr and pseType must be passed as 1.
- innerPrecise: Currently, 0 and 1 are reserved configuration values, and 2 enables invalid-row computation. This function prevents precision loss caused by full-row masks during computation but degrades performance. If the operator can determine that the invalid-row use case exists, for example, sparseMode set to 3 and Sq > Skv, invalid-row computation is automatically enabled.
- Constraints on sparseMode are as follows:
  - When all attenMaskOptional shapes are smaller than 2048 and identical, it is recommended to use the default mode to reduce memory usage.
  - When configured as 1, 2, 3, or 6, the user-configured preTokens and nextTokens do not take effect.
  - When configured as 0, 4, or 7, ensure that attenMaskOptional is consistent with the ranges of preTokens and nextTokens.
  - If not explicitly specified, it is recommended to pass 0.
  - For details about different sparse modes, see [sparseMode Introduction](../../../docs/en/context/sparse_mode_introduction.md).
  - When set to 1, 2, 3, 4, 6, 7, or 8, the correct corresponding attenMaskOptional should be passed; otherwise the computation result will be incorrect. When attenMaskOptional is None, the sparseMode, preTokens, and nextTokens parameters do not take effect and full computation is used by default.
  - When configured as 3, invalid row computation is not supported, and Sq <= Skv must be satisfied for each batch.
  - When configured as 7, the optional input realShiftOptional is not supported.
  - When configured as 8, the optional input realShiftOptional is supported when q and kv of each sequence have equal lengths, and pse is generated globally. Outward slicing in the q direction is supported, which requires that q and kv of each sequence have equal lengths before outward slicing, and after outward slicing, actualSeqQLenOptional[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0 (this feature is experimental).
- In some scenarios, if the computation workload is too large, the operator execution may time out (aicore error with errorStr: timeout or trap error). In this case, axis splitting is recommended. Note: The computation workload here is affected by parameters such as B, S, N, and D. Larger values result in a heavier workload.
- In the prefixOptional sparse computation scenario (sparseMode=6), when Sq > Skv, the N value of prefix ranges from [0, Skv]; when Sq <= Skv, the N value of prefix ranges from [Skv-Sq, Skv].
- In the band scenario, preTokens and nextTokens must have an intersection.
- The attenMaskOptional input does not support padding, meaning that attenMaskOptional cannot contain any row where all values are 1.
- The actualSeqQLenOptional input supports an S length of 0 for a certain batch. In this case, the optional input realShiftOptional is not supported. For example, if the actual S lengths are [2, 2, 0, 2, 2], the actualSeqQLenOptional passed in should be [2, 4, 4, 6, 8].
- softmaxOutLayout supports the following input values: an empty string or "same_as_input".

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"

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
  // Call aclrtMalloc to allocate memory on the device.
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

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
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
  std::vector<int64_t> attenmaskShape = {256, 256};
  std::vector<int64_t> sinkShape = {1};

  std::vector<int64_t> attentionOutShape = {256, 1, 128};
  std::vector<int64_t> softmaxMaxShape = {256, 1, 8};
  std::vector<int64_t> softmaxSumShape = {256, 1, 8};

  void* qDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* sinkDeviceAddr = nullptr;
  void* attentionOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* k = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* v = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* sink = nullptr;
  aclTensor* attentionOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxOut = nullptr;

  std::vector<float> qHostData(32768, 1);
  std::vector<float> qRopeHostData(16384, 1);
  std::vector<float> kHostData(32768, 1);
  std::vector<float> kRopeHostData(16384, 1);
  std::vector<float> vHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> sinkHostData(1, 3.0);
  std::vector<float> attentionOutHostData(32768, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);

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
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinkHostData, sinkShape, &sinkDeviceAddr, aclDataType::ACL_FLOAT, &sink);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_BF16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> prefixOp = {0};
  aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  std::vector<int64_t>  acSeqQLenOp = {256};
  std::vector<int64_t>  acSeqKvLenOp = {256};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;

  char layOut[5] = {'T', 'N', 'D', 0};
  char softmaxOutLayout[] = "same_as_input";


  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionVarLenScoreV5.
  ret = aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
            q, qRope, k, kRope, v, pse, dropMask, padding, attenmask, sink, prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
            scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise,
            sparseMode, pseType, softmaxOutLayout, softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // Allocate device memory using the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionVarLenScoreV5.
  ret = aclnnFlashAttentionVarLenScoreV5(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionVarLenScoreV5 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to end.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, and copy the results from the device memory to the host. Modify as required by the specific API definition.
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify as required by the specific API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(qRope);
  aclDestroyTensor(k);
  aclDestroyTensor(kRope);
  aclDestroyTensor(v);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(sink);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(sinkDeviceAddr);
  aclrtFree(attentionOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
