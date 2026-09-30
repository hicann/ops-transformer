# aclnnFlashAttentionVarLenScoreV5

## Applicable Products

|Product      | Supported or Not |
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

* Uses the FlashAttention algorithm to perform self-attention computation in training scenarios. This API is adapted to support the sink function of the gpt-oss model. Compared with the [aclnnFlashAttentionVarLenScoreV3](./aclnnFlashAttentionVarLenScoreV3_en.md) API, this API has an optional `sinkInOptional` parameter and retains the optional `softmaxOutLayout` parameter of the [aclnnFlashAttentionVarLenScoreV4](./aclnnFlashAttentionVarLenScoreV4_en.md) API.
* Formula
  The forward computation formula for attention is as follows:

  $$
  Attention=Dropout(Softmax(Mask(scale*(query*key^T + queryRope*keyRope^T) + pse),atten\_mask),keep\_prob)*value
  $$

  The following calculation logic applies after sink is added, with primary modifications made to the softmax_max and softmax_sum calculations:

  $$
  S = Q * K^{T}
  $$

  $$
  m = max(sink, max(S))
  $$

  $$
  Attention = \frac{e^{S - m} * V}{\sum e^{S-m} + S^{sink - m}}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFlashAttentionVarLenScoreV5` is called to perform computation.

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
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>Input</td>
        <td>query in the formula.</td>
        <td>The data type must be the same as that of key and value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryRope</td>
        <td>Optional input</td>
        <td>queryRope in the formula.</td>
        <td>The data type must be the same as that of key and value.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>key in the formula.</td>
        <td>The data type must be the same as that of query and value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyRope</td>
        <td>Optional input</td>
        <td>keyRope in the formula.</td>
        <td>The data type must be the same as that of query and value.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>value in the formula.</td>
        <td>The data type must be the same as that of query and key.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>realShiftOptional</td>
        <td>Optional input</td>
        <td>pse in the formula.</td>
        <td>Incompatible with queryRope and keyRope. When RoPE is not used, the data type must match that of query. Use this parameter with pseType.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[B,N,1024,Skv], [1,N,1024,Skv], [B,N], [N]</td>
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
        <td>attenMaskOptional</td>
        <td>Optional input</td>
        <td>atten_mask in the formula.</td>
        <td>A value of 1 indicates the bit is excluded from computation, while 0 indicates it is included.</td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sinkOptional</td>
        <td>Optional input</td>
        <td>sink in the formula.</td>
        <td>Provides the sink function.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[headNum]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>Optional input</td>
        <td>N of each batch in the prefix sparse computation scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>Input</td>
        <td>Sequence length of query corresponding to each batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>Input</td>
        <td>Sequence length of key/value corresponding to each batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>Optional input</td>
        <td>Global start index of the query sequence for the current chunk in an outer splitting scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>Optional input</td>
        <td>Global start index of the query sequence for the current chunk in an outer splitting scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Optional input</td>
        <td>scale in the formula, indicating the scale factor.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>Optional input</td>
        <td>Proportion of 1s in dropMaskOptional.</td>
        <td>The value range is (0, 1].</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Optional input</td>
        <td>Left boundary of the sliding window, used for sparse computation.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Optional input</td>
        <td>Right boundary of the sliding window, used for sparse computation.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Number of heads on a single device, that is, the length of the N axis of the input query.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Layout of the input query, key, and value.</td>
        <td>TND is supported.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Optional input</td>
        <td>Used to improve precision.</td>
        <td>Reserved.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Optional input</td>
        <td>Sparse mode.</td>
        <td>The value ranges from 0 to 8, excluding 5. When RoPE inputs are passed, 6 is not supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>Optional input</td>
        <td>Calculation sequence of multiplication and addition.</td>
        <td>The value ranges from 0 to 3 without RoPE, but is restricted to 1 when RoPE is enabled.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
          <td>softmaxOutLayout</td>
          <td>Optional input</td>
          <td>Softmax output layout in the TND scenario.</td>
          <td>When same_as_input is passed, the softmax output layout is the same as the input layout, which is TND. When an empty string is passed, the original logic is retained, and the softmax output layout is NTD.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>Output</td>
        <td>Intermediate result of the Max operation in softmax, used for backward computation.</td>
        <td>The output shape is determined by softmaxOutLayout.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[N,T,8] or [T,N,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>Output</td>
        <td>Intermediate result of the Sum operation in softmax, used for backward computation.</td>
        <td>The output shape is determined by softmaxOutLayout.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[N,T,8] or [T,N,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>Output</td>
        <td>Final output of the formula.</td>
        <td>The data type and shape must be the same as those of query.</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[T,N,D]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace required to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 286px">
  <col style="width: 118px">
  <col style="width: 746px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, sinkOptional, softmaxMaxOut, softmaxSumOut, softmaxOutOut, or attentionOutOut is not supported.</td>
    </tr>
    <tr>
      <td>The data format of query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, sinkOptional, softmaxMaxOut, softmaxSumOut, softmaxOutOut, or attentionOutOut is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>Internal API verification error, usually caused by unsupported input shape or attribute specifications.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionVarLenScoreV5

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1049px"><colgroup>
  <col style="width: 167px">
  <col style="width: 118px">
  <col style="width: 764px">
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation
  - `aclnnFlashAttentionVarLenScoreV5` defaults to a deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- **B** (batch size) of the input `query`, `key`, and `value` must be the same.
- **D** (Head-Dim) of the input `query`, `key`, and `value` must satisfy (qD == kD && kD >= vD).
- `input_layout` of the input `query`, `key`, and `value` must be the same.
- The input data types of `query`, `key`, `value`, and `realShiftOptional` must be the same.
- **N** of the input `query` can be different from **N** of the `key` or `value`, but they must be proportional. That is, Nq/Nkv must be a non-zero integer and the value of Nq ranges from 1 to 256. When Nq/Nkv > 1, it is a grouped-query attention (GQA). When Nkv=1, it is a multi-query attention (MQA). Unless otherwise specified, **N** in this document indicates Nq.
- Constraints on the data shape:
  - **T(B*S)**: The value ranges from 1 to 1M.
  - **B**: The value ranges from 1 to 20000. When `prefixOptional` is passed, **B** supports a maximum of 1K.
  - **N**: The value ranges from 1 to 256.
  - **S**: The value ranges from 1 to 1M.
  - **D**: The value ranges from 1 to 768.
- The data format of `query`, `key`, and `value` can only be TND. **T** indicates the data closely arranged on the B and S axes (SeqLenQ and SeqLenKV of each batch). **B (Batch)** indicates the batch size of the input sample, and **S (Seq-Length)** indicates the length of the input sample sequence. **H (Head-Size)** indicates the size of the hidden layer. **N (Head-Num)** indicates the number of heads. **D (Head-Dim)** indicates the minimum unit size of the hidden layer (D = H/N).
- `realShiftOptional`: If Sq is greater than 1024, Sq and Skv of each batch are of equal length, and it is a lower triangular mask scenario with `sparseMode` being 0, 2, or 3, ALiBi positional encoding compression can be enabled. In this case, only the last 1024 rows of the original PSE need to be input for memory optimization, that is, `alibi_compress = ori_pse[:, :, -1024:, :]`. Specifically:
  - If the parameters of each batch are different, the shape is BNHSkv (H=1024).
  - When each batch is the same, the shape is 1NHSkv (H=1024).
  - If `pseType` is 2 or 3, the data type must be FLOAT32, and the supported shapes are [B,N] and [N].
  - If this parameter is disabled, `realShiftOptional` must be set to `nullptr` and `pseType` must be set to 1.
- Restrictions on `sparseMode`
  - When the shapes of all `attenMaskOptional` are less than 2048 and are the same, the default mode is recommended to reduce memory usage.
  - When the value is set to 1, 2, 3, 5, or 6, the user-configured `preTokens` and `nextTokens` do not take effect.
  - When the value is set to 0, 4, or 7, ensure that the ranges of `attenMaskOptional`, `preTokens`, and `nextTokens` are consistent.
  - If no specific value is required, 0 is recommended.
  - For details about the sparse modes, see [Sparse Mode Description](../../../docs/en/context/sparse_mode_introduction.md).
  - When the value is set to 1, 2, 3, 4, 6, 7, or 8, the value of `attenMaskOptional` must be correct. Otherwise, the calculation result is incorrect. If `attenMaskOptional` is set to `None`, `sparseMode`, `preTokens`, and `nextTokens` do not take effect and all tokens are calculated.
  - When the value is set to 3, computation on invalid rows is not supported, and Sq <= Skv must be satisfied for each batch.
  - When the value is set to 7, `realShiftOptional` is not supported.
  - When the value is set to 8, `realShiftOptional` is supported when the q and kv of each sequence have the same length. PSE generation is performed globally. Outer splitting in the q direction is supported. q and kv of each sequence must have the same length before outer splitting, and `actualSeqQLenOptional` is passed after outer splitting.
- In some scenarios, if the computation load is too large, the operator execution may time out (an AI Core error is reported, and `errorStr` is `timeout or trap error`). In this case, you are advised to perform axis splitting. Note: The computation load is affected by parameters such as **B**, **S**, **N**, and **D**. Larger values indicate larger computation loads.
- The `prefixOptional` sparse computing scenario is `sparseMode=6`. When Sq > Skv, the value range of N of `prefix` is \[0, Skv\]. When Sq ≤ Skv, the value range of N of `prefix` is \[Skv – Sq, Skv\].
- In the band scenario, the values of `preTokens` and `nextTokens` must overlap.
 
- The `attenMaskOptional` input does not support padding. That is, `attenMaskOptional` cannot contain a row of all 1s.
- The `actualSeqQLenOptional` input supports the S length of 0 in a batch. In this case, the `realShiftOptional` input is not supported. If the actual S length is \[2,2,0,2,2\], the value of `actualSeqQLenOptional` is \[2,4,4,6,8\].
[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0 (experimental feature)
- `softmaxOutLayout` can be set to an empty string or `same_as_input`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  // (Boilerplate) Initialize resources.
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
  // Call aclrtMemcpy to copy data from the host to the device.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the acl API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
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


  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API aclnnFlashAttentionVarLenScoreV5.
  ret = aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
            q, qRope, k, kRope, v, pse, dropMask, padding, attenmask, sink, prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
            scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise,
            sparseMode, pseType, softmaxOutLayout, softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API aclnnFlashAttentionVarLenScoreV5.
  ret = aclnnFlashAttentionVarLenScoreV5(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionVarLenScoreV5 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
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
