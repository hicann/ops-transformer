# aclnnFlashAttentionScoreV3

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Description

- **API function**: Uses the FlashAttention algorithm to perform self-attention computation in training scenarios. This API is adapted to support the sink function of the gpt-oss model. **Compared with the [`aclnnFlashAttentionScoreV2`](./aclnnFlashAttentionVarLenScoreV2_en.md) API, this API has an optional `sinkInOptional` parameter.**

* Formula:
  The forward propagation formula for attention is as follows:
  
     - When `psetype` is set to 1, the calculation formula is the same as that of [`aclnnFlashAttentionScore`](./aclnnFlashAttentionScore_en.md).
  
     - When `psetype` is set to other values, the formula is as follows:
  
       $$
       attention\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\_mask),keep\_prob)*value
       $$
  
  The following calculation logic applies after sink is added, primarily modifying the calculation parts related to softmax_max and softmax_sum:
  
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

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFlashAttentionScoreV3GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFlashAttentionScoreV3` is called to perform computation.

```c++
aclnnStatus aclnnFlashAttentionScoreV3GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *realShiftOptional,
  const aclTensor   *dropMaskOptional,
  const aclTensor   *paddingMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *sinkOptional,           
  const aclIntArray *prefixOptional,
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
  const aclTensor   *softmaxMaxOut,
  const aclTensor   *softmaxSumOut,
  const aclTensor   *softmaxOutOut,
  const aclTensor   *attentionOutOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnFlashAttentionScoreV3(
  void             *workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor    *executor, 
  const aclrtStream stream)
```

## aclnnFlashAttentionScoreV3GetWorkspaceSize

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
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage Notes</th>
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
        <td>[BNSD], [BSND], [BSH], [SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td> key in the formula.</td>
        <td>The data type must be the same as that of query and value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[BNSD], [BSND], [BSH], [SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td> value in the formula.</td>
        <td>The data type must be the same as that of query and key.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[BNSD], [BSND], [BSH], [SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>realShiftOptional</td>
        <td>Optional input</td>
        <td>pse in the formula.</td>
        <td>The data type must match query. Use this parameter with pseType.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,N,1,Skv], [1,N,Sq,Skv], [B,N,1024,Skv], [1,N,1024,Skv], [B,N], [N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>Optional input</td>
        <td>Dropout in the formula.</td>
        <td>-</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Optional input</td>
        <td>atten_mask in the formula.</td>
        <td>A value of 1 indicates that the position does not participate in the calculation, while a value of 0 indicates that it does.</td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv] </td>
        <td>√</td>
      </tr>
  <tr>
        <td>sinkOptional</td>
        <td>Optional input</td>
        <td>sink in the formula.</td>
        <td>
          <ul>
              <li>Provides the sink function.</li>
              <li>The input shape type must be [headNum].</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>0, 1</td>
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
        <td>Number of heads on a single rank, that is, the length of the N axis of the input query.</td>
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
        <td>BSH, SBH, BSND, and BNSD are supported.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Optional input</td>
        <td>Used to improve precision.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Optional input</td>
        <td>Sparse mode.</td>
        <td>The value can be 0, 1, 2, 3, 4, 5, or 6.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>Optional input</td>
        <td>Calculation sequence of multiplication and addition.</td>
        <td>The value can be 0, 1, 2, or 3.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>Output</td>
        <td>Intermediate result of the Max operation in Softmax, used for backward calculation.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[B,N,Sq,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>Output</td>
        <td>Intermediate result of the Sum operation in Softmax, used for backward calculation.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[B,N,Sq,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>Output</td>
        <td>Final output of the formula.</td>
        <td>The data type and shape must be the same as those of query.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[BNSD], [BSND], [BSH], [SBH]</td>
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
  
  aclnnStatus: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  
  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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

## aclnnFlashAttentionScoreV3

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnFlashAttentionScoreV3GetWorkspaceSize.</td>
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

  aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnFlashAttentionScoreV3` defaults to a deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- The constraints for input `query`, `key`, and `value` are as follows:
  - The data format of `query`, `key`, and `value` can be interpreted from multiple dimensions. To be specific, **B** (Batch) indicates the size of an input sample batch, **S** (Seq-Length) indicates the length of the input sample sequence, **H** (Head-Size) indicates the size of the hidden layer, **N** (Head-Num) indicates the number of heads, and **D** (Head-Dim) indicates the minimum unit size of the hidden layer (**D** = **H**/**N**).
  - **B**: The batch sizes must be equal.
  - **D**: Head-Dim must satisfy (qD == kD && kD >= vD).
  - `inputLayout` must be consistent.
- The shapes of input `key` and `value` must be the same except D.
- You need to pay attention to constraints of the data shape. The following takes the `inputLayout` values BSND and BNSD as examples to describe the constraints (H = N\*D in BSH and SBH):
  
  - **B**: The value ranges from 1 to 2M. When `prefixOptional` is passed, **B** supports a maximum of 2K.
  - **N**: The value ranges from 1 to 256.
  - **S**: The value ranges from 1 to 1M.
  - **D**: The value ranges from 1 to 768.
- `realShiftOptional`: If Sq is greater than 1024, Sq and Skv of each batch are of equal length, and it is a lower triangular mask scenario with `sparseMode` being 0, 2, or 3, ALiBi positional encoding compression can be enabled. In this case, only the last 1024 rows of the original PSE need to be input for memory optimization, that is, `alibi_compress = ori_pse[:, :, -1024:, :]`. Specifically:
  - If the parameters of each batch are different, the shape is BNHSkv (H=1024).
  - When each batch is the same, the shape is 1NHSkv (H=1024).
  - If `pseType` is 2 or 3, the data type must be FLOAT32, and the supported shapes are [B,N] and [N].
  - If this parameter is not enabled, pass a null pointer to `realShiftOptional` and 1 to `pseType`.
- `innerPrecise`: 0 and 1 are reserved, and 2 indicates that invalid row calculation is enabled. This function is used to prevent precision loss caused by the mask of the entire row during calculation. However, this configuration deteriorates the performance. If the operator can determine that invalid rows exist, the invalid row computation is automatically enabled, such as in scenarios where `sparseMode` is set to 3 and Sq is greater than Skv.
- Meanings of `pseType` values:

    | pseType     | Meaning                             |      Remarks  |
    | ----------- | --------------------------------- | ----------|
    | 0           | A value is externally passed to `pse`, and multiplication is required before addition.             | - |
    | 1           | A value is externally passed to `pse`, and addition is required before multiplication.             | The implementation is the same as that of [FlashAttentionScore](./aclnnFlashAttentionScore_en.md).|
    | 2           | A value is internally passed to `pse`, and multiplication is required before addition.             | - |
    | 3           | A value is internally passed to `pse`, and multiplication and addition are required before square root operation.        | - |
    
- When `pseType` is set to 2 or 3, Sq and Skv must be of the same length.
- The constraints for `sparseMode` are as follows:
  - If the shape values of all `attenMaskOptional` are the same and less than 2048, you are advised to use the default mode to reduce memory usage.
  - When the value is set to 1, 2, 3, or 5, the user-configured `preTokens` and `nextTokens` do not take effect.
  - When the value is set to 0 or 4, ensure that the ranges of `attenMaskOptional`, `preTokens`, and `nextTokens` are consistent.
  - If no specific value is required, you are advised to set it to 0.
  - For details about the sparse modes, see [Sparse Mode Description](../../../docs/en/context/sparse_mode_introduction.md).
- In some scenarios, if the computation load is too large, the operator execution may time out (AI Core error, errorStr: timeout or trap error). In this case, you are advised to perform axis splitting. Note: The computation load is affected by parameters such as **B**, **S**, **N**, and **D**. Larger values indicate larger computation loads.
- In the band scenario, the values of `preTokens` and `nextTokens` must overlap.
- The `prefixOptional` sparse computing scenario is `sparseMode=5` or `sparseMode=6`. When Sq > Skv, the value range of N of `prefix` is \[0, Skv\]. When Sq ≤ Skv, the value range of N of `prefix` is \[Skv – Sq, Skv\].
- If Sq of `realShiftOptional` is greater than 1024, if BNHS and 1NHS are configured, Sq and Skv must have the same length.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
  
```C++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
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
  // (Fixed writing) Initialize resources.
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
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  int64_t B = 1;
  int64_t N1 = 1;
  int64_t N2 = 1;
  int64_t S1 = 256;
  int64_t S2 = 256;
  int64_t D = 128;
  int64_t H1 = N1 * D;
  int64_t H2 = N2 * D;

  int64_t q_size = S1 * B * H1;
  int64_t kv_size = S2 * B * H2;
  int64_t atten_mask_size = S1 * S2;
  int64_t softmax_size = B * N1 * S1 * 8;
  int64_t sink_size = N1;

  std::vector<int64_t> qShape = {S1, B, H1};
  std::vector<int64_t> kShape = {S2, B, H2};
  std::vector<int64_t> vShape = {S2, B, H2};
  std::vector<int64_t> attenmaskShape = {S1, S2};
  std::vector<int64_t> sinkShape = {N1};

  std::vector<int64_t> attentionOutShape = {S1, B, H1};
  std::vector<int64_t> softmaxMaxShape = {B, N1, S1, 8};
  std::vector<int64_t> softmaxSumShape = {B, N1, S1, 8};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* sinkDeviceAddr = nullptr;
  void* attentionOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
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

  std::vector<float> qHostData(q_size, 1.0);
  std::vector<float> kHostData(kv_size, 1.0);
  std::vector<float> vHostData(kv_size, 1.0);
  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> sinkHostData(sink_size, 3.0);
  std::vector<float> attentionOutHostData(q_size, 0);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinkHostData, sinkShape, &sinkDeviceAddr, aclDataType::ACL_FLOAT, &sink);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape , &attentionOutDeviceAddr, aclDataType::ACL_FLOAT, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  std::vector<int64_t> prefixOp = {0};
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  double scaleValue = 1.0/sqrt(128);
  double keepProb = 1.0;
  int64_t preTokens = INT32_MAX;
  int64_t nextTokens = INT32_MAX;
  int64_t headNum = N1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  char layOut[5] = {'S', 'B', 'H', 0};
  
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // Call the first-phase API of aclnnFlashAttentionScoreV3.
  ret = aclnnFlashAttentionScoreV3GetWorkspaceSize(
            q, k, v, pse, dropMask, padding, attenmask, sink, prefix, qStartIdx, kvStartIdx, scaleValue,
            keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise,
            sparseMode, pseType, softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // Call the second-phase API of aclnnFlashAttentionScoreV3.
  ret = aclnnFlashAttentionScoreV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV3 failed. ERROR: %d\n", ret); return ret);
  
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);
  
  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(sink);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyIntArray(prefix);
  aclDestroyIntArray(qStartIdx);
  aclDestroyIntArray(kvStartIdx);
  
  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
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
