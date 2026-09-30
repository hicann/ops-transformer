# NsaSelectedAttentionGrad

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Rearranges `key` and `value` data blocks with a size of `selectedBlockSize` based on `topkIndices`, and then computes the backward attention output in the training scenario.

- Formulas:

  Based on the input `topkIndices`, select `selectedBlockCount` blocks with a size of `selectedBlockSize` from `key` and `value` for rearrangement. The formula is as follows:

  $$
  selectedKey = Gather(key, topkIndices[i]),0<=i<selectedBlockCount \\
  selectedValue = Gather(value, topkIndices[i]),0<=i<selectedBlockCount
  $$

  Then, backward propagation of the attention mechanism is performed. The formula is as follows:

  $$
  V=P^TdY
  $$

  $$
  Q=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  K=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$
  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNsaSelectedAttentionGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnNsaSelectedAttentionGrad` is called to perform computation.

```c++
aclnnStatus aclnnNsaSelectedAttentionGradGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *attentionOut,
  const aclTensor   *attentionOutGrad,
  const aclTensor   *softmaxMax,
  const aclTensor   *softmaxSum,
  const aclTensor   *topkIndices,
  const aclIntArray *actualSeqQLenOptional,
  const aclIntArray *actualSeqKvLenOptional,
  const aclTensor   *attenMaskOptional,
  double             scaleValue,
  int64_t            selectedBlockSize,
  int64_t            selectedBlockCount,
  int64_t            headNum,
  char              *inputLayout,
  int64_t            sparseMode,
  const aclTensor         *dqOut,
  const aclTensor         *dkOut,
  const aclTensor         *dvOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaSelectedAttentionGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream);
```

## aclnnNsaSelectedAttentionGradGetWorkspaceSize

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td><code>query</code> in the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3–4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td><code>key</code> in the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3–4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td><code>value</code> in the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3–4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>topkIndices</td>
      <td>Input</td>
      <td><code>topkIndices</code> in the formula.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td><code>atten_mask</code> in the formula.</td>
      <td>-</td>
      <td>BOOL, UINT8</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqQLenOptional</td>
      <td>Input</td>
      <td>Cumulative sum of <code>S</code> of all batches in <code>query</code>.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqKvLenOptional</td>
      <td>Input</td>
      <td>Cumulative sum of <code>S</code> of all batches in <code>key</code>/<code>value</code>.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Scaling coefficient.</td>
      <td>Generally, this parameter is set to <code>D<sup>–0.5</sup></code>.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNum</td>
      <td>Input</td>
      <td>Number of heads.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputLayout</td>
      <td>Input</td>
      <td>Data layout of <code>query</code>/<code>key</code>/<code>value</code>.</td>
      <td>Currently, only <code>TND</code> is supported.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selectedBlockSize</td>
      <td>Input</td>
      <td>Length of each block.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selectedBlockCount</td>
      <td>Input</td>
      <td>Number of selected blocks.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Sparse mode.</td>
      <td><code>0</code> or <code>2</code> is supported.</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>Output</td>
      <td>Intermediate result of the Max operation in Softmax.</td>
      <td>Used for backward propagation.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>Output</td>
      <td>Intermediate result of the Sum operation in Softmax.</td>
      <td>Used for backward propagation.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>Output</td>
      <td>Final output of the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of <code>query</code>, <code>key</code>, <code>value</code>, <code>dy</code>, <code>pseShiftOptional</code>, <code>dropMaskOptional</code>, <code>paddingMaskOptional</code>, <code>attenMaskOptional</code>, <code>softmaxMaxOptional</code>, <code>softmaxSumOptional</code>, <code>softmaxInOptional</code>, <code>attentionInOptional</code>, <code>dqOut</code>, <code>dkOut</code>, or <code>dvOut</code> is not supported.</td>
    </tr>
    <tr>
      <td>The data format of <code>query</code>, <code>key</code>, <code>value</code>, <code>dy</code>, <code>pseShiftOptional</code>, <code>dropMaskOptional</code>, <code>paddingMaskOptional</code>, <code>attenMaskOptional</code>, <code>softmaxMaxOptional</code>, <code>softmaxSumOptional</code>, <code>softmaxInOptional</code>, <code>attentionInOptional</code>, <code>dqOut</code>, <code>dkOut</code>, or <code>dvOut</code> is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaSelectedAttentionGrad

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnNsaSelectedAttentionGradGetWorkspaceSize</code>.</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnNsaSelectedAttentionGrad` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- `B` (`batchsize`) of the input `query`, `key`, `value`, `attentionOut`, and `attentionOutGrad` must be the same.
- `N` (`numHead`) of the input `key` and `value` must be the same.
- `N` (`numHead`) of the input `query`, `attentionOut`, and `attentionOutGrad` must be the same.
- `D` (`HeadDim`) of the input `value`, `attentionOut`, and `attentionOutGrad` must be the same.
- `inputLayout` of the input `query`, `key`, `value`, `attentionOut`, and `attentionOutGrad` must be the same.
- The following uses the `inputLayout` `TND` as an example to describe the constraints on the data shape:
  - `T1`: The value ranges from 1 to 2M. It indicates the sum of `S` of all batches in `query`.
  - `T2`: The value ranges from 1 to 2M. It indicates the sum of `S` of all batches in `key`/`value`.
  - `B`: The value ranges from 1 to 2M.
  - `N1`: The value ranges from 1 to 128. It indicates `headNum` of `query`, and must be an integer multiple of `N2`.
  - `N2`: The value ranges from 1 to 128. It indicates `headNum` of `key` and `value`.
  - `G`: The value ranges from 1 to 32. `G` = `N1`/`N2`
  - `S`: The value ranges from 1 to 128K. The value of `S` for `key` and `value` must be greater than or equal to the product of `selectedBlockSize` and `selectedBlockCount`, and must be an integer multiple of `selectedBlockSize`.
  - `D`: The value can be `192` or `128`. `D` (`HeadDim`) of `key` and `value` can be different.
  - The value of `selectedBlockSize` must be less than or equal to `128` and be an integer multiple of 16.
  - The value range of `selectedBlockCount` is [1, 128]. The total size of the selected blocks (`selectedBlockCount * selectedBlockSize`) must be less than `128*64` (8K).
  - When the layout is `TND`, `S2` of each batch must be greater than `selectedBlockCount * selectedBlockSize`.
- The shape of the `softmaxMax` and `softmaxSum` parameters is restricted to `[T1, N1, 8]`.
- The shape of the `topkIndices` parameter is restricted to `[T1, N2, selectedBlockCount]`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_selected_attention_grad.h"

#define CHECK_RET(cond, return_expr)                   \
    do {                                               \
        if (!(cond)) {                                 \
            return_expr;                               \
        }                                              \
    } while (0)

#define LOG_PRINT(message, ...)                        \
    do {                                               \
        printf(message, ##__VA_ARGS__);                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    int64_t b = 1;
    int64_t s1 = 1;
    int64_t s2 = 1024;
    int64_t t1 = b * s1;
    int64_t t2 = b * s2;
    int64_t n1 = 1;
    int64_t n2 = 1;
    int64_t d = 192;

    int64_t sparseMode = 0;
    char inputLayout[5] = {'T', 'N', 'D', 0};
    double scaleValue = 1.0f;
    int64_t selectedBlockSize = 64;
    int64_t selectedBlockCount = 16;
    int32_t headNum = n1;

    std::vector<int64_t> queryShape = {t1, n1, d};
    std::vector<int64_t> keyShape = {t2, n2, d};
    std::vector<int64_t> valueShape = {t2, n2, d};
    std::vector<int64_t> attentionOutShape = {t1, n1, d};
    std::vector<int64_t> attentionOutGradShape = {t1, n1, d};
    std::vector<int64_t> softmaxMaxShape = {t1, n1, 8};
    std::vector<int64_t> softmaxSumShape = {t1, n1, 8};
    std::vector<int64_t> topkIndicesShape = {t1, n2, selectedBlockCount};
    std::vector<int64_t> actualSeqQLenOptionalShape = {b};
    std::vector<int64_t> actualSeqKvLenOptionalShape = {b};
    std::vector<int64_t> dqOutShape = {t1, n1, d};
    std::vector<int64_t> dkOutShape = {t2, n2, d};
    std::vector<int64_t> dvOutShape = {t2, n2, d};

    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;
    void* attentionOutGradDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* topkIndicesDeviceAddr = nullptr;
    void* dqOutDeviceAddr = nullptr;
    void* dkOutDeviceAddr = nullptr;
    void* dvOutDeviceAddr = nullptr;

    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* value = nullptr;
    aclTensor* attentionOut = nullptr;
    aclTensor* attentionOutGrad = nullptr;
    aclTensor* softmaxMax = nullptr;
    aclTensor* softmaxSum = nullptr;
    aclTensor* topkIndices = nullptr;
    aclTensor* dqOut = nullptr;
    aclTensor* dkOut = nullptr;
    aclTensor* dvOut = nullptr;

    std::vector<aclFloat16> queryHostData(GetShapeSize(queryShape), 2);
    std::vector<aclFloat16> keyHostData(GetShapeSize(keyShape), 2);
    std::vector<aclFloat16> valueHostData(GetShapeSize(valueShape), 2);
    std::vector<aclFloat16> attentionOutHostData(GetShapeSize(attentionOutShape), 2);
    std::vector<aclFloat16> attentionOutGradHostData(GetShapeSize(attentionOutGradShape), 2);
    std::vector<float> softmaxMaxHostData(GetShapeSize(softmaxMaxShape), 2);
    std::vector<float> softmaxSumHostData(GetShapeSize(softmaxSumShape), 2);
    std::vector<int32_t> topkIndicesHostData(GetShapeSize(topkIndicesShape), 1);
    std::vector<aclFloat16> dqOutHostData(GetShapeSize(dqOutShape), 2);
    std::vector<aclFloat16> dkOutHostData(GetShapeSize(dkOutShape), 2);
    std::vector<aclFloat16> dvOutHostData(GetShapeSize(dvOutShape), 2);

    for (int32_t i = 0; i < topkIndicesHostData.size(); i++) {
        topkIndicesHostData[i] = i;
    }

    // Create a query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an attentionOut aclTensor.
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an attentionOutGrad aclTensor.
    ret = CreateAclTensor(attentionOutGradHostData, attentionOutGradShape, &attentionOutGradDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOutGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a softmaxMax aclTensor.
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a softmaxSum aclTensor.
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a topkIndices aclTensor.
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    int64_t tempQ[1] = {1};
    int64_t tempK[1] = {1024};
    aclIntArray* actualSeqQLenOptional = aclCreateIntArray(tempQ, static_cast<uint64_t>(1));
    aclIntArray* actualSeqKvLenOptional = aclCreateIntArray(tempK, static_cast<uint64_t>(1));
    // Create a dq aclTensor.
    ret = CreateAclTensor(dqOutHostData, dqOutShape, &dqOutDeviceAddr, aclDataType::ACL_FLOAT16, &dqOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dk aclTensor.
    ret = CreateAclTensor(dkOutHostData, dkOutShape, &dkOutDeviceAddr, aclDataType::ACL_FLOAT16, &dkOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dv aclTensor.
    ret = CreateAclTensor(dvOutHostData, dvOutShape, &dvOutDeviceAddr, aclDataType::ACL_FLOAT16, &dvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnNsaSelectedAttentionGrad API call example
    // 3. Call the CANN operator library API. Change the API name to the actual one.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnNsaSelectedAttentionGrad.
    ret = aclnnNsaSelectedAttentionGradGetWorkspaceSize(query, key, value, attentionOut, attentionOutGrad, softmaxMax,
                                                        softmaxSum, topkIndices, actualSeqQLenOptional,
                                                        actualSeqKvLenOptional, nullptr, scaleValue, selectedBlockSize,
                                                        selectedBlockCount, headNum, inputLayout, sparseMode,
                                                        dqOut, dkOut, dvOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnNsaSelectedAttentionGrad.
    ret = aclnnNsaSelectedAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto dqSize = GetShapeSize(dqOutShape);
    std::vector<aclFloat16> dqResultData(dqSize, 0);
    ret = aclrtMemcpy(dqResultData.data(), dqResultData.size() * sizeof(dqResultData[0]), dqOutDeviceAddr,
                      dqSize * sizeof(dqResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dq from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dqSize; i++) {
        LOG_PRINT("result dq[%ld] is: %f\n", i, dqResultData[i]);
    }

    auto dkSize = GetShapeSize(dkOutShape);
    std::vector<aclFloat16> dkResultData(dkSize, 0);
    ret = aclrtMemcpy(dkResultData.data(), dkResultData.size() * sizeof(dkResultData[0]), dkOutDeviceAddr,
                      dkSize * sizeof(dkResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dk from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dkSize; i++) {
        LOG_PRINT("result dk[%ld] is: %f\n", i, dkResultData[i]);
    }

    auto dvSize = GetShapeSize(dvOutShape);
    std::vector<aclFloat16> dvResultData(dkSize, 0);
    ret = aclrtMemcpy(dvResultData.data(), dvResultData.size() * sizeof(dvResultData[0]), dkOutDeviceAddr,
                      dvSize * sizeof(dvResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dv from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dvSize; i++) {
        LOG_PRINT("result dv[%ld] is: %f\n", i, dkResultData[i]);
    }

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(attentionOut);
    aclDestroyTensor(attentionOutGrad);
    aclDestroyTensor(softmaxMax);
    aclDestroyTensor(softmaxSum);
    aclDestroyTensor(topkIndices);
    aclDestroyTensor(dqOut);
    aclDestroyTensor(dkOut);
    aclDestroyTensor(dvOut);
    aclDestroyIntArray(actualSeqQLenOptional);
    aclDestroyIntArray(actualSeqKvLenOptional);
    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(attentionOutDeviceAddr);
    aclrtFree(attentionOutGradDeviceAddr);
    aclrtFree(softmaxMaxDeviceAddr);
    aclrtFree(softmaxSumDeviceAddr);
    aclrtFree(topkIndicesDeviceAddr);
    aclrtFree(dqOutDeviceAddr);
    aclrtFree(dkOutDeviceAddr);
    aclrtFree(dvOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
