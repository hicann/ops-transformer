# aclnnNsaSelectedAttention

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

- Description: Implements selected attention computation in the Native Sparse Attention (NSA) algorithm for training scenarios.

- Formulas:
  The forward propagation formulas for selected attention are as follows:

  $$
  selected\_key = Gather(key, topk\_indices[i]),0<=i<selected\_block\_count \\
  selected\_value = Gather(value, topk\_indices[i]),0<=i<selected\_block\_count
  $$

  $$
  attention\_out = Softmax(Mask(scale * (query @ selected\_key^T), atten\_mask)) @ selected\_value
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNsaSelectedAttentionGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnNsaSelectedAttention` is called to perform computation.

```c++
aclnnStatus aclnnNsaSelectedAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *topkIndices,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *actualSeqQLenOptional,
  const aclIntArray *actualSeqKvLenOptional,
  double             scaleValue,
  int64_t            headNum,
  char              *inputLayout,
  int64_t            sparseMode,
  int64_t            selectedBlockSize,
  int64_t            selectedBlockCount,
  const aclTensor   *softmaxMaxOut,
  const aclTensor   *softmaxSumOut,
  const aclTensor   *attentionOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaSelectedAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnNsaSelectedAttentionGetWorkspaceSize

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
      <th>Usage Notes</th>
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
      <td>The data type must be the same as that of <code>key</code>/<code>value</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3–4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td><code>key</code> in the formula.</td>
      <td>The data type must be the same as that of <code>query</code>/<code>value</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3–4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td><code>value</code> in the formula.</td>
      <td>The data type must be the same as that of <code>query</code>/<code>key</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3–4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>topkIndices</td>
      <td>Input</td>
      <td><code>topk_indices</code> in the formula.</td>
      <td>The shape must be <code>[T_q, N_kv, selected_block_count]</code>, indicating the index of the selected data.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td><code>atten_mask</code> in the formula.</td>
      <td>
        <ul>
          <li>The value <code>true</code>/<code>1</code> indicates that the parameter is not involved in the computation.</li>
          <li>The value <code>false</code>/<code>0</code> indicates that the parameter is involved in the computation.</li>
        </ul>
      </td>
      <td>BOOL, UINT8</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqQLenOptional</td>
      <td>Input</td>
      <td>Cumulative sum of <code>S</code> of all batches in <code>query</code>.</td>
      <td>This parameter is required for <code>TND</code> layout. In other scenarios, input <code>nullptr</code>.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqKvLenOptional</td>
      <td>Input</td>
      <td>Cumulative sum of <code>S</code> of all batches in <code>key</code>/<code>value</code>.</td>
      <td>This parameter is required for <code>TND</code> layout. In other scenarios, input <code>nullptr</code>.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td><code>scale</code> in the formula, indicating the scaling coefficient.</td>
      <td>Generally, it is set to <code>D<sup>–0.5</sup></code>, where <code>D</code> indicates the head dimension of <code>query</code>.</td>
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
      <td>Size of each selected block.</td>
      <td>The value must be less than or equal to <code>128</code> and an integer multiple of 16.</td>
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
      <td>The value can be <code>0</code> or <code>2</code>.</td>
      <td>INT64</td>
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
      <td>The data type must be the same as that of <code>query</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3–4</td>
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

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 283px">
  <col style="width: 120px">
  <col style="width: 747px">
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
      <td>The data type or data format of <code>query</code>, <code>key</code>, <code>value</code>, <code>attenMaskOptional</code>, <code>softmaxMaxOut</code>, <code>softmaxSumOut</code>, or <code>attentionOut</code> is not supported.</td>
    </tr>
    <tr>
      <td>The input type of <code>inputLayout</code> is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaSelectedAttention

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnNsaSelectedAttentionGetWorkspaceSize</code>.</td>
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
  - `aclnnNsaSelectedAttention` defaults to a deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- `Batchsize` of the input `query`, `key`, and `value` must be the same. That is, the values of the input `actualSeqQLenOptional` and `actualSeqKvLenOptional` must be the same.
- `D` (`Head-Dim`) of the input `query`, `key`, and `value` must satisfy `D_q == D_k && D_k >= D_v`.
- The data types of the input `query`, `key`, and `value` must be the same.
- `inputLayout` of the input `query`, `key`, and `value` must be the same.
- The value range of `selectedBlockCount` is [1, 128]. The total size of the selected blocks (`selectedBlockCount * selectedBlockSize`) must be less than `128*64` (8K).
- When the layout is `TND`, `S2` of each batch must be greater than `selectedBlockCount * selectedBlockSize`.
- The `N` values of the input `query` and `key`/`value` can be different, but `N_q/N_kv` must be a non-zero integer, which is called `G` (`Group`), and `G` must be less than or equal to `32`.
- If `attenMaskOptional` is `nullptr`, the `sparseMode` parameter does not take effect and all tokens are computed.
- The following uses the `inputLayout` `TND` as an example to describe the restrictions on the data shape. (Note: T is the sum of `S` in all batches. When `S` in each batch is the same, `T` = `B*S`.)
  
  - `B` (`Batchsize`): The value ranges from 1 to 1024.
  - `N` (`Head-Num`): The value ranges from 1 to 128.
  - `G` (`Group`): The value ranges from 1 to 32.
  - `S` (`Seq-Length`): The value ranges from 1 to 128K. In addition, `S_kv` must be greater than or equal to the product of `selectedBlockSize` and `selectedBlockCount`, and be an integer multiple of `selectedBlockSize`.
  - `D` (`Head-Dim`): `D_qk` is `192` and `D_v` is `128`.

## Example

The following single-aclnn-operator calling example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_selected_attention.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T> void CopyOutResult(int64_t outIndex, std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<T> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    if(outIndex == 2) {
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("attention out result is: %f\n", i, resultData[i]);
        }
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // (Boilerplate) Initialize resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
        aclFinalize(); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret);
        aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtDestroyContext(context); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void FreeResource(aclTensor *q, aclTensor *k, aclTensor *v, aclTensor *attentionOut, aclTensor *softmaxMax,
    aclTensor *softmaxSum, void *qDeviceAddr, void *kDeviceAddr, void *vDeviceAddr, void *attentionOutDeviceAddr,
    void *softmaxMaxDeviceAddr, void *softmaxSumDeviceAddr, uint64_t workspaceSize, void *workspaceAddr,
    int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // Destroy aclTensors and aclScalars. Modify the code based on the API definition.
    if (q != nullptr) {
        aclDestroyTensor(q);
    }
    if (k != nullptr) {
        aclDestroyTensor(k);
    }
    if (v != nullptr) {
        aclDestroyTensor(v);
    }
    if (attentionOut != nullptr) {
        aclDestroyTensor(attentionOut);
    }
    if (softmaxMax != nullptr) {
        aclDestroyTensor(softmaxMax);
    }
    if (softmaxSum != nullptr) {
        aclDestroyTensor(softmaxSum);
    }

    // Release device resources.
    if (qDeviceAddr != nullptr) {
        aclrtFree(qDeviceAddr);
    }
    if (kDeviceAddr != nullptr) {
        aclrtFree(kDeviceAddr);
    }
    if (vDeviceAddr != nullptr) {
        aclrtFree(vDeviceAddr);
    }
    if (attentionOutDeviceAddr != nullptr) {
        aclrtFree(attentionOutDeviceAddr);
    }
    if (softmaxMaxDeviceAddr != nullptr) {
        aclrtFree(softmaxMaxDeviceAddr);
    }
    if (softmaxSumDeviceAddr != nullptr) {
        aclrtFree(softmaxSumDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (context != nullptr) {
        aclrtDestroyContext(context);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main()
{
    // 1. (Boilerplate) Initialize the device, context, and stream. For details, see the AscendCL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    // If you need to modify shape values, modify the shape values corresponding to query, key, and value generated in the test_nsa_selected_attention branch
    // in ../scripts/fa_generate_data.py, regenerate the data, and then execute the test script.
    int64_t batch = 2;
    int64_t s1 = 512;
    int64_t s2 = 2048;
    int64_t d1 = 192;
    int64_t d2 = 128;
    int64_t g = 4;
    int64_t n2 = 4;
    std::vector<int64_t> qShape = {batch * s1, n2 * g, d1};
    std::vector<int64_t> kShape = {batch * s2, n2, d1};
    std::vector<int64_t> vShape = {batch * s2, n2, d2};
    std::vector<int64_t> topKIndicesShape = {batch * s1, n2, 16};
    std::vector<int64_t> attentionOutShape = {batch * s1, n2 * g, d2};
    std::vector<int64_t> softmaxMaxShape = {batch * s1, n2 * g, 8};
    std::vector<int64_t> softmaxSumShape = {batch * s1, n2 * g, 8};
    
    double scaleValue = 1.0;
    int64_t headNum = 16;
    int64_t selectedBlockSize = 64;
    int64_t selectedBlockCount = 16;
    int64_t sparseMod = 2;
    char layOut[] = "TND";

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *topKIndicesDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void *softmaxMaxDeviceAddr = nullptr;
    void *softmaxSumDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *topKIndices = nullptr;
    aclTensor *attenMaskOptional = nullptr;
    aclTensor *softmaxMax = nullptr;
    aclTensor *softmaxSum = nullptr;
    aclTensor *attentionOut = nullptr;

    std::vector<int64_t> actualSeqQLenVec = {512, 1024};
    std::vector<int64_t> actualSeqKvLenVec = {2048, 4096};
    aclIntArray *actualSeqQLenOptional = aclCreateIntArray(actualSeqQLenVec.data(), actualSeqQLenVec.size());
    aclIntArray *actualSeqKvLenOptional = aclCreateIntArray(actualSeqKvLenVec.data(), actualSeqKvLenVec.size());

    std::vector<aclFloat16> qHostData(GetShapeSize(qShape), 1);
    std::vector<aclFloat16> kHostData(GetShapeSize(kShape), 1);
    std::vector<aclFloat16> vHostData(GetShapeSize(vShape), 1);
    std::vector<int32_t> topkIndicesHostData(GetShapeSize(topKIndicesShape), 2);
    std::vector<float> attentionOutHostData(GetShapeSize(attentionOutShape), 0.0);
    std::vector<float> softmaxMaxHostData(GetShapeSize(softmaxMaxShape), 0.0);
    std::vector<float> softmaxSumHostData(GetShapeSize(softmaxSumShape), 0.0);
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    // Create an aclTensor.
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(topkIndicesHostData, topKIndicesShape, &topKIndicesDeviceAddr, aclDataType::ACL_INT32, &topKIndices);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16,
                          &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 3. Call the CANN operator library API. Change the API name to the actual one.
    aclOpExecutor *executor;

    // Call the first-phase API of aclnnNsaSelectedAttention.
    ret = aclnnNsaSelectedAttentionGetWorkspaceSize(
        q, k, v, topKIndices, attenMaskOptional, actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue, headNum,
        layOut, sparseMod, selectedBlockSize, selectedBlockCount, softmaxMax, softmaxSum, attentionOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                deviceId, &context, &stream);
            return ret);
    }

    // Call the second-phase API of aclnnNsaSelectedAttention.
    ret = aclnnNsaSelectedAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttention failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    CopyOutResult<float>(0, softmaxMaxShape, &softmaxMaxDeviceAddr);
    CopyOutResult<float>(1, softmaxSumShape, &softmaxSumDeviceAddr);
    CopyOutResult<aclFloat16>(2, attentionOutShape, &attentionOutDeviceAddr);

    // 6. Destroy aclTensors and aclScalars. Modify the code based on the API definition. Release device resources.
    FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
        attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
        deviceId, &context, &stream);

    return 0;
}
```
