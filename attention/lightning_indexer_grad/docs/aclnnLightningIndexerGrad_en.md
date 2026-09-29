# aclnnLightningIndexerGrad

## Product Support

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 inference products</term>|      ×     |
|<term>Atlas A3 training products</term>|      √     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |

## Function

- Description: In the training scenario, the inputs `Query`, `Key`, `Weights`, `Dy`, and `Indices` are required to implement `LightningIndexer` backward computation. The backward computation extracts the `TopK` sequence from `Key` based on `Indices` in forward computation to reduce the MatMul computation workload.

- Formula:
  The formula for `LightningIndexer` backward computation is as follows:

  $$
  S = Relu(Matmul(Query, Gather(Key, Indices)))
  $$

  $$
  Y = Dy*Weights
  $$

  $$
  dW = Reduce(S * dy)
  $$

  $$
  dQ = Matmul(ReluGrad(Y, S), Gather(Key, Indices))
  $$

  $$
  dK = ScatterAdd(Matmul(ReluGrad(Y, S), Q), Indices)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLightningIndexerGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLightningIndexerGrad` is called to perform computation.

```c++
aclnnStatus aclnnLightningIndexerGradGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *dy,
  const aclTensor   *spareIndices,
  const aclTensor   *weights,
  const aclTensor   *actualSeqLengthsQuery,
  const aclTensor   *actualSeqLengthsKey,
  int64_t           headNum,  
  char              *layout,
  int64_t           sparseMode,
  int64_t           preTokens,
  int64_t           nextTokens,
  bool              deterministic,
  const aclTensor   *dQuery,
  const aclTensor   *dKey,
  const aclTensor   *dWeights,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnLightningIndexerGrad(
  void             *workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor    *executor, 
  aclrtStream       stream)
```

## aclnnLightningIndexerGradGetWorkspaceSize

- **Parameters**

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
      <td><code>query</code> in the formulas.</td>
      <td>The shape can be [B, S1, N1, D] or [T1, N1, D].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td><code>key</code> in the formulas.</td>
      <td>The shape can be [B, S2, N2, D] or [T2, N2, D].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td><code>value</code> in the formula.</td>
      <td>The shape can be [B, S1, N1, D] or [T1, N1, D].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sparseIndices</td>
      <td>Input</td>
      <td>sparseIndices in the formula.</td>
      <td>The shape can be [B, S1, K] or [T1, K].</td>
      <td>INT64</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>weights in the formula.</td>
      <td>The shape can be [B, S1, N1] or [T1, N1].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQuery</td>
      <td>Input</td>
      <td>Cumulative sum of <code>S</code> of all batches in <code>query</code>.</td>
      <td>This parameter is required for TND layout. In other scenarios, input <code>nullptr</code>.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKey</td>
      <td>Input</td>
      <td>Accumulated length of each batch S of the key.</td>
      <td>This parameter is required for TND layout. In other scenarios, input <code>nullptr</code>.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
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
      <td>layout</td>
      <td>Input</td>
      <td>Format of the query/key data.</td>
      <td>Currently, TND and BSND are supported.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Sparse mode.</td>
      <td>The value can be <code>0</code> or <code>3</code>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>Input</td>
      <td>Left boundary of the sliding window, used for sparse computation.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>Input</td>
      <td>Right boundary of the sliding window, used for sparse computation.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deterministic</td>
      <td>Input</td>
      <td>Indicates whether deterministic computation is supported.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dQuery</td>
      <td>Output</td>
      <td>dQuery gradient.</td>
      <td>The data type must be the same as that of <code>query</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dKey</td>
      <td>Output</td>
      <td>dKey gradient.</td>
      <td>The data type is the same as that of <code>key</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dWeights</td>
      <td>Output</td>
      <td>dWeights gradient.</td>
      <td>The data type is the same as that of <code>weights</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
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

  The first-phase API implements input parameter validation. The following error codes may be returned.

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
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and formats of query, key, dy, sparseIndices, and weights are not supported.</td>
    </tr>
    <tr>
      <td>The input type of <code>layout</code> is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnLightningIndexerGrad

- **Parameters**

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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnLightningIndexerGradGetWorkspaceSize</code>.</td>
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
  - `aclnnLightningIndexerGrad` defaults to non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- `inputLayout` can be TND or BSND.
- The following uses the BSND layout as an example to describe the restrictions on the data shape. Where:
  
  - `B` (Batchsize): The value ranges from 1 to 1024.
  - `N` (Head-Num): The value is `64`.
  - `G` (Group): The value is `64`.
  - `S1` (Seq-LengthQ): The value ranges from 1 to 128K.
  - `S2` (Seq-LengthK): The value ranges from *topK* to 128K.
  - `D` (Head-Dim): The value is `128`.
  - `TopK`: The value is `2048`.

## Example

The following is an example of aclnn single-operator calling. For details about the compilation and execution process, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_grad.h"

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

template <typename T> 
void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                            *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
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


void FreeResource(aclTensor *q, aclTensor *k, aclTensor *dy, aclTensor *sparseIndices, aclTensor *weights, 
                  aclTensor *dQuery, aclTensor *dKey, aclTensor *dWeights, void *qDeviceAddr, void *kDeviceAddr, 
                  void *dyDeviceAddr, void *sparseIndicesDeviceAddr, void *weightsDeviceAddr, void *dQueryAddr, 
                  void *dKeyAddr, void *dWeightsAddr, uint64_t workspaceSize, void *workspaceAddr, int32_t deviceId, 
                  aclrtContext *context, aclrtStream *stream)
{
    // Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    if (q != nullptr) {
        aclDestroyTensor(q);
    }
    if (k != nullptr) {
        aclDestroyTensor(k);
    }
    if (dy != nullptr) {
        aclDestroyTensor(dy);
    }
    if (sparseIndices != nullptr) {
        aclDestroyTensor(sparseIndices);
    }
    if (weights != nullptr) {
        aclDestroyTensor(weights);
    }
    if (dQuery != nullptr) {
        aclDestroyTensor(dQuery);
    }
    if (dKey != nullptr) {
        aclDestroyTensor(dKey);
    }
    if (dWeights != nullptr) {
        aclDestroyTensor(dWeights);
    }

    // Release device resources.
    if (qDeviceAddr != nullptr) {
        aclrtFree(qDeviceAddr);
    }
    if (kDeviceAddr != nullptr) {
        aclrtFree(kDeviceAddr);
    }
    if (dyDeviceAddr != nullptr) {
        aclrtFree(dyDeviceAddr);
    }
    if (sparseIndicesDeviceAddr != nullptr) {
        aclrtFree(sparseIndicesDeviceAddr);
    }
    if (weightsDeviceAddr != nullptr) {
        aclrtFree(weightsDeviceAddr);
    }
    if (dQueryAddr != nullptr) {
        aclrtFree(dQueryAddr);
    }
    if (dKeyAddr != nullptr) {
        aclrtFree(dKeyAddr);
    }
    if (dWeightsAddr != nullptr) {
        aclrtFree(dWeightsAddr);
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
    // 1. (Fixed writing) Initialize the device, context, and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on the API definition.
    // Query the shape values of <code>query</code>, <code>key</code>, <code>dy</code>, <code>sparseIndices</code>, and <code>weights</code>, regenerate data, and perform execution.
    int64_t batch = 2;
    int64_t s1 = 3;
    int64_t s2 = 2048;
    int64_t d = 128;
    int64_t g = 64;
    int64_t n2 = 1;
    int64_t topK = 2048;

    std::vector<int64_t> qShape = {batch, s1, n2 * g, d};
    std::vector<int64_t> kShape = {batch, s2, n2, d};
    std::vector<int64_t> dyShape = {batch, s1, n2 * g, d};
    std::vector<int64_t> sparseIndicesShape = {batch, s1, topK};
    std::vector<int64_t> weightsShape = {batch, s1, n2 * g};
    std::vector<int64_t> dQueryShape = {batch, s1, n2 * g, d};
    std::vector<int64_t> dKeyShape = {batch, s2, n2, d};
    std::vector<int64_t> dWeightsShape = {batch, s1, n2 * g};

    int64_t headNum = 64;
    int64_t sparseMode = 3;
    char layoutStr[] = "BSND";
    bool deteminstic = true;
    int64_t preToken = 65536;
    int64_t nextToken = 65536;

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *dyDeviceAddr = nullptr;
    void *sparseIndicesDeviceAddr = nullptr;
    void *weightsDeviceAddr = nullptr;
    void *dQueryAddr = nullptr;
    void *dKeyAddr = nullptr;
    void *dWeightsAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *dy = nullptr;
    aclTensor *sparseIndices = nullptr;
    aclTensor *weights = nullptr;
    aclTensor *dQuery = nullptr;
    aclTensor *dKey = nullptr;
    aclTensor *dWeights = nullptr;

    std::vector<aclFloat16> qHostData(GetShapeSize(qShape), 1.0);
    std::vector<aclFloat16> kHostData(GetShapeSize(kShape), 1.0);
    std::vector<aclFloat16> dyHostData(GetShapeSize(dyShape), 1.0);
    std::vector<int32_t> sparseIndicesHostData(GetShapeSize(sparseIndicesShape), 1);
    std::vector<aclFloat16> weightsHostData(GetShapeSize(weightsShape), 1.0);
    std::vector<aclFloat16> dQueryHostData(GetShapeSize(dQueryShape), 1.0);
    std::vector<aclFloat16> dKeyHostData(GetShapeSize(dKeyShape), 1.0);
    std::vector<aclFloat16> dWeightsHostData(GetShapeSize(dWeightsShape), 1.0);

    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    // Create an aclTensor.
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT16, &dy);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &sparseIndicesDeviceAddr, aclDataType::ACL_INT32, &sparseIndices);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(weightsHostData, weightsShape, &weightsDeviceAddr, aclDataType::ACL_FLOAT16, &weights);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    
    ret = CreateAclTensor(dQueryHostData, dQueryShape, &dQueryAddr, aclDataType::ACL_FLOAT16, &dQuery);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dKeyHostData, dKeyShape, &dKeyAddr, aclDataType::ACL_FLOAT16, &dKey);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);
    ret = CreateAclTensor(dWeightsHostData, dWeightsShape, &dWeightsAddr, aclDataType::ACL_FLOAT16, &dWeights);
    CHECK_RET(ret == ACL_SUCCESS,
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the specific one.
    aclOpExecutor *executor;

    // Call the first-phase API of aclnnLightningIndexerGrad.
    ret = aclnnLightningIndexerGradGetWorkspaceSize(
        q, k, dy, sparseIndices, weights, nullptr, nullptr, headNum,
        layoutStr, sparseMode, preToken, nextToken, deteminstic, dQuery, dKey, dWeights, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
            return ret);
    }

    // Call the second-phase API of aclnnLightningIndexerGrad.
    ret = aclnnLightningIndexerGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerGrad failed. ERROR: %d\n", ret);
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    PrintOutResult<aclFloat16>(dQueryShape, &dQueryAddr);
    PrintOutResult<aclFloat16>(dKeyShape, &dKeyAddr);
    PrintOutResult<aclFloat16>(dWeightsShape, &dWeightsAddr);

    // 6. Destroy aclTensors and aclScalars. Modify the code based on the API definition. Release device resources.
    FreeResource(q, k, dy, sparseIndices, weights, dQuery, dKey, dWeights, qDeviceAddr, kDeviceAddr, dyDeviceAddr,
                  sparseIndicesDeviceAddr, weightsDeviceAddr, dQueryAddr, dKeyAddr, dWeightsAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);

    return 0;
}
```
