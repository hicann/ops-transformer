# aclnnNsaCompressGrad

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Performs backward propagation of the aclnnNsaCompress operator.

- Formulas:
  The forward propagation formulas for selected attention are as follows:

    $$
    \text{dw} = \text{dk\_cmp} \cdot K^\top
    $$

    $$
    \text{dk} = W^\top \cdot \text{dk\_cmp}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNsaCompressGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnNsaCompressGrad` is called to perform computation.

```c++
aclnnStatus aclnnNsaCompressGradGetWorkspaceSize(
  const aclTensor   *outputGrad,
  const aclTensor   *input,
  const aclTensor   *weight,
  const aclIntArray *actSeqLenOptionalOptional,
  int64_t            compressBlockSize,
  int64_t            compressStride,
  int64_t            actSeqLenType,
  char              *layoutOptionalOptional,
  const aclTensor   *inputGradOut,
  const aclTensor   *weightGradOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaCompressGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnNsaCompressGradGetWorkspaceSize

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
      <td>outputGrad</td>
      <td>Input</td>
      <td>Backward gradient output by the forward operator.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Tensor to be compressed.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Compression weight. Its shape must meet the <code>broadcast</code> operation relationship with the shape of <code>input</code>.</td>
      <td>The data type must be the same as that of <code>input</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actSeqLenOptionalOptional</td>
      <td>Input</td>
      <td><code>S</code> corresponding to each batch. This parameter is required when the sequence lengths of batches are different.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <td>compressBlockSize</td>
      <td>Input</td>
      <td>Size of the compression sliding window.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td></td>
    </tr>
    <tr>
      <td>compressStride</td>
      <td>Input</td>
      <td>Sliding window interval between two compressions.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td></td>
    </tr>
    <tr>
      <td>actSeqLenType</td>
      <td>Input</td>
      <td>The value can be <code>0</code> or <code>1</code>. Currently, only <code>0</code> is supported.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td></td>
    </tr>
    <tr>
      <td>layoutOptionalOptional</td>
      <td>Input</td>
      <td>Layout of <code>input</code>, which can be <code>TND</code>.</td>
      <td>-</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
      <td></td>
    </tr>
    <tr>
      <td>inputGradOut</td>
      <td>Output</td>
      <td>Gradient of <code>input</code>. Its shape must be the same as the shape of <code>input</code>.</td>
      <td>The data type must be the same as that of <code>input</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightGradOut</td>
      <td>Output</td>
      <td>Gradient of <code>weight</code>. Its shape must be the same as the shape of <code>weight</code>.</td>
      <td>The data type must be the same as that of <code>weight</code>.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
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
      <td></td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td></td>
    </tr>
  </tbody>
  </table>

- **Returns**

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
      <td><code>input</code>, <code>weight</code>, <code>outputGrad</code>, <code>inputGrad</code>, or <code>weightGrad</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of <code>input</code> or <code>weight</code> is not supported.</td>
    </tr>
    <tr>
      <td>The shapes of <code>input</code> and <code>weight</code> are not broadcastable.</td>
    </tr>
    <tr>
      <td><code>layoutOptional</code> is invalid.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaCompressGrad

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnNsaCompressGradGetWorkspaceSize</code>.</td>
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

- Deterministic computation:
  - `aclnnNsaCompressGrad` defaults to a deterministic implementation.
- `compressBlockSize` and `compressStride` must be integer multiples of 16 and `compressBlockSize` must be greater than `compressStride`.

## Example

The following single-aclnn-operator calling example (based on Atlas A2 training products) is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_compress_grad.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr)     \
    do {                                 \
        if (!(cond)) {                   \
            return_expr;                 \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
      shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
    // (Boilerplate) Initialize AscendCL.
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
                    aclDataType dataType, aclTensor** xOrResult) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous xOrResult.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
    *xOrResult = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                 shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device, context, and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    int64_t headNum = 64;
    int64_t headDim = 128;
    int64_t blockSize = 32;
    int64_t blockStride = 16;
    int64_t blockNum = 15;
    int64_t seqLensSum = 272;
    int64_t seqLen = 3;
    std::vector<int64_t> outputGradShape = {blockNum, headNum, headDim};
    std::vector<int64_t> inputKVShape = {seqLensSum, headNum, headDim};
    std::vector<int64_t> weightShape = {blockSize, headNum};
    std::vector<int64_t> inputGradOutShape = {seqLensSum, headNum, headDim};
    std::vector<int64_t> weightGradOutShape = {blockSize, headNum};
    int64_t SeqLenType = 0;
    char layOut[] = "TND";

    void* outputGradDeviceAddr = nullptr;
    void* inputKVDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* inputGradOutDeviceAddr = nullptr;
    void* weightGradOutDeviceAddr = nullptr;

    aclTensor* outputGrad = nullptr;
    aclTensor* inputKV = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* inputGradOut = nullptr;
    aclTensor* weightGradOut = nullptr;

    std::vector<float> inputGradOutHostData(seqLensSum * headNum * headDim, 0.0);
    std::vector<float> weightGradOutHostData(blockSize * headNum, 0.0);

    std::vector<float> outputGradHostData(blockNum * headNum * headDim, 1.0);
    std::vector<float> inputKVHostData(seqLensSum * headNum * headDim, 1.0);
    std::vector<float> weightHostData(blockSize * headNum, 1.0);
    std::vector<int64_t> actSeqLenOptionalHostData = {0, 128, 272};

    aclIntArray *actSeqLenOptional = aclCreateIntArray(actSeqLenOptionalHostData.data(), actSeqLenOptionalHostData.size());

    // Create a dy aclTensor.
    ret = CreateAclTensor(outputGradHostData, outputGradShape, &outputGradDeviceAddr, aclDataType::ACL_FLOAT16,
                          &outputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an x aclTensor.
    ret = CreateAclTensor(inputKVHostData, inputKVShape, &inputKVDeviceAddr, aclDataType::ACL_FLOAT16, &inputKV);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gelu aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(inputGradOutHostData, inputGradOutShape, &inputGradOutDeviceAddr, aclDataType::ACL_FLOAT16, &inputGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightGradOutHostData, weightGradOutShape, &weightGradOutDeviceAddr, aclDataType::ACL_FLOAT16, &weightGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change the API name to the actual one.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnNsaCompressGrad.
    ret = aclnnNsaCompressGradGetWorkspaceSize(
        outputGrad, inputKV, weight, actSeqLenOptional, blockSize, blockStride, SeqLenType, layOut,
        inputGradOut, weightGradOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGradWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnNsaCompressGrad.
    ret = aclnnNsaCompressGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(inputGradOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputGradOutDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(outputGrad);
    aclDestroyTensor(inputKV);
    aclDestroyTensor(weight);
    aclDestroyTensor(inputGradOut);
    aclDestroyTensor(weightGradOut);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(outputGradDeviceAddr);
    aclrtFree(inputKVDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(inputGradOutDeviceAddr);
    aclrtFree(weightGradOutDeviceAddr);
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
