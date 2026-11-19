# aclnnNsaCompress

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:56:33.661Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description: In training use cases, uses the Nsa Compress algorithm to reduce the attention computation of long contexts by compressing along the KV sequence dimension.

- Formula:

    Each batch computes complete windows with W=compressBlockSize and S=compressStride as follows:

$$
\mathrm{output}_{i,n,d}=\sum_{j=0}^{W-1}\mathrm{input}_{iS+j,n,d}\cdot\mathrm{weight}_{j,n}
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnNsaCompressGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call "aclnnNsaCompress" to execute the computation.

```c++
aclnnStatus aclnnNsaCompressGetWorkspaceSize(
  const aclTensor   *input,
  const aclTensor   *weight,
  const aclIntArray *actSeqLenOptional,
  char              *layoutOptional,
  int64_t            compressBlockSize,
  int64_t            compressStride,
  int64_t            actSeqLenType,
  aclTensor         *output,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaCompress(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnNsaCompressGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
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
        <td>Input</td>
        <td>Input</td>
        <td>Indicates the tensor to be compressed.</td>
        <td>          <ul>
            <li>Empty tensor not supported.</li>
            <li>The data type must be the same as the weight data type.</li>
            <li>Supported shape: [T, N, D].</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Indicates the compression weight.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The data type must be the same as that of input.</li>
            <li>Shape [compressBlockSize,N], broadcast along D within each window.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>actSeqLenOptional</td>
        <td>Input</td>
        <td>Cumulative sequence lengths of all batches.</td>
        <td>          <ul>
            <li>Currently, this parameter cannot be empty.</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>layoutOptional</td>
        <td>Input</td>
        <td>Indicates the data format of input.</td>
        <td>          <ul>
            <li>Currently, only TND is supported.</li>
          </ul>
        </td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressBlockSize</td>
        <td>Input</td>
        <td>Size of the compression sliding window.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>Input</td>
        <td>Interval size between two compression sliding windows.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actSeqLenType</td>
        <td>Input</td>
        <td>Indicates the numeric type of actSeqLenOptional.</td>
        <td>          <ul>
            <li>Only 0 (cumulative lengths) is supported.</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>Output</td>
        <td>Output</td>
        <td>Compressed result.</td>
        <td>          <ul>
            <li>Empty tensor not supported.</li>
            <li>The data type must be the same as that of input.</li>
            <li>Shape [T_compressed, N, D], where T_compressed is the total number of complete windows across batches.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be requested on the Device.</td>
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

  - The input data layout can be interpreted from multiple dimensions: B (Batch) indicates input sample batch size, S (Seq-Length) indicates input sample sequence length, H (Head-Size) indicates hidden layer size, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the size of the smallest unit of the hidden layer, satisfying D=H/N; T denotes data tightly arranged on the merged axis of B and S (actSeqLen of each batch), where B (Batch) indicates input sample batch size, S (Seq-Length) indicates input sample sequence length, H (Head-Size) indicates hidden layer size, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the size of the smallest unit of the hidden layer, satisfying D=H/N.

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>input, weight, actSeqLenOptional, or output is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of input or weight is out of the supported range.</td>
    </tr>
    <tr>
      <td>The ranks of input/weight or the shape of weight are invalid.</td>
    </tr>
    <tr>
      <td>layoutOptional is invalid.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaCompress

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Workspace size allocated on the device. It is obtained by the first-phase API aclnnNsaCompressGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNsaCompress defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- weight is broadcast along D within each window, with input.shape[1]=weight.shape[1]. Empty inputs for input and weight are not supported.
- actSeqLenType supports only the value 0 currently, which means actSeqLenOptional must be in prefix-sum mode.
- Empty actSeqLenOptional is not supported currently.
- Currently, layoutOptional supports only TND. In this case, input.shape[0] must equal actSeqLenOptional[-1].
- input.shape[1]=weight.shape[1] must be less than or equal to 128.
- input.shape[2] must be a multiple of 16, with an upper limit of 256.
- weight.shape[0]=compressBlockSize must be a multiple of 16, with an upper limit of 128.
- compressStride must be a multiple of 16, and satisfy 16<=compressStride<=compressBlockSize.

## Example

The sample code for the call is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_compress.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
    {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<aclFloat16> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++)
    {
        LOG_PRINT("compressed result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // Boilerplate: resource initialization
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from host to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                            shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}
int main()
{
    // 1. (Boilerplate) Initialize device/context/stream. For details, see the list of AscendCL external APIs.
    // Set deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Customize the construction based on the API.
    void *inputDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclIntArray *actSeqLenOptional = nullptr;
    aclTensor *output = nullptr;

    // Customize inputs and attributes.
    int64_t compressBlockSize = 32;
    int64_t compressStride = 32;
    int64_t actSeqLenType = 0; // 0: prefix sum mode.
    char *layout = "TND";
    int32_t batchSize = 1;
    int32_t sampleLen = 64;
    int32_t headNum = 4;
    int32_t headDim = 32;

    std::vector<int64_t> inputShape = {batchSize * sampleLen, headNum, headDim};
    std::vector<int64_t> weightShape = {compressBlockSize, headNum};
    std::vector<int64_t> actSeqShape = {batchSize};
    std::vector<aclFloat16> inputHostData(batchSize * sampleLen * headNum * headDim);
    std::vector<aclFloat16> weightHostData(compressBlockSize * headNum);
    std::vector<int64_t> actSeqHostData(batchSize);

    for (int i = 0; i < inputHostData.size(); i++)
    {
        inputHostData[i] = aclFloatToFloat16(1.0);
    }
    for (int i = 0; i < weightHostData.size(); i++)
    {
        weightHostData[i] = aclFloatToFloat16(1.0);
    }

    int outputNum = 0;
    int preActSeqLen = 0;
    for (int i = 0; i < batchSize; i++)
    {
        actSeqHostData[i] = sampleLen + preActSeqLen;
        preActSeqLen = actSeqHostData[i];
        if (sampleLen >= compressBlockSize)
        {
            outputNum += (sampleLen - compressBlockSize) / compressStride + 1;
        }
    }

    std::vector<int64_t> outputShape = {outputNum, headNum, headDim};
    std::vector<aclFloat16> outputHostData(outputNum * headNum * headDim);

    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    actSeqLenOptional = aclCreateIntArray(actSeqHostData.data(), actSeqHostData.size());

    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace the API name with the actual one.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // Call the first-phase API aclnnNsaCompressGetWorkspaceSize.
    ret = aclnnNsaCompressGetWorkspaceSize(input, weight, actSeqLenOptional, layout, compressBlockSize, compressStride,
                                        actSeqLenType, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the aclnnNsaCompress second-phase API.
    ret = aclnnNsaCompress(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompress failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the result from the device memory to the host memory. Modify it according to the specific API definition.
    PrintOutResult(outputShape, &outputDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyIntArray(actSeqLenOptional);
    aclDestroyTensor(output);

    // 7. Release device resources.
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    // aclrtFree(actSeqDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
