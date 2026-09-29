# aclnnNsaCompress

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|     x      |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|     √      |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Implements compression in the KV sequence dimension in the training scenario by leveraging NSA Compress algorithm to reduce long-context attention computation.

- Formula:

    The forward propagation formula of NSA Compress is as follows:

$$
\tilde{K}_t^{\text{cmp}} = f_K^{\text{cmp}}(k_{:t}) = \left\{ \varphi(k_{id+1:id+l}) \bigg| 0 \leq i \leq \left\lfloor \frac{t-l}{d} \right\rfloor \right\}
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNsaCompressGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, `aclnnNsaCompress` is called to perform computation.

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
        <td>input</td>
        <td>Input</td>
        <td>Tensor to be compressed.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The data type must be the same as that of <code>weight</code>.</li>
            <li>The shape can be <code>[T, N, D]</code>.</li>
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
        <td>Compression weight.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The data type must be the same as that of <code>input</code>.</li>
            <li>The shapes of <code>weight</code> and <code>input</code> must meet the <code>broadcast</code> operation relationship.</li>
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
        <td><code>S</code> corresponding to each batch.</td>
        <td>
          <ul>
            <li>Cannot be left empty.</li>
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
        <td>Layout of <code>input</code>.</td>
        <td>
          <ul>
            <li>The value can be <code>BSH</code>, <code>SBH</code>, <code>BSND</code>, <code>BNSD</code>, or <code>TND</code>.</li>
            <li>Currently, only <code>TND</code> is supported.</li>
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
        <td>Sliding window interval between two compressions.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actSeqLenType</td>
        <td>Input</td>
        <td><code>actSeqLenOptional</code> value type.</td>
        <td>
          <ul>
            <li>The value can be <code>0</code> or <code>1</code>.</li>
            <li><code>0</code>: The value is the cumulative sum result. <code>1</code>: The value is the sequence size of each batch.</li>
            <li>Currently, only <code>0</code> is supported.</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>output</td>
        <td>Output</td>
        <td>Compressed result.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The data type must be the same as that of <code>input</code>.</li>
            <li>The shape can be <code>[T, N, D]</code>.</li>
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

  - The data layout of `input` can be interpreted from multiple dimensions. To be specific, `B` (`Batch`) indicates the size of an input sample batch, `S` (`Seq-Length`) indicates the length of the input sample sequence, `T` indicates the total length of all input sample sequences (`actSeqLen` of all batches), `H` (`Head-Size`) indicates the size of the hidden layer, `N` (`Head-Num`) indicates the number of heads, and `D` (`Head-Dim`) indicates the minimum unit size of the hidden layer (`D` = `H`/`N`).

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
      <td><code>input</code>, <code>weight</code>, <code>actSeqLenOptional</code>, or <code>output</code> is a null pointer.</td>
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

## aclnnNsaCompress

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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnNsaCompressGetWorkspaceSize</code>.</td>
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
  - `aclnnNsaCompress` defaults to a deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- `input` and `weight` must meet the broadcast relationship, that is `input.shape[1]` = `weight.shape[1]`. `input` and `weight` cannot be empty.
- Currently, `actSeqLenType` can only be set to `0`, that is, the cumulative sum mode is used for `actSeqLenOptional`.
- Currently, actSeqLenOptional cannot be empty.
- Currently, `layoutOptional` supports only `TND`. In this case, `input.shape[0]` must be equal to `actSeqLenOptional[-1]`.
- `input.shape[1]` must be equal to `weight.shape[1]`, and their value must be less than or equal to 128.
- `input.shape[2]` must be a multiple of 16, and cannot exceed 256.
- `weight.shape[0]` must be equal to `compressBlockSize`. Their value must be a multiple of 16, and cannot exceed 128.
- `compressStride` must be an integer multiple of 16, and `compressBlockSize` must be greater than or equal to `compressStride`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
        LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // (Fixed writing) Initialize resources.
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

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                            shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}
int main()
{
    // 1. (Fixed writing) Initialize the device, context, and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    void *inputDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclIntArray *actSeqLenOptional = nullptr;
    aclTensor *output = nullptr;

    // Custom inputs and attributes.
    int64_t compressBlockSize = 32;
    int64_t compressStride = 32;
    int64_t actSeqLenType = 0; // 0 indicates the cumulative sum mode, and 1 indicates the count mode.
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
        if (actSeqLenType == 0)
        {
            actSeqHostData[i] = sampleLen + preActSeqLen;
            preActSeqLen = actSeqHostData[i];
        }
        else if (actSeqLenType == 1)
        {
            actSeqHostData[i] = sampleLen;
        }
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

    // 3. Call the CANN operator library API. Change the API name to the actual one.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // Call the first-phase API aclnnNsaCompressGetWorkspaceSize.
    ret = aclnnNsaCompressGetWorkspaceSize(input, weight, actSeqLenOptional, layout, compressBlockSize, compressStride,
                                        actSeqLenType, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0)
    {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API aclnnNsaCompress.
    ret = aclnnNsaCompress(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompress failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    PrintOutResult(outputShape, &outputDeviceAddr);

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
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
