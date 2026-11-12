# aclnnNsaSelectedAttention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:04:32.186Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
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

- Description: Implements the selected-attention calculation in the NativeSparseAttention algorithm for training scenarios.

- Formula:
  The forward formula for selected-attention is as follows:

  $$
  selected\_key = Gather(key, topk\_indices[i]),0<=i<selected\_block\_count \\
  selected\_value = Gather(value, topk\_indices[i]),0<=i<selected\_block\_count
  $$

  $$
  attention\_out = Softmax(Mask(scale * (query @ selected\_key^T), atten\_mask)) @ selected\_value
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnNsaSelectedAttentionGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation flow, and then call "aclnnNsaSelectedAttention" to execute the computation.

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
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Shape</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>Input</td>
        <td>query in the formula.</td>
        <td>The data type must be the same as key/value.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>key in the formula.</td>
        <td>The data type must be the same as query/value.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>value in the formula.</td>
        <td>The data type must be the same as query/key.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topkIndices</td>
        <td>Input</td>
        <td>topk_indices in the formula.</td>
        <td>The shape must be [T_q, N_kv, selected_block_count], indicating the indices of the selected data.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>atten_mask in the formula.</td>
        <td>          <ul>
            <li>true/1 indicates that the tensor is excluded from the calculation.</li>
            <li>false/0 indicates that the tensor is included in the calculation.</li>
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
        <td>Represents the cumulative sum length of query S for each batch.</td>
        <td>Required for the TND layout. Enter nullptr in other use cases.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>Input</td>
        <td>Represents the cumulative sum length of key/value S for each batch.</td>
        <td>Required for TND layout; set to nullptr in other use cases.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>scale in the formula, representing the scaling coefficient.</td>
        <td>Generally set to D^-0.5, where D is the head dimension of query.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Represents the number of heads.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Specifies the data layout format of query/key/value.</td>
        <td>Currently, only TND is supported.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>selectedBlockSize</td>
        <td>Input</td>
        <td>Length of each block in the select.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>selectedBlockCount</td>
        <td>Input</td>
        <td>Number of select blocks.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Specifies the sparse mode.</td>
        <td>Supported values: 0 or 2.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>Output</td>
        <td>Maximum intermediate result of the Softmax computation.</td>
        <td>Used for backward computation.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>Output</td>
        <td>Intermediate Sum result of the Softmax computation.</td>
        <td>Used for backward computation.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>Final output of the formula.</td>
        <td>Data type is the same as query.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
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
        <td>Returns the op executor, which contains the operator computation flow.</td>
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

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input parameter is a required input, output, or attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types and data formats of query, key, value, attenMaskOptional, softmaxMaxOut, softmaxSumOut, and attentionOut are not in the supported range.</td>
    </tr>
    <tr>
      <td>The input type of inputLayout is not in the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaSelectedAttention

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
      <td>Pointer to the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnNsaSelectedAttentionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNsaSelectedAttention defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- The batch size of the input query, key, and value must be equal, that is, the passed actualSeqQLenOptional and actualSeqKvLenOptional must have the same length.
- The head dimension D of the input query, key, and value must satisfy (D_q == D_k &amp;&amp; D_k >= D_v).
- The data types of the input query, key, and value must be consistent.
- The inputLayout of the input query, key, and value must be consistent.
- sparseMode currently supports 0 and 2.
- selectedBlockSize supports ≤ 128 and must be an integer multiple of 16.
- selectedBlockCount: Supported range: [1-128]. The total selected size `selectedBlockCount * selectedBlockSize` must be < 128*64(8K)
- When Layout is TND, S2 of each batch must be greater than the total selected size `selectedBlockCount * selectedBlockSize`.
- inputLayout currently supports only TND.
- Supports unequal N of the input query and N of key/value, but they must be in a proportional relationship, that is, N_q / N_kv must be a non-zero integer, called G (group), and G must satisfy G <= 32.
- When the attenMaskOptional input is nullptr, the sparseMode parameter does not take effect and is fixed to full computation.
- For data shape constraints, take TND of inputLayout as an example (Note: T equals the cumulative sum of S lengths across batches. When S is equal across batches, T = B * S). Where:

  - B (Batchsize): The value range is 1 to 1024.
  - N (Head-Num): The value range is 1 to 128.
  - G (Group): The value range is 1 to 32.
  - S (Seq-Length): The value range is 1 to 128K. It must also satisfy S_kv >= selectedBlockSize * selectedBlockCount, and the S_kv length must be an integer multiple of selectedBlockSize.
  - D (Head-Dim): D_qk=192, D_v=128.

## Example

The following sample code shows how to call the operator through aclnn. It is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
    // Boilerplate: resource initialization
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

    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
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
    // Release aclTensor and aclScalar. Modify the code according to the specific API definition.
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
    // 1. (Boilerplate) device/context/stream initialization. For details, see the AscendCL public API list.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.
    // If you need to modify the shape values, also update the
    // query, key, and value shape values in the test_nsa_selected_attention branch of ../scripts/fa_generate_data.py, run gen data again, and then execute.
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
    int64_t sparseMode = 2;
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

    // Create the acl Tensor.
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

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    aclOpExecutor *executor;

    // Call the first-phase API of aclnnNsaSelectedAttention.
    ret = aclnnNsaSelectedAttentionGetWorkspaceSize(
        q, k, v, topKIndices, attenMaskOptional, actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue, headNum,
        layOut, sparseMode, selectedBlockSize, selectedBlockCount, softmaxMax, softmaxSum, attentionOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
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

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
                  attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
                  deviceId, &context, &stream);
              return ret);

    // 5. Obtain the output values. Copy the results from the device memory to the host side. Modify based on the specific API definition.
    CopyOutResult<float>(0, softmaxMaxShape, &softmaxMaxDeviceAddr);
    CopyOutResult<float>(1, softmaxSumShape, &softmaxSumDeviceAddr);
    CopyOutResult<aclFloat16>(2, attentionOutShape, &attentionOutDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition; release device resources.
    FreeResource(q, k, v, attentionOut, softmaxMax, softmaxSum, qDeviceAddr, kDeviceAddr, vDeviceAddr,
        attentionOutDeviceAddr, softmaxMaxDeviceAddr, softmaxSumDeviceAddr, workspaceSize, workspaceAddr,
        deviceId, &context, &stream);

    return 0;
}
```
