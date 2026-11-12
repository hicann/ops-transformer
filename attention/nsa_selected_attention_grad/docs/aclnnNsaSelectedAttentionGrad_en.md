# aclnnNsaSelectedAttentionGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:08:30.769Z -->

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

- Description: Rearranges data of a selected size (selectedBlockSize) from key and value based on topkIndices, and then computes the backward output of attention in training scenarios.

- Formula:

  Rearranges data of selectedBlockCount blocks, each of size selectedBlockSize, from key and value based on the input topkIndices. The formula is as follows:

  $$
  selectedKey = Gather(key, topkIndices[i]),0<=i<selectedBlockCount \\
  selectedValue = Gather(value, topkIndices[i]),0<=i<selectedBlockCount
  $$

  Then, the backward computation of the attention mechanism is performed. The formula is:

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnNsaSelectedAttentionGradGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnNsaSelectedAttentionGrad" API to perform the computation.

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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td>query in the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>key in the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>value in the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>Input</td>
      <td>Final output of the attention forward computation.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attentionOutGrad</td>
      <td>Input</td>
      <td>dY in the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxMax</td>
      <td>Input</td>
      <td>Max intermediate result of the Softmax computation.</td>
      <td>Used for backward computation.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSum</td>
      <td>Input</td>
      <td>Sum intermediate result of the Softmax computation.</td>
      <td>Used for backward computation.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>topkIndices</td>
      <td>Input</td>
      <td>topk_indices in the formula.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td>atten_mask in the formula.</td>
      <td>-</td>
      <td>BOOL, UINT8</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqQLenOptional</td>
      <td>Input</td>
      <td>Cumulative sum length of S for each batch of query.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqKvLenOptional</td>
      <td>Input</td>
      <td>Length of the cumulative sum of S for each batch of key/value.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Scaling factor scale.</td>
      <td>Generally D^-0.5.</td>
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
      <td>Data layout format of query/key/value.</td>
      <td>Currently only TND is supported.</td>
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
      <td>Sparse mode.</td>
      <td>Supports 0 or 2.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>Output</td>
      <td>dQ in the formula, indicating the gradient of query.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dkOut</td>
      <td>Output</td>
      <td>dK in the formula, indicating the gradient of keyIn.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dvOut</td>
      <td>Output</td>
      <td>dV in the formula, representing the gradient of value.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[BNSD], [BSND], [BSH], [SBH]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
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
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input parameter is a required input, output, or required attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of query, key, value, attentionOut, attentionOutGrad, dqOut, dkOut, and dvOut are not all the same or are not FLOAT16 or BFLOAT16.</td>
    </tr>
    <tr>
      <td>The data type of softmaxMax or softmaxSum is not FLOAT32, the data type of topkIndices is not INT32, or the data type of attenMaskOptional (when provided) is neither BOOL nor UINT8.</td>
    </tr>
    <tr>
      <td>An attribute value or tensor shape does not meet the constraints. Examples include: a non-null inputLayout other than TND (case-insensitive); sparseMode other than 0 or 2; selectedBlockSize outside [16, 128] or not a multiple of 16; selectedBlockCount outside [1, 128]; query, key, value, attentionOut, attentionOutGrad, softmaxMax, softmaxSum, or topkIndices not being three-dimensional, or a provided attenMaskOptional not being two-dimensional; headNum different from query's N1, N2 not positive, or N1 not divisible by N2; the head dimension of query or value outside {128, 192}, the head dimension of key different from query, the head dimension of value greater than query, or the T dimensions of key and value differing; softmaxMax or softmaxSum having a shape other than [T1, N1, 8], attentionOut having a shape other than [T1, N1, value head dimension], attentionOutGrad having a different shape from attentionOut, or topkIndices having a shape other than [T1, N2, selectedBlockCount]. When query is nonempty and the underlying operator successfully returns results, a view shape of dqOut, dkOut, or dvOut that differs from its corresponding result also returns this error.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaSelectedAttentionGrad

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
      <td>The workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnNsaSelectedAttentionGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNsaSelectedAttentionGrad defaults to a non-deterministic implementation. Deterministic computation can be enabled by calling aclrtSetSysParamOpt.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- The B (batchsize) of the input query, key, value, attentionOut, and attentionOutGrad must be equal.
- The N (numHead) of the input key and value must be consistent.
- The N (numHead) of the input query, attentionOut, and attentionOutGrad must be consistent.
- The D (HeadDim) of the input value, attentionOut, and attentionOutGrad must be consistent.
- The inputLayout of the input query, key, value, attentionOut, and attentionOutGrad must be consistent.
- For constraints on data shape, take TND of inputLayout as an example, where:
  - T1: Value range: 1 to 2M. T1 represents the sum of S across all batches of query.
  - T2: Value range: 1 to 2M. T2 represents the sum of S across all batches of key and value.
  - B: Value range: 1 to 2M.
  - N1: Value range: 1 to 128. Represents the headNum of query. N1 must be an integer multiple of N2.
  - N2: Value range: 1 to 128. Represents the headNum of key and value.
  - G: value range is 1 to 32. G = N1 / N2
  - S: value range is 1\~128K. For key, value, S must be greater than or equal to selectedBlockSize * selectedBlockCount, and must be an integer multiple of selectedBlockSize.
  - D: value range is 192 or 128. Supports unequal D (HeadDim) for K and V.
  - selectedBlockSize supports ≤ 128 and must be an integer multiple of 16.
  - selectedBlockCount: Supported range is [1-128]. The total selected size `selectedBlockCount * selectedBlockSize` < 128*64(8K)
  - When Layout is TND, S2 of each batch must be greater than the total selected size `selectedBlockCount * selectedBlockSize`.
- Constraints on the shape of the softmaxMax and softmaxSum parameters: [T1, N1, 8].
- Constraints on the shape of the topkIndices parameter: [T1, N2, selectedBlockCount].

## Example

The sample code is for reference only. For details about compiling and running the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
    // Call aclrtMalloc to allocate memory on the Device side
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data on the Host side to memory on the Device side
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize device/stream. For details, see the list of AscendCL external APIs
    // Fill in deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as needed
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API
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

    // Create query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create attentionOut aclTensor.
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create attentionOutGrad aclTensor.
    ret = CreateAclTensor(attentionOutGradHostData, attentionOutGradShape, &attentionOutGradDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOutGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create softmaxMax aclTensor.
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create softmaxSum aclTensor.
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create topkIndices aclTensor.
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    int64_t tempQ[1] = {1};
    int64_t tempK[1] = {1024};
    aclIntArray* actualSeqQLenOptional = aclCreateIntArray(tempQ, static_cast<uint64_t>(1));
    aclIntArray* actualSeqKvLenOptional = aclCreateIntArray(tempK, static_cast<uint64_t>(1));
    // Create dq aclTensor.
    ret = CreateAclTensor(dqOutHostData, dqOutShape, &dqOutDeviceAddr, aclDataType::ACL_FLOAT16, &dqOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create dk aclTensor.
    ret = CreateAclTensor(dkOutHostData, dkOutShape, &dkOutDeviceAddr, aclDataType::ACL_FLOAT16, &dkOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the dv aclTensor.
    ret = CreateAclTensor(dvOutHostData, dvOutShape, &dvOutDeviceAddr, aclDataType::ACL_FLOAT16, &dvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Example of calling the aclnnNsaSelectedAttentionGrad API
    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnNsaSelectedAttentionGrad.
    ret = aclnnNsaSelectedAttentionGradGetWorkspaceSize(query, key, value, attentionOut, attentionOutGrad, softmaxMax,
                                                        softmaxSum, topkIndices, actualSeqQLenOptional,
                                                        actualSeqKvLenOptional, nullptr, scaleValue, selectedBlockSize,
                                                        selectedBlockCount, headNum, inputLayout, sparseMode,
                                                        dqOut, dkOut, dvOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnNsaSelectedAttentionGrad.
    ret = aclnnNsaSelectedAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the result from device memory to the host. Modify it according to the definition of the specific API.
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
    std::vector<aclFloat16> dvResultData(dvSize, 0);
    ret = aclrtMemcpy(dvResultData.data(), dvResultData.size() * sizeof(dvResultData[0]), dvOutDeviceAddr,
                      dvSize * sizeof(dvResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result dv from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dvSize; i++) {
        LOG_PRINT("result dv[%ld] is: %f\n", i, dvResultData[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify it according to the definition of the specific API.
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
    // 7. Release device resources. Modify it according to the definition of the specific API.
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
