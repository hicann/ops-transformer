# aclnnNsaCompressAttention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:57:43.087Z -->

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

- **Description**: Computes the compress attention and select topk indexes in NSA. Paper: [Native Sparse Attention](https://arxiv.org/pdf/2502.11089).

- **Formula**: Compressed block size: $l$, selected block size: $l'$, compressed stride size: $d$

$$
P_{cmp} = Softmax(query*key^T) \\
$$

$$
attentionOut = Softmax(atten\_mask(scale*query*key^T, atten\_mask))*value
$$

$$
P_{slc}[j] = \sum_{m=0}^{l'/d-1}\sum_{n=0}^{l/d-1}P_{cmp} [l'/d*j-m-n],
$$

$$
P_{slc'} = \sum_{h=1}^{H}P_{slc}^{h}
$$

$$
P_{slc'} = topk\_mask(P_{slc'})
$$

$$
topkIndices = topk(P_{slc'})
$$

The data layout format of the query, key, and value inputs for NsaCompressAttention supports interpretation from multiple dimension layouts, which can be specified through inputLayout. Currently, only TND is supported.

- B: Indicates the batch size of the input samples (Batch).
- T: Indicates the length of the tightly packed B and S axes.
- S: Indicates the sequence length of the input samples (Seq-Length).
- H: Indicates the size of the hidden layer (Head-Size).
- N: Number of heads (Head-Num).
- D: Minimum unit size of the hidden layer, satisfying D = H/N (Head-Dim).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnNsaCompressAttentionGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnNsaCompressAttention" to execute the computation.

```c++
aclnnStatus aclnnNsaCompressAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *topkMaskOptional,
  const aclIntArray *actualSeqQLenOptional,
  const aclIntArray *actualCmpSeqKvLenOptional,
  const aclIntArray *actualSelSeqKvLenOptional,
  double             scaleValue,
  int64_t            headNum,
  char              *inputLayout,
  int64_t            sparseMode,
  int64_t            compressBlockSize,
  int64_t            compressStride,
  int64_t            selectBlockSize,
  int64_t            selectBlockCount,
  const aclTensor   *softmaxMaxOut,
  const aclTensor   *softmaxSumOut,
  const aclTensor   *attentionOut,
  const aclTensor   *topkIndicesOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaCompressAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnNsaCompressAttentionGetWorkspaceSize

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
        <td>The query in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>The key in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>The value in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>atten_mask in the formula.</td>
        <td>          <ul>
            <li>The input shape must be [S,S].</li>
            <li>Only the SS format is supported in TND use cases. SS is max(Sq) and max(CmpSkv), respectively.</li>
          </ul>
        </td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>Input</td>
        <td>Describes the query S size (Sq) corresponding to each batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualCmpSeqKvLenOptional</td>
        <td>Input</td>
        <td>Describes the key/value S size (CmpSkv) of each Batch in compress attention.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSelSeqKvLenOptional</td>
        <td>Input</td>
        <td>Describes the key/value S size (SelSkv) of each Batch after compression based on the importance score.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topkMaskOptional</td>
        <td>Input</td>
        <td>topk_mask in the formula.</td>
        <td>          <ul>
            <li>The input shape must be [S,S].</li>
            <li>Only the SS format is supported in TND use cases. SS refers to max(Sq) and max(SelSkv) respectively.</li>
            <li>If not used, pass nullptr.</li>
          </ul>
        </td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>scale in the formula, representing the scaling factor.</td>
        <td>Generally set to D^-0.5.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Indicates the number of query heads.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Indicates the data layout format of the input query, key, and value.</td>
        <td>Currently, TND is supported.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode selection.</td>
        <td>Only 0 and 1 are supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressBlockSize</td>
        <td>Input</td>
        <td>Corresponds to l in the formula.</td>
        <td>Compressed sliding window size.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>Input</td>
        <td>d in the corresponding formula.</td>
        <td>Interval size between two compression sliding windows.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>selectBlockSize</td>
        <td>Input</td>
        <td>Corresponds to l' in the formula.</td>
        <td>Selection block size.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>selectBlockCount</td>
        <td>Input</td>
        <td>Number of topK selections in the corresponding formula.</td>
        <td>Number of selected blocks.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>Output</td>
        <td>Max intermediate result of the Softmax computation.</td>
        <td>Used for backward propagation.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>Output</td>
        <td>Intermediate Sum result of the Softmax computation.</td>
        <td>Used for backward propagation.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>attentionOut in the formula.</td>
        <td>The data type and the first 2 dimensions of shape are the same as those of query, and the last dimension is the same as the last dimension of value.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topkIndicesOut</td>
        <td>Output</td>
        <td>topkIndices in the formula.</td>
        <td>-</td>
        <td>INT32</td>
        <td>-</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation process.</td>
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
      <td>The input query, key, or value is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of query, key, or value is not within the supported range.</td>
    </tr>
    <tr>
      <td>inputLayout is invalid.</td>
    </tr>
    <tr>
      <td>sparseMode is invalid.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaCompressAttention

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnNsaCompressAttentionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. Contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNsaCompressAttention defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- compressBlockSize, compressStride, and selectBlockSize must be integer multiples of 16, and satisfy: compressBlockSize>=compressStride && selectBlockSize>=compressBlockSize && selectBlockSize%compressStride==0
- compressBlockSize: 16-aligned, supports up to 128.
- compressStride: 16-aligned, supports up to 64.
- selectBlockSize: 16-aligned, supports up to 128.
- selectBlockCount: Supports [1, 32] and selectBlockCount <= min(SelSkv).
- actualSeqQLenOptional, actualCmpSeqKvLenOptional, and actualSelSeqKvLenOptional must be in prefix sum mode, and must be passed in TND format.
- Due to UB constraints, CmpSkv must satisfy: CmpSkv <= 14000
- SelSkv = CeilDiv(CmpSkv, selectBlockSize // compressStride)
- layoutOptional currently supports only TND.
- The data types of the input query, key, and value must be consistent.
- The batchSize of the input query, key, and value must be equal.
- The headDim of input query, key, and value must satisfy: qD == kD && kD >= vD
- The headDim of input query, key, and value must be in the range (0, 768]. aclnnNsaCompressAttentionVarLenScore follows the same constraint.
- The inputLayout of the input query, key, and value must be consistent.
- Let the headNum of input query be N1, and the headNum of input key and value be N2, then N1 >= N2 && N1 % N2 == 0
- Let G = N1 / N2. G must satisfy the following constraints: G < 128 && 128 % G == 0
- The use of attenMask and topkMask must comply with the description in the paper.

## Example

The sample code is as follows (taking <term>Atlas A2 products</term> as an example), for reference only. For the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_nsa_compress_attention.h"

using namespace std;

#define CHECK_RET(cond, return_expr)   \
    do {                               \
        if (!(cond)) {                 \
            return_expr;               \
        }                              \
    } while (0)

#define LOG_PRINT(message, ...)           \
    do {                                  \
        printf(message, ##__VA_ARGS__);   \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate. Resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host side to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Device/stream initialization. For details, see the AscendCL public API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Customize the construction as required by the API.
    int64_t T1 = 1024;
    int64_t T2 = 64;
    int64_t N1 = 16;
    int64_t N2 = 4;
    int64_t D1 = 192;
    int64_t D2 = 128;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 16;
    int64_t compressBlockSize = 32;
    int64_t compressStride = 16;
    std::vector<int64_t> qShape = {T1, N1, D1};
    std::vector<int64_t> kShape = {T2, N2, D1};
    std::vector<int64_t> vShape = {T2, N2, D2};
    std::vector<int64_t> attenmaskShape = {T1, T2};                        //[maxS1, maxS2]
    std::vector<int64_t> topkmaskShape = {T1, T1 / selectBlockSize};       //[maxS1, maxSelS2]
    std::vector<int64_t> softmaxMaxShape = {T1, N1, 8};
    std::vector<int64_t> softmaxSumShape = {T1, N1, 8};
    std::vector<int64_t> attenOutShape = {T1, N1, D2};                     //[T1, N1, D2]
    std::vector<int64_t> topkIndicesOutShape = {T1, N2, selectBlockCount}; //[T1, N2, selectBlockCount]

    void* qDeviceAddr = nullptr;
    void* kDeviceAddr = nullptr;
    void* vDeviceAddr = nullptr;
    void* attenmaskDeviceAddr = nullptr;
    void* topkmaskDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;
    void* topkIndicesOutDeviceAddr = nullptr;

    aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* attenmask = nullptr;
    aclTensor* topkmask = nullptr;
    aclTensor* softmaxMax = nullptr;
    aclTensor* softmaxSum = nullptr;
    aclTensor* attentionOut = nullptr;
    aclTensor* topkIndicesOut = nullptr;

    std::vector<op::fp16_t> qHostData(T1 * N1 * D1, 1.0);
    std::vector<op::fp16_t> kHostData(T2 * N2 * D1, 1.0);
    std::vector<op::fp16_t> vHostData(T2 * N2 * D2, 1.0);
    std::vector<uint8_t> attenmaskHostData(T1 * T2, 0);
    std::vector<uint8_t> topkmaskHostData(T1 * (T1 / selectBlockSize), 0);
    std::vector<float> softmaxMaxHostData(N1 * T1 * 8, 1.0);
    std::vector<float> softmaxSumHostData(N1 * T1 * 8, 1.0);
    std::vector<op::fp16_t> attenOutHostData(T1 * N1 * D2, 1.0);
    std::vector<int32_t> topkIndicesHostData(T1 * N2 * selectBlockCount, 1);

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkmaskHostData, topkmaskShape, &topkmaskDeviceAddr, aclDataType::ACL_UINT8, &topkmask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attenOutHostData, attenOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT16, &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesOutShape, &topkIndicesOutDeviceAddr, aclDataType::ACL_INT32, &topkIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualSeqQLenVec(1, T1);
    auto actualSeqQLen = aclCreateIntArray(actualSeqQLenVec.data(), actualSeqQLenVec.size());
    std::vector<int64_t> actualCmpKvSeqVec(1, T2);
    auto actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqVec.data(), actualCmpKvSeqVec.size());
    std::vector<int64_t> actualSelKvSeqVec(1, T1 / selectBlockSize);
    auto actualSelKvSeqLen = aclCreateIntArray(actualSelKvSeqVec.data(), actualSelKvSeqVec.size());

    double scale = 1.0;
    int64_t headNum = N1;
    char inputLayout[5] = {'T', 'N', 'D', 0};
    int64_t sparseMode = 1;

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API.
    ret = aclnnNsaCompressAttentionGetWorkspaceSize(q, k, v, attenmask, topkmask, actualSeqQLen, actualCmpKvSeqLen,
        actualSelKvSeqLen, scale, headNum, inputLayout, sparseMode, compressBlockSize, compressStride, selectBlockSize, selectBlockCount,
        softmaxMax, softmaxSum, attentionOut, topkIndicesOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory using the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API.
    ret = aclnnNsaCompressAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttention failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, copy the results on the device side to the host side, and modify the code based on the specific API definition.
    PrintOutResult(attenOutShape, &attentionOutDeviceAddr);
    PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
    PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);
    PrintOutResult(topkIndicesOutShape, &topkIndicesOutDeviceAddr);

    // 6. Release resources.
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(attenmask);
    aclDestroyTensor(topkmask);
    aclDestroyTensor(softmaxMax);
    aclDestroyTensor(softmaxSum);
    aclDestroyTensor(attentionOut);
    aclDestroyTensor(topkIndicesOut);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(attenmaskDeviceAddr);
    aclrtFree(topkmaskDeviceAddr);
    aclrtFree(softmaxMaxDeviceAddr);
    aclrtFree(softmaxSumDeviceAddr);
    aclrtFree(attentionOutDeviceAddr);
    aclrtFree(topkIndicesOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
