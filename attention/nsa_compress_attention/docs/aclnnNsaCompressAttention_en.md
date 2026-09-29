# aclnnNsaCompressAttention

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|     √      |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|     √      |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- **Description**: Calculates the compress attention and select topk indexes in NSA. For details, see the paper (<https://arxiv.org/pdf/2502.11089>).

- **Computation formulas:** compression block size: $l$; selection block size: $l'$; and compression stride: $d$

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

NsaCompressAttention input tensors (`query`, `key`, and `value`) support flexible data layouts interpretable across multiple dimensions. Use `inputLayout` to specify the desired format (only `TND` is supported).

- `B` (`Batch`): input batch size
- `T`: total token length with combined B and S dimensions
- `S` (`Seq-Length`): sequence length of input samples
- `H` (`Head-Size`): hidden-layer size
- `N` (`Head-Num`): number of heads
- `D` (`Head-Dim`): minimum unit size of the hidden layer (`D` = `H`/`N`)

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNsaCompressAttentionGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, `aclnnNsaCompressAttention` is called to perform computation.

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
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3–4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td><code>key</code> in the formulas.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3–4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td><code>value</code> in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3–4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td><code>atten_mask</code> in the formula.</td>
        <td>
          <ul>
            <li>The input shape must be [S, S].</li>
            <li>In the TND scenario, only the SS format is supported. S and S are max(Sq) and max(CmpSkv), respectively.</li>
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
        <td><code>S</code> of <code>query</code> (<code>Sq</code>) corresponding to each batch.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualCmpSeqKvLenOptional</td>
        <td>Input</td>
        <td><code>S</code> of <code>key</code>/<code>value</code> (<code>CmpSkv</code>) corresponding to each batch of the compressed attention.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSelSeqKvLenOptional</td>
        <td>Input</td>
        <td><code>S</code> of <code>key</code>/<code>value</code> (<code>SelSkv</code>) corresponding to each batch after the importance score computation and compression.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topkMaskOptional</td>
        <td>Input</td>
        <td><code>topk_mask</code> in the formula.</td>
        <td>
          <ul>
            <li>The input shape must be [S, S].</li>
            <li>In the TND scenario, only the [S, S] format is supported, which indicates <code>max(Sq)</code> and <code>max(SelSkv)</code>, respectively.</li>
            <li>If this parameter is not used, a null pointer can be passed.</li>
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
        <td><code>scale</code> in the formula, indicating the scaling coefficient.</td>
        <td>Generally, this parameter is set to D<sup>–0.5</sup>.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Number of heads in <code>query</code>.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Layout of the input <code>query</code>, <code>key</code>, and <code>value</code>.</td>
        <td>Currently, only <code>TND</code> is supported.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode.</td>
        <td>Only <code>0</code> and <code>1</code> are supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressBlockSize</td>
        <td>Input</td>
        <td>Corresponds to l in the formula.</td>
        <td>Size of the compression sliding window.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>Input</td>
        <td>Corresponds to d in the formula.</td>
        <td>Sliding window interval between two compressions.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>selectBlockSize</td>
        <td>Input</td>
        <td>Corresponds to l' in the formula.</td>
        <td>Size of the selected block.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>selectBlockCount</td>
        <td>Input</td>
        <td>Number of selected topK values in the formula.</td>
        <td>Number of selected blocks.</td>
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
        <td><code>attentionOut</code> in the formula.</td>
        <td>The data type and the first two dimensions of the shape are the same as those of the query, and the last dimension is the same as that of the value.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3–4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topkIndicesOut</td>
        <td>Output</td>
        <td><code>topkIndices</code> in the formula.</td>
        <td>-</td>
        <td>INT32</td>
        <td>-</td>
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
      <td>The input <code>query</code>, <code>key</code>, or <code>value</code> is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of <code>query</code>, <code>key</code>, or <code>value</code> is not supported.</td>
    </tr>
    <tr>
      <td><code>inputLayout</code> is invalid.</td>
    </tr>
    <tr>
      <td><code>sparseMode</code> is invalid.</td>
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API <code>aclnnNsaCompressAttentionGetWorkspaceSize</code>.</td>
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
  - `aclnnNsaCompressAttention` defaults to deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- The values of `compressBlockSize`, `compressStride`, and `selectBlockSize` must be integer multiples of 16 and meet the following requirements: `compressBlockSize>=compressStride && selectBlockSize>=compressBlockSize && selectBlockSize%compressStride==0`.
- compressBlockSize: 16-byte aligned, up to 128
- compressStride: 16-byte aligned, up to 64
- selectBlockSize: 16-byte aligned, up to 128
- selectBlockCount: supports [1-32] && selectBlockCount <= min(SelSkv).
- `actualSeqQLenOptional`, `actualCmpSeqKvLenOptional`, and `actualSelSeqKvLenOptional` must use the cumulative sum mode and must be passed in `TND` format.
- Due to the UB restriction, `CmpSkv` must be less than or equal to 14000.
- `SelSkv = CeilDiv(CmpSkv, selectBlockSize // compressStride)`
- Currently, layoutOptional supports only TND.
- The input data types of `query`, `key`, and `value` must be the same.
- The batch sizes of the input query, key, and value must be the same.
- The headDim of the input query, key, and value must meet the following requirement: qD == kD && kD >= vD
- `inputLayout` of the input `query`, `key`, and `value` must be the same.
- If `headNum` of input `query` is N1 and headNum of input `key` and `value` is N2, then `N1 >= N2 && N1% N2 == 0`.
- Assume G = N1/N2. G must meet the following constraint: `G < 128 && 128 % G == 0`.
- The usage of `attenMask` and `topkMask` must comply with the description in the paper.

## Example

The following example is for reference only (using <term>Atlas A2 training products/Atlas A2 inference products</term> as examples). For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
    // (Fixed writing) Initialize resources.
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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
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

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API.
    ret = aclnnNsaCompressAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttention failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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
