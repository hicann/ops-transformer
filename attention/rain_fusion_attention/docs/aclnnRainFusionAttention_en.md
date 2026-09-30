# aclnnRainFusionAttention

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Description

- **API function**: Performs RainFusionAttention sparse attention computation with flexible block-level sparsity patterns, achieving efficient sparse attention by using `selectIdx` to designate which key-value blocks are selected for each query block.

- **Formula**: Sparse block size of $blockShapeX \times blockShapeY$, with sparsity pattern defined by `selectIdx`

    $$
    attentionOut = Softmax(scale \cdot query \cdot key^T + atten\_mask) \cdot value
    $$

    RainFusionAttention input tensors (`query`, `key`, and `value`) support flexible data layouts interpretable across multiple dimensions. Use `qInputLayout` and `kvInputLayout` parameters to specify the desired format.
    - B: input batch size
    - T: total token length with combined B and S dimensions
    - S: sequence length of input samples
    - H: hidden layer size (Head-Size)
    - N: number of heads (Head-Num)
    - D: minimum unit size of the hidden layer (D=H/N)

    The following layouts are supported:
    - `qInputLayout`: TND and BNSD
    - `kvInputLayout`: TND and BNSD

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRainFusionAttentionGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRainFusionAttention` is called to perform computation.

```c++
aclnnStatus aclnnRainFusionAttentionGetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *selectIdx,
  const aclTensor   *selectNumIdx,
  const aclIntArray *blockShape,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  const aclTensor   *blockTableOptional,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  int64_t            maskType,
  double             scaleValue,
  int64_t            innerPrecise,
  int64_t            blockSize,
  const aclTensor   *attentionOut,
  const aclTensor   *softmaxLseOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnRainFusionAttention(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnRainFusionAttentionGetWorkspaceSize

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
      <th>Precaution</th>
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
      <td>query in the formula.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>key in the formula.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>value in the formula.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>selectIdx</td>
      <td>Input</td>
      <td>Sparse block index array specifying which key-value blocks each query block selects.</td>
      <td>
        <ul>
          <li>The shape is [QBlockNum, headNum, maxKvBlockNum].</li>
          <li>QBlockNum represents the total number of query blocks across all batches.</li>
          <li>Valid indexes are placed first in ascending order, with invalid positions padded with -1 at the end.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>selectNumIdx</td>
      <td>Input</td>
      <td>Number of key-value blocks actually selected by each query block.</td>
      <td>
        <ul>
          <li>The shape is [QBlockNum, headNum].</li>
          <li>Number of key-value blocks actually selected by each query block.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>blockShape</td>
      <td>Input</td>
      <td>Sparse block shape array.</td>
      <td>
        <ul>
          <li>At least two elements are required: [blockShapeX, blockShapeY].</li>
          <li>blockShapeX indicates the query block size.</li>
          <li>blockShapeY indicates the key-value block size.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td>atten_mask in the formula.</td>
      <td>This parameter is not supported currently. Pass nullptr.</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsOptional</td>
      <td>Input</td>
      <td>Query sequence length for each batch.</td>
      <td>
        <ul>
          <li>If this parameter is not used, a null pointer can be passed.</li>
          <li>This parameter is used in variable-length sequence scenarios.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional</td>
      <td>Input</td>
      <td>Key/Value sequence length for each batch.</td>
      <td>
        <ul>
          <li>If this parameter is not used, a null pointer can be passed.</li>
          <li>This parameter is used in variable-length sequence scenarios.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>Input</td>
      <td>Block table for PagedAttention.</td>
      <td>This parameter is not supported currently. Pass nullptr.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qInputLayout</td>
      <td>Input</td>
      <td>Layout of the input query.</td>
      <td> Currently, only TND and BNSD are supported.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvInputLayout</td>
      <td>Input</td>
      <td>Layout of the input key and value.</td>
      <td> Currently, only TND and BNSD are supported.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads</td>
      <td>Input</td>
      <td>Number of heads in key and value.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskType</td>
      <td>Input</td>
      <td>Mask type.</td>
      <td>0 indicates no mask. Other values indicate different mask types.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>scale in the formula, indicating the scale factor.</td>
      <td>Generally, set this parameter to D^-0.5.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise</td>
      <td>Input</td>
      <td>Softmax precision control.</td>
      <td>0 indicates float32 Softmax, and 1 indicates fp16 Softmax.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize</td>
      <td>Input</td>
      <td>Block size of PagedAttention.</td>
      <td>Used in the PagedAttention scenario. If this parameter is not used, set it to 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>Output</td>
      <td> attentionOut in the formula.</td>
      <td>The data type and shape must be the same as those of query.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>Output</td>
      <td>Intermediate result of the log-sum-exp operation in Softmax.</td>
      <td>This parameter is not supported currently. Pass nullptr.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
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

  aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

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
      <td>The input query, key, value, selectIdx, or selectNumIdx is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of query, key, or value is not supported.</td>
    </tr>
    <tr>
      <td>qInputLayout or kvInputLayout is invalid.</td>
    </tr>
    <tr>
      <td>blockShape is invalid (the number of elements is less than 2 or an element value is less than or equal to 0).</td>
    </tr>
    <tr>
      <td>innerPrecise is invalid (the value is not 0 or 1).</td>
    </tr>
  </tbody>
  </table>

## aclnnRainFusionAttention

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
  <col style="width: 144px">
  <col style="width: 125px">
  <col style="width: 700px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnRainFusionAttentionGetWorkspaceSize.</td>
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

  aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnRainFusionAttention` defaults to a deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Currently, `qInputLayout` can only be TND or BNSD.
- Currently, `kvInputLayout` can only be TND or BNSD.
- The input query, key, and value must be of the same data type, which can be either FLOAT16 or BFLOAT16.
- `blockShape` must contain at least two elements [blockShapeX, blockShapeY], and the element values must be greater than 0.
- The shape of `selectIdx` must be [T, headNum, maxKvBlockNum], where T is the total number of query blocks across all batches.
- The shape of `selectNumIdx` must be [T, headNum].
- `innerPrecise` must be 0 (for float32 Softmax) or 1 (for fp16 Softmax). If the `query` input is BFLOAT16, `innerPrecise` can only be set to 0.
- `qSeqle`n and `kvSeqlen` do not need to be exactly divided by `blockShape`. When they are not divisible, the actual number of blocks is determined by ceiling division.
- `qSeqlen` is required when `qInputLayout` is TND or BNSD. `kvSeqlen` is required when `kvInputLayout` is TND or BNSD.
- Sparse block indexes must be within the valid range. Fill invalid positions with -1.
- If headNum of input `query` is N1 and headNum of input `key` and `value` is N2, then `N1 >= N2 && N1% N2 == 0`.
- Assume G = N1/N2. G must meet the following constraint: `G < 128 && 128 % G == 0`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_rain_fusion_attention.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    // Check whether shape is valid.
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);
    
    // Check whether the size of hostData matches GetShapeSize(shape).
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", 
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }
    
    // Call aclrtMalloc to allocate memory on the device.
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);
    
    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}


int main() {
    // 1. (Fixed writing) Initialize the device and stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Set parameters.
    int32_t batch = 1;
    int32_t qSeqlen = 128;
    int32_t kvSeqlen = 128;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;
    
    // Calculate the dimensions in TND format.
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX; // Number of query blocks along the X dimension.
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY; // Number of key-value blocks along the Y dimension.
    // totalQBlocks = qBlockNum * numHeads (each query block corresponds to one head)
    int32_t totalQBlocks = qBlockNum * batch;
    int32_t maxKvBlockNum = kvBlockNum;
    
    
    // 3. Create the query tensor (TND format: [totalQTokens, numHeads, headDim]).
    void *queryDeviceAddr = nullptr;
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(totalQTokens * numHeads * headDim, 1.0f);
    aclTensor *queryTensor = nullptr;
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n"); return ret);
    
    // 4. Create a key-value tensor (TND format: [totalKvTokens, numKvHeads, headDim]).
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    std::vector<int64_t> kvShape = {totalKvTokens, numKvHeads, headDim};
    std::vector<op::fp16_t> keyHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    std::vector<op::fp16_t> valueHostData(totalKvTokens * numKvHeads * headDim, 1.0f);
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    ret = CreateAclTensor(keyHostData, kvShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n"); return ret);
    ret = CreateAclTensor(valueHostData, kvShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create value tensor\n"); return ret);
    
    // 5. Generate sparse indexes selectIdx and selectNumIdx.
    // selectIdx: [totalQBlocks, numHeads, maxKvBlockNum] - 3D tensor
    // selectNumIdx: [totalQBlocks, numHeads] - 2D tensor
    // The sparsity ratio is 1, that is, no sparsity is applied and each query block selects all key-value blocks.
    std::vector<int64_t> selectIdxHostData(totalQBlocks * numHeads * maxKvBlockNum, -1);
    std::vector<int64_t> selectNumIdxHostData(totalQBlocks * numHeads, 0);
    
    // With sparsity ratio of 1, each query block selects all key-value blocks, and indexes from 0 to maxKvBlockNum-1 are assigned.
    for (int32_t qb = 0; qb < totalQBlocks; ++qb) {
        for (int32_t h = 0; h < numHeads; ++h) {
            // selectNumIdx[qb, h] = maxKvBlockNum (Each query block selects all key-value blocks.)
            selectNumIdxHostData[qb * numHeads + h] = static_cast<int64_t>(maxKvBlockNum);
            
            // selectIdx[qb, h, k] = k (Indexes from 0 to maxKvBlockNum-1 are assigned.)
            int64_t baseIdx = static_cast<int64_t>((qb * numHeads + h) * maxKvBlockNum);
            for (int32_t k = 0; k < maxKvBlockNum; ++k) {
                selectIdxHostData[baseIdx + k] = static_cast<int64_t>(k);
            }
        }
    }
    
    void *selectIdxDeviceAddr = nullptr;
    void *selectNumIdxDeviceAddr = nullptr;
    std::vector<int64_t> selectIdxShape = {totalQBlocks, numHeads, maxKvBlockNum};
    std::vector<int64_t> selectNumIdxShape = {totalQBlocks, numHeads};
    aclTensor *selectIdxTensor = nullptr;
    aclTensor *selectNumIdxTensor = nullptr;
    ret = CreateAclTensor(selectIdxHostData, selectIdxShape, &selectIdxDeviceAddr, aclDataType::ACL_INT64, &selectIdxTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create selectIdx tensor\n"); return ret);
    ret = CreateAclTensor(selectNumIdxHostData, selectNumIdxShape, &selectNumIdxDeviceAddr, aclDataType::ACL_INT64, &selectNumIdxTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create selectNumIdx tensor\n"); return ret);
    
    // 6. Create an output tensor.
    void *outputDeviceAddr = nullptr;
    std::vector<int64_t> outputShape = {totalQTokens, numHeads, headDim};
    int64_t outputElementCount = totalQTokens * numHeads * headDim;
    std::vector<op::fp16_t> outputHostData(outputElementCount, 0.0f);
    aclTensor *outputTensor = nullptr;
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create output tensor\n"); return ret);
    
    // 7. Create a blockShape array.
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n"); return -1);
    
    // 8. Create mandatory parameters actualSeqLengths and actualSeqLengthsKv.
    std::vector<int64_t> actualSeqLengthsHost(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> actualSeqLengthsKvHost(batch, static_cast<int64_t>(kvSeqlen));
    
    void *actualSeqLengthsDevice = nullptr;
    void *actualSeqLengthsKvDevice = nullptr;
    size_t seqLengthsSize = batch * sizeof(int64_t);
    
    ret = aclrtMalloc(&actualSeqLengthsDevice, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengths memory\n"); return ret);
    ret = aclrtMalloc(&actualSeqLengthsKvDevice, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengthsKv memory\n"); 
              aclrtFree(actualSeqLengthsDevice); return ret);
    
    ret = aclrtMemcpy(actualSeqLengthsDevice, seqLengthsSize, actualSeqLengthsHost.data(), 
                     seqLengthsSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengths to device\n"); 
              aclrtFree(actualSeqLengthsDevice); aclrtFree(actualSeqLengthsKvDevice); return ret);
    ret = aclrtMemcpy(actualSeqLengthsKvDevice, seqLengthsSize, actualSeqLengthsKvHost.data(), 
                     seqLengthsSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengthsKv to device\n"); 
              aclrtFree(actualSeqLengthsDevice); aclrtFree(actualSeqLengthsKvDevice); return ret);
    
    // aclCreateIntArray expects a data pointer on the host instead of one on the device.
    aclIntArray *actualSeqLengths = aclCreateIntArray(actualSeqLengthsHost.data(), batch);
    aclIntArray *actualSeqLengthsKv = aclCreateIntArray(actualSeqLengthsKvHost.data(), batch);
    CHECK_RET(actualSeqLengths != nullptr && actualSeqLengthsKv != nullptr, 
              LOG_PRINT("Failed to create actualSeqLengths arrays\n"); 
              if (actualSeqLengthsDevice) aclrtFree(actualSeqLengthsDevice);
              if (actualSeqLengthsKvDevice) aclrtFree(actualSeqLengthsKvDevice); return -1);
    
    // 9. Prepare string parameters. Ensure that the buffer sizes are sufficient and include the null terminator.
    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);
    
    // 10. Calculate scaleValue.
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
    
    // 11. Call the first-phase API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    ret = aclnnRainFusionAttentionGetWorkspaceSize(
        queryTensor,           // query
        keyTensor,             // key
        valueTensor,           // value
        selectIdxTensor,       // selectIdx
        selectNumIdxTensor,    // selectNumIdx
        blockShape,            // blockShape
        nullptr,               // attenMaskOptional
        actualSeqLengths,      // actualSeqLengthsOptional
        actualSeqLengthsKv,    // actualSeqLengthsKvOptional
        nullptr,               // blockTableOptional
        qLayoutBuffer,         // qInputLayout
        kvLayoutBuffer,        // kvInputLayout
        numKvHeads,            // numKeyValueHeads
        0,                     // maskType
        scaleValue,            // scaleValue
        0,                     // innerPrecise (1=fp16 softmax)
        128,                   // blockSize
        outputTensor,          // attentionOut
        nullptr,               // softmaxLseOptional
        &workspaceSize,        // workspaceSize (out)
        &executor);            // executor (out)
    
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRainFusionAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    
    // 12. Allocate memory for workspace.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    
    // 12. Call the second-phase API.
    ret = aclnnRainFusionAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRainFusionAttention failed. ERROR: %d\n", ret); return ret);
    
    // 13. Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    
    // 14. Obtain the output value and copy the result from the device memory to the host.
    int64_t outputSize = GetShapeSize(outputShape);
    std::vector<op::fp16_t> resultData(outputSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(op::fp16_t), outputDeviceAddr,
                     outputSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    
    // 15. Print partial results.
    uint64_t printNum = 10;
    LOG_PRINT("Output results (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, static_cast<float>(resultData[i]));
    }
    
    // 16. Release resources.
    if (workspaceAddr) aclrtFree(workspaceAddr);
    if (queryDeviceAddr) aclrtFree(queryDeviceAddr);
    if (keyDeviceAddr) aclrtFree(keyDeviceAddr);
    if (valueDeviceAddr) aclrtFree(valueDeviceAddr);
    if (outputDeviceAddr) aclrtFree(outputDeviceAddr);
    if (selectIdxDeviceAddr) aclrtFree(selectIdxDeviceAddr);
    if (selectNumIdxDeviceAddr) aclrtFree(selectNumIdxDeviceAddr);
    if (actualSeqLengthsDevice) aclrtFree(actualSeqLengthsDevice);
    if (actualSeqLengthsKvDevice) aclrtFree(actualSeqLengthsKvDevice);
    
    if (queryTensor) aclDestroyTensor(queryTensor);
    if (keyTensor) aclDestroyTensor(keyTensor);
    if (valueTensor) aclDestroyTensor(valueTensor);
    if (outputTensor) aclDestroyTensor(outputTensor);
    if (selectIdxTensor) aclDestroyTensor(selectIdxTensor);
    if (selectNumIdxTensor) aclDestroyTensor(selectNumIdxTensor);
    if (blockShape) aclDestroyIntArray(blockShape);
    if (actualSeqLengths) aclDestroyIntArray(actualSeqLengths);
    if (actualSeqLengthsKv) aclDestroyIntArray(actualSeqLengthsKv);
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    LOG_PRINT("Test completed successfully!\n");
    return 0;
}

```
