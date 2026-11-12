# aclnnRainFusionAttention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:50:23.896Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

- **Description**: RainFusionAttention performs sparse attention computation, supporting flexible block-level sparse patterns. It specifies the KV blocks selected for each Q block through selectIdx, enabling efficient sparse attention computation.

- **Formula**: Sparse block size: $blockShapeX \times blockShapeY$, selectIdx specifies the sparse pattern

  $$
  attentionOut = Softmax(scale \cdot query \cdot key^T + atten\_mask) \cdot value
  $$

  The data layout format of the query, key, and value inputs for RainFusionAttention supports interpretation from multiple dimensional layouts, which can be passed in through qInputLayout and kvInputLayout.

  - B: Indicates the batch size of the input samples (Batch).
  - T: Indicates the length of B and S concatenated along the axis and tightly packed (Total tokens).
  - S: Indicates the sequence length of the input samples (Seq-Length).
  - H: Indicates the size of the hidden layer (Head-Size).
  - N: Number of heads (Head-Num).
  - D: Minimum unit size of the hidden layer, satisfying D = H/N (Head-Dim).

  Currently supported layouts:

  - qInputLayout: "TND" "BNSD"
  - kvInputLayout: "TND" "BNSD"

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnRainFusionAttentionGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call the "aclnnRainFusionAttention" API to execute the computation.

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
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>The key in the formula.</td>
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
      <td>Sparse block index array, specifying the KV block indices selected by each Q block.</td>
      <td>        <ul>
          <li>The shape is [QBlockNum, headNum, maxKvBlockNum].</li>
          <li>QBlockNum is the total number of blocks along the Q dimension in all batches.</li>
          <li>Stores the KV block indices selected by each Q block. Valid indices are placed at the front of the array in ascending order, and invalid positions are placed at the end and filled with -1.</li>
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
      <td>Number of KV blocks actually selected for each Q block.</td>
      <td>        <ul>
          <li>Shape is [QBlockNum, headNum].</li>
          <li>Stores the number of KV blocks actually selected for each Q block.</li>
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
      <td>Array of sparse block shapes.</td>
      <td>        <ul>
          <li>Must contain at least two elements [blockShapeX, blockShapeY].</li>
          <li>blockShapeX: block size along the Q direction.</li>
          <li>blockShapeY: block size along the KV direction.</li>
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
      <td>The atten_mask in the formula.</td>
      <td>Currently not supported. Pass in nullptr.</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsOptional</td>
      <td>Input</td>
      <td>Describes the query sequence length of each batch.</td>
      <td>        <ul>
          <li>Can be passed as nullptr if not used.</li>
          <li>Used for variable-length sequence use cases.</li>
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
      <td>Describes the key/value sequence length of each batch.</td>
      <td>        <ul>
          <li>Can be passed as nullptr if not used.</li>
          <li>Used for variable-length sequence use cases.</li>
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
      <td>Block table used for PagedAttention.</td>
      <td>Currently not supported. Pass in nullptr.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qInputLayout</td>
      <td>Input</td>
      <td>Represents the data layout format of the input query.</td>
      <td>Currently only "TND" and "BNSD" are supported.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kvInputLayout</td>
      <td>Input</td>
      <td>Represents the data layout format of the input key and value.</td>
      <td>Currently only "TND" and "BNSD" are supported.</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads</td>
      <td>Input</td>
      <td>Indicates the number of key/value heads.</td>
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
      <td>0 indicates no mask; other values indicate different mask types.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>scale in the formula. Denotes the scaling factor.</td>
      <td>Generally set to D^-0.5.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise</td>
      <td>Input</td>
      <td>Softmax precision control.</td>
      <td>0 indicates float32 softmax, and 1 indicates fp16 softmax.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize</td>
      <td>Input</td>
      <td>Size of the PagedAttention block.</td>
      <td>For PagedAttention use cases. Pass in 0 if not used.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>Output</td>
      <td>attentionOut in the formula.</td>
      <td>Data type and shape are the same as those of query.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxLseOptional</td>
      <td>Output</td>
      <td>Intermediate log-sum-exp result of the Softmax computation.</td>
      <td>Currently not supported. Pass in nullptr.</td>
      <td>FLOAT</td>
      <td>ND</td>
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
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**

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
      <td>A null pointer is passed in for query, key, value, selectIdx, or selectNumIdx.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of query, key, and value are not within the supported range.</td>
    </tr>
    <tr>
      <td>qInputLayout or kvInputLayout is invalid.</td>
    </tr>
    <tr>
      <td>blockShape is invalid (fewer than 2 elements or a value less than or equal to 0).</td>
    </tr>
    <tr>
      <td>innerPrecise is invalid (must be 0 or 1).</td>
    </tr>
  </tbody>
  </table>

## aclnnRainFusionAttention

- **Parameters:**

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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by the first-phase API aclnnRainFusionAttentionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor. Contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRainFusionAttention defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- qInputLayout currently supports only "TND" and "BNSD".
- kvInputLayout currently only supports "TND" and "BNSD".
- The data types of the input query, key, and value must be consistent, supporting FLOAT16 and BFLOAT16.
- blockShape must contain at least two elements [blockShapeX, blockShapeY], and their values must be greater than 0.
- The shape of selectIdx must be [T, headNum, maxKvBlockNum], where T is the total number of Q-direction blocks across all batches.
- The shape of selectNumIdx must be [T, headNum].
- innerPrecise must be 0 (float32 softmax) or 1 (fp16 softmax). When the query input is BFLOAT16, the current implementation supports only innerPrecise=0. If a non-zero value is passed, the operator automatically normalizes it to 0 and uses float32 softmax.
- qSeqlen and kvSeqlen do not need to be divisible by blockShape. Non-aligned scenarios are supported, and the actual number of blocks is calculated by rounding up.
- qSeqlen is mandatory when qInputLayout is "TND" or "BNSD". kvSeqlen is mandatory when kvInputLayout is "TND" or "BNSD".
- Sparse block indices must be within the valid range. Invalid positions are filled with -1.
- Let the headNum of the input query be N1 and the headNum of the input key and value be N2. Then N1 >= N2 && N1 % N2 == 0.
- Let G = N1 / N2. G must satisfy the following constraints: G < 128 && 128 % G == 0.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
    // Boilerplate for AscendCL initialization
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
    // Check whether the shape is valid.
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

    // Check whether the hostData size matches.
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    // Call aclrtMalloc to allocate device-side memory.
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);

    // Calculate the strides of the contiguous tensor.
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
    // 1. (boilerplate) Device/stream initialization.
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
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;  // Number of Q blocks in the X dimension
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;  // Number of KV blocks in the Y dimension
    // totalQBlocks = qBlockNum * numHeads (one head per Q block)
    int32_t totalQBlocks = qBlockNum * batch;
    int32_t maxKvBlockNum = kvBlockNum;


    // 3. Create the Query tensor (TND format: [totalQTokens, numHeads, headDim]).
    void *queryDeviceAddr = nullptr;
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<op::fp16_t> queryHostData(totalQTokens * numHeads * headDim, 1.0f);
    aclTensor *queryTensor = nullptr;
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n"); return ret);

    // 4. Create the Key/Value tensor (TND format: [totalKvTokens, numKvHeads, headDim]).
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
    // Sparsity rate is 1, meaning no sparsity. Each Q block selects all KV blocks.
    std::vector<int64_t> selectIdxHostData(totalQBlocks * numHeads * maxKvBlockNum, -1);
    std::vector<int64_t> selectNumIdxHostData(totalQBlocks * numHeads, 0);

    // Sparsity rate is 1: each Q block selects all KV blocks, directly using indexes 0 to maxKvBlockNum-1.
    for (int32_t qb = 0; qb < totalQBlocks; ++qb) {
        for (int32_t h = 0; h < numHeads; ++h) {
            // selectNumIdx[qb, h] = maxKvBlockNum (each Q block selects all KV blocks)
            selectNumIdxHostData[qb * numHeads + h] = static_cast<int64_t>(maxKvBlockNum);

            // selectIdx[qb, h, k] = k (directly assign indexes from 0 to maxKvBlockNum-1)
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

    // 6. Create the output tensor.
    void *outputDeviceAddr = nullptr;
    std::vector<int64_t> outputShape = {totalQTokens, numHeads, headDim};
    int64_t outputElementCount = totalQTokens * numHeads * headDim;
    std::vector<op::fp16_t> outputHostData(outputElementCount, 0.0f);
    aclTensor *outputTensor = nullptr;
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create output tensor\n"); return ret);

    // 7. Create the blockShape array.
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n"); return -1);

    // 8. Create actualSeqLengths and actualSeqLengthsKv (required parameters).
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

    // aclCreateIntArray expects the host-side data pointer instead of the device-side data.
    aclIntArray *actualSeqLengths = aclCreateIntArray(actualSeqLengthsHost.data(), batch);
    aclIntArray *actualSeqLengthsKv = aclCreateIntArray(actualSeqLengthsKvHost.data(), batch);
    CHECK_RET(actualSeqLengths != nullptr && actualSeqLengthsKv != nullptr,
              LOG_PRINT("Failed to create actualSeqLengths arrays\n");
              if (actualSeqLengthsDevice) aclrtFree(actualSeqLengthsDevice);
              if (actualSeqLengthsKvDevice) aclrtFree(actualSeqLengthsKvDevice); return -1);

    // 9. Prepare the string parameters (ensure the buffer size is sufficient and includes the null terminator).
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

    // 12. Allocate the workspace.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 12. Call the second-phase API.
    ret = aclnnRainFusionAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRainFusionAttention failed. ERROR: %d\n", ret); return ret);

    // 13. Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 14. Obtain the output value and copy the result from device-side memory to the host side.
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
