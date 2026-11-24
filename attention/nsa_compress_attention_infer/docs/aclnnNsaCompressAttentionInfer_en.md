# aclnnNsaCompressAttentionInfer

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:00:40.938Z -->

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

- Description: Computation of Compress Attention during Native Sparse Attention inference.
- Formula:

<center>

  $
  P_{cmp}= Softmax(scale * query · key^T) \\
  attentionOut = P_{cmp} · value\\
  P_{slc}[j] = \sum\limits_{m=0}^{l'/d -1} \sum\limits_{n = 0}^{l/d -1} P_{cmp} [l'/d * j -m - n]\\
  P_{slc'} = \sum\limits_{g=1}^{G}  P_{slc} ^g,\quad
  \text{where } G = \text{GroupSize, i.e.: } G = \frac{\text{numHeads}}{\text{numKeyValueHeads}} \\
  topkIndices = topk(P_{slc'})\\
  $

</center>

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnNsaCompressAttentionInferGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the computation process of the operator, and then call the "aclnnNsaCompressAttentionInfer" API to perform the computation.

```cpp
aclnnStatus aclnnNsaCompressAttentionInferGetWorkspaceSize(
    const aclTensor    *query,
    const aclTensor    *key,
    const aclTensor    *value,
    const aclTensor    *attentionMaskOptional,
    const aclTensor    *blockTableOptional,
    const aclIntArray  *actualQSeqLenOptional,
    const aclIntArray  *actualCmpKvSeqLenOptional,
    const aclIntArray  *actualSelKvSeqLenOptional,
    const aclTensor    *topKMaskOptional,
    int64_t             numHeads,
    int64_t             numKeyValueHeads,
    int64_t             selectBlockSize,
    int64_t             selectBlockCount,
    int64_t             compressBlockSize,
    int64_t             compressBlockStride,
    double              scaleValue,
    char               *layoutOptional,
    int64_t             pageBlockSize,
    int64_t             sparseMode,
    const aclTensor    *output,
    const aclTensor    *topKOutput,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor
)
```

```cpp
aclnnStatus aclnnNsaCompressAttentionInfer(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream
)
```

## aclnnNsaCompressAttentionInferGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1567px">
  <colgroup>
    <col style="width: 170px">  <!-- Parameter -->
    <col style="width: 120px">  <!-- Input/Output -->
    <col style="width: 300px">  <!-- Description -->
    <col style="width: 330px">  <!-- Instruction -->
    <col style="width: 212px">  <!-- Data Type -->
    <col style="width: 100px">  <!-- Data Format -->
    <col style="width: 190px">  <!-- Dimensions (shape) -->
    <col style="width: 145px">  <!-- Non-contiguous Tensor -->
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
      <td>Query input of the attention structure.</td>
      <td>        <ul style="list-style-type: circle;">
          <li>B in query is an integer in the range [1, 10000], and is equal to B in blockTableOptional and the length of the actualCmpKvSeqLenOptional array.</li>
          <li>The S axis of query is less than or equal to 4.</li>
          <li>The values of N in query and numHeads are equal, and the N axis must be an integer multiple of the N axis (H/D) of key/value. In addition, the ratio of the N axis of query to the N axis (H/D) of key/value (that is, the group size in GQA) is less than or equal to 128, and 128 is an integer multiple of the group.</li>
          <li>The values of D in query and D (H/numKeyValueHeads) of key are equal, less than or equal to 192, and greater than or equal to the D axis of value.</li>
          <li>The data types of the query, key, and value inputs are exactly the same, which is FLOAT16 or BFLOAT16.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[BSND], [TND]</td>
      <td>x</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Key input of the attention structure.</td>
      <td>        <ul style="list-style-type: circle;">
          <li>numBlocks in key is equal to numBlocks in the parameter value.</li>
          <li>blockSize in key is equal to pageBlockSize, blockSize is less than or equal to 128, and blockSize is an integer multiple of 16.</li>
          <li>The S axis of key is less than or equal to 8192.</li>
          <li>N in key is equal to numKeyValueHeads.</li>
          <li>The D axis of key is less than or equal to 192 and greater than or equal to the D axis of value.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads * headDimsQK]</td>
      <td>x</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Value input of the attention structure.</td>
      <td>        <ul style="list-style-type: circle;">
          <li>N in value is equal to numKeyValueHeads.</li>
          <li>D (H/numKeyValueHeads) in value is equal to D in output.</li>
     <li>The D axis of value is less than or equal to 128.</li>
          <li>blockSize in value is equal to pageBlockSize.</li>
          <li>The S axis of value is less than or equal to 8192.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads * headDimsV]</td>
      <td>x</td>
    </tr>
    <tr>
      <td>attentionMaskOptional</td>
      <td>Optional input</td>
      <td>Attention mask matrix.</td>
      <td>Takes effect only when Q_S is greater than 1</td>
      <td>-</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>Input</td>
      <td>Block mapping table used for KV storage in paged attention.</td>
      <td>Currently, only paged attention is supported. Therefore, this parameter is required.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualQSeqLenOptional</td>
      <td>Optional input</td>
      <td>Actual S-axis length of query.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>[B]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualCmpKvSeqLenOptional</td>
      <td>Optional input</td>
      <td>Actual S-axis length of the compressed key and value, that is, the actual S-axis length of the key and value processed by this operator.</td>
      <td>Currently, only paged attention is supported. Therefore, this parameter is required.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSelKvSeqLenOptional</td>
      <td>Optional input</td>
      <td>Actual length of the S axis of key and value before compression.</td>
      <td>Required for multi-token inference. The array length must equal the batch dimension of blockTableOptional.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topKMaskOptional</td>
      <td>Optional input</td>
      <td>Mask matrix in the topK computation.</td>
      <td>Reserved parameter, not used currently.</td>
      <td>-</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeads</td>
      <td>Input</td>
      <td>Number of heads.</td>
      <td>numHeads is an integer multiple of numKeyValueHeads.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads</td>
      <td>Input</td>
      <td>Number of kvHeads.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selectBlockSize</td>
      <td>Input</td>
      <td>Block size of the select phase, used for calculating the importance score.</td>
      <td>Only supports selectBlockSize values 16, 32, 48, 64, 80, 96, 112, and 128. selectBlockSize must be greater than or equal to compressBlockSize and an integer multiple of compressBlockStride.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selectBlockCount</td>
      <td>Input</td>
      <td>Number of blocks to be retained in the topK phase.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressBlockSize</td>
      <td>Input</td>
      <td>Sliding window size during compression.</td>
      <td>Only supports compressBlockSize values of 16, 32, 48, 64, 80, 96, 112, and 128, and the value must be greater than or equal to compressBlockStride.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressBlockStride</td>
      <td>Input</td>
      <td>Size of the sliding window interval between two compressions.</td>
      <td>compressBlockStride only supports values 16, 32, 48, and 64.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Scaling factor, used as the scalar value of Muls in the computation stream.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutOptional</td>
      <td>Input</td>
      <td>Data layout format of the input query, key, and value.</td>
      <td>Currently, the value can be "TND" or "BSND".</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pageBlockSize</td>
      <td>Input</td>
      <td>Size of a block in blockTableOptional.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Sparse mode. Controls sparse computation when attentionMaskOptional is provided.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Output</td>
      <td>Output</td>
      <td>Output of attention.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[BSND], [TND]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topKOutput</td>
      <td>Output</td>
      <td>Output of topK.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[T, N, selectBlockCount], [B, S, N, selectBlockCount]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
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

  <table style="undefined;table-layout: fixed; width: 1150px">
  <colgroup>
  <col style="width: 285px">
  <col style="width: 120px">
  <col style="width: 745px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The parameter passed in is a required input, output, or required attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types of query, key, value, blockTableOptional, attentionOut, and topKOutput are not in the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The API fails to call the npu runtime API for memory operations.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaCompressAttentionInfer

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
      <td>Pointer to the workspace memory allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the device, obtained by the first-phase API aclnnNsaCompressAttentionInfer.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  Check the return values of both API phases separately. For asynchronous execution, also check the return value of the stream synchronization API.

## Constraints

- Deterministic computation:
  - aclnnNsaCompressAttentionInfer defaults to a deterministic implementation.

- The query parameter only supports TND and BSND inputs. T is the tightly packed data of the B and S axes (actualQSeqLenOptional for each batch), B (batch) indicates the input sample batch size, S (qSeqlen) indicates the input sample sequence length, N (numHeads) indicates the number of multi-heads, and D (headDimsQK) indicates the minimum unit size of the hidden layer.

- The length before compression derived from the actual compressed KV length can be expressed as: actualSelKvSeqLenCeil = (actualCmpKvSeqLenOptional - 1) * compressBlockStride + compressBlockSize. The conditions ceil(actualSelKvSeqLenCeil / selectBlockSize) <= 4096 and selectBlockCount <= ceil(actualSelKvSeqLenCeil / selectBlockSize) must hold for every batch, where ceil rounds up. If actualSelKvSeqLenOptional does not satisfy actualCmpKvSeqLenOptional = (actualSelKvSeqLenOptional - compressBlockSize) / compressBlockStride + 1, or the length of actualSelKvSeqLenOptional differs from the batch dimension of blockTableOptional, the single-token inference scenario is entered by default.
- In multi-token inference scenarios, both actualQSeqLenOptional and actualSelKvSeqLenOptional must be passed in. The length of each array must equal the batch dimension of blockTableOptional. It only supports a maximum S-axis of query equal to 4, and in this case, each individual batch's actualQSeqLenOptional must be less than or equal to actualSelKvSeqLenOptional. If the length of actualQSeqLenOptional differs from the batch dimension of blockTableOptional, or the value of actualQSeqLenOptional is less than 1 or greater than 4, the single-token inference scenario is entered by default.

## Example

The sample code is as follows and is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_nsa_compress_attention_infer.h"

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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the list of AscendCL public APIs.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  int32_t batchSize = 20;
  int32_t headDimsQK = 192;
  int32_t blockNum = 640;
  int32_t headDimsV = 128;
  int32_t sequenceLengthK = 4096;
  int32_t maxNumBlocksPerSeq = 32;
  // attr
    int64_t numHeads = 64;
    int64_t numKeyValueHeads = 4;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 16;
    int64_t compressBlockSize = 32;
    int64_t compressBlockStride = 16;
    double scaleValue = 0.088388;
  string sLayerOut = "TND";
  char layOut[sLayerOut.length()+1];
  strcpy(layOut, sLayerOut.c_str());
    int64_t pageBlockSize = 128;
    int64_t sparseMode = 0;
  std::vector<int64_t> queryShape = {batchSize, numHeads, headDimsQK};
  std::vector<int64_t> keyShape = {blockNum, pageBlockSize, numKeyValueHeads * headDimsQK};
  std::vector<int64_t> valueShape = {blockNum, pageBlockSize, numKeyValueHeads * headDimsV};
  std::vector<int64_t> blockTableOptionalShape = {batchSize, maxNumBlocksPerSeq};
    std::vector<int64_t> outputShape = {batchSize, numHeads, headDimsV};
    std::vector<int64_t> topkIndicesShape = {batchSize, numKeyValueHeads, selectBlockCount};
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *blockTableOptionalDeviceAddr = nullptr;
  void *outputDeviceAddr = nullptr;
  void *topkIndicesDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *blockTableOptionalTensor = nullptr;
  aclTensor *outputTensor = nullptr;
  aclTensor *topkIndicesTensor = nullptr;
  std::vector<op::fp16_t> queryHostData(batchSize * numHeads * headDimsQK, 1.0);
  std::vector<op::fp16_t> keyHostData(blockNum * pageBlockSize * numKeyValueHeads * headDimsQK, 1.0);
  std::vector<op::fp16_t> valueHostData(blockNum * pageBlockSize * numKeyValueHeads * headDimsV, 1.0);
  std::vector<int32_t> blockTableOptionalHostData(batchSize * maxNumBlocksPerSeq, 1);
  std::vector<op::fp16_t> outputHostData(batchSize * numHeads * headDimsV, 1.0);
  std::vector<int32_t> topkIndicesHostData(batchSize * numKeyValueHeads * selectBlockCount, 1);

  // Create the query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the key aclTensor.
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the v aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the blockTableOptional aclTensor.
  ret = CreateAclTensor(blockTableOptionalHostData, blockTableOptionalShape, &blockTableOptionalDeviceAddr, aclDataType::ACL_INT32, &blockTableOptionalTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the output aclTensor.
  ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the topkIndices aclTensor.
  ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndicesTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualCmpKvSeqLenVector(batchSize, sequenceLengthK);
    auto actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqLenVector.data(), actualCmpKvSeqLenVector.size());

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnNsaCompressAttentionInferGetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, blockTableOptionalTensor, nullptr, actualCmpKvSeqLen,
        nullptr, nullptr,
        numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride,
        scaleValue, layOut, pageBlockSize, sparseMode, outputTensor, topkIndicesTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionInferGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory according to the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnNsaCompressAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionInfer failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task execution to complete synchronously.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, copy the results from device memory to the host side, and modify the code according to the specific API definition.
  auto size = GetShapeSize(outputShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
            size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [attn] result from device to host failed. ERROR: %d\n", ret); return ret);
  uint64_t printNum = 10;
  for (int64_t i = 0; i < printNum; i++) {
    std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }
    auto topksize = GetShapeSize(topkIndicesShape);
  std::vector<int32_t> topkresultData(topksize, 0);
  ret = aclrtMemcpy(topkresultData.data(), topkresultData.size() * sizeof(topkresultData[0]), topkIndicesDeviceAddr,
            topksize * sizeof(topkresultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [top k] result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < printNum; i++) {
    std::cout << "topk index: " << i << ": " << topkresultData[i] << std::endl;
  }

  // 6. Release resources.
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(blockTableOptionalTensor);
  aclDestroyIntArray(actualCmpKvSeqLen);
  aclDestroyTensor(outputTensor);
  aclDestroyTensor(topkIndicesTensor);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(blockTableOptionalDeviceAddr);
  aclrtFree(outputDeviceAddr);
  aclrtFree(topkIndicesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
