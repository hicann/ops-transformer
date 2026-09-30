# aclnnNsaCompressAttentionInfer

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Performs compressed attention computation during the Native Sparse Attention (NSA) inference.
- Formulas:

<center>

  $$
  P_{cmp}= Softmax(scale * query · key^T) \\
  attentionOut = P_{cmp} · value\\
  P_{slc}[j] = \sum\limits_{m=0}^{l'/d -1} \sum\limits_{n = 0}^{l/d -1} P_{cmp} [l'/d * j -m - n]\\
  P_{slc'} = \sum\limits_{g=1}^{G}  P_{slc} ^g,\quad 
  \text{where } G (group size) = \frac{\text{numHeads}}{\text{numKeyValueHeads}} \\
  topkIndices = topk(P_{slc'})\\
  $$

</center>

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNsaCompressAttentionInferGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnNsaCompressAttentionInfer` is called to perform computation.

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
    <col style="width: 170px">  <!-- Parameter Name -->
    <col style="width: 120px">  <!-- Input/Output -->
    <col style="width: 300px">  <!-- Description -->
    <col style="width: 330px">  <!-- Usage Notes -->
    <col style="width: 212px">  <!-- Data Type -->
    <col style="width: 100px">  <!-- Data Format -->
    <col style="width: 190px">  <!-- Dimension (Shape) -->
    <col style="width: 145px">  <!-- Non-contiguous Tensor -->
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
      <td>Query input of the attention structure.</td>
      <td>
        <ul style="list-style-type: circle;">
          <li>The <code>B</code> value of <code>query</code> is an integer within the range of [1, 10000], and is equal to the <code>B</code> value of <code>blockTableOptional</code> and the length of the <code>actualCmpKvSeqLenOptional</code> array.</li>
          <li>The <code>S</code> value of <code>query</code> is less than or equal to <code>4</code>.</li>
          <li>The <code>N</code> value of <code>query</code> is equal to the value of <code>numHeads</code>, and must be an integer multiple of the <code>N</code> value (<code>H/D</code>) of <code>key</code>/<code>value</code>. In addition, the ratio of the <code>N</code> value of <code>query</code> to the <code>N</code> value of <code>key</code>/<code>value</code> (that is, the group size in GQA) must be less than or equal to <code>128</code>, and <code>128</code> is an integer multiple of the group size.</li>
          <li>The <code>D</code> value of <code>query</code> is equal to the <code>D</code> value (<code>H/numKeyValueHeads</code>) of <code>key</code>, and must be less than or equal to <code>192</code> and greater than or equal to the <code>D</code> value of <code>value</code>.</li>
          <li>The input data types of <code>query</code>, <code>key</code>, and <code>value</code> must be the same, which can be FLOAT16 or BFLOAT16.</li>
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
      <td>
        <ul style="list-style-type: circle;">
          <li>The <code>numBlocks</code> values of <code>key</code> and <code>value</code> must be the same.</li>
          <li>The <code>blockSize</code> value of <code>key</code> is equal to the value of <code>pageBlockSize</code>, and must be less than or equal to <code>128</code> and be an integer multiple of 16.</li>
          <li>The <code>S</code> value of <code>key</code> is less than or equal to <code>8192</code>.</li>
          <li>The <code>N</code> value of <code>key</code> is equal to the value of <code>numKeyValueHeads</code>.</li>
          <li>The <code>D</code> value of <code>key</code> is less than or equal to <code>192</code> and greater than or equal to the <code>D</code> value of <code>value</code>.</li>
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
      <td>
        <ul style="list-style-type: circle;">
          <li>The <code>N</code> value of <code>value</code> is equal to the value of <code>numKeyValueHeads</code>.</li>
          <li>The <code>D</code> value (<code>H/numKeyValueHeads</code>) of <code>value</code> is equal to the <code>D</code> value of <code>output</code>.</li>
      <li>The <code>D</code> value of <code>value</code> is less than or equal to <code>128</code>.</li>
          <li>The <code>blockSize</code> value of <code>value</code> is equal to the value of <code>pageBlockSize</code>.</li>
          <li>The <code>S</code> value of <code>value</code> is less than or equal to <code>8192</code>.</li>
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
      <td>This parameter is valid only when <code>Q_S</code> is greater than 1.</td>
      <td>-</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>Input</td>
      <td>Block mapping table used for KV storage in paged attention.</td>
      <td>Currently, only paged attention is supported. Therefore, this parameter must be passed.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualQSeqLenOptional</td>
      <td>Optional input</td>
      <td>Actual <code>S</code> value of <code>query</code>.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>[B]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualCmpKvSeqLenOptional</td>
      <td>Optional input</td>
      <td>Actual <code>S</code> value of <code>key</code> and <code>value</code> after compression, that is, the actual <code>S</code> value of <code>key</code> and <code>value</code> processed by the operator.</td>
      <td>Currently, only paged attention is supported. Therefore, this parameter must be passed.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSelKvSeqLenOptional</td>
      <td>Optional input</td>
      <td>Actual <code>S</code> value of <code>key</code> and <code>value</code> before compression.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topKMaskOptional</td>
      <td>Optional input</td>
      <td>Mask matrix in topK computation.</td>
      <td>This parameter is reserved and not used currently.</td>
      <td>-</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeads</td>
      <td>Input</td>
      <td>Number of heads.</td>
      <td>The value of <code>numHeads</code> is a multiple of that of <code>numKeyValueHeads</code>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads</td>
      <td>Input</td>
      <td>Number of heads in <code>key</code>/<code>value</code>.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selectBlockSize</td>
      <td>Input</td>
      <td>Size of the selected block, which is used for computing the importance score.</td>
      <td>The value of <code>selectBlockSize</code> can only be <code>16</code>, <code>32</code>, <code>48</code>, <code>64</code>, <code>80</code>, <code>96</code>, <code>112</code>, or <code>128</code>, and must be greater than or equal to that of <code>compressBlockSize</code> and be an integer multiple of that of <code>compressBlockStride</code>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selectBlockCount</td>
      <td>Input</td>
      <td>Number of blocks to be retained.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressBlockSize</td>
      <td>Input</td>
      <td>Size of the compression sliding window.</td>
      <td>The value of <code>compressBlockSize</code> can only be <code>16</code>, <code>32</code>, <code>48</code>, <code>64</code>, <code>80</code>, <code>96</code>, <code>112</code>, or <code>128</code>, and must be greater than or equal to that of <code>compressBlockStride</code>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressBlockStride</td>
      <td>Input</td>
      <td>Sliding window interval between two compressions.</td>
      <td>The value of <code>compressBlockStride</code> can only be <code>16</code>, <code>32</code>, <code>48</code>, or <code>64</code>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Scaling coefficient, which is used as the scalar value of <code>Muls</code> in the computation stream.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutOptional</td>
      <td>Input</td>
      <td>Layout of the input <code>query</code>, <code>key</code>, and <code>value</code>.</td>
      <td>The value can be <code>TND</code> or <code>BSND</code>.</td>
      <td>CHAR</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pageBlockSize</td>
      <td>Input</td>
      <td>Size of a block in <code>blockTableOptional</code>.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Sparse mode, which controls sparse computation when <code>attentionMaskOptional</code> is input.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Attention output.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>[BSND], [TND]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topKOutput</td>
      <td>Output</td>
      <td>TopK output.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[T, N, selectBlockCount], [B, S, N, selectBlockCount]</td>
      <td>-</td>
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
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of <code>query</code>, <code>key</code>, <code>value</code>, <code>blockTableOptional</code>, <code>attentionOut</code>, or <code>topKOutput</code> is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurred when the NPU runtime API was called.</td>
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API <code>aclnnNsaCompressAttentionInferGetWorkspaceSize</code>.</td>
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
  - `aclnnNsaCompressAttentionInfer` defaults to a deterministic 
  implementation.

* `query` supports only TND and BSND inputs. `T` indicates the total length of all input sample sequences (`actualQSeqLenOptional` of all batches), `B` (`batch`) indicates the size of an input sample batch, `S` (`qSeqlen`) indicates the length of the input sample sequence, `N` (`numHeads`) indicates the number of heads, and `D` (`headDimsQK`) indicates the minimum unit size of the hidden layer.
* The upper limit of `kvSeqlen` before compression can be expressed as follows: `actualSelKvSeqLenCeil` = (`actualCmpKvSeqLenOptional` – 1) × `compressBlockStride` + `compressBlockSize`. The following conditions must be met: `actualSelKvSeqLenCeil/selectBlockSize<=4096` and `selectBlockCount<=actualSelKvSeqLenCeil/selectBlockSize`. If the value of `actualSelKvSeqLenOptional` does not meet the formula `actualCmpKvSeqLenOptional = (actualSelKvSeqLenOptional – compressBlockSize)/compressBlockStride + 1` or the value of `actualCmpKvSeqLenOptional` is different from the `batch` dimension of `blockTableOptional`, the single-token inference scenario is used by default.
* In the multi-token inference scenario, the `actualQSeqLenOptional` parameter must be passed, and the value of `actualQSeqLenOptional` must be the same as the `batch` dimension of `blockTableOptional`. In addition, the maximum `S` value of `query` is `4`, the value of `actualQSeqLenOptional` for each batch must be less than or equal to that of `actualSelKvSeqLenOptional`. If the value of `actualQSeqLenOptional` is different from the `batch` dimension of `blockTableOptional`, or the value of `actualQSeqLenOptional` is less than 1 or greater than 4, the single-token inference scenario is used by default.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  // (Boilerplate) Initialize resources.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
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
    int64_t compressStride = 16;
    double scaleValue = 0.088388;
  string sLayerOut = "TND";
  char layOut[sLayerOut.length()];
  strcpy(layOut, sLayerOut.c_str());
    int64_t pageBlockSize = 128;
    int64_t sparseMod = 0;
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

  // Create a query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a key aclTensor.
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a v aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a blockTableOptional aclTensor.
  ret = CreateAclTensor(blockTableOptionalHostData, blockTableOptionalShape, &blockTableOptionalDeviceAddr, aclDataType::ACL_INT32, &blockTableOptionalTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an output aclTensor.
  ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a topkIndices aclTensor.
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
        numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressStride,
        scaleValue, layOut, pageBlockSize, sparseMod, outputTensor, topkIndicesTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionInferGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnNsaCompressAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionInfer failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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
  std::vector<op::fp16_t> topkresultData(topksize, 0);
  ret = aclrtMemcpy(topkresultData.data(), topkresultData.size() * sizeof(topkresultData[0]), topkIndicesDeviceAddr,
            topksize * sizeof(topkresultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [top k] result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < printNum; i++) {
    std::cout << "topk index: " << i << ": " << static_cast<int32_t>(topkresultData[i]) << std::endl;
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
