# aclnnNsaSelectedAttentionInfer

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:13:02.664Z -->

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

- Description: Computation of Selected Attention during Native Sparse Attention inference.
- Formula:

  Self-attention leverages the relationships within the input samples themselves to construct an attention model. The principle is as follows: given an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector, each $d$-dimensional vector can be regarded as a token embedding. This sequence is transformed through three weight matrices to obtain three matrices of dimension $n*d$.

  The computation of Selected Attention combines topk index-based retrieval with attention computation, along with paged attention for fetching kvCache. First, $key_{topk}$ is retrieved from $key$ and $value_{topk}$ is retrieved from $value$ using the $topkIndices$ index, and the self-attention formula is computed as follows:

  $$
  Attention(query,key,value)=Softmax(\frac{query · key_{topk}^T}{\sqrt{d}})value_{topk}
  $$

  Here, the product of $query$ and $key_{topk}^T$ represents the attention of the input $x$. To prevent this value from becoming too large, it is typically scaled by dividing by the square root of $d$, and each row is normalized using softmax. After multiplying with $value_{topk}$, an $n*d$ matrix is obtained.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnNsaSelectedAttentionInferGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then call "aclnnNsaSelectedAttentionInfer" to perform the computation.

```c++
aclnnStatus aclnnNsaSelectedAttentionInferGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *topkIndices,
    const aclTensor     *attenMaskOptional,
    const aclTensor     *blockTableOptional,
    const aclIntArray   *actualQSeqLenOptional,
    const aclIntArray   *actualKvSeqLenOptional,
    char                *layoutOptional,
    int64_t              numHeads,
    int64_t              numKeyValueHeads,
    int64_t              selectBlockSize,
    int64_t              selectBlockCount,
    int64_t              pageBlockSize,
    double               scaleValue,
    int64_t              sparseMode,
    aclTensor           *output,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnNsaSelectedAttentionInfer(
    void                *workspace,
    uint64_t             workspaceSize,
    aclOpExecutor       *executor,
    const aclrtStream    stream)
```

## aclnnNsaSelectedAttentionInferGetWorkspaceSize

- **Parameters**

  <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1567px">
      <colgroup>
        <col style="width: 232px">
        <col style="width: 120px">
        <col style="width: 270px">
        <col style="width: 300px">
        <col style="width: 212px">
        <col style="width: 100px">
        <col style="width: 188px">
        <col style="width: 145px">
      </colgroup>
      <thead>
        <tr>
          <th style="font-weight: bold;">Parameter</th>
          <th style="font-weight: bold;">Input/Output</th>
          <th style="font-weight: bold;">Description</th>
          <th style="font-weight: bold;">Instruction</th>
          <th style="font-weight: bold;">Data Type</th>
          <th style="font-weight: bold;">Data Format</th>
          <th style="font-weight: bold;">Dimension (shape)</th>
          <th style="font-weight: bold;">Non-contiguous Tensor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="white-space: nowrap;">query</td>
          <td>Input</td>
          <td>The query input in the formula.</td>
          <td>            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>The data type is the same as that of key and value.</li>
              <li>Supports a value less than or equal to 16 for the ratio of the N axis of query to the N axis (H/D) of key/value (that is, the group size in GQA).</li>
              <li>Supports the D axis of query equal to 192.</li>
              <li>Only the S axis of query equal to 1 is supported in normal use cases.</li>
               <li>The N value in query is equal to numHeads, and numHeads is a multiple of numKeyValueHeads.</li>
               <li>The D value in query is equal to the D value (H/numKeyValueHeads) of key.</li>
            </ul>
          </td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3/4D</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">key</td>
          <td>Input</td>
          <td>Input key in the formula.</td>
          <td>            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>The data type is the same as that of query and value.</li>
              <li> Supports an N axis of key less than or equal to 256.</li>
               <li>Supports a D axis of key equal to 192.</li>
               <li>Supports a blockSize of key equal to 64 or 128.</li>
                <li>The N of key is equal to numHeads, and numHeads is a multiple of numKeyValueHeads.</li>
            </ul>
          </td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3D/4D</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">value</td>
          <td>Input</td>
          <td>Input value in the formula.</td>
          <td>            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>The data type is the same as that of query and key.</li>
              <li>Supports an N axis of value less than or equal to 256.</li>
               <li>Supports a D axis of value equal to 128.</li>
               <li>Supports a blockSize of value equal to 64 or 128.</li>
               <li>The N of value is equal to numHeads, and numHeads is a multiple of numKeyValueHeads.</li>
               <li>The D(H/numKeyValueHeads) of value is equal to the D of output.</li>
            </ul>
          </td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3/4 dimensions</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">topkIndices</td>
          <td>Input</td>
          <td>The topK indices in the formula.</td>
          <td>-
          </td>
          <td>INT32</td>
          <td>ND</td>
          <td>3/4 dimensions</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">attenMask</td>
          <td>Input</td>
          <td>Indicates the attention mask matrix.</td>
          <td>            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>Optional parameter.</li>
              <li>Pass nullptr if this function is not used.</li>
              <li>Reserved parameter, not used yet.</li>
            </ul>
          </td>
          <td>-</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">blockTableOptional</td>
          <td>Input</td>
          <td>Indicates the block mapping table used for KV storage in paged attention.</td>
          <td>            -
          </td>
          <td>INT32</td>
          <td>ND</td>
          <td>2D</td>
          <td>×</td>
        </tr>
        <tr>
          <td>actualQSeqLenOptional</td>
          <td>Input</td>
          <td>Indicates the actual length of the S axis of query.</td>
          <td>      If this feature is not used, pass nullptr.
          </td>
          <td>INT64</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>actualKvSeqLenOptional</td>
          <td>Input</td>
          <td>Indicates the actual length of the S axis of the key and value processed by the operator.</td>
          <td>-
          </td>
          <td>INT64</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">layoutOptional</td>
          <td>Input</td>
          <td>Indicates the data layout format of input query, key, and value.</td>
          <td>            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>Currently, BSH/BSND/TND are supported.</li>
              <li>If this parameter is not passed, the default value is "BSND", corresponding to the 3/4 dimensions of query, key, and value respectively.</li>
              <li>In the query data layout, B indicates Batch, S indicates Seq-Length, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the minimum unit size of the hidden layer, and D=H/N must be satisfied. The key and value data layouts currently (paged attention) support (blocknum, blocksize, H) and (blocknum, blocksize, N, D). H (Head-Size) indicates the size of the hidden layer, where H = N * D.</li>
            </ul>
          </td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">numHeads</td>
          <td>Input</td>
          <td>Represents the number of heads.</td>
          <td>            -
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">numKeyValueHeads</td>
          <td>Input</td>
          <td>Represents the number of kvHeads.</td>
          <td>            -
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">selectBlockSize</td>
          <td>Input</td>
          <td>Represents the block size in the select phase.</td>
          <td>            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>Used when computing importance score.</li>
              <li>Only selectBlockSize values that are integer multiples of 16 are supported, with a maximum of 128.</li>
            </ul>
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">selectBlockCount</td>
          <td>Input</td>
          <td>Represents the number of blocks to be retained in the topK phase.</td>
          <td>The upper limit of selectBlockCount satisfies selectBlockCount * selectBlockSize <= MaxKvSeqlen, where MaxKvSeqlen = Max(actualKvSeqLenOptional).
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">pageBlockSize</td>
          <td>Input</td>
          <td>Represents the block size of paged attention.</td>
          <td>      Used to fetch data from the kv cache.
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">scaleValue</td>
          <td>Input</td>
          <td>Reciprocal of the square root of d in the formula. Represents the scaling factor.</td>
          <td>      Serves as the scalar value of Muls in the computation stream.
          </td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">sparseMode</td>
          <td>Input</td>
          <td>Indicates the sparse mode. Controls sparse computation when attentionMask is input.</td>
          <td>      Reserved parameter, not used yet.
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">output</td>
          <td>Output</td>
          <td>Attention output in the formula.</td>
          <td>      Receives the attention computation result.
          </td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">workspaceSize</td>
          <td>Output</td>
          <td>Size of the workspace that the user needs to request on the Device.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">executor</td>
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
  </div>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <div style="overflow-x: auto;">
  <table style="table-layout: fixed; width: 1030px">  <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 650px">
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
        <td>The passed-in parameter is a required input, output, or required attribute, and is a null pointer.</td>
      </tr>
      <tr>
        <!-- Add the merged-cell class to merged cells for vertical centering. -->
        <td class="merged-cell" rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td class="merged-cell" rowspan="2">161002</td>
        <td>The data types of query, key, value, topkIndices, attenMask, blockTableOptional, actualQSeqLenOptional, actualKvSeqLenOptional, and output are not in the supported range.</td>
      </tr>
      <tr>
        <td>The data formats of query, key, value, topkIndices, attenMask, blockTableOptional, actualQSeqLenOptional, actualKvSeqLenOptional, and output are not in the supported range.</td>
      </tr>
    </tbody>
  </table>
  </div>

## aclnnNsaSelectedAttentionInfer

- **Parameters**

  <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 1030px">
      <colgroup>
          <col style="width: 250px">
          <col style="width: 130px">
          <col style="width: 650px">
      </colgroup>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Input/Output</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>workspace</td>
          <td>Input</td>
          <td>Workspace memory address requested by DevicenumHeads.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Workspace size requested on the Device side, obtained by the first-phase API aclnnNsaSelectedAttentionInferGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Op executor, containing the operator computation flow.</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>Input</td>
          <td>Specifies the Stream for task execution.</td>
        </tr>
      </tbody>
    </table>
  </div>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNsaSelectedAttentionInfer defaults to a deterministic implementation.
- The B axis must be less than or equal to 3072.
- Only paged attention is supported.
- In multi-token inference scenarios, the S axis of the query must be less than or equal to 4, and each batch must satisfy actualQSeqLen <= actualSelKvSeqLen.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_nsa_selected_attention_infer.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: resource initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main(int argc, char **argv)
{
    // 1. (Boilerplate) Initialize device/context/stream. See the AscendCL external API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Perform custom construction based on the API.
    // If you need to modify the shape value, synchronously modify the data generated under the test_nsa_selected_attention_infer branch in ../scripts/fa_generate_data.py.
    // Update the shape values of query, key, and value, regenerate data, and then execute.

    int64_t batch = 1;
    int64_t kvSeqlen = 48;
    aclIntArray * actualCmpKvSeqLen = nullptr;
    aclIntArray * actualCmpQSeqLen = nullptr;
    // Create actualCmpKvSeqLen aclIntArray.
    std::vector<int64_t> actualCmpKvSeqLenVector(batch, kvSeqlen);
    actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqLenVector.data(), actualCmpKvSeqLenVector.size());
    // Create actualCmpQSeqLen aclIntArray.
    int64_t qSeqlen = 1;
    std::vector<int64_t> actualCmpQSeqLenVector(batch, qSeqlen);
    actualCmpQSeqLen = aclCreateIntArray(actualCmpQSeqLenVector.data(), actualCmpQSeqLenVector.size());
    int64_t d1 = 192;
    int64_t d2 = 128;
    int64_t numHeads = 1;
    int64_t numKeyValueHeads = 1;
    int64_t blockSize = 64;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 1;
    int64_t blockTableLength = 1;
    int64_t numBlocks = batch * blockTableLength;
    std::vector<int64_t> queryShape = {batch, qSeqlen, numHeads, d1};
    std::vector<int64_t> keyShape = {numBlocks, blockSize, numKeyValueHeads, d1};
    std::vector<int64_t> valueShape = {numBlocks, blockSize, numKeyValueHeads, d2};
    std::vector<int64_t> topkIndicesShape = {batch, qSeqlen, numKeyValueHeads, selectBlockCount};
    std::vector<int64_t> blockTableOptionalShape = {batch, blockTableLength};
    std::vector<int64_t> outputShape = {batch, qSeqlen, numHeads, d2};

    long long queryShapeSize = GetShapeSize(queryShape);
    long long keyShapeSize = GetShapeSize(keyShape);
    long long valueShapeSize = GetShapeSize(valueShape);
    long long blockTableOptionalShapeSize = GetShapeSize(blockTableOptionalShape);
    long long outputShapeSize = GetShapeSize(outputShape);
    long long topkIndicesShapeSize = GetShapeSize(topkIndicesShape);

    std::vector<op::fp16_t> queryHostData(queryShapeSize, 1);
    std::vector<op::fp16_t> keyHostData(keyShapeSize, 1);
    std::vector<op::fp16_t> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> blockTableOptionalHostData(blockTableOptionalShapeSize, 0);
    std::vector<op::fp16_t> outputHostData(outputShapeSize, 1);

    std::vector<int32_t> topkIndicesHostData;
    for (int b = 0; b < batch; ++b) {
       for (int s = 0; s < qSeqlen; ++s) {
        for (int h = 0; h < numKeyValueHeads; ++h) {
            for (int k = 0; k < selectBlockCount; ++k) {
                if (k == 0) {
                    topkIndicesHostData.push_back(k);
                } else {
                    topkIndicesHostData.push_back(-1);
                }
            }
        }
       }
    }
    // attr
    double scaleValue = 1.0;
    int64_t sparseMode = 0;
    std::string sLayerOut = "BSND";
    char layOut[sLayerOut.length()+1];
    std::strcpy(layOut, sLayerOut.c_str());

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

    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    if (argv == nullptr || argv[0] == nullptr) {
        LOG_PRINT("Environment error, Argv=%p, Argv[0]=%p", argv, argv == nullptr ? nullptr : argv[0]);
        return 0;
    }
    // Create query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create blockTableOptional aclTensor.
    ret = CreateAclTensor(blockTableOptionalHostData, blockTableOptionalShape, &blockTableOptionalDeviceAddr, aclDataType::ACL_INT32, &blockTableOptionalTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create output aclTensor.
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create topkIndices aclTensor.
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndicesTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);

    // 3. Call the CANN operator library API. Replace Api with the actual API name.
    aclOpExecutor *executor;

    // Call the aclnnNsaSelectedAttention first-phase API.
    ret = aclnnNsaSelectedAttentionInferGetWorkspaceSize(queryTensor, keyTensor, valueTensor, topkIndicesTensor, nullptr,
                blockTableOptionalTensor, actualCmpQSeqLen, actualCmpKvSeqLen, layOut,
                numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount, blockSize,
                scaleValue, sparseMode, outputTensor,
                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInferGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the aclnnNsaSelectedAttention second-phase API.
    ret = aclnnNsaSelectedAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnn execute success : %d\n", ret);

    // 5. Obtain the output values and copy the results from device memory to host memory. Modify them based on the specific API definition.
    auto size = GetShapeSize(outputShape);
    std::vector<op::fp16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [attn] result from device to host failed. ERROR: %d\n", ret); return ret);
    uint64_t printNum = 10;
    for (int64_t i = 0; i < printNum; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }

    // 6. Release aclTensor and aclScalar, modify them based on the specific API definition; release device resources.
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(outputTensor);
    aclDestroyTensor(topkIndicesTensor);
    aclDestroyTensor(blockTableOptionalTensor);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(topkIndicesDeviceAddr);
    aclrtFree(blockTableOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
