# aclnnNsaSelectedAttentionInfer

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Performs selected attention computation during the Native Sparse Attention (NSA) inference.
- Formula:
  
  Self-attention constructs an attention model by leveraging the relationships within the input samples. The principle assumes there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. Such a sequence is transformed by three weight matrices to obtain three $n × d$ matrices.
  
  The calculation of selected attention is formed by combining the topk index data obtaining and attention calculation, and the paged attention obtains the kvCache. First, $key_{topk}$ is obtained from $key$ and $value_{topk}$ is obtained from $value$ by using $topkIndices$. The self-attention computation formula is as follows:
  
  $$
  Attention(query,key,value)=Softmax(\frac{query · key_{topk}^T}{\sqrt{d}})value_{topk}
  $$
  
  The product of $query$ and $key_{topk}^T$ represents the attention to the input $x$. To prevent this value from becoming excessively large, it is typically scaled by dividing by the square root of $d$, followed by row-wise softmax normalization. The result is then multiplied by $value_{topk}$ to produce an $n × d$ matrix.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNsaSelectedAttentionInferGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnNsaSelectedAttentionInfer` is called to perform computation.

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
          <th style="font-weight: bold;">Name</th>
          <th style="font-weight: bold;">Input/Output</th>
          <th style="font-weight: bold;">Description</th>
          <th style="font-weight: bold;">Usage Notes</th>
          <th style="font-weight: bold;">Data Type</th>
          <th style="font-weight: bold;">Data Format</th>
          <th style="font-weight: bold;">Dimension (Shape)</th>
          <th style="font-weight: bold;">Non-contiguous Tensor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="white-space: nowrap;">query</td>
          <td>Input</td>
          <td>Input <code>query</code> in the formula.</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>The data type must be the same as that of <code>key</code> and <code>value</code>.</li>
              <li>The ratio of the N-axis value of <code>query</code> to the N-axis value (<code>H</code>/<code>D</code>) of <code>key</code>/<code>value</code> (that is, the group size in GQA) can be less than or equal to 16.</li>
              <li>The D-axis value of <code>query</code> can be 192.</li>
              <li>In common scenarios, the S-axis value of <code>query</code> can only be 1.</li>
               <li>The <code>N</code> value of <code>query</code> is equal to the value of <code>numHeads</code>, and <code>numHeads</code> is a multiple of <code>numKeyValueHeads</code>.</li>
               <li>The <code>D</code> value of <code>query</code> is equal to the <code>D</code> value (<code>H</code>/<code>numKeyValueHeads</code>) of <code>key</code>.</li>
            </ul>
          </td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3/4</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">key</td>
          <td>Input</td>
          <td>Input <code>key</code> in the formula.</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>The data type must be the same as that of <code>query</code> and <code>value</code>.</li>
              <li>The N-axis value of <code>key</code> can be less than or equal to 256.</li>
               <li>The D-axis value of <code>key</code> can be 192.</li>
               <li><code>blockSize</code> of <code>key</code> can be 64 or 128.</li>
                <li>The <code>N</code> value of <code>key</code> is equal to the value of <code>numHeads</code>, and <code>numHeads</code> is a multiple of <code>numKeyValueHeads</code>.</li>
            </ul>
          </td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3/4</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">value</td>
          <td>Input</td>
          <td>Input <code>value</code> in the formula.</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>The data type must be the same as that of <code>query</code> and <code>key</code>.</li>
              <li>The N-axis value of <code>value</code> can be less than or equal to 256.</li>
               <li>The D-axis value of <code>value</code> can be 128.</li>
               <li><code>blockSize</code> of <code>value</code> can be 64 or 128.</li>
               <li>The <code>N</code> value of <code>value</code> is equal to the value of <code>numHeads</code>, and <code>numHeads</code> is a multiple of <code>numKeyValueHeads</code>.</li>
               <li>The <code>D</code> value (<code>H/numKeyValueHeads</code>) of <code>value</code> is equal to the <code>D</code> value of <code>output</code>.</li>
            </ul>
          </td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3/4</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">topkIndices</td>
          <td>Input</td>
          <td><code>topk</code> index in the formula.</td>
          <td>-
          </td>
          <td>INT32</td>
          <td>ND</td>
          <td>3/4</td>
          <td>×</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">attenMask</td>
          <td>Input</td>
          <td>Attention mask matrix.</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>This parameter is optional.</li>
              <li>If this parameter is not used, pass <code>nullptr</code>.</li>
              <li>This parameter is reserved and not used currently.</li>
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
          <td>Block mapping table used for KV storage in paged attention.</td>
          <td>
            - 
          </td>
          <td>INT32</td>
          <td>ND</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>actualQSeqLenOptional</td>
          <td>Input</td>
          <td>Actual <code>S</code> value of <code>query</code>.</td>
          <td>If this function is not used, nullptr can be passed.</td>
          <td>INT64</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>actualKvSeqLenOptional</td>
          <td>Input</td>
          <td>Actual <code>S</code> value of <code>key</code> and <code>value</code> processed by the operator.</td>
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
          <td>Layout of the input <code>query</code>, <code>key</code>, and <code>value</code>.</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>Currently, <code>BSH</code>, <code>BSND</code>, and <code>TND</code> are supported.</li>
              <li>If this parameter is not passed, the default value is <code>BSND</code>. The values of this parameter correspond to the 3D/4D format of <code>query</code>, <code>key</code>, and <code>value</code>.</li>
              <li>In the data layout of <code>query</code>, <code>B</code> (Batch) indicates the size of an input sample batch, <code>S</code> (Seq-Length) indicates the length of the input sample sequence, <code>N</code> (Head-Num) indicates the number of heads, and <code>D</code> (Head-Dim) indicates the minimum unit size of the hidden layer (<code>D</code> = <code>H</code>/<code>N</code>). The data layout of <code>key</code> and <code>value</code> support <code>(blocknum, blocksize, H)</code> and <code>(blocknum, blocksize, N, D)</code>. <code>H</code> (Head-Size) indicates the size of the hidden layer (<code>H</code> = <code>N</code> × <code>D</code>).</li>
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
          <td>Number of heads.</td>
          <td>
            - 
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">numKeyValueHeads</td>
          <td>Input</td>
          <td>Number of <code>key</code>/<code>value</code> heads.</td>
          <td>
            - 
          </td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">selectBlockSize</td>
          <td>Input</td>
          <td>Size of the selected block.</td>
          <td>
            <ul style="list-style-type: circle; margin: 0; padding-left: 20px;">
              <li>Used for computing the importance score.</li>
              <li>The value of <code>selectBlockSize</code> must be an integer multiple of 16, and the maximum value is 128.</li>  
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
          <td>Number of blocks to be retained in the topK phase.</td>
          <td>The upper limit of selectBlockCount meets the requirements of selectBlockCount * selectBlockSize < = MaxKvSeqlen and MaxKvSeqlen = Max(actualKvSeqLenOptional).</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">pageBlockSize</td>
          <td>Input</td>
          <td>Block size of paged attention.</td>
          <td>Used when data is obtained from the KV cache.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">scaleValue</td>
          <td>Input</td>
          <td>Reciprocal of the square root of <code>d</code> in the formula, which represents the scaling coefficient.</td>
          <td>Scalar value of Muls in the computation flow.</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">sparseMode</td>
          <td>Input</td>
          <td>Sparse mode, which controls sparse computation when <code>attenMask</code> is input.</td>
          <td>Reserved parameter. Not used currently.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">output</td>
          <td>Output</td>
          <td>Attention output in the formula.</td>
          <td>Reserved parameter. Not used currently.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">workspaceSize</td>
          <td>Output</td>
          <td>Returns the size of the workspace that needs to be applied for by the user in DevicenumHeads.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td style="white-space: nowrap;">executor</td>
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
  </div>
  
- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.
  
  <div style="overflow-x: auto;">
  <table style="table-layout: fixed; width: 1030px">  <colgroup>     
      <col style="width: 250px">     
      <col style="width: 130px">     
      <col style="width: 650px">   
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
        <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
      </tr>
      <tr>
        <!-- Merge cells and add the merged-cell class to center the cells. -->
        <td class="merged-cell" rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td class="merged-cell" rowspan="2">161002</td>
        <td>The data type of <code>query</code>, <code>key</code>, <code>value</code>, <code>topkIndices</code>, <code>attenMask</code>, <code>blockTableOptional</code>, <code>actualQSeqLenOptional</code>, <code>actualKvSeqLenOptional</code>, or <code>output</code> is not supported.</td>
      </tr>
      <tr>
        <td>The data format of <code>query</code>, <code>key</code>, <code>value</code>, <code>topkIndices</code>, <code>attenMask</code>, <code>blockTableOptional</code>, <code>actualQSeqLenOptional</code>, <code>actualKvSeqLenOptional</code>, or <code>output</code> is not supported.</td>
      </tr>
    </tbody>
  </table>
  </div>

## aclnnNsaSelectedAttentionInfer

- **Parameters**
  
  <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
          <col style="width: 250px">
          <col style="width: 130px">
          <col style="width: 650px">
      </colgroup><thead>
        <tr>
          <th>Name</th>
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
          <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnNsaSelectedAttentionInferGetWorkspaceSize</code>.</td>
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
  </div>

- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnNsaSelectedAttentionInfer` defaults to a deterministic implementation.
- The B axis must be less than or equal to 3072.
- Only paged attention is supported.
- In multi-token inference scenarios, the S axis of `query` can be at most 4 and for each batch, `actualQSeqLen` must be less than or equal to `actualSelKvSeqLen`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main(int argc, char **argv)
{
    // 1. (Fixed writing) Initialize the device, context, and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    // If you need to modify shape values, modify the shape values corresponding to query, key, and value generated in the test_nsa_selected_attention_infer branch
    // in ../scripts/fa_generate_data.py, regenerate the data, and then execute the test script.

    int64_t batch = 1;
    int sequenceLengthK = 48;
    aclIntArray * actualCmpKvSeqLen = nullptr;
    aclIntArray * actualCmpQSeqLen = nullptr;
    // Create an actualCmpKvSeqLen aclIntArray.
    std::vector<int64_t> actualCmpKvSeqLenVector(batch, sequenceLengthK);
    actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqLenVector.data(), actualCmpKvSeqLenVector.size());
    // Create an actualCmpQSeqLen aclIntArray.
    int64_t s1 = 1;
    std::vector<int64_t> actualCmpQSeqLenVector(batch, s1);
    actualCmpQSeqLen = aclCreateIntArray(actualCmpQSeqLenVector.data(), actualCmpQSeqLenVector.size());
    int64_t d1 = 192;
    int64_t d2 = 128;
    int64_t g = 1;
    
    int64_t n2 = 1;
    int64_t blockSize = 64;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 1;
    int64_t blockTableLength = 1;
    int64_t numBlocks = batch * blockTableLength;
    std::vector<int64_t> queryShape = {batch, s1, n2 * g, d1};
    std::vector<int64_t> keyShape = {numBlocks, blockSize, n2,d1};
    std::vector<int64_t> valueShape = {numBlocks, blockSize, n2,d2};
    std::vector<int64_t> topkIndicesShape = {batch, s1, n2, selectBlockCount};
    std::vector<int64_t> blockTableOptionalShape = {batch, blockTableLength};
    std::vector<int64_t> outputShape = {batch, s1, n2 * g, d2};

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
       for (int s = 0; s < s1; ++s) {
        for (int h = 0; h < n2; ++h) {
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
    int64_t sparseMod = 0;
    int64_t numHeads= static_cast<int64_t>(n2 * g);
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
    // Create a query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create a key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create a value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create a blockTableOptional aclTensor.
    ret = CreateAclTensor(blockTableOptionalHostData, blockTableOptionalShape, &blockTableOptionalDeviceAddr, aclDataType::ACL_INT32, &blockTableOptionalTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create an output aclTensor.
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // Create a topkIndices aclTensor.
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndicesTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);

    // 3. Call the CANN operator library API. Change the API name to the actual one.
    aclOpExecutor *executor;

    // Call the first-phase API of aclnnNsaSelectedAttention.
    ret = aclnnNsaSelectedAttentionInferGetWorkspaceSize(queryTensor, keyTensor, valueTensor, topkIndicesTensor, nullptr,
                blockTableOptionalTensor, actualCmpQSeqLen, actualCmpKvSeqLen, layOut,
                numHeads, n2, selectBlockSize, selectBlockCount, blockSize,
                scaleValue, sparseMod, outputTensor,
                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnNsaSelectedAttention.
    ret = aclnnNsaSelectedAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnn execute success : %d\n", ret);
    
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

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition. Release device resources.
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
