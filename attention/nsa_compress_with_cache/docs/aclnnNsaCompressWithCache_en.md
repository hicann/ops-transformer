# aclnnNsaCompressWithCache

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

- Description: Performs KV compression in the Native Sparse Attention (NSA) inference phase. A new token is generated for each batch during each inference. When the number of tokens in a batch reaches the size of a compression block, the operator compresses the last compression-block-sized tokens in the batch into a compressed token.
- Formulas:

$$
compressIdx=(s-compressBlockSize)/stride\\ 
outputCacheRef[slotMapping[i]] = input[compressIdx*stride : compressIdx*stride+compressBlockSize]*weight[:]
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnNsaCompressWithCacheGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnNsaCompressWithCache` is called to perform computation.

```c++
aclnnStatus aclnnNsaCompressWithCacheGetWorkspaceSize(
   const aclTensor   *input,
   const aclTensor   *weight,
   const aclTensor   *slotMapping,
   const aclIntArray *actSeqLenOptional,
   const aclTensor   *blockTableOptional,
   char              *layoutOptional,
   int64_t            compressBlockSize,
   int64_t            compressStride,
   int64_t            actSeqLenType,
   int64_t            pageBlockSize,
   aclTensor         *outputCache,
   uint64_t          *workspaceSize,
   aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaCompressWithCache(
   void          *workspace,
   uint64_t       workspaceSize,
   aclOpExecutor *executor,
   aclrtStream    stream)
```

## aclnnNsaCompressWithCacheGetWorkspaceSize

- **Parameters**
  
  <table style="undefined; table-layout: fixed; width: 1567px">
    <colgroup>
      <col style="width: 170px"> <!-- Name -->
      <col style="width: 120px"> <!-- Input/Output -->
      <col style="width: 300px"> <!-- Description -->
      <col style="width: 330px">  <!-- Usage Notes -->
      <col style="width: 212px"> <!-- Data Type -->
      <col style="width: 100px"> <!-- Data Format -->
      <col style="width: 190px"> <!-- Dimension (Shape) -->
      <col style="width: 145px"> <!-- Non-contiguous Tensor -->
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
        <td>input</td>
        <td>Input</td>
        <td>Tensor to be compressed.</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>Empty tensors are not supported.</li>
            <li><code>input</code> and <code>weight</code> must meet the <code>broadcast</code> operation relationship. The third-dimension size of <code>input</code> is equal to the second-dimension size of <code>weight</code>.</li>
            <li><code>headDim</code> must be less than or equal to 256 and is an integer multiple of 16.</li>
            <li><code>headNum</code> must be less than or equal to 64. If it is greater than 50, <code>headNum%2=0</code>.</li>
            <li><code>N</code> (<code>Head-Num</code>) indicates the number of heads, and <code>D</code> (<code>Head-Dim</code>) indicates the minimum unit size of the hidden layer.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[blockNum, pageBlockSize, N, D], [TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Compression weight.</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>Empty tensors are not supported.</li>
            <li>The data type must be the same as that of <code>input</code>.</li>
            <li><code>N</code> (<code>Head-Num</code>) indicates the number of heads.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[compressBlockSize, N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>slotMapping</td>
        <td>Input</td>
        <td>Index of the location where the compressed data at the end of each batch is stored.</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>Empty tensors are not supported.</li>
            <li>The values of <code>slotMapping</code> must be unique. Otherwise, the computation result will be unstable.</li>
          </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>[B]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>actSeqLenOptional</td>
        <td>Optional input</td>
        <td><code>S</code> corresponding to each batch.</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>This parameter is required for <code>TND</code> layout. In other scenarios, input <code>nullptr</code>.</li>
            <li><code>S</code> (<code>Seq-Length</code>) indicates the sequence length of input samples.</li>
            <li>The value of <code>actSeqLenOptional</code> should not exceed the maximum sequence length.</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>[B]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blockTableOptional</td>
        <td>Optional input</td>
        <td>Block mapping table used for KV storage in page attention.</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>If this parameter is not used, pass <code>nullptr</code>.</li>
            <li>The value of <code>blockTableOptional</code> cannot exceed the value of <code>blockNum</code>. Otherwise, an out-of-bounds error occurs.</li>
          </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>[batch, blockNumPerBatch]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>layoutOptional</td>
        <td>Optional input</td>
        <td>Layout of <code>input</code>.</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>Currently, only <code>TND</code> is supported. This parameter is invalid when <code>blockTableOptional</code> is passed. Otherwise, this parameter is mandatory.</li>
            <li><code>T</code> indicates the total length of all input sample sequences (<code>actSeqLen</code> of all batches), <code>B</code> (<code>Batch</code>) indicates the size of an input sample batch, <code>S</code> (<code>Seq-Length</code>) indicates the length of the input sample sequence, <code>N</code> (<code>Head-Num</code>) indicates the number of heads, and <code>D</code> (<code>Head-Dim</code>) indicates the minimum unit size of the hidden layer.</li>
          </ul>
        </td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressBlockSize</td>
        <td>Input</td>
        <td>Size of the compression sliding window.</td>
        <td>The value must be an integer multiple of 16 and must meet the following condition: <code>compressStride</code> ≤ <code>compressBlockSize</code> ≤ 64.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>Input</td>
        <td>Sliding window interval between two compressions.</td>
        <td>The value of compressStride can only be 16, 32, 48, or 64.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actSeqLenType</td>
        <td>Input</td>
        <td>Different expressions of <code>actSeqLenOptional</code></td>
        <td>This parameter takes effect when <code>actSeqLenOptional</code> is input. The value can be <code>0</code> or <code>1</code>. <code>0</code> indicates that the value in  <code>actSeqLenOptional</code> is the cumulative sum of the sequence sizes of the previous batches. <code>1</code> indicates that the value in <code>actSeqLenOptional</code> is the sequence size of each batch. Currently, only <code>1</code> is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pageBlockSize</td>
        <td>Input</td>
        <td>Block size of the page in the page attention scenario.</td>
        <td>The value can only be <code>64</code> or <code>128</code>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outputCache</td>
        <td>Output</td>
        <td>Compressed cache</td>
        <td>The data type must be the same as that of <code>input</code>.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[result_len, N, D]</td>
        <td>x</td>
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
  
  <table style="undefined;table-layout: fixed; width: 1153px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 732px">
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
      <td>The input and required output for computation are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types and formats of the computation input and output are not supported.</td>
    </tr>
    <tr>
      <td><code>input</code>, <code>weight</code>, and <code>outputCache</code> are empty tensors.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="6">561002</td>
      <td><code>input</code> and <code>weight</code> do not meet the <code>broadcast</code> operation relationship. That is, the third-dimension size of <code>input</code> is different from the second-dimension size of <code>weight</code>.</td>
    </tr>
    <tr>
      <td>The values of <code>activeNum</code>, <code>expertNum</code>, and <code>expertCapacity</code> are less than 0.</td>
    </tr>
    <tr>
      <td>compress_block_size and compress_stride are not integer multiples of 16.</td>
    </tr>
    <tr>
      <td><code>actSeqLenType</code> is not 1 or <code>layoutOptional</code> is not <code>BSH</code>, <code>SBH</code>, <code>BSND</code>, <code>BNSD</code>, or <code>TND</code>.</td>
    </tr>
    <tr>
      <td><code>pageBlockSize</code> is not <code>64</code> or <code>128</code>.</td>
    </tr>
    <tr>
      <td><code>headDim</code> is not 16-aligned.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaCompressWithCache

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnNsaCompressWithCacheGetWorkspaceSize</code>.</td>
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
  - `aclnnNsaCompressWithCache` defaults to a deterministic implementation.
- The N and D of outputCache are the same as those of input, and must meet the result_len>(blockNum*pageBlockSize-compressBlockSize)/compressStride condition.
- In the page attention scenario, the shape of `input` supports [blockNum, pageBlockSize, N, D]. In other scenarios, the shape of `input` supports [T, N, D].

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_compress_with_cache.h"
#include <iostream>
#include <vector>
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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // Set parameters related to the input shapes.
    constexpr int64_t compress_block_size = 32;
    constexpr int64_t compress_stride = 16;
    constexpr int64_t heads_num = 24;
    constexpr int64_t heads_dim = 192;
    constexpr int64_t batch_size = 4;
    constexpr int64_t page_block_size = 128;
    constexpr int64_t max_seq_len = 512;
    constexpr int64_t result_len = 512;
    constexpr int64_t block_num_per_batch = max_seq_len / page_block_size;
    constexpr int64_t blocks_num = block_num_per_batch * batch_size;
    // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Customize the returned error information as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> inputShape = {blocks_num, page_block_size, heads_num, heads_dim};
    std::vector<int64_t> weightShape = {compress_block_size, heads_num};
    std::vector<int64_t> slotMappingShape = {batch_size};
    std::vector<int64_t> outputCacheRefShape = {result_len, heads_num, heads_dim};
    std::vector<int64_t> actSeqLenShape = {batch_size};
    std::vector<int64_t> blockTableShape = {batch_size, block_num_per_batch};

    void *inputDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *slotMappingDeviceAddr = nullptr;
    void *outputCacheRefDeviceAddr = nullptr;
    void *actSeqLenDeviceAddr = nullptr;
    void *blockTableDeviceAddr = nullptr;

    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *slotMapping = nullptr;
    aclTensor *outputCacheRef = nullptr;
    aclIntArray *actSeqLen = nullptr;
    aclTensor *blockTable = nullptr;

    std::vector<aclFloat16> inputHostData(inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3],
                                          aclFloatToFloat16(1.0));
    std::vector<aclFloat16> weightHostData(weightShape[0] * weightShape[1], aclFloatToFloat16(1.0));
    std::vector<int32_t> slotMappingHostData(slotMappingShape[0], 0);
    std::vector<aclFloat16> outputCacheRefHostData(outputCacheRefShape[0] * outputCacheRefShape[1] *
                                                   outputCacheRefShape[2], aclFloatToFloat16(1.0));
    std::vector<int64_t> actSeqLenHostData(actSeqLenShape[0], 0);
    std::vector<int32_t> blockTableHostData(blockTableShape[0] * blockTableShape[1]);
    actSeqLenHostData[0]=32;
    // Create a self aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(slotMappingHostData, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32,
                          &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputCacheRefHostData, outputCacheRefShape, &outputCacheRefDeviceAddr,
                          aclDataType::ACL_FLOAT16, &outputCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    actSeqLen = aclCreateIntArray(actSeqLenHostData.data(), actSeqLenHostData.size());
    ret = CreateAclTensor(blockTableHostData, blockTableShape, &blockTableDeviceAddr, aclDataType::ACL_INT32,
                          &blockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    char layout[4] = "TND";
    int64_t actSeqLenType = 1;
    // 3. Call the CANN operator library API. Change the API name to the actual one.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnNsaCompressWithCache.
    ret = aclnnNsaCompressWithCacheGetWorkspaceSize(input, weight, slotMapping, actSeqLen, blockTable, layout,
                                                    compress_block_size, compress_stride, actSeqLenType,
                                                    page_block_size, outputCacheRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressWithCacheGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnNsaCompressWithCache.
    ret = aclnnNsaCompressWithCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressWithCache failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outputCacheRefShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(aclFloat16), outputCacheRefDeviceAddr,
                      size * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = heads_dim * heads_num - 16; i < heads_dim * heads_num + 16; i++) {
        printf("outputCache[%ld]:%f\n", i, aclFloat16ToFloat(resultData[i]));
    }
    // 6. Destroy aclTensor. Modify the code based on the API definition.
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(outputCacheRef);
    aclDestroyIntArray(actSeqLen);
    aclDestroyTensor(blockTable);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(outputCacheRefDeviceAddr);
    aclrtFree(blockTableDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
