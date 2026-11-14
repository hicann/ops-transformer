# aclnnNsaCompressWithCache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:04:53.440Z -->

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

- Description: Used for KV compression in the inference phase of Native-Sparse-Attention. Each batch generates a new token during each inference. When the number of tokens in a batch fills up one compress_block, this operator compresses the last compress_block tokens of that batch into one compress_token.
- Formula:

$$
compressIdx=(s-compressBlockSize)/stride\\
outputCacheRef[slotMapping[i]] = input[compressIdx*stride : compressIdx*stride+compressBlockSize]*weight[:]
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnNsaCompressWithCacheGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call the "aclnnNsaCompressWithCache" API to execute the computation.

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
      <col style="width: 170px"> <!-- Parameter -->
      <col style="width: 120px"> <!-- Input/Output -->
      <col style="width: 300px"> <!-- Description -->
      <col style="width: 330px"> <!-- Instruction -->
      <col style="width: 212px"> <!-- Data Type -->
      <col style="width: 100px"> <!-- Data Format -->
      <col style="width: 190px"> <!-- Dimensions (shape) -->
      <col style="width: 145px"> <!-- Non-contiguous Tensor -->
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Shape</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Input</td>
        <td>Input</td>
        <td>Tensor to be compressed.</td>
        <td>          <ul style="list-style-type: circle;">
            <li>Does not support null Tensor.</li>
            <li>input and weight satisfy the broadcast relationship; the third dimension of input equals the second dimension of weight.</li>
            <li>headDim is an integer multiple of 16, and headDim<=256.</li>
            <li>headNum<=64, and headNum%2=0 when headNum>50.</li>
            <li>N (Head-Num) indicates multi-head number, and D (Head-Dim) indicates the minimum unit size of the hidden layer.</li>
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
        <td>Weight for compression.</td>
        <td>          <ul style="list-style-type: circle;">
            <li>Does not support null Tensor.</li>
            <li>The data type is the same as input.</li>
            <li>N (Head-Num) indicates the multi-head number.</li>
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
        <td>Index of the storage location for compressed data at the end of each batch.</td>
        <td>          <ul style="list-style-type: circle;">
            <li>Does not support null Tensor.</li>
            <li>The values of slotMapping must be unique. Otherwise, the computation result becomes unstable.</li>
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
        <td>Size of S corresponding to each batch.</td>
        <td>          <ul style="list-style-type: circle;">
            <li>This input is required in the TND layout use case. Input nullptr in other use cases.</li>
            <li>S (Seq-Length) indicates input sample sequence length.</li>
            <li>The value of actSeqLenOptional should not exceed the maximum sequence length.</li>
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
        <td>Block mapping table used for KV storage in PageAttention.</td>
        <td>          <ul style="list-style-type: circle;">
            <li>Pass nullptr if this feature is not used.</li>
            <li>The value of blockTableOptional must not exceed blockNum. Otherwise, out-of-bounds access occurs.</li>
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
        <td>Data layout format of input.</td>
        <td>          <ul style="list-style-type: circle;">
            <li>Currently, only "TND" is supported. This parameter does not take effect when blockTableOptional is passed in; otherwise, it is mandatory.</li>
            <li>Specifically, T indicates the data of B and S tightly packed on the merged axis (actSeqLen of each batch), B (Batch) indicates input sample batch size, S (Seq-Length) indicates input sample sequence length, N (Head-Num) indicates multi-head, and D (Head-Dim) indicates the smallest unit size of the hidden layer.</li>
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
        <td>Compression sliding window size.</td>
        <td>Must be an integer multiple of 16, and compressBlockSize>=compressStride, compressBlockSize<=64.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>Input</td>
        <td>Sliding window interval between two compressions.</td>
        <td>Only compressStride values 16, 32, 48, and 64 are supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actSeqLenType</td>
        <td>Input</td>
        <td>A different representation of actSeqLenOptional.</td>
        <td>Takes effect when actSeqLenOptional is provided. Value range: 0 or 1. 0 indicates that values in actSeqLenOptional are cumsum results (cumulative sums) of the sequence sizes of preceding batches. 1 indicates that values in actSeqLenOptional are the sequence sizes of each batch. Currently, only 1 is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pageBlockSize</td>
        <td>Input</td>
        <td>Page block size in the page attention use case.</td>
        <td>Must be 64 or 128.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outputCache</td>
        <td>Output</td>
        <td>Compressed cache</td>
        <td>Data type is the same as that of input.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[result_len, N, D]</td>
        <td>x</td>
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

  <table style="undefined;table-layout: fixed; width: 1153px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 732px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The compute input or required compute output is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types and formats of the input and output are not in the supported range.</td>
    </tr>
    <tr>
      <td>input, weight, or outputCache is an empty tensor.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="7">561002</td>
      <td>input and weight do not satisfy the broadcast relationship. That is, the third dimension size of input is not equal to the second dimension size of weight.</td>
    </tr>
    <tr>
      <td>The value of activeNum, expertNum, or expertCapacity is less than 0.</td>
    </tr>
    <tr>
      <td>compress_block_size and compress_stride are not an integer multiple of 16.</td>
    </tr>
    <tr>
      <td>compress_stride is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>seq_lens_type!=1 or the value of layout is not one of BSH, SBH, BSND, BNSD, and TND.</td>
    </tr>
    <tr>
      <td>The value of page_block_size is not 64 or 128.</td>
    </tr>
    <tr>
      <td>headDim is not aligned to 16.</td>
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnNsaCompressWithCacheGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNsaCompressWithCache defaults to a deterministic implementation.
- The N and D of outputCache must be consistent with those of input, and result_len must satisfy result_len > (blockNum * pageBlockSize - compressBlockSize) / compressStride.
- In the page attention scenario, the shape of input supports [blockNum, pageBlockSize, N, D]; in other scenarios, the shape of input supports [T, N, D].

## Example

The following sample code is for reference only. For the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
    // Call aclrtMalloc to allocate memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Calculate the strides of the contiguous tensor.
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
    // Set parameters related to the input shape.
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
    // 1. Boilerplate: device/stream initialization. Refer to the ACL API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs according to the API definition.
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
    // Create the self aclTensor.
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
    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnNsaCompressWithCache.
    ret = aclnnNsaCompressWithCacheGetWorkspaceSize(input, weight, slotMapping, actSeqLen, blockTable, layout,
                                                    compress_block_size, compress_stride, actSeqLenType,
                                                    page_block_size, outputCacheRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressWithCacheGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnNsaCompressWithCache.
    ret = aclnnNsaCompressWithCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressWithCache failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate. Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value. Copy the result from the device memory to the host side. Modify it according to the definition of the specific API.
    auto size = GetShapeSize(outputCacheRefShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(aclFloat16), outputCacheRefDeviceAddr,
                      size * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = heads_dim * heads_num - 16; i < heads_dim * heads_num + 16; i++) {
        printf("outputCache[%ld]:%f\n", i, aclFloat16ToFloat(resultData[i]));
    }
    // 6. Release the aclTensor. Modify it according to the definition of the specific API.
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(outputCacheRef);
    aclDestroyIntArray(actSeqLen);
    aclDestroyTensor(blockTable);

    // 7. Release the device resources. Modify it according to the definition of the specific API.
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
