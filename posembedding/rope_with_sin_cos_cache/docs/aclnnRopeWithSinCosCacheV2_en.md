# aclnnRopeWithSinCosCacheV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:00:58.507Z -->

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

- Description: To improve performance, inference networks pass the sin and cos inputs through the cache and perform rotary position encoding computation. **Compared with [aclnnRopeWithSinCosCache](./aclnnRopeWithSinCosCache_en.md), this API adds the cacheMode parameter, which specifies how to concatenate cos and sin**:
  - If cacheMode=0, the implementation is the same as that of [aclnnRopeWithSinCosCache](./aclnnRopeWithSinCosCache_en.md), and cos and sin are concatenated in a segmented manner.
  - When cacheMode=1, cos and sin are concatenated in interleaved mode.
- Formula:

    1, **mrope mode**: The input shape of positions is [m, numTokens], where m is the number of elements in mropeSection, and 3 or 4 are supported:

    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    (1) When cacheMode is 0:
    - The number of elements in mropeSection is 3:

      $$
      cos0 = cos[0, :, :mropeSection[0]]
      $$

      $$
      cos1 = cos[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
      $$

      $$
      cos2 = cos[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
      $$

      $$
      cos = torch.cat((cos0, cos1, cos2), dim=-1)
      $$

      $$
      sin0 = sin[0, :, :mropeSection[0]]
      $$

      $$
      sin1 = sin[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
      $$

      $$
      sin2 = sin[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
      $$

      $$
      sin= torch.cat((sin0, sin1, sin2), dim=-1)
      $$

      $$
      queryRot = query[..., :rotaryDim]
      $$

      $$
      queryPass = query[..., rotaryDim:]
      $$

    - The number of elements in mropeSection is 4:

      $$
      cos = torch.cat([m[i]\ for\ i, m\ in\ enumerate(cos.split(mropeSection, dim=-1))], dim=-1)
      $$

      $$
      sin = torch.cat([m[i]\ for\ i, m\ in\ enumerate(sin.split(mropeSection, dim=-1))], dim=-1)
      $$

      $$
      queryRot = query[..., :rotaryDim]
      $$

      $$
      queryPass = query[..., rotaryDim:]
      $$

    (2) cacheMode is 1:

    $$
    cosTmp = cos
    $$

    $$
    cos [..., 1:mropeSection[1] * 3:3] = cosTmp[1, ..., 1:mropeSection[1] * 3:3]
    $$

    $$
    cos[..., 2:mropeSection[1] * 3:3] = cosTmp[2, ..., 2:mropeSection[1] * 3:3]
    $$

    $$
    sinTmp = sin
    $$

    $$
    sin[..., 1:mropeSection[1] * 3:3] = sinTmp [1, ..., 1:mropeSection[1] * 3:3]
    $$

    $$
    sin[..., 2:mropeSection[1] * 3:3] = sinTmp [2, ..., 2:mropeSection[1] * 3:3]
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    (1)rotate\_half(GPT-NeoX style) computation mode:

    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    (2)rotate\_interleaved(GPT-J style) computation mode:

    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    2.**rope mode**: The input shape of positions is [numTokens]:

    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    (1)rotate\_half(GPT-NeoX style) computation mode:

    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    (2)rotate\_interleaved(GPT-J style) computation mode:

    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnRopeWithSinCosCacheV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnRopeWithSinCosCacheV2" API to perform the computation.

```c++
aclnnStatus aclnnRopeWithSinCosCacheV2GetWorkspaceSize(
    const aclTensor   *positions,
    const aclTensor   *queryIn,
    const aclTensor   *keyIn,
    const aclTensor   *cosSinCache,
    const aclIntArray *mropeSection,
    int64_t            headSize,
    bool               isNeoxStyle,
    int64_t            cacheMode,
    aclTensor         *queryOut,
    aclTensor         *keyOut,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnRopeWithSinCosCacheV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnRopeWithSinCosCacheV2GetWorkspaceSize

- **Parameters**:

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 550px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
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
      </tr></thead>
      <tbody>
      <tr>
        <td>positions</td>
        <td>Input</td>
        <td>positions in the formula, used to select the position encoding tensor.</td>
        <td><ul><li>Does not support empty tensors.</li><li>In rope mode, the shape is (numTokens).</li><li>In mrope mode, the shape is (3, numTokens) or (4, numTokens).</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryIn</td>
        <td>Input</td>
        <td>query in the formula, the first tensor on which rotary position encoding is to be performed.</td>
        <td><ul><li>Does not support empty tensors.</li><li>Must be a 2D tensor with shape (numTokens,  numQHeads*headSize).</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyIn</td>
        <td>Input</td>
        <td>The second tensor on which to perform rotary position encoding.</td>
        <td><ul><li>Does not support empty tensors.</li><li>Must be a 2D tensor with shape (numTokens,  numKHeads*headSize).</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cosSinCache</td>
        <td>Input</td>
        <td>Represents the position encoding tensor used in the computation.</td>
        <td><ul><li>Does not support empty tensors.</li><li>Must be a 2D tensor with shape (maxSeqLen, rotaryDim). maxSeqLen represents the maximum length of the sequence processed by the model, and rotaryDim represents the dimension size of the rotary position embedding.</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mropeSection</td>
        <td>Input</td>
        <td>mropeSection in the formula. Used in mrope mode to integrate the position encoding tensor information of the input.</td>
        <td>The input mropeSection attribute indicates that mrope mode is enabled. If mrope mode is not enabled (rope mode), the input is nullptr.</td>
        <td>aclIntArray</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headSize</td>
        <td>Input</td>
        <td>Size of the dimension of each attention head.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeoxStyle</td>
        <td>Input</td>
        <td>Indicates whether to use the GPT-NeoX computation mode.</td>
        <td><ul><li>true indicates the GPT-NeoX style computation mode.</li><li>false indicates the GPT-J style computation mode.</li></ul></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cacheMode</td>
        <td>Input</td>
        <td>Indicates that cos and sin are concatenated in segmented or interleaved mode.</td>
        <td><ul><li>0 indicates segmented mode.</li><li>1 indicates interleaved mode.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryOut</td>
        <td>Output</td>
        <td>Result of query after performing rotary position encoding.</td>
        <td><ul><li>Same data type as query.</li><li>A 2D tensor with shape (numTokens,  numQHeads*headSize) is required.</li></ul></td>
        <td>FLOAT32, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>keyOut</td>
        <td>Output</td>
        <td>Result of key after performing rotary position encoding.</td>
        <td><ul><li>Same data type as key.</li><li>A 2D tensor with shape (numTokens,  numKHeads*headSize).</li></ul></td>
        <td>FLOAT32, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to request on the Device.</td>
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
    </tbody></table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thead>
        <tr>
        <th>Return value</th>
        <th>Error Code</th>
        <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td> ACLNN_ERR_PARAM_NULLPTR </td>
        <td> 161001 </td>
        <td>The mandatory input, output, or attribute passed in is a null pointer.</td>
        </tr>
        <tr>
        <td> ACLNN_ERR_PARAM_INVALID </td>
        <td> 161002 </td>
        <td>The data types and data formats of the input and output are not within the supported range.</td>
        </tr>
        <tr>
        <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
        <td rowspan="2"> 561002 </td>
        <td>The shape information of multiple input tensors does not match.</td>
        </tr>
        <tr>
        <td>The shape information between the input attributes and input tensors does not match.</td>
        </tr>
        <tr>
        <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
        <td rowspan="2">361001</td>
        <td>query or key is not 64-byte aligned.</td>
        </tr>
        <tr>
        <td>rotaryDim is greater than headSize.</td>
        </tr>
    </tbody></table>

## aclnnRopeWithSinCosCacheV2

- **Parameters:**
    <table style="undefined;table-layout: fixed; width: 1030px"> <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
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
        <td>Workspace memory address allocated on the device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device side, obtained by calling the first-phase API <code>aclnnRopeWithSinCosCacheV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
    </tr>
    </tbody></table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRopeWithSinCosCacheV2 defaults to a deterministic implementation.
- queryIn, keyIn, and cosSinCache only support 2-D shape inputs.
- The data types of input queryIn, keyIn, and cosSinCache must be the same.
- headSize: a multiple of 32 when the data type is BFLOAT16 or FLOAT16, and a multiple of 16 when the data type is FLOAT32.
- rotaryDim: always less than or equal to headSize; a multiple of 32 when the data type is BFLOAT16 or FLOAT16, and a multiple of 16 when the data type is FLOAT32. In mrope mode, the sum of all elements in mropeSection must be half of the rotaryDim value.
- The value of the input tensor positions must be less than maxSeqLen, the dimension-0 size of cosSinCache.
- In mrope mode, mropeSection: currently only supports [11, 11, 10], [16, 24, 24], [24, 20, 20], [8, 12, 12], and [16, 16, 16, 16].
- In mrope mode, cacheMode only supports 0 and 1. When mropeSection is [16, 16, 16, 16], only 0 is supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_rope_with_sin_cos_cache_v2.h"
#include <iostream>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // Boilerplate: resource initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the input and output, which need to be customized according to the API.
  std::vector<int64_t> positionsShape = {2};
  std::vector<int64_t> queryInShape = {2, 64};
  std::vector<int64_t> keyInShape = {2, 64};
  std::vector<int64_t> cosSinCacheShape = {2, 32};
  std::vector<int64_t> queryOutShape = {2, 64};
  std::vector<int64_t> keyOutShape = {2, 64};
  void* positionsDeviceAddr = nullptr;
  void* queryInDeviceAddr = nullptr;
  void* keyInDeviceAddr = nullptr;
  void* cosSinCacheDeviceAddr = nullptr;
  void* queryOutDeviceAddr = nullptr;
  void* keyOutDeviceAddr = nullptr;

  aclTensor* positions = nullptr;
  aclTensor* queryIn = nullptr;
  aclTensor* keyIn = nullptr;
  aclTensor* cosSinCache = nullptr;
  int64_t headSize = 32;
  bool isNeoxStyle = true;
  int64_t cacheMode = 1;
  aclTensor *queryOut = nullptr;
  aclTensor *keyOut = nullptr;

  std::vector<int64_t> positionsHostData = {0, 1};
  std::vector<float> queryInHostData = {74, 54, 84, 125, 23, 78, 37, 72, 27, 98, 34, 107, 29, 23, 54, 60, 70, 49,
                                        119, 54, 29, 54, 41, 99, 27, 62, 5, 46, 108, 39, 24, 123, 33, 82, 6, 40, 88,
                                        24, 6, 116, 38, 119, 110, 5, 30, 79, 87, 18, 29, 100, 90, 24, 21, 93, 63, 68,
                                        34, 112, 119, 48, 74, 43, 85, 64, 14, 49, 128, 59, 18, 37, 123, 76, 14, 63, 10,
                                        39, 107, 124, 79, 16, 17, 76, 80, 47, 90, 41, 58, 82, 75, 80, 69, 37, 74, 36, 54,
                                        26, 32, 54, 13, 100, 105, 15, 13, 69, 122, 26, 94, 59, 29, 14, 60, 8, 24, 17, 45,
                                        33, 107, 122, 63, 111, 75, 128, 68, 31, 105, 6, 82, 99};
  std::vector<float> keyInHostData = {112, 32, 66, 114, 69, 31, 117, 122, 77, 57, 78, 119, 115, 25, 54, 27, 122, 65, 15, 85,
                                      33, 16, 36, 6, 95, 15, 43, 6, 66, 91, 14, 101, 78, 51, 110, 74, 56, 30, 127, 61, 53, 29,
                                      32, 65, 114, 77, 26, 116, 89, 38, 75, 14, 96, 91, 87, 34, 25, 42, 57, 26, 51, 43, 23, 42,
                                      40, 17, 98, 117, 53, 75, 68, 75, 38, 41, 115, 76, 67, 22, 76, 10, 24, 46, 85, 54, 61, 114,
                                      10, 59, 6, 123, 58, 10, 115, 9, 13, 58, 66, 120, 23, 30, 83, 13, 11, 76, 18, 82, 57, 4,
                                      117, 105, 8, 73, 127, 5, 91, 56, 12, 125, 20, 3, 104, 40, 46, 18, 89, 63, 99, 104};
  std::vector<float> cosSinCacheHostData = {112, 32, 66, 114, 69, 31, 117, 122, 77, 57, 78, 119, 115, 25, 54, 27, 122, 65, 15, 85,
                                      33, 16, 36, 6, 95, 15, 43, 6, 66, 91, 14, 101, 78, 51, 110, 74, 56, 30, 127, 61, 53, 29,
                                      32, 65, 114, 77, 26, 116, 89, 38, 75, 14, 96, 91, 87, 34, 25, 42, 57, 26, 51, 43, 23, 42};
  std::vector<float> queryOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> keyOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  ret = CreateAclTensor(positionsHostData, positionsShape,
                        &positionsDeviceAddr, aclDataType::ACL_INT64,
                        &positions);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(queryInHostData, queryInShape, &queryInDeviceAddr,
                      aclDataType::ACL_FLOAT, &queryIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(keyInHostData, keyInShape, &keyInDeviceAddr,
                      aclDataType::ACL_FLOAT, &keyIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosSinCacheHostData, cosSinCacheShape, &cosSinCacheDeviceAddr,
                      aclDataType::ACL_FLOAT, &cosSinCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(queryOutHostData, queryOutShape, &queryOutDeviceAddr, aclDataType::ACL_FLOAT,
                        &queryOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(keyOutHostData, keyOutShape, &keyOutDeviceAddr, aclDataType::ACL_FLOAT,
                        &keyOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API and replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnRopeWithSinCosCacheV2.
  ret = aclnnRopeWithSinCosCacheV2GetWorkspaceSize(positions, queryIn, keyIn, cosSinCache, nullptr, headSize, isNeoxStyle, cacheMode,
                                               queryOut, keyOut, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnRopeWithSinCosCacheV2GetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnRopeWithSinCosCacheV2.
  ret = aclnnRopeWithSinCosCacheV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnRopeWithSinCosCacheV2 failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Synchronize and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result on the device memory to the host side. Modify it based on the specific API definition.
  PrintOutResult(queryOutShape, &queryOutDeviceAddr);
  PrintOutResult(keyOutShape, &keyOutDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
  aclDestroyTensor(positions);
  aclDestroyTensor(queryIn);
  aclDestroyTensor(keyIn);
  aclDestroyTensor(cosSinCache);
  aclDestroyTensor(queryOut);
  aclDestroyTensor(keyOut);

  // 7. Release device resources.
  aclrtFree(positionsDeviceAddr);
  aclrtFree(queryInDeviceAddr);
  aclrtFree(keyInDeviceAddr);
  aclrtFree(cosSinCacheDeviceAddr);
  aclrtFree(queryOutDeviceAddr);
  aclrtFree(keyOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
