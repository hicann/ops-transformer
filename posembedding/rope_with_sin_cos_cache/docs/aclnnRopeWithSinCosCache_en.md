# aclnnRopeWithSinCosCache

## Supported Products

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>     |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |    √     |
| <term>Atlas 200I/500 A2 inference product</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

* Description: To improve performance, the inference network passes the sin and cos inputs through a cache to perform the rotary position encoding calculation.
* Formula:

    1. **mrope mode**: The shape of the positions input is [3, numTokens]:

      $$
      cosSin[i] = cosSinCache[positions[i]]
      $$

      $$
      cos, sin = cosSin.chunk(2, dim=-1)
      $$

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

      (1) rotate\_half (GPT-NeoX style) computation formula:

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

      (2) rotate\_interleaved (GPT-J style) computation formula:

      $$
      x1 = queryRot[..., ::2]
      $$

      $$
      x2 = queryRot[..., 1::2]
      $$

      $$
      queryRot = torch.stack((o1, o2), dim=-1)
      $$

      $$
      query = torch.cat((queryRot, queryPass), dim=-1)
      $$

    2. **Rope mode**: The shape of the positions input is [numTokens]:

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

      (1) rotate\_half (GPT-NeoX style) computation formula:

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

      (2) rotate\_interleaved (GPT-J style) computation formula:
      
      $$
      x1 = query\_rot[..., ::2]
      $$

      $$
      x2 = query\_rot[..., 1::2]
      $$

      $$
      queryRot = torch.stack((o1, o2), dim=-1)
      $$

      $$
      query = torch.cat((queryRot, queryPass), dim=-1)
      $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md)calls. It is necessary to first call the `aclnnRopeWithSinCosCacheGetWorkspaceSize` interface to obtain the required workspace size for computation and the executor that includes the operator's computation process. Then call the `aclnnRopeWithSinCosCache` interface to execute the computation.

```c++
aclnnStatus aclnnRopeWithSinCosCacheGetWorkspaceSize(
    const aclTensor   *positions,
    const aclTensor   *queryIn,
    const aclTensor   *keyIn,
    const aclTensor   *cosSinCache,
    const aclIntArray *mropeSection,
    int64_t            headSize,
    bool               isNeoxStyle,
    aclTensor         *queryOut,
    aclTensor         *keyOut,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnRopeWithSinCosCache(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnRopeWithSinCosCacheGetWorkspaceSize

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
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage Notes</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
      <tbody>
      <tr>
        <td>positions</td>
        <td>Input</td>
        <td>The positions in the formula are used to select the position encoding tensor.</td>
        <td><ul><li>Empty tensors are not supported.</li><li>The shape for rope mode is (numTokens).</li><li>The shape for mrope mode is (3, numTokens).</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryIn</td>
        <td>Input</td>
        <td>The query in the formula, the first tensor to perform rotary position encoding.</td>
        <td><ul><li>Empty tensors are not supported.</li><li>It must be a 2D Tensor with a shape of (numTokens, numQHeads*headSize).</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyIn</td>
        <td>Input</td>
        <td>The second tensor to perform rotary position encoding.</td>
        <td><ul><li>Empty tensors are not supported.</li><li>It must be a 2D Tensor with a shape of (numTokens, numKHeads*headSize).</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cosSinCache</td>
        <td>Input</td>
        <td>Represents the position encoding tensor involved in the computation.</td>
        <td><ul><li>Empty tensors are not supported.</li><li>It must be a 2D Tensor with a shape of (maxSeqLen, rotaryDim), where maxSeqLen represents the maximum sequence length the model can handle, and rotaryDim represents the dimension size of the rotary position embedding.</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mropeSection</td>
        <td>Input</td>
        <td>The mropeSection in the formula is used to integrate the positional encoding tensor information of the input in mrope mode.</td>
        <td>The mropeSection attribute indicates enabling the mrope mode. If the mrope mode is not enabled (rope mode), the input should be nullptr.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headSize</td>
        <td>Input</td>
        <td>Dimension size of each attention head.</td>
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
        <td><ul><li>true indicates GPT-NeoX style computation mode.</li><li>false indicates GPT-J style computation mode.</li></ul></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryOut</td>
        <td>Output</td>
        <td>The result of the query after applying rotary position encoding.</td>
        <td><ul><li>The data type is the same as query.</li><li>It must be a 2D Tensor with a shape of (numTokens, numQHeads*headSize).</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>keyOut</td>
        <td>Output</td>
        <td>The result of applying rotary position encoding to the key.</td>
        <td><ul><li>The data type is the same as the key.</li><li>It must be a 2D Tensor with a shape of (numTokens, numKHeads*headSize).</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to apply for on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which includes the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
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
        <td> ACLNN_ERR_PARAM_NULLPTR </td>
        <td> 161001 </td>
        <td>The required input, required output, or required attribute passed in is a null pointer.</td>
        </tr>
        <tr>
        <td> ACLNN_ERR_PARAM_INVALID </td>
        <td> 161002 </td>
        <td>The data type and format of input and output are not within the supported range.</td>
        </tr>
        <tr>
        <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
        <td rowspan="2"> 561002 </td>
        <td>The shape information between multiple input tensors does not match.</td>
        </tr>
        <tr>
        <td>The shape information between the input attributes and the input tensor does not match.</td>
        </tr>
    </tbody></table>

## aclnnRopeWithSinCosCache

- **Parameters:**
    <table style="undefined;table-layout: fixed; width: 1030px">
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
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
        <td>The memory address of the workspace allocated on the device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The size of the workspace allocated on the device side, obtained by the first-phase API <code>aclnnRopeWithSinCosCacheGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor, which includes the operator computation process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for task execution.</td>
    </tr>
    </tbody></table>

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNormRopeConcatBackward defaults to deterministic implementation.

- `queryIn`, `keyIn`, and `cosSinCache` only support 2-dimensional shape input.
- The data types of `queryIn`, `keyIn`, and `cosSinCache` inputs need to be consistent.
- `headSize`: When the data type is BFLOAT16 or FLOAT16, it should be a multiple of 32; when the data type is FLOAT32, it should be a multiple of 16.
- `rotaryDim`: Always less than or equal to headSize; when the data type is BFLOAT16 or FLOAT16, it should be a multiple of 32, and when the data type is FLOAT32, it should be a multiple of 16; in mrope mode, it should satisfy rotaryDim = mropeSection[0] + mropeSection[1] + mropeSection[2].
- The value of the input tensor positions should be less than the `maxSeqLen` of the 0th dimension of `cosSinCache`.
- `mropeSection` (in mrope mode): the value is limited to [16, 24, 24], and the value of `rotaryDim` is 128.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_rope_with_sin_cos_cache.h"
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
  // Fixed writing, resource initialization.
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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // Call aclrtMemcpy to copy data from the host side to the device side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Calculate the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor interface to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) device/stream initialization, refer to the acl API manual.
  // Fill in the deviceId according to your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct input and output, which needs to be customized according to the API definition.
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
                      aclDataType::ACL_BF16, &queryIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(keyInHostData, keyInShape, &keyInDeviceAddr,
                      aclDataType::ACL_BF16, &keyIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosSinCacheHostData, cosSinCacheShape, &cosSinCacheDeviceAddr,
                      aclDataType::ACL_BF16, &cosSinCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(queryOutHostData, queryOutShape, &queryOutDeviceAddr, aclDataType::ACL_BF16,
                        &queryOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(keyOutHostData, keyOutShape, &keyOutDeviceAddr, aclDataType::ACL_BF16,
                        &keyOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnRopeWithSinCosCache.
  ret = aclnnRopeWithSinCosCacheGetWorkspaceSize(positions, queryIn, keyIn, cosSinCache, nullptr, headSize, isNeoxStyle, 
                                               queryOut, keyOut, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnRopeWithSinCosCacheGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize calculated from the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnRopeWithSinCosCache.
  ret = aclnnRopeWithSinCosCache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnRopeWithSinCosCache failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Fixed writing) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(queryOutShape, &queryOutDeviceAddr);
  PrintOutResult(keyOutShape, &keyOutDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
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
