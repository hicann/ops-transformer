# aclnnApplyRotaryPosEmb

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:38:39.896Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: To improve performance, the inference network fuses the query and key operators into a single operator. It performs rotary position embedding computation, and the computation result is updated in-place.
- Formula:

  $$
  query\_q1 = query[..., : query.shape[-1] // 2]
  $$

  $$
  query\_q2 = query[..., query.shape[-1] // 2 :]
  $$

  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1)
  $$

  $$
  key\_k1 = key[..., : key.shape[-1] // 2]
  $$

  $$
  key\_k2 = key[..., key.shape[-1] // 2 :]
  $$

  $$
  key\_rotate = torch.cat((-key\_k2, key\_k1), dim=-1)
  $$

  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$

  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnApplyRotaryPosEmbGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size according to the procedure, and then call the "aclnnApplyRotaryPosEmb" API to execute the calculation.

```cpp
aclnnStatus aclnnApplyRotaryPosEmbGetWorkspaceSize(
  aclTensor       *queryRef,
  aclTensor       *keyRef,
  const aclTensor *cos,
  const aclTensor *sin,
  int64_t         layout,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnApplyRotaryPosEmb(
  void          *workspace,
  uint64_t      workspaceSize,
  aclOpExecutor *executor,
  aclrtStream   stream)
```

## aclnnApplyRotaryPosEmbGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
    <col style="width: 157px">
    <col style="width: 125px">
    <col style="width: 282px">
    <col style="width: 319px">
    <col style="width: 196px">
    <col style="width: 122px">
    <col style="width: 230px">
    <col style="width: 145px">
  </colgroup>
  <tr>
    <th align="center">Parameter</th>
    <th align="center">Input/Output</th>
    <th align="center">Description</th>
    <th align="center">Instruction</th>
    <th align="center">Data Type</th>
    <th align="center">Data Format</th>
    <th align="center">Dimension (shape)</th>
    <th align="center">Non-contiguous Tensor</th>
  </tr>
  <tbody>
    <tr>
      <td>queryRef</td>
      <td>Input/Output</td>
      <td>First tensor on which rotary position embedding is performed, which is query in the formula. The calculation result is updated in place.</td>
      <td>        <ul>
          <li>Does not support empty tensors.</li>
          <li>The last dimension (D) of the shape must be equal to 128 or 64.</li>
        </ul>
      </td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4 (layout is 1) or 3 (layout is 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyRef</td>
      <td>Input/Output</td>
      <td>Second tensor on which rotary position embedding is performed, which is key in the formula. The calculation result is updated in place.</td>
      <td>        <ul>
          <li>Does not support empty tensors.</li>
          <li>The last dimension (D) of the shape must be equal to 128 or 64.</li>
        </ul>
      </td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4 (layout is 1) or 3 (layout is 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Position encoding tensor used for calculation, which is cos in the formula.</td>
      <td>        <ul>
          <li>Does not support empty tensors.</li>
          <li>The third dimension (N) of the shape must be equal to 1.</li>
          <li>The last dimension (D) of the shape must be equal to 128 or 64.</li>
          <li>The shapes of cos and sin must be the same.</li>
        </ul>
      </td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4 (layout is 1) or 3 (layout is 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Position encoding tensor used for calculation, which is sin in the formula.</td>
      <td>        <ul>
          <li>Does not support empty tensors.</li>
          <li>The 3rd dimension (N) of shape must be equal to 1.</li>
          <li>The last dimension (D) of shape must be equal to 128 or 64.</li>
          <li>The shapes of cos and sin must be the same.</li>
        </ul>
      </td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>4 (layout is 1) or 3 (layout is 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>Input</td>
      <td>Indicates the layout format of the input tensor.</td>
      <td>        <ul>
          <li>Value range: 1-BSND, 2-SBND, 3-BNSD, 4-TND.</li>
          <li>Currently, only the BSND layout format (value: 1) and the TND layout format (value: 4) are supported.</li>
        </ul>
      </td>
      <td>int64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: Do not support BFLOAT16

  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table>
  <tr>
  <td align="center" style="width:169px;">Return Value</td>
  <td align="center" style="width:125px;">Error Code</td>
  <td align="center" style="width:855px;">Description</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">The queryRef, keyRef, cos, or sin passed in is a null pointer.</td>
  </tr>
  <tr>
  <td rowspan="2" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="2" align="left">161002</td>
  <td align="left">The data type or data format of the queryRef, keyRef, cos, or sin passed in is not within the supported range, or the shape does not match.</td>
  </tr>
  <tr><td align="left">The layout parameter passed in is not within the supported range.</td></tr>
  </table>

## aclnnApplyRotaryPosEmb

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px">
  <colgroup>
    <col style="width: 169px">
    <col style="width: 125px">
    <col style="width: 855px">
  </colgroup>
  <tr>
    <th align="center">Parameter</th>
    <th align="center">Input/Output</th>
    <th align="center">Description</th>
  </tr>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>The memory address of the workspace applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The size of the workspace applied by the Device side, obtained by the first-phase API aclnnApplyRotaryPosEmbGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor that contains the operator calculation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: Return value (status code). For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnApplyRotaryPosEmb defaults to a deterministic implementation.

<!-- npu="A3,910b,310p" id8 -->
- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - When layout is 1, the first two dimensions (B, S) of the input shapes of queryRef, keyRef, cos, and sin must be equal; when layout is 4, the first dimension (T) must be equal.
  - The last dimension (D) of the queryRef, keyRef, cos, and sin input shapes must be equal.
  - The input tensors queryRef, keyRef, cos, and sin must have the same dtype.
  - When layout is 1, the shape of input queryRef is represented as (q_b, q_s, q_n, q_d), the shape of keyRef is represented as (q_b, q_s, k_n, q_d), and the shape of cos and sin is represented as (q_b, q_s, 1, q_d). Here, b indicates batch_size, s indicates seq_length, n indicates head_num, and d indicates head_dim. When layout is 4, the shape of input queryRef is represented as (q_t, q_n, q_d), the shape of keyRef is represented as (q_t, k_n, q_d), and the shape of cos and sin is represented as (q_t, 1, q_d). Here, t indicates the concatenated axis of b and s, n indicates head_num, and d indicates head_dim.

  - When the input is BFLOAT16, cast is 1, castSize is 4, and DtypeSize is 2.
  - When the input is FLOAT16 or FLOAT32, cast is 0, and castSize = DtypeSize (2 for FLOAT16, 4 for FLOAT32).

  Use lastDim to represent the value of head_dim, the last dimension of the input shape, and calculate the required UB space:
    `ub_required = (q_n + k_n) * lastDim * castSize * 2 + lastDim * DtypeSize * 4 + (q_n + k_n) * lastDim * castSize + (q_n + k_n) * lastDim * castSize * 2 + cast * (lastDim * 4 * 2)`,
  When the calculated `ub_required` exceeds the total UB space of the current AI processor, this fused operator is not supported.

<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT products</term>:
  - For any layout, queryRef and keyRef must have the same dimensions except the N dimension; the last dimension (D) of the queryRef and keyRef input shapes must be equal, and the last dimension (D) of the cos and sin input shapes must be equal and less than or equal to the last dimension (D) of the queryRef and keyRef input shapes.
  - The input tensors queryRef, keyRef, cos, and sin must have the same dtype.

<!-- end id9 -->

## Example

The sample code is as follows and is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb.h"
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
    // Call aclrtMalloc to allocate device-side memory.
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
    // 1. Boilerplate for device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check based on your needs.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API definition.
    std::vector<int64_t> queryShape = {1, 1, 1, 128};
    std::vector<int64_t> keyShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    int64_t layout = 1;

    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;

    std::vector<float> queryHostData = {74, 54, 84, 125, 23, 78, 37, 72, 27, 98, 34, 107, 29, 23, 54, 60, 70, 49,
                                        119, 54, 29, 54, 41, 99, 27, 62, 5, 46, 108, 39, 24, 123, 33, 82, 6, 40, 88,
                                        24, 6, 116, 38, 119, 110, 5, 30, 79, 87, 18, 29, 100, 90, 24, 21, 93, 63, 68,
                                        34, 112, 119, 48, 74, 43, 85, 64, 14, 49, 128, 59, 18, 37, 123, 76, 14, 63, 10,
                                        39, 107, 124, 79, 16, 17, 76, 80, 47, 90, 41, 58, 82, 75, 80, 69, 37, 74, 36, 54,
                                        26, 32, 54, 13, 100, 105, 15, 13, 69, 122, 26, 94, 59, 29, 14, 60, 8, 24, 17, 45,
                                        33, 107, 122, 63, 111, 75, 128, 68, 31, 105, 6, 82, 99};
    std::vector<float> keyHostData = {112, 32, 66, 114, 69, 31, 117, 122, 77, 57, 78, 119, 115, 25, 54, 27, 122, 65, 15, 85,
                                      33, 16, 36, 6, 95, 15, 43, 6, 66, 91, 14, 101, 78, 51, 110, 74, 56, 30, 127, 61, 53, 29,
                                      32, 65, 114, 77, 26, 116, 89, 38, 75, 14, 96, 91, 87, 34, 25, 42, 57, 26, 51, 43, 23, 42,
                                      40, 17, 98, 117, 53, 75, 68, 75, 38, 41, 115, 76, 67, 22, 76, 10, 24, 46, 85, 54, 61, 114,
                                      10, 59, 6, 123, 58, 10, 115, 9, 13, 58, 66, 120, 23, 30, 83, 13, 11, 76, 18, 82, 57, 4,
                                      117, 105, 8, 73, 127, 5, 91, 56, 12, 125, 20, 3, 104, 40, 46, 18, 89, 63, 99, 104};
    std::vector<float> cosHostData = {41, 37, 17, 25, 49, 25, 22, 24, 110, 120, 107, 3, 82, 66, 75, 86, 85, 115, 110, 56, 52,
                                      39, 86, 23, 36, 71, 20, 73, 113, 25, 114, 56, 125, 80, 95, 82, 31, 63, 99, 62, 23, 55, 30,
                                      99, 42, 121, 15, 24, 97, 87, 81, 67, 43, 21, 13, 9, 33, 29, 117, 10, 114, 61, 98, 15, 78,
                                      108, 48, 97, 1, 3, 78, 109, 57, 46, 47, 56, 50, 66, 81, 77, 17, 128, 68, 121, 47, 91, 114,
                                      125, 51, 108, 31, 15, 47, 78, 109, 115, 113, 26, 53, 97, 1, 111, 103, 58, 106, 68, 11,
                                      104, 22, 79, 61, 127, 86, 39, 33, 123, 102, 39, 64, 41, 119, 120, 61, 29, 94, 68, 36, 12};
    std::vector<float> sinHostData = {46, 56, 56, 101, 66, 10, 96, 16, 86, 57, 102, 66, 12, 105, 76, 58, 90, 6, 79, 128, 126,
                                      82, 41, 3, 45, 7, 66, 4, 46, 22, 31, 26, 37, 63, 97, 84, 91, 90, 47, 77, 90, 34, 41, 83,
                                      91, 108, 120, 13, 90, 32, 85, 37, 119, 31, 51, 82, 122, 125, 7, 116, 121, 108, 38, 56,
                                      100, 20, 97, 119, 10, 4, 53, 13, 46, 82, 103, 119, 124, 80, 23, 67, 78, 56, 119, 122, 40,
                                      58, 128, 27, 30, 52, 71, 42, 123, 69, 4, 5, 116, 97, 38, 107, 8, 4, 65, 120, 40, 22, 60,
                                      44, 48, 66, 68, 125, 4, 93, 112, 112, 113, 90, 94, 23, 104, 39, 85, 84, 64, 128, 96, 119};

    // Create a query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the cos aclTensor.
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the sin aclTensor.
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API aclnnApplyRotaryPosEmb.
    ret = aclnnApplyRotaryPosEmbGetWorkspaceSize(query, key, cos, sin, layout, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyRotaryPosEmbGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API aclnnApplyRotaryPosEmb.
    ret = aclnnApplyRotaryPosEmb(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyRotaryPosEmb failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate. Synchronously wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value. Copy the result from device memory to the host side. Modify it according to the specific API definition.
    auto size = GetShapeSize(queryShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(keyShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), keyDeviceAddr, size1 * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData1[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify it according to the specific API definition.
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);

    // 7. Release device resources.
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
