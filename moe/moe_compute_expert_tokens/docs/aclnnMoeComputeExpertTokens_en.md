# aclnnMoeComputeExpertTokens

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- **API Description:**

Searches for the location of the last row processed by each expert in binary search mode during MoE computation.

- **Formula:**

    $$
    for\: i\: in\: range(numExperts)
    $$

    $$
    out_{i}=BinarySearch(sortedExperts, i)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeComputeExpertTokensGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeComputeExpertTokens` is called to perform computation.

```cpp
aclnnStatus aclnnMoeComputeExpertTokensGetWorkspaceSize(
    const aclTensor *sortedExperts,
    int64_t          numExperts,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnMoeComputeExpertTokens(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeComputeExpertTokensGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 187px">
    <col style="width: 121px">
    <col style="width: 287px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 146px">
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
        <td>sortedExperts</td>
        <td>Input</td>
        <td>sortedExperts in the formula, representing the sorted expert array.</td>
        <td>The value range of the tensor is [0, numExperts – 1], and the shape size must be less than 2**24.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>numExperts</td>
        <td>Input</td>
        <td>Total number of experts.</td>
        <td>The value must be greater than 0 but cannot exceed 2048.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td>The shape size is equal to the number of experts.</td>
        <td>Same as sortedExperts</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
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
    </tbody></table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 253px">
    <col style="width: 140px">
    <col style="width: 762px">
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
        <td> ACLNN_ERR_PARAM_NULLPTR </td>
        <td> 161001 </td>
        <td>The passed sortedExperts is a null pointer.</td>
        </tr>
        <tr>
        <td rowspan="2"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="2"> 161002 </td>
        <td>The data type of sortedExperts is not supported.</td>
        </tr>
        <tr>
        <td>The data format of sortedExperts is not supported.</td>
        </tr>
        <tr>
        <td> ACLNN_ERR_INNER_TILING_ERROR </td>
        <td> 561002 </td>
        <td>The shape of sortedExperts or out is not equal to a 1D tensor.</td>
        </tr>
    </tbody></table>

## aclnnMoeComputeExpertTokens

- **Parameters**

    <table>
            <thead>
                <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
            </thead>
            <tbody>
                <tr><td>workspace</td><td>Input</td><td>The memory address of the workspace allocated on the device side.</td></tr>
                <tr><td>workspaceSize</td><td>Input</td><td>The workspace size allocated on the device side, obtained from the first API aclnnInplaceAddGetWorkspaceSize.</td></tr>
                <tr><td>executor</td><td>Input</td><td>The operator executor, which contains the computation process of the operator.</td></tr>
                <tr><td>stream</td><td>Input</td><td>Stream for executing the task. </td></tr>
            </tbody>
    </table>

- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnMoeComputeExpertTokens` defaults to a deterministic implementation.

- The size of the input shape cannot exceed the upper limit of the memory that can be allocated by the device. Otherwise, the program terminates abnormally.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_compute_expert_tokens.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor)
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
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external ACL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> sortedExpertForSourceRowShape = {6};
    std::vector<int64_t> outShape = {3};

    void* sortedExpertForSourceRowAddr = nullptr;
    void* outAddr = nullptr;

    aclTensor* sortedExperts = nullptr;
    aclTensor* out = nullptr;

    std::vector<int32_t> sortedExpertForSourceRowData = {0, 0, 1, 1, 2, 2};
    std::vector<int32_t> outData = {3, 4, 5};
    std::int32_t numExperts = 3;

    // Create an input aclTensor.
    ret = CreateAclTensor(sortedExpertForSourceRowData,
        sortedExpertForSourceRowShape,
        &sortedExpertForSourceRowAddr,
        aclDataType::ACL_INT32,
        &sortedExperts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    //Create an Out aclTensor.
    ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 3. Call the first-phase API of aclnnMoeComputeExpertTokens.
    ret = aclnnMoeComputeExpertTokensGetWorkspaceSize(
        sortedExperts, numExperts, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeComputeExpertTokensGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMoeComputeExpertTokens.
    ret = aclnnMoeComputeExpertTokens(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeComputeExpertTokens failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Destroy aclTensor. Modify the code based on the API definition.
    aclDestroyTensor(sortedExperts);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(sortedExpertForSourceRowAddr);
    aclrtFree(outAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
