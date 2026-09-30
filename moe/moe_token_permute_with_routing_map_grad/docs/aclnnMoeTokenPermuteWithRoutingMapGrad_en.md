# aclnnMoeTokenPermuteWithRoutingMapGrad

## Product Support

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Performs the backpropagation of `aclnnMoeTokenPermuteWithRoutingMap`.
- **Formula**:

    $$
    permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
    $$

    $$
    capacity = permutedTokenOutputGrad.size(0) / numExperts
    $$

    - When `probs` is not set to `None`:
    
    $$
    probsGradOutOptional = zeros(tokens_num, numExperts)
    $$
    
    - When `paddedMode` is set to `true`:
    
        $$
        probsGradOutOptional [sortedIndices[i], i/capacity] = permutedProbsOutputGradOptional[i]
        $$
    
    - When `paddedMode` is set to `false`:
    
        $$
        probsGradOutOptional = maskedscatter(probsGradOutOptional,routingMap,permutedProbsOutputGradOptional)
        $$
    
    - If `probs` is set to `None`:
    
        $$
        tokensGradout= zeros(restoreShapeOptional, dtype=permutedTokens.dtype, device=permutedTokens.device)
        $$
        
        $$
        tokensGradout[permuteTokenId[i]] += permutedTokens[outIndex[i]]
        $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMoeTokenPermuteWithRoutingMapGrad` is called to perform computation.

* `aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(const aclTensor *permutedTokenOutputGrad, const aclTensor *permutedProbsOutputGradOptional, const aclTensor *sortedIndices, const aclTensor *routingMapOptional, int64_t numExperts, int64_t tokensNum, bool dropAndPad, aclTensor *tokensGradOut, aclTensor *probsGradOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize

- **Parameters:**
  
  - `permutedTokenOutputGrad` (aclTensor\*, computation input): aclTensor on the device. It is the gradient of the forward output `permutedTokens`, and the value is a 2D tensor. In drop/pad-less mode, the shape must be 2D with size (tokens_num \* topK_num, hidden_size). In drop/pad mode, the shape must be a 2D with size (experts_num \* capacity, hidden_size). `topK_num` indicates the number of experts selected for each token, and `capacity` indicates the number of tokens selected by each expert. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported, but empty tensors are not supported.
  - `permutedProbsOutputGradOptional` (aclTensor\*, computation input): aclTensor on the device. This input is optional. If it is not passed, `probsGradOutOptional` does not need to be computed. In drop/pad-less mode, the shape must be 1D with size (tokens_num \* topK_num). In drop/pad mode, the shape must be 1D with size (experts_num \* capacity). `topK_num` indicates the number of experts selected for each token, and `capacity` indicates the number of tokens selected by each expert. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `sortedIndices` (aclTensor\*, input): aclTensor on the device. In drop/pad-less mode, the value must be a 1D shape with size (tokens_num \* topK_num,). The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. The index value range is [0, tokens_num \* topK_num – 1]. In drop/pad mode, the value must be a 1D tensor with shape (experts_num \* capacity). The data type can be INT32. The [data format](../../../docs/en/context/data_format.md) must be ND. The index value range is [0, experts_num \* capacity – 1]. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported.
  - `routingMap` (aclTensor\*, computation input): aclTensor on the device, which indicates the mapping between tokens and experts, that is, `routingMap` in the formula. The value must be a 2D shape with size (tokens_num, experts_num). The data type can be INT8 or BOOL. If the data type is INT8, the value can be `0` or `1`. If the data type is BOOL, the value can be `true` or `false`. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. In drop/pad-less mode, each row must contain *topK* `true` or `1` values.
  - `experts_num` (int64_t, computation input): number of experts involved in the computation.
  - `tokens_num` (int64_t, computation input): number of tokens involved in the computation.
  - `dropAndPad` (bool, computation input): `true` indicates that `dropPaddedMode` is enabled, and `false` indicates that `dropPaddedMode` is disabled.
  - `tokensGradOut` (aclTensor\*, output): gradient of the input `permutedTokens`. The value must be a 2D tensor with shape (tokens_num, hidden_size). The data type is the same as that of `permutedTokenOutputGrad`. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
  - `probsGradOutOptional` (aclTensor\*, output): gradient of the input `probs`. This is an optional output. The value must be a 2D tensor with shape (tokens_num, experts_num). The data type is the same as that of `permutedProbsOutputGradOptional`. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_format.md) must be ND. [non-contiguous tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
  - `workspaceSize` (uint64\_t\*, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.
- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

```cpp
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input and output tensors are null pointers.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
                                   2. The input and output shapes are not supported.
```

## aclnnMoeTokenPermuteWithRoutingMapGrad

- **Parameters:**

    - `workspace` (void\*, input): address of the workspace to be allocated on the device.
    - `workspaceSize` (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize`.
    - `executor` (aclOpExecutor\*, input): operator executor, containing the operator computation process.
    - `stream` (aclrtStream, input): stream for executing the task.
- **Returns:**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMoeTokenPermuteWithRoutingMapGrad` defaults to deterministic implementation.
- Non-dropPaddedMode scenario: The value of `topK_num` is less than or equal to `512`.
- Mixed precision input is not supported. That is, `permutedTokenOutputGrad`, `permutedProbsOutputGradOptional`, `tokensGradOut`, and `probsGradOutOptional` must have the same data type.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "aclnnop/aclnn_moe_token_permute_with_routing_map_grad.h"
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}


template <typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T>& hostData)
{
    size_t fileSize = 1;
    for (int64_t i : shape){
        fileSize *= i; 
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }
    // Obtain the file size.
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    hostData.reserve(fileSize);
    if (file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << "Failed to read the file." << std::endl;
        return 1;
    }
    file.close();
    return true;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        LOG_PRINT("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        LOG_PRINT("Write file Failed.");
        return false;
    }

    return true;
}
void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{

    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on the API definition.

    int64_t num_token = 4096;
    int64_t hidden_size = 7168;
    int64_t num_expert = 256;
    int64_t num_capacity = 16;
    std::vector<float> permuted_output_grad_Data(num_expert * num_capacity * hidden_size, 0);
    std::vector<int64_t> permuted_output_grad_Shape = {num_expert * num_capacity, hidden_size};
    void* permuted_output_grad_Addr = nullptr;
    aclTensor* permuted_output_grad = nullptr;

    ret = CreateAclTensor(permuted_output_grad_Data, permuted_output_grad_Shape, &permuted_output_grad_Addr,
                          aclDataType::ACL_FLOAT, &permuted_output_grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> permutedProbsOutputGradOptional(num_expert * num_capacity, 0.1);
    std::vector<int64_t> permutedProbsOutputGradOptionalShape = {num_expert * num_capacity};
    void* permutedProbsOutputGrad_Addr = nullptr;
    aclTensor* ppermutedProbsOutputGrad = nullptr;
    ret = CreateAclTensor(permutedProbsOutputGradOptional, permutedProbsOutputGradOptionalShape,
                          &permutedProbsOutputGrad_Addr, aclDataType::ACL_FLOAT, &ppermutedProbsOutputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int> sortedIndicesData(num_expert * num_capacity, 0);
    std::vector<int64_t> sortedIndicesShape = {num_expert * num_capacity};
    void* sortedIndicesAddr = nullptr;
    aclTensor* sortedIndices = nullptr;
    ReadFile("./sortedIndices.bin", sortedIndicesShape, sortedIndicesData);
    ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr, aclDataType::ACL_INT32,
                          &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<char> routingMapOptionalData(num_token * num_expert, 1);
    std::vector<int64_t> routingMapOptionalShape = {num_token , num_expert};
    void* routingMapOptionalAddr = nullptr;
    aclTensor* proutingMapOptional = nullptr;

    ret = CreateAclTensor(routingMapOptionalData, routingMapOptionalShape, &routingMapOptionalAddr,
                          aclDataType::ACL_INT8, &proutingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData(num_token * hidden_size, 0.0f);
    std::vector<int64_t> outShape = {num_token, hidden_size};
    // std::vector<int64_t> outShape = {num_token};
    void* outAddr = nullptr;
    aclTensor* out = nullptr;

    ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData2(num_token * num_expert, 0.0f);
    std::vector<int64_t> outShape2 = {num_token, num_expert};
    void* outAddr2 = nullptr;
    aclTensor* out2 = nullptr;

    ret = CreateAclTensor(outData2, outShape2, &outAddr2, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnMoeTokenPermuteGrad.
    ret = aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(permuted_output_grad, ppermutedProbsOutputGrad, sortedIndices, proutingMapOptional, num_expert, num_token, true,
                                                                 out, out2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnMoeTokenPermuteWithRoutingMapGrad.
    ret = aclnnMoeTokenPermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
              return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    PrintOutResult(outShape, &outAddr);
    PrintOutResult(outShape2, &outAddr2);

    // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
    aclDestroyTensor(permuted_output_grad);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(out);

    // 7. Release device resources.
    aclrtFree(permuted_output_grad_Addr);
    aclrtFree(sortedIndicesAddr);
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
