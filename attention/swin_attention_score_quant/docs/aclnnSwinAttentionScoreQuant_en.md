# aclnnSwinAttentionScoreQuant

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/attention/swin_attention_score_quant)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT|    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>|    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>|    ×     |
| <term>Atlas inference products</term>|    √     |
| <term>Atlas training products</term>|    ×     |

## Function

- Operator function: performs the attention calculation in the Swin-transformer scenario. Compared with the SwinAttentionScore operator, this operator supports int8 quantization.
- Formulas:

$$
out= Softmax(QK^T + bias1 + bias2)V
$$

## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnSwinAttentionScoreQuantGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnSwinAttentionScoreQuant** is called to perform computation.

```Cpp
aclnnStatus aclnnSwinAttentionScoreQuantGetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *value, 
    const aclTensor *scaleQuant, 
    const aclTensor *scaleDequant1, 
    const aclTensor *scaleDequant2, 
    const aclTensor *biasQuantOptional, 
    const aclTensor *biasDequant1Optional, 
    const aclTensor *biasDequant2Optional, 
    const aclTensor *paddingMask1Optional, 
    const aclTensor *paddingMask2Optional, 
    bool             queryTranspose, 
    bool             keyTranspose, 
    bool             valueTranspose, 
    int64_t          softmaxAxes, 
    const aclTensor *out, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwinAttentionScoreQuant(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnSwinAttentionScoreQuantGetWorkspaceSize

* **Parameters**:

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 300px">  
            <col style="width: 400px">  
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
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>query</td>
        <td>Input</td>
        <td>Query tensor of the input sample, Q in the formula.</td>
        <td>The input dimension [N, C, S, H] must be the same as that of the key and value. N indicates the batch size, C indicates the channel depth, S indicates the sequence length, and H indicates the headNum, S&lt;=1024, H=32/64. The NC dimension can be any value.</td>
        <td>INT8</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>Input</td>
        <td>Feature tensor of each position of the input sample, K in the formula.</td>
        <td>The input dimension must be the same as that of the query and value.</td>
        <td>INT8</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>Input</td>
        <td>Tensor of the value at each position after the attention is calculated, V in the formula.</td>
        <td>The input dimension must be the same as that of the query and key.</td>
        <td>INT8</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>scaleQuant</td>
        <td>Input</td>
        <td>Scale tensor for quantization after softmax normalization of attention</td>
        <td>Input shape: [1, S], S&lt;=1024</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>scaleDequant1</td>
        <td>Input</td>
        <td>Scale tensor for dequantization during attention calculation</td>
        <td>Input shape: [1, S], S&lt;=1024</td>
        <td>UINT64</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>scaleDequant2</td>
        <td>Input</td>
        <td>Scale tensor for dequantization of the output after attention calculation</td>
        <td>Input shape: [1, H], H=32/64</td>
        <td>UINT64</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>biasQuantOptional</td>
        <td>Input</td>
        <td>Offset tensor for quantization after softmax normalization of attention</td>
        <td>Input shape: [1, S], S&lt;=1024</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>biasDequant1Optional</td>
        <td>Input</td>
        <td>Offset tensor for dequantization during attention calculation</td>
        <td>Input shape: [1, S], S&lt;=1024</td>
        <td>INT32</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>biasDequant2Optional</td>
        <td>Input</td>
        <td>Offset tensor for dequantization of the output after attention calculation</td>
        <td>Input shape: [1, H], H=32/64</td>
        <td>INT32</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>paddingMask1Optional</td>
        <td>Input</td>
        <td>bias1 in the formula</td>
        <td>Dimension [1,C,S,S], S&lt;=1024. The C dimension can be any value.</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>paddingMask2Optional</td>
        <td>Input</td>
        <td>bias2 in the formula, reserved parameter</td>
        <td>Currently, only nullptr</td> is supported.
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryTranspose</td>
        <td>Input</td>
        <td>Whether to transpose the query</td>
        <td>Currently, only non-transposition is supported.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyTranspose</td>
        <td>Input</td>
        <td>Whether to transpose the key</td>
        <td>Currently, only non-transposition is supported.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueTranspose</td>
        <td>Input</td>
        <td>Whether to transpose the value</td>
        <td>Currently, only non-transposition is supported.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxAxes</td>
        <td>Input</td>
        <td>Dimension for softmax computation</td>
        <td>Currently, only -1 (the last dimension of the tensor) is supported.</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out</td>
        <td>Output</td>
        <td>Final output of the calculation formula</td>
        <td>-</td>
        <td>FLOAT16</td>
        <td>ND</td>
        <td>4</td>
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
        <td>Returns the operator executor, including the operator calculation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

* **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

   The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 281px">
    <col style="width: 119px">
    <col style="width: 749px">
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
        <td>The input tensor is a null pointer.</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data type or format of the input or output parameter is not supported.</td>
    </tr>
    </tbody>
    </table>

## aclnnSwinAttentionScoreQuant

* **Parameters**

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
        <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnSwinAttentionScoreQuantGetWorkspaceSize.</td>
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

* **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- When the QKV input dimension is [N, C, S, H], S <= 1024, H = 32/64, and the NC dimension can be any value.
- The input of the QKV transpose with the dimension [N, C, S, H] is not supported.
- Only asymmetric quantization is supported.
- Bias2 is not supported.
- Only the softmax operation can be performed on the last dimension of QK^T + bias1 + bias2.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swin_attention_score_quant.h"

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
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on API definitions.
    int64_t B = 1288;
    int64_t N = 3;
    int64_t S = 49;
    int64_t H = 32;
    std::vector<int64_t> qkvShape = {B, N, S, H};
    std::vector<int64_t> sShape = {1, S};
    std::vector<int64_t> hShape = {1, H};
    std::vector<int64_t> mask1Shape = {1, N, S, S};
    std::vector<int64_t> attentionScoreShape = {B, N, S, H};
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* scaleQuantDeviceAddr = nullptr;
    void* scaleDequant1DeviceAddr = nullptr;
    void* scaleDequant2DeviceAddr = nullptr;
    void* biasQuantDeviceAddr = nullptr;
    void* biasDequant1DeviceAddr = nullptr;
    void* biasDequant2DeviceAddr = nullptr;
    void* paddingMask1DeviceAddr = nullptr;
    void* attentionScoreDeviceAddr = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* value = nullptr;
    aclTensor* scaleQuant = nullptr;
    aclTensor* scaleDequant1 = nullptr;
    aclTensor* scaleDequant2 = nullptr;
    aclTensor* biasQuantOptional = nullptr;
    aclTensor* biasDequant1Optional = nullptr;
    aclTensor* biasDequant2Optional = nullptr;
    aclTensor* paddingMask1Optional = nullptr;
    aclTensor* paddingMask2Optional = nullptr;
    aclTensor* attentionScore = nullptr;
    std::vector<int8_t> queryHostData(B*N*S*H, 1);
    std::vector<int8_t> keyHostData(B*N*S*H, 1);
    std::vector<int8_t> valueHostData(B*N*S*H, 1);
    std::vector<uint16_t> scaleQuantHostData(S, 1);
    std::vector<uint64_t> scaleDequant1HostData(S, 1);
    std::vector<uint64_t> scaleDequant2HostData(H, 1);
    std::vector<uint16_t> biasQuantHostData(S, 1);
    std::vector<int32_t> biasDequant1HostData(S, 1);
    std::vector<int32_t> biasDequant2HostData(H, 1);
    std::vector<uint16_t> paddingMask1HostData(1*N*S*H, 1);
    std::vector<uint16_t> attentionScoreHostData(B*N*S*H, 0);
    // Create an input aclTensor.
    ret = CreateAclTensor(queryHostData, qkvShape, &queryDeviceAddr, aclDataType::ACL_INT8, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkvShape, &keyDeviceAddr, aclDataType::ACL_INT8, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, qkvShape, &valueDeviceAddr, aclDataType::ACL_INT8, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleQuantHostData, sShape, &scaleQuantDeviceAddr, aclDataType::ACL_FLOAT16, &scaleQuant);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleDequant1HostData, sShape, &scaleDequant1DeviceAddr, aclDataType::ACL_UINT64, &scaleDequant1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleDequant2HostData, hShape, &scaleDequant2DeviceAddr, aclDataType::ACL_UINT64, &scaleDequant2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasQuantHostData, sShape, &biasQuantDeviceAddr, aclDataType::ACL_FLOAT16, &biasQuantOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasDequant1HostData, sShape, &biasDequant1DeviceAddr, aclDataType::ACL_INT32, &biasDequant1Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasDequant2HostData, hShape, &biasDequant2DeviceAddr, aclDataType::ACL_INT32, &biasDequant2Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(paddingMask1HostData, mask1Shape, &paddingMask1DeviceAddr, aclDataType::ACL_FLOAT16, &paddingMask1Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(attentionScoreHostData, attentionScoreShape, &attentionScoreDeviceAddr, aclDataType::ACL_FLOAT16, &attentionScore);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnn API call example
    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnn.
    ret = aclnnSwinAttentionScoreQuantGetWorkspaceSize(query, key, value, scaleQuant, scaleDequant1, scaleDequant2,
        biasQuantOptional, biasDequant1Optional, biasDequant2Optional, paddingMask1Optional, paddingMask2Optional,
        false, false, false, -1, attentionScore, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinAttentionScoreQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclnnFakeQuantPerTensorAffineCachemask.
    ret = aclnnSwinAttentionScoreQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinAttentionScoreQuant failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host.
    auto size = GetShapeSize(attentionScoreShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), attentionScoreDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(scaleQuant);
    aclDestroyTensor(scaleDequant1);
    aclDestroyTensor(scaleDequant2);
    aclDestroyTensor(biasQuantOptional);
    aclDestroyTensor(biasDequant1Optional);
    aclDestroyTensor(biasDequant2Optional);
    aclDestroyTensor(paddingMask1Optional);
    aclDestroyTensor(attentionScore);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(scaleQuantDeviceAddr);
    aclrtFree(scaleDequant1DeviceAddr);
    aclrtFree(scaleDequant2DeviceAddr);
    aclrtFree(biasQuantDeviceAddr);
    aclrtFree(biasDequant1DeviceAddr);
    aclrtFree(biasDequant2DeviceAddr);
    aclrtFree(paddingMask1DeviceAddr);
    aclrtFree(attentionScoreDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
