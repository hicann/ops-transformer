# aclnnAlltoAllMatmul

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Fuses `AlltoAll` communication, `Permute` (to ensure contiguous memory addresses after communication), and `Matmul` computation using a **communication-before-computation** sequence.
- Formulas: Assume that the input `x1` shape is `(BS, H)`, and `rankSize` represents the number of NPUs.

  $$
  commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
  permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
  output = permutedOut @ x2 + bias \\
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAlltoAllMatmulGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnAlltoAllMatmul` is called to perform computation.

```cpp
aclnnStatus aclnnAlltoAllMatmulGetWorkspaceSize(
  const aclTensor*   x1, 
  const aclTensor*   x2,
  const aclTensor*   biasOptional,
  const aclIntArray* alltoAllAxesOptional,
  const char*        group,
  bool               transposeX1,
  bool               transposeX2,
  const aclTensor*   output,
  const aclTensor*   alltoAllOutOptional,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllMatmul(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAlltoAllMatmulGetWorkspaceSize

- ​**Parameter Description**

    <table style="undefined;table-layout: fixed; width: 1556px"><colgroup>
    <col style="width: 154px">
    <col style="width: 123px">
    <col style="width: 270px">
    <col style="width: 295px">
    <col style="width: 245px">
    <col style="width: 120px">
    <col style="width: 203px">
    <col style="width: 146px">
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
    <td>x1</td>
    <td>Input</td>
    <td>Left matrix input of the fused operator, corresponding to x1 in the formula.</td>
    <td>The result of AlltoAll communication and Permute operations on this input is used as the left matrix input for MatMul computation.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2D, with shape (BS, H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right matrix input of the fused operator, which is also the right matrix for MatMul computation.</td>
    <td>The input is directly used as the right matrix input for MatMul computation.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2D, with shape (H*rankSize, N)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>Optional input</td>
    <td>Bias to be accumulated after matrix multiplication, corresponding to bias in the formula.</td>
    <td>A null pointer can be passed. The restrictions on the data type vary according to the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>1D, with shape (N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllAxesOptional</td>
    <td>Input</td>
    <td>Direction of data exchange between AlltoAll and Permute.</td>
    <td>The value can be left empty or [-2, -1]. If the value is left empty, it is processed as [-2, -1] by default, indicating that the input is converted from (BS, H) to (BS/rankSize, rankSize*H).</td>
    <td>aclIntArray* (element type: INT64)</td>
    <td>-</td>
    <td>1D, shape: (2)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Input</td>
    <td>A string on the host identifying the communication domain name. The `commName` obtained via the HcclGetCommName API is used as the value for this parameter.</td>
    <td>The string length must be in the range of (0, 128).</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX1</td>
    <td>Input</td>
    <td>Whether the left matrix has been transposed.</td>
    <td>Currently, this parameter cannot be set to True.</td>
    <td>bool</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX2</td>
    <td>Input</td>
    <td>Whether the right matrix has been transposed.</td>
    <td>If this parameter is set to True, the shape of the right matrix is (N, rankSize*H).</td>
    <td>bool</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>output</td>
    <td>Input</td>
    <td>Final calculation result.</td>
    <td>The data type is the same as that of x1.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2D, with shape (BS/rankSize, N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllOutOptional</td>
    <td>Optional output</td>
    <td>Receives the content after AlltoAll and Permute.</td>
    <td>If nullptr is passed, no communication output is generated.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2D, with shape (BS/rankSize, H*rankSize)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace to be allocated on the device.</td>
    <td></td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td></td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width:282px">
    <col style="width:120px">
    <col style="width:747px">
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
      <td>Mandatory input and output tensors are null pointers.</td>
    </tr>
    <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The input and output data types are not supported.</td>
    </tr>
    <tr>
        <td>The input tensor is an empty tensor.</td>
    </tr>
    <tr>
        <td>alltoAllAxesOptional is invalid.</td>
    </tr>
    <tr>
        <td>transposeX1 is true.</td>
    </tr>
    <tr>
        <td>The communicator length is invalid.</td>
    </tr>
    <tr>
        <td>Invalid input or output tensor dimensions.</td>
    </tr>
    <tr>
        <td>The input and output formats are proprietary formats.</td>
    </tr>
      </tbody>
  </table>

## aclnnAlltoAllMatmul

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
        <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnAlltoAllMatmulGetWorkspaceSize API.</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

* Deterministic computing is supported by default.
* The number of NPUs (rankSize) varies depending on the device model.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: 2, 4, or 8 NPUs are supported.
  - <term>Atlas A3 training products/Atlas A3 inference products</term>: 2, 4, 8, or 16 NPUs are supported.
  - Ascend 950PR/Ascend 950DT: 2, 4, 8, or 16 NPUs are supported.
* The variable BS used in the shape must be exactly divided by the number of NPUs.
* The values of BS and N cannot exceed 2147483647 (INT32_MAX). The value of BS cannot be less than 0, and the value of N cannot be less than 1.
* The H*rankSize range varies depending on the device model.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: The value range is [1, 35000].
  - <term>Atlas A3 training products/Atlas A3 inference products</term> Ascend 950PR/Ascend 950DT: The value range is [2, 65535].
* The support for empty tensors varies depending on the device model.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: No empty tensor is supported.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> / Ascend 950PR/Ascend 950DT: Only empty tensors with the first dimension (BS) of the input x1 being 0 are supported.
* The support for non-contiguous tensors varies depending on the device model.
  - <term>Atlas A2/A3 training products/Atlas A2/A3 inference products</term>: No non-contiguous tensor is supported.
  - Ascend 950PR/Ascend 950DT: Only x2 can be a non-contiguous tensor. Other non-contiguous tensors are not supported.
* The data types of the input tensors x1 and x2 must be the same as those of the output tensors and alltoAllOutOptional. The input tensors x1, x2, and output are not null pointers.
* The data type of biasOptional is restricted by the device model.
  - <term>Atlas A2/A3 training products/Atlas A2/A3 inference products</term>: When the data type of the input tensors x1 and x2 is FLOAT16, the data type of the input tensor biasOptional can be FLOAT16. When the data type of the input tensors x1 and x2 is BFLOAT16, the data type of the input tensor biasOptional can be FLOAT32.
  - Ascend 950PR/Ascend 950DT: When the data type of the input tensors x1 and x2 is FLOAT16, the data type of the input tensor biasOptional can be FLOAT16 or FLOAT32. When the data type of the input tensors x1 and x2 is BFLOAT16, the data type of the input tensor biasOptional can be BFLOAT16 or FLOAT32.
* MC2 operators cannot be called concurrently, nor can different MC2 operators.
* Inter-super node communication is not supported. Only intra-super node communication is supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

Note: This sample code calls some HCCL collective communication library APIs, including HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclapiref/hcclcpp_07_0001.html>.

- <term>Atlas A2/A3 training products/Atlas A2/A3 inference products</term>:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_allto_all_matmul.h"
    
    int ndev = 2;
    
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
    
    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }
    
    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
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
    
    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };
    
    int launchOneThreadAlltoAllMatmul(Args &args)
    {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);
    
        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> alltoalloutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *alltoalloutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *out = nullptr;
        aclTensor *alltoallout = nullptr;
    
        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;
    
        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long alltoalloutShapeSize = GetShapeSize(alltoalloutShape);
        std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
        std::vector<op::fp16_t> x2HostData(x2ShapeSize, 1);
        std::vector<op::fp16_t> biasHostData(biasShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        std::vector<op::fp16_t> alltoalloutHostData(alltoalloutShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(alltoalloutHostData, alltoalloutShape, &alltoalloutDeviceAddr, aclDataType::ACL_FLOAT16, &alltoallout);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API.
        ret = aclnnAlltoAllMatmulGetWorkspaceSize(x1, x2, bias, alltoAllAxesOptional, hcom_name, false, false,
                                                out, alltoallout, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAlltoAllMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllMatmul failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllMatmul execute success \n", args.rankId);
        // Release device resources. Modify the code based on the API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (alltoallout != nullptr) {
            aclDestroyTensor(alltoallout);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (alltoalloutDeviceAddr != nullptr) {
            aclrtFree(alltoalloutDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }
    
    int main(int argc, char *argv[])
    {
        // This example is implemented based on Atlas A2 and can only run on Atlas A2.
        int ret = aclInit(nullptr);
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // Initialize the collective communication domain.
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // Enable multi-threading.
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllMatmul, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

- Ascend 950PR/Ascend 950DT:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnnop/aclnn_allto_all_matmul.h"
  
    int ndev = 2;
  
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
  
    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }
  
    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
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
  
    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };
  
    int launchOneThreadAlltoAllMatmul(Args &args)
    {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);
  
        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> alltoalloutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *alltoalloutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *out = nullptr;
        aclTensor *alltoallout = nullptr;
  
        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;
  
        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long alltoalloutShapeSize = GetShapeSize(alltoalloutShape);
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int16_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        std::vector<int16_t> alltoalloutHostData(alltoalloutShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(alltoalloutHostData, alltoalloutShape, &alltoalloutDeviceAddr, aclDataType::ACL_FLOAT16, &alltoallout);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API.
        ret = aclnnAlltoAllMatmulGetWorkspaceSize(x1, x2, bias, alltoAllAxesOptional, hcom_name, false, false,
                                                out, alltoallout, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAlltoAllMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllMatmul failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllMatmul execute success \n", args.rankId);
        // Release device resources. Modify the code based on the API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (alltoallout != nullptr) {
            aclDestroyTensor(alltoallout);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (alltoalloutDeviceAddr != nullptr) {
            aclrtFree(alltoalloutDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }
  
    int main(int argc, char *argv[])
    {
        // This sample is implemented based on the Atlas A5 and must run on the Atlas A5.
        int ret = aclInit(nullptr);
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // Initialize the collective communication domain.
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // Enable multi-threading.
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllMatmul, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
