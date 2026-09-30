# aclnnAllGatherMatmul

## Supported Products

| Product                                                             | Supported |
| :------------------------------------------------------------------ | :-------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term> |     √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |     √     |
| <term>Atlas 200I/500 A2 inference products</term>                   |     ×     |
| <term>Atlas inference products</term>                               |     ×     |
| <term>Atlas training products</term>                                |     ×     |

**Note**: When using this API, ensure that the driver firmware package and CANN package are in the 8.0.RC2 version or later. Otherwise, an error, such as BUS ERROR, will be reported.

## Description

- **API function**: Integrates AllGather communication and MatMul compute.
- **Formula**:

    $$
    output=allgather(x1)@x2+bias
    $$
    
    $$
    gatherOut=allgather(x1)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAllGatherMatmulGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnAllGatherMatmul` is called to perform computation.

```cpp
aclnnStatus aclnnAllGatherMatmulGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const char      *group,
    int64_t         gatherIndex,
    int64_t         commTurn,
    int64_t         streamMode,
    const aclTensor *output,
    const aclTensor *gatherOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnAllGatherMatmul(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnAllGatherMatmulGetWorkspaceSize

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1567px" class="_resize_120"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
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
          <td>x1</td>
          <td>Input</td>
          <td>Corresponds to x1 in the formula.</td>
          <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of x2. </li><li>The current version only supports two-dimensional input shapes and non-transposed scenarios.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Corresponds to x2 in the formula.</td>
          <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of x1. </li><li>The current version only supports two-dimensional input shapes and both transposed and non-transposed scenarios. </li><li>non-contiguous tensor can be constructed via transposition.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td> Corresponds to bias in the formula.</td>
          <td><term>For Atlas A3 training products and Atlas A3 inference products</term>, scenarios with non-zero bias input are not currently supported.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>A string on the host identifying the communication domain, that is, the name of the communication domain.</td>
          <td>It is obtained through the Hccl API extern HcclResult HcclGetCommName(HcclComm comm, char* commName);. commName corresponds to group.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>gatherIndex</td>
          <td>Input</td>
          <td> Identifies the Gather target.</td>
          <td><ul><li>0 indicates that the target is x1, and 1 indicates that the target is x2. </li><li>The current version supports only 0.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, the total data size divided by the communication size at a time.</td>
          <td>The current version supports only 0.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enumeration of stream modes.</td>
          <td>The current version supports only 1.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>Output</td>
          <td>Result of AllGather communication and MatMul computation, that is, output in the formula.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The data type must be the same as that of x1.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>gatherOut</td>
          <td>Output</td>
          <td>Result of AllGather communication, that is, gatherOut in the formula.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The data type must be the same as that of x1.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
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

    aclnnStatus: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
    
    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1180px"> <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
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
        <td>The input x1, x2, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of x1, x2, bias, or output is not supported.</td>
    </tr>
    <tr>
        <td>The value of streamMode is not within the valid range.</td>
    </tr>
    <tr>
        <td>x1 is an empty tensor.</td>
    </tr>
    </tbody>
    </table>

## aclnnAllGatherMatmul

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px">
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
        <td>Size of the workspace to be allocated on the device, which is obtained by calling <code>aclnnAllGatherMatmulGetWorkspaceSize</code>.</td>
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
    </tbody></table>
    
- **Returns**

    aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnAllGatherMatmul` defaults to a deterministic implementation.

- Input `x1` is 2D (m, k). `x2` must be 2D (k, n). The axes meet the input parameter requirements of the MatMul operator. Their k axes must be equal and fall within the range of [256, 65535).
- `x1` and `x2` support empty tensor scenarios. m and n can be empty, k cannot be empty, and the following conditions must be met:
    - m is empty, k is not empty, and n is not empty.
    - If m is not empty, k is not empty, and n is empty.
    - If m is empty, k is not empty, and n is empty.
- The output is 2D with shape (m*rank_size, n), where rank_size indicates the number of devices.
- <term>Atlas A2 training products and Atlas A2 inference products</term>: supports two, four, or eight devices and only the all mesh networking of HCCS links.
- <term>Atlas A3 training products and Atlas A3 inference products</term>: supports two, four, eight, sixteen, or thirty-two devices and only the double ring networking of HCCS links.
- <term>Atlas A2 training products and Atlas A2 inference products</term>: supports only one communication domain for MC2 operators within a model.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products, Atlas A2 inference products</term>, <term>Atlas A3 training products, and Atlas A3 inference products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <vector>
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_all_gather_matmul.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while(0)

    constexpr int DEV_NUM = 8;

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i +1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    struct Args {
        int rankId;
        HcclComm hcclComm;
        aclrtStream stream;
      };

    int launchOneThread_AllGatherMm(Args &args)
    {
        int ret = aclrtSetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
        std::vector<int64_t> x1Shape = {128, 256};
        std::vector<int64_t> x2Shape = {256, 512};
        std::vector<int64_t> biasShape = {512};
        std::vector<int64_t> outShape = {128 * DEV_NUM, 512};
        std::vector<int64_t> gatherOutShape = {128 * DEV_NUM, 256};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *gatherOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *out = nullptr;
        aclTensor *gatherOut = nullptr;

        int64_t gatherIndex = 0;
        int64_t commTurn = 0;
        int64_t streamMode = 1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long gatherOutShapeSize = GetShapeSize(gatherOutShape);

        std::vector<int16_t> x1HostData(x1ShapeSize, 0);
        std::vector<int16_t> x2HostData(x2ShapeSize, 0);
        std::vector<int16_t> biasHostData(biasShapeSize, 0);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        std::vector<int16_t> gatherOutHostData(gatherOutShapeSize, 0);

        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gatherOutHostData, gatherOutShape, &gatherOutDeviceAddr,
            aclDataType::ACL_FLOAT16, &gatherOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnAllGatherMatmulGetWorkspaceSize(
            x1, x2, bias, hcomName, gatherIndex, commTurn, streamMode, out, gatherOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnAllGatherMatmulGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);  return ret);
        }
        // Call the second-phase API.
        ret = aclnnAllGatherMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherMatmul failed. ret = %d \n", ret); return    ret);
        // (Fixed writing) Synchronously wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n",    ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnAllGatherMatmul execute successfully.\n", args.rankId);
        // Release device resources. Modify the configuration based on the API definition.
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
        if (gatherOut != nullptr) {
            aclDestroyTensor(gatherOut);
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
        if (gatherOutDeviceAddr != nullptr) {
            aclrtFree(gatherOutDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        ret = aclrtDestroyStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
        ret = aclrtResetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        // This sample is based on and can only run on Atlas A3.
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
        aclrtStream stream[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return   ret);
        }
        int32_t devices[DEV_NUM];
        for (int i = 0; i < DEV_NUM; i++) {
            devices[i] = i;
        }
        // Initialize the collective communication domain.
        HcclComm comms[DEV_NUM];
        ret = HcclCommInitAll(DEV_NUM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[DEV_NUM];
        // Start multiple threads.
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThread_AllGatherMm, std::ref(args [rankId])));    
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        for (int i = 0; i < DEV_NUM; i++) {
            auto hcclRet = HcclCommDestroy(comms[i]);
            CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return    -1);
        }
        aclFinalize();
        return 0;
    }
    ```
