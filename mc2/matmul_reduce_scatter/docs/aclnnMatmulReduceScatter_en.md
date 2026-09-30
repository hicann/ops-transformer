# aclnnMatmulReduceScatter

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

**Note**: When using this API, ensure that the driver firmware package and CANN package are in the 8.0.RC2 version or later. Otherwise, an error, such as BUS ERROR, will be reported.

## Function

- Description: Performs mm + reduce_scatter_base computation.

- Formula:

$$
output=reduce_scatter(x1@x2+bias)
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMatmulReduceScatterGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMatmulReduceScatter` is called to perform computation.

```cpp
aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* bias,
    const char*      group,
    const char*      reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor* output,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMatmulReduceScatter(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMatmulReduceScatterGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1237px"><colgroup>
    <col style="width: 252px">
    <col style="width: 121px">
    <col style="width: 523px">
    <col style="width: 216px">
    <col style="width: 125px">
    </colgroup>
    <thead>
    <tr>
    <th>Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>x1</td>
    <td>Input</td>
    <td>x1 in the formula. The data type is the same as x2. <br>The current version only supports two-dimensional input shapes and non-transposed scenarios.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>Input</td>
    <td>x2 in the formula. The data type is the same as x1. Supports non-contiguous tensor constructed via transposition. <br>The current version only supports two-dimensional inputs.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>bias</td>
    <td>Input</td>
    <term>For Atlas A3 training products and Atlas A3 inference products</term>, scenarios with non-zero bias input are not currently supported.
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Input</td>
    <td>A string on the host identifying the communication domain, that is, the name of the communication domain. The value is obtained by calling the HcclGetCommName API.</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>reduceOp</td>
    <td>Input</td>
    <td>Reduce operation type on the host. <br>Currently, only sum is supported.</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>commTurn</td>
    <td>Input</td>
    <td>Number of split communication data copies, equal to the value of total data volume divided by single communication volume. <br>Currently, only 0 is supported.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>streamMode</td>
    <td>Input</td>
    <td>Enumeration of the stream mode. Currently, only the enumerated value 1 is supported.</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output</td>
    <td>Output</td>
    <td>Result of mm computation + ReduceScatter communication. The data type is the same as x1.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace required to be allocated on the device.</td>
    <td>UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td>aclOpExecutor*</td>
    <td>ND</td>
    </tr>
    </tbody></table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter verification. The following errors may be thrown:

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
    <td>The input x1, x2, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of x1, x2, bias, or output does not meet the requirements.</td>
    </tr>
    <tr>
        <td>The value of streamMode is not within the valid range.</td>
    </tr>
    <tr>
        <td>The shape of x1, x2, or output does not meet the requirements.</td>
    </tr>
    </tbody></table>

## aclnnMatmulReduceScatter

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
    <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnMatmulReduceScatterGetWorkspaceSize.</td>
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

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - `aclnnMatmulReduceScatter` defaults to a non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic compute.

- Input `x1` must be 2D with shape `(m, k)`. m must be an integer multiple of `rank_size`.
- Input `x2` must be 2D with shape `(k, n)`. The axes must meet the input parameter requirements of the `mm` operator. The k axes must be equal and fall within the range of [256, 65535).
- `x1` and `x2` support empty tensors. m and n can be empty, but k cannot be empty. The following conditions must be met:
  - m is empty, k is not empty, and n is not empty.
  - If m is not empty, k is not empty, and n is empty.
  - If m is empty, k is not empty, and n is empty.
- Matrix `x2` can be transposed or not transposed, and matrix `x1` can only be not transposed.
- The data type of `x1` and `x2` must be the same as that of output.
- `bias` does not support non-zero input.
- The output is 2D with shape (m/rank_size, n), where `rank_size` indicates the number of devices.
- <term>Atlas A2 training products and Atlas A2 inference products</term>: supports two, four, or eight devices and only the all mesh networking of HCCS links.
- <term>Atlas A3 training products and Atlas A3 inference products</term>: supports two, four, eight, sixteen, or thirty-two devices and only the double ring networking of HCCS links.
- <term>Atlas A2 training products and Atlas A2 inference products</term>: supports only one communication domain for MC2 operators within a model.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <vector>
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_matmul_reduce_scatter.h"

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

    int launchOneThread_MmReduceScatter(Args &args)
    {
        int ret = aclrtSetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);

        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ERROR: %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
        std::vector<int64_t> x1Shape = {1024, 256};
        std::vector<int64_t> x2Shape = {256, 512};
        std::vector<int64_t> biasShape = {512};
        std::vector<int64_t> outShape = {1024 / DEV_NUM, 512};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *out = nullptr;

        int64_t commTurn = 0;
        int64_t streamMode = 1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);

        std::vector<int16_t> x1HostData(x1ShapeSize, 0);
        std::vector<int16_t> x2HostData(x2ShapeSize, 0);
        std::vector<int16_t> biasHostData(biasShapeSize, 0);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnMatmulReduceScatterGetWorkspaceSize(
            x1, x2, bias, hcomName, "sum", commTurn, streamMode, out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMatmulReduceScatterGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMatmulReduceScatter(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMatmulReduceScatter failed. ret = %d \n", ret); return ret);
        // (Fixed writing) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMatmulReduceScatter execute successfully.\n", args.rankId);
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
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }
        int32_t devices[DEV_NUM];
        for (int i = 0; i < DEV_NUM; i++) {
            devices[i] = i;
        }
        // Initialize the collective communicator.
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
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThread_MmReduceScatter, std::ref(args [rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        for (int i = 0; i < DEV_NUM; i++) {
            auto hcclRet = HcclCommDestroy(comms[i]);
            CHECK_RET(hcclRet == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return -1);
        }
        aclFinalize();
        return 0;
    }
    ```
