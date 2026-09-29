# aclnnQuantReduceScatter

## Supported Products

| Product                                                                           | Supported|
| :------------------------------------------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT                                               | √       | 
| <term>Atlas A3 training products/Atlas A3 inference products</term>                       | ×       |
| <term>Atlas A2 training products/Atlas A2 inference products</term>| ×       |
| <term>Atlas 200I/500 A2 inference products</term>                                        | ×       |
| <term>Atlas inference products</term>                                               | ×       |
| <term>Atlas training products</term>                                                | ×       |

**Note**: When using this API, ensure that the driver firmware package and CANN package are in the 8.0.RC2 version or later. Otherwise, an error, such as BUS ERROR, will be reported.

## Function

- API function: Implements quant+reduceScatter fusion computation.
- **Formula**:

    $$
    output=Reduce(AllToAllScales * AllToAllData)
    $$

    $$
    AllToAllData=AllToAll(x)
    $$

    $$
    AllToAllScales=AllToAll(scales)
    $$

    The Reduce computation is performed on data from different ranks.

## Prototype

This operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnQuantReduceScatterGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnQuantReduceScatter` is called to perform computation.

```cpp
aclnnStatus aclnnQuantReduceScatterGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *scales, 
    const char      *group, 
    const char      *reduceOp,
    aclTensor       *output, 
    uint64_t        *workspaceSize, 
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantReduceScatter(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    const aclrtStream stream)
```

## aclnnQuantReduceScatterGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1556px"><colgroup>
    <col style="width: 161px">
    <col style="width: 141px">
    <col style="width: 245px">  
    <col style="width: 408px">  
    <col style="width: 191px">  
    <col style="width: 120px"> 
    <col style="width: 145px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>x</td>
        <td>Input</td>
        <td>x in the formula.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The supported shapes are (BS, H) or (B, S, H). B indicates the batch size, S indicates the sequence length, and H indicates the hidden size. In the current version, the value of H in the input x supports any 128-aligned generalization from 1024 to 8192.</li></ul></td>
        <td>INT8, HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
    </tr>
    <tr>
        <td>scales</td>
        <td>Input</td>
        <td>Input scales in the formula.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>When the data type of scales is FLOAT8_E8M0, the data type of x must be FLOAT8_E4M3FN, FLOAT8_E5M2, the shape of x is (BS, H) or (B, S, H), and the shape of scales must be (BS, H/64, 2) or (B, S, H/64, 2) corresponding to the shape of x. </li><li>When the data type of scales is FLOAT, the data type of x must be INT8, HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, the shape of x is (BS, H) or (B, S, H), and the shape of scales must be (BS, H/128) or (B, S, H/128) corresponding to the shape of x.</li></ul></td>
        <td>FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>2-4</td>
        <td>√</td>
    </tr>
    <tr>
        <td>group</td>
        <td>Input</td>
        <td>Communicator ID.</td>
        <td>Communicator ID.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp</td>
        <td>Input</td>
        <td>Reduce operation type in the formula.</td>
        <td>Currently, only "sum" is supported.</td>
        <td>string</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>If the shape of x is (BS, H), the shape of the output must be (BS/rankNum, H). If the shape of x is (B, S, H), the shape of the output must be (B*S/rankNum, H). rankNum indicates the communicator size.</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
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

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md). 

    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
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
        <td>The pointer of x, scales, or output is null.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of x, scales, or output is not supported.</td>
    </tr>
    <tr>
        <td>The data type and shape of x and scales do not match.</td>
    </tr>
    <tr>
        <td>The dimensions of x and scales are not supported.</td>
    </tr>
    </tbody>
    </table>

## aclnnQuantReduceScatter

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
        <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnQuantReduceScatterGetWorkspaceSize.</td>
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

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- When the data type of x is FLOAT8_E4M3FN, FLOAT8_E5M2 and that of scales is FLOAT8_E8M0, the quantization mode of the input data is mx quantization.
- When the data type of x is INT8, HIFLOAT8, or FLOAT8_E4M3FN, FLOAT8_E5M2 and that of scales is FLOAT, the quantization mode of the input data is pertoken-pergroup quantization (groupSize = 128).
- This feature is enabled only on the Ascend 950 series platforms.
- Empty tensor input is not supported.
- The size of the communicator can be 2, 4, or 8.
- Restrictions on the use of the communicator: The aclnnQuantAllReduce and aclnnQuantReduceScatter operators can be executed only in sequence in the same communicator, and no other communication operators are allowed in the communicator.
- HCCL_BUFFSIZE: Before calling this operator, check whether the value of the HCCL_BUFFSIZE environment variable is proper. This environment variable indicates the buffer size occupied by a single communication domain, in MB. If this environment variable is not set, the default value 200 MB is used. The following condition must be met: HCCL_BUFFSIZE >= 2 *(xDataSize + scalesDataSize + 1). xDataSize indicates the size of the input x data, which is calculated as follows: xDataSize = BS* H *1 (byte). scalesDataSize indicates the size of the scales data. When the quantization mode is pertoken-pergroup quantization, the calculation formula is as follows: scalesDataSize = BS* H / 128 *4 (byte). When the quantization mode is mx quantization, the calculation formula is as follows: scalesDataSize = BS* H / 32 * 1 (byte).
- The value of H must be 128-aligned and within the range of [1024, 8192].

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- Ascend 950PR/Ascend 950DT:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <vector>
    #include <string>
    #include <cstring>
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_quant_reduce_scatter.h"
    
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

    constexpr int DEV_NUM = 2;

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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret);
                return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret);
                return ret);
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
        aclrtContext context;
    };

    int LaunchOneThreadQtReduceScatter(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret);
                return ret);
        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret);
                return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
        std::vector<int64_t> xShape = {1024, 5120};
        std::vector<int64_t> scalesShape = {1024, 40};
        std::vector<int64_t> outputShape = {1024 / DEV_NUM, 5120};
        void *xDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *outputDeviceAddr = nullptr;
        void *workspaceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *output = nullptr;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        long long xShapeSize = GetShapeSize(xShape);
        long long scalesShapeSize = GetShapeSize(scalesShape);
        long long outputShapeSize = GetShapeSize(outputShape);

        std::vector<int8_t> xHostData(xShapeSize, 0);
        std::vector<int8_t> scalesHostData(scalesShapeSize, 0);
        std::vector<int16_t> outputHostData(outputShapeSize, 0);

        // Create a tensor.
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnQuantReduceScatterGetWorkspaceSize(
            x, scales, hcomName, "sum", output, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnQuantReduceScatterGetWorkspaceSize failed. ret = %d \n", ret);
                    return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret);
                    return ret);
        }
        // Call the second-phase API.
        ret = aclnnQuantReduceScatter(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantReduceScatter failed. ret = %d \n", ret);
                return ret);
        // (Fixed writing) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnQuantReduceScatter execute successfully.\n", args.rankId);
        // Release device resources. Modify the configuration based on the API definition.
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (scales != nullptr) {
            aclDestroyTensor(scales);
        }
        if (output != nullptr) {
            aclDestroyTensor(output);
        }

        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (scalesDeviceAddr != nullptr) {
            aclrtFree(scalesDeviceAddr);
        }
        if (outputDeviceAddr != nullptr) {
            aclrtFree(outputDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        ret = aclrtDestroyStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret);
                return ret);
        
        ret = HcclCommDestroy(args.hcclComm);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret);
                return ret);

        ret = aclrtDestroyContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret);
                return ret);

        ret = aclrtResetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret);
                return ret);

        return 0;
    }
    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
        aclrtStream stream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
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
            args[rankId].context = context[rankId];
            args[rankId].stream = stream[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadQtReduceScatter, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
