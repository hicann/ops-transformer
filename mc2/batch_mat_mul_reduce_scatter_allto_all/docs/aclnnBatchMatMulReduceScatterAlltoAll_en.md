# aclnnBatchMatMulReduceScatterAlltoAll

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/batch_mat_mul_reduce_scatter_allto_all)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Description

- Function: BatchMatMulReduceScatterAllToAll is a computing and communication fusion operator that implements parallelism of BatchMatMul computation and ReduceScatter and AllToAll collective communication.

- The calculation formula is as follows: BatchMatMul computation --> Transpose (required when yShardType is 0) --> ReduceScatter collective communication --> Add --> AllToAll collective communication. Below shows the computation logic, where `y` is the output.

$$
temp1 = BatchMatMul(x, weight)
$$

$$
temp2 = ReduceScatter(temp1)
$$

$$
temp3 = Add(temp2, bias)
$$

$$
y = AllToAll(temp3)
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnBatchMatMulReduceScatterAlltoAll` is called to perform computation.

```cpp
aclnnStatus aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* biasOptional,
    const char*      groupEp,
    const char*      groupTp,
    int64_t          epWorldSize,
    int64_t          tpWorldSize,
    int64_t          yShardType,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnBatchMatMulReduceScatterAlltoAll(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
    <col style="width: 170px">
    <col style="width: 140px">
    <col style="width: 310px">
    <col style="width: 350px">
    <col style="width: 150px">
    <col style="width: 120px">
    <col style="width: 150px">
    <col style="width: 190px">
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
    <td>x(aclTensor*)</td>
    <td>Input</td>
    <td>Left matrix computed by BatchMatMul</td>
    <td>Must be 3-dimensional.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>×</td>
    </tr>
    <tr>
    <td>weight(aclTensor*)</td>
    <td>Input</td>
    <td>Right matrix computed by BatchMatMul</td>
    <td><ul><li>The data type is the same as that of x. </li><li>Must be 3-dimensional.</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>√ (only for transposition)</td>
    </tr>
    <tr>
    <td>biasOptional(aclTensor*)</td>
    <td>Input</td>
    <td>bias computed by Add</td>
    <td><ul><li>The Add operation must be performed after the ReduceScatter communication.</li><li>If x is of type FLOAT16, biasOptional must be of type FLOAT16. If x is of type BFLOAT16, biasOptional must be of type FLOAT32. </li><li>The two-dimensional or three-dimensional data is supported. Null pointers can be passed.</li></ul></td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>2 or 3</td>
    <td>×</td>
    </tr>
    <tr>
    <td>groupEp(char*)</td>
    <td>Input</td>
    <td>Name of the communicator for expert parallelism</td>
    <td>The string length must be greater than 0 and less than 128.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp(char*)</td>
    <td>Input</td>
    <td>Name of the communicator for tensor parallelism</td>
    <td>The string length must be greater than 0 and less than 128.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize (int64_t) </td>
    <td>Input</td>
    <td>ep communicator size</td>
    <td>Value range: 2, 4, 8, 16, or 32.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize (int64_t) </td>
    <td>Input</td>
    <td>tp Communication domain size</td>
    <td>Value range: 2, 4, 8, 16, 32.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>yShardType (int64_t) </td>
    <td>Input</td>
    <td>Integer</td>
    <td><ul><li>0 indicates that ReduceScatter is performed on the H dimension (the second dimension of the BatchMatMul calculation result, the result has three dimensions, and the dimension indexes are 0, 1, and 2) by tp. </li><li>1 indicates that ReduceScatter is performed on dimension C (the first dimension of the BatchMatMul computation result) by tp.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>out(aclTensor*)</td>
    <td>Output</td>
    <td>: batch_matmul calculation + reduce_scatter calculation + all_to_all communication result </td>;
    <td><ul><li> The data type is the same as that of input x. </li><li> must be three-dimensional.</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t) </td>
    <td>Output</td>
    <td>Returns the workspace size.</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor(aclOpExecutor**)</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

- **Returns**

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
    <td>1. The input x, weight, groupEp, groupTp, or out is a null pointer.</td>
    </tr>
    <tr>
    <td class="merged-cell" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
    <td class="merged-cell" rowspan="5">161002</td>
    <td>1. The length of the groupEp or groupTp string is invalid.</td>
    </tr>
    <tr>
    <td>2. The input data type is not supported.</td>
    </tr>
    <tr>
    <td>3. The attribute value is invalid.</td>
    </tr>
    <tr>
    <td>4. The aclTensor dimension is invalid.</td>
    </tr>
    <tr>
    <td>5. The aclTensor shape is invalid.</td>
    </tr>
    </tbody></table>

## aclnnBatchMatMulReduceScatterAlltoAll

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
    <td>Size of the workspace to be allocated on the device, which is obtained by calling `aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize`.</td>
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

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnBatchMatMulReduceScatterAlltoAll` defaults to a deterministic implementation.

  - Due to the requirements of collective communication and BatchMatMul computation, the input and output shapes must meet the following mathematical relationship (ep = epWorldSize and tp = tpWorldSize):
  - ReduceScatter by H axis (yShardType = 0):
   
      - x: (E/ep, ep*C, M/tp) 
      - weight: (E/ep, M/tp, H)
      - biasOptional: (E/ep, 1, H/tp) for 3D and (E/ep, H/tp) for 2D when the pointer is not empty
      - y: (E, C, H/tp)

  - ReduceScatter scenario by the C axis, that is, the scenario where yShardType is 1:
     - x: (E/ep, ep*tp*C/tp, M/tp)
     - weight: (E/ep, M/tp, H)
     - biasOptional: (E/ep, 1, H) for 3D and (E/ep, H) for 2D when the pointer is not empty
     - y: (E, C/tp, H)

- Data relationship description:
  - For example, if x.size(0) is equal to E/tp and y.size(0) is equal to E, it indicates that y.size(0) = ep*x.size(0), where y.size (0) is an integer multiple of ep. Other relationships are similar to this.
  - The value range of E is [2, 512], and E is an integer multiple of ep.
  - The value range of H is [1, 65535]. When yShardType is 0, H is an integer multiple of tp.
  - The value range of M/tp is [1, 65535].
  - The value range of E/ep is [1, 32].
  - ep and tp can only be 2, 4, 8, 16, or 32.
  - groupEp and groupTp cannot have the same name.
  - When C is greater than 0, its maximum value is the upper limit of the operator device memory. When yShardType is 1, C is an integer multiple of tp.
  - MC2 operators cannot be called concurrently, nor can different MC2 operators.
  - Cross-supernode operations are not supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A3 training products/Atlas A3 inference products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_batch_matmul_reduce_scatter_all_to_all.h"

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

    constexpr int EP_WORLD_SIZE = 4;
    constexpr int TP_WORLD_SIZE = 2;
    constexpr int DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

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
        HcclComm hcclEpComm;
        HcclComm hcclTpComm;
        aclrtStream stream;
        aclrtContext context;
    };

    int LaunchOneThreadBatchMMRSAlltoAll(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
        char hcomTpName[128] = {0};
        ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, stream = %p, context = %p\n", args.rankId,
            hcomEpName, hcomTpName, args.stream, args.context);

        int64_t E = 4 * EP_WORLD_SIZE;
        int64_t C = 6 * TP_WORLD_SIZE;
        int64_t H = 2 * TP_WORLD_SIZE;
        int64_t M = 6 * TP_WORLD_SIZE;
        int64_t xShardType = 1;
        
        std::vector<int64_t> xShape;
        std::vector<int64_t> weightShape;
        std::vector<int64_t> biasShape;
        std::vector<int64_t> yOutShape;

        if (xShardType == 1) {
            xShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, M / TP_WORLD_SIZE};
            weightShape = {E / EP_WORLD_SIZE, M / TP_WORLD_SIZE, H};
            biasShape = {E / EP_WORLD_SIZE, 1, H};
            yOutShape = {E, C / TP_WORLD_SIZE, H};
        } else if (xShardType == 0) {
            xShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, M / TP_WORLD_SIZE};
            weightShape = {E / EP_WORLD_SIZE, M / TP_WORLD_SIZE, H};
            biasShape = {E / EP_WORLD_SIZE, 1, H / TP_WORLD_SIZE};
            yOutShape = {E, C, H / TP_WORLD_SIZE};
        } else {
            LOG_PRINT("[ERROR] unsupported xShardType = %ld.\n", xShardType);
            return -1;
        }

        printf("x_shape: %d %d %d\n", xShape[0], xShape[1], xShape[2]);
        printf("weight_shape: %d %d %d\n", weightShape[0], weightShape[1], weightShape[2]);
        printf("bias_shape: %d %d %d\n", biasShape[0], biasShape[1], biasShape[2]);
        printf("y_shape: %d %d %d\n", yOutShape[0], yOutShape[1], yOutShape[2]);

        void *xDeviceAddr = nullptr;
        void *weightDeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *yOutDeviceAddr = nullptr;
        aclTensor *x = nullptr;
        aclTensor *weight = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *yOut = nullptr;

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long xShapeSize = GetShapeSize(xShape);
        long long weightShapeSize = GetShapeSize(weightShape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long yOutShapeSize = GetShapeSize(yOutShape);

        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int16_t> weightHostData(weightShapeSize, 2);
        std::vector<int16_t> biasHostData(biasShapeSize, 3);
        std::vector<int16_t> y1OutHostData(yOutShapeSize, 0);

        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(y1OutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT16, &yOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(x, weight, bias, hcomEpName, hcomTpName, EP_WORLD_SIZE,
            TP_WORLD_SIZE, xShardType, yOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnBatchMatMulReduceScatterAlltoAll(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnBatchMatMulReduceScatterAlltoAll failed. ret = %d \n", ret);
            return ret);
        // (Fixed writing) Synchronously wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnBatchMatMulReduceScatterAlltoAll execute successfully.\n", args.rankId);
        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        // Release device resources. Modify the configuration based on the API definition.
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (weight != nullptr) {
            aclDestroyTensor(weight);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (yOut != nullptr) {
            aclDestroyTensor(yOut);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (weightDeviceAddr != nullptr) {
            aclrtFree(weightDeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (yOutDeviceAddr != nullptr) {
            aclrtFree(yOutDeviceAddr);
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
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }

        int32_t devicesEp[DEV_NUM];
        int32_t devicesTp[DEV_NUM];

        //Initialize the EP domain. ep = 4 {0,2,4,6} {1,3,5,7}
        HcclComm commsEp[DEV_NUM];
        for (int i = 0; i < TP_WORLD_SIZE; i++) {
            for (int j =0; j < EP_WORLD_SIZE; j++) {
                devicesEp[j + i * EP_WORLD_SIZE] = i + j * TP_WORLD_SIZE;
            }
            ret = HcclCommInitAll(EP_WORLD_SIZE, &devicesEp[i * EP_WORLD_SIZE], &commsEp[i * EP_WORLD_SIZE]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d \n", i, ret);
                return ret);
        }

        //Initialize the TP domain. tp = 4 {0,1},{2,3},{4,5},{6,7}
        HcclComm commsTp[DEV_NUM];
        for (int i = 0; i < EP_WORLD_SIZE; i++) {
            for (int j =0; j < TP_WORLD_SIZE; j++) {
                devicesTp[j + i * TP_WORLD_SIZE] = j + i * TP_WORLD_SIZE;
            }
            ret = HcclCommInitAll(TP_WORLD_SIZE, &devicesTp[i * TP_WORLD_SIZE], &commsTp[i * TP_WORLD_SIZE]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d \n", i, ret);
                return ret);
        }

        Args args[DEV_NUM];
        // Start multiple threads.
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclEpComm = commsEp[rankId % TP_WORLD_SIZE * EP_WORLD_SIZE + rankId / TP_WORLD_SIZE];
            args[rankId].hcclTpComm = commsTp[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new std::thread(&LaunchOneThreadBatchMMRSAlltoAll, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
