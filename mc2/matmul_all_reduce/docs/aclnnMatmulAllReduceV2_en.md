# aclnnMatmulAllReduceV2

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

**Note**: When using this API, ensure that the driver firmware package and CANN package are in the 8.0.RC2 version or later. Otherwise, an error, such as BUS ERROR, will be reported.

## Function

- **Description**: Integrates the MatMul computation and AllReduce communication.
- **Formula**:

    $$
    output = allreduce(x1 @ x2 + bias + x3)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMatmulAllReduceV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnMatmulAllReduceV2` is called to perform computation.

```cpp
aclnnStatus aclnnMatmulAllReduceV2GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *x3,
    const char      *group,
    const char      *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnMatmulAllReduceV2(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnMatmulAllReduceV2GetWorkspaceSize

- **Parameters:**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
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
          <td>Left matrix of MatMul computation, that is, <code>x1</code> in the formula.</td>
          <td><ul><li>The current version supports only 2D or 3D inputs. </li><li>The non-transpose scenario is supported.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of MatMul computation, that is, <code>x2</code> in the formula.</td>
          <td><ul><li>The current version supports only 2D inputs. </li><li> The transpose and non-transpose scenarios are supported. </li><li>non-contiguous tensor are supported when the last two axes are transposed.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3</td>
          <td>Input</td>
          <td>Addition after MatMul computation, that is, <code>x3</code> in the formula.</td>
          <td>The shape is the same as that after MatMul computation.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td>Corresponds to <code>bias</code> in the formula.</td>
          <td>The current version supports only 1D inputs.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>0-1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communication domain name.</td>
          <td>It is obtained through the <code>extern HcclResult HcclGetCommName(HcclComm comm, char* commName);</code> API provided by HCCL, where <code>commName</code> is the same as <code>group</code>.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>Reduce operation type.</td>
          <td>The current version supports only <code>sum</code>.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, the total data volume divided by single communication volume.</td>
          <td>The current version supports only <code>0</code>.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enumeration of the stream mode.</td>
          <td>The current version supports only <code>1</code>.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>Output</td>
          <td>Result of MatMul computation and AllReduce communication, that is, <code>output</code> in the formula.</td>
          <td>The number of dimensions of <code>output</code> is the same as that of <code>x1</code>.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
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
        <td>The input <code>x1</code>, <code>x2</code>, or <code>output</code> is passed as a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of <code>x1</code>, <code>x2</code>, <code>bias</code>, <code>x3</code>, or <code>output</code> is not supported.</td>
    </tr>
    <tr>
        <td>The value of <code>reduceOp</code> or <code>streamMode</code> is invalid.</td>
    </tr>
    <tr>
        <td>The shape of <code>x1</code>, <code>x2</code>, <code>bias</code>, <code>x3</code>, or <code>output</code> does not meet the requirements.</td>
    </tr>
    </tbody>
    </table>

## aclnnMatmulAllReduceV2

- **Parameters:**

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
        <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API <code>aclnnMatmulAllReduceV2GetWorkspaceSize</code>.</td>
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

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMatmulAllReduceV2` defaults to non-deterministic implementation. You can call `aclrtCtxSetSysParamOpt` to enable deterministic computation.

- MC2 is disabled in incremental scenarios but enabled in full scenarios.
- The input `x1` can be 2D or 3D with shape (m, k) or (b, s, k). `x2` must be 2D with shape (k, n). The axes meet the input parameter requirements of the MatMul operator, and the k axes of `x1` and `x2` are equal. If `bias` is not empty, its shape is (n).
- The values of b*s, `m`, `k`, and `n` cannot exceed `2147483647` (INT32_MAX).
- When the shape of the input `x1` is (b, s, k), the shape of `output` is (b, s, n). When the shape of the input `x1` is (m, k), the shape of `output` is (m, n).
- The data types of inputs `x1`,`x2`, and `bias` must be the same as the data type of `output`. The input `x1`, `x2`, `x3`, or `output` cannot be a null pointer.
- Only the all-mesh networking of the HCCS link is supported.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: 1, 2, 4, and 8 ranks are supported.
- <term>Atlas A2 training products/Atlas A2 inference products</term>: Only one communication domain for MC2 operators within a model is supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    ```Cpp
    #include <iostream>
    #include <vector>
    #include <thread>
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_matmul_all_reduce_v2.h"

    int ndev = 8;

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

    int launchOneThreadMatmulAllReduce(Args &args) {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128];
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x3Shape = {32, 128};
        std::vector<int64_t> outShape = {32, 128};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x3DeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x3 = nullptr;
        aclTensor *out = nullptr;

        int64_t commTurn = 0;
        int64_t streamMode = 1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x3ShapeSize = GetShapeSize(x3Shape);
        long long outShapeSize = GetShapeSize(outShape);
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int16_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> x3HostData(x3ShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API and pass out to x3.
        ret = aclnnMatmulAllReduceV2GetWorkspaceSize(x1, x2, bias, x3, hcom_name, "sum", commTurn, streamMode, out, &   workspaceSize, &executor);

        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnMatmulAllReduceV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMatmulAllReduceV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulAllReduceV2 failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnMatmulAllReduceV2 execute success \n", args.rankId);
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
        if (x3 != nullptr) {
            aclDestroyTensor(x3);
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
        if (x3DeviceAddr != nullptr) {
            aclrtFree(x3DeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
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

    int main(int argc, char *argv[]) {
        int ret;
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // Initialize the collective communicator.
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
        // Start multiple threads.
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadMatmulAllReduce, std::ref(args  [rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
