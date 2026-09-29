# aclnnMatmulReduceScatterV2

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_reduce_scatter_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------:|
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description
    `aclnnMatmulReduceScatterV2` extends the functionality of `aclnnMatmulReduceScatter`. Building upon the support for FLOAT16/BFLOAT16 input types for x1 and x2:
    - Ascend 950PR/Ascend 950DT:
        - The low-precision data type FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 is supported. It supports per-tensor, per-block, and mx [quantization mode](../../../docs/en/context/quant_mode_introduction.md).
    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
        - Support for the low-precision data type INT8 is added. It supports per-token/per-channel [quantization methods](../../../docs/en/context/quant_mode_introduction.md).

- Formulas:
    - Case 1: If the data types of x1 and x2 are FLOAT16 or BFLOAT16, the matmul operation is performed on the input parameters x1, x2, and bias, and then the ReduceScatter communication is performed.

        $$
        output=ReduceScatter(x1@x2 + bias_{optional})
        $$

    - Scenario 2: If the data type of x1 and x2 is FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 in the per-tensor scenario, or INT8 in the per-channel or per-token scenario, and amaxOut is not output, x1 and x2 undergo MatMul and dequantization operations, followed by a ReduceScatter communication operation.

        $$
        output=ReduceScatter((x1Scale*x2Scale)*(x1@x2 + bias_{optional}))
        $$

    - Case 3: If the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 in the perblock scenario, and amaxOut is not output, when x1 is (a0, a1) and x2 is (b0, b1), x1Scale is (ceildiv(a0, 128), ceildiv(a1, 128)) and x2Scale is (ceildiv(b0, 128), ceildiv(b1, 128)). After matmul and dequant are performed on the input parameters x1 and x2, the ReduceScatter communication is performed.

        $$
        output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
        $$

    - Case 4: If the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2 in the mx quantization scenario, and amaxOut is not output, when x1 is (a0, a1) and x2 is (b0, b1), x1Scale is (a0, ceildiv(a1, 64), 2) and x2Scale is (b0, ceildiv(b1, 64), 2). After matmul and dequant are performed on the input parameters x1 and x2, the ReduceScatter communication is performed.

        $$
        output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
        $$
    
## Function Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnMatmulReduceScatterV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, `aclnnMatmulReduceScatterV2` is called to perform computation.

```cpp
aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(
    const aclTensor* x1, 
    const aclTensor* x2, 
    const aclTensor* bias, 
    const aclTensor* x1Scale, 
    const aclTensor* x2Scale, 
    const aclTensor* quantScale, 
    int64_t          blockSize, 
    const char*      group, 
    const char*      reduceOp, 
    int64_t          commTurn, 
    int64_t          streamMode, 
    int64_t          groupSize, 
    const char*      commMode, 
    aclTensor*       output, 
    aclTensor*       amaxOutOptional, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMatmulReduceScatterV2(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnMatmulReduceScatterV2GetWorkspaceSize

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
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x1(aclTensor*)</td>
        <td>Input</td>
        <td>MM left matrix, that is, x1 in the calculation formula.</td>
        <td>The current version supports only two-dimensional input, with the shape of [m, k], and supports only the non-transposed scenario.</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2(aclTensor*)</td>
        <td>Input</td>
        <td>MM right matrix, that is, x2 in the calculation formula.</td>
        <td><ul><li>The current version supports only two-dimensional input, with the shape of [m, k], and supports both the transposed and non-transposed scenarios. </li><li>Only non-contiguous tensors are supported when two axes are transposed. In other scenarios, <a href="../../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensors</a> are not supported.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT8</td>
        <td>ND, FRACTAL_NZ</td>
        <td>2</td>
        <td>√ (only for transposition)</td>
    </tr>
    <tr>
        <td>bias(aclTensor*)</td>
        <td>Input</td>
        <td>Corresponds to <code>bias</code> in the formula.</td>
        <td><ul><li>Null pointers can be passed. </li><li>The current version supports only 1D inputs.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x1Scale(aclTensor*)</td>
        <td>Input</td>
        <td>Dequantization parameter for the left matrix of the MatMul operation.</td>
        <td><ul><li>Null pointers can be passed.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1-3</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2Scale(aclTensor*)</td>
        <td>Input</td>
        <td>mm right matrix dequantization parameter.</td>
        <td>The scenario where a null pointer is passed is supported.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1-3</td>
        <td>√ (only for transposition)</td>
    </tr>
    <tr>
        <td>quantScale(aclTensor*)</td>
        <td>Input</td>
        <td>Quantization scale of the output matrix.</td>
        <td>Currently, only the scenario where a null pointer is passed is supported.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>blockSize (int64_t)</td>
        <td>Input</td>
        <td>Quantization parameter indicating the number of quantization operations that can be performed on the M and N axes of the output matrix of the mm operation.</td>
        <td>blockSize is formed by three values: blockSizeM, blockSizeN, and blockSizeK. Each value occupies 16 bits. The calculation formula is blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32. The mm output matrix does not involve the K axis, and blockSizeK is fixed at 0. In the current version, only blockSizeM=blockSizeN=0 is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group(char*)</td>
        <td>Input</td>
        <td>Communication domain name.</td>
        <td>It is obtained through the <code>extern HcclResult HcclGetCommName(HcclComm comm, char* commName);</code> API provided by HCCL, where <code>commName</code> is the same as <code>group</code>.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp(char*)</td>
        <td>Input</td>
        <td><code>reduce</code> operation type.</td>
        <td>In the current version, only sum is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn (int64_t)</td>
        <td>Input</td>
        <td>Number of communication data splits, that is, the total data volume divided by single communication volume.</td>
        <td>The current version supports only <code>0</code>.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode (int64_t)</td>
        <td>Input</td>
        <td>Enumeration of the stream mode.</td>
        <td>Currently, only <code>1</code> is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize (int64_t)</td>
        <td>Input</td>
        <td>Indicates the number of x1 or x2 inputs that can be used for dequantization in the corresponding dimension direction of the x1Scale or x2Scale input in dequantization.</td>
        <td>The groupSize input consists of three values: groupSizeM, groupSizeN, and groupSizeK in three directions. Each value occupies 16 bits. The calculation formula is groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode(char*)</td>
        <td>Input</td>
        <td>Communication mode.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output(aclTensor*)</td>
        <td>Output</td>
        <td>Result of AllGather communication and MatMul computation, that is, output in the formula.</td>
        <td>Empty tensors are supported only when the output type is FLOAT16 or BFLOAT16.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>amaxOutOptional(aclTensor*)</td>
        <td>Output</td>
        <td>Maximum value calculated by the MM, that is, amaxOut in the formula.</td>
        <td>In the current version, only nullptr or an empty tensor is supported.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor(aclOpExecutor**)</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
        - x1 and x2: When commMode is set to aiv, the data type can be FLOAT16, BFLOAT16, or INT8. The data format of x1 supports only ND, and the data format of x2 supports ND and FRACTAL_NZ.
        - bias: When commMode is set to aiv, only nullptr is supported in the current version.
        - x1Scale: When commMode is set to aiv, the data type can be FLOAT. When the data type of x1 and x2 is FLOAT16/BFLOAT16, only `nullptr` is supported. In per-token scenarios, the shape is (m, 1).
        - x2Scale: When commMode is set to aiv, the data type can be FLOAT or INT64, and the data format is ND. The INT64 data type is only supported when the output data type is FLOAT16. When the data type of x1 and x2 is FLOAT16/BFLOAT16, only `nullptr` is supported. In per-channel scenarios, the shape is (1, n).
        - groupSize: In the current version, only 0 is supported.
        - commMode: Currently, only the aiv mode is supported. In the `aiv` mode, the AI Vector Core is used to complete communication tasks. In the current version, only "aiv" is supported.
        - output: The data type can be FLOAT16 or BFLOAT16. If the data type of x1 is FLOAT16 or BFLOAT16, the data type of output is the same as that of x1.
    - Ascend 950PR/Ascend 950DT:
        - x1 and x2: The data type can be FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2 or HIFLOAT8, and the data format is ND.
        - bias: If the data type of x1 is FLOAT16 or BFLOAT16, the data type of bias must be FLOAT16 or BFLOAT16. If the data type of x1 is FLOAT8_E4M3FN, FLOAT8_E5M2 or HIFLOAT8, the data type of bias must be FLOAT in the pertensor and mx quantization scenarios. In the perblock scenario, only nullptr is supported.
        - x1Scale: When the data types of x1 and x2 are FLOAT16 or BFLOAT16, the input can only be nullptr. In the pertensor scenario, the shape is [1]. In the perblock scenario, the shape is [ceildiv(m, 128), ceildiv(k, 128)]. In the pertensor and perblock scenarios, the data type can be FLOAT. In the mx quantization scenario, the data type is FLOAT8_E8M0 and the shape is (m, ceilDiv(k, 64), 2).
        - x2Scale: When the data types of x1 and x2 are FLOAT16 or BFLOAT16, the input can only be nullptr. In the pertensor scenario, the shape is [1]. In perblock mode, the shape is [ceildiv(k, 128), ceildiv(n, 128)]. In pertensor and perblock modes, the data type is FLOAT. In the mx scenario, the data type is FLOAT8_E8M0 and the shape is (n, ceilDiv(k, 64), 2).
        - groupSize: In perblock mode, when the x1Scale and x2Scale inputs are both 2D and the data type is FLOAT, the value of [groupSizeM, groupSizeN, groupSizeK] can only be [128, 128, 128], and the corresponding groupSize value is 549764202624. In the mx quantization scenario, when the x1Scale and x2Scale inputs are both 3D and the data type is FLOAT8_E8M0, the value of [groupSizeM, groupSizeN, groupSizeK] can only be [1, 1, 32], and the corresponding groupSize value is 4295032864. In other scenarios, only 0 is supported.
        - commMode: Only "ccu" is supported in the current version.
        - output: If the x1 type is FLOAT16 or BFLOAT16, the output type is the same as that of x1. If the x1 type is FLOAT8_E4M3FN, FLOAT8_E5M2 or HIFLOAT8, the data type can be FLOAT16, BFLOAT16, or FLOAT.

            $$
            groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
            $$

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
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
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data format or data type of the input x1, x2, output, bias (non-null scenario), x1Scale (non-null scenario), x2Scale (non-null scenario), or quantScale (non-null scenario) is not supported.</td>
    </tr>
    </tbody>
    </table>

## aclnnMatmulReduceScatterV2

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 173px">
    <col style="width: 133px">
    <col style="width: 860px">
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
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API `aclnnMatmulReduceScatterV2GetWorkspaceSize`.</td>
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

- Deterministic compute:
  - By default, aclnnMatmulReduceScatterV2 uses deterministic computing.
- Ascend 950PR/Ascend 950DT:
    - The x1 matrix cannot be transposed. The x2 matrix can be transposed or not transposed.
    - Input x1 must be 2D with shape \(m, k\). m must be an integer multiple of rank\_size.
    - Input x2 must be 2D (k, n). The axes must meet the input parameter requirements of the `mm` operator. The k axes must be equal and fall within the range of [256, 65535).
    - Input bias must be 1D (n,).
    - The output is 2D, and its shape is \(m/rank\_size, n\), where rank\_size indicates the number of devices.
    - When the data type of x1 and x2 is FLOAT16 or BFLOAT16, x1 and x2 support empty tensors. In this case, m and n can be empty, but k cannot be empty and must meet the following conditions:
        - m is empty, k is not empty, and n is not empty.
        - If m is not empty, k is not empty, and n is empty.
        - If m is empty, k is not empty, and n is empty.
    - When the data type of x1 and x2 is FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, empty tensors are not supported.
    - When the data type of x1 and x2 is FLOAT16, BFLOAT16, or HIFLOAT8, the data types of x1 and x2 must be the same.
    - When the data type of x1 and x2 is FLOAT8_E4M3FN/FLOAT8_E5M2, the data of x1 and x2 can be either of them.
    - 2, 4, 8, 16, 32, or 64 cards are supported.
    - The total size of ReduceScatter collective communication data cannot exceed 16256 MB. The total size of collective communication data is calculated as follows: m x n x sizeof(output_dtype). The internal implementation of the operator may vary according to the shape. The actual supported total communication volume may be slightly less than this value.

- <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
    - The x1 matrix cannot be transposed. The x2 matrix can be transposed or not transposed.
    - Input x1 must be 2D with shape \(m, k\). m must be an integer multiple of rank\_size.
    - Input x2 must be 2D (k, n). The axes must meet the input parameter requirements of the `mm` operator. The k axes must be equal and fall within the range of [256, 65535).
    - Input bias must be 1D (n,).
    - The output is 2D, and its shape is \(m/rank\_size, n\), where rank\_size indicates the number of devices.
    - Empty tensors are not supported.
    - The data types of x1 and x2 must be the same.
    - Two, four, and eight devices are supported.

## Calling Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

Note: This sample code calls some HCCL collective communication library APIs: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [<<HCCL API (C)>>](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclapiref/hcclcpp_07_0001.html).

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    ```c++
    #include <iostream>
    #include <vector>
    #include <thread>
    #include "hccl/hccl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_matmul_reduce_scatter_v2.h"

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
        aclrtContext context;
    };

    int LaunchOneThreadMmReduceScatterV2(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
        std::vector<int64_t> x1Shape = {1024, 256};
        std::vector<int64_t> x2Shape = {256, 512};
        std::vector<int64_t> biasShape = {512};
        std::vector<int64_t> x1ScaleShape = {1024};
        std::vector<int64_t> x2ScaleShape = {512};
        std::vector<int64_t> outShape = {1024 / DEV_NUM, 512};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x1ScaleDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;

        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1Scale = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor *quantScale = nullptr;
        aclTensor *out = nullptr;
        aclTensor *amaxOut = nullptr;

        int32_t commTurn = 0;
        int32_t streamMode = 1;
        int32_t blockSize = 0;
        int32_t groupSize = 0;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);

        std::vector<int8_t> x1HostData(x1ShapeSize, 0);
        std::vector<int8_t> x2HostData(x2ShapeSize, 0);
        std::vector<int32_t> biasHostData(biasShapeSize, 0);
        std::vector<float> x1ScaleHostData(x1ScaleShapeSize, 0);
        std::vector<float> x2ScaleHostData(x2ScaleShapeSize, 0);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnMatmulReduceScatterV2GetWorkspaceSize(
            x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, hcomName, "sum", commTurn, streamMode, groupSize, "aiv",
            out, amaxOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMatmulReduceScatterV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2 failed. ret = %d \n", ret); return ret);
        // (Fixed writing) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMatmulReduceScatterV2 execute successfully.\n", args.rankId);
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
        if (x1Scale != nullptr) {
            aclDestroyTensor(x1Scale);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (quantScale != nullptr) {
            aclDestroyTensor(quantScale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (amaxOut != nullptr) {
            aclDestroyTensor(amaxOut);
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
        if (x1ScaleDeviceAddr != nullptr) {
            aclrtFree(x1ScaleDeviceAddr);
        }
        if (x2ScaleDeviceAddr != nullptr) {
            aclrtFree(x2ScaleDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        ret = HcclCommDestroy(args.hcclComm);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
        ret = aclrtDestroyStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
        ret = aclrtResetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtDestroyContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
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
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
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
            args[rankId].context = context[rankId];
            args[rankId].stream = stream[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadMmReduceScatterV2, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

- Ascend 950PR/Ascend 950DT:

    ```c++
    #include <iostream>
    #include <vector>
    #include <thread>
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_matmul_reduce_scatter_v2.h"

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
        aclrtContext context;
    };

    int LaunchOneThreadMmReduceScatterV2(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);
        std::vector<int64_t> x1Shape = {1024, 256};
        std::vector<int64_t> x2Shape = {256, 512};
        std::vector<int64_t> biasShape = {512};
        std::vector<int64_t> x1ScaleShape = {1};
        std::vector<int64_t> x2ScaleShape = {1};
        std::vector<int64_t> outShape = {1024 / DEV_NUM, 512};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x1ScaleDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;

        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1Scale = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor *quantScale = nullptr;
        aclTensor *out = nullptr;
        aclTensor *amaxOut = nullptr;

        int32_t commTurn = 0;
        int32_t streamMode = 1;
        int32_t blockSize = 0;
        int32_t groupSize = 0;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);

        std::vector<int8_t> x1HostData(x1ShapeSize, 0);
        std::vector<int8_t> x2HostData(x2ShapeSize, 0);
        std::vector<int32_t> biasHostData(biasShapeSize, 0);
        std::vector<int32_t> x1ScaleHostData(x1ScaleShapeSize, 0);
        std::vector<int32_t> x2ScaleHostData(x2ScaleShapeSize, 0);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnMatmulReduceScatterV2GetWorkspaceSize(
            x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, hcomName, "sum", commTurn, streamMode, groupSize, "ccu",
            out, amaxOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMatmulReduceScatterV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2 failed. ret = %d \n", ret); return ret);
        // (Fixed writing) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMatmulReduceScatterV2 execute successfully.\n", args.rankId);
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
        if (x1Scale != nullptr) {
            aclDestroyTensor(x1Scale);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (quantScale != nullptr) {
            aclDestroyTensor(quantScale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (amaxOut != nullptr) {
            aclDestroyTensor(amaxOut);
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
        if (x1ScaleDeviceAddr != nullptr) {
            aclrtFree(x1ScaleDeviceAddr);
        }
        if (x2ScaleDeviceAddr != nullptr) {
            aclrtFree(x2ScaleDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        ret = HcclCommDestroy(args.hcclComm);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
        ret = aclrtDestroyStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
        ret = aclrtResetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtDestroyContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
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
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
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
            args[rankId].context = context[rankId];
            args[rankId].stream = stream[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadMmReduceScatterV2, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
