# aclnnAllGatherMatmulV2

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/mc2/all_gather_matmul_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------:|
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Description

- **API function**:

  The `aclnnAllGatherMatmulV2` API extends the functions of the `aclnnAllGatherMatmul` API. In addition to supporting the x1 and x2 input types of FLOAT16/BFLOAT16, the following functions are added:

  - Ascend 950PR/Ascend 950DT:

    The low-precision data type FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 is supported. It supports pertensor, perblock, and mx [quantization modes](../../../docs/en/context/quant_mode_introduction.md).

  - <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>:

    The low-precision data types INT8 and INT4 are supported. It supports per-token/per-channel [quantization methods](../../../docs/en/context/quant_mode_introduction.md).

- **Formula**:

  - Case 1: If the data types of x1 and x2 are FLOAT16 or BFLOAT16, MatMul is performed on x1 and x2 after AllGather is performed on x1.

    $$
    output=AllGather(x1)@x2 + bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

  - Case 2: If the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 in the pertensor scenario, or INT8/INT4 in the perchannel or pertoken scenario, and amaxOut is not output, MatMul is performed on x1 and x2 after AllGather is performed on x1, and then dequant is performed.

    $$
    output=(x1Scale*x2Scale)*(AllGather(x1)@x2 + bias)
    $$

    $$
    gatherOut=AllGather(x1)
    $$

  - Case 3: If the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 in the perblock scenario, and amaxOut is not output, when x1 is (m, k) and x2 is (k, n), x1Scale is (ceilDiv(m, 128), ceilDiv(k, 128)) and x2Scale is (ceilDiv(k, 128), ceilDiv(n, 128)). After AllGather is performed on x1 and x1Scale, perblock quantized MatMul is performed on x1 and x2, and then dequant is performed.

    $$
    output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
    $$

    $$
    gatherOut=AllGather(x1)
    $$

  - Case 4: If the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2 in the mx quantization scenario, x1 is (a0, a1), x2 is (b0, b1), x1Scale is (a0, ceilDiv(a1, 64), 2), and x2Scale is (b0, ceilDiv(b1, 64), 2), after AllGather is performed on x1 and x1Scale, MatMul is performed on x1 and x2, and then dequant is performed.

    $$
    output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
    $$

    $$
    gatherOut=AllGather(x1)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAllGatherMatmulV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, `aclnnAllGatherMatmulV2` is called to perform computation.

```cpp
aclnnStatus aclnnAllGatherMatmulV2GetWorkspaceSize(
    const aclTensor* x1, 
    const aclTensor* x2, 
    const aclTensor* bias, 
    const aclTensor* x1Scale, 
    const aclTensor* x2Scale, 
    const aclTensor* quantScale, 
    int64_t          blockSize, 
    const char*      group, 
    int64_t          gatherIndex, 
    int64_t          commTurn, 
    int64_t          streamMode, 
    int64_t          groupSize, 
    const char*      commMode, 
    aclTensor*       output, 
    aclTensor*       gatherOut, 
    aclTensor*       amaxOut, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnAllGatherMatmulV2(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
```

## aclnnAllGatherMatmulV2GetWorkspaceSize

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1607px"><colgroup>
    <col style="width: 190px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 120px">
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
        <td>x1 (aclTensor*)</td>
        <td>Input</td>
        <td>MM left matrix, that is, x1 in the calculation formula.</td>
        <td>The current version supports only two-dimensional input, with the shape being [m, k], and only the non-transposition scenario is supported.</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT8, INT4</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2 (aclTensor*)</td>
        <td>Input</td>
        <td>MM right matrix, that is, x2 in the calculation formula.</td>
        <td><ul><li>The current version supports only two-dimensional input, with the shape being [k, n]. Both transposed and non-transposed scenarios are supported. </li><li><a href="../../../docs/en/context/non_contiguous_tensor.md">Non-contiguous tensors</a> are supported only when two axes are transposed.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT8, INT4</td>
        <td>ND</td>
        <td>2</td>
        <td>√ (only for transposition)</td>
    </tr>
    <tr>
        <td>bias (aclTensor*)</td>
        <td>Input</td>
        <td> Corresponds to `bias` in the formula.</td>
        <td><ul><li>Ascend 950PR/Ascend 950DT: A one-dimensional input or null pointer can be passed. </li><li><term>Atlas A2 training products/Atlas A2 inference products</term>: Only null pointers can be passed in the current version.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x1Scale (aclTensor*)</td>
        <td>Input</td>
        <td>mm left matrix dequantization parameter.</td>
        <td>The scenario where a null pointer is passed is supported.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1-3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2Scale (aclTensor*)</td>
        <td>Input</td>
        <td>mm right matrix dequantization parameter.</td>
        <td>The scenario where a null pointer is passed is supported.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1-3</td>
        <td>√ (only for transposition)</td>
    </tr>
    <tr>
        <td>quantScale (aclTensor*)</td>
        <td>Input</td>
        <td> Corresponds to `bias` in the formula.</td>
        <td>Currently, only the scenario where a null pointer is passed is supported.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize (int64_t) </td>
        <td>Input</td>
        <td>Quantization number of the mm output matrix in the M axis and N axis directions. This parameter indicates the number of quantization values in the corresponding direction.</td>
        <td>blockSize is composed of blockSizeM, blockSizeN, and blockSizeK. Each value occupies 16 bits. The calculation formula is blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32. The output matrix of the mm does not involve the K axis, and blockSizeK is fixed at 0. In the current version, only blockSizeM=blockSizeN=0 is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group (char*)</td>
        <td>Input</td>
        <td>Communication domain name.</td>
        <td>It is obtained through the `extern HcclResult HcclGetCommName(HcclComm comm, char* commName);` API provided by HCCL, where `commName` is the same as `group`.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gatherIndex (int64_t) </td>
        <td>Input</td>
        <td>Gather target identifier.</td>
        <td><ul><li>0 indicates that the target is x1, and 1 indicates that the target is x2. </li><li>The current version supports only 0.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn (int64_t) </td>
        <td>Input</td>
        <td>Number of communication data splits, that is, the total data volume divided by single communication volume.</td>
        <td>The current version supports only `0`.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode (int64_t) </td>
        <td>Input</td>
        <td>Enumeration of the stream mode.</td>
        <td>Currently, only `1` is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize (int64_t) </td>
        <td>Input</td>
        <td>Number of x1 or x2 inputs that can be used for dequantization in the corresponding dimension direction of the x1Scale or x2Scale input in dequantization.</td>
        <td>The groupSize input consists of three values: groupSizeM, groupSizeN, and groupSizeK in three directions. Each value occupies 16 bits. The calculation formula is groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode (char*)</td>
        <td>Input</td>
        <td>Communication mode.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output (aclTensor*)</td>
        <td>Output</td>
        <td>Result of AllGather communication and MatMul computation, that is, output in the formula.</td>
        <td><ul><li>Ascend 950PR/Ascend 950DT: Empty tensors are supported. </li><li><term>Atlas A2 training products/Atlas A2 inference products</term>: Empty tensors are not supported.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gatherOut (aclTensor*)</td>
        <td>Output</td>
        <td>Only the result after AllGather communication is output. corresponding to gatherOut in the formula.</td>
        <td><ul><li>Ascend 950PR/Ascend 950DT: Empty tensors are supported. </li><li><term>Atlas A2 training products/Atlas A2 inference products</term>: Empty tensors are not supported. </li><li>The data type must be the same as that of x1.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT8, INT4</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>amaxOut (aclTensor*)</td>
        <td>Output</td>
        <td>Maximum value calculated by the MM, that is, amaxOut in the formula.</td>
        <td>In the current version, only nullptr or empty tensors are supported.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize (uint64_t*) </td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor (aclOpExecutor**)</td>
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
        - x1 and x2: The data type can be FLOAT16, BFLOAT16, INT8, or INT4.
        - bias: When commMode is set to aiv, the current version supports only nullptr as the input.
        - x1Scale: The data type can be FLOAT. When the data type of x1 and x2 is FLOAT16/BFLOAT16, only `nullptr` is supported. In per-token scenarios, the shape is (m, 1).
        - x2Scale: The data type can be FLOAT or INT64. The INT64 data type is supported only when the data type of x1 and x2 is INT4 or the data type of output is FLOAT16. When the data type of x1 and x2 is FLOAT16/BFLOAT16, only `nullptr` is supported. In per-channel scenarios, the shape is (1, n).
        - groupSize: In the current version, only 0 is supported.
        - commMode: Currently, only the aiv mode is supported. In the `aiv` mode, the AI Vector Core is used to complete communication tasks. In the current version, only "aiv" is supported.
        - output: The data type can be FLOAT16 or BFLOAT16. If the data type of x1 is FLOAT16 or BFLOAT16, the data type of output is the same as that of x1.
        - gatherOut: The data type can be FLOAT16, BFLOAT16, INT8, or INT4.
    - Ascend 950PR/Ascend 950DT:
        - x1 and x2: The data type can be FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2 or HIFLOAT8.
        - bias: If the data type of x1 is FLOAT16 or BFLOAT16, the data type of bias must be FLOAT16 or BFLOAT16. If the data type of x1 is FLOAT8_E4M3FN, FLOAT8_E5M2 or HIFLOAT8, the data type of bias must be FLOAT in the pertensor and mx quantization scenarios. In the perblock scenario, only nullptr can be input.
        - x1Scale: When the data type of x1 and x2 is FLOAT16 or BFLOAT16, only nullptr can be input. In the pertensor scenario, the shape is [1]. In perblock mode, the shape is [ceilDiv(m, 128), ceilDiv(k, 128)]. In pertensor and perblock modes, the data type is FLOAT. In the mx quantization scenario, the data type is FLOAT8_E8M0 and the shape is (m, ceilDiv(k, 64), 2).
        - x2Scale: When the data types of x1 and x2 are FLOAT16 or BFLOAT16, only nullptr is supported. In pertensor mode, the shape is [1]. In perblock mode, the shape is [ceilDiv(k, 128), ceilDiv(n, 128)]. In pertensor and perblock modes, the data type is FLOAT. In the mx scenario, the data type is FLOAT8_E8M0 and the shape is (ceilDiv(k, 64), n, 2). Only the transpose scenario is supported.
        - groupSize: When both x1Scale and x2Scale are 2D inputs and their data types are FLOAT, the value of groupSize must be 549764202624, which corresponds to the value of [groupSizeM, groupSizeN, groupSizeK] being [128, 128, 128]. When both x1Scale and x2Scale are 3D inputs and their data types are FLOAT8_E8M0, the value of groupSize must be 4295032864, which corresponds to the value of [groupSizeM, groupSizeN, groupSizeK] being [1, 1, 32]. In other scenarios, only 0 is supported.
        - commMode: Only "ccu" is supported in the current version.
        - output: If the type of x1 is FLOAT16 or BFLOAT16, the type of output is the same as that of x1. If the type of x1 is FLOAT8_E4M3FN, FLOAT8_E5M2 or HIFLOAT8, the data type can be FLOAT16, BFLOAT16, or FLOAT.
        - gatherOut: The data type can be FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2 or HIFLOAT8.
        - Restrictions on groupSize:
            - The groupSize value is valid only when x1Scale and x2Scale are 2D or higher-dimensional inputs. In other scenarios, the input must be 0.
            - The input groupSize is decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formulas. If one or more of them are 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the input shape of x1/x2/x1Scale/x2Scale for calculation. Principle: If groupSizeM is 0, the quantization group size in the m direction is inferred by the API. The inference formula is groupSizeM = m/scaleM (m must be exactly divided by scaleM). m is the same as that in the x1 shape, and scaleM is the same as that in the x1Scale shape.

            $$
            groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
            $$

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
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
        <td>The input `x1`, `x2`, or `output` is passed as a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type and dimension of the input x1, x2, x1Scale, x2Scale, bias, quantScale, output, gatherOut, or amaxOut are not supported.</td>
    </tr>
    </tbody></table>

## aclnnAllGatherMatmulV2

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
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
        <td>Size of the workspace allocated on the device, which is obtained by the first API `aclnnAllGatherMatmulV2GetWorkspaceSize`.</td>
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

- Deterministic computation:
  - The default deterministic implementation of `aclnnAllGatherMatmulV2` is used.
- Ascend 950PR/Ascend 950DT:
  - The input **x1** is 2-dimensional, and its dimension is (m, k). x2 must be 2-dimensional, with the shape of (k, n). The axis meets the input parameter requirements of the mm operator, the k axis is equal, and the value range of the k axis is [256, 65535).
  - Input `bias` must be 1D (n,).
  - The output is 2-dimensional, with the shape of (m x rank_size, n), where rank_size indicates the number of devices.
  - The gatherout is 2-dimensional, with the shape of (m x rank_size, k), where rank_size indicates the number of devices.
  - When the data type of x1 and x2 is FLOAT16 or BFLOAT16, the output data type is the same as that of x1 and x2.
  - When the data type of x1 and x2 is FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, the output data type can be FLOAT16, BFLOAT16, or FLOAT.
  - When the data type of x1 and x2 is FLOAT16, BFLOAT16, or HIFLOAT8, the data types of x1 and x2 must be the same.
  - When the data type of x1 and x2 is FLOAT8_E4M3FN/FLOAT8_E5M2, the data type of x1 and x2 can be either of them.
  - When the data type of x1 and x2 is FLOAT16/BFLOAT16/HIFLOAT8/FLOAT8_E4M3FN/FLOAT8_E5M2, x2 can be transposed or not, and x1 can only be not transposed.
  - When groupSize is set to 549764202624, bias must be empty.
  - 2, 4, 8, 16, 32, or 64 devices are supported.
  - The total size of the AllGather (x1) collective communication data cannot exceed 16 x 256 MB. The total size of the collective communication data is calculated as follows: m x k x sizeof(x1_dtype) x Number of devices. The internal implementation of the operator may vary according to the shape. Therefore, the actual supported total communication volume may be slightly less than this value.

- <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>:
  - The `x1` matrix cannot be transposed. The `x2` matrix can be transposed or not transposed.
  - Input `x1` is 2D (m, k).
  - Input `x2` must be 2D (k, n). The axes must meet the input parameter requirements of the MatMul operator. The k axes of `x1` and `x2` must be equal and fall within the range of [256, 65535).
  - The bias supports only nullptr as the input.
  - The output is 2D (m*rank_size, n). rank_size indicates the number of devices.
  - Empty tensors are not supported.
  - The data types of `x1` and `x2` must be the same.
  - When the data types of x1 and x2 are INT4, k and n must be even numbers.
  - Two, four, and eight devices are supported.

## Example

Note: This sample code calls some HCCL collective communication library APIs, including HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [<<HCCL API (C)>>](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclapiref/hcclcpp_07_0001.html).

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term>

    ```c++
    #include <iostream>
    #include <vector>
    #include <thread>
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_all_gather_matmul_v2.h"

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

    int LaunchOneThreadAllGatherMmV2(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p, context = %p\n", args.rankId, hcomName,
        args.stream, args.context);
        std::vector<int64_t> x1Shape = {32, 256};
        std::vector<int64_t> x2Shape = {256, 128};
        std::vector<int64_t> x1ScaleShape = {32, 1};
        std::vector<int64_t> x2ScaleShape = {1, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> outShape = {32 * DEV_NUM, 128};
        std::vector<int64_t> gatherOutShape = {32 * DEV_NUM, 256};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *x1ScaleDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *gatherOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *x1Scale = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *quantScale = nullptr;
        aclTensor *out = nullptr;
        aclTensor *gatherOut = nullptr;
        aclTensor *amax = nullptr;

        int64_t gatherIndex = 0;
        int64_t commTurn = 0;
        int64_t streamMode = 1;
        int64_t blockSize = 0;
        int64_t groupSize = 0;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long gatherOutShapeSize = GetShapeSize(gatherOutShape);

        std::vector<int8_t> x1HostData(x1ShapeSize, 0);
        std::vector<int8_t> x2HostData(x2ShapeSize, 0);
        std::vector<int32_t> x1ScaleHostData(x1ScaleShapeSize, 0);
        std::vector<int32_t> x2ScaleHostData(x2ScaleShapeSize, 0);
        std::vector<int32_t> biasHostData(biasShapeSize, 0);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        std::vector<int8_t> gatherOutHostData(gatherOutShapeSize, 0);

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
        ret = CreateAclTensor(gatherOutHostData, gatherOutShape, &gatherOutDeviceAddr,
                              aclDataType::ACL_INT8, &gatherOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnAllGatherMatmulV2GetWorkspaceSize(
            x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, hcomName, gatherIndex, commTurn, streamMode, groupSize, "aiv",
            out, gatherOut, amax, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnAllGatherMatmulV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAllGatherMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherMatmulV2 failed. ret = %d \n", ret); return ret);
        // (Fixed writing) Synchronously wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnAllGatherMatmulV2 execute successfully.\n", args.rankId);
        // Release device resources. Modify the configuration based on the API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (x1Scale != nullptr) {
            aclDestroyTensor(x1Scale);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (quantScale != nullptr) {
            aclDestroyTensor(quantScale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (gatherOut != nullptr) {
            aclDestroyTensor(gatherOut);
        }
        if (amax != nullptr) {
            aclDestroyTensor(amax);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (x1ScaleDeviceAddr != nullptr) {
            aclrtFree(x1ScaleDeviceAddr);
        }
        if (x2ScaleDeviceAddr != nullptr) {
            aclrtFree(x2ScaleDeviceAddr);
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
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadAllGatherMmV2, std::ref(args[rankId])));
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
    #include "aclnnop/aclnn_all_gather_matmul_v2.h"

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

    int LaunchOneThreadAllGatherMmV2(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p, context = %p\n", args.rankId, hcomName,
            args.stream, args.context);
        std::vector<int64_t> x1Shape = {32, 256};
        std::vector<int64_t> x2Shape = {256, 128};
        std::vector<int64_t> x1ScaleShape = {1};
        std::vector<int64_t> x2ScaleShape = {1};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> outShape = {32 * DEV_NUM, 128};
        std::vector<int64_t> gatherOutShape = {32 * DEV_NUM, 256};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *x1ScaleDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *gatherOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *x1Scale = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *quantScale = nullptr;
        aclTensor *out = nullptr;
        aclTensor *gatherOut = nullptr;
        aclTensor *amax = nullptr;

        int64_t gatherIndex = 0;
        int64_t commTurn = 0;
        int64_t streamMode = 1;
        int64_t blockSize = 0;
        int64_t groupSize = 0;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long gatherOutShapeSize = GetShapeSize(gatherOutShape);

        std::vector<int8_t> x1HostData(x1ShapeSize, 0);
        std::vector<int8_t> x2HostData(x2ShapeSize, 0);
        std::vector<int32_t> x1ScaleHostData(x1ScaleShapeSize, 0);
        std::vector<int32_t> x2ScaleHostData(x2ScaleShapeSize, 0);
        std::vector<int32_t> biasHostData(biasShapeSize, 0);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        std::vector<int8_t> gatherOutHostData(gatherOutShapeSize, 0);

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
        ret = CreateAclTensor(gatherOutHostData, gatherOutShape, &gatherOutDeviceAddr,
                            aclDataType::ACL_FLOAT8_E4M3FN, &gatherOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnAllGatherMatmulV2GetWorkspaceSize(
            x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, hcomName, gatherIndex, commTurn, streamMode, groupSize, "ccu",
            out, gatherOut, amax, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnAllGatherMatmulV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the computed workspaceSize.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAllGatherMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherMatmulV2 failed. ret = %d \n", ret); return ret);
        // (Fixed writing) Synchronously wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnAllGatherMatmulV2 execute successfully.\n", args.rankId);
        // Release device resources. Modify the configuration based on the API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (x1Scale != nullptr) {
            aclDestroyTensor(x1Scale);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (quantScale != nullptr) {
            aclDestroyTensor(quantScale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (gatherOut != nullptr) {
            aclDestroyTensor(gatherOut);
        }
        if (amax != nullptr) {
            aclDestroyTensor(amax);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (x1ScaleDeviceAddr != nullptr) {
            aclrtFree(x1ScaleDeviceAddr);
        }
        if (x2ScaleDeviceAddr != nullptr) {
            aclrtFree(x2ScaleDeviceAddr);
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
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadAllGatherMmV2, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```
