# MatmulReduceScatterV2

English | [简体中文](./README.md)

## Product Support

| Product                                                      | Supported |
| :----------------------------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- **Interface Function**:
    The aclnnMatmulReduceScatterV2 API is a functional extension of the aclnnMatmulReduceScatter API. On the basis of supporting the input types FLOAT16/BFLOAT16 for x1 and x2,
    - <term>Ascend 950PR&950DT products</term>:
        - Adds support for the low-precision data types FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1. The pertensor, perblock, and mx [quantization modes](../../docs/zh/context/quant_mode_introduction_en.md) are supported.
    - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
        - Adds support for the low-precision data type INT8. The pertoken/perchannel [quantization modes](../../docs/zh/context/quant_mode_introduction_en.md) are supported.

- **Computation Formula**:
    - Case 1: If the data types of x1 and x2 are FLOAT16/BFLOAT16, the inputs x1, x2, and bias are first processed by the matmul computation and then by the ReduceScatter communication.

        $$
        output=ReduceScatter(x1@x2 + bias_{optional})
        $$

    - Case 2: If x1 and x2 are of the data types FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 in the pertensor scenario, or of the data type INT8 in the perchannel and pertoken scenarios, and amaxOut is not output, the inputs x1 and x2 are first processed by the matmul computation and the dequant computation and then by the ReduceScatter communication.

        $$
        output=ReduceScatter((x1Scale*x2Scale)*(x1@x2) + bias_{optional})
        $$

    - Case 3: If x1 and x2 are of the data types FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 in the perblock scenario, amaxOut is not output, the shape of x1 is (m, k), the shape of x2 is (k, n), the shape of x1Scale is (ceildiv(m, 128), ceildiv(k, 128)), and the shape of x2Scale is (ceildiv(k, 128), ceildiv(n, 128)), the inputs x1 and x2 are first processed by the matmul computation and the dequant computation and then by the ReduceScatter communication.

        $$
        output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
        $$

    - Case 4: If x1 and x2 are of the data types FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E2M1 in the mx quantization scenario, amaxOut is not output, the shape of x1 is (m, k), and the shape of x1Scale is (m, ceildiv(k, 64), 2), the inputs x1 and x2 are first processed by the matmul computation and the dequant computation and then by the ReduceScatter communication. x2 supports the transpose and non-transpose scenarios: when x2 is transposed, the shape of x2 is (n, k) and the shape of x2Scale is (n, ceildiv(k, 64), 2); when x2 is not transposed, the shape of x2 is (k, n) and the shape of x2Scale is (ceildiv(k, 64), n, 2).

        $$
        output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
        $$

## Function Prototype

Each operator is divided into a two-stage API. You must first call the `aclnnMatmulReduceScatterV2GetWorkspaceSize` API to obtain the workspace size required for the computation and an executor that contains the operator computation process, and then call the `aclnnMatmulReduceScatterV2` API to perform the computation.

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

- **Parameter Description:**
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
      <th>Parameter Name</th>
      <th>Input/Output</th>
        <th>Description</th>
        <th>Usage Instructions</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimensions (shape)</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x1</td>
        <td>Input</td>
        <td>Left matrix of the MM computation, that is, x1 in the computation formula.</td>
        <td>The current version supports only two-dimensional input with the shape [m, k], and supports only the non-transpose scenario.</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Right matrix of the MM computation, that is, x2 in the computation formula.</td>
        <td><ul><li>The current version supports only two-dimensional input and supports the transpose and non-transpose scenarios. The shape is [n, k] in the transpose scenario and [k, n] in the non-transpose scenario.</li><li>Only
        <a href="../../docs/zh/context/non_contiguous_tensor_en.md">[non-contiguous tensors]</a> in the two-axis transpose case are supported; non-contiguous tensors in other scenarios are not supported.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8</td>
        <td>ND, FRACTAL_NZ</td>
        <td>2</td>
        <td>√ (applicable only in the transpose scenario)</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Input</td>
        <td>bias in the computation formula.</td>
        <td><ul><li>Passing a null pointer is supported.</li><li>The current version supports only one-dimensional input.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x1Scale</td>
        <td>Input</td>
        <td>Dequantization parameter of the left MM matrix.</td>
        <td>Passing a null pointer is supported.</td>
        <td>FLOAT16, BFLOAT16, FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-3</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2Scale</td>
        <td>Input</td>
        <td>Dequantization parameter of the right MM matrix.</td>
        <td>Passing a null pointer is supported. In the MX quantization scenario, x2Scale supports transposed and non-transposed input.</td>
        <td>FLOAT16, BFLOAT16, FLOAT, INT64, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-3</td>
        <td>√ (applicable only in the transpose scenario)</td>
    </tr>
    <tr>
        <td>quantScale</td>
        <td>Input</td>
        <td>Quantization scale of the output matrix.</td>
        <td>The current version supports only passing a null pointer.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>Input</td>
        <td>Indicates the number of elements along the M axis and the N axis of the MM output matrix that are quantized together in the corresponding direction.</td>
        <td>blockSize is composed of three values: blockSizeM, blockSizeN, and blockSizeK, and each value occupies 16 bits. The computation formula is blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32. The MM output matrix does not involve the K axis, so blockSizeK is fixed to 0. The current version supports only blockSizeM=blockSizeN=0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>Input</td>
        <td>Communicator name.</td>
        <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp</td>
        <td>Input</td>
        <td>Reduce operation type.</td>
        <td>The current version supports only "sum".</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn</td>
        <td>Input</td>
        <td>Number of communication data slices, that is, the total data volume divided by the data volume per communication.</td>
        <td>The current version supports only the input value 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode</td>
        <td>Input</td>
        <td>Enumeration of the stream mode.</td>
        <td>The current version supports only the enumerated value 1.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>Input</td>
        <td>Indicates, in dequantization, how many elements of the x1/x2 input along the corresponding dimension can be dequantized using one element of the x1Scale/x2Scale input.</td>
        <td>The groupSize input is composed of three values: groupSizeM, groupSizeN, and groupSizeK, and each value occupies 16 bits. The computation formula is groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode</td>
        <td>Input</td>
        <td>Communication mode.</td>
        <td>The input values "ai_cpu", "ccu", and "aiv" are supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output</td>
        <td>Output</td>
        <td>Result of the ReduceScatter communication and the Matmul computation, that is, output in the computation formula.</td>
        <td>Empty tensors are supported only when the output type is FLOAT16 or BFLOAT16.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>amaxOutOptional</td>
        <td>Output</td>
        <td>Maximum value result of the MM computation, that is, amaxOut in the formula.</td>
        <td>The current version supports only nullptr or an empty tensor.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace that needs to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
        - x1 and x2: When commMode is aicpu, the supported data types are FLOAT16 and BFLOAT16. When commMode is aiv, the supported data types are FLOAT16, BFLOAT16, and INT8. The data format of x1 supports only ND, and the data formats of x2 support ND and FRACTAL_NZ.
        - bias: When commMode is aicpu, the supported data types are FLOAT16 and BFLOAT16, and only the input value 0 is supported. When commMode is aiv, if the data types of x1 and x2 are INT8, the data type of bias can be FLOAT16, BFLOAT16, or FLOAT; if the data types of x1 and x2 are FLOAT16, the data type of bias must be FLOAT16 or FLOAT; if the data types of x1 and x2 are BFLOAT16, the data type of bias must be BFLOAT16 or FLOAT.
        - x1Scale: When commMode is aicpu, only the input nullptr is supported. When commMode is aiv, the supported data type is FLOAT. When the data types of x1 and x2 are FLOAT16/BFLOAT16, only the input nullptr is supported. In the pertoken scenario, the shape is (m, 1).
        - x2Scale: When commMode is aicpu, only the input nullptr is supported. When commMode is aiv, the supported data types are FLOAT and INT64, and the supported data format is ND. The INT64 data type is supported only when the data type of output is FLOAT16. When the data types of x1 and x2 are FLOAT16 or BFLOAT16, only the input nullptr is supported. In the perchannel scenario, the shape is (1, n).
        - groupSize: The current version supports only the input value 0.
        - commMode: Only the aiv mode is supported currently. In aiv mode, AI VECTOR cores are used to complete communication tasks. The current version supports only the input value "aiv".
        - output: The supported data types are FLOAT16 and BFLOAT16. If the type of x1 is FLOAT16 or BFLOAT16, the type of output is the same as that of x1.
    - <term>Ascend 950PR&950DT products</term>:
        - x1 and x2: The supported data types are FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1, and the supported data format is only ND.
        - bias: If the data type of x1 is FLOAT16 or BFLOAT16, the data type of bias must be FLOAT16 or BFLOAT16. If the data type of x1 is FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, or FLOAT4_E2M1, the data type of bias must be FLOAT in the pertensor and mx quantization scenarios. In the perblock scenario, only the input nullptr is supported.
        - x1Scale: When the data types of x1 and x2 are FLOAT16 or BFLOAT16, only the input nullptr is supported. In the pertensor scenario, the shape is [1]. In the perblock scenario, the shape is [ceildiv(m, 128), ceildiv(k, 128)]. In the pertensor and perblock scenarios, the supported data type is FLOAT. In the mx quantization scenario, the data type is FLOAT8_E8M0 and the shape is (m, ceilDiv(k, 64), 2).
        - x2Scale: When the data types of x1 and x2 are FLOAT16 or BFLOAT16, only the input nullptr is supported. In the pertensor scenario, the shape is [1]. In the perblock scenario, the shape is [ceildiv(k, 128), ceildiv(n, 128)]. In the pertensor and perblock scenarios, the supported data type is FLOAT. In the mx scenario, the data type is FLOAT8_E8M0. When x2 is transposed, the shape is (n, ceilDiv(k, 64), 2); when x2 is not transposed, the shape is (ceilDiv(k, 64), n, 2).
        - groupSize:
            - The value of groupSize is valid only when both the x1Scale and x2Scale inputs are two-dimensional or higher; in other scenarios, pass 0.
            - The value of groupSize can be derived by formula: The input groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. If one or more of them are 0, groupSizeM, groupSizeN, and groupSizeK used for computation are reset based on the shapes of the x1/x2/x1Scale/x2Scale inputs. Reset principle: If groupSizeM=0, the quantization group value along the m direction is derived by the API, and the derivation formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the same as m in the shape of x1 and scaleM is the same as m in the shape of x1Scale; if groupSizeK=0, the quantization group value along the k direction is derived by the API, and the derivation formula is groupSizeK = k / scaleK (k must be divisible by scaleK), where k is the same as k in the shape of x1 and scaleK is the same as k in the shape of x1Scale; if groupSizeN=0, the quantization group value along the n direction is derived by the API, and the derivation formula is groupSizeN = n / scaleN (n must be divisible by scaleN), where n is the same as n in the shape of x2 and scaleN is the same as n in the shape of x2Scale.
            $$
            groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
            $$
            - If the reset conditions are met, in general cases, when both the x1Scale and x2Scale inputs are two-dimensional and their data types are both FLOAT, the combination of [groupSizeM, groupSizeN, groupSizeK] values is derived as [128, 128, 128], corresponding to the groupSize value 549764202624; when both the x1Scale and x2Scale inputs are three-dimensional and their data types are both FLOAT8_E8M0, the combination of [groupSizeM, groupSizeN, groupSizeK] values is derived as [1, 1, 32], corresponding to the groupSize value 4295032864.
        - commMode: The current version supports the input values "ai_cpu" or "ccu".
        - output: If the type of x1 is FLOAT16 or BFLOAT16, the type of output is the same as that of x1. If the type of x1 is FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, or FLOAT4_E2M1, the supported data types are FLOAT16, BFLOAT16, and FLOAT.

            $$
            groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
            $$

- **Return Values**

    Returns the aclnnStatus status code. For details, see [aclnn Return Codes](../../docs/zh/context/aclnn_return_code_en.md).

    The first-stage API performs input parameter validation and reports an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
    </colgroup>
    <thead>
    <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x1, x2, or output is a null pointer.</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data format or data type of the input x1, x2, output, bias (non-null scenario), x1Scale (non-null scenario), x2Scale (non-null scenario), or quantScale (non-null scenario) is not within the supported range.</td>
    </tr>
    </tbody>
    </table>

## aclnnMatmulReduceScatterV2

- **Parameter Description:**

    <table style="undefined;table-layout: fixed; width: 1166px"> <colgroup>
    <col style="width: 173px">
    <col style="width: 133px">
    <col style="width: 860px">
    </colgroup>
    <thead>
    <tr>
    <th>Parameter Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Memory address of the workspace allocated on the Device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the Device, obtained by the first-stage API <code>aclnnMatmulReduceScatterV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream on which tasks are executed.</td>
    </tr>
    </tbody></table>

- **Return Values**

    Returns the aclnnStatus status code. For details, see [aclnn Return Codes](../../docs/zh/context/aclnn_return_code_en.md).

## Constraints

- Communication engine constraints:
  - <term>Atlas A2 products</term>: Only the commMode value "aiv" is supported.
  - <term>Atlas A3 products</term>: Only the commMode value "aiv" is supported.
  - <term>Ascend 950PR&950DT products</term>: The current version supports the commMode input values "ai_cpu" or "ccu".

- Deterministic computation:
  - `aclnnMatmulReduceScatterV2` uses a deterministic computation implementation by default.
- <term>Ascend 950PR&950DT products</term>:
    - Only the x2 matrix supports the transpose and non-transpose scenarios, and the x1 matrix supports only the non-transpose scenario.
    - The input x1 is two-dimensional, its shape is \(m, k\), and m must be an integer multiple of rank\_size (the number of cards).
    - The input x2 is two-dimensional, its viewshape must be \(k, n\), the axes meet the input parameter requirements of the mm operator, the k axes are equal, and the value range of the k axis is \[256, 65535\).
    - bias is one-dimensional and its shape is \(n,\).
    - The output is two-dimensional, its shape is \(m/rank\_size, n\), and rank\_size is the number of cards.
    - When the data types of x1 and x2 are FLOAT16/BFLOAT16, in the empty tensor scenarios supported by x1/x2, m and n can be empty, k cannot be empty, and the following conditions must be met:
        - m is empty, k is not empty, and n is not empty;
        - m is not empty, k is not empty, and n is empty;
        - m is empty, k is not empty, and n is empty.
    - When the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1, empty tensors are not supported.
    - When the data types of x1 and x2 are FLOAT16/BFLOAT16/HIFLOAT8/FLOAT4_E2M1, the data types of x1 and x2 must be the same.
    - When the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2, the data types of x1 and x2 can be any one of the two.
    - In the mx quantization scenario, x2/x2Scale supports transposed and non-transposed input.
    - In the mx quantization scenario, when the x1 and x2 inputs are FLOAT4_E2M1 (MXFP4 quantization), k must be an even number, and n must also be an even number when x2 is input without transpose.
    - Supports 2, 4, 8, 16, 32, and 64 cards; supports CCU communication and AICPU communication. CCU supports only the interconnection within a UB domain on a single machine, and AICPU supports the interconnection within a UB domain across machines.
    - Under CCU communication, the total data volume of the ReduceScatter collective communication cannot exceed 63 * 256MB. The total data volume is calculated as follows: m * n * sizeof(output_dtype). Because the internal implementation of the operator may vary with the shape, the actually supported total communication volume may be slightly smaller than this value.

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
    - Only the x2 matrix supports the transpose and non-transpose scenarios, and the x1 matrix supports only the non-transpose scenario.
    - The input x1 is two-dimensional, its shape is \(m, k\), and m must be an integer multiple of rank\_size (the number of cards).
    - The input x2 must be two-dimensional, its shape is \(k, n\), the axes meet the input parameter requirements of the mm operator, the k axes are equal, and the value range of the k axis is \[256, 65535\).
    - bias is one-dimensional and its shape is \(n,\).
    - The output is two-dimensional, its shape is \(m/rank\_size, n\), and rank\_size is the number of cards.
    - Empty tensors are not supported.
    - The data types of x1 and x2 must be the same.
    - Supports 2, 4, and 8 cards.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_matmul_reduce_scatter_v2.cpp](./examples/test_aclnn_matmul_reduce_scatter_v2.cpp) | Calls the MatmulReduceScatterV2 operator through the [aclnnMatmulReduceScatterV2](docs/aclnnMatmulReduceScatterV2_en.md) API. |
