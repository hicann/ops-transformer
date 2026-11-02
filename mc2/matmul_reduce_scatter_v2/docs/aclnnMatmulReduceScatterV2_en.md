# aclnnMatmulReduceScatterV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:49:33.765Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description:
    The aclnnMatmulReduceScatterV2 API is a functional extension of the aclnnMatmulReduceScatter API. In addition to supporting x1 and x2 inputs of type FLOAT16/BFLOAT16,

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>:
        - Adds support for the low-precision data types FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1. Supports pertensor, perblock, and mx [quantization modes](../../../docs/en/context/quant_mode_introduction.md). mx quantization supports the MXFP8 and MXFP4 quantization scenarios.
    <!-- end id7 -->
    <!-- npu="A3,910b" id8 -->
    - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
        - Added support for the low-precision data type INT8. Supports pertoken/perchannel [quantization modes](../../../docs/en/context/quant_mode_introduction.md).

    <!-- end id8 -->

- Formula:
    - Case 1: If the data types of x1 and x2 are FLOAT16/BFLOAT16, perform ReduceScatter communication after matmul computation on the input parameters x1, x2, and bias.

        $$
        output=ReduceScatter(x1@x2 + bias_{optional})
        $$

    - Case 2: If the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 in the pertensor scenario, or the data types of x1 and x2 are INT8 in the perchannel or pertoken scenario, and amaxOut is not output, perform ReduceScatter communication after matmul computation and dequant computation on the input parameters x1 and x2.

        $$
        output=ReduceScatter((x1Scale*x2Scale)*(x1@x2) + bias_{optional})
        $$

    - Case 3: If the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8 in the perblock scenario, and amaxOut is not output, when the shape of x1 is (m, k), the shape of x2 is (k, n), the shape of x1Scale is (ceildiv(m, 128), ceildiv(k, 128)), and the shape of x2Scale is (ceildiv(k, 128), ceildiv(n, 128)), the inputs x1 and x2 undergo matmul calculation and dequant calculation, and then ReduceScatter communication is performed.

        $$
        output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
        $$

    - Case 4: If the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E2M1 in the mx quantization scenario, and amaxOut is not output, when the shape of x1 is (m, k) and the shape of x1Scale is (m, ceildiv(k, 64), 2), the inputs x1 and x2 undergo matmul calculation and dequant calculation, and then ReduceScatter communication is performed. x2 supports the transposed/non-transposed scenarios: when x2 is transposed, the shape of x2 is (n, k) and the shape of x2Scale is (n, ceildiv(k, 64), 2); when x2 is not transposed, the shape of x2 is (k, n) and the shape of x2Scale is (ceildiv(k, 64), n, 2).

        $$
        output=ReduceScatter(\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
        $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnMatmulReduceScatterV2GetWorkspaceSize` API to obtain the workspace size required for calculation and the executor containing the operator calculation process, and then call the `aclnnMatmulReduceScatterV2` API to perform the calculation.

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
      <th>Parameter</th>
      <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-Contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x1(aclTensor*)</td>
        <td>Input</td>
        <td>MM left matrix, namely x1 in the Formula.</td>
        <td>The current version only supports two-dimensional input with shape [m, k] and only supports the non-transpose use case.</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2(aclTensor*)</td>
        <td>Input</td>
        <td>MM right matrix, namely x2 in the Formula.</td>
        <td><ul><li>The current version supports only two-dimensional input, in transpose/non-transpose scenarios. When transposed, the shape is [n, k]; when not transposed, the shape is [k, n].</li><li>Only non-contiguous tensors with both axes transposed are supported; <a href="../../../docs/en/context/non_contiguous_tensor.md">[Non-contiguous Tensor]</a> in other scenarios is not supported.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8</td>
        <td>ND, FRACTAL_NZ</td>
        <td>2</td>
        <td>√ (only applicable to transpose scenarios)</td>
    </tr>
    <tr>
        <td>bias(aclTensor*)</td>
        <td>Input</td>
        <td>The bias in the Formula.</td>
        <td><ul><li>Supports passing in a null pointer.</li><li>The current version supports only one-dimensional input.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x1Scale(aclTensor*)</td>
        <td>Input</td>
        <td>Dequantization parameter of the left matrix of mm.</td>
        <td>Supports passing in a null pointer in the scenario.</td>
        <td>FLOAT16, BFLOAT16, FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-3</td>
        <td>×</td>
    </tr>
    <tr>
        <td>x2Scale(aclTensor*)</td>
        <td>Input</td>
        <td>Dequantization parameter of the right matrix in mm.</td>
        <td>Supports passing in a null pointer. In the MX quantization scenario, x2Scale supports transposed or non-transposed input.</td>
        <td>FLOAT16, BFLOAT16, FLOAT, INT64, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-3</td>
        <td>√ (applicable only in the transpose scenario)</td>
    </tr>
    <tr>
        <td>quantScale(aclTensor*)</td>
        <td>Input</td>
        <td>Quantization scale for the output matrix.</td>
        <td>Currently, only passing in a null pointer is supported.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>blockSize (int64_t)</td>
        <td>Input</td>
        <td>Indicates how many elements along the M-axis and N-axis of the mm output matrix are used for quantization in the corresponding direction.</td>
        <td>blockSize is a concatenation of three values: blockSizeM, blockSizeN, and blockSizeK, each occupying 16 bits. The formula is blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32. The mm output matrix does not involve the K axis, so blockSizeK is fixed to 0. The current version only supports blockSizeM=blockSizeN=0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group(char*)</td>
        <td>Input</td>
        <td>Communicator name.</td>
        <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>reduceOp(char*)</td>
        <td>Input</td>
        <td>Type of the reduce operation.</td>
        <td>The current version only supports "sum".</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn (int64_t)</td>
        <td>Input</td>
        <td>Number of communication data splits, that is, total data volume/single communication volume.</td>
        <td>The current version only supports input 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode (int64_t)</td>
        <td>Input</td>
        <td>Enumeration of stream modes.</td>
        <td>Currently, only enumeration value 1 is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize (int64_t)</td>
        <td>Input</td>
        <td>Indicates the number of x1/x2 input values that a single x1Scale/x2Scale input value can dequantize along the corresponding dimension.</td>
        <td>The groupSize input is formed by concatenating three values: groupSizeM, groupSizeN, and groupSizeK in three directions. Each value occupies 16 bits. Formula: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode(char*)</td>
        <td>Input</td>
        <td>Communication mode.</td>
        <td>Supports "ai_cpu", "ccu", or "aiv".</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output(aclTensor*)</td>
        <td>Output</td>
        <td>Result of the ReduceScatter communication and MatMul calculation, that is, the output in the Formula.</td>
        <td>Empty tensors are supported only when the output type is FLOAT16 or BFLOAT16.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
    </tr>
    <tr>
        <td>amaxOutOptional(aclTensor*)</td>
        <td>Output</td>
        <td>Maximum value result of the MM calculation, that is, amaxOut in the formula.</td>
        <td>The current version only supports nullptr or an empty tensor.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
    </tr>
    <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be side-applied on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor(aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator calculation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
        - x1, x2: When commMode is aiv, the data types FLOAT16, BFLOAT16, and INT8 are supported. x1 only supports the ND data format, and x2 supports the ND and FRACTAL_NZ data formats.
        - bias: When commMode is aiv, if the data type of x1 and x2 is INT8, the data type of bias can be FLOAT16, BFLOAT16, or FLOAT; if the data type of x1 and x2 is FLOAT16, the data type of bias must be FLOAT16 or FLOAT; if the data type of x1 and x2 is BFLOAT16, the data type of bias must be BFLOAT16 or FLOAT.
        - x1Scale: When commMode is aiv, the FLOAT data type is supported. When the data types of x1 and x2 are FLOAT16 or BFLOAT16, only nullptr is supported as input. In the pertoken scenario, the shape is (m, 1).
        - x2Scale: When commMode is aiv, the data types FLOAT and INT64 are supported, and the ND data format is supported. The INT64 data type is only supported when the output data type is FLOAT16. When the data types of x1 and x2 are FLOAT16 or BFLOAT16, only nullptr is supported as input. In the perchannel scenario, the shape is (1, n).
        - groupSize: The current version only supports an input of 0.
        - commMode: Currently only supports aiv mode. In aiv mode, AI VECTOR cores are used to complete communication tasks. The current version only supports passing in "aiv".
        - output: The supported data types are FLOAT16 and BFLOAT16. If the type of x1 is FLOAT16 or BFLOAT16, the type of output is the same as that of x1.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>:
        - x1 and x2: The supported data types are FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1; only ND is supported for the data format.
        - bias: If the data type of x1 is FLOAT16 or BFLOAT16, the data type of bias must be FLOAT16 or BFLOAT16. If the data type of x1 is FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, or FLOAT4_E2M1, the data type of bias must be FLOAT in the pertensor and mx quantization scenarios. In the perblock scenario, only nullptr is supported as the input.
        - x1Scale: When the data types of x1 and x2 are FLOAT16 or BFLOAT16, only nullptr is supported as the input. In the pertensor scenario, the shape is [1]. In the perblock scenario, the shape is [ceildiv(m, 128), ceildiv(k, 128)]. In the pertensor and perblock scenarios, the supported data type is FLOAT. In the mx quantization scenario (MXFP8 and MXFP4), the data type is FLOAT8_E8M0 and the shape is (m, ceilDiv(k, 64), 2).
        - x2Scale: When the data types of x1 and x2 are FLOAT16 or BFLOAT16, only nullptr is supported as the input. In the pertensor scenario, the shape is [1]. In the perblock scenario, the shape is [ceildiv(k, 128), ceildiv(n, 128)]. In the pertensor and perblock scenarios, the supported data type is FLOAT. In the mx scenario (MXFP8 and MXFP4), the data type is FLOAT8_E8M0, the shape is (n, ceilDiv(k, 64), 2) when x2 is transposed, and (ceilDiv(k, 64), n, 2) when x2 is not transposed.
        - groupSize:
            - The groupSize value is valid only when both x1Scale and x2Scale inputs are 2-dimensional or higher. In other scenarios, pass 0.
            - The groupSize value supports formula derivation: the passed-in groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK by the following formula. If 1 or more of them is 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the input shapes of x1/x2/x1Scale/x2Scale for calculation. Setting principle: if groupSizeM=0, the quantization group value in the m direction is determined by interface derivation, and the derivation formula is groupSizeM = m / scaleM (must ensure that m is divisible by scaleM), where m is the same as m in the shape of x1, and scaleM is the same as m in the shape of x1Scale; if groupSizeK=0, the quantization group value in the k direction is determined by interface derivation, and the derivation formula is groupSizeK = k / scaleK (must ensure that k is divisible by scaleK), where k is the same as k in the shape of x1, and scaleK is the same as k in the shape of x1Scale; if groupSizeN=0, the quantization group value in the n direction is determined by interface derivation, and the derivation formula is groupSizeN = n / scaleN (must ensure that n is divisible by scaleN), where n is the same as n in the shape of x2, and scaleN is the same as n in the shape of x2Scale.
            $$
            groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
            $$
            - If the reset condition is met, in general, when both the x1Scale and x2Scale inputs are 2-dimensional and the data type is FLOAT, the value combination of [groupSizeM, groupSizeN, groupSizeK] is derived as [128, 128, 128], and the corresponding groupSize value is 549764202624; when both the x1Scale and x2Scale inputs are 3-dimensional and the data type is FLOAT8_E8M0, the value combination of [groupSizeM, groupSizeN, groupSizeK] is derived as [1, 1, 32], and the corresponding groupSize value is 4295032864.
        - commMode: The current version supports "ai_cpu" or "ccu" as the input.
        - output: If the type of x1 is FLOAT16 or BFLOAT16, the type of output is the same as that of x1. If the type of x1 is FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, or FLOAT4_E2M1, the supported data types are FLOAT16, BFLOAT16, and FLOAT.

            $$
            groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
            $$

    <!-- end id10 -->

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

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
        <td>The passed-in x1, x2, or output is a null pointer.</td>
    </tr>
    <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data format or data type of the passed-in x1, x2, output, bias (non-empty scenario), x1Scale (non-empty scenario), x2Scale (non-empty scenario), and quantScale (non-empty scenario) is not within the supported range.</td>
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
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Memory address of the workspace side-applied on the Device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace side-applied on the Device, obtained by the first-phase API <code>aclnnMatmulReduceScatterV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor that contains the operator computation process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for executing tasks.</td>
    </tr>
    </tbody></table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Communication constraints:

  <!-- npu="910b" id11 -->
  - <term>Atlas A2 products</term>: Only commMode "aiv" is supported, and the communication buffer must be at least 200MB.
  <!-- end id11 -->
  <!-- npu="A3" id12 -->
  - <term>Atlas A3 products</term>: Only commMode "aiv" is supported, and the communication buffer must be at least 200MB.
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: The current version only supports passing in commMode as "ai_cpu" or "ccu".

  <!-- end id13 -->

- Deterministic computation:
  - `aclnnMatmulReduceScatterV2` defaults to a deterministic implementation.

<!-- npu="950" id14 -->
- <term>Ascend 950PR&950DT products</term>:
    - Only supports transposed or non-transposed x2 matrix. The x1 matrix only supports the non-transposed scenario.
    - Input x1 is 2-dimensional with shape \((m, k)\), where m must be an integer multiple of the number of cards rank_size.
    - The input x2 is 2-dimensional, and its viewshape must be \(k, n\). The axes must meet the input requirements of the mm operator, the k axes must be equal, and the value range of the k axis is \[256, 65535\).
    - bias is 1-dimensional, with shape \(n,\).
    - The output is 2-dimensional, with shape \(m/rank\_size, n\), where rank\_size is the number of cards.
    - When the data types of x1 and x2 are FLOAT16/BFLOAT16, x1/x2 support the null tensor scenario, where m and n can be null, k cannot be null, and the following conditions must be met:
        - m is null, k is not null, and n is not null.
        - m is not null, k is not null, and n is null.
        - m is null, k is not null, and n is null.
    - When the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1, empty tensors are not supported.
    - When the data types of x1 and x2 are FLOAT16/BFLOAT16/HIFLOAT8/FLOAT4_E2M1, the data types of x1 and x2 must be consistent.
    - When the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2, the data types of x1 and x2 can be either of them.
    - In the mx quantization scenario, x2/x2Scale supports transposed/non-transposed input.
    - In the mx quantization scenario, when x1 and x2 are FLOAT4_E2M1 (MXFP4 quantization) inputs, k must be an even number. When x2 is a non-transpose input, n must also be an even number.
    - Supports 2, 4, 8, 16, 32, and 64 devices; supports CCU communication and AICPU communication. CCU only supports interconnection within the UB domain of a single machine, and AICPU supports interconnection across machines within the UB domain.
    - In case of CCU communication, the total data volume of ReduceScatter collective communication must not exceed 63 * 256MB. The total data volume is calculated as follows: m * n * sizeof(output_dtype). The internal implementation of the operator may vary due to shape differences. The actual supported total communication volume may be slightly less than this value.

<!-- end id14 -->
<!-- npu="A3,910b" id15 -->
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - Only supports transposed or non-transposed x2 matrix. The x1 matrix only supports the non-transposed scenario.
    - Input x1 is 2-dimensional with shape \((m, k)\), where m must be an integer multiple of the number of cards rank_size.
    - Input x2 must be 2-dimensional, with shape \(k, n\). The axes meet the mm operator input parameter requirements, the k axes are equal, and the k-axis value range is \([256, 65535)\).
    - bias is 1-dimensional, with shape \(n,\).
    - The output is 2-dimensional, with shape \(m/rank\_size, n\), where rank\_size is the number of cards.
    - Empty tensors are not supported.
    - The data types of x1 and x2 must remain consistent.
    - Supports 2, 4, and 8 cards.

<!-- end id15 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

NOTE: The sample code calls some HCCL collective communication library APIs: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="A3,910b" id16 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
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
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
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
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMatmulReduceScatterV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2 failed. ret = %d \n", ret); return ret);
        // (Boilerplate) Synchronize and wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMatmulReduceScatterV2 execute successfully.\n", args.rankId);
        // Release device resources. Modify the code according to the specific API definition.
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
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadMmReduceScatterV2, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id16 -->
<!-- npu="950" id17 -->
- <term>Ascend 950PR&950DT products</term>:

    ```c++
    #include <iostream>
    #include <vector>
    #include <thread>
    #include <cstdlib>
    #include <cstring>
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
        } while (0)

    constexpr int DEV_NUM = 2;
    constexpr const char *COMM_MODE = "ccu";

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
            strides[i] = shape[i + 1] * strides[i + 1];
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
        // Create tensors.
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
            x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, hcomName, "sum", commTurn, streamMode, groupSize, COMM_MODE,
            out, amaxOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMatmulReduceScatterV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMatmulReduceScatterV2 failed. ret = %d \n", ret); return ret);
        // (Boilerplate) Synchronously wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMatmulReduceScatterV2 execute successfully.\n", args.rankId);
        // Release device resources, which need to be modified based on the specific API definition.
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
        class EnvGuard {
        public:
            EnvGuard(const char *key, const char *val, bool enable = true)
                : key_(enable ? key : nullptr)
            {
                if (!enable) return;
                const char *old = getenv(key);
                if (old != nullptr) {
                    strncpy(saved_, old, sizeof(saved_) - 1);
                }
                setenv(key, val, 1);
            }
            ~EnvGuard()
            {
                if (key_ == nullptr) return;
                if (saved_[0] != '\0') {
                    setenv(key_, saved_, 1);
                } else {
                    unsetenv(key_);
                }
            }
        private:
            const char *key_;
            char saved_[256] = {0};
        };

        EnvGuard envGuard("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", strcmp(COMM_MODE, "ccu") == 0);

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
        // Initialize the collective communication domain.
        HcclComm comms[DEV_NUM];
        ret = HcclCommInitAll(DEV_NUM, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[DEV_NUM];
        // Start multiple threads
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

<!-- end id17 -->
