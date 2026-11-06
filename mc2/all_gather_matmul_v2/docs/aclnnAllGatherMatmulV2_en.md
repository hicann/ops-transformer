# aclnnAllGatherMatmulV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:17:10.598Z -->

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

- **Description**:

    The `aclnnAllGatherMatmulV2` API is an extension of the `aclnnAllGatherMatmul` API. In addition to supporting FLOAT16/BFLOAT16 input types for x1 and x2, it adds the following features:

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>:

        Adds support for low-precision data types FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1. Supports the pertensor, perblock, and mx [quantization modes](../../../docs/en/context/quant_mode_introduction.md).

    <!-- end id7 -->
    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:

        Adds support for the low-precision data types INT8/INT4. Supports the pertoken/perchannel [quantization modes](../../../docs/en/context/quant_mode_introduction.md).

    <!-- end id8 -->

- **Formula**:

    - Scenario 1: When the data types of x1 and x2 are FLOAT16/BFLOAT16, AllGather is performed on input parameter x1, followed by MatMul computation on x1 and x2.

    $$
    output=AllGather(x1)@x2 + bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - Scenario 2: In the pertensor scenario where the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, or in the perchannel/pertoken scenario where the data types of x1 and x2 are INT8/INT4, and amaxOut is not output, after AllGather is performed on the input parameter x1, MatMul computation is performed on x1 and x2, followed by a dequant operation.

    $$
    output=(x1Scale*x2Scale)*(AllGather(x1)@x2 + bias)
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - Scenario 3: In the perblock scenario where the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, and amaxOut is not output, when x1 is (m, k), x2 is (k, n), x1Scale is (ceilDiv(m, 128), ceilDiv(k, 128)), and x2Scale is (ceilDiv(k, 128), ceilDiv(n, 128)), after AllGather is performed on the input parameters x1 and x1Scale, perblock quantized MatMul computation is performed on x1 and x2, followed by a dequant operation.

        $$
        output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
        $$

        $$
        gatherOut=AllGather(x1)
        $$

    - Case 4: If x1 and x2 are of the data type FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E2M1 in the mx quantization scenario, x1 is (m, k), x2 is (k, n) in the non-transpose scenario and (n, k) in the transpose scenario, x1Scale is (m, ceilDiv(k, 64), 2), and x2Scale is (ceilDiv(k, 64), n, 2) in the non-transpose scenario and (n, ceilDiv(k, 64), 2) in the transpose scenario. After AllGather is performed on the input parameters x1 and x1Scale, MatMul is performed on x1 and x2, followed by the dequant operation;

        $$
        output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
        $$

        $$
        gatherOut=AllGather(x1)
        $$

## Function Prototype

Each operator is implemented as a two-phase API. First, call the `aclnnAllGatherMatmulV2GetWorkspaceSize` API to obtain the workspace size required for the computation and the executor containing the operator computation process. Then, call the `aclnnAllGatherMatmulV2` API to perform the computation.

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

- **Parameters:**
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
      <th>Parameter</th>
      <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x1 (aclTensor*)</td>
        <td>Input</td>
        <td>MM left matrix, which is x1 in the Formula.</td>
        <td>The current version supports only two-dimensional input with shape [m, k], and only the non-transpose scenario is supported.</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8, INT4</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2 (aclTensor*)</td>
        <td>Input</td>
        <td>MM right matrix, which is x2 in the Formula.</td>
        <td><ul><li>The current version supports only two-dimensional input with a shape of [k, n], in both transpose and non-transpose scenarios.</li><li>Supports only <a href="../../../docs/en/context/non_contiguous_tensor.md">[non-contiguous tensor]</a> with both axes transposed.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8, INT4</td>
        <td>ND</td>
        <td>2</td>
        <td>√ (transpose scenario only)</td>
    </tr>
    <tr>
        <td>bias (aclTensor*)</td>
        <td>Input</td>
        <td>The bias in the Formula.</td>
        <td><ul><li><term>Ascend 950PR&950DT products</term>: Supports a one-dimensional input or a null pointer.</li><li><term>Atlas A2 products</term>: The current version supports only a null pointer.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x1Scale (aclTensor*)</td>
        <td>Input</td>
        <td>Dequantization parameter for the mm left matrix.</td>
        <td>Supports passing a null pointer.</td>
        <td>FLOAT, FLOAT8_E8M0 </td>
        <td>ND</td>
        <td>1-3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2Scale (aclTensor*)</td>
        <td>Input</td>
        <td>Dequantization parameter of the right matrix in mm computation.</td>
        <td>Supports passing in a null pointer.</td>
        <td>FLOAT, INT64, FLOAT8_E8M0 </td>
        <td>ND</td>
        <td>1-3</td>
        <td>√ (only applicable to the transpose scenario)</td>
    </tr>
    <tr>
        <td>quantScale (aclTensor*)</td>
        <td>Input</td>
        <td>Quantization parameter.</td>
        <td>Currently, only a null pointer can be passed.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize (int64_t)</td>
        <td>Input</td>
        <td>Indicates the number of elements along the M and N axes of the mm output matrix that can be used for quantization in the corresponding direction.</td>
        <td>blockSize is concatenated from three values: blockSizeM, blockSizeN, and blockSizeK, each occupying 16 bits. The formula is blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32. The mm output matrix does not involve the K axis, so blockSizeK is fixed to 0. The current version supports only blockSizeM=blockSizeN=0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group (char*)</td>
        <td>Input</td>
        <td>Communicator name.</td>
        <td>Obtained by calling the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gatherIndex (int64_t)</td>
        <td>Input</td>
        <td>Identifies the gather target.</td>
        <td><ul><li>0 indicates the target is x1, and 1 indicates the target is x2.</li><li>The current version supports only input 0.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn (int64_t)</td>
        <td>Input</td>
        <td>Number of communication data splits, that is, total data volume/single communication volume.</td>
        <td>The current version supports only 0.</td>
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
        <td>Indicates how many values of the x1/x2 input along the corresponding dimension can be dequantized by a single value of the x1Scale/x2Scale input.</td>
        <td>The groupSize input is a concatenation of three values for the three directions: groupSizeM, groupSizeN, and groupSizeK, each occupying 16 bits. Formula: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode (char*)</td>
        <td>Input</td>
        <td>Communication mode.</td>
        <td>Supported values: "ai_cpu", "ccu", or "aiv"</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output (aclTensor*)</td>
        <td>Output</td>
        <td>The result of AllGather communication and MatMul computation, that is, output in the Formula.</td>
        <td><ul><li><term>Ascend 950PR&950DT products</term>: Empty tensors are supported.</li><li><term>Atlas A2 products</term>: Empty tensors are not supported.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gatherOut (aclTensor*)</td>
        <td>Output</td>
        <td>Outputs only the result after all_gather communication, that is, gatherOut in the formula.</td>
        <td><ul><li><term>Ascend 950PR&950DT products</term>: Supports empty tensors.</li><li><term>Atlas A2 products</term>: Not supported empty tensors.</li><li>The data type is the same as that of x1.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8, INT4</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>amaxOut (aclTensor*)</td>
        <td>Output</td>
        <td>Maximum value result of the MM computation, that is, amaxOut in the formula.</td>
        <td>The current version supports only nullptr or an empty tensor.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns an op executor that contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
        - x1, x2: Supports FLOAT16, BFLOAT16, INT8, and INT4 data types.
        - bias: When commMode is aiv, the current version only supports passing in nullptr.
        - x1Scale: The data type supports FLOAT. When the data types of x1 and x2 are FLOAT16/BFLOAT16, only nullptr is supported as input. In the pertoken scenario, the shape is (m, 1).
        - x2Scale: supports the FLOAT and INT64 data types. The INT64 data type is supported only when x1 and x2 are of the INT8 data type or output is of the FLOAT16 data type. When x1 and x2 are of the FLOAT16/BFLOAT16 data type, only nullptr can be input. In the perchannel use case, the shape is (1, n).
        - groupSize: The current version only supports an input of 0.
        - commMode: Currently only supports aiv mode. In aiv mode, AI VECTOR cores are used to complete communication tasks. The current version only supports passing in "aiv".
        - output: supports the FLOAT16 and BFLOAT16 data types. If x1 is of the FLOAT16 or BFLOAT16 data type, the data type of output is the same as that of x1.
        - gatherOut: data type supports FLOAT16, BFLOAT16, INT8, and INT4.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>:
        - x1, x2: support the FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1 data types.
        - bias: If x1 is of the FLOAT16 or BFLOAT16 data type, the data type of bias must be FLOAT16 or BFLOAT16. If x1 is of the FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, or FLOAT4_E2M1 data type, the data type of bias must be FLOAT in the pertensor and mx quantization scenarios. In the perblock use case, only nullptr can be input.
        - x1Scale: When the data types of x1 and x2 are FLOAT16 or BFLOAT16, only nullptr is supported as input. In the pertensor scenario, the shape is [1]. In the perblock scenario, the shape is [ceilDiv(m, 128), ceilDiv(k, 128)]. In the pertensor and perblock scenarios, the data type supports FLOAT. In the mx quantization scenario, the data type is FLOAT8_E8M0, and the shape is (m, ceilDiv(k, 64), 2).
        - x2Scale: When x1 and x2 are of the FLOAT16 or BFLOAT16 data type, only nullptr can be input. In the pertensor use case, the shape is [1]. In the perblock use case, the shape is [ceilDiv(k, 128), ceilDiv(n, 128)]. In the pertensor and perblock use cases, the data type is FLOAT. In the mx use case, the data type is FLOAT8_E8M0. The shape is (n, ceilDiv(k, 64), 2) in the transpose scenario and (ceilDiv(k, 64), n, 2) in the non-transpose scenario.
        - commMode: The current version supports only "ai_cpu" or "ccu" as the input.
        - output: If x1 is of type FLOAT16 or BFLOAT16, output is of the same type as x1. If x1 is of type FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, or FLOAT4_E2M1, the supported data types are FLOAT16, BFLOAT16, and FLOAT.
        - gatherOut: Supported data types are FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1.
        - groupSize:
            - The groupSize value is valid only when both x1Scale and x2Scale inputs are 2-dimensional or higher. In other scenarios, pass 0.
            - The groupSize value supports formula-based derivation: The passed-in groupSize is decomposed internally into groupSizeM, groupSizeN, and groupSizeK by the following formula. If one or more of them is 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the input shapes of x1/x2/x1Scale/x2Scale for computation. Reset rules: If groupSizeM=0, the quantization group value in the m direction is obtained through interface derivation, and the derivation formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the same as m in the x1 shape, and scaleM is the same as m in the x1Scale shape; if groupSizeK=0, the quantization group value in the k direction is obtained through interface derivation, and the derivation formula is groupSizeK = k / scaleK (k must be divisible by scaleK), where k is the same as k in the x1 shape, and scaleK is the same as k in the x1Scale shape; if groupSizeN=0, the quantization group value in the n direction is obtained through interface derivation, and the derivation formula is groupSizeN = n / scaleN (n must be divisible by scaleN), where n is the same as n in the x2 shape, and scaleN is the same as n in the x2Scale shape.
            $$
            groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
            $$
            - If the reset conditions are met, in the general case, when both x1Scale and x2Scale are 2D inputs of the FLOAT data type, the [groupSizeM, groupSizeN, groupSizeK] combination is derived as [128, 128, 128], corresponding to a groupSize value of 549764202624; when both x1Scale and x2Scale are 3D inputs of the FLOAT8_E8M0 data type, the [groupSizeM, groupSizeN, groupSizeK] combination is derived as [1, 1, 32], corresponding to a groupSize value of 4295032864.

    <!-- end id10 -->

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
        <td>The input x1, x2, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data types and shapes (dimensions/sizes) of the input x1, x2, x1Scale, x2Scale, bias, quantScale, output, gatherOut, or amaxOut are out of the supported range.</td>
    </tr>
    <tr>
        <td>streamMode or commMode is not in the valid range.</td>
    </tr>
    <tr>
        <td>The format of the input x1, x2, output, bias, gatherOut, or amaxOut is not ND, the format of x1Scale or x2Scale is not ND/NCL (the FRACTAL_NZ format is not supported when x2 is INT4), or x1Scale and x2Scale are null pointers in quantization scenarios, or amaxOut is not a null pointer in fp16/bf16 scenarios.</td>
    </tr>
    </tbody></table>

## aclnnAllGatherMatmulV2

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
        <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device, obtained by calling the first-phase API <code>aclnnAllGatherMatmulV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for task execution.</td>
    </tr>
    </tbody></table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnAllGatherMatmulV2` defaults to a deterministic implementation.

<!-- npu="950" id11 -->
- <term>Ascend 950PR&950DT products</term>:
    - Input x1 is 2-dimensional, and its dimensions are \(m, k\). x2 must be 2-dimensional, and its dimensions are \(k, n\). The axes must meet the input parameter requirements of the mm operator, and the k axes must be equal. The value range of the k axis is \[256, 65535\). The values of m and n must not exceed 2147483647 (INT32_MAX).
    - For the empty tensor scenario supported by x1/x2, m and n can be empty, k cannot be empty, and the following conditions must be met:
        - m is null, k is not null, and n is not null.
        - m is not null, k is not null, and n is null.
        - m is null, k is not null, and n is null.
    - bias is 1-dimensional, with shape \(n,\).
    - The output is 2-dimensional with dimensions \(m*rank\_size, n\), where rank\_size is the number of devices.
    - The gatherout is 2-dimensional with dimensions \(m*rank\_size, k\), where rank\_size is the number of devices.
    - When the data type of x1 and x2 is FLOAT16/BFLOAT16, the output data type is consistent with x1 and x2.
    - When the data type of x1 and x2 is FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1, the output data type of output can be FLOAT16, BFLOAT16, or FLOAT.
    - When the data type of x1 and x2 is FLOAT16/BFLOAT16/HIFLOAT8/FLOAT4_E2M1, the data types of x1 and x2 must be the same.
    - When the data types of x1 and x2 are FLOAT8_E4M3FN or FLOAT8_E5M2, the data types of x1 and x2 can be either one.
    - When the data types of x1 and x2 are FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN, or FLOAT8_E5M2, the x2 matrix supports both transpose and non-transpose scenarios, while the x1 matrix supports only the non-transpose scenario.
    - When the data type of x1 and x2 is FLOAT4_E2M1, the x2 matrix supports the transpose/non-transpose scenario, the x1 matrix supports only the non-transpose scenario, the k axis must be even, and the n axis must also be even when the x2 matrix is non-transposed.
    - When groupSize is set to 549764202624, bias must be empty.
    - Supports 2, 4, 8, 16, 32, or 64 cards.
    - Both the CCU communication engine and the AICPU communication engine are supported. The CCU communication engine supports interconnection only within a UB domain on a single device, and the AICPU communication engine supports interconnection across UB domains on multiple devices.
    - When the CCU communication engine is used, the total amount of allgather(x1) collective communication data in a single communicator cannot exceed 63 \* 256MB. The total amount of collective communication data is calculated as follows: m \* k \* sizeof(x1_dtype) \* number of devices. Due to different shapes, the internal implementation of the operator may vary, and the total communication volume actually supported may be slightly less than this value.

<!-- end id11 -->
<!-- npu="A3,910b" id12 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
    - Only supports transposed or non-transposed x2 matrix. The x1 matrix only supports the non-transposed scenario.
    - The input x1 must be 2-dimensional, with shape \(m, k\).
    - Input x2 must be 2-dimensional, with shape \(k, n\). The axes meet the mm operator input parameter requirements, the k axes are equal, and the k-axis value range is \([256, 65535)\).
    - bias supports only nullptr as input.
    - The output is 2-dimensional with shape \(m*rank\_size, n\), where rank\_size is the number of cards.
    - Empty tensors are not supported.
    - The data types of x1 and x2 must remain consistent.
    - When the data type of x1 and x2 is INT4, k and n must be even numbers.
    - Supports 2, 4, and 8 cards.
    - The communication buffer must be greater than or equal to 200MB.

<!-- end id12 -->

## Example

NOTE: The sample code calls some APIs of the HCCL collective communication library: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="A3,910b" id13 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

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
        // Allocate device memory based on the workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAllGatherMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherMatmulV2 failed. ret = %d \n", ret); return ret);
        // (Boilerplate) Wait synchronously for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnAllGatherMatmulV2 execute successfully.\n", args.rankId);
        // Release device resources. Modify based on the specific API definition.
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
        // Initialize the collective communication communicator.
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

<!-- end id13 -->
<!-- npu="950" id14 -->
- <term>Ascend 950PR&950DT products</term>:

    ```c++
    #include <iostream>
    #include <vector>
    #include <thread>
    #include <cstdlib>
    #include <cstring>
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
            x1, x2, bias, x1Scale, x2Scale, quantScale, blockSize, hcomName, gatherIndex, commTurn, streamMode, groupSize, COMM_MODE,
            out, gatherOut, amax, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnAllGatherMatmulV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAllGatherMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherMatmulV2 failed. ret = %d \n", ret); return ret);
        // (Boilerplate) Synchronously wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnAllGatherMatmulV2 execute successfully.\n", args.rankId);
        // Release device resources. Modify based on the specific API definition.
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
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }
        int32_t devices[DEV_NUM];
        for (int i = 0; i < DEV_NUM; i++) {
            devices[i] = i;
        }
        // Initialize the collective communication communicator.
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

<!-- end id14 -->
