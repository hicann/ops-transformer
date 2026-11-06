# AllGatherMatmulV2

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>   | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | x |
| <term>Atlas inference products</term> | x |
| <term>Atlas training products</term> | x |

## Functional Description

- **Operator function**:
  Fuses AllGather communication with MatMul computation. In addition to supporting FLOAT16/BFLOAT16 input types for x1 and x2, low-precision data types are also supported. In this case, the operator performs the corresponding dequantization computation after Matmul computation. The supported low-precision data types and quantization methods are as follows:

  - <term>Ascend 950PR&950DT products</term>:

    Adds support for the low-precision data types FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8/FLOAT4_E2M1. The pertensor, perblock, and mx [quantization methods](../../docs/zh/context/quant_mode_introduction_en.md) are supported.

  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    Adds support for the low-precision data types INT8/INT4. The pertoken/perchannel [quantization methods](../../docs/zh/context/quant_mode_introduction_en.md) are supported.

- **Computation formula**:
    - Case 1: When the data types of x1 and x2 are FLOAT16/BFLOAT16, after the input x1 undergoes AllGather, MatMul is performed on x1 and x2.

    $$
    output=AllGather(x1)@x2 + bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - Case 2: In the pertensor scenario where the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, or in the perchannel and pertoken scenarios where the data types of x1 and x2 are INT8/INT4, if amaxOut is not output, after the input x1 undergoes AllGather, MatMul is performed on x1 and x2, and then the dequant operation is performed.

    $$
    output=(x1Scale*x2Scale)*(AllGather(x1)@x2 + bias)
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - Case 3: In the perblock scenario where the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, if amaxOut is not output, when x1 is (m, k) and x2 is (k, n), with x1Scale being (ceildiv(m, 128), ceildiv(k, 128)) and x2Scale being (ceildiv(k, 128), ceildiv(n, 128)), after the inputs x1 and x1Scale undergo AllGather, perblock quantized MatMul is performed on x1 and x2, and then the dequant operation is performed.

        $$
        output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
        $$

        $$
        gatherOut=AllGather(x1)
        $$

    - Case 4: In the mx quantization scenario where the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E2M1, with x1 being (m, k), x2 being (k, n) in the non-transposed scenario and (n, k) in the transposed scenario, x1Scale being (m, ceilDiv(k, 64), 2), and x2Scale being (ceilDiv(k, 64), n, 2) in the non-transposed scenario and (n, ceilDiv(k, 64), 2) in the transposed scenario, after the inputs x1 and x1Scale undergo AllGather, MatMul is performed on x1 and x2, and then the dequant operation is performed;

        $$
        output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
        $$

        $$
        gatherOut=AllGather(x1)
        $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
    </colgroup>
    <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>x1 </td>
        <td>Input</td>
        <td>Left matrix of MM, that is, x1 in the computation formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8, INT4</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>x2 </td>
        <td>Input</td>
        <td>Right matrix of MM, that is, x2 in the computation formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8, INT4</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>bias </td>
        <td>Optional input</td>
        <td>bias in the computation formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>x1Scale </td>
        <td>Optional input</td>
        <td>Dequantization parameter of the mm left matrix.</td>
        <td>FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>x2Scale </td>
        <td>Optional input</td>
        <td>Dequantization parameter of the mm right matrix.</td>
        <td>FLOAT, INT64, FLOAT8_E8M0</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>quantScale </td>
        <td>Optional input</td>
        <td>bias in the computation formula.</td>
        <td>FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>output </td>
        <td>Output</td>
        <td>Result of AllGather communication and MatMul computation, that is, output in the computation formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gatherOut </td>
        <td>Output</td>
        <td>Outputs only the result after all_gather communication, that is, gatherOut in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, INT8, INT4</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>amaxOut </td>
        <td>Optional output</td>
        <td>Maximum value result of the MM computation, that is, amaxOut in the formula.</td>
        <td>FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>Attribute</td>
        <td>Indicates the number of values that the mm output matrix can quantify in the corresponding direction along the M axis and the N axis.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>Attribute</td>
        <td>Communicator name.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gatherIndex</td>
        <td>Attribute</td>
        <td>Identifies the gather target.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn</td>
        <td>Attribute</td>
        <td>Number of communication data splits, that is, the total data volume divided by the volume of each communication.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode</td>
        <td>Attribute</td>
        <td>Enum of the stream mode.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>Optional attribute</td>
        <td>Indicates, in dequantization, the number of x1/x2 input values along the corresponding dimension direction that a single value of the x1Scale/x2Scale input can dequantize along that direction.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode</td>
        <td>Attribute</td>
        <td>Communication mode.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    </tbody></table>

## Constraints

- Deterministic computation:
  - This operator is deterministic by default.
- <term>Ascend 950PR&950DT products</term>:
    - Input x1 is 2D with dimension \(m, k\). x2 must be 2D with dimension \(k, n\). The axes must satisfy the input requirements of the mm operator, the k axes must be equal, and the value range of the k axis is \[256, 65535\).
    - bias is 1D with shape \(n,\).
    - The output `output` is 2D with dimension \(m*rank\_size, n\), where rank\_size is the number of cards.
    - The output gatherout is 2D with dimension \(m*rank\_size, k\), where rank\_size is the number of cards.
    - When the data types of x1 and x2 are FLOAT16/BFLOAT16, the output computation output data type is consistent with x1 and x2.
    - When the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, the output data type supports FLOAT16, BFLOAT16, and FLOAT.
    - When the data types of x1 and x2 are FLOAT16/BFLOAT16/HIFLOAT8, the data types of x1 and x2 must be consistent.
    - When the data types of x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2, the data types of x1 and x2 can be either of the two.
    - When the data types of x1 and x2 are FLOAT16/BFLOAT16/HIFLOAT8/FLOAT8_E4M3FN/FLOAT8_E5M2, the x2 matrix supports both the transposed and non-transposed scenarios, and the x1 matrix supports only the non-transposed scenario.
    - When the data types of x1 and x2 are FLOAT4_E2M1, the x2 matrix supports both the transposed and non-transposed scenarios, the x1 matrix supports only the non-transposed scenario, the k axis must be even, and when the x2 matrix is non-transposed, the n axis must also be even.
    - When groupSize is 549764202624, bias must be empty.
    - Supports 2, 4, 8, 16, 32, and 64 cards.
    - Supports the CCU communication engine and the AICPU communication engine. CCU supports interconnection only within the UB domain of a single machine, and AICPU supports interconnection across machines within the UB domain.
    - The total data volume of allgather(x1) collective communication cannot exceed 63*256MB. The total communication data volume is calculated as m* k *sizeof(x1_dtype)* number of cards. Because shapes differ, the internal implementation of the operator may vary, and the actually supported total communication volume may be slightly smaller than this value.

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
    - Only the transposed and non-transposed x2 matrix is supported, and the x1 matrix supports only the non-transposed scenario.
    - Input x1 must be 2D with shape \(m, k\).
    - Input x2 must be 2D with shape \(k, n\). The axes must satisfy the input requirements of the mm operator, the k axes must be equal, and the value range of the k axis is \[256, 65535\).
    - bias supports only nullptr input.
    - The output is 2D with shape \(m*rank\_size, n\), where rank\_size is the number of cards.
    - Empty tensors are not supported.
    - The data types of x1 and x2 must be consistent.
    - When the data types of x1 and x2 are INT4 (A4W4), k and n must be even; x2 supports only the ND format and does not support the FRACTAL_NZ (NZ) format.
    - Supports 2, 4, and 8 cards.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_all_gather_matmul_v2](./examples/test_aclnn_all_gather_matmul_v2.cpp) | Calls the AllGatherMatmulV2 operator through the [aclnnAllGatherMatmulV2](docs/aclnnAllGatherMatmulV2_en.md) API. |
