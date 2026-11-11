# AlltoAllMatmul

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: fuses AlltoAll communication, Permute (which keeps addresses contiguous after communication), and the Matmul computation, with **communication before computation**. Supports the non-quantization, K-C quantization, K-C dynamic quantization, and mx [quantization modes](../../docs/zh/context/quant_mode_introduction_en.md).
- Computation formula: assume the input x1 has shape (BS, H); in the mx quantization scenario, the input x1Scale has shape (BS, ceil(H/64), 2); rankSize is the number of NPU cards.

    - <term>Atlas A2 products</term>:
      - Non-quantization scenario:

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        output = permutedOut @ x2 + bias \\
        $$

      - K-C quantization scenario:

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        output_{quant} = permutedOut @ x2 \\
        output = output_{quant} \times x1_{scale} \times x2_{scale} \\
        output = output + bias
        $$

      - K-C dynamic quantization scenario:

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        x1_{quant}, x1_{scale} = Quant(permutedOut) \\
        output_{quant} = x1_{quant} @ x2 \\
        output = output_{quant} \times x1_{scale} \times x2_{scale} \\
        output = output + bias
        $$

    - <term>Atlas A3 products</term>:
      - Non-quantization scenario:

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        output = permutedOut @ x2 + bias \\
        $$

    - <term>Ascend 950PR&950DT products</term>:
      - Non-quantization scenario:

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        output = permutedOut @ x2 + bias \\
        $$

      - K-C dynamic quantization scenario:

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        dynQuantX1, dynQuantX1Scale = dynamicQuant(permutedOut) \\
        output = (dynQuantX1@x2 + bias) \times dynQuantX1Scale \times x2Scale
        $$

      - mx quantization scenario:

        $$
        commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
        permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
        commScale = AlltoAll(x1Scale.view(rankSize, BS/rankSize, ceil(H/64), 2)) \\
        permutedScale = commScale.permute(1, 0, 2, 3).view(BS/rankSize, ceil(H/64)*rankSize, 2) \\
        output = \sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (permutedOut @ x2 * (permutedScale * x2Scale)) + bias
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
        <th>Parameter Name</th>
        <th>Input/Output/Attribute</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr></thead>
    <tbody>
    <tr>
    <td>x1</td>
    <td>Input</td>
    <td>Left matrix of the fused operator, that is, the input x1 in the formula.</td>
    <td>FLOAT16, BFLOAT16, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right matrix of the fused operator and the right matrix of the Matmul, that is, the input x2 in the formula.</td>
    <td>FLOAT16, BFLOAT16, INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>bias</td>
    <td>Optional input</td>
    <td>Optional input, the bias accumulated after the matrix multiplication, corresponding to bias in the formula.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x1_scale</td>
    <td>Optional input</td>
    <td>Quantization coefficient of the left matrix, corresponding to x1Scale in the formula.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x2_scale</td>
    <td>Optional input</td>
    <td>Quantization coefficient of the right matrix, corresponding to x2Scale in the formula.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>comm_scale</td>
    <td>Optional input</td>
    <td>Reserved parameter, the quantization coefficient for low-bit communication.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1_offset</td>
    <td>Optional input</td>
    <td>Reserved parameter, the quantization offset of the left matrix.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2_offset</td>
    <td>Optional input</td>
    <td>Reserved parameter, the quantization offset of the right matrix.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Result of computation + communication, that is, output in the formula.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>alltoall_out</td>
    <td>Optional output</td>
    <td>Corresponds to all2all_out in the IR. When alltoall_out_flag is true, retains the result of AlltoAll and Permute, that is, permutedOut in the formula. When false, this optional output is not retained; the main computation still performs AlltoAll and Permute.</td>
    <td>Same as the input x1</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Required attribute</td>
    <td>String that identifies the column group on the Host side, that is, the communicator name. Obtain commName through the Hccl interface HcclGetCommName as this parameter. The string length must be (0, 128).</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>world_size</td>
    <td>Required attribute</td>
    <td>Number of NPU cards used, that is, rankSize in the formula. It must match the actual number of cards in the communicator specified by group.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>all2all_axes</td>
    <td>Optional attribute</td>
    <td>Direction of the data exchange for AlltoAll and Permute. Supports empty or [-2, -1]. When empty is passed, [-2, -1] applies by default, which converts the input from (BS, H) to (BS/rankSize, H*rankSize).</td>
    <td>aclIntArray* (element type INT64)</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y_dtype</td>
    <td>Optional attribute</td>
    <td>Data type of output y. The IR default is DT_UNDEFINED. Current data type inference does not accept DT_UNDEFINED, so callers must set a supported output type for the platform and computation scenario.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1_quant_mode</td>
    <td>Optional attribute</td>
    <td>Quantization mode of the left matrix. Configure it according to the actual scenario.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2_quant_mode</td>
    <td>Optional attribute</td>
    <td>Quantization mode of the right matrix. Configure it according to the actual scenario.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>comm_quant_mode</td>
    <td>Optional attribute</td>
    <td>Quantization mode for low-bit communication. Reserved parameter. Currently only 0 can be configured, indicating no quantization.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>comm_quant_dtype</td>
    <td>Optional attribute</td>
    <td>Quantization data type for low-bit communication. Reserved parameter. Currently only -1 can be configured, indicating ACL_DT_UNDEFINED.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1_quant_dtype</td>
    <td>Optional attribute</td>
    <td>Quantization data type for quantizing the left matrix of the Matmul. The result of AlltoAll communication and Permute is quantized according to this parameter configuration and then used as the left matrix input of the Matmul computation. Configure it according to the actual scenario.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transpose_x1</td>
    <td>Optional attribute</td>
    <td>Indicates whether the left matrix is transposed. Setting it to True is not yet supported.</td>
    <td>bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transpose_x2</td>
    <td>Optional attribute</td>
    <td>Indicates whether the right matrix is transposed. When set to True, the shape of the right matrix is (N, H*rankSize).</td>
    <td>bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group_size</td>
    <td>Optional attribute</td>
    <td>Quantization group sizes along three directions for the Matmul computation. The value takes effect only when all scale inputs are two-dimensional or higher; in other scenarios, pass 0 by default.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>comm_mode</td>
    <td>Optional attribute</td>
    <td>Communication mode. The IR default is "ai_cpu". Atlas A2 products require "aiv"; Atlas A3 products support "ai_cpu"; Ascend 950PR&950DT products support "ai_cpu" or "ccu".</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>alltoall_out_flag</td>
    <td>Optional attribute</td>
    <td>Indicates whether to keep the result after AlltoAll and Permute. Defaults to true and controls the optional output alltoall_out.</td>
    <td>bool</td>
    <td>-</td>
    </tr>
    </tbody></table>

The enumeration values of x1QuantMode, x2QuantMode, and commQuantMode map to the [quantization modes](../../docs/zh/context/quant_mode_introduction_en.md) as follows:

* 0: no quantization
* 1: pertensor
* 2: perchannel
* 3: pertoken
* 4: pergroup
* 5: perblock
* 6: mx quantization
* 7: pertoken dynamic quantization

## Constraints

* Deterministic computation is supported by default.
* The number of NPU cards (rankSize) has different limits depending on the device model:
    - <term>Atlas A2 products</term>: supports 2, 4, and 8 cards.
    - <term>Atlas A3 products</term>: supports 2, 4, 8, and 16 cards.
    - <term>Ascend 950PR&950DT products</term>: supports 2, 4, 8, and 16 cards.
* Support for empty tensors and non-contiguous tensors has different limits depending on the device model:
    - <term>Atlas A2 products</term>: does not support any empty tensors; does not support any non-contiguous tensors.
    - <term>Atlas A3 products</term>: does not support any empty tensors. Supports only the transposed non-contiguous tensor of input x2; other non-contiguous tensors are not supported.
    - <term>Ascend 950PR&950DT products</term>: supports only the empty tensor whose first dimension (BS) of input x1 is 0 in the non-quantization scenario; other empty tensors are not supported. Supports only the transposed non-contiguous tensor of input x2; other non-contiguous tensors are not supported.
* The input x1 must be two-dimensional with shape (BS, H). BS must be divisible by the number of NPU cards. The values of BS and N must not exceed 2147483647 (INT32_MAX). Transposition is not supported.
* The input x2 must be two-dimensional with shape (H\*rankSize, N). The range of H*rankSize has different constraints depending on the chip model and scenario. For details, see [quantization aclnn constraints](docs/aclnnAlltoAllQuantMatmul_en.md#constraints)
  and [non-quantization aclnn constraints](docs/aclnnAlltoAllMatmul_en.md#constraints). In the mx quantization scenario, x2 must be transposed with shape (N, H\*rankSize), and transpose_x2 is set to True.
* If bias is not empty, it must be one-dimensional with shape (N).
* If x1_scale is not empty, it is three-dimensional with shape (BS, ceil(H/64), 2) in the mx quantization scenario, one-dimensional with shape (BS) in the K-C quantization scenario, and one-dimensional with shape (H*rankSize) in the K-C dynamic quantization scenario.
* If x2_scale is not empty, it is three-dimensional with shape (N, ceil(H\*rankSize/64), 2) in the mx quantization scenario and one-dimensional with shape (N) in other scenarios.
* all2all_axes is a one-dimensional array, and its shape must be (2).
* The currently supported quantization modes have different limits depending on the device model:
    - <term>Atlas A2 products</term>: supports the K-C quantization and K-C dynamic quantization modes, with x1QuantMode=3 or 7 and x2QuantMode=2.
    - <term>Atlas A3 products</term>: currently does not support quantization scenarios.
    - <term>Ascend 950PR&950DT products</term>: supports the K-C dynamic quantization mode with x1QuantMode=7 and x2QuantMode=2, and the mx quantization mode with x1QuantMode=6 and x2QuantMode=6.
* In the non-quantization scenario, the data types of the computation inputs x1 and x2 must be the same as those of the computation outputs output and alltoAllOutOptional, and the passed-in x1, x2, and output must not be null pointers.
* In the quantization scenario, the data type of x1 must be the same as that of alltoAllOutOptional, and the passed-in x1, x2, x2Scale, and output must not be null pointers.
* The data types of the computation inputs x1, x2, and bias have different limits depending on the device model:
    - <term>Atlas A2 products</term>:
        - In the non-quantization scenario, when the data type of the computation output `output` is FLOAT16, the supported data type of the computation input bias is FLOAT16; when the data type of output is BFLOAT16, the supported data type of bias is FLOAT32.
        - In the quantization scenario, see [quantization aclnn constraints](docs/aclnnAlltoAllQuantMatmul_en.md#constraints) for the supported data type combinations.
    - <term>Atlas A3 products</term>:
        - In the non-quantization scenario, when the data type of the computation output `output` is FLOAT16, the supported data type of the computation input bias is FLOAT16; when the data type of output is BFLOAT16, the supported data type of bias is FLOAT32.
        - A3 currently does not support quantization scenarios.
    - <term>Ascend 950PR&950DT products</term>:
        - In the non-quantization scenario, when the data types of the computation inputs x1/x2 are FLOAT16, the supported data types of the computation input bias are FLOAT16 and FLOAT32; when the data types of x1/x2 are BFLOAT16, the supported data types of bias are BFLOAT16 and FLOAT32.
        - In the quantization scenario, the K-C dynamic quantization mode and the mx quantization mode are supported. The data type of the computation input x1 can be FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, or FLOAT4_E2M1, the data type of the computation input x2 can be FLOAT8_E4M3FN, FLOAT8_E5M2, or FLOAT4_E2M1, and bias is of type FLOAT32 or empty. For the supported type combinations, see [quantization aclnn constraints](docs/aclnnAlltoAllQuantMatmul_en.md#constraints).
        - In the mx quantization mode, when the data types of x1 and x2 are FLOAT4_E2M1, their data types must be the same.
* Communication-computation fused operators do not support concurrent calls, and different communication-computation fused operators do not support concurrent calls either.
* Cross-supernode communication is not supported; only intra-supernode communication is supported.
* Communication engine constraints:
   - <term>Atlas A2 products</term>: supports MTE communication.
   - <term>Atlas A3 products</term>: supports AI\_CPU communication.
   - <term>Ascend 950PR&950DT products</term>: supports CCU communication and AI\_CPU communication. CCU communication supports only intra-machine interconnection within the UB domain, and AI_CPU supports interconnection across machines within the UB domain.

## Calling Instructions

| Calling Method   | Sample Code                                                                                      | Description                                                                                           |
| ---------------- |-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn interface  | [test_aclnn_allto_all_matmul.cpp](./examples/test_aclnn_allto_all_matmul.cpp)             | Calls the AlltoAllMatMul operator in the non-quantization scenario through the [aclnnAlltoAllMatMul](docs/aclnnAlltoAllMatmul_en.md) interface.          |
| aclnn interface  | [test_aclnn_allto_all_quant_matmul.cpp](./examples/test_aclnn_allto_all_quant_matmul.cpp) | Calls the AlltoAllMatMul operator in the quantization scenario through the [aclnnAlltoAllQuantMatMul](docs/aclnnAlltoAllQuantMatmul_en.md) interface. |
