# MatmulAlltoAll

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Functional Description

- Operator function: Fuses the quantized MatMul computation, Permute (which ensures address continuity after communication), and AlltoAll communication. **Computation is performed before communication.** The non-quantized, K-C quantization, and mx [quantization modes](../../docs/zh/context/quant_mode_introduction_en.md) are supported.
- Computation formula: Assume that the shape of x1 is (BS, H1), the shape of x2 is (H1, H2), and rankSize is the number of NPU cards.

    - <term>Atlas A2 products</term>:
        - Non-quantized scenario:

          $$
          computeOut = x1 @ x2 + bias \\
          permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2) \\
          output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
          $$

        - K-C quantization scenario:

          $$
          computeOut = (x1 @ x2) * x1Scale * x2Scale  + bias \\
          permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
          output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
          $$

    - <term>Atlas A3 products</term>:
        - Non-quantized scenario:

          $$
          computeOut = x1 @ x2 + bias \\
          permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2) \\
          output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
          $$

    - <term>Ascend 950PR&950DT products</term>:
        - Non-quantized scenario:

          $$
          computeOut = x1 @ x2 + bias \\
          permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2) \\
          output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
          $$

        - K-C quantization scenario:

          $$
          computeOut = (x1 @ x2 + bias) * x1Scale * x2Scale \\
          permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
          output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
          $$

        - mx quantization scenario:

          $$
          computeOut = \sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (x1 @ x2 * (x1Scale * x2Scale)) + bias \\
          permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
          output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
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
    <td>x1</td>
    <td>Input</td>
    <td>Left matrix of the MatMul, that is, input x1 in the formula.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT8</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right matrix of the MatMul, that is, input x2 in the formula.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT8</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>bias</td>
    <td>Optional input</td>
    <td>Optional input. Bias accumulated after the matrix multiplication, corresponding to bias in the formula.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x1_scale</td>
    <td>Optional input</td>
    <td>Quantization scale of the left matrix, corresponding to x1Scale in the formula.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x2_scale</td>
    <td>Optional input</td>
    <td>Quantization scale of the right matrix, corresponding to x2Scale in the formula.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>comm_scale</td>
    <td>Optional input</td>
    <td>Reserved parameter. Quantization scale for low-bit communication.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1_offset</td>
    <td>Optional input</td>
    <td>Reserved parameter. Quantization offset of the left matrix.</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2_offset</td>
    <td>Optional input</td>
    <td>Reserved parameter. Quantization offset of the right matrix.</td>
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
    <td>group</td>
    <td>Required attribute</td>
    <td>String on the Host that identifies the column group, that is, the communicator name. Obtain commName through the Hccl API HcclGetCommName and use it as this parameter. The string length must be (0, 128).</td>
    <td>STRING</td>
    <td>-</td>
    </tr>
    <tr>
    <td>world_size</td>
    <td>Required attribute</td>
    <td>Number of NPU cards used, corresponding to rankSize in the formula.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>all2all_axes</td>
    <td>Optional attribute</td>
    <td>Direction of data exchange between AlltoAll and Permute. The value can be empty or [-1, -2]. If an empty value is passed, it is processed as [-1, -2] by default, which converts the input from (BS, H2) to (BS*rankSize, H2/rankSize).</td>
    <td>aclIntArray* (element type INT64)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1_quant_mode</td>
    <td>Optional attribute</td>
    <td>Quantization mode of the left matrix. Set the value based on the actual scenario.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2_quant_mode</td>
    <td>Optional attribute</td>
    <td>Quantization mode of the right matrix. Set the value based on the actual scenario.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>comm_quant_mode</td>
    <td>Optional attribute</td>
    <td>Quantization mode for low-bit communication. This is a reserved parameter. Currently, only 0 can be configured, indicating no quantization.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>comm_quant_dtype</td>
    <td>Optional attribute</td>
    <td>Quantization type for low-bit communication. This is a reserved parameter. Currently, only -1 can be configured, indicating ACL_DT_UNDEFINED.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transpose_x1</td>
    <td>Optional attribute</td>
    <td>Identifies whether the left matrix is transposed. Setting the value to True is not yet supported.</td>
    <td>bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transpose_x2</td>
    <td>Optional attribute</td>
    <td>Identifies whether the right matrix is transposed. If the value is set to True, the shape of the right matrix is (H2, H1).</td>
    <td>bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group_size</td>
    <td>Optional attribute</td>
    <td>Quantization group size used in three directions of the MatMul computation. The value takes effect only when all scale inputs are 2D or higher. In other scenarios, pass 0 by default.</td>
    <td>INT</td>
    <td>-</td>
    </tr>
    </tbody></table>

The relationship between the enumeration values of x1QuantMode, x2QuantMode, and commQuantMode and the [quantization modes](../../docs/zh/context/quant_mode_introduction_en.md) is as follows:

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
* The number of NPU cards (world_size) varies depending on the product model:
    - <term>Atlas A2 products</term>: 2, 4, and 8 cards are supported.
    - <term>Atlas A3 products</term>: 2, 4, 8, and 16 cards are supported.
    - <term>Ascend 950PR&950DT products</term>: 2, 4, 8, and 16 cards are supported.
* Support for empty tensors and non-contiguous tensors varies depending on the product model:
    - <term>Atlas A2 products</term>: No empty tensor is supported. No non-contiguous tensor is supported.
    - <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: Only empty tensors where the first dimension (BS) of input x1 is 0 in the non-quantized scenario are supported, and no other empty tensor is supported. Only the transposed non-contiguous tensor of input x2 is supported, and no other non-contiguous tensor is supported.
* Input x1 must be 2D with shape (BS, H1). The values of BS*rankSize and H2 must not exceed 2147483647 (INT32_MAX).
* Input x2 must be 2D with shape (H1, H2). H2 must be evenly divided by the number of NPU cards. The value range of H1 only supports [1, 65535], and the value of H2 must not exceed 2147483647 (INT32_MAX).
* If bias is not empty, it must be 1D with shape (H2).
* If x1_scale is not empty, it is 3D with shape (BS, ceil(H1/64), 2) in the mx quantization scenario, and 1D with shape (BS) in other scenarios.
* If x2_scale is not empty, it is 3D with shape (H2, ceil(H1/64), 2) in the mx quantization scenario, and 1D with shape (H2) in other scenarios.
* all2all_axes is a 1D array, and its shape must be (2).
* The currently supported quantization modes vary depending on the product model:
    - <term>Atlas A2 products</term>: The K-C quantization mode is supported, with x1QuantMode=3 and x2QuantMode=2.
    - <term>Atlas A3 products</term>: Quantization scenarios are not supported currently.
    - <term>Ascend 950PR&950DT products</term>: The K-C quantization mode is supported, with x1QuantMode=3 and x2QuantMode=2, and the mx quantization mode is supported, with x1QuantMode=6 and x2QuantMode=6.
* In the non-quantized scenario, the data types of the computation inputs x1 and x2 must be the same as the data type of the computation output `output`, and none of x1, x2, or output passed in is a null pointer.
* In the quantization scenario, none of x1, x2, x1Scale, x2Scale, or output passed in is a null pointer, and
    - <term>Atlas A2 products</term>: A null pointer cannot be passed in for bias.
* The data types of the computation inputs x1, x2, and bias vary depending on the product model:
    - <term>Atlas A2 products</term>:
        - In the non-quantized scenario, when the data type of the computation inputs x1/x2 is FLOAT16, the data type of the computation input bias supports FLOAT16. When the data type of the computation inputs x1/x2 is BFLOAT16, the data type of the computation input bias supports FLOAT32.
        - In the quantization scenario, the K-C quantization mode with bias added afterwards is supported. The data types of the computation inputs x1 and x2 must be INT8. When the data type of the computation output `output` is BFLOAT16, the data type of bias is FLOAT32 or BFLOAT16. When the data type of output is FLOAT16, the data type of bias is FLOAT16.
    - <term>Atlas A3 products</term>:
        - In the non-quantized scenario, when the data type of the computation output `output` is FLOAT16, the data type of the computation input bias supports FLOAT16. When the data type of the computation output `output` is BFLOAT16, the data type of the computation input bias supports FLOAT32.
        - Quantization scenarios are not supported on A3 currently.
    - <term>Ascend 950PR&950DT products</term>:
        - In the non-quantized scenario, when the data type of the computation inputs x1/x2 is FLOAT16, the data type of the computation input bias supports FLOAT16 and FLOAT32. When the data type of the computation inputs x1/x2 is BFLOAT16, the data type of the computation input bias supports BFLOAT16 and FLOAT32.
        - In the quantization scenario, the K-C quantization mode and mx quantization mode are supported. The data types of the computation inputs x1 and x2 are FLOAT8_E4M3FN, FLOAT8_E5M2, and FLOAT4_E2M1. The data type of bias is FLOAT32, or bias is empty. For details about the specific type combinations, see [Quantization aclnn Constraints](docs/aclnnQuantMatmulAlltoAll_en.md#constraints).
        - In the mx quantization scenario, when the data types of x1 and x2 are FLOAT4_E2M1, the data types of the two must be the same.
* Communication-computation fusion operators do not support concurrent calls, and different communication-computation fusion operators do not support concurrent calls either.
* Cross-supernode communication is not supported. Only communication within a supernode is supported.
* Communication engine constraints:
   - <term>Atlas A2 products</term>: MTE communication is supported.
   - <term>Atlas A3 products</term>: AI CPU communication is supported.
   - <term>Ascend 950PR&950DT products</term>: CCU communication and AI CPU communication are supported. CCU supports only interconnection within the UB domain on a single machine, and AI CPU supports interconnection across machines within the UB domain.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_matmul_allto_all.cpp](./examples/test_aclnn_matmul_allto_all.cpp) | Calls the MatMulAlltoAll operator in the non-quantized scenario through the [aclnnMatMulAlltoAll](docs/aclnnMatmulAlltoAll_en.md) API. |
| aclnn API  | [test_aclnn_quant_matmul_allto_all.cpp](./examples/test_aclnn_quant_matmul_allto_all.cpp) | Calls the MatMulAlltoAll operator in the quantization scenario through the [aclnnQuantMatMulAlltoAll](docs/aclnnQuantMatmulAlltoAll_en.md) API. |
