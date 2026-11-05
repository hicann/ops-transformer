# FFN

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    ×     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: this FFN operator provides the computation functions of MoeFFN and FFN. When there is no expert grouping (expertTokens is empty), it is FFN; when there is expert grouping, it is MoeFFN. Both are collectively called FFN and belong to the Moe structure. MoE (Mixture-of-Experts) is a technology for training models with trillions of parameters. MoE decomposes the prediction modeling task into several subtasks, trains an expert model on each subtask, and develops a gating model, which allocates one or more experts based on the input data. Finally, the computation results of multiple experts are integrated as the prediction result. A model with the Mixture-of-Experts structure allocates the input data to the one or more most relevant experts, and determines the final result by integrating the computation results of all involved experts.
- Computation formula:

  - **Non-quantized scenario:**

	$$
    y=activation(x * W1 + b1) * W2 + b2
	$$

  - **Quantized scenario:**

	$$
    y=((activation((x * W1 + b1) * deqScale1) * scale + offset) * W2 + b2) * deqScale2
	$$

  - **Pseudo-quantized scenario:**

	$$
    y=activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
	$$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
<col style="width: 150px">
<col style="width: 300px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
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
    <td>x</td>
    <td>Input</td>
    <td>Required parameter. aclTensor on the Device side, which is the input x in the formula. The input supports at least 2 dimensions [M, K1] and at most 8 dimensions.</td>
    <td>FLOAT16, BFLOAT16, INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight1</td>
    <td>Input</td>
    <td>Required parameter. aclTensor on the Device side, which is the weight data of the experts, that is, W1 in the formula. The input is [E, K1, N1]/[K1, N1] with/without experts.</td>
    <td>FLOAT16, BFLOAT16, INT8, INT4</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight2</td>
    <td>Input</td>
    <td>Required parameter. aclTensor on the Device side, which is the weight data of the experts, that is, W2 in the formula. The input is [E, K2, N2]/[K2, N2] with/without experts.</td>
    <td>FLOAT16, BFLOAT16, INT8, INT4</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>expertTokens</td>
    <td>Input</td>
    <td>Optional parameter. aclIntArray on the Host side, representing the number of tokens of each expert. If it is not empty, the maximum supported length is 256.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias1</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is the correction value of the weight data, that is, b1 in the formula. The input is [E, N1]/[N1] with/without experts.</td>
    <td>FLOAT16, FLOAT32, INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias2</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is the correction value of the weight data, that is, b2 in the formula. The input is [E, N2]/[N2] with/without experts.</td>
    <td>FLOAT16, FLOAT32, INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is a quantization parameter, that is, the quantization scaling factor. In the per-tensor mode, the input is a one-dimensional vector with/without experts, and the number of input elements is [E]/[1] with/without experts. In the per-channel mode, the input is a two-dimensional vector/one-dimensional vector with/without experts, and the number of input elements is [E, N1]/[N1] with/without experts.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>offset</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is a quantization parameter, that is, the quantization offset. It is a one-dimensional vector, and the number of input elements is [E]/[1] with/without experts.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>deqScale1</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is a quantization parameter, that is, the dequantization scaling factor of the first matmul. The input is [E, N1]/[N1] with/without experts.</td>
    <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>deqScale2</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is a quantization parameter, that is, the dequantization scaling factor of the second matmul. The input is [E, N2]/[N2] with/without experts.</td>
    <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>antiquantScale1</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is a pseudo-quantization parameter, that is, the scaling factor of the first matmul. In the per-channel mode, the input is [E, N1]/[N1] with/without experts; in the per-group mode, the input is [E, G, N1]/[G, N1] with/without experts.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>antiquantScale2</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is a pseudo-quantization parameter, that is, the scaling factor of the second matmul. In the per-channel mode, the input is [E, N2]/[N2] with/without experts; in the per-group mode, the input is [E, G, N2]/[G, N2] with/without experts.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>antiquantOffset1</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is a pseudo-quantization parameter, that is, the offset of the first matmul. In the per-channel mode, the input is [E, N1]/[N1] with/without experts; in the per-group mode, the input is [E, G, N1]/[G, N1] with/without experts.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>antiquantOffset2</td>
    <td>Input</td>
    <td>Optional parameter. aclTensor on the Device side, which is a pseudo-quantization parameter, that is, the offset of the second matmul. In the per-channel mode, the input is [E, N2]/[N2] with/without experts; in the per-group mode, the input is [E, G, N2]/[G, N2] with/without experts.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

- When experts exist, the total amount of expert data must be consistent with M of x.
- When the activation layer is geglu/swiglu/reglu, only the FLOAT16 high-performance scenario without expert grouping is supported (the FLOAT16 scenario refers to the scenario where the data types of all required parameters of the aclTensor type are FLOAT16), and N1=2\*K2.
- When the activation layer is gelu/fastgelu/relu/silu, the FLOAT16 high-precision and high-performance scenarios with or without expert grouping, the BFLOAT16 scenario, the quantized scenario, and the pseudo-quantized scenario are supported, and N1=K2.
- All scenarios must satisfy K1=N2, K1<65536, K2<65536, and the M axis, after being aligned to 32 bytes, must be smaller than the maximum value of INT32.
- The non-quantized scenario cannot take quantization or pseudo-quantization parameters as input, the quantized scenario cannot take pseudo-quantization parameters as input, and the pseudo-quantized scenario cannot take quantization parameters as input.
- Parameter types in the quantized scenario: x is INT8, weight is INT8, bias is INT32, scale is FLOAT32, and offset is FLOAT32. The other parameter types fall into two cases depending on y:
  - When y is FLOAT16, the data types supported by deqScale are UINT64, INT64, and FLOAT32.
  - When y is BFLOAT16, the data type supported by deqScale is BFLOAT16.
  - deqScale1 and deqScale2 must have the same data type.
- Parameter types supported by scale in the per-channel mode in the quantized scenario: x is INT8, weight is INT8, bias is INT32, scale is FLOAT32, and offset is FLOAT32. The other parameter types fall into two cases depending on y:
  - When y is FLOAT16, the data types supported by deqScale are UINT64 and INT64.
  - When y is BFLOAT16, the data type supported by deqScale is BFLOAT16.
  - deqScale1 and deqScale2 must have the same data type.
- The pseudo-quantized scenario supports two different parameter types:
  - y is FLOAT16, x is FLOAT16, bias is FLOAT16, antiquantScale is FLOAT16, and antiquantOffset is FLOAT16. weight supports the data types INT8 and INT4.
  - y is BFLOAT16, x is BFLOAT16, bias is FLOAT32, antiquantScale is BFLOAT16, and antiquantOffset is BFLOAT16. weight supports the data types INT8 and INT4.
- When the data type of weight1/weight2 is INT4, the last dimension of its shape must be an even number.
- In the pseudo-quantized scenario and per-group mode, the number of groups G in antiquantScale1 and antiquantOffset1 must be divisible by K1, and the number of groups G in antiquantScale2 and antiquantOffset2 must be divisible by K2.
- In the pseudo-quantized scenario, the per-group mode currently only supports the scenario where weight is of the INT4 data type.
- In the BFLOAT16 non-quantized scenario, the innerPrecise parameter can only be set to 0. In the FLOAT16 non-quantized scenario, it can be set to 0 or 1. In the quantized or pseudo-quantized scenario, both 0 and 1 can be set, but the setting does not take effect.

## Invocation Description

| Invocation Method      | Invocation Sample                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_ffn.cpp](examples/test_aclnn_ffn.cpp) | Invokes the [FFN](docs/aclnnFFN_en.md) operator through the interface. |
