# InplaceMatmulAllReduceAddRmsNorm

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| ---- | :----: |
| <term>Atlas A3 products</term>   | x |
| <term>Atlas A2 products</term> | √ |

## Description

- Operator function: Performs the in-place mm + all_reduce + add + rms_norm computation.
- Computation formula:
  - Scenario 1:

    $$
    mm\_out = allReduce(x1 @ x2 + bias)
    $$

    $$
    y = mm\_out + residual
    $$

    $$
    normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
    $$

  - Scenario 2:

    $$
    mm_out = allReduce(dequantScale * (x1_{int8}@x2_{int8} + bias_{int32}))
    $$

    $$
    y = mm_out + residual
    $$

    $$
    normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
    $$

  - Scenario 3:

    $$
    mm\_out = allReduce(x1 @ (x2*antiquantScale + antiquantOffset) + bias)
    $$

    $$
    y = mm\_out + residual
    $$

    $$
    normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
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
      <td>Input/Output</td>
      <td>The left matrix of MatMul, that is, the input x1 in the formula. It is also the output, that is, y in the formula.</td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The right matrix of MatMul, that is, the input x2 in the formula.</td>
      <td>FLOAT16, BFLOAT16, INT8, INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>The Add item after the Matmul computation, that is, the input bias in the formula.</td>
      <td>FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>residual</td>
      <td>Input</td>
      <td>The residual input of the AddRmsNorm fused operator, that is, the input residual in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>The RmsNorm computation input of the AddRmsNorm fused operator, that is, the input gamma in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquant_scale</td>
      <td>Optional input</td>
      <td>The input antiquantScale in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquant_offset</td>
      <td>Optional input</td>
      <td>The offset parameter for the pseudo-quantization computation on x2, that is, the input antiquantOffset in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dequant_scale</td>
      <td>Optional input</td>
      <td>The full-quantization coefficient after the mm computation, that is, the input dequantScale in the formula.</td>
      <td>FLOAT16, BFLOAT16, UINT64, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>norm_out</td>
      <td>Output</td>
      <td><ul><li>The output normOut in the formula.</li><li>The result of mm + all_reduce + add + rms_norm.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>Attribute</td>
      <td><ul><li>The communicator name.</li><li>Obtained through the HCCL API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</li></ul></td>
      <td>CHAR*, STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduceOp</td>
      <td>Optional attribute</td>
      <td><ul><li>The reduce operation type.</li><li>The default value is "sum".</li></ul></td>
      <td>CHAR*, STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_a</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to transpose x1 before the matrix multiplication.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_b</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to transpose x2 before the matrix multiplication.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>commTurn</td>
      <td>Optional attribute</td>
      <td><ul><li>The number of communication data splits, that is, the total data volume divided by the data volume per communication.</li><li>The default value is 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_group_size</td>
      <td>Optional attribute</td>
      <td><ul><li>In pseudo-quantization pergroup mode, the groupSize input for the dequantization computation on x2.</li><li>The default value is 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Used to prevent division-by-zero errors, that is, the input epsilon in the formula.</li><li>The value of epsilon must be in the range (0,1).</li><li>The default value is 1e-6.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

* The usage scenarios are the same as those of the fused operator aclnnWeightQuantMatmulAllReduce: MC2 is disabled for incremental scenarios and enabled for full scenarios.
* The input x1 can be 2D or 3D, with shape (b, s, k) or (s, k). x2 must be 2D, with shape (k, n). The axes must meet the input requirements of the mm operator, and the k axes must be equal. m is in the range [1, 2147483647], and k and n are in the range [1, 65535]. If bias is not empty, bias is 1D with shape (n). bias is optional and can be empty. When bias is not empty, the current version supports only 1D input.
* The input residual must be 3D, with shape (b, s, n). When x1 is 2D, the (b*s) of residual equals the s of x1. Non-contiguous tensors are not supported. The input gamma must be 1D, with shape (n). Non-contiguous tensors are not supported.
* In the pertensor scenario, the shape of antiquantScale is (1). In the perchannel scenario, the shape is (1,n)/(n). In the pergroup scenario, the shape is (ceil(k,antiquantGroupSize),n). antiquantOffset is optional and can be empty. When antiquantOffset is not empty, its shape is the same as that of antiquantScale.
* In the pertensor scenario, the shape of dequantScale is (1). In the perchannel scenario, the shape is (n)/(1, n).
* The dimensions and data types of the outputs y and normOut are the same as those of residual. If bias is not empty, its shape size equals the last dimension of normOut.
* The data type of x2 must be int8 or int4, and the data types of the computation inputs x1, bias, residual, gamma, y, and normOut must be the same.
* When the pergroup scenario is not supported, pass 0 for antiquantGroupSize. When the pergroup scenario is supported, the value must be in the range [32, min(k-1,INT_MAX)] and be a multiple of 32. The value range of k is the same as that of the mm API.
* Empty tensors with (b*s) or n equal to 0 are supported, but empty tensors with k equal to 0 are not supported.
* Only the x2 matrix supports transpose/no-transpose, while the x1 matrix supports only the no-transpose scenario.
* The current version supports only "sum" as the input of the attribute reduceOp.
* The current version supports only 0 as the input of the attribute commTurn.
* 1, 2, 4, and 8 devices are supported, and only all mesh networking over hccs links is supported.
* <term>Atlas A2 products</term>: For the communication-computation fusion MC2 operators in one model, only the same communicator is supported. The types must be the same.

## Calling Description

| Calling Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_inplace_matmul_all_reduce_add_rms_norm.cpp](./examples/test_aclnn_inplace_matmul_all_reduce_add_rms_norm.cpp) | Call the InplaceMatmulAllReduceAddRmsNorm operator through the [aclnnInplaceMatmulAllReduceAddRmsNorm.md](docs/aclnnInplaceMatmulAllReduceAddRmsNorm_en.md) API. |
