# MatmulAllReduceAddRmsNorm

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----: |
| <term>Atlas A3 products</term>   | x |
| <term>Atlas A2 products</term> | √ |

## Functional Description

- Operator function: Performs the mm + all_reduce + add + rms_norm computation.
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
    mm_out = allReduce(dequant_scale * (x1_{int8}@x2_{int8} + bias_{int32}))
    $$

    $$
    y = mm_out + residual
    $$

    $$
    normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
    $$

  - Scenario 3:

    $$
    mm\_out = allReduce(x1 @ (x2*antiquant_scale + antiquant_offset) + bias)
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
      <td>Input</td>
      <td>Left matrix of the MatMul, that is, input x1 in the formula.</td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Right matrix of the MatMul, that is, input x2 in the formula.</td>
      <td>FLOAT16, BFLOAT16, INT8, INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Add term after the Matmul computation, that is, input bias in the formula.</td>
      <td>FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>residual</td>
      <td>Input</td>
      <td>Residual input of the AddRmsNorm fused operator, that is, input residual in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>RmsNorm computation input of the AddRmsNorm fused operator, that is, input gamma in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquant_scale</td>
      <td>Optional input</td>
      <td>Input antiquant_scale in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquant_offset</td>
      <td>Optional input</td>
      <td>Offset parameter for the pseudo-quantization computation on x2, that is, input antiquant_offset in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dequant_scale</td>
      <td>Optional input</td>
      <td>Full quantization scale after the mm computation, that is, input dequant_scale in the formula.</td>
      <td>FLOAT16, BFLOAT16, UINT64, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Result of mm + all_reduce + add.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>norm_out</td>
      <td>Output</td>
      <td><ul><li>Output norm_out in the formula.</li><li>Result of mm + all_reduce + add + rms_norm.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>Attribute</td>
      <td><ul><li>Communicator name.</li><li>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</li></ul></td>
      <td>CHAR*, STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduceOp</td>
      <td>Optional attribute</td>
      <td><ul><li>Reduce operation type.</li><li>The default value is "sum".</li></ul></td>
      <td>CHAR*, STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_a</td>
      <td>Optional attribute</td>
      <td><ul><li>Determines whether x1 is transposed before matrix multiplication.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_trans_b</td>
      <td>Optional attribute</td>
      <td><li>Determines whether x2 is transposed before matrix multiplication.</li><li>The default value is false.</li></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>commTurn</td>
      <td>Optional attribute</td>
      <td><ul><li>Number of communication data splits, that is, the total data volume divided by the volume of each communication.</li><li>The default value is 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_group_size</td>
      <td>Optional attribute</td>
      <td><ul><li>In pseudo-quantization pergroup mode, groupSize input for the dequantization computation on x2.</li><li>The default value is 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Used to prevent division by zero, that is, input epsilon in the formula.</li><li>The value of epsilon is in the range (0,1).</li><li>The default value is 1e-6.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

* MC2 is not enabled in the incremental scenario and is enabled in the full scenario.
* Input x1 can be 2D or 3D, with shape (b, s, k) or (s, k). x2 must be 2D with shape (k, n). The axes must satisfy the input requirements of the mm operator, and the k axes must be equal. The value range of m is [1, 2147483647], that of k is [1, 65535], and that of n is [0, 65535]. If bias is not empty, it is 1D with shape (n). bias is optional and can be empty. When it is not empty, the current version supports only 1D input.
* Input residual must be 3D with shape (b, s, n). When x1 is 2D, the (b*s) of residual equals the s of x1. Non-contiguous tensors are not supported. Input gamma must be 1D with shape (n). Non-contiguous tensors are not supported.
* For antiquant_scale, the shape is (1) in the pertensor scenario, (1,n)/(n) in the perchannel scenario, and (ceil(k,antiquant_group_size),n) in the pergroup scenario. antiquant_offset is optional and can be empty. When it is not empty, its shape is the same as that of antiquant_scale.
* The shape of dequant_scale is (1) in the pertensor scenario and (n)/(1, n) in the perchannel scenario.
* The dimensions and data types of outputs y and normOut are the same as those of residual. If bias is not empty, its shape size is equal to the last dimension of normOut.
* The data types of the computation inputs bias, residual, gamma, y, and normOut must be consistent.
* In scenarios that do not support pergroup, pass 0 for antiquant_group_size. In scenarios that support pergroup, the value range is [32, min(k-1,INT_MAX)], and the value must be a multiple of 32. The value range of k is the same as that of the mm API.
* Empty tensors with (b*s) or n being 0 are supported. Empty tensors with k being 0 are not supported.
* Only the transposition and non-transposition of the x2 matrix are supported, and the x1 matrix supports only the non-transposed scenario.
* In the current version, attribute reduceOp supports only the input "sum".
* In the current version, attribute commTurn supports only the input 0.
* 1, 2, 4, and 8 cards are supported, and only the all mesh networking over HCCS links is supported.
* <term>Atlas A2 products</term>: In one model, the communication-computation fusion MC2 operators support only the same communicator. The data types must be consistent.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_matmul_all_reduce_add_rms_norm.cpp](./examples/test_aclnn_matmul_all_reduce_add_rms_norm.cpp) | Calls the MatmulAllReduceAddRmsNorm operator through the [aclnnMatmulAllReduceAddRmsNorm](docs/aclnnMatmulAllReduceAddRmsNorm_en.md) API. |
