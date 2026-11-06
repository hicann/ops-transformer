# MatmulAllReduce

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>   | x |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | x |
| <term>Atlas inference products</term>    | √ |
| <term>Atlas training products</term> | x |

## Description

- Operator function: Fuses the MatMul computation with AllReduce communication.
- Computation formula:

  Non-quantization scenario:
  - Case 1:

    $$
    y_i=x_i\times weight_i + bias_i
    $$

  - Case 2:

    $$
    output = AllReduce(x1 @ x2 + bias + x3)
    $$

  - Case 3: Performs MatMul on the quantized inputs x1 and x2, then performs the Dequant computation, then performs the Add operation with x3, and finally performs the AllReduce computation.

    $$
    output= AllReduce(dequantScale*(x1_{int8}@x2_{int8} + bias_{int32}) + x3)
    $$

  - Case 4: Performs MatMul on the quantized inputs x1 and x2, then performs the Dequant and pertoken computations, then performs the Add operation with x3, and finally performs the AllReduce computation.

    $$
    output= AllReduce(dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional)
    $$

  - Case 5: Performs MatMul, Dequant, and pertoken computations on the quantized inputs x1 and x2, then performs the Add operation with x3, then performs perchannel quantization on the output, then performs AllToAll communication, performs the reduceSum computation on the result of the first communication, then performs AllGather communication, and finally performs Dequant on the result of the second communication to obtain the final output.

    $$
    matmulAddOutput = (dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional);
    $$

    $$
    alltoallOutPut_{int8} = AllToAll(matmulAddOutput / commQuantScale1Optional);
    $$

    $$
    reduceSumOutPut_{int8} = (add(alltoallOutPut_{int8}) * (commQuantScale1Optional / commQuantScale2Optional));
    $$

    $$
    outPut = (AllGather(reduceSumOutPut_{int8}) * commQuantScale2Optional);
    $$

  - Case 6:
    - When commQuantScale1Optional and commQuantScale2Optional are not empty:

      $$
      matmulAddOutput = (x2Scale * x1ScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional);
      $$

      $$
      alltoallOutput_{int8} = AllToAll(matmulAddOutput / commQuantScale1Optional);
      $$

      $$
      reduceSumOutput_{int8} = (add(alltoallOutput_{int8}) * (commQuantScale1Optional / commQuantScale2Optional));
      $$

      $$
      output = (AllGather(reduceSumOutput_{int8}) * commQuantScale2Optional);
      $$

    - x1 and x2 are INT8, x1ScaleOptional is absent, x2Scale is INT64/UINT64, the optional biasOptional is INT32, and out is BFLOAT16/FLOAT16:

      $$
      output = AllReduce((x1@x2 + biasOptional) * x2Scale + x3Optional)
      $$

    - x1 and x2 are INT8, x1ScaleOptional is FLOAT32, x2Scale is FLOAT32/BFLOAT16, the optional biasOptional is INT32, and out is FLOAT16/BFLOAT16:

      $$
      output = AllReduce((x1@x2 + biasOptional) * x2Scale * x1ScaleOptional + x3Optional)
      $$

    - x1 and x2 are FLOAT4_E2M1/FLOAT8_E4M3FN/FLOAT8_E5M2, x1ScaleOptional is FLOAT8_E8M0, x2Scale is FLOAT8_E8M0, the optional biasOptional is FLOAT32, and out is FLOAT16/BFLOAT16/FLOAT32:

      $$
      output = AllReduce((x1* x1ScaleOptional)@(x2* x2Scale) + biasOptional + x3Optional)
      $$

    - x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1ScaleOptional is FLOAT32, x2Scale is FLOAT32, the optional bias is FLOAT32, and out is FLOAT16/BFLOAT16/FLOAT32:

      $$
      output = AllReduce((x1@x2 + biasOptional) * x2Scale * x1ScaleOptional + x3Optional)
      $$

    - x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1ScaleOptional is FLOAT32, x2Scale is FLOAT32, and biasOptional is absent. When x1 is (a0, a1) and x2 is (b0, b1), x1ScaleOptional is (ceildiv(a0, 128), ceildiv(a1, 128)), x2Scale is (ceildiv(b0, 128), ceildiv(b1, 128)), and out is FLOAT16/BFLOAT16/FLOAT32:

      $$
      output_{pq} = AllReduce(\sum_{0}^{\left \lfloor \frac{k}{128} \right \rfloor} (x1_{pr}@x2_{rq}*(x1ScaleOptional_{pr}*x2Scale_{rq})) + x3)
      $$

  - Case 7:

    $$
    output = Allreduce(x1 @ ((x2 + antiquantOffset) *antiquantScale) + bias+ x3)
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
      <td>The left matrix of MatMul, that is, the input x1 in the formula.</td>
      <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The right matrix of MatMul, that is, the input x2 in the formula.</td>
      <td>FLOAT16, BFLOAT16, INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
      <td>ND, FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>The input bias in the formula.</td>
      <td>FLOAT16, BFLOAT16, INT32, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Optional input</td>
      <td>The Add computation after the MatMul computation, that is, the input x3 in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantScale</td>
      <td>Optional input</td>
      <td>The input antiquantScale in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantOffset</td>
      <td>Optional input</td>
      <td>The input antiquantOffset in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dequantScale</td>
      <td>Optional input</td>
      <td>The dequantization coefficient after the MatMul computation, that is, the input dequantScale in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT, UINT64, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pertoken_scale</td>
      <td>Optional input</td>
      <td>The pertoken dequantization coefficient after the MatMul computation, that is, the input pertokenScaleOptional in the formula.</td>
      <td>FLOAT, BFLOAT16, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>comm_quant_scale_1</td>
      <td>Optional input</td>
      <td>The perchannel quantization coefficient after the matmulAdd computation, that is, the input comm_quant_scale_1 in the formula.</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>comm_quant_scale_2</td>
      <td>Optional input</td>
      <td>The perchannel quantization coefficient after the allGather computation, that is, the input comm_quant_scale_2 in the formula.</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>The result of computation + communication, that is, output in the formula.</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>Attribute</td>
      <td><ul><li>A string on the host side that identifies the column group, that is, the communicator name.</li><li>Obtained through the HCCL API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</li></ul></td>
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
      <td>In pseudo-quantization pergroup mode, the groupSize input for the dequantization computation on x2.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_size</td>
      <td>Optional attribute</td>
      <td><ul><li>In the input tensors x1/x2, the number of values in the M, N, and K dimensions that jointly corresponds to one dequantization coefficient.</li><li>The default value is 0.</li><li>Reserved attribute</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>comm_quant_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>The flag for static quantization and dynamic quantization. The values are 0 and 1.</li><li>The default value is 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

* MC2 is disabled for incremental scenarios and enabled for full scenarios.
* The input x1 can be 2D or 3D, with shape (b, s, k) or (m, k). Non-contiguous input is not supported.
* The input x2 must be 2D, with shape (k, n). The k axis must meet the input requirements of the mm operator, and the k axes must be equal. m is in the range [1, 2147483647], k is in the range [0, 65535], and n is in the range [1, 65535]. When k is 0, only the empty K scenario is supported.
* Data formats of the input x2:
    * <term>Atlas A2 products</term>: Support the ND format (the current version supports only 2D input) and the FRACTAL_NZ format (the current version supports only 4D input).
    * <term>Ascend 950PR&950DT products</term>: Support the ND format.
    * When the data format of x2 is FRACTAL_NZ, use it together with the conversion to the NZ data format performed by aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight. Non-contiguous tensors support only the transpose scenario. When the data format of x2 is ND, the current version supports only 2D input.
* The passed-in x1, x2, antiquantScale, or output is not a null pointer.
* When the shape of the input x1 is (b, s, k), the shape of x3 (non-empty scenario) and the output `output` is (b, s, n), and the shape of pertoken_scale is (b*s). When the shape of the input x1 is (m, k), the shape of x3 (non-empty scenario) and the output `output` is (m, n), and the shape of pertoken_scale is (m).
* The inputs comm_quant_scale_1 and comm_quant_scale_2 are optional and can be empty. When x2 is (k, n), the shape can be (n) or (1,n).
* The input dequantScale is optional and can be empty. The shape is (1) in the pertensor scenario and (n)/(1, n) in the perchannel scenario. When the output is BFLOAT16, pass the BFLOAT16-type dequantScale directly to this API. When the output is FLOAT16, if pertokenScaleOptional is not empty, you can pass the FLOAT32-type dequantScale directly to this API. If pertokenScaleOptional is empty, you need to call the aclnn API of the TransQuantParamV2 operator in advance to convert dequantScale to the INT64/UINT64 data type.
* If bias is not empty, the current version supports only 1D, and its shape size equals the size of the last dimension of output. In the pertensor scenario, the shape of antiquantScale is (1). In the perchannel scenario, the shape is (1,n)/(n). In the pergroup scenario, the shape is (ceil(k,antiquantGroupSize), n). If antiquantOffset is not empty, its shape is the same as that of antiquantScale.
* The data types and data formats of x1, x2, x3 (non-empty scenario), antiquantScale, antiquantOffset (non-empty scenario), output, and bias (non-empty scenario) must be within the supported ranges.
* The data types of x1, antiquantScale, antiquantOffset (non-empty scenario), x3 (non-empty scenario), bias (non-empty scenario), and output are the same. When the pergroup scenario is not supported, pass 0 for antiquantGroupSize. When the pergroup scenario is supported, the value must be in the range [32, min(k-1,INT_MAX)] and be a multiple of 32. The value range of k is the same as that of the mm API.
* In the perblock scenario, group_size supports only 549764202624. In other scenarios, group_size supports only 0.
* Only the x2 matrix supports transpose/no-transpose, while the x1 matrix does not support the transpose scenario.
* The current version supports only "sum" as the input of the attribute reduceOp.
* The current version supports only 0 as the input of the attribute commTurn.
* In long-sequence scenarios, OOM errors or computation timeouts may occur as b/s or m increases.
* Only all mesh networking over hccs links is supported.
    * <term>Atlas A2 products</term>: Support 1, 2, 4, and 8 devices.
    * <term>Ascend 950PR&950DT products</term>: Support 1, 2, 4, 8, 16, 32, and 64 devices.
* <term>Atlas A2 products</term>: For the communication-computation fusion MC2 operators in one model, only the same communicator is supported.

## Calling Description

| Calling Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_matmul_all_reduce.cpp](./examples/test_aclnn_matmul_all_reduce.cpp) | Call the MatMulAllReduce operator through the [aclnnMatMulAllReduce](docs/aclnnMatmulAllReduce_en.md) API. |
