# GroupedMatmulFinalizeRouting

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: a fusion operator of GroupedMatmul and MoeFinalizeRouting. The output computed by GroupedMatmul is combined by index

## Parameter Description

> Superscript notes in the Data Type column: <sup>1</sup> indicates data types supported only by <term>Ascend 950PR&950DT products</term>; <sup>2</sup> indicates data types supported only by <term>Atlas A2 products</term> and <term>Atlas A3 products</term>. No superscript indicates that the data type is supported by all products. For detailed parameter constraints of each product, see the corresponding [aclnn interface document](#interface-invocation).

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 120px">
  <col style="width: 301px">
  <col style="width: 219px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
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
      <td>Input x (left matrix).</td>
      <td>INT8, FLOAT8_E5M2<sup>1</sup>, FLOAT8_E4M3FN<sup>1</sup>, HIFLOAT8<sup>1</sup>, FLOAT4_E2M1<sup>1</sup></td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input weight (right matrix)</td>
      <td>INT4<sup>2</sup>, INT8, INT32<sup>2</sup>, FLOAT8_E5M2<sup>1</sup>, FLOAT8_E4M3FN<sup>1</sup>, HIFLOAT8<sup>1</sup>, FLOAT4_E2M1<sup>1</sup></td>
      <td>ND, NZ</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>Input</td>
      <td>Scaling factor in the quantization parameters, a perchannel quantization parameter</td>
      <td>INT64<sup>2</sup>, BFLOAT16, FLOAT32, FLOAT8_E8M0<sup>1</sup></td>
      <td>ND</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Offset of the matrix</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>Input</td>
      <td>Offset of asymmetric quantization</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>Input</td>
      <td>Scaling factor of pseudo-quantization</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>Input</td>
      <td>Offset of pseudo-quantization</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pertokenScaleOptional</td>
      <td>Input</td>
      <td>Dequantization parameter of the matrix computation</td>
      <td>FLOAT32, FLOAT8_E8M0<sup>1</sup></td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupListOptional</td>
      <td>Input</td>
      <td>Matmul size distribution of the input and output along the grouping axis</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sharedInputOptional</td>
      <td>Input</td>
      <td>Output of the shared expert in the MoE computation. It needs to be combined with the output of the MoE experts</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logitOptional</td>
      <td>Input</td>
      <td>Logit value of each token computed by the MoE experts</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rowIndexOptional</td>
      <td>Input</td>
      <td>The output of the MoE experts is combined according to this rowIndex, and the values in it are the indexes used by combine to perform scatter add</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Output type of the computation: 0: FLOAT32; 1: FLOAT16; 2: BFLOAT16. Currently, only 0 is supported.</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>Attribute</td>
      <td>Coefficient used to combine the shared expert with the MoE experts. sharedInput is first multiplied by this parameter and then accumulated with the result of the MoE experts.</td>
      <td>FLOAT</td>
      <td></td>
    </tr>
    <tr>
      <td>sharedInputOffset</td>
      <td>Attribute</td>
      <td>Offset of the shared expert output in the total output.</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>transposeX</td>
      <td>Attribute</td>
      <td>Specifies whether the left matrix is transposed. Only false is supported.</td>
      <td>BOOL</td>
      <td></td>
    </tr>
    <tr>
      <td>transposeW</td>
      <td>Attribute</td>
      <td>Specifies whether the right matrix is transposed. <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support only false; <term>Ascend 950PR&950DT products</term> support true or false.</td>
      <td>BOOL</td>
      <td></td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>Attribute</td>
      <td>Grouping mode: 0: cumsum mode, that is, prefix sum; 1: count mode.</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>tuningConfigOptional</td>
      <td>Attribute</td>
      <td>The first element in the array indicates the expected number of tokens processed by each expert. During operator tiling, the tiling is properly split based on the first element of the array, achieving better performance. Elements starting from the second one are reserved. You do not need to set them, and they will be extended in the future. For backward compatibility, if you do not use this parameter, simply do not pass it in (that is, nullptr).</td>
      <td>INT64</td>
      <td></td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## Constraint Description

<details>
<summary><term>Atlas A2 products</term> and <term>Atlas A3 products</term></summary>

The inputs and outputs support the following data type combinations:

| x1   | x2         | scaleOptional | biasOptional | offsetOptional | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupListOptional | sharedInputOptional | logitOptional | rowIndexOptional | out     |
| ---- | ---------- | ------------- | ------------ | -------------- | ---------------------- | ----------------------- | --------------------- | ----------------- | ------------------- | ------------- | ---------------- | ------- |
| INT8 | INT4       | INT64         | FLOAT32      | FLOAT32        | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT32 |
| INT8 | INT4       | INT64         | FLOAT32      | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT32 |
| INT8 | INT8 (NZ) | FLOAT32       | null         | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |
| INT8 | INT8 (NZ) | FLOAT32       | null         | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |
| INT8 | INT4 (NZ) | INT64         | FLOAT32      | FLOAT32        | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |
| INT8 | INT4 (NZ) | INT64         | FLOAT32      | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT   |

</details>

<details>
<summary><term>Ascend 950PR&950DT products</term></summary>

**ND format (aclnnGroupedMatmulFinalizeRoutingV3) supported data type combinations:**

Only the MX full quantization scenario is supported. For related information, see [Quantization Introduction](../../docs/zh/context/quant_mode_introduction_en.md). offsetOptional, antiquantScaleOptional, and antiquantOffsetOptional must be set to empty.

| MX Quantization Scenario | x1                        | x2                         | scaleOptional | biasOptional     | pertokenScaleOptional | groupListOptional | sharedInputOptional | logitOptional | rowIndexOptional | out     |
| ---------- | ------------------------- | -------------------------- | ------------- | ---------------- | --------------------- | ----------------- | ------------------- | ------------- | ---------------- | ------- |
| MXFP8      | FLOAT8_E4M3FN / FLOAT8_E5M2 | FLOAT8_E4M3FN / FLOAT8_E5M2 | FLOAT8_E8M0   | BFLOAT16 / null      | FLOAT8_E8M0           | INT64             | BFLOAT16 / null         | FLOAT32       | INT64            | FLOAT32 |
| MXFP4      | FLOAT4_E2M1               | FLOAT4_E2M1                | FLOAT8_E8M0   | BFLOAT16 / null      | FLOAT8_E8M0           | INT64             | BFLOAT16 / null         | FLOAT32       | INT64            | FLOAT32 |

- The dimensions of x2 are (e,k,n), or (e,n,k) when transposed. The value range of e is [1,1024].
- The data type of scaleOptional is FLOAT8_E8M0, and its transpose attribute must be consistent with that of x2.
- The data type of pertokenScaleOptional is FLOAT8_E8M0.

**NZ format (aclnnGroupedMatmulFinalizeRoutingWeightNzV2) supported data type combinations:**

| Quantization Scenario | x1                | x2                | scale           | bias          | pertokenScaleOptional | groupList | sharedInput | logit   | rowIndex | out   |
| ------- | ----------------- | ----------------- | --------------- | ------------- | --------------------- | --------- | ----------- | ------- | -------- | ----- |
| Full quantization   | INT8              | INT8              | FLOAT/BFLOAT16      | BFLOAT16/null     | FLOAT/null            | INT64     | BFLOAT16        | FLOAT   | INT64/INT32 | FLOAT |
| Full quantization   | FLOAT8_E4M3FN     | FLOAT8_E4M3FN     | FLOAT/BFLOAT16      | BFLOAT16/null     | FLOAT/null            | INT64     | BFLOAT16        | FLOAT   | INT64     | FLOAT |
| Full quantization   | HIFLOAT8          | HIFLOAT8          | FLOAT/BFLOAT16      | BFLOAT16/null     | FLOAT/null            | INT64     | BFLOAT16        | FLOAT   | INT64     | FLOAT |
| Pseudo-quantization   | FLOAT8_E4M3FN     | FLOAT4_E2M1       | FLOAT8_E8M0     | BFLOAT16/null     | FLOAT8_E8M0           | INT64     | BFLOAT16        | FLOAT   | INT64     | FLOAT |

- The value range of e of x2 is [1,1024], and the transpose attribute can be true or false.
- In the full quantization scenario, the shape of scale is (e, 1, n), and the shape of pertokenScaleOptional is (m).
- In the pseudo-quantization scenario, the format of x2 is NZ_C0_32, transposeX2 is fixed to true, k must satisfy k%32==0, and n must satisfy n%32==0.

</details>

## Interface Invocation

| Invocation Method | Sample | Description |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_grouped_matmul_finalize_routing_v3](examples/test_aclnn_grouped_matmul_finalize_routing_v3.cpp) | Calls the GroupedMatmulFinalizeRouting operator in the mode of the [aclnnGroupedMatmulFinalizeRoutingV3](docs/aclnnGroupedMatmulFinalizeRoutingV3_en.md) API. |
| aclnn invocation | [test_aclnn_grouped_matmul_finalize_routing_weight_nz](examples/arch35/test_aclnn_grouped_matmul_finalize_routing_weightnz.cpp) | Calls the GroupedMatmulFinalizeRoutingWeightNz operator in the mode of the [aclnnGroupedMatmulFinalizeRoutingWeightNzV2](docs/aclnnGroupedMatmulFinalizeRoutingWeightNzV2_en.md) API. |
| aclnn invocation | [test_aclnn_grouped_matmul_finalize_routing_v3_mx](examples/arch35/test_aclnn_grouped_matmul_finalize_routing_v3_mx.cpp) | Calls the GroupedMatmulFinalizeRouting operator (MX quantization scenario) in the mode of the [aclnnGroupedMatmulFinalizeRoutingV3](docs/aclnnGroupedMatmulFinalizeRoutingV3_en.md) API on <term>Ascend 950PR&950DT products</term>. |
| aclnn invocation | [test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4](examples/arch35/test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4.cpp) | Calls the GroupedMatmulFinalizeRoutingWeightNz operator (MxA8W4 quantization scenario) in the mode of the [aclnnGroupedMatmulFinalizeRoutingWeightNzV2](docs/aclnnGroupedMatmulFinalizeRoutingWeightNzV2_en.md) API on <term>Ascend 950PR&950DT products</term>. |
