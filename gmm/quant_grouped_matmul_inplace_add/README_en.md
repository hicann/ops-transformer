# QuantGroupedMatmulInplaceAdd

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Functional Description

- Operator function: In micro-batch training scenarios, gradient accumulation across micro-batches is required, and a large number of fused scenarios where GroupedMatMul is followed by InplaceAdd exist. The QuantGroupedMatmulInplaceAdd operator fuses the preceding operators to improve network performance. It implements grouped matrix multiplication and addition, and its basic function is a combination of matrix multiplication and addition. For example, in the T-C quantization scenario, $y_i[m,n]=(x1_i[m,k_i] \times x2_i[k_i,n]) * scale2_i[n] * scale1_i + y_i[m,n], i=1...g$, where g is the number of groups and $m/k_i/n$ are the corresponding dimensions.

    Compared with the [GroupedMatmulV4](../grouped_matmul/docs/aclnnGroupedMatmulV4_en.md) API, **the changes of this API are:**
    - The input and output parameters are all of the aclTensor type.
    - The InplaceAdd computation is added after the GroupedMatMul computation.
    - Only quantization scenarios are supported (1. mx quantization; 2. T-C quantization; 3. T-T quantization). For quantization methods, refer to [Quantization Introduction](../../docs/zh/context/quant_mode_introduction_en.md).
    - Only inputs where x1 and x2 are FLOAT8_E5M2, FLOAT8_E4M3FN, or HIFLOAT8 are supported.

- Computation formula:
    - **mx quantization:**

    $$
     y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice_i * x2Slice_i)) * (scale1_i[m, j] * scale2_i[j, n])) + y_i[m,n]
    $$

    where gsK denotes the quantization block size along the K axis, that is, 32; $x1Slice_i$ denotes the vector of length gsK in row m of $x1_i$; $x2Slice_i$ denotes the vector of length gsK in column n of $x2_i$; slicing along the K axis starts at $j*gsK$ in both cases; the value range of j is [0, kLoops), where kLoops=ceil($K_i$ / gsK); a final slice shorter than gsK is supported.
    - **T-T/T-C quantization:**

    $$
     y_i=(x1_i\times x2_i) * scale2_i * scale1_i + y_i
    $$

## Parameter Description

<table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
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
    <th style="white-space: nowrap">Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x1</td>
    <td rowspan="1">Input</td>
    <td>Input x1 in the formula, indicating the input matrix (left matrix).</td>
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td rowspan="1">Input</td>
    <td>Input x2 in the formula, indicating the weight matrix (right matrix).</td>
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale1</td>
    <td rowspan="1">Input</td>
    <td>Scaling factor introduced by x1 quantization in the quantization parameters. For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale2</td>
    <td rowspan="1">Input</td>
    <td>Scaling factor introduced by x2 quantization in the quantization parameters. For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>group_list</td>
    <td rowspan="1">Input</td>
    <td>Distribution of matmul sizes along the group axis, in the form of a prefix-sum list of group indexes. When group_list_type is 0, the values must form a non-negative monotonic non-decreasing sequence; when group_list_type is 1, the values must be non-negative.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td rowspan="1">Input/Output</td>
    <td>Input/output y in the formula, which serves as the accumulator for InplaceAdd. After the matmul result is computed, it is added to y and written back.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>group_list_type</td>
    <td rowspan="1">Attribute</td>
    <td>Integer attribute, specifying how group_list is parsed: 0 indicates cumulative sum (cumsum), and 1 indicates the size of each group (count).</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>group_size</td>
    <td rowspan="1">Attribute</td>
    <td>Integer attribute, specifying the quantization group size along the m/n/k directions.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## Constraints

  - After 32-byte alignment, each dimension of x1 and x2 must be smaller than the maximum int32 value 2147483647, and the inner-axis size must be smaller than 2097152.
    - The input types supported in the dynamic quantization (T-T/T-C quantization) scenario are:
      - For non-empty parameters, the supported data type combinations must satisfy the following table:

        | x1       | x2  | scale2 | scale1Optional |yRef     |
        |:-------:|:-------:| :------      | :------   | :------ |
        |HIFLOAT8  |HIFLOAT8| FLOAT32    | FLOAT32   | FLOAT32 |

      - scale1Optional/scale2 must satisfy the following constraints (where g is the number of matmul groups, that is, the number of groups):

        | Parameter | Shape Restriction |
        |:---------:| :------ |
        |scale1Optional| A 2D tensor or a 1D tensor, with shape (g, 1) or (g,)|
        |scale2| In the pertensor scenario: a 2D tensor or a 1D tensor, with shape (g, 1) or (g,); in the perchannel scenario: a 2D tensor, with shape (g, N)|

    - The data types supported in the dynamic quantization (mx quantization) scenario are:
      - The data type combinations must satisfy the following table:

        | x1       | x2  |  scale2  | scale1Optional |yRef     |
        |:-------:|:-------:| :-------    | :------   | :------ |
        |FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT8_E8M0   | FLOAT8_E8M0    | FLOAT32 |

      - scale1Optional/scale2 must satisfy the following constraints (where g is the number of matmul groups, that is, the number of groups, and g\_i is the i-th group, with the subscript starting from 0):

        | Parameter | Shape Restriction |
        |:---------:| :------ |
        |scale1Optional| A 3D tensor with shape ((K / 64) + g, M, 2). The start address offset of scale\_i is ((K\_0 + K\_1 + ...+ K\_{i-1})/ 64 + g\_i) \* M \* 2, that is, the start address offset of scale_0 is 0, the start address offset of scale_1 is (K\_0 / 64 + 1) \* M \* 2, the start address offset of scale_2 is ((K\_0 + K\_1) / 64 + 2) \* M \* 2, and so on|
        |scale2| A 3D tensor with shape ((K / 64) + g, N, 2). The start address offset is determined in the same way as scale1Optional|

## Invocation Description

| Invocation Method | Invocation Sample         | Description                                                  |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_quant_grouped_matmul_inplace_add](examples/arch35/test_aclnn_quant_grouped_matmul_inplace_add.cpp) | Calls the [QuantGroupedMatmulInplaceAdd](docs/aclnnQuantGroupedMatmulInplaceAdd_en.md) operator through the API. |
