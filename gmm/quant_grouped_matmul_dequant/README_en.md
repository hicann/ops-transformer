# QuantGroupedMatmulDequant

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: performs quantization, grouped matrix multiplication, and dequantization on the input x. It is mainly used for the grouped matrix multiplication computation of quantized weights (INT8, FRACTAL_NZ layout) in MoE inference.

- Computation formula:

  1. If smooth_scale is input:

  $$
      x = x\cdot scale_{smooth}
  $$

  2. If x_scale is not input, dynamic quantization is performed, and the quantization coefficient of x needs to be computed:

  $$
      scale_{x}=row\_max(abs(x))/max_{quantDataType}
  $$

  3. Quantization:

  $$
      x_{quantized}=round(x/scale_{x})
  $$

  4. Grouped matrix multiplication + dequantization:
    - 4.1 If the data type of the input $scale_{weight}$ is FLOAT32:

  $$
    \begin{aligned}
      x^{*}_{quantized} &= x_{quantized}[group[i-1]:group[i]]\\
      out^{*}_{quantized} &= out_{quantized}[group[i-1]:group[i]]\\
      scale^{*}_{x} &= \begin{cases}
                      scale_{x}[group[i-1]:group[i]] & pertoken \\
                      scale_{x} & pertensor \\
                      \end{cases} \\
      out &= (x^{*}_{quantized}@weight_{quantized}[i] + bias) * scale_{weight}[i] * scale^{*}_{x}
    \end{aligned}
  $$

    - 4.2 If the data type of the input $scale_{weight}$ is INT64, it indicates that $scale_{weight}$ has already been multiplied by $scale_{x}$ before being input, and this step is omitted in the internal computation of the operator. Only the pertensor static quantization scenario is supported:

  $$
      out = (x^{*}_{quantized}@weight_{quantized}[i] + bias) * scale_{weight}[i]
  $$

## Parameter Description

<table style="undefined;table-layout: fixed;width: 1200px"><colgroup>
<col style="width: 170px">
<col style="width: 120px">
<col style="width: 380px">
<col style="width: 220px">
<col style="width: 160px">
<col style="width: 150px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th style="white-space: nowrap">Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>Input x in the formula, indicating the input matrix (left matrix). This parameter is required.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantized_weight</td>
    <td>Input</td>
    <td>$weight_{quantized}$ in the formula, indicating the quantized weight matrix (right matrix). This parameter is required.</td>
    <td>INT8</td>
    <td>FRACTAL_NZ</td>
  </tr>
  <tr>
    <td>weight_scale</td>
    <td>Input</td>
    <td>$scale_{weight}$ in the formula, indicating the quantization coefficient of the weight. This parameter is required. When the data type is INT64, it indicates that the coefficient has been pre-multiplied with the quantization coefficient of $x$, and only the pertensor scenario is supported. For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>group_list</td>
    <td>Input</td>
    <td>group in the formula, indicating the prefix sum of the number of rows in each group. This parameter is required.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Input</td>
    <td>bias in the formula, indicating the offset of the matrix multiplication. This parameter is optional. The current implementation does not support this input, and a null pointer must be passed.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_scale</td>
    <td>Input</td>
    <td>$scale_{x}$ in the formula, indicating the quantization coefficient of x. This parameter is optional. When it is empty, dynamic quantization is performed; in pertoken mode, the shape is (m,), and in pertensor mode, the shape is (1,).</td>
    <td>FLOAT32, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_offset</td>
    <td>Input</td>
    <td>Indicates the offset of x. This parameter is optional. The current implementation does not support this input, and a null pointer must be passed.</td>
    <td>FLOAT32, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>smooth_scale</td>
    <td>Input</td>
    <td>$scale_{smooth}$ in the formula, indicating the smoothing coefficient of x. This parameter is optional, with the shape (k,).</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>out in the formula, indicating the computation result, with the shape (m,n).</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_quant_mode</td>
    <td>Attribute</td>
    <td>String attribute, specifying the quantization mode of the input x: pertoken indicates that each token (a row) uses independent quantization parameters; pertensor indicates that the entire tensor uses unified quantization parameters. Only pertoken is supported for dynamic quantization. The default value is "pertoken".</td>
    <td>String</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transpose_weight</td>
    <td>Attribute</td>
    <td>Boolean attribute, indicating whether the input weight is transposed. Currently, only true is supported. The default value is true.</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## Constraints

- Both n and k must be integer multiples of 16.
- When the data type of weight_scale is INT64, the data type of x_scale must be FLOAT16 and x_quant_mode must be pertensor; when the data type of x_scale is FLOAT16, the data type of weight_scale must be INT64 and x_quant_mode must be pertensor.

## Invocation Description

| Invocation Method      | Sample Code                 | Description                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn call | [test_aclnn_quant_grouped_matmul_dequant](examples/test_aclnn_quant_grouped_matmul_dequant.cpp) | Invokes the QuantGroupedMatmulDequant operator through the two-phase [QuantGroupedMatmulDequant](docs/aclnnQuantGroupedMatmulDequant_en.md) API. |
