# QuantGroupedMatmulDequant

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      ×     |
|<term>Atlas A3系列产品</term>|      ×     |
|<term>Atlas A2系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas推理系列产品</term>|      √     |
|<term>Atlas训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：对输入x进行量化、分组矩阵乘以及反量化，主要用于MoE推理中量化权重（INT8、FRACTAL_NZ排布）的分组矩阵乘计算。

- 计算公式：

  1.若输入smooth_scale，则

  $$
      x = x\cdot scale_{smooth}
  $$

  2.若不输入x_scale，则为动态量化，需要计算x量化系数

  $$
      scale_{x}=row\_max(abs(x))/max_{quantDataType}
  $$

  3.量化

  $$
      x_{quantized}=round(x/scale_{x})
  $$

  4.分组矩阵乘+反量化
    - 4.1 若输入的$scale_{weight}$数据类型为FLOAT32，则：

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

    - 4.2 若输入的$scale_{weight}$数据类型为INT64，则说明$scale_{weight}$输入前已经和$scale_{x}$做过乘运算，算子内部计算时省略该步骤，仅支持pertensor静态量化场景：

  $$
      out = (x^{*}_{quantized}@weight_{quantized}[i] + bias) * scale_{weight}[i]
  $$

## 参数说明

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
    <th>参数名</th>
    <th style="white-space: nowrap">输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>公式中的输入x，表示输入矩阵（左矩阵），必选。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantized_weight</td>
    <td>输入</td>
    <td>公式中的$weight_{quantized}$，表示量化后的权重矩阵（右矩阵），必选。</td>
    <td>INT8</td>
    <td>FRACTAL_NZ</td>
  </tr>
  <tr>
    <td>weight_scale</td>
    <td>输入</td>
    <td>公式中的$scale_{weight}$，表示权重的量化系数，必选。数据类型为INT64时表示已与$x$的量化系数预乘，仅支持pertensor场景。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>group_list</td>
    <td>输入</td>
    <td>公式中的group，表示各分组的行数前缀和，必选。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>公式中的bias，表示矩阵乘的偏移量，可选。当前实现不支持该输入，须传空指针。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_scale</td>
    <td>输入</td>
    <td>公式中的$scale_{x}$，表示x的量化系数，可选。为空时进行动态量化；pertoken模式shape为(m,)，pertensor模式shape为(1,)。</td>
    <td>FLOAT32、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_offset</td>
    <td>输入</td>
    <td>表示x的偏移量，可选。当前实现不支持该输入，须传空指针。</td>
    <td>FLOAT32、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>smooth_scale</td>
    <td>输入</td>
    <td>公式中的$scale_{smooth}$，表示x的平滑系数，可选，shape为(k,)。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的out，表示计算结果，shape为(m,n)。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_quant_mode</td>
    <td>属性</td>
    <td>字符串型属性，指定输入x的量化模式：pertoken表示每个token（某一行）使用独立的量化参数；pertensor表示整个张量使用统一的量化参数。动态量化时仅支持pertoken。默认值为"pertoken"。</td>
    <td>String</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transpose_weight</td>
    <td>属性</td>
    <td>布尔型属性，表示输入权重是否转置，当前仅支持true。默认值为true。</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## 约束说明

- n、k都需要是16的整数倍。
- weight_scale数据类型为INT64时，必须要求x_scale数据类型为FLOAT16，且x_quant_mode为pertensor；x_scale数据类型为FLOAT16时，必须要求weight_scale数据类型为INT64，且x_quant_mode为pertensor。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_quant_grouped_matmul_dequant](examples/test_aclnn_quant_grouped_matmul_dequant.cpp) | 通过两段式接口调用[QuantGroupedMatmulDequant](docs/aclnnQuantGroupedMatmulDequant.md)算子。 |
