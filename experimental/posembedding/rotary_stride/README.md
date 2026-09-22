# 算子名称：RotaryStride

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2训练系列产品</term>                                         | 是       |

## 功能说明

- 算子功能：带跨步的位置编码。
- 计算公式：
  $$
  y_m^{i} = x_m^{i} \cos(m\theta_i) - x_m^{i+1} \sin(m\theta_i) \\
  y_m^{i+1} = x_m^{i} \sin(m\theta_i) + x_m^{i+1} \cos(m\theta_i)
  $$
- 在基本RoPE位置编码的基础上，追加了跨步功能。输入是shape为（Batch, Seqlen, Headnum, Stride）的张量，Stride为跨步大小，仅计算Stride的后head_dim维。head_dim为实际需要计算旋转位置编码的维度大小，Stride大于等于head_dim。

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>blockDim</td>
      <td>输入</td>
      <td>AI CORE的数量，比如：Ascend910B是40。</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
	<tr>
      <td>in</td>
      <td>输入</td>
      <td>公式中的输入张量x，shape为(B, S, N, stride)</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>sin</td>
      <td>输入</td>
      <td>公式中的输入张量sin，shape为(MaxS, D)，MaxS指最大序列长度</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>cos</td>
      <td>输入</td>
      <td>公式中的输入张量x，shape为(MaxS, D)，MaxS指最大序列长度</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>out</td>
      <td>输入</td>
      <td>公式中的输出张量y，shape为(B, S, N, stride)</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
	<tr>
      <td>gbD</td>
      <td>输入</td>
      <td>head_dim维度的大小</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- x/y仅支持BFLOAT16/HAFL类型。
- stride >= head_dim。

## 调用说明
```
torch.ops.npu_ops_transformer_ext.rotary_stride(blockDim, in, sin, cos, out, gbD)
```
