# SwinAttentionFFN

## 产品支持情况

<table style="undefined;table-layout: fixed; width: 700px"><colgroup>
<col style="width: 600px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th style="text-align: center;">产品</th>
    <th style="text-align: center;">是否支持</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Ascend 950PR/Ascend 950DT</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas A3训练系列产品/Atlas A3推理系列产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas A2训练系列产品/Atlas A2推理系列产品</td>
    <td style="text-align: center;">√</td>
  </tr>
  <tr>
    <td>Atlas 200I/500 A2推理产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas 推理系列产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Atlas训练系列产品</td>
    <td style="text-align: center;">×</td>
  </tr>
  <tr>
    <td>Kirin X90处理器系列产品</td>
    <td style="text-align: center;">√</td>
  </tr>
  <tr>
    <td>Kirin 9030处理器系列产品</td>
    <td style="text-align: center;">√</td>
  </tr>
</tbody>
</table>

## 功能说明

- 算子功能：swin transformer场景下的ffn算子，用于将输入x1与x2做矩阵乘法，加上bias偏置后与x3做残差加法，输出最终结果。仅支持float16数据类型。

- 计算公式：

    $$
    y=x1*x2+bias +x3
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
<col style="width: 150px">
<col style="width: 200px">
<col style="width: 450px">
<col style="width: 200px">
<col style="width: 100px">
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
    <td>x1</td>
    <td>输入</td>
    <td>必选参数，Device侧的aclTensor，公式中的输入x1，支持输入的维度为3维[B,M,K]，其中B为batch size,[M,K]仅支持[64,128]。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>必选参数，Device侧的aclTensor，公式中的输入x2，支持输入的维度为2维[K, N]，[K, N]仅支持[128,128]。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>必选参数，Device侧的aclTensor，公式中的输入bias，支持输入的维度为1维[N]，[N]仅支持[128]。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x3</td>
    <td>输入</td>
    <td>可选参数，Device侧的aclTensor，公式中的输入x3，支持输入的维度为3维[B,M,N]，其中B为batch size,[M,N]仅支持[64,128]。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>必选参数，Device侧的aclTensor，公式中的输出y。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

  当前不支持用户直接调用
