# SwinAttentionFFN


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：全量推理场景的FlashAttention算子，支持sparse优化、支持actualSeqLengthsKv优化、支持int8量化功能，支持高精度或者高性能模式选择。

- 计算公式：


    $$
    y=x1*x2+bias +x3
    $$


## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x1</td>
    <td>输入</td>
    <td>公式中的输入Q。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x3</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>  
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明
无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_swin_attention_ffn](examples/test_geir_swin_attention_ffn.cpp)  | 通过[算子IR](op_graph/swin_attention_ffn_proto.h)构图方式调用SwinAttentionFFN算子。         |


