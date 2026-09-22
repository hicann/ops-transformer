# MhcPostBackward

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      √     |
|<term>Atlas A3系列产品</term>|      √     |
|<term>Atlas A2系列产品</term>|      √     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas推理系列产品</term>|      ×     |
|<term>Atlas训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：mhc_post基于一系列计算对MHC（Manifold-Constrained Hyper-Connection）架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_j$进行ResMapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。该算子实现前述过程的反向功能。

- 计算公式：
  $$
  grad\_x = H_{l}^{res} \times grad\_y\\
  grad\_h\_res = x_{l} \times {grad\_y}^{T}
  $$
  $$
  grad\_h\_out=({grad\_y} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
  grad\_h\_post=({grad\_y} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 952px"><colgroup>
<col style="width: 106px">
<col style="width: 87px">
<col style="width: 445px">
<col style="width: 209px">
<col style="width: 105px">
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
    <td>grad_y</td>
    <td>输入</td>
    <td>待计算的数据，表示网络中MHC层的输入数据</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>待计算的数据，表示网络中MHC层的输入数据</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>h_res</td>
    <td>输入</td>
    <td>mHC的h_res变换矩阵</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>h_out</td>
    <td>输入</td>
    <td>Atten/MLP层的输出</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>h_post</td>
    <td>输入</td>
    <td>MHC的h_post变换矩阵</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad_x</td>
    <td>输出</td>
    <td>网络中MHC层的输入数据x的梯度</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
   <tr>
    <td>grad_h_res</td>
    <td>输出</td>
    <td>网络中MHC层的输入数据h_res的梯度</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
   <tr>
    <td>grad_h_out</td>
    <td>输出</td>
    <td>网络中MHC层的输入数据h_out的梯度</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
   <tr>
    <td>grad_h_post</td>
    <td>输出</td>
    <td>网络中MHC层的输入数据h_post的梯度</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

无

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mhc_post_backward](examples/test_aclnn_mhc_post_backward.cpp) | 通过[aclnnMhcpostBackward](docs/aclnnMhcPostBackward.md)接口方式调用算子。|
