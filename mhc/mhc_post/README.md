# MhcPost

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：MhcPost基于一系列计算对mHC架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_l$进行Res Mapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。

- 计算公式：

  当 h_res 提供时：

  $$
  x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}
  $$

  当 h_res 缺省时（仅 Ascend 950 支持）：

  $$
  x_{l+1} = x_l + h_{l}^{out} \otimes H_{t}^{post}
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
      <td>x</td>
      <td>输入</td>
      <td>待计算的张量，表示网络中mHC层的输入数据。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_res</td>
      <td>输入（可选）</td>
      <td>mHC的h_res变换矩阵，是做完sinkhorn变换后的双随机矩阵。缺省时退化为直接残差连接（仅 Ascend 950 支持）。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_out</td>
      <td>输入</td>
      <td>Atten/MLP层的输出。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_post</td>
      <td>输入</td>
      <td>mHC的h_post变换矩阵。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>网络中mHC层的输出数据，作为下一层的输入。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：h_res 支持传入 nullptr，此时退化为直接残差连接。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：h_res 为必传参数，不支持传入 nullptr。

  | 规格项   | 规格               | 规格说明                                |
  | :------- | :----------------- | :------------------------------------- |
  | n        | 4                  | 固定为4                              |
  | d        | 范围1到100000       | 128的倍数                             |

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mhc_post](examples/test_aclnn_mhc_post.cpp) | 通过[aclnnMhcPost](docs/aclnnMhcPost.md)接口方式调用MhcPost算子。 |
| PyTorch API | - | 通过[mhc_post](../../mhc/mhc_post/docs/torchapi_mhc_post.md)接口方式调用MhcPost算子。 |
