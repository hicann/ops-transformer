# SwinTransformerLnQKV

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
| <term>Ascend 950PR/Ascend 950DT</term>              |    ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
|<term>Kirin X90 处理器系列产品</term> | √ |
|<term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：完成fp16权重场景下的Swin Transformer网络模型的Q、K、V的计算。

- 计算公式：

    $$
    (Q,K,V)=((Layernorm(inputX)).transpose() * weight).transpose().split()
    $$  

  其中，weight是Q、K、V三个矩阵权重的拼接。

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
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>inputX</td>
    <td>输入</td>
    <td>表示待进行归一化计算的目标张量，公式中的inputX， Device侧的aclTensor，数据类型支持FLOAT16。只支持维度为[B,S,H]，其中B为batch size且只支持[1,32],S为原始图像长宽的乘积，H为序列长度和通道数的乘积且小于等于1024。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma</td>
    <td>输入</td>
    <td>表示layernorm计算中尺度缩放的大小，维度只支持1维且为[H]，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>表示layernorm计算中尺度偏移的大小，维度只支持1维且维度为[H]，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>输入</td>
    <td>表示目标张量转换使用的权重矩阵，维度只支持2维且维度为[H, 3 * H],Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>表示目标张量转换使用的偏移矩阵，维度只支持1维且维度为[3 * H]，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr> 
  <tr>
    <td>query_output</td>
    <td>输出</td>
    <td>表示转换之后的张量，公式中的Q，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key_output</td>
    <td>输出</td>
    <td>表示转换之后的张量，公式中的K，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value_output</td>
    <td>输出</td>
    <td>表示转换之后的张量，公式中的V，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr> 
</tbody>
</table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持FLOAT16。

## 约束说明

- 当前不支持用户通过aclnn直接调用。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_swin_transformer_ln_qkv.cpp](examples/test_geir_swin_transformer_ln_qkv.cpp) | 通过[算子IR](op_graph/swin_transformer_ln_qkv_proto.h)构图方式调用SwinTransformerLnQKV算子。 |
