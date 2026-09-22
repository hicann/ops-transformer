# SwinTransformerLnQkvQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    ×     |
| <term>Atlas A3系列产品</term>     |    ×     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    ×     |
| <term>Kirin X90处理器系列产品</term> | √ |
| <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：Swin Transformer网络模型完成Q、K、V的计算。
- 计算公式：

    $$
    (Q,K,V)=(Quant(Layernorm(x).transpose()) * weight).dequant().transpose().split()
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
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>表示待进行归一化计算的目标张量，公式中的x， Device侧的aclTensor，数据类型支持FLOAT16。只支持维度为[B,S,H]，其中B为batch size且只支持[1,32],S为原始图像长宽的乘积，H为序列长度和通道数的乘积且小于等于1024。不支持非连续的Tensor。</td>
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
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>表示目标张量转换使用的偏移矩阵，维度只支持1维且维度为[3 * H]，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantScale</td>
    <td>输入</td>
    <td>表示目标张量量化使用的缩放参数，维度只支持1维且维度为[H]，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantOffset</td>
    <td>输入</td>
    <td>表示目标张量转换使用的偏移矩阵，维度只支持1维且维度为[H]，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dequantScale</td>
    <td>输入</td>
    <td>表示目标张量乘以权重矩阵之后反量化使用的缩放参数，维度只支持1维且维度为[3 * H]，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>UINT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>headNum</td>
    <td>输入</td>
    <td>表示转换使用的通道数；支持范围[1,32]。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>seqLength</td>
    <td>输入</td>
    <td> 表示转换使用的序列长度。只支持32/64两种。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>layernorm计算除0保护值；为了保证精度，建议小于等于1e-4。</td>
    <td>float</td>
    <td>-</td>
  </tr>
  <tr>
    <td>oriHeight</td>
    <td>输入</td>
    <td>layernorm中S轴transpose的维度；oriHeight*oriWeight需等于输入x的第二维S的大小，且为hWinSize的整数倍。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>oriWeight</td>
    <td>输入</td>
    <td>layernorm中S轴transpose的维度；oriHeight*oriWeight需等于输入x的第二维S的大小，且为wWinSize的整数倍。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>hWinSize</td>
    <td>属性</td>
    <td>使用的特征窗宽度大小；支持范围[7,32]。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>wWinSize</td>
    <td>属性</td>
    <td>使用的特征窗宽度大小；支持范围[7,32]。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>weightTranspose</td>
    <td>属性</td>
    <td>weight矩阵需要转置，当前不支持不转置场景。</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>queryOutputOut</td>
    <td>输出</td>
    <td>表示转换之后的张量，公式中的Q，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>keyOutputOut</td>
    <td>输出</td>
    <td>表示转换之后的张量，公式中的K，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>valueOutputOut</td>
    <td>输出</td>
    <td>表示转换之后的张量，公式中的V，Device侧的aclTensor。不支持非连续的Tensor。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- seqLength只支持32/64。
- oriHeight*oriWeight=输入x Tensor的第二维度，且oriHeight为hWinSize的整数倍，oriWeight为wWinSize的整数倍。
- hWinSize和wWinSize范围只支持7~32。
- 输入x Tensor的第一维度B只支持1~32。
- weight需要转置。
