# FusedFloydAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      ×     |
|<term>Atlas A3系列产品</term>|     √     |
|<term>Atlas A2系列产品</term>|     √     |
|<term>Atlas 200I/500 A2推理产品</term>|     ×      |
|<term>Atlas推理系列产品</term>|     ×     |
|<term>Atlas训练系列产品</term>|     ×     |

## 功能说明

- 算子功能：训练场景下，计算Floyd注意力的反向输出，FloydAttn相较于传统FA主要是计算qk/pv注意力时会额外将seq作为batch轴从而转换为batchMatmul。

- 计算公式：

    已知注意力的正向计算公式为：

    $$
    P=Softmax(Mask(scale*(Q*K_1^T + Q*K_2^T), atten\_mask)) \\
    Y=(P*V_1+P*V_2)
    $$

    则注意力的反向计算公式为：

    $$
    S=Softmax(S)
    $$

    $$
    dV_1=P^TdY
    $$

    $$
    dV_2=P^TdY
    $$

    $$
    dQ=\frac{((dS)*K_1)}{\sqrt{d}}+\frac{((dS)*K_2)}{\sqrt{d}}
    $$

    $$
    dK_1=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

    $$
    dK_2=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的输入Q。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key1</td>
      <td>输入</td>
      <td>公式中的输入K1。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value1</td>
      <td>输入</td>
      <td>公式中的输入V1。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key2</td>
      <td>输入</td>
      <td>公式中的输入K2。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value2</td>
      <td>输入</td>
      <td>公式中的输入V2。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的输入dY。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>可选输入</td>
      <td>公式中的atten_mask，表示注意力掩码，取值为1代表该位不参与计算（不生效），为0代表该位参与计算。</td>
      <td>BOOL、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>公式中的scale，表示缩放系数，作为计算流中Muls的scalar值。</li>
          <li>默认值为1.0。</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dqOut</td>
      <td>输出</td>
      <td>公式中的dQ，表示query的梯度。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dk1Out</td>
      <td>输出</td>
      <td>公式中的dK1，表示key1的梯度。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dv1Out</td>
      <td>输出</td>
      <td>公式中的dV1，表示value1的梯度。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dk2Out</td>
      <td>输出</td>
      <td>公式中的dK2，表示key2的梯度。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dv2Out</td>
      <td>输出</td>
      <td>公式中的dV2，表示value2的梯度。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配
- 关于数据shape的约束，其中：
  - B：取值范围为1\~2K。
  - H：取值范围为1\~256。
  - N：取值范围为16\~1M且N%16==0。
  - M：取值范围为128\~1M且M%128==0。
  - K：取值范围为128\~1M且K%128==0。
  - D：取值范围为32/64/128。

- query与key1的第0/2/4轴需相同。
- key1与value1 shape需相同。
- key2与value2 shape需相同。
- query与dy/attentionIn shape需相同。
- softmaxMax与softmaxSum shape需相同。
- D只支持32/64/128。

## 调用说明

| 调用方式           | 调用样例                                                                                                              | 说明                                                                                                                    |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_fused_floyd_attention_grad](./examples/test_aclnn_fused_floyd_attention_grad.cpp)                     | 通过接口方式调用[aclnnFusedFloydAttentionGrad](./docs/aclnnFusedFloydAttentionGrad.md)算子。                   |
