# MoeGatingTopKBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：完成MoE（Mixture of Experts）门控Top-K选择的反向梯度计算。该算子是MoeGatingTopK的反向算子，根据前向算子输出的归一化得分（xNorm）、上游梯度（gradY）和专家索引（expertIdx），计算输入得分矩阵的梯度（gradX）。支持sigmoid模式（normType=1）。
- 计算公式（sigmoid模式，normType=1）：

  1. 缩放梯度

    $$
    gradYScaled_{ip} = routedScalingFactor \cdot gradY_{ip}
    $$

  2. 正向renorm的反向传播

    $$
    wPrime_{ip} = xNorm_{i,\ expertIdx_{ip}}
    $$

    $$
    D_i = \sum_{p} wPrime_{ip} + eps
    $$

    $$
    w_{ip} = \frac{wPrime_{ip}}{D_i}
    $$

    $$
    beta_i = \sum_{p} w_{ip} \cdot gradYScaled_{ip}
    $$

    $$
    gradWPrime_{ip} = \frac{gradYScaled_{ip} - beta_i}{D_i}
    $$

  3. 散射到完整维度

    $$
    gradNormX_{ij} = \sum_{p:\ expertIdx_{ip}=j} gradWPrime_{ip}
    $$

  4. Sigmoid反向传播

    $$
    gradX_{ij} = xNorm_{ij} \cdot (1 - xNorm_{ij}) \cdot gradNormX_{ij}
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1110px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 300px">
  <col style="width: 300px">
  <col style="width: 170px">
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
    <td>x_norm</td>
    <td>输入</td>
    <td>计算的输入，对应公式中的xNorm。要求是一个2D的Tensor，维度为[M,N]。最后一维（专家数N）要求大于等于2，并小于等于2048。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad_y</td>
    <td>输入</td>
    <td>表示前向算子输出yOut的上游梯度，对应公式中的gradY。要求是一个2D的Tensor，维度为[M,K]，K的范围要求大于等于1，并小于等于N。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>expert_idx</td>
    <td>输入</td>
    <td>表示前向算子输出的top-k专家索引，对应公式中的expertIdx。shape要求与grad_y一致，维度为[M,K]。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>renorm</td>
    <td>可选属性</td>
    <td>表示前向算子在softmax模式下renorm标记。0：不做renorm；1：需要做renorm；预留参数，当前仅支持sigmoid模式。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>norm_type</td>
    <td>可选属性</td>
    <td>表示norm函数类型。1表示使用Sigmoid函数，0表示Softmax函数。当前仅支持1。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>routed_scaling_factor</td>
    <td>可选属性</td>
    <td>表示前向计算中使用的routed_scaling_factor系数，对应公式中的routedScalingFactor。默认值为1.0。</td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>eps</td>
    <td>可选属性</td>
    <td>表示前向计算使用的防除零常数，对应公式中的eps。默认值为1e-20。</td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>grad_x</td>
    <td>输出</td>
    <td>表示前向算子输入参数x的梯度，对应公式中的gradX。数据类型与grad_y需要保持一致。shape与x_norm需要一致，维度为[M,N]。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
  - x_norm最后一维的大小（即专家数N）取值范围为[2, 2048]。
  - grad_y最后一维的大小（即K）取值范围为[1, N]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_gating_top_k_backward](examples/test_aclnn_moe_gating_top_k_backward.cpp) | 通过[aclnnMoeGatingTopKBackward](docs/aclnnMoeGatingTopKBackward.md)接口方式调用MoeGatingTopKBackward算子。 |
