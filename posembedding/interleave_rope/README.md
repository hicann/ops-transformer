# InterleaveRope

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     ×    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：针对单输入x进行旋转位置编码。
- 计算公式：
  $$
  q = \text{reshape}(x, [B, N, S, D//2, 2]) \cdot \text{transpose}(-1, -2) \cdot \text{reshape}([B, N, S, D])
  $$

  $$
  q_{\text{embed}} = q \cdot \text{cos} + \text{RotateHalf}(q) \cdot \sin
  $$

  其中：RotateHalf(q)表示将q的D维后半部分元素移至前半部分并乘以 -1，后半部分用前半部分的值。

  $$
  \text{RotateHalf}(q)_{\text{i}} =
  \begin{cases}
  -q_{i+D//2} & \text{if } i < D//2 \\
  q_{i-D//2} & \text{otherwise}
  \end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
  <col style="width: 213px">
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
      <td>表示待处理张量，对应公式中的x。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>表示RoPE旋转位置的余弦分量，对应公式中的cos。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>表示RoPE旋转位置的正弦分量，对应公式中的sin。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示旋转编码后的结果，对应公式中的q_embed。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- Kirin X90/Kirin 9030处理器系列产品: 不支持BFLOAT16。

## 约束说明

  * 该接口支持推理场景下使用。
  * x，y要求为4 维张量，shape为（B，N，S，D）。
  * cos，sin要求为4 维张量，shape为（B，N，S，D），S可以为1 或与x的S相同，数据类型、数据格式与x一致。
  * 输入x、cos、sin的D维度必须等于64。
  * cos、sin的N维度必须等于1。
  * x、cos、sin、y都不支持非连续的Tensor。

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_interleave_rope](./examples/test_aclnn_interleave_rope.cpp) | 通过[aclnnInterleaveRope](./docs/aclnnInterleaveRope.md)接口方式调用InterleaveRope算子。             |
