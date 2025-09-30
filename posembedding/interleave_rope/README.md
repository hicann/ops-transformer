# InterleaveRope

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>昇腾910_95 AI处理器</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明

- 算子功能：针对单输入 x 进行旋转位置编码。
- 计算公式：
  $$
  q = \text{reshape}(x, [B, N, S, D//2, 2]) \cdot \text{transpose}(-1, -2) \cdot \text{reshape}([B, N, S, D])
  $$

  $$
  q_{\text{embed}} = q \cdot \text{cos} + \text{RotateHalf}(q) \cdot \sin
  $$

  其中：RotateHalf(q) 表示将 q 的 D 维后半部分元素移至前半部分并乘以 -1，后半部分用前半部分的值。
  $$
  \text{RotateHalf}(q)_{\text{i}} = 
  \begin{cases} 
  -q_{i+D//2} & \text{if } i < D//2 \\
  q_{i+D//2} & \text{otherwise}
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
      <td>表示待处理张量，对应公式中的 x。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>表示 RoPE 旋转位置的余弦分量，对应公式中的 cos。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>表示 RoPE 旋转位置的正弦分量，对应公式中的 sin。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示旋转编码后的结果，对应公式中的 q_embed。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    
  </tbody></table>

## 约束说明

  * 该接口支持推理场景下使用。
  * x，y 要求为 4 维张量，shape 为（B，N，S，D）。
  * cos，sin 要求为 4 维张量，shape 为（B，N，S，D），S 可以为 1 或与 x 的 S 相同，数据类型、数据格式与 x 一致。
  * 输入x、cos、sin 的 D 维度必须等于 64。
  * cos、sin 的 N 维度必须等于 1。
  * x、cos、sin、y 都不支持非连续的 Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_interleave_rope](examples/test_geir_interleave_rope.cpp)  | 通过[算子IR](op_graph/interleave_rope_proto.h)构图方式调用InterleaveRope算子。         |