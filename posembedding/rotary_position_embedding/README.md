# RotaryPositionEmbedding

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明
-  **算子功能**：执行单路旋转位置编码计算。
-  **计算公式**：

    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：

    （1）half模式（mode等于0）：
    $$
    x1 = x[..., : x.shape[-1] // 2]
    $$

    $$
    x2 = x[..., x.shape[-1] // 2 :]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1)
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$
    （2）interleave模式（mode等于1）：
    $$
    x1 = x[..., ::2].view(-1, 1)
    $$

    $$
    x2 = x[..., 1::2].view(-1, 1)
    $$    
    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    $$    
    $$
    y = x * cos + x\_rotate * sin
    $$

    
    （3）quarter模式（mode等于2）：
    $$
    x1 = x[..., : x.shape[-1] // 4]
    $$

    $$
    x2 = x[..., x.shape[-1] // 4 : x.shape[-1] // 2]
    $$    
    $$
    x3 = x[..., x.shape[-1] // 2 : x.shape[-1] // 4 * 3]
    $$    
    $$
    x4 = x[..., x.shape[-1] // 4 * 3 :]
    $$

    $$
    x\_rotate = torch.cat((-x2, x1, -x4, x3), dim=-1)
    $$

    $$
    y = x * cos + x\_rotate * sin
    $$    
    （4）interleave-half模式（mode等于3），该模式会先将奇数位的输入抽取到前半部分，将偶数位的输入抽取到后半部分，再进行half处理：
    $$
    x1 = x[..., ::2]
    $$

    $$
    x2 = x[..., 1::2]
    $$    
    $$
    x\_part1 = torch.cat((x1, x2), dim=-1)
    $$

    $$
    x\_part2 = torch.cat((-x2, x1), dim=-1)
    $$    
    $$
    y = x\_part1 * cos + x\_part2 * sin
    $$  

## 参数说明

<table style="undefined;table-layout: fixed; width: 1200px">
<colgroup>
  <col style="width: 50px">
  <col style="width: 50px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 50px">
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
    <td>x</td>
    <td>输入</td>
    <td>公式中的x，待执行旋转位置编码的张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>输入</td>
    <td>公式中的cos，参与计算的位置编码张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>输入</td>
    <td>公式中的sin，参与计算的位置编码张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mode</td>
    <td>输入</td>
    <td>公式中的旋转模式。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>公式中的y，旋转位置编码结果张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>


## 约束说明

    用(B, S, N, D)表示四维输入x的shape，在该表示下，各参数的shape约束可以描述如下：
    - 输入张量x、cos、sin及输出张量y的D维度大小必须相同，且小于等于1024。对于half、interleave和interleave-half模式，D必须能被2整除，对于quarter模式，D必须能被4整除。
    - 输入张量x和输出张量y的shape必须完全相同。
    - 输入张量cos和sin的shape必须完全相同，且必须满足下列条件之一：
      - 前三维大小都为1，即shape为(1, 1, 1, D)。
      - 前三维的大小和x前三维的大小完全相等，即shape为(B, S, N, D)。
      - 前三维中，第二维和第三维中的一个大小为1，剩余的维度及第一维大小与x的对应维度相等，即shape为(B, 1, N, D)或(B, S, 1, D)。
      - 前三维中，两个维度大小为1，剩余的一个维度大小与x的对应维度相等，即shape为(1, 1, N, D)，(1, S, 1, D)或(B, 1, 1, D)。
    当x为空tensor时，输出也为空tensor，且不受上述shape约束限制。

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    
    输入张量x支持BNSD、BSND、SBND排布。
    输入张量x、cos、sin及输出张量y的D维度大小必须相同，满足D<896，且必须为2的倍数。
    输入张量x和输出张量y的shape必须完全相同。
    输入张量cos和sin的shape必须完全相同.
    - half模式：
      - B，N < 1000;
      - 当x为(B, N, S, D)时，cos、sin支持(1, 1, S, D)、(B, 1, S, D)、(B, N, S, D)
      - 当x为(B, S, N, D)时，cos、sin支持(1, S, 1, D)、(B, S, 1, D)、(B, S, N, D)
      - 当x为(S, B, N, D)时，cos、sin支持(S, 1, 1, D)、(S, B, 1, D)、(S, B, N, D)
    - interleave模式：
      - B * N < 1000
      - 当x为(B, N, S, D)时，cos、sin支持(1, 1, S, D)
      - 当x为(B, S, N, D)时，cos、sin支持(1, S, 1, D)
      - 当x为(S, B, N, D)时，cos、sin支持(S, 1, 1, D)

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_rotary_position_embedding](./examples/test_aclnn_rotary_position_embedding.cpp) | 通过[aclnnRotaryPositionEmbedding](./docs/aclnnRotaryPositionEmbedding.md)接口方式调用RotaryPositionEmbedding算子。             |