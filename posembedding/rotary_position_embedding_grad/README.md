# RotaryPositionEmbeddingGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √    |
| <term>Atlas A2系列产品</term> |    √    |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×    |
| <term>Atlas推理系列产品</term>                             |    ×    |
| <term>Atlas训练系列产品</term>                              |    ×    |

## 功能说明

- **算子功能**：执行单路旋转位置编码[RotaryPositionEmbedding](../rotary_position_embedding/README.md)的反向计算。

- **计算公式**：

    取旋转位置编码的正向计算中，broadcast的轴列表为`dims`，则计算公式可表达如下：

    - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：

    （1）half模式（mode等于0）：
    $$
    dy1, dy2 = chunk(dy, chunks=2, dim=-1)
    $$

    $$
    cos1, cos2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin1, sin2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    x1, x2 = chunk(x, chunks=2, dim=-1)
    $$

    $$
    dx = cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1)
    $$

    $$
    dcos = sum(dy * x, dims)
    $$

    $$
    dsin = sum(dy * cat((-x2, x1), dim=-1), dims)
    $$

    （2）interleave模式（mode等于1）：
    $$
    dy1, dy2 = dy[..., :: 2], dy[..., 1 :: 2]
    $$

    $$
    cos1, cos2 = cos[..., :: 2], cos[..., 1 :: 2]
    $$

    $$
    sin1, sin2 = sin[..., :: 2], sin[..., 1 :: 2]
    $$

    $$
    x1, x2 = x[..., :: 2], x[..., 1 :: 2]
    $$

    $$
    dx = stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
    $$

    $$
    dcos = sum(dy * x, dims)
    $$

    $$
    dsin = sum(dy * stack((-x2, x1), dim=-1).reshape(dy.shape), dims)
    $$
    - <term>Ascend 950PR&950DT系列产品</term>：

    （3）quarter模式（mode等于2）：
    $$
    dy1, dy2, dy3, dy4 = chunk(dy, chunks=4, dim=-1)
    $$

    $$
    cos1, cos2, cos3, cos4 = chunk(cos, chunks=4, dim=-1)
    $$

    $$
    sin1, sin2, sin3, sin4 = chunk(sin, chunks=4, dim=-1)
    $$

    $$
    x1, x2, x3, x4 = chunk(x, chunks=4, dim=-1)
    $$

    $$
    dx = cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1, cos3 * dy3 + sin4 * dy4, cos4 * dy4 - sin3 * dy3), dim=-1)
    $$

    $$
    dcos = sum(dy * x, dims)
    $$

    $$
    dsin = sum(dy * cat((-x2, x1, -x4, x3), dim=-1), dims)
    $$

    （4）interleave-half模式（mode等于3）：
    $$
    dy1, dy2 = chunk(dy, chunks=2, dim=-1)
    $$

    $$
    cos1, cos2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin1, sin2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    x1, x2 = x[..., :: 2], x[..., 1 :: 2]
    $$

    $$
    dx = stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
    $$

    $$
    dcos = sum(dy * cat((x1, x2), dim=-1), dims)
    $$

    $$
    dsin = sum(dy * cat((-x2, x1), dim=-1), dims)
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
    <td>dy</td>
    <td>输入</td>
    <td>公式中的dy，表示正向计算输出y的导数。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>输入</td>
    <td>公式中的cos，正向计算输入，需与dy数据类型一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>输入</td>
    <td>公式中的sin，正向计算输入，需与dy数据类型一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>xOptional</td>
    <td>可选输入</td>
    <td>公式中的x，正向计算输入。如果为空指针，则不计算dcosOut和dsinOut。</td>
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
    <td>dxOut</td>
    <td>输出</td>
    <td>公式中的dx，输入x的导数。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dcosOut</td>
    <td>输出</td>
    <td>公式中的dcos，输入cos的导数，仅当xOptional非空时有效。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dsinOut</td>
    <td>输出</td>
    <td>公式中的dsin，输入sin的导数，仅当xOptional非空时有效。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

  - <term>Ascend 950PR&950DT系列产品</term>：
    输入张量dy支持BNSD、BSND、SBND、TND排布。各参数的shape约束可以描述如下：
    - 输入张量dy、cos、sin及输出张量dx的最后一维大小必须相同，且小于等于1024。对于half、interleave和interleave-half模式，最后一维必须能被2整除，对于quarter模式，最后一维必须能被4整除。
    - 输入张量dy和输出张量dx的shape必须完全相同。
    - 输入张量cos和sin的shape必须完全相同，cos和sin的shape需要与dy满足[broadcast关系](../../docs/zh/context/broadcast_relationship.md)，且广播后的shape必须等于dy的shape。
    - 当dy为TND时，cos、sin支持T1D、TND。

  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：

    - 输入张量dy支持BNSD、BSND、SBND、TND排布。
    - 输入张量dy、cos、sin、xOptional及输出张量dxOut、dcosOut、dsinOut的D维度大小必须相同，满足D<896，且必须为2的倍数。
    - 输入张量dy、xOptional和输出张量dxOut的shape必须完全相同。
    - 输入张量cos、sin和输出张量dcosOut、dsinOut的shape必须完全相同，且cos和sin的shape必须完全相同。
    - half模式：
      - B，N < 1000；当需要计算dsin、dcos时，B * N <= 1024
      - 当dy为BNSD时，cos、sin支持11SD、B1SD、BNSD；当cos、sin为B1SD时需满足B < S
      - 当dy为BSND时，cos、sin支持1S1D、BS1D、BSND；当cos、sin为BS1D时需满足B < S
      - 当dy为SBND时，cos、sin支持S11D、SB1D、SBND
      - 当dy为TND时，cos、sin支持T1D、TND
    - interleave模式：
      - B * N < 1000（N < 1000当dy为TND时）
      - 当dy为BNSD时，cos、sin支持11SD
      - 当dy为BSND时，cos、sin支持1S1D
      - 当dy为SBND时，cos、sin支持S11D
      - 当dy为TND时，cos、sin支持T1D

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_rotary_position_embedding_grad](./examples/test_aclnn_rotary_position_embedding_grad.cpp) | 通过[aclnnRotaryPositionEmbeddingGrad](./docs/aclnnRotaryPositionEmbeddingGrad.md)接口方式调用RotaryPositionEmbeddingGrad算子。             |
