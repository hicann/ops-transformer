# LightningIndexerKLLoss

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    x    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- 算子功能：`lightning_indexer_kl_loss` 计算 Lightning Indexer 中 teacher 分布与 student 分布之间的 KL 散度损失函数。

  - **teacher 侧**（target_score）：压缩段未归一化的原始主注意力分数（sum ≠ 1），用 `clamp_min` 防止 y=0 处 log(0) 导致 NaN。
  - **student 侧**（index_probs）：indexer softmax 后的概率分布，用 `+eps` 保住 Y→0 处的梯度。
- 计算公式：

  $$
  y = \text{target\_score}, \quad Y = \text{index\_probs}
  $$

  $$
  P = \frac{y}{\text{sum}(y, \text{dim}=-1, \text{keepdim=True}) + \varepsilon}
  $$

  $$
  \log\_P = \log(\text{clamp\_min}(\tilde{y}, \varepsilon))
  $$

  $$
  \log\_Y = \log(Y + \varepsilon)
  $$

  $$
  \text{loss} = \sum((\log\_P - \log\_Y) \cdot \text{weight})
  $$

  其中 $\varepsilon$ 为 `eps` 参数，默认值 $10^{-9}$。

  weight 的选择由 `weight_type` 控制：

  - `'logits'`（默认）：weight = y，即原始未归一化分数
  - `'probs'`：weight = P，即归一化概率

## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 150px">
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
      <td>target_score</td>
      <td>输入</td>
      <td>公式中的 y，表示 teacher 未归一化的原始主注意力分数，不支持空 tensor 和非连续。shape 为 (B, D) 或 (B, H, D)。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index_probs</td>
      <td>输入</td>
      <td>公式中的 Y，表示 student softmax 后的概率分布，不支持空 tensor 和非连续。shape 与 target_score 一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>输出</td>
      <td>损失函数值，标量 (1,)。数据类型与 target_score 保持一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>数值稳定常数，默认值为 1e-9。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight_type</td>
      <td>属性</td>
      <td>外层权重选择，可选值为 'logits' 或 'probs'。'logits'（默认）用原始 y 作为外层权重，'probs' 用归一化概率 p = y / sum(y) 作为外层权重。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deterministic</td>
      <td>属性</td>
      <td>确定性计算标志。默认值为 false。true 时使用 per-core workspace 进行确定性累加；false 时使用 atomicAdd 进行跨核累加。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 输入 tensor 不支持空 tensor 和非连续。
- target_score 和 index_probs 的 shape 必须一致，支持 shape 为 (B, S, K) 或 (T, K)，B的取值范围为1\~512，最后一维 K 的取值范围为 1\~8192。
- 数据排布仅支持 ND 格式。
- 支持 FLOAT16、BFLOAT16、FLOAT 三种数据类型，两个输入的数据类型必须一致。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_lightning_indexer_kl_loss.cpp">test_aclnn_lightning_indexer_kl_loss
    </a>
    </td>
    <td class="tg-lboi">
    通过 aclnnLightningIndexerKLLoss 接口方式调用算子
    </td>
  </tr>
</tbody></table>
