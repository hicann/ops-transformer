# LightningIndexerKLLoss

English | [简体中文](./README.md)

## Product Support

| Product                                                     | Supported |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    x    |
| <term>Atlas A3 products</term> |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                  |    ×    |
| <term>Atlas inference products</term>                          |    ×    |
| <term>Atlas training products</term>                          |    ×    |

## Function Description

- Operator function: `lightning_indexer_kl_loss` computes the KL divergence loss between the teacher distribution and the student distribution in the Lightning Indexer.

  - **Teacher side** (target_score): the raw, unnormalized main attention scores of the compressed segments (sum ≠ 1). Use `clamp_min` to prevent NaN caused by log(0) at y = 0.
  - **Student side** (index_probs): the probability distribution after the indexer softmax. Use `+eps` to preserve the gradient as Y approaches 0.
- Computation formula:

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

  Here, $\varepsilon$ is the `eps` parameter with a default value of $10^{-9}$.

  The `weight_type` parameter controls the selection of weight:

  - `'logits'` (default): weight = y, that is, the raw unnormalized scores
  - `'probs'`: weight = P, that is, the normalized probabilities

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 150px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>target_score</td>
      <td>Input</td>
      <td>y in the formula. It represents the raw, unnormalized main attention scores on the teacher side. Empty tensors and non-contiguous tensors are not supported. The shape is (B, D) or (B, H, D).</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index_probs</td>
      <td>Input</td>
      <td>Y in the formula. It represents the probability distribution after the softmax on the student side. Empty tensors and non-contiguous tensors are not supported. The shape is the same as that of target_score.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>Output</td>
      <td>Loss value, a scalar of shape (1,). The data type is the same as that of target_score.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>Constant for numerical stability. The default value is 1e-9.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight_type</td>
      <td>Attribute</td>
      <td>Selection of the outer weight. The value can be 'logits' or 'probs'. 'logits' (default) uses the raw y as the outer weight, and 'probs' uses the normalized probability p = y / sum(y) as the outer weight.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deterministic</td>
      <td>Attribute</td>
      <td>Deterministic computation flag. The default value is false. If the value is true, a per-core workspace is used for deterministic accumulation; if the value is false, atomicAdd is used for cross-core accumulation.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## Constraints

- Empty tensors and non-contiguous tensors are not supported for input tensors.
- The shapes of target_score and index_probs must be the same. The supported shapes are (B, S, K) or (T, K). The value of B ranges from 1\~512, and the value of the last dimension K ranges from 1\~8192.
- Only the ND format is supported for data layout.
- FLOAT16, BFLOAT16, and FLOAT are supported. The data types of the two inputs must be the same.

## Invocation Example

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Invocation Method</th>
    <th class="tg-0pky">Sample Code</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8">aclnn interface</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_lightning_indexer_kl_loss.cpp">test_aclnn_lightning_indexer_kl_loss
    </a>
    </td>
    <td class="tg-lboi">
    Call the operator through the aclnnLightningIndexerKLLoss interface.
    </td>
  </tr>
  <tr>
    <td class="tg-9wq8">PyTorch API</td>
    <td class="tg-0pky">-</td>
    <td class="tg-lboi">Call the lightning_indexer_kl_loss operator through the <a href="../../attention/lightning_indexer_kl_loss/docs/torchapi_lightning_indexer_kl_loss_en.md">lightning_indexer_kl_loss</a> interface.</td>
  </tr>
</tbody></table>
