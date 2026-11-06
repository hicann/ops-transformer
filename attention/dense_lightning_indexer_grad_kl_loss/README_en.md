# DenseLightningIndexerGradKLLoss

English | [简体中文](./README.md)

## Product Support Status

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: The DenseLightningIndexerGradKlLoss operator is the backward operator of LightningIndexer and additionally integrates the loss computation function. The LightningIndexer operator selects the TopK tokens with the strongest intrinsic relevance between QueryToken and KeyToken, reducing the computation workload of Attention in long-sequence scenarios and accelerating the inference and training performance of long-sequence networks. In dense scenarios, the inputs query, key, query_index, and key_index of LightningIndexerGrad do not require sparsification processing.

- Calculation formulas:

  1. Formula for calculating the Top-k value:

  $$
  I_{t,:}=W_{t,:}@ReLU(\tilde{q}_{t,:}@\tilde{K}_{:t,:}^\top)
  $$

    - $W_{t,:}$ is the $weights$ corresponding to the $t$-th token;
    - $\tilde{q}_{t,:}$ is the result of merging the $G$ query heads corresponding to the $t$-th token of the $\tilde{q}$ matrix into one axis;
    - $\tilde{K}_{:t,:}$ is the $\tilde{K}$ matrix of $t$ rows.

  2. Formula corresponding to the forward Softmax:

  $$
  p_{t,:} = \text{Softmax}(q_{t,:} @ K_{:t,:}^\top/\sqrt{d})
  $$

    - $p_{t,:}$ is the Softmax result corresponding to the $t$-th token;
    - $q_{t,:}$ is the result of merging the $G$ query heads corresponding to the $t$-th token of the $q$ matrix into one axis;
    - ${K}_{:t,:}$ is the $K$ matrix of $t$ rows.

  3. npu_lightning_indexer is trained separately, and the corresponding loss function is:

  $$
  Loss{=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
  $$

  Here, $p_{t,:}$ is the target distribution, obtained by summing the main attention score over all heads and then applying L1 normalization to the sum along the context direction. $D_{KL}$ is the KL divergence, expressed as:

  $$
  D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
  $$

  4. The gradient expression of the Loss is obtained by differentiation:

  $$
  dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

  The gradients of the weights, query, and key matrices can be computed using the chain rule:
  $$
  dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left) \left) \mathop{{}}\nolimits^{\top}\right. \right. \right. \right.
  $$

  $$
  d\mathop{{\tilde{q}}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@\tilde{K}\mathop{{}}\nolimits_{{:t,:}}
  $$

  $$
  d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}=\left(dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{\top}@\tilde{q}\mathop{{}}\nolimits_{{:t, :}}\right. \right.
  $$

  Here, $S$ is the result of the matrix multiplication of $\tilde{q}$ and $K$.

  <!-- - **Description**:

   <blockquote>The query, key, and value data layout formats can be interpreted from multiple dimensions, where B (Batch) indicates the batch size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the minimum unit size of the hidden layer, D=H/N is satisfied, and T indicates the sum of sequence lengths of input samples across all batches.
   </blockquote> -->

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
      <tbody>
      <tr>
          <td>query</td>
          <td>Input</td>
          <td>Input Q of the attention structure.</td>
          <td>FLOAT16, BFLOAT16 </td>
          <td>ND</td>
      </tr>
      <tr>
          <td>key</td>
          <td>Input</td>
          <td>Input K of the attention structure.</td>
          <td>FLOAT16, BFLOAT16 </td>
          <td>ND</td>
      </tr>
      <tr>
          <td>queryIndex</td>
          <td>Input</td>
          <td>Input queryIndex of the lightningIndexer structure.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>keyIndex</td>
          <td>Input</td>
          <td>Input keyIndex of the lightningIndexer structure.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>weights</td>
          <td>Input</td>
          <td>Weights.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxMax</td>
          <td>Input</td>
          <td>aclTensor on the Device side. Intermediate output of the forward attention computation.</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxSum</td>
          <td>Input</td>
          <td>aclTensor on the Device side. Intermediate output of the forward attention computation.</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxMaxIndex</td>
          <td>Input</td>
          <td>aclTensor on the Device side. Intermediate output of the forward attention computation.</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxSumIndex</td>
          <td>Input</td>
          <td>aclTensor on the Device side. Intermediate output of the forward attention computation.</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>queryRope</td>
          <td>Input</td>
          <td>MLA rope part: Output of the Query position encoding.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>keyRope</td>
          <td>Input</td>
          <td>MLA rope part: Output of the Key position encoding.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actualSeqLengthsQuery</td>
          <td>Input</td>
          <td>Number of valid Query tokens in each batch.</td>
          <td>INT64</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actualSeqLengthsKey</td>
          <td>Input</td>
          <td>Number of valid Key tokens in each batch.</td>
          <td>INT64</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>scaleValue</td>
          <td>Input</td>
          <td>Scaling factor.</td>
          <td>double</td>
          <td>-</td>
      </tr>
      <tr>
          <td>layout</td>
          <td>Input</td>
          <td>Layout format.</td>
          <td>char*</td>
          <td>-</td>
      </tr>
      <tr>
          <td>sparseMode</td>
          <td>Input</td>
          <td>Sparse mode.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>preTokens</td>
          <td>Input</td>
          <td>Used for sparse computation. Indicates that Attention needs to compute relevance with the preceding tokens.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>nextTokens</td>
          <td>Input</td>
          <td>Used for sparse computation. Indicates that Attention needs to compute relevance with the following tokens.</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dQueryIndex</td>
          <td>Output</td>
          <td>Gradient of QueryIndex.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>dKeyIndex</td>
          <td>Output</td>
          <td>Gradient of KeyIndex.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>dWeights</td>
          <td>Output</td>
          <td>Gradient of Weights.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>loss</td>
          <td>Output</td>
          <td>Loss function value.</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      </tbody>
  </table>

## Constraints

None

## Invocation Description

| Invocation Method | Invocation Sample                                                            | Description                                                            |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_dense_lightning_indexer_grad_kl_loss](examples/test_aclnn_dense_lightning_indexer_grad_kl_loss.cpp) | Calls the dense_lightning_indexer_grad_kl_loss operator through the [aclnnDenseLightningIndexerGradKLLoss](docs/aclnnDenseLightningIndexerGradKLLoss_en.md) API. |
