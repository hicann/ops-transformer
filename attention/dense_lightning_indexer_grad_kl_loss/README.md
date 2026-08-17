# DenseLightningIndexerGradKLLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：DenseLightningIndexerGradKlLoss算子是LightningIndexer的反向算子，再额外融合了Loss计算功能。LightningIndexer算子将QueryToken和KeyToken之间的最高内在联系的TopK个筛选出来，从而减少长序列场景下Attention的计算量，加速长序列的网络的推理和训练的性能。稠密场景下的LightningIndexerGrad的输入query、key、query_index、key_index不用做稀疏化处理。

- 计算公式：

  1. Top-k value的计算公式：

  $$
  I_{t,:}=W_{t,:}@ReLU(\tilde{q}_{t,:}@\tilde{K}_{:t,:}^\top)
  $$

    - $W_{t,:}$是第$t$个token对应的$weights$；
    - $\tilde{q}_{t,:}$是$\tilde{q}$矩阵第$t$个token对应的$G$个query头合轴后的结果；
    - $\tilde{K}_{:t,:}$为$t$行$\tilde{K}$矩阵。

  2. 正向的Softmax对应公式：

  $$
  p_{t,:} = \text{Softmax}(q_{t,:} @ K_{:t,:}^\top/\sqrt{d})
  $$

    - $p_{t,:}$是第$t$个token对应的Softmax结果；
    - $q_{t,:}$是$q$矩阵第$t$个token对应的$G$个query头合轴后的结果；
    - ${K}_{:t,:}$为$t$行$K$矩阵。

  3. npu_lightning_indexer会单独训练，对应的loss function为：

  $$
  Loss{=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
  $$

  其中，$p_{t,:}$是target distribution，通过对main attention score进行所有的head的求和，然后把求和结果沿着上下文方向进行L1正则化得到。$D_{KL}$为KL散度，其表达式为：

  $$
  D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
  $$

  4. 通过求导可得Loss的梯度表达式：

  $$
  dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

  利用链式法则可以进行weights，query和key矩阵的梯度计算：
  $$
  dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left) \left) \mathop{{}}\nolimits^{\top}\right. \right. \right. \right.
  $$

  $$
  d\mathop{{\tilde{q}}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@\tilde{K}\mathop{{}}\nolimits_{{:t,:}}
  $$

  $$
  d\tilde{K}\mathop{{}}\nolimits_{{:t,:}}=\left(dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{\top}@\tilde{q}\mathop{{}}\nolimits_{{:t, :}}\right. \right.
  $$

  其中，$S$为$\tilde{q}$和$K$矩阵乘的结果。

  <!-- - **说明**：

   <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
   </blockquote> -->

## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
          <td>query</td>
          <td>输入</td>
          <td>attention结构的输入Q。</td>
          <td>FLOAT16、BFLOAT16 </td>
          <td>ND</td>
      </tr>
      <tr>
          <td>key</td>
          <td>输入</td>
          <td>attention结构的输入K。</td>
          <td>FLOAT16、BFLOAT16 </td>
          <td>ND</td>
      </tr>
      <tr>
          <td>queryIndex</td>
          <td>输入</td>
          <td>lightningIndexer结构的输入queryIndex。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>keyIndex</td>
          <td>输入</td>
          <td>lightningIndexer结构的输入keyIndex。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>weights</td>
          <td>输入</td>
          <td>权重。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxMax</td>
          <td>输入</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出。</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxSum</td>
          <td>输入</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出。</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxMaxIndex</td>
          <td>输入</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出。</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmaxSumIndex</td>
          <td>输入</td>
          <td>Device侧的aclTensor，注意力正向计算的中间输出。</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>queryRope</td>
          <td>输入</td>
          <td>MLA rope部分：Query位置编码的输出。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>keyRope</td>
          <td>输入</td>
          <td>MLA rope部分：Key位置编码的输出。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actualSeqLengthsQuery</td>
          <td>输入</td>
          <td>每个Batch中，Query的有效token数。</td>
          <td>INT64</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actualSeqLengthsKey</td>
          <td>输入</td>
          <td>每个Batch中，Key的有效token数。</td>
          <td>INT64</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>scaleValue</td>
          <td>输入</td>
          <td>缩放系数。</td>
          <td>double</td>
          <td>-</td>
      </tr>
      <tr>
          <td>layout</td>
          <td>输入</td>
          <td>layout格式。</td>
          <td>char*</td>
          <td>-</td>
      </tr>
      <tr>
          <td>sparseMode</td>
          <td>输入</td>
          <td>sparse的模式。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>preTokens</td>
          <td>输入</td>
          <td>用于稀疏计算，表示Attention需要和前几个token计算关联。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>nextTokens</td>
          <td>输入</td>
          <td>用于稀疏计算，表示Attention需要和后几个token计算关联。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dQueryIndex</td>
          <td>输出</td>
          <td>QueryIndex的梯度。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>dKeyIndex</td>
          <td>输出</td>
          <td>KeyIndex的梯度。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>dWeights</td>
          <td>输出</td>
          <td>Weights的梯度。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>loss</td>
          <td>输出</td>
          <td>损失函数值。</td>
          <td>FLOAT32</td>
          <td>ND</td>
      </tr>
      </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dense_lightning_indexer_grad_kl_loss](examples/test_aclnn_dense_lightning_indexer_grad_kl_loss.cpp) | 通过[aclnnDenseLightningIndexerGradKLLoss](docs/aclnnDenseLightningIndexerGradKLLoss.md)接口方式调用dense_lightning_indexer_grad_kl_loss算子。 |
