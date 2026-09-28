# MoeTokenUnpermuteWithRoutingMapGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：aclnnMoeTokenUnpermuteWithRoutingMap的反向传播。
- **计算公式**：

  (1) probs非None：

  $$
  permutedTokensGrad[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  $$
  permutedProbsGrad = permutedTokensGrad * permutedTokensOptional
  $$

  $$
  probsGradExpertOrder = \sum_{j=0}^{hidden\_size}(permutedProbsGrad_{i,j})
  $$

    - dropAndPad为false时

  $$
  probsGradOut = masked\_scatter(routingMapOptional^T,probsGradExpertOrder)
  $$

  $$
  permutedProbs = probsOptional^T.masked\_select(routingMapOptional^T)
  $$

  $$
  permutedTokensGradOut = permutedProbs.unsqueeze(-1) * permutedTokensGrad
  $$

    - dropAndPad为true时

  $$
  probsGradOut[permuteTokenId[i], outIndex[i]/capacity] = probsGradExpertOrder[outIndex[i]]
  $$

  $$
  permutedProbs[outIndex[i]] = probsOptional.view(1)[i]
  $$

  $$
  permutedTokensGradOut = permutedProbs * permutedTokensGrad
  $$

    (2) probs为None：

  $$
  permutedTokensGradOut[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]
  $$

  1. hidden_size指unpermutedTokensGrad的第1维大小。
  2. dropAndPad等于true时，每个专家固定能够处理capacity个token。输入routingMapOptional的第1维是experts_num，即专家个数，输入outIndex的第0维是experts_num * capacity，根据这两个维度可以算出capacity。
  3. dropAndPad等于false时，每个token能被小于等于topK_num个专家处理。输入unpermutedTokensGrad的第0维是tokens_num，即token的个数，输入outIndex的第0维是tokens_num * topK_num，根据这两个维度可以算出topK_num。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1296px"><colgroup>
  <col style="width: 212px">
  <col style="width: 151px">
  <col style="width: 541px">
  <col style="width: 273px">
  <col style="width: 119px">
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
        <td>unpermutedTokensGrad</td>
        <td>输入</td>
        <td>Device侧的aclTensor。计算公式中的unpermutedTokensGrad，代表正向输出unpermutedTokens的梯度。</td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>outIndex</td>
        <td>输入</td>
        <td>Device侧的aclTensor。计算公式中outIndex，代表输出位置索引。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permuteTokenId</td>
        <td>输入</td>
        <td>Device侧的aclTensor。计算公式中的permuteTokenId，代表输入permutedTokens每个位置对应的Token序号。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>routingMapOptional</td>
        <td>可选输入</td>
        <td>Device侧的aclTensor，可选输入，当输入probsOptional为空指针时不需要此输入，应该传入空指针。计算公式中的routingMapOptional，代表对应位置的Token是否被对应专家处理。</td>
        <td>INT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permutedTokensOptional</td>
        <td>可选输入</td>
        <td>Device侧的aclTensor，可选输入，当输入probsOptional为空指针时不需要此输入，应该传入空指针。</td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>probsOptional</td>
        <td>可选输入</td>
        <td>Device侧的aclTensor，可选输入，当不需要时为空指针。</td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dropAndPad</td>
        <td>属性</td>
        <td>host侧的BOOL。true表示开启dropAndPad，false表示关闭dropAndPad。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>restoreShapeOptional</td>
        <td>属性</td>
        <td>host侧的aclIntArray。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permutedTokensGradOut</td>
        <td>输出</td>
        <td>输入permutedTokens的梯度，要求是一个2D的Tensor。</td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>probsGradOutOptional</td>
        <td>可选输出</td>
        <td>当不需要时为空指针。输入probs的梯度。</td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- 当输入probsOptional非空，且dropAndPad为false时
    - 要求topK_num <= 512且topK_num <= experts_num。
    - 要求experts_num满足(ubSize - (probTypeLen + 1) *numExpertAlign-(tokenTypeLen + 8)* 256) / (6 * tokenTypeLen + 12) >= 1，其中ubSize是芯片ub空间大小，probTypeLen是输入probsOptional的数据类型对应的字节数，tokenTypeLen是输入unpermutedTokensGrad的数据类型对应的字节数，numExpertAlign是experts_num对32做向上对齐的结果。
- 当输入probsOptional非空，且dropAndPad为true时
    - 要求capacity <= tokens_num。
    - 要求hidden_size <= 256 * (ubSize - 2080) / (8 + tokenTypeLen)，其中ubSize是芯片ub空间大小，tokenTypeLen是输入unpermutedTokensGrad的数据类型对应的字节数。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_token_unpermute_with_routing_map_grad](examples/test_aclnn_moe_token_unpermute_with_routing_map_grad.cpp) | 通过[aclnnMoeTokenUnpermuteWithRoutingMapGrad](docs/aclnnMoeTokenUnpermuteWithRoutingMapGrad.md)接口方式调用MoeTokenUnpermuteWithRoutingMapGrad算子。 |
