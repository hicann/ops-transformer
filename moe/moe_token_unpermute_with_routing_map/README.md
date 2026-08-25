# MoeTokenUnpermuteWithRoutingMap

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：对经过aclnnMoeTokenpermuteWithRoutingMap处理的permutedTokens，累加回原unpermutedTokens。根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘，最后进行累加求和，并输出计算结果。
- 计算公式：

  $$
  topK\_num= permutedTokens.size(0) // routingMapOptional.size(0)
  $$

  其中`topK_num`表示每个token预留的最大专家槽位数。`paddedMode`为`false`时，每个token实际选择的专家数可以小于等于`topK_num`；未使用的槽位在`sortedIndices`中以`-1`表示，计算时跳过该槽位。

  $$
  numExperts = probs.size(1)
  $$

  $$
  numTokens = probs.size(0)
  $$

  $$
  capacity = sortedIndices.size(0) // numExperts
  $$

  (1)probs不为None，paddedMode为true时：

  $$
  permutedProbs  [i//capacity,sortedIndices[i]]=probs[i]
  $$

  $$
  permutedTokens = permutedTokens  * permutedProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (2)probs不为None，paddedMode为false时:

  $$
  permutedProbs = probs.T.maskedSelect(routingMap.T)
  $$

  $$
  permutedTokens = permutedTokens  * permutedProbs
  $$

  $$
  unpermutedTokens= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$

  $$
  if sortedIndices[i] >= 0:
      unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]] * permutedProbs[i]
  $$

  (3)probs为None,paddedMode为true时:

  $$
  permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
  $$

  $$
  unpermutedTokens[permuteTokenId[i]] += permutedTokens[outIndex[i]]
  $$

  (4)probs为None,paddedMode为false时:

  $$
  if sortedIndices[i] >= 0:
      unpermutedTokens[i//topK\_num] += permutedTokens[sortedIndices[i]]
  $$

## 参数说明

  <table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>permutedTokens</td>
      <td>输入</td>
      <td>待计算输入，对应公式中的`permutedTokens`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>输入</td>
      <td>对应公式中的`sortedIndices`。`paddedMode`为false时，非负值表示`permutedTokens`中的有效行号，-1表示该槽位没有对应专家输出，计算时跳过。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>routingMapOptional</td>
      <td>可选输入</td>
      <td>代表对应位置的Token是否被对应专家处理，对应公式中的`routingMapOptional`。</td>
      <td>INT8、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>可选输入</td>
      <td>代表对应位置的Token被对应专家处理后的结果在最终结果中的权重，对应公式中的`probs`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>属性</td>
      <td>true表示开启paddedMode，false表示关闭paddedMode。开启paddedMode时，每个专家固定处理capacity个token。关闭paddedMode时，每个token预留topK_num个专家槽位，实际被小于等于topK_num个专家处理，未使用槽位在sortedIndices中以-1表示。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional</td>
      <td>属性</td>
      <td>代表unpermutedTokens的shape。</td>
      <td>aclIntArray*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>unpermutedTokens</td>
      <td>输出</td>
      <td>对应公式中的`unpermutedTokens`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outIndex</td>
      <td>输出</td>
      <td>对应公式中的`outIndex`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permuteTokenId</td>
      <td>输出</td>
      <td>对应公式中的`permuteTokenId`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permuteProbs</td>
      <td>输出</td>
      <td>表示输出经过排序后的probs，对应公式中的`permutedProbs`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

- topkNum <= 512。paddedMode为false时，每个token最多预留topkNum个专家槽位，routingMap中每行为1或true的个数小于等于topkNum；sortedIndices中允许使用-1表示无效槽位。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_token_unpermute_with_routing_map](examples/test_aclnn_moe_token_unpermute_with_routing_map.cpp) | 通过[aclnnMoeTokenUnpermuteWithRoutingMap](docs/aclnnMoeTokenUnpermuteWithRoutingMap.md)接口方式调用MoeTokenUnpermuteWithRoutingMap算子。 |
