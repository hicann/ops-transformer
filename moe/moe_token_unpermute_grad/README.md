# MoeTokenUnpermuteGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT系列产品</term>                      |     √    |
| <term>Atlas A3系列产品</term>      |    √    |
| <term>Atlas A2系列产品</term>      |    √    |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：aclnnMoeTokenUnpermute的反向传播。
- 计算公式：

  - probs非None：

    $$
    unpermutedTokens[i] = permutedTokens[sortedIndices[i]]
    $$

    $$
    unpermutedTokens = unpermutedTokens.reshape(-1, topK\_num, hiddenSize)
    $$

    $$
    unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
    $$

    $$
    probsGrad = \sum_{k=0}^{K}(unpermutedTokens_{i,j,k})
    $$

    $$
    permutedTokensGrad[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
    $$

  - probs为None：

    $$
    permutedTokensGrad[sortedIndices[i]] = unpermutedOutputGrad[i]
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
      <td>permutedTokensOptional</td>
      <td>可选输入</td>
      <td>输入token，对应公式中的`permutedTokens`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>unpermutedTokensGrad</td>
      <td>输入</td>
      <td>正向输出unpermutedTokens的梯度，对应公式中的`unpermutedTokensGrad`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>输入</td>
      <td>对应公式中的`sortedIndices`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probs</td>
      <td>可选输入</td>
      <td>对应公式中的`probs`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>属性</td>
      <td>目前仅支持false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>restoreShapeOptional</td>
      <td>属性</td>
      <td>目前仅支持nullptr。</td>
      <td>aclIntArray*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permutedTokensGradOut</td>
      <td>输出</td>
      <td>对应公式中的`permutedTokensGrad`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsGradOut</td>
      <td>输出</td>
      <td>对应公式中的`probsGrad`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：topK_num <= 512。
- <term>Ascend 950PR&950DT系列产品</term>：
  在调用本接口时，框架内部会转调用[aclnnMoeFinalizeRoutingV2Grad](../moe_finalize_routing_v2_grad/docs/aclnnMoeFinalizeRoutingV2Grad.md)接口，如果出现参数错误提示，请参考以下参数映射关系：
  - permutedTokensOptional输入等同于aclnnMoeFinalizeRoutingV2Grad接口的expandedXOptional输入。
  - unpermutedTokensGrad输入等同于aclnnMoeFinalizeRoutingV2Grad接口的gradY输入。
  - sortedIndices输入等同于aclnnMoeFinalizeRoutingV2Grad接口的expandedRowIdx输入。
  - probsOptional输入等同于aclnnMoeFinalizeRoutingV2Grad接口的scalesOptional输入。
  - paddedMode输入等同于aclnnMoeFinalizeRoutingV2Grad接口的dropPadMode输入。
  - permutedTokensGradOut输出等同于aclnnMoeFinalizeRoutingV2Grad接口的gradExpandedXOut输出。
  - probsGradOut输出等同于aclnnMoeFinalizeRoutingV2Grad接口的gradScalesOut输出。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_token_unpermute_grad](examples/test_aclnn_moe_token_unpermute_grad.cpp) | 通过[aclnnMoeTokenUnpermuteGrad](docs/aclnnMoeTokenUnpermuteGrad.md)接口方式调用MoeTokenUnpermuteGrad算子。 |
