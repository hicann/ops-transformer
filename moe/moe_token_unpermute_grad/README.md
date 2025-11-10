# MoeTokenUnpermuteGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：aclnnMoeTokenUnpermuteGrad的反向传播。
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
    permutedTokensGrad[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddensize))[i]
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
      <td>permutedTokens</td>
      <td>输入</td>
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
      <td>probsOptional</td>
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

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：topK_num <= 512。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_token_unpermute_grad](examples/test_aclnn_moe_token_unpermute_grad.cpp) | 通过[aclnnMoeTokenUnpermuteGrad](docs/aclnnMoeTokenUnpermuteGrad.md)接口方式调用MoeTokenUnpermuteGrad算子。 |