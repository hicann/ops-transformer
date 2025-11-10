# MoeTokenUnpermuteWithEp

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：根据sortedIndices存储的下标位置，去获取permutedTokens中的输入数据与probs相乘，并进行合并累加。
- 计算公式：

$$
sortedIndices = sortedIndices[rangeOptional[0]<=i<rangeOptional[1]]
$$

（1）probs非None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$，$j \in {0, 1, 2, ..., numTopk - 1}$，$k \in {0, 1, 2, ..., num\_tokens * numTopk}$：

$$
permutedTokens = permutedTokens.indexSelect(0, sortedIndices)
$$

$$
permutedTokens_{k} = permutedTokens_{k} * probs_{i,j}
$$

$$
out_{i} = \sum_{k=i*numTopk}^{(i+1)*numTopk - 1 } permutedTokens_{k}
$$

（2）probs为None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$，$j \in {0, 1, 2, ..., numTopk - 1}$：

$$
permutedTokens = permutedTokens.indexSelect(0, sortedIndices)
$$

$$
out_{i} = \sum_{k=i*numTopk}^{(i+1)*numTopk - 1 } permutedTokens_{k}
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
      <td>表示经过扩展并排序过的tokens，对应公式中的`permutedTokens`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>输入</td>
      <td>表示需要计算的数据在permutedTokens中的位置，对应公式中的`sortedIndices`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>可选输入</td>
      <td>表示输入tokens对应的专家概率，对应公式中的`probs`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numTopk</td>
      <td>属性</td>
      <td>被选中的专家个数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rangeOptional</td>
      <td>属性</td>
      <td>ep切分的有效范围，size为2。</td>
      <td>aclIntArray*</td>
      <td>-</td>
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
      <td>out</td>
      <td>输出</td>
      <td>表示permutedTokens反重排的输出结果，对应公式中的`out`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- topK_num <= 512。
- 不支持paddedMode为`True`。
- 当rangeOptional为空时，忽略numTopk，执行逻辑回退到[aclnnMoeTokenUnpermute](../moe_token_unpermute/docs/aclnnMoeTokenUnpermute.md)。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_token_unpermute_with_ep](examples/test_aclnn_moe_token_unpermute_with_ep.cpp) | 通过[aclnnMoeTokenUnpermuteWithEp](docs/aclnnMoeTokenUnpermuteWithEp.md)接口方式调用MoeTokenUnpermuteWithEp算子。 |