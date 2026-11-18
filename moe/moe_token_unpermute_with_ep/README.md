# MoeTokenUnpermuteWithEp

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：根据sortedIndices存储的下标位置，获取permutedTokens中的输入数据与probs相乘，并进行合并累加。
- 计算公式：

记 sortedIndices 的第 k 个元素为 `sortedIndices[k]`，`num_tokens = len(sortedIndices) / numTopk`，`rangeOptional = [start, end)`。仅当 `start <= sortedIndices[k] < end` 时该槽位取数，取数行为 `permutedTokens[sortedIndices[k] - start]`；否则该槽位贡献 0。`rangeOptional` 为空时等价于默认值 `{0, 0}`，所有槽位均不满足条件，输出全 0。

（1）probs非None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$，$j \in {0, 1, 2, ..., numTopk - 1}$，槽位下标$k = i * numTopk + j$：

$$
out_{i} = \sum_{j=0}^{numTopk - 1} \begin{cases}
permutedTokens[sortedIndices[k] - start] * probs_{i,j}, & start \le sortedIndices[k] < end \\
0, & \text{otherwise}
\end{cases}
$$

（2）probs为None计算公式如下，其中$i \in {0, 1, 2, ..., num\_tokens - 1}$，$j \in {0, 1, 2, ..., numTopk - 1}$，槽位下标$k = i * numTopk + j$：

$$
out_{i} = \sum_{j=0}^{numTopk - 1} \begin{cases}
permutedTokens[sortedIndices[k] - start], & start \le sortedIndices[k] < end \\
0, & \text{otherwise}
\end{cases}
$$

## 参数说明

  <table style="table-layout: fixed; width: 1576px">
  <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
  </colgroup>
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
      <td>表示经过扩展并排序过的tokens，对应公式中的<code>permutedTokens</code>。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sortedIndices</td>
      <td>输入</td>
      <td>表示需要计算的数据在permutedTokens中的位置，对应公式中的<code>sortedIndices</code>。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>可选输入</td>
      <td>表示输入tokens对应的专家概率，对应公式中的<code>probs</code>。</td>
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
      <td>ep切分的有效范围[start, end)，size为2，元素类型为INT64。</td>
      <td>INT64（aclIntArray*）</td>
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
      <td>INT64（aclIntArray*）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示permutedTokens反重排的输出结果，对应公式中的<code>out</code>。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>: 不支持BFLOAT16。

## 约束说明

- numTopk必须在[1, 512]范围内。
- sortedIndices必须为1D，长度为num_tokens*numTopk，且num_tokens大于等于1，不支持空Tensor。
- rangeOptional非空时，必须满足0 <= rangeOptional[0] <= rangeOptional[1] <= rangeOptional[0] + permutedTokens的第0维长度。
- 不支持Broadcast。
- 不支持paddedMode为`True`。
- 当rangeOptional为空时，使用默认值{0,0}，输出为全0，不会回退调用其他算子。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_token_unpermute_with_ep](examples/test_aclnn_moe_token_unpermute_with_ep.cpp) | 通过[aclnnMoeTokenUnpermuteWithEp](docs/aclnnMoeTokenUnpermuteWithEp.md)接口方式调用MoeTokenUnpermuteWithEp算子。 |
