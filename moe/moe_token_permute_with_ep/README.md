# MoeTokenPermuteWithEp

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    ×     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

算子功能：MoE(Mixture of Experts)的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。

计算公式：

- paddedMode`false`时

    $$
    sortedIndicesFirst=argSort(indices)
    $$

    $$
    sortedIndicesOut=argSort(sortedIndicesFirst)
    $$

    当rangeOptional[0] <= sortedIndicesOut[i] < rangeOptional[1]时

    $$
    permuteTokensOut[sortedIndicesOut[i]-range[0]]=tokens[i//topK]
    $$

    $$
    permuteProbsOut[sortedIndicesOut[i]-rangeOptional[0]]=probsOptional[i]
    $$

  - paddedMode为`true`时

    $$
    permuteTokensOut[i]=tokens[indices[i]]
    $$

    $$
    sortedIndicesOut=indices
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
   <td>tokens</td>
   <td>输入</td>
   <td>permute中的输入tokens，公式中的`tokens`。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>indices</td>
   <td>输入</td>
   <td>输入tokens对应的专家索引，公式中的`indices`。</td>
   <td>INT32、INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>probsOptional</td>
   <td>输入</td>
   <td>可选输入，输入tokens对应的专家概率，公式中的`probsOptional`。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>rangeOptional</td>
   <td>属性</td>
   <td>ep切分的有效范围。</td>
   <td>aclIntArray</td>
   <td>-</td>
  </tr>
  <tr>
   <td>numOutTokens</td>
   <td>属性</td>
   <td>有效输出token数，在rangeOptional为空时生效。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>paddedMode</td>
   <td>属性</td>
   <td>为true时表示indices已被填充为代表每个专家选中的token索引。</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>permuteTokensOut</td>
   <td>输出</td>
   <td>indices进行扩展并排序过的tokens，公式中的`permuteTokensOut`。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndicesOut</td>
   <td>输出</td>
   <td>排序后的输出结果。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>permuteProbsOut</td>
   <td>输出</td>
   <td>permute之后的输出。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## 约束说明

- indices要求元素个数小于`16777215`，值大于等于`0`且小于`16777215`(单点支持int32或int64的最大或最小值，其余值不在范围内排序结果不正确)。
- topK小于等于`512`。
- 不支持paddedMode为`true`。
- 当rangeOptional为空时，忽略probsOptional和permuteTokensOut，执行逻辑回退到[aclnnMoeTokenPermute](../moe_token_permute/docs/aclnnMoeTokenPermute.md)。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_token_permute_with_ep.cpp](examples/test_aclnn_moe_token_permute_with_ep.cpp) | 通过[aclnnMoeTokenPermuteWithEp](docs/aclnnMoeTokenPermuteWithEp.md)接口方式调用MoeTokenPermuteWithEp算子。 |
