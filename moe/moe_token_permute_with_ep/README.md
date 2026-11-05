# MoeTokenPermuteWithEp

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

算子功能：MoE(Mixture of Experts)的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。

计算公式：

- paddedMode为`false`时，公式如下。topK表示每个token选择的专家数量：indices为2维时topK等于indices最后一维的大小，indices为1维时topK为1。

    $$
    sortedIndicesFirst=argSort(flatten(indices))
    $$

    $$
    sortedIndicesOut=argSort(sortedIndicesFirst)
    $$

    当rangeOptional[0] <= sortedIndicesOut[i] < rangeOptional[1]时

    $$
    permuteTokensOut[sortedIndicesOut[i]-rangeOptional[0]]=tokens[i//topK]
    $$

    $$
    permuteProbsOut[sortedIndicesOut[i]-rangeOptional[0]]=probsOptional[i]
    $$

- paddedMode为`true`时（暂不支持）

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
   <td>可选输入</td>
   <td>输入tokens对应的专家概率，公式中的`probsOptional`。数据类型需与tokens相同。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>rangeOptional</td>
   <td>可选属性</td>
   <td>ep切分的有效范围，公式中的`rangeOptional`。默认值为[0,0]。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>numOutTokens</td>
   <td>可选属性</td>
   <td>输出token数量相关参数。仅当rangeOptional为空时生效，此时切片起点为0、终点为该参数值；rangeOptional非空时该参数被忽略。默认值为0。需要指定输出范围时，请使用rangeOptional。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>paddedMode</td>
   <td>可选属性</td>
   <td>为true时表示indices已被填充为代表每个专家选中的token索引，当前仅支持false。默认值为false。</td>
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

- tokens要求为2D，shape为`(num_tokens, hidden_size)`，且第0维与indices第0维相同；indices要求为1D或2D。
- indices要求元素个数（`num_tokens * topK`）小于`16777215`，即不超过`16777214`；元素值要求大于等于`0`且小于等于`16777214`。除该值域外，仅单点支持int32或int64的最大值或最小值（用作无效位置标记），其余越界值的排序结果不正确。
- topK表示每个token选择的专家数量，indices为2维时topK等于indices最后一维的大小，indices为1维时topK为1；topK取值范围为`[1, 512]`。
- 输入tokens、indices不支持空tensor（`num_tokens`与`hidden_size`均需大于0，`num_tokens`为0的调用不在支持范围内）；probsOptional为可选输入，可不传入，传入时其shape与indices相同且不得为空tensor。
- probsOptional传入时，其数据类型需与tokens相同；不传入时不输出permuteProbsOut，此时permuteProbsOut不参与计算与比对。
- rangeOptional非空时，size为`2`，两个元素start、end均要求在`[0, num_tokens*topK]`范围内，否则返回失败。
- numOutTokens仅在rangeOptional为空时生效：取值为`0`时表示不删除任何token（输出`num_tokens*topK`行），取值为正整数时保留排序结果的前numOutTokens行；取值范围为`[0, num_tokens*topK]`，不支持负数与超过上限的取值。
- 不支持paddedMode为`true`。
- 当rangeOptional为空时，忽略probsOptional和permuteProbsOut，执行逻辑回退到[aclnnMoeTokenPermute](../moe_token_permute/docs/aclnnMoeTokenPermute.md)。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_moe_token_permute_with_ep.cpp](examples/test_aclnn_moe_token_permute_with_ep.cpp) | 通过[aclnnMoeTokenPermuteWithEp](docs/aclnnMoeTokenPermuteWithEp.md)接口方式调用MoeTokenPermuteWithEp算子。 |
