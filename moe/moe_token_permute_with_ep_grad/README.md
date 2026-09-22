# MoeTokenPermuteWithEpGrad

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

算子功能：aclnnMoeTokenPermuteWithEp的反向传播计算。

计算公式：
  $$
  sortedIndices = sortedIndices[rangeOptional[0]<=i<rangeOptional[1]]
  $$

  $$
  tokenGradOut = permutedTokensOutputGrad.indexSelect(0, sortedIndices)
  $$

  $$
  tokenGradOut = tokenGradOut.reshape(-1, numTopk, hiddenSize)
  $$

  $$
  tokenGradOut = tokenGradOut.sum(dim = 1)
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
   <td>permutedTokensOutputGrad</td>
   <td>输入</td>
   <td>正向输出permutedTokens的梯度，公式中的`permutedTokensOutputGrad`。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndices</td>
   <td>输入</td>
   <td>正向输出的permuteTokensOut和正向输入的tokens的映射关系，公式中的`sortedIndices`。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>permutedProbsOutputGradOptional</td>
   <td>输入</td>
   <td>可选计算输入，与计算输出probsGradOut对应，传入空则不输出probsGradOut。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
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
   <td>ep切分的有效范围。</td>
   <td>aclIntArray</td>
   <td>-</td>
  </tr>
  <tr>
   <td>paddedMode</td>
   <td>属性</td>
   <td>true表示开启paddedMode，false表示关闭paddedMode，目前仅支持false。</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tokenGradOut</td>
   <td>输出</td>
   <td>输入token的梯度。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>probsGradOut</td>
   <td>输出</td>
   <td>输入probs的梯度。</td>
   <td>FLOAT、FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## 约束说明

 - numTopk <= 512。
 - 不支持paddedMode为`True`。
 - 当rangeOptional为空时，忽略permutedProbsOutputGradOptional和probsGradOut，执行逻辑回退到[aclnnMoeTokenPermuteGrad](../moe_token_permute_grad/docs/aclnnMoeTokenPermuteGrad.md)。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_token_permute_with_ep_grad.cpp](examples/test_aclnn_moe_token_permute_with_ep_grad.cpp) | 通过[aclnnMoeTokenPermuteWithEpGrad](docs/aclnnMoeTokenPermuteWithEpGrad.md)接口方式调用MoeTokenPermuteWithEpGrad算子。 |
