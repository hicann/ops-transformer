# aclnnMoeTokenPermuteWithRoutingMapGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

算子功能：aclnnMoeTokenPermuteWithRoutingMap的反向传播。

计算公式：

$$
permuteTokenId, outIndex= sortedIndices.sort(dim=-1)
$$

$$
capacity = permutedTokenOutputGrad.size(0) / experts\_num
$$

- probs不为None：
  
  $$
  probsGradOutOptional = zeros(tokens\_num, experts\_num)
  $$
  
  - paddedMode为true时
  
  $$
  probsGradOutOptional [sortedIndices[i], i/capacity] = permutedProbsOutputGradOptional[i]
  $$
  
  - paddedMode为false时
  
  $$
  probsGradOutOptional = maskedscatter(probsGradOutOptional,routingMap,permutedProbsOutputGradOptional)
  $$
- probs为None：
  
  $$
  tokensGradOut= zeros(restoreShape, dtype=permutedTokens.dtype, device=permutedTokens.device)
  $$
  
  $$
  tokensGradOut[permuteTokenId[i]] += permutedTokens[outIndex[i]]
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
   <td>permutedTokenOutputGrad</td>
   <td>输入</td>
   <td>正向输出permutedTokens的梯度。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>permutedProbsOutputGradOptional</td>
   <td>输入</td>
   <td>可选输入，不传则表示不需要计算probsGradOutOptional。</td>
   <td>INT8、BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndices</td>
   <td>输入</td>
   <td>非droppad模式要求shape为一个1D的（tokens_num \* topK_num，）。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>routingMap</td>
   <td>输入</td>
   <td>代表token到expert的映射关系。</td>
   <td>INT8</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>experts_num</td>
   <td>属性</td>
   <td>参与运算的专家个数。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tokens_num</td>
   <td>属性</td>
   <td>参与运算的token个数。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>-</td>
  </tr>
  <tr>
   <td>dropAndPad</td>
   <td>属性</td>
   <td>true表示开启dropPaddedMode，false表示关闭dropPaddedMode。</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>tokensGradOut</td>
   <td>输出</td>
   <td>permutedTokens的梯度。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>probsGradOutOptional</td>
   <td>输出</td>
   <td>permutedProbs的梯度，可选输出。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>


## 约束说明

 - 非dropPaddedMode 场景topK_num <= 512。
 - 不支持混合精度输入，即permutedTokenOutputGrad、permutedProbsOutputGradOptional、tokensGradOut、probsGradOutOptional需要保持相同的数据类型。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_token_permute_with_routing_map_grad.cpp](examples/test_aclnn_moe_token_permute_with_routing_map_grad.cpp) | 通过[aclnnMoeTokenPermuteWithRoutingMap](docs/aclnnMoeTokenPermuteWithRoutingMapGrad.md)接口方式调用MoeTokenPermuteWithRoutingMap算子。 |
