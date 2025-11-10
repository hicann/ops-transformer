# MoeTokenPermuteWithRoutingMap

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

算子功能：MoE的permute计算，将token和expert的标签作为routingMap传入，根据routingMap将tokens和可选probsOptional广播后排序。

计算公式：
  tokens\_num 为routingMap的第0维大小，expert\_num为routingMap的第1维大小。
 
 1.dropAndPad为`false`时：
  
  $$
  expertIndex=arrange(tokens\_num).expand(expert\_num,-1)
  $$
  
  $$
  sortedIndicesFirst=expertIndex.maskedselect(routingMap.T)
  $$
  
  $$
  sortedIndicesOut=argSort(sortedIndicesFirst)
  $$
    
  $$
  topK = numOutTokens // tokens\_num
  $$
  
  $$
  outToken = topK * tokens\_num
  $$

  $$
  permuteTokens[sortedIndicesOut[i]]=tokens[i//topK]
  $$
  
  $$
  permuteProbsOutOptional=probsOptional.T.maskedselect(routingMap.T)
  $$
  
2.dropAndPad为`true`时：
  $$
  capacity = numOutTokens // expert\_num
  $$
  $$
  outToken = capacity * expert\_num
  $$

  $$
  sortedIndicesOut = argsort(routingMap.T,dim=-1)[:, :capacity]
  $$
  
  $$
  permutedTokensOut = tokens.index_select(0, sortedIndicesOut)
  $$
  
- 如果probs不是`none`时：
  
  $$
  probs\_T\_1D = probsOptional.T.view(-1)
  $$
  
  $$
  indices\_dim0 = arange(expert\_num)
  $$
  
  $$
  indices\_dim1 = sortedIndicesOut.view(expert\_num, capacity)
  $$
  
  $$
  indices\_1D = (indices_dim0 * tokens\_num + indices\_dim1).view(-1)
  $$
  
  $$
  permuteProbsOutOptional = probs\_T\_1D.index_select(0, indices_1D)
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
   <td>permute中的输入tokens，公式中的tokens。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>routingMap</td>
   <td>输入</td>
   <td>公式中的routingMap，代表token到expert的映射关系。</td>
   <td>INT8、BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>probsOptional</td>
   <td>输入</td>
   <td>可选输入，公式中的probsOptional。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>numOutTokens</td>
   <td>属性</td>
   <td>公式中的numOutTokens，用于计算公式中topK 和capacity 的有效输出token数。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>dropAndPad</td>
   <td>属性</td>
   <td>公式中的dropAndPad，表示是否开启dropAndPad模式。</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>permutedTokensOut</td>
   <td>输出</td>
   <td>公式中的permutedTokensOut，根据indices进行扩展并排序筛选过的tokens。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndicesOut</td>
   <td>输出</td>
   <td>公式中的sortedIndicesOut，permute_tokens和tokens的映射关系。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>permuteProbsOutOptional</td>
   <td>输出</td>
   <td>公式中的permuteProbsOutOptional，根据indices进行排序并筛选过的probsOptional。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>



## 约束说明

 - tokens_num和expert_num要求小于`16777215`。
 - pad模式为false时routingMap中每行为1或true的个数固定且小于`512`。
 
## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_token_permute_with_routing_map](examples/test_aclnn_moe_token_permute_with_routing_map.cpp) | 通过[aclnnMoeTokenPermuteWithRoutingMap](docs/aclnnMoeTokenPermuteWithRoutingMap.md)接口方式调用MoeTokenPermuteWithRoutingMap算子。 |
