# MoeTokenPermuteGrad

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

算子功能：aclnnMoeTokenPermute的反向传播计算。

计算公式：
  $$
  inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)
  $$

  $$
  inputGrad = inputGrad.reshape(-1, numTopk, hiddenSize)
  $$

  $$
  inputGrad = inputGrad.sum(dim = 1)
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
   <td>permutedOutputGrad</td>
   <td>输入</td>
   <td>正向输出permutedTokens的梯度。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sortedIndices</td>
   <td>输入</td>
   <td>排序的索引值。</td>
   <td>INT32</td>
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
   <td>paddedMode</td>
   <td>属性</td>
   <td>pad模式的开关。</td>
   <td>BOOL</td>
   <td>-</td>
  </tr>
  <tr>
   <td>out</td>
   <td>输出</td>
   <td>输出token的梯度。</td>
   <td>BFLOAT16、FLOAT16、FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody></table>

## 约束说明

- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：numTopk <= 512。
- <term>Ascend 950PR&950DT系列产品</term>：
  在调用本接口时，框架内部会转调用[aclnnMoeInitRoutingV2Grad](../moe_init_routing_v2_grad/docs/aclnnMoeInitRoutingV2Grad.md)接口，如果出现参数错误提示，请参考以下参数映射关系：
  - permutedOutputGrad输入等同于aclnnMoeInitRoutingV2Grad接口的gradExpandedX输入。
  - sortedIndices输入等同于aclnnMoeInitRoutingV2Grad接口的expandedRowIdx输入。
  - numTopk输入等同于aclnnMoeInitRoutingV2Grad接口的topK输入。
  - paddedMode输入等同于aclnnMoeInitRoutingV2Grad接口的dropPadMode输入。
  - out输出等同于aclnnMoeInitRoutingV2Grad接口的out输出。

- <term>Atlas A3系列产品</term>: 单卡通信量取值范围[2MB，100MB]。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_token_permute_grad.cpp](examples/test_aclnn_moe_token_permute_grad.cpp) | 通过[aclnnMoeTokenPermuteGrad](docs/aclnnMoeTokenPermuteGrad.md)接口方式调用MoeTokenPermuteGrad算子。 |
