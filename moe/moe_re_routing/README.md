# MoeReRouting

## 产品支持情况
|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。

- 计算公式：  
    通过双重求和计算当前token在源位置的偏移量：
    $$
    SrcOffset = 
    \sum_{i=0}^{cur\_rank} \left( \sum_{j=0}^{cur\_expert} {expert\_token\_num\_per\_rank}(i,j) \right)
    $$
    通过双重求和计算当前token在目标位置的偏移量：
    $$
    DstOffset = 
    \sum_{j=0}^{cur\_expert} \left( \sum_{i=0}^{cur\_rank} {expert\_token\_num\_per\_rank}(i,j) \right)
    $$

    - SrcOffset指当前需要移动的token源偏移，根据输入expert_token_num_per_rank的值进行计算。
    - DstOffset指当前需要移动的token目的偏移。
    - cur_rank是expert_token_num_per_rank的纵轴索引，表示该token原本在的卡。
    - cur_expert是expert_token_num_per_rank的横轴索引，表示该token由卡上专家cur_expert计算。

## 参数说明
<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>tokens</td>
      <td>输入</td>
      <td>表示待重新排布的token。</td>
      <td>FLOAT16、BF16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expert_token_num_per_rank</td>
      <td>输入</td>
      <td>表示每张卡上各个专家处理的token数，对应公式中的`expert_token_num_per_rank`。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>per_token_scales</td>
      <td>可选输入</td>
      <td>表示每个token对应的scale，需要随token同样进行重新排布。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permute_tokens</td>
      <td>输出</td>
      <td>表示重新排布后的token。</td>
      <td>FLOAT16、BF16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permute_per_token_scales</td>
      <td>输出</td>
      <td>表示重新排布后的per_token_scales。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>permute_token_idx</td>
      <td>输出</td>
      <td>表示每个token在原排布方式的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expert_token_num</td>
      <td>输出</td>
      <td>表示每个专家处理的token数。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expert_token_num_type</td>
      <td>可选属性</td>
      <td>表示输出expert_token_num的模式。0为cumsum模式，1为count模式，默认值为1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>idx_type</td>
      <td>可选属性</td>
      <td>表示输出permute_token_idx的索引类型。0为gather索引，1为scatter索引，默认值为0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>


## 约束说明
- Tensor中shape使用的变量说明：
  - A：表示token个数，取值要求Sum(expert_token_num_per_rank)=A。
  - H：表示token长度，取值要求 0 < H < 16384。
  - N：表示卡数，取值无限制。
  - E：表示卡上的专家数，取值无限制。
- 输入值域限制
  - expert_token_num_type，即输出expert_token_num的模式。0为cumsum模式，1为count模式，默认值为1。当前只支持为1。
  - idx_type，即输出permute_token_idx的索引类型。0为gather索引，1为scatter索引，默认值为0。当前只支持为0。
