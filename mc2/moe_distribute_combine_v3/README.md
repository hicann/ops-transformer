# MoeDistributeCombineV3

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950DT</term>                             |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                               |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：进行AllToAllV通信，最后将接收的数据整合（乘权重再相加）。TP（Tensor Parallelism）域通信当前版本不再支持，TP相关参数为预留参数。

    相较于MoeDistributeCombineV2算子，该算子变更如下：
    - 新增`context`入参，存入通信域相关信息；
    - 新增`ccl_buffer_size`入参，指定当前通信域大小；
    - 减少`group_ep`以及`group_tp`通信域名称入参;
    详细说明请参考以下参数说明。
- 计算公式：

    $$
    ataOut = AllToAllV(expand_x)\\
    x_out = Sum(expert_scales * ataOut + expert_scales * sharedExpertX)
    $$

    注意该算子必须与MoeDistributeDispatchV3配套使用，相当于按MoeDistributeDispatchV3算子收集数据的路径原路返还。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>context</td>
   <td>输入</td>
   <td>本卡通信域信息数据。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expand_x</td>
   <td>输入</td>
   <td>根据expertIds进行扩展过的token特征。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_ids</td>
   <td>输入</td>
   <td>每个token的topK个专家索引。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>assist_info_for_combine</td>
   <td>输入</td>
   <td>表示同一专家收到的token个数。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_send_counts</td>
   <td>输入</td>
   <td>从EP通信域各卡接收的token数。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_scales</td>
   <td>输入</td>
   <td>每个token的topK个专家的权重。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_send_counts_optional</td>
   <td>可选输入</td>
   <td>预留参数，TP域通信不再支持，传空指针即可。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x_active_mask_optional</td>
   <td>可选输入</td>
   <td>表示token是否参与通信，可传有效数据或空指针；1D时true需排在false前（例：{true, false, true}非法），2D时token对应K个值全为false则不参与通信；默认所有token参与通信；各卡BS不一致时所有token需有效。</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activation_scale_optional</td>
   <td>可选输入</td>
   <td>预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight_scale_optional</td>
   <td>可选输入</td>
   <td>预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>group_list_optional</td>
   <td>可选输入</td>
   <td>预留参数，当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expand_scales_optional</td>
   <td>可选输入</td>
   <td>表示本卡token的权重，对应MoeDistributeDispatchV3中的expand_scales输出。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shared_expert_x_optional</td>
   <td>可选输入</td>
   <td>表示共享专家计算后的token，数据类型需与expand_x一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elastic_info_optional</td>
   <td>可选输入</td>
   <td>EP通信域动态缩容信息。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ori_x_optional</td>
   <td>可选输入</td>
   <td>表示未经过FFN（Feed-Forward Neural network）的token数据，在开启copyExpert或开启constExpert的场景下需要本输入数据。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>const_expert_alpha_1_optional</td>
   <td>可选输入</td>
   <td>在开启constExpert的场景下需要输入的计算系数。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>const_expert_alpha_2_optional</td>
   <td>可选输入</td>
   <td>在开启constExpert的场景下需要输入的计算系数。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>const_expert_v_optional</td>
   <td>可选输入</td>
   <td>在开启constExpert的场景下需要输入的计算系数。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>performance_info_optional</td>
   <td>可选输入</td>
   <td>表示本卡等待各卡数据的通信时间，单位为us（微秒）。单次算子调用各卡通信耗时会累加到该Tensor上，算子内部不进行自动清零，因此用户每次启用此Tensor开始记录耗时前需对Tensor清零。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_world_size</td>
   <td>属性</td>
   <td>EP通信域大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_rank_id</td>
   <td>属性</td>
   <td>EP域本卡Id，取值范围[0, ep_world_size)，同一个EP通信域中各卡的ep_rank_id不重复。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moe_expert_num</td>
   <td>属性</td>
   <td>MoE专家数量，满足moe_expert_num % (ep_world_size - shared_expert_num) = 0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ccl_buffer_size</td>
   <td>属性</td>
   <td><li>当前通信域Buffer大小。</li><li>默认值为""。</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_world_size</td>
   <td>可选属性</td>
   <td><li>TP通信域大小（预留参数，当前版本不支持，仅允许0或1）。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_rank_id</td>
   <td>可选属性</td>
   <td><li>TP域本卡Id（预留参数，当前版本不支持，传0即可）。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_shard_type</td>
   <td>可选属性</td>
   <td><li>表示共享专家卡分布类型，当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shared_expert_num</td>
   <td>可选属性</td>
   <td><li>表示共享专家数量（一个共享专家可复制部署到多个卡上）。</li><li>默认值为1。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shared_expert_rank_num</td>
   <td>可选属性</td>
   <td><li>表示共享专家卡数量，取值范围[0, ep_world_size)；为0时需满足shared_expert_num为0或1，不为0时需满足shared_expert_rank_num % shared_expert_num = 0。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>global_bs</td>
   <td>可选属性</td>
   <td><li>EP域全局的batch size大小；各rank BS一致时，global_bs = BS * ep_world_size或0；各rank BS不一致时，global_bs = max_bs * ep_world_size（max_bs为单卡BS最大值）。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>out_dtype</td>
   <td>可选属性</td>
   <td><li>用于指定输出x的数据类型，预留参数，当前版本不支持，传0即可。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>comm_quant_mode</td>
   <td>可选属性</td>
   <td><li>通信量化类型，取值范围0或2；0表示通信不量化，2表示通信int8量化。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>group_list_type</td>
   <td>可选属性</td>
   <td><li>group_list格式，预留参数，当前版本不支持，传0即可。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>comm_alg</td>
   <td>可选属性</td>
   <td><li>表示通信亲和内存布局算法。</li><li>默认值为""。</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>zero_expert_num</td>
   <td>可选属性</td>
   <td><li>零专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>copy_expert_num</td>
   <td>可选属性</td>
   <td><li>copy专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>const_expert_num</td>
   <td>可选属性</td>
   <td><li>常量专家数量。</li><li>默认值为0。</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x_out</td>
   <td>输出</td>
   <td>表示处理后的token，数据类型、数据格式与expand_x保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
    * 不支持`expand_scales_optional`。
    * 不支持`comm_alg`。

## 约束说明

- `MoeDistributeDispatchV3`与`CombineV3`系列算子必须配套使用，具体参考调用示例。

- 在不同产品型号、不同通信算法或不同版本中，`MoeDistributeDispatchV3`的Tensor输出`assist_info_for_combine_out`、`ep_recv_counts_out`、`expand_scales_out`中的元素值可能不同，使用时直接将上述Tensor传给`CombineV3`系列算子对应参数即可，模型其他业务逻辑不应对其存在依赖。`tp_recv_counts_out`为预留输出，当前版本不支持，传空指针即可。

- 调用算子过程中使用的`ep_world_size`、`moe_expert_num`、`ccl_buffer_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_num`、`global_bs`、`comm_alg`参数取值所有卡需保持一致，网络中不同层中也需保持一致，且和`MoeDistributeDispatchV3`算子对应参数也保持一致。`tp_world_size`为预留参数，当前版本不支持。

- 参数说明里shape格式说明：
    - `A`：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足`A` = `BS` *`ep_world_size`* `shared_expert_num` / `shared_expert_num`。
        - 对于MoE专家，当`global_bs`为0时，要满足`A` >= `BS` *`ep_world_size`* min(`local_expert_num`, `K`)；当`global_bs`非0时，要满足`A` >= `global_bs` * min(`local_expert_num`, `K`)。
    - `K`：表示选取topK个专家，取值范围为0 < `K` ≤ 16同时满足0 < `K` ≤ `moe_expert_num` + `zero_expert_num` + `copy_expert_num` + `const_expert_num`。
    - `local_expert_num`：表示本卡专家数量。
        - 对于共享专家卡，`local_expert_num` = 1
        - 对于MoE专家卡，`local_expert_num` = `moe_expert_num` / (`ep_world_size` - `shared_expert_num`)。当前版本不支持TP域通信。

- 参数约束：
    - `zero_expert_num`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1,合法的零专家的ID的值是[`moe_expert_num`, `moe_expert_num` + `zero_expert_num`)。
    - `copy_expert_num`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1,合法的copy专家的ID的值是[`moe_expert_num` + `zero_expert_num`, `moe_expert_num` + `zero_expert_num` + `copy_expert_num`)。
    - `const_expert_num`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1,合法的常量专家的ID的值是[`moe_expert_num` + `zero_expert_num` + `copy_expert_num`, `moe_expert_num` + `zero_expert_num` + `copy_expert_num` + `const_expert_num`)。
    - `ori_x_optional`：可选择传入有效数据或填空指针，当`copy_expert_num`不为0或`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`BS`, `H`)，数据类型需与`expand_x`保持一致。
    - `const_expert_alpha_1_optional`：可选择传入有效数据或填空指针，当`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`const_expert_num`,)，数据类型需与`expand_x`保持一致。
    - `const_expert_alpha_2_optional`：可选择传入有效数据或填空指针，当`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`const_expert_num`,)，数据类型需与`expand_x`保持一致。
    - `const_expert_v_optional`：可选择传入有效数据或填空指针，当`const_expert_num`不为0时必须传入有效输入；当传入有效数据时，要求shape为(`const_expert_num`, `H`)，数据类型需与`expand_x`保持一致。

- 本文公式中的"/"表示整除。
- 通信域使用约束：
    - 一个模型中的`MoeDistributeCombineV3`和`MoeDistributeDispatchV3`仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 当前不支持TP域通信。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
    - 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围[1024, 8192]。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。
    - 参数约束：
        - `ep_world_size`：取值支持8、16、32、64、128、144、256、288。
        - `moe_expert_num`：取值范围(0, 1024]。
        - `tp_world_size`：预留参数，当前版本不支持，仅允许0或1。
        - `tp_rank_id`：预留参数，当前版本不支持，传0即可。
        - `shared_expert_num`：当前取值范围[0, 4]。
        - `comm_quant_mode`：int8量化支持。
        - `performance_info_optional`：预留参数，当前版本不支持，传空指针即可。
        - `ccl_buffer_size`：调用get_low_latency_ccl_buffer_size接口(../../torch_extension/cann_ops_transformer/ops/deep_ep.py)。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| low_latency_combine接口 | [deepep.py](../../torch_extension/cann_ops_transformer/ops/deep_ep.py) | 通过[low_latency_combine](../../torch_extension/cann_ops_transformer/docs/zh/low_latency_combine.md)接口方式调用moe_distribute_combine_v3算子。 |
