# MoeDistributeDispatchV3

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                               |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

算子功能：对token数据进行量化（可选），进行EP（Expert Parallelism）域的AllToAllV通信。TP（Tensor Parallelism）域通信当前版本不再支持，TP相关参数为预留参数。

- 情形1：如果quant_mode=0（非量化场景）：

$$
allToAllXOut = AllToAllV(X)\\
expand\_x\_out = AllToAllV(X)
$$

- 情形2：如果quant_mode=2（pertoken动态量化场景）：

$$
xFp32 = CastToFp32(X) \times scales \\
dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
quantOut = CastToInt8(xFp32 \times dynamicScales) \\
allToAllXOut = AllToAllV(quantOut) \\
allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
expand\_x\_out = AllToAllV(quantOut) \\
dynamic\_scales\_out = allToAllDynamicScalesOut
$$

其中，$emax$表示该类型最大正规数对应的指数部分的值。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：该算子必须与`MoeDistributeCombineV3`一起使用。

相较于`MoeDistributeDispatchV3`算子，该算子变更如下：

- 新增`context`入参，存入通信域相关信息；
- 新增`ccl_buffer_size`入参，指定当前通信域大小；
- 减少`group_ep`以及`group_tp`通信域名称入参;

详细说明请参考以下参数说明。

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
   <td>x</td>
   <td>输入</td>
   <td>本卡发送的token数据。</td>
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
   <td>scales_optional</td>
   <td>可选输入</td>
   <td>每个专家的量化平滑参数，非量化场景传空指针，动态量化可传有效数据或空指针。</td>
   <td>FLOAT32</td>
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
   <td>expert_scales_optional</td>
   <td>可选输入</td>
   <td>每个token的topK个专家权重。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elastic_info_optional</td>
   <td>可选输入</td>
   <td>EP通信域动态缩容信息。</td>
   <td>FLOAT32</td>
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
   <td><li>TP通信域大小（预留参数，当前版本不支持，仅支持传0或1）。</li><li>默认值为0。</li></td>
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
   <td>quant_mode</td>
   <td>可选属性</td>
   <td><li>表示量化模式，支持0：非量化，2：动态量化。</li><li>默认值为0。</li></td>
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
   <td>expert_token_nums_type</td>
   <td>可选属性</td>
   <td><li>输出expert_token_nums中值的语义类型，支持0：expert_token_nums中的输出为每个专家处理的token数的前缀和，1：expert_token_nums中的输出为每个专家处理的token数量。</li><li>默认值为1。</li></td>
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
   <td>expand_x_out</td>
   <td>输出</td>
   <td>根据expert_ids进行扩展过的token特征。</td>
   <td>FLOAT16、BFLOAT16、INT8</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>dynamic_scales_out</td>
   <td>输出</td>
   <td>量化场景下，表示本卡输出Token的量化系数，仅quant_mode=2时有该输出。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>assist_info_for_combine_out</td>
   <td>输出</td>
   <td>表示给同一专家发送的token个数。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_token_nums_out</td>
   <td>输出</td>
   <td>表示每个专家收到的token个数。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_recv_count_out</td>
   <td>输出</td>
   <td>从EP通信域各卡接收的token数。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_recv_count_out</td>
   <td>输出</td>
   <td>从TP通信域各卡接收的token数（预留输出，当前版本不支持，传空指针即可），无TP域通信则无该输出。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expand_scales_out</td>
   <td>输出</td>
   <td>表示本卡输出token的权重。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

* <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
    * 不支持`expand_scales_out`。

## 约束说明

- `MoeDistributeDispatchV3`与`CombineV3`系列算子必须配套使用，具体参考调用示例。

- 在不同产品型号、不同通信算法或不同版本中，`MoeDistributeDispatchV3`的Tensor输出`assist_info_for_combine_out`、`ep_recv_count_out`、`tp_recv_count_out`、`expand_scales_out`中的元素值可能不同，使用时直接将上述Tensor传给`CombineV3`系列算子对应参数即可，模型其他业务逻辑不应对其存在依赖。

- 调用算子过程中使用的`ep_world_size`、`moe_expert_num`、`ccl_buffer_size`、`tp_world_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`global_bs`、`comm_alg`参数取值所有卡需保持一致，网络中不同层中也需保持一致，且和`CombineV3`系列算子对应参数也保持一致。

- 参数说明里shape格式说明：
    - `A`：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足`A` = `BS` *`ep_world_size`* `shared_expert_num` / `shared_expert_rank_num`。
        - 对于MoE专家，当`global_bs`为0时，要满足`A` >= `BS` *`ep_world_size`* min(`local_expert_num`, `K`)；当`global_bs`非0时，要满足`A` >= `global_bs` * min(`local_expert_num`, `K`)。
    - `K`：表示选取topK个专家，取值范围为0 < `K` ≤ 16同时满足0 < `K` ≤ `moe_expert_num` + `zero_expert_num` + `copy_expert_num` + `const_expert_num`。
    - `local_expert_num`：表示本卡专家数量。
        - 对于共享专家卡，`local_expert_num` = 1
        - 对于MoE专家卡，`local_expert_num` = `moe_expert_num` / (`ep_world_size` - `shared_expert_rank_num`)，`local_expert_num` > 1时，不支持TP域通信。

- 属性约束：
    - `zero_expert_num`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1,合法的零专家的ID的值是[`moe_expert_num`, `moe_expert_num` + `zero_expert_num`)。
    - `copy_expert_num`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1,合法的copy专家的ID的值是[`moe_expert_num` + `zero_expert_num`, `moe_expert_num` + `zero_expert_num` + `copy_expert_num`)。
    - `const_expert_num`：取值范围：[0, MAX_INT32)，MAX_INT32 = 2^31 - 1,合法的常量专家的ID的值是[`moe_expert_num` + `zero_expert_num` + `copy_expert_num`, `moe_expert_num` + `zero_expert_num` + `copy_expert_num` + `const_expert_num`)。

- 本文公式中的"/"表示整除。
- 通信域使用约束：
    - 一个模型中的`CombineV3`系列算子和`MoeDistributeDispatchV3`仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 当前不支持TP域通信。

- 通信方式约束：
    - <term>Ascend 950DT</term>：仅支持UB Memory通信。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  ：
    - 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
    - 参数约束：
        - `elastic_info_optional`：当前版本不支持，传空指针即可。
        - `ep_world_size`：取值范围[2, 768]。
        - `moe_expert_num`：取值范围(0, 1024]。
        - `shared_expert_num`：取值支持[0, 4]。
        - `comm_alg`：当前版本仅支持""，"fullmesh_v1"，"fullmesh_v2"三种输入方式。
            - ""：默认值，开启fullmesh_v1模板。
            - "fullmesh_v1"：开启fullmesh_v1模板。
            - "fullmesh_v2"：开启fullmesh_v2模板，其中`comm_alg`仅在`tp_world_size`取值为1时生效，且不支持在各卡`BS`不一致、输入xActiveMask和特殊专家场景下开启。
        - `ep_recv_count_out`：要求shape为(`ep_world_size` * `local_expert_num`,)。
        - `performance_Info_optional`：预留参数，当前版本不支持，传空指针即可。
        - `ccl_buffer_size`：调用get_low_latency_ccl_buffer_size接口(../../torch_extension/cann_ops_transformer/ops/deep_ep.py)。
    - 参数说明里shape格式说明：
        - `H`：表示hidden size隐藏层大小，取值范围[1024, 8192]。
        - `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为[1, 512]。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| low_latency_dispatch接口 | [deepep.py](../../torch_extension/cann_ops_transformer/ops/deep_ep.py) | 通过[low_latency_dispatch](../../torch_extension/cann_ops_transformer/docs/zh/low_latency_dispatch.md)接口方式调用moe_distribute_dispatch_v3算子。 |
