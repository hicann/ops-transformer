# low_latency_combine

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

    需与[low_latency_dispatch](low_latency_dispatch.md)配套使用，相当于按low_latency_dispatch算子收集数据的路径原路返回。
  - 支持数据整合功能，进行alltoallv通信，最后将接收的数据整合（乘权重再相加）；
  - 支持特殊专家场景。

- **计算公式**：

  - 数据整合功能：

      $$ata\_out = AlltoAllv(expand\_x)$$

      $$x = Sum(topk\_weights * ata\_out + topk\_weights * shared\_expert\_x)$$

  - 特殊专家场景：

      零专家场景，即`zero_expert_num`不为0：

      $$Moe(ori\_x)=0$$

      拷贝专家场景，即`copy_expert_num`不为0：

      $$Moe(ori\_x)=ori\_x$$

      常量专家场景，即`const_expert_num`不为0：

      $$Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v$$

## 函数原型

```python
MoeDistributeBuffer.low_latency_combine(x, topk_idx, topk_weights, assist_info_for_combine, ep_send_counts, *, num_experts=0, comm_alg="", comm_quant_mode=0, x_active_mask=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, zero_expert_num=0, copy_expert_num=0, const_expert_num=0, expert_shared_type=0, shared_expert_num=1, shared_expert_rank_num=0, num_max_dispatch_tokens_per_rank=0) -> Tensor
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>根据`topk_idx`进行扩展过的token特征，数据格式为ND。</td>
        <td>bfloat16、float16</td>
        <td>(A, H)</td>
    </tr>
    <tr>
        <td>topk_idx</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token的topK个专家索引，数据格式为ND。对应[low_latency_dispatch](low_latency_dispatch.md)的`topk_idx`输入，张量里value取值范围为[0, num_experts)，且同一行中的K个value不能重复。</td>
        <td>int32</td>
        <td>(BS, K)</td>
    </tr>
    <tr>
        <td>topk_weights</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示每个token的topK个专家的权重，其中共享专家不需要乘权重系数，直接相加即可，支持传None，数据格式为ND。</td>
        <td>float</td>
        <td>(BS, K)</td>
    </tr>
    <tr>
        <td>assist_info_for_combine</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示给同一专家发送的token个数，数据格式为ND。对应[low_latency_dispatch](low_latency_dispatch.md)的`assist_info_for_combine`输出。</td>
        <td>int32</td>
        <td>(A*128, )</td>
    </tr>
    <tr>
        <td>ep_send_counts</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示本卡每个专家发给EP（Expert Parallelism）域每个卡的token数（token数以前缀和的形式表示），数据格式为ND。对应[low_latency_dispatch](low_latency_dispatch.md)的`ep_recv_counts`输出。</td>
        <td>int32</td>
        <td>(ep_world_size*local_expert_num, )</td>
    </tr>
    <tr>
        <td>num_experts</td>
        <td>int</td>
        <td>可选</td>
        <td>MoE专家数量，取值范围[1, 1024]，并且满足以下条件：num_experts%(ep_world_size - shared_expert_rank_num)=0。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>可选</td>
        <td>表示通信亲和内存布局算法，当前版本仅支持""。默认值为""。</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>表示通信量化类型。支持取0和2。0表示通信时不量化，2表示通信时进行int8量化。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_active_mask</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>表示token是否参与通信。要求为一个1维或2维张量，数据格式要求为ND。当输入为1维时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true}为非法输入；当输入为2维时，参数为true表示当前token对应的`topk_idx`参与通信，若当前token对应的K个bool值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。默认值为None。</td>
        <td>bool</td>
        <td>
            <ul>
                <li>(BS, )</li>
                <li>(BS, K)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>expand_scales</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>对应[low_latency_dispatch](low_latency_dispatch.md)的`expand_scales`输出。暂不支持该参数，使用默认值即可。默认值为None。</td>
        <td>float</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_x</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>数据类型需与`expand_x`保持一致。仅在共享专家卡数量`shared_expert_rank_num`为0的场景下使用，表示共享专家token，在combine_v2后需要做add的值。当张量为2D时，shape为(BS, H)；当张量为3D时，前两维的乘积需等于BS，第三维需等于H。默认值为None。</td>
        <td>与x一致</td>
        <td>
            <ul>
                <li>(BS, H)</li>
                <li>(*, *, H)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>elastic_info</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>预留参数，当前版本不支持，传默认值None即可。默认值为None。</td>
        <td>int32</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ori_x</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>表示未经过FFN的token数据，在`copy_expert_num`不为0或`const_expert_num`不为0的场景下需要本输入数据，数据类型需跟`expand_x`保持一致，数据格式要求为ND。可选择传入有效数据或填None，当`copy_expert_num`不为0或`const_expert_num`不为0时必须传入有效数据。默认值为None。</td>
        <td>与x一致</td>
        <td>(BS, H)</td>
    </tr>
    <tr>
        <td>const_expert_alpha_1</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>在`const_expert_num`不为0的场景下需要输入的计算系数，数据类型需跟`expand_x`保持一致，数据格式要求为ND。可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入。默认值为None。</td>
        <td>与x一致</td>
        <td>(const_expert_num, H)</td>
    </tr>
    <tr>
        <td>const_expert_alpha_2</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>在`const_expert_num`不为0的场景下需要输入的计算系数，数据类型需跟`expand_x`保持一致，数据格式要求为ND。可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入。默认值为None。</td>
        <td>与x一致</td>
        <td>(const_expert_num, H)</td>
    </tr>
    <tr>
        <td>const_expert_v</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>在`const_expert_num`不为0的场景下需要输入的计算系数，数据类型需跟`expand_x`保持一致，数据格式要求为ND。可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入。默认值为None。</td>
        <td>与x一致</td>
        <td>(const_expert_num, H)</td>
    </tr>
    <tr>
        <td>zero_expert_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示零专家的数量。取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的零专家的ID值是[num_experts, num_experts+zero_expert_num)。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>copy_expert_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示拷贝专家的数量。取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的拷贝专家的ID值是[num_experts+zero_expert_num, num_experts+zero_expert_num+copy_expert_num)。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>const_expert_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示常量专家的数量。取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的常量专家的ID值是[num_experts+zero_expert_num+copy_expert_num, num_experts+zero_expert_num+copy_expert_num+const_expert_num)。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expert_shared_type</td>
        <td>int</td>
        <td>可选</td>
        <td>表示共享专家卡排布类型。当前仅支持0，表示共享专家卡排在MoE专家卡前面。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示共享专家数量，一个共享专家可以复制部署到多个卡上。取值范围[0, 4]，0表示无共享专家，默认值为1。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_rank_num</td>
        <td>int</td>
        <td>可选</td>
        <td>表示共享专家卡数量。取值范围[0, ep_world_size)。取0表示无共享专家，不取0需满足shared_expert_rank_num%shared_expert_num=0。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_max_dispatch_tokens_per_rank</td>
        <td>int</td>
        <td>可选</td>
        <td>表示每张卡上的token数量。当每个rank的BS不同时，最大的BS大小，当每个rank上BS相同时，默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示处理后的token，类型与输入`expand_x`保持一致，数据格式为ND，不支持非连续的Tensor。</td>
        <td>bfloat16、float16</td>
        <td>(BS, H)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- 各张量参数的list[Tensor]长度、是否转置、是否支持非连续Tensor约束如下：

    <table style="undefined;table-layout: fixed; width:900px"><colgroup>
    <col style="width: 240px">
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 200px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>list[Tensor]长度</th>
        <th>是否转置</th>
        <th>是否支持非连续Tensor</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>x</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>topk_idx</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>topk_weights</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>assist_info_for_combine</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>ep_send_counts</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>x_active_mask</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>expand_scales</td>
            <td>-</td>
            <td>否</td>
            <td>-</td>
        </tr>
        <tr>
            <td>shared_expert_x</td>
            <td>-</td>
            <td>否</td>
            <td>-</td>
        </tr>
        <tr>
            <td>elastic_info</td>
            <td>-</td>
            <td>否</td>
            <td>-</td>
        </tr>
        <tr>
            <td>ori_x</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>const_expert_alpha_1</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>const_expert_alpha_2</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>const_expert_v</td>
            <td>-</td>
            <td>否</td>
            <td>支持</td>
        </tr>
    </tbody>
    </table>

- 参数里Shape使用的变量如下：
  - A：表示本卡接收的最大token数量，取值范围如下
    - 对于共享专家，要满足A=BS\*ep_world_size\*shared_expert_num/shared_expert_rank_num。
    - 对于MoE专家，当num_max_dispatch_tokens_per_rank为0时，要满足A >= BS\*ep_world_size \* min(local_expert_num, K)；当num_max_dispatch_tokens_per_rank不为0时，要满足A >= num_max_dispatch_tokens_per_rank \* ep_world_size \* min(local_expert_num, K)。
  - H：表示hidden size隐藏层大小。取值范围[1024, 8192]。
  - BS：表示待发送的token数量。取值范围为0<BS≤512。
  - K：表示选取topK个专家，取值范围为0<K≤16，同时满足0 < K ≤ num_experts + zero_expert_num + copy_expert_num + const_expert_num。
  - local_expert_num：表示本卡专家数量。
    - 对于共享专家卡，local_expert_num=1。
    - 对于MoE专家卡，local_expert_num=num_experts/(ep_world_size-shared_expert_rank_num)。
    - 应满足0 < local_expert_num \* ep_world_size ≤ 2048。

- `low_latency_dispatch`和`low_latency_combine`必须配套使用。
- 在不同产品型号、不同通信算法或不同版本中，`low_latency_dispatch`的Tensor输出`assist_info_for_combine`、`ep_recv_counts`、`expand_scales`中的元素值可能不同，使用时直接将上述Tensor传给`low_latency_combine`对应参数即可，模型其他业务逻辑不应对其存在依赖。
- 调用接口过程中使用的`group_ep`、`ep_world_size`、`num_experts`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`num_max_dispatch_tokens_per_rank`参数取值所有卡需保持一致，`group_ep`、`ep_world_size`、`expert_shard_type`、`num_max_dispatch_tokens_per_rank`网络中不同层中也需保持一致，且和[low_latency_dispatch](low_latency_dispatch.md)对应参数也保持一致。
- 该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
- num_experts + zero_expert_num + copy_expert_num + const_expert_num < MAX_int32。
- 相关约束：
  <!-- npu="A3" id7 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - `topk_weights`仅支持传入有效Tensor并进行加权聚合。
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - `topk_weights`可传有效Tensor或None或空Tesenor，传有效Tensor时使能topK专家权重功能，传None或空Tesenor时不使能并直接对专家输出求和
  <!-- end id8 -->
- HCCL通信域缓存区大小：

    调用本接口前需检查通信域缓存区大小取值是否合理，单位MB，不配置时默认为200MB。
  - 该场景仅支持通过环境变量HCCL_BUFFSIZE配置，该环境变量按通信域粒度管理，每个通信域独占一组“2\*HCCL_BUFFSIZE”大小的内存。
  - ep通信域内：设置大小要求 >= 2且满足>= 2 \* (local_expert_num \* max_bs \* ep_world_size \* Align512(Align32(2 \* h) + 64) + (k + shared_expert_num) \* max_bs\* Align512(2 \* h))。
  - 其中480Align512(x) = ((x+480-1)/480)\*512,Align512(x) = ((x+512-1)/512)\*512,Align32(x) = ((x+32-1)/32)\*32。
  - 通信域内开设大小可通过调用MoeDistributeBuffer.get_low_latency_ccl_buffer_size接口计算。

- 本文公式中的“/”表示整除。

- 通信域使用约束：

  - 一个模型中的`low_latency_dispatch`和`low_latency_combine`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。
  - 一个通信域内的节点需在一个超节点内，不支持跨超节点。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import os
  import torch
  import random
  import torch_npu
  import numpy as np
  from torch.multiprocessing import Process
  import torch.distributed as dist
  from torch.distributed import ReduceOp
  from cann_ops_transformer.ops import MoeDistributeBuffer

  # 控制模式
  quant_mode = 2  # 2为动态量化
  is_dispatch_scales = True  # 动态量化可选择是否传scales
  input_dtype = torch.bfloat16  # 输出dtype
  server_num = 1
  server_index = 0
  port = 50001
  master_ip = '127.0.0.1'
  dev_num = 16
  world_size = server_num * dev_num
  rank_per_dev = int(world_size / server_num)  # 每个host有几个die
  shared_expert_rank_num = 0  # 共享专家数
  num_experts = 32  # moe专家数
  bs = 8  # token数量
  h = 7168  # 每个token的长度
  k = 8
  random_seed = 0
  ep_world_size = world_size
  moe_rank_num = ep_world_size - shared_expert_rank_num
  local_moe_expert_num = num_experts // moe_rank_num
  globalBS = bs * ep_world_size
  is_shared = (shared_expert_rank_num > 0)
  is_quant = (quant_mode > 0)
  zero_expert_num = 1
  copy_expert_num = 1
  const_expert_num = 0


  def get_new_group(rank):
      ep_ranks = list(range(ep_world_size))
      ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
      if rank in ep_ranks:
          print(f"rank:{rank} ep_ranks:{ep_ranks}")
      return ep_group


  def get_hcomm_info(rank, comm_group):
      if torch.__version__ > '2.0.1':
          hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
      else:
          hcomm_info = comm_group.get_hccl_comm_name(rank)
      return hcomm_info

  def run_npu_process(rank):
      torch_npu.npu.set_device(rank)
      rank = rank + 16 * server_index
      # 初始化分布式进程组
      dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
      ep_group = get_new_group(rank)
      ep_hcomm_info = get_hcomm_info(rank, ep_group)

      # 创建输入tensor
      x = torch.randn(bs, h, dtype=input_dtype).npu()
      topk_idx = torch.tensor([[5, 7, 17, 4, 2, 6, 11, 16],
                              [10, 12, 13, 15, 19, 4, 18, 1],
                              [19, 33, 1, 17, 9, 5, 0, 32],
                              [19, 11, 17, 0, 10, 5, 7, 9],
                              [10, 16, 11, 17, 33, 8, 9, 3],
                              [12, 19, 5, 7, 1, 3, 18, 16],
                              [11, 9, 13, 16, 12, 33, 17, 14],
                              [16, 4, 9, 5, 0, 10, 11, 17]], dtype=torch.int32).npu()
      topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()

      const_expert_alpha_1 = None
      const_expert_alpha_2 = None
      const_expert_v = None

      # 构造通信域buffer
      distribute_buffer = MoeDistributeBuffer(ep_group)

      # 先调用dispatch，完成token的量化与EP域alltoallv分发，combine所需的assist_info_for_combine、ep_recv_counts来自本步输出
      expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, _ = distribute_buffer.low_latency_dispatch(
          x=x,
          topk_idx=topk_idx,
          shared_expert_num=0,
          shared_expert_rank_num=shared_expert_rank_num,
          num_experts=num_experts,
          quant_mode=quant_mode,
          num_max_dispatch_tokens_per_rank=globalBS,
          zero_expert_num=zero_expert_num,
          copy_expert_num=copy_expert_num,
          const_expert_num=const_expert_num)

      if is_quant:
          expand_x = expand_x.to(input_dtype)

      # 再调用combine，将dispatch的输出原路返回回收：assist_info_for_combine、ep_recv_counts直接配套传入，按topk_weights乘权重再相加
      x = distribute_buffer.low_latency_combine(x=expand_x,
                                              topk_idx=topk_idx,
                                              topk_weights=topk_weights,
                                              assist_info_for_combine=assist_info_for_combine,
                                              ep_send_counts=ep_recv_counts,
                                              shared_expert_num=0,
                                              shared_expert_rank_num=shared_expert_rank_num,
                                              num_experts=num_experts,
                                              num_max_dispatch_tokens_per_rank=globalBS,
                                              ori_x=x,
                                              const_expert_alpha_1=const_expert_alpha_1,
                                              const_expert_alpha_2=const_expert_alpha_2,
                                              const_expert_v=const_expert_v,
                                              zero_expert_num=zero_expert_num,
                                              copy_expert_num=copy_expert_num,
                                              const_expert_num=const_expert_num)
      print(f'rank {rank} npu finished! \n')


  if __name__ == "__main__":
      print(f"bs={bs}")
      print(f"num_max_dispatch_tokens_per_rank={globalBS}")
      print(f"shared_expert_rank_num={shared_expert_rank_num}")
      print(f"num_experts={num_experts}")
      print(f"k={k}")
      print(f"quant_mode={quant_mode}", flush=True)
      print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
      print(f"ep_world_size={ep_world_size}", flush=True)

      if shared_expert_rank_num > ep_world_size:
          print("shared_expert_rank_num不能大于ep_world_size")
          exit(0)
      if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
          print("ep_world_size必须是shared_expert_rank_num的整数倍")
          exit(0)
      if num_experts % moe_rank_num != 0:
          print("num_experts必须是moe_rank_num的整数倍")
          exit(0)
      p_list = []
      for rank in range(rank_per_dev):
          p = Process(target=run_npu_process, args=(rank,))
          p_list.append(p)

      for p in p_list:
          p.start()
      for p in p_list:
          p.join()

      print("run npu success.")
  ```

- 图模式调用：

    ```python
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time
    from cann_ops_transformer.ops import MoeDistributeBuffer

    # 控制模式
    quant_mode = 2  # 2为动态量化
    is_dispatch_scales = True  # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16  # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    shared_expert_rank_num = 0  # 共享专家数
    num_experts = 32  # moe专家数
    bs = 8  # token数量
    h = 7168  # 每个token的长度
    k = 8
    random_seed = 0
    ep_world_size = world_size
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_num_experts = num_experts // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)

    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 0

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self, group_ep):
            super().__init__()
            self.group = group_ep
            self.distribute_buffer = MoeDistributeBuffer(group_ep)
        def forward(self, x, topk_idx, group_ep,
                    ep_world_size, ep_rank_id, expert_shard_type, shared_expert_rank_num, num_experts,
                    scales, quant_mode, num_max_dispatch_tokens_per_rank, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
            output_dispatch_npu = self.distribute_buffer.low_latency_dispatch(
                x=x,
                topk_idx=topk_idx,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            output_combine_npu = self.distribute_buffer.low_latency_combine(
                x=expand_x_npu,
                topk_idx=topk_idx,
                topk_weights=expert_scales,
                assist_info_for_combine=assist_info_for_combine_npu,
                ep_send_counts=ep_recv_counts_npu,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                ori_x=x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]


    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1


    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2


    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v


    def get_new_group(rank):
        ep_ranks = list(range(ep_world_size))
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            print(f"rank:{rank} ep_ranks:{ep_ranks}")
        return ep_group


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def get_dispatch_kwargs_warmup(
        x_warm_up, topk_idx_warm_up,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()
        return {
            'x': x_warm_up,
            'topk_idx': topk_idx_warm_up,
            'x_active_mask': None,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'num_experts': num_experts,
            'scales': None,
            'quant_mode': 2,
            'num_max_dispatch_tokens_per_rank': 1,
        }


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        ep_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        topk_idx = torch.tensor([
            [0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]
        ], dtype=torch.int32).npu()
        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        elastic_info = None
        const_expert_alpha_1 = None
        const_expert_alpha_2 = None
        const_expert_v = None
        model = MOE_DISTRIBUTE_GRAPH_Model(ep_group)
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(
            x, topk_idx, ep_hcomm_info, ep_world_size,
            rank, 0, shared_expert_rank_num, num_experts, scales,
            quant_mode, globalBS, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
            zero_expert_num, copy_expert_num, const_expert_num
        )
        torch.npu.synchronize()
        print(f'rank {rank} npu finished! \n')

        time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"num_max_dispatch_tokens_per_rank={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"num_experts={num_experts}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_num_experts={local_num_experts}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num不能大于ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size必须是shared_expert_rank_num的整数倍")
            exit(0)

        if num_experts % moe_rank_num != 0:
            print("num_experts必须是moe_rank_num的整数倍")
            exit(0)

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```
