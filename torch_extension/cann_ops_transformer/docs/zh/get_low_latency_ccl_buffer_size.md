# get_low_latency_ccl_buffer_size

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

    需与[low_latency_dispatch](low_latency_dispatch.md)和[low_latency_combine](low_latency_combine.md)配套使用，用于计算dispatch_v3和combine_v3算子所需的HCCL通信buffer_size大小（单位：MB）。该接口为静态方法，可在初始化`MoeDistributeBuffer`前调用。

- **计算公式**：

  - 计算token实际长度：

        $$token\_actual\_len = Align32(hidden * max\_out\_dtype\_size) + scale\_expand\_index\_buffer$$

        其中max_out_dtype_size为token类型转为字节数，scale_expand_index_buffer为scale（32B）+ expand_idx（3 × 4B）

  - 根据`comm_alg`计算dispatch阶段每个token所需大小：

        若`comm_alg`为"fullmesh_v2"：

        $$token\_need\_size\_dispatch = Align480(token\_actual\_len) / full\_mesh\_data\_align * win\_addr\_align$$

        若`comm_alg`为"fullmesh_v1"或""：

        $$token\_need\_size\_dispatch = Align512(token\_actual\_len)$$

  - 计算combine阶段每个token所需大小：

        $$token\_need\_size\_combine = Align512(hidden * max\_out\_dtype\_size)$$

        其中，full_mesh_data_align为480B对齐，win_addr_align为512B对齐

  - 计算ccl_buffer_size大小：

        $$dispatch\_buffer\_size = 2 * num\_max\_dispatch\_tokens\_per\_rank * token\_need\_size\_dispatch * world\_size * local\_moe\_expert\_num$$
        $$combine\_buffer\_size = 2 * num\_max\_dispatch\_tokens\_per\_rank * token\_need\_size\_combine * (topk + num\_shared\_expert)$$
        $$ccl\_buffer\_size = Align2(Align1MB(dispatch\_buffer\_size + combine\_buffer\_size + 1MB) / 1MB) / 2$$

## 函数原型

```python
MoeDistributeBuffer.get_low_latency_ccl_buffer_size(world_size, num_max_dispatch_tokens_per_rank, hidden, num_moe_expert, topk, num_shared_expert=0, num_shared_expert_ranks=0, comm_alg="") -> int
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
        <td>world_size</td>
        <td>int</td>
        <td>必选</td>
        <td>表示通信域的大小。取值范围[2, 768]。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_max_dispatch_tokens_per_rank</td>
        <td>int</td>
        <td>必选</td>
        <td>表示每张卡上的token数量。当每个rank的BS不同时，最大的BS大小。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>hidden</td>
        <td>int</td>
        <td>必选</td>
        <td>表示hidden size隐藏层大小。取值范围[1024, 8192]。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_moe_expert</td>
        <td>int</td>
        <td>必选</td>
        <td>MoE专家数量，取值范围[1, 1024]，并且满足以下条件：num_experts % (ep_world_size - shared_expert_rank_num) = 0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>topk</td>
        <td>int</td>
        <td>必选</td>
        <td>表示选取topK个专家，取值范围为0 < K ≤ 16，同时满足0 < K ≤ num_experts + zero_expert_num + copy_expert_num + const_expert_num。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_shared_expert</td>
        <td>int</td>
        <td>可选</td>
        <td>表示共享专家数量，一个共享专家可以复制部署到多个卡上。取值范围[0, 4]，0表示无共享专家，默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_shared_expert_ranks</td>
        <td>int</td>
        <td>可选</td>
        <td>表示共享专家卡数量。必须满足world_size - num_shared_expert_ranks > 0，默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>可选</td>
        <td>表示通信算法。支持取值："fullmesh_v1"、"fullmesh_v2"或""（默认）。</td>
        <td>str</td>
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
        <td>ccl_buffer_size</td>
        <td>int</td>
        <td>必选</td>
        <td>表示计算得到的ccl_buffer_size大小，单位为MB。</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和torchair图模式调用。
- get_low_latency_ccl_buffer_size、low_latency_dispatch和low_latency_combine必须配套使用。
- 本文公式中的“/”表示整除。

变量说明：

- BS：表示batch sequence size，即本卡最终输出的token数量。取值范围为0<BS≤512，当comm_alg为"fullmesh_v2"时，需满足0<BS≤256。

list[Tensor]长度 / 是否转置 / 是否支持非连续Tensor：

本接口无张量入参。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import os
  import torch
  import torch_npu
  from cann_ops_transformer.ops import MoeDistributeBuffer

  # 1. 准备通信域与算子规格参数
  server_num = 1
  dev_num = 16
  world_size = server_num * dev_num
  shared_expert_num = 1
  shared_expert_rank_num = 1  # 共享专家卡数
  num_experts = 32  # moe专家数
  bs = 8  # token数量
  h = 7168  # 每个token的长度
  k = 8
  comm_alg = "fullmesh_v1"

  # 2. 静态方法调用，计算dispatch/combine所需的CCL通信buffer大小（单位MB）
  buffer_size = MoeDistributeBuffer.get_low_latency_ccl_buffer_size(world_size, bs, h, num_experts, k,\
      shared_expert_num, shared_expert_rank_num, comm_alg)

  # 3. 将计算结果写入HCCL_BUFFSIZE环境变量，供后续low_latency_dispatch/low_latency_combine使用
  os.environ['HCCL_BUFFSIZE'] = f'buffer_size'
  print(f"buffer_size is {buffer_size}")
  print("run ccl_buffer_size success.")
  ```
