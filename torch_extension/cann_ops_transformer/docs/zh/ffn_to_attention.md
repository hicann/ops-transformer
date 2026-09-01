# ffn_to_attention

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
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

  - ffn_to_attention算子将FFN节点上的token数据发送至Attention节点，用于MoE（Mixture of Experts）场景下Attention与FFN分离部署时的反向数据通信。算子根据每个token所属的Attention Worker索引，将FFN计算完成后的token数据路由回目标Attention卡，完成FFN到Attention的数据回传。
  - 该算子提供了ffn_to_attention与get_buffer_for_ffn_to_attention等接口配套使用。
  - get_buffer_for_ffn_to_attention：用于封装输入参数并创建通信上下文（context），生成`context`、`ccl_buffer_size`等ffn_to_attention算子运行所需信息，返回buffer对象。

- **计算公式**：

  - 输入：
    - $\mathbf{X} \in \mathbb{R}^{\text{Y} \times \text{H}}$：FFN节点上的token数据矩阵，对应入参 `x`。$\text{Y}$ 是本卡需要分发的最大token数量，$\text{H}$ 是隐藏层维度。
    - $\mathbf{S} \in \mathbb{Z}^{\text{Y}}$：每个token所属的Attention Worker索引，对应入参 `session_ids`。取值范围为$[0, \text{attnRankNum}-1]$。
    - $\mathbf{MB} \in \mathbb{Z}^{\text{Y}}$：每个token的microBatch索引，对应入参 `micro_batch_ids`。取值范围为$[0, \text{microBatchNum}-1]$。
    - $\mathbf{T} \in \mathbb{Z}^{\text{Y}}$：每个token在microBatch中的token索引，对应入参 `token_ids`。取值范围为$[0, \text{BS}-1]$。
    - $\mathbf{EO} \in \mathbb{Z}^{\text{Y}}$：每个token在专家维度的偏移，对应入参 `expert_offsets`。取值范围为$[0, \text{expertNumPerToken}-1]$。
    - $\text{N}$：本卡发送的实际token总数，对应入参 `actual_token_num`。取值范围为$[0, \text{Y}]$。
  - 输出：
    - 无host可见输出。数据通过HCCL窗口发送至目标Attention卡，Attention卡从其CCL窗口中读取token数据。
  - 约定：
    - $\text{attnRankNum}$：Attention Worker数量。
    - $\text{ffnRankNum}$：FFN Worker数量。
    - $\text{microBatchNum}$：micro batch数量。
    - $\text{expertNumPerToken}$：每个token对应的专家总数（含共享专家）。
    - $\text{HS}$：token数据表的隐藏层大小（含scale存储空间），满足$\text{HS} \ge \text{H}$。

- 计算说明：

    **数据路由**

    对于FFN Worker上的每个token $\text{token}_i$（$i \in \{0, 1, \dots, \text{N}-1\}$），根据其所属的Attention Worker索引 $\mathbf{S}[i]$，将token数据 $\mathbf{x}_i$ 路由至目标Attention卡。

    目标Attention卡的Rank Id通过以下方式确定：
    - 若提供了 `attn_rank_table`：$\text{toRankId}_i = \text{attnRankTable}[\mathbf{S}[i]]$
    - 若未提供 `attn_rank_table`（默认）：$\text{toRankId}_i = \mathbf{S}[i]$，即Attention Worker索引等于Rank Id

    token数据写入目标Attention卡CCL窗口中对应的数据区域，位置由 $\mathbf{MB}[i]$（microBatch索引）、$\mathbf{T}[i]$（token索引）和 $\mathbf{EO}[i]$（专家偏移）共同确定。

    $$\text{sendData}_i = \mathbf{x}_i \quad \in \mathbb{R}^{1 \times \text{H}}$$

    token数据以原始精度（FP16/BF16）直接传输，不支持量化。

## 函数原型

先用get_buffer_for_ffn_to_attention接口封装输入参数并创建通信上下文（buffer），再调用ffn_to_attention接口进行数据发送。

```python
get_buffer_for_ffn_to_attention(group, world_size, token_info_table_shape, token_data_shape) -> FFNToAttentionBuffer
```

```python
ffn_to_attention(buffer, x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, *, attn_rank_table=None) -> None
```

## 参数说明

### get_buffer_for_ffn_to_attention

<table style="undefined;table-layout: fixed; width:840px"><colgroup>
<col style="width: 180px">
<col style="width: 140px">
<col style="width: 80px">
<col style="width: 440px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>group</td>
        <td>torch.distributed.ProcessGroup</td>
        <td>必选</td>
        <td>EP通信域的ProcessGroup对象。</td>
    </tr>
    <tr>
        <td>world_size</td>
        <td>int</td>
        <td>必选</td>
        <td>通信域大小。</td>
    </tr>
    <tr>
        <td>token_info_table_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>Token信息表格的shape，长度为3，格式为<code>[microBatchNum, BS, expertNumPerToken]</code>。</td>
    </tr>
    <tr>
        <td>token_data_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>Token数据表格的shape，长度为4，格式为<code>[microBatchNum, BS, expertNumPerToken, HS]</code>。</td>
    </tr>
</tbody>
</table>

该接口返回<code>FFNToAttentionBuffer</code>对象，内部自动计算CCL通信缓冲区大小并创建通信上下文，供<code>ffn_to_attention</code>使用。

### ffn_to_attention

<table style="undefined;table-layout: fixed; width:1400px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 90px">
<col style="width: 320px">
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 260px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>buffer</td>
        <td>FFNToAttentionBuffer</td>
        <td>必选</td>
        <td>由<a href="#get_buffer_for_ffn_to_attention">get_buffer_for_ffn_to_attention</a>创建的通信buffer，内部封装了<code>context</code>、<code>group</code>、<code>world_size</code>、<code>token_info_table_shape</code>、<code>token_data_shape</code>及<code>ccl_buffer_size</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(contextSize,)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡发送的token数据。</td>
        <td>float16、bfloat16</td>
        <td>ND</td>
        <td>(Y, H)</td>
    </tr>
    <tr>
        <td>session_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token所属的Attention Worker索引。元素取值范围为<code>[0, attnRankNum-1]</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>micro_batch_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token的microBatch索引。元素取值范围为<code>[0, microBatchNum-1]</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>token_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token在microBatch中的token索引。元素取值范围为<code>[0, BS-1]</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>expert_offsets</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token在专家维度的偏移。元素取值范围为<code>[0, expertNumPerToken-1]</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>actual_token_num</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡发送的实际token总数。取值范围为<code>[0, Y]</code>。</td>
        <td>int64</td>
        <td>ND</td>
        <td>(1,)</td>
    </tr>
    <tr>
        <td>attn_rank_table</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>映射每个Attention Worker对应的卡Id。若为None，采用默认策略：每张卡的Id作为对应Attention Worker的Id。默认值为None。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(attnRankNum,)</td>
    </tr>
</tbody>
</table>

## 返回值说明

无返回值。数据通过HCCL窗口发送至目标Attention卡，无host可见的输出tensor。

## 约束说明

- 调用算子过程中使用的`group`、`world_size`、`token_info_table_shape`、`token_data_shape`参数及`ccl_buffer_size`取值所有卡需保持一致，网络中不同层中也需保持一致。其中`group`、`world_size`、`token_info_table_shape`、`token_data_shape`及`ccl_buffer_size`由`get_buffer_for_ffn_to_attention`创建的buffer封装，调用`ffn_to_attention`时无需单独传入。

- `ccl_buffer_size`为HBM上分配的CCL通信缓冲区**总大小**（Bytes），由`get_buffer_for_ffn_to_attention`内部自动计算，需满足：

$$ccl\_buffer\_size \ge \mathrm{CeilAlign}(\mathrm{tokenInfoSize} + \mathrm{tokenDataSize},\ 2\,\mathrm{MB})$$

其中：
  - `tokenInfoSize = microBatchNum × BS × expertNumPerToken × 4B`
  - `tokenDataSize = microBatchNum × BS × expertNumPerToken × HS × 2B`

该大小由`get_buffer_for_ffn_to_attention`内部自动计算，用户无需自行计算或设置`ccl_buffer_size`。

- 参数说明里shape格式说明：
    - `Y`：表示本卡需要分发的最大token数量。
    - `BS`：表示各Attention节点上的token数，取值范围为0 < `BS` ≤ 512。
    - `H`：表示hidden size（隐藏层大小），取值范围为1024 ≤ `H` ≤ 8192。
    - `HS`：表示hidden与scale隐藏层大小，取值范围为1152 ≤ `HS` ≤ 8320，满足`HS` ≥ `H`。
    - `microBatchNum`：表示micro batch数量。
    - `expertNumPerToken`：表示每个token对应的专家总数（含共享专家）。
    - `attnRankNum`：表示Attention Worker数量，取值范围为0 < `attnRankNum` < `worldSize`。
    - `ffnRankNum`：表示FFN Worker数量，取值范围为0 < `ffnRankNum` < `worldSize`。
    - `sharedExpertNum`：表示共享专家数量，取值范围为0 ≤ `sharedExpertNum` ≤ 4。
    - `worldSize`：通信域大小，取值区间[2, 768]。

- 通信域使用约束：
    - ffn_to_attention算子的通信域中不允许有其他算子。
    - 通信域各节点的驱动版本应当相同。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  下面示例展示了ffn_to_attention的完整调用流程：先初始化通信域，再用get_buffer_for_ffn_to_attention创建通信buffer，最后调用ffn_to_attention发送token数据。

  ```python
  import os
  import torch
  import torch_npu
  import torch.distributed as dist
  import torch.multiprocessing as mp
  from torch.multiprocessing import Process
  from cann_ops_transformer.ops import (
      ffn_to_attention,
      get_buffer_for_ffn_to_attention,
  )

  WORLD_SIZE = 8
  ATTENTION_WORKER_NUM = 3
  FFN_WORKER_NUM = WORLD_SIZE - ATTENTION_WORKER_NUM
  BS = 8
  H = 7168
  HS = 7168
  K = 4
  sharedExpertNum = 1
  expertNumPerToken = K + sharedExpertNum
  microBatchNum = 1


  def set_device(rank):
      torch_npu.npu.set_device(rank % (WORLD_SIZE // 2))


  def init_hccl_comm(rank):
      dist.init_process_group(
          backend="hccl",
          rank=rank,
          world_size=WORLD_SIZE,
          init_method="tcp://127.0.0.1:50001",
      )
      ep_group = dist.new_group(backend="hccl", ranks=list(range(WORLD_SIZE)))
      return ep_group


  def run_ffn_to_attention(rank):
      set_device(rank)
      ep_group = init_hccl_comm(rank)

      token_info_table_shape = [microBatchNum, BS, expertNumPerToken]
      token_data_shape = [microBatchNum, BS, expertNumPerToken, HS]

      # 步骤1：创建通信buffer
      buffer = get_buffer_for_ffn_to_attention(
          ep_group,
          WORLD_SIZE,
          token_info_table_shape,
          token_data_shape,
      )

      # FFN Worker发送token数据至Attention Worker
      if rank >= ATTENTION_WORKER_NUM:
          # 步骤2：构造输入数据
          tokens_per_rank = (
              BS * microBatchNum * ATTENTION_WORKER_NUM * expertNumPerToken // FFN_WORKER_NUM
          )
          x = torch.randn((tokens_per_rank, H), dtype=torch.bfloat16, device="npu")
          session_ids = torch.randint(
              0, ATTENTION_WORKER_NUM, (tokens_per_rank,), dtype=torch.int32, device="npu"
          )
          micro_batch_ids = torch.zeros(tokens_per_rank, dtype=torch.int32, device="npu")
          token_ids = torch.randint(0, BS, (tokens_per_rank,), dtype=torch.int32, device="npu")
          expert_offsets = torch.randint(
              0, expertNumPerToken, (tokens_per_rank,), dtype=torch.int32, device="npu"
          )
          actual_token_num = torch.tensor([tokens_per_rank], dtype=torch.int64, device="npu")
          attn_rank_table = torch.arange(ATTENTION_WORKER_NUM, dtype=torch.int32, device="npu")

          # 步骤3：调用ffn_to_attention发送token数据
          ffn_to_attention(
              buffer,
              x,
              session_ids,
              micro_batch_ids,
              token_ids,
              expert_offsets,
              actual_token_num,
              attn_rank_table=attn_rank_table,
          )

      torch.npu.synchronize()
      buffer.destroy()
      dist.barrier()
      dist.destroy_process_group()
      print(f"[INFO] rank {rank} ffn_to_attention finished")


  if __name__ == "__main__":
      mp.set_start_method("forkserver", force=True)
      proc_list = []
      for rank in range(WORLD_SIZE):
          proc = Process(target=run_ffn_to_attention, args=(rank,))
          proc.start()
          proc_list.append(proc)
      for proc in proc_list:
          proc.join()
  ```
