# ffn_to_attention

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
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
get_buffer_for_ffn_to_attention(group, world_size, token_info_table_shape, token_data_shape, *, window_addr=None, window_size=None) -> FFNToAttentionBuffer
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
    <tr>
        <td>window_addr</td>
        <td>int</td>
        <td>可选</td>
        <td>本卡（ffn侧）通信窗口内存的设备地址，与<code>window_size</code>必须成对传入。传入后该内存替代框架内部分配的通信窗口：框架会将其清零一次并注册进通信域，不会分配或释放该内存；调用方需保证该内存在buffer销毁前持续有效。仅channel后端（Ascend950）支持。默认值为None，表示由框架内部分配。</td>
    </tr>
    <tr>
        <td>window_size</td>
        <td>int</td>
        <td>可选</td>
        <td>本卡通信窗口内存大小（Bytes），即ffn侧内存大小，同时作为算子<code>ccl_buffer_size</code>（等于实际注册的窗口大小）。需不小于FFN侧窗口所需大小（<code>attention_to_ffn</code>全部attention worker的下发接收区 + 本算子flag source区与地址表区）；算子tiling会按会话布局校验该值（调用<code>ffn_to_attention</code>传入<code>attn_rank_table</code>时按其dim0精确校验，否则按保守最大会话数校验），不满足时算子执行报错。默认值为None，表示由框架按保守容量（覆盖<code>world_size - 1</code>个会话）内部分配。</td>
    </tr>
</tbody>
</table>

该接口返回<code>FFNToAttentionBuffer</code>对象，内部自动计算FFN侧CCL通信缓冲区大小（覆盖<code>attention_to_ffn</code>的下发接收区及本算子scratch区，作为算子<code>ccl_buffer_size</code>属性，等于实际注册的窗口大小），并创建通信上下文，供<code>ffn_to_attention</code>使用；传入<code>window_addr</code>/<code>window_size</code>时，本卡通信窗口以该地址和大小注册。本接口与<code>attention_to_ffn</code>共用注册tag以便对端rank查询本卡窗口。

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

- 本算子与`attention_to_ffn`共用同一rank通信窗口（注册tag相同），本算子的flag source区与地址表/count行区布局在全部N个会话的下发接收区之后，与对侧会话数据区互不重叠。会话数`N`（即对侧`attentionWorkerNum`/`ffn_token_data_shape[0]`）在算子侧的确定方式：调用`ffn_to_attention`传入`attn_rank_table`时`N`为其dim0（精确布局，须满足`0 < N < world_size`且与对侧一致）；未传入时按`ccl_buffer_size`（即实际注册的窗口大小）所能容纳的最大会话数推导布局，此时窗口须按保守容量（`world_size - 1`个会话）分配，保证推导值不小于真实值。

- `ccl_buffer_size`为HBM上分配的CCL通信缓冲区**总大小**（Bytes），等于实际注册的窗口大小：自定义内存时为`window_size`，否则由`get_buffer_for_ffn_to_attention`内部按保守容量自动计算，需满足：

$$ccl\_buffer\_size \ge \mathrm{CeilAlign}\big(\mathrm{CeilAlign}(\mathrm{recvInfoSize} + \mathrm{recvDataSize} + 32\,\mathrm{KiB},\,512) + \mathrm{tableBytes},\ 2\,\mathrm{MB}\big)$$

其中（`sessionNum`为布局会话数：传入`attn_rank_table`时为其dim0，未传入时为保守上界`worldSize - 1`；接口内部默认按`worldSize - 1`计算）：
  - `recvInfoSize = CeilAlign(sessionNum × microBatchNum × (2 + BS × expertNumPerToken) × 4B, 512)`（`attention_to_ffn` 下发的 N 会话 token 信息表区，镜像其对侧布局公式）
  - `recvDataSize = CeilAlign(sessionNum × microBatchNum × BS × expertNumPerToken × HS × 2B, 512)`（`attention_to_ffn` 下发的 N 会话 token 数据区；2B 为对侧非量化元素宽度，量化模式下对侧按 1B 下发，此处为保守上界）
  - `tableBytes = world_size × rankTableStride + 1025 × countRowStride`（各rank地址表与AIV计数行；1025 = 1条总计数行 + 1024条AIV预留计数行）
  - `rankTableStride = CeilAlign(world_size × microBatchNum × BS × expertNumPerToken × 16B, 512)`
  - `countRowStride = CeilAlign(world_size × 4B, 512)`

传入`attn_rank_table`时tiling按其dim0精确校验`ccl_buffer_size`是否覆盖N会话布局，不满足时报错；未传入时按可容纳的最大会话数推导布局。对端attention卡通信窗口大小由attention侧`get_buffer_for_attention_to_ffn`计算。

- 自定义通信窗口约束（`window_addr`/`window_size`）：
    - 两个参数必须成对传入，`window_size`为本卡窗口内存的实际大小（Bytes），同时作为算子`ccl_buffer_size`。
    - FFN卡上传入的为ffn侧内存大小，需不小于FFN侧窗口所需大小（全部attention worker的下发接收区 + 本算子flag source区与地址表区）；传入`attn_rank_table`时按其dim0精确校验，否则需覆盖保守容量（`world_size - 1`个会话）。
    - 窗口内存由调用方管理，需在`buffer.destroy()`之前保持有效；框架仅负责清零一次，不负责分配与释放。
    - 仅channel后端（Ascend950）支持；同一进程同一group下只应创建一个通信buffer（attn卡创建`attention_to_ffn`的buffer，FFN卡创建`ffn_to_attention`的buffer）。

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

      # 步骤1：准备本卡通信窗口内存并创建通信buffer。
      # attention卡窗口需覆盖attention_to_ffn自身的staging/flag区（由get_buffer_for_attention_to_ffn
      # 内部计算，见其文档）；FFN卡窗口需覆盖attention_to_ffn的下发接收区（每会话一个
      # info表区+数据区）及本算子flag source/地址表区。传入attn_rank_table时按其dim0（即
      # ATTENTION_WORKER_NUM）精确校验，未传入时按保守最大会话数（world_size - 1）校验；
      # 此处为演示window_addr/window_size用法按保守容量手工计算。
      if rank < ATTENTION_WORKER_NUM:
          window_size = microBatchNum * BS * expertNumPerToken * (4 + HS * 2)
      else:
          # 保守容量：最大会话数 = world_size - 1；infoTableLastDim与attention_to_ffn的
          # ffn_token_info_table_shape最后一维一致；另需为flag source/地址表区预留余量。
          max_sessions = WORLD_SIZE - 1
          info_table_last_dim = 2 + BS * expertNumPerToken
          window_size = max_sessions * microBatchNum * (
              info_table_last_dim * 4 + BS * expertNumPerToken * H * 2
          ) + 2 * 1024 * 1024
      window_size = (
          window_size + 2 * 1024 * 1024 - 1
      ) // (2 * 1024 * 1024) * (2 * 1024 * 1024)
      # 本卡窗口内存由调用方管理，需在buffer销毁前保持有效
      window = torch.empty(window_size, dtype=torch.uint8, device="npu")
      window_addr = window.data_ptr()
      buffer = get_buffer_for_ffn_to_attention(
          ep_group,
          WORLD_SIZE,
          token_info_table_shape,
          token_data_shape,
          window_addr=window_addr,
          window_size=window_size,
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
