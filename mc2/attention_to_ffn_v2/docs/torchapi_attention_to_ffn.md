# attention_to_ffn

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

  - attention_to_ffn算子将Attention节点上的token数据发送至FFN节点，用于MoE（Mixture of Experts）场景下Attention与FFN分离部署时的前向数据通信。算子根据每个token选择的专家索引及专家到FFN卡的映射表，将token数据路由到目标FFN卡，并支持可选的量化传输以降低通信带宽开销。
  - 该算子提供了attention_to_ffn与get_buffer_for_attention_to_ffn等接口配套使用。
  - get_buffer_for_attention_to_ffn：用于封装输入参数并创建通信上下文（context），生成`context`、`ccl_buffer_size`等attention_to_ffn算子运行所需信息，返回buffer对象。

- **计算公式**：

  - 输入：
    - $\mathbf{X} \in \mathbb{R}^{\text{X} \times \text{BS} \times \text{H}}$：Attention节点上的token数据矩阵，对应入参 `x`。$\text{X}$ 是micro batch sequence size，$\text{BS}$ 是batch sequence size，$\text{H}$ 是隐藏层维度。
    - $\mathbf{E} \in \mathbb{Z}^{\text{X} \times \text{BS} \times \text{K}}$：每个token选择的topK个专家索引矩阵，对应入参 `expert_ids`。$\text{K}$ 是每个token选择的专家数量。
    - $\mathbf{R} \in \mathbb{Z}^{\text{L} \times (\text{moeExpertNum} + \text{sharedExpertNum}) \times \text{M}}$：专家到FFN卡的部署映射表，对应入参 `expert_rank_table`。$\text{L}$ 是模型层数，$\text{M}$ 是映射表最后一维长度。
    - $\text{sessionId}$：当前Attention Worker节点Id，对应入参 `session_id`。
    - $\text{microBatchId}$：当前micro batch的Id，对应入参 `micro_batch_id`。
    - $\text{layerId}$：当前模型层数Id，对应入参 `layer_id`。
  - 输出：
    - 无host可见输出。数据通过HCCL窗口发送至目标FFN卡，FFN卡从其CCL窗口中读取token数据。
  - 约定：
    - $\text{expertNumPerToken} = \text{K} + \text{sharedExpertNum}$，表示每个token对应的专家总数（含共享专家）。
    - $\text{attnWorkerNum}$：Attention Worker数量。
    - $\text{ffnWorkerNum}$：FFN Worker数量。
    - $\text{ffnStartRankId}$：FFN节点的起始Rank Id。

- 计算说明：

    **数据路由**

    对于Attention Worker上的每个token $\text{token}_i$（$i \in \{0, 1, \dots, \text{BS}-1\}$），根据其选择的第 $k$ 个专家 $e_{i,k} = \mathbf{E}[\text{microBatchId}, i, k]$（$k \in \{0, \dots, \text{K}-1\}$），通过专家到FFN卡的映射表 $\mathbf{R}$ 查找到该专家所在的FFN卡 $\text{toRankId}_{i,k}$。

    <details>
    <summary> 非量化场景（quantMode=0）</summary>

    在非量化场景下，token数据以原始精度（FP16/BF16）直接传输。对于每个token $\text{token}_i$ 及其选中的第 $k$ 个专家，将token数据 $\mathbf{x}_i$ 写入目标FFN卡 $\text{toRankId}_{i,k}$ 的CCL窗口中对应的token数据区域，同时将元信息（sessionId、microBatchId、tokenId、expertOffset等）写入该FFN卡的信息表区域。

    $$\text{sendData}_{i,k} = \mathbf{x}_i \quad \in \mathbb{R}^{1 \times \text{H}}$$

    </details>

    <details>
    <summary> PERTOKEN+INT8量化场景（quantMode=2）</summary>

    在PERTOKEN量化场景下，对每个token的隐藏状态向量动态计算缩放因子并量化为INT8后传输。

    对token $\text{token}_i$，计算其逐token缩放因子：

    $$s^{X}_i = \frac{\max(|\mathbf{X}[\text{microBatchId}, i, :]|)}{127} \in \mathbb{R}$$

    然后量化得到INT8表示：

    $$\mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[\text{microBatchId}, i, :]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{H}}$$

    将量化后的数据 $\mathbf{q}_i$ 和缩放因子 $s^{X}_i$ 一并发送至目标FFN卡。

    </details>

    <details>
    <summary> MXFP量化场景（quantMode=3/4/5）</summary>

    在MX（Microscaling）量化场景下，对token数据按每32个连续元素为一组计算共享指数缩放因子（E8M0格式），并量化为目标低精度类型。

    对token $\text{token}_i$ 的隐藏状态向量 $\mathbf{x}_i \in \mathbb{R}^{1 \times \text{H}}$，按group size = 32分组：

    $$\text{shared\_exp}_g = \text{floor}(\log_2(\max(|\mathbf{x}_{i,g}|))) - \text{emax}, \quad g \in \{0, 1, \dots, \lceil \text{H}/32 \rceil - 1\}$$

    $$s^{X}_{i,g} = 2^{\text{shared\_exp}_g}$$

    其中 $\text{emax}$ 为目标数据类型的最大指数值（FP8_E5M2为7，FP8_E4M3为3，FP4_E2M1为1），$\mathbf{x}_{i,g}$ 表示第 $g$ 组的32个元素。

    量化结果为：

    $$\mathbf{q}_{i,g} = \text{Cast}\left(\frac{\mathbf{x}_{i,g}}{s^{X}_{i,g}}, \text{dstType}\right)$$

    各quantMode对应的输出数据类型：

    | quantMode | 输出数据类型 | 说明 |
    | :-------: | :----------: | :---: |
    | 3 | FLOAT8_E5M2 | MX量化，输出FP8_E5M2 |
    | 4 | FLOAT8_E4M3 | MX量化，输出FP8_E4M3 |
    | 5 | FLOAT4_E2M1 | MX量化，输出FP4_E2M1 |

    </details>

    <details>
    <summary> MX_CLIP量化场景（quantMode=6/7）</summary>

    MX_CLIP模式在MX量化基础上引入1e-4下限clipping，防止缩放因子过小导致精度损失。

    $$\text{max\_abs}_g = \max(|\mathbf{x}_{i,g}|)$$

    $$\text{max\_abs\_clamp}_g = \max(\text{max\_abs}_g, 10^{-4})$$

    $$\text{shared\_exp}_g = \text{ceil}(\log_2(\text{max\_abs\_clamp}_g / \text{fp8\_max}))$$

    $$s^{X}_{i,g} = 2^{\text{shared\_exp}_g}$$

    $$\mathbf{q}_{i,g} = \text{Cast}\left(\frac{\mathbf{x}_{i,g}}{s^{X}_{i,g}}, \text{dstType}\right)$$

    各quantMode对应的输出数据类型：

    | quantMode | 输出数据类型 | 说明 |
    | :-------: | :----------: | :---: |
    | 6 | FLOAT8_E5M2 | MX_CLIP量化，输出FP8_E5M2 |
    | 7 | FLOAT8_E4M3 | MX_CLIP量化，输出FP8_E4M3 |

    > MX_CLIP模式仅支持FP8输出，不支持FP4。

    </details>

## 函数原型

先用get_buffer_for_attention_to_ffn接口封装输入参数并创建通信上下文（buffer），再调用attention_to_ffn接口进行数据发送。

```python
get_buffer_for_attention_to_ffn(group, world_size, ffn_token_info_table_shape, ffn_token_data_shape, *, quant_mode=0) -> AttentionToFfnBuffer
```

```python
attention_to_ffn(buffer, x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, attn_token_info_table_shape, moe_expert_num, *, sync_flag=0, ffn_start_rank_id=0, scales=None, active_mask=None) -> None
```

## 参数说明

### get_buffer_for_attention_to_ffn

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
        <td>ffn_token_info_table_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>FFN节点上token信息表格的shape，长度为3，格式为<code>[attnWorkerNum, X, infoTableLastDim]</code>。</td>
    </tr>
    <tr>
        <td>ffn_token_data_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>FFN节点上token数据表格的shape，长度为5，格式为<code>[attnWorkerNum, microBatchNum, BS, K+sharedExpertNum, HS]</code>。</td>
    </tr>
    <tr>
        <td>quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>量化模式。0表示非量化，2表示PERTOKEN+INT8，3表示MX+FP8_E5M2，4表示MX+FP8_E4M3，5表示MX+FP4_E2M1，6表示MX_CLIP+FP8_E5M2，7表示MX_CLIP+FP8_E4M3。默认值为0。</td>
    </tr>
</tbody>
</table>

该接口返回<code>AttentionToFfnBuffer</code>对象，内部自动计算CCL通信缓冲区大小并创建通信上下文，供<code>attention_to_ffn</code>使用。

### attention_to_ffn

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
        <td>AttentionToFfnBuffer</td>
        <td>必选</td>
        <td>由<a href="#get_buffer_for_attention_to_ffn">get_buffer_for_attention_to_ffn</a>创建的通信buffer，内部封装了<code>context</code>、<code>group</code>、<code>world_size</code>、<code>ffn_token_info_table_shape</code>、<code>ffn_token_data_shape</code>、<code>quant_mode</code>及<code>ccl_buffer_size</code>。</td>
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
        <td>(X, BS, H)</td>
    </tr>
    <tr>
        <td>session_id</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>当前Attention Worker节点的Id。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(X,)</td>
    </tr>
    <tr>
        <td>micro_batch_id</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>当前micro batch的Id。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(X,)</td>
    </tr>
    <tr>
        <td>layer_id</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>当前模型层数的Id。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(X,)</td>
    </tr>
    <tr>
        <td>expert_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token的topK个专家索引。元素取值范围为<code>[0, moe_expert_num)</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(X, BS, K)</td>
    </tr>
    <tr>
        <td>expert_rank_table</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>专家Id到FFN卡部署的映射表（外部需保证值正确）。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(L, moeExpertNum + sharedExpertNum, M)</td>
    </tr>
    <tr>
        <td>attn_token_info_table_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>Attention节点上token信息表格的shape，长度为3。</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moe_expert_num</td>
        <td>int</td>
        <td>必选</td>
        <td>MoE路由专家数量。</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sync_flag</td>
        <td>int</td>
        <td>可选</td>
        <td>同步模式。0表示异步，1表示同步。默认值为0。</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ffn_start_rank_id</td>
        <td>int</td>
        <td>可选</td>
        <td>FFN节点的起始Rank Id。默认值为0。</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scales</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>每个专家的量化平滑参数。MX/MX_CLIP量化模式下不支持传入，需为None。默认值为None。</td>
        <td>float32</td>
        <td>ND</td>
        <td>(L, moeExpertNum + sharedExpertNum, H)</td>
    </tr>
    <tr>
        <td>active_mask</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>表示token是否参与通信。默认值为None。</td>
        <td>bool</td>
        <td>ND</td>
        <td>(X, BS)</td>
    </tr>
</tbody>
</table>

## 返回值说明

无返回值。数据通过HCCL窗口发送至目标FFN卡，无host可见的输出tensor。

## 约束说明

- 调用算子过程中使用的`group`、`world_size`、`ffn_token_info_table_shape`、`ffn_token_data_shape`、`attn_token_info_table_shape`参数及`ccl_buffer_size`取值所有卡需保持一致，网络中不同层中也需保持一致。其中`group`、`world_size`、`ffn_token_info_table_shape`、`ffn_token_data_shape`、`quant_mode`及`ccl_buffer_size`由`get_buffer_for_attention_to_ffn`创建的buffer封装，调用`attention_to_ffn`时无需单独传入。

- `ccl_buffer_size`为HBM上分配的CCL通信缓冲区**总大小**（Bytes），由`get_buffer_for_attention_to_ffn`内部自动计算，需满足：

$$ccl\_buffer\_size \ge \mathrm{CeilAlign}(\mathrm{tokenInfoSize} + \mathrm{tokenDataSize},\ 2\,\mathrm{MB})$$

其中：

  - `tokenInfoSize = attnWorkerNum × microBatchNum × infoTableLastDim × 4B`
  - 非量化模式：`tokenDataSize = attnWorkerNum × microBatchNum × BS × (K+shared) × HS × 2B`
  - INT8/FP8量化模式：`tokenDataSize = attnWorkerNum × microBatchNum × BS × (K+shared) × HS × 1B`
  - FP4量化模式：`HS`已按打包字节计算，`tokenDataSize = attnWorkerNum × microBatchNum × BS × (K+shared) × HS × 1B`

该大小由`get_buffer_for_attention_to_ffn`内部自动计算，用户无需自行计算或设置`ccl_buffer_size`。

- 量化模式说明：

| quantMode | 算法 | 输出类型 | 说明 |
| :--------: | :--: | :------: | :---: |
| 0 | 非量化 | 跟随输入 | FP16/BF16直接传输 |
| 2 | PERTOKEN | INT8 | 每个token动态计算scale并量化为INT8 |
| 3 | MX | FP8_E5M2 | 每32个元素为一组计算MX scale（E8M0格式），输出FP8_E5M2 |
| 4 | MX | FP8_E4M3 | 每32个元素为一组计算MX scale（E8M0格式），输出FP8_E4M3 |
| 5 | MX | FP4_E2M1 | 每32个元素为一组计算MX scale（E8M0格式），输出FP4_E2M1 |
| 6 | MX_CLIP | FP8_E5M2 | 每32个元素为一组计算MX scale（带1e-4下限clipping），输出FP8_E5M2 |
| 7 | MX_CLIP | FP8_E4M3 | 每32个元素为一组计算MX scale（带1e-4下限clipping），输出FP8_E4M3 |

- MX/MX_CLIP量化约束：
  - MX模式（quantMode 3/4/5）及MX_CLIP模式（quantMode 6/7）不支持传入scales参数，需为None。
  - FP4输出（quantMode=5）时H维度需为偶数。
  - MX_CLIP模式（quantMode 6/7）仅支持FP8输出（E5M2/E4M3），不支持FP4。

- 参数说明里shape格式说明：
    - `X`：表示micro batch sequence size（token组数），当前版本只支持`X` = 1。
    - `BS`：表示batch sequence size（本卡最终输出的token数量），取值范围为0 < `BS` ≤ 512。
    - `K`：表示选取topK个专家，取值范围为0 < `K` ≤ 16且满足0 < `K` ≤ moeExpertNum。
    - `H`：表示hidden size（隐藏层大小），取值范围为1024 ≤ `H` ≤ 8192。
    - `L`：表示模型层数，当前版本只支持`L` = 1。
    - `M`：表示expertRankTable最后一维的长度，具体体现为部署在FFN节点上数量最多的专家部署信息列表的长度。
    - `moeExpertNum`：表示MoE专家数量，取值范围为0 < `moeExpertNum` ≤ 1024。
    - `sharedExpertNum`：表示共享专家数量，取值范围为0 ≤ `sharedExpertNum` ≤ 4。
    - `worldSize`：通信域大小，取值区间[2, 1024]。

- 通信域使用约束：
    - attention_to_ffn算子的通信域中不允许有其他算子。
    - 通信域各节点的驱动版本应当相同。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  下面示例展示了attention_to_ffn的完整调用流程：先初始化通信域，再用get_buffer_for_attention_to_ffn创建通信buffer，最后调用attention_to_ffn发送token数据。

  ```python
  import os
  import torch
  import torch_npu
  import torch.distributed as dist
  import torch.multiprocessing as mp
  from torch.multiprocessing import Process, Manager
  from cann_ops_transformer.ops import (
      attention_to_ffn,
      get_buffer_for_attention_to_ffn,
  )

  WORLD_SIZE = 8
  FFN_WORKER_NUM = 5
  ATTENTION_WORKER_NUM = WORLD_SIZE - FFN_WORKER_NUM
  X = 1
  BS = 8
  H = 7168
  K = 4
  sharedExpertNum = 1
  moeExpertNum = 8
  expertNumPerToken = K + sharedExpertNum
  quantMode = 0
  syncFlag = 0
  ffnStartRankId = 0


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


  def run_attention_to_ffn(rank):
      set_device(rank)
      ep_group = init_hccl_comm(rank)

      ffn_token_info_table_shape = [ATTENTION_WORKER_NUM, X, 2 + BS * expertNumPerToken]
      ffn_token_data_shape = [ATTENTION_WORKER_NUM, X, BS, expertNumPerToken, H]
      attn_token_info_table_shape = [X, BS, expertNumPerToken]

      # 步骤1：创建通信buffer
      buffer = get_buffer_for_attention_to_ffn(
          ep_group,
          WORLD_SIZE,
          ffn_token_info_table_shape,
          ffn_token_data_shape,
          quant_mode=quantMode,
      )

      # 步骤2：构造输入数据
      x = torch.randn((X, BS, H), dtype=torch.bfloat16, device="npu")
      session_id = torch.tensor([rank - FFN_WORKER_NUM], dtype=torch.int32, device="npu")
      micro_batch_id = torch.tensor([0], dtype=torch.int32, device="npu")
      layer_id = torch.tensor([0], dtype=torch.int32, device="npu")
      expert_ids = torch.randint(0, moeExpertNum, (X, BS, K), dtype=torch.int32, device="npu")
      M = 2 * (FFN_WORKER_NUM - 1) + 1
      expert_rank_table = torch.randint(0, FFN_WORKER_NUM, (1, moeExpertNum + sharedExpertNum, M),
                                        dtype=torch.int32, device="npu")

      # 步骤3：调用attention_to_ffn发送token数据
      attention_to_ffn(
          buffer,
          x,
          session_id,
          micro_batch_id,
          layer_id,
          expert_ids,
          expert_rank_table,
          attn_token_info_table_shape,
          moeExpertNum,
          sync_flag=syncFlag,
          ffn_start_rank_id=ffnStartRankId,
      )

      torch.npu.synchronize()
      buffer.destroy()
      dist.barrier()
      dist.destroy_process_group()
      print(f"[INFO] rank {rank} attention_to_ffn finished")


  if __name__ == "__main__":
      mp.set_start_method("forkserver", force=True)
      proc_list = []
      for rank in range(WORLD_SIZE):
          proc = Process(target=run_attention_to_ffn, args=(rank,))
          proc.start()
          proc_list.append(proc)
      for proc in proc_list:
          proc.join()
  ```
