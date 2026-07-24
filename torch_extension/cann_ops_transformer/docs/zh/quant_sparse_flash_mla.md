# quant_sparse_flash_mla

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  `quant_sparse_flash_mla_metadata`接口用于生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及Q和K的分块的索引，供后续quant_sparse_flash_mla算子使用。
  `quant_sparse_flash_mla`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`QuantSparseFlashMla`算子完成共享KV（Key和Value使用同一份输入）的稀疏注意力计算。该接口支持以下三类计算模式：

  - **SWA（Sliding Window Attention）**：仅使用`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时使用`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和TopK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时使用`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。

  `quant_sparse_flash_mla_metadata`是`QuantSparseFlashMla`的torch扩展接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子必须传入该metadata。典型调用流程如下：

  1. 准备`q`、`ori_kv`、`cmp_kv`、序列长度、`block table`、`sinks`等输入。
  2. 调用`quant_sparse_flash_mla_metadata`生成`metadata`。
  3. 调用`quant_sparse_flash_mla`，将上一步得到的`metadata`传入主算子。

- 计算公式：

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

  其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的$KV$，由`ori_kv`的滑动窗口部分和`cmp_kv`的压缩部分共同组成，实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`决定。

> [!NOTE]
>
> `cmp_residual_kv`同时是`sparse_flash_mla`和`sparse_flash_mla_metadata`的可选输入。该参数用于恢复压缩前KV长度：`ori_len_for_cmp_mask = cmp_len * cmp_ratio + cmp_residual_kv[b]`。

## 函数原型

调用quant_sparse_flash_mla接口之前，先调用前置接口quant_sparse_flash_mla_metadata，完成quant_sparse_flash_mla负载均衡的计算。

```python
cann_ops_transformer.quant_sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    batch_size=0,
    max_seqlen_q=0,
    max_seqlen_ori_kv=0,
    max_seqlen_cmp_kv=0,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=True
) -> Tensor
```

```python
cann_ops_transformer.quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    q_descale=None,
    ori_kv_descale=None,
    cmp_kv_descale=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## 参数说明

### quant_sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示Query的head个数，支持1-128。 | int32 | - |
| num_heads_kv | int | 必选 | 表示Key和Value对应的多头数，仅支持1。 | int32 | - |
| head_dim | int | 必选 | 表示注意力头的维度，仅支持512。 | int32 | - |
| quant_mode | int | 必选 | 表示量化模式，1表示qkv为hifloat8 perTensor量化。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | 表示不同Batch中Query的有效Sequence Length，仅layout_q为TND场景需传入。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| cu_seqlens_ori_kv | Tensor | 可选 | 表示不同Batch中ori_kv的有效Sequence Length，仅layout_kv为TND场景需传入。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| cu_seqlens_cmp_kv | Tensor | 可选 | 表示不同Batch中cmp_kv的有效Sequence Length，仅layout_kv为TND场景需传入。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| seqused_q | Tensor | 可选 | 表示不同Batch中Query实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| seqused_ori_kv | Tensor | 可选 | 表示不同Batch中ori_kv实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| seqused_cmp_kv | Tensor | 可选 | 表示不同Batch中cmp_kv实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| cmp_residual_kv | Tensor | 可选 | 表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算。cmp_mask_mode=3且cmp_ratio≠1时必须传入。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| ori_topk_length | Tensor | 可选 | 用于标识ori_kv sparseindices实际参与计算的长度。数据格式为ND，支持非连续的Tensor。 | int32 | (B, S1, N2)或(T1, N2) |
| cmp_topk_length | Tensor | 可选 | 用于标识ori_kv sparseindices实际参与计算的长度。数据格式为ND，支持非连续的Tensor。 | int32 | (B, S1, N2)或(T1, N2) |
| batch_size | int | 可选 | 表示Batch数量，默认值为0。 | int32 | - |
| max_seqlen_q | int | 可选 | 表示Query的最长Sequence Length，默认值为0。 | int32 | - |
| max_seqlen_ori_kv | int | 可选 | 表示ori_kv的最长Sequence Length，默认值为0。 | int32 | - |
| max_seqlen_cmp_kv | int | 可选 | 表示cmp_kv的最长Sequence Length，默认值为0。 | int32 | - |
| ori_topk | int | 可选 | 预留参数，当前不生效，表示ori_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景，默认值为0。 | int32 | - |
| cmp_topk | int | 可选 | 表示cmp_kv中筛选出的关键稀疏token的个数。默认值为0。 | int32 | - |
| cmp_ratio | int | 可选 | 表示对cmp_kv的压缩率，支持1到128。默认值为1。 | int32 | - |
| ori_mask_mode | int | 可选 | 表示q和ori_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，4表示sliding window模式，默认值为0。 | int32 | - |
| cmp_mask_mode | int | 可选 | 表示q和cmp_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，默认值为0。 | int32 | - |
| ori_win_left | int | 可选 | 表示q和ori_kv计算中q对过去token计算的数量，-1表示无穷大，默认值为-1。 | int32 | - |
| ori_win_right | int | 可选 | 表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大，默认值为-1。 | int32 | - |
| layout_q | str | 可选 | 表示Query的排列格式，支持"BSND"、"TND"，默认值为"BSND"。 | string | - |
| layout_kv | str | 可选 | 表示Key的排列格式，支持"BSND"、"TND"、"PA_BBND"，默认值为"BSND"。 | string | - |
| has_ori_kv | bool | 可选 | 用于标识是否含有ori_kv，默认值为True。 | bool | - |
| has_cmp_kv | bool | 可选 | 用于标识是否含有cmp_kv，默认值为True。 | bool | - |

### quant_sparse_flash_mla

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| q | 输入 | Query输入。 | HIFLOAT8 | ND |
| ori_kv | 可选输入 | 原始量化KV输入，Key和Value共享同一份数据。| HIFLOAT8 | ND |
| cmp_kv | 可选输入 | 压缩量化KV输入，Key和Value共享同一份数据。| HIFLOAT8 | ND |
| q_descale | 可选输入 | q对应的量化参数。 | FLOAT | ND |
| ori_kv_descale | 可选输入 | ori_kv对应的量化参数。| FLOAT | ND |
| cmp_kv_descale | 可选输入 | cmp_kv对应的量化参数。| FLOAT | ND |
| ori_sparse_indices | 可选输入 | 原始KV稀疏索引，无效位置填-1。 | INT32 | ND |
| cmp_sparse_indices | 可选输入 | 压缩KV TopK索引，无效位置填-1。 | INT32 | ND |
| ori_block_table | 可选输入 | PageAttention场景下`ori_kv`使用的block映射表。 | INT32 | ND |
| cmp_block_table | 可选输入 | PageAttention场景下`cmp_kv`使用的block映射表。 | INT32 | ND |
| cu_seqlens_q | 可选输入 | TND场景下`q`的累积序列长度。 | INT32 | ND |
| cu_seqlens_ori_kv | 可选输入 | TND场景下`ori_kv`的累积序列长度。 | INT32 | ND |
| cu_seqlens_cmp_kv | 可选输入 | TND场景下`cmp_kv`的累积序列长度。 | INT32 | ND |
| seqused_q | 可选输入 | 不同batch中`q`实际参与计算的token数。 | INT32 | ND |
| seqused_ori_kv | 可选输入 | 不同batch中`ori_kv`实际参与计算的token数。 | INT32 | ND |
| seqused_cmp_kv | 可选输入 | 不同batch中`cmp_kv`实际参与计算的token数。 | INT32 | ND |
| cmp_residual_kv | 可选输入 | 压缩KV余数，用于恢复cmp侧mask使用的压缩前KV长度。 | INT32 | ND |
| ori_topk_length | 可选输入 | 用于标识ori_kv sparseindices实际参与计算的长度。 | INT32 | ND |
| cmp_topk_length | 可选输入 | 用于标识cmp_kv sparseindices实际参与计算的长度 | INT32 | ND |
| sinks | 可选输入 | attention sinks输入。 | FLOAT | ND |
| metadata | 输入 | `quant_sparse_flash_mla_metadata`生成的任务切分结果。 | INT32 | ND |
| quant_mode | 必选属性 | 表示量化模式，1表示qkv为hifloat8 perTensor量化。 | INT | - |
| softmax_scale | 可选属性 | QK矩阵乘后的缩放系数。默认值为1.0。 | FLOAT | - |
| cmp_ratio | 可选属性 | 表示`cmp_kv`相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度；仅传入`ori_kv`时不参与压缩KV计算。支持1到128。默认值为1。 | INT | - |
| ori_mask_mode | 可选属性 | 表示`q`和`ori_kv`计算的mask模式。<br>0: No Mask。<br>3: RightDownCausal模式。<br>4: Band模式。默认值为0。 | INT | - |
| cmp_mask_mode | 可选属性 | 表示`q`和`cmp_kv`计算的mask模式。<br>0: No Mask。<br>3: RightDownCausal模式。默认值为0。 | INT | - |
| ori_win_left | 可选属性 | 表示`q`和`ori_kv`计算中`q`对过去token计算的数量，支持-1或非负数，其中-1表示窗口不受限。默认值为-1。 | INT | - |
| ori_win_right | 可选属性 | 表示`q`和`ori_kv`计算中`q`对未来token计算的数量，支持-1或非负数，其中-1表示窗口不受限。默认值为-1。 | INT | - |
| layout_q | 可选属性 | 表示输入`q`的数据排布格式，支持"BSND"和"TND"。默认值为"BSND"。 | STRING | - |
| layout_kv | 可选属性 | 表示输入`ori_kv`和`cmp_kv`的数据排布格式，支持"BSND"、"TND"和"PA_BBND"。默认值为"BSND"。 | STRING | - |
| topk_value_mode | 可选属性 | 表示TopK索引取值模式，仅支持1。默认值为1。 | INT | - |
| return_softmax_lse | 可选属性 | 表示是否返回softmax的log-sum-exp结果。默认值为False。 | BOOL | - |
| attention_out | 输出 | attention计算输出。 | BFLOAT16 | ND |
| softmax_lse | 输出 | softmax的log-sum-exp结果；未使能返回时为占位Tensor。 | FLOAT | ND |

## 返回值说明

### quant_sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个cube核上FlashAttention计算任务的Batch、Head、以及 Q 和 K 的分块的索引，以及每个vector核上FlashDecode的规约任务索引。数据格式为ND，不支持非连续的Tensor。 | int32 | (1024, ) |

### quant_sparse_flash_mla

- **attention_out**：`quant_sparse_flash_mla`的第一个输出，shape和`q`一致，dtype和`q`一致。
- **softmax_lse**：`quant_sparse_flash_mla`的第二个输出。`return_softmax_lse=False`时返回FLOAT32标量占位Tensor；`return_softmax_lse=True`时返回FLOAT32的log-sum-exp结果。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和aclgraph模式。
- quant_sparse_flash_mla_metadata接口需与quant_sparse_flash_mla算子配套使用。
- `layout_q`支持"BSND"和"TND"；`layout_q="BSND"`时，`q`必须为4维；`layout_q="TND"`时，`q`必须为3维且必须传入`cu_seqlens_q`。
- `layout_kv`支持"BSND"、"TND"和"PA_BBND"；`layout_kv="BSND"`或`layout_kv="PA_BBND"`时，`ori_kv`和`cmp_kv`必须为4维；`layout_kv="TND"`时，`ori_kv`和`cmp_kv`必须为3维。
- `layout_kv="TND"`时必须传入`cu_seqlens_ori_kv`；传入`cmp_kv`时，还必须传入`cu_seqlens_cmp_kv`。
- B（Batch）表示输入样本批量大小。
- 参数`cu_seqlens_q`、`cu_seqlens_ori_kv`及`cu_seqlens_cmp_kv`要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
- 参数`seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv`要求其值表示每个Batch中的有效token数。
- 参数`cmp_residual_kv`需满足`cmp_residual_kv\[i\]` < `cmp_ratio`。
- `layout_kv="PA_BBND"`时必须传入`seqused_ori_kv`和`ori_block_table`；传入`cmp_kv`时，还必须传入`cmp_block_table`。
- `seqused_cmp_kv`为所有`layout_kv`下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度。
- `ori_mask_mode`及`cmp_mask_mode`所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../../docs/zh/context/sparse_mode_introduction.md)。
- `metadata`固定为1024个INT32元素，`topk_value_mode`仅支持1，`ori_sparse_indices`、`ori_topk_length`和`cmp_topk_length`当前版本不支持传入。
- `ori_kv`和`cmp_kv`允许存在行间padding类非连续内存，接口会通过aclnn获取stride信息传给底层算子。

- 规格约束：
  - 公共参数约束：
    - `head_dim`仅支持512，`num_heads_kv`仅支持1。
    - `num_heads_q / num_heads_kv`仅支持1-128。
    - `cmp_ratio`仅支持1到128。
    - PageAttention的block_size支持1到1024。
  - SWA：
    - 仅传入`ori_kv`时，`cmp_ratio`不参与压缩KV计算，需保持默认值1。
    - 不传入`cmp_kv`、`cmp_sparse_indices`和`cmp_block_table`。
    - `cmp_topk`传0，`cmp_mask_mode`传0。
  - CSA：
    - `cmp_mask_mode`仅支持3。
    - `cmp_sparse_indices`必须传入，最后一维支持泛化；`cmp_topk`对应传非0。
    - `cmp_residual_kv`必须传入，长度必须等于batch大小。
  - HCA：
    - `cmp_mask_mode`仅支持3。
    - 不传入`cmp_sparse_indices`；`cmp_topk`传0。
    - `cmp_residual_kv`必须传入，长度必须等于batch大小。

## 确定性计算

- 默认支持确定性计算

## 调用说明

### SWA，BSND输入，aclnn直调
```python
import torch
import torch_npu
import math
import cann_ops_transformer

torch_npu.npu.set_device(0)

qDtype = torch.uint8 # 使用uint8伪装hifloat8
kvDtype = torch.uint8
B = 1
S1 = 16
S2 = 64
N1 = 64
N2 = 1
Dq = 512
Dkv = 512
quant_mode = 1
cmp_ratio = 1  # SWA示例仅传ori_kv，cmp_ratio不参与压缩KV计算，保持默认值1。

seqused_q = torch.tensor([16], dtype=torch.int32, device="npu")
seqused_ori_kv = torch.tensor([64], dtype=torch.int32, device="npu")
q = torch.full((B, S1, N1, Dq), 8, dtype=qDtype, device="npu")  # uint8的8 等于 hifloat8的1.0
ori_kv = torch.full((B, S2, N2, Dkv), 8 ,dtype=kvDtype, device="npu")
cmp_kv = None
q_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
ori_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
cmp_kv_descale = None
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

layout_q = "BSND"
layout_kv = "BSND"

metadata = torch.ops.cann_ops_transformer.quant_sparse_flash_mla_metadata(
    num_heads_q = N1,
    num_heads_kv = N2,
    head_dim = Dq,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    quant_mode=quant_mode,
    batch_size = B,
    max_seqlen_q = max(seqused_q),
    max_seqlen_ori_kv = max(seqused_ori_kv),
    ori_topk = 0,
    cmp_ratio = cmp_ratio,
    ori_mask_mode = 4,
    cmp_mask_mode = 0,
    ori_win_left = 127,
    ori_win_right = 0,
    layout_q = layout_q,
    layout_kv = layout_kv,
    has_ori_kv = ori_kv is not None,
    has_cmp_kv = cmp_kv is not None,
    )

npu_result, _ = torch.ops.cann_ops_transformer.quant_sparse_flash_mla(
    q=q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    q_descale = q_descale,
    ori_kv_descale = ori_kv_descale,
    cmp_kv_descale = cmp_kv_descale,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    sinks=sinks,
    metadata=metadata,
    quant_mode=quant_mode,
    softmax_scale=1.0 / math.sqrt(Dq),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q=layout_q,
    layout_kv=layout_kv,
    topk_value_mode=1,
    return_softmax_lse=False)

torch.npu.synchronize()
```

### HCA，LayoutQ BSND，LayoutKv PA_BBND输入，aclnn直调

```python
import torch
import torch_npu
import math
import cann_ops_transformer

torch_npu.npu.set_device(0)

def scatter_to_pa(kv_bnsd, block_size, block_num, max_block_num_per_batch):
    B, N2, S, D = kv_bnsd.shape
    kv_expand = torch.zeros(B, N2, max_block_num_per_batch * block_size, D, dtype=kv_bnsd.dtype, device=kv_bnsd.device)
    kv_expand[:, :, :S, :] = kv_bnsd
    kv_pa = torch.zeros(block_num, block_size, N2, D, dtype=kv_bnsd.dtype, device=kv_bnsd.device)
    block_table = torch.zeros(B, max_block_num_per_batch, dtype=torch.int32, device=kv_bnsd.device)
    cur_block_id = 0
    for i_B in range(B):
        for i_block in range(max_block_num_per_batch):
            block_table[i_B, i_block] = cur_block_id
            block_start = i_block * block_size
            for i_N2 in range(N2):
                kv_pa[cur_block_id, :, i_N2, :] = kv_expand[i_B, i_N2, block_start:block_start + block_size, :]
            cur_block_id += 1
    return kv_pa, block_table

qDtype = torch.uint8 # 使用uint8伪装hifloat8
kvDtype = torch.uint8
B = 1
S1 = 16
S2 = 2050
N1 = 64
N2 = 1
Dq = 512
Dkv = 512
quant_mode = 1
cmp_ratio = 128

block_size1 = 128
block_size2 = 128
ori_max_block_num_per_batch = math.ceil(S2 / block_size1)
block_num1 = ori_max_block_num_per_batch * B

seqused_ori_kv = torch.tensor([S2], dtype=torch.int32, device="npu")
cmp_max_s2 = S2 // cmp_ratio
seqused_cmp_kv = torch.tensor([cmp_max_s2], dtype=torch.int32, device="npu")
cmp_residual_kv = torch.tensor([S2 % cmp_ratio], dtype=torch.int32, device="npu")
seqused_q = torch.tensor([16], dtype=torch.int32, device="npu")

q = torch.full((B, S1, N1, Dq), 8, dtype=qDtype, device="npu") # uint8的8 等于 hifloat8的1.0
ori_kv_bnsd = torch.full((B, N2, S2, Dkv), 8, dtype=kvDtype, device="npu")
ori_kv, ori_block_table = scatter_to_pa(ori_kv_bnsd, block_size1, block_num1, ori_max_block_num_per_batch)

if cmp_max_s2 > 0:
    cmp_max_block_num_per_batch = math.ceil(cmp_max_s2 / block_size2)
    block_num2 = cmp_max_block_num_per_batch * B
    cmp_kv_bnsd = torch.full((B, N2, cmp_max_s2, Dkv), 8, dtype=kvDtype, device="npu")
    cmp_kv, cmp_block_table = scatter_to_pa(cmp_kv_bnsd, block_size2, block_num2, cmp_max_block_num_per_batch)
else:
    block_num2 = 0
    cmp_kv = None
    cmp_block_table = None

q_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
ori_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
cmp_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")

sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

layout_q = "BSND"
layout_kv = "PA_BBND"

metadata = torch.ops.cann_ops_transformer.quant_sparse_flash_mla_metadata(
    num_heads_q = N1,
    num_heads_kv = N2,
    head_dim = Dq,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    quant_mode=quant_mode,
    batch_size = B,
    max_seqlen_q = max(seqused_q),
    max_seqlen_ori_kv = max(seqused_ori_kv),
    max_seqlen_cmp_kv = max(seqused_cmp_kv),
    ori_topk = 0,
    cmp_topk = 0,
    cmp_ratio = cmp_ratio,
    ori_mask_mode = 4,
    cmp_mask_mode = 3,
    ori_win_left = 127,
    ori_win_right = 0,
    layout_q = layout_q,
    layout_kv = layout_kv,
    has_ori_kv = ori_kv is not None,
    has_cmp_kv = cmp_kv is not None,
    )

npu_result, _ = torch.ops.cann_ops_transformer.quant_sparse_flash_mla(
    q=q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    q_descale=q_descale,
    ori_kv_descale=ori_kv_descale,
    cmp_kv_descale=cmp_kv_descale,
    ori_block_table=ori_block_table,
    cmp_block_table=cmp_block_table,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    quant_mode=quant_mode,
    softmax_scale=1.0 / math.sqrt(Dq),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q=layout_q,
    layout_kv=layout_kv,
    topk_value_mode=1,
    return_softmax_lse=False)

torch.npu.synchronize()
```

### CSA，TND输入，graph调用
```python
import torch
import torch_npu
import math
import cann_ops_transformer
import torchair
from torchair.configs.compiler_config import CompilerConfig

torch_npu.npu.set_device(0)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, q, ori_kv, cmp_kv, q_descale, ori_kv_descale, cmp_kv_descale,
                cmp_sparse_indices, cu_seqlens_q, seqused_q, cu_seqlens_ori_kv, seqused_ori_kv,
                cu_seqlens_cmp_kv, seqused_cmp_kv, cmp_residual_kv,
                sinks, metadata, quant_mode, softmax_scale, cmp_ratio,
                ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
                layout_q, layout_kv, topk_value_mode, return_softmax_lse):
        npu_result, _ = torch.ops.cann_ops_transformer.quant_sparse_flash_mla(
            q=q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            q_descale=q_descale,
            ori_kv_descale=ori_kv_descale,
            cmp_kv_descale=cmp_kv_descale,
            cmp_sparse_indices=cmp_sparse_indices,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            seqused_ori_kv=seqused_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks,
            metadata=metadata,
            quant_mode=quant_mode,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            topk_value_mode=topk_value_mode,
            return_softmax_lse=return_softmax_lse)
        return npu_result

qDtype = torch.uint8 # 使用uint8伪装hifloat8
kvDtype = torch.uint8
B = 1
S1 = 16
S2 = 2050
N1 = 64
N2 = 1
Dq = 512
Dkv = 512
K = 512
quant_mode = 1
cmp_ratio = 128

q_seqlens = [S1] * B
ori_kv_seqlens = [S2] * B

cu_seqlens_q = torch.tensor([sum(q_seqlens[:i]) for i in range(B + 1)], dtype=torch.int32, device="npu")
cu_seqlens_ori_kv = torch.tensor([sum(ori_kv_seqlens[:i]) for i in range(B + 1)], dtype=torch.int32, device="npu")
cu_seqlens_cmp_kv = torch.tensor([sum(s // cmp_ratio for s in ori_kv_seqlens[:i]) for i in range(B + 1)], dtype=torch.int32, device="npu")

total_q = cu_seqlens_q[-1].item()
total_ori_kv = cu_seqlens_ori_kv[-1].item()
total_cmp_kv = cu_seqlens_cmp_kv[-1].item()

seqused_q = torch.tensor(q_seqlens, dtype=torch.int32, device="npu")
seqused_ori_kv = torch.tensor(ori_kv_seqlens, dtype=torch.int32, device="npu")
seqused_cmp_kv = torch.tensor([s // cmp_ratio for s in ori_kv_seqlens], dtype=torch.int32, device="npu")
cmp_residual_kv = torch.tensor([s % cmp_ratio for s in ori_kv_seqlens], dtype=torch.int32, device="npu")

q = torch.full((total_q, N1, Dq), 8, dtype=qDtype, device="npu") # uint8的8 等于 hifloat8的1.0
ori_kv = torch.full((total_ori_kv, N2, Dkv), 8, dtype=kvDtype, device="npu")
cmp_kv = torch.full((total_cmp_kv, N2, Dkv), 8, dtype=kvDtype, device="npu")
q_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
ori_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")
cmp_kv_descale = torch.tensor([2.0], dtype=torch.float32, device="npu")

cmp_sparse_indices = torch.full((total_q, N2, K), fill_value=-1, dtype=torch.int32, device="npu")
for i_B in range(B):
    cur_act_q = seqused_q[i_B].item()
    s1_prefix = cu_seqlens_q[i_B].item()
    cur_ori_kv = seqused_ori_kv[i_B].item()
    for i_N2 in range(N2):
        for i_S1 in range(cur_act_q):
            cur_valid_s2_max = math.floor((cur_ori_kv - cur_act_q + i_S1 + 1) / cmp_ratio)
            valid_blocks_max = max(0, cur_valid_s2_max)
            block_indices = torch.randperm(valid_blocks_max, device="npu").to(torch.int32)
            valid_blocks_topk = min(valid_blocks_max, K)
            cmp_sparse_indices[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]

sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

layout_q = "TND"
layout_kv = "TND"

print("quant_sparse_flash_mla_metadata...")
metadata = torch.ops.cann_ops_transformer.quant_sparse_flash_mla_metadata(
    num_heads_q = N1,
    num_heads_kv = N2,
    head_dim = Dq,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
    seqused_q=seqused_q,
    seqused_ori_kv=seqused_ori_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    quant_mode=quant_mode,
    batch_size = B,
    max_seqlen_q = max(q_seqlens),
    max_seqlen_ori_kv = max(ori_kv_seqlens),
    max_seqlen_cmp_kv = max(s // cmp_ratio for s in ori_kv_seqlens),
    ori_topk = 0,
    cmp_topk = K,
    cmp_ratio = cmp_ratio,
    ori_mask_mode = 4,
    cmp_mask_mode = 3,
    ori_win_left = 127,
    ori_win_right = 0,
    layout_q = layout_q,
    layout_kv = layout_kv,
    has_ori_kv = ori_kv is not None,
    has_cmp_kv = cmp_kv is not None,
    )
torch.npu.synchronize()
metadata.npu()

print("torch.compile...")
torch._dynamo.reset()
npu_mode = Network().npu()
config = CompilerConfig()
config.mode = "reduce-overhead"
config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
config.experimental_config.frozen_parameter = True
npu_backend = torchair.get_npu_backend(compiler_config=config)
npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)

print("quant_sparse_flash_mla (graph)...")
npu_result = npu_mode(
    q=q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    q_descale=q_descale,
    ori_kv_descale=ori_kv_descale,
    cmp_kv_descale=cmp_kv_descale,
    cmp_sparse_indices=cmp_sparse_indices,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    seqused_ori_kv=seqused_ori_kv,
    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    quant_mode=quant_mode,
    softmax_scale=1.0 / math.sqrt(Dq),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q=layout_q,
    layout_kv=layout_kv,
    topk_value_mode=1,
    return_softmax_lse=False)

torch.npu.synchronize()
```
