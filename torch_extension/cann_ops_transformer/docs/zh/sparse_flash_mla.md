# sparse_flash_mla

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
<!-- npu="310p" id8 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id8 -->
<!-- npu="910" id9 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id9 -->

## 功能说明

- **接口功能**：

  `sparse_flash_mla_metadata`接口用于生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及Q和K的分块的索引，供后续sparse_flash_mla算子使用。

  `sparse_flash_mla`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`SparseFlashMla`算子完成共享KV（Key和Value使用同一份输入）的稀疏注意力计算。该接口支持以下三类计算模式：

  - **SWA（Sliding Window Attention）**：仅使用`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时使用`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和TopK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时使用`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。

  `sparse_flash_mla_metadata`是`sparse_flash_mla`的metadata前置接口，用于在主接口执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主接口必须传入该metadata。典型调用流程如下：

  1. 准备`q`、`ori_kv`、`cmp_kv`、序列长度、`block table`、`sinks`等输入。
  2. 调用`sparse_flash_mla_metadata`生成`metadata`。
  3. 调用`sparse_flash_mla`，将上一步得到的`metadata`传入主算子。

- **计算公式**：

  $$
  O = \text{softmax}(Q \cdot \tilde{K}^{T} \cdot \text{softmax\_scale}) \cdot \tilde{V}
  $$

  其中$\tilde{K}=\tilde{V}$，由`ori_kv`的滑动窗口部分和`cmp_kv`的压缩部分共同组成，实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`决定。

> [!NOTE]
>
> `cmp_residual_kv`同时是`sparse_flash_mla`和`sparse_flash_mla_metadata`的可选输入。该参数用于恢复压缩前KV长度：`ori_len_for_cmp_mask = cmp_len * cmp_ratio + cmp_residual_kv[b]`。

## 函数原型

调用sparse_flash_mla接口之前，先调用前置接口sparse_flash_mla_metadata，完成sparse_flash_mla负载均衡的计算。

```python
cann_ops_transformer.sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
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
cann_ops_transformer.sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
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

### sparse_flash_mla_metadata

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| num_heads_q | 属性 | Query head数，支持1、2、4、8、16、32、64、128。 | INT | - |
| num_heads_kv | 属性 | KV head数，仅支持1。 | INT | - |
| head_dim | 属性 | 每个注意力头的维度，仅支持512。 | INT | - |
| cu_seqlens_q | 可选输入 | TND场景下`q`的累积序列长度。 | int32 | ND |
| cu_seqlens_ori_kv | 可选输入 | TND场景下`ori_kv`的累积序列长度。 | int32 | ND |
| cu_seqlens_cmp_kv | 可选输入 | TND场景下`cmp_kv`的累积序列长度。 | int32 | ND |
| seqused_q | 可选输入 | 不同batch中`q`实际参与计算的token数。 | int32 | ND |
| seqused_ori_kv | 可选输入 | 不同batch中`ori_kv`实际参与计算的token数。 | int32 | ND |
| seqused_cmp_kv | 可选输入 | 不同batch中`cmp_kv`实际参与计算的token数。 | int32 | ND |
| cmp_residual_kv | 可选输入 | 压缩KV余数，用于恢复cmp侧mask使用的压缩前KV长度。 | int32 | ND |
| ori_topk_length | 可选输入 | 预留输入，当前版本不支持传入非空Tensor。 | int32 | ND |
| cmp_topk_length | 可选输入 | 预留输入，当前版本不支持传入非空Tensor。 | int32 | ND |
| batch_size | 可选属性 | batch大小。默认值为0。 | INT | - |
| max_seqlen_q | 可选属性 | 所有batch中`q`的最大有效长度。默认值为0。 | INT | - |
| max_seqlen_ori_kv | 可选属性 | 所有batch中`ori_kv`的最大有效长度。默认值为0。 | INT | - |
| max_seqlen_cmp_kv | 可选属性 | 所有batch中`cmp_kv`的最大有效长度。默认值为0。 | INT | - |
| ori_topk | 可选属性 | 原始KV TopK长度。默认值为0。 | INT | - |
| cmp_topk | 可选属性 | 压缩KV TopK长度，默认值为0。 | INT | - |
| cmp_ratio | 可选属性 | 表示`cmp_kv`相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度；仅传入`ori_kv`时不参与压缩KV计算。支持1到128。默认值为1。 | INT | - |
| ori_mask_mode | 可选属性 | 表示`q`和`ori_kv`计算的mask模式。<br>0: No Mask。<br>3: RightDownCausal模式。<br>4: Band模式。默认值为0。 | INT | - |
| cmp_mask_mode | 可选属性 | 表示`q`和`cmp_kv`计算的mask模式。<br>0: No Mask。<br>3: RightDownCausal模式。默认值为0。 | INT | - |
| ori_win_left | 可选属性 | 表示`q`和`ori_kv`计算中`q`对过去token计算的数量，支持-1或非负数，其中-1表示窗口不受限。默认值为-1。 | INT | - |
| ori_win_right | 可选属性 | 表示`q`和`ori_kv`计算中`q`对未来token计算的数量，支持-1或非负数，其中-1表示窗口不受限。默认值为-1。 | INT | - |
| layout_q | 可选属性 | 表示输入`q`的数据排布格式，支持"BSND"和"TND"。默认值为"BSND"。 | STRING | - |
| layout_kv | 可选属性 | 表示输入`ori_kv`和`cmp_kv`的数据排布格式，支持"BSND"、"TND"和"PA_BBND"。默认值为"BSND"。 | STRING | - |
| has_ori_kv | 可选属性 | 表示`SparseFlashMla`主算子是否传入`ori_kv`。默认值为True。 | BOOL | - |
| has_cmp_kv | 可选属性 | 表示`SparseFlashMla`主算子是否传入`cmp_kv`。默认值为True。 | BOOL | - |
| metadata | 输出 | 表示`SparseFlashMla`主算子使用的任务切分结果。 | int32 | ND |

### sparse_flash_mla

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| q | 输入 | Query输入。 | float16、bfloat16 | ND |
| ori_kv | 可选输入 | 原始KV输入，Key和Value共享同一份数据。 | float16、bfloat16 | ND |
| cmp_kv | 可选输入 | 压缩KV输入，Key和Value共享同一份数据。 | float16、bfloat16 | ND |
| ori_sparse_indices | 可选输入 | 原始KV稀疏索引，当前版本不支持传入非空Tensor。 | int32 | ND |
| cmp_sparse_indices | 可选输入 | 压缩KV TopK索引，无效位置填-1。 | int32 | ND |
| ori_block_table | 可选输入 | PageAttention场景下`ori_kv`使用的block映射表。 | int32 | ND |
| cmp_block_table | 可选输入 | PageAttention场景下`cmp_kv`使用的block映射表。 | int32 | ND |
| cu_seqlens_q | 可选输入 | TND场景下`q`的累积序列长度。 | int32 | ND |
| cu_seqlens_ori_kv | 可选输入 | TND场景下`ori_kv`的累积序列长度。 | int32 | ND |
| cu_seqlens_cmp_kv | 可选输入 | TND场景下`cmp_kv`的累积序列长度。 | int32 | ND |
| seqused_q | 可选输入 | 不同batch中`q`实际参与计算的token数。 | int32 | ND |
| seqused_ori_kv | 可选输入 | 不同batch中`ori_kv`实际参与计算的token数。 | int32 | ND |
| seqused_cmp_kv | 可选输入 | 不同batch中`cmp_kv`实际参与计算的token数。 | int32 | ND |
| cmp_residual_kv | 可选输入 | 压缩KV余数，用于恢复cmp侧mask使用的压缩前KV长度。 | int32 | ND |
| ori_topk_length | 可选输入 | 预留输入，当前版本不支持传入非空Tensor。 | int32 | ND |
| cmp_topk_length | 可选输入 | 预留输入，当前版本不支持传入非空Tensor。 | int32 | ND |
| sinks | 可选输入 | attention sinks输入。 | float | ND |
| metadata | 输入 | `sparse_flash_mla_metadata`生成的任务切分结果。 | int32 | ND |
| softmax_scale | 可选属性 | QK矩阵乘后的缩放系数。默认值为1.0。 | float | - |
| cmp_ratio | 可选属性 | 表示`cmp_kv`相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度；仅传入`ori_kv`时不参与压缩KV计算。支持1到128。默认值为1。 | INT | - |
| ori_mask_mode | 可选属性 | 表示`q`和`ori_kv`计算的mask模式。<br>0: No Mask。<br>3: RightDownCausal模式。<br>4: Band模式。默认值为0。 | INT | - |
| cmp_mask_mode | 可选属性 | 表示`q`和`cmp_kv`计算的mask模式。<br>0: No Mask。<br>3: RightDownCausal模式。默认值为0。 | INT | - |
| ori_win_left | 可选属性 | 表示`q`和`ori_kv`计算中`q`对过去token计算的数量，支持-1或非负数，其中-1表示窗口不受限。默认值为-1。 | INT | - |
| ori_win_right | 可选属性 | 表示`q`和`ori_kv`计算中`q`对未来token计算的数量，支持-1或非负数，其中-1表示窗口不受限。默认值为-1。 | INT | - |
| layout_q | 可选属性 | 表示输入`q`的数据排布格式，支持"BSND"和"TND"。默认值为"BSND"。 | STRING | - |
| layout_kv | 可选属性 | 表示输入`ori_kv`和`cmp_kv`的数据排布格式，支持"BSND"、"TND"和"PA_BBND"。默认值为"BSND"。 | STRING | - |
| topk_value_mode | 可选属性 | 表示TopK索引取值模式，仅支持1。默认值为1。 | INT | - |
| return_softmax_lse | 可选属性 | 表示是否返回softmax的log-sum-exp结果。默认值为False。 | BOOL | - |
| attention_out | 输出 | attention计算输出。 | float16、bfloat16 | ND |
| softmax_lse | 输出 | softmax的log-sum-exp结果；未使能返回时为占位Tensor。 | float | ND |

## 返回值说明

### sparse_flash_mla_metadata

- **metadata**：`sparse_flash_mla_metadata`的输出，shape固定为`[1024]`，dtype为`int32`。

### sparse_flash_mla

- **attention_out**：`sparse_flash_mla`的第一个输出，shape和`q`一致，dtype和`q`一致。
- **softmax_lse**：`sparse_flash_mla`的第二个输出。`return_softmax_lse=False`时返回float32标量占位Tensor；`return_softmax_lse=True`时返回float32的log-sum-exp结果。

## 约束说明

- 公共约束：
  - 适用场景：该接口支持训练、推理场景下使用。
  - 调用方式：该接口支持单算子模式和TorchAir图模式调用。
  - `q`、`ori_kv`、`cmp_kv`的数据类型必须一致，支持float16和bfloat16。
  - `layout_q`和`layout_kv`组合仅支持"BSND"/"BSND"、"TND"/"TND"、"BSND"/"PA_BBND"、"TND"/"PA_BBND"；非PA_BBND场景下`layout_q`和`layout_kv`必须一致。
  - `layout_q`支持"BSND"和"TND"；`layout_q="BSND"`时，`q`必须为4维；`layout_q="TND"`时，`q`必须为3维且必须传入`cu_seqlens_q`。
  - `layout_kv`支持"BSND"、"TND"和"PA_BBND"；`layout_kv="BSND"`或`layout_kv="PA_BBND"`时，`ori_kv`和`cmp_kv`必须为4维；`layout_kv="TND"`时，`ori_kv`和`cmp_kv`必须为3维。
  - `layout_kv="TND"`时必须传入`cu_seqlens_ori_kv`；传入`cmp_kv`时，还必须传入`cu_seqlens_cmp_kv`。
  - 参数`cu_seqlens_q`、`cu_seqlens_ori_kv`及`cu_seqlens_cmp_kv`要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
  - 参数`seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv`要求其值表示每个Batch中的有效token数。
  - 参数`cmp_residual_kv`需满足`cmp_residual_kv\[i\]` < `cmp_ratio`。
  - `layout_kv="PA_BBND"`时必须传入`seqused_ori_kv`和`ori_block_table`；传入`cmp_kv`时，还必须传入`cmp_block_table`。
  - `seqused_cmp_kv`为所有`layout_kv`下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度。
  - `ori_mask_mode`及`cmp_mask_mode`所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../../docs/zh/context/sparse_mode_introduction.md)。
  - `metadata`固定为1024个int32元素，`topk_value_mode`仅支持1，`ori_sparse_indices`、`ori_topk_length`和`cmp_topk_length`当前版本不支持传入非空Tensor。
  - `ori_kv`和`cmp_kv`允许存在行间padding类非连续内存，接口会通过aclnn获取stride信息传给底层算子。

- 规格约束：
  - 公共参数约束：
    - `head_dim`仅支持512，`num_heads_kv`仅支持1。
    - `num_heads_q / num_heads_kv`支持1、2、4、8、16、32、64、128。
    - `ori_mask_mode`仅支持4，`ori_win_left`仅支持127，`ori_win_right`仅支持0。
    - PageAttention的block_size支持1到1024。

  - 产品型号约束如下：
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：`num_heads_q`/`num_heads_kv`不支持1。
    <!-- end id5 -->
    <!-- npu="A3,910b" id6 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：cmp_ratio在SWA场景保持默认值1，CSA支持传入4，HCA支持传入128；block_size支持16的倍数，且不超过1024；ori_sparse_indices当前暂不支持，cmp_sparse_indices的最后一维压缩KV TopK长度，支持0、512、1024。默认值为0。
    <!-- end id6 -->
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

- 默认支持确定性计算。
- 默认支持batch invariance。

## 调用说明

### SWA，BSND输入

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
B = 1
S1 = 16
S2 = 64
N1 = 64
N2 = 1
D = 512
cmp_ratio = 1  # SWA示例仅传ori_kv，cmp_ratio不参与压缩KV计算，保持默认值1。

q = torch.randn(B, S1, N1, D, dtype=dtype, device="npu")
ori_kv = torch.randn(B, S2, N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    N1,
    N2,
    D,
    batch_size=B,
    max_seqlen_q=S1,
    max_seqlen_ori_kv=S2,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=False,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### HCA，BSND输入

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
B = 1
S1 = 16
S2 = 128
S3 = 1
N1 = 64
N2 = 1
D = 512
cmp_ratio = 128

q = torch.randn(B, S1, N1, D, dtype=dtype, device="npu")
ori_kv = torch.randn(B, S2, N2, D, dtype=dtype, device="npu")
cmp_kv = torch.randn(B, S3, N2, D, dtype=dtype, device="npu")
cmp_residual_kv = torch.zeros(B, dtype=torch.int32, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    N1,
    N2,
    D,
    batch_size=B,
    max_seqlen_q=S1,
    max_seqlen_ori_kv=S2,
    max_seqlen_cmp_kv=S3,
    cmp_residual_kv=cmp_residual_kv,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=True,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### CSA，TND输入并使能cmp_residual_kv

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.float16
B = 1
q_lens = [1]
ori_lens = [6]
cmp_lens = [1]
N1 = 64
N2 = 1
D = 512
K = 512
cmp_ratio = 4

cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cu_ori = torch.tensor([0, 6], dtype=torch.int32, device="npu")
cu_cmp = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cmp_residual_kv = torch.tensor([2], dtype=torch.int32, device="npu")

q = torch.randn(sum(q_lens), N1, D, dtype=dtype, device="npu")
ori_kv = torch.randn(sum(ori_lens), N2, D, dtype=dtype, device="npu")
cmp_kv = torch.randn(sum(cmp_lens), N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

cmp_sparse_indices = torch.full((sum(q_lens), N2, K), -1, dtype=torch.int32, device="npu")
cmp_sparse_indices[:, :, :1] = torch.arange(1, dtype=torch.int32, device="npu").view(1, 1, 1)

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    N1,
    N2,
    D,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    max_seqlen_q=max(q_lens),
    max_seqlen_ori_kv=max(ori_lens),
    max_seqlen_cmp_kv=max(cmp_lens),
    ori_topk=0,
    cmp_topk=K,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    cmp_residual_kv=cmp_residual_kv,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="TND",
    has_ori_kv=True,
    has_cmp_kv=True,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_sparse_indices=cmp_sparse_indices,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="TND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### CP切分示例，TND + CSA，rank0切开第二个seq

下面示例用单进程顺序模拟两个CP rank，说明全局TND数据与每个rank入参之间的关系。假设全局有2个序列，`cmp_ratio=4`：

| 视角 | q范围 | ori_kv范围 | cmp_kv范围 | cu_seqlens_q | cu_seqlens_ori_kv | cu_seqlens_cmp_kv | cmp_residual_kv |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 全局 | seq0 `[0,16)`，seq1 `[0,18)` | seq0 `[0,16)`，seq1 `[0,18)` | seq0 `[0,4)`，seq1 `[0,4)` | `[0,16,34]` | `[0,16,34]` | `[0,4,8]` | `[0,2]` |
| rank0 | seq0 `[0,16)`，seq1 `[0,8)` | seq0 `[0,16)`，seq1 `[0,8)` | seq0 `[0,4)`，seq1 `[0,2)` | `[0,16,24]` | `[0,16,24]` | `[0,4,6]` | `[0,0]` |
| rank1 | seq1 `[8,18)` | seq1 `[0,18)` | seq1 `[0,4)` | `[0,10]` | `[0,18]` | `[0,4]` | `[2]` |

rank1虽然只计算seq1的`[8,18)`，但`ori_kv`和`cmp_kv`需要传到当前位置结束为止的前缀。此时`ori_prefix_len - q_len = 18 - 10 = 8`，kernel推导出的q起点正好是CP切分点。每个本地batch都需要满足`cmp_len * cmp_ratio + cmp_residual_kv[b] == ori_prefix_len`。

```python
import math
import torch
import torch_npu
import cann_ops_transformer


torch_npu.npu.set_device(0)

dtype = torch.float16
cmp_ratio = 4
K = 512
N1 = 64
N2 = 1
D = 512

# 全局packed TND视角：seq0长度16，seq1长度18。
global_q_lens = [16, 18]
global_ori_lens = [16, 18]
global_cmp_lens = [4, 4]
global_cmp_residual = [0, 2]

q_global = torch.randn(sum(global_q_lens), N1, D, dtype=dtype, device="npu")
ori_global = torch.randn(sum(global_ori_lens), N2, D, dtype=dtype, device="npu")
cmp_global = torch.randn(sum(global_cmp_lens), N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")


def make_cu(lengths):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32, device="npu")


def make_cmp_sparse_indices(q_lens, ori_prefix_lens, cmp_lens):
    indices = torch.full((sum(q_lens), N2, K), -1, dtype=torch.int32, device="npu")
    q_base = 0
    for q_len, ori_prefix_len, cmp_len in zip(q_lens, ori_prefix_lens, cmp_lens):
        q_start = ori_prefix_len - q_len
        for row in range(q_len):
            q_pos = q_start + row
            cmp_end = min(cmp_len, (q_pos + 1) // cmp_ratio)
            if cmp_end > 0:
                indices[q_base + row, :, :cmp_end] = torch.arange(
                    cmp_end, dtype=torch.int32, device="npu"
                ).view(1, cmp_end)
        q_base += q_len
    return indices


def run_one_rank(name, q, ori_kv, cmp_kv, q_lens, ori_prefix_lens, cmp_lens, residuals):
    cu_q = make_cu(q_lens)
    cu_ori = make_cu(ori_prefix_lens)
    cu_cmp = make_cu(cmp_lens)
    cmp_residual_kv = torch.tensor(residuals, dtype=torch.int32, device="npu")
    cmp_sparse_indices = make_cmp_sparse_indices(q_lens, ori_prefix_lens, cmp_lens)

    metadata = cann_ops_transformer.sparse_flash_mla_metadata(
        N1,
        N2,
        D,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        max_seqlen_q=max(q_lens),
        max_seqlen_ori_kv=max(ori_prefix_lens),
        max_seqlen_cmp_kv=max(cmp_lens),
        ori_topk=0,
        cmp_topk=K,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        cmp_residual_kv=cmp_residual_kv,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )

    attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(D),
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == q.shape, name
    assert attn_out.dtype == q.dtype, name
    assert softmax_lse.shape == torch.Size([]), name
    assert torch.isfinite(attn_out.float()).all().item(), name
    return attn_out


# rank0：包含完整seq0，并切到seq1前8个token。
rank0_q = torch.cat([q_global[0:16], q_global[16:24]], dim=0)
rank0_ori = torch.cat([ori_global[0:16], ori_global[16:24]], dim=0)
rank0_cmp = torch.cat([cmp_global[0:4], cmp_global[4:6]], dim=0)
rank0_out = run_one_rank(
    "rank0", rank0_q, rank0_ori, rank0_cmp,
    q_lens=[16, 8], ori_prefix_lens=[16, 8], cmp_lens=[4, 2], residuals=[0, 0]
)

# rank1：只算seq1后10个token，但ori_kv/cmp_kv传seq1到18为止的前缀。
rank1_q = q_global[24:34]
rank1_ori = ori_global[16:34]
rank1_cmp = cmp_global[4:8]
rank1_out = run_one_rank(
    "rank1", rank1_q, rank1_ori, rank1_cmp,
    q_lens=[10], ori_prefix_lens=[18], cmp_lens=[4], residuals=[2]
)

seq0_out = rank0_out[:16]
seq1_out = torch.cat([rank0_out[16:24], rank1_out], dim=0)
assert seq0_out.shape == (16, N1, D)
assert seq1_out.shape == (18, N1, D)
```
