# flash_mla_with_kvcache

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

  `flash_mla_with_kvcache`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于非量化 MLA 注意力计算。输入`q`和`k_cache`已合并512维nope与64维rope，算子不执行KV压缩或RoPE变换。

  `k_cache`同时提供Key和Value：Key使用全部576维，Value使用前512维，不单独传入`v`、`q_rope`或`k_rope`。

  `flash_mla_with_kvcache_metadata`是主算子的元数据生成接口，记录AICore/AIVCore任务切分结果。典型调用流程如下：

  1. 准备Query、分页KV缓存、块表和序列长度。
  2. 调用`flash_mla_with_kvcache_metadata`生成`metadata`。
  3. 将`metadata`和对应输入传入`flash_mla_with_kvcache`。

- **计算公式**：

  对每个batch，按`block_table`取出有效KV缓存，计算：

  $$
  K = K_{cache},\qquad V = K_{cache}[..., :DV]
  $$

  $$
  S = softmax\_scale \cdot QK^T + Mask
  $$

  $$
  Attention(Q,K,V) = Softmax(S)V
  $$

  开启`return_softmax_lse`后：

  $$
  softmax\_lse = \log\sum_j e^{S_j}
  $$

  `Mask`在允许访问的位置为0，在屏蔽位置为负无穷。

> [!NOTE]
>
> B表示batch数，Q_T表示所有batch物理Query序列长度之和，Q_N表示Query头数，KV_N表示KV头数，D表示Query/Key逻辑头维，DV表示Value头维。

## 函数原型

调用flash_mla_with_kvcache接口之前，请先调用前置接口flash_mla_with_kvcache_metadata，完成负载均衡的计算。

```python
cann_ops_transformer.flash_mla_with_kvcache_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=None,
    layout_q="BSND",
) -> Tensor
```

```python
cann_ops_transformer.flash_mla_with_kvcache(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
) -> (Tensor, Tensor)
```

## 参数说明

### flash_mla_with_kvcache_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cache_seqlens | Tensor | 必选 | 各batch有效KV长度，与主算子一致 | int32 | ND | (B,) |
| num_heads_q | int | 必选 | Query头数，配套当前主算子时支持64、96 | int32 | - | - |
| num_heads_kv | int | 必选 | KV头数，配套当前主算子时为1 | int32 | - | - |
| cu_seqlens_q | Tensor | 必选 | TND累积Query长度，首项为0 | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 各batch实际使用的Query长度 | int32 | ND | (B,) |
| max_seqlen_q | int | 可选 | 配套主算子时仅传-1或省略；默认None归一化为-1 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 配套主算子时仅传-1或省略；默认None归一化为-1 | int32 | - | - |
| head_dim_qk | int | 可选 | Query/Key逻辑头维，固定576，不能传nope部分的512 | int32 | - | - |
| head_dim_v | int | 可选 | Value头维，固定512 | int32 | - | - |
| mask_mode | int | 可选 | 0或3；默认None归一化为0，与主算子一致 | int32 | - | - |
| layout_q | string | 可选 | 当前须显式传TND，不使用历史默认值BSND | string | - | - |

### flash_mla_with_kvcache

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 已合并nope和rope的Query | bfloat16/float16 | ND | (Q_T, Q_N, D) |
| k_cache | Tensor | 必选 | 同时提供Key和Value的分页缓存 | bfloat16/float16 | ND | PA_BBND：(num_blocks, block_size, KV_N, D)；PA_NZ：(num_blocks, KV_N, D / D0, block_size, D0) |
| block_table | Tensor | 必选 | 各batch逻辑页到物理页的索引映射 | int32 | ND | (B, max_num_blocks_per_seq) |
| cache_seqlens | Tensor | 必选 | 各batch有效KV长度 | int32 | ND | (B,) |
| cu_seqlens_q | Tensor | 必选 | 累积Query长度，首项0，末项Q_T | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 各batch实际使用的Query长度 | int32 | ND | (B,) |
| attn_mask | Tensor | 可选 | mask_mode=3时必传；mask_mode=0时不传 | int8 | ND | (2048, 2048) |
| metadata | Tensor | 必选 | 配套metadata接口生成的调度数据 | int32 | ND | (max_schedule_size,) |
| head_dim_v | int | 可选 | Value头维，固定512，默认512 | int32 | - | - |
| softmax_scale | float | 可选 | 注意力分数缩放系数，默认1.0 | float32 | - | - |
| mask_mode | int | 可选 | 仅0、3，默认0 | int32 | - | - |
| max_seqlen_q | int | 可选 | 仅-1或省略，默认-1 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 仅-1或省略，默认-1 | int32 | - | - |
| layout_q | string | 可选 | 当前须显式传TND | string | - | - |
| layout_kv | string | 可选 | 仅PA_BBND、PA_NZ，默认PA_BBND | string | - | - |
| layout_out | string | 可选 | 当前须显式传NTD | string | - | - |
| return_softmax_lse | bool | 可选 | 是否输出有效LSE，默认False | bool | - | - |

## 返回值说明

### flash_mla_with_kvcache_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | 主算子的任务切分数据 | int32 | ND | (max_schedule_size,) |

metadata的shape根据batch数和设备核数动态计算。

### flash_mla_with_kvcache

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | 必选 | 注意力输出，与q的数据类型相同 | bfloat16/float16 | ND | (Q_N, Q_T, DV) |
| softmax_lse | Tensor | 可选 | Softmax的LSE结果，由return_softmax_lse控制 | float32 | ND | 开启：(Q_N, Q_T)；关闭：(0,) |

**说明**

- attn_out：D维度为head_dim_v，其余维度由query按layout_out排布。
- softmax_lse：return_softmax_lse为True时，shape为(Q_N, Q_T)；为False时，shape为(0,)。

## 约束说明

- `block_table`、序列长度、`attn_mask`和`metadata`中的值由调用者保证正确。Tiling阶段不能全面校验设备Tensor内容，非法值可能导致精度错误或非法内存访问。
- metadata接口和主算子的序列长度、头数、头维、掩码模式及Query布局必须一致。输入变化后应重新生成对应metadata。

### 特性参数组

| 特性参数组 | 参数字段名称 | 字段分组 | 字段类型 |
| :--- | :--- | :--- | :--- |
| 公共参数组 | q、k_cache、metadata | INPUT | Tensor |
| | head_dim_v | ATTR(OPTIONAL) | int |
| | softmax_scale | ATTR(OPTIONAL) | float |
| | layout_q、layout_kv、layout_out | ATTR(OPTIONAL) | string |
| | attn_out | OUTPUT | Tensor |
| Mask参数组 | mask_mode | ATTR(OPTIONAL) | int |
| | attn_mask | INPUT(OPTIONAL) | Tensor |
| Paged Attention参数组 | block_table | INPUT | Tensor |
| SeqLens参数组 | cache_seqlens、cu_seqlens_q | INPUT | Tensor |
| | seqused_q | INPUT(OPTIONAL) | Tensor |
| | max_seqlen_q、max_seqlen_kv | ATTR(OPTIONAL) | int |
| SoftmaxLSE参数组 | return_softmax_lse | ATTR(OPTIONAL) | bool |
| | softmax_lse | OUTPUT | Tensor |

### 基准信息说明

| 命名 | 含义 |
| :--- | :--- |
| B | batch数 |
| Q_N / N1 | 输入q tensor的头数 |
| KV_N / N2 | 输入k_cache tensor的头数 |
| Q_S / KV_S | 单个batch的有效Query/KV长度 |
| Q_T / T | 所有batch物理Query序列长度之和 |
| D | Query/Key逻辑头维 |
| DV | Value及attn_out头维 |
| num_blocks | KV缓存物理块数 |
| max_num_blocks_per_seq | 每个batch的块表容量 |
| block_size | 每个物理块的token容量 |
| D0 | PA_NZ布局的最内层分块维度 |

### 参数组约束

#### 公共参数组

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
| :--- | :--- | :--- | :--- | :--- |
| q | float16/bfloat16；TND，(Q_T, Q_N, D) | 必须存在 | 与k_cache、attn_out数据类型一致 | 0 < B < 65536；Q_T > 0；Q_N支持64、96；KV_N=1、D=576 |
| k_cache | float16/bfloat16；PA_BBND：(num_blocks, block_size, KV_N, D)；PA_NZ：(num_blocks, KV_N, D / D0, block_size, D0)；非连续Tensor：PA_BBND仅dim0支持，PA_NZ支持dim0或dim1 | 必须存在 | 逻辑D与q相同 | block_size=128；D0=16 |
| attn_out | float16/bfloat16；NTD，(Q_N, Q_T, DV) | 必须存在 | Q_T、Q_N与q对应 | DV=head_dim_v=512 |
| metadata | int32、一维、ND、非空；容量不小于当前设备和B所需容量 | 必须传入 | 必须对应本次调用参数 | 由配套metadata接口生成 |
| head_dim_v | 仅512 | 可省略，默认512 | head_dim_v+64=576 | Value取缓存前512维 |
| softmax_scale | float，默认1.0 | 可省略 | 无 | 不自动按D缩放 |

layout匹配关系表：

| layout_q | layout_kv | layout_out | layout_softmax_lse |
| :--- | :--- | :--- | :--- |
| TND | PA_BBND、PA_NZ | NTD | (Q_N, Q_T) |

#### Mask参数组

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
| :--- | :--- | :--- | :--- | :--- |
| mask_mode | 仅0、3 | 可省略，默认0 | 与metadata生成时相同 | 0无掩码；3右对齐因果掩码 |
| attn_mask | int8、ND、(2048, 2048) | mode=0不传；mode=3必传 | 对角线及其下方为0，上方为1 | 配合mode=3解释为右对齐因果掩码 |

#### Paged Attention参数组

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
| :--- | :--- | :--- | :--- | :--- |
| block_table | int32、ND；shape为(B, max_num_blocks_per_seq)；max_num_blocks_per_seq > 0；block_size=128 | 必须传入 | B与cache_seqlens、cu_seqlens_q对应 | 必须传入cache_seqlens；layout_kv为PA_NZ时，D为D0的倍数 |

#### SeqLengths参数组

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
| :--- | :--- | :--- | :--- | :--- |
| cache_seqlens | int32、ND、(B,)，元素非负 | 必须传入 | 与metadata生成时一致 | 与block_table的batch数一致 |
| cu_seqlens_q | int32、ND、(B+1,)，非递减，首项0、末项Q_T | TND场景必须传入 | B与cache_seqlens一致 | 相邻差值表示各batch物理Query长度 |
| seqused_q | int32、ND、(B,)，元素非负 | 可选 | 与metadata生成时一致 | 每项不超过对应cu_seqlens_q相邻差值；省略时使用相邻差值 |
| max_seqlen_q | int | 可省略 | 仅允许-1或省略 | 由Query序列长度信息推导，不接受0、正数或其他负数 |
| max_seqlen_kv | int | 可省略 | 仅允许-1或省略 | 由cache_seqlens推导，不接受0、正数或其他负数 |

#### SoftmaxLSE参数组

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
| :--- | :--- | :--- | :--- | :--- |
| return_softmax_lse | bool，仅True/False | 可省略，默认False | 控制LSE输出是否有效 | 不改变attn_out的布局和shape |
| softmax_lse | float32 | 始终占返回二元组的第二项 | True：(Q_N, Q_T)；False：(0,) | 无 |

## 调用示例

以下示例使用 TND Query、PA_NZ 缓存、NTD 输出及右对齐因果掩码，省略两个最大序列长度参数。

```python
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)
device = "npu"
dtype = torch.bfloat16

q = torch.randn(8, 96, 576, dtype=dtype, device=device)
# 两个 batch 各使用一个 128-token 物理块。
k_cache = torch.randn(2, 1, 36, 128, 16, dtype=dtype, device=device)
block_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
cache_seqlens = torch.tensor([128, 96], dtype=torch.int32, device=device)
cu_seqlens_q = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
attn_mask = torch.triu(
    torch.ones(2048, 2048, dtype=torch.int8, device=device), diagonal=1
)

metadata = cann_ops_transformer.ops.flash_mla_with_kvcache_metadata(
    cache_seqlens,
    num_heads_q=96,
    num_heads_kv=1,
    cu_seqlens_q=cu_seqlens_q,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=3,
    layout_q="TND",
)

attn_out, softmax_lse = cann_ops_transformer.ops.flash_mla_with_kvcache(
    q,
    k_cache,
    block_table=block_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,
    attn_mask=attn_mask,
    metadata=metadata,
    head_dim_v=512,
    softmax_scale=576 ** -0.5,
    mask_mode=3,
    layout_q="TND",
    layout_kv="PA_NZ",
    layout_out="NTD",
    return_softmax_lse=True,
)
torch_npu.npu.synchronize()
assert attn_out.shape == (96, 8, 512)
assert softmax_lse.shape == (96, 8)
assert attn_out.dtype == dtype
assert softmax_lse.dtype == torch.float32
```
