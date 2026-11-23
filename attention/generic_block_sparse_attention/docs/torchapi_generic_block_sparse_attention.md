# generic_block_sparse_attention

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

  `generic_block_sparse_attention`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`GenericBlockSparseAttention`算子完成任意粒度块稀疏注意力计算。

  `generic_block_sparse_attention_metadata`是`generic_block_sparse_attention`的元数据生成接口，用于在主算子执行前生成metadata。metadata记录AI Core/AIV Core的任务切分结果，当前主算子必须传入配套metadata；虽然Python签名默认值为None，实际调用不支持省略，也不会自动生成metadata。典型调用流程如下：

  1. 准备`q`、`k`、`v`、`sparse_block_idx`、`sparse_block_count`等输入。
  2. 调用`generic_block_sparse_attention_metadata`生成`metadata`。
  3. 调用`generic_block_sparse_attention`，将上一步得到的`metadata`传入主算子。

- **计算公式**：

  本算子在标准 Attention 基础上，沿序列维按块进行稀疏计算。稀疏块大小为$blockShapeX \times blockShapeY$：Q 按$blockShapeX$、KV 按$blockShapeY$划分；`sparse_block_idx`给出每个Q块选中的KV块索引，`sparse_block_count`给出每个Q块实际参与计算的KV块个数，仅对这些块完成$q k^{T}$、Softmax及与$v$的乘积。

  $$
  attention\_out = Softmax(softmax\_scale \cdot q \cdot k_{sparse}^{T} + attn\_mask) \cdot v_{sparse}
  $$

  其中$softmax\_scale$为缩放系数；$k_{sparse}$、$v_{sparse}$表示按上述索引选取后的KV块。

> [!NOTE]
>
> q、k、v数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、Q_T表示所有query Batch输入样本序列长度的累加和，KV_T表示所有k、v Batch输入样本序列长度的累加和。
> Q_S表示seqlen_q，Q_N表示nheads_q，KV_S表示seqlen_kv，KV_N表示nheads_kv。

## 函数原型

调用generic_block_sparse_attention接口之前，请先调用前置接口generic_block_sparse_attention_metadata，完成generic_block_sparse_attention负载均衡的计算。

```python
cann_ops_transformer.generic_block_sparse_attention_metadata(
    sparse_block_idx,
    sparse_block_count,
    num_heads_q,
    num_heads_kv,
    head_dim,
    block_shape,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="TND",
    layout_kv="PA_BBND",
    layout_sparse_pattern=4,
    mask_mode=1,
    quant_mode=0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
    residual_block_mode=0,
    is_consistent_topk=False,
) -> Tensor
```

```python
cann_ops_transformer.generic_block_sparse_attention(
    q,
    k,
    v,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    *,
    metadata=None,
    attn_mask=None,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    p_quant_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    layout_sparse_pattern=4,
    softmax_scale=0.0,
    mask_mode=1,
    quant_mode=0,
    dst_type_max=0.0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
    return_softmax_lse=False,
    residual_block_mode=0,
    is_consistent_topk=False,
    attention_out_dtype=None,
) -> (Tensor, Tensor)
```

## 枚举说明

`quant_mode`、`mask_mode`与`residual_block_mode`在 Python 接口中支持传入 `IntEnum` 枚举或对应 int 值，枚举定义于 `cann_ops_transformer.ops.generic_block_sparse_attention`：

### quant_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `NO_QUANT` | 0 | 非量化（默认值） |
| `FP8_E4M3_STATIC_PER_GROUP` | 1 | FP8_E4M3 静态 per-group |
| `FP8_E4M3_DYNAMIC_MX` | 2 | FP8_E4M3 动态 MX |
| `FP4_E2M1_DYNAMIC_OCP` | 3 | FP4_E2M1 动态 OCP，预留、未实现 |
| `FP4_E2M1_DYNAMIC_CX` | 4 | A4C4 CX，Q/K/P group32、V per-channel |
| `FP8_E4M3_STATIC_CAST_P` | 5 | FP8_E4M3 静态 cast P |
| `MXFP8_E4M3_PER_CHANNEL_V` | 6 | A8C8：Q/K 按 D 维每 32 个元素使用一个 E8M0 scale，V 按 channel 量化；须显式指定 softmax_precision=0。 |
| `INT8_FULL_QUANT` | 7 | INT8 全量化，四个外部 scale 均必传 |
| `Q_FP8_PER_TENSOR_KV_FP4_PER_TOKEN_PER_GROUP` | 8 | Q 为 FP8 E4M3FN，K/V 为打包的 FP4 E2M1，K/V scale 为 BF16。 |

<!-- npu="A3,910b" id17 -->
- <term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>支持非量化（0）和INT8全量化（7）。
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950PR&950DT系列产品</term>支持非量化（0）、A4C4（4）、常规FP8（5）、A8C8（6）和A8C4（8）。
<!-- end id18 -->

### mask_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `NO_MASK` | 0 | 不加 mask |
| `CAUSAL` | 1 | Causal 模式（默认值） |
| `WINDOW` | 2 | Window 模式 |
| `UNION` | 3 | Union exactMask 模式 |

### residual_block_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `MARKED_BY_SPARSE_BLK_IDX` | 0 | 尾块是否参与计算由sparseBlockIdx确定 |
| `INCOMPLETE_BLK_KEPT_BUT_NOT_IN_SPARSE_BLK_IDX` | 1 | 不完整的尾块一定参与计算，但不包含在sparseBlockIdx中 |

> [!NOTE]
>
> 枚举可传 `IntEnum` 或对应整数。`quant_mode` 支持 0（非量化）、4（A4C4）、5（FP8）、6（A8C8）、7（<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term> INT8）和 8（FP8 Q 和 FP4 K/V）。4、5、6 和 8 适用于 <term>Ascend 950PR&950DT系列产品</term>。INT8 支持 `mask_mode=0/1`，要求 `softmax_precision=0`。各模式的详细条件见产品差异约束。

## layout_sparse_pattern 取值与sparse_block_idx/sparse_block_count的shape对应关系说明

| 值 | sparse_block_idx的shape | sparse_block_count的shape | 含义 |
| :---: | :--- | :--- | :--- |
| 0 | [batch, numKeyValueHeads, maxQBlockCount, maxKvBlockCount] | [batch, numKeyValueHeads, maxQBlockCount] | 同一个Group中的qHead共享sparsePattern，表示每个Q块选择了哪些KV块 |
| 1 | [batch, numKeyValueHeads, maxKvBlockCount, maxQBlockCount] | [batch, numKeyValueHeads, maxKvBlockCount] | 同一个Group中的qHead共享sparsePattern，表示每个KV块选择了哪些Q块 |
| 2 | [batch, headNum, maxQBlockCount, maxKvBlockCount] | [batch, headNum, maxQBlockCount] | 同一个Group中的qHead有独立的sparsePattern，表示每个Q块选择了哪些KV块 |
| 3 | [batch, headNum, maxKvBlockCount, maxQBlockCount] | [batch, headNum, maxKvBlockCount] | 同一个Group中的qHead有独立的sparsePattern，表示每个KV块选择了哪些Q块 |
| 4 | [numKeyValueHeads, totalQBlocks, maxKvBlockCount] | [numKeyValueHeads, totalQBlocks] | 同一个Group中的qHead共享sparsePattern，表示每个Q块选择了哪些KV块（当前默认值） |
| 5 | [numKeyValueHeads, totalKBlocks, maxQBlockCount] | [numKeyValueHeads, totalKBlocks] | 同一个Group中的qHead共享sparsePattern，表示每个KV块选择了哪些Q块 |
| 6 | [headNum, totalQBlocks, maxKvBlockCount] | [headNum, totalQBlocks] | 同一个Group中的qHead有独立的sparsePattern，表示每个Q块选择了哪些KV块 |
| 7 | [headNum, totalKBlocks, maxQBlockCount] | [headNum, totalKBlocks] | 同一个Group中的qHead有独立的sparsePattern，表示每个KV块选择了哪些Q块 |

> [!NOTE]
>
> 当前仅支持 layout_sparse_pattern = 4。

## 参数说明

### generic_block_sparse_attention_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sparse_block_idx | Tensor | 必选 | 每个Q块选择的KV块索引 | int32 | ND | (KV_N, totalQBlocks, maxKvBlockCount) |
| sparse_block_count | Tensor | 必选 | 每个Q块实际保留的KV块数量 | int32 | ND | (KV_N, totalQBlocks) |
| num_heads_q | int | 必选 | Query head数；A4C4的num_heads_q/num_heads_kv范围为[1,16]。 | int32 | - | - |
| num_heads_kv | int | 必选 | Key/Value head数 | int32 | - | - |
| head_dim | int | 必选 | 每个注意力头的维度；A4C4使用逻辑D=192或256。 | int32 | - | - |
| block_shape | list[int] | 必选 | 稀疏块形状 `[block_x, block_y]`；A4C4使用[1,Y]，Y∈{4,8,16,32,64,128}。 | int64 | - | 长度为2 |
| cu_seqlens_q | Tensor | 条件必选 | TND必须传入Q存储长度前缀和；BSND/BNSD不使用，建议传入None；第一个元素为0 | int64 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | 条件必选 | TND KV必须传入KV存储长度前缀和；BSND/BNSD不使用，建议传入None；PA_BBND/PA_BNBD必须为None | int64 | ND | (B+1,) |
| seqused_q | Tensor | 条件必选 | BSND/BNSD 必传。TND 下，非量化或 `quant_mode=8` 时可选，INT8 时传 `None`。表示每个 batch 的实际 Q 长度。 | int32 | ND | (B,) |
| seqused_kv | Tensor | 条件必选 | PA_BBND/PA_BNBD及BSND/BNSD KV必须传入；TND KV可选，指定每batch实际KV长度 | int32 | ND | (B,) |
| max_seqlen_q | int | 条件必选 | BSND/BNSD必须传入Q的shape容量Sq；TND为可选长度上限 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 允许-1或非负值，不用于推导BSND/BNSD的实际KV长度 | int32 | - | - |
| layout_q | string | 可选 | 定义输入query张量的布局格式 | string | - | - |
| layout_kv | string | 可选 | 定义输入key/value张量的布局格式 | string | - | - |
| layout_sparse_pattern | int | 可选 | 代表输入的sparse_block_idx、sparse_block_count的数据排布格式，取值见「layout_sparse_pattern 取值与sparse_block_idx/sparse_block_count的shape对应关系说明」 | int64 | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」 | int32 | - | - |
| quant_mode | int/QuantMode | 可选 | 量化模式，支持传入枚举或对应 int 值，枚举定义见「quant_mode 枚举」 | int32 | - | - |
| softmax_precision | int | 可选 | Softmax精度模式 | int32 | - | - |
| win_left | int | 可选 | window左界限 | int32 | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - |
| residual_block_mode | int/ResidualBlockMode | 可选 | 表示KV序列以blockShapeY为单位进行稀疏后，末尾块的状态，枚举定义见「residual_block_mode 枚举」 | int64 | - | - |
| is_consistent_topk | bool | 可选 | 前置算子进行稀疏块选择时，同一batch同一head内每个Q块选择的KV块最大数量是否一致 | bool | - | - |

### generic_block_sparse_attention

表格中的D表示逻辑headDim；A4C4的Torch张量物理末维为D/2，打包存储及scale约定见[A4C4说明](#a4c4quantmode4)。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的q；模式4使用packed uint8，物理末维为D/2。 | bfloat16/float16/float8_e4m3fn/int8/uint8 | ND | TND: (Q_T,Q_N,D)；BSND: (B,Sq,Q_N,D)；BNSD: (B,Q_N,Sq,D) |
| k | Tensor | 必选 | 公式中的 k。`quant_mode=4/8` 时，Torch 张量末轴 `D_kv` 为逻辑 `D/2`，其余模式为 `D`。模式8要求 K/V 的数据类型一致，均为 `float4_e2m1fn_x2` 或均为存有打包 FP4 数据的 `uint8`。 | bfloat16/float16/float8_e4m3fn/int8/float4_e2m1fn_x2/uint8 | ND | PA_BBND: (num_blocks, block_size, KV_N, D_kv)；PA_BNBD: (num_blocks, KV_N, block_size, D_kv)；TND: (KV_T,KV_N,D_kv)；BSND: (B,Sk,KV_N,D_kv)；BNSD: (B,KV_N,Sk,D_kv) |
| v | Tensor | 必选 | 公式中的 v。`D_kv` 和支持的数据类型与 k 相同。 | bfloat16/float16/float8_e4m3fn/int8/float4_e2m1fn_x2/uint8 | ND | 与 k 相同 |
| sparse_block_idx | Tensor | 必选 | 每个Q块选择的KV块索引 | int32 | ND | (KV_N, totalQBlocks, maxKvBlockCount) |
| sparse_block_count | Tensor | 必选 | 每个Q块实际保留的KV块数量 | int32 | ND | (KV_N, totalQBlocks) |
| block_shape | list[int] | 必选 | 稀疏块形状 `[block_x, block_y]`；A4C4使用[1,Y]，Y∈{4,8,16,32,64,128}。 | int64 | - | 长度为2 |
| metadata | Tensor | 必选 | 必须传入`generic_block_sparse_attention_metadata`生成的配套任务切分结果 | int32 | ND | (1024,) |
| attn_mask | Tensor | 可选 | mask_mode=0/1时须为None；mask_mode=3（Union）时须为INT32 exactMask | bool/int32 | ND | Union: (KV_N, G, U×block_y) |
| q_dequant_scale | Tensor | INT8、A4C4必传 | quant_mode=0/5/8为None；INT8全量化（quant_mode=7）须传连续、有限且大于0的float32 Tensor；A4C4（模式4）必传E8M0或uint8编码。 | float32/float8_e8m0fnu/uint8 | ND | TND: [Q_T,Q_N]；BSND: [B,Q_S,Q_N]；BNSD: [B,Q_N,Q_S] <br>A4C4：[scale形状](#a4c4quantmode4)。 |
| k_dequant_scale | Tensor | INT8和`quant_mode=8`必传；A4C4必传 | quant_mode=0/5为None；INT8全量化（quant_mode=7）须传连续、有限且大于0的float32；A4C4（模式4）必传E8M0或uint8编码。 Tensor | float32/bfloat16/uint8 | ND | PA_BBND/PA_BNBD: [num_blocks,KV_N,128]；TND: [KV_N,KV_T]；BSND/BNSD: [B,KV_N,KV_S]<br>`quant_mode=8` 时，按 K/V 的逻辑形状将末轴 D 替换为 8。 <br>A4C4：[scale形状](#a4c4quantmode4)。 |
| v_dequant_scale | Tensor | INT8和`quant_mode=8`必传；A4C4必传 | quant_mode=0/5为None；INT8全量化（quant_mode=7）须传连续、有限且大于0的float32；A4C4（模式4）必传E8M0或uint8编码。 Tensor | float32/bfloat16/uint8 | ND | [KV_N,D]<br>`quant_mode=8` 时，按 K/V 的逻辑形状将末轴 D 替换为 8。 <br>A4C4：[scale形状](#a4c4quantmode4)。 |
| p_quant_scale | Tensor | INT8必传 | quant_mode=0/5/8为None；INT8全量化（quant_mode=7）须传连续、有限且大于0的float32 Tensor | float32 | ND | [1] |
| cu_seqlens_q | Tensor | 条件必选 | TND必须传入Q存储长度前缀和；BSND/BNSD不使用，建议传入None；第一个元素为0 | int64 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | 条件必选 | TND KV必须传入KV存储长度前缀和；BSND/BNSD不使用，建议传入None；PA_BBND/PA_BNBD必须为None | int64 | ND | (B+1,) |
| seqused_q | Tensor | 条件必选 | BSND/BNSD 必传。TND 下，非量化或 `quant_mode=8` 时可选，INT8 时传 `None`。表示每个 batch 的实际 Q 长度。 | int32 | ND | (B,) |
| seqused_kv | Tensor | 条件必选 | PA_BBND/PA_BNBD及BSND/BNSD KV必须传入；TND KV可选，指定每batch实际KV长度 | int32 | ND | (B,) |
| block_table | Tensor | 条件必选 | PA_BBND/PA_BNBD必须传入页表；TND/BSND/BNSD KV必须为None | int32 | ND | (B, max_num_blocks_per_seq) |
| layout_q | string | 可选 | 定义输入query张量的布局格式 | string | - | - |
| layout_kv | string | 可选 | 定义输入key/value张量的布局格式 | string | - | - |
| layout_sparse_pattern | int | 可选 | 代表输入的sparse_block_idx、sparse_block_count的数据排布格式，取值见「layout_sparse_pattern 取值与sparse_block_idx/sparse_block_count的shape对应关系说明」 | int64 | - | - |
| softmax_scale | float | 可选 | 可显式设置缩放因子，覆盖默认计算 | float32 | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」 | int32 | - | - |
| quant_mode | int/QuantMode | 可选 | 量化模式，支持传入枚举或对应 int 值，枚举定义见「quant_mode 枚举」 | int32 | - | - |
| dst_type_max | float | 可选 | 模式4中，0表示默认CX=6，显式值须有限且在[6,12]范围内；其他模式须为0。 | float32 | - | - |
| softmax_precision | int | 可选 | Softmax精度模式 | int32 | - | - |
| win_left | int | 可选 | window左界限 | int32 | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - |
| return_softmax_lse | bool | 可选 | 是否需要获取softmax的LSE结果 | bool | - | - |
| residual_block_mode | int/ResidualBlockMode | 可选 | 表示KV序列以blockShapeY为单位进行稀疏后，末尾块的状态，枚举定义见「residual_block_mode 枚举」 | int64 | - | - |
| is_consistent_topk | bool | 可选 | 前置算子进行稀疏块选择时，同一batch同一head内每个Q块选择的KV块最大数量是否一致 | bool | - | - |
| attention_out_dtype | dtype | 可选 | 输出dtype；`quant_mode!=0`时必填，`quant_mode=0`且未指定时与q一致 | ScalarType | - | - |

## 返回值说明

### generic_block_sparse_attention_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | generic_block_sparse_attention的任务切分数据 | int32 | ND | (1024,) |

### generic_block_sparse_attention

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 必选 | generic_block_sparse_attention的计算输出 | bfloat16/float16 | ND | TND: (Q_T,Q_N,D)；BSND: (B,Sq,Q_N,D)；BNSD: (B,Q_N,Sq,D) |
| softmax_lse | Tensor | 可选 | softmax的LSE结果 | float32 | ND | TND: (Q_T,Q_N,1)；BSND/BNSD: (B,Q_N,Sq,1)；或空 |

**说明**

- attention_out：限制：非MXFP4模式的shape与query一致；MXFP4模式的query以uint8存两个FP4值，输出末维为query末维的2倍；`quant_mode=0`时dtype默认与q一致，`quant_mode!=0`时由attention_out_dtype指定。
- softmax_lse：return_softmax_lse为True时，TND布局下输出shape为(Q_T,Q_N,1)的Tensor，BSND/BNSD布局下输出(B,Q_N,Sq,1)；return_softmax_lse为False时，则输出空Tensor。

## 约束说明

- 参数cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv、sparse_block_idx、sparse_block_count及block_table属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- generic_block_sparse_attention_metadata和generic_block_sparse_attention的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由用户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 产品差异约束

- <term>Ascend 950PR&950DT系列产品</term>的A4C4（quant_mode=4）：D=192/256，block_shape=[1,Y]，Y∈{4,8,16,32,64,128}，GQA分组大小1～16，mask_mode=0/1、residual_block_mode=0/1、softmax_precision=0。Q/K/V使用FLOAT4_E2M1，scale为E8M0，输出FP16/BF16，可选输出FP32 LSE。详见[A4C4接口约束](#a4c4quantmode4)。
- <term>Ascend 950PR&950DT系列产品</term>上，Metadata 与主算子均设 `quant_mode=QuantMode.Q_FP8_PER_TENSOR_KV_FP4_PER_TOKEN_PER_GROUP`（8）。先调用 `generic_block_sparse_attention_metadata`，再将返回张量显式传给主算子的 `metadata` 参数。逻辑 D=128，`block_shape=[1,128]`，GQA 分组大小 1～32，`mask_mode=0`（非因果）或 `1`（因果），`softmax_precision=0`，`residual_block_mode=0`。显式指定 `attention_out_dtype` 为 float16 或 bfloat16，可选 LSE。
  - `max_seqlen_q` 和每个 batch 的实际 Q 序列长度均不超过 16。BSND 和 BNSD 的 Q 张量序列轴长度也不超过 16。KV 布局为 PA_BBND、BSND 或 TND 时，`num_heads_kv` 不超过 5。BNSD 和 PA_BNBD 不受此 5 个 KV head 的限制。
  - Q 支持 `TND`、`BSND`、`BNSD`。非分页 K/V 与 Q 使用同名布局；这三种 Q 布局也都可搭配 `PA_BBND` 或 `PA_BNBD`，页大小为 128。输出形状见 [layout 匹配关系表](#layout匹配关系表)。
  - Q 使用 `torch.float8_e4m3fn`，算子固定使用 Q scale `1.0`，`q_dequant_scale` 传 `None`。K/V 均使用 `torch.float4_e2m1fn_x2`，或均使用承载已打包 FP4 数据的 `torch.uint8`。两种表示的 Torch 张量末轴均为逻辑 `D/2`（D=128 时为 64 字节，每字节低 4 位和高 4 位分别存储偶数和奇数 D 索引的 FP4 E2M1 值）。对于每个 token 和每个 KV head，K 和 V 沿 D 轴每 16 个元素各用一份 BF16 scale，对应的 scale 张量末轴为逻辑 `D/16`。非分页 scale 使用连续 ND 张量。`p_quant_scale` 传 `None`。
  - 分页 K、V、K scale、V scale 的首轴 stride 可独立取值，各张量页内连续。打包 K/V 的 Torch 首轴 stride 以字节计，至少为 `block_size × KV_N × 64`，且是 `KV_N × 64` 的整数倍。两份 scale 的首轴 stride 以 BF16 元素计，至少为 `block_size × KV_N × 8`，且是 `KV_N × 8` 的整数倍。详见 [ACLNN 约束](./aclnnGenericBlockSparseAttention.md)。

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>非量化FLOAT16/BFLOAT16场景：`head_dim`支持1～256的任意值；`layout_q/layout_kv`支持`TND/PA_BBND`、`TND/PA_BNBD`、`TND/TND`、`BSND/BSND`、`BNSD/BNSD`、`BSND/PA_BBND`、`BNSD/PA_BBND`、`BSND/PA_BNBD`和`BNSD/PA_BNBD`，不支持其他组合；`mask_mode`支持0（非因果）、1（causal）和3（union）；非union时`block_shape=[1,Y]`（`1≤Y≤128`），union（`mask_mode=3`）时`block_shape=[unionSize,Y]`，`unionSize`从`block_shape[0]`读取且**当前仅支持8**；`residual_block_mode`支持0或1（union仅支持0）。
- <term>Ascend 950PR&950DT系列产品</term> Union（`mask_mode=3`）仅支持 `quant_mode=0`（FP16/BF16 非量化）和 `quant_mode=5`（FP8）。须传 INT32 exactMask，`block_shape=[8,Y]`（`1≤Y≤128`），`residual_block_mode=0`；FP8 Union 另要求 `head_dim=128`，分页 KV 下要求 Y 整除页大小 128。
- <term>Ascend 950PR&950DT系列产品</term> 常规 FP8（`quant_mode=5`）要求 `head_dim=128`、`mask_mode=1`、`residual_block_mode=0`、`block_shape=[1,Y]`（`1≤Y≤128`）。
- <term>Ascend 950PR&950DT系列产品</term> 常规 FP8（`quant_mode=5`、`mask_mode=1`）高精模式（`softmax_precision=0`）仅支持 `head_dim=128` 和 `1≤group_size≤16`。Q/KV 支持 `TND/TND`、`BSND/BSND`、`BNSD/BNSD`，以及上述三种 Q 布局分别搭配 `PA_BBND` 或 `PA_BNBD`。`return_softmax_lse` 支持 0 或 1。低精模式（`softmax_precision=1`）仅支持 `TND/PA_BBND`，且关闭 LSE。
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>非量化FLOAT16/BFLOAT16场景：`head_dim`支持1～256的任意值；`layout_q/layout_kv`支持`TND/PA_BBND`、`TND/PA_BNBD`、`TND/TND`、`BSND/BSND`、`BNSD/BNSD`、`BSND/PA_BBND`、`BNSD/PA_BBND`、`BSND/PA_BNBD`和`BNSD/PA_BNBD`；`block_shape=[1,Y]`，Y支持1、2、4、8、16、32、64、128；`mask_mode`支持0（非因果）、1（causal）和3（union，见下方Union约束）；`residual_block_mode`支持0或1。TND KV须传入`cu_seqlens_kv`，dense Q/K/V按对应布局连续存储；causal模式下各batch须满足实际KV长度不小于实际Q长度。
- <term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term> Union（`mask_mode=3`）仅支持 `quant_mode=0`（FP16/BF16 非量化）、`block_shape=[8,Y]`、`residual_block_mode=0`。Y 支持 1、2、4、8、16、32、64、128，分页 KV 的页大小为 128。支持 TND、BSND、BNSD 的同布局 dense KV 以及 PA_BBND、PA_BNBD 分页 KV。须传 INT32 exactMask，GQA 分组大小与 blockShapeX 的乘积须不大于 128。
<!-- end id8 -->
<!-- npu="950,A3,910b" id9 -->
- <term>Ascend 950PR&950DT系列产品</term> 常规 FP8 量化：`head_dim`仅支持128，`mask_mode=1`、`residual_block_mode=0`、`block_shape=[1,Y]`（`1≤Y≤128`）。
- <term>Ascend 950PR&950DT系列产品</term> 常规 FP8（`quant_mode=5`、`mask_mode=1`）高精模式（`softmax_precision=0`）仅支持 `head_dim=128` 和 `1≤groupSize≤16`。Q/KV 支持 `TND/TND`、`BSND/BSND`、`BNSD/BNSD`，以及上述三种 Q 布局分别搭配 `PA_BBND` 或 `PA_BNBD`。 `return_softmax_lse` 支持 0 或 1。 低精模式（`softmax_precision=1`）仅支持 `TND/PA_BBND`，且关闭 LSE。
- <term>Ascend 950PR&950DT系列产品</term> 常规低精 FP8（quantType=5，softmax_precision=1）：`head_dim`仅支持128，`layout_q/layout_kv`仅支持`TND/PA_BBND`，`mask_mode=1`、`residual_block_mode=0`、`block_shape=[1,Y]`（`1≤Y≤128`）。FP8 Union 见上条 Union 约束；Union 不支持 `quantType=6`。
- <term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term> INT8路径支持`residual_block_mode=0/1`；mode=1的尾块不写入`sparse_block_idx`，不计入`sparse_block_count`，metadata和计算接口必须使用同一模式。
<!-- end id9 -->
<!-- npu="950,A3,910b" id10 -->
- 稀疏索引最后一维容量：<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>非量化和INT8全量化支持1～INT32_MAX，<term>Ascend 950PR&950DT系列产品</term>非量化FLOAT16/BFLOAT16及常规 FP8（含 FP8 Union）支持1～INT32_MAX；容量均须不小于稀疏块有效计数的最大值。
<!-- end id10 -->
<!-- npu="950,A3,910b" id11 -->
- `quant_mode`：<term>Ascend 950PR&950DT系列产品</term>支持0（非量化）、4（A4C4）、5（FP8量化）或8（FP8 Q 和 FP4 K/V）；<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>支持0（非量化）或7（INT8全量化）。
<!-- end id11 -->
<!-- npu="950,A3,910b" id12 -->
- `softmax_precision`：<term>Ascend 950PR&950DT系列产品</term>非Union场景（mask_mode=0/1）下，非量化FLOAT16/BFLOAT16支持0或1，quant_mode=5的FP8量化场景支持0或1；Union场景（mask_mode=3）下，非量化与FP8量化均仅支持0；<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>非Union场景下FLOAT16支持0或1，BFLOAT16仅支持0，Union场景（`mask_mode=3`）下仅支持0。
- <term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>非量化场景支持在上述布局组合及`residual_block_mode=0`或1时输出LSE。常规低精 FP8（quant_mode=5、softmax_precision=1）不支持输出LSE；常规高精 FP8（`quant_mode=5`、`softmax_precision=0`）和 FP8 Union（D=128）均支持输出 LSE。
<!-- end id12 -->
- 每组有效稀疏索引须非负、严格升序且不重复，位于前`sparse_block_count`个位置；随机升序即可，不要求等步长或物理页连续。
<!-- npu="A3,910b" id14 -->
- <term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>在`mask_mode=1`时，第q个有效Q token的KV可见长度为`kvLen-qLen+q+1`，q从0开始；`mask_mode=0`时可见长度为`kvLen`。`residual_block_mode=1`时显式列表只包含可见范围内的完整KV块，尾部不完整块由算子隐式加入，不计入`sparse_block_count`及topK容量。
<!-- end id14 -->

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |         k         |      INPUT      |   Tensor   |
|                      |         v         |      INPUT      |   Tensor   |
|                      |  sparse_block_idx  |      INPUT      |   Tensor   |
|                      | sparse_block_count  |      INPUT      |   Tensor   |
|                      |     block_shape     |   ATTR(REQUIRED) |  int[]  |
|                      |       metadata       | INPUT（当前必传） |   Tensor   |
|                      |    softmax_scale    | ATTR(OPTIONAL) |   float   |
|                      |  softmax_precision  | ATTR(OPTIONAL) |   int   |
|                      |      layout_q       | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |  layout_sparse_pattern  | ATTR(OPTIONAL) |   int64   |
|                      | residual_block_mode | ATTR(OPTIONAL) |   int64   |
|                      | is_consistent_topk | ATTR(OPTIONAL) |   bool   |
|                      |   attention_out   |     OUTPUT     |   Tensor   |
|   metadata参数组   |   max_seqlen_q   | ATTR(OPTIONAL) |   int   |
|                      |  max_seqlen_kv  | ATTR(OPTIONAL) |   int   |
|                      |   num_heads_q   | ATTR(REQUIRED) |   int   |
|                      |  num_heads_kv  | ATTR(REQUIRED) |   int   |
|                      |     head_dim     | ATTR(REQUIRED) |   int   |
|                      |       metadata       |     OUTPUT     |   Tensor   |
|      Mask参数组      |      mask_mode      | ATTR(OPTIONAL) |   int   |
|                      |      win_left       | ATTR(OPTIONAL) |   int   |
|                      |     win_right      | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | INPUT(OPTIONAL) |   Tensor   |
| SeqLengths参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seqlens_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |    seqused_q    | INPUT(OPTIONAL) |  Tensor  |
|                      |   seqused_kv   | INPUT(OPTIONAL) |  Tensor  |
| Paged Attention参数组 |     block_table     | INPUT(OPTIONAL) |   Tensor   |
|    Quant参数组    |    quant_mode    | ATTR(OPTIONAL) |   int   |
|                      | q_dequant_scale | INPUT(OPTIONAL) |  Tensor  |
|                      | k_dequant_scale | INPUT(OPTIONAL) |  Tensor  |
|                      | v_dequant_scale | INPUT(OPTIONAL) |  Tensor  |
|                      |  p_quant_scale  | INPUT(OPTIONAL) |  Tensor  |
|                      |  dst_type_max  | ATTR(OPTIONAL) |  float  |
|                      | attention_out_dtype | ATTR(OPTIONAL) | ScalarType |
|   SoftmaxLSE参数组   | return_softmax_lse | ATTR(OPTIONAL) |    bool    |
|                      |    softmax_lse    |     OUTPUT     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch，表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|    KV_N    |    输入k/v tensor的头数，对应k/v shape中的N    |
|     Q_T     |          输入q tensor所有batch序列长度的累加和          |
|    KV_T    |          输入k/v所有batch序列长度的累加和          |
|     Q_S     |      各batch的query逻辑序列长度，TND下一般由cu_seqlens_q差分得到      |
|    KV_S    |  各batch的key/value逻辑序列长度；TND下一般由cu_seqlens_kv差分得到，Paged Attention下为block_table映射对应的逻辑KV长度  |
|     D     |          输入q/k/v tensor隐藏层最小的单元尺寸headdim         |
|  maxKvBlockCount  | sparse_block_idx最后一维，表示每个Q块最多选择的KV块数，须不小于sparseBlockCount中所有元素的最大值，容量上限见<a href="#产品差异约束">产品差异约束</a> |
| totalQBlocks | 按blockShapeX分块后的Q块总数；$\sum_i\mathrm{ceil}(Q\_S_i / block\_x)$，为`sparse_block_idx`/`sparse_block_count`第二维 |
| totalKBlocks | 按blockShapeY分块后的KV块总数，即$\sum_i\mathrm{ceilDiv}(kvStorageLen_i, blockShapeY)$ |
| num_blocks | Paged KV Cache物理页总数 |
| block_size | Paged KV Cache单页token数 |
| blockShapeX / blockShapeY | 稀疏块在Q方向、KV方向的块大小 |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
  - 空Tensor指必选输入和输出的shape size为0，即有任意轴为0。
  - 触发空Tensor的用例将全部拦截报错。

- q、k、v、attention_out校验：

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 147px">
    <col style="width: 232px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    </colgroup>
    <thead>
    <tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>q</td>
        <td>
            <ul>
                <li>tensor_type支持bfloat16、float16、float8_e4m3fn、int8；MXFP4 为 packed uint8，物理末维 D/2</li>
                <li>TND -> (Q_T, Q_N, D)</li>
            </ul>
        </td>
        <td rowspan="4">
            必须存在
        </td>
        <td rowspan="4">
            <ul>
                <li>quant_mode=8 时使用 FP8 E4M3FN Q 和打包的 FP4 E2M1 K/V。其他模式要求 q、k、v 类型相同。</li>
                <li>k、v的shape需相同</li>
                <li>Layout校验规则见<a href="#layout匹配关系表">layout匹配关系表</a></li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>Q_T > 0</li>
                <li>Q_N > 0</li>
                <li>KV_N > 0</li>
                <li>headDim取值范围见<a href="#产品差异约束">产品差异约束</a></li>
                <li>Q_N % KV_N == 0且Q_N / KV_N > 0</li>
                <li>groupSize = Q_N / KV_N 当前须不超过128</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="2">
            <ul>
                <li>tensor_type支持bfloat16、float16、float8_e4m3fn、int8；MXFP4 为 packed uint8，物理末维 D/2</li>
                <li>PA_BBND -> (num_blocks, block_size, KV_N, D)</li>
                <li>PA_BNBD -> (num_blocks, KV_N, block_size, D)</li>
                <li>PA的block_size固定为128，与稀疏block_shape[1]独立</li>
                <li>TND -> (KV_T, KV_N, D)，适用范围见<a href="#产品差异约束">产品差异约束</a></li>
                <li>非连续Tensor约束：模式0/5的PA_BBND/PA_BNBD仅dim0支持非连续；模式4/6会整理为连续输入；模式4的布局组合见A4C4契约</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>v</td>
    </tr>
    <tr>
        <td>attention_out</td>
        <td>
            <ul>
                <li>quant_mode=0时data_type与q一致；quant_mode!=0时由attention_out_dtype指定</li>
                <li>模式0/5的shape与q一致；模式4输出逻辑D，末维为packed q末维的2倍</li>
            </ul>
        </td>
    </tr>
    </tbody>
    </table>

- layout匹配关系表：<a name="layout匹配关系表"></a>

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 293px">
    </colgroup>
    <thead>
    <tr>
        <th>layout_q</th>
        <th>layout_kv</th>
        <th>attention_out</th>
        <th>softmax_lse</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>TND</td>
            <td>PA_BBND、PA_BNBD或TND，适用范围见<a href="#产品差异约束">产品差异约束</a></td>
            <td>(Q_T, Q_N, D)</td>
            <td>(Q_T, Q_N, 1)</td>
        </tr>
        <tr>
            <td>BSND</td><td>BSND或PA_BBND或PA_BNBD</td><td>(B,Sq,Q_N,D)</td><td>(B,Q_N,Sq,1)</td>
        </tr>
        <tr>
            <td>BNSD</td><td>BNSD或PA_BBND或PA_BNBD</td><td>(B,Q_N,Sq,D)</td><td>(B,Q_N,Sq,1)</td>
        </tr>
    </tbody>
    </table>

- sparse_block_idx、sparse_block_count、block_shape校验：

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 147px">
    <col style="width: 232px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    </colgroup>
    <thead>
    <tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>sparse_block_idx</td>
        <td>
            <ul>
                <li>tensor_type仅支持int32</li>
                <li>shape为(KV_N, totalQBlocks, maxKvBlockCount)</li>
            </ul>
        </td>
        <td rowspan="3">必须存在</td>
        <td rowspan="3">
            <ul>
                <li>metadata接口与主算子须传入相同的sparse_block_idx、sparse_block_count、block_shape</li>
                <li>maxKvBlockCount须不小于sparse_block_count中所有元素的最大值</li>
            </ul>
        </td>
        <td rowspan="3">
            <ul>
                <li>maxKvBlockCount：容量上限见<a href="#产品差异约束">产品差异约束</a></li>
                <li>非union：block_shape=[1,Y]；union（mask_mode=3）：block_shape=[unionSize,Y]，unionSize当前仅支持8。</li>
                <li>Y的取值范围见表后说明。</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_block_count</td>
        <td>
            <ul>
                <li>tensor_type仅支持int32</li>
                <li>shape为(KV_N, totalQBlocks)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>block_shape</td>
        <td>
            <ul>
                <li>长度为2的int列表</li>
            </ul>
        </td>
    </tr>
    </tbody>
    </table>

<!-- npu="A3,910b" id15 -->
- block_shape=[1,Y]：<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term>的Y支持1、2、4、8、16、32、64、128。
<!-- end id15 -->
<!-- npu="950" id16 -->
- <term>Ascend 950PR&950DT系列产品</term>非量化与常规 FP8：`X=1`，`1≤Y≤128`；Union：`[8,Y]`（`1≤Y≤128`）；union（mask_mode=3）时X为unionSize，当前仅支持8。
<!-- end id16 -->

- metadata校验：

    <table style="undefined;table-layout: fixed; width:1625px">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 232px">
            <col style="width: 293px">
            <col style="width: 185px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>单参数校验</th>
                <th>存在性校验</th>
                <th>一致性校验</th>
                <th>特性交叉校验</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>metadata</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持int32</li>
                        <li>shape固定为(1024,)</li>
                        <li>当前不支持不传入，未传入将发出拦截报警</li>
                    </ul>
                </td>
                <td>当前必传</td>
                <td>无</td>
                <td>必须使用与当前输入和属性配套的generic_block_sparse_attention_metadata输出</td>
            </tr>
        </tbody>
    </table>

- softmax_precision校验：

    <table style="undefined;table-layout: fixed; width:1625px">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 232px">
            <col style="width: 293px">
            <col style="width: 185px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>单参数校验</th>
                <th>存在性校验</th>
                <th>一致性校验</th>
                <th>特性交叉校验</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>softmax_precision</td>
                <td>
                    <ul>
                        <li>data_type支持INT</li>
                        <li>取值为0或1，适用范围见<a href="#产品差异约束">产品差异约束</a></li>
                    </ul>
                </td>
                <td>可选属性，默认值为1</td>
                <td>metadata接口与主算子须一致</td>
                <td>无</td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_mode参数解释见「mask_mode 枚举」。

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mask_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>支持0（非因果）、1（causal）。模式3（union）的适用范围见<a href="#产品差异约束">产品差异约束</a></li>
                </ul>
            </td>
            <td>可选输入，默认值为1</td>
            <td rowspan="4">
                <ul>
                    <li>metadata接口与主算子的mask_mode须一致</li>
                    <li>当前win_left和win_right须为-1</li>
                </ul>
            </td>
            <td rowspan="4">
                <ul>
                    <li>mask_mode=1时使用算子内置causal mask</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>
                <ul>
                    <li>mask_mode=0/1：须为None</li>
                    <li>mask_mode=3：须传INT32 exactMask，shape [KV_N, G, U×block_y]；bit i 表示组内第 i 个 Q 对该列可见</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>win_left</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>当前仅支持-1</li>
                </ul>
            </td>
            <td rowspan="2">可选输入，默认值为-1</td>
        </tr>
        <tr>
            <td>win_right</td>
        </tr>
    </tbody>
</table>

#### SeqLengths参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>seqused_q</td>
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持int32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>仅支持非负整数</li>
                    <li>seqused_q中的值需小于等于Q_S</li>
                    <li>seqused_kv中的值需小于等于KV_S</li>
                </ul>
            </td>
            <td>BSND/BNSD必须传入；TND非量化时可选，INT8时必须为None</td>
            <td rowspan="4">无</td>
            <td rowspan="4">无</td>
        </tr>
        <tr>
            <td>seqused_kv</td>
            <td>PA_BBND/PA_BNBD及BSND/BNSD必须传入；TND可选</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持int64</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列</li>
                    <li>TND Q：cu_seqlens_q第一个元素为0且最后一个元素等于Q_T</li>
                    <li>TND KV：cu_seqlens_kv第一个元素为0且最后一个元素等于KV_T</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>当layout_q为TND时，cu_seqlens_q必须传入</li>
                    <li>当layout_q为BSND/BNSD时，cu_seqlens_q不参与计算，建议传入None</li>
                    <li>当layout_kv为TND时，cu_seqlens_kv必须传入</li>
                    <li>当layout_kv为BSND/BNSD时，cu_seqlens_kv不参与计算，建议传入None；PA_BBND/PA_BNBD时不传</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_kv</td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td rowspan="2">max_seqlen_q条件必选；max_seqlen_kv可选</td>
            <td rowspan="2">
                <ul>
                    <li>max_seqlen_q在BSND/BNSD下必须为Q的shape容量Sq；max_seqlen_kv允许-1或非负值，不参与实际KV长度计算</li>
                </ul>
            </td>
            <td rowspan="2">无</td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
        </tr>
        </tbody>
</table>

#### Paged Attention参数组

当block_table不为空时，开启Paged Attention：
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>block_table</td>
            <td>
                <ul>
                    <li>tensor_type仅支持int32</li>
                    <li>tensor_shape为(B, max_num_blocks_per_seq)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>PA_BBND/PA_BNBD必须传入；TND/BSND/BNSD必须为None</td>
            <td>无</td>
            <td>
                <ul>
                    <li>PagedAttention开启情况下，必须传入seqused_kv</li>
                    <li>PagedAttention开启情况下，block_table必须不为空</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Quant参数组

<term>Atlas A2系列产品</term>和<term>Atlas A3系列产品</term> INT8全量化设置`quant_mode=7`、`softmax_precision=0`，Q/K/V均为int8，GQA分组为1～128，D为1～256，Y为1/2/4/8/16/32/64/128。四个scale均必传，形状见上方参数表，要求float32、连续、有限且大于0；`attention_out_dtype`必须指定为torch.float16或torch.bfloat16。TND Q的`seqused_q`必须为None。量化公式和PA scale示例见[aclnn接口的量化相关说明](aclnnGenericBlockSparseAttention.md#量化相关说明)。

quant_mode参数解释见「quant_mode 枚举」。

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>quant_mode</td>
            <td>data_type支持INT；当前支持0（非量化）、4（A4C4）、5（FP8）、6（MXFP8）、7（INT8全量化）和8（A8C4）。</td>
            <td>可选输入，默认值为0</td>
            <td rowspan="5">
                <ul>
                    <li>metadata接口与主算子的quant_mode须一致</li>
                </ul>
            </td>
            <td rowspan="5">
                <ul>
                    <li>quant_mode=0/5 时，Q/K/V scale 须为 None。quant_mode=8 时，K/V scale 为独立 BF16 张量，Q/P scale 为 None。</li>
                    <li>quant_mode!=0时attention_out_dtype必须传入，当前量化场景输出dtype须为float16或bfloat16</li>
                </ul>
             A4C4（模式4）必传三项E8M0 dequant scale，p_quant_scale=None；见<a href="#a4c4quantmode4">A4C4说明</a>。</td>
        </tr>
        <tr>
            <td>q_dequant_scale</td>
            <td rowspan="3">
                <ul>
                    <li>quant_mode=0/5 须为 None。quant_mode=8 时，Q scale 固定为 1.0，q_dequant_scale 传 None，K/V scale 为 BF16，形状见入参表。</li>
                    <li>非量化场景须为None；INT8场景必须提供float32 Tensor</li>
                </ul>
            </td>
            <td rowspan="3">可选参数</td>
        </tr>
        <tr>
            <td>k_dequant_scale</td>
        </tr>
        <tr>
            <td>v_dequant_scale</td>
        </tr>
        <tr>
            <td>p_quant_scale</td>
            <td>
                <ul>
                    <li>当前须为None</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
    </tbody>
</table>

#### SoftmaxLSE参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>return_softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持BOOL</li>
                    <li>非量化（quant_mode=0）、A4C4（quant_mode=4）、INT8（quant_mode=7）和 A8C4（quant_mode=8）均支持 False 和 True；高精 FP8（quant_mode=5、softmax_precision=0、D=128）支持False/True；低精 FP8（quant_mode=5、softmax_precision=1）仅支持False</li>
                </ul>
            </td>
            <td>可选属性，默认值为False</td>
            <td rowspan="2">
                <ul>
                    <li>return_softmax_lse为False时，softmax_lse输出空Tensor</li>
                    <li>return_softmax_lse为True时，输出shape见<a href="#layout匹配关系表">layout匹配关系表</a></li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持float32</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

## 调用示例

- generic_block_sparse_attention_metadata + generic_block_sparse_attention联合调用示例（TND + PA_BBND）

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    B, Q_N, KV_N, Q_S, KV_S, D = 1, 32, 8, 128, 256, 128
    block_x, block_y = 1, 128
    block_size = 128
    top_k = 16

    Q_T = B * Q_S
    num_blocks = math.ceil(KV_S / block_size)
    total_q_blocks = math.ceil(Q_S / block_x)

    q = torch.randn(Q_T, Q_N, D, dtype=torch.float16, device="npu")
    k = torch.randn(num_blocks, block_size, KV_N, D, dtype=torch.float16, device="npu")
    v = torch.randn(num_blocks, block_size, KV_N, D, dtype=torch.float16, device="npu")

    # top_k 是索引张量的容量，有效数量不超过逻辑 KV 块数。
    num_sparse_kv_blocks = math.ceil(KV_S / block_y)
    valid_count = min(top_k, num_sparse_kv_blocks)
    sparse_block_idx = torch.full(
        (KV_N, total_q_blocks, top_k), -1, dtype=torch.int32, device="npu"
    )
    sparse_block_idx[..., :valid_count] = torch.arange(valid_count, dtype=torch.int32, device="npu")
    sparse_block_count = torch.full((KV_N, total_q_blocks), valid_count, dtype=torch.int32, device="npu")

    # 有效索引必须非负、严格升序且不重复，容量中的其余位置不参与计算。
    valid_idx = sparse_block_idx[..., :valid_count].cpu()
    assert torch.all(sparse_block_count.cpu() == valid_count).item()
    assert torch.all((valid_idx >= 0) & (valid_idx < num_sparse_kv_blocks)).item()
    assert torch.all(valid_idx[..., 1:] > valid_idx[..., :-1]).item()
    cu_seqlens_q = torch.tensor([0, Q_S], dtype=torch.int64, device="npu")
    seqused_kv = torch.tensor([KV_S], dtype=torch.int32, device="npu")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="npu").view(1, -1)
    block_shape = [block_x, block_y]

    metadata = cann_ops_transformer.ops.generic_block_sparse_attention_metadata(
        sparse_block_idx,
        sparse_block_count,
        Q_N,
        KV_N,
        D,
        block_shape,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        max_seqlen_q=Q_S,
        max_seqlen_kv=KV_S,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        mask_mode=1,
        quant_mode=0,
        softmax_precision=1,
        residual_block_mode=0,
        is_consistent_topk=False,
    )

    attention_out, softmax_lse = cann_ops_transformer.ops.generic_block_sparse_attention(
        q,
        k,
        v,
        sparse_block_idx,
        sparse_block_count,
        block_shape,
        metadata=metadata,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        block_table=block_table,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        softmax_scale=1.0 / (D ** 0.5),
        mask_mode=1,
        quant_mode=0,
        softmax_precision=1,
        return_softmax_lse=False,
        residual_block_mode=0,
        is_consistent_topk=False,
    )
    torch_npu.npu.synchronize()
    assert attention_out.shape == q.shape
    assert attention_out.dtype == q.dtype

    # 本示例页表为顺序映射，选择前 valid_count 个逻辑 KV 块，并使用右对齐 causal mask。
    selected_kv = min(valid_count * block_y, KV_S)
    q_ref = q.cpu().float()
    k_ref = k.cpu().float().reshape(-1, KV_N, D)[:selected_kv]
    v_ref = v.cpu().float().reshape(-1, KV_N, D)[:selected_kv]
    k_ref = k_ref.repeat_interleave(Q_N // KV_N, dim=1)
    v_ref = v_ref.repeat_interleave(Q_N // KV_N, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q_ref, k_ref) / math.sqrt(D)
    visible_kv = KV_S - Q_S + torch.arange(Q_S) + 1
    visible_mask = torch.arange(selected_kv)[None, :] < visible_kv[:, None]
    scores.masked_fill_(~visible_mask[None, :, :], float("-inf"))
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v_ref)
    torch.testing.assert_close(attention_out.cpu().float(), reference, atol=1e-2, rtol=1e-2)
    ```

### A8C8 的精度及 LSE

`quant_mode=6` 仅接受 `softmax_precision=0`；Metadata 与 Attention 均须显式指定 0。Python 通用接口的默认值仍为 1，以保持其他模式的接口行为；不会为 A8C8 静默转换。`return_softmax_lse=True` 可返回 FP32 LSE，TND 为 `[Tq,Nq,1]`，BSND/BNSD 为 `[B,Nq,Sq,1]`。LSE 基于反量化 Q/K、在 P 量化之前计算，空行及 padding 输出 `-FLT_MAX`；关闭时仍返回空 Tensor。输入约束见 [A8C8 量化契约](./aclnnGenericBlockSparseAttention.md#a8c8quanttype6)。

<a name="a4c4quantmode4"></a>

## A4C4（MXFP4 CX，quant_mode=4）

`QuantMode.FP4_E2M1_DYNAMIC_CX`（4）选择A4C4。Metadata和主算子均使用模式4及 `softmax_precision=0`；
模式6仅用于MXFP8，模式3 OCP尚不支持。逻辑D为192或256，`block_shape=[1,Y]`，Y∈{4,8,16,32,64,128}，Q/KV head数之比在[1,16]。
支持 `mask_mode=0/1`、`residual_block_mode=0/1`，以及可选FP32 LSE；九种布局组合见[ACLNN A4C4说明](./aclnnGenericBlockSparseAttention.md#a4c4quanttype4)。

### 输入与输出

| 参数 | Torch类型与约定 |
| --- | --- |
| q、k、v | packed uint8，每字节两个E2M1值，物理末维D/2；偶数D元素占低4 bit |
| q_dequant_scale、k_dequant_scale | E8M0或uint8编码，保持对应数据前缀轴，末维D/2替换为[D/64,2] |
| v_dequant_scale | E8M0或uint8编码；布局及各batch独立对齐方式见ACLNN A4C4 scale表 |
| p_quant_scale | None，P在算子内部动态量化 |
| dst_type_max | 0表示默认CX=6；显式值须有限且在[6,12]内 |
| attention_out_dtype | 必须显式指定torch.float16或torch.bfloat16 |
| attention_out | 与q布局一致，末维恢复逻辑D |
| softmax_lse | FP32；TND为[Tq,Nq,1]，BNSD/BSND为[B,Nq,Sq,1]；未启用时为空Tensor |

三项dequant scale均必传。同一有效序列、KV head和channel的V scale保持一致，并按接口shape重复存储。
TND V scale须按 `cu_seqlens_kv` 的每批存储长度分别向上对齐后拼接；不能按总长度一次对齐，也不能按较短的 `seqused_kv` 重新确定batch起点。
PA的K/V及scale共用 `block_table` 的物理页编号，允许页编号不连续或乱序，接口整理为连续存储。

### 长度与数值说明

TND Q须传 `cu_seqlens_q`，TND KV须传 `cu_seqlens_kv`；PA KV须传 `block_table` 和 `seqused_kv`，且 `cu_seqlens_kv=None`。
BNSD/BSND Q须传 `seqused_q`；BNSD/BSND KV须传 `seqused_kv`。Metadata的 `head_dim` 使用逻辑D，相关输入及属性必须与主算子一致。
因果模式下，各batch的实际KV长度须不小于实际Q长度。

P使用CX动态量化，Softmax分母及LSE使用量化前FP32指数；调用者无需额外反量化。
量化误差、块配对和FD累加顺序可能导致不同布局或调度下的输出不逐位一致，具体数值与scale约定见[ACLNN A4C4说明](./aclnnGenericBlockSparseAttention.md#a4c4quanttype4)。
