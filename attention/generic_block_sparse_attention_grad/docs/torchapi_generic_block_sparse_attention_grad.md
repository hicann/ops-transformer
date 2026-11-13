# generic_block_sparse_attention_grad

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

  `generic_block_sparse_attention_grad`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`GenericBlockSparseAttentionGrad`算子完成块稀疏注意力的反向梯度计算。`layout_sparse_pattern=1`时通过`sparse_block_idx`指定每个KV块选择的Q块/token；`layout_sparse_pattern=0`时通过该参数指定每个Q token选择的完整KV块。算子返回`dq`/`dk`/`dv`。

  `layout_sparse_pattern=1`时使用`generic_block_sparse_attention_grad_metadata`生成AI Core任务切分与负载均衡信息，再传入主算子。`layout_sparse_pattern=0`时不调用该接口，主算子的`metadata`须传`None`。`layout_sparse_pattern=1`的典型调用流程如下：

  1. 准备`q`、`k`、`v`、`dout`、`attn_out`、`softmax_lse`、`sparse_block_idx`、`sparse_block_count`、`cu_seqlens_q`、`cu_seqlens_kv`等输入。
  2. 调用`generic_block_sparse_attention_grad_metadata`生成`metadata`。
  3. 调用`generic_block_sparse_attention_grad`，将上一步得到的`metadata`传入主算子。
- **计算公式**：

  稀疏块大小为$blockShapeX \times blockShapeY$。依据`sparse_block_idx`/`sparse_block_count`选取参与计算的Q–KV块对后：

  $$
  P = SimpleSoftmax(Mask(Q @ selectedK^{T} \cdot scale), lse)
  $$

  $$
  dP = dO @ selectedV^{T}
  $$

  $$
  dS = P \odot (dP - SoftmaxGrad(dO, O))
  $$

  $$
  dQ = dS @ selectedK \cdot scale
  $$

  $$
  dK = dS^{T} @ Q \cdot scale
  $$

  $$
  dV = P^{T} @ dO
  $$

  其中$scale$由`softmax_scale`指定，建议值为$D^{-0.5}$。

> [!NOTE]
> q、k、v数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim。
> TND中的N为q的headNum时记为N1（Q_N），k/v的headNum记为N2（KV_N）。T1表示所有q Batch输入样本序列长度的累加和；T2表示所有kv Batch输入样本序列长度的累加和。
> $J = \lceil max\_seqlen\_kv / blockShapeY \rceil$。`layout_sparse_pattern=1`时的`sparse_block_idx`/`sparse_block_count`为**KV→Q**方向；`layout_sparse_pattern=0`时为**Q→KV**方向。

## 函数原型

`layout_sparse_pattern=1`时调用`generic_block_sparse_attention_grad`前须先调用`generic_block_sparse_attention_grad_metadata`；`layout_sparse_pattern=0`时不调用`generic_block_sparse_attention_grad_metadata`接口。

```python
cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
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
    max_seqlen_q=None,
    max_seqlen_kv=None,
    layout_q="TND",
    layout_kv="TND",
    layout_sparse_pattern=1,
    mask_mode=MaskMode.CAUSAL,
    softmax_precision=0,
    win_left=-1,
    win_right=-1,
    residual_block_mode=ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX,
    is_consistent_topk=False,
) -> Tensor
```

```python
cann_ops_transformer.generic_block_sparse_attention_grad(
    q,
    k,
    v,
    dout,
    attn_out,
    softmax_lse,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    *,
    metadata=None,
    attn_mask=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    layout_q="TND",
    layout_kv="TND",
    layout_sparse_pattern=1,
    softmax_scale=1.0,
    mask_mode=MaskMode.CAUSAL,
    softmax_precision=0,
    win_left=-1,
    win_right=-1,
    residual_block_mode=ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX,
    is_consistent_topk=False,
) -> (Tensor, Tensor, Tensor)
```

## 枚举说明

`MaskMode`和`ResidualBlockMode`均为`IntEnum`，定义于`cann_ops_transformer.ops.generic_block_sparse_attention_grad`，Python封装接口支持传入枚举、对应int值或枚举名字符串。

### mask_mode枚举

| 枚举名     | 值 | 含义                              |
| :--------- | :-: | :-------------------------------- |
| `CAUSAL` | 1 | Causal模式，默认值；两种模式均支持 |
| `NO_MASK` | 0 | 无掩码；仅`layout_sparse_pattern=0`时支持 |

> [!NOTE]
> 枚举为`IntEnum`，可直接作为int传入底层算子；接口支持传入枚举、对应int值或枚举名字符串。具体取值限制见“约束说明”。

### residual_block_mode枚举

| 枚举名 | 值 | 含义 |
| :--- | :-: | :--- |
| `ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX` | 0 | 尾部不完整KV块是否参与计算由`sparse_block_idx`决定；`layout_sparse_pattern=1`时使用，默认值 |
| `ResidualBlockMode.INCOMPLETE_BLK_KEPT_BUT_NOT_IN_SPARSE_BLK_IDX` | 1 | 尾部不完整KV块必定参与计算，但不包含在`sparse_block_idx`中；`layout_sparse_pattern=0`时使用 |

`generic_block_sparse_attention_grad`与`generic_block_sparse_attention_grad_metadata`接口均默认使用`ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX`。主算子按模式分别使用0或1；可传对应整数或枚举名字符串。字符串解析与`mask_mode`一致：忽略大小写及首尾空格。

## 参数说明

### generic_block_sparse_attention_grad_metadata

| 参数名             | 参数类型     | 可选/必选 | 描述                                                                                                | 数据类型 | 数据格式 | 维度                                                      |
| :----------------- | :----------- | :-------- | :-------------------------------------------------------------------------------------------------- | :------- | :------- | :-------------------------------------------------------- |
| sparse_block_idx   | Tensor       | 必选      | 稀疏块索引数组，指定每个KV块选择的Q块/token索引 | int32    | ND       | TND：`(B,N2,J,maxS1)`；BNSD/BSND：`(B,N2,J,S1)` |
| sparse_block_count | Tensor       | 必选      | 指定每个KV块实际选择的Q数量 | int32    | ND       | `(B,N2,J)` |
| num_heads_q        | int          | 必选      | Query head数（N1）                                                                                  | int32    | -        | -                                                         |
| num_heads_kv       | int          | 必选      | Key/Value head数（N2）                                                                              | int32    | -        | -                                                         |
| head_dim           | int          | 必选      | 每个注意力头的维度，当前固定128 | int32    | -        | -                                                         |
| block_shape        | list[int]    | 必选      | 稀疏块形状[block_x,block_y]，分别为Q、KV方向的token数；block_x仅支持1；block_y须≥128且为64的倍数 | int64    | -        | 长度为2                                                   |
| cu_seqlens_q       | Tensor       | 可选      | q累积序列长度，第一个元素必须为0                                                                    | int64    | ND       | `(B+1,)`                                                |
| cu_seqlens_kv      | Tensor       | 可选      | k/value累积序列长度，第一个元素必须为0                                                              | int64    | ND       | `(B+1,)`                                                |
| seqused_q          | Tensor       | 可选      | 各batch中q实际使用的序列长度                                                                        | int32    | ND       | `(B,)`                                                  |
| seqused_kv         | Tensor       | 可选      | 各batch中kv实际使用的序列长度                                                                       | int32    | ND       | `(B,)`                                                  |
| max_seqlen_q       | int          | 可选      | 所有batch中q序列长度的最大值；省略时由扩展推断（优先`seqused_q`/`cu_seqlens_q`，否则取`sparse_block_idx`最后一维）；显式传入须≥0 | int32    | -        | -                                                         |
| max_seqlen_kv      | int          | 可选      | 所有batch中kv序列长度的最大值，用于计算$J$；省略时由扩展推断（优先`seqused_kv`/`cu_seqlens_kv`，否则取`J * block_y`）；显式传入须≥0 | int32    | -        | -                                                         |
| layout_q           | string       | 可选      | q布局，支持`"TND"`/`"BNSD"`/`"BSND"`，默认`"TND"`                                           | string   | -        | -                                                         |
| layout_kv          | string       | 可选      | k/value布局，须与`layout_q`一致，默认`"TND"`                                                    | string   | -        | -                                                         |
| layout_sparse_pattern | int | 可选 | sparse_block_idx、sparse_block_count的数据排布；仅支持1（KV→Q），默认1 | int64 | - | - |
| mask_mode          | int/MaskMode | 可选      | 掩码模式，1（CAUSAL）表示因果掩码，默认1；当前仅支持1 | int32    | -        | -                                                         |
| softmax_precision  | int          | 可选      | Softmax精度级别，当前仅支持0，默认0                                                                   | int32    | -        | -                                                         |
| win_left           | int          | 可选      | 滑窗向前包含token数，不使能时必须为-1，默认-1                                                       | int32    | -        | -                                                         |
| win_right          | int          | 可选      | 滑窗向后包含token数，不使能时必须为-1，默认-1                                                       | int32    | -        | -                                                         |
| residual_block_mode | int/ResidualBlockMode/str | 可选 | 尾部不完整KV块的处理模式；默认0，由稀疏索引决定是否参与；当前仅支持0；可传枚举、对应整数或枚举名字符串 | int64 | - | - |
| is_consistent_topk | bool | 可选 | 前置稀疏选择中，同一batch同一head内每个Q块选择的KV块最大数量是否一致，默认False；False代表不一致，True代表一致；两种值均走通用计算路径 | bool | - | - |

### generic_block_sparse_attention_grad

| 参数名             | 参数类型     | 可选/必选 | 描述                                                                                  | 数据类型         | 数据格式 | 维度                                                                     |
| :----------------- | :----------- | :-------- | :------------------------------------------------------------------------------------ | :--------------- | :------- | :----------------------------------------------------------------------- |
| q                  | Tensor       | 必选      | 公式中的$Q$                                                                         | bfloat16/float16 | ND       | TND：`(T1, N1, D)`；BNSD：`(B, N1, S1, D)`；BSND：`(B, S1, N1, D)` |
| k                  | Tensor       | 必选      | 公式中的$K$                                                                         | bfloat16/float16 | ND       | TND：`(T2, N2, D)`；BNSD：`(B, N2, S2, D)`；BSND：`(B, S2, N2, D)` |
| v                  | Tensor       | 必选      | 公式中的$V$，shape与k相同                                                           | bfloat16/float16 | ND       | 与k一致                                                                  |
| dout               | Tensor       | 必选      | 注意力输出梯度$dO$                                                                  | bfloat16/float16 | ND       | 与q一致                                                                  |
| attn_out           | Tensor       | 必选      | 注意力正向输出$O$                                                                   | bfloat16/float16 | ND       | 与q一致                                                                  |
| softmax_lse        | Tensor       | 必选      | 正向保存的softmax LSE                                                                | float32          | ND       | TND：`(T1,N1,1)`；BNSD：`(B,N1,S1,1)`；BSND：`(B,S1,N1,1)`或`(B,N1,S1,1)`，详见约束说明 |
| sparse_block_idx   | Tensor       | 必选      | 稀疏块索引数组：`layout_sparse_pattern=1`时指定每个KV块选择的Q块/token索引；=0时指定每个Q token选择的完整KV块索引 | int32 | ND | =1：TND为`(B,N2,J,maxS1)`，BNSD/BSND为`(B,N2,J,S1)`；=0：`(B,N2,maxQBlockCount,maxKvBlockCount)` |
| sparse_block_count | Tensor       | 必选      | `layout_sparse_pattern=1`时指定每个KV块实际选择的Q数量；=0时指定每个Q token选中的完整可见KV块数，不包含自动计算的尾部不完整块 | int32 | ND | =1：`(B,N2,J)`；=0：`(B,N2,maxQBlockCount)` |
| metadata           | Tensor       | 可选      | `generic_block_sparse_attention_grad_metadata`生成的任务切分结果，须通过关键字参数传入；layout_sparse_pattern=1时必传，=0时须为None | int32            | ND       | `(metaSize,)`，见返回值说明 |
| block_shape        | list[int]    | 必选      | 稀疏块形状[block_x,block_y]，分别为Q、KV方向的token数；block_x仅支持1；=1时block_y须≥128且为64的倍数；=0时1≤block_y≤baseKtile，G=N1/N2≤128时baseKtile=128，否则为64 | int64            | -        | 长度为2 |
| attn_mask          | Tensor       | 可选      | 掩码矩阵，当前暂不支持，应传`None`                                                  | bool             | ND       | -                                                                        |
| cu_seqlens_q       | Tensor       | 可选      | q累积序列长度；`layout_q="TND"`时必传                                               | int64            | ND       | `(B+1,)`                                                               |
| cu_seqlens_kv      | Tensor       | 可选      | kv累积序列长度；`layout_kv="TND"`时必传                                             | int64            | ND       | `(B+1,)`                                                               |
| seqused_q          | Tensor       | 可选      | 各batch中q实际使用的序列长度                                                          | int32            | ND       | `(B,)`                                                                 |
| seqused_kv         | Tensor       | 可选      | 各batch中kv实际使用的序列长度                                                         | int32            | ND       | `(B,)`                                                                 |
| layout_q           | string       | 可选      | q布局，支持`"TND"`/`"BNSD"`/`"BSND"`，默认`"TND"`                             | string           | -        | -                                                                        |
| layout_kv          | string       | 可选      | k/value布局，须与`layout_q`一致，默认`"TND"`                                      | string           | -        | -                                                                        |
| layout_sparse_pattern | int | 可选 | sparse_block_idx、sparse_block_count的数据排布；1表示KV→Q，0表示Q→KV；同一KV head对应的Q head共享稀疏索引，默认1 | int64 | - | - |
| softmax_scale      | float        | 可选      | 缩放因子，建议值$1/\sqrt{D}$，默认1.0 | float32 | - | - |
| mask_mode          | int/MaskMode | 可选      | 掩码模式，1（CAUSAL）表示因果掩码，默认1；0（NO_MASK）表示无掩码，仅layout_sparse_pattern=0支持；支持枚举、对应整数或枚举名字符串 | int32 | - | - |
| softmax_precision  | int          | 可选      | Softmax精度级别，当前仅支持0，默认0                                           | int32            | -        | -                                                                        |
| win_left           | int          | 可选      | 滑窗向前包含token数，不使能时必须为-1，默认-1                                         | int32            | -        | -                                                                        |
| win_right          | int          | 可选      | 滑窗向后包含token数，不使能时必须为-1，默认-1                                         | int32            | -        | -                                                                        |
| residual_block_mode | int/ResidualBlockMode/str | 可选 | 尾部不完整KV块的处理模式；默认0，由稀疏索引决定是否参与；当layout_sparse_pattern=1时使用0；当layout_sparse_pattern=0时必须使用1，尾部不完整块自动参与计算，且不包含在sparse_block_idx中；可传枚举、对应整数或枚举名字符串 | int64 | - | - |
| is_consistent_topk | bool | 可选 | 前置稀疏选择中，同一batch同一head内每个Q块选择的KV块最大数量是否一致，默认False；False代表不一致，True代表一致；=0 时False和True均按每行实际索引和数量计算；=1 时须与generic_block_sparse_attention_grad_metadata接口的配置一致 | bool | - | - |

## 返回值说明

### generic_block_sparse_attention_grad_metadata（仅`layout_sparse_pattern=1`时）

| 参数名   | 参数类型 | 可选/必选 | 描述                     | 数据类型 | 数据格式 | 维度            |
| :------- | :------- | :-------- | :----------------------- | :------- | :------- | :-------------- |
| metadata | Tensor   | 必选      | 主算子分核与负载均衡数据 | int32    | ND       | `(metaSize,)` |

其中：
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>

    $$
    metaSize = 80 + B \times N1 \times J \times 4,\quad J = \lceil max\_seqlen\_kv / blockShapeY \rceil
    $$

<!-- end id7 -->

<!-- npu="A3,910b" id8 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：

    $$
    metaSize = 198
    $$

<!-- end id8 -->

### generic_block_sparse_attention_grad

| 参数名 | 参数类型 | 可选/必选 | 描述    | 数据类型         | 数据格式 | 维度    |
| :----- | :------- | :-------- | :------ | :--------------- | :------- | :------ |
| dq     | Tensor   | 必选      | q的梯度 | bfloat16/float16 | ND       | 与q一致 |
| dk     | Tensor   | 必选      | k的梯度 | bfloat16/float16 | ND       | 与k一致 |
| dv     | Tensor   | 必选      | v的梯度 | bfloat16/float16 | ND       | 与v一致 |

**说明**

- `dq`/`dk`/`dv`的dtype与对应输入`q`/`k`/`v`保持一致。

## 约束说明

- 确定性计算：`generic_block_sparse_attention_grad`默认不支持确定性计算。

**`layout_sparse_pattern=1`：**

- 要求`residual_block_mode=0`、`mask_mode=1`，`is_consistent_topk`支持False和True。须先调用`generic_block_sparse_attention_grad_metadata`生成`metadata`，两次调用的`layout_sparse_pattern`、`residual_block_mode`、`is_consistent_topk`须保持一致。
- Softmax LSE：TND为`(T1,N1,1)`，BNSD为`(B,N1,S1,1)`；BSND在A5上GSAG须为`(B,N1,S1,1)`，在A2/A3上GSAG为`(B,S1,N1,1)`。
- `softmax_scale`须为float32可表示的有限值（非NaN/Inf，且转float后仍有限）。
- `max_seqlen_q`/`max_seqlen_kv`可省略，由Torch扩展推断；显式传入时须≥0，且满足`J=ceilDiv(max_seqlen_kv, block_y)`与`maxS1≥max_seqlen_q`。

**`layout_sparse_pattern=0`：**

- Softmax LSE与前向输出一致：TND为`(T1,N1)`或`(T1,N1,1)`；BNSD和BSND均为`(B,N1,S1)`或`(B,N1,S1,1)`，不随query的head/seq轴顺序改变。
- 要求`residual_block_mode=1`、`mask_mode=0`（NO_MASK）或`1`（CAUSAL）；`is_consistent_topk`支持False和True，两种取值均走通用计算路径。该排布不调用`generic_block_sparse_attention_grad_metadata`接口，`metadata`和`attn_mask`须为`None`；尾部不完整KV块自动参与计算。
- `q`、`k`、`v`、`dout`、`attn_out`及`dq`、`dk`、`dv`均为float16或bfloat16且类型一致；`v`与`k`同shape，`dout`、`attn_out`、`dq`与`q`同shape，`dk`、`dv`与`k`同shape；必选张量不支持零维长度或负维度。
- 要求B、N1、N2为正数，N1%N2=0、G=N1/N2≤256，Q/K的D相同且位于[1,256]；`block_shape=[1,block_y]`，`1≤block_y≤baseKtile`（G≤128时为128，否则为64）。`softmax_scale`须为有限值，`softmax_precision=0`，`win_left=win_right=-1`。
- `sparse_block_idx`在TND布局为`(B,N2,maxQBlockCount,maxKvBlockCount)`，在BNSD/BSND布局为`(B,N2,S1,maxKvBlockCount)`；`sparse_block_count`分别为`(B,N2,maxQBlockCount)`和`(B,N2,S1)`。`maxQBlockCount`是Q方向的索引容量；`maxKvBlockCount`是每个Q token可保存的完整KV块数容量，取值为[1,INT32_MAX]。
- 同一`(B,N2)`下的Q head group共享索引。`sparse_block_count`只统计选中的完整可见KV块，不得超过`maxKvBlockCount`；NONE模式的可见长度`limit=klen`，CAUSAL模式为`max(0,min(klen,klen-qlen+q_index+1))`，`q_index`从0开始。有效块索引须在`[0,floor(limit/block_y))`内；尾部不完整块不写入索引，由算子自动加入计算。
- TND布局要求`maxQBlockCount≥ceil(T1/B)`且不小于每个batch的实际Q长度；须传入INT64 `(B+1,)`的`cu_seqlens_q`和`cu_seqlens_kv`，`seqused_q`/`seqused_kv`可选，传入时须为INT32 `(B,)`。BNSD/BSND布局下这四个参数须为`None`。
- `cu_seqlens_q`、`cu_seqlens_kv`、`seqused_q`、`seqused_kv`、`sparse_block_idx`、`sparse_block_count`的实际数值不在tiling阶段校验，调用方须保证其语义和范围合法。

### 特性参数组

|   特性参数组   |    参数字段名称    |    字段分组    |   字段类型   |
| :------------: | :----------------: | :-------------: | :----------: |
|   公共参数组   |         q         |      INPUT      |    Tensor    |
|                |         k         |      INPUT      |    Tensor    |
|                |         v         |      INPUT      |    Tensor    |
|                |        dout        |      INPUT      |    Tensor    |
|                |      attn_out      |      INPUT      |    Tensor    |
|                |    softmax_lse    |      INPUT      |    Tensor    |
|                |  sparse_block_idx  |      INPUT      |    Tensor    |
|                | sparse_block_count |      INPUT      |    Tensor    |
|                |      metadata      |      INPUT      |    Tensor    |
|                |    block_shape    | ATTR(REQUIRED) |    int[]    |
|                |   softmax_scale   | ATTR(OPTIONAL) |    float    |
|                |      layout_q      | ATTR(OPTIONAL) |    string    |
|                |     layout_kv     | ATTR(OPTIONAL) |    string    |
|                | layout_sparse_pattern | ATTR(OPTIONAL) | int |
|                | residual_block_mode | ATTR(OPTIONAL) | int/ResidualBlockMode/str |
|                | is_consistent_topk | ATTR(OPTIONAL) | bool |
|                |         dq         |     OUTPUT     |    Tensor    |
|                |         dk         |     OUTPUT     |    Tensor    |
|                |         dv         |     OUTPUT     |    Tensor    |
| metadata参数组 |    max_seqlen_q    | ATTR(OPTIONAL) |     int     |
|                |   max_seqlen_kv   | ATTR(OPTIONAL) |     int     |
|                |    num_heads_q    | ATTR(REQUIRED) |     int     |
|                |    num_heads_kv    | ATTR(REQUIRED) |     int     |
|                |      head_dim      | ATTR(REQUIRED) |     int     |
|                |      metadata      |     OUTPUT     |    Tensor    |
|   Mask参数组   |     mask_mode     | ATTR(OPTIONAL) | int/MaskMode |
|                |      win_left      | ATTR(OPTIONAL) |     int     |
|                |     win_right     | ATTR(OPTIONAL) |     int     |
|                |     attn_mask     | INPUT(OPTIONAL) |    Tensor    |
| SeqLens参数组 |    cu_seqlens_q    | INPUT(OPTIONAL) |    Tensor    |
|                |   cu_seqlens_kv   | INPUT(OPTIONAL) |    Tensor    |
|                |     seqused_q     | INPUT(OPTIONAL) |    Tensor    |
|                |     seqused_kv     | INPUT(OPTIONAL) |    Tensor    |
| Softmax参数组 | softmax_precision | ATTR(OPTIONAL) |     int     |

### 基准信息说明

资料约束中，常见字段释义如下：

|   命名   |                          含义                          |
| :-------: | :-----------------------------------------------------: |
|     B     |               Batch,表示输入样本批量大小               |
| N1 / Q_N |                   输入q tensor的头数                   |
| N2 / KV_N |                 输入key/v tensor的头数                 |
| S1 / Q_S |                      输入q序列长度                      |
| S2 / KV_S |                  输入key/value序列长度                  |
|    T1    |             输入q所有batch序列长度的累加和             |
|    T2    |             输入kv所有batch序列长度的累加和             |
|     D     |          `layout_sparse_pattern=1`固定128；`layout_sparse_pattern=0`时`1<=D<=256`          |
|     J     | $\lceil max\_seqlen\_kv / blockShapeY \rceil$，KV块数 |
|   maxS1   | `layout_sparse_pattern=1`时为`sparse_block_idx`最后一维容量，须≥`max_seqlen_q` |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：

  - 空Tensor指必选输入和输出的shape size为0，即有任意轴为0。
  - 触发空Tensor的用例将全部拦截报错。
- q、k、v、dout、attn_out、dq、dk、dv校验:

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
                <li>tensor_type支持bfloat16、float16</li>
                <li>TND → (T1, N1, D)；BNSD → (B, N1, S1, D)；BSND → (B, S1, N1, D)</li>
            </ul>
        </td>
        <td rowspan="5">必须存在</td>
        <td rowspan="5">
            <ul>
                <li>q、k、v、dout、attn_out、dq、dk、dv的数据类型需相同</li>
                <li>dout、attn_out、dq的shape与q一致</li>
                <li>k、v、dk、dv的shape需相同</li>
                <li>layout_q与layout_kv须一致，支持TND/BNSD/BSND</li>
            </ul>
        </td>
        <td rowspan="5">
            轴校验：
            <ul>
                <li>`layout_sparse_pattern=1`：D=128，N1、N2取值范围[1,128]，N1 % N2 == 0</li>
                <li>`layout_sparse_pattern=0`：1≤D≤256，N1、N2为正数，N1 % N2 == 0，N1/N2≤256</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k / v</td>
        <td>
            <ul>
                <li>tensor_type支持bfloat16、float16</li>
                <li>TND → (T2, N2, D)；BNSD → (B, N2, S2, D)；BSND → (B, S2, N2, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>dout / attn_out</td>
        <td>
            <ul>
                <li>dtype与q一致</li>
                <li>shape与q一致</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>dq</td>
        <td>shape/dtype与q一致</td>
    </tr>
    <tr>
        <td>dk / dv</td>
        <td>shape/dtype与k一致</td>
    </tr>
    </tbody>
    </table>
- lse校验:

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
        <td>softmax_lse</td>
        <td>
            <ul>
                <li>tensor_type仅支持float32</li>
                <li>TND：(T1,N1,1)；BNSD：(B,N1,S1,1)；BSND：(B,S1,N1,1)或(B,N1,S1,1)，详见约束说明</li>
            </ul>
        </td>
        <td>必须存在</td>
        <td>BSND轴序见约束说明</td>
        <td>无</td>
    </tr>
    </tbody>
    </table>
- sparse_block_idx、sparse_block_count、block_shape校验:

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
                <li>`layout_sparse_pattern=1`：TND为(B, N2, J, maxS1)，BNSD/BSND为(B, N2, J, S1)</li>
                <li>`layout_sparse_pattern=0`：TND为(B, N2, maxQBlockCount, maxKvBlockCount)，BNSD/BSND为(B, N2, S1, maxKvBlockCount)</li>
            </ul>
        </td>
        <td rowspan="3">必须存在</td>
        <td rowspan="3">
            <ul>
                <li>`layout_sparse_pattern=1`：`generic_block_sparse_attention_grad_metadata`接口与主算子须传入相同的sparse_block_idx、sparse_block_count、block_shape</li>
                <li>`layout_sparse_pattern=1`：J=ceilDiv(max_seqlen_kv, block_y)，maxS1 / S1须≥max_seqlen_q及sparse_block_count中的最大值</li>
                <li>`layout_sparse_pattern=0`：sparse_block_count不得超过maxKvBlockCount</li>
            </ul>
        </td>
        <td rowspan="3">
            <ul>
                <li>`layout_sparse_pattern=1`：block_x=1，block_y>=128且为64的倍数</li>
                <li>`layout_sparse_pattern=1`：residual_block_mode=0</li>
                <li>`layout_sparse_pattern=0`：block_x=1，1≤block_y≤baseKtile；layout_sparse_pattern=0，residual_block_mode=1</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_block_count</td>
        <td>
            <ul>
                <li>tensor_type仅支持int32</li>
                <li>`layout_sparse_pattern=1`：shape为(B, N2, J)</li>
                <li>`layout_sparse_pattern=0`：TND为(B, N2, maxQBlockCount)，BNSD/BSND为(B, N2, S1)</li>
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
- metadata校验:

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
                        <li>`layout_sparse_pattern=1`：tensor_type仅支持int32，不同产品长度支持范围不一样，具体见下方说明</li>
                    </ul>
                </td>
                <td>`layout_sparse_pattern=1`必须存在；`layout_sparse_pattern=0`须为None</td>
                <td>`layout_sparse_pattern=1`须为generic_block_sparse_attention_grad_metadata的输出</td>
                <td>`layout_sparse_pattern=1`任务数上界B×N1×J ≤ 1048576</td>
            </tr>
        </tbody>
    </table>

<!-- npu="A3,910b" id9 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
    metadata长度≥198。
<!-- end id9 -->

<!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT系列产品</term>：metadata长度≥80+B×N1×J×4。
<!-- end id10 -->

#### Mask参数组

mask_mode参数解释：

- `MaskMode.CAUSAL`（1）：两种模式均支持，默认值。
- `MaskMode.NO_MASK`（0）：仅`layout_sparse_pattern=0`时支持。
- win_left / win_right：不使能时必须为-1；attn_mask当前应传`None`

#### SeqLens参数组

- `layout_q="TND"`时，`cu_seqlens_q`必选；`layout_kv="TND"`时，`cu_seqlens_kv`必选。
- BNSD/BSND布局下，上述累积序列长度传`None`。
- `cu_seqlens_q`/`cu_seqlens_kv`首元素必须为0，末元素分别等于T1/T2。

#### Softmax参数组

- `softmax_precision`当前仅支持0。

## 调用示例

- 以下示例演示`layout_sparse_pattern=1`时`generic_block_sparse_attention_grad_metadata`接口与主算子的联合调用（BNSD）。`layout_sparse_pattern=0`时不调用`generic_block_sparse_attention_grad_metadata`接口，参数限制见“约束说明”。

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer
  from cann_ops_transformer.ops.generic_block_sparse_attention_grad import MaskMode, ResidualBlockMode

  torch_npu.npu.set_device(0)

  B, N1, N2, S1, S2, D = 1, 1, 1, 128, 128, 128
  block_x, block_y = 1, 128
  J = math.ceil(S2 / block_y)
  scale = 1.0 / math.sqrt(D)
  block_shape = [block_x, block_y]

  q = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
  k = torch.randn(B, N2, S2, D, dtype=torch.float16, device="npu")
  v = torch.randn(B, N2, S2, D, dtype=torch.float16, device="npu")
  dout = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
  attn_out = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
  softmax_lse = torch.randn(B, N1, S1, 1, dtype=torch.float32, device="npu")

  sparse_block_idx = torch.full((B, N2, J, S1), -1, dtype=torch.int32, device="npu")
  sparse_block_count = torch.zeros((B, N2, J), dtype=torch.int32, device="npu")
  for qi in range(S1):
      sparse_block_idx[0, 0, 0, qi] = qi
  sparse_block_count[0, 0, 0] = S1

  metadata = cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
      sparse_block_idx,
      sparse_block_count,
      N1,
      N2,
      D,
      block_shape,
      layout_q="BNSD",
      layout_kv="BNSD",
      layout_sparse_pattern=1,
      mask_mode=MaskMode.CAUSAL,
      softmax_precision=0,
      win_left=-1,
      win_right=-1,
      residual_block_mode=ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX,
      is_consistent_topk=False,
  )

  dq, dk, dv = cann_ops_transformer.generic_block_sparse_attention_grad(
      q,
      k,
      v,
      dout,
      attn_out,
      softmax_lse,
      sparse_block_idx,
      sparse_block_count,
      block_shape,
      metadata=metadata,
      attn_mask=None,
      layout_q="BNSD",
      layout_kv="BNSD",
      layout_sparse_pattern=1,
      softmax_scale=scale,
      mask_mode=MaskMode.CAUSAL,
      softmax_precision=0,
      win_left=-1,
      win_right=-1,
      residual_block_mode=ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX,
      is_consistent_topk=False,
  )
  torch_npu.npu.synchronize()
  assert dq.shape == q.shape
  assert dk.shape == k.shape
  assert dv.shape == v.shape
  assert dq.dtype == q.dtype
  ```
