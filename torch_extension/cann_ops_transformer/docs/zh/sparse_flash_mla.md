# sparse_flash_mla <a name="ZH-CN_TOPIC_SPARSE_FLASH_MLA"></a>

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

## 功能说明<a name="zh-cn_topic_sparse_flash_mla_function"></a>

- **接口功能**：

  `sparse_flash_mla`是`cann_ops_transformer`扩展torch接口，用于调用`SparseFlashMla`算子完成共享KV（Key和Value使用同一份输入）的稀疏注意力计算。接口支持训练和推理场景，并通过`ori_kv`和可选的`cmp_kv`组合KV上下文。

  `sparse_flash_mla_metadata`是主算子的前置metadata生成接口，用于生成AI Core/AI Vector Core的任务切分结果。`metadata`必须由与主算子完全一致的参数生成，并传入`sparse_flash_mla`。当前场景必须传入该metadata。典型调用流程如下：

  1. 根据对应场景，准备`q`、`ori_kv`、可选的`cmp_kv`及序列长度、稀疏索引或Block Table等输入。
  2. 调用`sparse_flash_mla_metadata`生成`metadata`，作为`sparse_flash_mla`接口的入参。
  3. 调用`sparse_flash_mla`，并传入`sparse_flash_mla_metadata`的计算结果。

  该接口支持以下三类典型计算模式：

  - **SWA（Sliding Window Attention）**：仅使用`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时使用`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和TopK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时使用`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。

- **计算公式**：

  $$
  O = \text{softmax}(Q \cdot \tilde{K}^{T} \cdot \text{softmax\_scale}) \cdot \tilde{V}
  $$

  其中$\tilde{K}=\tilde{V}$，由`ori_kv`的滑动窗口部分与`cmp_kv`的压缩部分共同组成。实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`共同决定。

  开启`return_softmax_lse`后，第二个输出为每个Query位置的log-sum-exp：

  $$
  \operatorname{LSE}=\log\sum\exp(S-\max(S))+\max(S),\quad S=Q\tilde{K}^{T}\cdot\text{softmax\_scale}
  $$

## 函数原型<a name="zh-cn_topic_sparse_flash_mla_prototype"></a>

调用`sparse_flash_mla`接口前，请先调用前置接口`sparse_flash_mla_metadata`完成分核。两次调用中的参数必须保持一致。

> [!NOTE]
>
> `sparse_flash_mla_metadata`的mask和窗口参数默认值为`ori_mask_mode=0`、`cmp_mask_mode=0`、`ori_win_left=-1`、`ori_win_right=-1`，而`sparse_flash_mla`的对应默认值为`4`、`3`、`-1`、`-1`。为保证两段调用的参数一致，请显式传入这四个参数，不要同时依赖两接口的默认值。

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
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_sparse_flash_mla_parameters"></a>

### 基准信息说明

| 命名 | 含义 |
| :--- | :--- |
| b | 表示输入样本batch大小。 |
| q_n / kv_n | `q`的头数 / kv的头数。`kv_n`当前仅支持1。 |
| q_s / ori_kv_s / cmp_kv_s | `q`、`ori_kv`、`cmp_kv`的单Batch序列长度。 |
| q_t / ori_kv_t / cmp_kv_t | TND布局下`q`、`ori_kv`、`cmp_kv`所有Batch序列长度的累加和。 |
| d | 每个注意力头的维度，当前仅支持512。 |
| ori_kv_k / cmp_kv_k | 输入`ori_sparse_indices`、`cmp_sparse_indices`中topK选出的token个数。 |
| ori_kv_s_max / cmp_kv_s_max | 输入`ori_kv`、`cmp_kv`的最大序列长度。 |
| ori_kv_block_size / cmp_kv_block_size | 输入`ori_kv`、`cmp_kv`在PagedAttention场景下的block大小。 |
| ori_kv_block_nums / cmp_kv_block_nums | 输入`ori_kv`、`cmp_kv`在PagedAttention场景下的block数量。 |

### sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int32 | 必选 | `q`的头数，支持1~128。 | int32 | - | - |
| num_heads_kv | int32 | 必选 | `ori_kv`和`cmp_kv`的头数，仅支持1。 | int32 | - | - |
| head_dim | int32 | 必选 | 每个注意力头的维度，仅支持512。 | int32 | - | - |
| cu_seqlens_q | Tensor | 可选 | TND场景下`q`各Batch有效token数的累积和，第一个元素必须为0。 | int32 | ND | (b+1,) |
| cu_seqlens_ori_kv | Tensor | 可选 | TND场景下`ori_kv`各Batch有效token数的累积和，第一个元素必须为0。 | int32 | ND | (b+1,) |
| cu_seqlens_cmp_kv | Tensor | 可选 | TND场景下`cmp_kv`各Batch有效token数的累积和，第一个元素必须为0。 | int32 | ND | (b+1,) |
| seqused_q | Tensor | 可选 | 每个Batch中`q`实际参与计算的token数。 | int32 | ND | (b,) |
| seqused_ori_kv | Tensor | 可选 | 每个Batch中`ori_kv`实际参与计算的token数。 | int32 | ND | (b,) |
| seqused_cmp_kv | Tensor | 可选 | 每个Batch中`cmp_kv`实际参与计算的token数。 | int32 | ND | (b,) |
| cmp_residual_kv | Tensor | 可选 | 每个Batch压缩前kv长度除以`cmp_ratio`的余数，用于恢复cmp侧mask长度。 | int32 | ND | (b,) |
| ori_topk_length | Tensor | 可选 |  表示ori_sparse_indices实际参与计算的长度。 | int32 | ND | `BSND`：(b, q_s, kv_n)<br>`TND`：(q_t, kv_n) |
| cmp_topk_length | Tensor | 可选 |  表示cmp_sparse_indices实际参与计算的长度。 | int32 | ND | `BSND`：(b, q_s, kv_n)<br>`TND`：(q_t, kv_n) |
| batch_size | int32 | 可选 | Batch大小；BSND场景使用该值，默认值为0。 | int32 | - | - |
| max_seqlen_q | int32 | 可选 | `q`的最大有效序列长度，TND场景需与实际最大长度一致。默认值为0。 | int32 | - | - |
| max_seqlen_ori_kv | int32 | 可选 | `ori_kv`的最大有效序列长度，默认值为0。 | int32 | - | - |
| max_seqlen_cmp_kv | int32 | 可选 | `cmp_kv`的最大有效序列长度，默认值为0。 | int32 | - | - |
| ori_topk | int32 | 可选 | 表示`ori_kv`中筛选出的关键稀疏token的个数。0表示非稀疏场景。默认值为0 。 | int32 | - | - |
| cmp_topk | int32 | 可选 | 表示`cmp_kv`中筛选出的关键稀疏token的个数。0表示非稀疏场景。默认值为0。 | int32 | - | - |
| cmp_ratio | int32 | 可选 | `cmp_kv`相对于压缩前kv的压缩倍率；`cmp_kv`不传时，固定为1，`cmp_kv`传入时，取值1-128。默认值为1。 | int32 | - | - |
| ori_mask_mode | int32 | 可选 | `q`与`ori_kv`的mask模式：0（No Mask）、3（RightDownCausal）或4（Band）。 | int32 | - | - |
| cmp_mask_mode | int32 | 可选 | `q`与`cmp_kv`的mask模式：0（No Mask）或3（RightDownCausal）。 | int32 | - | - |
| ori_win_left | int32 | 可选 | `ori_kv`滑动窗口左边界，取值为-1或不小于0；-1表示不限制。 | int32 | - | - |
| ori_win_right | int32 | 可选 | `ori_kv`滑动窗口右边界，取值为-1或不小于0；-1表示不限制。 | int32 | - | - |
| layout_q | string | 可选 | `q`的数据布局，支持`BSND`和`TND`。默认值为`BSND`。 | string | - | - |
| layout_kv | string | 可选 | `ori_kv`和`cmp_kv`的数据布局，支持`BSND`、`TND`和`PA_BBND`。默认值为`BSND`。 | string | - | - |
| has_ori_kv | bool | 可选 | 主算子是否传入`ori_kv`。默认值为True。 | bool | - | - |
| has_cmp_kv | bool | 可选 | 主算子是否传入`cmp_kv`。默认值为True。 | bool | - | - |
| metadata | Tensor | 输出 | 主算子使用的任务切分结果。 | int32 | ND | (1024,) |

### sparse_flash_mla

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的Query。 | bfloat16/float16 | ND | `BSND`：(b, q_s, q_n, d)<br>`TND`：(q_t, q_n, d) |
| ori_kv | Tensor | 可选 | 原始kv输入，Key和Value共享同一份数据。 | bfloat16/float16 | ND | `BSND`：(b, ori_kv_s, kv_n, d)<br>`TND`：(ori_kv_t, kv_n, d)<br>`PA_BBND`：(ori_kv_block_nums, ori_kv_block_size, kv_n, d) |
| cmp_kv | Tensor | 可选 | 压缩kv输入，Key和Value共享同一份数据。 | bfloat16/float16 | ND | `BSND`：(b, cmp_kv_s, kv_n, d)<br>`TND`：(cmp_kv_t, kv_n, d)<br>`PA_BBND`：(cmp_kv_block_nums, cmp_kv_block_size, kv_n, d) |
| ori_sparse_indices | Tensor | 可选 | 原始kv稀疏索引；预留字段。 | int32 | ND | `TND`：(q_t, kv_n, ori_kv_k) `BSND`：(b, q_s, kv_n, ori_kv_k) |
| cmp_sparse_indices | Tensor | 可选 | 压缩kv的TopK索引，无效位置填-1；仅CSA场景传入。 | int32 | ND | `BSND`：(b, q_s, kv_n, cmp_kv_k)<br>`TND`：(q_t, kv_n, cmp_kv_k) |
| ori_block_table | Tensor | 可选 | PageAttention场景下`ori_kv`使用的Block映射表。 | int32 | ND | (b, max_num_blocks_per_seq) |
| cmp_block_table | Tensor | 可选 | PageAttention场景下`cmp_kv`使用的Block映射表。 | int32 | ND | (b, max_num_blocks_per_seq) |
| cu_seqlens_q | Tensor | 可选 | TND场景下`q`各Batch有效token数的累积和。 | int32 | ND | (b+1,) |
| cu_seqlens_ori_kv | Tensor | 可选 | TND场景下`ori_kv`各Batch有效token数的累积和。 | int32 | ND | (b+1,) |
| cu_seqlens_cmp_kv | Tensor | 可选 | TND场景下`cmp_kv`各Batch有效token数的累积和。 | int32 | ND | (b+1,) |
| seqused_q | Tensor | 可选 | 每个Batch中`q`实际参与计算的token数。 | int32 | ND | (b,) |
| seqused_ori_kv | Tensor | 可选 | 每个Batch中`ori_kv`实际参与计算的token数。 | int32 | ND | (b,) |
| seqused_cmp_kv | Tensor | 可选 | 每个Batch中`cmp_kv`实际参与计算的token数。 | int32 | ND | (b,) |
| cmp_residual_kv | Tensor | 可选 | 每个Batch的压缩余数；`cmp_kv`存在且`cmp_mask_mode=3`时必须传入。 | int32 | ND | (b,) |
| ori_topk_length | Tensor | 可选 | 表示`ori_sparse_indices`实际参与计算的长度。 | int32 | ND | - |
| cmp_topk_length | Tensor | 可选 | 表示`cmp_sparse_indices`实际参与计算的长度。 | int32 | ND | - |
| sinks | Tensor | 可选 | 表示各注意力头设置独立可学习虚拟偏移项，用于维持长文本推理时的稳定性。 | float32 | ND | (q_n,) |
| metadata | Tensor | 必选 | 由`sparse_flash_mla_metadata`生成的分核信息。 | int32 | ND | (1024,) |
| softmax_scale | float | 可选 | QK矩阵乘后的缩放系数。默认值为1.0。 | float32 | - | - |
| cmp_ratio | int32 | 可选 | 压缩倍率，取值范围1-128。；SWA场景固定为1，CSA和HCA场景取值1-128。默认值为1。 | int32 | - | - |
| ori_mask_mode | int32 | 可选 | `q`与`ori_kv`的mask模式。0：No mask。3：rightDownCausal模式。4：sliding window模式。默认值为4。 | int32 | - | - |
| cmp_mask_mode | int32 | 可选 | `q`与`cmp_kv`的mask模式。0：No mask。3：rightDownCausal模式。默认值为3。 | int32 | - | - |
| ori_win_left | int32 | 可选 | `ori_kv`滑动窗口左边界，表示`q`和`ori_kv`计算中`q`对历史token计算的数量，取值为-1或不小于0。-1表示不限制。默认值为-1。 | int32 | - | - |
| ori_win_right | int32 | 可选 | `ori_kv`滑动窗口右边界，表示`q`和`ori_kv`计算中`q`对未来token计算的数量，取值为-1或不小于0。-1表示不限制。默认值为-1。 | int32 | - | - |
| layout_q | string | 可选 | `q`的数据布局，支持`BSND`和`TND`。默认值为`BSND`。 | string | - | - |
| layout_kv | string | 可选 | kv的数据布局，支持`BSND`、`TND`和`PA_BBND`。默认值为`BSND`。 | string | - | - |
| topk_value_mode | int32 | 可选 | TopK索引取值模式，仅支持1。默认值为1。 | int32 | - | - |
| return_softmax_lse | bool | 可选 | 是否返回softmax的log-sum-exp结果。默认值为False。 | bool | - | - |
| attention_out | Tensor | 必选 | Attention计算输出，shape和数据类型与`q`一致。 | bfloat16/float16 | ND | 与`q`一致 |
| softmax_lse | Tensor | 可选 | softmax的log-sum-exp结果。 | float32 | ND | `BSND`：(b, kv_n, q_s, q_n/kv_n)<br>`TND`：(kv_n, q_t, q_n/kv_n) |

## 返回值说明<a name="zh-cn_topic_sparse_flash_mla_returns"></a>

### sparse_flash_mla_metadata

| 参数名 | 参数类型 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | `sparse_flash_mla`的分核信息。 | int32 | ND | (1024,) |

### sparse_flash_mla

| 参数名 | 参数类型 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 注意力计算输出，shape和数据类型与`q`一致。 | bfloat16/float16 | ND | 与`q`一致 |
| softmax_lse | Tensor | `return_softmax_lse=True`时返回softmax的log-sum-exp；否则返回float32标量占位Tensor。 | float32 | ND | `BSND`：(b, kv_n, q_s, q_n/kv_n)<br>`TND`：(kv_n, q_t, q_n/kv_n) |

## 约束说明<a name="zh-cn_topic_sparse_flash_mla_constraints"></a>

- 声明
  - `cu_seqlens_q`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`、`seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv`、`cmp_residual_kv`、稀疏索引及Block Table均为Tensor。算子在Tiling阶段无法校验其具体值，用户必须保证其合法性；传入非法值可能导致精度异常或非法内存访问。
  - 当输入为PA_BBND时，`seqused_ori_kv`和`ori_block_table`必须传入；当输入为BSND时，`seqused_ori_kv`可用于表达每个batch的`ori_kv`有效长度；当输入为TND时，`ori_kv`最大长度由`cu_seqlens_ori_kv`表达，若传了`seqused_ori_kv`，则有效长度由`seqused_ori_kv`表达。`cmp_kv`同理。
  - `cu_seqlens_q`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`须满足首元素为0，且序列整体呈非递减排列，即任一元素不小于其前一个元素。
  - 当layout_kv为PA_BBND时，`ori_kv`和`cmp_kv`支持0轴非连续。
  - 当`ori_mask_mode`、`cmp_mask_mode`为0时，`ori_kv_k`、`cmp_kv_k`需要大于等于`ori_topk_length`、`cmp_topk_length`的最大值。
  - `ori_topk_length`、`cmp_topk_length`表示ori/cmp sparse_indices实际参与计算的长度。其值不能大于sparse_indices的最后一维大小，且当`seqused_q`传入时，topk_length对应有效部分的值需要大于等于0。
  - `cmp_residual_kv`配合`cmp_ratio`使用，可恢复压缩前KV长度。且每个batch的值需要小于`cmp_ratio`，即`cmp_residual_kv[i]` < `cmp_ratio`。
  - `attention_out`：tensor类型，公式中的输出，数据类型支持bfloat16和float16。数据格式支持ND。限制：该输出参数的shape与入参q的shape保持一致，dtype与q一致。
  - `return_softmax_lse`为False时返回shape为[1]且值为0的tensor；`return_softmax_lse`为True时返回float32的log-sum-exp结果。
  - `cu_seqlens_q`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`须满足首元素为0，且序列整体呈非递减排列，即任一元素不小于其前一个元素。
  - `sparse_flash_mla_metadata`和`sparse_flash_mla`分两段调用。两次调用中参与任务切分的入参必须一致；不一致时可能产生未定义行为。
  - 本接口支持单算子模式和TorchAir图模式调用，可用于训练和推理场景。

### 特性参数组

| 特性参数组 | 参数字段名称 |
| :--- | :--- |
| 公共参数组 | layout_q、layout_kv、q、ori_kv、cmp_kv、attention_out、softmax_scale |
| metadata参数组 | num_heads_q、num_heads_kv、head_dim、batch_size、max_seqlen_q、max_seqlen_ori_kv、max_seqlen_cmp_kv、metadata、has_ori_kv、has_cmp_kv |
| 稀疏压缩参数组 | ori_topk、cmp_topk、cmp_ratio、ori_sparse_indices、cmp_sparse_indices |
| Mask和窗口参数组 | ori_mask_mode、cmp_mask_mode、ori_win_left、ori_win_right、cmp_residual_kv |
| Paged Attention参数组 | ori_block_table、cmp_block_table |
| SeqLengths参数组 | cu_seqlens_q、cu_seqlens_ori_kv、cu_seqlens_cmp_kv、seqused_q、seqused_ori_kv、seqused_cmp_kv |
| Sinks参数组 | sinks |
| SoftmaxLse参数组 | return_softmax_lse、softmax_lse |

### 参数组约束

下表按“单参数校验、存在性拦截、一致性拦截、特性交叉拦截”说明参数要求。对于Tensor中的具体数值，Tiling阶段无法读取或完整校验的部分以“用户保证”标记；此类约束未满足时可能不会在接口入口报错，但会导致未定义行为。

#### 前置metadata接口参数组

`sparse_flash_mla_metadata`与`sparse_flash_mla`中参与任务切分的参数必须成对传入且值完全一致。特别是布局、长度、压缩率、TopK长度、Mask模式、窗口和kv存在性标志不一致时，应拦截调用。

| 参数 | 单参数校验 | 存在性拦截 | 一致性拦截 | 特性交叉拦截 |
| :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int32；范围1~128。 | 必选。 | 必须与`q`的`q_n`维一致。 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值非1、2、4、8、16、32、64、128时拦截；<term>Ascend 950PR/Ascend 950DT</term>：取值小于1或大于128时拦截。 |
| num_heads_kv | int32；仅支持1。 | 必选。 | 必须与`ori_kv`、`cmp_kv`的kv_n维一致。 | 与`num_heads_q`、`softmax_lse`的分组维度一致。 |
| head_dim | int32；仅支持512。 | 必选。 | 必须与`q`、`ori_kv`、`cmp_kv`的d维一致。 | 不满足时拦截；不允许依赖自动推导。 |
| batch_size | int32；必须大于0。 | BSND场景必传。 | 必须与BSND布局下`q`和kv的b维一致。 | TND场景由`cu_seqlens_q`长度推导时，仍须与所有长度类Tensor一致。 |
| max_seqlen_q | int32；必须大于0。 | TND场景必传。 | 必须等于`q`各Batch实际长度的最大值。 | 与`cu_seqlens_q`及q的q_t维一致。 |
| max_seqlen_ori_kv | int32；必须大于0。 | `ori_kv`为TND布局时必传。 | 必须等于`ori_kv`各Batch实际长度的最大值。 | 与`cu_seqlens_ori_kv`及ori_kv_t一致。 |
| max_seqlen_cmp_kv | int32；必须大于0。 | `cmp_kv`为TND布局时必传。 | 必须等于`cmp_kv`各Batch实际长度的最大值。 | 与`cu_seqlens_cmp_kv`及cmp_kv_t一致。 |
| ori_topk | int32；当前仅支持0。 | 可选，默认0。 | 必须与`ori_sparse_indices`和`ori_topk_length`的传入状态一致。 | 当前不支持`ori_sparse_indices`非空，因此必须为0。 |
| cmp_topk | int32；SWA/HCA场景取值为0，CSA场景取值为压缩kv的TopK长度且大于0。 | CSA场景必传且非0；其他场景为0。 | 必须等于`cmp_sparse_indices`最后一维。 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：CSA取非512或1024、或SWA/HCA取非0时拦截；<term>Ascend 950PR/Ascend 950DT</term>：CSA取小于等于0、或SWA/HCA取非0时拦截。 |
| cmp_ratio | int32；SWA场景取值为1，CSA/HCA场景取值1-128。 | 可选，默认1。 | 必须与主接口、`cmp_residual_kv`和`cmp_kv`压缩关系一致。 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：SWA取非1、CSA取非4、HCA取非128时拦截；<term>Ascend 950PR/Ascend 950DT</term>：SWA取非1或CSA/HCA取非1-128时拦截。 |
| ori_mask_mode | int32；接口定义支持0、3、4。 | 可选。 | 无。 | 当传入4时，与`ori_win_left`、`ori_win_right`组合使用。 |
| cmp_mask_mode | int32；接口定义支持0、3。 | 可选。 | 无。 | SWA为0；CSA/HCA为3。 |
| ori_win_left | int32；接口定义为-1或非负数。 | 可选。 | 无。 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取非127时拦截；<term>Ascend 950PR/Ascend 950DT</term>：取值小于-1时拦截。仅作用于`ori_kv`侧。 |
| ori_win_right | int32；接口定义为-1或非负数。 | 可选。 | 无。 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取非0时拦截；<term>Ascend 950PR/Ascend 950DT</term>：取值小于-1时拦截。仅作用于`ori_kv`侧。 |
| layout_q | string；仅支持`BSND`、`TND`。 | 可选，默认`BSND`。 | 必须与q的维度、主接口和长度类Tensor一致。 | 仅支持与`layout_kv`组合为`BSND`/`BSND`、`TND`/`TND`、`BSND`/`PA_BBND`或`TND`/`PA_BBND`。 |
| layout_kv | string；仅支持`BSND`、`TND`、`PA_BBND`。 | 可选，默认`BSND`。 | 必须与`ori_kv`、`cmp_kv`的维度和主接口一致。 | 非PA场景必须与`layout_q`相同；PA场景要求Block Table和`seqused_ori_kv`。 |
| has_ori_kv | bool。 | 必选，与主接口实际传入状态一致。 | 必须等价于`ori_kv is not None`。 | 三种已支持场景均要求为True。 |
| has_cmp_kv | bool。 | 必选，与主接口实际传入状态一致。 | 必须等价于`cmp_kv is not None`。 | SWA为False；CSA/HCA为True。 |
| metadata（输出） | Tensor；`int32`、ND、shape为(1024,)。 | 必然输出。 | 必须作为同一组参数下主接口的`metadata`输入。 | 不允许复用由不同布局、长度或模式参数生成的`metadata`。 |

长度与可选Tensor校验：

| 参数 | 单参数校验 | 存在性拦截 | 一致性拦截 | 特性交叉拦截 |
| :--- | :--- | :--- | :--- | :--- |
| cu_seqlens_q | `int32`、ND、shape为(b+1,)；首元素为0，单调非递减，末元素为q_t。 | `layout_q="TND"`时必传；其他布局不传。 | 必须与`q`的q_t、b和`max_seqlen_q`一致。 | Tensor值由用户保证。 |
| cu_seqlens_ori_kv | `int32`、ND、shape为(b+1,)；首元素为0，单调非递减，末元素为ori_kv_t。 | `layout_kv="TND"`且传入`ori_kv`时必传；其他布局不传。 | 必须与`ori_kv`的ori_kv_t、b和`max_seqlen_ori_kv`一致。 | Tensor值由用户保证。 |
| cu_seqlens_cmp_kv | `int32`、ND、shape为(b+1,)；首元素为0，单调非递减，末元素为cmp_kv_t。 | `layout_kv="TND"`且传入`cmp_kv`时必传；其他布局不传。 | 必须与`cmp_kv`的cmp_kv_t、b和`max_seqlen_cmp_kv`一致。 | Tensor值由用户保证。 |
| seqused_q | `int32`、ND、shape为(b,)；每项非负且不超过对应`q`长度。 | 可选。 | b必须与`q`一致。 | Paged Attention不依赖此参数。Tensor值由用户保证。 |
| seqused_ori_kv | `int32`、ND、shape为(b,)；每项非负且不超过对应`ori_kv`长度。 | PA场景必传；其他场景可选。 | b必须与`ori_kv`和Block Table一致。 | PA场景决定每个序列的有效kv前缀。Tensor值由用户保证。 |
| seqused_cmp_kv | `int32`、ND、shape为(b,)；每项非负且不超过对应`cmp_kv`长度。 | 可选。 | b必须与`cmp_kv`一致。 | 显式传入时覆盖cmp侧逻辑有效长度。Tensor值由用户保证。 |
| cmp_residual_kv | `int32`、ND、shape为(b,)；每项满足`0 <= value < cmp_ratio`。 | CSA/HCA必传；SWA不传。 | 必须同时传给`metadata`接口和主接口。 | 满足`cmp_len * cmp_ratio + residual = ori_len_for_cmp_mask`。Tensor值由用户保证。 |
| ori_topk_length | `int32`、ND、shape为(b, q_s, kv_n)或(q_t, kv_n)。 | ori_mask_mode=0且`ori_sparse_indices`不为空时，必须传入；其他场景不支持传入。 | 与`ori_topk=0`、`ori_sparse_indices=None`一致。 | 当ori_mask_mode不为0时，不支持传入。 |
| cmp_topk_length | `int32`、ND、shape为(b, q_s, kv_n)或(q_t, kv_n)。 | 只有`cmp_kv`传入才校验；cmp_mask_mode=0且`cmp_sparse_indices`不为空时，必须传入；其他场景不支持传入。 | 与`cmp_topk`和`cmp_sparse_indices`的状态一致。 | 无。 |

#### 公共参数组

- `q`、`ori_kv`、`cmp_kv`的数据类型必须一致，且仅支持`float16`和`bfloat16`。
- `layout_q`和`layout_kv`仅支持`BSND`/`BSND`、`TND`/`TND`、`BSND`/`PA_BBND`、`TND`/`PA_BBND`组合。非`PA_BBND`场景下，两者必须一致。
- `layout_q="BSND"`时`q`必须为4维；`layout_q="TND"`时`q`必须为3维，并且必须传入`cu_seqlens_q`。
- `layout_kv="BSND"`或`layout_kv="PA_BBND"`时，kv必须为4维；`layout_kv="TND"`时，kv必须为3维。`layout_kv="TND"`时必须传入`cu_seqlens_ori_kv`；传入`cmp_kv`时，还必须传入`cu_seqlens_cmp_kv`。
- `metadata`必须为1024个`int32`元素；`topk_value_mode`仅支持1。
- `ori_kv`和`cmp_kv`允许存在行间padding类非连续内存，接口会通过aclNN获取stride信息并传递给底层算子。

| 参数 | 单参数校验 | 存在性拦截 | 一致性拦截 | 特性交叉拦截 |
| :--- | :--- | :--- | :--- | :--- |
| q | `float16`或`bfloat16`；`BSND`为(b, q_s, q_n, d)，`TND`为(q_t, q_n, d)；任一轴为0时拦截。 | 必传。 | `q`、`attention_out`的dtype、shape需相同。 | b > 0；d必须为512，q_n/kv_n必须为正整数；布局必须符合下表。q_s > 0；0 < q_n <= 128；q_t > 0。 |
| ori_kv | 与`q`相同的数据类型；`BSND`为(b, ori_kv_s, kv_n, d)，`TND`为(ori_kv_t, kv_n, d)，`PA_BBND`为(ori_kv_block_nums, ori_kv_block_size, kv_n, d)。 | 已支持的SWA、CSA、HCA场景均必传。 | layout_kv不为PA_BBND时，layout_q和layout_kv需保持一致；layout_kv为PA_BBND时，layout_q可为BSND或TND | PA场景必须有`ori_block_table`和`seqused_ori_kv`；block_size满足产品规格。ori_kv_s > 0；ori_kv_n = 1；ori_kv_t > 0；ori_kv_block_nums > 0；1 <= ori_kv_block_size <= 1024。 |
| cmp_kv | 与`q`相同的数据类型；`BSND`为(b, cmp_kv_s, kv_n, d)，`TND`为(cmp_kv_t, kv_n, d)，`PA_BBND`为(cmp_kv_block_nums, cmp_kv_block_size, kv_n, d)。 | SWA必须不传；CSA/HCA必传。 | 若cmp_kv传入，ori_kv与cmp_kv的dtype需一致。 | CSA要求`cmp_sparse_indices`；HCA禁止传该索引；PA场景要求`cmp_block_table`。cmp_kv_s > 0；cmp_kv_n = 1；cmp_kv_t > 0；cmp_kv_block_nums > 0；1 <= cmp_kv_block_size <= 1024。 |
| metadata | `int32`、ND、shape为(1024,)。 | 必传；缺失应拦截。 | 必须是使用本次layout、序列长度、模式、窗口、压缩率和kv存在性生成的结果。 | 与前置接口不一致或复用其他输入生成的metadata应拦截。 |
| softmax_scale | float，必须为有限值。 | 可选，默认1.0。 | 应与用户期望的缩放策略一致。 | 不参与metadata生成；常用设置为`1.0 / sqrt(d)`。 |
| layout_q | 仅支持`BSND`、`TND`。 | 可选，默认`BSND`。 | 必须与q的rank和`cu_seqlens_q`传入状态一致。 | 见布局匹配关系表。 |
| layout_kv | 仅支持`BSND`、`TND`、`PA_BBND`。 | 可选，默认`BSND`。 | 必须与所有非空kv的rank和对应长度/Block Table参数一致。 | 见布局匹配关系表。 |
| attention_out（输出） | 数据类型、rank和shape均与`q`一致。 | 必然输出。 | 与q一一对应。 | `q`为空或shape非法时应在输出构造前拦截。 |

layout匹配关系表：

| layout_q | layout_kv | q/kv维度 | softmax_lse维度（使能时） |
| :--- | :--- | :--- | :--- |
| `BSND` | `BSND` | q：(b, q_s, q_n, d)<br>kv：(b, ori_kv_s, kv_n, d) | (b, kv_n, q_s, q_n/kv_n) |
| `TND` | `TND` | q：(q_t, q_n, d)<br>kv：(ori_kv_t, kv_n, d) | (kv_n, q_t, q_n/kv_n) |
| `BSND` | `PA_BBND` | q：(b, q_s, q_n, d)<br>kv：(ori_kv_block_nums, ori_kv_block_size, kv_n, d) | (b, kv_n, q_s, q_n/kv_n) |
| `TND` | `PA_BBND` | q：(q_t, q_n, d)<br>kv：(ori_kv_block_nums, ori_kv_block_size, kv_n, d) | (kv_n, q_t, q_n/kv_n) |

#### 计算模式参数组

| 场景 | 必选输入与属性 | 不允许或固定的输入与属性 |
| :--- | :--- | :--- |
| SWA | 仅传入`ori_kv`；`cmp_ratio=1`。 | 不传`cmp_kv`、`cmp_sparse_indices`和`cmp_block_table`；`cmp_topk=0`、`cmp_mask_mode=0`。 |
| CSA | 传入`ori_kv`、`cmp_kv`、`cmp_sparse_indices`和`cmp_residual_kv`；`cmp_mask_mode=3`且`cmp_topk`为非0。 | - |
| HCA | 传入`ori_kv`、`cmp_kv`和`cmp_residual_kv`；`cmp_mask_mode=3`。 | 不传`cmp_sparse_indices`；`cmp_topk=0`。 |

| 参数 | 单参数校验 | 存在性拦截 | 一致性拦截 | 特性交叉拦截 |
| :--- | :--- | :--- | :--- | :--- |
| ori_sparse_indices | `int32`、ND。 | 必须不传。 | 与`ori_topk=0`和`ori_topk_length=None`一致。 | 当mask mode ！=0时，有效长度必须与参与计算的序列长度保持一致， 且不支持传入topk_length。当mask mode ==0时，ori_kv_k 需要大于等于对应的topklength。 |
| cmp_sparse_indices | `int32`、ND；`BSND`为(b, q_s, kv_n, cmp_kv_k)，`TND`为(q_t, kv_n, cmp_kv_k)；值必须为-1或有效的cmp token索引。 | 仅CSA必传；SWA/HCA必须不传。 | b/q_t、kv_n必须与`q`一致，cmp_kv_k必须与`cmp_topk`一致。 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：cmp_kv_k取非512或1024时拦截；<term>Ascend 950PR/Ascend 950DT</term>：cmp_kv_k取小于等于0时拦截。无效位置填-1，索引具体取值由用户保证。 |
| ori_topk_length | `int32`、ND、shape为(b, q_s, kv_n)或(q_t, kv_n)。 | `ori_topk_length` 在ori+cmp稀疏时必传。 | 与`ori_sparse_indices=None`和`ori_topk=0`一致。 | CSA/ALL_CSA场景可选，其他场景不能传；传入时，不需要传入`seqused_ori_kv`。 |
| cmp_topk_length | `int32`、ND、shape为(b, q_s, kv_n)或(q_t, kv_n)。 | `cmp_topk_length` 在ori+cmp稀疏时必传。 | 与`cmp_sparse_indices`和`cmp_topk`的状态一致。 | CSA/ALL_CSA场景可选，其他场景不能传；传入时，不需要传入`seqused_cmp_kv`。 |
| cmp_ratio | int32；SWA场景取值为1，CSA/HCA场景取值范围1-128。 | 可选，默认1。 | 必须同时与`metadata`、`cmp_kv`长度和`cmp_residual_kv`一致。 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：SWA取非1、CSA取非4、HCA取非128时拦截；<term>Ascend 950PR/Ascend 950DT</term>：SWA取非1或CSA/HCA取非1-128时拦截。 |
| topk_value_mode | int32；仅支持1。 | 可选，默认1。 | 必须与`cmp_sparse_indices`的索引取值约定一致。 | 不参与`metadata`生成；取非1值应拦截。 |

#### SeqLengths和Mask参数组

- `cu_seqlens_q`、`cu_seqlens_ori_kv`和`cu_seqlens_cmp_kv`必须为当前Batch与前序Batch有效token数的累加值，第一个元素为0，后一个元素不得小于前一个元素。
- `seqused_q`、`seqused_ori_kv`和`seqused_cmp_kv`表示各Batch的实际有效token数。`seqused_cmp_kv`在所有kv布局下均可选，显式传入时用于覆盖cmp侧逻辑有效长度。
- `cmp_residual_kv[i]`必须小于`cmp_ratio`；CSA和HCA场景下，其长度必须等于Batch大小。
- `ori_mask_mode`及`cmp_mask_mode`的详细含义请参见[sparse_mode参数说明](../../../../docs/zh/context/sparse_mode_introduction.md)。当前规格中`ori_mask_mode`支持0、3、4和`cmp_mask_mode`支持3和4，`ori_win_left`和`ori_win_right`支持-1和非负数。

| 参数 | 单参数校验 | 存在性拦截 | 一致性拦截 | 特性交叉拦截 |
| :--- | :--- | :--- | :--- | :--- |
| cu_seqlens_q | `int32`、ND、shape为(b+1,)；首元素为0、单调非递减、末元素为q_t。 | `layout_q="TND"`时必传；其他布局必须不传。 | 与`q`的q_t、B、`max_seqlen_q`和`metadata`一致。 | 具体长度值由用户保证。 |
| cu_seqlens_ori_kv | `int32`、ND、shape为(b+1,)；首元素为0、单调非递减、末元素为ori_kv_t。 | `layout_kv="TND"`且`ori_kv`非空时必传；其他布局必须不传。 | 与`ori_kv`的ori_kv_t、b、`max_seqlen_ori_kv`和`metadata`一致。 | 具体长度值由用户保证。 |
| cu_seqlens_cmp_kv | `int32`、ND、shape为(b+1,)；首元素为0、单调非递减、末元素为cmp_kv_t。 | `layout_kv="TND"`且`cmp_kv`非空时必传；其他布局必须不传。 | 与`cmp_kv`的cmp_kv_t、b、`max_seqlen_cmp_kv`和`metadata`一致。 | 具体长度值由用户保证。 |
| seqused_q | `int32`、ND、shape为(b,)；每项非负整数且不超过对应q长度。 | 可选。 | b必须与`q`、`metadata`一致。 | Tensor具体值由用户保证。 |
| seqused_ori_kv | `int32`、ND、shape为(b,)；每项非负整数且不超过对应`ori_kv`长度。 | PA场景必传；其他场景可选。 | b必须与`ori_kv`、`ori_block_table`和`metadata`一致。 | 无。 |
| seqused_cmp_kv | `int32`、ND、shape为(b,)；每项非负整数且不超过对应`cmp_kv`长度。 | 可选。 | b必须与`cmp_kv`和`metadata`一致。 | 无。 |
| cmp_residual_kv | `int32`、ND、shape为(b,)；每项范围[0, cmp_ratio)。 | cmp_mask_mode=0时可不传 | 必须与`metadata`、`cmp_ratio`和`cmp_kv`长度一致。 | 恢复长度必须满足`cmp_len * cmp_ratio + residual = ori_len_for_cmp_mask`。 |
| ori_mask_mode | `int32`；接口定义支持0、3、4。 | 可选。 | 必须与`metadata`一致。 | 当前支持0、3、4。 |
| cmp_mask_mode | `int32`；接口定义支持0、3。 | 可选。 | 必须与`metadata`一致。 | 当前支持0、3。 |
| ori_win_left | `int32`；接口定义为-1或非负数。 | 可选。 | 必须与`metadata`一致。 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取非127时拦截；<term>Ascend 950PR/Ascend 950DT</term>：取值小于-1时拦截。 |
| ori_win_right | `int32`；接口定义为-1或非负数。 | 可选。 | 必须与`metadata`一致。 | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取非0时拦截；<term>Ascend 950PR/Ascend 950DT</term>：取值小于-1时拦截。 |

#### Paged Attention参数组

- `layout_kv="PA_BBND"`时，必须传入`seqused_ori_kv`和`ori_block_table`；传入`cmp_kv`时，还必须传入`cmp_block_table`。
- PageAttention的`block_size`取值必须大于0。对于<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>和<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>，取非16的倍数或大于1024时拦截；对于<term>Ascend 950PR/Ascend 950DT</term>，取小于等于0时拦截。
- topk_value_mode=2时 `ori_block_table`、`cmp_block_table`均可不传。

| 参数 | 单参数校验 | 存在性拦截 | 一致性拦截 | 特性交叉拦截 |
| :--- | :--- | :--- | :--- | :--- |
| ori_block_table | `int32`、ND、shape为(b, max_num_blocks_per_seq)；值只能为正整数。 | `layout_kv="PA_BBND"`时必传；其他布局必须不传。 | b必须与`q`、`ori_kv`、`seqused_ori_kv`和metadata一致。 | ori_block_table存在时，必须传入`seqused_ori_kv`；PagedAttention开启情况下，`ori_block_table`必须不为空。 |
| cmp_block_table | `int32`、ND、shape为(b, max_num_blocks_per_seq)；值只能为正整数。 | `layout_kv="PA_BBND"`且`cmp_kv`非空时必传；其他情况必须不传。 | b必须与`q`、`cmp_kv`、`seqused_cmp_kv`（若传入）和metadata一致。 | `cmp_block_table`存在时，必须传入`seqused_cmp_kv`；PagedAttention开启情况下，`cmp_block_table`必须不为空。 |

#### Sinks参数组

| 参数 | 单参数校验 | 存在性拦截 | 一致性拦截 | 特性交叉拦截 |
| :--- | :--- | :--- | :--- | :--- |
| sinks | `float32`、ND、shape为(q_n,)。 | 当前版本必传。 | 长度必须等于`q`的q_n。 | 无 |

#### SoftmaxLse参数组

| 参数 | 单参数校验 | 存在性拦截 | 一致性拦截 | 特性交叉拦截 |
| :--- | :--- | :--- | :--- | :--- |
| return_softmax_lse | bool；true代表开启softmax_lse，false代表关闭softmax_lse。 | 可选参数，默认False。 | 当return_softmax_lse为false时，输出shape为[1]的值为0的tensor; 当return_softmax_lse为true时，softmax_lse的shape与layout_q的关系如下：layout_q为BSND时，softmax_lse的shape为(b, kv_n, q_s, q_n/kv_n);layout_q为TND时，softmax_lse的shape为(kv_n, q_t, q_n/kv_n)。 | 无 |
| softmax_lse | `float32`、ND；BSND为(b, kv_n, q_s, q_n/kv_n)，TND为(kv_n, q_t, q_n/kv_n)。 | `return_softmax_lse=True`时返回有效结果。 | kv_n、q_s/q_t和q_n/kv_n必须与`q`、kv一致。 | `return_softmax_lse=False`时为float32标量占位Tensor，不应读取为有效LSE。 |

## 确定性计算<a name="zh-cn_topic_sparse_flash_mla_deterministic"></a>

- 默认支持确定性计算。
- 默认支持batch invariance。

## 调用示例<a name="zh-cn_topic_sparse_flash_mla_examples"></a>

### SWA场景（BSND输入）

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
b = 1
q_s = 16
ori_kv_s = 64
q_n = 64
kv_n = 1
d = 512
cmp_ratio = 1  # SWA示例仅传ori_kv，cmp_ratio不参与压缩kv计算，保持默认值1。

q = torch.randn(b, q_s, q_n, d, dtype=dtype, device="npu")
ori_kv = torch.randn(b, ori_kv_s, kv_n, d, dtype=dtype, device="npu")
sinks = torch.zeros(q_n, dtype=torch.float32, device="npu")

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    q_n,
    kv_n,
    d,
    batch_size=b,
    max_seqlen_q=q_s,
    max_seqlen_ori_kv=ori_kv_s,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
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
    softmax_scale=1.0 / math.sqrt(d),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
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

### HCA场景（BSND输入）

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
b = 1
q_s = 16
ori_kv_s = 128
cmp_kv_s = 1
q_n = 64
kv_n = 1
d = 512
cmp_ratio = 128

q = torch.randn(b, q_s, q_n, d, dtype=dtype, device="npu")
ori_kv = torch.randn(b, ori_kv_s, kv_n, d, dtype=dtype, device="npu")
cmp_kv = torch.randn(b, cmp_kv_s, kv_n, d, dtype=dtype, device="npu")
cmp_residual_kv = torch.zeros(b, dtype=torch.int32, device="npu")
sinks = torch.zeros(q_n, dtype=torch.float32, device="npu")

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    q_n,
    kv_n,
    d,
    batch_size=b,
    max_seqlen_q=q_s,
    max_seqlen_ori_kv=ori_kv_s,
    max_seqlen_cmp_kv=cmp_kv_s,
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
    softmax_scale=1.0 / math.sqrt(d),
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

### CSA场景（TND输入，使用cmp_residual_kv）

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.float16
b = 1
q_lens = [1]
ori_lens = [6]
cmp_lens = [1]
q_n = 64
kv_n = 1
d = 512
cmp_kv_k = 512
cmp_ratio = 4

cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cu_ori = torch.tensor([0, 6], dtype=torch.int32, device="npu")
cu_cmp = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cmp_residual_kv = torch.tensor([2], dtype=torch.int32, device="npu")

q = torch.randn(sum(q_lens), q_n, d, dtype=dtype, device="npu")
ori_kv = torch.randn(sum(ori_lens), kv_n, d, dtype=dtype, device="npu")
cmp_kv = torch.randn(sum(cmp_lens), kv_n, d, dtype=dtype, device="npu")
sinks = torch.zeros(q_n, dtype=torch.float32, device="npu")

cmp_sparse_indices = torch.full((sum(q_lens), kv_n, cmp_kv_k), -1, dtype=torch.int32, device="npu")
cmp_sparse_indices[:, :, :1] = torch.arange(1, dtype=torch.int32, device="npu").view(1, 1, 1)

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    q_n,
    kv_n,
    d,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    max_seqlen_q=max(q_lens),
    max_seqlen_ori_kv=max(ori_lens),
    max_seqlen_cmp_kv=max(cmp_lens),
    ori_topk=0,
    cmp_topk=cmp_kv_k,
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
    softmax_scale=1.0 / math.sqrt(d),
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

### CP切分示例（TND + CSA，rank0切开第二个序列）

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
cmp_kv_k = 512
q_n = 64
kv_n = 1
d = 512

# 全局packed TND视角：seq0长度16，seq1长度18。
global_q_lens = [16, 18]
global_ori_lens = [16, 18]
global_cmp_lens = [4, 4]
global_cmp_residual = [0, 2]

q_global = torch.randn(sum(global_q_lens), q_n, d, dtype=dtype, device="npu")
ori_global = torch.randn(sum(global_ori_lens), kv_n, d, dtype=dtype, device="npu")
cmp_global = torch.randn(sum(global_cmp_lens), kv_n, d, dtype=dtype, device="npu")
sinks = torch.zeros(q_n, dtype=torch.float32, device="npu")


def make_cu(lengths):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32, device="npu")


def make_cmp_sparse_indices(q_lens, ori_prefix_lens, cmp_lens):
    indices = torch.full((sum(q_lens), kv_n, cmp_kv_k), -1, dtype=torch.int32, device="npu")
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
        q_n,
        kv_n,
        d,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        max_seqlen_q=max(q_lens),
        max_seqlen_ori_kv=max(ori_prefix_lens),
        max_seqlen_cmp_kv=max(cmp_lens),
        ori_topk=0,
        cmp_topk=cmp_kv_k,
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
        softmax_scale=1.0 / math.sqrt(d),
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
assert seq0_out.shape == (16, q_n, d)
assert seq1_out.shape == (18, q_n, d)
```
