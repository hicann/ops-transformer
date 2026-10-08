# quant_flash_mla_with_kvcache

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

  `quant_flash_mla_with_kvcache`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`QuantFlashMlaWithKvcache`算子完成量化场景下的MLA（Multi-head Latent Attention）注意力计算，训练推理归一化。该接口支持单算子直调与aclgraph两种调用方式。

  该接口为MLA全量化KV-cache（Paged Attention）场景，量化形式如下：

  - **Q**：per-token-head 动态量化，数据类型支持 fp8_e4m3/hifloat8；
  - **K（KV cache）**：per-tensor 静态量化，数据类型支持 fp8_e4m3/hifloat8；
  - **Q_rope / K_rope**：数据类型支持 fp8_e4m3/hifloat8；
  - **输出 attn_out**：数据类型为 bfloat16。

  其中Q的量化参数`q_descale`数据类型为FP32，shape取决于`layout_q`；K的量化参数`k_descale`为per-tensor静态量化，shape为(1,)，数据类型为FP32。

  `quant_flash_mla_with_kvcache_metadata`是`quant_flash_mla_with_kvcache`的元数据生成接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子传入该metadata以优化调度。典型调用流程如下：

  1. 准备`q`、`k_cache`、`q_descale`、`k_descale`、`block_table`、`cache_seqlens`等输入。
  2. 调用`quant_flash_mla_with_kvcache_metadata`生成`metadata`。
  3. 调用`quant_flash_mla_with_kvcache`，将上一步得到的`metadata`传入主算子。

**计算公式**:

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n \times d$的矩阵。

  self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到的矩阵，且可以统一到一个特征空间中。$Q$、$K$、$V$以低精度格式输入，并携带对应的反量化scale。公式及算子名称中的"Attention"为"self-attention"的简写。

  $$
  Attention(Q,K,V)=Score(Q,\ K) V
  $$

  本算子中Score函数采用Softmax函数，self-attention计算公式为:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$\sqrt{d}$进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n \times d$的矩阵。

  开启**return_softmax_lse**之后，返回值softmax_lse计算逻辑如下所示：

  $$
  S = \frac{QK^T}{\sqrt{d}}
  $$

  $$
  softmax\_max = max(S)
  $$

  $$
  softmax\_lse = log{\sum e^{S-softmax\_max}} + softmax\_max
  $$

> [!NOTE]
>
> Q、K、V数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、Q_T表示所有query Batch输入样本序列长度的累加和，KV_T表示所有K、V Batch输入样本序列长度的累加和。Q_S表示输入q tensor的序列长度，Q_N表示输入q tensor的头数，KV_S表示输入k/v tensor的序列长度，KV_N表示输入k/v tensor的头数。

> [!NOTE]
 >
 > MLA中K与V复用同一份数据（KV复用/KV共享），因此主算子`quant_flash_mla_with_kvcache`的输入中只有`k_cache`，不单独接收`v_cache`。算子内部基于这份共用的KV cache同时完成Key与Value的计算，无需（也不支持）额外传入V。对应的量化参数仅需提供K的`k_descale`（per-tensor静态量化，shape为(1,)）。
> 该接口仅支持Paged Attention（PA）KV-cache场景，即必须传入`block_table`与`cache_seqlens`。

## 函数原型

调用quant_flash_mla_with_kvcache接口之前，请先调用前置接口quant_flash_mla_with_kvcache_metadata，完成quant_flash_mla_with_kvcache负载均衡的计算。

```python
cann_ops_transformer.quant_flash_mla_with_kvcache_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=0,
    layout_q="BSND"
) -> Tensor
```

```python
cann_ops_transformer.quant_flash_mla_with_kvcache(
    q,
    k_cache,
    q_descale,
    k_descale,
    block_table,
    cache_seqlens,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=None,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BNBD",
    layout_out="BSND",
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## 枚举说明

`quant_mode` 与 `mask_mode` 在 Python 接口中支持传入 `IntEnum` 枚举或对应 int 值，枚举定义于 `cann_ops_transformer.ops.quant_flash_mla_with_kvcache`：

### quant_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `A8C8_Q_HIF8_PER_TOKEN_HEAD_KV_HIF8_PER_TENSOR` | 0 | A8C8 Q HIFLOAT8 per-token-head，KV HIFLOAT8 per-tensor |
| `A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR` | 1 | A8C8 Q FP8_E4M3 per-token-head，KV FP8_E4M3 per-tensor |

> [!NOTE]
>
> `quant_mode=0` 时为 HIF8 场景，`quant_mode=1` 时为 FP8_E4M3 场景，对应 q/k_cache 的数据类型分别为 hifloat8 与 float8_e4m3fn。

### mask_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `NO_MASK` | 0 | 全计算模式（默认值） |
| `CAUSAL` | 3 | Causal 模式 |

> [!NOTE]
>
> 枚举为 `IntEnum`，可直接作为 int 传入底层算子；接口仅支持传入枚举或对应 int 值。当前仅支持 mask_mode = 0/3，不支持其他值。

## 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch，表示输入样本批量大小                |
|     Q_N     |       输入q tensor的头数，对应q shape中的N，即nheads_q        |
|    KV_N    |    输入k_cache tensor的头数，对应k_cache shape中的N，即nheads_kv    |
|     Q_S     |      输入q tensor的序列长度，对应q shape中的S      |
|     D     |          隐藏层最小的单元尺寸headdim。q/k的head_dim_qk为576，v的head_dim_v为512         |
|     Bs     |          Paged Attention场景下的KV cache的块大小block_size          |
|     Bn     |          Paged Attention场景下KV cache的块数，即num_blocks          |

## 参数说明

### quant_flash_mla_with_kvcache_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cache_seqlens | Tensor | 必选 | 每个batch的KV序列长度 | int32 | ND | (batch_size,) | × |
| num_heads_q | int | 必选 | Query head数 | int32 | - | - | - |
| num_heads_kv | int | 必选 | Key/Value head数 | int32 | - | - | - |
| quant_mode | int | 必选 | 量化模式，支持传入枚举或对应 int 值 | int32 | - | - | - |
| cu_seqlens_q | Tensor | 可选 | Q的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (batch_size+1,) | × |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (batch_size,) | × |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - | - |
| head_dim_qk | int | 可选 | q/k的每个注意力头维度，默认值为576 | int32 | - | - | - |
| head_dim_v | int | 可选 | v的每个注意力头维度，默认值为512 | int32 | - | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - | - |

### quant_flash_mla_with_kvcache

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的Q | float8_e4m3fn/hifloat8 | ND | <ul><li>(B, Q_S, Q_N, D)</li><li>(B, Q_N, Q_S, D)</li><li>(Q_T, Q_N, D)</li></ul> | × |
| k_cache | Tensor | 必选 | 公式中的K（KV cache）。仅PA场景支持非连续tensor，详细约束见<a href="#Paged Attention参数组">Paged Attention参数组</a>特性交叉校验 | float8_e4m3fn/hifloat8 | ND | <ul><li>(Bn, Bs, KV_N, D)</li><li>(Bn, KV_N, Bs, D)</li><li>(Bn, KV_N, D/32, Bs, 32)</li></ul> | √ |
| q_descale | Tensor | 必选 | q的反量化scale，per-token-head动态量化 | float32 | ND | <ul><li>(B, Q_S, Q_N)</li><li>(B, Q_N, Q_S)</li><li>(Q_T, Q_N)</li></ul> | × |
| k_descale | Tensor | 必选 | k的反量化scale，per-tensor静态量化 | float32 | ND | (1,) | × |
| block_table | Tensor | 必选 | 用于分块注意力计算中的块索引映射 | int32 | ND | (B, Bn) | × |
| cache_seqlens | Tensor | 必选 | 每个batch的KV序列长度。当layout_q为TND时，为各batch序列长度的累积序列长度 | int32 | ND | (batch_size,) | × |
| cu_seqlens_q | Tensor | 可选 | Q的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (batch_size+1,) | × |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (batch_size,) | × |
| attn_mask | Tensor | 可选 | 掩码矩阵，仅在mask_mode为3（CAUSAL）时需要 | int8/uint8/bool | ND | (2048, 2048) | × |
| metadata | Tensor | 可选 | `quant_flash_mla_with_kvcache_metadata`生成的任务切分结果，传入后可优化调度 | int32 | ND | (max_schedule_size,) | x |
| quant_mode | int | 必选 | 量化模式，支持传入枚举或对应 int 值 | int32 | - | - | - |
| head_dim_v | int | 可选 | v的每个注意力头维度，默认值为512，当前仅支持512 | int32 | - | - | - |
| softmax_scale | float | 可选 | 可显式设置缩放因子。不传入时默认取 1/sqrt(headdim) | float32 | - | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」 | int32 | - | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - | - |
| layout_kv | string | 可选 | 定义输入k_cache张量的布局格式 | string | - | - | - |
| layout_out | string | 可选 | 定义输出attn_out张量的布局格式 | string | - | - | - |
| return_softmax_lse | bool | 可选 | 是否需要获取softmax的LSE结果 | BOOL | - | - | - |

## 返回值说明

### quant_flash_mla_with_kvcache_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | quant_flash_mla_with_kvcache的任务切分数据 | int32 | ND | shape根据batch_size和num_heads_kv动态计算 | x |

### quant_flash_mla_with_kvcache

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | 必选 | quant_flash_mla_with_kvcache的计算输出 | bfloat16 | ND | <ul><li>(B, Q_S, Q_N, D)</li><li>(B, Q_N, Q_S, D)</li><li>(Q_T, Q_N, D)</li><li>(Q_N, Q_T, D)</li></ul> | × |
| softmax_lse | Tensor | 可选 | softmax的LSE结果。`return_softmax_lse`为True时输出；`return_softmax_lse`为False时输出空Tensor | float32 | ND | <ul><li>(B, Q_N, Q_S)</li><li>(Q_N, Q_T)</li></ul> | × |

## 约束说明

- 参数cu_seqlens_q、seqused_q、block_table、cache_seqlens及attn_mask属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- quant_flash_mla_with_kvcache_metadata和quant_flash_mla_with_kvcache的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- 该接口仅支持Paged Attention（PA）场景，不传入block_table将被拦截。

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |       k_cache       |      INPUT      |   Tensor   |
|                      |         metadata        |      INPUT(OPTIONAL)      | Tensor |
|                      |      softmax_scale      | ATTR(OPTIONAL) |   double   |
|                      |      head_dim_v      | ATTR(OPTIONAL) |   int   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |      layout_out      | ATTR(OPTIONAL) |   string   |
|                      |     attn_out     |     OUTPUT     |   Tensor   |
|      全量化参数组      |       quant_mode       | ATTR |   int   |
|                      |       q_descale       | INPUT |   Tensor   |
|                      |       k_descale       | INPUT |   Tensor   |
|      Mask参数组      |       mask_mode       | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | INPUT(OPTIONAL) |   Tensor   |
| SeqLens参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_q  | INPUT(OPTIONAL) |  Tensor  |
|                      |  max_seqlen_q  | ATTR(OPTIONAL) |  int  |
|                      |  max_seqlen_kv  | ATTR(OPTIONAL) |  int  |
| Paged Attention参数组 |      block_table      | INPUT |   Tensor   |
|                      |      cache_seqlens      | INPUT |   Tensor   |
|   SoftmaxLSE参数组   |    return_softmax_lse    | ATTR(OPTIONAL) |    bool    |
|                      |      softmax_lse      |     OUTPUT(OPTIONAL)     |   Tensor   |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
  - 空Tensor指必选输入和输出的shape size为0，即有任意轴为0。
  - 触发空Tensor的用例将全部拦截报错。

- q、k_cache、attn_out校验：

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
                    <li>tensor_type支持float8_e4m3fn/hifloat8</li>
                    <li>shape dim支持3、4</li>
                </ul>
            </td>
            <td rowspan="3">
                必须存在
            </td>
            <td rowspan="3">
                <ul>
                    <li>q、k_cache的数据类型必须相同</li>
                </ul>
            </td>
            <td rowspan="3">
                <ul>
                    <li>轴校验：
                        <ul>
                            <li>S ∈ [1, 16]</li>
                            <li>Q_N ∈ {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}</li>
                            <li>KV_N = 1</li>
                            <li>q/k的head_dim_qk仅支持576（nope+rope）</li>
                            <li>v的head_dim_v仅支持512</li>
                            <li>Q_T ≥ 0</li>
                        </ul>
                    </li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>k_cache</td>
            <td>
                <ul>
                    <li>tensor_type支持float8_e4m3fn/hifloat8</li>
                    <li>shape dim支持4、5</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_out</td>
            <td>
                <ul>
                    <li>data_type仅支持bfloat16</li>
                    <li>shape dim支持3、4</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>layout_q</td>
            <td>支持BSND/BNSD/TND/TND_NTD</td>
            <td rowspan="3">当前不支持不传入，未传入将发出拦截报警</td>
            <td rowspan="3">无</td>
            <td rowspan="3">无</td>
        </tr>
        <tr>
            <td>layout_kv</td>
            <td>支持PA_BBND/PA_BNBD/PA_NZ</td>
        </tr>
        <tr>
            <td>layout_out</td>
            <td>支持BSND/BNSD/TND/NTD</td>
        </tr>
        <tr>
            <td>metadata</td>
            <td>
                <ul>
                    <li>tensor_type仅支持int32</li>
                    <li>shape由quant_flash_mla_with_kvcache_metadata动态计算</li>
                    <li>当前不支持不传入，未传入将发出拦截报警</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>传入时需与quant_flash_mla_with_kvcache_metadata生成的结果一致</td>
        </tr>
    </tbody>
    </table>

#### 全量化参数组

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
            <td>
                <ul>
                    <li>data_type支持int32</li>
                </ul>
            </td>
            <td>必选属性</td>
            <td rowspan="3">
                <ul>
                    <li>q、k_cache的数据类型与quant_mode量化模式匹配：q/k为fp8_e4m3或hifloat8</li>
                    <li>q_descale、k_descale的dtype与shape匹配关系见下表</li>
                </ul>
            </td>
            <td rowspan="3">
                <ul>
                    <li>不支持非连续Tensor</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>q_descale</td>
            <td>
                <ul>
                    <li>tensor_type仅支持float32</li>
                    <li>per-token-head动态量化，shape与layout_q匹配</li>
                </ul>
            </td>
            <td>必须存在</td>
        </tr>
        <tr>
            <td>k_descale</td>
            <td>
                <ul>
                    <li>tensor_type仅支持float32</li>
                    <li>per-tensor静态量化，shape为(1,)</li>
                </ul>
            </td>
            <td>必须存在</td>
        </tr>
    </tbody>
</table>

- q_descale shape匹配关系表：

    <table style="undefined;table-layout: fixed; width:1625px">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 293px">
        </colgroup>
        <thead>
            <tr>
                <th>layout_q</th>
                <th>参数</th>
                <th>shape</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>BSND</td>
                <td>q_descale</td>
                <td>(B, Q_S, Q_N)</td>
            </tr>
            <tr>
                <td>BNSD</td>
                <td>q_descale</td>
                <td>(B, Q_N, Q_S)</td>
            </tr>
            <tr>
                <td>TND/TND_NTD</td>
                <td>q_descale</td>
                <td>(Q_T, Q_N)</td>
            </tr>
        </tbody>
    </table>

- k_descale shape匹配关系表：

    <table style="undefined;table-layout: fixed; width:1625px">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 293px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>layout_kv</th>
                <th>shape</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>k_descale</td>
                <td>-</td>
                <td>(1,)<br>per-tensor 量化，1D</td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_mode参数解释
<ul>
    <li>mask_mode=0，NO_MASK，全计算模式（默认值）</li>
    <li>mask_mode=3，CAUSAL，Causal模式</li>
</ul>

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
                    <li>data_type支持int32</li>
                    <li>支持输入为0/3</li>
                </ul>
            </td>
            <td>
                可选属性，默认值为0
            </td>
            <td rowspan="2">
                <ul>
                    <li>当mask_mode为0时，不支持传入attn_mask</li>
                    <li>当mask_mode为3时，必须传入attn_mask矩阵</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>当前仅支持mask_mode=0/3，不支持其他值</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>
                <ul>
                    <li>tensor_type支持int8</li>
                    <li>tensor_shape为(2048, 2048)</li>
                </ul>
            </td>
            <td>
                可选输入
            </td>
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
            <td>
                <ul>
                    <li>tensor_type支持int32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>值仅支持非负整数</li>
                    <li>seqused_q中的值需小于等于Q_S</li>
                </ul>
            </td>
            <td rowspan="4">可选参数</td>
            <td rowspan="4">无</td>
            <td>
                <ul>
                    <li>当layout_q为TND时，seqused_q与max_seqlen_q至少传入其中一个</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>
                <ul>
                    <li>tensor_type支持int32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于Q_T</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，必须传入</li>
                    <li>当layout_q不为TND时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td>
                <ul>
                    <li>data_type支持int32</li>
                    <li>值需 ≥ -1</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时与seqused_q至少传入其中一个</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
            <td>
                <ul>
                    <li>data_type支持int32</li>
                    <li>值需 ≥ -1</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
    </tbody>
</table>

> [!NOTE]
>
> 算子不校验 `seqused_q` 中的最大值是否与 `max_seqlen_q` 一致。若同时传入这两个参数，用户需自行保证 `max(seqused_q) <= max_seqlen_q`。

#### Paged Attention参数组 <a name="Paged Attention参数组"></a>

该接口为PA（Paged Attention）专用场景，必须传入block_table与cache_seqlens。
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
                    <li>tensor_shape为(B, Bn)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>必选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>PA开启情况下，block_table必须不为空</li>
                    <li>当layout_kv=PA_BNBD/PA_NZ时，k_cache仅支持0轴或0轴1轴非连续；当layout_kv=PA_BBND时，k_cache仅支持0轴非连续</li>
                    <li>不支持v_cache传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cache_seqlens</td>
            <td>
                <ul>
                    <li>tensor_type仅支持int32</li>
                    <li>tensor_shape为(batch_size,)</li>
                    <li>值仅支持非负整数</li>
                </ul>
            </td>
            <td>必选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>-</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>k_cache</td>
            <td>
                <ul>
                    <li>tensor_type支持float8_e4m3fn/hifloat8</li>
                    <li>b、d数量需满足head_dim_qk=576，KV_N=1</li>
                </ul>
            </td>
            <td>必选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>layout_kv仅支持PA_BBND/PA_BNBD/PA_NZ</li>
                    <li>layout_kv=PA_BBND时，k_cache为(Bn, Bs, N, D)排布</li>
                    <li>layout_kv=PA_BNBD时，k_cache为(Bn, N, Bs, D)排布</li>
                    <li>layout_kv=PA_NZ时，k_cache为(Bn, N, D/32, Bs, 32)排布</li>
                </ul>
            </td>
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
                    <li>值仅支持True和False，True代表开启softmax_lse，False代表关闭softmax_lse</li>
                </ul>
            </td>
            <td>可选属性，默认值为False</td>
            <td rowspan="2">
                <ul>
                    <li>当return_softmax_lse为False时，输出空Tensor</li>
                    <li>当return_softmax_lse为True时，softmax_lse必须非空</li>
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

- quant_flash_mla_with_kvcache_metadata + quant_flash_mla_with_kvcache联合调用示例（TND + PA场景，causal mask）

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    dtype = torch.float8_e4m3fn
    out_dtype = torch.bfloat16
    B = 2
    Q_S = 16
    Q_N = 8
    KV_N = 1
    head_dim_qk = 576
    head_dim_v = 512
    pa_block_size = 512

    # TND排布，Q_T为各batch序列长度累加和
    Q_T = B * Q_S
    num_blocks_per_seq = 1
    total_blocks = num_blocks_per_seq * B

    q = torch.randn(Q_T, Q_N, head_dim_qk, dtype=dtype, device="npu")
    k_cache = torch.randn(total_blocks, KV_N, pa_block_size, head_dim_qk, dtype=dtype, device="npu")

    # descale：q为per-token-head动态量化，k为per-tensor静态量化
    q_descale = torch.randn(Q_T, Q_N, dtype=torch.float32, device="npu")
    k_descale = torch.randn(1, dtype=torch.float32, device="npu")

    # block_table
    block_table = torch.arange(total_blocks, dtype=torch.int32, device="npu").reshape(B, num_blocks_per_seq)

    # cache_seqlens：layout_q为TND时，为各batch序列长度的累积序列长度
    cache_seqlens = torch.tensor([pa_block_size, pa_block_size * 2], dtype=torch.int32, device="npu")

    # 累计序列长度（带前导0）
    cu_seqlens_q = torch.tensor([0, Q_S, Q_S * 2], dtype=torch.int32, device="npu")

    # attn_mask (causal, 2048*2048)
    attn_mask = torch.tril(torch.ones(2048, 2048, dtype=torch.int8, device="npu"))

    metadata = cann_ops_transformer.ops.quant_flash_mla_with_kvcache_metadata(
        cache_seqlens,
        Q_N,
        KV_N,
        quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=Q_S,
        max_seqlen_kv=pa_block_size,
        mask_mode=3,
        layout_q="TND",
    )

    attn_out, softmax_lse = cann_ops_transformer.ops.quant_flash_mla_with_kvcache(
        q, k_cache,
        q_descale, k_descale,
        block_table, cache_seqlens,
        quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        attn_mask=attn_mask,
        metadata=metadata,
        softmax_scale=1.0 / (head_dim_qk ** 0.5),
        mask_mode=3,
        max_seqlen_q=Q_S,
        max_seqlen_kv=pa_block_size,
        layout_q="TND",
        layout_kv="PA_BNBD",
        layout_out="TND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == (Q_T, Q_N, head_dim_qk)
    assert attn_out.dtype == out_dtype
    assert torch.isfinite(attn_out.float()).all().item()
    ```
