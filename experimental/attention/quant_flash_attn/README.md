# quant_flash_attn（MxFP4/MxFP8 Softmax FP16）

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

- **接口功能**:

  `quant_flash_attn`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`QuantFlashAttn`算子完成全量化注意力计算，训练推理归一。当前支持 MxFP4 与 MxFP8 Softmax FP16 场景：

  - **MxFP4场景**（`quant_mode=5`）：Q/K/V 均采用 MxFP4，P 采用 MxFP4，Softmax 在 FP16 下计算。
  - **MxFP8 Softmax FP16场景**（`quant_mode=3`）：Q/K/V 均采用 MxFP8（float8_e4m3fn），P 采用 FP8_E4M3（per-tensor），Softmax 在 FP16 下计算。

  `quant_flash_attn_metadata`是`quant_flash_attn`的元数据生成接口，用于在主算子执行前生成metadata。metadata记录AI Core/AIVCore的任务切分结果，主算子可选择传入该metadata以优化调度。典型调用流程如下：

  1. 准备`q`、`k`、`v`等输入。
  2. 调用`quant_flash_attn_metadata`生成`metadata`。
  3. 调用`quant_flash_attn`，将上一步得到的`metadata`传入主算子。

- **计算公式**:

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
> - Q、K、V数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、G = Q_N / KV_N 表示GQA分组数。
> - MxFP4场景（`quant_mode=5`）下，仅支持 D=128，Q/Out 支持 BNSD、TND 排布，KV 支持 BNSD、TND 与 Paged Attention（PA_BNBD/PA_BBND/PA_NZ），支持 GQA 与 CAUSAL mask。
> - MxFP8 Softmax FP16场景（`quant_mode=3`）下，仅支持 BNSD 排布，仅支持 D=128，不支持 Paged Attention。

## 函数原型

调用quant_flash_attn接口之前，请先调用前置接口quant_flash_attn_metadata，完成quant_flash_attn负载均衡的计算。

```python
cann_ops_transformer.quant_flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    v_descale=None,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND"
) -> Tensor
```

```python
cann_ops_transformer.quant_flash_attn(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    quant_mode,
    *,
    block_table=None,
    p_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## 枚举说明

`quant_mode` 与 `mask_mode` 在 Python 接口中支持传入 `IntEnum` 枚举或对应 int 值，枚举定义于 `cann_ops_transformer.ops.quant_flash_attn`：

### quant_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `A8C8_QKV_HIF8_PER_TENSOR_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32` | 0 | A8C8 Q/KV HIF8 per-tensor，P FP8_E4M3 per-tensor，Softmax FP32 |
| `A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32` | 1 | A8C8 Q/KV MxFP8，P FP8_E4M3 per-tensor，Softmax FP32 |
| `A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP32` | 2 | A8C8 Q/KV MxFP8，P MxFP8，Softmax FP32 |
| `A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16` | 3 | A8C8 Q/KV MxFP8，P FP8_E4M3 per-tensor，Softmax FP16 |
| `A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP16` | 4 | A8C8 Q/KV MxFP8，P MxFP8，Softmax FP16 |
| `A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16` | 5 | A4C4 Q/KV MxFP4，P MxFP4，Softmax FP16 |
| `A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32` | 6 | A8C8 Q/K FP8_E4M3 per-token-head、V FP8_E4M3 per-head、P FP8_E4M3 per-tensor，Softmax FP32 |
| `A4C4_QKV_HIF4_P_HIF4_LEVEL1_SOFTMAX_FP16` | 7 | A4C4 Q/KV HIF4，P HIF4 Level1，Softmax FP16 |
| `A4C4_QKV_HIF4_P_HIF4_LEVEL2_SOFTMAX_FP16` | 8 | A4C4 Q/KV HIF4，P HIF4 Level2，Softmax FP16 |
| `A4C4_QKV_HIF4_P_HIF4_LEVEL3_SOFTMAX_FP16` | 9 | A4C4 Q/KV HIF4，P HIF4 Level3，Softmax FP16 |
| `A4C4_QKV_HIF4_P_FP8_E4M3_SOFTMAX_FP16` | 10 | A4C4 Q/KV HIF4，P FP8_E4M3，Softmax FP16 |

> [!NOTE]
>
> 当前实现 `quant_mode=5`（MxFP4）与 `quant_mode=3`（MxFP8 Softmax FP16）场景。

### mask_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `NO_MASK` | 0 | 全计算模式（默认值） |
| `CAUSAL` | 3 | Causal 模式 |
| `SLIDING_WINDOW` | 4 | Sliding Window 模式 |

> [!NOTE]
>
> - MxFP4 场景（`quant_mode=5`）支持 mask_mode = 0（`NO_MASK`）与 3（`CAUSAL`）。
> - MxFP8 Softmax FP16 场景（`quant_mode=3`）仅支持 mask_mode = 0（`NO_MASK`）。

## 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|    KV_N    |    输入k/v tensor的头数，对应k/v shape中的N    |
|     Q_S     |      输入q tensor的序列长度，对应q shape中的S      |
|    KV_S    |  输入k/v tensor的序列长度，对应k/v shape中的S  |
|     D     |          输入q/k/v tensor以及输出attn_out隐藏层最小的单元尺寸headdim         |
|     G     |          GQA分组数，G = Q_N / KV_N          |

## 参数说明

### quant_flash_attn_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | 必选 | Query head数 | int32 | - | - | - |
| num_heads_kv | int | 必选 | Key/Value head数 | int32 | - | - | - |
| head_dim | int | 必选 | 每个注意力头的维度 | int32 | - | - | - |
| quant_mode | int/QuantMode | 必选 | 量化模式，支持传入枚举或对应 int 值，枚举定义见「quant_mode 枚举」 | int32 | - | - | - |
| cu_seqlens_q | Tensor | 可选 | Q的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| cu_seqlens_kv | Tensor | 可选 | KV的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| seqused_q | Tensor | 可选 | q的指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| seqused_kv | Tensor | 可选 | kv的指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| v_descale | Tensor | 可选 | v的反量化scale，用于校验 | float8_e8m0 | ND | - | × |
| batch_size | int | 可选 | batch大小。若未传入，则从cu_seqlens_q或seqused_q推导。默认值为None | int32 | - | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」 | int32 | - | - | - |
| win_left | int | 可选 | window左界限 | int32 | - | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - | - |
| layout_q_descale | string | 可选 | 定义输入q_descale张量的布局格式 | string | - | - | - |
| layout_kv | string | 可选 | 定义输入k和v张量的布局格式 | string | - | - | - |
| layout_out | string | 可选 | 定义输出张量的布局格式 | string | - | - | - |

### quant_flash_attn

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的Q | float4_e2m1（MxFP4）/ float8_e4m3fn（MxFP8 Softmax FP16） | ND | BNSD: (B, Q_N, Q_S, D)；TND: (Q_T, Q_N, D) | × |
| k | Tensor | 必选 | 公式中的K | float4_e2m1（MxFP4）/ float8_e4m3fn（MxFP8 Softmax FP16） | ND | BNSD: (B, KV_N, KV_S, D)；TND: (KV_T, KV_N, D) | × |
| v | Tensor | 必选 | 公式中的V | float4_e2m1（MxFP4）/ float8_e4m3fn（MxFP8 Softmax FP16） | ND | BNSD: (B, KV_N, KV_S, D)；TND: (KV_T, KV_N, D) | × |
| q_descale | Tensor | 必选 | q的反量化scale | float8_e8m0 | ND | BNSD: (B, Q_N, Q_S, D/64, 2)；TND: N2TGD (KV_N, Q_T, G, D/64, 2) | × |
| k_descale | Tensor | 必选 | k的反量化scale | float8_e8m0 | ND | BNSD: (B, KV_N, KV_S, D/64, 2)；TND: (KV_T, KV_N, D/64, 2) | × |
| v_descale | Tensor | 必选 | v的反量化scale | float8_e8m0 | ND | BNSD: (B, KV_N, ceil(KV_S/64), D, 2)；TND: (sum(ceil(packed_kv_len_i/64)), KV_N, D, 2)，`packed_kv_len_i` 为第 i 个 batch 的**物理打包长度**（即 `cu_seqlens_kv[i+1]-cu_seqlens_kv[i]`，可大于 `seqused_kv[i]`）；PA 见[descale shape匹配关系表](#descale_shape匹配关系表) | × |
| quant_mode | int/QuantMode | 必选 | 量化模式，支持传入枚举或对应 int 值，枚举定义见「quant_mode 枚举」 | int32 | - | - | - |
| block_table | Tensor | 可选 | 用于分块注意力计算中的块索引映射 | int32 | ND | (B, Bn) | × |
| cu_seqlens_q | Tensor | 可选 | Q的累积序列长度，用于处理变长序列，第一个元素必须为0。layout_q为TND时必传，非TND时禁止传入 | int32 | ND | (B+1,) | × |
| cu_seqlens_kv | Tensor | 可选 | KV的累积序列长度，用于处理变长序列，第一个元素必须为0。layout_kv为TND时必传，非TND时禁止传入 | int32 | ND | (B+1,) | × |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算。layout_q不为TND时与max_seqlen_q至少传入其中一个 | int32 | ND | (B,) | × |
| seqused_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算。layout_kv不为TND且不为PA场景时与max_seqlen_kv至少传入其中一个 | int32 | ND | (B,) | × |
| p_scale | Tensor | 可选 | P的量化参数 | float32 | ND | (1,) | × |
| sinks | Tensor | 可选 | 暂不支持 | float32 | ND | (Q_N,) | × |
| attn_mask | Tensor | 可选 | 掩码矩阵 | int8 | ND | (2048, 2048) | × |
| metadata | Tensor | 可选 | `quant_flash_attn_metadata`生成的任务切分结果，传入后可优化调度 | int32 | ND | (max_schedule_size,) | x |
| softmax_scale | float | 可选 | 可显式设置缩放因子，覆盖默认计算 | float32 | - | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」 | int32 | - | - | - |
| win_left | int | 可选 | window左界限 | int32 | - | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - | - |
| layout_q_descale | string | 可选 | 定义输入q_descale张量的布局格式 | string | - | - | - |
| layout_kv | string | 可选 | 定义输入k和v张量的布局格式 | string | - | - | - |
| layout_out | string | 可选 | 定义输出张量的布局格式 | string | - | - | - |
| return_softmax_lse | bool | 可选 | 是否需要获取softmax的LSE结果 | BOOL | - | - | - |

> [!NOTE]
>
> 本文档中 q/k/v/attn_out 的 shape 一律按量化后的**逻辑元素数**计（MxFP4 为 fp4 元素，MxFP8 Softmax FP16 为 e4m3 元素，两场景 D 均为 128）。MxFP4 下 q/k/v 按 4bit 物理存储（2 个元素占 1 byte），以 `uint8` 或 `float4_e2m1fn_x2` 传入时末维写 D/2；`attn_out`、descale、`softmax_lse` 无打包，按表中 shape 构造。

## 返回值说明

### quant_flash_attn_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | quant_flash_attn的任务切分数据 | int32 | ND | shape根据batch_size和num_heads_kv动态计算 | x |

### quant_flash_attn

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | 必选 | quant_flash_attn的计算输出。 | bfloat16 | ND | BNSD: (B, Q_N, Q_S, D)；TND: (Q_T, Q_N, D) | × |
| softmax_lse | Tensor | 可选 | softmax的LSE结果。`return_softmax_lse`为True时，BNSD输出shape为(B, Q_N, Q_S)，TND输出shape为(Q_T, Q_N)；`return_softmax_lse`为False时，则输出空Tensor。 | float32 | ND | BNSD: (B, Q_N, Q_S)；TND: (Q_T, Q_N) | × |

## 约束说明

- 参数cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv、block_table及attn_mask属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- quant_flash_attn_metadata和quant_flash_attn的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |          k          |      INPUT      | Tensor |
|                      |         v         |      INPUT      | Tensor |
|                      |         metadata        |      INPUT(OPTIONAL)      | Tensor |
|                      |      softmax_scale      | ATTR(OPTIONAL) |   double   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_q_descale      | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |      layout_out      | ATTR(OPTIONAL) |   string   |
|                      |     attn_out     |     OUTPUT     |   Tensor   |
|      全量化参数组      |       quant_mode       | ATTR |   int   |
|                      |       q_descale       | INPUT |   Tensor   |
|                      |       k_descale       | INPUT |   Tensor   |
|                      |       v_descale       | INPUT |   Tensor   |
|                      |       p_scale       | INPUT(OPTIONAL) |   Tensor   |
|      Mask参数组      |       mask_mode       | ATTR(OPTIONAL) |   int   |
|                      |       win_left       | ATTR(OPTIONAL) |   int   |
|                      |      win_right      | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | INPUT(OPTIONAL) |   Tensor   |
| SeqLens参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seqlens_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_q  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |  max_seqlen_q  | ATTR(OPTIONAL) |  int  |
|                      |  max_seqlen_kv  | ATTR(OPTIONAL) |  int  |
| Paged Attention参数组 |      block_table      | INPUT(OPTIONAL) |   Tensor   |
|  Sinks参数组  |     sinks     | INPUT(OPTIONAL) |   Tensor   |
|   SoftmaxLSE参数组   |    return_softmax_lse    | ATTR(OPTIONAL) |    bool    |
|                      |      softmax_lse      |     OUTPUT(OPTIONAL)     |   Tensor   |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
  - 空Tensor指必选输入和输出的shape size为0,即有任意轴为0。
  - 触发空Tensor的用例将全部拦截报错。

- q、k、v、attn_out校验:

    <table style="width:100%;table-layout:auto"><colgroup>
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
                    <li>tensor_type支持float8_e4m3fn、float4_e2m1</li>
                    <li>shape dim支持3、4</li>
                </ul>
            </td>
            <td rowspan="4">
                必须存在
            </td>
            <td rowspan="4">
                <ul>
                    <li>q、k、v的数据类型必须相同（均为float4_e2m1 或均为 float8_e4m3fn）</li>
                </ul>
            </td>
            <td rowspan="4">
                轴校验：
                <ul>
                    <li>65536 > B > 0</li>
                    <li>Q_S ≥ 0；KV_S ≥ 0</li>
                    <li>D仅支持64或128</li>
                    <li>Q_N % KV_N == 0且Q_N / KV_N > 0</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>k</td>
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持float8_e4m3fn、float4_e2m1</li>
                    <li>shape dim支持3、4、5</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>v</td>
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
            <td>支持BSND/BNSD/TND</td>
            <td rowspan="4">当前不支持不传入，未传入将发出拦截报警</td>
            <td rowspan="4">无</td>
            <td rowspan="4">无</td>
        </tr>
        <tr>
            <td>layout_q_descale</td>
            <td>支持BSND/BNSD/TND/N2TGD</td>
        </tr>
        <tr>
            <td>layout_kv</td>
            <td>支持BSND/BNSD/TND/PA_BNBD/PA_BBND/PA_NZ</td>
        </tr>
        <tr>
            <td>layout_out</td>
            <td>支持BSND/BNSD/TND</td>
        </tr>
        <tr>
            <td>metadata</td>
            <td>
                <ul>
                    <li>tensor_type仅支持int32</li>
                    <li>shape由quant_flash_attn_metadata动态计算</li>
                    <li>当前不支持不传入，未传入将发出拦截报警</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>传入时需与quant_flash_attn_metadata生成的结果一致</td>
        </tr>
    </tbody>
    </table>

- softmax_scale校验:
    - softmax_scale 必须大于 0。

#### 全量化参数组

- quant_mode参数解释（枚举定义见「quant_mode 枚举」）:

    <table style="width:100%;table-layout:auto">
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
                        <li>支持输入值为 A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16（5）</li>
                        <li>支持输入值为 A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16（3）</li>
                    </ul>
                </td>
                <td>必选属性</td>
                <td>无</td>
                <td rowspan="5">
                    <ul>
                        <li>不支持非连续Tensor</li>
                        <li>Layout校验规则见<a href="#layout匹配关系表">layout匹配关系表</a></li>
                        <li>q、k、v、attn_out shape校验规则见<a href="#qkv_attn_out_shape匹配关系表">q/k/v/attn_out shape匹配关系表</a></li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>q_descale</td>
                <td>
                    <ul>
                        <li>tensor_type支持float8_e8m0、float32</li>
                        <li>shape dim支持4、5</li>
                    </ul>
                </td>
                <td rowspan="3">必须存在</td>
                <td rowspan="3">
                    <ul>
                        <li>descale shape校验规则见<a href="#descale_shape匹配关系表">descale_shape匹配关系表</a></li>
                        <li>descale dtype校验规则见<a href="#descale_dtype匹配关系表">descale_dtype匹配关系表</a></li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>k_descale</td>
                <td>
                    <ul>
                        <li>tensor_type支持float8_e8m0、float32</li>
                        <li>shape dim支持4、5、6</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>
                    <ul>
                        <li>tensor_type支持float8_e8m0、float32</li>
                        <li>shape dim支持4、5、6</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>p_scale</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持float32</li>
                        <li>shape仅支持(1)</li>
                    </ul>
                </td>
                <td>可选参数，默认值为 [1.0f]</td>
                <td>无</td>
            </tr>
        </tbody>
    </table>

- layout匹配关系表：<a name="layout匹配关系表"></a>

    <table style="width:100%;table-layout:auto"><colgroup>
    <col style="width: 110px">
    <col style="width: 130px">
    <col style="width: 130px">
    <col style="width: 180px">
    <col style="width: 110px">
    <col style="width: 160px">
    </colgroup>
    <thead>
    <tr>
        <th>quant_mode</th>
        <th>layout_q</th>
        <th>layout_q_descale</th>
        <th>layout_kv</th>
        <th>layout_out</th>
        <th>layout_softmax_lse</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>quant_mode=5（MxFP4）</td>
            <td>BNSD</td>
            <td>BNSD</td>
            <td>BNSD</td>
            <td>BNSD</td>
            <td>(B, Q_N, Q_S)</td>
        </tr>
        <tr>
            <td>quant_mode=5（MxFP4）</td>
            <td>BNSD</td>
            <td>BNSD</td>
            <td>PA_BNBD/PA_BBND/PA_NZ</td>
            <td>BNSD</td>
            <td>(B, Q_N, Q_S)</td>
        </tr>
        <tr>
            <td>quant_mode=5（MxFP4）</td>
            <td>TND</td>
            <td>N2TGD</td>
            <td>TND</td>
            <td>TND</td>
            <td>(Q_T, Q_N)</td>
        </tr>
        <tr>
            <td>quant_mode=5（MxFP4）</td>
            <td>TND</td>
            <td>N2TGD</td>
            <td>PA_BNBD/PA_BBND/PA_NZ</td>
            <td>TND</td>
            <td>(Q_T, Q_N)</td>
        </tr>
        <tr>
            <td>quant_mode=3（MxFP8 Softmax FP16）</td>
            <td>BNSD</td>
            <td>BNSD</td>
            <td>BNSD</td>
            <td>BNSD</td>
            <td>(B, Q_N, Q_S)</td>
        </tr>
    </tbody>
    </table>

- q/k/v descale dtype匹配关系表: <a name="descale_dtype匹配关系表"></a>

    <table style="width:100%;table-layout:auto">
        <colgroup>
            <col style="width: 120px">
            <col style="width: 147px">
            <col style="width: 293px">
        </colgroup>
        <thead>
            <tr>
                <th>quant_mode</th>
                <th>参数</th>
                <th>dtype</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3">5</td>
                <td>q_descale</td>
                <td>float8_e8m0</td>
            </tr>
            <tr>
                <td>k_descale</td>
                <td>float8_e8m0</td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>float8_e8m0</td>
            </tr>
            <tr>
                <td rowspan="3">3</td>
                <td>q_descale</td>
                <td>float8_e8m0</td>
            </tr>
            <tr>
                <td>k_descale</td>
                <td>float8_e8m0</td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>float8_e8m0</td>
            </tr>
        </tbody>
    </table>

- q_descale/k_descale/v_descale shape匹配关系表：<a name="descale_shape匹配关系表"></a>

    <table style="width:100%;table-layout:auto">
        <colgroup>
            <col style="width: 120px">
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 293px">
        </colgroup>
        <thead>
            <tr>
                <th>quant_mode</th>
                <th>参数</th>
                <th>layout</th>
                <th>shape</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3">5</td>
                <td>q_descale</td>
                <td>BNSD</td>
                <td>(B, Q_N, Q_S, D/64, 2)</td>
            </tr>
            <tr>
                <td>k_descale</td>
                <td>BNSD</td>
                <td>(B, KV_N, KV_S, D/64, 2)</td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>BNSD</td>
                <td>(B, KV_N, ceil(KV_S/64), D, 2)</td>
            </tr>
            <tr>
                <td rowspan="3">5（TND）</td>
                <td>q_descale</td>
                <td>N2TGD</td>
                <td>(KV_N, Q_T, G, D/64, 2)</td>
            </tr>
            <tr>
                <td>k_descale</td>
                <td>TND</td>
                <td>(KV_T, KV_N, D/64, 2)</td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>TND</td>
                <td>(sum(ceil(kv_len_i/64)), KV_N, D, 2)，kv_len_i 为第 i 个 batch 的 KV 有效长度</td>
            </tr>
            <tr>
                <td rowspan="2">5（PA_BNBD）</td>
                <td>k_descale</td>
                <td>PA_BNBD</td>
                <td>(Bn, KV_N, Bs, D/64, 2)</td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>PA_BNBD</td>
                <td>(Bn, KV_N, ceil(Bs/64), D, 2)</td>
            </tr>
            <tr>
                <td rowspan="2">5（PA_BBND）</td>
                <td>k_descale</td>
                <td>PA_BBND</td>
                <td>(Bn, Bs, KV_N, D/64, 2)</td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>PA_BBND</td>
                <td>(Bn, ceil(Bs/64), KV_N, D, 2)</td>
            </tr>
            <tr>
                <td rowspan="2">5（PA_NZ）</td>
                <td>k_descale</td>
                <td>PA_NZ</td>
                <td>(Bn, KV_N, ceil(Bs/16), 2, 16, 2)</td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>PA_NZ</td>
                <td>(Bn, KV_N, D/16, ceil(Bs/64), 16, 2)</td>
            </tr>
            <tr>
                <td rowspan="3">3</td>
                <td>q_descale</td>
                <td>BNSD</td>
                <td>(B, Q_N, Q_S, D/64, 2)</td>
            </tr>
            <tr>
                <td>k_descale</td>
                <td>BNSD</td>
                <td>(B, KV_N, KV_S, D/64, 2)</td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>BNSD</td>
                <td>(B, KV_N, ceil(KV_S/64), D, 2)</td>
            </tr>
        </tbody>
    </table>

    > [!NOTE]
    >
    > v_descale 的 S 维 group 数约定为 **ceil(KV_S / 64)**，其中 KV_S 取 k/v 的物理长度：BNSD 为 kv shape 中的 S 维；TND 为逐 batch 的物理打包长度 `cu_seqlens_kv[i+1]-cu_seqlens_kv[i]`，总 group 数为 `sum(ceil(packed_kv_len_i / 64))`。主算子 `quant_flash_attn` 仅校验 v_descale 的 dtype 与维数（4D/5D/6D），不做逐维 shape 校验，正确性由用户按上述约定构造保证。BNSD 非 PA 场景下主算子按有效 kv 长度计算 head 段距（有 `seqused_kv` 时取 `max(seqused_kv)`，否则取 `max_seqlen_kv`），若该有效长度小于物理 `KV_S`，group 维需按有效长度构造。
    >
    > - `q_descale` 的布局与 shape **仅由 `layout_q` 决定**，与 `layout_kv` 无关：`layout_q=BNSD` 时为 `(B, Q_N, Q_S, D/64, 2)`，`layout_q=TND` 时强制为 N2TGD `(KV_N, Q_T, G, D/64, 2)`（G = Q_N / KV_N，任意值含 G=1）。因此 `layout_q=TND` + PA 时仍按 N2TGD 构造。
    > - `k_descale`/`v_descale` 的布局与 shape 由 `layout_kv` 决定，与 `layout_q` 无关。
    > - descale 为 E8M0，每个 scale 占 1 字节，不做打包；`D/64` 为 MX 量化组数（每 32 个元素对应 1 个 scale），故 D=128 时 descale 末维为 2。

- q/k/v/attn_out shape匹配关系表： <a id="qkv_attn_out_shape匹配关系表"></a>

    <table style="width:100%;table-layout:auto">
        <colgroup>
            <col style="width: 120px">
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 293px">
        </colgroup>
        <thead>
            <tr>
                <th>quant_mode</th>
                <th>参数</th>
                <th>layout</th>
                <th>shape</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3">5</td>
                <td>q</td>
                <td>BNSD</td>
                <td>(B, Q_N, Q_S, D)</td>
            </tr>
            <tr>
                <td>k/v</td>
                <td>BNSD</td>
                <td>(B, KV_N, KV_S, D)</td>
            </tr>
            <tr>
                <td>attn_out</td>
                <td>BNSD</td>
                <td>(B, Q_N, Q_S, D)</td>
            </tr>
            <tr>
                <td rowspan="3">5（TND）</td>
                <td>q</td>
                <td>TND</td>
                <td>(Q_T, Q_N, D)</td>
            </tr>
            <tr>
                <td>k/v</td>
                <td>TND</td>
                <td>(KV_T, KV_N, D)</td>
            </tr>
            <tr>
                <td>attn_out</td>
                <td>TND</td>
                <td>(Q_T, Q_N, D)</td>
            </tr>
            <tr>
                <td rowspan="4">5（PA）</td>
                <td>q</td>
                <td>BNSD/TND</td>
                <td>BNSD: (B, Q_N, Q_S, D)；TND: (Q_T, Q_N, D)</td>
            </tr>
            <tr>
                <td>k</td>
                <td>PA_BNBD/PA_BBND/PA_NZ</td>
                <td>PA_BNBD: (Bn, KV_N, Bs, D)；PA_BBND: (Bn, Bs, KV_N, D)；PA_NZ: (Bn, KV_N, D/64, Bs, 64)</td>
            </tr>
            <tr>
                <td>v</td>
                <td>PA_BNBD/PA_BBND/PA_NZ</td>
                <td>同 k</td>
            </tr>
            <tr>
                <td>attn_out</td>
                <td>BNSD/TND</td>
                <td>与 layout_out 一致：BNSD (B, Q_N, Q_S, D)；TND (Q_T, Q_N, D)</td>
            </tr>
            <tr>
                <td rowspan="3">3</td>
                <td>q</td>
                <td>BNSD</td>
                <td>(B, Q_N, Q_S, D)</td>
            </tr>
            <tr>
                <td>k/v</td>
                <td>BNSD</td>
                <td>(B, KV_N, KV_S, D)</td>
            </tr>
            <tr>
                <td>attn_out</td>
                <td>BNSD</td>
                <td>(B, Q_N, Q_S, D)</td>
            </tr>
        </tbody>
    </table>

    > [!NOTE]
    >
    > - 表中 D 为量化后的逻辑元素数（MxFP4 为 fp4；q/k/v 以 `uint8` 打包传入时末维写 D/2，PA_NZ 末维 64 对应 32 Byte）。
    > - `Bn` 为块池总数（即 k/v tensor 第 0 维），`Bs` 为 block_size（仅支持 64/128/256/512/1024）。
    > - q/attn_out 的排布仅由 `layout_q`/`layout_out` 决定，与 `layout_kv` 是否为 PA 无关；q/k/v 末维必须一致（`vHeadDim` 与 `qkHeadDim` 均为 128）。

- MxFP4 TND（变长）补充说明:

    > [!NOTE]
    >
    > - B 由 cu_seqlens 推导：B = len(cu_seqlens_q) - 1，且 cu_seqlens_kv 的 B 必须与之相同；Q_T = q.shape[0]、KV_T = k.shape[0]，等于各 batch 有效长度之和。
    > - q_descale 的 N2TGD 布局第 0 维是 **KV_N**（不是 Q_N），第 1 维是 **Q_T**（按 token 连续展开，非 per-batch 分段），第 2 维为 G = Q_N / KV_N。易错点：不要按 (Q_N, Q_T, ...) 构造。
    > - k_descale 第 0 维为 KV_T；v_descale 第 0 维为 **Σ_i ceil(kv_len_i / 64)**（按 batch 分段累计），不等于 KV_T。
    > - S1G 合轴：tile 内 M 行按 t-major 排列（r = t*G + g），causal 可见性按该 batch 内的 q 行号判定；G 任意（含 G=1，此时退化为逐 token 平铺）。

- MxFP8 Softmax FP16 特殊约束:

    <table style="width:100%;table-layout:auto">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 293px">
        </colgroup>
        <thead>
            <tr>
                <th>约束项</th>
                <th>约束内容</th>
                <th>说明</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>q/k/v dtype</td>
                <td>仅支持 float8_e4m3fn</td>
                <td>PyTorch 原生支持该 dtype，直接构造输入，无需 uint8 伪装，D 维大小即实际 head_dim</td>
            </tr>
            <tr>
                <td>D</td>
                <td>仅支持 128</td>
                <td>MxFP8 Softmax FP16 仅支持 head_dim = 128</td>
            </tr>
            <tr>
                <td>排布</td>
                <td>仅支持 BNSD</td>
                <td>layout_q/layout_kv/layout_out 仅支持 BNSD，不支持 Paged Attention</td>
            </tr>
            <tr>
                <td>GQA</td>
                <td>支持</td>
                <td>支持 G = Q_N / KV_N（GQA）与 MHA（Q_N == KV_N）</td>
            </tr>
            <tr>
                <td>S1/S2</td>
                <td>任意长度</td>
                <td>S1/S2 支持任意值</td>
            </tr>
            <tr>
                <td>return_softmax_lse</td>
                <td>支持</td>
                <td>正常行输出 lse；padding 行（超过该 batch 有效长度的行）为 0；seqused_kv 为 0 的 batch，其有效行输出 -inf（softmax 空集，对应 log(0)）</td>
            </tr>
            <tr>
                <td>seqused</td>
                <td>支持 per-batch 变长</td>
                <td>支持 per-batch 变长（每个 batch 独立有效长度）；支持 seqused_q/seqused_kv 中个别 batch 取值为 0（该 batch 无有效 token）；张量 S 维大小取 max(seqused)</td>
            </tr>
        </tbody>
    </table>

- MxFP4 特殊约束:

    <table style="width:100%;table-layout:auto">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 293px">
        </colgroup>
        <thead>
            <tr>
                <th>约束项</th>
                <th>约束内容</th>
                <th>说明</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>GQA</td>
                <td>支持</td>
                <td>支持 G = Q_N / KV_N（GQA）与 MHA（Q_N == KV_N），要求 Q_N % KV_N == 0；TND/N2TGD 下按 S1×G 合轴（M 行 t-major，r = t*G + g），任意 G 均支持（含 G=1）</td>
            </tr>
            <tr>
                <td>q/k/v dtype</td>
                <td>float4_e2m1（4bit，打包存储）</td>
                <td>PyTorch 侧以 `uint8` 或 `float4_e2m1fn_x2` 传入（2 个 fp4 元素打包为 1 byte），末维写 D/2；本文档 shape 均按 fp4 逻辑元素 D 计。q、k、v 三者 dtype 必须一致</td>
            </tr>
            <tr>
                <td>D</td>
                <td>仅支持 128</td>
                <td>qkHeadDim 与 vHeadDim 均仅支持 128（通用层的 64 在 MxFP4 被拦截）</td>
            </tr>
            <tr>
                <td>排布</td>
                <td>支持 BNSD、TND、PA</td>
                <td>layout_q/layout_out 支持 BNSD、TND（两者需一致）；layout_kv 支持 BNSD、TND、PA_BNBD、PA_BBND、PA_NZ。q 与 kv 排布独立组合，即 TND 的 q 可与 PA 的 KV 组合使用</td>
            </tr>
            <tr>
                <td>layout_q_descale</td>
                <td>TND 时必须为 N2TGD，非 TND 时与 layout_q 一致</td>
                <td>TND 统一 N2TGD 5D (KV_N, Q_T, G, D/64, 2)，S1G 合轴，G 任意（G=1 含内）；BNSD 摊平，与 layout_q 一致；4D TND 形态将被拦截</td>
            </tr>
            <tr>
                <td>Paged Attention</td>
                <td>支持 PA_BNBD/PA_BBND/PA_NZ</td>
                <td>必须传入 block_table (B, maxBlockNumPerBatch) 与 seqused_kv；blockSize 仅支持 64/128/256/512/1024；S2 = maxBlockNumPerBatch * blockSize；Bn 取 k/v 第 0 维（块池总数）；非 PA 场景传入 block_table 将被拦截</td>
            </tr>
            <tr>
                <td>mask_mode</td>
                <td>支持 0（NO_MASK）、3（CAUSAL）</td>
                <td>mask_mode=0 不允许传入 attn_mask；mask_mode=3 必须传入 attn_mask（int8，固定 shape (2048, 2048)）；取值必须在 {0, 3, 4} 内，其中 4（SLIDING_WINDOW）在本场景不支持，传入将被拦截</td>
            </tr>
            <tr>
                <td>win_left/win_right</td>
                <td>取值 ≥ -1</td>
                <td>仅在 mask_mode=4 下生效；MxFP4 不支持 mask_mode=4，故该两项对 MxFP4 无实际作用（kernel 不消费）</td>
            </tr>
            <tr>
                <td>metadata</td>
                <td>必须传入</td>
                <td>须由 quant_flash_attn_metadata 以与主算子一致的入参生成；当前实现未传入将被拦截报错</td>
            </tr>
            <tr>
                <td>p_scale</td>
                <td>可选，仅校验 dtype(float32) 与 shape (1,)</td>
                <td>MxFP4 的 P 由 kernel 内部量化，p_scale 不被消费</td>
            </tr>
            <tr>
                <td>v_tail / sinks</td>
                <td>v_tail 三件套不消费；sinks 不支持</td>
                <td>v_tail、block_table_tail、seqused_v_tail 在 MxFP4 下不被消费；传入 sinks 将被拦截报错</td>
            </tr>
            <tr>
                <td>SeqLengths</td>
                <td>见下方 SeqLengths 参数组</td>
                <td>TND：cu_seqlens_q/cu_seqlens_kv 必传（int32，(B+1,)，首元素 0，末元素等于 Q_T/KV_T），seqused_* 与 max_seqlen_* 均可选；非 TND：禁止传 cu_seqlens_*，seqused_q 与 max_seqlen_q 至少传一个，seqused_kv 与 max_seqlen_kv 至少传一个（PA 场景必须传 seqused_kv）</td>
            </tr>
            <tr>
                <td>输出</td>
                <td>attn_out 为 bfloat16，softmax_lse 为 float32</td>
                <td>attn_out：BNSD (B, Q_N, Q_S, D) / TND (Q_T, Q_N, D)；softmax_lse 仅在 return_softmax_lse=True 时输出，TND 为 (Q_T, Q_N)（t-major），BNSD 为 (B, Q_N, Q_S)，关闭时输出空 Tensor</td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_mode参数解释（枚举定义见「mask_mode 枚举」）

<table style="width:100%;table-layout:auto">
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
                    <li>取值必须在0/3/4内，其他取值直接拦截</li>
                </ul>
            </td>
            <td>
                可选属性，默认值为0
            </td>
            <td rowspan="3">
                <ul>
                    <li>当mask_mode为0时，不支持传入attn_mask</li>
                    <li>当mask_mode为3或4时，必须传入attn_mask矩阵</li>
                </ul>
            </td>
            <td rowspan="3">
                <ul>
                    <li>MxFP4场景支持mask_mode=0（NO_MASK）与mask_mode=3（CAUSAL），传入其他值将被拦截报错</li>
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
        <tr>
            <td>win_left<br>win_right</td>
            <td>
                <ul>
                    <li>data_type支持int32</li>
                    <li>值需要 ≥ -1</li>
                </ul>
            </td>
            <td>
                可选属性，仅在mask_mode=4时生效
                <li>默认值为-1，表示无穷（极大值）</li>
            </td>
        </tr>
    </tbody>
</table>

> [!NOTE]
>
> - `attn_mask` 为 int8 的 (2048, 2048) 掩码矩阵，按 (s1, s2) 相对下标索引（不含 B 维）；TND 下按每个 batch 内的 (s1, s2) 相对位置取用。
> - mask_mode=3 为 causal 语义（已按 batch 内 q 行号对齐），仍需按上述 shape 传入 attn_mask；mask_mode=0 时不应传入。

#### SeqLengths参数组

<table style="width:100%;table-layout:auto">
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
                    <li>值仅支持非负整数</li>
                </ul>
            </td>
            <td rowspan="6">可选参数</td>
            <td rowspan="6">无</td>
            <td>
                <ul>
                    <li>当layout_q不为TND时，seqused_q与max_seqlen_q至少传入其中一个</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>seqused_kv</td>
            <td>
                <ul>
                    <li>当layout_kv不为TND且不为PA场景时，seqused_kv与max_seqlen_kv至少传入其中一个</li>
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
                    <li>当layout_q为TND时，必须传入（此时seqused_q与max_seqlen_q均为可选）</li>
                    <li>当layout_q不为TND时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_kv</td>
            <td>
                <ul>
                    <li>tensor_type支持int32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于KV_T</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_kv为TND时，必须传入（此时seqused_kv与max_seqlen_kv均为可选）</li>
                    <li>当layout_kv不为TND时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td>
                <ul>
                    <li>data_type支持int32</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q不为TND时，seqused_q与max_seqlen_q至少传入其中一个</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
            <td>
                <ul>
                    <li>data_type支持int32</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_kv不为TND且不为PA场景时，seqused_kv与max_seqlen_kv至少传入其中一个</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

> [!NOTE]
>
> 同时传入 `seqused_kv` 与 `max_seqlen_kv` 时，算子按 `max(seqused_kv)` 寻址（BNSD 场景的 v_descale 段距以它为准），`max_seqlen_kv` 仅在未传 `seqused_kv` 时生效；`max(seqused_kv) > max_seqlen_kv` 不报错，此时 `max_seqlen_kv` 被忽略。
>
> 由于 tiling 阶段无法读取 `seqused_q` / `seqused_kv` 的数值，以下约束算子不做校验，**正确性需用户自行保证**，违反时表现为静默精度错误或非法内存访问：
>
> - `seqused_q[i]` 不大于 Q 的物理序列长度（BNSD 为 `Q_S`；TND 为 `cu_seqlens_q[i+1]-cu_seqlens_q[i]`）。
> - `seqused_kv[i]` 不大于 K/V 的物理序列长度（BNSD 为 `KV_S`；TND 为 `cu_seqlens_kv[i+1]-cu_seqlens_kv[i]`）。
> - `max_seqlen_q` / `max_seqlen_kv` 取值为 `-1`（表示未传）或 `1 ~ 物理序列长度`（BNSD 为 shape 的 S 维，TND 为 `Q_T` / `KV_T`）；超过物理长度会多派任务，造成越界读写。
> - `max_seqlen_q` / `max_seqlen_kv` 传 `0` 不表示“未传”，会被当作长度为 0 处理，输出全 0。
> - 未传 `seqused_*` 时，`max_seqlen_*` 即所有 batch 的统一实际长度。
> - BNSD 非 PA 场景下，`v_descale` 的 S 维 group 数必须等于 `ceil(L/64)`，其中 `L` 为算子实际使用的 kv 长度（有 `seqused_kv` 时取 `max(seqused_kv)`，否则取 `max_seqlen_kv`）；建议 `max_seqlen_kv` 取物理 `KV_S`，TND 场景的构造约定见上方 descale 说明。

#### Paged Attention参数组

当 layout_kv 为 PA_BNBD/PA_BBND/PA_NZ 时，视为 Paged Attention 场景。

<table style="width:100%;table-layout:auto">
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
            <td>可选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>Paged Attention 场景下，block_table 必须不为空</li>
                    <li>Paged Attention 场景下，必须传入 seqused_kv</li>
                    <li>非 PA 场景（含 MxFP4 BNSD）中，block_table 不应传入，传入将被拦截</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Sinks参数组

<table style="width:100%;table-layout:auto">
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
            <td>sinks</td>
            <td>
                <ul>
                    <li>暂不支持</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>暂不支持，传入将被拦截报错</td>
        </tr>
    </tbody>
</table>

#### SoftmaxLSE参数组

<table style="width:100%;table-layout:auto">
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
                    <li>当return_softmax_lse为True时，softmax_lse必须非空，输出shape见<a href="#layout匹配关系表">layout匹配关系表</a></li>
                </ul>
            </td>
            <td>MxFP4（quant_mode=5）不支持开启，置True将被拦截报错</td>
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

- quant_flash_attn_metadata + quant_flash_attn 联合调用示例（BNSD，MxFP4场景）

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

    torch_npu.npu.set_device(0)

    B = 5
    N_q = 28
    N_kv = 28
    D = 128
    max_seqlen_q = 501
    max_seqlen_kv = 501

    # MxFP4 场景下 4 bit 元素以 uint8 存储，2 个元素打包为 1 byte
    # q/k/v shape 中 D 维实际大小为 D // 2 = 64
    q = torch.randn(B, N_q, max_seqlen_q, D // 2, dtype=torch.uint8, device="npu")
    k = torch.randn(B, N_kv, max_seqlen_kv, D // 2, dtype=torch.uint8, device="npu")
    v = torch.randn(B, N_kv, max_seqlen_kv, D // 2, dtype=torch.uint8, device="npu")

    # descale: q_descale/k_descale shape 为 (B, N, S, D/64, 2)
    q_descale = torch.randn(B, N_q, max_seqlen_q, D // 64, 2, dtype=torch.uint8, device="npu")
    k_descale = torch.randn(B, N_kv, max_seqlen_kv, D // 64, 2, dtype=torch.uint8, device="npu")
    # v_descale shape 为 (B, N, ceil(S/64), D, 2)
    v_descale = torch.randn(B, N_kv, (max_seqlen_kv + 63) // 64, D, 2, dtype=torch.uint8, device="npu")

    # 每 batch 实际使用的序列长度，截断冗余运算
    seqused_q = torch.tensor([501, 501, 501, 501, 501], dtype=torch.int32, device="npu")
    seqused_kv = torch.tensor([489, 501, 489, 489, 489], dtype=torch.int32, device="npu")

    metadata = quant_flash_attn_metadata(
        num_heads_q=N_q,
        num_heads_kv=N_kv,
        head_dim=D,
        quant_mode=5,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        v_descale=v_descale,
        batch_size=B,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BNSD",
        layout_q_descale="BNSD",
        layout_kv="BNSD",
        layout_out="BNSD",
    )

    attn_out, softmax_lse = quant_flash_attn(
        q, k, v,
        q_descale, k_descale, v_descale,
        quant_mode=5,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        metadata=metadata,
        softmax_scale=1.0 / (D ** 0.5),
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q="BNSD",
        layout_q_descale="BNSD",
        layout_kv="BNSD",
        layout_out="BNSD",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == (B, N_q, max_seqlen_q, D)
    assert attn_out.dtype == torch.bfloat16
    ```

- TND 排布示例（MxFP4场景，变长序列，仅传 cu_seqlens）

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

    torch_npu.npu.set_device(0)

    B = 3
    N_q = 4
    N_kv = 2
    D = 128
    # 每 batch 有效序列长度，T 总 token 数为其累加
    seq_lens_q = [128, 300, 45]
    seq_lens_kv = [512, 640, 100]
    Q_T = sum(seq_lens_q)
    KV_T = sum(seq_lens_kv)

    # TND: q shape 为 (Q_T, N_q, D//2)，k/v shape 为 (KV_T, N_kv, D//2)
    q = torch.randn(Q_T, N_q, D // 2, dtype=torch.uint8, device="npu")
    k = torch.randn(KV_T, N_kv, D // 2, dtype=torch.uint8, device="npu")
    v = torch.randn(KV_T, N_kv, D // 2, dtype=torch.uint8, device="npu")

    # descale: q_descale 为 N2TGD 5D (N_kv, Q_T, G, D/64, 2)；k_descale shape 为 (KV_T, N_kv, D/64, 2)
    q_descale = torch.randn(N_kv, Q_T, N_q // N_kv, D // 64, 2, dtype=torch.uint8, device="npu")
    k_descale = torch.randn(KV_T, N_kv, D // 64, 2, dtype=torch.uint8, device="npu")
    # v_descale 第 0 维为 sum(ceil(len/64))，len 为每 batch 的 KV 有效长度
    v_descale = torch.randn(sum((l + 63) // 64 for l in seq_lens_kv), N_kv, D, 2,
                            dtype=torch.uint8, device="npu")

    # cu_seqlens: TND 下必传（seqused/max_seqlen 此时可选）
    cu_seqlens_q = torch.tensor([0, 128, 428, 473], dtype=torch.int32, device="npu")
    cu_seqlens_kv = torch.tensor([0, 512, 1152, 1252], dtype=torch.int32, device="npu")

    metadata = quant_flash_attn_metadata(
        num_heads_q=N_q,
        num_heads_kv=N_kv,
        head_dim=D,
        quant_mode=5,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        v_descale=v_descale,
        batch_size=B,
        layout_q="TND",
        layout_q_descale="N2TGD",
        layout_kv="TND",
        layout_out="TND",
    )

    attn_out, softmax_lse = quant_flash_attn(
        q, k, v,
        q_descale, k_descale, v_descale,
        quant_mode=5,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        metadata=metadata,
        softmax_scale=1.0 / (D ** 0.5),
        layout_q="TND",
        layout_q_descale="N2TGD",
        layout_kv="TND",
        layout_out="TND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == (Q_T, N_q, D)
    assert attn_out.dtype == torch.bfloat16
    ```

- Paged Attention 示例（MxFP4场景，PA_BNBD，decode 形态）

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn

    torch_npu.npu.set_device(0)

    B = 2
    N_q = 4
    N_kv = 2
    D = 128
    block_size = 128              # 仅支持 64/128/256/512/1024
    max_block_num_per_batch = 4   # 即 Bn，S2 = 4 * 128 = 512
    Bn = 6                       # 块池总数，必须 ≥ 各 batch 实际占用块数之和

    # q 仍为 BNSD：decode 形态 S1 = 1
    q = torch.randn(B, N_q, 1, D // 2, dtype=torch.uint8, device="npu")
    # PA_BNBD: k/v shape 为 (Bn, KV_N, block_size, D/2)
    k = torch.randn(Bn, N_kv, block_size, D // 2, dtype=torch.uint8, device="npu")
    v = torch.randn(Bn, N_kv, block_size, D // 2, dtype=torch.uint8, device="npu")

    q_descale = torch.randn(B, N_q, 1, D // 64, 2, dtype=torch.uint8, device="npu")
    # k_descale: (Bn, KV_N, Bs, D/64, 2)；v_descale: (Bn, KV_N, ceil(Bs/64), D, 2)
    k_descale = torch.randn(Bn, N_kv, block_size, D // 64, 2, dtype=torch.uint8, device="npu")
    v_descale = torch.randn(Bn, N_kv, (block_size + 63) // 64, D, 2, dtype=torch.uint8, device="npu")

    block_table = torch.randint(0, Bn, (B, max_block_num_per_batch), dtype=torch.int32, device="npu")
    # PA 场景 seqused_kv 必传
    seqused_kv = torch.tensor([300, 480], dtype=torch.int32, device="npu")

    metadata = quant_flash_attn_metadata(
        num_heads_q=N_q,
        num_heads_kv=N_kv,
        head_dim=D,
        quant_mode=5,
        seqused_kv=seqused_kv,
        v_descale=v_descale,
        batch_size=B,
        max_seqlen_q=1,
        max_seqlen_kv=max_block_num_per_batch * block_size,
        layout_q="BNSD",
        layout_q_descale="BNSD",
        layout_kv="PA_BNBD",
        layout_out="BNSD",
    )

    attn_out, softmax_lse = quant_flash_attn(
        q, k, v,
        q_descale, k_descale, v_descale,
        quant_mode=5,
        block_table=block_table,
        seqused_kv=seqused_kv,
        metadata=metadata,
        softmax_scale=1.0 / (D ** 0.5),
        max_seqlen_q=1,
        max_seqlen_kv=max_block_num_per_batch * block_size,
        layout_q="BNSD",
        layout_q_descale="BNSD",
        layout_kv="PA_BNBD",
        layout_out="BNSD",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == (B, N_q, 1, D)
    assert attn_out.dtype == torch.bfloat16
    ```
