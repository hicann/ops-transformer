# quant_flash_attn（MxFP4）

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

- **接口功能**:

  `quant_flash_attn`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`QuantFlashAttn`算子完成 MxFP4 量化场景下的全量化注意力计算，训练推理归一化。当前支持 MxFP4 场景：

  - **MxFP4场景**（`quant_mode=5`）：Q/K/V 均采用 MxFP4，P 采用 MxFP4，Softmax 在 FP16 下计算。

  `quant_flash_attn_metadata`是`quant_flash_attn`的元数据生成接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子可选择传入该metadata以优化调度。典型调用流程如下：

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
> - MxFP4场景（`quant_mode=5`）下，仅支持 Q_N == KV_N（即 G=1，不支持 GQA），仅支持 D=128，仅支持 BNSD 排布。

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
| `A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32` | 1 | A8C8 Q/KV MxFP8，P FP8_E4M3 per-tensor，Softmax FP32 |
| `A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP32` | 2 | A8C8 Q/KV MxFP8，P MxFP8，Softmax FP32 |
| `A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16` | 3 | A8C8 Q/KV MxFP8，P FP8_E4M3 per-tensor，Softmax FP16 |
| `A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP16` | 4 | A8C8 Q/KV MxFP8，P MxFP8，Softmax FP16 |
| `A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16` | 5 | A4C4 Q/KV MxFP4，P MxFP4，Softmax FP16 |
| `A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32` | 6 | A8C8 Q/K FP8_E4M3 per-token-head、V FP8_E4M3 per-head、P FP8_E4M3 per-tensor，Softmax FP32 |
| `A8C8_QKV_HIF8_PER_TENSOR_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32` | 7 | A8C8 Q/KV HIF8 per-tensor，P FP8_E4M3 per-tensor，Softmax FP32 |
| `A4C4_QKV_HIF4_P_HIF4_LEVEL1_SOFTMAX_FP16` | 8 | A4C4 Q/KV HIF4，P HIF4 Level1，Softmax FP16 |
| `A4C4_QKV_HIF4_P_HIF4_LEVEL2_SOFTMAX_FP16` | 9 | A4C4 Q/KV HIF4，P HIF4 Level2，Softmax FP16 |
| `A4C4_QKV_HIF4_P_HIF4_LEVEL3_SOFTMAX_FP16` | 10 | A4C4 Q/KV HIF4，P HIF4 Level3，Softmax FP16 |

> [!NOTE]
>
> 当前仅实现 `quant_mode=5`（MxFP4）场景。

### mask_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `NO_MASK` | 0 | 全计算模式（默认值） |
| `CAUSAL` | 3 | Causal 模式 |
| `SLIDING_WINDOW` | 4 | Sliding Window 模式 |

> [!NOTE]
>
> MxFP4 场景（`quant_mode=5`）仅支持 mask_mode = 0（`NO_MASK`）。

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
| q | Tensor | 必选 | 公式中的Q | float4_e2m1（MxFP4场景） | ND | (B, Q_N, Q_S, D) | × |
| k | Tensor | 必选 | 公式中的K | float4_e2m1（MxFP4场景） | ND | (B, KV_N, KV_S, D) | × |
| v | Tensor | 必选 | 公式中的V | float4_e2m1（MxFP4场景） | ND | (B, KV_N, KV_S, D) | × |
| q_descale | Tensor | 必选 | q的反量化scale | float8_e8m0 | ND | (B, Q_N, Q_S, D/64, 2) | × |
| k_descale | Tensor | 必选 | k的反量化scale | float8_e8m0 | ND | (B, KV_N, KV_S, D/64, 2) | × |
| v_descale | Tensor | 必选 | v的反量化scale | float8_e8m0 | ND | - | × |
| quant_mode | int/QuantMode | 必选 | 量化模式，支持传入枚举或对应 int 值，枚举定义见「quant_mode 枚举」 | int32 | - | - | - |
| block_table | Tensor | 可选 | 用于分块注意力计算中的块索引映射 | int32 | ND | (B, Bn) | × |
| cu_seqlens_q | Tensor | 可选 | Q的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| cu_seqlens_kv | Tensor | 可选 | KV的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| seqused_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
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

## 返回值说明

### quant_flash_attn_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | quant_flash_attn的任务切分数据 | int32 | ND | shape根据batch_size和num_heads_kv动态计算 | x |

### quant_flash_attn

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | 必选 | quant_flash_attn的计算输出。 | bfloat16 | ND | (B, Q_N, Q_S, D) | × |
| softmax_lse | Tensor | 可选 | softmax的LSE结果。`return_softmax_lse`为True时，输出shape为(B, Q_N, Q_S)的Tensor；`return_softmax_lse`为False时，则输出空Tensor。 | float32 | ND | (B, Q_N, Q_S) | × |

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
            <td>支持BSND/BNSD/TND</td>
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

- quant_mode参数解释:

    <ul>
        <li>quant_mode=5，A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16（MxFP4场景）</li>
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
                <td>quant_mode</td>
                <td>
                    <ul>
                        <li>data_type支持int32</li>
                        <li>支持输入值为 A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16（5）</li>
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

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
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
    </tbody>
    </table>

- q/k/v descale dtype匹配关系表: <a name="descale_dtype匹配关系表"></a>

    <table style="undefined;table-layout: fixed; width:1625px">
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
        </tbody>
    </table>

- q_descale/k_descale/v_descale shape匹配关系表：<a name="descale_shape匹配关系表"></a>

    <table style="undefined;table-layout: fixed; width:1625px">
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
        </tbody>
    </table>

    > [!NOTE]
    >
    > v_descale 的逐维 shape 校验由 `quant_flash_attn_metadata` 执行：BNSD 场景下校验 S 维 group 数须等于 ceil(max(seqused_kv)/64)，其中 KV_S 取 max(seqused_kv)（未传入 seqused_kv 时取 max_seqlen_kv）；主算子 `quant_flash_attn` 仅校验 v_descale 的 dtype 与维数（4D/5D/6D），不做逐维 shape 校验。

- q/k/v/attn_out shape匹配关系表： <a id="qkv_attn_out_shape匹配关系表"></a>

    <table style="undefined;table-layout: fixed; width:1625px">
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
                <td rowspan="4">5</td>
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

- MxFP4 特殊约束:

    <table style="undefined;table-layout: fixed; width:1625px">
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
                <td>layout_q_descale</td>
                <td>仅支持 BNSD</td>
                <td>MxFP4 仅支持 layout_q_descale = BNSD</td>
            </tr>
            <tr>
                <td>GQA</td>
                <td>Q_N == KV_N（G=1）</td>
                <td>MxFP4 不支持 GQA，Q_N 必须等于 KV_N</td>
            </tr>
            <tr>
                <td>q/k/v dtype</td>
                <td>仅支持 float4_e2m1</td>
                <td>MxFP4 仅支持 Q/K/V 的 dtype 为 float4_e2m1</td>
            </tr>
            <tr>
                <td>D</td>
                <td>仅支持 128</td>
                <td>MxFP4 仅支持 head_dim = 128</td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_mode参数解释
<ul>
    <li>mask_mode=0，NO_MASK，全计算模式（默认值）</li>
    <li>mask_mode=3，CAUSAL，Causal模式</li>
    <li>mask_mode=4，SLIDING_WINDOW，Window模式</li>
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
                    <li>支持输入为0/3/4</li>
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
                    <li>MxFP4场景仅支持mask_mode=0（NO_MASK），传入其他值将被拦截报错</li>
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
                    <li>值仅支持非负整数</li>
                </ul>
            </td>
            <td rowspan="6">可选参数</td>
            <td rowspan="6">无</td>
            <td rowspan="2">
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
> 算子不校验 `seqused_kv` 中的最大值是否与 `max_seqlen_kv` 一致。若同时传入这两个参数，用户需自行保证 `max(seqused_kv) <= max_seqlen_kv`。

#### Paged Attention参数组

当 layout_kv 为 PA_BNBD/PA_BBND/PA_NZ 时，视为 Paged Attention 场景。

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
                    <li>当return_softmax_lse为True时，softmax_lse必须非空，输出shape见<a href="#layout匹配关系表">layout匹配关系表</a></li>
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
