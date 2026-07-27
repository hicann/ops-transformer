# flash_attn

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

  `flash_attn`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`FlashAttn`算子完成共享KV（Key和Value使用同一份输入）的非量化注意力计算，训练推理归一化。

  `flash_attn_metadata`是`flash_attn`的元数据生成接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子可选择传入该metadata以优化调度。典型调用流程如下：

  1. 准备`q`、`k`、`v`等输入。
  2. 调用`flash_attn_metadata`生成`metadata`。
  3. 调用`flash_attn`，将上一步得到的`metadata`传入主算子。

- **计算公式**：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n \times d$的矩阵。

  self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  本算子中Score函数采用Softmax函数，self-attention计算公式为:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$\sqrt{d}$进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n \times d$的矩阵。

  增加**sink**之后计算逻辑如下所示，主要修改相关softmax_max和softmax_sum逻辑计算部分。

  $$
  S = \frac{QK^T}{\sqrt{d}}
  $$

  $$
  m = max(sink, max(S))
  $$

  $$
  Attention = \frac{e^{S - m} \times V}{\sum e^{S-m} + e^{sink - m}}
  $$

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
> q、k、v数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、Q_T表示所有query Batch输入样本序列长度的累加和，KV_T表示所有k、v Batch输入样本序列长度的累加和。
> Q_S表示seqlen_q，Q_N表示nheads_q，KV_S表示seqlen_kv，KV_N表示nheads_kv。

## 函数原型

调用flash_attn接口之前，请先调用前置接口flash_attn_metadata，完成flash_attn负载均衡的计算。

```python
cann_ops_transformer.flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    mask_mode=None,
    win_left=None,
    win_right=None,
    layout_q=None,
    layout_kv=None,
    layout_out=None
) -> Tensor
```

```python
cann_ops_transformer.flash_attn(
    q,
    k,
    v,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    attn_mask=None,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## 参数说明

### flash_attn_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | 必选 | Query head数 | int32 | - | - |
| num_heads_kv | int | 必选 | Key/Value head数 | int32 | - | - |
| head_dim | int | 必选 | 每个注意力头的维度 | int32 | - | - |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,)
| cu_seqlens_kv | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,)
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,)
| seqused_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,)
| batch_size | int | 可选 | batch大小。若未传入，则从cu_seqlens_q或seqused_q推导。默认值为None | int32 | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - |
| mask_mode | int | 可选 | 掩码模式 | int32 | - | - |
| win_left | int | 可选 | window左界限 | int32 | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - |
| layout_kv | string | 可选 | 定义输入k和v张量的布局格式 | string | - | - |
| layout_out | string | 可选 | 定义输出张量的布局格式 | string | - | - |

### flash_attn

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的q | bfloat16/float16 | ND | <ul><li>(B, Q_S, Q_N, D)</li><li>(B, Q_N, Q_S, D)</li><li>(Q_T, Q_N, D)</li></ul> |
| k | Tensor | 必选 | 公式中的k | bfloat16/float16 | ND | <ul><li>(B, KV_S, KV_N, D)</li><li>(B, KV_N, KV_S, D)</li><li>(KV_T, KV_N, D)</li><li>(num_blocks, block_size, KV_N, D)</li><li>(num_blocks, KV_N, block_size, D)</li></ul> |
| v | Tensor | 必选 | 公式中的v | bfloat16/float16 | ND | <ul><li>(B, KV_S, KV_N, D)</li><li>(B, KV_N, KV_S, D)</li><li>(KV_T, KV_N, D)</li><li>(num_blocks, block_size, KV_N, D)</li><li>(num_blocks, KV_N, block_size, D)</li></ul> |
| block_table | Tensor | 可选 | 用于分块注意力计算中的块索引映射 | int32 | ND | (B, max_num_blocks_per_seq) |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) |
| seqused_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) |
| sinks | Tensor | 可选 | 注意力汇聚（sink）参数，用于改善自注意力计算的数值稳定性 | float32 | ND | (Q_N,) |
| metadata | Tensor | 可选 | `flash_attn_metadata`生成的任务切分结果，传入后可优化调度 | int32 | ND | (max_schedule_size,) |
| softmax_scale | float | 可选 | 可显式设置缩放因子，覆盖默认计算 | float32 | - | - |
| mask_mode | int | 可选 | 掩码模式 | int32 | - | - |
| attn_mask | Tensor | 可选 | 掩码矩阵 | int8 | ND | (2048, 2048)
| win_left | int | 可选 | window左界限 | int32 | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - |
| layout_kv | string | 可选 | 定义输入k和v张量的布局格式 | string | - | - |
| layout_out | string | 可选 | 定义输出张量的布局格式 | string | - | - |
| return_softmax_lse | bool | 可选 | 是否需要获取softmax的LSE结果 | bool | - | - |

## 返回值说明

### flash_attn_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | flash_attn的任务切分数据 | int32 | ND | shape根据batch_size和num_heads_kv动态计算 |

### flash_attn

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | 必选 | flash_attn的计算输出 | bfloat16/float16 | ND | <ul><li>(B, Q_S, Q_N, D)</li><li>(B, Q_N, Q_S, D)</li><li>(Q_T, Q_N, D )</li></ul> |
| softmax_lse | Tensor | 可选 | softmax的LSE结果 | float32 | ND | <ul><li>(B, Q_N, Q_S)</li><li>(Q_N, Q_T)</li></ul> |

**说明**

- attn_out：限制：该输出参数的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。
- softmax_lse：return_softmax_lse为True时，一般情况下，输出shape为(B, Q_N, Q_S)的Tensor，当input_q为TND时，输出shape为(Q_N, Q_T)的Tensor；return_softmax_lse为False时，则输出shape为[1]的值为0的Tensor。

## 约束说明

- 参数cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv、block_table及attn_mask属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- flash_attn_metadata和flash_attn的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |          k          |      INPUT      | Tensor |
|                      |         v         |      INPUT      | Tensor |
|                      |         metadata        |      INPUT(OPTIONAL)      | Tensor |
|                      |      softmax_scale      | ATTR(OPTIONAL) |   float   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |      layout_out      | ATTR(OPTIONAL) |   string   |
|                      |     attn_out     |     OUTPUT     |   Tensor   |
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
|                      |      softmax_lse      |     OUTPUT     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|    KV_N    |    输入k/v tensor的头数，对应k/v shape中的N    |
|     Q_S     |      输入q tensor的序列长度，对应q shape中的S      |
|    KV_S    |  输入k/v tensor的序列长度，对应k/v shape中的S  |
|     Q_T     |          输入q tensor所有batch序列长度的累加和          |
|     KV_T     |          输入k/v tensor所有batch序列长度的累加和          |
|     D     |          输入q/k/v tensor以及输出attn_out隐藏层最小的单元尺寸headdim         |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
  - 空Tensor指必选输入和输出的shape size为0,即有任意轴为0。
  - 触发空tensor的用例将全部拦截报错。

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
                <li>tensor_type支持bfloat16和float16</li>
                <li>BNSD -> (B, N_Q, S_Q, D)</li>
                <li>BSND -> (B, S_Q, N_Q, D)</li>
                <li>TND -> (T_Q, N_Q, D)</li>
            </ul>
        </td>
        <td rowspan="4">
            必须存在
        </td>
        <td rowspan="4">
            <ul>
                <li>q、k、v、attn_out的数据类型需相同</li>
                <li>k、v的shape需相同</li>
                <li>Layout校验规则见layout匹配关系表</li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>65536 > B > 0</li>
                <li>Q_T > 0</li>
                <li>KV_T > 0</li>
                <li>Q_N > 0</li>
                <li>KV_N > 0</li>
                <li>Q_S > 0</li>
                <li>KV_S > 0</li>
                <li>D支持64、128、256</li>
                <li>q与k/v的D维度必须相等</li>
                <li>Q_N % KV_N == 0且Q_N / KV_N > 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="2">
            <ul>
                <li>tensor_type支持bfloat16和float16</li>
                <li>BNSD -> (B, N_KV, S_KV, D)</li>
                <li>BSND -> (B, S_KV, N_KV, D)</li>
                <li>TND -> (T_KV, N_KV, D)</li>
                <li>PA_BBND -> (num_blocks, block_size, KV_N, D)</li>
                <li>PA_BNBD -> (num_blocks, KV_N, block_size, D)</li>
                <li>PA_NZ -> (num_blocks, KV_N, D//16, block_size, 16)</li>
                <li>1024 >= block_size >= 16，block_size % 16 == 0</li>
                <li>非连续Tensor约束：
                    <ul>
                        <li>BNSD/BSND/TND：k/v必须连续，传入非连续Tensor会报错</li>
                        <li>PA_BBND：仅dim0支持非连续</li>
                        <li>PA_BNBD/PA_NZ：dim0或dim1支持非连续</li>
                    </ul>
                </li>
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
                <li>data_type支持bfloat16和float16</li>
                <li>shape维度支持3、4</li>
            </ul>
        </td>
    </tr>
    </tbody>
    </table>

- layout匹配关系表：

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 132px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    <col style="width: 119px">
    <col style="width: 272px">
    <col style="width: 145px">
    </colgroup>
    <thead>
    <tr>
        <th>layout_q</th>
        <th>layout_kv</th>
        <th>layout_out</th>
        <th>layout_softmax_lse</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>BNSD</td>
            <td>
            <li>BNSD</li>
            <li>PA_BBND</li>
            <li>PA_BNBD</li>
            <li>PA_NZ</li>
            </td>
            <td>
            <li>BNSD</li>
            <li>BSND</li>
            </td>
            <td>(B, Q_N, Q_S)</td>
        </tr>
        <tr>
            <td>BSND</td>
            <td>
            <li>BSND</li>
            <li>PA_BBND</li>
            <li>PA_BNBD</li>
            <li>PA_NZ</li>
            </td>
            <td>BSND</td>
            <td>(B, Q_N, Q_S)</td>
        </tr>
        <tr>
            <td>TND</td>
            <td>
            <li>TND</li>
            <li>PA_BBND</li>
            <li>PA_BNBD</li>
            <li>PA_NZ</li>
            </td>
            <td>TND</td>
            <td>(Q_N, Q_T)</td>
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
                        <li>tensor_type仅支持int32</li>
                        <li>shape由flash_attn_metadata动态计算</li>
                        <li>当前不支持不传入，未传入将发出拦截报警</li>
                    </ul>
                </td>
                <td>可选参数</td>
                <td>无</td>
                <td>传入时需与flash_attn_metadata生成的结果一致</td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_mode参数解释
<ul>
    <li>mask_mode=0，全计算模式（默认值）</li>
    <li>mask_mode=3，Causal模式</li>
    <li>mask_mode=4，Window模式</li>
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
                    <li>data_type支持INT</li>
                    <li>支持输入范围仅为0、3、4，默认值为0</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为0
            </td>
            <td rowspan="4">
                <ul>
                    <li>当mask_mode为0时，win_left和win_right必须为-1以及不支持传入attn_mask</li>
                    <li>当mask_mode为3时，win_left和win_right必须为-1且必须传入attn_mask矩阵</li>
                    <li>当mask_mode为4时，必须传入attn_mask矩阵</li>
                </ul>
            </td>
            <td rowspan="4">
                <ul>
                    <li>无</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>
                <ul>
                    <li>tensor_type支持int8</li>
                    <li>tensor_shape维度为(2048, 2048)，该矩阵固定为2048*2048的下三角矩阵对角线全0</li>
                </ul>
            </td>
            <td rowspan="3">无</td>
        </tr>
        <tr>
            <td>win_left</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>不传或者传入-1,代表该值为正无穷</li>
                    <li>值需要 >= -1</li>
                </ul>
            </td>
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
            <td rowspan="2">可选参数</td>
            <td rowspan="4">无</td>
            <td rowspan="6">无</td>
        </tr>
        <tr>
            <td>seqused_kv</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持int32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列</li>
                    <li>cu_seqlens_q：第一个元素为0且最后一个元素等于Q_T</li>
                    <li>cu_seqlens_kv：第一个元素为0且最后一个元素等于KV_T</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>当layout_q为TND时，cu_seqlens_q必须传入</li>
                    <li>当layout_q不为TND时，cu_seqlens_q不支持传入</li>
                    <li>当layout_kv为TND时，cu_seqlens_kv必须传入</li>
                    <li>当layout_kv不为TND时，cu_seqlens_kv不支持传入</li>
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
            <td rowspan="2">
                <ul>
                    <li>可选参数</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>值必须大于等于-1；传入时必须等于实际的最大序列长度，否则行为未定义</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
        </tr>
        </tbody>
</table>

#### Paged Attention参数组

当block_table不为空时，开启Paged Attention
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
                    <li>1024 >= block_size >= 16，block_size % 16 == 0</li>
                </ul>
            </td>
            <td>可选参数</td>
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
            <td> 可选参数 </td>
            <td> 无 </td>
            <td> 暂不支持 </td>
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
                    <li>当return_softmax_lse为False时，输出shape为[1]的值为0的Tensor</li>
                    <li>当return_softmax_lse为True时，输出shape见layout匹配关系表</li>
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

- flash_attn_metadata + flash_attn联合调用示例（BSND）

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    dtype = torch.bfloat16
    B = 2
    S = 16
    Q_N = 8
    KV_N = 2
    D = 128

    q = torch.randn(B, S, Q_N, D, dtype=dtype, device="npu")
    k = torch.randn(B, S, KV_N, D, dtype=dtype, device="npu")
    v = torch.randn(B, S, KV_N, D, dtype=dtype, device="npu")

    metadata = cann_ops_transformer.ops.flash_attn_metadata(
        Q_N,
        KV_N,
        D,
        batch_size=B,
        max_seqlen_q=S,
        max_seqlen_kv=S,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
    )

    attn_out, softmax_lse = cann_ops_transformer.ops.flash_attn(
        q, k, v,
        metadata=metadata,
        softmax_scale=1.0 / (D ** 0.5),
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == q.shape
    assert attn_out.dtype == q.dtype
    assert torch.isfinite(attn_out.float()).all().item()
    ```
