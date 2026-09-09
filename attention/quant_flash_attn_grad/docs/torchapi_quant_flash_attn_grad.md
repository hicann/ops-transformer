# quant_flash_attn_grad

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

  `quant_flash_attn_grad`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`QuantFlashAttnGrad`算子完成HIFLOAT8量化场景下的注意力反向梯度计算。该接口为`quant_flash_attn`正向算子的配套反向接口，用于计算Query、Key、Value的梯度（dq、dk、dv）以及sink梯度（dsink）。当前支持HIFLOAT8量化数据类型，支持BSND、BNSD两种数据排布格式。
   `quant_flash_attn_grad`的元数据生成接口复用`quant_flash_attn_metadata`，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子可选择传入该metadata以优化调度。典型调用流程如下：

  1. 准备`q`、`k`、`v`等输入。
  2. 调用`quant_flash_attn_metadata`生成`metadata`。
  3. 调用`quant_flash_attn_grad`，将上一步得到的`metadata`传入主算子。

- **计算公式**:

  量化注意力反向梯度计算分为以下几个阶段：

  阶段一：反量化输入数据，将量化输入转换到高精度浮点域

  $$
  Q_{fp} = Q_{quant} \times q\_descale
  $$

  $$
  K_{fp} = K_{quant} \times k\_descale
  $$

  $$
  V_{fp} = V_{quant} \times v\_descale
  $$

  $$
  dO_{fp} = dO_{quant} \times do\_descale
  $$

  阶段二：计算Softmax梯度（PreSfmg阶段）

  $$
  dP = dO \times V
  $$

  $$
  dS = dP \times p\_scale
  $$

  $$
  dS_{grad} = (dS - ds\_scale \times softmax(lse)) \times softmaxLse
  $$

  阶段三：计算Key/Value梯度（Cube主计算阶段）

  $$
  dV = dS_{grad}^{T} \times Q
  $$

  $$
  dK = dS_{grad}^{T} \times Q
  $$

  $$
  dQ = dS_{grad} \times K
  $$

  阶段四：后处理（Post阶段），将中间结果转换输出dtype

  $$
  dq = dQ \times softmax\_scale
  $$

  $$
  dk = dK \times softmax\_scale
  $$

  $$
  dv = dV \times softmax\_scale
  $$

  $$
  dsink = reduce(dS_{grad}, dim=S)
  $$

  其中，$q\_descale$、$k\_descale$、$v\_descale$、$do\_descale$为量化缩放因子，$p\_scale$为P矩阵的缩放因子，$ds\_scale$为反量化缩放因子，$softmax\_scale$为缩放系数（建议值为$\sqrt{D}$的倒数）。

> [!NOTE]
>
> Q、K、V数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim。Q_S表示输入q tensor的序列长度，Q_N表示输入q tensor的头数，KV_S表示输入k/v tensor的序列长度，KV_N表示输入k/v tensor的头数。

## 函数原型
调用quant_flash_attn_grad接口之前，请先调用前置接口quant_flash_attn_metadata，完成quant_flash_attn_grad负载均衡的计算。

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
    layout_out="BSND",
    is_grad_enabled=True
) -> Tensor
```

```python
cann_ops_transformer.quant_flash_attn_grad(
    q,
    k,
    v,
    dout,
    attn_out,
    q_descale,
    k_descale,
    v_descale,
    do_descale,
    p_scale,
    ds_scale,
    softmax_lse,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    quant_mode=0,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="BSND"
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 枚举说明

`quant_mode` 与 `mask_mode` 在 Python 接口中支持传入 `IntEnum` 枚举或对应 int 值，枚举定义于 `cann_ops_transformer.ops.quant_flash_attn_grad`：

### quant_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `HIF8_PER_TENSOR` | 0 | HIFLOAT8 per-tensor 量化 |

### mask_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `ALL` | 0 | 全计算模式（默认值） |
| `CAUSAL` | 3 | Causal 模式 |
| `WINDOW` | 4 | Sliding Window 模式 |

> [!NOTE]
>
> 枚举为 `IntEnum`，可直接作为 int 传入底层算子；接口仅支持传入枚举或对应 int 值。当前版本仅支持 mask_mode = 0（`ALL`），其他模式暂不支持。

## 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|    KV_N    |    输入k/v tensor的头数，对应k/v shape中的N    |
|     Q_S     |      输入q tensor的序列长度，对应q shape中的S      |
|    KV_S    |  输入k/v tensor的序列长度，对应k/v shape中的S  |
|     D     |          输入q/k/v tensor以及输出dq/dk/dv隐藏层最小的单元尺寸headdim         |

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的Q，量化数据 | uint8 | ND | BSND: (B, Q_S, Q_N, D)<br>BNSD: (B, Q_N, Q_S, D) | × |
| k | Tensor | 必选 | 公式中的K，量化数据 | uint8 | ND | BSND: (B, KV_S, KV_N, D)<br>BNSD: (B, KV_N, KV_S, D) | × |
| v | Tensor | 必选 | 公式中的V，量化数据 | uint8 | ND | BSND: (B, KV_S, KV_N, D)<br>BNSD: (B, KV_N, KV_S, D) | × |
| dout | Tensor | 必选 | 正向输出attn_out对应的梯度，量化数据 | uint8 | ND | 与q的shape一致 | × |
| attn_out | Tensor | 必选 | 正向计算输出的attn_out | bfloat16 | ND | 与q的shape一致 | × |
| q_descale | Tensor | 必选 | q的反量化缩放因子 | float32 | ND | (1,) | × |
| k_descale | Tensor | 必选 | k的反量化缩放因子 | float32 | ND | (1,) | × |
| v_descale | Tensor | 必选 | v的反量化缩放因子 | float32 | ND | (1,) | × |
| do_descale | Tensor | 必选 | dout的反量化缩放因子 | float32 | ND | (1,) | × |
| p_scale | Tensor | 必选 | P矩阵的量化缩放因子 | float32 | ND | (1,) | × |
| ds_scale | Tensor | 必选 | 反量化缩放因子 | float32 | ND | (1,) | × |
| softmax_lse | Tensor | 必选 | 注意力正向计算的输出softmaxLse | float32 | ND | (B, Q_N, Q_S, 1) | × |
| cu_seqlens_q | Tensor | 可选 | Q的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| cu_seqlens_kv | Tensor | 可选 | KV的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| seqused_q | Tensor | 可选 | 指定每batch中q实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| seqused_kv | Tensor | 可选 | 指定每batch中kv实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| sinks | Tensor | 可选 | sink场景下的输入tensor。当前版本不支持，传None即可 | float32 | ND | (Q_N,) | × |
| attn_mask | Tensor | 可选 | 掩码矩阵 | int8/uint8/bool | ND | (2048, 2048) | × |
| metadata | Tensor | 可选 | tiling下沉的aicpu算子输出结果 | int32 | ND | (2, max_schedule_size) | × |
| quant_mode | int/QuantMode | 必选 | 量化模式，支持传入枚举或对应 int 值，枚举定义见「quant_mode 枚举」 | int32 | - | - | - |
| softmax_scale | float | 可选 | 缩放系数，默认值为1.0。推荐值：sqrt(head_dim)的倒数 | float32 | - | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」。当前版本仅支持0 | int32 | - | - | - |
| win_left | int | 可选 | window左界限，默认值为-1 | int32 | - | - | - |
| win_right | int | 可选 | window右界限，默认值为-1 | int32 | - | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限，-1表示自动推导。默认值为-1 | int32 | - | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限，-1表示自动推导。默认值为-1 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式，支持"BSND"、"BNSD"，默认值为"BSND" | string | - | - | - |
| layout_kv | string | 可选 | 定义输入k/v张量的布局格式，支持"BSND"、"BNSD"，默认值为"BSND" | string | - | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| dq | Tensor | 必选 | Query的梯度 | bfloat16 | ND | BSND: (B, Q_S, Q_N, D)<br>BNSD: (B, Q_N, Q_S, D) | × |
| dk | Tensor | 必选 | Key的梯度 | bfloat16 | ND | BSND: (B, KV_S, KV_N, D)<br>BNSD: (B, KV_N, KV_S, D) | × |
| dv | Tensor | 必选 | Value的梯度 | bfloat16 | ND | BSND: (B, KV_S, KV_N, D)<br>BNSD: (B, KV_N, KV_S, D) | × |
| dsink | Tensor | 必选 | sink的梯度 | float32 | ND | (Q_N,) | × |

## 约束说明

- 确定性说明：quant_flash_attn_grad默认确定性实现。
- 入参为空处理：q为空Tensor时直接返回。
- 仅支持BSND或BNSD layout，且layout_q与layout_kv必须保持一致。
- 参数cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv、attn_mask属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。
- quant_flash_attn_metadata和quant_flash_attn_grad的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |          k          |      INPUT      | Tensor |
|                      |         v         |      INPUT      | Tensor |
|                      |         dout         |      INPUT      | Tensor |
|                      |         attn_out         |      INPUT      | Tensor |
|                      |         metadata        |      INPUT(OPTIONAL)      | Tensor |
|                      |      softmax_scale      | ATTR(OPTIONAL) |   float   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |     dq     |     OUTPUT     |   Tensor   |
|                      |     dk     |     OUTPUT     |   Tensor   |
|                      |     dv     |     OUTPUT     |   Tensor   |
|      全量化参数组      |       quant_mode       | ATTR |   int   |
|                      |       q_descale       | INPUT |   Tensor   |
|                      |       k_descale       | INPUT |   Tensor   |
|                      |       v_descale       | INPUT |   Tensor   |
|                      |       do_descale       | INPUT |   Tensor   |
|                      |       p_scale       | INPUT |   Tensor   |
|                      |       ds_scale       | INPUT |   Tensor   |
|                      |       softmax_lse       | INPUT |   Tensor   |
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
|  Sinks参数组  |     sinks     | INPUT(OPTIONAL) |   Tensor   |
|   DSink输出参数组   |    dsink    |     OUTPUT     |    Tensor    |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
  - 空Tensor指必选输入和输出的shape size为0，即有任意轴为0。
  - 触发空Tensor的用例将全部拦截报错。

- q、k、v、dout、attn_out校验:

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
                    <li>tensor_type仅支持uint8</li>
                    <li>shape dim仅支持4</li>
                </ul>
            </td>
            <td rowspan="5">
                必须存在
            </td>
            <td rowspan="5">
                <ul>
                    <li>q、k、v、dout的数据类型必须相同（均为uint8）</li>
                    <li>q与dout、attn_out的shape必须一致</li>
                    <li>k与v的shape必须一致</li>
                </ul>
            </td>
            <td rowspan="5">
                轴校验：
                <ul>
                    <li>65536 > B > 0</li>
                    <li>Q_S ≥ 0；KV_S ≥ 0</li>
                    <li>D仅支持128</li>
                    <li>Q_N % KV_N == 0且Q_N / KV_N > 0（GQA约束）</li>
                    <li>Q_N ≤ 128；KV_N ≤ 128</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>k</td>
            <td>
                <ul>
                    <li>tensor_type仅支持uint8</li>
                    <li>shape dim仅支持4</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>v</td>
            <td>
                <ul>
                    <li>tensor_type仅支持uint8</li>
                    <li>shape dim仅支持4</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>dout</td>
            <td>
                <ul>
                    <li>tensor_type仅支持uint8</li>
                    <li>shape dim仅支持4</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_out</td>
            <td>
                <ul>
                    <li>data_type仅支持bfloat16</li>
                    <li>shape dim仅支持4</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>layout_q</td>
            <td>支持BSND/BNSD</td>
            <td rowspan="2">当前不支持不传入，未传入将发出拦截报警</td>
            <td rowspan="2">layout_q与layout_kv必须一致</td>
            <td rowspan="2">无</td>
        </tr>
        <tr>
            <td>layout_kv</td>
            <td>支持BSND/BNSD</td>
        </tr>
        <tr>
            <td>metadata</td>
            <td>
                <ul>
                    <li>tensor_type仅支持int32</li>
                    <li>shape为(2, max_schedule_size)</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
    </table>

#### 全量化参数组

- quant_mode参数解释:

    <ul>
        <li>quant_mode=0，HIF8_PER_TENSOR（HIFLOAT8 per-tensor量化场景）</li>
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
                        <li>支持输入范围为0</li>
                    </ul>
                </td>
                <td>必选属性</td>
                <td>无</td>
                <td rowspan="8">
                    <ul>
                        <li>不支持非连续Tensor</li>
                        <li>D固定为128</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>q_descale</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持float32</li>
                        <li>shape仅支持(1,)</li>
                    </ul>
                </td>
                <td rowspan="7">必须存在</td>
                <td rowspan="7">
                    <ul>
                        <li>所有descale/scale的dtype必须为float32</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>k_descale</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持float32</li>
                        <li>shape仅支持(1,)</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>v_descale</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持float32</li>
                        <li>shape仅支持(1,)</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>do_descale</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持float32</li>
                        <li>shape仅支持(1,)</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>p_scale</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持float32</li>
                        <li>shape仅支持(1,)</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>ds_scale</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持float32</li>
                        <li>shape仅支持(1,)</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>softmax_lse</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持float32</li>
                        <li>shape为(B, Q_N, Q_S, 1)</li>
                    </ul>
                </td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_mode参数解释
<ul>
    <li>mask_mode=0，ALL，全计算模式（默认值）</li>
    <li>mask_mode=3，CAUSAL，Causal模式</li>
    <li>mask_mode=4，WINDOW，Window模式</li>
</ul>

> [!NOTE]
>
> 当前版本仅支持mask_mode=0（`ALL`），其他模式暂不支持。

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
                    <li>当前版本仅支持输入为0</li>
                </ul>
            </td>
            <td>
                可选属性，默认值为0
            </td>
            <td rowspan="3">
                <ul>
                    <li>当mask_mode为0时，不支持传入attn_mask</li>
                </ul>
            </td>
            <td rowspan="3">
                <ul>
                    <li>当前版本仅支持mask_mode=0，其他模式暂不支持</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>
                <ul>
                    <li>tensor_type支持int8/uint8/bool</li>
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
                    <li>seqused_q中的值需小于等于Q_S</li>
                    <li>seqused_kv中的值需小于等于KV_S</li>
                </ul>
            </td>
            <td rowspan="6">可选参数</td>
            <td rowspan="6">无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>seqused_kv</td>
            <td>无</td>
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
            <td>无</td>
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
            <td>无</td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td>
                <ul>
                    <li>data_type支持int32</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
            <td>
                <ul>
                    <li>data_type支持int32</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td>无</td>
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
                    <li>当前版本不支持</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>当前版本不支持，传None即可</td>
        </tr>
    </tbody>
</table>

## 调用示例

- quant_flash_attn_grad调用示例（BSND layout，HIFLOAT8量化场景）

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch_npu.npu.set_device(0)

    B = 1
    Q_S = 512
    KV_S = 512
    Q_N = 1
    KV_N = 1
    D = 128

    # BSND layout
    q = torch.zeros(B, Q_S, Q_N, D, dtype=torch.uint8, device="npu")
    k = torch.zeros(B, KV_S, KV_N, D, dtype=torch.uint8, device="npu")
    v = torch.zeros(B, KV_S, KV_N, D, dtype=torch.uint8, device="npu")
    dout = torch.zeros(B, Q_S, Q_N, D, dtype=torch.uint8, device="npu")
    attn_out = torch.zeros(B, Q_S, Q_N, D, dtype=torch.bfloat16, device="npu")

    # descale / scale: FP32, shape=(1,)
    q_descale = torch.ones(1, dtype=torch.float32, device="npu")
    k_descale = torch.ones(1, dtype=torch.float32, device="npu")
    v_descale = torch.ones(1, dtype=torch.float32, device="npu")
    do_descale = torch.ones(1, dtype=torch.float32, device="npu")
    p_scale = torch.ones(1, dtype=torch.float32, device="npu")
    ds_scale = torch.ones(1, dtype=torch.float32, device="npu")

    # softmax_lse: FP32, shape=(B, Q_N, Q_S, 1)
    softmax_lse = torch.zeros(B, Q_N, Q_S, 1, dtype=torch.float32, device="npu")

    try:
        metadata = cann_ops_transformer.quant_flash_attn_metadata(
            num_heads_q=Q_N,
            num_heads_kv=KV_N,
            head_dim=D,
            quant_mode=0,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=None,
            batch_size=B,
            max_seqlen_q=Q_S,
            max_seqlen_kv=KV_S,
            mask_mode=0,
            layout_q="BSND",
            layout_kv="BSND",
            is_grad_enabled=True
        )
    except Exception as e:
        logger.error("[MAIN_WRAPPER] quant_flash_attn_metadata 重建失败: %s", str(e))
        raise

    dq, dk, dv, dsink = cann_ops_transformer.ops.quant_flash_attn_grad(
        q, k, v, dout, attn_out,
        q_descale, k_descale, v_descale, do_descale, p_scale, ds_scale, softmax_lse,
        metadata=metadata,
        quant_mode=0,
        softmax_scale=1.0 / (D ** 0.5),
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=Q_S,
        max_seqlen_kv=KV_S,
        layout_q="BSND",
        layout_kv="BSND",
    )
    torch_npu.npu.synchronize()
    assert dq.shape == (B, Q_S, Q_N, D)
    assert dk.shape == (B, KV_S, KV_N, D)
    assert dv.shape == (B, KV_S, KV_N, D)
    assert dq.dtype == torch.bfloat16
    ```
