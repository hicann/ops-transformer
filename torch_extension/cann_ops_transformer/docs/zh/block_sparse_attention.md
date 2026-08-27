# block_sparse_attention

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
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

> [!NOTE]
>
> <!-- npu="950" id10 -->
> <term>Ascend 950PR/Ascend 950DT</term>支持`quant_mode=0`（非量化的float16/bfloat16）、`quant_mode=1`（FP8量化）、`quant_mode=2/3`（MXFP4量化）。
> <!-- end id10 -->
> <!-- npu="910b,A3" id11 -->
> <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>仅支持`quant_mode=0`（非量化的float16/bfloat16输入）。
> <!-- end id11 -->

## 功能说明

- **接口功能**：

  `block_sparse_attention`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`BlockSparseAttention`算子完成块级稀疏注意力计算。通过`block_sparse_mask`指定每个Q块需要参与计算的KV块，实现高效的稀疏注意力计算，训练推理归一化。

  当前支持以下量化场景：

  - **非量化场景**（`quant_mode=0`，默认）：Q/K/V 均为 float16 或 bfloat16；
  - **FP8场景**（`quant_mode=1`）：Q/K/V 均为 float8_e4m3fn，携带 FP32 反量化缩放因子，输出反量化为 float16 或 bfloat16；
  - **MXFP4场景**（`quant_mode=2/3`）：Q/K/V 为 float4_e2m1fn_x2（以 uint8 传入），携带 float8_e8m0fnu descale 张量，输出反量化为 float16 或 bfloat16。

- **计算公式**：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n \times d$的矩阵。

  self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。稀疏块大小为$blockShapeX \times blockShapeY$，由`block_sparse_mask`指定稀疏模式。

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T + atten\_mask) \cdot value_{sparse}
  $$

  其中$scale$为缩放系数，一般取$scale = D^{-0.5}$，$key_{sparse}$、$value_{sparse}$为`block_sparse_mask`选取的KV块。

> [!NOTE]
>
> query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、T表示B和S合轴紧密排列的长度（Total tokens）。Q_S表示输入query tensor的序列长度，Q_N表示输入query tensor的头数，KV_S表示输入key/value tensor的序列长度，KV_N表示输入key/value tensor的头数。

## 函数原型

```python
cann_ops_transformer.block_sparse_attention(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    *,
    q_input_layout="TND",
    kv_input_layout="TND",
    num_key_value_heads=1,
    scale_value=1.0,
    inner_precise=1,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    return_softmax_lse=False,
    mask_type=0,
    quant_mode=0,
    block_size=0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    dst_type_max=0.0,
    atten_mask=None,
    block_table=None,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    p_quant_scale=None,
    attention_out_dtype=None
) -> (Tensor, Tensor)
```

> [!NOTE]
>
> `*`之后的所有参数仅支持以关键字方式传入。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | 公式中的query | float16/bfloat16/float8_e4m3fn/uint8 | ND | <ul><li>TND: [Tq, Nq, D]</li><li>BNSD: [B, Nq, Sq, D]</li><li>BSND: [B, Sq, Nq, D]</li></ul>（MXFP4量化时D为逻辑headDim，实际存储为uint8、形状末维为D//2） | × |
| key | Tensor | 必选 | 公式中的key | float16/bfloat16/float8_e4m3fn/uint8 | ND | <ul><li>TND: [Tkv, Nkv, D]</li><li>BNSD: [B, Nkv, Skv, D]</li><li>BSND: [B, Skv, Nkv, D]</li></ul>（MXFP4量化时D为逻辑headDim，实际存储为uint8、形状末维为D//2） | × |
| value | Tensor | 必选 | 公式中的value | float16/bfloat16/float8_e4m3fn/uint8 | ND | <ul><li>TND: [Tkv, Nkv, D]</li><li>BNSD: [B, Nkv, Skv, D]</li><li>BSND: [B, Skv, Nkv, D]</li></ul>（MXFP4量化时D为逻辑headDim，实际存储为uint8、形状末维为D//2） | × |
| block_sparse_mask | Tensor | 必选 | 块级稀疏mask，1表示保留该块参与计算，0表示跳过该块 | int8 | ND | [B, N, ceil(Sq/block_x), ceil(Skv/block_y)] | × |
| block_shape | list[int] | 必选 | 稀疏块形状，必须包含两个元素[block_x, block_y] | int64 | - | 长度为2 | - |
| q_input_layout | string | 可选 | 输入query张量的数据排布格式，默认"TND" | string | - | - | - |
| kv_input_layout | string | 可选 | 输入key/value张量的数据排布格式，默认"TND" | string | - | - | - |
| num_key_value_heads | int | 可选 | key/value的head个数，默认1 | int64 | - | - | - |
| scale_value | float | 可选 | 公式中的scale，默认1.0，一般设置为D^-0.5 | double | - | - | - |
| inner_precise | int | 可选 | Softmax计算采取的精度级别，默认1，支持0、1、4 | int64 | - | - | - |
| actual_seq_lengths | list[int] | TND时必选/其余可选 | 每个batch对应的query实际序列长度 | int64 | - | 长度为B | - |
| actual_seq_lengths_kv | list[int] | TND时必选/其余可选 | 每个batch对应的key/value实际序列长度 | int64 | - | 长度为B | - |
| return_softmax_lse | bool | 可选 | 是否使能softmaxLse输出，默认False | BOOL | - | - | - |
| mask_type | int | 可选 | attention计算中的掩码类型，默认0，当前仅支持0 | int64 | - | - | - |
| quant_mode | int | 可选 | 量化模式，默认0。取值含义：0-非量化；1-FP8量化，Q/K/V为float8_e4m3fn；2-MXFP4 OCP量化，量化scale向下截断；3-MXFP4 CX量化，自定义量化量程，量化scale向上截断。MXFP4量化公式见「约束说明」 | int64 | - | - | - |
| block_size | int | 可选 | PagedAttention的block大小，默认0，当前不支持PagedAttention，仅支持0 | int64 | - | - | - |
| pre_tokens | int | 可选 | 滑窗attention需要向前包含的token数，默认2147483647，当前仅支持2147483647 | int64 | - | - | - |
| next_tokens | int | 可选 | 滑窗attention需要向后包含的token数，默认2147483647，当前仅支持2147483647 | int64 | - | - | - |
| dst_type_max | float | 可选 | MXFP4 CX量化时传入的自定义量化量程，默认0.0 | double | - | - | - |
| atten_mask | Tensor | 可选 | 公式中的atten_mask，当前不支持，必须为None | int8 | ND | 2 | × |
| block_table | Tensor | 可选 | Block表，用于PagedAttention，当前不支持，必须为None | int32 | ND | 2 | × |
| q_dequant_scale | Tensor | 可选 | query的反量化缩放因子 | float32/float8_e8m0fnu | ND | 见「约束说明」 | × |
| k_dequant_scale | Tensor | 可选 | key的反量化缩放因子 | float32/float8_e8m0fnu | ND | 见「约束说明」 | × |
| v_dequant_scale | Tensor | 可选 | value的反量化缩放因子 | float32/float8_e8m0fnu | ND | 见「约束说明」 | × |
| p_quant_scale | Tensor | 可选 | softmax结果p的量化缩放因子，预留参数，当前不支持，必须为None | float32 | ND | - | × |
| attention_out_dtype | ScalarType | 可选 | attentionOut的数据类型，默认None。quant_mode=0时默认与query一致；quant_mode≠0时必选，仅支持float16/bfloat16 | - | - | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 必选 | 公式中的attentionOut | float16/bfloat16 | ND | 与query一致（MXFP4量化时，query以uint8伪装传入且存储维为D/2，attentionOut为反量化后的浮点输出，最后一维为逻辑headDim，即2*(D/2)=D） | × |
| softmax_lse | Tensor | 可选 | Softmax计算的log-sum-exp中间结果，`return_softmax_lse`为True时输出，否则输出空Tensor {0} | float32 | ND | <ul><li>TND: [Tq, Nq, 1]</li><li>BNSD: [B, Nq, Sq, 1]</li><li>BSND: [B, Sq, Nq, 1]</li></ul> | × |

## 约束说明

- q_input_layout与kv_input_layout必须保持一致，且仅支持"TND"、"BNSD"、"BSND"。
- query、key、value的数据类型必须一致，支持float16和bfloat16；FP8（`quant_mode=1`）时三者必须同时为float8_e4m3fn；MXFP4（`quant_mode=2/3`）时三者必须同时为uint8（float4_e2m1fn_x2伪装，每字节打包2个fp4数值）。
- query、key、value的D轴当前仅支持64或128。
- query的头数N1（由query的shape推断，接口不显式传入）与num_key_value_heads（N2）需满足 N1 >= N2 && N1 % N2 == 0。
- block_shape必须包含两个元素[block_x, block_y]，值必须大于0，各产品倍数约束如下：
  <!-- npu="950" id7 -->
  - 在<term>Ascend 950PR/Ascend 950DT</term>上：block_y须为16的倍数；MXFP4量化（`quant_mode=2/3`）时，block_x和block_y均只支持64的倍数。
  <!-- end id7 -->
  <!-- npu="910b,A3" id8 -->
  - 在<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>上：block_y须为128的倍数。
  <!-- end id8 -->
- block_sparse_mask当前必须传入，且shape必须为[batch, headNum, ceilDiv(maxQS, block_x), ceilDiv(maxKVS, block_y)]。
- actual_seq_lengths在q_input_layout为"TND"时必选；actual_seq_lengths_kv在kv_input_layout为"TND"时必选；两者必须同时配置或同时不配置，仅配置其中之一将被拦截。
- qSeqlen和kvSeqlen不需要被block_shape整除，支持非对齐场景，实际分块数通过向上取整计算。
- inner_precise仅支持配置0、1或4：
  - 0：online softmax和rescale全部采取fp32数据类型；
  - 1：仅支持query、key、value均为fp16时配置，全部采取fp16数据类型，性能更好但精度较低；
  - 4：混合精度运算，online softmax采取fp16/bf16，rescale采取fp32。
  <!-- npu="950,A3,910b" id9 -->
  - <term>Ascend 950PR/Ascend 950DT</term>仅支持配置为4；<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>仅支持配置为0或1。
  <!-- end id9 -->
- mask_type当前仅支持0，表示不加mask。
- block_size当前仅支持0，表示不支持paged cache。
- pre_tokens和next_tokens当前仅支持2147483647，表示当前token的前后所有token都参与attention运算，即不支持滑窗attention。
- atten_mask、block_table、p_quant_scale为预留参数，当前必须为None。
- quant_mode=0（非量化）时，q_dequant_scale、k_dequant_scale、v_dequant_scale必须为None，attention_out_dtype默认与query一致。
- quant_mode≠0（量化）时，attention_out_dtype必选，且仅支持float16或bfloat16。

### quant_mode=1（FP8量化）相关约束

- 仅<term>Ascend 950PR/Ascend 950DT</term>支持。
- query、key、value必须同时为float8_e4m3fn，且必须同时提供q_dequant_scale、k_dequant_scale、v_dequant_scale三个反量化缩放因子。
- 反量化缩放因子的数据类型必须为float32：
  - q_dequant_scale：shape为[Batch, HeadNum, ceilDiv(maxQSeqLength, 128), 1]，在QK矩阵乘法时对query进行反量化；
  - k_dequant_scale：shape为[Batch, KVHeadNum, ceilDiv(maxKVSeqLength, 256), 1]或[Batch, KVHeadNum, ceilDiv(maxKVSeqLength, 512), 1]，在QK矩阵乘法时对key进行反量化；
  - v_dequant_scale：shape与k_dequant_scale一致，在PV矩阵乘法时对value进行反量化。
- block_shape必须传入，且q、kv的量化块大小必须与block_shape的两个元素大小分别保持一致。

### quant_mode=2/3（MXFP4量化）相关约束

- 仅<term>Ascend 950PR/Ascend 950DT</term>支持。
- query、key、value为float4_e2m1fn_x2（以uint8伪装传入，D轴为uint8元素个数，逻辑headDim为2*D），需要提供float8_e8m0fnu类型的descale张量：
  - q_dequant_scale：MX量化模式，仅支持Rowwise E8M0 Micro Scaling，shape为：
    - BNSD: [B, Nq, Sq, ceilDiv(D, 64), 2]；
    - BSND: [B, Sq, Nq, ceilDiv(D, 64), 2]；
    - TND: [Tq, Nq, ceilDiv(D, 64), 2]。
  - k_dequant_scale：MX量化模式，仅支持Rowwise E8M0 Micro Scaling，shape为：
    - BNSD: [B, Nkv, Skv, ceilDiv(D, 64), 2]；
    - BSND: [B, Skv, Nkv, ceilDiv(D, 64), 2]；
    - TND: [Tkv, Nkv, ceilDiv(D, 64), 2]。
  - v_dequant_scale：MX量化模式，仅支持Columnwise E8M0 Micro Scaling，shape为：
    - BNSD: [B, Nkv, ceilDiv(Skv, 64), D, 2]；
    - BSND: [B, ceilDiv(Skv, 64), Nkv, D, 2]；
    - TND: [BTkv//64, Nkv, D, 2]，其中$BTkv = \sum_{b=0}^{B-1}\lceil\frac{S2_b}{64}\rceil$。
- attention_out_dtype必须为float16或bfloat16，attentionOut的最后一维为2*D（对应逻辑headDim，两个fp4数值打包进一个uint8字节）。
- block_x和block_y均须为64的倍数。
- dst_type_max可设置0.0或[6.0, 12.0]：0.0代表Amax(DType)为量化结果数据类型的最大值；6.0~12.0代表Amax(DType)为传入值。
- **量化公式**：以P（softmax中间结果）的量化为例，其中$P_{\text{max}}$为一个group（32个元素）内元素绝对值的最大值。
  - OCP量化公式（`quant_mode=2`，量化scale向下截断）：

    $$
    P_{\text{Scale}} = 2^{\lfloor \log_2 P_{\text{max}} \rfloor - 2}
    $$

  - CX量化公式（`quant_mode=3`，量化scale向上截断）：

    $$
    P_{\text{Scale}} = 2^{\lceil \log_2 \left( P_{\text{max}} / dst\_type\_max \right) \rceil} = 2^{\lceil \log_2 P_{\text{max}} - \log_2 dst\_type\_max \rceil}
    $$

- 暂不支持atten_mask、PagedAttention、softmaxLse等高阶特性。

## 调用示例

- 基础示例（TND布局，float16非量化）

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    B, Nq, Nkv, Sq, Skv, D = 1, 32, 32, 128, 128, 128
    block_x, block_y = 128, 128

    # TND排布，Tq/Tkv为各batch序列长度累加和
    Tq, Tkv = B * Sq, B * Skv
    query = torch.randn(Tq, Nq, D, dtype=torch.float16, device="npu")
    key = torch.randn(Tkv, Nkv, D, dtype=torch.float16, device="npu")
    value = torch.randn(Tkv, Nkv, D, dtype=torch.float16, device="npu")

    # block_sparse_mask: [B, N, ceil(Sq/block_x), ceil(Skv/block_y)]
    mask = torch.ones(B, Nq, Sq // block_x, Skv // block_y, dtype=torch.int8, device="npu")

    # block_shape / actual_seq_lengths 为int列表
    block_shape = [block_x, block_y]
    actual_seq_lengths = [Sq] * B
    actual_seq_lengths_kv = [Skv] * B

    attention_out, softmax_lse = cann_ops_transformer.block_sparse_attention(
        query, key, value, mask, block_shape,
        q_input_layout="TND",
        kv_input_layout="TND",
        num_key_value_heads=Nkv,
        scale_value=1.0 / (D ** 0.5),
        inner_precise=4,                 # 950PR/950DT 仅支持 4；不传(默认1)会直接崩溃
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        return_softmax_lse=True,
    )
    torch_npu.npu.synchronize()
    assert attention_out.shape == (Tq, Nq, D)
    assert softmax_lse.shape == (Tq, Nq, 1)
    ```

- MXFP4量化示例（TND布局，quant_mode=2）

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    # D为逻辑headDim；float4_e2m1fn_x2以uint8伪装传入，存储维为D//2
    B, Nq, Nkv, Sq, Skv, D = 1, 32, 32, 128, 128, 128
    block_x, block_y = 128, 128

    Tq, Tkv = B * Sq, B * Skv
    query = torch.randint(0, 256, (Tq, Nq, D // 2), dtype=torch.uint8, device="npu")
    key = torch.randint(0, 256, (Tkv, Nkv, D // 2), dtype=torch.uint8, device="npu")
    value = torch.randint(0, 256, (Tkv, Nkv, D // 2), dtype=torch.uint8, device="npu")

    mask = torch.ones(B, Nq, Sq // block_x, Skv // block_y, dtype=torch.int8, device="npu")
    block_shape = [block_x, block_y]
    actual_seq_lengths = [Sq] * B
    actual_seq_lengths_kv = [Skv] * B

    # descale为float8_e8m0fnu，以uint8伪装传入；维数按逻辑headDim D 计算：
    #   Q/K scale: [T, N, D//64, 2]；V scale: [BTkv, N, D, 2]，BTkv=B*ceilDiv(Skv,64)
    q_dequant_scale = torch.randint(0, 256, (Tq, Nq, D // 64, 2), dtype=torch.uint8, device="npu")
    k_dequant_scale = torch.randint(0, 256, (Tkv, Nkv, D // 64, 2), dtype=torch.uint8, device="npu")
    v_dequant_scale = torch.randint(0, 256, (Tkv // 64, Nkv, D, 2), dtype=torch.uint8, device="npu")

    attention_out, _ = cann_ops_transformer.block_sparse_attention(
        query, key, value, mask, block_shape,
        q_input_layout="TND",
        kv_input_layout="TND",
        num_key_value_heads=Nkv,
        scale_value=1.0 / (D ** 0.5),
        inner_precise=4,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        quant_mode=2,
        q_dequant_scale=q_dequant_scale,
        k_dequant_scale=k_dequant_scale,
        v_dequant_scale=v_dequant_scale,
        attention_out_dtype=torch.float16,
    )
    torch_npu.npu.synchronize()
    # MXFP4量化时输出为逻辑headDim D（Q/K/V存储维D//2，反量化输出D）
    assert attention_out.shape == (Tq, Nq, D)
    assert attention_out.dtype == torch.float16
    ```
