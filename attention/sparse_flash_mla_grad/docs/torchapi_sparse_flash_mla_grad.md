# sparse_flash_mla_grad

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

## 功能说明

- **接口功能**：

  - `sparse_flash_mla_grad`：计算`SparseFlashMla`训练场景下注意力的反向输出，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。
  - `sparse_flash_mla_grad_metadata`：接口用于生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及Q和K的分块的索引，供后续`sparse_flash_mla_grad`算子使用。

- **计算公式**：

    阶段一：根据不同cmp_ratio场景，对输入ori_kv与cmp_kv进行选择

  - 当cmp_ratio = 1 (SWA)：

    $$
    selectedKv\text{ }=\text{ }orikv
    $$

  - 当cmp_ratio = 4 (SCFA)：

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }Gather \left( cmpkv,topkIndices \left[ i \left]  \left)) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

  - else (CFA):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }cmpkv)
    $$

    阶段二：计算P、dP、dS

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), lse)
    $$

    $$
    dP = dO \text{ }@\text{ } selectedKv^{{T}}
    $$

    $$
    dS = P \times (dP\text{ } -\text{ } SoftmaxGrad(dO, O))
    $$

    阶段三：计算dQ, dKV, dSinks

    $$
    dQ = dS \text{ } @ \text{ } selectedKv \text{ }  \cdot \text{ } scale
    $$

    $$
    dKV = dS^{{T}} \text{ } @ \text{ } Q \text{ } \cdot \text{ } scale + P^{{T}}  @ \text{ } dO
    $$

    $$
    dSinks = ReduceSum(-P \text{ }\times\text{ } dP \text{ }\times\text{ } SimpleSoftmax(sinks, lse), dim=-1)
    $$

## 函数原型

```python
cann_ops_transformer.sparse_flash_mla_grad_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *，
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    ori_topk=None,
    cmp_topk=None,
    cmp_ratio=None,
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
cann_ops_transformer.sparse_flash_mla_grad(
    q,
    dout,
    attn_out,
    softmax_lse,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None, metadata=None,
    softmax_scale=None,
    cmp_ratio=None,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND"
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

### sparse_flash_mla_grad

- **q**（`Tensor`）：必选参数，对应公式中的$Q$。`layout_q`="BSND" 时shape为 `[B, S1, N1, D]`；`layout_q`="TND" 时shape为 `[T1, N1, D]`。B：支持泛化；S1：支持泛化；D：512；T1：B × S1。支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`。

- **dout**（`Tensor`）：必选参数，注意力正向输出矩阵的梯度，对应公式中的$dO$。数据类型和shape均与q保持一致。支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`。

- **attn_out**（`Tensor`）：必选参数，注意力正向输出矩阵，对应公式中的$O$。数据类型和shape均与q保持一致。支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`。

- **softmax_lse**（`Tensor`）：必选参数，注意力正向计算的输出lse。`layout_q`="BSND" 时shape为 `[B, N2, S1, G]`；`layout_q`="TND" 时shape为 `[N2, T1, G]`。B：与q的B保持一致；N2：1；S1：与q的S1 保持一致；G：N1/N2；T1：B × S1。支持非连续，数据格式支持ND，数据类型支持`float32`。

- **ori_kv**（`Tensor`）：可选参数，对应公式中的$oriKv$。`layout_kv`="BSND" 时shape为 `[B, S2, N2, D]`；`layout_kv`="TND" 时shape为 `[T2, N2, D]`。B：与q的B保持一致；S2：支持泛化；N2：1；D：512；T2：B × S2。支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`。

- **cmp_kv**（`Tensor`）：可选参数，对应公式中的$cmpkv$。`layout_kv`="BSND" 时shape为 `[B, S3, N2, D]`；`layout_kv`="TND" 时shape为 `[T3, N2, D]`。B：与q的B保持一致；S3：支持泛化；N2：1；D：512；T3：B × S3。传None时按SWA场景计算。支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`。

- **ori_sparse_indices**（`Tensor`）：可选参数，对应oriKv部分的topk索引。`layout_q`="BSND" 时shape为 `[B, S1, N2, K1]`；`layout_q`="TND" 时shape为 `[T1, N2, K1]`。B：与q的B保持一致；S1：与q的S1 保持一致；N2：1；K1：支持泛化；T1：B × S1。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **cmp_sparse_indices**（`Tensor`）：可选参数，对应公式中的$topkIndices$。`layout_q`="BSND" 时shape为 `[B, S1, N2, K2]`；`layout_q`="TND" 时shape为 `[T1, N2, K2]`。B：与q的B保持一致；S1：与q的S1 保持一致；N2：1；K2：支持泛化；T1：B × S1。若cmp_kv不为None，此时cmp_sparse_indices不为None时按SCFA场景计算，为None时按CFA场景计算；若cmp_kv为None，则cmp_sparse_indices只能为None，此时按SWA场景计算。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **cu_seqlens_q**（`Tensor`）：可选参数，每个Batch中q的有效token数的累加和形式。`layout_q`="TND" 时必传。shape为 `[B+1]`，累加和与T1 保持一致。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **cu_seqlens_ori_kv**（`Tensor`）：可选参数，每个Batch中ori_kv的有效token数的累加和形式。`layout_kv`="TND" 时必传。shape为 `[B+1]`，累加和与T2 保持一致。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **cu_seqlens_cmp_kv**（`Tensor`）：可选参数，每个Batch中cmp_kv的有效token数的累加和形式。`layout_kv`="TND" 时必传。shape为 `[B+1]`，累加和与T3 保持一致。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **seqused_q**（`Tensor`）：可选参数，表示不同batch中query实际参与运算的token数。shape为 `[B]`。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **seqused_ori_kv**（`Tensor`）：可选参数，表示不同batch中ori_kv实际参与运算的token数。shape为 `[B]`。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **seqused_cmp_kv**（`Tensor`）：可选参数，表示不同batch中cmp_kv实际参与运算的token数。shape为 `[B]`。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **cmp_residual_kv**（`Tensor`）：可选参数，表示每个batch S2 // cmpRatio后的余数。shape为 `[B]`。当cmp_kv不为空且cmp_mask_mode=3 时必须传入。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **ori_topk_length**（`Tensor`）：可选参数，表示每行query对应的ori_kv实际可选的topk长度。shape为 `[B, S1, N2]`（BSND）或 `[T1, N2]`（TND）。当ori_mask_mode=0 且ori_sparse_indices不为None时必须传入且必须为准确值。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **cmp_topk_length**（`Tensor`）：可选参数，表示每行query对应的cmp_kv实际可选的topk长度。shape为 `[B, S1, N2]`（BSND）或 `[T1, N2]`（TND）。当cmp_mask_mode=0 且cmp_sparse_indices不为None时必须传入且必须为准确值。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **sinks**（`Tensor`）：可选参数，注意力下沉tensor。shape为 `[N1]`。支持非连续，数据格式支持ND，数据类型支持`float32`。

- **metadata**（`Tensor`）：可选参数，表示tiling下沉的aicpu算子输出结果。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **softmax_scale**（`double`）：可选参数，代表缩放系数。数据类型支持`double`，默认值：1.0 / sqrt(D)。

- **cmp_ratio**（`int`）：可选参数，代表压缩率，取值范围1~128。数据类型支持`int`，默认值：1。

- **ori_mask_mode**（`int`）：可选参数，q和ori_kv计算的mask模式。模式0 为不做mask操作；模式3 为rightDownCausal；模式4 为band（滑窗，起点右下角）。数据类型支持`int`。

- **cmp_mask_mode**（`int`）：可选参数，q和cmp_kv计算的mask模式。模式0 为不做mask操作；模式3 为rightDownCausal。数据类型支持`int`。

- **ori_win_left**（`int`）：可选参数，q和ori_kv计算中q对过去token计算的数量。当前仅支持取值127。数据类型支持`int`。

- **ori_win_right**（`int`）：可选参数，q和ori_kv计算中q对未来token计算的数量。当前仅支持取值0。数据类型支持`int`。

- **layout_q**（`str`）：可选参数，q的数据排布格式。支持 "BSND"、"TND"。数据类型支持`str`。

- **layout_kv**（`str`）：可选参数，ori_kv、cmp_kv的数据排布格式。支持 "BSND"、"TND"，当前必须与layout_q保持一致。数据类型支持`str`。

<!-- npu="A3,910b" id5 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持seqused_q、seqused_ori_kv、seqused_cmp_kv、ori_topk_length、cmp_topk_lengt、metadata参数。
<!-- end id5 -->

### sparse_flash_mla_grad_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示公式中$Q$的头数（即N1），当前支持[1, 128]。 | int32 | - |
| num_heads_kv | int | 必选 | 表示公式中$oriKv$或$cmpkv$的头数（即N2），当前仅支持1。 | int32 | - |
| head_dim | int | 必选 | 表示头的维度（即D），当前仅支持512。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | 代表每个Batch中，q的有效token数的累加和形式，当`layout_q`为TND时该参数必传，支持非连续，数据格式支持ND，累加和与T1保持一致。 | int32 | (B+1, ) |
| cu_seqlens_ori_kv | Tensor | 可选 | 代表每个Batch中，ori_kv的有效token数的累加和形式，当`layout_kv`为TND时该参数必传，支持非连续，数据格式支持ND，累加和与T2保持一致。 | int32 | (B+1, ) |
| cu_seqlens_cmp_kv | Tensor | 可选 | 代表每个Batch中，cmp_kv的有效token数的累加和形式，当`layout_kv`为TND时该参数必传，支持非连续，数据格式支持ND，累加和与T3保持一致。 | int32 | (B+1, ) |
| seqused_q | Tensor | 可选 | 表示不同batch中query实际参与运算的token数。 | int32 | (B, ) |
| seqused_ori_kv | Tensor | 可选 | 表示不同batch中ori_kv实际参与运算的token数。 | int32 | (B, ) |
| seqused_cmp_kv | Tensor | 可选 | 表示不同batch中cmp_kv实际参与运算的token数。 | int32 | (B, ) |
| cmp_residual_kv | Tensor | 可选 | 表示每个batch实际ori_s2 // cmpRatio后的余数，支持非连续，数据格式支持ND，当cmp_kv不为空且cmp_mask_mode=3时必须传入。 | int32 | (B, ) |
| ori_topk_length | Tensor | 可选 | 表示每行query对应的ori_kv实际可选的topk长度。 | int32 | (B, S1, N2)或(T1, N2) |
| cmp_topk_length | Tensor | 可选 | 表示每行query对应的cmp_kv实际可选的topk长度。 | int32 | (B, S1, N2)或(T1, N2) |
| batch_size | int | 可选 | 表示输入样本批量大小（即B），默认值为0。 | int32 | - |
| max_seqlen_q | int | 可选 | 表示TND场景下输入q的最大序列长度，默认值为0。 | int32 | - |
| max_seqlen_ori_kv | int | 可选 | 表示TND场景下输入ori_kv的最大序列长度，默认值为0。 | int32 | - |
| max_seqlen_cmp_kv | int | 可选 | 表示TND场景下输入cmp_kv的最大序列长度，默认值为0。 | int32 | - |
| ori_topk | int | 可选 | 表示ori_kv的topk长度，默认值为0。 | int32 | - |
| cmp_topk | int | 可选 | 表示cmp_kv的topk长度，默认值为0。 | int32 | - |
| cmp_ratio | int | 可选 | 代表压缩率，取值范围：1~128，默认值为1。 | int32 | - |
| ori_mask_mode | int | 可选 | 表示q和ori_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，4表示sliding window模式，默认值为0。 | int32 | - |
| cmp_mask_mode | int | 可选 | 表示q和cmp_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，默认值为0。 | int32 | - |
| ori_win_left | int | 可选 | 表示q和ori_kv计算中q对过去token计算的数量，-1表示无穷大，默认值为-1。 | int32 | - |
| ori_win_right | int | 可选 | 表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大，默认值为-1。 | int32 | - |
| layout_q | str | 可选 | 表示q的数据排布格式，支持"BSND"、"TND"，默认值为BSND。 | string | - |
| layout_kv | str | 可选 | 表示ori_kv、cmp_kv的数据排布格式，支持"BSND"、"TND"，当前必须与layout_q保持一致，默认值为BSND。 | string | - |
| has_ori_kv | bool | 可选 | 表示是否传入ori_kv，默认值为true。 | bool | - |
| has_cmp_kv | bool | 可选 | 表示是否传入cmp_kv，默认值为true。 | bool | - |

<!-- npu="A3,910b" id6 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持seqused_q、seqused_ori_kv、seqused_cmp_kv、ori_topk_length、cmp_topk_length，ori_mask_mode仅支持4，cmp_mask_mode仅支持3，ori_win_left仅支持127，ori_win_right仅支持0。
<!-- end id6 -->

## 返回值说明

### sparse_flash_mla_grad

- **dq**（`Tensor`）：对应公式中的$dQ$，支持非连续，数据格式支持ND，数据类型、shape与输入q保持一致。

- **dori_kv**（`Tensor`）：可选输出，表示输入ori_kv的梯度，支持非连续，数据格式支持ND，数据类型、shape与输入ori_kv保持一致。

- **dcmp_kv**（`Tensor`）：可选输出，表示输入cmp_kv的梯度，支持非连续，数据格式支持ND，数据类型、shape与输入cmp_kv保持一致；当cmp_kv为None时，dcmp_kv也为None。

- **dsinks**（`Tensor`）：可选输出，表示输入sinks的梯度，支持非连续，数据格式支持ND，数据类型支持`float32`，shape与输入sinks保持一致。

- **ori_softmax_l1norm**（`Tensor`）：可选输出，表示q与ori_kv计算得出的softmax的L1Norm结果，公式为reduceG(softmax)/G；数据类型为`float32`。`layout_q`为BSND时shape为`[B,S1,N2,K1]`，当`layout_q`为TND时shape为`[T1,N2,K1]`。当ori_sparse_indices不为None时该输出不为空，其他场景下输出为None。

- **cmp_softmax_l1norm**（`Tensor`）：可选输出，表示q与cmp_kv计算得出的softmax的L1Norm结果，公式为reduceG(softmax)/G；当cmp_sparse_indices不为None时该输出不为空，其他场景下输出为None。

### sparse_flash_mla_grad_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个cube核上FlashAttention计算任务的Batch、Head、以及Q和K的分块的索引，以及每个vector核上FlashDecode的规约任务索引。数据格式为ND，不支持非连续的Tensor。 | int32 | (1024, ) |

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和aclgraph模式。
- 参数q、dout、attn_out、ori_kv、cmp_kv的数据类型必须保持一致。
- 入参为空的场景处理：q为空Tensor时直接返回。
- 各个场景关于cmp_kv、cmp_sparse_indices的使用说明如下：
  - SWA场景：要求cmp_kv == None && cmp_sparse_indices == None
  - SCFA场景：要求cmp_kv != None && cmp_sparse_indices != None
  - CFA场景：要求cmp_kv != None && cmp_sparse_indices == None

- **Mask模式支持**：

    | 模式 | 含义 | 备注 |
    | :--- | :--- | :--- |
    | 0 | 不做mask操作 | 支持 |
    | 3 | rightDownCausal | 支持 |
    | 4 | band（滑窗，起点右下角） | oriMaskMode支持 |

- **规格约束**：

    | 规格项 | 规格 | 规格说明 |
    | :--- | :--- | :--- |
    | B | 支持泛化 | - |
    | S1、S2 | 支持泛化 | 支持S1、S2不等长。 |
    | N1 | 1~128 | - |
    | N2 | 1 | 当前仅支持N2=1。 |
    | D | 512 | q、ori_kv、cmp_kv最后一维需保持一致。 |
    | layout_q/kv | BSND / TND，必须一致 | - |
    | cmp_ratio | 1~128 | - |

  - `ori_kv`/`cmp_kv`传None的支持情况:
    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持。
    <!-- end id7 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
    <!-- end id10 -->
  - `ori_sparse_indices`的支持情况:
    <!-- npu="A3,910b" id11 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持。
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
    <!-- end id12 -->
  - `seqused_q`的支持情况:
    <!-- npu="A3,910b" id13 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持（需传None）。
    <!-- end id13 -->
    <!-- npu="950" id14 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
    <!-- end id14 -->
  - `ori_topk_length`/`cmp_topk_length`的支持情况:
    <!-- npu="A3,910b" id15 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持（需传None）。
    <!-- end id15 -->
    <!-- npu="950" id16 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
    <!-- end id16 -->
  - `sinks` 传None的支持情况:
    <!-- npu="A3,910b" id17 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持。
    <!-- end id17 -->
    <!-- npu="950" id18 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：支持。
    <!-- end id18 -->
  - `metadata`的支持情况:
    <!-- npu="A3,910b" id19 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持传None。
    <!-- end id19 -->
    <!-- npu="950" id20 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：必须传。
    <!-- end id20 -->
  - `ori_mask_mode`的支持情况:
    <!-- npu="A3,910b" id21 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持模式4。
    <!-- end id21 -->
    <!-- npu="950" id22 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：支持模式0、3、4。
    <!-- end id22 -->
  - `cmp_mask_mode`的支持情况:
    <!-- npu="A3,910b" id23 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持模式3。
    <!-- end id23 -->
    <!-- npu="950" id24 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：支持模式0、3。
    <!-- end id24 -->

## 确定性计算

默认支持确定性计算

## 调用示例

- 单算子模式调用

    ```python
    import math
    import random
    import numpy as np

    import torch
    import torch_npu
    import cann_ops_transformer

    S1 = 256
    S2 = 1024
    cmp_ratio = 4
    actual_seq_q = [S1]
    actual_seq_ori_kv = [S2]
    actual_seq_cmp_kv = [S2 // cmp_ratio]
    cmp_residual = [i % cmp_ratio for i in actual_seq_ori_kv]

    T1 = sum(actual_seq_q)
    T2 = sum(actual_seq_ori_kv)
    T3 = sum(actual_seq_cmp_kv)

    B = 1
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    scaleValue = 1 / math.sqrt(D)
    dtype = torch.float16
    input_layout = "TND"

    q_shape = tuple((T1, N1, D))
    ori_kv_shape = tuple((T2, N2, D))
    cmp_kv_shape = tuple((T3, N2, D))
    out_shape = tuple((T1, N1, D))
    lse_shape = tuple((N2, T1, N1 // N2))

    cu_seq_qlen = [0] + [sum(actual_seq_q[:x+1]) for x in range(len(actual_seq_q))]
    cu_seq_ori_kvlen = [0] + [sum(actual_seq_ori_kv[:x+1]) for x in range(len(actual_seq_ori_kv))]
    cu_seq_cmp_kvlen = [0] + [sum(actual_seq_cmp_kv[:x+1]) for x in range(len(actual_seq_cmp_kv))]

    q = (torch.rand(q_shape).to(dtype)) * 2
    ori_kv = (torch.rand(ori_kv_shape).to(dtype)) * 2
    cmp_kv = (torch.rand(cmp_kv_shape).to(dtype)) * 2 if cmp_ratio != 1 else None
    dy = (torch.rand(out_shape).to(dtype)) * 2
    out = (torch.rand(out_shape).to(dtype)) * 2 # 实际使用场景中应使用前向输出attn_out
    sinks = (torch.rand(N1).to(torch.float32)) * (128)
    lse = (torch.rand(lse_shape).to(torch.float32)) # 实际使用场景中应使用前向输出lse

    if cmp_ratio == 4:
        cmp_sparse_indices = torch.ones([T1, N2, K], dtype=torch.int32) * -1
        accum_t = 0
        for i in range(B):
            cur_s1 = actual_seq_q[i]
            cur_ori_s2 = actual_seq_cmp_kv[i] * cmp_ratio + cmp_residual[i]
            delta_s = cur_ori_s2 - cur_s1
            for s1_idx in range(cur_s1):
                threshold = delta_s + s1_idx + 1 if delta_s + s1_idx + 1 >= 0 else 0
                max_cmp_s2 = threshold // cmp_ratio
                cur_s2_idxs = np.arange(max_cmp_s2)
                if max_cmp_s2 < K:
                    cmp_sparse_indices[accum_t, :, :max_cmp_s2] = torch.tensor(np.random.choice(cur_s2_idxs, size=max_cmp_s2, replace=False))
                else:
                    cmp_sparse_indices[accum_t, :, :] = torch.tensor(np.random.choice(cur_s2_idxs, size=K, replace=False))
                accum_t += 1
    else:
        cmp_sparse_indices = None

    cmp_kv_npu = cmp_kv.npu() if cmp_kv != None else None
    cmp_sparse_indices_npu = cmp_sparse_indices.npu() if cmp_sparse_indices != None else None
    cu_seq_qlen = None if input_layout != "TND" else torch.tensor(cu_seq_qlen).to(torch.int32).npu()
    cu_seq_ori_kvlen = None if input_layout != "TND" else torch.tensor(cu_seq_ori_kvlen).to(torch.int32).npu()
    cu_seq_cmp_kvlen = None if input_layout != "TND" else torch.tensor(cu_seq_cmp_kvlen).to(torch.int32).npu()
    cmp_residual_kv = torch.tensor(cmp_residual).to(torch.int32).npu() if cmp_residual != None else None

    metadata = cann_ops_transformer.ops.sparse_flash_mla_grad_metadata(N1, N2, D)
    npu_out = cann_ops_transformer.ops.sparse_flash_mla_grad(
            q.npu(), dy.npu(), out.npu(), lse.npu(),
            ori_kv=ori_kv.npu(), cmp_kv=cmp_kv_npu,
            cmp_sparse_indices=cmp_sparse_indices_npu,
            cu_seqlens_q=cu_seq_qlen,
            cu_seqlens_ori_kv=cu_seq_ori_kvlen,
            cu_seqlens_cmp_kv=cu_seq_cmp_kvlen,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks.npu(),
            softmax_scale=scaleValue,
            cmp_ratio=cmp_ratio,
            layout_q=input_layout,
            layout_kv=input_layout,
            ori_win_left=127,
            ori_win_right=0,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            metadata=metadata
        )
    ```
