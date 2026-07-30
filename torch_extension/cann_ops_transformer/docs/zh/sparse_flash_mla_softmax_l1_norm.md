# sparse_flash_mla_softmax_l1_norm

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

- 接口功能：计算`SparseFlashMla`注意力的Softmax L1Norm结果，支持Compressed Attention以及Sparse Compressed Attention场景。该接口为`dense_lightning_indexer_kl_loss_grad`反向算子的配套接口，输出可用于反向梯度计算。调用过程中，需先调用`sparse_flash_mla_softmax_l1_norm_metadata`接口完成负载均衡计算，再调用`sparse_flash_mla_softmax_l1_norm`接口完成算子计算过程。
    - `sparse_flash_mla_softmax_l1_norm_metadata`接口：根据主算子的shape、layout、mask等信息，采用strided方式将任务均衡切分到可用AIC核上，输出metadata供主算子使用。
    - `sparse_flash_mla_softmax_l1_norm`接口：根据metadata中的分核信息，对Q和K计算Softmax L1Norm。

- 计算公式：

    阶段一：根据是否为sparse场景，对输入key进行选择

    * 当为sparse时：

    $$
    selectedKv\text{ }=\text{ }Gather \left( K, sparseIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

    * else：

    $$
    selectedKv\text{ }=\text{ }K
    $$

    阶段二：计算P（SimpleSoftmax）

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), lse)
    $$

    阶段三：计算Softmax L1Norm

    $$
    softmaxL1Norm = \frac{ReduceSum(P, dim=G)}{G}
    $$

    其中，$G$ 为group数（$G = N1 / N2$），$ReduceSum$ 在G维度（query head group维度）上对softmax概率$P$求和后取平均。

## 函数原型

调用`sparse_flash_mla_softmax_l1_norm`接口之前，先调用前置接口`sparse_flash_mla_softmax_l1_norm_metadata`，完成负载均衡的计算。

```python
cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    topk=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None
) -> Tensor
```

```python
cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
    q,
    k,
    softmax_lse,
    *,
    sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    metadata=None,
    softmax_scale=1.0,
    max_seqlen_k=0,
    cmp_ratio=1,
    mask_mode=0,
    layout_q="BSND",
    layout_k="BSND"
) -> Tensor
```

## 参数说明

### sparse_flash_mla_softmax_l1_norm

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| q | Tensor | 必选 | attention结构的输入Q，对应公式中的$Q$。`layout_q`="BSND"时shape为(B,S1,N1,D)；`layout_q`="TND"时shape为(T1,N1,D)。 | float16、bfloat16 | (B,S1,N1,D)、(T1,N1,D) |
| k | Tensor | 必选 | attention结构的输入K，对应公式中的$K$。与q的B保持一致，S2支持泛化，N2=1。 | float16、bfloat16 | (B,S2,N2,D)、(T2,N2,D) |
| softmax_lse | Tensor | 必选 | 注意力正向计算的输出softmax_lse。与q的B保持一致，N2=1，S1与q的S1保持一致，G=N1/N2。 | float32 | (B,N2,S1,G)、(N2,T1,G) |
| sparse_indices | Tensor | 可选 | 稀疏场景下选择的k中权重较高的注意力索引。与q的B/S1保持一致，N2=1，K支持泛化。 | int32 | (B,S1,N2,K)、(T1,N2,K) |
| cu_seqlens_q | Tensor | 可选 | 每个Batch中q的有效token数的累加和形式。`layout_q`为TND时该参数必传，长度与B+1保持一致，累加和与T1保持一致。 | int32 | (B+1,) |
| cu_seqlens_k | Tensor | 可选 | 每个Batch中k的有效token数的累加和形式。`layout_k`为TND时该参数必传，长度与B+1保持一致，累加和与T2保持一致。 | int32 | (B+1,) |
| seqused_q | Tensor | 可选 | 表示不同batch中q实际参与运算的token数。 | int32 | (B,) |
| seqused_k | Tensor | 可选 | 表示不同batch中k实际参与运算的token数。 | int32 | (B,) |
| cmp_residual_k | Tensor | 可选 | 表示每个batch S2 // cmpRatio后的余数。当k不为空且mask_mode=3时必须传入。 | int32 | (B,) |
| topk_length | Tensor | 可选 | 表示每行q对应的k实际可选的topk长度。mask_mode=0且sparse_indices不为空时需要传，且必须为准确值。 | int32 | (B,S1,N2)、(T1,N2) |
| metadata | Tensor | 必选 | 表示tiling下沉的aicpu算子输出结果，由`sparse_flash_mla_softmax_l1_norm_metadata`算子生成。 | int32 | (x) |
| softmax_scale | float | 可选 | 表示缩放系数，默认值为1.0。推荐值：sqrt(head_dim)的倒数。 | float32 | - |
| max_seqlen_k | int | 可选 | 表示k的最大序列长度，TND dense场景用于输出shape推导，默认值为0。 | int | - |
| cmp_ratio | int | 可选 | 表示对k的压缩率，取值范围1~128，默认值为1。 | int | - |
| mask_mode | int | 可选 | 表示q和k计算的mask模式。0：No mask；3：rightDownCausal模式。默认值为0。 | int | - |
| layout_q | str | 可选 | 表示输入q的数据排布格式，支持"BSND"、"TND"，默认值为"BSND"。需与layout_k保持一致。 | string | - |
| layout_k | str | 可选 | 表示输入k的数据排布格式，支持"BSND"、"TND"，默认值为"BSND"。需与layout_q保持一致。 | string | - |

### sparse_flash_mla_softmax_l1_norm_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示公式中$Q$的头数（即N1），当前支持1~128。 | int | - |
| num_heads_k | int | 必选 | 表示公式中$key$的头数（即N2），当前仅支持1。 | int | - |
| head_dim | int | 必选 | 表示头的维度（即D），当前仅支持512。 | int | - |
| cu_seqlens_q | Tensor | 可选 | 表示每个Batch中q的有效token数的累加和形式，当`layout_q`为TND时该参数必传，累加和与T1保持一致。 | int32 | (B+1,) |
| cu_seqlens_k | Tensor | 可选 | 表示每个Batch中k的有效token数的累加和形式，当`layout_k`为TND时该参数必传，累加和与T2保持一致。 | int32 | (B+1,) |
| seqused_q | Tensor | 可选 | 表示不同batch中q实际参与运算的token数。 | int32 | (B,) |
| seqused_k | Tensor | 可选 | 预留接口参数，表示不同batch中k实际参与运算的token数，当前kernel路径暂不使用。 | int32 | (B,) |
| cmp_residual_k | Tensor | 可选 | 预留接口参数，表示不同batch中k的sequence length与cmpRatio相关的残差，当前kernel路径暂不使用。 | int32 | (B,) |
| topk_length | Tensor | 可选 | 表示每行q对应的k实际可选的topk长度。 | int32 | (B,S1,N2)、(T1,N2) |
| batch_size | int | 可选 | 表示输入样本批量大小（即B），默认值为None（BSND场景需传正数，TND场景可为None自动推导）。 | int | - |
| max_seqlen_q | int | 可选 | 表示q的最大序列长度，默认值为None。BSND场景必须为正数。 | int | - |
| max_seqlen_k | int | 可选 | 表示k的最大序列长度，默认值为None。BSND场景必须为正数。 | int | - |
| topk | int | 可选 | 表示从k中筛选出的关键token个数，0表示无稀疏，默认值为None。 | int | - |
| layout_q | str | 可选 | 表示q的数据排布格式，支持"BSND"、"TND"，默认值为None（内部转为"BSND"）。 | string | - |
| layout_k | str | 可选 | 表示k的数据排布格式，支持"BSND"、"TND"，默认值为None（内部转为"BSND"）。 | string | - |
| mask_mode | int | 可选 | 表示q和k计算的mask模式，0表示No mask，3表示rightDownCausal模式，默认值为None（内部转为0）。 | int | - |
| cmp_ratio | int | 可选 | 表示对k的压缩率，取值范围1~128，默认值为None（内部转为1）。 | int | - |

## 返回值说明

### sparse_flash_mla_softmax_l1_norm

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| softmax_l1_norm | Tensor | 必选 | q与k计算得出的softmax L1Norm结果，公式为reduceG(softmax)/G。若存在sparse_indices则该输出不为空，其他场景下输出为空。 | float32 | (B,S1,N2,S2)、(T1,N2,T2) |

### sparse_flash_mla_softmax_l1_norm_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 负载均衡结果输出，采用strided切分方式，每个核处理的seq索引按核间步进排列。 | int32 | (64,) |

Metadata字段布局：

| 字段 | index | 说明 |
|:---|:---|:---|
| totalNum | 0 | 参与负载均衡的seq总数。 |
| formerCoreProcessNum | 1 | 常规核处理的seq数，即ceil(totalNum / totalCoreNum)。 |
| remainCoreProcessNum | 2 | 尾核处理的seq数，即floor(totalNum / totalCoreNum)。 |
| remainCoreNum | 3 | 尾核数目。 |
| totalCoreNum | 4 | 实际使用的AIC核数，取min(totalNum, aicCoreNum, 36)。 |

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- 参数q、k的数据类型必须保持一致。
- 入参为空处理：q为空Tensor时直接返回。

- Mask模式支持：

    | 模式 | 含义 | 备注 |
    | :--- | :--- | :--- |
    | 0 | 不做mask操作 | 支持 |
    | 3 | rightDownCausal | 支持 |

- 规格约束：

    | 规格项 | 规格 | 规格说明 |
    | :--- | :--- | :--- |
    | B | 支持泛化 | - |
    | S1、S2 | 支持泛化 | 支持S1、S2不等长。 |
    | N1 | 1~128 | num_heads_q必须能被num_heads_k整除。 |
    | N2 | 1 | 当前仅支持N2=1。 |
    | D | 512 | q、k最后一维需保持一致。 |
    | layout_q/k | BSND / TND，必须一致 | - |
    | cmp_ratio | 1~128 | - |

  - BSND场景：必传batch_size、max_seqlen_q和max_seqlen_k参数。
  - TND场景：必传cu_seqlens_q和cu_seqlens_k参数，batch_size可为None（通过cu_seqlens_q推导）。
  - metadata必须传入，由`sparse_flash_mla_softmax_l1_norm_metadata`算子生成。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    S1 = 16
    S2 = 2048
    cmp_ratio = 128
    actual_seq_q = [S1]
    actual_seq_k = [S2]

    T1 = sum(actual_seq_q)
    T2 = sum(actual_seq_k)

    B = 1
    N1 = 128
    N2 = 1
    D = 512
    scale_value = 1.0 / math.sqrt(D)
    dtype = torch.float16
    input_layout = "TND"

    q_shape = (T1, N1, D)
    k_shape = (T2, N2, D)
    softmax_lse_shape = (N2, T1, N1 // N2)
    softmax_l1_norm_shape = (T1, N2, T2)

    cu_seq_qlen = [0] + [sum(actual_seq_q[:x+1]) for x in range(len(actual_seq_q))]
    cu_seq_klen = [0] + [sum(actual_seq_k[:x+1]) for x in range(len(actual_seq_k))]

    q = (torch.rand(q_shape).to(dtype)) * 2
    k = (torch.rand(k_shape).to(dtype)) * 2
    softmax_lse = (torch.rand(softmax_lse_shape).to(torch.float32))

    cu_seq_qlen_tensor = torch.tensor(cu_seq_qlen).to(torch.int32).npu()
    cu_seq_klen_tensor = torch.tensor(cu_seq_klen).to(torch.int32).npu()

    cmp_residual_k = torch.zeros(B, dtype=torch.int32, device="npu")

    # 调用sparse_flash_mla_softmax_l1_norm_metadata完成负载均衡计算
    metadata = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
        N1, N2, D,
        cu_seqlens_q=cu_seq_qlen_tensor,
        cu_seqlens_k=cu_seq_klen_tensor,
        cmp_residual_k=cmp_residual_k,
        max_seqlen_q=S1,
        max_seqlen_k=S2,
        topk=0,
        cmp_ratio=cmp_ratio,
        mask_mode=3,
        layout_q=input_layout,
        layout_k=input_layout,
    )

    # 调用sparse_flash_mla_softmax_l1_norm执行算子计算
    softmax_l1_norm = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
        q.npu(),
        k.npu(),
        softmax_lse.npu(),
        cu_seqlens_q=cu_seq_qlen_tensor,
        cu_seqlens_k=cu_seq_klen_tensor,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        softmax_scale=scale_value,
        max_seqlen_k=S2,
        cmp_ratio=cmp_ratio,
        mask_mode=3,
        layout_q=input_layout,
        layout_k=input_layout,
    )

    torch_npu.npu.synchronize()
    assert softmax_l1_norm.shape == softmax_l1_norm_shape
    assert softmax_l1_norm.dtype == torch.float32
    assert torch.isfinite(softmax_l1_norm.float()).all().item()
    ```

- TorchAir图模式调用：

    ```python
    import math
    import torch
    import torch_npu
    import torchair
    import cann_ops_transformer


    class SparseFlashMlaSoftmaxL1NormModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q, k, softmax_lse, cu_seq_qlen_tensor, cu_seq_klen_tensor,
                    cmp_residual_k, max_seqlen_k, cmp_ratio, mask_mode, layout_q, layout_k):
            # 调用sparse_flash_mla_softmax_l1_norm_metadata完成负载均衡计算
            metadata = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
                N1, N2, D,
                cu_seqlens_q=cu_seq_qlen_tensor,
                cu_seqlens_k=cu_seq_klen_tensor,
                cmp_residual_k=cmp_residual_k,
                max_seqlen_q=S1,
                max_seqlen_k=max_seqlen_k,
                topk=0,
                cmp_ratio=cmp_ratio,
                mask_mode=mask_mode,
                layout_q=layout_q,
                layout_k=layout_k,
            )
            # 调用sparse_flash_mla_softmax_l1_norm执行算子计算
            return cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
                q, k, softmax_lse,
                cu_seqlens_q=cu_seq_qlen_tensor,
                cu_seqlens_k=cu_seq_klen_tensor,
                cmp_residual_k=cmp_residual_k,
                metadata=metadata,
                softmax_scale=scale_value,
                max_seqlen_k=max_seqlen_k,
                cmp_ratio=cmp_ratio,
                mask_mode=mask_mode,
                layout_q=layout_q,
                layout_k=layout_k,
            )


    torch_npu.npu.set_device(0)

    S1 = 16
    S2 = 2048
    cmp_ratio = 128
    actual_seq_q = [S1]
    actual_seq_k = [S2]

    T1 = sum(actual_seq_q)
    T2 = sum(actual_seq_k)

    B = 1
    N1 = 128
    N2 = 1
    D = 512
    scale_value = 1.0 / math.sqrt(D)
    dtype = torch.float16
    input_layout = "TND"

    q_shape = (T1, N1, D)
    k_shape = (T2, N2, D)
    softmax_lse_shape = (N2, T1, N1 // N2)

    cu_seq_qlen = [0] + [sum(actual_seq_q[:x+1]) for x in range(len(actual_seq_q))]
    cu_seq_klen = [0] + [sum(actual_seq_k[:x+1]) for x in range(len(actual_seq_k))]

    q = (torch.rand(q_shape).to(dtype)) * 2
    k = (torch.rand(k_shape).to(dtype)) * 2
    softmax_lse = (torch.rand(softmax_lse_shape).to(torch.float32))

    cu_seq_qlen_tensor = torch.tensor(cu_seq_qlen).to(torch.int32).npu()
    cu_seq_klen_tensor = torch.tensor(cu_seq_klen).to(torch.int32).npu()

    cmp_residual_k = torch.zeros(B, dtype=torch.int32, device="npu")

    model = SparseFlashMlaSoftmaxL1NormModel()
    config = torchair.CompilerConfig()
    # 获取图模式后端
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    softmax_l1_norm = model(q, k, softmax_lse, cu_seq_qlen_tensor, cu_seq_klen_tensor,
                            cmp_residual_k, S2, cmp_ratio, 3, input_layout, input_layout)
    ```
