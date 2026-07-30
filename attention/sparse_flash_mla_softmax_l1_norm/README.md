# SparseFlashMlaSoftmaxL1Norm

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：计算`SparseFlashMla`注意力的Softmax L1Norm结果，支持Compressed Attention以及Sparse Compressed Attention场景。该算子为`aclnnDenseLightningIndexerKLLossGrad`反向算子的配套正向接口，输出可用于反向梯度计算。调用过程中，需先调用`SparseFlashMlaSoftmaxL1NormMetadata`算子完成负载均衡计算，再调用本算子完成计算过程。

- 计算公式：

    阶段一：根据是否为sparse场景，对输入key进行选择

    * 当为sparse时：

    $$
    selectedKv = Gather(K, sparseIndices[i]),\ 0 <= i < selectBlockCount
    $$

    * else：

    $$
    selectedKv = K
    $$

    阶段二：计算P（SimpleSoftmax）

    $$
    P = SimpleSoftmax(Mask(Q @ selectedKv^{T} \cdot scale), softmaxLse)
    $$

    阶段三：计算Softmax L1Norm

    $$
    softmaxL1Norm = \frac{ReduceSum(P, dim=G)}{G}
    $$

    其中，$G$ 为group数（$G = N1 / N2$）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | -------------- | ---- | -------- | -------- |
| q | 输入 | attention结构的输入Q。 | FLOAT16、BFLOAT16 | ND |
| k | 输入 | attention结构的输入K(V)。 | FLOAT16、BFLOAT16 | ND |
| softmax_lse | 输入 | 注意力正向计算的输出softmaxLse，计算公式详见sparse_flash_mla文档。 | FLOAT32 | ND |
| sparse_indices | 可选输入 | 稀疏场景下选择的k中权重较高的注意力索引。 | INT32 | ND |
| cu_seqlens_q | 可选输入 | 每个Batch中Query的有效token数的累加和形式。layout为TND时该参数必传。 | INT32 | ND |
| cu_seqlens_k | 可选输入 | 每个Batch中Key的有效token数的累加和形式。layout为TND时该参数必传。 | INT32 | ND |
| seqused_q | 可选输入 | 表示不同batch中q实际参与运算的token数。 | INT32 | ND |
| seqused_k | 可选输入 | 表示不同batch中k实际参与运算的token数。 | INT32 | ND |
| cmp_residual_k | 可选输入 | 表示每个batch S2 // cmpRatio后的余数。当k不为空且mask_mode=3时必须传入。 | INT32 | ND |
| topk_length | 可选输入 | 表示每行q对应的k实际可选的topk长度。mask_mode=0且存在稀疏索引时需要传。 | INT32 | ND |
| metadata | 可选输入 | 表示tiling下沉的aicpu算子输出结果，由`SparseFlashMlaSoftmaxL1NormMetadata`算子生成。 | INT32 | ND |
| softmax_scale | 可选属性 | 缩放系数，默认值为1.0。推荐值：sqrt(head_dim)的倒数。 | FLOAT32 | - |
| max_seqlen_k | 可选属性 | 表示k的最大序列长度，TND dense场景用于输出shape推导，默认值为0。 | INT64 | - |
| cmp_ratio | 可选属性 | 表示对k的压缩率，取值范围1~128，默认值为1。 | INT64 | - |
| mask_mode | 可选属性 | 表示q和k计算的mask模式。0：No mask；3：rightDownCausal模式。默认值为0。 | INT64 | - |
| layout_q | 可选属性 | 表示输入q的数据排布格式，支持"BSND"、"TND"，默认值为"BSND"。 | STRING | - |
| layout_k | 可选属性 | 表示输入k的数据排布格式，支持"BSND"、"TND"，默认值为"BSND"。 | STRING | - |
| softmax_l1_norm | 输出 | 表示q与k计算得出的softmax L1Norm结果，公式为reduceG(softmax)/G。若存在sparse_indices则该输出不为空，其他场景下输出为空。 | FLOAT32 | ND |

## 约束说明

- 确定性说明：aclnnSparseFlashMlaSoftmaxL1Norm默认确定性实现。
- 仅支持BSND或TND layout，且layout_q与layout_k必须保持一致。
- 关于数据shape的约束：
    - B：泛化支持。
    - S1、S2：泛化支持，支持S1、S2不等长。
    - N1：支持1~128，且num_heads_q必须能被num_heads_k整除。
    - N2：仅支持1。
    - D：仅支持512。
- mask_mode支持：
    - 0：不做mask操作。
    - 3：rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。
- cmp_ratio取值范围：1~128。
- TND场景：必传cu_seqlens_q和cu_seqlens_k参数。
- metadata必须传入，由`SparseFlashMlaSoftmaxL1NormMetadata`算子生成。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| aclnn API | [test_aclnn_sparse_flash_mla_softmax_l1_norm](examples/test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp) | 通过[aclnnSparseFlashMlaSoftmaxL1Norm](docs/aclnnSparseFlashMlaSoftmaxL1Norm.md)接口方式调用算子。 |
| PyTorch API | - | 通过[sparse_flash_mla_softmax_l1_norm](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla_softmax_l1_norm.md)接口调用算子。 |
