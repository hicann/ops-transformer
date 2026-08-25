# SparseFlashMlaSoftmaxL1NormMetadata

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

- 算子功能：该算子为AICPU算子，是`SparseFlashMlaSoftmaxL1Norm`算子的前置算子。根据`SparseFlashMlaSoftmaxL1Norm`算子的输入shape、layout、mask和压缩比例信息，计算并输出分核切分metadata，减少主算子tiling阶段对host array的访问。

  **该算子不建议单独使用，建议与`SparseFlashMlaSoftmaxL1Norm`算子配合使用，形成完整的工作流。**

  计算流程：
  1. 接收主算子的shape信息，包括batchSize、maxSeqLenQ、maxSeqLenK、numHeadsQ、numHeadsK、headDim、topk、layout和mask信息。
  2. 根据layout和可选输入seqUsedQOptional计算参与负载均衡的seq总数，采用strided方式将任务均衡切分到可用AIC核上。
  3. 输出metadata，作为`SparseFlashMlaSoftmaxL1Norm`算子的metadataOptional输入使用。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | -------------- | ---- | -------- | -------- |
| cu_seqlens_q | 可选输入 | 表示每个Batch中Query的有效token数的累加和形式。TND场景下必传，第一个值固定为0。 | INT32 | ND |
| cu_seqlens_k | 可选输入 | 表示每个Batch中Key的有效token数的累加和形式。TND场景下必传，第一个值固定为0。 | INT32 | ND |
| seqused_q | 可选输入 | 表示不同batch中q实际参与运算的token数。当该入参存在时，seq总数按每个batch实际用到的seq累加计算。 | INT32 | ND |
| seqused_k | 可选输入 | 预留接口参数，表示不同batch中k实际参与运算的token数。当前kernel路径暂不使用。 | INT32 | ND |
| cmp_residual_k | 可选输入 | 预留接口参数，表示不同batch中key的sequence length与cmpRatio相关的残差。当前kernel路径暂不使用。 | INT32 | ND |
| topk_length | 可选输入 | 表示每行q对应的k实际可选的topk长度。maskMode=0且存在稀疏索引时需要传。 | INT32 | ND |
| batch_size | 可选属性 | 表示batch数量，支持非负数。TND场景可填0并通过cu_seqlens_q推导，默认值为0。 | INT64 | - |
| max_seqlen_q | 可选属性 | 表示q的最大序列长度，支持非负数。BSND场景必须为正数，默认值为0。 | INT64 | - |
| max_seqlen_k | 可选属性 | 表示k的最大序列长度，支持非负数。BSND场景必须为正数，默认值为0。 | INT64 | - |
| num_heads_q | 必选属性 | 表示q的head个数，必须为正数且能被num_heads_k整除。 | INT64 | - |
| num_heads_k | 必选属性 | 表示k的head个数，必须为正数。 | INT64 | - |
| head_dim | 必选属性 | 表示q/k的head dimension，必须为正数。 | INT64 | - |
| topk | 可选属性 | 表示从k中筛选出的关键token个数，支持非负数。0表示无稀疏，默认值为0。 | INT64 | - |
| cmp_ratio | 可选属性 | 表示k的压缩率，取值范围[1, 128]，默认值为1。 | INT64 | - |
| mask_mode | 可选属性 | 表示sparse mask模式。0：No mask；3：rightDownCausal模式。默认值为0。 | INT64 | - |
| layout_q | 可选属性 | 表示q侧的数据排布格式，支持BSND/TND，默认值为"BSND"。 | STRING | - |
| layout_k | 可选属性 | 表示k侧的数据排布格式，支持BSND/TND，默认值为"BSND"。 | STRING | - |
| metadata | 输出 | 表示负载均衡结果输出，shape固定为(64,)，输出结果作为`SparseFlashMlaSoftmaxL1Norm`的metadataOptional输入。 | INT32 | ND |

## 约束说明

- 确定性说明：aclnnSparseFlashMlaSoftmaxL1NormMetadata默认确定性实现。
- BSND场景：必传batch_size、max_seqlen_q和max_seqlen_k参数以获取shape信息。
- TND场景：必传cu_seqlens_q和cu_seqlens_k参数以获取正确shape信息。当batch_size为0时，通过cu_seqlens_q的shape推导batch。
- layout约束：layout_q必须为BSND或TND，layout_k支持BSND和TND，layout_k需与layout_q保持一致。
- head约束：num_heads_q、num_heads_k必须为正数，且num_heads_q必须能被num_heads_k整除, head_dim只支持512。
- 负载均衡约束：
    - topk支持非负数，0表示无稀疏。
    - cmp_ratio取值范围为[1, 128]。
    - mask_mode当前仅支持0和3。

## 问题定位说明

- 关于AI CPU算子Kernel常见执行问题或异常错误，问题定位方法请参考《故障处理》中“[故障案例集>算子执行问题>AI CPU算子Kernel执行报错](https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)”。

## Metadata输出布局

metadata输出为INT32 Tensor，shape固定为(64,)，字段布局如下：

| 字段 | index | 说明 |
| ---- | ----- | ---- |
| totalNum | 0 | 参与负载均衡的seq总数。TND场景为各batch的seqQ累加值；BSND场景为B×S；当seqused_q存在时，按每个batch实际用到的seq累加。 |
| formerCoreProcessNum | 1 | 常规核处理的seq数，即ceil(totalNum / totalCoreNum)。 |
| remainCoreProcessNum | 2 | 尾核处理的seq数，即floor(totalNum / totalCoreNum)。 |
| remainCoreNum | 3 | 尾核数目。当totalNum能被totalCoreNum整除时为0，否则为totalCoreNum - (totalNum % totalCoreNum)。 |
| totalCoreNum | 4 | 实际使用的AIC核数，取min(totalNum, aicCoreNum, 36)。 |

采用strided切分方式，每个核处理的seq索引按核间步进totalCoreNum排列。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| aclnn API | - | 通过[aclnnSparseFlashMlaSoftmaxL1NormMetadata](docs/aclnnSparseFlashMlaSoftmaxL1NormMetadata.md)接口方式调用算子，输出metadata供`SparseFlashMlaSoftmaxL1Norm`使用。 |
| PyTorch API | - | 通过[sparse_flash_mla_softmax_l1_norm_metadata](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla_softmax_l1_norm.md)（内置在主算子接口中）调用算子。 |
