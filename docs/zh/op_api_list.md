# 算子接口（aclnn）

## 使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀API，接口列表如下），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

- **头文件、库文件**

  调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在`${INSTALL_DIR}/include/aclnnop`，库文件默认在`${INSTALL_DIR}/lib64`，具体文件如下：

  - 头文件：方式1 （推荐）：引用算子仓总头文件aclnn\_ops\_\$\{ops\_project\}.h。方式2：引用单个算子API的头文件aclnn\_\*.h。
  - 库文件：引用算子仓对应的库文件libopapi_${ops_project}.so。注意，原所有算子仓总库文件libopapi.so后续会废弃，不推荐使用，也不支持与单个算子仓库文件同时使用。

  ${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓名（如math、nn、cv、transformer），请改为实际算子仓名。

- **V版本演进说明**

  请注意，部分API存在多个V版本，使用时选择最高V版本即可（高版本API已兼容低版本API的所有能力）。

## 接口列表

> **确定性简介**：
>
> - 配置说明：因CANN或NPU型号不同等原因，可能无法保证同一个算子多次运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分算子接口可通过`aclrtCtxSetSysParamOpt`（参见[《Runtime运行时API》](https://hiascend.com/document/redirect/CannCommunityRuntimeApi)）开启确定性算法，使多次运行结果一致。
> - 性能说明：同一个算子采用确定性计算通常比非确定性慢，因此模型单次运行性能可能会下降。但在实验、调试调测等需要保证多次运行结果相同来定位问题的场景，确定性计算可以提升效率。
> - 线程说明：同一线程中只能设置一次确定性状态，多次设置以最后一次有效设置为准。有效设置是指设置确定性状态后，真正执行了一次算子任务下发。如果仅设置，没有算子下发，只能是确定性变量开启但未下发给算子，因此不执行算子。
>   解决方案：暂不推荐一个线程多次设置确定性。该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。
> - 符号说明：表中"-"符号表示该接口暂不支持当前列产品。

|    接口名   |   说明     | 确定性说明（A2/A3）  | 确定性说明（Ascend 950） |
| ----------- | ------------------- | ---------  | --------- |
|[aclnnAllGatherAdd](../../examples/mc2/all_gather_add/docs/aclnnAllGatherAdd.md)|完成[AllGather](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0873.html)通信和[Add](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0035.html)加法的融合。|默认确定性实现| - |
|[aclnnAllGatherMatmul](../../mc2/all_gather_matmul/docs/aclnnAllGatherMatmul.md)|完成AllGather通信与MatMul计算融合。|默认确定性实现| 默认确定性实现 |
|[aclnnAllGatherMatmulV2](../../mc2/all_gather_matmul_v2/docs/aclnnAllGatherMatmulV2.md)|aclnnAllGatherMatmulV2接口是对aclnnAllGatherMatmul接口的功能拓展。|默认确定性实现| 默认确定性实现 |
|[aclnnAlltoAllAllGatherBatchMatMul](../../mc2/allto_all_all_gather_batch_mat_mul/docs/aclnnAlltoAllAllGatherBatchMatMul.md)|完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。|默认确定性实现| - |
|[aclnnAlltoAllMatmul](../../mc2/allto_all_matmul/docs/aclnnAlltoAllMatmul.md)|完成AlltoAll通信与MatMul计算融合。|默认确定性实现| 默认确定性实现 |
|[aclnnAlltoAllMatmulV2](../../mc2/allto_all_matmul/docs/aclnnAlltoAllMatmulV2.md)|兼容[aclnnAlltoAllMatmul](../../mc2/allto_all_matmul/docs/aclnnAlltoAllMatmul.md)支持的功能，在此基础上新增commMode参数，供用户指定通信引擎参数。|默认确定性实现| 默认确定性实现 |
|[aclnnAlltoAllQuantMatmul](../../mc2/allto_all_matmul/docs/aclnnAlltoAllQuantMatmul.md)|完成AlltoAll通信、量化计算、MatMul计算和反量化计算的融合。|默认确定性实现| 默认确定性实现 |
|[aclnnAlltoAllQuantMatmulV2](../../mc2/allto_all_matmul/docs/aclnnAlltoAllQuantMatmulV2.md)|兼容[aclnnAlltoAllQuantMatmul](../../mc2/allto_all_matmul/docs/aclnnAlltoAllQuantMatmul.md)支持的功能，在此基础上新增commMode参数，供用户指定通信引擎参数。|默认确定性实现| 默认确定性实现 |
|[aclnnAlltoAllvGroupedMatMul](../../mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMul.md)|完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合。|默认确定性实现| 默认确定性实现 |
|[aclnnAlltoAllvGroupedMatMulV2](../../mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMulV2.md)|兼容[aclnnAlltoAllvGroupedMatMul](../../mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMul.md)支持的功能，在此基础上新增commMode参数，供用户指定通信引擎参数。|默认确定性实现| 默认确定性实现 |
|[aclnnAlltoAllvQuantGroupedMatMul](../../mc2/allto_allv_quant_grouped_mat_mul/docs/aclnnAlltoAllvQuantGroupedMatMul.md)|完成路由专家AlltoAllv、Permute、量化GroupedMatMul计算融合并实现与共享专家量化MatMul并行融合。| - | 默认确定性实现 |
|[aclnnAlltoAllvQuantGroupedMatMulV2](../../mc2/allto_allv_quant_grouped_mat_mul/docs/aclnnAlltoAllvQuantGroupedMatMulV2.md)|兼容[aclnnAlltoAllvQuantGroupedMatMul](../../mc2/allto_allv_quant_grouped_mat_mul/docs/aclnnAlltoAllvQuantGroupedMatMul.md)支持的功能，在此基础上新增commMode参数，供用户指定通信引擎参数。| - | 默认确定性实现 |
|[aclnnApplyRotaryPosEmb](../../posembedding/apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmb.md)|将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。|默认确定性实现| 默认确定性实现 |
|[aclnnApplyRotaryPosEmbGrad](../../posembedding/apply_rotary_pos_emb_grad/docs/aclnnApplyRotaryPosEmbGrad.md)|执行双路旋转位置编码aclnnApplyRotaryPosEmb的反向计算，同时计算query和key的rope反向梯度，融合为一次kernel调用。| - | 默认确定性实现 |
|[aclnnApplyRotaryPosEmbV2](../../posembedding/apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmbV2.md)|将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。|默认确定性实现| 默认确定性实现 |
|[aclnnAttentionUpdate](../../attention/attention_update/docs/aclnnAttentionUpdate.md)|将各SP域PA算子的输出的中间结果lse，localOut两个局部变量结果更新成全局结果。|默认确定性实现| 默认确定性实现 |
|[aclnnAttentionWorkerCombine](../../attention/attention_worker_combine/docs/aclnnAttentionWorkerCombine.md)|Attention和FFN分离部署场景下，Attention侧数据融合算子，将多个计算单元处理的注意力token数据进行融合，结合专家权重对结果进行加权，输出最终的注意力融合结果，并更新层ID。|默认确定性实现| - |
|[aclnnBatchMatMulReduceScatterAlltoAll](../../mc2/batch_mat_mul_reduce_scatter_allto_all/docs/aclnnBatchMatMulReduceScatterAlltoAll.md)|BatchMatMulReduceScatterAllToAll是通算融合算子，实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行的算子。|默认确定性实现| - |
|[aclnnBlitzSparseAttention](../../experimental/attention/blitz_sparse_attention/docs/aclnnBlitzSparseAttention.md)|全量推理场景的FlashAttention算子，支持sparse优化、actualSeqLengthsKv优化、int8量化功能、innerPrecise参数（用于支持高精度或者高性能模式选择）。|-|-|
|[aclnnBlockAttnResPrepare](../../attention/block_attn_res_prepare/docs/aclnnBlockAttnResPrepare.md)|计算历史残差块与伪 Query 的块间注意力，输出 softmax 加权分子、logit 最大值和指数和，供后续阶段融合当前残差。|-|默认确定性实现|
|[aclnnBlockAttnResUpdate](../../attention/block_attn_res_update/docs/aclnnBlockAttnResUpdate.md)|将`delta`原地累加到`partialBlockRef`，计算更新后`partialBlockRef`的RMSNorm score，并与历史online softmax中间状态合并，输出当前层结果`h`。|-|默认确定性实现|
|[aclnnBlockSparseAttention](../../attention/block_sparse_attention/docs/aclnnBlockSparseAttention.md)|BlockSparseAttention通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。|默认确定性实现|默认确定性实现|
|[aclnnBlockSparseAttentionV2](../../attention/block_sparse_attention/docs/aclnnBlockSparseAttentionV2.md)|BlockSparseAttention通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。|默认确定性实现|默认确定性实现|
|[aclnnBlockSparseAttentionV3](../../attention/block_sparse_attention/docs/aclnnBlockSparseAttentionV3.md)|BlockSparseAttention通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算，支持MXFP4量化模式。|默认确定性实现|默认确定性实现|
|[aclnnBlockSparseAttentionGrad](../../attention/block_sparse_attention_grad/docs/aclnnBlockSparseAttentionGrad.md)|BlockSparseAttentionGrad通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。|默认确定性实现| - |
|[aclnnBSASelectBlockMask](../../attention/bsa_select_block_mask/docs/aclnnBSASelectBlockMask.md)|BSASelectBlockMask是BSA（BlockSparseAttention）的前置算子，根据Query和Key的内容动态生成blockSparseMask，使BSA的调用链从"手动提供掩码"变为"根据Q/K内容自适应选择稀疏模式"。|默认确定性实现| - |
|[aclnnCausalConv1dFn](../../mamba/causal_conv1d/docs/aclnnCausalConv1dFn.md)|完成因果一维卷积（Causal Conv1d）的前向计算（prefill / chunk-prefill）。算子内部支持SiLU激活、缓存索引（cacheIndices）、初始状态模式（initialStateMode）等特性。| - | 默认确定性实现 |
|[aclnnCausalConv1dUpdate](../../mamba/causal_conv1d/docs/aclnnCausalConv1dUpdate.md)|完成因果一维卷积（Causal Conv1d）的状态更新（decode / update）计算。算子内部支持SiLU激活、缓存索引（cacheIndices）、投机解码（numAcceptedTokens）等特性。| - | 默认确定性实现 |
|[aclnnFusedCausalConv1d](../../attention/fused_causal_conv1d/docs/aclnnFusedCausalConv1d.md)|对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接等特性。| - | 默认确定性实现 |
|[aclnnFusedCausalConv1dV2](../../attention/fused_causal_conv1d/docs/aclnnFusedCausalConv1dV2.md)|对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接等特性。| - | 默认确定性实现 |
|[aclnnInplaceFusedCausalConv1d](../../attention/inplace_fused_causal_conv1d/docs/aclnnInplaceFusedCausalConv1d.md)|对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接、原地更新等特性。| - | 默认确定性实现 |
|[aclnnInplaceFusedCausalConv1dV2](../../attention/inplace_fused_causal_conv1d/docs/aclnnInplaceFusedCausalConv1dV2.md)|对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接、原地更新等特性。| - | 默认确定性实现 |
|[aclnnAttentionToFFN](../../mc2/attention_to_ffn/docs/aclnnAttentionToFFN.md)|将Attention节点上数据发往FFN节点。|默认确定性实现| - |
|[aclnnChunkGatedDeltaRule](../../attention/chunk_gated_delta_rule/docs/aclnnChunkGatedDeltaRule.md)|完成chunk版的Gated Delta Rule计算。|默认确定性实现|默认确定性实现|
|[aclnnChunkKdaFwd](../../attention/chunk_kda_fwd/docs/aclnnChunkKdaFwd.md)|完成不涉及CP切分的KDA分块正向计算。|默认确定性实现|默认确定性实现|
|[aclnnCompressor](../../attention/compressor/docs/aclnnCompressor.md)|Compressor将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。|默认确定性实现| 默认确定性实现 |
|[aclnnCompressorGrad](../../attention/compressor_grad/docs/aclnnCompressorGrad.md)|Compressor的反向算子，用于计算输入X、权重W^KV/W^Gate与位置编码Ape的梯度，前向在gradEnabled为true时导出softmax_score与kv中间结果作为本算子输入。|- | 默认确定性实现 |
|[aclnnFFNToAttention](../../mc2/ffn_to_attention/docs/aclnnFFNToAttention.md)|将FFN节点上的token数据发往Attention节点。|默认确定性实现| - |
|[aclnnFfnWorkerBatching](../../ffn/ffn_worker_batching/docs/aclnnFfnWorkerBatching.md)|Attention和FFN分离场景下，FFN侧数据扫描及token重排。|默认确定性实现| - |
|[aclnnDequantRopeQuantKvcache](../../posembedding/dequant_rope_quant_kvcache/docs/aclnnDequantRopeQuantKvcache.md)|对输入张量进行dequant后，对尾轴进行切分，划分为q、k、vOut，对q、k进行旋转位置编码，并进行量化。|默认确定性实现| 默认确定性实现 |
|[aclnnDistributeBarrier](../../mc2/distribute_barrier/docs/aclnnDistributeBarrier.md)|完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。|默认确定性实现| 默认确定性实现 |
|[aclnnDistributeBarrierV2](../../mc2/distribute_barrier/docs/aclnnDistributeBarrierV2.md)|完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。|默认确定性实现| 默认确定性实现 |
|[aclnnDenseLightningIndexerGradKLLoss](../../attention/dense_lightning_indexer_grad_kl_loss/docs/aclnnDenseLightningIndexerGradKLLoss.md)|dense场景LightningIndexer的反向算子，再额外融合了Loss计算功能。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnDenseLightningIndexerSoftmaxLse](../../attention/dense_lightning_indexer_softmax_lse/docs/aclnnDenseLightningIndexerSoftmaxLse.md)|dense场景DenseLightningIndexerGradKlLoss算子计算Softmax输入的一个分支算子。|默认确定性实现| 默认确定性实现 |
|[aclnnDenseLightningIndexerSoftmaxLseV2](../../attention/dense_lightning_indexer_softmax_lse_v2/docs/aclnnDenseLightningIndexerSoftmaxLseV2.md)|dense场景DenseLightningIndexerGradKlLoss算子计算Softmax输入的一个分支算子。相比aclnnDenseLightningIndexerSoftmaxLse，新增了压缩注意力（Compressed Attention）支持，并支持通过metadata前置算子进行分核负载均衡。|- | 默认确定性实现 |
|[aclnnDenseLightningIndexerSoftmaxLseV2Metadata](../../attention/dense_lightning_indexer_softmax_lse_v2_metadata/docs/aclnnDenseLightningIndexerSoftmaxLseV2Metadata.md)| aclnnDenseLightningIndexerSoftmaxLseV2接口的前置接口，用于计算aclnnDenseLightningIndexerSoftmaxLseV2的负载均衡。| - | 默认确定性实现 |
|[aclnnFFN](../../ffn/ffn/docs/aclnnFFN.md)|该FFN算子提供MoeFFN和FFN的计算功能。|默认非确定性实现，支持配置开启| - |
|[aclnnFFNV2](../../ffn/ffn/docs/aclnnFFNV2.md)|该FFN算子提供MoeFFN和FFN的计算功能。|默认非确定性实现，支持配置开启| - |
|[aclnnFFNV3](../../ffn/ffn/docs/aclnnFFNV3.md)|该FFN算子提供MoeFFN和FFN的计算功能。|默认非确定性实现，支持配置开启| - |
|[aclnnFlashAttentionScore](../../attention/flash_attention_score/docs/aclnnFlashAttentionScore.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现| 默认确定性实现 |
|[aclnnFlashAttentionScoreV2](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV2.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现| 默认确定性实现 |
|[aclnnFlashAttentionScoreV3](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV3.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。对标竞品适配gptoss模型支持sink功能。|默认确定性实现| 默认确定性实现 |
|[aclnnFlashAttentionScoreV4](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV4.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。对标竞品适配gptoss模型支持sink功能。|默认确定性实现| 默认确定性实现 |
|[aclnnFlashAttentionScoreGrad](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGrad.md)|训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnFlashAttentionScoreGradV2](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV2.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionScoreV2](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV2.md)的反向计算。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnFlashAttentionScoreGradV3](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV3.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionScoreV3](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV3.md)的反向计算。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnFlashAttentionScoreGradV4](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV4.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionScoreV4](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV4.md)的反向计算。该接口query、key、value参数支持多个长度相等或者长度不相等的sequence。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnFlashAttentionUnpaddingScoreGrad](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGrad.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionVarLenScore](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScore.md)的反向计算。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnFlashAttentionUnpaddingScoreGradV2](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV2.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionVarLenScoreV2](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV2.md)的反向计算。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnFlashAttentionUnpaddingScoreGradV3](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV3.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionVarLenScoreV3](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV3.md)的反向计算。该接口相较于[aclnnFlashAttentionUnpaddingScoreGradV2](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV2.md)接口，新增queryRope、keyRope、dqRope和dkRope参数。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnFlashAttentionUnpaddingScoreGradV4](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV4.md)|训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnFlashAttentionUnpaddingScoreGradV5](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV5.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。增加`sinkInOptional`可选输入。本接口为反向输出。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnFlashAttentionVarLenScore](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScore.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现| 默认确定性实现 |
|[aclnnFlashAttentionVarLenScoreV2](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV2.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现| 默认确定性实现 |
|[aclnnFlashAttentionVarLenScoreV3](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV3.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现| 默认确定性实现 |
|[aclnnFlashAttentionVarLenScoreV4](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV4.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现| 默认确定性实现 |
|[aclnnFlashAttentionVarLenScoreV5](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV5.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。对标竞品适配gptoss模型支持sink功能。|默认确定性实现| 默认确定性实现 |
|[aclnnFusedInferAttentionScoreV4](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV4.md)|适配Decode & Prefill场景的FlashAttention算子。|默认确定性实现| - |
|[aclnnFusedInferAttentionScoreV5](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md)|适配增量&全量推理场景的FlashAttention算子。|默认确定性实现| 默认确定性实现 |
|[aclnnGatherPaKvCache](../../attention/gather_pa_kv_cache/docs/aclnnGatherPaKvCache.md)|根据blockTables中的blockId值、seqLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。|默认确定性实现| 默认确定性实现 |
|[aclnnGenericBlockSparseAttention](../../attention/generic_block_sparse_attention/docs/aclnnGenericBlockSparseAttention.md)|GenericBlockSparseAttention通过sparseBlockIdx指定每个Q块选择的KV块，sparseBlockCount指定每个Q块保留的KV块数量，实现高效的稀疏注意力计算。|默认确定性实现|默认确定性实现|
|[aclnnGenericBlockSparseAttentionMetadata](../../attention/generic_block_sparse_attention_metadata/docs/aclnnGenericBlockSparseAttentionMetadata.md)|aclnnGenericBlockSparseAttention接口的前置接口，用于计算aclnnGenericBlockSparseAttention的负载均衡。|默认确定性实现| 默认确定性实现 |
|[aclnnGroupedMatmulV5](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现| 默认确定性实现 |
|[aclnnGroupedMatmulAdd](../../gmm/grouped_matmul_add/docs/aclnnGroupedMatmulAdd.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现| 默认确定性实现 |
|[aclnnGroupedMatmulAddV2](../../gmm/grouped_matmul_add/docs/aclnnGroupedMatmulAddV2.md)|实现分组矩阵乘计算后原地累加，每组矩阵乘的维度大小可以不同。|默认确定性实现|-|
|[aclnnGroupedMatmulActivationQuantWeightNz](../../gmm/grouped_matmul_activation_quant/docs/aclnnGroupedMatmulActivationQuantWeightNz.md)|融合GMM、激活函数和量化算子，完成分组矩阵乘、激活和量化计算，输出量化结果及量化因子。|-| 默认确定性实现 |
|[aclnnGroupedMatMulAlltoAllv](../../mc2/grouped_mat_mul_allto_allv/docs/aclnnGroupedMatMulAlltoAllv.md)|完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合。|默认确定性实现| 默认确定性实现 |
|[aclnnGroupedMatMulAlltoAllvV2](../../mc2/grouped_mat_mul_allto_allv/docs/aclnnGroupedMatMulAlltoAllvV2.md)|兼容[aclnnGroupedMatMulAlltoAllv](../../mc2/grouped_mat_mul_allto_allv/docs/aclnnGroupedMatMulAlltoAllv.md)支持的功能，在此基础上新增commMode参数，供用户指定通信引擎参数。|默认确定性实现| 默认确定性实现 |
|[aclnnQuantGroupedMatMulAlltoAllv](../../mc2/quant_grouped_mat_mul_allto_allv/docs/aclnnQuantGroupedMatMulAlltoAllv.md)|完成路由专家量化GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家量化MatMul并行融合。| - | 默认确定性实现 |
|[aclnnQuantGroupedMatMulAlltoAllvV2](../../mc2/quant_grouped_mat_mul_allto_allv/docs/aclnnQuantGroupedMatMulAlltoAllvV2.md)|兼容[aclnnQuantGroupedMatMulAlltoAllv](../../mc2/quant_grouped_mat_mul_allto_allv/docs/aclnnQuantGroupedMatMulAlltoAllv.md)支持的功能，在此基础上新增commMode参数，供用户指定通信引擎参数。| - | 默认确定性实现 |
|[aclnnGroupedMatmulFinalizeRouting](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRouting.md)|GroupedMatMul和MoeFinalizeRouting的融合算子。|默认非确定性实现，支持配置开启| - |
|[aclnnGroupedMatmulFinalizeRoutingV2](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingV2.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。|默认非确定性实现，支持配置开启| - |
|[aclnnGroupedMatmulFinalizeRoutingV3](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingV3.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnGroupedMatmulFinalizeRoutingWeightNz](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingWeightNz.md)|GroupedMatMul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持输入Weight为AI处理器亲和数据排布格式(NZ)。|默认非确定性实现，支持配置开启| - |
|[aclnnGroupedMatmulFinalizeRoutingWeightNzV2](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingWeightNzV2.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持输入Weight为AI处理器亲和数据排布格式(NZ)。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnGroupedMatmulSwigluQuant](../../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md)|融合GroupedMatMul、Dequant、Swiglu和Quant。|默认确定性实现| - |
|[aclnnGroupedMatmulSwigluQuantV2](../../gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md)|融合GroupedMatmul 、Dequant、Swiglu和Quant。|默认确定性实现| 默认确定性实现 |
|[aclnnGroupedMatmulSwigluQuantWeightNZ](../../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md)|融合GroupedMatMul、Dequant、Swiglu和Quant，输入权重Weight会被强制视为NZ格式。|默认确定性实现| - |
|[aclnnGroupedMatmulSwigluQuantWeightNzV2](../../gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantWeightNzV2.md)|融合GroupedMatMul、Dequant、Swiglu和Quant，输入权重Weight会被强制视为NZ格式。|默认确定性实现| 默认确定性实现 |
|[aclnnGroupedMatmulWeightNz](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同，输入权重Weight会被强制视为NZ格式。|默认确定性实现| 默认确定性实现 |
|[aclnnIncreFlashAttentionV4](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)|在全量推理场景的FlashAttention算子的基础上实现增量推理。|默认确定性实现| - |
|[aclnnIndexerQuantCache](../../attention/indexer_quant_cache/docs/aclnnIndexerQuantCache.md)|在Indexer注意力机制的Epilog阶段，对KV Cache进行原地逐块动态量化压缩更新，支持MX-FP8、Normal、HiFloat8、MX-FP4四种量化模式。|- | 默认确定性实现 |
|[aclnnInplaceAttentionWorkerScheduler](../../attention/attention_worker_scheduler/docs/aclnnInplaceAttentionWorkerScheduler.md)|Attention和FFN分离部署场景下，Attention侧数据扫描算子。该算子接收来自FFNToAttention算子的输出数据，并对数据进行逐步扫描，确保数据准备就绪。|默认确定性实现| 默认确定性实现 |
|[aclnnInplaceFfnWorkerScheduler](../../ffn/ffn_worker_scheduler/docs/aclnnInplaceFfnWorkerScheduler.md)|Attention和FFN分离场景下，FFN侧数据扫描算子。该算子接收AttentionToFFN算子发送的数据，进行扫描并完成数据整理。|默认确定性实现| 默认确定性实现 |
|[aclnnInplacePartialRotaryMul](../../posembedding/inplace_partial_rotary_mul/docs/aclnnInplacePartialRotaryMul.md)|执行单路旋转位置编码的Inplace计算，直接修改输入张量，不产生新的输出张量。|默认确定性实现| 默认确定性实现 |
|[aclnnInplacePartialRotaryMulGrad](../../posembedding/inplace_partial_rotary_mul_grad/docs/aclnnInplacePartialRotaryMulGrad.md)|执行局部旋转位置编码InplacePartialRotaryMul的反向计算，对输入dy的D维度上切片[start, end)区域执行旋转位置编码梯度计算，结果inplace写回dy。|- | 默认确定性实现 |
|[aclnnInterleaveRope](../../posembedding/interleave_rope/docs/aclnnInterleaveRope.md)|针对单输入x进行旋转位置编码。|- | 默认确定性实现 |
|[aclnnKvCompressEpilog](../../attention/kv_compress_epilog/docs/aclnnKvCompressEpilog.md)|在KV Cache的Epilog阶段，对缓存进行原地逐组动态量化压缩更新，将bfloat16激活值压缩为FP8格式并按slotMapping散写。|- | 默认确定性实现 |
|[aclnnLightningIndexer](../../attention/lightning_indexer/docs/aclnnLightningIndexer.md)|稀疏attention前处理的计算，目的是选出关键的稀疏token位置。|默认确定性实现| 默认确定性实现 |
|[aclnnLightningIndexerGrad](../../attention/lightning_indexer_grad/docs/aclnnLightningIndexerGrad.md)|训练场景下，实现LightningIndexer反向，其中输入有Query, Key, Weights, Dy, Indices，反向主要利用正向计算的Indices从Key中提取TopK序列从而降低Matmul计算量。|默认非确定性实现，不支持配置开启| -|
|[aclnnLightningIndexerKLLoss](../../attention/lightning_indexer_kl_loss/docs/aclnnLightningIndexerKLLoss.md)|训练场景下，计算LightningIndexer中teacher分布与student分布之间的KL散度损失函数。|默认非确定性实现，支持配置开启| - |
|[aclnnLightningIndexerV2](../../attention/lightning_indexer_v2/docs/aclnnLightningIndexerV2.md)|稀疏attention前处理的计算，目的是选出关键的稀疏token位置。支持KV压缩场景。|默认确定性实现| 默认确定性实现 |
|[aclnnLightningIndexerV2Metadata](../../attention/lightning_indexer_v2_metadata/docs/aclnnLightningIndexerV2Metadata.md)| aclnnLightningIndexerV2接口的前置接口，用于计算aclnnLightningIndexerV2的负载均衡。|默认确定性实现|默认确定性实现|
|[aclnnMaskedCausalConv1d](../../attention/masked_causal_conv1d/docs/aclnnMaskedCausalConv1d.md)|对hidden层的token之间进行带mask的因果一维分组卷积操作。| - | 默认确定性实现 |
|[aclnnMaskedCausalConv1dBackward](../../attention/masked_causal_conv1d_backward/docs/aclnnMaskedCausalConv1dBackward.md)|对hidden层的token之间进行一维分组卷积操作的反向梯度计算。| - | 默认确定性实现 |
|[aclnnMatmulAlltoAll](../../mc2/matmul_allto_all/docs/aclnnMatmulAlltoAll.md)|完成MatMul计算与AlltoAll通信融合。|默认确定性实现| 默认确定性实现 |
|[aclnnMatmulAlltoAllV2](../../mc2/matmul_allto_all/docs/aclnnMatmulAlltoAllV2.md)|兼容[aclnnMatmulAlltoAll](../../mc2/matmul_allto_all/docs/aclnnMatmulAlltoAll.md)支持的功能，在此基础上新增commMode参数，供用户指定通信引擎参数。|默认确定性实现| 默认确定性实现 |
|[aclnnMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnMatmulAllReduce.md)|完成MatMul计算与AllReduce通信融合。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnMatmulAllReduceV2](../../mc2/matmul_all_reduce/docs/aclnnMatmulAllReduceV2.md)|完成MatMul计算与AllReduce通信融合。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnMatmulReduceScatter](../../mc2/matmul_reduce_scatter/docs/aclnnMatmulReduceScatter.md)|完成mm + reduce_scatter_base计算。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnMatmulReduceScatterV2](../../mc2/matmul_reduce_scatter_v2/docs/aclnnMatmulReduceScatterV2.md)|aclnnMatmulReduceScatterV2接口是对[aclnnMatmulReduceScatter](../../mc2/matmul_reduce_scatter/docs/aclnnMatmulReduceScatter.md)接口的功能扩展。|默认确定性实现| 默认确定性实现 |
|[aclnnMixedQuantSparseFlashMla](../../attention/mixed_quant_sparse_flash_mla/docs/aclnnMixedQuantSparseFlashMla.md)|支持量化场景下SWA、CSA、HCA三类Attention计算场景。|-| 默认确定性实现 |
|[aclnnMixedQuantSparseFlashMlaMetadata](../../attention/mixed_quant_sparse_flash_mla_metadata/docs/aclnnMixedQuantSparseFlashMlaMetadata.md)| aclnnMixedQuantSparseFlashMla接口的前置接口，用于计算aclnnMixedQuantSparseFlashMla的负载均衡。| - | 默认确定性实现 |
|[aclnnQuantSparseFlashMla](../../attention/quant_sparse_flash_mla/docs/aclnnQuantSparseFlashMla.md)|支持全量化场景下SWA、CSA、HCA三类Attention计算场景。|-| 默认确定性实现 |
|[aclnnQuantSparseFlashMlaMetadata](../../attention/quant_sparse_flash_mla_metadata/docs/aclnnQuantSparseFlashMlaMetadata.md)| aclnnQuantSparseFlashMla接口的前置接口，用于计算aclnnQuantSparseFlashMla的负载均衡。| - | 默认确定性实现 |
|[aclnnMlaPreprocess](../../attention/mla_preprocess/docs/aclnnMlaPreprocess.md)|Multi-Head Latent Attention前处理的计算。|默认确定性实现| - |
|[aclnnMlaPreprocessV2](../../attention/mla_preprocess_v2/docs/aclnnMlaPreprocessV2.md)|推理场景，Multi-Head Latent Attention前处理的计算。|默认确定性实现| - |
|[aclnnMlaProlog](../../attention/mla_prolog/docs/aclnnMlaProlog.md)|Multi-Head Latent Attention前处理的计算。|默认确定性实现| - |
|[aclnnMlaPrologV2WeightNz](../../attention/mla_prolog_v2/docs/aclnnMlaPrologV2WeightNz.md)|Multi-Head Latent Attention前处理的计算。|默认确定性实现| - |
|[aclnnMlaPrologV3WeightNz](../../attention/mla_prolog_v3/docs/aclnnMlaPrologV3WeightNz.md)|Multi-Head Latent Attention前处理的计算。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnMlaPrologV4WeightNz](../../attention/mla_prolog_v3/docs/aclnnMlaPrologV4WeightNz.md)|Multi-Head Latent Attention前处理的计算。支持doRope参数控制是否执行旋转位置编码。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnMoeComputeExpertTokens](../../moe/moe_compute_expert_tokens/docs/aclnnMoeComputeExpertTokens.md)|MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeCombine](../../mc2/moe_distribute_combine/docs/aclnnMoeDistributeCombine.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeCombineV2](../../mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV2.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合。expertScales非空时乘权重再相加，为空Tensor时直接相加。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeCombineV3](../../mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV3.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合。expertScales非空时乘权重再相加，为空Tensor时直接相加。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeCombineV4](../../mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV4.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合。expertScales非空时乘权重再相加，为空Tensor时直接相加。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeCombineAddRmsNorm](../../mc2/moe_distribute_combine_add_rms_norm/docs/aclnnMoeDistributeCombineAddRmsNorm.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合，之后完成Add + RmsNorm融合。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeCombineAddRmsNormV2](../../mc2/moe_distribute_combine_add_rms_norm/docs/aclnnMoeDistributeCombineAddRmsNormV2.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合，之后完成Add + RmsNorm融合。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeCombineSetup](../../mc2/moe_distribute_combine_setup/docs/aclnnMoeDistributeCombineSetup.md)| 进行AlltoAllV通信，将数据写入对端GM。| - | 默认确定性实现 |
|[aclnnMoeDistributeCombineTeardown](../../mc2/moe_distribute_combine_teardown/docs/aclnnMoeDistributeCombineTeardown.md)| 接收aclnnMoeDistributeCombineSetup发来的数据并整合数据（乘权重再相加）。 | - | 默认确定性实现 |
|[aclnnMoeDistributeDispatch](../../mc2/moe_distribute_dispatch/docs/aclnnMoeDistributeDispatch.md)|对Token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeDispatchV2](../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV2.md)|对token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeDispatchV3](../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3.md)|对token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeDispatchV4](../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md)|对token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeDistributeDispatchSetup](../../mc2/moe_distribute_dispatch_setup/docs/aclnnMoeDistributeDispatchSetup.md)|对Token数据进行量化（可选），根据token选择的topK专家在EP（Expert Parallelism）域的AllToAllV通信，只进行数据发送和通信状态发送，通信指令发出后算子即刻退出，无需等待通信完成。数据的接收和后处理由aclnnMoeDistributeDispatchTeardown接口完成。| - | 默认确定性实现 |
|[aclnnMoeDistributeDispatchTeardown](../../mc2/moe_distribute_dispatch_teardown/docs/aclnnMoeDistributeDispatchTeardown.md)| 接收MoE层EP（Expert Parallelism）域的AllToAllV通信发过来的数据，数据发送端由aclnnMoeDistributeDispatchSetup完成，本接口内完成通信状态确认和数据整理。| - | 默认确定性实现 |
|[aclnnMoeFinalizeRouting](../../moe/moe_finalize_routing/docs/aclnnMoeFinalizeRouting.md)|MoE计算中，最后处理合并MoE FFN的输出结果。|默认确定性实现| - |
|[aclnnMoeFinalizeRoutingV2](../../moe/moe_finalize_routing_v2/docs/aclnnMoeFinalizeRoutingV2.md)|MoE计算中，最后处理合并MoE FFN的输出结果，支持配置dropPadMode。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeFinalizeRoutingV2Grad](../../moe/moe_finalize_routing_v2_grad/docs/aclnnMoeFinalizeRoutingV2Grad.md)|aclnnMoeFinalizeRoutingV2的反向传播。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeFinalizeRoutingV3](../../moe/moe_finalize_routing_v2/docs/aclnnMoeFinalizeRoutingV3.md)| MoE计算中，最后处理合并MoE FFN的输出结果。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeFusedTopk](../../moe/moe_fused_topk/docs/aclnnMoeFusedTopk.md)|MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。|默认确定性实现| 默认确定性实现 |
| [aclnnMoeGatingTopK](../../moe/moe_gating_top_k/docs/aclnnMoeGatingTopK.md)|MoE计算中，对输入x做Sigmoid、SoftMax或者SqrtSoftplus计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。|默认确定性实现| 默认确定性实现 |
| [aclnnMoeGatingTopKV2](../../moe/moe_gating_top_k/docs/aclnnMoeGatingTopKV2.md)|MoE计算中，对输入x做Sigmoid、SoftMax或者SqrtSoftplus计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家，支持两种模式。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeGatingTopKBackward](../../moe/moe_gating_top_k_backward/docs/aclnnMoeGatingTopKBackward.md)|aclnnMoeGatingTopK的反向算子。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeGatingTopKSoftmax](../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md)|MoE计算中，对x的输出做Softmax计算，取TopK操作。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeGatingTopKSoftmaxV2](../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)|MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取TopK操作；如果renorm=1，先对x的输出做TopK操作，再进行Softmax操作。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeInitRouting](../../moe/moe_init_routing/docs/aclnnMoeInitRouting.md)|MoE的routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做Routing处理。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeInitRoutingV2](../../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)|该算子对应MoE中的Routing计算，以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeInitRoutingV3](../../moe/moe_init_routing_v3/docs/aclnnMoeInitRoutingV3.md)|MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理，支持不量化和动态量化模式。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeInitRoutingV4](../../moe/moe_init_routing_v4/docs/aclnnMoeInitRoutingV4.md)|MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理，支持非量化、静态量化和动态量化模式。| - | 默认确定性实现 |
|[aclnnMoeInitRoutingQuant](../../moe/moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant.md)|MoE的Routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做Routing处理，并对结果进行量化。|默认确定性实现| - |
|[aclnnMoeInitRoutingQuantV2](../../moe/moe_init_routing_quant_v2/docs/aclnnMoeInitRoutingQuantV2.md)|MoE的Routing计算，根据aclnnMoeGatingTopKSoftmaxV2的计算结果做Routing处理。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeInitRoutingV2Grad](../../moe/moe_init_routing_v2_grad/docs/aclnnMoeInitRoutingV2Grad.md)|[aclnnMoeInitRoutingV2](../../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)的反向传播，完成Tokens的加权求和。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeTokenPermute](../../moe/moe_token_permute/docs/aclnnMoeTokenPermute.md)|MoE的permute计算，根据索引indices将tokens广播并排序。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeTokenPermuteGrad](../../moe/moe_token_permute_grad/docs/aclnnMoeTokenPermuteGrad.md)|[aclnnMoeTokenPermute](../../moe/moe_token_permute/docs/aclnnMoeTokenPermute.md)的反向传播计算。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeTokenPermuteWithEp](../../moe/moe_token_permute_with_ep/docs/aclnnMoeTokenPermuteWithEp.md)|MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。|默认确定性实现| - |
|[aclnnMoeTokenPermuteWithEpGrad](../../moe/moe_token_permute_with_ep_grad/docs/aclnnMoeTokenPermuteWithEpGrad.md)|[aclnnMoeTokenPermuteWithEp](../../moe/moe_token_permute_with_ep/docs/aclnnMoeTokenPermuteWithEp.md)的反向传播计算。|默认确定性实现| - |
|[aclnnMoeTokenPermuteWithRoutingMap](../../moe/moe_token_permute_with_routing_map/docs/aclnnMoeTokenPermuteWithRoutingMap.md)|MoE的permute计算，将token和expert的标签作为routingMap传入，根据routingMaps将tokens和可选probsOptional广播后排序|默认确定性实现| 默认确定性实现 |
|[aclnnMoeTokenPermuteWithRoutingMapGrad](../../moe/moe_token_permute_with_routing_map_grad/docs/aclnnMoeTokenPermuteWithRoutingMapGrad.md)|[aclnnMoeTokenPermuteWithRoutingMap](../../moe/moe_token_permute_with_routing_map/docs/aclnnMoeTokenPermuteWithRoutingMap.md)的反向传播。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnMoeTokenUnpermute](../../moe/moe_token_unpermute/docs/aclnnMoeTokenUnpermute.md)|根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘；最后进行累加求和，并输出计算结果。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeTokenUnpermuteGrad](../../moe/moe_token_unpermute_grad/docs/aclnnMoeTokenUnpermuteGrad.md)|[aclnnMoeTokenUnpermute](../../moe/moe_token_unpermute/docs/aclnnMoeTokenUnpermute.md)的反向传播。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeTokenUnpermuteWithEp](../../moe/moe_token_unpermute_with_ep/docs/aclnnMoeTokenUnpermuteWithEp.md)|根据sortedIndices存储的下标位置，去获取permutedTokens中的输入数据与probs相乘，并进行合并累加。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeTokenUnpermuteWithEpGrad](../../moe/moe_token_unpermute_with_ep_grad/docs/aclnnMoeTokenUnpermuteWithEpGrad.md)|[aclnnMoeTokenUnpermuteWithEp](../../moe/moe_token_unpermute_with_ep/docs/aclnnMoeTokenUnpermuteWithEp.md)的反向传播。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeTokenUnpermuteWithRoutingMap](../../moe/moe_token_unpermute_with_routing_map/docs/aclnnMoeTokenUnpermuteWithRoutingMap.md)|对经过aclnnMoeTokenUpermuteWithRoutingMap处理的permutedTokens，累加回原unpermutedTokens。|默认确定性实现| 默认确定性实现 |
|[aclnnMoeTokenUnpermuteWithRoutingMapGrad](../../moe/moe_token_unpermute_with_routing_map_grad/docs/aclnnMoeTokenUnpermuteWithRoutingMapGrad.md)|[aclnnMoeTokenUnpermuteWithRoutingMap](../../moe/moe_token_unpermute_with_routing_map/docs/aclnnMoeTokenUnpermuteWithRoutingMap.md)的反向传播。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnMoeUpdateExpert](../../mc2/moe_update_expert/docs/aclnnMoeUpdateExpert.md)|本API支持负载均衡和专家剪枝功能。经过映射后的专家表和Mask可传入MoE层进行数据分发和处理。|默认确定性实现| 默认确定性实现 |
|[aclnnMhcPre](../../mhc/mhc_pre/docs/aclnnMhcPre.md)|基于一系列计算得到MHC架构中hidden层的$H^{res}$和$H^{post}$投影矩阵以及Attention或MLP层的输入矩阵$h^{in}$。|默认确定性实现|默认确定性实现|
|[aclnnMhcPreV2](../../mhc/mhc_pre/docs/aclnnMhcPreV2.md)|兼容[aclnnMhcPre](../../mhc/mhc_pre/docs/aclnnMhcPre.md)支持的功能，在此基础上新增opImplMode参数，供用户选择MhcPre计算模式。| - |默认确定性实现|
|[aclnnMhcPreBackward](../../mhc/mhc_pre_backward/docs/aclnnMhcPreBackward.md)|[aclnnMhcPre](../../mhc/mhc_pre/docs/aclnnMhcPre.md)的反向传播。|默认确定性实现|默认确定性实现|
|[aclnnMhcPreSinkhorn](../../mhc/mhc_pre_sinkhorn/docs/aclnnMhcPreSinkhorn.md)| 基于一系列计算得到MHC架构中hidden层的$\mathbf{H}'_{\text{res}}$和$\mathbf{H}_{\text{post}}$投影矩阵以及Attention或MLP层的输入矩阵$\mathbf{h}_{\text{in}}$。对$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出中间计算结果，用于反向梯度计算。 |默认确定性实现| - |
|[aclnnMhcPost](../../mhc/mhc_post/docs/aclnnMhcPost.md)|基于一系列计算对mHC架构中上一层输出进行Post Mapping，对上一层的输入进行Res Mapping，然后对二者进行残差连接，得到下一层的输入。|默认确定性实现| 默认确定性实现 |
|[aclnnMhcPostBackward](../../mhc/mhc_post_backward/docs/aclnnMhcPostBackward.md)|mhc_post基于一系列计算对mHC架构中上一层输出进行Post Mapping，对上一层的输入进行Res Mapping，然后对二者进行残差连接，得到下一层的输入。该算子实现前述过程的反向。|默认确定性实现| 默认确定性实现 |
|[aclnnMhcSinkhorn](../../mhc/mhc_sinkhorn/docs/aclnnMhcSinkhorn.md)| 对mHC架构中的$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出迭代过程中的中间归一化结果（norm_out）和求和结果（sum_out），用于反向梯度计算。 | - | 默认确定性实现 |
|[aclnnMhcSinkhornBackward](../../mhc/mhc_sinkhorn_backward/docs/aclnnMhcSinkhornBackward.md)|[aclnnMhcSinkhorn](../../mhc/mhc_sinkhorn/docs/aclnnMhcSinkhorn.md)的反向传播。| - | 默认确定性实现 |
|[aclnnMhcPreSinkhornBackward](../../mhc/mhc_pre_sinkhorn_backward/docs/aclnnMhcPreSinkhornBackward.md)|[aclnnMhcPreSinkhorn](../../mhc/mhc_pre_sinkhorn/docs/aclnnMhcPreSinkhorn.md)的反向传播。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnNormRopeConcat](../../posembedding/norm_rope_concat/docs/aclnnNormRopeConcat.md)|transformer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）。|默认确定性实现| 默认确定性实现 |
|[aclnnNormRopeConcatBackward](../../posembedding/norm_rope_concat_grad/docs/aclnnNormRopeConcatBackward.md)|transformer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）融合算子功能反向推导。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnNsaCompress](../../attention/nsa_compress/docs/aclnnNsaCompress.md)|训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。|默认确定性实现| - |
|[aclnnNsaCompressAttention](../../attention/nsa_compress_attention/docs/aclnnNsaCompressAttention.md)|NSA中compress attention以及select topk索引计算。|默认确定性实现| - |
|[aclnnNsaCompressAttentionInfer](../../attention/nsa_compress_attention_infer/docs/aclnnNsaCompressAttentionInfer.md)|Native Sparse Attention推理过程中，Compress Attention的计算。|默认确定性实现| - |
|[aclnnNsaCompressGrad](../../attention/nsa_compress_grad/docs/aclnnNsaCompressGrad.md)|[aclnnNsaCompress](../../attention/nsa_compress/docs/aclnnNsaCompress.md)算子的反向计算。|默认确定性实现| - |
|[aclnnNsaCompressWithCache](../../attention/nsa_compress_with_cache/docs/aclnnNsaCompressWithCache.md)|实现Native-Sparse-Attention推理阶段的KV压缩。|默认确定性实现| - |
|[aclnnNsaSelectedAttention](../../attention/nsa_selected_attention/docs/aclnnNsaSelectedAttention.md)|训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算。|默认确定性实现| - |
|[aclnnNsaSelectedAttentionGrad](../../attention/nsa_selected_attention_grad/docs/aclnnNsaSelectedAttentionGrad.md)|根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启| - |
|[aclnnNsaSelectedAttentionInfer](../../attention/nsa_selected_attention_infer/docs/aclnnNsaSelectedAttentionInfer.md)|Native Sparse Attention推理过程中，Selected Attention的计算。|默认确定性实现| - |
|[aclnnPromptFlashAttentionV3](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttentionV3.md)|全量推理场景的FlashAttention算子。|默认确定性实现| - |
|[aclnnQkvRmsNormRopeCache](../../posembedding/qkv_rms_norm_rope_cache/docs/aclnnQkvRmsNormRopeCache.md)|输入qkv融合张量，通过SplitVD拆分q、k、v张量，执行RmsNorm、ApplyRotaryPosEmb、Quant、Scatter融合操作，输出qOut、kCache、vCache、qBeforeQuant(可选)、kBeforeQuant(可选)、vBeforeQuant(可选)。|默认确定性实现| - |
|[aclnnQkvRmsNormRopeCacheWithKScale](../../posembedding/qkv_rms_norm_rope_cache_with_k_scale/docs/aclnnQkvRmsNormRopeCacheWithKScale.md)|输入Q/K/V融合张量，对Q/K执行RMSNorm、RoPE/M-RoPE及场景对应的量化，并按slotMapping更新KV Cache和K scale Cache，支持M-RoPE MX场景。|-|默认确定性实现|
|[aclnnQuantAllReduce](../../mc2/quant_all_reduce/docs/aclnnQuantAllReduce.md)|实现quant + allReduce融合计算。|- | 默认非确定性说明，支持配置开启 |
|[aclnnQuantCompressor](../../attention/quant_compressor/docs/aclnnQuantCompressor.md)|Compressor的量化版本，将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。|- | 默认确定性实现 |
|[aclnnQuantFlashAttentionScore](../../attention/flash_attention_score/docs/aclnnQuantFlashAttentionScore.md)| 量化的训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|- | 默认确定性说明 |
|[aclnnQuantGroupedMatmulDequantWeightNZ](../../gmm/quant_grouped_matmul_dequant/docs/aclnnQuantGroupedMatmulDequantWeightNZ.md)|对输入x进行量化，分组矩阵乘以及反量化，输入权重Weight会被强制视为NZ格式。| - | - |
|[aclnnQuantLightningIndexer](../../attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md)|QuantLightningIndexer在LightningIndexer的基础上支持了Per-Token-Head量化输入。| - | 默认确定性实现 |
|[aclnnQuantGroupedMatmulInplaceAdd](../../gmm/quant_grouped_matmul_inplace_add/docs/aclnnQuantGroupedMatmulInplaceAdd.md)|实现分组矩阵乘计算和加法计算，基本功能为矩阵乘和加法的组合。| - | 默认确定性实现 |
|[aclnnQuantLightningIndexerV2](../../attention/quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md)|稀疏attention前处理的计算，选出关键的稀疏token，支持KV压缩和量化输入。|默认确定性实现|默认确定性实现|
|[aclnnQuantLightningIndexerV2Metadata](../../attention/quant_lightning_indexer_v2_metadata/docs/aclnnQuantLightningIndexerV2Metadata.md)| aclnnQuantLightningIndexerV2接口的前置接口，用于计算aclnnQuantLightningIndexerV2的负载均衡。|默认确定性实现|默认确定性实现|
|[aclnnQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduce.md)|对量化后的入参x1、x2进行MatMul计算后，接着进行Dequant计算，接着与x3进行Add操作，最后做AllReduce计算。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnQuantMatmulAllReduceV2](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV2.md)|aclnnQuantMatmulAllReduceV2接口是对[aclnnQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduce.md)接口的功能扩展。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnQuantMatmulAllReduceV3](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV3.md)|aclnnQuantMatmulAllReduceV3接口是对[aclnnQuantMatmulAllReduceV2](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV2.md)接口的功能扩展。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnQuantMatmulAllReduceV4](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV4.md)|兼容[aclnnQuantMatmulAllReduceV3](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV3.md)支持的功能，在此基础上新增perblock量化方式的支持。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnQuantMatmulAlltoAll](../../mc2/matmul_allto_all/docs/aclnnQuantMatmulAlltoAll.md)|对量化后的入参x1、x2进行MatMul计算后，接着进行Dequant计算，最后做AlltoAll通信。|默认确定性实现| 默认确定性实现 |
|[aclnnQuantMatmulAlltoAllV2](../../mc2/matmul_allto_all/docs/aclnnQuantMatmulAlltoAllV2.md)|兼容[aclnnQuantMatmulAlltoAll](../../mc2/matmul_allto_all/docs/aclnnQuantMatmulAlltoAll.md)支持的功能，在此基础上新增commMode参数，供用户指定通信引擎参数。|默认确定性实现| 默认确定性实现 |
|[aclnnQuantGroupedMatmulDequant](../../gmm/quant_grouped_matmul_dequant/docs/aclnnQuantGroupedMatmulDequant.md)|对输入x进行量化，分组矩阵乘以及反量化。|默认确定性实现| 默认确定性实现 |
|[aclnnQuantReduceScatter](../../mc2/quant_reduce_scatter/docs/aclnnQuantReduceScatter.md)|实现quant + reduceScatter融合计算。|默认确定性实现| 默认确定性实现 |
|[aclnnRainFusionAttention](../../attention/rain_fusion_attention/docs/aclnnRainFusionAttention.md)|RainFusionAttention稀疏注意力计算，支持灵活的块级稀疏模式，通过selectIdx指定每个Q块选择的KV块，实现高效的稀疏注意力计算。|默认确定性实现| - |
|[aclnnRecurrentGatedDeltaRule](../../attention/recurrent_gated_delta_rule/docs/aclnnRecurrentGatedDeltaRule.md)|完成变步长的Recurrent Gated Delta Rule计算。|默认确定性实现| 默认确定性实现 |
|[aclnnRecurrentKda](../../attention/recurrent_kda/docs/aclnnRecurrentKda.md)|完成KDA（Kimi Delta Attention）的递归前向计算，面向decode和MTP短序列场景。|默认确定性实现| 默认确定性实现 |
|[aclnnRingAttentionUpdate](../../attention/ring_attention_update/docs/aclnnRingAttentionUpdate.md)|将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|默认确定性实现| 默认确定性实现 |
|[aclnnRingAttentionUpdateV2](../../attention/ring_attention_update/docs/aclnnRingAttentionUpdateV2.md)|指定softmax的输入排布，将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|默认确定性实现| 默认确定性实现 |
|[aclnnRopeWithSinCosCache](../../posembedding/rope_with_sin_cos_cache/docs/aclnnRopeWithSinCosCache.md)|推理网络为了提升性能，将sin和cos输入通过cache传入，执行旋转位置编码计算。|默认确定性实现| 默认确定性实现 |
|[aclnnRopeWithSinCosCacheV2](../../posembedding/rope_with_sin_cos_cache/docs/aclnnRopeWithSinCosCacheV2.md)|对比V1增加cacheMode属性，指示cos和sin的拼接方式。|默认确定性实现| - |
|[aclnnRotaryPositionEmbedding](../../posembedding/rotary_position_embedding/docs/aclnnRotaryPositionEmbedding.md)|执行单路旋转位置编码计算。|默认确定性实现| 默认确定性实现 |
|[aclnnRotaryPositionEmbeddingV2](../../posembedding/rotary_position_embedding/docs/aclnnRotaryPositionEmbeddingV2.md)|执行单路旋转位置编码计算。本接口相较于[aclnnRotaryPositionEmbedding](../../posembedding/rotary_position_embedding/docs/aclnnRotaryPositionEmbedding.md)，新增入参rotate。|默认确定性实现| 默认确定性实现 |
|[aclnnRotaryPositionEmbeddingGrad](../../posembedding/rotary_position_embedding_grad/docs/aclnnRotaryPositionEmbeddingGrad.md)|单路旋转位置编码[aclnnRotaryPositionEmbedding](../../posembedding/rotary_position_embedding/docs/aclnnRotaryPositionEmbedding.md)的反向计算。|默认确定性实现| 默认确定性实现 |
|[aclnnScatterPaCache](../../attention/scatter_pa_cache/docs/aclnnScatterPaCache.md)|更新KCache中指定位置的key。|-| 默认确定性实现 |
|[aclnnScatterPaKvCache](../../attention/scatter_pa_kv_cache/docs/aclnnScatterPaKvCache.md)|更新KvCache中指定位置的key和value。|默认确定性实现| 默认确定性实现 |
|[aclnnSparseFlashAttention](../../attention/sparse_flash_attention/docs/aclnnSparseFlashAttention.md)|根据sparse_indices选取重要性较高的key和value进行attention运算，得到attention_out输出。|默认确定性实现| 默认确定性实现 |
|[aclnnSparseFlashAttentionV2](../../attention/sparse_flash_attention/docs/aclnnSparseFlashAttentionV2.md)|根据sparse_indices选取重要性较高的key和value进行attention运算，得到attention_out输出。|默认确定性实现| 默认确定性实现 |
|[aclnnSparseFlashAttentionGrad](../../attention/sparse_flash_attention_grad/docs/aclnnSparseFlashAttentionGrad.md)|根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnSparseFlashAttentionGradV2](../../attention/sparse_flash_attention_grad/docs/aclnnSparseFlashAttentionGradV2.md)|根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。本接口相较于[aclnnSparseFlashAttentionGrad](../../attention/sparse_flash_attention_grad/docs/aclnnSparseFlashAttentionGrad.md)，新增可选OSS Sink输入与对应梯度输出。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnSparseFlashMla](../../attention/sparse_flash_mla/docs/aclnnSparseFlashMla.md)|支持SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）、HCA（Heavily Compressed Attention）三类Attention计算场景。|默认确定性实现| 默认确定性实现 |
|[aclnnSparseFlashMlaMetadata](../../attention/sparse_flash_mla_metadata/docs/aclnnSparseFlashMlaMetadata.md)|生成aclnnSparseFlashMla主算子使用的任务切分metadata。|默认确定性实现| 默认确定性实现 |
|[aclnnSparseFlashMlaGrad](../../attention/sparse_flash_mla_grad/docs/aclnnSparseFlashMlaGrad.md)|计算SparseFlashMla训练场景下注意力的反向输出，支持SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）以及HCA（Heavily Compressed Attention）。|默认非确定性实现，不支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnSparseFlashMlaGradMetadata](../../attention/sparse_flash_mla_grad_metadata/docs/aclnnSparseFlashMlaGradMetadata.md)| aclnnSparseFlashMlaGrad接口的前置接口，用于计算aclnnSparseFlashMlaGrad的负载均衡。| - | 默认确定性实现 |
|[aclnnSparseLightningIndexerGradKLLoss](../../attention/sparse_lightning_indexer_grad_kl_loss/docs/aclnnSparseLightningIndexerGradKLLoss.md)|LightningIndexer的反向算子，再额外融合了Loss计算功能。|默认非确定性实现，不支持配置开启| 默认确定性实现 |
|[aclnnSparseLightningIndexerGradKLLossV2](../../attention/sparse_lightning_indexer_grad_kl_loss/docs/aclnnSparseLightningIndexerGradKLLossV2.md)|LightningIndexer的反向算子，再额外融合了Loss计算功能,该接口支持A5的sinks场景计算P概率归一化。|默认非确定性实现，不支持配置开启| 默认确定性实现 |
|[aclnnSparseLightningIndexerKLLossGrad](../../attention/sparse_lightning_indexer_kl_loss_grad/docs/aclnnSparseLightningIndexerKLLossGrad.md)|LightningIndexer的反向算子，支持输出Loss计算所需Index部分的分数。|默认非确定性实现，支持配置开启| 默认非确定性实现，支持配置开启 |
|[aclnnSparseLightningIndexerKLLossGradMetadata](../../attention/sparse_lightning_indexer_kl_loss_grad_metadata/docs/aclnnSparseLightningIndexerKLLossGradMetadata.md)| aclnnSparseLightningIndexerKLLossGrad接口的前置接口，用于计算aclnnSparseLightningIndexerKLLossGrad的负载均衡。| 默认确定性实现 | 默认确定性实现 |
|[aclnnStemOamPrepPagedKv](../../attention/stem_oam_prep_paged_kv/docs/aclnnStemOamPrepPagedKv.md)| 大模型推理动态稀疏注意力机制的前置评分模块，为block-sparse-attention的前置评分模块。| - |默认确定性实现|
|[aclnnSwigluGatedMlp](../../experimental/ffn/swiglu_gated_mlp/docs/aclnnSwigluGatedMlp.md)|完成融合SwiGLU门控MLP计算，包括首个MatMul、SwiGLU激活和第二个MatMul。|默认确定性实现| - |
|[aclnnStemOamPrepVarlenQ](../../attention/stem_oam_prep_varlen_q/docs/aclnnStemOamPrepVarlenQ.md)|Stem OAM block-sparse attention中Q侧预处理，将变长Q tensor转化为按stem block分组的flattened qFlat输出。| - | 默认确定性实现 |
|[aclnnSwinAttentionScoreQuant](../../attention/swin_attention_score_quant/docs/aclnnSwinAttentionScoreQuant.md)|完成swin-transformer场景的Attention计算。|默认确定性实现| - |
|[aclnnSwinTransformerLnQkvQuant](../../ffn/swin_transformer_ln_qkv_quant/docs/aclnnSwinTransformerLnQkvQuant.md)|Swin Transformer网络模型完成Q、K、V的计算。| - | - |
|[aclnnWeightQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnWeightQuantMatmulAllReduce.md)|对入参x2进行伪量化计算后，完成MatMul和AllReduce计算。|默认非确定性实现，支持配置开启| 默认确定性实现 |
|[aclnnKvRmsNormRopeCache](../../posembedding/kv_rms_norm_rope_cache/docs/aclnnKvRmsNormRopeCache.md)|对输入张量（kv）的尾轴，拆分出左半边用于rms_norm计算，右半边用于RoPE计算，再将计算结果分别scatter到两块cache中。|默认确定性实现|默认确定性实现|
|[aclnnKvRmsNormRopeCacheV2](../../posembedding/kv_rms_norm_rope_cache/docs/aclnnKvRmsNormRopeCacheV2.md)|融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算（对应$rms\_size$）与RoPE（Rotary Position Embedding）位置编码（对应$rope\_size$)，以及更新KVCache的ScatterUpdate操作。本接口向下兼容aclnnKvRmsNormRopeCache。|默认确定性实现|默认确定性实现|
|[aclnnFusedFloydAttention](../../attention/fused_floyd_attention/docs/aclnnFusedFloydAttention.md)|训练场景下，使用FloydAttention算法实现多维自注意力的计算。|默认确定性实现| - |
|[aclnnFusedFloydAttentionGrad](../../attention/fused_floyd_attention_grad/docs/aclnnFusedFloydAttentionGrad.md)|训练场景下，计算Floyd注意力的反向输出，FloydAttn相较于传统FA主要是计算qk/pv注意力时会额外将seq作为batch轴从而转换为batchMatmul。|默认非确定性实现，不支持配置开启| - |
|[aclnnScatterPaKvCacheWithKScale](../../attention/scatter_pa_kv_cache_with_k_scale/docs/aclnnScatterPaKvCacheWithKScale.md)|训练场景下，更新KvCache中指定位置的key和value，同时更新key的scale值。|默认确定性实现| - |
|[aclnnDenseLightningIndexerKLLossGrad](../../attention/dense_lightning_indexer_kl_loss_grad/docs/aclnnDenseLightningIndexerKLLossGrad.md)|LightningIndexer的反向算子，支持输出Loss计算所需Index部分的分数。| - | 默认非确定性实现，支持配置开启 |
|[aclnnDenseLightningIndexerKLLossGradMetadata](../../attention/dense_lightning_indexer_kl_loss_grad_metadata/docs/aclnnDenseLightningIndexerKLLossGradMetadata.md)| aclnnDenseLightningIndexerKLLossGrad接口的前置接口，用于计算aclnnDenseLightningIndexerKLLossGrad的负载均衡。| - | 默认确定性实现 |
|[aclnnSparseFlashMlaSoftmaxL1Norm](../../attention/sparse_flash_mla_softmax_l1_norm/docs/aclnnSparseFlashMlaSoftmaxL1Norm.md)|aclnnDenseLightningIndexerKLLossGradd的前置接口，用于计算attnSoftmaxL1Norm。| - | 默认确定性实现 |
|[aclnnSparseFlashMlaSoftmaxL1NormMetadata](../../attention/sparse_flash_mla_softmax_l1_norm_metadata/docs/aclnnSparseFlashMlaSoftmaxL1NormMetadata.md)| aclnnSparseFlashMlaSoftmaxL1NormGrad接口的前置接口，用于计算aclnnSparseFlashMlaSoftmaxL1Norm的负载均衡。| - | 默认确定性实现 |
|[aclnnUndGenQkvRmsNormRopeCache](../../posembedding/und_gen_qkv_rms_norm_rope_cache/docs/aclnnUndGenQkvRmsNormRopeCache.md)|把理解阶段与生成阶段两段QKV输入按catIndices间接寻址拼成一条输出序列，逐token拆分Q/K/V后对Q/K执行RMSNorm和MRoPE，V透传；Q作为独立输出返回，K/V按slotMapping原地写入分页KV Cache。|-|默认确定性实现|

## 废弃接口

|    废弃接口   |   说明     |
| --------------- | ----------------------- |
|[aclnnFusedInferAttentionScore](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScore.md)|此接口后续版本会废弃，请使用最新接口[aclnnFusedInferAttentionScoreV5](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md)。 |
|[aclnnFusedInferAttentionScoreV2](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV2.md)|此接口后续版本会废弃，请使用最新接口[aclnnFusedInferAttentionScoreV5](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md)。 |
|[aclnnFusedInferAttentionScoreV3](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV3.md)|此接口后续版本会废弃，请使用最新接口[aclnnFusedInferAttentionScoreV5](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md)。 |
|[aclnnGroupedMatMulAllReduce](../../mc2/grouped_mat_mul_all_reduce/docs/aclnnGroupedMatMulAllReduce.md)|此接口后续版本会废弃，请勿使用该接口。|
|[aclnnIncreFlashAttention](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttention.md)|此接口后续版本会废弃，请使用最新接口[aclnnIncreFlashAttentionV4](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)。 |
|[aclnnIncreFlashAttentionV2](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV2.md)|此接口后续版本会废弃，请使用最新接口[aclnnIncreFlashAttentionV4](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)。 |
|[aclnnIncreFlashAttentionV3](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV3.md)|此接口后续版本会废弃，请使用最新接口[aclnnIncreFlashAttentionV4](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)。 |
|[aclnnPromptFlashAttention](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttention.md)|此接口后续版本会废弃，请使用最新接口[aclnnPromptFlashAttentionV3](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttentionV3.md)。 |
|[aclnnPromptFlashAttentionV2](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttentionV2.md)|此接口后续版本会废弃，请使用最新接口[aclnnPromptFlashAttentionV3](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttentionV3.md)。 |
|[aclnnGroupedMatmul](../../gmm/grouped_matmul/docs/aclnnGroupedMatmul.md)|此接口后续版本会废弃，请使用最新接口[aclnnGroupedMatmulV5](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)。 |
|[aclnnGroupedMatmulV2](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV2.md)|此接口后续版本会废弃，请使用最新接口[aclnnGroupedMatmulV5](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)。 |
|[aclnnGroupedMatmulV3](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV3.md)|此接口后续版本会废弃，请使用最新接口[aclnnGroupedMatmulV5](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)。 |
|[aclnnGroupedMatmulV4](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md)|此接口后续版本会废弃，请使用最新接口[aclnnGroupedMatmulV5](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)。 |
|[aclnnMlaProlog](../../attention/mla_prolog/docs/aclnnMlaProlog.md)|此接口后续版本会废弃，请使用最新接口[aclnnMlaPrologV3WeightNz](../../attention/mla_prolog_v3/docs/aclnnMlaPrologV3WeightNz.md)。|
|[aclnnMlaPrologV2WeightNz](../../attention/mla_prolog_v2/docs/aclnnMlaPrologV2WeightNz.md)|此接口后续版本会废弃，请使用最新接口[aclnnMlaPrologV3WeightNz](../../attention/mla_prolog_v3/docs/aclnnMlaPrologV3WeightNz.md)。 |
|[aclnnMatmulAllReduceAddRmsNorm](../../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnMatmulAllReduce.md)和[aclnnRmsNorm](https://gitcode.com/cann/ops-nn/blob/master/norm/add_rms_norm/docs/aclnnAddRmsNorm.md)。|
|[aclnnQuantMatmulAllReduceAddRmsNorm](../../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnQuantMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnQuantMatmulAllReduceV2](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV2.md)和[aclnnRmsNorm](https://gitcode.com/cann/ops-nn/blob/master/norm/add_rms_norm/docs/aclnnAddRmsNorm.md)。|
|[aclnnWeightQuantMatmulAllReduceAddRmsNorm](../../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnWeightQuantMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnWeightQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnWeightQuantMatmulAllReduce.md)和[aclnnRmsNorm](https://gitcode.com/cann/ops-nn/blob/master/norm/add_rms_norm/docs/aclnnAddRmsNorm.md)。|
|[aclnnInplaceMatmulAllReduceAddRmsNorm](../../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnMatmulAllReduce.md)和[aclnnRmsNorm](https://gitcode.com/cann/ops-nn/blob/master/norm/add_rms_norm/docs/aclnnAddRmsNorm.md)。|
|[aclnnInplaceQuantMatmulAllReduceAddRmsNorm](../../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceQuantMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnQuantMatmulAllReduceV2](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV2.md)和[aclnnRmsNorm](https://gitcode.com/cann/ops-nn/blob/master/norm/add_rms_norm/docs/aclnnAddRmsNorm.md)。|
|[aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm](../../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm.md)|此接口后续版本会废弃，请替换为[aclnnWeightQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnWeightQuantMatmulAllReduce.md)和[aclnnRmsNorm](https://gitcode.com/cann/ops-nn/blob/master/norm/add_rms_norm/docs/aclnnAddRmsNorm.md)。 |
