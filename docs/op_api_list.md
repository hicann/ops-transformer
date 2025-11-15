# 算子接口（aclnn）

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

算子接口列表如下：

|    接口名   |      说明     |
|-----------|------------|
|[aclnnAllGatherMatmul](../mc2/all_gather_matmul/docs/aclnnAllGatherMatmul.md)|完成AllGather通信与MatMul计算融合。
|[aclnnAlltoAllAllGatherBatchMatMul](../mc2/allto_all_all_gather_batch_mat_mul/docs/aclnnAlltoAllAllGatherBatchMatMul.md)|完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。
|[aclnnAlltoAllvGroupedMatMul](../mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMul.md)|完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合。
|[aclnnBatchMatMulReduceScatterAlltoAll](../mc2/batch_mat_mul_reduce_scatter_allto_all/docs/aclnnBatchMatMulReduceScatterAlltoAll.md)|BatchMatMulReduceScatterAllToAll是通算融合算子，实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行的算子。
|[aclnnDistributeBarrier](../mc2/distribute_barrier/docs/aclnnDistributeBarrier.md)|完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。
|[aclnnDistributeBarrierV2](../mc2/distribute_barrier/docs/aclnnDistributeBarrierV2.md)|完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。
|[aclnnFFN](../ffn/ffn/docs/aclnnFFN.md)|该FFN算子提供MoeFFN和FFN的计算功能。
|[aclnnFlashAttentionScore](../attention/flash_attention_score/docs/aclnnFlashAttentionScore.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。
|[aclnnFlashAttentionScoreGrad](../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGrad.md)|训练场景下计算注意力的反向输出。
|[aclnnFlashAttentionUnpaddingScoreGradV4](../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV4.md)|训练场景下计算注意力的反向输出。
|[aclnnFlashAttentionVarLenScoreV4](../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV4.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。
|[aclnnFusedInferAttentionScoreV4](../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV4.md)|适配decode & prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。
|[aclnnGroupedMatMulAlltoAllv](../mc2/grouped_mat_mul_allto_allv/docs/aclnnGroupedMatMulAlltoAllv.md)|完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合。
|[aclnnGroupedMatmulFinalizeRouting](../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRouting.md)|GroupedMatmul和MoeFinalizeRouting的融合算子。
|[aclnnGroupedMatmulFinalizeRoutingWeightNz](../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingWeightNz.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持w为昇腾亲和数据排布格式(NZ)。
|[aclnnGroupedMatmulSwigluQuant](../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md)|融合GroupedMatmul 、dquant、swiglu和quant。
|[aclnnGroupedMatmulSwigluQuantWeightNZ](../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md)|融合GroupedMatmul、dquant、swiglu和quant。
|[aclnnGroupedMatmulV5](../gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)|实现分组矩阵乘计算。
|[aclnnGroupedMatmulWeightNz](../gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。
|[aclnnIncreFlashAttentionV4](../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)|在全量推理场景的FlashAttention算子的基础上实现增量推理|
|[aclnnInplaceMatmulAllReduceAddRmsNorm](../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。
|[aclnnInplaceQuantMatmulAllReduceAddRmsNorm](../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。
|[aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm](../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。
|[aclnnMatmulAllReduce](../mc2/matmul_all_reduce/docs/aclnnMatmulAllReduce.md)|完成MatMul计算与AllReduce通信融合。
|[aclnnMatmulAllReduceAddRmsNorm](../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。
|[aclnnMatmulReduceScatter](../mc2/matmul_reduce_scatter/docs/aclnnMatmulReduceScatter.md)|完成mm + reduce_scatter_base计算。
|[aclnnMlaProlog](../attention/mla_prolog/docs/aclnnMlaProlog.md)|Multi-Head Latent Attention前处理的计算 。|
|[aclnnMlaPrologV2WeightNz](../attention/mla_prolog_v2/docs/aclnnMlaPrologV2WeightNz.md)|Multi-Head Latent Attention前处理的计算 。|
|[aclnnMoeComputeExpertTokens](../moe/moe_compute_expert_tokens/docs/aclnnMoeComputeExpertTokens.md)|MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。
|[aclnnMoeDistributeBufferReset](../mc2/moe_distribute_buffer_reset/docs/aclnnMoeDistributeBufferReset.md)|故障检测流程中，对EP通信域做数据区与状态区的清理。
|[aclnnMoeDistributeCombine](../mc2/moe_distribute_combine/docs/aclnnMoeDistributeCombine.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。
|[aclnnMoeDistributeCombineAddRmsNormV2](../mc2/moe_distribute_combine_add_rms_norm/docs/aclnnMoeDistributeCombineAddRmsNormV2.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加），之后完成Add + RmsNorm融合。
|[aclnnMoeDistributeCombineV3](../mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV3.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。
|[aclnnMoeDistributeDispatchV3](../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3.md)|对token数据进行量化（可选）。
|[aclnnMoeFinalizeRouting](../moe/moe_finalize_routing/docs/aclnnMoeFinalizeRouting.md)|MoE计算中，最后处理合并MoE FFN的输出结果。
|[aclnnMoeFinalizeRoutingV2](../moe/moe_finalize_routing_v2/docs/aclnnMoeFinalizeRoutingV2.md)|MoE计算中，最后处理合并MoE FFN的输出结果。
|[aclnnMoeFinalizeRoutingV2Grad](../moe/moe_finalize_routing_v2_grad/docs/aclnnMoeFinalizeRoutingV2Grad.md)|aclnnMoeFinalizeRoutingV2的反向传播。
|[aclnnMoeGatingTopK](../moe/moe_gating_top_k/docs/aclnnMoeGatingTopK.md)|MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。
|[aclnnMoeGatingTopKSoftmax](../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md)|MoE计算中，对x的输出做Softmax计算，取topk操作。
|[aclnnMoeGatingTopKSoftmaxV2](../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)|MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取topk操作；如果renorm=1，先对x的输出做topk操作，再进行Softmax操作。
|[aclnnMoeInitRouting](../moe/moe_init_routing/docs/aclnnMoeInitRouting.md)|MoE的routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做routing处理。
|[aclnnMoeInitRoutingQuant](../moe/moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant.md)|MoE的routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做routing处理，并对结果进行量化。
|[aclnnMoeInitRoutingQuantV2](../moe/moe_init_routing_quant_v2/docs/aclnnMoeInitRoutingQuantV2.md)|MoE的routing计算，根据aclnnMoeGatingTopKSoftmaxV2的计算结果做routing处理。
|[aclnnMoeInitRoutingV2](../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)|该算子对应MoE（Mixture of Experts，混合专家模型）中的Routing计算，以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用。
|[aclnnMoeInitRoutingV2Grad](../moe/moe_init_routing_v2_grad/docs/aclnnMoeInitRoutingV2Grad.md)|aclnnMoeInitRoutingV2的反向传播，完成tokens的加权求和。
|[aclnnMoeUpdateExpert](../mc2/moe_update_expert/docs/aclnnMoeUpdateExpert.md)|本API支持负载均衡和专家剪枝功能。经过映射后的专家表和mask可传入Moe层进行数据分发和处理。
|[aclnnNsaCompress](../attention/nsa_compress/docs/aclnnNsaCompress.md)|训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。
|[aclnnNsaCompressAttention](../attention/nsa_compress_attention/docs/aclnnNsaCompressAttention.md)|NSA中compress attention以及select topk索引计算。
|[aclnnNsaCompressAttentionInfer](../attention/nsa_compress_attention_infer/docs/aclnnNsaCompressAttentionInfer.md)|Native Sparse Attention推理过程中，Compress Attention的计算。
|[aclnnNsaCompressWithCache](../attention/nsa_compress_with_cache/docs/aclnnNsaCompressWithCache.md)|实现Native-Sparse-Attention推理阶段的KV压缩。|
|[aclnnNsaSelectedAttention](../attention/nsa_selected_attention/docs/aclnnNsaSelectedAttention.md)|训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算。
|[aclnnNsaSelectedAttentionGrad](../attention/nsa_selected_attention_grad/docs/aclnnNsaSelectedAttentionGrad.md)|根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。
|[aclnnNsaSelectedAttentionInfer](../attention/nsa_selected_attention_infer/docs/aclnnNsaSelectedAttentionInfer.md)|Native Sparse Attention推理过程中，Selected Attention的计算。
|[aclnnPromptFlashAttentionV3](../attention/prompt_flash_attention/docs/aclnnPromptFlashAttentionV3.md)|全量推理场景的FlashAttention算子。
|[aclnnQuantMatmulAllReduce](../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduce.md)|对量化后的入参x1、x2进行MatMul计算后，接着进行Dequant计算，接着与x3进行Add操作，最后做AllReduce计算。
|[aclnnQuantMatmulAllReduceAddRmsNorm](../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。
|[aclnnWeightQuantMatmulAllReduce](../mc2/matmul_all_reduce/docs/aclnnWeightQuantMatmulAllReduce.md)|对入参x2进行伪量化计算后，完成Matmul和AllReduce计算。
|[aclnnWeightQuantMatmulAllReduceAddRmsNorm](../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnWeightQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。