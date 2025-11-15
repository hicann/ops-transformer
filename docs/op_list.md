# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](context/dir_structure.md)。


项目提供的所有算子分类和算子列表如下：

| 算子分类 |  算子目录  |  算子执行位置 |     说明  |
| ------- |-----------|-----------|-----------|
| attention   | [flash_attention_score](../attention/flash_attention_score/README.md) | AI Core | 使用FlashAttention算法实现self-attention（自注意力）的计算 |
| attention   | [flash_attention_score_grad](../attention/flash_attention_score_grad/README.md) | AI Core | 训练场景下计算注意力的反向输出，即FlashAttentionScore的反向计算 |
| attention   | [fused_infer_attention_score](../attention/fused_infer_attention_score/README.md) | AI Core | decode & prefill场景的FlashAttention算子 |
| attention   | [incre_flash_attention](../attention/incre_flash_attention/README.md) | AI Core | 增量推理场景的FlashAttention算子  |
| attention   | [mla_prolog](../attention/mla_prolog/README.md) | AI Core | 推理MlaProlog算子  |
| attention   | [mla_prolog_v2](../attention/mla_prolog_v2/README.md) | AI Core | 推理MlaPrologV2WeightNz算子 |
| attention   | [nsa_compress](../attention/nsa_compress/README.md) | AI Core | 训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩 |
| attention   | [nsa_compress_attention](../attention/nsa_compress_attention/README.md) | AI Core |  NSA中compress attention以及select topk索引计算 |
| attention   | [nsa_compress_attention_infer](../attention/nsa_compress_attention_infer/README.md) | AI Core | 实现Native Sparse Attention推理过程中，Compress Attention的计算 |
| attention   | [nsa_compress_grad](../attention/nsa_compress_grad/README.md) | AI Core | aclnnNsaCompress算子的反向计算 |
| attention   | [nsa_compress_with_cache](../attention/nsa_compress_with_cache/README.md) | AI Core | 实现Native-Sparse-Attention推理阶段的KV压缩 |
| attention   | [nsa_selected_attention_infer](../attention/nsa_selected_attention_infer/README.md) | AI Core | 实现Native Sparse Attention推理过程中，Selected Attention的计算 |
| attention   | [nsa_selected_attention](../attention/nsa_selected_attention/README.md) | AI Core | 训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算 |
| attention   | [nsa_selected_attention_grad](../attention/nsa_selected_attention_grad/README.md) | AI Core | 根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出 |
| attention   | [prompt_flash_attention](../attention/prompt_flash_attention/README.md) | AI Core | 全量推理场景的FlashAttention算子 |
| ffn         | [ffn](../ffn/ffn/README.md) | AI Core | 提供MoeFFN和FFN的计算功能 |
| ffn         | [swin_attention_ffn](../ffn/swin_attention_ffn/README.md) | AI Core | 全量推理场景的FlashAttention算子 |
| ffn         | [swin_transformer_ln_qkv](../ffn/swin_transformer_ln_qkv/README.md) | AI Core | 完成fp16权重场景下的Swin Transformer 网络模型的Q、K、V 的计算 |
| gmm         | [grouped_matmul](../gmm/grouped_matmul/README.md) | AI Core | 实现分组矩阵乘计算。 |
| gmm         | [grouped_matmul_add](../gmm/grouped_matmul_add/README.md) | AI Core | 实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。 |
| gmm         | [grouped_matmul_finalize_routing](../gmm/grouped_matmul_finalize_routing/README.md) | AI Core | GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作 |
| gmm         | [grouped_matmul_swiglu_quant](../gmm/grouped_matmul_swiglu_quant/README.md) | AI Core | 融合GroupedMatmul 、dquant、swiglu和quant |
| mc2         | [all_gather_matmul](../mc2/all_gather_matmul/README.md) | AI Core | 完成AllGather通信与MatMul计算融合 |
| mc2         | [allto_all_all_gather_batch_mat_mul](../mc2/allto_all_all_gather_batch_mat_mul/README.md) | AI Core | 完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。|
| mc2         | [allto_allv_grouped_mat_mul](../mc2/allto_allv_grouped_mat_mul/README.md) | AI Core | 完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，**先通信后计算** |
| mc2         | [batch_mat_mul_reduce_scatter_allto_all](../mc2/batch_mat_mul_reduce_scatter_allto_all/README.md) | AI Core | 实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行 |
| mc2         | [distribute_barrier](../mc2/distribute_barrier/README.md) | AI Core | 完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作 |
| mc2         | [grouped_mat_mul_allto_allv](../mc2/grouped_mat_mul_allto_allv/README.md) | AI Core | 完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合，**先计算后通信** |
| mc2         | [inplace_matmul_all_reduce_add_rms_norm](../mc2/inplace_matmul_all_reduce_add_rms_norm/README.md) | AI Core | 完成mm + all_reduce + add + rms_norm计算 |
| mc2         | [matmul_all_reduce](../mc2/matmul_all_reduce/README.md) | AI Core | 完成MatMul计算与AllReduce通信融合 |
| mc2         | [matmul_all_reduce_add_rms_norm](../mc2/matmul_all_reduce_add_rms_norm/README.md) | AI Core | 完成mm + all_reduce + add + rms_norm计算 |
| mc2         | [matmul_reduce_scatter](../mc2/matmul_reduce_scatter/README.md) | AI Core | 完成mm + reduce_scatter_base计算 |
| mc2         | [moe_distribute_buffer_reset](../mc2/moe_distribute_buffer_reset/README.md) | AI Core | 故障检测流程中，对EP通信域做数据区与状态区的清理 |
| mc2         | [moe_distribute_combine](../mc2/moe_distribute_combine/README.md) | AI Core | 当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）|
| mc2         | [moe_distribute_combine_add_rms_norm](../mc2/moe_distribute_combine_add_rms_norm/README.md) | AI Core | 当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加），之后完成Add + RmsNorm融合 |
| mc2         | [moe_distribute_combine_v2](../mc2/moe_distribute_combine_v2/README.md) | AI Core | 当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加） |
| mc2         | [moe_distribute_dispatch](../mc2/moe_distribute_dispatch/README.md) | AI Core | 对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信 |
| mc2         | [moe_distribute_dispatch_v2](../mc2/moe_distribute_dispatch_v2/README.md) | AI Core | 对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信 |
| mc2         | [moe_update_expert](../mc2/moe_update_expert/README.md) | AI Core | 完成每个token的topK个专家逻辑专家号到物理卡号的映射 |
| moe         | [moe_compute_expert_tokens](../moe/moe_compute_expert_tokens/README.md) | AI Core | MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置 |
| moe         | [moe_finalize_routing](../moe/moe_finalize_routing/README.md) | AI Core | MoE计算中，最后处理合并MoE FFN的输出结果 |
| moe         | [moe_finalize_routing_v2](../moe/moe_finalize_routing_v2/README.md) | AI Core | MoE计算中，最后处理合并MoE FFN的输出结果 |
| moe         | [moe_finalize_routing_v2_grad](../moe/moe_finalize_routing_v2_grad/README.md) | AI Core | aclnnMoeFinalizeRoutingV2的反向传播 |
| moe         | [moe_gating_top_k](../moe/moe_gating_top_k/README.md) | AI Core | MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家 |
| moe         | [moe_gating_top_k_softmax](../moe/moe_gating_top_k_softmax/README.md) | AI Core | MoE计算中，对x的输出做Softmax计算，取topk操作。 |
| moe         | [moe_gating_top_k_softmax_v2](../moe/moe_gating_top_k_softmax_v2/README.md) | AI Core | MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取topk操作；如果renorm=1，先对x的输出做topk操作，再进行Softmax操作 |
| moe         | [moe_init_routing](../moe/moe_init_routing/README.md) | AI Core | MoE的routing计算，根据[aclnnMoeGatingTopKSoftmax](../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md)的计算结果做routing处理 |
| moe         | [moe_init_routing_quant](../moe/moe_init_routing_quant/README.md) | AI Core | MoE的routing计算，根据[aclnnMoeGatingTopKSoftmax](../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md)的计算结果做routing处理，并对结果进行量化 |
| moe         | [moe_init_routing_quant_v2](../moe/moe_init_routing_quant_v2/README.md) | AI Core | MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理 |
| moe         | [moe_init_routing_v2](../moe/moe_init_routing_v2/README.md) | AI Core | 以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用 |
| moe         | [moe_init_routing_v2_grad](../moe/moe_init_routing_v2_grad/README.md) | AI Core | [aclnnMoeInitRoutingV2](../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)的反向传播，完成tokens的加权求和 |
| moe         | [moe_init_routing_v3](../moe/moe_init_routing_v3/README.md) | AI Core | MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](README.md)的计算结果做routing处理，支持不量化和动态量化模式 |
| moe         | [moe_re_routing](../moe/moe_re_routing/README.md) | AI Core | MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列 |
| moe         | [moe_token_permute](../moe/moe_token_permute/README.md) | AI Core | MoE的permute计算，根据索引indices将tokens广播并排序 |
| moe         | [moe_token_permute_grad](../moe/moe_token_permute_grad/README.md) | AI Core | aclnnMoeTokenPermute的反向传播计算 |
| moe         | [moe_token_permute_with_ep](../moe/moe_token_permute_with_ep/README.md) | AI Core | MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片 |
| moe         | [moe_token_permute_with_ep_grad](../moe/moe_token_permute_with_ep_grad/README.md) | AI Core | aclnnMoeTokenPermuteWithEp的反向传播计算 |
| moe         | [moe_token_permute_with_routing_map](../moe/moe_token_permute_with_routing_map/README.md) | AI Core | aclnnMoeTokenPermuteWithRoutingMap的反向传播 |
| moe         | [moe_token_permute_with_routing_map_grad](../moe/moe_token_permute_with_routing_map_grad/README.md) | AI Core | MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片 |
| moe         | [moe_token_unpermute](../moe/moe_token_unpermute/README.md) | AI Core | 根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘；最后进行累加求和，并输出计算结果 |
| moe         | [moe_token_unpermute_grad](../moe/moe_token_unpermute_grad/README.md) | AI Core | aclnnMoeTokenUnpermuteGrad的反向传播 |
| moe         | [moe_token_unpermute_with_ep](../moe/moe_token_unpermute_with_ep/README.md) | AI Core | 根据sortedIndices存储的下标位置，去获取permutedTokens中的输入数据与probs相乘，并进行合并累加 |
| moe         | [moe_token_unpermute_with_ep_grad](../moe/moe_token_unpermute_with_ep_grad/README.md) | AI Core | aclnnMoeTokenUnpermuteWithEp的反向传播 |
| moe         | [moe_token_unpermute_with_routing_map](../moe/moe_token_unpermute_with_routing_map/README.md) | AI Core | 对经过aclnnMoeTokenpermuteWithRoutingMap处理的permutedTokens，累加回原unpermutedTokens。根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘，最后进行累加求和，并输出计算结果 |
| moe         | [moe_token_unpermute_with_routing_map_grad](../moe/moe_token_unpermute_with_routing_map_grad/README.md) | AI Core | aclnnMoeTokenUnpermuteWithRoutingMap的反向传播 |
| posembedding| [apply_rotary_pos_emb](../posembedding/apply_rotary_pos_emb/README.md) | AI Core | 执行旋转位置编码计算，推理网络为了提升性能，将query和key两路算子融合成一路 |
| posembedding| [dequant_rope_quant_kvcache](../posembedding/dequant_rope_quant_kvcache/README.md) | AI Core | 对输入张量（x）进行dequant（可选）后，按`sizeSplits`（为切分的长度）对尾轴进行切分，划分为q、k、vOut，对q、k进行旋转位置编码，生成qOut和kOut，之后对kOut和vOut进行量化并按照`indices`更新到kCacheRef和vCacheRef上 |
| posembedding| [interleave_rope](../posembedding/interleave_rope/README.md) | AI Core | 针对单输入 x 进行旋转位置编码 |
| posembedding| [rope_quant_kvcache](../posembedding/rope_quant_kvcache/README.md) | AI Core | 对输入张量的尾轴进行切分 |
| posembedding| [rope_with_sin_cos_cache](../posembedding/rope_with_sin_cos_cache/README.md) | AI Core | 推理网络为了提升性能，将sin和cos输入通过cache传入，执行旋转位置编码计算 |
| posembedding| [rotary_position_embedding](../posembedding/rotary_position_embedding/README.md) | AI Core | 执行单路旋转位置编码计算 |
| posembedding| [rotary_position_embedding_grad](../posembedding/rotary_position_embedding_grad/README.md) | AI Core | 执行单路旋转位置编码的反向计算 |
