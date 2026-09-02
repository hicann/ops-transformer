# 算子列表

> **使用说明**：
>
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./install/dir_structure.md)。
> - **算子执行硬件单元**：大部分算子运行在AI Core，少部分算子运行在AI CPU。默认情况下，项目中提到的算子一般指AI Core算子。关于AI Core和AI CPU详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)，其中版本号大于等于8.5.0中对应章节为“硬件实现”，其余版本中对应章节为“概念原理和术语 > 硬件架构与数据处理原理”。
> - **算子接口列表**：为方便调用算子，CANN提供一套C API执行算子，一般以aclnn为前缀，全量接口参见[aclnn列表](op_api_list.md)。
> - **V版本演进说明**：部分算子存在多个V版本，使用时选择最高V版本即可（高版本算子已兼容低版本算子的所有能力）。

项目提供的所有算子分类和算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子分类</th>
    <th rowspan="2">算子目录</th>
    <th colspan="2">算子实现</th>
    <th>aclnn调用</th>
    <th>图模式调用</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/attention_update/README.md">attention_update</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将各SP域PA算子的输出的中间结果lse，localOut两个局部变量结果更新成全局结果。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/attention_worker_combine/README.md">attention_worker_combine</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将多个计算单元处理的注意力token数据进行融合，结合专家权重对结果进行加权，输出最终的注意力融合结果。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/attention_worker_scheduler/README.md">attention_worker_scheduler</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Attention和FFN分离部署场景下，Attention侧数据扫描算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/bsa_select_block_mask/README.md">bsa_select_block_mask</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>BSA（BlockSparseAttention）的前置算子，负责根据 Query 和 Key 的内容动态生成 blockSparseMask，使 BSA 的调用链从"手动提供掩码"变为"根据 Q/K 内容自适应选择稀疏模式"。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_attn_res_prepare/README.md">block_attn_res_prepare</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>BlockAttnRes第一阶段准备算子，用于并行计算全部目标slot的块间注意力，输出softmax加权分子及统计量，供下一阶段融合当前残差。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_attn_res_update/README.md">block_attn_res_update</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>BlockAttnRes第二阶段更新算子，用于更新partial block并计算在线Softmax状态对应的注意力输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_sparse_attention_grad/README.md">block_sparse_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下计算注意力的反向输出，即BlockSparseAttention的反向计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/chunk_gated_delta_rule/README.md">chunk_gated_delta_rule</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>推理场景下线性注意力Gated Delta Rule的chunk版本，用于prefill节点。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_softmax_lse_v2/README.md">dense_lightning_indexer_softmax_lse_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>DenseLightningIndexerGradKlLoss算子计算Softmax输入的一个分支算子。相比dense_lightning_indexer_softmax_lse，新增了压缩注意力（Compressed Attention）支持，并支持通过metadata前置算子进行分核负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_softmax_lse_v2_metadata/README.md">dense_lightning_indexer_softmax_lse_v2_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>dense_lightning_indexer_softmax_lse_v2算子的前置算子，用于计算dense_lightning_indexer_softmax_lse_v2的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attn/README.md">flash_attn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>基于FlashAttention算法实现self-attention（自注意力）的计算，训练推理归一化，仅支持非量化场景。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attention_score/README.md">flash_attention_score</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>使用FlashAttention算法实现self-attention（自注意力）的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attention_score_grad/README.md">flash_attention_score_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下计算注意力的反向输出，即FlashAttentionScore的反向计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_causal_conv1d/README.md">fused_causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接等特性，并且可以通过传入参数maxDraftTokens来指定最大投机个数。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_floyd_attention/README.md">fused_floyd_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下，使用FloydAttention算法实现多维自注意力的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_floyd_attention_grad/README.md">fused_floyd_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下，计算Floyd注意力的反向输出，FloydAttn相较于传统FA主要是计算qk/pv注意力时会额外将seq作为batch轴从而转换为batchMatmul。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_infer_attention_score/README.md">fused_infer_attention_score</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>decode & prefill场景的FlashAttention算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/gather_pa_kv_cache/README.md">gather_pa_kv_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据blockTables中的blockId值、seqLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/generic_block_sparse_attention_metadata/README.md">generic_block_sparse_attention_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>generic_block_sparse_attention接口的前置接口，用于计算generic_block_sparse_attention的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/generic_block_sparse_attention/README.md">generic_block_sparse_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>通过sparseBlockIdx/sparseBlockCount指定稀疏块模式的注意力正向计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/incre_flash_attention/README.md">incre_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>增量推理场景的FlashAttention算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/indexer_quant_cache/README.md">indexer_quant_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>在Indexer注意力机制的Epilog阶段对KV Cache进行原地逐块动态量化压缩更新，支持MX-FP8、Normal、HiFloat8、MX-FP4四种量化模式。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/inplace_fused_causal_conv1d/README.md">inplace_fused_causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接、原地更新等特性，并且可以通过传入参数maxDraftTokens来指定最大投机个数。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kv_compress_epilog/README.md">kv_compress_epilog</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>在KV Cache的Epilog阶段对缓存进行原地逐组动态量化压缩更新，将bfloat16激活值压缩为FP8格式并按slotMapping散写。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kv_quant_sparse_flash_attention/README.md">kv_quant_sparse_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>在Sparse Flash Attention的基础上支持了[Per-Token-Head-Tile-128量化]输入。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer/README.md">lightning_indexer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>基于一系列操作得到每一个token对应的Top-k个位置。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_grad/README.md">lightning_indexer_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>lightning_indexer的反向梯度计算算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_kl_loss/README.md">lightning_indexer_kl_loss</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算Lightning Indexer中teacher分布与student分布之间的KL散度损失。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_v2/README.md">lightning_indexer_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>基于一系列操作得到每一个token对应的Top-k个位置。支持KV压缩场景。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_v2_metadata/README.md">lightning_indexer_v2_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>lightning_indexer_v2算子的前置算子，用于计算lightning_indexer_v2的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/masked_causal_conv1d/README.md">masked_causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对hidden层的token之间进行带mask的因果一维分组卷积操作。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/masked_causal_conv1d_backward/README.md">masked_causal_conv1d_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>对hidden层的token之间进行一维分组卷积操作的反向梯度计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mixed_quant_sparse_flash_mla/README.md">mixed_quant_sparse_flash_mla</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mixed_quant_sparse_flash_mla_metadata/README.md">mixed_quant_sparse_flash_mla_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>mixed_quant_sparse_flash_mla算子的前置算子，用于计算mixed_quant_sparse_flash_mla的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_sparse_flash_mla/README.md">quant_sparse_flash_mla</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>支持全量化场景下的Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_sparse_flash_mla_metadata/README.md">quant_sparse_flash_mla_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>quant_sparse_flash_mla算子的前置算子，用于计算quant_sparse_flash_mla的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_preprocess/README.md">mla_preprocess</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>推理MlaPreprocess算子</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog/README.md">mla_prolog</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>推理MlaProlog算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog_v2/README.md">mla_prolog_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>推理MlaPrologV2WeightNz算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog_v3/README.md">mla_prolog_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>推理MlaPrologV3WeightNz算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress/README.md">nsa_compress</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_attention/README.md">nsa_compress_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>NSA中compress attention以及select topk索引计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_attention_infer/README.md">nsa_compress_attention_infer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现Native Sparse Attention推理过程中，Compress Attention的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_grad/README.md">nsa_compress_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>aclnnNsaCompress算子的反向计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_with_cache/README.md">nsa_compress_with_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现Native-Sparse-Attention推理阶段的KV压缩。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_selected_attention_infer/README.md">nsa_selected_attention_infer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现Native Sparse Attention推理过程中，Selected Attention的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_selected_attention/README.md">nsa_selected_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_lightning_indexer_grad_kl_loss/README.md">sparse_lightning_indexer_grad_kl_loss</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>SparselightningIndexerGradKlLoss算子是LightningIndexer的反向算子，并额外融合了Loss计算功能输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_selected_attention_grad/README.md">nsa_selected_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/prompt_flash_attention/README.md">prompt_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>全量推理场景的FlashAttention算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_compressor/README.md">quant_compressor</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>推理场景下SMLA和QLI的前处理算子，是Compressor的量化版本，用于将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_lightning_indexer/README.md">quant_lightning_indexer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>推理场景下，SparseFlashAttention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_lightning_indexer_v2/README.md">quant_lightning_indexer_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>推理场景下，稀疏attention前处理的计算，选出关键的稀疏token，支持对输入query和key进行INT8/FLOAT8_e4m3fn/HIFLOAT8量化实现存8算8，获取最大收益。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_lightning_indexer_v2_metadata/README.md">quant_lightning_indexer_v2_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>quant_lightning_indexer_v2算子的前置算子，用于计算quant_lightning_indexer_v2的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/recurrent_gated_delta_rule/README.md">recurrent_gated_delta_rule</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>增量推理场景的Recurrent Gated Delta Rule算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/recurrent_kda/README.md">recurrent_kda</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>完成KDA（Kimi Delta Attention）的递归前向计算，面向decode和MTP短序列场景。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/ring_attention_update/README.md">ring_attention_update</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下，更新两次FlashAttention的结果。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/scatter_pa_cache/README.md">scatter_pa_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>更新KCache中指定位置的key。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_attention/README.md">sparse_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>针对大序列长度推理场景的高效注意力计算模块。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_attention_grad/README.md">sparse_flash_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下，计算sparse_flash_attention注意力的反向输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla/README.md">sparse_flash_mla</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>支持SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）和HCA（Heavily Compressed Attention）。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_grad/README.md">sparse_flash_mla_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下，计算sparse_flash_mla注意力的反向输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_grad_metadata/README.md">sparse_flash_mla_grad_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>sparse_flash_mla_grad算子的前置算子，用于计算sparse_flash_mla_grad的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_metadata/README.md">sparse_flash_mla_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>sparse_flash_mla算子的前置算子，用于计算sparse_flash_mla的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_lightning_indexer_kl_loss_grad/README.md">sparse_lightning_indexer_kl_loss_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>SparselightningIndexerKlLossGrad算子是LightningIndexer的反向算子，支持输出Loss计算所需Index部分的分数。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_lightning_indexer_kl_loss_grad_metadata/README.md">sparse_lightning_indexer_kl_loss_grad_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>sparse_lightning_indexer_kl_loss_grad算子的前置算子，用于计算sparse_lightning_indexer_kl_loss_grad的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/compressor/README.md">compressor</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>推理场景下SMLA和QLI的前处理算子，用于将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/compressor_grad/README.md">compressor_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>Compressor的反向算子，用于计算输入X、权重W^KV/W^Gate与位置编码Ape的梯度，前向在gradEnabled为true时导出softmax_score与kv中间结果作为本算子输入。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/stem_oam_prep_varlen_q/README.md">stem_oam_prep_varlen_q</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Stem OAM block-sparse attention中Q侧预处理算子，将变长Q tensor从paged存储格式转化为按stem block分组的flattened qFlat输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/scatter_pa_kv_cache_with_k_scale/README.md">scatter_pa_kv_cache_with_k_scale</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>训练场景下，更新KvCache中指定位置的key和value，同时更新key的scale值。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_kl_loss_grad/README.md">dense_lightning_indexer_kl_loss_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>DenselightningIndexerKlLossGrad算子是LightningIndexer的反向算子，支持输出Loss计算所需Index部分的分数。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_kl_loss_grad_metadata/README.md">dense_lightning_indexer_kl_loss_grad_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>dense_lightning_indexer_kl_loss_grad算子的前置算子，用于计算dense_lightning_indexer_kl_loss_grad的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_softmax_l1_norm/README.md">sparse_flash_mla_softmax_l1_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下，作为dense_lightning_indexer_kl_loss_grad的前置接口，用于计算attnSoftmaxL1Norm。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_flash_attn_grad/README.md">quant_flash_attn_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算量化注意力（QuantFlashAttention）的反向梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_softmax_l1_norm_metadata/README.md">sparse_flash_mla_softmax_l1_norm_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>sparse_flash_mla_softmax_l1_norm算子的前置算子，用于计算sparse_flash_mla_softmax_l1_norm的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/stem_oam_prep_paged_kv/README.md">stem_oam_prep_paged_kv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>大模型推理动态稀疏注意力机制的前置评分模块，为block-sparse-attention的前置评分模块。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/ffn/README.md">ffn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>提供MoeFFN和FFN的计算功能。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/swin_attention_ffn/README.md">swin_attention_ffn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>面向Swin Transformer的注意力输出后处理融合算子，完成输出线性投影（MatMul+Bias）、窗口还原、反向循环移位以及残差加法。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../experimental/ffn/swiglu_gated_mlp/README.md">swiglu_gated_mlp</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成融合SwiGLU门控MLP计算，包括首个MatMul、SwiGLU激活和第二个MatMul。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/ffn_worker_scheduler/README.md">ffn_worker_scheduler</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>Attention和FFN分离场景下，FFN侧数据扫描算子。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/ffn_worker_batching/README.md">ffn_worker_batching</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Attention和FFN分离场景下，FFN侧数据扫描及token重排。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/swin_transformer_ln_qkv/README.md">swin_transformer_ln_qkv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成fp16权重场景下的Swin Transformer网络模型的Q、K、V的计算。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/swin_transformer_ln_qkv_quant/README.md">swin_transformer_ln_qkv_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>Swin Transformer网络模型 完成Q、K、V的计算。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul/README.md">grouped_matmul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现分组矩阵乘计算。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_add/README.md">grouped_matmul_add</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_activation_quant/README.md">grouped_matmul_activation_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>融合GMM、激活函数和量化算子，完成分组矩阵乘、激活和量化计算，输出量化结果及量化因子。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_finalize_routing/README.md">grouped_matmul_finalize_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_swiglu_quant/README.md">grouped_matmul_swiglu_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>融合GroupedMatmul 、dequant、swiglu和quant。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_swiglu_quant_v2/README.md">grouped_matmul_swiglu_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>融合GroupedMatmul 、dequant、swiglu和quant，新增了MXFP8量化场景（仅Ascend 950PR/Ascend 950DT AI处理器支持）</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/quant_grouped_matmul_inplace_add/README.md">quant_grouped_matmul_inplace_add</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现分组矩阵乘计算和加法计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/all_gather_matmul/README.md">all_gather_matmul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成AllGather通信与MatMul计算融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_all_all_gather_batch_mat_mul/README.md">allto_all_all_gather_batch_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_all_matmul/README.md">allto_all_matmul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>完成AlltoAll通信与MatMul计算融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_allv_grouped_mat_mul/README.md">allto_allv_grouped_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，先通信后计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/ffn_to_attention/README.md">ffn_to_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>一个通信域内的FFN节点对Attention节点发送数据并写状态位，以检测通信链路是否正常。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/ffn_to_attention_v2/README.md">ffn_to_attention_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>一个通信域内的FFN节点对Attention节点发送数据并写状态位，以检测通信链路是否正常。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/attention_to_ffn/README.md">attention_to_ffn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>一个通信域内的Attention节点对FFN节点发送数据并写状态位，以检测通信链路是否正常。</td>
  </tr>
    <tr>
    <td>mc2</td>
    <td><a href="../../mc2/attention_to_ffn_v2/README.md">attention_to_ffn_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>一个通信域内的Attention节点对FFN节点发送数据并写状态位，以检测通信链路是否正常。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/batch_mat_mul_reduce_scatter_allto_all/README.md">batch_mat_mul_reduce_scatter_allto_all</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/distribute_barrier/README.md">distribute_barrier</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。</td>
  </tr>
   <tr>
    <td>mc2</td>
    <td><a href="../../mc2/distribute_barrier_extend/README.md">distribute_barrier_extend</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/grouped_mat_mul_all_reduce/README.md">grouped_mat_mul_all_reduce</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>在融合GroupedMatMul的基础上实现多卡并行AllReduce功能，实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/grouped_mat_mul_allto_allv/README.md">grouped_mat_mul_allto_allv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合，先计算后通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/inplace_matmul_all_reduce_add_rms_norm/README.md">inplace_matmul_all_reduce_add_rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成mm + all_reduce + add + rms_norm计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_allto_all/README.md">matmul_allto_all</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>完成MatMul计算与AlltoAll通信融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_all_reduce/README.md">matmul_all_reduce</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成MatMul计算与AllReduce通信融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_all_reduce_add_rms_norm/README.md">matmul_all_reduce_add_rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td> 完成mm + all_reduce + add + rms_norm计算</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_reduce_scatter/README.md">matmul_reduce_scatter</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成mm + reduce_scatter_base计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/mega_moe/README.md">mega_moe</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>完成dispatch + group_matmul1 + swiglu_quant + group_matmul2 + combine的端到端融合计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine/README.md">moe_distribute_combine</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_add_rms_norm/README.md">moe_distribute_combine_add_rms_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加），之后完成Add + RmsNorm融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_v2/README.md">moe_distribute_combine_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合。expertScales非空时乘权重再相加，为空Tensor时直接相加。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch/README.md">moe_distribute_dispatch</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_v2/README.md">moe_distribute_dispatch_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_v3/README.md">moe_distribute_dispatch_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_update_expert/README.md">moe_update_expert</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成每个token的topK个专家逻辑专家号到物理卡号的映射。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_compute_expert_tokens/README.md">moe_compute_expert_tokens</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_finalize_routing/README.md">moe_finalize_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE计算中，最后处理合并MoE FFN的输出结果。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_finalize_routing_v2/README.md">moe_finalize_routing_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE计算中，最后处理合并MoE FFN的输出结果。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_finalize_routing_v2_grad/README.md">moe_finalize_routing_v2_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>aclnnMoeFinalizeRoutingV2的反向传播。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k/README.md">moe_gating_top_k</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE计算中，对输入x做Sigmoid、SoftMax或者SqrtSoftplus计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k_backward/README.md">moe_gating_top_k_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoeGatingTopK的反向算子。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k_softmax/README.md">moe_gating_top_k_softmax</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE计算中，对x的输出做Softmax计算，取TopK操作。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k_softmax_v2/README.md">moe_gating_top_k_softmax_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取topk操作；如果renorm=1，先对x的输出做topk操作，再进行Softmax操作。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing/README.md">moe_init_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE的routing计算，根据<a href="../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md">aclnnMoeGatingTopKSoftmax</a>的计算结果做routing处理。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_quant/README.md">moe_init_routing_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE的routing计算，根据<a href="../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md">aclnnMoeGatingTopKSoftmax</a>的计算结果做routing处理，并对结果进行量化。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_quant_v2/README.md">moe_init_routing_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE的routing计算，根据<a href="../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md">aclnnMoeGatingTopKSoftmaxV2</a>的计算结果做routing处理。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v2/README.md">moe_init_routing_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v2_grad/README.md">moe_init_routing_v2_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td><a href="../../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md">aclnnMoeInitRoutingV2</a>的反向传播，完成tokens的加权求和。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v3/README.md">moe_init_routing_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE的routing计算，根据<a href="../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md">aclnnMoeGatingTopKSoftmaxV2</a>的计算结果做routing处理，支持不量化和动态量化模式。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v4/README.md">moe_init_routing_v4</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE的routing计算，根据<a href="../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md">aclnnMoeGatingTopKSoftmaxV2</a>的计算结果做routing处理，支持非量化、静态量化和动态量化模式。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_re_routing/README.md">moe_re_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_re_routing_v2/README.md">moe_re_routing_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列，支持对topkWeight重排。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute/README.md">moe_token_permute</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE的permute计算，根据索引indices将tokens广播并排序。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_grad/README.md">moe_token_permute_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenPermute的反向传播计算。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_ep/README.md">moe_token_permute_with_ep</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_ep_grad/README.md">moe_token_permute_with_ep_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenPermuteWithEp的反向传播计算。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_routing_map/README.md">moe_token_permute_with_routing_map</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE的permute计算，将token和expert的标签作为routingMap传入，根据routingMap将tokens和可选probsOptional广播后排序。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_routing_map_grad/README.md">moe_token_permute_with_routing_map_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute/README.md">moe_token_unpermute</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘；最后进行累加求和，并输出计算结果。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_grad/README.md">moe_token_unpermute_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenUnpermuteGrad的反向传播。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_ep/README.md">moe_token_unpermute_with_ep</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>根据sortedIndices存储的下标位置，去获取permutedTokens中的输入数据与probs相乘，并进行合并累加。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_ep_grad/README.md">moe_token_unpermute_with_ep_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenUnpermuteWithEp的反向传播。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_routing_map/README.md">moe_token_unpermute_with_routing_map</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>对经过aclnnMoeTokenpermuteWithRoutingMap处理的permutedTokens，累加回原unpermutedTokens。根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘，最后进行累加求和，并输出计算结果。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_routing_map_grad/README.md">moe_token_unpermute_with_routing_map_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenUnpermuteWithRoutingMap的反向传播。</td>
  </tr>
    <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_sinkhorn/README.md">mhc_sinkhorn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>基于Sinkhorn-Knopp迭代算法将超连接的混合矩阵投影到双随机矩阵流形，以此稳定深度网络信号传播、解决梯度消失 / 爆炸问题</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_sinkhorn_backward/README.md">mhc_sinkhorn_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>mhc_sinkhorn的反向算子</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_post/README.md">mhc_post</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>基于一系列计算对mHC架构中上一层输出进行Post Mapping，对上一层的输入进行Res Mapping，然后对二者进行残差连接，得到下一层的输入</td>
  </tr>
   <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_post_backward/README.md">mhc_post_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>mhc_post基于一系列计算对mHC架构中上一层输出进行Post Mapping，对上一层的输入进行Res Mapping，然后对二者进行残差连接，得到下一层的输入。该算子实现前述过程的反向计算。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre/README.md">mhc_pre</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>基于一系列计算得到MHC架构中hidden层的$H^{res}$和$H^{post}$投影矩阵以及Attention或MLP层的输入矩阵$h^{in}$。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre_backward/README.md">mhc_pre_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>mhc_pre算子的反向传播，基于一系列计算得到MHC架构中hidden层的梯度。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre_sinkhorn/README.md">mhc_pre_sinkhorn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>mhc_pre_sinkhorn算子，支持MHC架构中Hin、Hpre、Hres和Hpost矩阵输出。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre_sinkhorn_backward/README.md">mhc_pre_sinkhorn_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>mhc_pre_sinkhorn算子的反向传播，基于一系列计算得到MHC架构中hidden层的梯度。</td>
  </tr>
  <tr>
    <td>mamba</td>
    <td><a href="../../mamba/causal_conv1d/README.md">causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成因果一维卷积（Causal Conv1d）计算，支持前向计算（prefill/chunk-prefill）和状态更新（decode/update）两种运行模式，支持APC（Automatic Prefix Caching）、MTP（投机解码）等特性。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/apply_rotary_pos_emb/README.md">apply_rotary_pos_emb</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>执行旋转位置编码计算，推理网络为了提升性能，将query和key两路算子融合成一路。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/apply_rotary_pos_emb_grad/README.md">apply_rotary_pos_emb_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>执行双路旋转位置编码ApplyRotaryPosEmb的反向计算，同时计算query和key的rope反向梯度，融合为一次kernel调用。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/dequant_rope_quant_kvcache/README.md">dequant_rope_quant_kvcache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量（x）进行dequant（可选）后，按`sizeSplits`（为切分的长度）对尾轴进行切分，划分为q、k、vOut，对q、k进行旋转位置编码，生成qOut和kOut，之后对kOut和vOut进行量化并按照`indices`更新到kCacheRef和vCacheRef上。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/interleave_rope/README.md">interleave_rope</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>针对单输入x进行旋转位置编码。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/inplace_partial_rotary_mul/README.md">inplace_partial_rotary_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>执行单路旋转位置编码的Inplace计算，直接修改输入张量，不产生新的输出张量。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/inplace_partial_rotary_mul_grad/README.md">inplace_partial_rotary_mul_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>执行局部旋转位置编码InplacePartialRotaryMul的反向计算，对输入dy的D维度上切片[start, end)区域执行旋转位置编码梯度计算，结果inplace写回dy。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/qkv_rms_norm_rope_cache/README.md">qkv_rms_norm_rope_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>输入qkv融合张量，通过SplitVD拆分q、k、v张量，执行RmsNorm、ApplyRotaryPosEmb、Quant、Scatter融合操作，输出q_out、k_cache、v_cache、q_out_before_quant(可选)、k_out_before_quant(可选)、v_out_before_quant(可选)。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/qkv_rms_norm_rope_cache_with_k_scale/README.md">qkv_rms_norm_rope_cache_with_k_scale</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>融合Q/K/V拆分、Q/K RMSNorm、RoPE/M-RoPE、量化和KV Cache更新，支持M-RoPE MX场景。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rope_quant_kvcache/README.md">rope_quant_kvcache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量按照‘sizeSplits’对尾轴进行切分，划分为q、k、vOut，对q、k进行旋转位置编码，生成qOut与kOut，之后对kOut与vOut进行量化并按照‘indices’更新到kCacheRef和vCacheRef上。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rope_with_sin_cos_cache/README.md">rope_with_sin_cos_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>推理网络为了提升性能，将sin和cos输入通过cache传入，执行旋转位置编码计算。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rotary_position_embedding/README.md">rotary_position_embedding</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>执行单路旋转位置编码计算。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rotary_position_embedding_grad/README.md">rotary_position_embedding_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>执行单路旋转位置编码的反向计算。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/kv_rms_norm_rope_cache/README.md">kv_rms_norm_rope_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对输入张量(kv)的尾轴，拆分出左半边用于rms_norm计算，右半边用于rope计算，再将计算结果分别scatter到两块cache中。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/und_gen_qkv_rms_norm_rope_cache/README.md">und_gen_qkv_rms_norm_rope_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>输入understand段与generate段两路QKV融合张量，按cat_indices重排后拆分Q、K、V；对Q、K分别用两段各自的权重执行RMSNorm并做MRoPE多轴旋转位置编码，输出q；K、V按slot_mapping散写更新分页k_cache与v_cache。</td>
  </tr>
</tbody>
</table>
