# Ascend 950算子列表

> **使用说明**：
>
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./install/dir_structure.md)。
> - **算子执行硬件单元**：大部分算子运行在AI Core，少部分算子运行在AI CPU。默认情况下，项目中提到的算子一般指AI Core算子。关于AI Core和AI CPU详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)，其中版本号大于等于8.5.0中对应章节为"硬件实现"，其余版本中对应章节为"概念原理和术语 > 硬件架构与数据处理原理"。
> - **算子接口列表**：为方便调用算子，CANN提供一套C API执行算子，一般以aclnn为前缀，全量接口参见[aclnn列表](op_api_list.md)。
> - **V版本演进说明**：部分算子存在多个V版本，使用时选择最高V版本即可（高版本算子已兼容低版本算子的所有能力）。

Ascend 950支持的算子分类和算子列表如下：

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
    <td>✓</td>
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
    <td>将多个计算单元处理的注意力token数据进行融合，结合专家权重对结果进行加权，输出最终的注意力融合结果，并更新层ID。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_sparse_attention/README.md">block_sparse_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>训练场景下BlockSparseAttention的正向计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_softmax_lse/README.md">dense_lightning_indexer_softmax_lse</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>DenseLightningIndexerSoftmaxLse算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attention_score/README.md">flash_attention_score</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>使用FlashAttention算法实现self-attention（自注意力）的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attention_score_grad/README.md">flash_attention_score_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>训练场景下计算注意力的反向输出，即FlashAttentionScore的反向计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attn/README.md">flash_attn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算FlashAttention前向输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attn_grad/README.md">flash_attn_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算FlashAttention反向梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_causal_conv1d/README.md">fused_causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接等特性。</td>
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
    <td><a href="../../attention/generic_block_sparse_attention_grad/README.md">generic_block_sparse_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>通用块稀疏注意力反向算子，依据rsvdBlockIdx/rsvdBlockCount在被选中的Q块上计算并传播梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/generic_block_sparse_attention_grad_metadata/README.md">generic_block_sparse_attention_grad_metadata</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI CPU</td>
    <td>generic_block_sparse_attention_grad算子的前置算子，用于计算generic_block_sparse_attention_grad的负载均衡。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/inplace_fused_causal_conv1d/README.md">inplace_fused_causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接、原地更新等特性。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kv_quant_sparse_flash_attention/README.md">kv_quant_sparse_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>在Sparse Flash Attention的基础上支持了`Per-Token-Head-Tile-128量化`输入。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer/README.md">lightning_indexer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>基于一系列操作得到每一个token对应的Top-k个位置。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_v2/README.md">lightning_indexer_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>根据Query、Key和权重计算每个token的Top-k位置。</td>
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
    <td><a href="../../attention/mla_prolog_v3/README.md">mla_prolog_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>推理MlaPrologV3WeightNz算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_lightning_indexer/README.md">quant_lightning_indexer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>推理场景下，SparseFlashAttention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/chunk_kda_fwd/README.md">chunk_kda_fwd</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>完成不涉及CP切分的KDA分块正向计算。</td>
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
    <td>✓</td>
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
    <td><a href="../../attention/scatter_pa_kv_cache/README.md">scatter_pa_kv_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>更新KCache和VCache中指定位置的key和value。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_attention/README.md">sparse_flash_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
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
    <td>SparseFlashAttention的反向梯度计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_lightning_indexer_grad_kl_loss/README.md">sparse_lightning_indexer_grad_kl_loss</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>SparselightningIndexerGradKlLoss算子是LightningIndexer的反向算子，再额外融合了Loss计算功能输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/stem_oam_prep_paged_kv/README.md">stem_oam_prep_paged_kv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>大模型推理动态稀疏注意力机制的前置评分模块，为block-sparse-attention的前置评分模块。</td>
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
    <td><a href="../../attention/block_attn_res_prepare/README.md">block_attn_res_prepare</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算历史残差块间注意力，输出Softmax加权结果和统计量。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_attn_res_update/README.md">block_attn_res_update</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>更新局部残差，并结合在线Softmax状态计算注意力输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/block_sparse_attention_grad/README.md">block_sparse_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算块稀疏注意力的反向梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/bsa_select_block_mask/README.md">bsa_select_block_mask</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>根据Query和Key生成块稀疏注意力掩码。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/chunk_gated_delta_rule/README.md">chunk_gated_delta_rule</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算分块Gated Delta Rule。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_grad_kl_loss/README.md">dense_lightning_indexer_grad_kl_loss</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算Dense LightningIndexer的反向梯度和KL Loss。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_kl_loss_grad/README.md">dense_lightning_indexer_kl_loss_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算Dense LightningIndexer KL Loss的反向梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/dense_lightning_indexer_softmax_lse_v2/README.md">dense_lightning_indexer_softmax_lse_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算Dense LightningIndexer的注意力评分和LSE。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_mla_with_kvcache/docs/torchapi_flash_mla_with_kvcache.md">flash_mla_with_kvcache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>使用KV Cache计算非量化MLA注意力。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_floyd_attention/README.md">fused_floyd_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算FloydAttention前向输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_floyd_attention_grad/README.md">fused_floyd_attention_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算FloydAttention反向梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/indexer_quant_cache/README.md">indexer_quant_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>量化输入并按指定位置更新Indexer缓存。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kda_input_proj/README.md">kda_input_proj</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>对隐藏状态做投影，生成Recurrent KDA所需的qkv、beta、gate和g。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/key_pool/README.md">key_pool</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>压缩注意力计算中的Key。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kv_compress_epilog/README.md">kv_compress_epilog</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>压缩并原地更新KV Cache。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kv_quant_sparse_flash_attention_v2/README.md">kv_quant_sparse_flash_attention_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算支持Per-Token-Head-Tile-128量化输入的稀疏注意力。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer_grad/README.md">lightning_indexer_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>根据稀疏索引计算Query、Key和权重的梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mixed_quant_sparse_flash_mla/README.md">mixed_quant_sparse_flash_mla</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算支持KV量化输入的稀疏MLA注意力。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog/README.md">mla_prolog</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算MLA注意力的前处理结果。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog_v2/README.md">mla_prolog_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算MLA注意力的前处理结果。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/msa_index_score/README.md">msa_index_score</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算MSA索引分支的块评分。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/pool_key_indexer/README.md">pool_key_indexer</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算每个token对应的Top-k位置。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_compressor/README.md">quant_compressor</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算Compressor的量化输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_flash_attn/docs/torchapi_quant_flash_attn.md">quant_flash_attn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算量化FlashAttention前向输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_flash_attn_grad/README.md">quant_flash_attn_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算量化FlashAttention反向梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_flash_mla_with_kvcache/docs/torchapi_quant_flash_mla_with_kvcache.md">quant_flash_mla_with_kvcache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>使用KV Cache计算量化MLA注意力。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_lightning_indexer_v2/README.md">quant_lightning_indexer_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>根据量化Query和Key计算Top-k稀疏索引。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_sparse_flash_mla/README.md">quant_sparse_flash_mla</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算全量化稀疏MLA注意力。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/rain_fusion_attention/README.md">rain_fusion_attention</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>按selectIdx指定的稀疏块计算注意力。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/scatter_pa_kv_cache_with_k_scale/README.md">scatter_pa_kv_cache_with_k_scale</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>更新KV Cache中指定位置的Key、Value及Key的量化scale。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla/README.md">sparse_flash_mla</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算稀疏MLA注意力。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_grad/README.md">sparse_flash_mla_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算稀疏MLA注意力的反向梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_mla_softmax_l1_norm/README.md">sparse_flash_mla_softmax_l1_norm</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算稀疏MLA注意力的Softmax L1范数。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_lightning_indexer_kl_loss_grad/README.md">sparse_lightning_indexer_kl_loss_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算稀疏LightningIndexer KL Loss的反向梯度。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/stem_oam_prep_varlen_q/README.md">stem_oam_prep_varlen_q</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算变长Query的Stem OAM注意力前处理结果。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul/README.md">grouped_matmul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
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
    <td><a href="../../gmm/grouped_matmul_finalize_routing/README.md">grouped_matmul_finalize_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_swiglu_quant_v2/README.md">grouped_matmul_swiglu_quant_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>融合GroupedMatmul 、dequant、swiglu和quant，新增了MXFP8量化场景（仅Ascend 950PR&950DT系列产品 AI处理器支持）。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/quant_grouped_matmul_inplace_add/README.md">quant_grouped_matmul_inplace_add</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>实现分组矩阵乘计算和加法计算。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_activation_quant/README.md">grouped_matmul_activation_quant</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>融合分组矩阵乘、激活和量化计算。</td>
  </tr>
  <tr>
    <td>mamba</td>
    <td><a href="../../mamba/causal_conv1d/README.md">causal_conv1d</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算因果一维卷积并更新状态。</td>
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
    <td><a href="../../mc2/all_gather_matmul_v2/README.md">all_gather_matmul_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成AllGather通信与MatMul计算融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/all_gather_matmul_v3/docs/torchapi_all_gather_quant_matmul.md">all_gather_matmul_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>融合AllGather通信与MX量化矩阵乘。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_all_matmul/README.md">allto_all_matmul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
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
    <td>完成路由专家AlltoAllV、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，先通信后计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_allv_quant_grouped_mat_mul/README.md">allto_allv_quant_grouped_mat_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成路由专家AlltoAllV、Permute、QuantGroupedMatMul融合并实现与共享专家MatMul并行融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/3rd/batch_mat_mul_v3/README.md">batch_mat_mul_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现批量矩阵乘计算。</td>
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
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成通信域内的全卡同步扩展版本。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/grouped_mat_mul_allto_allv/README.md">grouped_mat_mul_allto_allv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成路由专家GroupedMatMul、Unpermute、AlltoAllV融合并实现与共享专家MatMul并行融合，先计算后通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/3rd/mat_mul_v3/README.md">mat_mul_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>实现矩阵乘计算。</td>
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
    <td><a href="../../mc2/matmul_allto_all/README.md">matmul_allto_all</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成MatMul计算与AlltoAll通信融合。</td>
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
    <td><a href="../../mc2/matmul_reduce_scatter_v2/README.md">matmul_reduce_scatter_v2</a></td>
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
    <td>✓</td>
    <td>✓</td>
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
    <td><a href="../../mc2/moe_distribute_combine_setup/README.md">moe_distribute_combine_setup</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoeDistributeCombine的setup阶段，用于初始化通信资源。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_teardown/README.md">moe_distribute_combine_teardown</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoeDistributeCombine的teardown阶段，用于释放通信资源。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_v2/README.md">moe_distribute_combine_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_v3/README.md">moe_distribute_combine_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch/README.md">moe_distribute_dispatch</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AlltoAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AlltoAllV通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_setup/README.md">moe_distribute_dispatch_setup</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoeDistributeDispatch的setup阶段，用于初始化通信资源。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_teardown/README.md">moe_distribute_dispatch_teardown</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoeDistributeDispatch的teardown阶段，用于释放通信资源。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_v2/README.md">moe_distribute_dispatch_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AlltoAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AlltoAllV通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_v3/README.md">moe_distribute_dispatch_v3</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AlltoAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AlltoAllV通信。</td>
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
    <td>mc2</td>
    <td><a href="../../mc2/quant_all_reduce/README.md">quant_all_reduce</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成量化后的AllReduce通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/quant_grouped_mat_mul_allto_allv/README.md">quant_grouped_mat_mul_allto_allv</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成路由专家QuantGroupedMatMul、Unpermute、AlltoAllV融合并实现与共享专家MatMul并行融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/quant_reduce_scatter/README.md">quant_reduce_scatter</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>完成量化后的ReduceScatter通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_all_matmul_v2/docs/torchapi_all_to_all_quant_matmul.md">allto_all_matmul_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>融合AlltoAll通信与MX量化矩阵乘。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/attention_to_ffn_v2/README.md">attention_to_ffn_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将Attention节点的数据发送到FFN节点。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/engram_fetch">engram_fetch</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>按索引获取Engram数据。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/engram_fetch_grad">engram_fetch_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算Engram数据获取操作的反向梯度。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/engram_fetch_wait">engram_fetch_wait</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>等待Engram数据获取完成。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/ffn_to_attention_v2/README.md">ffn_to_attention_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>将FFN节点的数据发送到Attention节点。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_ep_combine">moe_ep_combine</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>合并MoE专家并行计算后的token。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_ep_combine_epilogue">moe_ep_combine_epilogue</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>处理MoE专家并行token合并的结果。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_ep_dispatch">moe_ep_dispatch</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>向MoE专家并行节点分发token。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_ep_dispatch_epilogue">moe_ep_dispatch_epilogue</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>处理MoE专家并行token分发的结果。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_post/README.md">mhc_post</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>基于一系列计算对mHC架构中上一层输出进行Post Mapping，对上一层的输入进行Res Mapping，然后对二者进行残差连接，得到下一层的输入。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_post_backward/README.md">mhc_post_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>mhc_post算子的反向传播。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre/README.md">mhc_pre</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
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
    <td><a href="../../mhc/mhc_sinkhorn/README.md">mhc_sinkhorn</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>基于用Sinkhorn-Knopp迭代算法将超连接的混合矩阵投影到双随机矩阵流形，以此稳定深度网络信号传播、解决梯度消失/爆炸问题。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_sinkhorn_backward/README.md">mhc_sinkhorn_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>mhc_sinkhorn的反向算子。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/block_attention_residuals_grad/README.md">block_attention_residuals_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>BlockAttentionResiduals的反向算子，计算partial_block、block_res、proj_weight、norm_weight的梯度。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/block_attention_residuals/README.md">block_attention_residuals</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>对残差块做归一化、投影打分和Softmax加权求和。</td>
  </tr>
  <tr>
    <td>mhc</td>
    <td><a href="../../mhc/mhc_pre_sinkhorn_backward/README.md">mhc_pre_sinkhorn_backward</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算MhcPreSinkhorn反向梯度。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_compute_expert_tokens/README.md">moe_compute_expert_tokens</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。</td>
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
    <td>MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。</td>
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
    <td>✓</td>
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
    <td><a href="../../moe/moe_re_routing/README.md">moe_re_routing</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_routing_map/README.md">moe_token_permute_with_routing_map</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_routing_map_grad/README.md">moe_token_permute_with_routing_map_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenPermuteWithRoutingMap的反向传播。</td>
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
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenUnpermuteWithRoutingMap的反向传播。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/3rd/moe_inplace_index_add/README.md">moe_inplace_index_add</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE中根据索引进行原地加法操作。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/3rd/moe_inplace_index_add_with_sorted/README.md">moe_inplace_index_add_with_sorted</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE中根据排序后的索引进行原地加法操作。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/3rd/moe_masked_scatter/README.md">moe_masked_scatter</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>MoE中根据mask进行scatter操作。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_fused_topk/README.md">moe_fused_topk</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>对输入做Sigmoid和分组排序，选出Top-k专家。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v4/README.md">moe_init_routing_v4</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>根据专家选择结果重排token，支持非量化、静态量化和动态量化。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_re_routing_v2/README.md">moe_re_routing_v2</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>将AlltoAll接收的token按专家顺序重排。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_ep/README.md">moe_token_permute_with_ep</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>按专家并行索引重排token和可选权重，并截取指定范围。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_ep_grad/README.md">moe_token_permute_with_ep_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算专家并行token重排的反向梯度。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_ep/README.md">moe_token_unpermute_with_ep</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>按索引还原token顺序，并加权合并。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_ep_grad/README.md">moe_token_unpermute_with_ep_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>计算专家并行token还原操作的反向梯度。</td>
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
    <td>✓</td>
    <td>AI Core</td>
    <td>执行双路旋转位置编码ApplyRotaryPosEmb的反向计算，同时计算query和key的rope反向梯度，融合为一次kernel调用。</td>
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
    <td><a href="../../posembedding/norm_rope_concat/README.md">norm_rope_concat</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>执行RmsNorm、RoPE和Concat融合计算。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/norm_rope_concat_grad/README.md">norm_rope_concat_grad</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>NormRopeConcat的反向梯度计算。</td>
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
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>执行单路旋转位置编码的反向计算。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/dequant_rope_quant_kvcache/README.md">dequant_rope_quant_kvcache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>融合反量化、Q/K/V拆分、旋转位置编码、量化和KV Cache更新。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/inplace_partial_rotary_mul/README.md">inplace_partial_rotary_mul</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>原地计算单路旋转位置编码。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/qkv_rms_norm_rope_cache/README.md">qkv_rms_norm_rope_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>融合Q/K/V拆分、RMS归一化、旋转位置编码、量化和KV Cache更新。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/qkv_rms_norm_rope_cache_with_k_scale/README.md">qkv_rms_norm_rope_cache_with_k_scale</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✗</td>
    <td>AI Core</td>
    <td>拆分Q/K/V并更新KV Cache及Key的量化scale。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rope_quant_kvcache/README.md">rope_quant_kvcache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>拆分Q/K/V，对Q/K做旋转位置编码，并量化更新KV Cache。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/und_gen_qkv_rms_norm_rope_cache/README.md">und_gen_qkv_rms_norm_rope_cache</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>计算多模态模型的Q/K/V前处理结果。</td>
  </tr>
  <tr>
    <td>examples</td>
    <td><a href="../../examples/add_example/README.md">add_example</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>示例算子，用于演示算子开发流程。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/ffn_worker_batching/README.md">ffn_worker_batching</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>重排Attention与FFN分离部署时FFN节点上的token。</td>
  </tr>
</tbody>
</table>
