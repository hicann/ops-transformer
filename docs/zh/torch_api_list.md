# torch_extension接口

## 使用说明

为简化算子调用，项目提供了一套兼容PyTorch原生风格的API。该API通过PyTorch的JIT机制（`torch.utils.cpp_extension.load`），在首次调用时即时编译C++ Kernel Wrapper，将PyTorch函数桥接到CANN的aclnn API，同时通过GE Converter支持TorchAir图模式，便于开发者构建模型与应用。

- **软件包说明**

  调用torch\_extension接口时，请确保已安装CANN Toolkit包、ops-transformer包、Ascend for PyTorch包。

- **调用方式**：

  调用torch\_extension接口时，依赖`cann-ops-transformer`模块，定义在`${INSTALL_DIR}/python/sitepackage/cann-ops-transformer`，\$\{INSTALL\_DIR\}表示CANN安装后文件路径。

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer
  ```

- **V版本演进说明**

  请注意，部分API存在多个V版本，使用时选择最高V版本即可（高版本API已兼容低版本API的所有能力）。

## 接口列表

> **确定性简介**：因CANN或NPU型号不同等原因，可能无法保证同一个API运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分接口可通过PyTorch中控制算法确定性的全局开关[torch.use_deterministic_algorithms](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py)开启确定性算法，使多次运行结果一致。

|    接口名   |   说明     |  确定性说明（A2/A3）  | 确定性说明（Ascend 950） |
| ----------- | ------------------- | ------------------- | ------------------- |
|[causal_conv1d_fn](../../torch_extension/cann_ops_transformer/docs/zh/causal_conv1d_fn.md)| 因果一维卷积前向计算（prefill/chunk-prefill），封装aclnnCausalConv1dFn。| - | 默认支持确定性计算 |
|[causal_conv1d_update](../../torch_extension/cann_ops_transformer/docs/zh/causal_conv1d_update.md)| 因果一维卷积状态更新（decode/update），封装aclnnCausalConv1dUpdate。 | - | 默认支持确定性计算 |
|[compressor](../../torch_extension/cann_ops_transformer/docs/zh/compressor.md)| 将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。| 默认支持确定性计算 | 默认支持确定性计算  |
|[dense_lightning_indexer_softmax_lse](../../torch_extension/cann_ops_transformer/docs/zh/dense_lightning_indexer_softmax_lse.md)| dense场景DenseLightningIndexerGradKlLoss算子计算Softmax输入的一个分支算子。支持压缩注意力（Compressed Attention），并支持通过metadata前置算子进行分核负载均衡。需与`dense_lightning_indexer_softmax_lse_metadata`配套使用。|-|默认确定性实现|
|[dense_lightning_indexer_softmax_lse_metadata](../../torch_extension/cann_ops_transformer/docs/zh/dense_lightning_indexer_softmax_lse.md)| dense_lightning_indexer_softmax_lse接口的前置接口，用于计算dense_lightning_indexer_softmax_lse的负载均衡。|-|默认确定性实现|
|[flash_attn](../../torch_extension/cann_ops_transformer/docs/zh/flash_attn.md)| 调用`FlashAttn`算子完成共享KV（Key和Value使用同一份输入）的非量化注意力计算，训练推理归一化。需与`flash_attn_metadata`配套使用。 | - | 默认支持确定性计算  |
|[get_low_latency_ccl_buffer_size](../../torch_extension/cann_ops_transformer/docs/zh/get_low_latency_ccl_buffer_size.md)|计算low_latency_dispatch/low_latency_combine所需的HCCL通信buffer_size（单位MB），为MoeDistributeBuffer的静态方法，可在初始化前调用。|默认支持确定性计算|默认支持确定性计算|
|[grouped_matmul_activation_quant](../../torch_extension/cann_ops_transformer/docs/zh/grouped_matmul_activation_quant.md)|融合GMM、激活函数和量化算子，完成分组矩阵乘、激活和量化计算，输出量化结果及量化因子。|-|默认确定性实现|
|[indexer_quant_cache](../../torch_extension/cann_ops_transformer/docs/zh/indexer_quant_cache.md)| 在Indexer注意力机制的Epilog阶段对KV Cache进行原地动态量化压缩更新，封装aclnnIndexerQuantCache。  |-|默认确定性实现|
|[inplace_partial_rotary_mul](../../torch_extension/cann_ops_transformer/docs/zh/inplace_partial_rotary_mul.md)|执行单路旋转位置编码的Inplace计算，直接修改输入张量，不产生新的输出张量。|默认确定性实现|默认确定性实现|
|[inplace_partial_rotary_mul_backward](../../torch_extension/cann_ops_transformer/docs/zh/inplace_partial_rotary_mul_backward.md)|执行`inplace_partial_rotary_mul`的反向计算，对输入梯度张量执行inplace更新，切片内替换为RoPE梯度，切片外保持不变。|-|默认支持确定性计算|
|[kv_compress_epilog](../../torch_extension/cann_ops_transformer/docs/zh/kv_compress_epilog.md)| 在KV Cache的Epilog阶段对cache进行原地量化压缩更新，封装aclnnKvCompressEpilog。|默认确定性实现|-|
|[lightning_indexer](../../torch_extension/cann_ops_transformer/docs/zh/lightning_indexer.md)| 基于一系列操作得到每一个token对应的Top-k个位置。支持KV压缩场景。|默认确定性实现|-|
|[lightning_indexer_metadata](../../torch_extension/cann_ops_transformer/docs/zh/lightning_indexer.md)| lightning_indexer接口的前置接口，用于计算lightning_indexer的负载均衡。|默认确定性实现|默认确定性实现|
|[low_latency_dispatch](../../torch_extension/cann_ops_transformer/docs/zh/low_latency_dispatch.md)|完成MoE并行部署下token的低时延dispatch分发，支持动态量化与EP域alltoallv通信，需与low_latency_combine配套使用。|默认支持确定性计算|默认支持确定性计算|
|[low_latency_combine](../../torch_extension/cann_ops_transformer/docs/zh/low_latency_combine.md)|与low_latency_dispatch配套，按dispatch原路返回完成token的低时延combine反向聚合；topk_weights非空时乘路由权重再相加，为None时直接相加。|默认支持确定性计算|默认支持确定性计算|
|[mega_moe](../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md)|MoE端到端通算融合算子，将Dispatch+GroupMatmul1+SwiGLUQuant+GroupMatmul2+Combine融合为单算子；配套get_mega_moe_ccl_buffer_size、get_symm_buffer_for_mega_moe使用。|-|默认支持确定性计算|
|[mhc_post](../../torch_extension/cann_ops_transformer/docs/zh/mhc_post.md)|实现MHC Post组件的前向计算，用于Transformer模型中多层残差连接的后处理阶段。该算子将残差矩阵变换与输出状态投影融合为单次计算，避免多次独立算子调用带来的额外开销。|默认确定性实现|-|
|[mhc_pre_sinkhorn](../../torch_extension/cann_ops_transformer/docs/zh/mhc_pre_sinkhorn.md)|基于一系列计算得到MHC架构中hidden层的$\mathbf{H}'_{\text{res}}$和$\mathbf{H}_{\text{post}}$投影矩阵以及Attention或MLP层的输入矩阵$\mathbf{h}_{\text{in}}$。对$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出中间计算结果，用于反向梯度计算。|默认确定性实现|默认确定性实现|
|[mixed_quant_sparse_flash_mla](../../torch_extension/cann_ops_transformer/docs/zh/mixed_quant_sparse_flash_mla.md)|量化场景下基于共享KV完成MixedQuantSparseFlashMla稀疏注意力计算。需与`mixed_quant_sparse_flash_mla_metadata`配套使用。|默认确定性实现|默认确定性实现|
|[moe_finalize_routing](../../torch_extension/cann_ops_transformer/docs/zh/moe_finalize_routing.md)|将各专家FFN的输出结果按路由权重加权合并，还原为原始token序列。|默认支持确定性计算。|默认支持确定性计算。|
|[moe_init_routing](../../torch_extension/cann_ops_transformer/docs/zh/moe_init_routing.md)|MoE的routing计算，根据moe_gating_top_k_softmax的计算结果做routing处理，支持不量化、静态量化和动态量化模式。|默认确定性实现|默认确定性实现|
|[moe_re_routing](../../torch_extension/cann_ops_transformer/docs/zh/moe_re_routing.md)|MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。支持对topkWeight重排。|默认确定性实现|默认确定性实现|
|[moe_token_permute](../../torch_extension/cann_ops_transformer/docs/zh/moe_token_permute.md)|根据专家索引扩展并排序token。|默认支持确定性计算。|
|[qkv_rms_norm_rope_cache_with_k_scale](../../torch_extension/cann_ops_transformer/docs/zh/qkv_rms_norm_rope_cache_with_k_scale.md)|融合Q/K/V拆分、Q/K RMSNorm、RoPE、共享rotation矩阵乘、FP8量化和KV Cache更新，返回更新后的cache副本。|-|默认支持确定性计算。|
|[quant_compressor](../../torch_extension/cann_ops_transformer/docs/zh/quant_compressor.md)|Compressor的量化版本，将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。|-|默认支持确定性计算|
|[quant_flash_attn](../../torch_extension/cann_ops_transformer/docs/zh/quant_flash_attn.md)| 调用`QuantFlashAttn`算子完成MxFP8/HiF8/MxFP4量化场景下的全量化注意力计算，训练推理归一化。|默认支持确定性计算。|
|[quant_lightning_indexer](../../torch_extension/cann_ops_transformer/docs/zh/quant_lightning_indexer.md)| 基于一系列操作得到每一个token对应的top-k个位置。|默认支持确定性计算。|
|[quant_sparse_flash_mla](../../torch_extension/cann_ops_transformer/docs/zh/quant_sparse_flash_mla.md)|调用`QuantSparseFlashMla`算子完成共享KV（Key和Value使用同一份输入）的稀疏注意力计算。|默认支持确定性计算。|
|[scatter_pa_kv_cache_with_k_scale](../../torch_extension/cann_ops_transformer/docs/zh/scatter_pa_kv_cache_with_k_scale.md)|训练场景下，更新KvCache中指定位置的key和value，同时更新key的scale值。|-|默认支持确定性计算|
|[sparse_flash_mla](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md)|基于共享KV完成SparseFlashMla稀疏注意力计算。需与`sparse_flash_mla_metadata`配套使用。 |默认确定性实现|默认确定性实现|
|[sparse_flash_mla_grad](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla_grad.md)|计算`SparseFlashMla`训练场景下注意力的反向输出，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。需与`sparse_flash_mla_grad_metadata`配套使用。 |-|默认确定性实现|
|[sparse_lightning_indexer_kl_loss_grad](../../torch_extension/cann_ops_transformer/docs/zh/sparse_lightning_indexer_kl_loss_grad.md)| Lightning Indexer KL Loss训练场景下的反向输出。需与`sparse_lightning_indexer_kl_loss_grad_metadata`配套使用。|默认确定性实现|默认确定性实现|
|[stem_oam_prep_varlen_q](../../torch_extension/cann_ops_transformer/docs/zh/stem_oam_prep_varlen_q.md)|完成Stem OAM block-sparse attention中Q侧预处理计算，将变长Q tensor转化为按stem block分组的flattened qFlat输出。|-|默认确定性实现|
|[dense_lightning_indexer_kl_loss_grad](../../torch_extension/cann_ops_transformer/docs/zh/dense_lightning_indexer_kl_loss_grad.md)| Lightning Indexer KL Loss训练Dense场景下的反向输出。需与`dense_lightning_indexer_kl_loss_grad_metadata`配套使用。| - |默认确定性实现|
|[sparse_flash_mla_softmax_l1_norm](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla_softmax_l1_norm.md)|`dense_lightning_indexer_kl_loss_grad`的前置接口，生成attn_softmax_l1_norm。需与`sparse_flash_mla_softmax_l1_norm_metadata`配套使用。 |-|默认确定性实现|
|[stem_oam_prep_paged_kv](../../torch_extension/cann_ops_transformer/docs/zh/stem_oam_prep_paged_kv.md)| 大模型推理动态稀疏注意力机制的前置评分模块，为block-sparse-attention的前置评分模块。| - |默认确定性实现|
