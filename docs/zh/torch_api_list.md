# torch_extension接口

## 使用说明

为简化算子调用，项目提供了一套兼容PyTorch原生风格的API。该API通过PyTorch的JIT机制（`torch.utils.cpp_extension.load`），在首次调用时即时编译C++ Kernel Wrapper，将PyTorch函数桥接到CANN的aclnn API，同时通过GE Converter支持TorchAir图模式，便于开发者构建模型与应用。

- **软件包说明**

  调用`torch_extension`接口时，请确保已安装CANN Toolkit包、ops-transformer包、Ascend for PyTorch包。

- **调用方式**：

  调用`torch_extension`接口时，依赖如下模块，其中`cann-ops-transformer`模块定义在`${INSTALL_DIR}/python/sitepackage/cann-ops-transformer`目录。

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer
  ```
  
- **V版本演进说明**

  请注意，部分API存在多个V版本，使用时选择最高V版本即可（高版本API已兼容低版本API的所有能力）。

## 接口列表

> **确定性简介**：因CANN或NPU型号不同等原因，可能无法保证同一个API运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分接口可通过PyTorch中控制算法确定性的全局开关[torch.use_deterministic_algorithms](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py)开启确定性算法，使多次运行结果一致。

|    接口名   |   说明     | 确定性说明（A2/A3）  | 确定性说明（Ascend 950） |
| ----------- | ------------------- | ------------------- | ------------------- |
|[flash_attn](../../torch_extension/cann_ops_transformer/docs/zh/flash_attn.md)|FlashAttention非量化注意力计算。|-|默认支持确定性计算|
|[get_low_latency_ccl_buffer_size](../../torch_extension/cann_ops_transformer/docs/zh/get_low_latency_ccl_buffer_size.md)|需与low_latency_dispatch和low_latency_combine配套使用，用于计算dispatch_v3和combine_v3算子所需的HCCL通信buffer_size大小（单位：MB）。|-|默认确定性实现|
|[lightning_indexer](../../torch_extension/cann_ops_transformer/docs/zh/lightning_indexer.md)|基于一系列操作得到每一个token对应的Top-k个位置。支持KV压缩场景。|默认确定性实现|-|
|[low_latency_combine](../../torch_extension/cann_ops_transformer/docs/zh/low_latency_combine.md)|需与low_latency_dispatch配套使用，相当于按low_latency_dispatch算子收集数据的路径原路返回。|默认确定性实现|-|
|[low_latency_dispatch](../../torch_extension/cann_ops_transformer/docs/zh/low_latency_dispatch.md)|需与low_latency_combine配套使用，完成MoE的并行部署下的token的dispatch和combine。|默认确定性实现|-|
|[mega_moe](../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md)|将MoE层的专家FFN完整计算流程及前后数据通信（即 Dispatch + Linear1 + SwiGLU + Linear2 + Combine）融合为单个算子，实现通信和计算的掩盖。|默认确定性实现|默认确定性实现|
|[mhc_post](../../torch_extension/cann_ops_transformer/docs/zh/mhc_post.md)|实现MHC Post组件的前向计算，用于Transformer模型中多层残差连接的后处理阶段。该算子将残差矩阵变换与输出状态投影融合为单次计算，避免多次独立算子调用带来的额外开销。|默认确定性实现|-|
|[mhc_pre_sinkhorn](../../torch_extension/cann_ops_transformer/docs/zh/mhc_pre_sinkhorn.md)|基于一系列计算得到MHC架构中hidden层的$\mathbf{H}'_{\text{res}}$和$\mathbf{H}_{\text{post}}$投影矩阵以及Attention或MLP层的输入矩阵$\mathbf{h}_{\text{in}}$。对$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出中间计算结果，用于反向梯度计算。|默认确定性实现|-|
|[sparse_flash_mla](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md)|基于共享KV完成SparseFlashMla稀疏注意力计算。|默认确定性实现|-|
|[sparse_flash_mla_grad](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla_grad.md)|训练场景下计算SparseFlashMla注意力反向输出，即[sparse_flash_mla](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md)的反向计算。|默认不支持确定性计算|-|
|[sparse_lightning_indexer_kl_loss_grad](../../torch_extension/cann_ops_transformer/docs/zh/sparse_lightning_indexer_kl_loss_grad.md)|计算 Lightning Indexer KL Loss训练场景下的反向输出。|默认确定性实现|-|
