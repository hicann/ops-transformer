## 1 attention layer 算子分类原则

attention layer 算子按是否使用 Softmax 分为两大类：Softmax Attention（基于 Softmax 的注意力）和 Linear Attention（线性注意力）。

| 大类 | 子类 | 核心特征 | 典型网络 | 对应算子 |
| :--- | :--- | :--- | :--- | :--- |
| Softmax Attention | GQA | 标准 MHA/GQA 稠密注意力，对全部 KV 执行 Softmax(QK^T/√d)V，无稀疏筛选；覆盖训练正向/反向与推理全量/增量场景 | Llama、Qwen、GPT | flash_attention_score、flash_attention_score_grad、flash_attn、fused_infer_attention_score、quant_flash_attn |
| Softmax Attention | MLA | Multi-head Latent Attention，K/V 共享同一份低维 latent 表示，通过下采样+RmsNorm+上采样+RoPE 前处理生成 query/query_rope 并更新 KV cache | DeepSeek-V2/V3.1 | fused_infer_attention_score、mla_prolog、mla_prolog_v2、mla_prolog_v3 |
| Softmax Attention | 稀疏 | 稀疏 Flash Attention / Sparse Flash MLA，先通过 Lightning Indexer 选取 Top-K 关键 KV 子集，再对选中 KV 执行注意力；含 KV 压缩（Compressor）、双路稀疏等变体 | DeepSeek-V3.2/V4（SMLA/QLI） | lightning_indexer、lightning_indexer_grad、quant_lightning_indexer、sparse_flash_attention、sparse_flash_attention_grad、kv_quant_sparse_flash_attention、compressor、compressor_grad、quant_compressor、lightning_indexer_v2、quant_lightning_indexer_v2、sparse_flash_mla、sparse_flash_mla_grad、sparse_flash_mla_softmax_l1_norm、quant_sparse_flash_mla、mixed_quant_sparse_flash_mla |
| Linear Attention | GDN | 基于 Gated DeltaNet 论文的 Gated Delta Rule，通过门控状态递归更新；分 Recurrent（Decode 逐步递归）和 Chunk（Prefill 切块并行）两种计算模式 | Qwen3.5 | recurrent_gated_delta_rule、chunk_gated_delta_rule |
| Linear Attention | KDA | Kernel Delta Attention，基于分块核方法将 delta rule 的递归状态更新 kernel 化实现，通过 chunk 切分完成 gate 累计、QK 系数计算、状态递归更新和注意力输出，同时输出最终状态及可选的反向中间量 | Kimi-K3 | chunk_kda_fwd |

---

## 2 对外接口

下表列出 attention layer 算子的全部算子，算子名与算子目录名保持一致。

| 算子名 | 算子概述 | sample |
| :--- | :--- | :--- |
| [flash_attention_score](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md) | 训练场景 FlashAttention 正向计算，执行 Dropout(Softmax(Mask(scale·Q·K^T+pse)))·V，支持 pse/mask/dropout/RoPE | [test_aclnn_flash_attention_score.cpp](./flash_attention_score/examples/test_aclnn_flash_attention_score.cpp) |
| [flash_attention_score_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_flash_attention_score_grad.md) | FlashAttentionScore 的反向梯度计算，求 Q/K/V 的梯度 | [test_aclnn_flash_attention_score_grad_v4_fp16.cpp](./flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v4_fp16.cpp) |
| [flash_attn](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md) + [flash_attn_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_flash_attn_metadata.md) | 统一训练/推理 FlashAttention 接口，支持 BSND/BNSD/TND 布局、分页注意力、变长、滑动窗口；metadata 为其预计算 tiling 切分方案 | [test_flash_attn.py](./flash_attn/tests/pytests/test_flash_attn.py) |
| [fused_infer_attention_score](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md) | 融合推理注意力，统一支持全量(Prompt)与增量(Incre)场景，支持量化/反量化/分页/共享前缀/RoPE | [test_aclnn_fused_infer_attention_score_v5.cpp](./fused_infer_attention_score/examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp) |
| [incre_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_incre_flash_attention.md) | 自回归推理增量注意力，query 的 S 轴固定为 1，KV 叠加历史 state；950 已不支持，推荐使用 flash_attn | [test_aclnn_incre_flash_attention.cpp](./incre_flash_attention/examples/test_aclnn_incre_flash_attention.cpp) |
| [prompt_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_prompt_flash_attention.md) | 全量推理 Prefill 阶段 FlashAttention，支持 sparse 优化、INT8 量化；950 已不支持，推荐使用 flash_attn | [test_aclnn_prompt_flash_attention.cpp](./prompt_flash_attention/examples/test_aclnn_prompt_flash_attention.cpp) |
| [quant_flash_attn](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_flash_attn.md) + [quant_flash_attn_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_flash_attn.md) | MXFP8 量化版 FlashAttention，Q/K/V 为 FP8_E4M3，softmax 在 FP32 计算，输出 BF16；metadata 为其预计算 tiling 切分方案；prompt_flash_attention / incre_flash_attention 量化场景推荐使用 quant_flash_attn | [test_quant_flash_attn_func_rdv.py](./quant_flash_attn/tests/pytest/qfa_mxfp8_test/test_quant_flash_attn_func_rdv.py) |
| [lightning_indexer](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer.md) | 基于 Q@K 相关性 + ReLU + 权重 W 选取每个 token 的 Top-K 稀疏位置索引，作为 SparseFlashAttention 前处理 | [test_aclnn_lightning_indexer.cpp](./lightning_indexer/examples/test_aclnn_lightning_indexer.cpp) |
| [lightning_indexer_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer_grad.md) | lightning_indexer 的反向算子，计算 query/key/weights 的梯度 | [test_aclnn_lightning_indexer_grad.cpp](./lightning_indexer_grad/examples/test_aclnn_lightning_indexer_grad.cpp) |
| [quant_lightning_indexer](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_lightning_indexer.md) | 量化版 Lightning Indexer，query/key 为 INT8/FP8/HIF8 量化输入，存8算8 | [test_aclnn_quant_lightning_indexer.cpp](./quant_lightning_indexer/examples/test_aclnn_quant_lightning_indexer.cpp) |
| [sparse_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_attention.md) | 面向大序列推理的稀疏注意力，通过 sparse_indices 选取关键 KV 执行 Softmax(QK^T/√d)V | [test_aclnn_sparse_flash_attention_v2.cpp](./sparse_flash_attention/examples/test_aclnn_sparse_flash_attention_v2.cpp) |
| [sparse_flash_attention_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_attention_grad.md) | sparse_flash_attention 的反向算子，根据 sparseIndices 重排 KV 后计算 dQ/dK/dV | [test_aclnn_sparse_flash_attention_grad.cpp](./sparse_flash_attention_grad/examples/test_aclnn_sparse_flash_attention_grad.cpp) |
| [kv_quant_sparse_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_kv_quant_sparse_flash_attention.md) | 支持 Per-Token-Head-Tile-128 量化 KV 输入的稀疏注意力，仅 IR 图模式算子，无 aclnn C 接口 | [test_npu_kv_quant_sparse_flash_attention.py](./kv_quant_sparse_flash_attention/examples/test_npu_kv_quant_sparse_flash_attention.py) |
| [mla_prolog](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog.md) | MLA 前处理：下采样 + RmsNorm + 上采样 + RoPE，生成 query/query_rope 并更新 kv_cache/kr_cache | [test_aclnn_mla_prolog_nq_bsh.cpp](./mla_prolog/examples/test_aclnn_mla_prolog_nq_bsh.cpp) |
| [mla_prolog_v2](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v2.md) | MLA 前处理 v2，新增全量化场景支持（token_x/weight 均可量化），输出 dequant_scale_q_nope | [test_aclnn_mla_prolog_v2_fq.cpp](./mla_prolog_v2/examples/test_aclnn_mla_prolog_v2_fq.cpp) |
| [mla_prolog_v3](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v3.md) | MLA 前处理 v3，新增尺度矫正因子(αq/αkv)、query_norm 输出、per-token/per-group 量化模式 | [test_aclnn_mla_prolog_v3.cpp](./mla_prolog_v3/examples/test_aclnn_mla_prolog_v3.cpp) |
| [compressor](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_compressor.md) | KV 压缩前处理，将每 4 或 128 个 token 的 KV cache 压缩成 1 个，输出 cmp_kv | [test_aclnn_compressor.cpp](./compressor/examples/test_aclnn_compressor.cpp) |
| [compressor_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_compressor_grad.md) | compressor 的反向算子，计算 X、W^KV/W^Gate 与 Ape 的梯度 | [test_aclnn_compressor_grad.cpp](./compressor_grad/examples/test_aclnn_compressor_grad.cpp) |
| [quant_compressor](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_compressor.md) | 量化版 Compressor，输入 x/W^KV/W^Gate 为 HIFLOAT8，硬件原生 Matmul 后反量化 | [test_aclnn_quant_compressor.cpp](./quant_compressor/examples/arch35/test_aclnn_quant_compressor.cpp) |
| [lightning_indexer_v2](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer_v2.md) + [lightning_indexer_v2_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer_v2_metadata.md) | V2 版稀疏索引选取，支持 maskMode/cmpRatio，基于 Q@K 相关性 + Top-K 选出稀疏位置；metadata 为其生成负载均衡方案 | [test_aclnn_lightning_indexer_v2.cpp](./lightning_indexer_v2/examples/test_aclnn_lightning_indexer_v2.cpp) |
| [quant_lightning_indexer_v2](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_lightning_indexer_v2.md) + [quant_lightning_indexer_v2_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_lightning_indexer_v2_metadata.md) | 量化版 V2 稀疏索引，query/key 为 INT8/FP8/HIF8 量化输入；metadata 为其生成负载均衡方案 | [test_aclnn_quant_lightning_indexer_v2.cpp](./quant_lightning_indexer_v2/examples/test_aclnn_quant_lightning_indexer_v2.cpp) |
| [sparse_flash_mla](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla.md) + [sparse_flash_mla_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_metadata.md) | MLA 稀疏注意力主算，支持双路（ori 原始 KV + cmp 压缩 KV）稀疏注意力，支持 SWA/CSA/HCA 掩码；metadata 为其生成负载均衡方案 | [test_aclnn_sparse_flash_mla.cpp](./sparse_flash_mla/examples/test_aclnn_sparse_flash_mla.cpp) |
| [sparse_flash_mla_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_grad.md) + [sparse_flash_mla_grad_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_grad_metadata.md) | sparse_flash_mla 训练场景反向算子；metadata 为其生成含 Batch/Head/Q/K 分块索引的任务列表 | [test_aclnn_sparse_flash_mla_grad.cpp](./sparse_flash_mla_grad/examples/test_aclnn_sparse_flash_mla_grad.cpp) |
| [sparse_flash_mla_softmax_l1_norm](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_softmax_l1_norm.md) + [sparse_flash_mla_softmax_l1_norm_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_softmax_l1_norm_metadata.md) | 计算 sparse_flash_mla 注意力的 Softmax L1Norm 结果，为 KL Loss Grad 反向配套；metadata 为其生成前置分核方案 | [test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp](./sparse_flash_mla_softmax_l1_norm/examples/test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp) |
| [quant_sparse_flash_mla](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_sparse_flash_mla.md) + [quant_sparse_flash_mla_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_sparse_flash_mla_metadata.md) | 全量化版 MLA 稀疏注意力，Q/KV 为 HIFLOAT8 per-tensor 量化；metadata 为其生成负载均衡方案 | [test_aclnn_quant_sparse_flash_mla.cpp](./quant_sparse_flash_mla/examples/test_aclnn_quant_sparse_flash_mla.cpp) |
| [mixed_quant_sparse_flash_mla](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mixed_quant_sparse_flash_mla.md) + [mixed_quant_sparse_flash_mla_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mixed_quant_sparse_flash_mla_metadata.md) | 混合量化版 MLA 稀疏注意力，支持 per-token-group 量化 KV；metadata 为其生成负载均衡方案 | [test_aclnn_mixed_quant_sparse_flash_mla.cpp](./mixed_quant_sparse_flash_mla/examples/test_aclnn_mixed_quant_sparse_flash_mla.cpp) |
| [recurrent_gated_delta_rule](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_recurrent_gated_delta_rule.md) | 逐时间步递归更新隐状态 S_t 并输出 o_t，适用于 Decode 阶段；支持变步长、状态索引 | [test_aclnn_recurrent_gated_delta_rule.cpp](./recurrent_gated_delta_rule/examples/test_aclnn_recurrent_gated_delta_rule.cpp) |
| [chunk_gated_delta_rule](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_chunk_gated_delta_rule.md) | 序列切块并行计算 Gated Delta Rule，长上下文效率高于 Recurrent 版，适用于 Prefill 阶段；输出每步 o_t 及最终状态 S_L | [test_aclnn_chunk_gated_delta_rule.cpp](./chunk_gated_delta_rule/examples/test_aclnn_chunk_gated_delta_rule.cpp) |
| [chunk_kda_fwd](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_chunk_kda_fwd.md) | KDA 分块前向计算，通过分块核方法将 delta rule 递归状态更新 kernel 化实现并行，适用于 Prefill 阶段 | [test_chunk_kda_fwd.py](./chunk_kda_fwd/tests/pytest/test_chunk_kda_fwd.py) |

---

## 3 Metadata 算子机制

### 3.1 概述

Metadata 算子是与主算子配套的 AICPU 前置算子（命名以 `_metadata` 结尾），运行在 device 侧的 AI CPU 上，负责根据输入的序列元信息（`cuSeqlens`、`seqused`、batch_size、max_seqlen 等）计算负载均衡的分核切分方案，输出固定 shape 的 metadata tensor 供主算子消费。

典型调用流程为两段式，Metadata 算子先行执行生成分核方案，主算子随后消费 metadata 执行注意力计算：

```
步骤1: 调用 Metadata 算子 (AICPU)          步骤2: 调用主算子 (AICore)

  输入: 序列元信息                            输入: Q, K, V + metadata
  (cu_seqlens, seqused, ...)                        |
         |                                          v
         v                                   +-------------+
   +-------------+      metadata             |   主算子     |
   |  Metadata   |     (分核方案)            |  Attention   |
   |   算子      |  ---------------------->  |   计算       |
   +-------------+                           +-------------+
                                                   |
                                                   v
                                              输出: attn_out
```

### 3.2 解决的核心问题：Tiling 下沉

#### 3.2.1 什么是 Tiling 下沉

传统模式下，主算子的分核切分在 host 端 `GetWorkspaceSize`（tiling）阶段完成。Tiling 下沉是一种图执行模式，将原本在 host 端 tiling 阶段做的分核计算，下沉到 device 侧（AICPU）执行。Metadata 算子就是 tiling 下沉的具体实现手段。

#### 3.2.2 为什么需要 Tiling 下沉

| 问题 | 说明 |
| :--- | :--- |
| **Host 端无法读取 device tensor 的值** | 变长序列（TND）场景下，`cuSeqlens`、`seqused` 等 tensor 在 device 侧，host 端 tiling 阶段拿不到实际数值，无法做精准负载均衡 |
| **减少 host 端开销** | 复杂的分核逻辑在 host 端执行会阻塞 launch 流水，下沉到 AICPU 后主算子 tiling 阶段可直接跳过切分计算，减少对 host array 的访问 |
| **支持 aclgraph 图模式** | aclgraph 要求静态编译时确定执行图，而 tiling 依赖运行时数据。通过 AICPU 算子提前算好 metadata，使主算子能在图模式下执行 |

#### 3.2.3 整体下沉

"整体下沉"指将 tiling 计算从 host 端整体下沉到 AICPU 算子中，主算子 tiling 阶段不再自行计算切分方案，而是直接消费 AICPU 算子的输出结果。主算子文档将 metadata 描述为"tiling 下沉的 AICPU 算子输出结果"。

### 3.3 Metadata 算子与主算子的值依赖及分工

Metadata 算子的输出 tensor 直接作为主算子的输入 tensor（`metadataOptional` 参数），是 device 侧的运行时数据依赖，而非 host 侧属性依赖。主算子 tiling 阶段读取 metadata tensor 填充分核结构，kernel 阶段按分核结果执行计算。两者分工如下：

| 维度 | Metadata 算子（AICPU） | 主算子（AICore） |
| :--- | :--- | :--- |
| 执行位置 | Device 侧 AI CPU | Device 侧 AI Core / Vector |
| 功能 | 负载均衡分核、任务划分 | Attention 数值计算 |
| 数据访问 | 可读 device tensor 实际值（cuSeqlens、seqused 等） | 按 metadata 指示执行 |
| 执行时机 | 主算子之前（前置算子） | 消费 metadata 后执行 |
| workspace | 通常为 0 | 需要较大 workspace |
| 流水关系 | 两者可流水并行，metadata 可复用 | 消费前置结果 |

---

## 4 公共概念

### 4.1 Layout（数据布局）

attention layer 算子中，输入输出张量的排布格式通过 `inputLayout` 参数指定。维度字母含义如下：

| 字母 | 含义 |
| :--- | :--- |
| B | Batch Size，输入样本批量大小 |
| S | Sequence Length，单条序列长度 |
| T | Total Tokens，所有 Batch 序列长度的累加和（T = ΣS_i） |
| N | Head Num，多头数 |
| D | Head Dim，单个 head 的隐藏维度 |
| H | Head Size，隐藏层大小（H = N × D） |

各布局格式定义如下：

| 格式名 | shape | 特点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| BSH | `[B, S, H]`，H = N × D | N 和 D 合并为一个维度 H，3 维格式，最简单 | FlashAttentionScore 训练；PromptFlashAttention / IncreFlashAttention / FusedInferAttentionScore 推理（推荐默认布局） |
| BSND | `[B, S, N, D]` | N 和 D 拆分为独立维度，4 维格式 | 训练和推理全场景通用 |
| BNSD | `[B, N, S, D]` | 同 BSND，但 N 和 S 轴互换 | 训练和推理全场景通用 |
| TND | `[T, N, D]` | 将多个 batch 的序列拼接为一维张量，消除 padding 浪费；需配合 `cuSeqlens` 标记 batch 边界 | 变长序列（VarLen）场景，训练和推理均支持 |
| NTD | `[N, T, D]` | 同 TND，但 N 和 T 轴互换 | 量化注意力全量化场景；常以 `NTD_TND`（输入 NTD、输出 TND）组合使用 |
| BnBsH | `[BlockNum, BlockSize, H]` | PA 场景 KV Cache 布局，H = N×D，3 维最简格式；H 超过 65535 时不可用 | PageAttention KV Cache 存储 |
| PA_BSND / BnNBsD / PA_BBND | `[BlockNum, BlockSize, N, D]` 或 `[BlockNum, N, BlockSize, D]` | PA 场景 KV Cache 布局，D 维连续存放，4 维格式；性能优于 BnBsH，推荐优先选择 | PageAttention KV Cache 存储 |
| PA_NZ / NZ | `[BlockNum, N, D/D0, BlockSize, D0]` | PA 场景 KV Cache 布局，D 维拆分为 D/D0 × D0 的 NZ 分形结构，5 维；最内层 D0=32（FP8/INT8）或 16（FP16/BF16/keyRope）；量化场景专用，INT8 时**仅支持** NZ | PageAttention 量化场景 KV Cache 存储 |

> **补充说明**：
> - **BNSD_BSND**：特殊组合布局，输入为 BNSD 时输出格式为 BSND。
> - PA_BSND 与 PA_NZ 的核心区别：PA_BSND 的 D 维连续存放（4 维），PA_NZ 将 D 维拆分为 NZ 分形（5 维），量化场景下对 Cube 指令更友好。

### 4.2 PageAttention（分页注意力）

#### 4.2.1 原理与设计动机

在大模型自回归推理中，每个 token 生成时都需读取历史 KV Cache。传统 KV Cache 要求每个 batch 在 S 维（序列长度）上预留一段连续物理内存，并按 `maxSeqLen` 预先 padding，导致显存碎片化和浪费。

PagedAttention 借鉴操作系统虚拟内存分页机制，将 KV Cache 切分为固定大小的 block，存储在一片连续物理内存中，通过 `blockTable`（页表）建立"逻辑序列顺序 → 物理块地址"的映射。其核心价值：

- **消除显存碎片**：按需分配 block，无需预分配整段连续内存
- **提升显存利用率**：不同 batch 共享同一片物理 block 池，支持更大 batch 或更长序列
- **逻辑连续、物理离散**：逻辑上连续的 KV 序列可散布在不连续的物理 block 中

#### 4.2.2 工作机制

PagedAttention 的开启必要条件是 `blockTable` 存在且有效，同时 key/value 按 `blockTable` 中的索引在连续内存中排布。开启后 key/value 的 `inputLayout` 参数失效，KV Cache 的排布由其自身 shape 维度决定。

**blockTable 查表过程**（以 `B=2`、`blockSize=128`、`actualSeqLengthsKv=[256, 512]` 为例）：

`maxBlockNumPerSeq = ceil(512/128) = 4`，blockTable shape = `[2, 4]`：

```
blockTable = [
  [ 3,  7,  1,  0],   # batch 0: 逻辑块0→物理block 3, 逻辑块1→物理block 7 (实际只用2个block)
  [ 5,  2,  9,  4],   # batch 1: 逻辑块0→物理block 5, 逻辑块1→物理block 2, 逻辑块2→物理block 9, 逻辑块3→物理block 4
]
```

访问 batch 0 的 token 0~255 时：
- token 0~127 → 逻辑块 0 → 查表得 block_id=3 → 读物理 `kvCache[3]`
- token 128~255 → 逻辑块 1 → 查表得 block_id=7 → 读物理 `kvCache[7]`

block id 可乱序分配、可跨 batch 共享物理池，用户需自行保证 block id 合法性（host 侧不校验）。

#### 4.2.3 blockTable

| 项目 | 内容 |
| :--- | :--- |
| 数据类型 | INT32 |
| shape | `[B, maxBlockNumPerSeq]` |
| 第一维 | 等于 Batch 数 B |
| 第二维 | 不小于最大 KV 序列长度对应的 block 数，即 `ceil(max_KV_S / blockSize)` |
| 元素含义 | block id，指向 KV Cache 物理池中第几个 block |
| 约束 | block id 合法性由用户自行保证，host 侧不校验 |

#### 4.2.4 blockSize

| 项目 | 内容 |
| :--- | :--- |
| 含义 | 每个 block 中的最大 token 数，用户自定义参数 |
| 通用对齐 | FP16/BF16 需 16 对齐；INT8/HIFLOAT8/FP8_E4M3 需 32 对齐；INT4/FLOAT4_E2M1 需 64 对齐 |
| 最大值 | 1024（部分早期版本为 512） |
| 推荐值 | 128（兼容性最好，多数全量化场景的强制值） |
| 性能影响 | 调大 blockSize 有一定性能收益（减少查表次数、提升单次搬运数据量），但会增加尾部 block 显存浪费 |

#### 4.2.5 PA 模式与普通模式对比

| 维度 | 普通模式 | PA 模式 |
| :--- | :--- | :--- |
| KV 物理排布 | S 轴连续 | 按 block 散布，靠 blockTable 映射 |
| inputLayout（KV） | 生效 | 失效 |
| blockTable | 不传 | 必须传 |
| actualSeqLengthsKv | 可选 | 必须传 |
| tensorlist | 支持 | 不支持 |
| 左 padding | 支持 | 不支持 |
| 公共前缀 | 支持 | 不支持 |
| D 不等长 | 支持 | 不支持 |
| KV Cache 布局 | BSH/BSND/BNSD/TND 等 | BnBsH / BnNBsD / NZ |

#### 4.2.6 性能特点

- **吞吐量提升 vs 单次延迟下降**：PA 通过提升显存利用率支持更大 batch，从而提高吞吐量；但 blockTable 间接寻址带来额外查表开销和非连续访问，单次推理延迟略有下降
- **KV Cache 排布影响性能**：BnNBsD 性能优于 BnBsH（N 轴前置便于按 head 拉取连续 D 数据），推荐优先选择 BnNBsD 格式
- **BnBsH 限制**：当 KV_N × D 超过 65535 时，受硬件指令约束报错，需改用 BnNBsD 格式

### 4.3 Mask Mode（掩码模式）

Mask Mode（`sparseMode` / `mask_mode`）控制注意力计算中的掩码策略。不同算子中参数名不同但语义一致。

#### 4.3.1 取值定义

| 取值 | 名称 | 含义 | attenMask 要求 | preTokens/nextTokens |
| :---: | :--- | :--- | :--- | :--- |
| 0 | defaultMask | 默认模式；不传 mask 则全计算；传 mask 则按 preTokens/nextTokens 做 causal 或 band | 可选 | 生效 |
| 3 | rightDownCausal | 右下顶点划分的下三角场景 | 压缩下三角矩阵（2048×2048） | 不生效 |
| 4 | band | 计算 preTokens 和 nextTokens 之间的部分，起点为右下角 | 压缩下三角矩阵（2048×2048） | 生效 |

#### 4.3.2 rightDownCausal（mode=3）

右下顶点因果掩码，以注意力矩阵右下角为顶点，保留下三角部分（含对角线），屏蔽上三角。即 query token i 只能 attend 到 key token 0~i，不能看到未来 token。适用于标准因果语言模型（GPT/Llama 等）的自回归训练和推理。

```
       key:  0  1  2  3  4
          ┌───────────────┐
    q 0   │  1  0  0  0  0 │
    q 1   │  1  1  0  0  0 │
    q 2   │  1  1  1  0  0 │
    q 3   │  1  1  1  1  0 │
    q 4   │  1  1  1  1  1 │
          └───────────────┘
       1 = 计算    0 = 屏蔽
```

传入的 attenMask 为压缩下三角矩阵（2048×2048），preTokens/nextTokens 不生效。

#### 4.3.3 band（mode=4）

带状掩码，以右下角为起点，通过 `preTokens` 和 `nextTokens` 定义一个带状区域。query token i 只能 attend 到 key token `[i - preTokens + 1, i + nextTokens]` 范围内的 token。适用于滑动窗口注意力（Sliding Window Attention）场景。

以 `preTokens=3, nextTokens=0` 为例：

```
       key:  0  1  2  3  4
          ┌───────────────┐
    q 0   │  1  0  0  0  0 │
    q 1   │  1  1  0  0  0 │
    q 2   │  1  1  1  0  0 │
    q 3   │  0  1  1  1  0 │
    q 4   │  0  0  1  1  1 │
          └───────────────┘
       1 = 计算    0 = 屏蔽
       窗口宽度 = preTokens = 3
```

传入的 attenMask 为压缩下三角矩阵（2048×2048），`preTokens`/`nextTokens` 生效。

### 4.4 cuSeqlens & seqused

这两个参数用于描述变长序列中每个 batch 的实际有效长度。不同 layout 场景下使用方式不同：TND 布局将多个 batch 的序列拼接为一维张量，需通过 `cuSeqlens` 标记 batch 边界；非 TND 布局（BSND/BNSD/BSH 等）每个 batch 有独立的 S 维度，需通过 `seqused` 标记实际有效长度。

| 项目 | cuSeqlens | seqused |
| :--- | :--- | :--- |
| 含义 | 累积前缀和（cumulative sum），标记各 batch 的边界；每个元素为当前 batch 及前序 batch 有效 token 数的累加值 | 每个 batch 实际参与计算的 token 数 |
| shape | `(B+1,)` | `(B,)` |
| 数据类型 | INT32 | INT32 |
| 首元素 | 固定为 0 | 无要求 |
| 数值约束 | 单调非递减；末元素等于 T（总 token 数） | 不超过 shape 中的 S |
| 数值关系 | `cuSeqlens[i+1] - cuSeqlens[i] = seqused[i]` | 可由 cuSeqlens 差分得到 |
| TND 布局 | 必传（数据按 batch 紧凑排列，需标记边界） | 可选（总长度由 cuSeqlens 表达，seqused 可显式覆盖有效长度） |
| 非 TND 布局 | 禁止传入 | 可选（用于标记每个 batch 的有效长度） |

#### 4.4.1 TND 布局下的等效 padding 效果

TND 布局将多个 batch 的序列紧凑拼接为一维张量，无需 padding。`cuSeqlens` 标记各 batch 边界，`seqused` 标记各 batch 有效长度，两者配合实现等效 padding 效果。

以 `B=3`、各 batch 有效长度分别为 `2, 1, 3` 为例：

```
TND (紧凑拼接，零浪费):              BSND (需 padding 对齐到 maxLen=3):

┌────┬────┬────┬────┬────┬────┐    ┌────┬────┬─────┐
│ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │    │ t0 │ t1 │ pad │  batch0
└────┴────┴────┴────┴────┴────┘    ├────┼────┼─────┤
  b0: 2   b1:1   b2:  3           │ t2 │pad │ pad │  batch1
                                    ├────┼────┼─────┤
cuSeqlens = [0, 2, 3, 6]            │ t3 │ t4 │ t5  │  batch2
seqused   = [2, 1, 3]               └────┴────┴─────┘
                                     占 9 位，浪费 3 位
占 6 位，零浪费
```

### 4.5 稀疏索引参数（topk / sparse_indices）

稀疏注意力算子通过 TopK 选取关键 KV 子集，仅对选中 KV 执行注意力计算。`topk` 控制选取数量，`sparse_indices` 记录选取的位置索引，两者构成 Lightning Indexer → Sparse Flash Attention 的生产-消费链路。

#### 4.5.1 topk（TopK 选取数量）

稀疏注意力的核心思想是不对全部 KV 做注意力计算，而是先通过相关性打分为每个 query token 筛选出最相关的若干个 KV 位置，只对这部分 KV 执行 `Softmax(QK^T)V`，从而大幅减少计算量。这一筛选过程在 Lightning Indexer 算子中完成。

`topk` 即控制"选多少个"的数量参数：topk 越大，参与计算的 KV 越多，注意力结果越逼近稠密注意力的精度，但计算量和访存开销也随之增加；topk 越小，计算越快但可能遗漏关键信息，导致精度下降。用户需要根据模型精度要求和序列长度在性能与精度之间权衡。`topk` 直接决定了下游 `sparse_indices` 输出的最后一维大小。

以序列长度 `KV_S=8192` 为例：

```
topk=2048: 选取 25% 的 KV，计算量减少 75%
topk=512:  选取 6% 的 KV，计算量减少 94%
```

#### 4.5.2 sparse_indices（稀疏位置索引）

`sparse_indices` 是 Lightning Indexer 的输出、Sparse Flash Attention / Sparse Flash MLA 的输入，记录每个 query token 应关注的 KV 位置索引。它将 topk 筛选的结果以 INT32 索引数组的形式传递给注意力主算子，主算子据此离散 gather 对应位置的 KV 后执行注意力计算，而非对全部 KV 做稠密计算。

索引数组中，有效位置存放 KV 的实际 block 索引，无效位置填充 -1。消费侧约定有效值排列在前半部分、无效值在后半部分，以便主算子只需处理前 `topk` 个有效条目，跳过尾部无效部分。

**生产-消费链路**：

```
lightning_indexer(query, key, weights, topk) → sparse_indices (输出)
                                                    ↓
sparse_flash_attention(query, key, value, sparse_indices) → attn_out (输入)
```

Lightning Indexer 计算 `ReLU(Q@K^T)` 得到相关性分数，乘以权重后沿 group 聚合，TopK 选取前 `topk` 个索引输出。Sparse Flash Attention 据此离散 gather KV 后执行 `Softmax(QK^T)@V`。整个链路将"选哪些 KV"和"对选中的 KV 做注意力"解耦为两个独立算子，便于分别优化和灵活组合。

### 4.6 量化场景概念

attention layer 算子按 Q/K/V 是否量化及量化粒度，分为非量化、伪量化和全量化三类场景。

#### 4.6.1 量化类型划分

| 量化类型 | Q | K/V | 反量化参数 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| 非量化 | FP16/BF16 | FP16/BF16 | 无 | Q/K/V 均为高精度，直接参与计算 |
| 伪量化 | FP16/BF16 | INT8/FP8/HIF8 等 | `antiquantScale` / `antiquantOffset` | 仅 K/V 量化存储，算子内反量化后参与计算；Q 与输出仍为高精度 |
| 全量化 | INT8/FP8/HIF8 等 | INT8/FP8/HIF8 等 | `dequantScaleQuery` + `antiquantScale` | Q/K/V 均量化输入，算子内全部反量化后计算 |

#### 4.6.2 反量化参数体系

| 参数 | 作用 | 适用场景 |
| :--- | :--- | :--- |
| `antiquantScale` / `antiquantOffset` | 对 K/V 伪量化的反量化因子和偏移 | 伪量化（对称/非对称） |
| `keyAntiquantScale` / `valueAntiquantScale` | K/V 分离反量化因子（KV 可使用不同 scale） | 伪量化/全量化（KV 分离模式） |
| `dequantScaleQuery` | 对 Q 反量化的因子 | **全量化专用** |
| `deqScale1` / `quantScale1` | 对 QK 结果反量化 / 对 P 量化 | INT8 在线量化（Q 高精度场景） |
| `deqScale2` / `quantScale2` / `quantOffset2` | 对 PV 结果反量化 / 对输出量化及偏移 | INT8 在线量化（输出 INT8 场景） |

#### 4.6.3 GQA 全量化（FP8）

GQA 全量化场景下 Q/K/V 均为 FLOAT8_E4M3FN 格式，通过 FLOAT32 反量化因子在算子内恢复高精度计算。

| 项目 | 内容 |
| :--- | :--- |
| Q/K/V 数据类型 | FLOAT8_E4M3FN |
| 反量化因子 dtype | FLOAT32 |
| 量化粒度 | `queryQuantMode=3`（per-token 叠加 per-head），`keyAntiquantMode=3`（per-token 叠加 per-head），`valueAntiquantMode=2`（per-tensor 叠加 per-head） |
| HeadDim 约束 | 仅支持 128 |
| blockSize 约束 | 仅支持 128 |
| 典型算子 | fused_infer_attention_score |

#### 4.6.4 MXFP8 全量化

MXFP8（Microscaling FP8）是 FP8_E4M3FN 数据配合 FLOAT8_E8M0 缩放因子的量化格式，每 32 个元素共享一个 E8M0 scale。

| 项目 | 内容 |
| :--- | :--- |
| Q/K/V 数据类型 | FLOAT8_E4M3FN |
| 缩放因子 dtype | FLOAT8_E8M0 |
| 量化粒度 | `queryQuantMode/keyAntiquantMode=6`（per-token-group），`valueAntiquantMode=8`（per-channel-group） |
| scale shape | `(1, B, N, >=KV_S, D/32)`，每 32 个 D 元素共享一个 E8M0 scale |
| HeadDim 约束 | 仅支持 64 或 128 |
| blockSize 约束 | 支持 64/128/256/512/1024 |
| 典型算子 | quant_flash_attn、fused_infer_attention_score |

### 4.7 Batch 一致性

Batch 一致性（确定性计算）是指算子在不同 batch 大小、不同 batch 组合下执行时，计算结果保持一致的能力。默认情况下，部分算子（如 compressor / quant_compressor）的计算结果可能与输入 batch 在整体数据中的位置和 batch 划分方式有关，即同一组数据在不同 batch 划分下可能产生微小数值差异。

通过 `aclrtSetSysParamOpt()` 配置 `ACL_OPT_DETERMINISTIC=3` 可开启 batch 一致性模式，开启后算子内部采用确定性计算策略，保证计算结果与所在批次大小、位置无关。

| 项目 | 内容 |
| :--- | :--- |
| 含义 | 计算结果与 batch 大小、batch 在数据中的位置无关 |
| 开启方式 | `aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3)` |
| 适用算子 | compressor、quant_compressor 等 |
| 默认状态 | 关闭（非确定性） |
| 代价 | 开启后可能带来一定性能下降 |

---
