English | [简体中文](./README.md)

## 1 attention layer Operator Classification Principles

attention layer operators are divided into two major categories by whether Softmax is used: Softmax Attention (Softmax-based attention) and Linear Attention (linear attention).

| Major category | Subcategory | Core features | Typical networks | Corresponding operators |
| :--- | :--- | :--- | :--- | :--- |
| Softmax Attention | GQA | Standard MHA/GQA dense attention. Executes Softmax(QK^T/√d)V on all KV with no sparse filtering. Covers training forward/backward and inference full/incremental scenarios | Llama, Qwen, GPT | flash_attention_score, flash_attention_score_grad, flash_attn, fused_infer_attention_score, quant_flash_attn |
| Softmax Attention | MLA | Multi-head Latent Attention. K/V share the same low-dimensional latent representation. Generates query/query_rope through downsampling+RmsNorm+upsampling+RoPE preprocessing and updates the KV cache | DeepSeek-V2/V3.1 | fused_infer_attention_score, mla_prolog, mla_prolog_v2, mla_prolog_v3 |
| Softmax Attention | Sparse | Sparse Flash Attention / Sparse Flash MLA. First uses the Lightning Indexer to select the Top-K key KV subset, and then performs attention on the selected KV. Includes variants such as KV compression (Compressor) and dual-path sparsity | DeepSeek-V3.2/V4 (SMLA/QLI) | lightning_indexer, lightning_indexer_grad, quant_lightning_indexer, sparse_flash_attention, sparse_flash_attention_grad, kv_quant_sparse_flash_attention, compressor, compressor_grad, quant_compressor, lightning_indexer_v2, quant_lightning_indexer_v2, sparse_flash_mla, sparse_flash_mla_grad, sparse_flash_mla_softmax_l1_norm, quant_sparse_flash_mla, mixed_quant_sparse_flash_mla |
| Linear Attention | GDN | Gated Delta Rule based on the Gated DeltaNet paper. Recursively updates states through gating. Provides two computation modes: Recurrent (Decode step-by-step recursion) and Chunk (Prefill chunk-parallel) | Qwen3.5 | recurrent_gated_delta_rule, chunk_gated_delta_rule |
| Linear Attention | KDA | Kernel Delta Attention. Implements kernelized recursion state updates of the delta rule based on the blockwise kernel method. Completes gate accumulation, QK coefficient computation, state recursion updates, and attention output through chunk splitting, and also outputs the final state and optional backward intermediates | Kimi-K3 | chunk_kda_fwd |

---

## 2 External Interfaces

The following table lists all operators of the attention layer. Operator names are consistent with operator directory names.

| Operator name | Operator description | Sample |
| :--- | :--- | :--- |
| [flash_attention_score](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md) | FlashAttention forward computation in the training scenario. Executes Dropout(Softmax(Mask(scale·Q·K^T+pse)))·V. Supports pse/mask/dropout/RoPE | [test_aclnn_flash_attention_score.cpp](./flash_attention_score/examples/test_aclnn_flash_attention_score.cpp) |
| [flash_attention_score_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_flash_attention_score_grad.md) | Backward gradient computation of FlashAttentionScore. Computes the gradients of Q/K/V | [test_aclnn_flash_attention_score_grad_v4_fp16.cpp](./flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v4_fp16.cpp) |
| [flash_attn](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md) + [flash_attn_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_flash_attn_metadata.md) | Unified training/inference FlashAttention interface. Supports BSND/BNSD/TND layouts, paged attention, variable length, and the sliding window. The metadata operator pre-computes the tiling split scheme for it | [test_flash_attn.py](./flash_attn/tests/pytests/test_flash_attn.py) |
| [fused_infer_attention_score](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md) | Fused inference attention. Uniformly supports the full (Prompt) and incremental (Incre) scenarios. Supports quantization/dequantization/paging/shared prefix/RoPE | [test_aclnn_fused_infer_attention_score_v5.cpp](./fused_infer_attention_score/examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp) |
| [incre_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_incre_flash_attention.md) | Incremental attention for autoregressive inference. The S axis of query is fixed to 1, and KV is appended to the historical state. No longer supported on 950; flash_attn is recommended | [test_aclnn_incre_flash_attention_v4.cpp](./incre_flash_attention/examples/test_aclnn_incre_flash_attention_v4.cpp) |
| [prompt_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_prompt_flash_attention.md) | FlashAttention for the full inference Prefill stage. Supports sparse optimization and INT8 quantization. No longer supported on 950; flash_attn is recommended | [test_aclnn_prompt_flash_attention_v3.cpp](./prompt_flash_attention/examples/test_aclnn_prompt_flash_attention_v3.cpp) |
| [quant_flash_attn](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_flash_attn.md) + [quant_flash_attn_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_flash_attn.md) | MXFP8 quantized FlashAttention. Q/K/V are FP8_E4M3, softmax is computed in FP32, and the output is BF16. The metadata operator pre-computes the tiling split scheme for it. quant_flash_attn is recommended for the quantization scenarios of prompt_flash_attention / incre_flash_attention | [test_quant_flash_attn_func_rdv.py](./quant_flash_attn/tests/pytest/qfa_mxfp8_test/test_quant_flash_attn_func_rdv.py) |
| [lightning_indexer](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer.md) | Selects the Top-K sparse position index of each token based on Q@K correlation + ReLU + weight W, serving as the preprocessing of SparseFlashAttention | [test_aclnn_lightning_indexer.cpp](./lightning_indexer/examples/test_aclnn_lightning_indexer.cpp) |
| [lightning_indexer_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer_grad.md) | Backward operator of lightning_indexer. Computes the gradients of query/key/weights | [test_aclnn_lightning_indexer_grad.cpp](./lightning_indexer_grad/examples/test_aclnn_lightning_indexer_grad.cpp) |
| [quant_lightning_indexer](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_lightning_indexer.md) | Quantized Lightning Indexer. query/key are INT8/FP8/HIF8 quantized inputs, with 8-bit storage and 8-bit computation | [test_aclnn_quant_lightning_indexer.cpp](./quant_lightning_indexer/examples/test_aclnn_quant_lightning_indexer.cpp) |
| [sparse_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_attention.md) | Sparse attention for large-sequence inference. Selects key KV through sparse_indices and performs Softmax(QK^T/√d)V | [test_aclnn_sparse_flash_attention_v2.cpp](./sparse_flash_attention/examples/test_aclnn_sparse_flash_attention_v2.cpp) |
| [sparse_flash_attention_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_attention_grad.md) | Backward operator of sparse_flash_attention. Rearranges KV according to sparseIndices and then computes dQ/dK/dV | [test_aclnn_sparse_flash_attention_grad.cpp](./sparse_flash_attention_grad/examples/test_aclnn_sparse_flash_attention_grad.cpp) |
| [kv_quant_sparse_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_kv_quant_sparse_flash_attention.md) | Sparse attention supporting Per-Token-Head-Tile-128 quantized KV input. Only the IR graph mode operator, with no aclnn C interface | [test_npu_kv_quant_sparse_flash_attention.py](./kv_quant_sparse_flash_attention/examples/test_npu_kv_quant_sparse_flash_attention.py) |
| [mla_prolog](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog.md) | MLA preprocessing: downsampling + RmsNorm + upsampling + RoPE. Generates query/query_rope and updates kv_cache/kr_cache | [test_aclnn_mla_prolog_nq_bsh.cpp](./mla_prolog/examples/test_aclnn_mla_prolog_nq_bsh.cpp) |
| [mla_prolog_v2](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v2.md) | MLA preprocessing v2. Adds full quantization scenario support (both token_x/weight can be quantized) and outputs dequant_scale_q_nope | [test_aclnn_mla_prolog_v2_weight_nz_fq.cpp](./mla_prolog_v2/examples/test_aclnn_mla_prolog_v2_weight_nz_fq.cpp) |
| [mla_prolog_v3](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v3.md) | MLA preprocessing v3. Adds scale correction factors (αq/αkv), query_norm output, and per-token/per-group quantization modes | [test_aclnn_mla_prolog_v3_weight_nz.cpp](./mla_prolog_v3/examples/test_aclnn_mla_prolog_v3_weight_nz.cpp) |
| [compressor](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_compressor.md) | KV compression preprocessing. Compresses the KV cache of every 4 or 128 tokens into 1 token and outputs cmp_kv | [test_aclnn_compressor.cpp](./compressor/examples/test_aclnn_compressor.cpp) |
| [compressor_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_compressor_grad.md) | Backward operator of compressor. Computes the gradients of X, W^KV/W^Gate, and Ape | [test_aclnn_compressor_grad.cpp](./compressor_grad/examples/test_aclnn_compressor_grad.cpp) |
| [quant_compressor](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_compressor.md) | Quantized Compressor. Inputs x/W^KV/W^Gate are HIFLOAT8, with hardware-native Matmul followed by dequantization | [test_aclnn_quant_compressor.cpp](./quant_compressor/examples/arch35/test_aclnn_quant_compressor.cpp) |
| [lightning_indexer_v2](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer_v2.md) + [lightning_indexer_v2_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_lightning_indexer_v2_metadata.md) | V2 sparse index selection. Supports maskMode/cmpRatio and selects sparse positions based on Q@K correlation + Top-K. The metadata operator generates the load balancing scheme for it | [test_aclnn_lightning_indexer_v2.cpp](./lightning_indexer_v2/examples/test_aclnn_lightning_indexer_v2.cpp) |
| [quant_lightning_indexer_v2](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_lightning_indexer_v2.md) + [quant_lightning_indexer_v2_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_lightning_indexer_v2_metadata.md) | Quantized V2 sparse index. query/key are INT8/FP8/HIF8 quantized inputs. The metadata operator generates the load balancing scheme for it | [test_aclnn_quant_lightning_indexer_v2.cpp](./quant_lightning_indexer_v2/examples/test_aclnn_quant_lightning_indexer_v2.cpp) |
| [sparse_flash_mla](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla.md) + [sparse_flash_mla_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_metadata.md) | MLA sparse attention main operator. Supports dual-path (ori original KV + cmp compressed KV) sparse attention and SWA/CSA/HCA masks. The metadata operator generates the load balancing scheme for it | [test_aclnn_sparse_flash_mla.cpp](./sparse_flash_mla/examples/test_aclnn_sparse_flash_mla.cpp) |
| [sparse_flash_mla_grad](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_grad.md) + [sparse_flash_mla_grad_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_grad_metadata.md) | Backward operator of sparse_flash_mla in the training scenario. The metadata operator generates a task list containing Batch/Head/Q/K block indices for it | [test_aclnn_sparse_flash_mla_grad.cpp](./sparse_flash_mla_grad/examples/test_aclnn_sparse_flash_mla_grad.cpp) |
| [sparse_flash_mla_softmax_l1_norm](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_softmax_l1_norm.md) + [sparse_flash_mla_softmax_l1_norm_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_mla_softmax_l1_norm_metadata.md) | Computes the Softmax L1Norm result of the sparse_flash_mla attention, paired with the KL Loss Grad backward. The metadata operator generates the preceding core-splitting scheme for it | [test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp](./sparse_flash_mla_softmax_l1_norm/examples/test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp) |
| [quant_sparse_flash_mla](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_sparse_flash_mla.md) + [quant_sparse_flash_mla_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_sparse_flash_mla_metadata.md) | Fully quantized MLA sparse attention. Q/KV are HIFLOAT8 per-tensor quantized. The metadata operator generates the load balancing scheme for it | [test_aclnn_quant_sparse_flash_mla.cpp](./quant_sparse_flash_mla/examples/test_aclnn_quant_sparse_flash_mla.cpp) |
| [mixed_quant_sparse_flash_mla](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mixed_quant_sparse_flash_mla.md) + [mixed_quant_sparse_flash_mla_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mixed_quant_sparse_flash_mla_metadata.md) | Mixed quantized MLA sparse attention. Supports per-token-group quantized KV. The metadata operator generates the load balancing scheme for it | [test_aclnn_mixed_quant_sparse_flash_mla.cpp](./mixed_quant_sparse_flash_mla/examples/test_aclnn_mixed_quant_sparse_flash_mla.cpp) |
| [mixed_quant_flash_attn](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mixed_quant_flash_attn.md) + [mixed_quant_flash_attn_metadata](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mixed_quant_flash_attn_metadata.md) | Pseudo-quantized FlashAttention. Q remains BF16/FP16 without quantization, and KV is stored in MXFP4/HiFP4 quantization and dequantized for recovery. The metadata operator pre-computes the tiling split scheme for it | [test_aclnn_mixed_quant_flash_attn_metadata.cpp](./mixed_quant_flash_attn_metadata/examples/test_aclnn_mixed_quant_flash_attn_metadata.cpp) |
| [recurrent_gated_delta_rule](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_recurrent_gated_delta_rule.md) | Recursively updates the hidden state S_t at each time step and outputs o_t. Suitable for the Decode stage. Supports variable step length and state indexing | [test_aclnn_recurrent_gated_delta_rule.cpp](./recurrent_gated_delta_rule/examples/test_aclnn_recurrent_gated_delta_rule.cpp) |
| [chunk_gated_delta_rule](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_chunk_gated_delta_rule.md) | Computes the Gated Delta Rule in parallel by sequence chunks. More efficient for long contexts than the Recurrent version. Suitable for the Prefill stage. Outputs o_t at each step and the final state S_L | [test_aclnn_chunk_gated_delta_rule.cpp](./chunk_gated_delta_rule/examples/test_aclnn_chunk_gated_delta_rule.cpp) |
| [chunk_kda_fwd](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_chunk_kda_fwd.md) | KDA chunked forward computation. Implements parallelism by kernelizing the delta rule recursion state updates through the blockwise kernel method. Suitable for the Prefill stage | [test_chunk_kda_fwd.py](./chunk_kda_fwd/tests/pytest/test_chunk_kda_fwd.py) |

---

## 3 Metadata Operator Mechanism

### 3.1 Overview

A Metadata operator is an AICPU preprocessing operator paired with a main operator (named with the `_metadata` suffix). It runs on the AI CPU on the device side, computes a load-balanced core-splitting scheme based on the input sequence metadata (such as `cuSeqlens`, `seqused`, batch_size, and max_seqlen), and outputs a fixed-shape metadata tensor for the main operator to consume.

The typical invocation flow consists of two stages. The Metadata operator runs first to generate the core-splitting scheme, and the main operator then consumes the metadata to perform attention computation:

```
Step 1: Call the Metadata operator (AICPU)          Step 2: Call the main operator (AICore)

  Input: sequence metadata                            Input: Q, K, V + metadata
  (cu_seqlens, seqused, ...)                        |
         |                                          v
         v                                   +-------------+
   +-------------+      metadata             |    Main     |
   |  Metadata   |    (core-splitting        |  Attention  |
   |   operator  |      scheme) ---------->  |  computing  |
   +-------------+                           +-------------+
                                                    |
                                                    v
                                               Output: attn_out
```

### 3.2 Core Problem Solved: Tiling Sinking

#### 3.2.1 What Is Tiling Sinking

In the traditional mode, core splitting of the main operator is completed in the host-side `GetWorkspaceSize` (tiling) stage. Tiling sinking is a graph execution mode that sinks the core-splitting computation originally performed in the host-side tiling stage to the device side (AICPU). The Metadata operator is the specific implementation of tiling sinking.

#### 3.2.2 Why Tiling Sinking Is Needed

| Problem | Description |
| :--- | :--- |
| **The host side cannot read the values of device tensors** | In the variable-length sequence (TND) scenario, tensors such as `cuSeqlens` and `seqused` reside on the device side. The host-side tiling stage cannot obtain their actual values and therefore cannot perform precise load balancing |
| **Reduce host-side overhead** | Complex core-splitting logic executed on the host side blocks the launch pipeline. After the logic is sunk to the AICPU, the main operator can directly skip the splitting computation in the tiling stage, reducing access to host arrays |
| **Support the aclgraph graph mode** | aclgraph requires the execution graph to be determined at static compilation time, while tiling depends on runtime data. Computing the metadata in advance through AICPU operators enables the main operator to execute in graph mode |

#### 3.2.3 Overall Sinking

"Overall sinking" means sinking the tiling computation from the host side as a whole into AICPU operators. The main operator no longer computes the splitting scheme itself in the tiling stage but directly consumes the output result of the AICPU operator. The main operator documentation describes the metadata as "the output result of the AICPU operator for tiling sinking".

### 3.3 Value Dependency and Division of Work Between the Metadata Operator and the Main Operator

The output tensor of the Metadata operator directly serves as the input tensor of the main operator (the `metadataOptional` parameter). This is a device-side runtime data dependency rather than a host-side attribute dependency. In the tiling stage, the main operator reads the metadata tensor to fill the core-splitting structure, and in the kernel stage, it performs computation according to the core-splitting result. The division of work is as follows:

| Dimension | Metadata operator (AICPU) | Main operator (AICore) |
| :--- | :--- | :--- |
| Execution location | Device-side AI CPU | Device-side AI Core / Vector |
| Function | Load-balanced core splitting and task division | Attention numerical computation |
| Data access | Can read the actual values of device tensors (cuSeqlens, seqused, and so on) | Executes as indicated by the metadata |
| Execution timing | Before the main operator (preprocessing operator) | Executes after consuming the metadata |
| workspace | Usually 0 | Requires a large workspace |
| Pipeline relationship | The two can run in parallel in a pipeline, and metadata can be reused | Consumes the preceding result |

---

## 4 Common Concepts

### 4.1 Layout (Data Layout)

In attention layer operators, the layout format of input and output tensors is specified by the `inputLayout` parameter. The meanings of the dimension letters are as follows:

| Letter | Meaning |
| :--- | :--- |
| B | Batch Size, the batch size of input samples |
| S | Sequence Length, the length of a single sequence |
| T | Total Tokens, the cumulative sum of sequence lengths of all batches (T = ΣS_i) |
| N | Head Num, the number of heads |
| D | Head Dim, the hidden dimension of a single head |
| H | Head Size, the size of the hidden layer (H = N × D) |

The layout formats are defined as follows:

| Format name | shape | Characteristics | Applicable scenarios |
| :--- | :--- | :--- | :--- |
| BSH | `[B, S, H]`, H = N × D | N and D are merged into one dimension H. 3D format, the simplest | FlashAttentionScore training; PromptFlashAttention / IncreFlashAttention / FusedInferAttentionScore inference (recommended default layout) |
| BSND | `[B, S, N, D]` | N and D are split into independent dimensions. 4D format | Common to all training and inference scenarios |
| BNSD | `[B, N, S, D]` | Same as BSND, but with the N and S axes swapped | Common to all training and inference scenarios |
| TND | `[T, N, D]` | Concatenates the sequences of multiple batches into a 1D tensor, eliminating padding waste; must be used with `cuSeqlens` to mark batch boundaries | Variable-length sequence (VarLen) scenarios; supported for both training and inference |
| NTD | `[N, T, D]` | Same as TND, but with the N and T axes swapped | Full quantization scenarios of quantized attention; often used in the `NTD_TND` combination (input NTD, output TND) |
| BnBsH | `[BlockNum, BlockSize, H]` | PA scenario KV Cache layout, H = N×D, the simplest 3D format; unavailable when H exceeds 65535 | PageAttention KV Cache storage |
| PA_BSND / BnNBsD / PA_BBND | `[BlockNum, BlockSize, N, D]` or `[BlockNum, N, BlockSize, D]` | PA scenario KV Cache layout, with the D dimension stored contiguously. 4D format; performance is better than BnBsH, and this format is recommended first | PageAttention KV Cache storage |
| PA_NZ / NZ | `[BlockNum, N, D/D0, BlockSize, D0]` | PA scenario KV Cache layout, with the D dimension split into an NZ fractal structure of D/D0 × D0. 5D; the innermost D0=32 (FP8/INT8) or 16 (FP16/BF16/keyRope); dedicated to quantization scenarios, and **only NZ is supported** for INT8 | PageAttention KV Cache storage in quantization scenarios |

> **Supplementary notes**:
> - **BNSD_BSND**: A special combination layout. When the input is BNSD, the output format is BSND.
> - Core difference between PA_BSND and PA_NZ: the D dimension of PA_BSND is stored contiguously (4D), while PA_NZ splits the D dimension into an NZ fractal (5D), which is more Cube-instruction-friendly in quantization scenarios.

### 4.2 PageAttention (Paged Attention)

#### 4.2.1 Principle and Design Motivation

In LLM autoregressive inference, the historical KV Cache must be read every time a token is generated. The traditional KV Cache requires each batch to reserve a segment of contiguous physical memory along the S dimension (sequence length) and pre-pad according to `maxSeqLen`, resulting in memory fragmentation and waste.

PagedAttention borrows the virtual memory paging mechanism of the operating system, splits the KV Cache into fixed-size blocks stored in a segment of contiguous physical memory, and establishes the mapping from logical sequence order to physical block addresses through `blockTable` (page table). Its core values:

- **Eliminates memory fragmentation**: allocates blocks on demand, with no need to pre-allocate an entire segment of contiguous memory
- **Improves memory utilization**: different batches share the same physical block pool, supporting larger batches or longer sequences
- **Logically contiguous, physically discrete**: logically contiguous KV sequences can be scattered across non-contiguous physical blocks

#### 4.2.2 Working Mechanism

PagedAttention is enabled only when `blockTable` exists and is valid, and meanwhile key/value are arranged in contiguous memory according to the indices in `blockTable`. After PagedAttention is enabled, the `inputLayout` parameter of key/value becomes invalid, and the layout of the KV Cache is determined by its own shape dimensions.

**blockTable lookup process** (taking `B=2`, `blockSize=128`, `actualSeqLengthsKv=[256, 512]` as an example):

`maxBlockNumPerSeq = ceil(512/128) = 4`, blockTable shape = `[2, 4]`:

```
blockTable = [
  [ 3,  7,  1,  0],   # batch 0: logical block 0 → physical block 3, logical block 1 → physical block 7 (actually only 2 blocks are used)
  [ 5,  2,  9,  4],   # batch 1: logical block 0 → physical block 5, logical block 1 → physical block 2, logical block 2 → physical block 9, logical block 3 → physical block 4
]
```

When accessing tokens 0~255 of batch 0:
- token 0~127 → logical block 0 → table lookup gives block_id=3 → read physical `kvCache[3]`
- token 128~255 → logical block 1 → table lookup gives block_id=7 → read physical `kvCache[7]`

Block ids can be allocated out of order and can share the physical pool across batches. Users need to ensure the validity of block ids themselves (not verified on the host side).

#### 4.2.3 blockTable

| Item | Content |
| :--- | :--- |
| Data type | INT32 |
| shape | `[B, maxBlockNumPerSeq]` |
| First dimension | Equals the number of batches B |
| Second dimension | Not smaller than the number of blocks corresponding to the maximum KV sequence length, that is, `ceil(max_KV_S / blockSize)` |
| Element meaning | block id, pointing to the n-th block in the KV Cache physical pool |
| Constraint | Users must ensure the validity of block ids themselves; not verified on the host side |

#### 4.2.4 blockSize

| Item | Content |
| :--- | :--- |
| Meaning | The maximum number of tokens in each block; a user-defined parameter |
| General alignment | FP16/BF16 requires alignment to 16; INT8/HIFLOAT8/FP8_E4M3 requires alignment to 32; INT4/FLOAT4_E2M1 requires alignment to 64 |
| Maximum | 1024 (512 in some earlier versions) |
| Recommended value | 128 (best compatibility; the mandatory value in most full quantization scenarios) |
| Performance impact | Increasing blockSize brings certain performance benefits (reducing table lookups and increasing the data volume per transfer), but increases the memory waste of the tail block |

#### 4.2.5 Comparison Between PA Mode and Normal Mode

| Dimension | Normal mode | PA mode |
| :--- | :--- | :--- |
| KV physical layout | Contiguous along the S axis | Scattered by block, mapped through blockTable |
| inputLayout (KV) | Effective | Invalid |
| blockTable | Not required | Required |
| actualSeqLengthsKv | Optional | Required |
| tensorlist | Supported | Not supported |
| Left padding | Supported | Not supported |
| Shared prefix | Supported | Not supported |
| Unequal D | Supported | Not supported |
| KV Cache layout | BSH/BSND/BNSD/TND and so on | BnBsH / BnNBsD / NZ |

#### 4.2.6 Performance Characteristics

- **Throughput improvement vs. single-run latency decrease**: PA supports larger batches by improving memory utilization, thereby increasing throughput. However, the indirect addressing of blockTable introduces extra table lookup overhead and non-contiguous access, slightly decreasing single-run inference latency
- **KV Cache layout affects performance**: BnNBsD performs better than BnBsH (placing the N axis first facilitates fetching contiguous D data by head). BnNBsD is recommended first
- **BnBsH limitation**: when KV_N × D exceeds 65535, an error is reported due to hardware instruction constraints; switch to the BnNBsD format

### 4.3 Mask Mode

Mask Mode (`sparseMode` / `mask_mode`) controls the masking strategy in attention computation. Parameter names differ across operators but the semantics are the same.

#### 4.3.1 Value Definitions

| Value | Name | Meaning | attenMask requirement | preTokens/nextTokens |
| :---: | :--- | :--- | :--- | :--- |
| 0 | defaultMask | Default mode; if no mask is passed, all is computed; if a mask is passed, causal or band is applied based on preTokens/nextTokens | Optional | Effective |
| 3 | rightDownCausal | Lower triangular scenario divided by the bottom-right vertex | Compressed lower triangular matrix (2048×2048) | Not effective |
| 4 | band | Computes the part between preTokens and nextTokens, starting from the bottom-right corner | Compressed lower triangular matrix (2048×2048) | Effective |

#### 4.3.2 rightDownCausal (mode=3)

Bottom-right vertex causal mask. Taking the bottom-right corner of the attention matrix as the vertex, the lower triangular part (including the diagonal) is retained and the upper triangle is masked. That is, query token i can only attend to key tokens 0~i and cannot see future tokens. Applicable to autoregressive training and inference of standard causal language models (GPT/Llama, and so on).

```
       key:  0  1  2  3  4
          ┌───────────────┐
    q 0   │  1  0  0  0  0 │
    q 1   │  1  1  0  0  0 │
    q 2   │  1  1  1  0  0 │
    q 3   │  1  1  1  1  0 │
    q 4   │  1  1  1  1  1 │
          └───────────────┘
       1 = computed    0 = masked
```

The passed attenMask is a compressed lower triangular matrix (2048×2048), and preTokens/nextTokens do not take effect.

#### 4.3.3 band (mode=4)

Band mask. Starting from the bottom-right corner, a band-shaped region is defined through `preTokens` and `nextTokens`. Query token i can only attend to key tokens within the range `[i - preTokens + 1, i + nextTokens]`. Applicable to the sliding window attention (Sliding Window Attention) scenario.

Taking `preTokens=3, nextTokens=0` as an example:

```
       key:  0  1  2  3  4
          ┌───────────────┐
    q 0   │  1  0  0  0  0 │
    q 1   │  1  1  0  0  0 │
    q 2   │  1  1  1  0  0 │
    q 3   │  0  1  1  1  0 │
    q 4   │  0  0  1  1  1 │
          └───────────────┘
       1 = computed    0 = masked
       window width = preTokens = 3
```

The passed attenMask is a compressed lower triangular matrix (2048×2048), and `preTokens`/`nextTokens` take effect.

### 4.4 cuSeqlens & seqused

These two parameters describe the actual valid length of each batch in variable-length sequences. They are used differently in different layout scenarios: the TND layout concatenates the sequences of multiple batches into a 1D tensor, requiring `cuSeqlens` to mark batch boundaries; in non-TND layouts (BSND/BNSD/BSH, and so on), each batch has an independent S dimension, requiring `seqused` to mark the actual valid length.

| Item | cuSeqlens | seqused |
| :--- | :--- | :--- |
| Meaning | Cumulative prefix sum, marking the boundaries of each batch; each element is the accumulated number of valid tokens of the current batch and preceding batches | The number of tokens that each batch actually participates in computation |
| shape | `(B+1,)` | `(B,)` |
| Data type | INT32 | INT32 |
| First element | Fixed to 0 | No requirement |
| Numeric constraint | Monotonically non-decreasing; the last element equals T (the total number of tokens) | Not greater than S in the shape |
| Numeric relationship | `cuSeqlens[i+1] - cuSeqlens[i] = seqused[i]` | Can be obtained by differencing cuSeqlens |
| TND layout | Required (data is compactly arranged by batch, and boundaries need to be marked) | Optional (the total length is expressed by cuSeqlens, and seqused can explicitly override the valid length) |
| Non-TND layout | Forbidden to pass | Optional (used to mark the valid length of each batch) |

#### 4.4.1 Equivalent Padding Effect in the TND Layout

The TND layout compactly concatenates the sequences of multiple batches into a 1D tensor, requiring no padding. `cuSeqlens` marks the boundaries of each batch, and `seqused` marks the valid length of each batch. The two work together to achieve the equivalent padding effect.

Taking `B=3` and the valid lengths of batches `2, 1, 3` as an example:

```
TND (compact concatenation, zero waste):              BSND (requires padding to maxLen=3):

┌────┬────┬────┬────┬────┬────┐    ┌────┬────┬─────┐
│ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │    │ t0 │ t1 │ pad │  batch0
└────┴────┴────┴────┴────┴────┘    ├────┼────┼─────┤
  b0: 2   b1:1   b2:  3           │ t2 │pad │ pad │  batch1
                                    ├────┼────┼─────┤
cuSeqlens = [0, 2, 3, 6]            │ t3 │ t4 │ t5  │  batch2
seqused   = [2, 1, 3]               └────┴────┴─────┘
                                     occupies 9 slots, wastes 3 slots
occupies 6 slots, zero waste
```

### 4.5 Sparse Index Parameters (topk / sparse_indices)

Sparse attention operators select the key KV subset through TopK and perform attention computation only on the selected KV. `topk` controls the number of selections, and `sparse_indices` records the selected position indices. The two form the production-consumption link of Lightning Indexer → Sparse Flash Attention.

#### 4.5.1 topk (Number of TopK Selections)

The core idea of sparse attention is not to perform attention computation on all KV. Instead, the most relevant KV positions for each query token are filtered out first through correlation scoring, and `Softmax(QK^T)V` is executed only on this part of KV, greatly reducing the computation load. This filtering process is completed in the Lightning Indexer operator.

`topk` is the quantity parameter that controls how many to select: the larger the topk, the more KV participates in computation, and the closer the attention result is to the precision of dense attention, but the computation and memory access overhead also increase; the smaller the topk, the faster the computation, but key information may be missed, resulting in precision degradation. Weigh performance against precision based on the model precision requirements and sequence length. `topk` directly determines the size of the last dimension of the downstream `sparse_indices` output.

Taking the sequence length `KV_S=8192` as an example:

```
topk=2048: selects 25% of the KV, computation reduced by 75%
topk=512:  selects 6% of the KV, computation reduced by 94%
```

#### 4.5.2 sparse_indices (Sparse Position Indices)

`sparse_indices` is the output of the Lightning Indexer and the input of Sparse Flash Attention / Sparse Flash MLA, recording the KV position indices that each query token should attend to. It passes the topk filtering result to the attention main operator in the form of an INT32 index array. The main operator discretely gathers the KV at the corresponding positions based on it and then performs attention computation, instead of performing dense computation on all KV.

In the index array, valid positions store the actual block indices of the KV, and invalid positions are filled with -1. The consumer side convention is that valid values are arranged in the first half and invalid values in the second half, so that the main operator only needs to process the first `topk` valid entries and skip the invalid part at the tail.

**Production-consumption link**:

```
lightning_indexer(query, key, weights, topk) → sparse_indices (output)
                                                    ↓
sparse_flash_attention(query, key, value, sparse_indices) → attn_out (input)
```

Lightning Indexer computes `ReLU(Q@K^T)` to obtain correlation scores, multiplies them by weights and aggregates along the group, and selects and outputs the first `topk` indices through TopK. Sparse Flash Attention discretely gathers the KV accordingly and then executes `Softmax(QK^T)@V`. The entire link decouples selecting which KV and performing attention on the selected KV into two independent operators, facilitating separate optimization and flexible combination.

### 4.6 Quantization Scenario Concepts

According to whether Q/K/V are quantized and the quantization granularity, attention layer operators fall into the non-quantized, pseudo-quantized, and fully quantized scenarios.

#### 4.6.1 Quantization Type Classification

| Quantization type | Q | K/V | Dequantization parameters | Description |
| :--- | :--- | :--- | :--- | :--- |
| Non-quantized | FP16/BF16 | FP16/BF16 | None | Q/K/V are all high-precision and directly participate in computation |
| Pseudo-quantized | FP16/BF16 | INT8/FP8/HIF8 and so on | `antiquantScale` / `antiquantOffset` | Only K/V is quantized for storage and dequantized within the operator before computation; Q and the output remain high-precision |
| Fully quantized | INT8/FP8/HIF8 and so on | INT8/FP8/HIF8 and so on | `dequantScaleQuery` + `antiquantScale` | Q/K/V are all quantized inputs, all dequantized within the operator before computation |

#### 4.6.2 Dequantization Parameter System

| Parameter | Function | Applicable scenario |
| :--- | :--- | :--- |
| `antiquantScale` / `antiquantOffset` | Dequantization factor and offset for K/V pseudo-quantization | Pseudo-quantization (symmetric/asymmetric) |
| `keyAntiquantScale` / `valueAntiquantScale` | Separate dequantization factors for K/V (K/V can use different scales) | Pseudo-quantization/full quantization (K/V separate mode) |
| `dequantScaleQuery` | Factor for dequantizing Q | **Dedicated to full quantization** |
| `deqScale1` / `quantScale1` | Dequantizes the QK result / quantizes P | INT8 online quantization (Q high-precision scenario) |
| `deqScale2` / `quantScale2` / `quantOffset2` | Dequantizes the PV result / quantizes the output and offset | INT8 online quantization (output INT8 scenario) |

#### 4.6.3 GQA Full Quantization (FP8)

In the GQA full quantization scenario, Q/K/V are all in FLOAT8_E4M3FN format, and high-precision computation is restored within the operator through FLOAT32 dequantization factors.

| Item | Content |
| :--- | :--- |
| Q/K/V data type | FLOAT8_E4M3FN |
| Dequantization factor dtype | FLOAT32 |
| Quantization granularity | `queryQuantMode=3` (per-token plus per-head), `keyAntiquantMode=3` (per-token plus per-head), `valueAntiquantMode=2` (per-tensor plus per-head) |
| HeadDim constraint | Only 128 is supported |
| blockSize constraint | Only 128 is supported |
| Typical operator | fused_infer_attention_score |

#### 4.6.4 MXFP8 Full Quantization

MXFP8 (Microscaling FP8) is a quantization format combining FP8_E4M3FN data with FLOAT8_E8M0 scaling factors, where every 32 elements share one E8M0 scale.

| Item | Content |
| :--- | :--- |
| Q/K/V data type | FLOAT8_E4M3FN |
| Scaling factor dtype | FLOAT8_E8M0 |
| Quantization granularity | `queryQuantMode/keyAntiquantMode=6` (per-token-group), `valueAntiquantMode=8` (per-channel-group) |
| scale shape | `(1, B, N, >=KV_S, D/32)`, every 32 D elements share one E8M0 scale |
| HeadDim constraint | Only 64 or 128 is supported |
| blockSize constraint | Supports 64/128/256/512/1024 |
| Typical operator | quant_flash_attn, fused_infer_attention_score |

### 4.7 Batch Consistency

Batch consistency (deterministic computation) refers to the ability of an operator to keep computation results consistent when executing under different batch sizes and different batch combinations. By default, the computation results of some operators (such as compressor / quant_compressor) may be related to the position of the input batch in the overall data and the batch division method. That is, the same group of data may produce slight numerical differences under different batch divisions.

Configuring `ACL_OPT_DETERMINISTIC=3` through `aclrtSetSysParamOpt()` enables the batch consistency mode. After enabling, the operator internally adopts a deterministic computation strategy, ensuring that computation results are independent of the batch size and position.

| Item | Content |
| :--- | :--- |
| Meaning | Computation results are independent of the batch size and the position of the batch in the data |
| Enabling method | `aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3)` |
| Applicable operators | compressor, quant_compressor, and so on |
| Default state | Disabled (non-deterministic) |
| Cost | Enabling may cause certain performance degradation |

---
