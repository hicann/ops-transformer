# pool\_key\_indexer

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: `pool_key_indexer` obtains the Top-$k$ positions corresponding to each token through a series of operations. Different from `lightning_indexer`, `pool_key_indexer` introduces the **pooling** mechanism on the Key side. The original Key sequence is aggregated in groups of `pool_size` tokens (each pool is one token). TopK selection is first performed at the pool level, and then the selected pools are expanded into original token indices.

- Calculation formula:

    $$
    Top\text{-}k \leftarrow Expand\left(Top\text{-}\frac{k}{p}\left\{[1]_{1\times g}@\left[\left(W@[1]_{1\times S_{k}^{pool}}\right)\odot ReLU\left(\left(Q_{index}@{K_{index}^{Pool}}^T\right)\odot\frac{1}{\sqrt{d}}\right)\right]\right\}\right) + AppendTail
    $$

    where $p = pool\_size$, $S_{k}^{pool} = \lfloor S_{k} / pool\_size \rfloor$, and $d$ is the attention head dimension (headDim). The scaling coefficient $\frac{1}{\sqrt{d}}$ prevents the dot product from becoming too large. $Expand$ expands each pool index into $p$ consecutive token indices, and $AppendTail$ appends the `pool_tail_k` valid tokens in the tail pool. The incomplete tail pool is not included in $pool\_key$; $pool\_key$ contains only complete pools. The indices of the tail tokens are the $pool\_tail\_k$ consecutive original token indices starting from the number of complete pools $\times pool\_size$.

    The main computation process is as follows:

    1. Multiplies the input parameter `query` ($Q_{index}\in\R^{g\times d}$) corresponding to a token by the pooled context `pool_key` ($K_{index}^{Pool}\in\R^{S_{k}^{pool}\times d}$), then multiplies the result by the scaling coefficient $\frac{1}{\sqrt{d}}$ to obtain relevance scores.
    2. Filters invalid negative correlation signals through the activation function $ReLU$ to obtain the pool-level relevance score vector between the current token and all preceding tokens.
    3. Multiplies the vector by the weight coefficient `weights` ($W$) and aggregates it along the g direction.
    4. Selects the first $Top\text{-}k / pool\_size$ pool indices in the pool space and expands them into $Top\text{-}k$ token indices.
    5. Appends the valid tokens in the tail pool that are not fully covered by pool_size (specified by `pool_tail_k`), and fills invalid positions with -1.

- Quantization support: `quant_mode` is -1 (no quantization, FP16/BF16), 0 (FP8 per-token-head quantization), 1 (mxFP8 quantization), or 2 (int8 quantization).

## Parameter Description

> **Description:**<br>
> Parameter dimensions: B indicates the batch size. S1 and S2 indicate the sequence lengths of query and pool_key respectively (S2 is at the pool level). N1 and N2 indicate the head nums of query and pool_key respectively. D indicates the head dim (only 128 is supported). T1 and T2 indicate the total tokens of query and pool_key respectively. block_num and block_size indicate the total number of blocks and the number of pools contained in each block in the PageAttention scenario respectively. N2 supports only 1.

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| query | Input | Input Q in the formula. Empty tensors are not supported. When layout_q is BSND, the shape is [B, S1, N1, D]; when layout_q is TND, the shape is [T1, N1, D]. | FLOAT16, BFLOAT16, FLOAT8_E4M3FN, INT8 | ND |
| pool_key | Input | Pooled input K in the formula. Empty tensors are not supported. When layout_k is PA_BBND, the shape is [block_num, block_size, N2, D]; when layout_k is BSND, the shape is [B, S2, N2, D]; when layout_k is TND, the shape is [T2, N2, D]. The data type must be the same as that of query. | FLOAT16, BFLOAT16, FLOAT8_E4M3FN, INT8 | ND |
| weights | Input | Input W in the formula. Empty tensors are not supported. When layout_q is BSND, the shape is [B, S1, N1]; when layout_q is TND, the shape is [T1, N1]. | FLOAT16, BFLOAT16 | ND |
| pool_tail_k | Input | Number of valid tokens in the incomplete tail pool of each batch. The value range is [0, pool_size-1]. The value 0 indicates no valid tail tokens. The shape is [B, ]. | INT64 | ND |
| actual_seq_q | Input | Number of valid tokens of Query in each batch. Required when layout_q is TND. The value of each element indicates the total number of tokens of the current batch and all preceding batches (prefix sum). The shape is [B, ]. | INT64 | ND |
| actual_seq_k | Input | Number of valid pools of pool_key in each batch. Required when layout_k is TND (prefix sum); required when layout_k is PA_BBND (non-prefix sum, indicating the number of pools of the current batch). The shape is [B, ]. | INT64 | ND |
| block_table | Input | Indicates the block mapping table used for KV storage in PageAttention. Required when layout_k is PA_BBND. The shape is [B, maxBlockNumPerSeq]. | INT32 | ND |
| q_descale | Input | Dequantization coefficient of Query. Valid only when quant_mode>=0. When quant_mode is 0, the shape is [B, S1, N1] (BSND) or [T1, N1] (TND); when quant_mode is 1, the shape is [B, S1, N1, D/64, 2] (BSND) or [T1, N1, D/64, 2] (TND); when quant_mode is 2, the shape is [B, S1, N1] (BSND) or [T1, N1] (TND). | FLOAT, FLOAT8_E8M0, FLOAT16 | ND |
| k_descale | Input | Dequantization coefficient of pool_key. Valid only when quant_mode>=0. When quant_mode is 0, the shape is [B, S2, N2] (BSND), [T2, N2] (TND), or [block_num, block_size, N2] (PA_BBND); when quant_mode is 1, [D/64, 2] is appended to the last dimension; when quant_mode is 2, the shape is [B, S2, N2] (BSND), [T2, N2] (TND), or [block_num, block_size, N2] (PA_BBND). Non-contiguous axis 0 is supported only in the PA_BBND layout. | FLOAT, FLOAT8_E8M0, FLOAT16 | ND |
| layout_q | Attribute | Identifies the data layout format of input Query. Supports BSND and TND. The default value is BSND. | STRING | - |
| layout_k | Attribute | Identifies the data layout format of input pool_key. Supports PA_BBND, BSND, and TND. The default value is BSND. | STRING | - |
| topk | Attribute | Number of tokens to be retained after expansion. Supports [1, 2048] as well as 3072, 4096, 5120, 6144, 7168, and 8192. topk % pool_size == 0 must be satisfied. The default value is 2048. | INT64 | - |
| pool_size | Attribute | Number of tokens contained in each pool. Supports [1, 128]. The default value is 16. The value 1 indicates no pooling, degrading to the lightning_indexer behavior. | INT64 | - |
| mask_mode | Attribute | Indicates the mask mode. The value 0 indicates the defaultMask mode, and the value 3 indicates the mask of the rightDownCausal mode, corresponding to the lower triangle scenario divided by the right vertex. The default value is 3. | INT64 | - |
| quant_mode | Attribute | Indicates the quantization mode of Query/Key. The value -1 indicates no quantization, the value 0 indicates FP8 per-token-head quantization (the dequantization coefficient is FLOAT), the value 1 indicates mxFP8 quantization (the dequantization coefficient is FLOAT8_E8M0), and the value 2 indicates int8 quantization (query/pool_key is INT8, weights is FLOAT16, q_descale is FLOAT16 at the per-token-head granularity, and k_descale is FLOAT16 at the per-token (pool row) granularity). The default value is -1. | INT64 | - |
| return_value | Attribute | Indicates whether to output sparse_values. True indicates output, and False indicates no output (an empty tensor with the shape (0,) is output). The default value is False. | BOOL | - |
| key_stride0 | Attribute | Stride of axis 0 of pool_key (in elements), used for addressing the pool_key with a non-contiguous axis 0 when layout_k is PA_BBND. The value -1 indicates that it is not specified, and the stride is derived from the shape as contiguous input. The default value is -1. | INT64 | - |
| k_descale_stride0 | Attribute | Stride of axis 0 of k_descale (in elements), used for addressing the k_descale with a non-contiguous axis 0 when layout_k is PA_BBND. Valid only in quantization scenarios. The value -1 indicates that it is not specified, and the stride is derived from the shape as contiguous input. The default value is -1. | INT64 | - |
| sparse_indices | Output | Indices output in the formula, which is the expanded sparse token index, with invalid parts filled with -1. When layout_q is BSND, the shape is [B, S1, topk + pool_size - 1]; when layout_q is TND, the shape is [T1, topk + pool_size - 1]. | INT32 | ND |
| sparse_values | Output | Values output corresponding to Indices in the formula, that is, the score values of the selected pools, with invalid parts filled with -inf. When return_value is True, the shape is [B, S1, topk // pool_size] when layout_q is BSND, and [T1, topk // pool_size] when layout_q is TND; when return_value is False, the shape is (0,). | FLOAT | ND |

## Constraints

- This API is used in inference scenarios and supports invocation in the single-operator mode, TorchAir (GE graph) mode, and TorchAir (aclgraph) graph mode.
- The N1 of query supports values less than or equal to 64. The N2 of pool_key supports only 1. D (HeadDim) supports only 128.
- The data types of query and pool_key should be the same. In non-quantization scenarios (quant_mode is -1), the data type of weights should be the same as those of query and pool_key. In quantization scenarios (quant_mode is 0 or 1), weights is FLOAT16 or BFLOAT16. In the int8 quantization scenario (quant_mode is 2), weights supports only FLOAT16.
- The data layouts of query and pool_key (layout_q and layout_k) must be the same, except in the PageAttention scenario (layout_k is PA_BBND). In non-PA scenarios, when layout_q and layout_k are inconsistent (for example, a BSND/TND or TND/BSND mix), the host side reports an error and rejects the invocation.
- topk must satisfy topk % pool_size == 0. topk supports [1, 2048] as well as 3072, 4096, 5120, 6144, 7168, and 8192. pool_size supports [1, 128].
- When pool_size=1 and pool_tail_k=0, the behavior degrades to that of lightning_indexer, and the output dimension is topk.
- When layout_k is PA_BBND, block_table and actual_seq_k must be passed in. When layout_k is not PA_BBND, passing in block_table is not supported.
- When layout_q is BSND, passing in actual_seq_q is not supported. When layout_q is TND, actual_seq_q must be passed in. When layout_k is BSND, passing in actual_seq_k is not supported. When layout_k is TND, actual_seq_k must be passed in.
- In the PA_BBND scenario, block_size is a multiple of 16, with a maximum of 2048.
- pool_key and k_descale support a non-contiguous axis 0 only in the PA_BBND layout (which requires the key_stride0/k_descale_stride0 attributes). The BSND/TND layouts require contiguous input.
- When quant_mode=-1, passing in q_descale and k_descale is not supported.
- The value check of pool_tail_k is performed on the host side only for CPU input. NPU input directly passes device data through the aclnn Tensor variant, and no value check is performed on the host side. The caller must ensure that the value is valid; otherwise, the kernel behavior is undefined.
- Invalid parts of sparse_indices are filled with -1, and invalid parts of sparse_values are filled with -inf.
- For details about the mask modes indicated by mask_mode, see [sparse_mode Parameter Description](../../docs/zh/context/sparse_mode_introduction_en.md).
- <term>Ascend 950PR&950DT products</term>:
  - query and pool_key support the FLOAT8_E4M3FN data type.
  - quant_mode supports -1 (no quantization), 0 (FP8 per-token-head quantization), 1 (mxFP8 quantization), and 2 (int8 per-token-head quantization).
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - The FLOAT8_E4M3FN and FLOAT8_E8M0 data types are not supported, and the FP8 quantization scenario (quant_mode 0/1) is not supported.
  - quant_mode supports -1 (no quantization) and 2 (int8 per-token-head quantization).

## Calling Description

| Calling Method | Sample Code | Description |
|----------|----------|------|
| aclnn API | [test_aclnn_pool_key_indexer](./examples/test_aclnn_pool_key_indexer.cpp) | Calls the operator through the [aclnnPoolKeyIndexer](docs/aclnnPoolKeyIndexer_en.md) API |
| PyTorch API | - | Calls the pool_key_indexer operator through the [torchapi_pool_key_indexer](docs/torchapi_pool_key_indexer_en.md) API |
