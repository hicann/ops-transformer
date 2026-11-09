# QuantLightningIndexerV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Support |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- API function: In inference scenarios, QuantLightningIndexerV2 performs the preprocessing computation of sparse attention. It selects key sparse tokens and quantizes the inputs q and k to implement 8-bit storage and 8-bit computation, achieving maximum benefit.

- Computation formula:
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    The main computation process is:
    1. Multiply the input parameter `q` of a token ($Q_{index}^{Quant}\in\R^{g\times d}$) by the given context `k` ($K_{index}^{Quant}\in\R^{S_{k}\times d}$) to obtain the relevance.
    2. Multiply the relevance result by the dequantization coefficients `q_descale` ($Scale_Q$) and `k_descale` ($Scale_K^T$) corresponding to `q` and `k`, filter invalid negative relevance signals through the activation function $ReLU$, and obtain the relevance score vector between the current token and all preceding tokens.
    3. After multiplying it by the weight coefficient `w` ($W$), select the first $Top-k$ index values along the g direction to obtain the output $out$, which serves as the input of Attention.

## Parameter Description

| Parameter Name                     | Input/Output/Attribute | Description  | Data Type       | Data Format   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| q                     | Input      | $Q_{index}^{Quant}\in\R^{g\times d}$ in the formula, representing input Index Query; non-contiguous is not supported.| INT8, FLOAT8_e4m3fn, HIFLOAT8, FLOAT4_e2m1 | ND         |
| k                   | Input      | $K_{index}^{Quant}\in\R^{S_{k}\times d}$ in the formula, representing the compressed input Index Key; non-contiguity on axis 0 is supported.| INT8, FLOAT8_e4m3fn, HIFLOAT8, FLOAT4_e2m1 | ND |
| w                 | Input      | $W$ in the formula, representing the weight coefficient; non-contiguous is not supported. | FLOAT16, FLOAT32 | ND |
| q_descale             | Input      | $Scale_Q$ in the formula, representing the dequantization coefficient of Index Query; non-contiguous is not supported. When `quant_mode` is 3/5, the shape is obtained by replacing the D axis of `q` with (D/64, 2); when `quant_mode` is 4, the shape is (1,); in other scenarios, the shape is consistent with w | FLOAT16, FLOAT32, FLOAT8_e8m0     | ND         |
| k_descale            | Input      | $Scale_K$ in the formula, representing the dequantization coefficient of Index Key; non-contiguity on axis 0 is supported. When `quant_mode` is 3/5, the shape is obtained by replacing the D axis of `k` with (D/64, 2); when `quant_mode` is 4, the shape is (1,); in other scenarios, the shape is obtained by removing the D axis of `k` | FLOAT16, FLOAT32, FLOAT8_e8m0       | ND         |
| cu_seqlens_q                    | Optional input      | Must be passed in when layout_q is TND, representing the prefix sum of the number of valid tokens of `q` in each batch; it cannot be passed in when layout_q is BSND. The default value is None. | INT32       | ND         |
| cu_seqlens_k                    | Optional input      | Must be passed in when layout_k is TND, representing the prefix sum of the number of valid tokens of `k` in each batch; it cannot be passed in when layout_k is PA_BBND or BSND. The default value is None. | INT32       | ND         |
| seqused_q                    | Optional input      | Optionally passed in when layout_q is BSND, representing the number of valid tokens of `q` in each batch. The default value is None. | INT32       | ND         |
| seqused_k                    | Optional input      | Used when layout_k is PA_BBND or BSND, representing the number of valid tokens of `k` in each batch. The default value is None.| INT32       | ND         |
| cmp_residual_k                    | Optional input      | Residual length of Key in compression scenarios, which must satisfy 0 \<= cmp_residual_k\[i\] \< cmp_ratio. The default value is None.| INT32       | ND         |
| block_table                    | Optional input      | Represents the block mapping table used for KV storage in PageAttention. The default value is None. | INT32       | ND         |
| output_idx_offset                    | Optional input      | Offset of the output indices. The default value is None. | INT32       | ND         |
| metadata                    | Optional input      | Core partitioning information passed in by the QuantLightningIndexerV2Metadata operator, including the number of used cores, block size, and the start point of the data processed by each core. The default value is None. | INT32       | ND         |
| quant_mode                 | Attribute      | Identifies the quantization mode of inputs. | INT64          | -         |
| max_seqlen_q                 | Optional attribute| Maximum sequence length of Query; the default value -1 means any possible length | INT64 | -         |
| layout_q                 | Optional attribute| Identifies the data layout format of input `q`, with a default value of "BSND". | STRING | -         |
| layout_k      | Optional attribute      | Identifies the data layout format of input `k`, with a default value of "BSND".| STRING          | -         |
| topk  | Attribute      | Represents the number of indices to retain in the topK stage, with a default value of 2048. | INT64          | -         |
| mask_mode | Optional attribute      | Represents the mask mode, with a default value of 0. | INT64          | -         |
| cmp_ratio      | Optional attribute      | Used for sparse computation; represents the compression ratio of k, with a default value of 1. | INT64          | -         |
| return_value      |  Optional attribute     | Indicates whether to output `sparse_values`, with a default value of 0. | INT64          | -         |
| sparse_indices     | Output      | Output Out in the formula, representing the token index values that participate in the sparse attention computation. | INT32          | ND         |
| sparse_values           | Output      | Value corresponding to the Indices output in the formula. When `return_value` is 1, the shape is consistent with `sparse_indices`; when `return_value` is 0, the shape is (0,) | BFLOAT16         | ND          |

## Constraint Description

- <term>Ascend 950PR&950DT products</term>:
  - `q` and `k` support FLOAT8_e4m3fn when `quant_mode` is 1/3, INT8 when `quant_mode` is 2, HIFLOAT8 when `quant_mode` is 4, and FLOAT4_e2m1 when `quant_mode` is 5.
  - `q_descale` and `k_descale` support FLOAT32 when `quant_mode` is 1/4, FLOAT16 when `quant_mode` is 2, and FLOAT8_e8m0 when `quant_mode` is 3/5.
  - `w` supports FLOAT16 when `quant_mode` is 2 and FLOAT32 when `quant_mode` is 1/3/4/5.
  - When `quant_mode` is 3/5, the number of dimensions of `q_descale` and `k_descale` is 1 more than that of `q` and `k` respectively, the prefix dimensions are consistent, and the last two dimensions are (D/64,2); D must be a multiple of 64, and each scale corresponds to 32 consecutive logical elements along the D axis.
  - N of `q` supports [1, 64], and N of `k` supports only 1.
  - `topk` supports [1, 8192].
  - When the passed-in parameter layout_q is TND, cu_seqlens_q must be passed in. If seqused_q is also passed in, ensure that the q length of each batch passed in by seqused_q does not exceed the q sequence length of each batch computed from cu_seqlens_q. When the q sequence length seqlen1 of a batch passed in by seqused_q is less than the q length seqlen2 computed from cu_seqlens_q, the TND Padding feature is enabled, and the sparse_indices and sparse_values of the q output for the difference between seqlen2 and seqlen1 of that batch are all set to invalid values. In some long-sequence scenarios, if too much invalid data needs to be padded, aicore execution may time out due to hardware limitations. You can compute the amount of data to pad through (seqlen2 - seqlen1) * topk, and it is recommended to keep this amount within 400 million.

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - `quant_mode` supports only 2.
  - `q` and `k` support INT8 and do not support FLOAT8_e4m3fn, HIFLOAT8, or FLOAT4_e2m1.
  - `q_descale` and `k_descale` support FLOAT16 and do not support FLOAT32 or FLOAT8_e8m0.
  - `w` supports FLOAT16 and does not support FLOAT32.
  - `output_idx_offset` and `return_value` are not supported.
  - `layout_k` supports PA_BBND and TND: when N of `q` is 64, only PA_BBND is supported (`seqused_k` must be passed in); when N is 24/32, PA_BBND (`seqused_k` must be passed in) or TND (`cu_seqlens_k` must be passed in, and `seqused_k` is not supported) is supported.
  - N of `q` supports 64, 32, and 24, and N of `k` supports only 1; when N is 24/32, `layout_q` must be TND, `mask_mode` must be 3, and `cmp_ratio` must be 1; otherwise the API reports an error.
  - When N is 24/32, the TND Padding feature is not supported: passing in `seqused_q` when `layout_q` is TND causes the API to report an error.
  - `topk` supports [1, 2048].

## Invocation Example

| Invocation Method | Invocation Example | Description |
|----------|----------|------|
| PyTorch API | - | Call the QuantLightningIndexerV2 operator through the [torch.ops.cann_ops_transformer.quant_lightning_indexer](docs/torchapi_quant_lightning_indexer_en.md) interface. |
| aclnn API | [test_aclnn_quant_lightning_indexer_v2](examples/test_aclnn_quant_lightning_indexer_v2.cpp) | Call the QuantLightningIndexerV2 operator through the [aclnnQuantLightningIndexerV2](docs/aclnnQuantLightningIndexerV2_en.md) two-phase interface. |
