# sparse_flash_attention_antiquant_v2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:06:08.709Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description:
  `sparse_flash_attention_antiquant_v2` is an extended `torch` API of `cann_ops_transformer`, used to call the `SparseFlashAttentionAntiquantV2` operator to perform attention computation in quantization and sparsity use cases.

  As the context length of large models increases, Sparse Attention is becoming increasingly important. This technique substantially reduces computation by "computing only key parts". However, it introduces a large number of discrete memory accesses, increasing data movement time and degrading overall performance. On top of `sparse_flash_attention`, `sparse_flash_attention_antiquant_v2` supports Per-Token-Head-Tile-128 quantized inputs, and is carefully optimized for discrete memory accesses through instruction reduction and data movement aggregation.

- Formula:

  $$
  Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
  $$

  where $\tilde{K},\tilde{V}$ are the Key and Value with high importance obtained by a selection algorithm (such as `LightningIndexer`). They are generally sparse or block-sparse. $d_k$ represents the dimension of each head of $Q,\tilde{K}$. $\text{Dequant}(\cdot,\cdot)$ is the dequantization function.

## Function Prototype

```python
cann_ops_transformer.sparse_flash_attention_antiquant_v2(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=0,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
    key_dtype=None,
    value_dtype=None,
    sinks=None,
    return_softmax_lse=False
) -> (Tensor, Tensor, Tensor)
```

## Parameters

### Common Field Descriptions

|    Naming    |                            Meaning                            |
| :---------: | :---------------------------------------------------------: |
|      B      |      Batch size of input samples.                                        |
|     Q_S     |      Sequence length of the input query.                                       |
|    KV_S     |      Sequence length of the input key/value.                                   |
|     Q_N     |      Number of heads of the input query.                                          |
|    KV_N     |      Number of heads of the input key/value.                                       |
|     Q_D     |      Attention head dimension of the input query.                                   |
|    KV_D     |      Attention head dimension of the input key/value.                                |
|     Q_T     |      Sum of sequence lengths of all batches of the input query (Total Tokens).           |
|    KV_T     |      Sum of sequence lengths of all batches of the input key/value (Total Tokens).       |
|  sparse_size |      Number of blocks selected discretely at a time.                                      |
|   block_num |      Total number of blocks in the PageAttention use case.                            |
|  block_size |      Number of tokens in each block in the PageAttention use case.                      |

### sparse_flash_attention_antiquant_v2

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | Required | Q input of the attention structure. Non-contiguous input is not supported. The input is obtained by concatenating `q_nope` and `q_rope` of the same data type along dimension D. The shape is `(B, Q_S, Q_N, Q_D)` when `layout_query` is `BSND`, and `(Q_T, Q_N, Q_D)` when `layout_query` is `TND`. Q_D supports only 576, that is, q_nope+q_rope=512+64. Q_N supports 1-128 on Ascend 950, and 1/2/4/8/16/32/48/64/128 on other platforms. | float16, bfloat16 | ND | `BSND`: (B, Q_S, Q_N, Q_D)<br>`TND`: (Q_T, Q_N, Q_D) |
| key | Tensor | Required | K input of the attention structure. The input is obtained by concatenating `k_nope`, `k_rope` of the same data type as query, and quantization parameters of float32 along dimension D. The shape is `(B, KV_S, KV_N, KV_D)` when `layout_kv` is `BSND`, `(KV_T, KV_N, KV_D)` when `layout_kv` is `TND`, and `(block_num, block_size, KV_N, KV_D)` when `layout_kv` is `PA_BSND` (PageAttention), where block_size is a multiple of 16 and supports a maximum of 1024. KV_N supports only 1. KV_D supports only 656, that is, nope+rope*2+dequant_scale*4=512+64*2+4*4. | float8_e4m3, int8, hifloat8 | ND | `BSND`: (B, KV_S, KV_N, KV_D)<br>`TND`: (KV_T, KV_N, KV_D)<br>`PA_BSND`: (block_num, block_size, KV_N, KV_D) |
| value | Tensor | Required | V input of the attention structure. The data type is the same as that of key. | float8_e4m3, int8, hifloat8 | ND | Same as key |
| sparse_indices | Tensor | Required | Represents the indices for discrete access to kvCache. Non-contiguous input is not supported. The shape is `(B, Q_S, KV_N, sparse_size)` when `layout_query` is `BSND`, and `(Q_T, KV_N, sparse_size)` when `layout_query` is `TND`. In each row, valid values are in the first half, invalid values (-1) are in the second half, and sparse_size must be greater than 0. If key and value are of the hifloat8 type, sparse_size supports only 2048. | int32 | ND | `BSND`: (B, Q_S, KV_N, sparse_size)<br>`TND`: (Q_T, KV_N, sparse_size) |
| scale_value | float | Required | Reciprocal of the square root of $d_k$ in the formula. Represents the scaling factor, used as the scalar value of Muls after the matrix multiplication of query and key. | float32 | - | - |
| key_quant_mode | int64 | Required | Represents the quantization mode of key. Only 2 is supported, indicating the per_tile quantization mode. | int64 | - | - |
| value_quant_mode | int64 | Required | Represents the quantization mode of value. Only 2 is supported, indicating the per_tile quantization mode. | int64 | - | - |
| key_dequant_scale | Tensor | Optional | Reserved parameter. | - | - | - |
| value_dequant_scale | Tensor | Optional | Reserved parameter. | - | - | - |
| block_table | Tensor | Optional | Represents the block mapping table used for kvCache storage in PageAttention. The shape is `(B, ceil(KV_S_max/block_size))`, where the first dimension is B, and the second dimension is no less than the number of blocks corresponding to the maximum KV_S across all batches, that is, the rounded-up value of KV_S_max/block_size. | int32 | ND | (B, ceil(KV_S_max/block_size)) |
| actual_seq_lengths_query | Tensor | Optional | Represents the number of valid tokens of query in different batches. If not passed (None), the value is the same as Q_S of query. The shape is `(B,)`. The number of valid tokens in each batch must not exceed Q_S of query and must not be less than 0. When `layout_query` is `TND`, this parameter must be passed, and B is determined by the number of elements in this parameter. Each element represents the total number of tokens (prefix sum) of the current batch and all preceding batches, and each element must be greater than or equal to the previous one. | int32 | ND | (B,) |
| actual_seq_lengths_kv | Tensor | Optional | Represents the number of valid tokens of key and value in different batches. If not passed (None), the value is the same as KV_S. The shape is `(B,)`. The number of valid tokens in each batch must not exceed KV_S of key/value and must not be less than 0. This parameter must be passed when `layout_kv` is `TND` or `PA_BSND`. When `layout_kv` is `TND`, each element represents the total number of tokens (prefix sum) of the current batch and all preceding batches, and each element must be greater than or equal to the previous one. | int32 | ND | (B,) |
| sinks | Tensor | Optional | Represents the learnable sinks information in the attention structure, used to maintain stability in long-text inference. Non-contiguous input is not supported. The shape is `(Q_N,)`. | float32 | ND | (Q_N,) |
| sparse_block_size | int64 | Optional | Represents the block size in the sparse phase. When sparse_block_size is 1, it is the Token-wise sparsification use case. When sparse_block_size is greater than 1 and less than or equal to 128, it is the Block-wise sparsification use case, where tokens in a block share the same sparsification decision. The default value is 1. | int64 | - | - |
| layout_query | string | Optional | Identifies the data layout format of input query. `BSND` and `TND` are supported. The default value is `BSND`. | string | - | - |
| layout_kv | string | Optional | Identifies the data layout format of input key. `BSND`, `TND`, and `PA_BSND` are supported. `PA_BSND` is used when PageAttention is enabled. The default value is `BSND`. | string | - | - |
| sparse_mode | int64 | Optional | Represents the sparse mode. When sparse_mode is 0, full computation is performed. When sparse_mode is 3, the mask is in rightDownCausal mode, corresponding to the lower-triangular use case with the dividing line extending from the bottom-right vertex to the top-left. The default value is 3. | int64 | - | - |
| pre_tokens | int64 | Optional | Used for sparse computation. Represents the number of preceding tokens for attention computation. Only $2^{63}-1$ is supported. The default value is $2^{63}-1$. | int64 | - | - |
| next_tokens | int64 | Optional | Used for sparse computation. Represents the number of succeeding tokens for attention computation. Only $2^{63}-1$ is supported. The default value is $2^{63}-1$. | int64 | - | - |
| attention_mode | int64 | Optional | Represents the attention mode. Only 2 is supported, indicating the MLA-absorb mode, where D of QK contains both rope and nope, and KV shares the same copy. | int64 | - | - |
| quant_scale_repo_mode | int64 | Optional | Represents the storage mode of quantization parameters. Only 1 is supported, indicating the combine mode, where quantization parameters are stored together with data. The default value is 1. | int64 | - | - |
| tile_size | int64 | Optional | Represents the data block size of each parameter in per_tile mode. Valid only in per_tile mode. Only 128 is supported. The default value is 128. | int64 | - | - |
| rope_head_dim | int64 | Optional | Represents the rope_head_dim size in the MLA architecture. Valid only when attention_mode is 2. Only 64 is supported. The default value is 64. | int64 | - | - |
| key_dtype | int | Optional | Represents the data type identifier of key. If key is of the hifloat8 type, `torch_npu.hifloat8` can be passed. For other data types, the default value None is kept. | int64 | - | - |
| value_dtype | int | Optional | Represents the data type identifier of value. If value is of the hifloat8 type, `torch_npu.hifloat8` can be passed. For other data types, the default value None is kept. | int64 | - | - |
| return_softmax_lse | bool | Optional | Indicates whether to return the LSE result of softmax. If True, softmax_max and softmax_sum are returned together with attention_out. If False, an empty placeholder tensor is returned. The default value is False. | bool | - | - |

## Return Values

### sparse_flash_attention_antiquant_v2

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | Required | Represents the output Attention in the formula. The output shape is the same as the shape of the input parameter query. If `layout_query` is `BSND`, the shape is `(B, Q_S, Q_N, Q_out_D)`. If `layout_query` is `TND`, the shape is `(Q_T, Q_N, Q_out_D)`, where Q_out_D=Q_D-rope_head_dim. | float16, bfloat16 | ND | `BSND`: (B, Q_S, Q_N, Q_D-rope_head_dim)<br>`TND`: (Q_T, Q_N, Q_D-rope_head_dim) |
| softmax_max | Tensor | Optional | Indicates the maximum value at each query position during the softmax computation (in the numerator $x-\max(x)$ of softmax $\max(x)$). Returns valid results when `return_softmax_lse=True`. | float32 | ND | `BSND`: (B, KV_N, Q_S, Q_N/KV_N)<br>`TND`: (KV_N, Q_T, Q_N/KV_N) |
| softmax_sum | Tensor | Optional | Indicates the sum at each query position during the softmax computation (the denominator of softmax $\sum\exp(x-\max(x))$). Returns valid results when `return_softmax_lse=True`. | float32 | ND | `BSND`: (B, KV_N, Q_S, Q_N/KV_N)<br>`TND`: (KV_N, Q_T, Q_N/KV_N) |

## Constraints

- This API can be called in single-operator mode.
- In non-PageAttention use cases, `layout_query` and `layout_kv` must be set to the same value.
- If `return_softmax_lse=False`, an empty float32 placeholder tensor of shape `[0]` is returned. If `return_softmax_lse=True`, the float32 softmax_max and softmax_sum results are returned.
- <term>Ascend 950PR&950DT products</term>:
  - The data types of key and value support only float8_e4m3, int8, and hifloat8.
  - The parameter `sparse_block_size` supports only 1.
  - key supports non-contiguous axis 0 only when `layout_kv` is `PA_BSND`.
  - The optional input parameter sinks is supported.
  - If key/value is hifloat8, `sparse_size` (the last dimension of sparse_indices) supports only 2048.

## Deterministic Computation

- Defaults to a deterministic implementation.
- batch consistency: The computation result is independent of the batch size and batch position.

## Example

### PageAttention use case (layout_query=BSND, layout_kv=PA_BSND)

```python
import torch
import torch_npu
import math
import random
import cann_ops_transformer

torch_npu.npu.set_device(0)

query_dtype = torch.bfloat16
kv_dtype = torch.int8
B = 1
Q_S = 1
KV_S = 8192
KV_S_act = 4096            # Actual KV length involved in computation
Q_N = 128
KV_N = 1
Q_D = 512                  # q_nope length
D_rope = 64
tile_size = 128
block_size = 256
sparse_size = 2048
scale_value = 1.0 / math.sqrt(Q_D)

# query: q_nope(512, bf16) + q_rope(64, bf16) concatenated along the D dimension
query = torch.randn(B, Q_S, Q_N, Q_D + D_rope, dtype=query_dtype).npu()

# key: k_nope(512, int8) + k_rope(64, bf16) + dequant_scale(512/128, fp32) concatenated by byte along the D dimension
block_num = B * (KV_S // block_size)
k_nope = torch.randint(-128, 127, (block_num, block_size, KV_N, Q_D), dtype=kv_dtype).npu()
k_rope = torch.randn(block_num, block_size, KV_N, D_rope, dtype=query_dtype).npu()
dequant_scale = torch.rand(block_num, block_size, KV_N, Q_D // tile_size, dtype=torch.float32).npu()
key = torch.cat((k_nope, k_rope.view(torch.int8), dequant_scale.view(torch.int8)), dim=-1)
value = key.clone()

# sparse_indices: block indices discretely selected for each query position, invalid positions filled with -1
sparse_indices = torch.full((B, Q_S, KV_N, sparse_size), -1, dtype=torch.int32).npu()
idxs = random.sample(range(KV_S_act - Q_S + 1), sparse_size)
sparse_indices[0, 0, 0, :] = torch.tensor(idxs, dtype=torch.int32).npu()

# block_table: block mapping table for PageAttention
block_table = torch.arange(block_num, dtype=torch.int32).reshape(B, -1).npu()

actual_seq_q = torch.tensor([Q_S] * B, dtype=torch.int32).npu()
actual_seq_kv = torch.tensor([KV_S_act] * B, dtype=torch.int32).npu()
sinks = torch.zeros(Q_N, dtype=torch.float32).npu()

attention_out, softmax_max, softmax_sum = torch.ops.cann_ops_transformer.sparse_flash_attention_antiquant_v2(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode=2,
    value_quant_mode=2,
    block_table=block_table,
    actual_seq_lengths_query=actual_seq_q,
    actual_seq_lengths_kv=actual_seq_kv,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=tile_size,
    rope_head_dim=64,
    sinks=sinks,
    return_softmax_lse=False)

torch.npu.synchronize()
```

### Full computation use case (layout_query=BSND, layout_kv=BSND, return_softmax_lse enabled)

```python
import torch
import torch_npu
import math
import cann_ops_transformer

torch_npu.npu.set_device(0)

query_dtype = torch.float16
kv_dtype = torch.int8
B = 1
Q_S = 16
KV_S = 128
Q_N = 8
KV_N = 1
Q_D = 512
D_rope = 64
tile_size = 128
sparse_size = 128
scale_value = 1.0 / math.sqrt(Q_D)

query = torch.randn(B, Q_S, Q_N, Q_D + D_rope, dtype=query_dtype).npu()

k_nope = torch.randint(-128, 127, (B, KV_S, KV_N, Q_D), dtype=kv_dtype).npu()
k_rope = torch.randn(B, KV_S, KV_N, D_rope, dtype=query_dtype).npu()
dequant_scale = torch.rand(B, KV_S, KV_N, Q_D // tile_size, dtype=torch.float32).npu()
key = torch.cat((k_nope, k_rope.view(torch.int8), dequant_scale.view(torch.int8)), dim=-1)
value = key.clone()

sparse_indices = torch.full((B, Q_S, KV_N, sparse_size), -1, dtype=torch.int32).npu()
for s in range(Q_S):
    sparse_indices[0, s, 0, :sparse_size] = torch.arange(sparse_size, dtype=torch.int32)

attention_out, softmax_max, softmax_sum = torch.ops.cann_ops_transformer.sparse_flash_attention_antiquant_v2(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode=2,
    value_quant_mode=2,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=0,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=tile_size,
    rope_head_dim=64,
    return_softmax_lse=True)

torch.npu.synchronize()
```
