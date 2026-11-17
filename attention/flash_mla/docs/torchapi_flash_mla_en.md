# flash_mla

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:23:25.041Z -->

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

- **Description**:

  `flash_mla_with_kvcache` is a `cann_ops_transformer` extension API based on `TorchNPU`, used for non-quantized MLA attention computation. The inputs `q` and `k_cache` have merged 512-dim nope and 64-dim rope. The operator does not perform KV compression or RoPE transformation.

  `k_cache` provides both Key and Value: Key uses all 576 dims, and Value uses the first 512 dims. `v`, `q_rope`, or `k_rope` is not passed separately.

  `flash_mla_with_kvcache_metadata` is the metadata generation API of the main operator. It records the AICore/AIVCore task partitioning result. The typical call flow is as follows:

  1. Prepare the Query, paged KV cache, block table, and sequence length.
  2. Call `flash_mla_with_kvcache_metadata` to generate `metadata`.
  3. Pass `metadata` and the corresponding inputs to `flash_mla_with_kvcache`.

- **Formula**:

  For each batch, retrieve the valid KV cache based on `block_table` and compute:

  $$
  K = K_{cache},\qquad V = K_{cache}[..., :DV]
  $$

  $$
  S = softmax\_scale \cdot QK^T + Mask
  $$

  $$
  Attention(Q,K,V) = Softmax(S)V
  $$

  If `return_softmax_lse` is enabled:

  $$
  softmax\_lse = \log\sum_j e^{S_j}
  $$

  `Mask` is 0 at positions where access is allowed, and negative infinity at masked positions.

> [!NOTE]
>
> B indicates the batch size, Q_T indicates the sum of physical Query sequence lengths across all batches, Q_N indicates the number of Query heads, KV_N indicates the number of KV heads, D indicates the Query/Key logical head dimension, and DV indicates the Value head dimension.

## Function Prototype

Before calling the flash_mla_with_kvcache API, call the prerequisite API flash_mla_with_kvcache_metadata first to complete the load balancing computation.

```python
cann_ops_transformer.flash_mla_with_kvcache_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=None,
    layout_q="BSND",
) -> Tensor
```

```python
cann_ops_transformer.flash_mla_with_kvcache(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
) -> (Tensor, Tensor)
```

## Parameters

### flash_mla_with_kvcache_metadata

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cache_seqlens | Tensor | Required | Valid KV length of each batch, consistent with the main operator | int32 | ND | (B,) |
| num_heads_q | int | Required | Number of Query heads; 64 and 96 are supported when paired with the current main operator | int32 | - | - |
| num_heads_kv | int | Required | Number of KV heads; 1 when paired with the current main operator | int32 | - | - |
| cu_seqlens_q | Tensor | Required | TND cumulative Query length; the first item is 0 | int32 | ND | (B+1,) |
| seqused_q | Tensor | Optional | Query length actually used by each batch | int32 | ND | (B,) |
| max_seqlen_q | int | Optional | Only -1 can be passed or omitted when paired with the main operator; None is normalized to -1 by default | int32 | - | - |
| max_seqlen_kv | int | Optional | Only -1 can be passed or omitted when paired with the main operator; None is normalized to -1 by default | int32 | - | - |
| head_dim_qk | int | Optional | Query/Key logical head dimension, fixed to 576. The 512 of the nope part cannot be passed | int32 | - | - |
| head_dim_v | int | Optional | Value head dimension, fixed to 512 | int32 | - | - |
| mask_mode | int | Optional | 0 or 3; None is normalized to 0 by default, consistent with the main operator | int32 | - | - |
| layout_q | string | Optional | Currently must be explicitly passed as TND; the historical default value BSND is not used | string | - | - |

### flash_mla_with_kvcache

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | Required | Query with nope and rope merged. | bfloat16/float16 | ND | (Q_T, Q_N, D) |
| k_cache | Tensor | Required | Paged cache providing both Key and Value. | bfloat16/float16 | ND | PA_BBND: (num_blocks, block_size, KV_N, D); PA_NZ: (num_blocks, KV_N, D / D0, block_size, D0) |
| block_table | Tensor | Required | Index mapping from logical pages to physical pages for each batch. | int32 | ND | (B, max_num_blocks_per_seq) |
| cache_seqlens | Tensor | Required | Valid KV length for each batch. | int32 | ND | (B,) |
| cu_seqlens_q | Tensor | Required | Cumulative Query length. The first item is 0 and the last item is Q_T. | int32 | ND | (B+1,) |
| seqused_q | Tensor | Optional | Query length actually used for each batch. | int32 | ND | (B,) |
| attn_mask | Tensor | Optional | Must be passed when mask_mode=3; not passed when mask_mode=0. | int8 | ND | (2048, 2048) |
| metadata | Tensor | Required | Scheduling data generated by the companion metadata API. | int32 | ND | (max_schedule_size,) |
| head_dim_v | int | Optional | Value head dimension. Fixed to 512, defaults to 512. | int32 | - | - |
| softmax_scale | float | Optional | Scaling factor for attention scores, defaults to 1.0. | float32 | - | - |
| mask_mode | int | Optional | Only 0 or 3, defaults to 0. | int32 | - | - |
| max_seqlen_q | int | Optional | Only -1 or omitted, defaults to -1. | int32 | - | - |
| max_seqlen_kv | int | Optional | Only -1 or omitted, defaults to -1. | int32 | - | - |
| layout_q | string | Optional | Currently must be explicitly passed as TND. | string | - | - |
| layout_kv | string | Optional | Only PA_BBND or PA_NZ, defaults to PA_BBND. | string | - | - |
| layout_out | string | Optional | Currently must be explicitly passed as NTD. | string | - | - |
| return_softmax_lse | bool | Optional | Specifies whether to output valid LSE, defaults to False. | bool | - | - |

## Return Values

### flash_mla_with_kvcache_metadata

| Name | Type | Required/Optional | Description | data type | data format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | Required | Task splitting data of the main operator. | int32 | ND | (max_schedule_size,) |

The shape of metadata is dynamically calculated based on the number of batches and the number of device cores.

### flash_mla_with_kvcache

| Name | Type | Required/Optional | Description | data type | data format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | Required | Attention output, with the same data type as q | bfloat16/float16 | ND | (Q_N, Q_T, DV) |
| softmax_lse | Tensor | Optional | LSE result of Softmax, controlled by return_softmax_lse | float32 | ND | Enabled: (Q_N, Q_T); Disabled: (0,) |

**Note**

- attn_out: The D dimension is head_dim_v, and other dimensions are arranged by query according to layout_out.
- softmax_lse: When return_softmax_lse is True, the shape is (Q_N, Q_T); when it is False, the shape is (0,).

## Constraints

- The values in `block_table`, sequence lengths, `attn_mask`, and `metadata` are guaranteed to be correct by the caller. Tiling cannot fully validate device tensor content. Invalid values may cause accuracy errors or illegal memory access.
- The metadata API and the main operator must be consistent in sequence lengths, number of heads, head dimensions, mask modes, and Query layouts. Regenerate the corresponding metadata after the inputs change.

### Feature Parameter Group

| Feature Parameter Group | Parameter Field Name | Field Grouping | Field Type |
| :--- | :--- | :--- | :--- |
| Common Parameter Group | q, k_cache, metadata | INPUT | Tensor |
| | head_dim_v | ATTR(OPTIONAL) | int |
| | softmax_scale | ATTR(OPTIONAL) | float |
| | layout_q, layout_kv, layout_out | ATTR(OPTIONAL) | string |
| | attn_out | OUTPUT | Tensor |
| Mask Parameter Group | mask_mode | ATTR(OPTIONAL) | int |
| | attn_mask | INPUT(OPTIONAL) | Tensor |
| Paged Attention Parameter Group | block_table | INPUT | Tensor |
| SeqLens Parameter Group | cache_seqlens, cu_seqlens_q | INPUT | Tensor |
| | seqused_q | INPUT(OPTIONAL) | Tensor |
| | max_seqlen_q, max_seqlen_kv | ATTR(OPTIONAL) | int |
| SoftmaxLSE Parameter Group | return_softmax_lse | ATTR(OPTIONAL) | bool |
| | softmax_lse | OUTPUT | Tensor |

### Baseline Information

| Name | Meaning |
| :--- | :--- |
| B | Number of batches. |
| Q_N / N1 | Number of heads of the input q tensor. |
| KV_N / N2 | Number of heads of the input k_cache tensor. |
| Q_S / KV_S | Valid Query/KV length of a single batch. |
| Q_T / T | Sum of physical Query sequence lengths of all batches. |
| D | Logical head dimension of Query/Key. |
| DV | Head dimension of Value and attn_out. |
| num_blocks | Number of physical KV cache blocks. |
| max_num_blocks_per_seq | Block table capacity of each batch. |
| block_size | Token capacity of each physical block. |
| D0 | Innermost blocking dimension of the PA_NZ layout. |

### Parameter Group Constraints

#### Common Parameter Group

| Parameter | Single-Parameter Validation | Existence Validation | Consistency Validation | Feature Cross-Validation |
| :--- | :--- | :--- | :--- | :--- |
| q | float16/bfloat16; TND, (Q_T, Q_N, D) | Must Exist | Consistent with the data types of k_cache and attn_out. | 0 < B < 65536; Q_T > 0; Q_N supports 64 and 96; KV_N=1, D=576 |
| k_cache | float16/bfloat16; PA_BBND: (num_blocks, block_size, KV_N, D); PA_NZ: (num_blocks, KV_N, D / D0, block_size, D0); non-contiguous tensor: PA_BBND supports dim0 only, PA_NZ supports dim0 or dim1 | Must Exist | Logical D is the same as that of q. | block_size=128; D0=16 |
| attn_out | float16/bfloat16; NTD, (Q_N, Q_T, DV) | Must Exist | Q_T and Q_N correspond to those of q. | DV=head_dim_v=512 |
| metadata | int32, 1D, ND, non-empty; capacity no less than that required by the current device and B | Must Be Passed | Must correspond to the parameters of the current call. | Generated by the corresponding metadata API. |
| head_dim_v | 512 only | Optional, defaults to 512 | head_dim_v+64=576 | Value takes the first 512 dimensions of the cache. |
| softmax_scale | float, defaults to 1.0 | Optional | None. | No automatic scaling by D. |

layout matching relationship table:

| layout_q | layout_kv | layout_out | layout_softmax_lse |
| :--- | :--- | :--- | :--- |
| TND | PA_BBND, PA_NZ | NTD | (Q_N, Q_T) |

#### Mask Parameter Group

| Parameter | Single-Parameter Validation | Existence Validation | Consistency Validation | Feature Cross-Validation |
| :--- | :--- | :--- | :--- | :--- |
| mask_mode | Only 0 or 3 | Optional, defaults to 0. | Same as during metadata generation. | 0: no mask; 3: right-aligned causal mask. |
| attn_mask | int8, ND, (2048, 2048) | Not passed when mode=0; must be passed when mode=3. | 0 on and below the diagonal, 1 above the diagonal. | Interpreted as a right-aligned causal mask together with mode=3. |

#### Paged Attention Parameter Group

| Parameter | Single-Parameter Validation | Existence Validation | Consistency Validation | Feature Cross-Validation |
| :--- | :--- | :--- | :--- | :--- |
| block_table | int32, ND; shape is (B, max_num_blocks_per_seq); max_num_blocks_per_seq > 0; block_size=128 | Must be passed | B corresponds to cache_seqlens and cu_seqlens_q. | cache_seqlens must be passed; when layout_kv is PA_NZ, D must be a multiple of D0. |

#### SeqLengths Parameter Group

| Parameter | Single-Parameter Validation | Existence Validation | Consistency Validation | Feature Cross-Validation |
| :--- | :--- | :--- | :--- | :--- |
| cache_seqlens | int32, ND, (B,), non-negative elements | Must Be Passed | Consistent with values used for metadata generation | Consistent with the number of batches in block_table |
| cu_seqlens_q | int32, ND, (B+1,), non-decreasing, first element 0 and last element Q_T | Must Be Passed for TND use cases | B consistent with cache_seqlens | Adjacent differences indicate the physical Query length of each batch. |
| seqused_q | int32, ND, (B,), non-negative elements | Optional | Consistent with values used for metadata generation | Each element does not exceed the adjacent difference of the corresponding cu_seqlens_q; if omitted, adjacent differences are used. |
| max_seqlen_q | int | Optional | Only -1 allowed or omitted | Derived from the Query sequence length information; 0, positive values, or other negative values are not accepted. |
| max_seqlen_kv | int | Optional | Only -1 allowed or omitted | Derived from cache_seqlens; 0, positive values, or other negative values are not accepted. |

#### SoftmaxLSE Parameter Group

| Parameter | Single-Parameter Validation | Existence Validation | Consistency Validation | Feature Cross-Validation |
| :--- | :--- | :--- | :--- | :--- |
| return_softmax_lse | bool, True/False only | Optional, defaults to False | Controls whether the LSE output is valid. | Does not change the layout or shape of attn_out. |
| softmax_lse | float32 | Always occupies the second item of the returned tuple. | True: (Q_N, Q_T); False: (0,) | None. |

## Example

The following example uses a TND query, PA_NZ cache, NTD output, and a right-aligned causal mask, and omits the two maximum sequence length parameters.

```python
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)
device = "npu"
dtype = torch.bfloat16

q = torch.randn(8, 96, 576, dtype=dtype, device=device)
# Each of the two batches uses one 128-token physical block.
k_cache = torch.randn(2, 1, 36, 128, 16, dtype=dtype, device=device)
block_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
cache_seqlens = torch.tensor([128, 96], dtype=torch.int32, device=device)
cu_seqlens_q = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
attn_mask = torch.triu(
    torch.ones(2048, 2048, dtype=torch.int8, device=device), diagonal=1
)

metadata = cann_ops_transformer.ops.flash_mla_with_kvcache_metadata(
    cache_seqlens,
    num_heads_q=96,
    num_heads_kv=1,
    cu_seqlens_q=cu_seqlens_q,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=3,
    layout_q="TND",
)

attn_out, softmax_lse = cann_ops_transformer.ops.flash_mla_with_kvcache(
    q,
    k_cache,
    block_table=block_table,
    cache_seqlens=cache_seqlens,
    cu_seqlens_q=cu_seqlens_q,
    attn_mask=attn_mask,
    metadata=metadata,
    head_dim_v=512,
    softmax_scale=576 ** -0.5,
    mask_mode=3,
    layout_q="TND",
    layout_kv="PA_NZ",
    layout_out="NTD",
    return_softmax_lse=True,
)
torch_npu.npu.synchronize()
assert attn_out.shape == (96, 8, 512)
assert softmax_lse.shape == (96, 8)
assert attn_out.dtype == dtype
assert softmax_lse.dtype == torch.float32
```
