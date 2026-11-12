# stem\_oam\_prep\_paged\_kv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:38:08.548Z -->

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
<!-- npu="310p" id8 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id8 -->
<!-- npu="910" id9 -->
- <term>Atlas training products</term>: Not supported
<!-- end id9 -->

## Function

- **Description**:

  `stem_oam_prep_paged_kv` is a `cann_ops_transformer` extension API based on `torch_npu`. It calls the `StemOamPrepPagedKv` operator to perform the pre-scoring computation of the Stem OAM (Output-Aware Metric) dynamic sparse attention mechanism. The operator extracts K/V data from the Paged KV Cache, computes it through K Processing (per-token K-scale × group sum + anti-diagonal flip) and V Processing (per-head vScale × L2 Norm → Log → Global Normalize → ReLU → Block Average), and outputs kFlat and vBias for Stem OAM score computation.

- **The inputs and outputs support the following data use cases**:

    ```text
    k_cache: [total_blocks, kv_block_size, H_kv, 128] or [total_blocks, H_kv, kv_block_size, 128]
    v_cache: [total_blocks, kv_block_size, H_kv, 128] or [total_blocks, H_kv, kv_block_size, 128]
    kv_indices: [batch, max_kv_blocks]
    kv_seq_lens: [batch]
    k_scale_cache: [total_blocks, kv_block_size, H_kv, 1] or [total_blocks, H_kv, kv_block_size, 1]
    v_scale: [H_kv]
    k_flat: [batch, H_kv, max_Kb, stem_stride * 128]
    v_bias: [batch, H_kv, max_Kb]
    ```

- k_cache/v_cache supports two layouts, specified by the `kv_layout` attribute: "BBND" indicates [total_blocks, kv_block_size, H_kv, 128], and "BNBD" indicates [total_blocks, H_kv, kv_block_size, 128]. Currently, only "BNBD" is supported.
- The k_scale_cache layout changes with `kv_layout` and is consistent with the layout of k_cache/v_cache (with only the last dimension D=1 instead of D=128).
- Only the FP8_E4M3FN input path is supported.

## Function Prototype

```python
cann_ops_transformer.stem_oam_prep_paged_kv(
    k_cache,
    v_cache,
    kv_indices,
    kv_seq_lens,
    k_scale_cache,
    v_scale,
    lambda_mag=0.3,
    kv_layout="BNBD",
    stem_block_size=128,
    stem_stride=16
) -> Tuple[Tensor, Tensor]
```

## Parameters

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k_cache | Tensor | Required | Paged K cache. Does not support empty tensors. When kv_layout="BBND", the shape is [total_blocks, kv_block_size, H_kv, 128]; when kv_layout="BNBD", the shape is [total_blocks, H_kv, kv_block_size, 128]. The first three dimensions can be non-contiguous (stride>shape), the last dimension must be contiguous, and the maximum value of H_kv is 8. | float8_e4m3fn | 4 |
| v_cache | Tensor | Required | Paged V cache. Does not support empty tensors. The shape is the same as that of k_cache. | Same as k_cache | 4 |
| kv_indices | Tensor | Required | Array of KV block indices for each batch. Does not support empty tensors. shape[1]=max_kv_blocks. The maximum value of batch is 16, and the maximum value of max_kv_blocks is 2048. | int32 | 2 |
| kv_seq_lens | list[int] | Required | KV sequence length for each batch. Does not support empty lists. The maximum value of the KV sequence length is 262144. This value is used to derive max_kv_blocks (the second dimension of kv_indices) and max_Kb in the output shape. | int32 | 1 |
| k_scale_cache | Tensor | Optional | Per-token per-head K scale. Required when the data type of k_cache is FP8, and can be omitted for other types. Changes with kv_layout: kv_layout="BBND": [total_blocks, kv_block_size, H_kv, 1]; kv_layout="BNBD": [total_blocks, H_kv, kv_block_size, 1]. The first three dimensions can be non-contiguous (stride>shape), and the last dimension must be contiguous. | float32 | 4 |
| v_scale | Tensor | Optional | Per-head V scale. Required when the data type of k_cache is FP8, and can be omitted for other types. | float32 | 1 ([H_kv]) |
| lambda_mag | float | Optional | V bias multiplier. Value range: [0,1]. Default: 0.3. | float | - |
| kv_layout | str | Optional | KV cache layout. "BBND" indicates [total_blocks, kv_block_size, H_kv, 128], and "BNBD" indicates [total_blocks, H_kv, kv_block_size, 128]. Currently, only "BNBD" is supported. Default: "BNBD". | str | - |
| stem_block_size | int | Optional | Stem block size. %32==0 and ≤256. Default: 128. | int | - |
| stem_stride | int | Optional | Stride size. %16==0, ≤64, and ≤stem_block_size, and stem_block_size must be an integer multiple of stem_stride. Default: 16. | int | - |

## Return Values

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k_flat | Tensor | Required | K group sum + anti-diag flip result. Does not support empty tensor. The shape is [batch, H_kv, max_Kb, kflat_dim], where kflat_dim=stem_stride×D, and max_Kb is calculated based on the kv_seq_lens input. | bfloat16 | 4 |
| v_bias | Tensor | Required | V block bias result. Does not support empty tensor. The shape is [batch, H_kv, max_Kb], and max_Kb is calculated based on the kv_seq_lens input. | float32 | 3 |

## Constraints

- The data type of k_cache/v_cache must be float8_e4m3fn.
- The data type of k_scale_cache/v_scale must be float32. Required when the data type of k_cache is FP8; can be omitted for other types.
- The shape of k_scale_cache changes with kv_layout: "BBND" is `[total_blocks, kv_block_size, H_kv, 1]`, and "BNBD" is `[total_blocks, H_kv, kv_block_size, 1]`. Currently only supports "BNBD".
- v_scale shape: `[H_kv]`.
- kv_block_size is derived from the k_cache shape: k_cache shape[1] for the BBND layout and k_cache shape[2] for the BNBD layout, with values of {64, 128}.
- kv_indices shape[1]=max_kv_blocks; the maximum value of max_kv_blocks is 2048.
- kv_seq_lens: the maximum KV sequence length is 262144.
- stem_block_size % 32 == 0 and ≤256; stem_stride % 16 == 0 and ≤64, stem_stride ≤ stem_block_size, and stem_block_size must be an integer multiple of stem_stride.
- Derived values: R = stem_block_size / stem_stride, kflat_dim = stem_stride × 128, k_down_len = num_Kb × R.
- Boundaries: When kv_seq_lens[b]=0, the k_flat/v_bias outputs of the corresponding batch are all zeros; k_scale_cache padding rows (beyond kv_len) → zero.
- Supports only the arch35 architecture (Ascend 950PR&950DT products).

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Call in single-operator mode

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    # Define shapes.
    batch = 1
    total_blocks = 8
    kv_block_size = 64
    num_kv_heads = 4
    dim_qk = 128
    max_kv_blocks = 2
    stem_block_size = 128
    stem_stride = 16
    kv_layout = "BNBD"

    # Construct k_cache/v_cache (FP8_E4M3FN).
    k_cache_shape = (total_blocks, num_kv_heads, kv_block_size, dim_qk)
    k_cache = torch.randn(k_cache_shape, dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)
    v_cache = torch.randn(k_cache_shape, dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)

    # Construct kv_indices and kv_seq_lens.
    kv_indices = torch.randint(0, total_blocks, (batch, max_kv_blocks), dtype=torch.int32, device="npu")
    kv_seq_lens = (128,)

    # Construct k_scale_cache and v_scale.
    k_scale_cache_shape = (total_blocks, num_kv_heads, kv_block_size, 1)
    k_scale_cache = torch.randn(k_scale_cache_shape, dtype=torch.float32, device="npu")
    v_scale = torch.randn(num_kv_heads, dtype=torch.float32, device="npu")

    # Call the operator.
    k_flat, v_bias = cann_ops_transformer.stem_oam_prep_paged_kv(
        k_cache,
        v_cache,
        kv_indices,
        kv_seq_lens,
        k_scale_cache=k_scale_cache,
        v_scale=v_scale,
        lambda_mag=0.3,
        kv_layout=kv_layout,
        stem_block_size=stem_block_size,
        stem_stride=stem_stride,
    )

    torch_npu.npu.synchronize()
    print(k_flat.shape, k_flat.dtype)
    print(v_bias.shape, v_bias.dtype)
    ```

- Call in graph mode (torchair):

    ```python
    import torch
    import torch_npu
    import torch.nn as nn
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    # Define shapes.
    batch = 1
    total_blocks = 8
    kv_block_size = 64
    num_kv_heads = 4
    dim_qk = 128
    max_kv_blocks = 2
    stem_block_size = 128
    stem_stride = 16
    kv_layout = "BNBD"

    # Construct k_cache/v_cache (FP8_E4M3FN).
    k_cache_shape = (total_blocks, num_kv_heads, kv_block_size, dim_qk)
    k_cache = torch.randn(k_cache_shape, dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)
    v_cache = torch.randn(k_cache_shape, dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)

    # Construct kv_indices and kv_seq_lens.
    kv_indices = torch.randint(0, total_blocks, (batch, max_kv_blocks), dtype=torch.int32, device="npu")
    kv_seq_lens = (128,)

    # Construct k_scale_cache and v_scale.
    k_scale_cache_shape = (total_blocks, num_kv_heads, kv_block_size, 1)
    k_scale_cache = torch.randn(k_scale_cache_shape, dtype=torch.float32, device="npu")
    v_scale = torch.randn(num_kv_heads, dtype=torch.float32, device="npu")

    class StemOamPrepPagedKvNetwork(nn.Module):
        def __init__(self):
            super(StemOamPrepPagedKvNetwork, self).__init__()

        @torch._dynamo.disable
        def forward(self, k_cache, v_cache, kv_indices, k_scale_cache, v_scale,
                    kv_seq_lens, lambda_mag, kv_layout,
                    stem_block_size, stem_stride):
            return torch.ops.cann_ops_transformer.stem_oam_prep_paged_kv(
                k_cache, v_cache, kv_indices, kv_seq_lens,
                k_scale_cache=k_scale_cache, v_scale=v_scale,
                lambda_mag=lambda_mag, kv_layout=kv_layout,
                stem_block_size=stem_block_size, stem_stride=stem_stride)

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(StemOamPrepPagedKvNetwork(), backend=npu_backend, dynamic=False)

    k_flat, v_bias = npu_mode(
        k_cache, v_cache, kv_indices, k_scale_cache, v_scale,
        kv_seq_lens, 0.3, kv_layout,
        stem_block_size, stem_stride,
    )

    print(k_flat.shape, k_flat.dtype)
    print(v_bias.shape, v_bias.dtype)
    ```
