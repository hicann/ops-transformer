# fused_qkv_proj_norm_rope_cache

English | [简体中文](./torchapi_fused_qkv_proj_norm_rope_cache.md)

## Product Support

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

## Description

- **Interface function**: `cann_ops_transformer.fused_qkv_proj_norm_rope_cache` fuses QKV projection, K RMSNorm, Q/K RoPE, BF16 conversion, and KV cache scatter into a single Torch custom operator call. This interface directly calls the DSL implementation; QKV intermediate tensors stay on-chip and are not written back to GM. For complete computation semantics, scheduling, and precision details, see [FusedQkvProjNormRopeCache](../README_en.md).

- **Computation semantics**:

    Let the input activation be $X\in\mathbb{R}^{M\times2048}$ and the weight be $W\in\mathbb{R}^{N\times2048}$, where $N$ is 2048 or 2560. Then:

    $$
    Y = XW^{T}\in\mathbb{R}^{M\times N}
    $$

    When $N=2560$ (kv-mirror layer):

    $$
    q = \operatorname{RoPE}\left(Y[:,0:1536]\right),\quad
    k = \operatorname{RoPE}\left(\operatorname{RMSNorm}\left(Y[:,1536:1792]\right)\right)
    $$

    $$
    v = \operatorname{BF16}\left(Y[:,1792:2048]\right),\quad
    mirror\_k = \operatorname{BF16}\left(Y[:,2048:2304]\right),\quad
    mirror\_v = \operatorname{BF16}\left(Y[:,2304:2560]\right)
    $$

    When $N=2048$ (plain QKV layer), only $q$, $k$, and $v$ are produced; the Torch interface returns empty `mirror_k` and `mirror_v` tensors with shape `(0,256)`.

    K RMSNorm is:

    $$
    k_{norm} = \frac{Y_k}{\sqrt{\operatorname{mean}(Y_k^{2})+\epsilon}}\odot\gamma
    $$

    RoPE is rotate-half: the last 64 dimensions of each 256-dimensional head are split into $x_1=tail[0:32]$ and $x_2=tail[32:64]$, and the current row of `cos_sin_cache` is used as follows:

    $$
    x_1' = x_1\cos - x_2\sin,\qquad x_2' = x_1\sin + x_2\cos
    $$

    The first 32 columns of each `cos_sin_cache` row are cos, and the last 32 columns are sin. When `slot_mapping` is non-negative, $k$ and $v$ are written in place to the corresponding cache rows; tokens with value -1 skip the scatter.

## Function Prototype

```python
import cann_ops_transformer

cann_ops_transformer.fused_qkv_proj_norm_rope_cache(
    hidden: Tensor,
    weight: Tensor,
    k_gamma: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    slot_mapping: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    *,
    eps: float = 1e-6,
    positions_contiguous: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
```

You can also import the interface with `from cann_ops_transformer.ops import fused_qkv_proj_norm_rope_cache`. The registered name is `torch.ops.cann_ops_transformer.fused_qkv_proj_norm_rope_cache`. The Torch interface always returns five outputs. For `return_v=False` or explicit validation control, call the DSL entry in [op_dsl/fused_qkv_proj_norm_rope_cache.py](../op_dsl/fused_qkv_proj_norm_rope_cache.py) directly.

## Parameter Description

| Parameter | Type | Optional/Mandatory | Description | Data Type | Shape |
| :--- | :--- | :--- | :--- | :--- | :--- |
| hidden | Tensor | Mandatory | Projection input; must be contiguous. The data format is ND. | bfloat16 | (M, 2048) |
| weight | Tensor | Mandatory | `nn.Linear` `(out,in)` layout weight; must be contiguous. N supports only 2048 or 2560. The data format is ND. | bfloat16 | (N, 2048) |
| k_gamma | Tensor | Mandatory | K RMSNorm weight. The data format is ND. | bfloat16 | (256) |
| positions | Tensor | Mandatory | Current token positions; prefill requires contiguous positions, decode may be arbitrary. The data format is ND. | int32, int64 | (M) |
| cos_sin_cache | Tensor | Mandatory | RoPE table; the first 32 columns are cos and the last 32 columns are sin. The data format is ND. | float32 | (P, 64) |
| slot_mapping | Tensor | Mandatory | Cache row indices; -1 means skip the token. The data format is ND. | int32, int64 | (M) |
| k_cache | Tensor | Mandatory | Paged K cache, updated in place; the last dimension is 256 and it must be flattenable to (slots, 256). The data format is ND. | bfloat16 | (slots, 256) |
| v_cache | Tensor | Mandatory | Paged V cache, updated in place; same shape as `k_cache`. The data format is ND. | bfloat16 | (slots, 256) |
| eps | float | Optional | Numerical stability term for K RMSNorm; default is `1e-6`. | float | - |
| positions_contiguous | bool | Optional | Whether to use the prefill contiguous-position path; default is `True`. Set to `False` for the decode per-token path. | bool | - |

## Return Value Description

| Parameter | Type | Description | Data Type | Shape |
| :--- | :--- | :--- | :--- | :--- |
| q | Tensor | Q projection and RoPE result. The data format is ND. | bfloat16 | (M, 1536) |
| k | Tensor | K projection, RMSNorm, and RoPE result. The data format is ND. | bfloat16 | (M, 256) |
| v | Tensor | V projection result. The Torch interface always returns this output. The data format is ND. | bfloat16 | (M, 256) |
| mirror_k | Tensor | mirror K output; `(M,256)` when N=2560 and an empty `(0,256)` tensor when N=2048. The data format is ND. | bfloat16 | (M, 256) or (0, 256) |
| mirror_v | Tensor | mirror V output; `(M,256)` when N=2560 and an empty `(0,256)` tensor when N=2048. The data format is ND. | bfloat16 | (M, 256) or (0, 256) |

`k_cache` and `v_cache` are updated in place and are not included in the return values.

## Constraints

- This interface supports training and inference scenarios.
- This interface supports single-operator mode and torch.compile (npugraph_ex) graph mode.
- During graph capture (ACL Graph), parameter checks that read device values back (`positions` contiguity/range, `slot_mapping` range and uniqueness) are skipped to avoid synchronizing the capturing stream. The caller must keep the indices valid for both capture and replay. Single-operator mode still performs all checks.
- Supported only on Ascend 950PR/950DT (ARCH 3510).
- `M >= 1`; `hidden`, `weight`, `k_gamma`, and `cos_sin_cache` must be contiguous and reside on the same NPU.
- The N dimension of `weight` supports only 2048 (plain QKV layer) or 2560 (kv-mirror layer).
- `k_cache` and `v_cache` must be contiguous, their last dimension must be 256, and they must be flattenable to `(slots,256)`; both must have the same shape.
- When `positions_contiguous=True` and `M > 1`, `positions` must be strictly contiguous; the value range is `[0, P-1]`.
- `slot_mapping` values must be in `[-1, slots-1]`; -1 skips the scatter, and non-negative slots must be unique. Duplicate slots are rejected.
- Only the BF16 weight path is covered. The modelslim-mxfp8 weight and E8M0 scale path is not included.
- The current implementation provides forward only, not backward.

## Determinism/Batch Consistency

- Determinism: Deterministic computation is supported by default.
- Batch consistency: Batch consistency is supported by default.

## Example

- Single-operator mode

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import fused_qkv_proj_norm_rope_cache

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    tokens, units, max_pos, slots = 72, 10, 4096, 1024

    hidden = (torch.randn(tokens, 2048, device=device) * 0.1).bfloat16()
    weight = (torch.randn(units * 256, 2048, device=device) * 0.05).bfloat16()
    k_gamma = (torch.randn(256, device=device) * 0.1 + 1.0).bfloat16()
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    cos_sin_cache = torch.randn(max_pos, 64, dtype=torch.float32, device=device)
    slot_mapping = torch.arange(tokens, dtype=torch.int64, device=device)
    k_cache = torch.zeros(slots, 256, dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros(slots, 256, dtype=torch.bfloat16, device=device)

    q, k, v, mirror_k, mirror_v = fused_qkv_proj_norm_rope_cache(
        hidden, weight, k_gamma, positions, cos_sin_cache, slot_mapping,
        k_cache, v_cache, eps=1e-6, positions_contiguous=True,
    )
    print(q.shape, k.shape, v.shape, mirror_k.shape, mirror_v.shape)
    ```

- Graph mode

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import fused_qkv_proj_norm_rope_cache

    class QkvModel(torch.nn.Module):
        def forward(self, hidden, weight, k_gamma, positions, cos_sin_cache,
                    slot_mapping, k_cache, v_cache):
            return fused_qkv_proj_norm_rope_cache(
                hidden, weight, k_gamma, positions, cos_sin_cache, slot_mapping,
                k_cache, v_cache, eps=1e-6, positions_contiguous=True,
            )

    model = torch.compile(
        QkvModel().eval(), backend="npugraph_ex", fullgraph=True, dynamic=False,
    )
    out = model(hidden, weight, k_gamma, positions, cos_sin_cache,
                slot_mapping, k_cache, v_cache)
    ```
