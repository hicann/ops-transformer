# FusedQkvProjNormRopeCache

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| ---- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

- Operator function: A CANNBot-DSL based fully fused WeLMv4 QKV projection operator. A single `MIX_AIC_1_2` composite core performs QKV projection, K RMSNorm, Q/K RoPE, BF16 conversion, and KV cache scatter. QKV intermediate tensors stay on-chip and are not written back to GM.

- Calculation formula:

    Let the input activation be $X\in\mathbb{R}^{M\times2048}$ and the weight be $W\in\mathbb{R}^{N\times2048}$, where $N$ is 2048 or 2560. Then:

    $$
    Y = XW^{T}\in\mathbb{R}^{M\times N}
    $$

    When $N=2560$ (kv-mirror layer), split by head unit:

    $$
    q = \operatorname{RoPE}\left(Y[:,0:1536]\right),\quad
    k = \operatorname{RoPE}\left(\operatorname{RMSNorm}\left(Y[:,1536:1792]\right)\right)
    $$

    $$
    v = \operatorname{BF16}\left(Y[:,1792:2048]\right),\quad
    mirror\_k = \operatorname{BF16}\left(Y[:,2048:2304]\right),\quad
    mirror\_v = \operatorname{BF16}\left(Y[:,2304:2560]\right)
    $$

    When $N=2048$ (plain QKV layer), only $q$, $k$, and $v$ are produced; no mirror outputs are generated.

    K RMSNorm is:

    $$
    k_{norm} = \frac{Y_k}{\sqrt{\operatorname{mean}(Y_k^{2})+\epsilon}}\odot\gamma
    $$

    RoPE is rotate-half: the last 64 dimensions of each 256-dimensional head are split into $x_1=tail[0:32]$ and $x_2=tail[32:64]$, and the current row of `cos_sin_cache` is used as follows:

    $$
    x_1' = x_1\cos - x_2\sin,\qquad x_2' = x_1\sin + x_2\cos
    $$

    The first 32 columns of each `cos_sin_cache` row are cos, and the last 32 columns are sin. When `slot_mapping` is non-negative, $k$ and $v$ are written in place to the corresponding cache rows; tokens with value -1 skip the scatter.

- Implementation notes:

    The weight is split into $N/256$ N tiles by head unit. Cube processes rows in `PAIR_M=256` pairs, so a pair of 128-row tiles shares one A load and one B slice, reducing L1 traffic. FP32 results in L0C are delivered by Fixpipe as two half-frames directly to the UB of the paired AIVs. The vector stage performs RoPE, RMSNorm, BF16 conversion, and scatter in place. The launch core count is derived as `min(ceil(M/256)*N/256, available AIC groups on the current stream)`; a stream quota change triggers an independent compile and recompile.

## Parameter Description

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| hidden | Input | Projection input with shape `[M,2048]`; must be contiguous. | BFLOAT16 | ND |
| weight | Input | `nn.Linear` `(out,in)` layout with shape `[N,2048]`; `N` supports only 2048 or 2560. | BFLOAT16 | ND |
| k_gamma | Input | K RMSNorm weight with shape `[256]`. | BFLOAT16 | ND |
| positions | Input | Current token positions with shape `[M]`; prefill requires contiguous positions, decode may be arbitrary. | INT32, INT64 | ND |
| cos_sin_cache | Input | RoPE table with shape `[P,64]`; the first 32 columns are cos and the last 32 columns are sin. | FLOAT32 | ND |
| slot_mapping | Input | Cache row indices with shape `[M]`; -1 means skip the token. | INT32, INT64 | ND |
| k_cache | Input/Output | Paged K cache, updated in place; the last dimension is 256 and the tensor must be flattenable to `[slots,256]`. | BFLOAT16 | ND |
| v_cache | Input/Output | Paged V cache, updated in place; same shape as `k_cache`. | BFLOAT16 | ND |
| eps | Attribute | Numerical stability term for K RMSNorm; default is `1e-6`. | FLOAT | - |
| positions_contiguous | Attribute | Whether to use the prefill contiguous-position path; default is `True`. When `False`, the decode per-token path is used. | BOOL | - |

## Constraints

- During ACL Graph capture, parameter checks that read device values back (`positions` contiguity/range, `slot_mapping` range and uniqueness) are skipped; the caller must keep the indices valid for capture and replay.
- Supported only on Ascend 950PR/950DT (ARCH 3510).
- `M >= 1`; `hidden`, `weight`, `k_gamma`, and `cos_sin_cache` must be contiguous and reside on the same NPU.
- The N dimension of `weight` supports only 2048 (plain QKV layer) or 2560 (kv-mirror layer).
- `k_cache` and `v_cache` must be contiguous, their last dimension must be 256, and they must be flattenable to `[slots,256]`; both must have the same shape.
- When `positions_contiguous=True` and `M > 1`, `positions` must be strictly contiguous; the value range is `[0, P-1]`.
- `slot_mapping` values must be in `[-1, slots-1]`; -1 skips the scatter, and non-negative slots must be unique. Duplicate slots are rejected.
- Only the BF16 weight path is covered. The modelslim-mxfp8 weight and E8M0 scale path is not included.
- The current implementation provides forward only, not backward.

## Calling Instructions

| Calling Method | Sample Code | Description |
| :--- | :--- | :--- |
| PyTorch API | [torchapi_fused_qkv_proj_norm_rope_cache_en.md](docs/torchapi_fused_qkv_proj_norm_rope_cache_en.md) | Calls through `cann_ops_transformer.fused_qkv_proj_norm_rope_cache` and always returns five outputs `(q,k,v,mirror_k,mirror_v)`. |
| Direct DSL call | [op_dsl/fused_qkv_proj_norm_rope_cache.py](op_dsl/fused_qkv_proj_norm_rope_cache.py) | Adds `experimental/attention/fused_qkv_proj_norm_rope_cache/op_dsl` to `PYTHONPATH`; supports `return_v`, `validate`, and `positions_contiguous=None` auto-detection. |
| Graph mode | [torchapi_fused_qkv_proj_norm_rope_cache_en.md](docs/torchapi_fused_qkv_proj_norm_rope_cache_en.md) | Calls through the `torch.compile` / `npugraph_ex` backend. |
| Unit test | [tests/pytest](tests/pytest) | 51 NPU precision cases. |

### PyTorch API Example

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

### Graph Mode

```python
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

### Precision Validation

`tests/pytest/test_fused_qkv_proj_norm_rope_cache_dsl.py` uses a CPU FP32 golden and contains 51 cases covering:

- prefill contiguous positions M=1/2/7/8/16/31/33/63/64/65/72/100/127/128/200/255/256/257/300/511/512/513/1000
- decode arbitrary positions, non-zero position start, and automatic position mode
- 8/10 unit layers (N=2048/2560)
- scatter skipping with strides 2/3/5, `eps=1e-1/1e-3/1e-5`, and `return_v=False`
- small cache, paged 3D cache, `int32` indices, and small/large cos/sin tables
- stream AIC/AIV quota change (32→8) triggering independent compilation and correct recompilation
- Meta/eager rank, shape, and dtype consistency for the 8-unit layer through the official Torch registration entry

```bash
source ${install_path}/ascend-toolkit/set_env.sh
cd experimental/attention/fused_qkv_proj_norm_rope_cache/tests/pytest
python -m pytest -q test_fused_qkv_proj_norm_rope_cache_dsl.py
```

## References

- [PyTorch API documentation (Chinese)](docs/torchapi_fused_qkv_proj_norm_rope_cache.md)
- [PyTorch API documentation (English)](docs/torchapi_fused_qkv_proj_norm_rope_cache_en.md)
- [Operator unit test](tests/pytest/test_fused_qkv_proj_norm_rope_cache_dsl.py)
