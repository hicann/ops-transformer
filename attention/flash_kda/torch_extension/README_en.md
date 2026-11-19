# FlashKDA PyTorch API

## Overview

`flash_kda_metadata` generates scheduling metadata. `flash_kda` returns the forward output, final state, and chunk entry states. `flash_kda_fwd` also returns the caches required by the backward pass. See [FlashKDA](../README_en.md) for input types, shapes, and constraints.

## Signatures

```python
def flash_kda(
    q, k, v, g, beta, metadata,
    initial_state=None, A_log=None, dt_bias=None,
    cu_seqlens=None, cu_seqlens_cpu=None, seqused=None, *,
    scale=128**(-0.5), output_final_state=False,
    return_intermediate_states=False, use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False, disable_recompute=False,
    lower_bound=None, layout="TND",
):
    ...
```

`flash_kda_fwd` takes the same arguments and uses the same defaults.

```python
def flash_kda_metadata(
    cu_seqlens=None, seqused=None, *,
    batch_size, nheads_qk, nheads_v, seqlens,
    headdim_qk=128, headdim_v=128, enable_grad=False, layout="TND",
):
    ...
```

`batch_size`, `nheads_qk`, and `nheads_v` correspond to `B`, `Nqk`, and `Nv`. Set `seqlens` to the fixed storage length `S` or the total packed storage length `T`. Dimension parameters must be positive integers within the INT32 range, and both head dimensions must be 128. Use the same `cu_seqlens`, `seqused`, and `layout` for metadata generation and the forward call. Set `enable_grad` to `not torch.is_inference_mode_enabled()`. The API creates a one-dimensional INT32 tensor on the current NPU. Its length depends on the implementation. Pass the tensor directly to the forward call without modifying its contents.

## Return Values

`flash_kda` always returns `(out, final_state, chunk_states)`. `flash_kda_fwd` returns a tuple of 16 entries. Entries that are not requested or applicable are set to `None`.

| Index | Name | Data type | Return condition |
| --- | --- | --- | --- |
| 0 | out | BFLOAT16 | Always returned. |
| 1 | final_state | FLOAT32 | `output_final_state=True`. |
| 2 | chunk_states | BFLOAT16 | Inference with `return_intermediate_states=True`, or training with `disable_recompute=True`. |
| 3 | Aqk | BFLOAT16 | Training. Causal QK matrix within each chunk, including the scale factor. |
| 4 | Akk | BFLOAT16 | Training. Inverse matrix within each chunk. |
| 5 | w | BFLOAT16 | Training with `disable_recompute=True`. |
| 6 | u | BFLOAT16 | Training with `disable_recompute=True`. |
| 7 | qg | BFLOAT16 | Training with `disable_recompute=True`. |
| 8 | kg | BFLOAT16 | Training with `disable_recompute=True`. |
| 9 | v_new | BFLOAT16 | Training with `disable_recompute=True`. |
| 10 | g_cumsum | FLOAT32 | Training with `disable_recompute=True`. Cumulative log gate within each chunk. |
| 11 | q_norm | BFLOAT16 | Training with `use_qk_l2norm_in_kernel=True`. |
| 12 | q_rstd | FLOAT32 | Training with `use_qk_l2norm_in_kernel=True`. Inverse norm used to normalize Q. |
| 13 | k_norm | BFLOAT16 | Training with `use_qk_l2norm_in_kernel=True`. |
| 14 | k_rstd | FLOAT32 | Training with `use_qk_l2norm_in_kernel=True`. Inverse norm used to normalize K. |
| 15 | beta_activated | FLOAT32 | Training with `use_beta_sigmoid_in_kernel=True`. |

For fixed-length layouts, `C=ceil(S/64)`. In TND, `Ctotal` is the allocated chunk state capacity, including chunk alignment. Read only the states corresponding to valid sequences.

| Output | BNSD or BSND input | TND input |
| --- | --- | --- |
| out | `(B,S,Nv,128)` | `(T,Nv,128)` |
| final_state | `(B,Nv,128,128)` | `(B,Nv,128,128)` |
| chunk_states | `(B,C,Nv,128,128)` | `(Ctotal,Nv,128,128)` |
| Aqk, Akk | `(B,Nv,S,64)` | `(Nv,T,64)` |
| w, u, qg, kg, v_new, g_cumsum | `(B,Nv,S,128)` | `(Nv,T,128)` |
| q_norm, k_norm | Same as q and k | Same as q and k |
| q_rstd, k_rstd | `(B,Nqk,S)` | `(Nqk,T)` |
| beta_activated | `(B,Nv,S)` | `(Nv,T)` |

The training output contract remains active under `torch.no_grad()` and is independent of the inputs' `requires_grad` flags. Use `flash_kda_fwd` when you need caches for the [FlashKDAGrad backward API](../../flash_kda_grad/README_en.md).

## Example

Install compatible versions of CANN, PyTorch, torch_npu, and CANNBot DSL in a Linux NPU environment. Follow the [extension build instructions](../../../torch_extension/README_en.md) to build and install the package. The example imports the dedicated `cann_ops_transformer_flash_kda` package built with `--ops=flash_kda --vendor_name=flash_kda`. Include the native build artifacts for your device in the package.

```python
import torch
import torch_npu
import cann_ops_transformer_flash_kda as ops

torch.npu.set_device(0)
batch, length, n_qk, n_v, dim = 1, 65, 2, 4, 128

# For BSND, g and beta use (B,S,Nv,D) and (B,S,Nv), respectively.
q = torch.randn(batch, length, n_qk, dim, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(batch, length, n_v, dim, device="npu", dtype=torch.bfloat16)
g = torch.randn(batch, length, n_v, dim, device="npu", dtype=torch.float32)
beta = torch.randn(batch, length, n_v, device="npu", dtype=torch.float32)
A_log = torch.zeros(n_v, device="npu", dtype=torch.float32)

# Use the same storage layout and execution mode for both calls.
with torch.inference_mode():
    metadata = ops.flash_kda_metadata(
        batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
        seqlens=length, enable_grad=False, layout="BSND",
    )
    out, final_state, chunk_states = ops.flash_kda(
        q, k, v, g, beta, metadata, A_log=A_log,
        layout="BSND", output_final_state=True,
        return_intermediate_states=True,
    )

torch.npu.synchronize()
assert out.shape == (batch, length, n_v, dim)
assert final_state.shape == (batch, n_v, dim, dim)
assert chunk_states.shape == (batch, 2, n_v, dim, dim)
```
