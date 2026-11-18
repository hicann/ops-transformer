# FlashKDA

## Supported Products

| Product | Supported |
| --- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

FlashKDA computes the forward pass of Kimi Delta Attention (KDA). It maintains the attention state with per-channel gated decay and delta updates. It supports variable-length sequences, grouped-query attention (GQA), fused Q/K L2 normalization, fused gate activation, and fused beta sigmoid.

For a single value head in one sequence, let $\bar q_t$ and $\bar k_t$ be the query and key used for computation, $\ell_t$ the per-token log gate, and $b_t$ the activated beta. The state $H_t$ is stored in $[D_v,D_k]$ order.

$$
\widetilde H_t=H_{t-1}\operatorname{diag}(\exp(\ell_t))
$$

$$
\Delta v_t=b_t(v_t-\widetilde H_t\bar k_t),\qquad
H_t=\widetilde H_t+\Delta v_t\bar k_t^T
$$

$$
o_t=scale\cdot H_t\bar q_t
$$

The initial state $H_0$ is zero when `initial_state` is omitted. When enabled, L2 normalization applies to the last dimension of Q and K. In GQA, each consecutive group of value heads shares a Q/K head.

The operator processes chunks of 64 tokens. The host selects a staged or persistent schedule using the effective sequence lengths, head counts, and available device cores. The recurrence uses FLOAT32 and matrix computation uses BFLOAT16. `flash_kda` returns three results, while `flash_kda_fwd` also provides caches used by the backward pass.

## Parameters

The PyTorch parameters are listed in the order defined in `torch_extension/flash_kda.py`. `B` is the number of sequences, `S` is the fixed storage length, and `T` is the total packed storage length. `Nqk` and `Nv` are the Q/K and V head counts, respectively. `Dk=Dv=128`. Tensor format is ND and the logical layout is selected by `layout`.

| Parameter | Role | Description | Data type | Format |
| --- | --- | --- | --- | --- |
| q | Input | Query tensor. Corresponds to $\bar q_t$ or Q before normalization. See the layout table below. | BFLOAT16 | ND |
| k | Input | Key tensor with the same shape as q. Corresponds to $\bar k_t$ or K before normalization. | BFLOAT16 | ND |
| v | Input | Value tensor corresponding to $v_t$. | BFLOAT16 | ND |
| g | Input | Gate input before activation when gate activation is fused. Otherwise, this is the per-token log gate $\ell_t$. | BFLOAT16, FLOAT32 | ND |
| beta | Input | Input before sigmoid when sigmoid is fused. Otherwise, this is the update coefficient $b_t$. | BFLOAT16, FLOAT32 | ND |
| metadata | Input | Required one-dimensional schedule tensor produced by `flash_kda_metadata`. | INT32 | ND |
| initial_state | Optional input | Initial state $H_0$ of shape `(B,Nv,128,128)`. Defaults to `None`, which uses a zero state. | FLOAT32 | ND |
| A_log | Optional input | Per-head gate parameter of shape `(Nv,)`. Defaults to `None`. Required with `use_gate_in_kernel=True`. | FLOAT32 | ND |
| dt_bias | Optional input | Gate bias of shape `(Nv,128)` or `(Nv*128,)`. Defaults to `None`, which uses zero bias. | FLOAT32 | ND |
| cu_seqlens | Optional input | Cumulative sequence lengths of shape `(B+1,)`. Defaults to `None`. Required for TND. | INT32 | ND |
| cu_seqlens_cpu | Optional input | Reserved parameter. Only the default value `None` is supported. | INT list | - |
| seqused | Optional input | Effective length of each sequence, of shape `(B,)`. Defaults to `None`, which uses the full stored sequence. | INT32 | ND |
| scale | Optional attribute | Output scale. Defaults to `128**(-0.5)`. Accepts a finite number or `None` to use the default. | FLOAT | - |
| output_final_state | Optional attribute | Whether to return the final state. Accepts `True` or `False`, and defaults to `False`. | BOOL | - |
| return_intermediate_states | Optional attribute | Whether to return chunk entry states during inference. Accepts `True` or `False`, and defaults to `False`. | BOOL | - |
| use_qk_l2norm_in_kernel | Optional attribute | Whether to fuse Q/K L2 normalization. Accepts `True` or `False`, and defaults to `True`. | BOOL | - |
| use_gate_in_kernel | Optional attribute | Whether to fuse gate activation. Accepts `True` or `False`, and defaults to `True`. | BOOL | - |
| use_beta_sigmoid_in_kernel | Optional attribute | Whether to fuse beta sigmoid. Accepts `True` or `False`, and defaults to `True`. | BOOL | - |
| allow_neg_eigval | Optional attribute | Whether to double the fused sigmoid result. Accepts `True` or `False`, and defaults to `False`. `True` requires fused beta sigmoid. | BOOL | - |
| disable_recompute | Optional attribute | Whether to save chunk entry states and backward caches during training. Accepts `True` or `False`, and defaults to `False`. | BOOL | - |
| lower_bound | Optional attribute | Lower bound for fused gate activation. Defaults to `None`. Accepts a finite value in `[-5,0)` when gate activation is fused. | FLOAT | - |
| layout | Optional attribute | Logical layout of Q/K/V. Accepts `BNSD`, `BSND`, or `TND`, and defaults to `TND`. | STRING | - |
| out | Output | Forward result $o_t$, of shape `(B,S,Nv,128)` for fixed-length layouts and `(T,Nv,128)` for TND. | BFLOAT16 | ND |
| final_state | Optional output | Final state of shape `(B,Nv,128,128)`. Returns `None` unless requested. | FLOAT32 | ND |
| chunk_states | Optional output | State at each chunk entry. The shape is `(B,C,Nv,128,128)` for fixed-length layouts and `(Ctotal,Nv,128,128)` for TND. See the API reference for return conditions. | BFLOAT16 | ND |

For fixed-length layouts, the sequence dimension of `g` and `beta` precedes the head dimension.

| Layout | q, k | v | g | beta |
| --- | --- | --- | --- | --- |
| BNSD | `(B,Nqk,S,128)` | `(B,Nv,S,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` |
| BSND | `(B,S,Nqk,128)` | `(B,S,Nv,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` |
| TND | `(T,Nqk,128)` | `(T,Nv,128)` | `(T,Nv,128)` | `(T,Nv)` |

## Constraints

- `B` and the applicable storage length `S` or `T` are positive. Head counts satisfy `1 <= Nqk <= Nv <= 96` and `Nv % Nqk == 0`. The Q/K and V head dimensions are both 128.
- Input tensors must be contiguous. Compute runs on an NPU. Meta tensors are used only for output specification inference.
- For TND, `cu_seqlens` starts at zero, is nondecreasing, and ends at `T`. In fixed-length layouts, when `cu_seqlens` is supplied, consecutive differences cannot exceed `S`. Each `seqused` value is nonnegative and is bounded by the corresponding sequence's storage length.
- Metadata must match the layout, storage dimensions, head counts, and effective lengths of the current call. Regenerate metadata when these change.
- `torch.is_inference_mode_enabled()` determines training or inference behavior. Use `torch.inference_mode()` for inference. `torch.no_grad()` still uses the training output contract. See the [PyTorch API reference](torch_extension/README_en.md) for `flash_kda_fwd` caches and return conditions.
- With gate fusion disabled, `g` provides the per-token log gate and the operator ignores `A_log`, `dt_bias`, and `lower_bound`. With beta sigmoid fusion disabled, `beta` provides the update coefficient directly.

## Invocation

| Interface | Example | Description |
| --- | --- | --- |
| PyTorch API | [Inference example](torch_extension/README_en.md#example) | Call [flash_kda, flash_kda_fwd, or flash_kda_metadata](torch_extension/README_en.md). |

See the [PyTorch extension guide](../../torch_extension/README.md) for build and installation instructions. This DSL implementation also requires compatible versions of CANN, torch_npu, and the CANNBot DSL runtime.

## References

- [Test cases](tests/ttk/kda_cases.py)
- [Performance measurement guide](tests/ttk/PERFORMANCE_en.md)
- [FlashKDAGrad backward operator](../flash_kda_grad/README_en.md)
