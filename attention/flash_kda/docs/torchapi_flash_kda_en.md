# flash_kda

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

## Function Description

- Computes the forward pass of Kimi Delta Attention (KDA), maintaining an attention state through channel-wise gated decay and delta updates. Supports training, inference, variable sequence lengths, grouped query attention (GQA), and fused activations.
  - First call `cann_ops_transformer.flash_kda_metadata()` to generate scheduling data from storage dimensions, valid sequence lengths, and execution mode, then pass the result to `cann_ops_transformer.flash_kda()`.
  - `flash_kda` returns the output, optional final state, and optional chunk entry states. During training, PyTorch autograd invokes the companion [FlashKDAGrad](../../flash_kda_grad/docs/torchapi_flash_kda_grad.md) operator.
- For one sequence and one value head, let $q_t$, $k_t$, and $v_t$ denote query, key, and value. With `use_qk_l2norm_in_kernel=True`, $\bar q_t$ and $\bar k_t$ are Q/K normalized along the last dimension; otherwise they are the original Q/K. Let $x_t=g_t+dt\_bias$, using zero bias when omitted. The per-step log gate is:

  $$
  \ell_t=\begin{cases}
  g_t, & \text{gate fusion disabled} \\
  -\exp(A_{log})\operatorname{softplus}(x_t), & \text{gate fusion enabled, lower\_bound is None} \\
  lower\_bound\cdot\sigma(\exp(A_{log})x_t), & \text{gate fusion enabled, lower\_bound specified}
  \end{cases}
  $$

  Here $\sigma$ is sigmoid. The update coefficient $b_t$ is `beta` without sigmoid fusion, $\sigma(beta_t)$ with fusion, or $2\sigma(beta_t)$ when `allow_neg_eigval=True` as well. State $H_t$ uses the $[D_v,D_k]$ layout:

  $$
  \widetilde H_t=H_{t-1}\operatorname{diag}(\exp(\ell_t)),\qquad
  \Delta v_t=b_t(v_t-\widetilde H_t\bar k_t)
  $$

  $$
  H_t=\widetilde H_t+\Delta v_t\bar k_t^T,\qquad
  o_t=scale\cdot H_t\bar q_t
  $$

  `initial_state` supplies $H_0$, or zero when omitted. GQA maps each consecutive group of `Nv/Nqk` value heads to one Q/K head. The implementation processes chunks of 64 tokens, with float32 state recurrence and bfloat16 matrix operations.

## Prototype

The prototype and examples use `import cann_ops_transformer`. Generate the scheduling data before calling `cann_ops_transformer.flash_kda`, and pass it unchanged as `metadata`. The spelling `A_log` follows the actual software interface.

First call `flash_kda_metadata`. Pass `batch_size` and all subsequent parameters by keyword; `batch_size`, `nheads_qk`, `nheads_v`, and `seqlens` are required and have no defaults. The same-named variables below represent caller-provided values.

```python
cann_ops_transformer.flash_kda_metadata(
    cu_seqlens=None,
    seqused=None,
    batch_size=batch_size,
    nheads_qk=nheads_qk,
    nheads_v=nheads_v,
    seqlens=seqlens,
    headdim_qk=128,
    headdim_v=128,
    enable_grad=False,
    layout="TND"
)
```

`q`, `k`, `v`, `g`, `beta`, and `metadata` are required. Pass `scale` and all subsequent parameters by keyword.

```python
cann_ops_transformer.flash_kda(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    seqused=None,
    scale=0.08838834764831845,
    output_final_state=False,
    return_intermediate_states=False,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=False,
    lower_bound=None,
    layout="TND"
)
```

## Parameters

`B` is the number of sequences, `S` the dense storage length, `T` the total packed storage length, and `Nqk`/`Nv` the Q/K and V head counts. `Dk=Dv=128`. Tensor storage format is ND.

### flash_kda_metadata

These scheduling parameters must match the subsequent `flash_kda` call. Explicitly provide `batch_size`, `nheads_qk`, `nheads_v`, and `seqlens`; they are not inferred from the length tensors. Integer arguments must be positive and fit in int32.

| Name | Parameter type | Required/Optional | Description | Data type | Data format | Shape |
| --- | --- | --- | --- | --- | --- | --- |
| cu_seqlens | Tensor | Optional | Cumulative sequence lengths. Default: `None`. Required for TND: starts at zero, strictly increases, and ends at `seqlens`. For dense layouts, supplied adjacent differences must be positive and at most `seqlens`. | int32 | ND | `(B+1,)` |
| seqused | Tensor | Optional | Valid sequence lengths. Default: `None`, using adjacent differences from `cu_seqlens`. If both tensors are omitted, each dense sequence uses `seqlens`. If both are supplied, entries must equal the corresponding adjacent differences. | int32 | ND | `(B,)` |
| batch_size | int | Required | Number of sequences `B`, with no default. The Q/K/V batch dimension for dense layouts; `cu_seqlens.numel()-1` for TND. | - | - | - |
| nheads_qk | int | Required | Q/K head count `Nqk`, with no default. Must satisfy `1 <= Nqk <= Nv`. | - | - | - |
| nheads_v | int | Required | V head count `Nv`, with no default. Must satisfy `Nv <= 96` and `Nv % Nqk == 0`. | - | - | - |
| seqlens | int | Required | Storage length, with no default. Pass dense storage length `S` for BNSD/BSND or total packed storage length `T` for TND. Do not substitute the maximum valid sequence length for the storage length. | - | - | - |
| headdim_qk | int | Optional | Q/K head dimension `Dk`. Default: 128; only 128 is supported. | - | - | - |
| headdim_v | int | Optional | V head dimension `Dv`. Default: 128; only 128 is supported. | - | - | - |
| enable_grad | bool | Optional | Generates scheduling data required for training. Default: `False`. Use `False` for inference and `True` for training; `True` includes both forward and backward scheduling data. This argument does not change PyTorch gradient or inference mode. | - | - | - |
| layout | str | Optional | Logical Q/K/V layout. Default: `"TND"`. Supports `"BNSD"`, `"BSND"`, and `"TND"`; must match the main interface. | - | - | - |

### flash_kda

| Name | Parameter type | Required/Optional | Description | Data type | Data format | Shape |
| --- | --- | --- | --- | --- | --- | --- |
| q | Tensor | Required | Query $q_t$. | bfloat16 | ND | See the input layout table. |
| k | Tensor | Required | Key $k_t$, with the same shape as `q`. | bfloat16 | ND | Same as `q`. |
| v | Tensor | Required | Value $v_t$. | bfloat16 | ND | See the input layout table. |
| g | Tensor | Required | Pre-activation gate input when fused; otherwise per-step log gate $\ell_t$. | bfloat16, float32 | ND | See the input layout table. |
| beta | Tensor | Required | Pre-activation input when sigmoid is fused; otherwise update coefficient $b_t$. | bfloat16, float32 | ND | See the input layout table. |
| metadata | Tensor | Required | Scheduling data returned by `flash_kda_metadata`. Must not be `None` during execution. | int32 | ND | One-dimensional; generated by the metadata interface. |
| initial_state | Tensor | Optional | Initial state $H_0$. Default: `None`, interpreted as zero. | float32 | ND | `(B,Nv,128,128)` |
| A_log | Tensor | Optional | Per-value-head gate parameter. Default: `None`; required when `use_gate_in_kernel=True`. | float32 | ND | `(Nv,)` |
| dt_bias | Tensor | Optional | Gate bias. Default: `None`, interpreted as zero. | float32 | ND | `(Nv,128)` or `(Nv*128,)` |
| cu_seqlens | Tensor | Optional | Cumulative sequence lengths. Default: `None`; required for TND, matching metadata generation. | int32 | ND | `(B+1,)` |
| cu_seqlens_cpu | List[int] | Optional | Reserved; only the default `None` is supported. | - | - | - |
| seqused | Tensor | Optional | Valid sequence lengths. Default: `None`, using differences from `cu_seqlens`, or `S` if both length tensors are omitted. | int32 | ND | `(B,)` |
| scale | float | Optional | Output scale. Default: `0.08838834764831845`, equal to `128**(-0.5)`. Accepts a finite value or `None`; `None` uses the default scale. | - | - | - |
| output_final_state | bool | Optional | Returns the final state. Default: `False`. | - | - | - |
| return_intermediate_states | bool | Optional | Returns chunk entry states during inference. Default: `False`; training uses `disable_recompute` instead. | - | - | - |
| use_qk_l2norm_in_kernel | bool | Optional | Fuses Q/K L2 normalization. Default: `True`. | - | - | - |
| use_gate_in_kernel | bool | Optional | Fuses gate activation. Default: `True`. | - | - | - |
| use_beta_sigmoid_in_kernel | bool | Optional | Fuses beta sigmoid. Default: `True`. | - | - | - |
| allow_neg_eigval | bool | Optional | Multiplies the fused sigmoid result by 2. Default: `False`; `True` requires `use_beta_sigmoid_in_kernel=True`. | - | - | - |
| disable_recompute | bool | Optional | Saves additional backward caches during training to avoid recomputation. Default: `False`. | - | - | - |
| lower_bound | float | Optional | Gate activation lower bound. Default: `None`; fused gate accepts finite values in `[-5,0)`. | - | - | - |
| layout | str | Optional | Logical Q/K/V layout. Default: `"TND"`. Supports `"BNSD"`, `"BSND"`, and `"TND"`. | - | - | - |

For dense layouts, `g` and `beta` always place the sequence dimension before the head dimension.

| layout | q, k | v | g | beta |
| --- | --- | --- | --- | --- |
| BNSD | `(B,Nqk,S,128)` | `(B,Nv,S,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` |
| BSND | `(B,S,Nqk,128)` | `(B,S,Nv,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` |
| TND | `(T,Nqk,128)` | `(T,Nv,128)` | `(T,Nv,128)` | `(T,Nv)` |

## Return Values

### flash_kda_metadata

| Name | Parameter type | Required/Optional | Description | Data type | Data format | Shape |
| --- | --- | --- | --- | --- | --- | --- |
| metadata | Tensor | Required | Contiguous one-dimensional scheduling tensor. Contains forward scheduling data when `enable_grad=False`, and also backward scheduling data when `True`. Pass the returned tensor unchanged as the `metadata` argument to `flash_kda`. | int32 | ND | One-dimensional; length is generated from the input specifications and execution mode. |

### flash_kda

Returns `(out, final_state, chunk_states)`. Unrequested optional outputs are `None`; the tuple always has three elements.

| Name | Parameter type | Required/Optional | Description | Data type | Data format | Shape |
| --- | --- | --- | --- | --- | --- | --- |
| out | Tensor | Required | Forward result $o_t$. BNSD input also produces BSND output. | bfloat16 | ND | Dense: `(B,S,Nv,128)`; TND: `(T,Nv,128)`. |
| final_state | Tensor | Optional | Final state when `output_final_state=True`. | float32 | ND | `(B,Nv,128,128)` |
| chunk_states | Tensor | Optional | Chunk entry states when inference uses `return_intermediate_states=True`, or training uses `disable_recompute=True`. Non-differentiable. | bfloat16 | ND | Dense: `(B,C,Nv,128,128)`; TND: `(Ctotal,Nv,128,128)`. |

`C=ceil(S/64)`. `Ctotal=ceil(T/64)+B` is the TND state capacity, including chunk alignment space. For sequence `b`, valid states start at `floor(cu_seqlens[b]/64)+b` and occupy `ceil(length_b/64)` slots, where `length_b` is its valid length. Only consume these valid slots.

## Constraints

- Supports training and inference in eager mode.
- `B` and `S` or `T` must be positive. Head counts satisfy `1 <= Nqk <= Nv <= 96` and `Nv % Nqk == 0`; Q/K and V head dimensions are 128. Metadata integer dimensions and the sum of valid sequence lengths must fit in int32.
- Non-contiguous tensors: None of the computational tensor inputs (for both `flash_kda_metadata` and `flash_kda`) support non-contiguous tensors; passing a non-contiguous tensor raises a validation error.
- Empty tensors: None of the computational tensor inputs (for both `flash_kda_metadata` and `flash_kda`) support empty tensors (any dimension equal to 0 or zero elements); passing an empty tensor raises a validation error. Pass `None` when omitting an optional tensor argument.
- Prepare inputs and metadata on the same NPU and set the current device before calling. Meta tensors are for shape inference only.
- For TND, `cu_seqlens` starts at zero, is strictly increasing, and ends at `T`. For dense layouts, any supplied adjacent differences must be positive and at most `S`. `seqused` entries must be positive and at most the storage length. If both length tensors are supplied, each `seqused` entry must equal the corresponding adjacent difference; smaller values cannot truncate the interval.
- Metadata must match layout, storage dimensions, head counts, valid lengths, mode, and device. Regenerate it when these change; do not reuse it across devices or implementation versions. Always pass storage length `S` for dense layouts or total length `T` for TND.
- `torch.is_inference_mode_enabled()` controls the forward output/cache protocol. Use `torch.inference_mode()` with `enable_grad=False` for inference; `torch.no_grad()` retains the training protocol. Training uses `enable_grad=True`; `disable_recompute=False` retains necessary caches and recomputes during backward, while `True` retains additional caches.
- Use `flash_kda` for autograd. When gradient recording is enabled and at least one input requires gradients, it saves backward caches internally and invokes FlashKDAGrad. Second-order gradients are unsupported.
- Without gate fusion, `g` is the per-step log gate and `A_log`, `dt_bias`, and `lower_bound` are unused. Without beta sigmoid fusion, `beta` is the update coefficient and `allow_neg_eigval` must be `False`.
- Output storage can include chunk alignment space, and returned tensors are not guaranteed contiguous. Call `.contiguous()` when required by a subsequent interface. Only consume valid token/chunk regions; unused storage has no valid-value guarantee.

## Determinism and Batch Consistency

- Deterministic computation is supported by default.
- Batch consistency is not supported by default. Changing the batch or sequence grouping can change scheduling; bitwise equality across batch compositions is not guaranteed.

## Examples

These eager examples import `cann_ops_transformer`. The first two run inference; the third also requires the companion FlashKDAGrad extension.

### Dense Inference

```python
import torch
import torch_npu
import cann_ops_transformer

torch.npu.set_device(0)
batch, length, n_qk, n_v, dim = 1, 65, 2, 4, 128

# BSND inputs; g and beta place sequence before head.
q = torch.randn(batch, length, n_qk, dim, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(batch, length, n_v, dim, device="npu", dtype=torch.bfloat16)
g = torch.randn(batch, length, n_v, dim, device="npu", dtype=torch.float32)
beta = torch.randn(batch, length, n_v, device="npu", dtype=torch.float32)
A_log = torch.zeros(n_v, device="npu", dtype=torch.float32)

with torch.inference_mode():
    metadata = cann_ops_transformer.flash_kda_metadata(
        batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
        seqlens=length, enable_grad=False, layout="BSND",
    )
    out, final_state, chunk_states = cann_ops_transformer.flash_kda(
        q, k, v, g, beta, metadata, A_log=A_log,
        layout="BSND", output_final_state=True,
        return_intermediate_states=True,
    )

torch.npu.synchronize()
assert out.shape == (batch, length, n_v, dim)
assert final_state.shape == (batch, n_v, dim, dim)
assert chunk_states.shape == (batch, 2, n_v, dim, dim)
```

### Packed TND Inference

```python
import torch
import torch_npu
import cann_ops_transformer

torch.npu.set_device(0)
lengths = [65, 33]
batch, total, n_qk, n_v, dim = 2, sum(lengths), 2, 4, 128
cu_seqlens = torch.tensor([0, 65, 98], device="npu", dtype=torch.int32)
q = torch.randn(total, n_qk, dim, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(total, n_v, dim, device="npu", dtype=torch.bfloat16)
g = torch.randn(total, n_v, dim, device="npu", dtype=torch.float32)
beta = torch.randn(total, n_v, device="npu", dtype=torch.float32)
A_log = torch.zeros(n_v, device="npu", dtype=torch.float32)

# Pass total storage length T; reuse cu_seqlens in forward.
with torch.inference_mode():
    metadata = cann_ops_transformer.flash_kda_metadata(
        cu_seqlens, batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
        seqlens=total, enable_grad=False, layout="TND",
    )
    out, final_state, chunk_states = cann_ops_transformer.flash_kda(
        q, k, v, g, beta, metadata, A_log=A_log,
        cu_seqlens=cu_seqlens, layout="TND", output_final_state=True,
    )

torch.npu.synchronize()
assert out.shape == (total, n_v, dim)
assert final_state.shape == (batch, n_v, dim, dim)
assert chunk_states is None
```

### Autograd Training

```python
import torch
import torch_npu
import cann_ops_transformer

torch.npu.set_device(0)
batch, length, n_qk, n_v, dim = 1, 64, 1, 1, 128

# Training inputs require gradients; do not enter inference_mode.
q = torch.randn(batch, length, n_qk, dim, device="npu",
                dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(batch, length, n_v, dim, device="npu",
                dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(batch, length, n_v, dim, device="npu",
                dtype=torch.float32, requires_grad=True)
beta = torch.randn(batch, length, n_v, device="npu",
                   dtype=torch.float32, requires_grad=True)
A_log = torch.zeros(n_v, device="npu", dtype=torch.float32, requires_grad=True)

# Generate both schedules; flash_kda manages backward caches internally.
metadata = cann_ops_transformer.flash_kda_metadata(
    batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
    seqlens=length, enable_grad=True, layout="BSND",
)
out, final_state, chunk_states = cann_ops_transformer.flash_kda(
    q, k, v, g, beta, metadata, A_log=A_log, layout="BSND",
)
out.float().sum().backward()
torch.npu.synchronize()
assert q.grad is not None and q.grad.shape == q.shape
assert v.grad is not None and v.grad.shape == v.shape
assert final_state is None and chunk_states is None
```
