# flash_kda_grad / flash_kda_grad_metadata

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR/950DT series</term> (`dav-3510`) | √ |
| <term>Atlas A3 series</term> | × |
| <term>Atlas A2 series</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference series</term> | × |
| <term>Atlas training series</term> | × |

The current implementation uses a fixed chunk size `C=64` and supports `Dk=Dv=128`. Support on other products
is not declared by this implementation.

## Description

FlashKDAGrad computes the training backward pass of Flash Kimi Delta Attention (KDA). It takes the forward
inputs, the forward caches, the output gradient `do` and the final-state gradient `dht`, and produces the
gradients of Q, K, V, the initial state, the gate, beta and the gate parameters. Variable-length sequences,
GQA, Q/K L2 normalization, gate activation and beta sigmoid fusion are supported.

For one sequence and one value head, let $\bar q_t$ and $\bar k_t$ be the Q/K used by attention, $\ell_t$ the
per-step log gate, $b_t$ the activated beta, and store the state $H_t$ as `[Dv,Dk]`. The forward semantics are:

$$
\widetilde H_t = H_{t-1}\operatorname{diag}(\exp(\ell_t))
$$

$$
\Delta v_t = b_t\left(v_t-\widetilde H_t\bar k_t\right),\qquad
H_t = \widetilde H_t+\Delta v_t\bar k_t^T
$$

$$
o_t = scale\cdot H_t\bar q_t
$$

The backward pass treats `do` and `dht` as the upstream gradients of $o_t$ and of the final state, and returns:

$$
(dq,dk,dv,dh0,dg,dbeta,dA,dBias)
=
\left(
\frac{\partial L}{\partial q},
\frac{\partial L}{\partial k},
\frac{\partial L}{\partial v},
\frac{\partial L}{\partial H_0},
\frac{\partial L}{\partial g},
\frac{\partial L}{\partial beta},
\frac{\partial L}{\partial A_{log}},
\frac{\partial L}{\partial dt_{bias}}
\right)
$$

`dA` is the gradient with respect to `A_log`. With Q/K L2 normalization enabled, `dq/dk` are the gradients of
the pre-normalization Q/K. The formulas above describe the computational semantics; the implementation uses chunk
decomposition and a reverse scan over state gradients, with BF16 Cube operands and FP32 state caches and state
gradients.

The host automatically selects persistent or staged scheduling from the effective sequence lengths, the head
counts and the number of device cores; both strategies share one public interface. The main backward launches a
single mixed Cube/Vector compute kernel and by default executes the backward on the seven caches saved by the
forward. With `disable_recompute=False` the same kernel first rebuilds the required caches and then runs the
backward. Scheduling information is produced by a separate AICPU metadata interface.

**Recompute** lives in [flash_kda_grad_recompute.py](../op_dsl/flash_kda_grad_recompute.py) and neither calls
nor imports the forward implementation. The precision standard is the STC FP32 reference and the native
`cross_check L2`: raw `g`, `beta` and the saved beta activation cache keep the input precision; the `W/U` multiply
inputs and the FP32 recursion state use a BF16 main value plus a residual Cube product pair as compensation, and
the token caches finally stored remain BF16. The retained `Akk`, normalized Q/K and beta activation caches are
reused; token caches of invalid tokens in padded layouts are cleared.

**Staged rebuild**: both backward schedules share one rebuild flow. `qg/kg/g_cumsum/W/U` are first computed per
chunk across cores; after a full-core sync and on-chip resource reclamation, `chunk_states/v_new` are recursed
per value slice. The rounded BF16 `U` is read first, the compensated `W @ H` is subtracted in FP32, `v_new` is
rounded to BF16, and the state is updated with FP32 multiply-add. Existing metadata and partition interfaces such
as `_staged_chunk_start` are reused; after the rebuild a full-core sync precedes the original persistent or
staged backward. The whole flow still launches a single compute kernel.

Recompute calls with tail chunks use staged backward so that channel acquire/release between the rebuild and
backward phases stays deterministic; saved-cache mode and tail-free persistent recompute are unaffected.

Consistency between recompute and saved caches is validated against a reference cache that keeps FP32
intermediate math, using native TTK L2 on the final gradients. Because intermediate precision is higher,
recompute no longer requires bit-exact reproduction of the extra BF16 operand rounding of the real forward.

## Interface

The Torch entry is [torch_extension/flash_kda_grad.py](../torch_extension/flash_kda_grad.py), registered as
`torch.ops.cann_ops_transformer.flash_kda_grad`. The DSL main entry and the forward-cache adapter are in
[op_dsl/flash_kda_grad.py](../op_dsl/flash_kda_grad.py); the metadata entry is in
[op_dsl/flash_kda_grad_metadata.py](../op_dsl/flash_kda_grad_metadata.py). Callers generate the backward
metadata first, then call the main entry or the adapter.

### PyTorch prototype

```python
cann_ops_transformer.flash_kda_grad(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    do: Tensor,
    dht: Tensor,
    Aqk: Tensor,
    Akk: Tensor,
    metadata: Optional[Tensor] = None,
    *,
    initial_state: Optional[Tensor] = None,
    chunk_states: Optional[Tensor] = None,
    w: Optional[Tensor] = None,
    u: Optional[Tensor] = None,
    v_new: Optional[Tensor] = None,
    qg: Optional[Tensor] = None,
    kg: Optional[Tensor] = None,
    g_cumsum: Optional[Tensor] = None,
    q_rtsd: Optional[Tensor] = None,
    k_rstd: Optional[Tensor] = None,
    beta_sigmoid: Optional[Tensor] = None,
    A_log: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
```

The Torch entry adds no numerical processing and dispatches to the DSL implementation through PrivateUse1.
`disable_recompute` defaults to `False` (recompute, memory saving) on the Torch entry and to `True` (consume saved
caches) on the DSL main entry; the semantics are identical.

### DSL main entry prototype

```python
flash_kda_grad_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    metadata: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qg: Optional[torch.Tensor] = None,
    kg: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
    q_rtsd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = True,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
)
```

`q_rtsd` is the actual parameter name and differs in spelling from `k_rstd`. The main entry has no separate
`q_norm/k_norm` parameters; with normalization enabled, pass the forward-saved normalized results as `q/k` in the
input layout.

### Metadata prototype

```python
flash_kda_grad_metadata(
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    batch_size: int,
    n_heads_qk: int,
    n_heads_v: int,
    seqlens: int,
    headdim_qk: int = 128,
    headdim_v: int = 128,
    layout: str = "TND",
) -> torch.Tensor
```

### Forward-cache adapter prototype

```python
flash_kda_grad_from_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    forward_outputs: tuple[torch.Tensor | None, ...],
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    metadata: torch.Tensor,
    scale: Optional[float] = None,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    disable_recompute: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states_fp32: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
)
```

The adapter takes the raw `q/k` and the 16 values returned by the training forward. With normalization enabled it
extracts `q_norm/k_norm` from `forward_outputs`, converts them to the input layout and forwards them to the main
entry. `A_log` and `dt_bias` have no default in this entry; pass `None` explicitly when unused.

## Parameters

Dimension numbering: "dimension N" in this document starts at 1; code expressions such as `shape[0]` start at 0.

Dimension symbols:

- `B`: number of logical sequences, `B >= 1`.
- `S`: physical sequence length of BSND/BNSD inputs.
- `T`: total number of tokens of a TND input, equal to the sum of all effective lengths.
- `Nqk`: number of Q/K heads.
- `Nv`: number of V heads, requiring `Nv % Nqk == 0`.
- `R`: number of value heads per Q/K head, `R = Nv / Nqk`.
- `Dk`, `Dv`: head dims of Q/K and V, both 128 currently.
- `C`: chunk size, fixed at 64.
- `Cmax`: chunk capacity of padded layouts, `Cmax = ceil(S / C)`.
- `Nc`: chunk cache capacity of TND, `Nc = ceil(T / C) + B`, including reserved slots.

### Input layouts

The shapes below are the shapes actually received by the interface. `layout` applies to `q/k/v/g/beta/do`.

| Parameter | TND | BNSD | BSND |
| :--- | :--- | :--- | :--- |
| `q`, `k` | `[T,Nqk,Dk]` | `[B,Nqk,S,Dk]` | `[B,S,Nqk,Dk]` |
| `v`, `do` | `[T,Nv,Dv]` | `[B,Nv,S,Dv]` | `[B,S,Nv,Dv]` |
| `g` | `[T,Nv,Dk]` | `[B,Nv,S,Dk]` | `[B,S,Nv,Dk]` |
| `beta` | `[T,Nv]` | `[B,Nv,S]` | `[B,S,Nv]` |
| `initial_state`, `dht` | `[B,Nv,Dv,Dk]` | `[B,Nv,Dv,Dk]` | `[B,Nv,Dv,Dk]` |

### Main interface parameters

"Cache layout" below refers to the forward-cache shapes defined in the next section; FLOAT32 and BFLOAT16
correspond to `torch.float32` and `torch.bfloat16`.

| Parameter | In/Out | Description | Usage | dtype | Format | Shape |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `q`, `k` | in | Query, Key. | Mandatory; normalized forward values when L2Norm is on, raw values when off. | BFLOAT16 | ND | see input layouts |
| `v` | in | Value. | Mandatory; identical to the forward input. | BFLOAT16 | ND | see input layouts |
| `g` | in | Gate input. | Mandatory; raw gate input when fused, per-step log gate when not fused. | BFLOAT16 / FLOAT32 | ND | see input layouts |
| `beta` | in | Update coefficient per token and value head. | Mandatory; pre-sigmoid input when fused, activated coefficient when not fused. | BFLOAT16 / FLOAT32 | ND | see input layouts |
| `do` | in | Gradient of the forward output. | Mandatory; pass a zero Tensor when absent, `None` is rejected. | BFLOAT16 | ND | same as `v` |
| `dht` | in | Gradient of the final state. | Mandatory; pass a zero Tensor when absent, `None` is rejected. | FLOAT32 | ND | `[B,Nv,Dv,Dk]` |
| `Aqk` | in | Forward causal QK matrix, already scaled. | Mandatory in both modes, never recomputed. | BFLOAT16 | ND | cache layout |
| `Akk` | in | Inverse of the unit lower-triangular matrix of the forward WY transform. | Mandatory in both modes, never recomputed. | BFLOAT16 | ND | cache layout |
| `metadata` | in | Backward scheduling information. | Mandatory; produced by `flash_kda_grad_metadata`, a forward metadata cannot substitute. | INT32 | ND | `[8+7*B*Nv]` |
| `initial_state` | in | Forward initial state. | Used in recompute mode; `None` means a zero state; unused in saved-cache mode. | FLOAT32 | ND | `[B,Nv,Dv,Dk]` |
| `chunk_states` | in | Entry state of each chunk. | Mandatory in saved-cache mode; ignored and regenerated in recompute mode. | FLOAT32 | ND | cache layout |
| `w` | in | Key cache after the WY transform. | Mandatory in saved-cache mode; regenerated in recompute mode. | BFLOAT16 | ND | cache layout |
| `u` | in | Value cache after the WY transform. | Still required in saved-cache mode; the current backward math does not read the saved values. | BFLOAT16 | ND | cache layout |
| `v_new` | in | Value corrected with the chunk entry state. | Mandatory in saved-cache mode; regenerated in recompute mode. | BFLOAT16 | ND | cache layout |
| `qg`, `kg` | in | Q/K caches combined with the cumulative gate. | Mandatory in saved-cache mode; `qg` does not include `scale`. | BFLOAT16 | ND | cache layout |
| `g_cumsum` | in | Cumulative log gate inside a chunk. | Mandatory in saved-cache mode; the per-token raw `g` cannot substitute. | FLOAT32 | ND | cache layout |
| `q_rtsd`, `k_rstd` | in | Inverse norms of the forward Q/K L2Norm. | Mandatory when Q/K normalization is on; never recomputed. | FLOAT32 | ND | cache layout |
| `beta_sigmoid` | in | Activated beta from the forward. | Mandatory when beta sigmoid fusion is on; includes the factor 2 when negative eigenvalues are enabled. | FLOAT32 | ND | cache layout |
| `A_log` | in | Gate parameter per value head. | Mandatory when `use_gate_in_kernel=True`. | FLOAT32 | ND | `[Nv]` |
| `dt_bias` | in | Gate bias per value head and Key channel. | Optional; `None` means zero bias when the gate is fused. | FLOAT32 | ND | `[Nv,Dk]` |
| `cu_seqlens` | in | TND token prefix sums. | May be omitted on the main backward; when provided it must match the metadata. | INT32 | ND | `[B+1]` |
| `seqused` | in | Effective lengths of BSND/BNSD inputs. | May be omitted on the main backward; when provided it must match the metadata. | INT32 | ND | `[B]` |
| `scale` | attr | Output/Query scaling factor. | Defaults to `None` meaning `1/sqrt(Dk)`; an explicit value must be finite and match the forward. | float / None | - | - |
| `use_qk_l2norm_in_kernel` | attr | Whether to compute the Q/K normalization backward. | Default `True`. | bool | - | - |
| `use_gate_in_kernel` | attr | Whether to compute the gate activation backward. | Default `True`. | bool | - | - |
| `use_beta_sigmoid_in_kernel` | attr | Whether to compute the beta sigmoid backward. | Default `True`. | bool | - | - |
| `allow_neg_eigval` | attr | Whether `2*sigmoid(beta)` is used. | Default `False`; when `True`, beta sigmoid fusion must be on. | bool | - | - |
| `disable_recompute` | attr | Whether the seven saved caches are used. | Defaults to `True` on the DSL entry (full caches required) and to `False` on the Torch entry. | bool | - | - |
| `lower_bound` | attr | Lower bound of the fused gate and the formula selection. | Default `None` uses softplus; a non-`None` value must be in `[-5,0)` and selects bounded sigmoid. | float / None | - | - |
| `layout` | attr | Layout of inputs and corresponding gradients. | Default `"TND"`; `"TND"`, `"BNSD"` and `"BSND"` are supported. | str | - | - |
| `dq`, `dk`, `dv` | out | Q/K/V gradients. | GQA Q/K gradients are summed over the corresponding value heads. | BFLOAT16 | ND | same as `q/k/v` |
| `dh0` | out | Gradient of the initial state. | Always an FP32 V-first state. | FLOAT32 | ND | `[B,Nv,Dv,Dk]` |
| `dg`, `dbeta` | out | Gradients of the gate/beta inputs. | Keep the dtype and shape of `g/beta`. | BFLOAT16 / FLOAT32 | ND | same as `g/beta` |
| `dA` | out | Gradient of `A_log`. | `None` when gate fusion is off. | FLOAT32 | ND | `[Nv]` |
| `dBias` | out | Gradient of `dt_bias`. | `None` when gate fusion is off or `dt_bias` was not passed. | FLOAT32 | ND | `[Nv,Dk]` |

### Forward cache layout

Both BNSD and BSND inputs use head-major forward caches. The caches must not be converted to `[B,S,N,...]` along
with a BSND input; the main entry creates the required views internally.

| Cache | TND | BNSD / BSND | dtype |
| :--- | :--- | :--- | :--- |
| `q_norm`, `k_norm` (read by the adapter from the tuple only) | `[T,Nqk,Dk]` | `[B,Nqk,S,Dk]` | BFLOAT16 |
| `q_rtsd`, `k_rstd` | `[T,Nqk]` | `[B,Nqk,S]` | FLOAT32 |
| `beta_sigmoid` | `[T,Nv]` | `[B,Nv,S]` | FLOAT32 |
| `Aqk`, `Akk` | `[Nc,Nv,C,C]` | `[B,Nv,Cmax,C,C]` | BFLOAT16 |
| `w`, `qg`, `kg` | `[T,Nv,Dk]` | `[B,Nv,S,Dk]` | BFLOAT16 |
| `u`, `v_new` | `[T,Nv,Dv]` | `[B,Nv,S,Dv]` | BFLOAT16 |
| `g_cumsum` | `[T,Nv,Dk]` | `[B,Nv,S,Dk]` | FLOAT32 |
| `chunk_states` | `[Nc,Nv,Dv,Dk]` | `[B,Nv,Cmax,Dv,Dk]` | FLOAT32 |

In TND, chunk `j` (0-based) of sequence `s` (0-based) occupies slot `floor(cu_seqlens[s]/64) + s + j`. `Nc` is not
the sum of the effective chunks; reserved slots do not participate. Matrix and state caches must keep the chunk
axis; the entry does not accept the legacy compact caches or flattened matrix caches.

`forward_outputs` must follow the 16-item return order of the training forward:

```python
(
    out, final_state, q_norm, k_norm, q_rtsd, k_rstd, beta_sigmoid,
    Aqk, Akk, w, u, qg, kg, v_new, g_cumsum, saved_chunk_states,
)
```

The adapter ignores the values of the first two items, and `do/dht` are still mandatory. `chunk_states_fp32` may
override `saved_chunk_states` in saved-cache mode, provided its dtype, shape and storage order match the table
above; recompute mode ignores the override.

### Metadata parameters

| Parameter | In/Out | Description | Usage | dtype | Format | Shape |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cu_seqlens` | in | TND sequence boundaries. | Mandatory for TND; first item 0, non-decreasing, last item T. | INT32 | ND | `[B+1]` |
| `seqused` | in | Effective lengths of padded inputs. | Optional for BSND/BNSD, each item in `[0,S]`; omitted means all sequences have length S. | INT32 | ND | `[B]` |
| `batch_size` | attr | Number of logical sequences. | Mandatory positive integer. | int | - | - |
| `n_heads_qk` | attr | Number of Q/K heads. | Mandatory positive integer. | int | - | - |
| `n_heads_v` | attr | Number of V heads. | Mandatory; must be a positive multiple of `n_heads_qk`. | int | - | - |
| `seqlens` | attr | Physical storage length. | Mandatory non-negative integer; T for TND, S for BSND/BNSD. | int | - | - |
| `headdim_qk`, `headdim_v` | attr | Head dims of Q/K and V. | Default 128; only 128 is supported currently. | int | - | - |
| `layout` | attr | Input layout. | Default `"TND"`, same options as the main entry. | str | - | - |
| returned Tensor | out | Backward scheduling snapshot. | 8 header fields plus 7 task fields per `(sequence,value_head)`. | INT32 | ND | `[8+7*B*Nv]` |

Length Tensors must be contiguous and on the NPU. The output lives on the device of the length Tensor; for
BSND/BNSD without a length Tensor it lives on the current NPU.

## Returns

The main entry, the adapter and the Torch entry all return `(dq, dk, dv, dh0, dg, dbeta, dA, dBias)` in this order:

- `dq/dk/dv` are BF16 and keep the shapes and logical layouts of Q/K/V; with normalization enabled `dq/dk`
  already include the L2Norm backward.
- `dh0` is FP32 `[B,Nv,Dv,Dk]`; for an empty sequence `dh0` equals `dht`.
- `dg/dbeta` keep the dtype and shape of the original `g/beta`. When the corresponding fusion is off, the
  gradients with respect to the processed gate/beta are returned.
- `dA` is returned only when gate fusion is on; `dBias` additionally requires `dt_bias`, otherwise the slot is `None`.

Rows of invalid tokens in BSND/BNSD write zeros into `dq/dk/dv/dg/dbeta`. Outputs are not guaranteed contiguous;
for example `dbeta` may be a transposed view and `dA` a column view of internal storage. Downstream consumers that
need contiguous tensors should call `.contiguous()` explicitly.

## Constraints

### Constraint levels

| Level | Meaning |
| :--- | :--- |
| Per-parameter | Shape, dtype, device and contiguity of a single Tensor, plus attribute value ranges. |
| Existence | Parameters that must be provided or omitted for a given layout, fusion switch or recompute mode. |
| Consistency | Agreement of shapes, head counts, sequence boundaries, layouts and cache semantics between parameters. |
| Feature cross | Requirements driven jointly by fusion, GQA, recompute and the forward caches. |

### Feature parameter groups

| Group | Fields | Kind | Type |
| :--- | :--- | :--- | :--- |
| Common | `q/k/v/do/dht` | INPUT | Tensor |
| Common | `scale/layout` | ATTR | float / str |
| Common | `dq/dk/dv/dh0` | OUTPUT | Tensor |
| Forward cache and recompute | `Aqk/Akk/chunk_states/w/u/v_new/qg/kg/g_cumsum/initial_state` | INPUT | Tensor / None |
| Forward cache and recompute | `disable_recompute` | ATTR | bool |
| Q/K normalization | `q_rtsd/k_rstd` | INPUT | Tensor / None |
| Q/K normalization | `use_qk_l2norm_in_kernel` | ATTR | bool |
| Gate | `g/A_log/dt_bias` | INPUT | Tensor / None |
| Gate | `use_gate_in_kernel/lower_bound` | ATTR | bool / float / None |
| Gate | `dg/dA/dBias` | OUTPUT | Tensor / None |
| Beta | `beta/beta_sigmoid` | INPUT | Tensor / None |
| Beta | `use_beta_sigmoid_in_kernel/allow_neg_eigval` | ATTR | bool |
| Beta | `dbeta` | OUTPUT | Tensor |
| Sequence length and metadata | `cu_seqlens/seqused/metadata` | INPUT | Tensor / None |

### Base information

| Name | Meaning |
| :--- | :--- |
| TND | Concatenates the effective tokens of all sequences and describes boundaries with `cu_seqlens`. |
| BNSD / BSND | Keeps the batch and the physical length S and describes effective lengths with `seqused`, allowing padding. |
| GQA | Each Q/K head corresponds to R consecutive value heads; value head `h` uses Q/K head `h // R`. |
| V-first | The last two state dims are Dv then Dk; state caches and `initial_state/dht/dh0` follow this convention. |
| Saved-cache mode | `disable_recompute=True`, consuming the seven caches saved by the forward. |
| Recompute mode | `disable_recompute=False`, rebuilding the seven caches inside the backward kernel. |
| persistent / staged | Two internally selected backward execution strategies; not public attributes. |

### Group constraints

#### Common group

- **Per-parameter**: fixed `Dk=Dv=128`, `C=64`. Q/K/V, `do`, matrix caches and ordinary token caches use BF16;
  states and state gradients use FP32. All Tensors actually consumed must be on the same NPU.
- **Existence**: `do/dht` are both mandatory. Even when only one of the output or the final state is differentiated,
  a shape- and type-matching zero Tensor must be provided for the other.
- **Consistency**: Q and K have identical shapes; the batch and token dims of Q/K/V agree; `1 <= Nqk <= Nv` and
  `Nv % Nqk == 0`. `g` has the same shape as `v`, and `beta` has shape `v.shape[:-1]`.
- **Feature cross**: `scale`, layout, fusion switches, `allow_neg_eigval` and `lower_bound` must match the forward
  that produced the caches. Under GQA, Q/K inputs and gradients are still stored per Nqk heads while the gate, beta,
  matrix and state caches are stored per Nv heads.

#### Forward cache and recompute group

| Parameter or behavior | `disable_recompute=False` | `disable_recompute=True` |
| :--- | :--- | :--- |
| `w/u/v_new/qg/kg/g_cumsum/chunk_states` | May be omitted or `None`; regenerated even when supplied. | All seven mandatory, matching the forward cache layout. |
| `initial_state` | `None` means a zero state; a non-zero forward initial state must be passed. | Unused. |
| `Aqk/Akk` | Mandatory, never recomputed. | Mandatory. |
| `q_rtsd/k_rstd/beta_sigmoid` | Mandatory when the corresponding fusion is on, never recomputed. | Mandatory when the corresponding fusion is on. |

- **Per-parameter**: `chunk_states` is the FP32 chunk entry state. In recompute mode `initial_state` is FP32
  `[B,Nv,Dv,Dk]` and its last-dim stride must be 1.
- **Consistency**: caches come from the same forward and keep the shapes and storage order returned by the current
  forward. Caches for BSND inputs remain head-major; TND must use the `Nc` capacity that includes reserved slots.
- **Feature cross**: the current training forward ties the normalization and beta activation caches to forward
  `disable_recompute=True`. Therefore enabling recompute in the backward does not allow omitting these fusion
  caches. Saving the state cache also requires forward `return_intermediate_state=True`; the `initial_state`
  needed by backward recompute is not part of the forward return tuple.

#### Q/K normalization group

- **Existence**: with `use_qk_l2norm_in_kernel=True`, `q/k` of the main entry must be the BF16 normalized values
  saved by the forward, and FP32 `q_rtsd/k_rstd` must be provided. When off, pass raw Q/K and no inverse norms.
- **Consistency**: inverse norms are stored per Nqk heads. For BSND inputs, `q/k` of the main entry are
  `[B,S,Nqk,Dk]` while the inverse norms remain `[B,Nqk,S]`.
- **Feature cross**: the switch controls the normalization backward and never re-normalizes raw Q/K inside the main
  entry. The adapter entry may take raw Q/K and extract the normalized results from the forward tuple.

#### Gate group

Let $x=g+dt_{bias}$ (0 when the bias is absent) and $a=\exp(A_{log})$. The per-step log gate of the fused gate is:

$$
\ell =
\begin{cases}
-a\cdot\operatorname{softplus}(x), & lower\_bound=\mathrm{None},\\
lower\_bound\cdot\sigma(a\cdot x), & -5\le lower\_bound<0.
\end{cases}
$$

- **Per-parameter**: `g` supports BF16 or FP32; `A_log` is FP32 `[Nv]`, `dt_bias` is FP32 `[Nv,Dk]`.
  `lower_bound` may only be `None` or a value in `[-5,0)`; the backward entry rejects 0.
- **Existence**: with fusion on, `A_log` is mandatory and `dt_bias` is optional. With fusion off, `g` represents the
  already-processed per-step non-positive log gate and the gate activation parameters are not used for gradients.
- **Consistency**: `g_cumsum` accumulates the per-step log gate inside a chunk. Parameter gradients are reduced over
  all effective tokens and sequences.
- **Feature cross**: the backward performs FP32 Vector math directly on FP32 `g`; BF16 inputs are promoted to FP32
  inside the kernel. `dg` matches `g.dtype` when written back. If the forward producer pre-quantizes an FP32 gate,
  the caller must ensure the gate values passed to the backward match those used to build the caches.

#### Beta group

- **Per-parameter**: `beta` independently supports BF16 or FP32, one value per token and value head;
  `beta_sigmoid` must be FP32.
- **Existence**: with `use_beta_sigmoid_in_kernel=True`, the forward activation cache is mandatory; when off, `beta`
  is used directly as the update coefficient.
- **Consistency**: with `allow_neg_eigval=False` the activation cache is `sigmoid(beta)`; with `True` it is
  `2*sigmoid(beta)` and already includes the factor 2.
- **Feature cross**: `allow_neg_eigval=True` requires beta sigmoid fusion. `dbeta.dtype` follows the original
  `beta.dtype`, not the FP32 activation cache.

#### Sequence length and metadata group

- **Per-parameter**: `metadata` is a contiguous 1-D NPU INT32 Tensor of length `8+7*B*Nv`. Its integer dimension
  attributes must fit in int32, and so must the capacity.
- **Existence**: TND metadata requires `cu_seqlens` and forbids `seqused`; BSND/BNSD accepts only `seqused`, and
  omitting it generates full-length S. The two length Tensors are mutually exclusive. The main backward may pass
  only the generated metadata.
- **Consistency**: TND prefix sums start at 0, are non-decreasing and end at T; padded effective lengths are in
  `[0,S]`. When the main backward receives lengths explicitly, they are validated against the metadata.
- **Feature cross**: empty sequences and tail chunks shorter than 64 tokens are allowed; an empty sequence passes
  `dht` to `dh0` and padding gradients are zeroed. Forward and backward metadata have different ABIs and must be
  generated separately. Changing dims, layout or effective lengths requires regenerating the backward metadata;
  the task table must not be edited by hand.

On first use the main backward validates the metadata magic, ABI version, status code and input shapes; an invalid
snapshot raises `ValueError`, and reading device data may introduce a synchronization. Identical, unmodified
metadata can be reused; the generated metadata carries a production-complete event for cross-stream waiting.

Required inputs and the caches used by the active path must be Tensors; invalid types or dims raise `ValueError`.
Conversion and finiteness checks of an explicit `scale` happen before metadata parsing and recompute cache
allocation, so invalid values never reach compute preparation.

## Example

The example below uses BNSD, GQA and a bounded gate; it runs the training forward to produce the caches and then
calls the backward. Load the CANN environment (including `ASCEND_HOME_PATH`) and install matching `cannbotdsl`,
PyTorch and `torch_npu` beforehand; the first call may trigger DSL/AICPU compilation.

```python
import torch
import torch_npu

import cann_ops_transformer as ops

torch.npu.set_device(0)
device = torch.device("npu:0")
B, S, Nqk, Nv, D = 1, 64, 1, 2, 128
layout = "BNSD"
scale = D ** -0.5
lower_bound = -0.06

q = torch.randn(B, Nqk, S, D, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, Nv, S, D, device=device, dtype=torch.bfloat16)
g = torch.randn_like(v)
beta = torch.randn(B, Nv, S, device=device, dtype=torch.bfloat16)
initial_state = torch.zeros(B, Nv, D, D, device=device, dtype=torch.float32)
A_log = torch.zeros(Nv, device=device, dtype=torch.float32)
dt_bias = torch.zeros(Nv, D, device=device, dtype=torch.float32)

# Forward and backward metadata are generated separately.
fwd_outputs = forward_op(          # training forward, returns 16 items
    q, k, v, g, beta, scale=scale, initial_state=initial_state,
    A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
    layout=layout, return_intermediate_state=True,
)
out, final_state = fwd_outputs[:2]
do = torch.randn_like(out)
dht = torch.zeros_like(final_state)  # only `out` receives a non-zero upstream gradient here.

bwd_metadata = ops.flash_kda_grad_metadata(
    batch_size=B, n_heads_qk=Nqk, n_heads_v=Nv, seqlens=S, layout=layout,
)

# Saved-cache mode: consume the seven caches returned by the forward.
dq, dk, dv, dh0, dg, dbeta, dA, dBias = ops.flash_kda_grad(
    q, k, v, g, beta, do, dht,
    fwd_outputs[7], fwd_outputs[8],           # Aqk, Akk
    bwd_metadata,
    initial_state=initial_state,
    chunk_states=fwd_outputs[15],
    w=fwd_outputs[9], u=fwd_outputs[10], v_new=fwd_outputs[13],
    qg=fwd_outputs[11], kg=fwd_outputs[12], g_cumsum=fwd_outputs[14],
    q_rtsd=fwd_outputs[4], k_rstd=fwd_outputs[5],
    beta_sigmoid=fwd_outputs[6],
    A_log=A_log, dt_bias=dt_bias,
    scale=scale,
    disable_recompute=True,
    allow_neg_eigval=False,
    lower_bound=lower_bound,
    layout=layout,
)
```

For recompute mode, set `disable_recompute=False`, pass `initial_state` and omit the seven recomputable caches:

```python
dq, dk, dv, dh0, dg, dbeta, dA, dBias = ops.flash_kda_grad(
    q, k, v, g, beta, do, dht,
    fwd_outputs[7], fwd_outputs[8], bwd_metadata,
    initial_state=initial_state,
    q_rtsd=fwd_outputs[4], k_rstd=fwd_outputs[5], beta_sigmoid=fwd_outputs[6],
    A_log=A_log, dt_bias=dt_bias,
    scale=scale,
    disable_recompute=False,
    lower_bound=lower_bound,
    layout=layout,
)
```

The normalization and beta activation caches returned by the forward are still required; the seven recomputable
caches are ignored.

When calling the DSL main entry `flash_kda_grad_npu` directly, split the forward caches according to the prototype;
this example is BNSD, so `q/k` should be replaced by the normalized values in `fwd_outputs[2:4]`. For BSND these
two items must be converted from `[B,Nqk,S,Dk]` to `[B,S,Nqk,Dk]`; the other caches keep the forward return layout.

Unit tests and reference formulas live in [tests/ut](../tests/ut); TTK precision and performance acceptance is
described in [tests/ttk/README.md](../tests/ttk/README.md).
