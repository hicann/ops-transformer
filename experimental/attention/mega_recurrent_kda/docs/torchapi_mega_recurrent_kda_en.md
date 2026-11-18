# mega_recurrent_kda PyTorch API

[简体中文](torchapi_mega_recurrent_kda.md)

## Product Support

| Product | Supported |
| --- | :---: |
| <term>Ascend 950PR and 950DT series products</term> | √ |
| <term>Atlas A3 series products</term> | × |
| <term>Atlas A2 series products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference series products</term> | × |
| <term>Atlas training series products</term> | × |

## Description

Runs fused KDA recurrence for BSH or TH input, reads the paged checkpoint, and updates the Conv and replay pools. See the [operator overview](../README_en.md#description) for the equations and stages.

## Interface

Call `torch.ops.cann_ops_transformer.mega_recurrent_kda`.

### PyTorch signature

```python
def mega_recurrent_kda(
    hidden_states: torch.Tensor,
    qkv_projection_weight: torch.Tensor,
    decay_projection_a_weight: torch.Tensor,
    decay_projection_b_weight: torch.Tensor,
    beta_projection_weight: torch.Tensor,
    output_gate_projection_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

## Parameters

Dimension numbers in prose are one-based. Code expressions such as `shape[2]` and `stride(0)` use zero-based indexing.

### Dimension symbols

- `B` Number of logical requests. For BSH, use the first input dimension. For TH, use `cu_seqlens.numel() - 1`.
- `b` Model batch used in the specialized-path condition, where `B = b * TP`.
- `seq_len` Number of tokens per request in BSH.
- `total_tokens` Total tokens across all requests in TH.
- `H` Hidden dimension, fixed at 7168.
- `N` Heads per TP instance. TP4, TP8, and TP16 use 24, 12, and 6 heads, respectively.
- `D` Dimension per head, fixed at 128.
- `P` Projection width, `P = N * D`.
- `block_num` Number of physical blocks shared by all five pools, given by the first dimension of each pool.
- `L` Rows per Conv block, given by the second dimension of `conv_state`. Must cover the longest sequence plus two rows.
- `S` Token capacity per head and replay block, given by the third dimension of each replay pool. Must cover the longest sequence.

### Parameter table

| Parameter | Input/output/attribute | Required/optional | Dtype/format | Shape or value | Description |
| --- | --- | --- | --- | --- | --- |
| `hidden_states` | Input | Required | BF16 / ND | `[B, seq_len, H]` or `[total_tokens, H]` | Hidden states in BSH or TH layout. |
| `qkv_projection_weight` | Input | Required | BF16 / NZ | `[3*P,H]` | Joint query, key, and value projection weights. |
| `decay_projection_a_weight` | Input | Required | BF16 / NZ | `[D,H]` | First decay projection, from hidden width to head width. |
| `decay_projection_b_weight` | Input | Required | BF16 / NZ | `[P,D]` | Second decay projection, from head width to all head channels. |
| `beta_projection_weight` | Input | Required | BF16 / NZ | `[N,H]` | Projects one beta logit per head. |
| `output_gate_projection_weight` | Input | Required | BF16 / NZ | `[P,H]` | Projects the output gate logits. |
| `output_norm_weight` | Input | Required | FP32 / ND | `[D]` | Per-channel RMSNorm weight shared by all heads. |
| `output_projection_weight` | Input | Required | BF16 / NZ | `[H,P]` | Projects concatenated head outputs to hidden width. |
| `conv1d_weight` | Input | Required | BF16 / ND | `[4,3*P]`, `[3,N,4,D]` or `[3,4,D]` | Four-tap causal convolution weights. |
| `conv_state` | Input/output | Required | BF16 / ND | `[block_num,L,3*P]`, writable | Convolution history pool. |
| `checkpoint_state` | Input | Required | FP32 / ND | `[block_num,N,D,D]`, read-only | Read-only initial recurrent state. |
| `replay_u` | Input/output | Required | FP32 / ND | `[block_num,N,S,D]`, writable update vectors | Replay update vectors. |
| `replay_k` | Input/output | Required | FP16 / ND | `[block_num,N,S,D]`, writable normalized keys | Replay normalized keys. |
| `replay_decay` | Input/output | Required | FP32 / ND | `[block_num,N,S,D]`, writable exponentiated decay factors | Replay exponentiated decay factors. |
| `state_indices` | Input | Required | int32 / ND | `[B]`, unique block indices | Shared physical block index for each request. |
| `conv_num_accepted_tokens` | Input | Required | int32 / ND | `[B]`, Conv history counts in `[0,L-2]` | Selects the convolution history window for each request. |
| `A_log` | Input | Required | FP32 / ND | `[N]`, log decay parameter | Log decay parameter per head. |
| `dt_bias` | Input | Required | FP32 / ND | `[N,D]`, decay bias | Decay bias per head and channel. |
| `scale` | Attribute | Required | FP32 scalar | Query scale, typically `D**-0.5` | Scales normalized queries. |
| `lower_bound` | Attribute | Required | FP32 scalar | Decay bound in `[-5,0]` | Lower bound of the log decay gate. |
| `rms_norm_eps` | Attribute | Optional (default `1e-5`) | FP32 scalar | Positive normal value | Epsilon for output RMSNorm. |
| `cu_seqlens` | Input | Optional (required for TH) | int32/int64 / ND | `[B+1]` prefix sums for TH, `None` for BSH | Cumulative token counts for TH. Use None for BSH. |
| `out` | Output | — | BF16 / ND | `[B,seq_len,H]` / `[total_tokens,H]` | Main output, with the same shape as the input. |

## Returns

The result is `(out, conv_state, replay_u, replay_k, replay_decay)`. `out` is BF16 with the shape of `hidden_states`. The other four entries alias their corresponding input pools. Their storage layouts are preserved. `checkpoint_state` is read-only.

## Constraints

### Constraint types

- **Single-parameter constraints** Covers a tensor's shape, dtype, format, and values, or a scalar attribute's range.
- **Presence constraints** Specifies which parameters a scenario requires or omits.
- **Consistency constraints** Covers dimension, dtype, or storage relationships within a group.
- **Cross-feature constraints** Covers relationships between groups, including dimensions, capacities, and execution conditions.

### Feature parameter groups

Groups follow the computation stages and field roles. Supply all required inputs. The four `INPUT/OUTPUT` tensors are updated in place and returned.

| Feature parameter group | Parameter field | Field role | Field type |
| --- | --- | --- | --- |
| Common | `hidden_states` | INPUT | Tensor |
|  | `out` | OUTPUT | Tensor |
| QKV projection | `qkv_projection_weight` | INPUT | Tensor |
| Causal Conv1D | `conv1d_weight` | INPUT | Tensor |
|  | `conv_state` | INPUT/OUTPUT | Tensor |
|  | `conv_num_accepted_tokens` | INPUT | Tensor |
| Decay projection and gating | `decay_projection_a_weight` | INPUT | Tensor |
|  | `decay_projection_b_weight` | INPUT | Tensor |
|  | `A_log` | INPUT | Tensor |
|  | `dt_bias` | INPUT | Tensor |
|  | `lower_bound` | ATTR | float |
| Beta projection and query scaling | `beta_projection_weight` | INPUT | Tensor |
|  | `scale` | ATTR | float |
| Paged recurrence and replay | `state_indices` | INPUT | Tensor |
|  | `checkpoint_state` | INPUT | Tensor |
|  | `replay_u` | INPUT/OUTPUT | Tensor |
|  | `replay_k` | INPUT/OUTPUT | Tensor |
|  | `replay_decay` | INPUT/OUTPUT | Tensor |
| Output normalization, gating, and projection | `output_norm_weight` | INPUT | Tensor |
|  | `rms_norm_eps` | ATTR(OPTIONAL) | float |
|  | `output_gate_projection_weight` | INPUT | Tensor |
|  | `output_projection_weight` | INPUT | Tensor |
| Variable-length input | `cu_seqlens` | INPUT(OPTIONAL) | Tensor |

### Reference quantities

| Quantity | Meaning |
| --- | --- |
| `max_seq_len` | `seq_len` for BSH, or the largest consecutive difference in `cu_seqlens` for TH. |
| `slot` | Physical block for the current request, taken from `state_indices[b]`. |
| `pool.stride(0)` | Distance in elements between adjacent physical blocks in a state pool. |

### Constraints by parameter group

Each constraint is annotated with its interception behavior:

- [Enforced] Checked synchronously at call time. Violations raise `ValueError` or `TypeError`, and no state pool is modified (atomic rejection). The checks cover shapes, dtypes, formats, layouts, scalar ranges, and storage aliasing. They do not validate tensor contents (such as weight, `A_log`, `dt_bias`, or checkpoint values); content correctness is the caller's responsibility.
- [Enforced on host] Value checks that must copy device tensors to the host. These value checks are skipped during graph capture; the caller must write valid values before replay (see [Graph capture](#graph-capture)).
- [Not enforced] Cases where the operator performs no check or none is needed, including content and semantic contracts owned by the caller, and results guaranteed by the operator's own execution behavior.
- Within a single constraint, mixed statements are annotated individually.

#### Common

- **Single-parameter constraints** [Enforced] `hidden_states` is BF16/ND with hidden width 7168. BSH requires positive batch and sequence lengths. TH requires a positive token count. The batch satisfies `1 <= B <= block_num`. The total token count is below `2**31` in both layouts (host-side value check, skipped during graph capture).

- **Presence constraints** [Enforced] The interface is for inference. Supply every input except `rms_norm_eps` and `cu_seqlens`; missing arguments are rejected by the interface schema. Use ordinary tensors and call `.detach()` on Parameter objects first; undetached Parameter inputs are rejected.

- **Consistency constraints** [Enforced] All tensors reside on the same NPU; device consistency is checked per tensor, and NPU devices are required. The main output matches the input shape, layout, and dtype (not enforced: the operator allocates the output from the input shape and layout, so this holds by construction).

- **Cross-feature constraints** [Enforced] Convert the six projection weights to NZ before calling. Other tensors use ND. NZ/ND formats are checked per tensor. Input tensors outside the five state pools must be contiguous.

#### QKV projection

- **Single-parameter constraints** [Enforced] `qkv_projection_weight` is BF16/NZ with shape `[3*P,H]`, in Q, K, V order. The Q, K, V channel ordering is a content contract (not enforced: the operator splits channels in a fixed order and does not inspect the weight contents).

- **Cross-feature constraints** [Enforced] Its input width matches `H` in `hidden_states`. Its `3*P` output channels match the Conv weights and history pool (the Conv weight shape enumeration and the third dimension of `conv_state` are both checked against the same `3*P`).

#### Causal Conv1D

- **Single-parameter constraints** [Enforced] `conv1d_weight` accepts `[4,3*P]`, `[3,N,4,D]`, or `[3,4,D]` with four taps. `conv_state` is BF16/ND with shape `[block_num,L,3*P]`, `L >= 3`, and a positive `block_num`.

- **Consistency constraints** [Enforced on host] `conv_num_accepted_tokens` is an int32 tensor of shape `[B]` with values in `[0,L-2]`. Reading three history rows starting at `max(count-1,0)`, copying the last two rows to the start of the block, and appending the current raw QKV projections are execution behavior (not enforced); whether the count is semantically consistent with the true history length is the caller's responsibility (not enforced).

- **Cross-feature constraints** [Enforced] Select history blocks with `state_indices`. For a longest request of `max_seq_len` tokens, require `L >= max_seq_len + 2` (checked at the entry point for BSH; for TH it depends on host-side sequence lengths and is skipped during graph capture).

#### Decay projection and gating

- **Single-parameter constraints** [Enforced] Both projection weights are BF16/NZ with shapes `[D,H]` and `[P,D]`, respectively. `A_log` and `dt_bias` are FP32/ND. `lower_bound` is finite FP32 in `[-5,0]`.

- **Consistency constraints** [Enforced] The first projection output width matches the second projection input width `D`. The second output is grouped as `[N,D]`, aligned by head with `A_log[N]` and `dt_bias[N,D]`.

#### Beta projection and query scaling

- **Single-parameter constraints** [Enforced] `beta_projection_weight` is BF16/NZ with shape `[N,H]`. `scale` must be representable as finite FP32; its actual value is unrestricted (not enforced: the common value is `D**-0.5`, and the numerical effect is the caller's responsibility). Q/K normalization uses a fixed `1e-6` (a compile-time constant, no check needed).

- **Cross-feature constraints** [Enforced] `N` is 24, 12, or 6 for TP4, TP8, or TP16. All weights and state pools use the same `N` and `D=128`.

#### Paged recurrence and replay

- **Single-parameter constraints** [Enforced on host] `state_indices` is an int32 tensor of shape `[B]`. Its indices are distinct within a call and in `[0,block_num)` (including block 0, which is an ordinary valid block). Index range and uniqueness checks read values to the host and are skipped during graph capture. `checkpoint_state` being read-only is a behavioral promise (not enforced: the operator never writes to this pool, so no check is needed).

- **Consistency constraints** [Enforced] The replay pools share shape `[block_num,N,S,D]`. They store FP32 update vectors, FP16 normalized keys, and FP32 exponentiated decay factors, with a positive `S`. Writing only valid token prefixes in the selected blocks while preserving other blocks, record tails, and inter-block gaps is execution behavior (not enforced).

- **Cross-feature constraints** [Enforced] Conv, checkpoint, and the three replay pools share `block_num` and the physical block selected for each request. Each pool can have its own `stride(0)` of at least one block in elements. The remaining axes must be contiguous. The four writable pools require `requires_grad=False` and storage regions disjoint from other inputs (aliasing is checked pairwise). Replay capacity must satisfy `S >= max_seq_len` (checked at the entry point for BSH; for TH it depends on host-side sequence lengths and is skipped during graph capture).

For `state_indices=[5,0]` and sequence lengths of one and three, the call writes the first replay record in block 5 and the first three records in block 0. Other records retain their values. A Conv count of 2 for the second request starts the history read at row 1 in block 0.

Compute the replay offset in elements of the corresponding pool's dtype.

```text
slot * replay.stride(0) + (head * S + token) * D + channel
```

#### Output normalization, gating, and projection

- **Single-parameter constraints** [Enforced] `output_norm_weight` is an FP32/ND tensor of shape `[D]`. Gate and output projection weights are BF16/NZ with shapes `[P,H]` and `[H,P]`. `rms_norm_eps` must be a positive normal FP32 value.

- **Presence constraints** [Enforced] The default `rms_norm_eps` is `1e-5` (guaranteed by the interface default value).

- **Cross-feature constraints** [Enforced] All heads share the RMSNorm weight. The gate projection aligns with the `P` concatenated channels, and the final projection restores width `H`.

#### Variable-length input

- **Single-parameter constraints** [Enforced] `cu_seqlens` is a contiguous int32 or int64 ND tensor of shape `[B+1]`.

- **Presence constraints** [Enforced] TH requires `cu_seqlens`. BSH uses `None`.

- **Consistency constraints** [Enforced on host] Prefix sums start at zero, increase strictly, and end at `total_tokens`. Each consecutive difference gives the corresponding request length. The value check reads data to the host and is skipped during graph capture.

- **Cross-feature constraints** [Enforced on host] The largest consecutive difference determines `max_seq_len` for Conv and replay capacity checks. TH reads prefix sums on the host and cannot be captured (an execution restriction rather than a check; it cannot be intercepted inside the operator, so callers must avoid capturing TH inputs).

### Graph capture

Graph capture here means ACL Graph stream capture through `torch_npu.npu.NPUGraph` / `torch_npu.npu.graph`.
BSH supports graph capture. Warm up operator compilation and workspace allocation before capture. Static parameter checks remain enabled; device-value checks and wrapper event waits and recording are skipped during capture.
Repeated zero block indices may be used as placeholders during capture. Before replay, supply valid, distinct block indices and valid Conv counts. Ordinary calls, including warm-up before capture, still perform value checks.
The shared call lock and completion event protect workspace and pool reuse. The caller manages graph and shared workspace lifetimes and serializes their use.
The general BSH path generates equally spaced sequence prefixes on the device. TH reads `cu_seqlens` on the host to select its execution path and cannot be captured.

The Torch extension registers Meta and PrivateUse1 implementations for `torch.ops.cann_ops_transformer.mega_recurrent_kda`. The operator can be called directly when building graphs with TorchDynamo.

### Execution path and core quotas

| Implementation | Path | Matrix-core quotas | Vector cores per matrix core |
|---|---|---|---|
| `arch35` | Uses the specialized path for equal lengths of four or eight tokens when `B=b*TP` and `1 <= b <= 32`. Other valid inputs use the general path. | 8, 16, 24, 28, 32 | 2 |
| `arch92` | Uses the general path for BSH and TH. | 8, 16, 24, 32, 64 | 1 |

The quota comes from the active stream. Callers can specify it through stream resource limits.

Set `KDA_ARCH=arch92` to select the arch92 cases when running pytest or exporting cases. See the [case guide](../tests/ttk/README_en.md) for TTK cases and commands.

## Usage example

This BSH example uses BF16/ND `hidden_states` of shape `[2,4,7168]`. Prepare the six NZ projection weights, `conv1d_weight`, `output_norm_weight`, `A_log`, and `dt_bias` as specified in the parameter table. Install the operator package and PyTorch extension built from this repository and import the extension to register the interface. The example uses the default extension package name `cann_ops_transformer`; the name depends on the build configuration.

```python
import torch
import cann_ops_transformer

block_num, N, D = 6, 12, 128
S, L = 8, 10
P = N * D
device = hidden_states.device

conv_state = torch.zeros(2*block_num, L, 3*P, dtype=torch.bfloat16, device=device)[::2]
checkpoint_state = torch.zeros(3*block_num, N, D, D, dtype=torch.float32, device=device)[::3]
replay_u = torch.zeros(2*block_num, N, S, D, dtype=torch.float32, device=device)[::2]
replay_k = torch.zeros(3*block_num, N, S, D, dtype=torch.float16, device=device)[::3]
replay_decay = torch.zeros(4*block_num, N, S, D, dtype=torch.float32, device=device)[::4]
state_indices = torch.tensor([5, 0], dtype=torch.int32, device=device)
conv_num_accepted_tokens = torch.zeros(2, dtype=torch.int32, device=device)

out, conv_state, replay_u, replay_k, replay_decay = torch.ops.cann_ops_transformer.mega_recurrent_kda(
    hidden_states=hidden_states,
    qkv_projection_weight=qkv_projection_weight,
    decay_projection_a_weight=decay_projection_a_weight,
    decay_projection_b_weight=decay_projection_b_weight,
    beta_projection_weight=beta_projection_weight,
    output_gate_projection_weight=output_gate_projection_weight,
    output_norm_weight=output_norm_weight,
    output_projection_weight=output_projection_weight,
    conv1d_weight=conv1d_weight,
    conv_state=conv_state,
    checkpoint_state=checkpoint_state,
    replay_u=replay_u,
    replay_k=replay_k,
    replay_decay=replay_decay,
    state_indices=state_indices,
    conv_num_accepted_tokens=conv_num_accepted_tokens,
    A_log=A_log,
    dt_bias=dt_bias,
    scale=D**-0.5,
    lower_bound=-5.0,
)
# The two requests write replay records to slots 5 and 0.
```

TH uses the same interface. For two requests with one and three tokens, pack the input in request order into shape `[4,7168]` and pass an int32 or int64 tensor `cu_seqlens=[0,1,4]` on the same device. Supply one block index and one Conv count per request.
