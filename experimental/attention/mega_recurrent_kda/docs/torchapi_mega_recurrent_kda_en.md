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

| Parameter | Input/output | Dtype/format | Shape or value | Description |
| --- | --- | --- | --- | --- |
| `hidden_states` | Input | BF16 / ND | `[B, seq_len, H]` or `[total_tokens, H]` | Hidden states in BSH or TH layout. |
| `qkv_projection_weight` | Input | BF16 / NZ | `[3*P,H]` | Joint query, key, and value projection weights. |
| `decay_projection_a_weight` | Input | BF16 / NZ | `[D,H]` | First decay projection, from hidden width to head width. |
| `decay_projection_b_weight` | Input | BF16 / NZ | `[P,D]` | Second decay projection, from head width to all head channels. |
| `beta_projection_weight` | Input | BF16 / NZ | `[N,H]` | Projects one beta logit per head. |
| `output_gate_projection_weight` | Input | BF16 / NZ | `[P,H]` | Projects the output gate logits. |
| `output_norm_weight` | Input | FP32 / ND | `[D]` | Per-channel RMSNorm weight shared by all heads. |
| `output_projection_weight` | Input | BF16 / NZ | `[H,P]` | Projects concatenated head outputs to hidden width. |
| `conv1d_weight` | Input | BF16 / ND | `[4,3*P]`, `[3,N,4,D]` or `[3,4,D]` | Four-tap causal convolution weights. |
| `conv_state` | Input/output | BF16 / ND | `[block_num,L,3*P]`, writable | Convolution history pool. |
| `checkpoint_state` | Input | FP32 / ND | `[block_num,N,D,D]`, read-only | Read-only initial recurrent state. |
| `replay_u` | Input/output | FP32 / ND | `[block_num,N,S,D]`, writable update vectors | Replay update vectors. |
| `replay_k` | Input/output | FP16 / ND | `[block_num,N,S,D]`, writable normalized keys | Replay normalized keys. |
| `replay_decay` | Input/output | FP32 / ND | `[block_num,N,S,D]`, writable exponentiated decay factors | Replay exponentiated decay factors. |
| `state_indices` | Input | int32 / ND | `[B]`, unique block indices | Shared physical block index for each request. |
| `conv_num_accepted_tokens` | Input | int32 / ND | `[B]`, Conv history counts in `[0,L-2]` | Selects the convolution history window for each request. |
| `A_log` | Input | FP32 / ND | `[N]`, log decay parameter | Log decay parameter per head. |
| `dt_bias` | Input | FP32 / ND | `[N,D]`, decay bias | Decay bias per head and channel. |
| `scale` | Input | FP32 scalar | Query scale, typically `D**-0.5` | Scales normalized queries. |
| `lower_bound` | Input | FP32 scalar | Decay bound in `[-5,0]` | Lower bound of the log decay gate. |
| `rms_norm_eps` | Input | FP32 scalar | Positive normal value, default `1e-5` | Epsilon for output RMSNorm. |
| `cu_seqlens` | Input | int32/int64 / ND | `[B+1]` prefix sums for TH, `None` for BSH | Cumulative token counts for TH. Use None for BSH. |
| `out` | Output | BF16 / ND | `[B,seq_len,H]` / `[total_tokens,H]` | Main output, with the same shape as the input. |

## Returns

The result is `(out, conv_state, replay_u, replay_k, replay_decay)`. `out` is BF16 with the shape of `hidden_states`. The other four entries alias their corresponding input pools.

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

#### Common

- **Single-parameter constraints** `hidden_states` is BF16/ND with hidden width 7168. BSH requires positive batch and sequence lengths. TH requires a positive token count. The total token count is below `2**31` in both layouts.

- **Presence constraints** The interface is for inference. Supply every input except `rms_norm_eps` and `cu_seqlens`, which are optional. Use ordinary tensors and call `.detach()` on Parameter objects first.

- **Consistency constraints** All tensors reside on the same NPU. The main output matches the input shape, layout, and dtype.

- **Cross-feature constraints** Convert the six projection weights to NZ before calling. Other tensors use ND. Input tensors outside the five state pools must be contiguous.

#### QKV projection

- **Single-parameter constraints** `qkv_projection_weight` is BF16/NZ with shape `[3*P,H]`, in Q, K, V order.

- **Cross-feature constraints** Its input width matches `H` in `hidden_states`. Its `3*P` output channels match the Conv weights and history pool.

#### Causal Conv1D

- **Single-parameter constraints** `conv1d_weight` accepts `[4,3*P]`, `[3,N,4,D]`, or `[3,4,D]` with four taps. `conv_state` is BF16/ND with shape `[block_num,L,3*P]` and `L >= 3`.

- **Consistency constraints** `conv_num_accepted_tokens` is an int32 tensor of shape `[B]` with values in `[0,L-2]`. Read three history rows starting at `max(count-1,0)`. Copy the last two rows to the start of the block and append the current raw QKV projections.

- **Cross-feature constraints** Select history blocks with `state_indices`. For a longest request of `max_seq_len` tokens, require `L >= max_seq_len + 2`.

#### Decay projection and gating

- **Single-parameter constraints** Both projection weights are BF16/NZ with shapes `[D,H]` and `[P,D]`, respectively. `A_log` and `dt_bias` are FP32/ND. `lower_bound` is finite FP32 in `[-5,0]`.

- **Consistency constraints** The first projection output width matches the second projection input width `D`. The second output is grouped as `[N,D]`, aligned by head with `A_log[N]` and `dt_bias[N,D]`.

#### Beta projection and query scaling

- **Single-parameter constraints** `beta_projection_weight` is BF16/NZ with shape `[N,H]`. `scale` must be representable as finite FP32, commonly `D**-0.5`. Q/K normalization uses a fixed `1e-6`.

- **Cross-feature constraints** `N` is 24, 12, or 6 for TP4, TP8, or TP16. All weights and state pools use the same `N` and `D=128`.

#### Paged recurrence and replay

- **Single-parameter constraints** `state_indices` is an int32 tensor of shape `[B]`. Its indices are distinct within a call and in `[0,block_num)`. Block 0 is valid. `checkpoint_state` is read-only.

- **Consistency constraints** The replay pools share shape `[block_num,N,S,D]`. They store FP32 update vectors, FP16 normalized keys, and FP32 exponentiated decay factors. Writes cover only valid token prefixes in the selected blocks, preserving other blocks, record tails, and inter-block gaps.

- **Cross-feature constraints** Conv, checkpoint, and the three replay pools share `block_num` and the physical block selected for each request. Each pool can have its own `stride(0)` of at least one block in elements. The remaining axes must be contiguous. The four writable pools require `requires_grad=False` and storage regions disjoint from other inputs. Replay capacity must satisfy `S >= max_seq_len`.

Compute the replay offset in elements of the corresponding pool's dtype.

```text
slot * replay.stride(0) + (head * S + token) * D + channel
```

#### Output normalization, gating, and projection

- **Single-parameter constraints** `output_norm_weight` is an FP32/ND tensor of shape `[D]`. Gate and output projection weights are BF16/NZ with shapes `[P,H]` and `[H,P]`. `rms_norm_eps` must be a positive normal FP32 value.

- **Presence constraints** The default `rms_norm_eps` is `1e-5`.

- **Cross-feature constraints** All heads share the RMSNorm weight. The gate projection aligns with the `P` concatenated channels, and the final projection restores width `H`.

#### Variable-length input

- **Single-parameter constraints** `cu_seqlens` is a contiguous int32 or int64 ND tensor of shape `[B+1]`.

- **Presence constraints** TH requires `cu_seqlens`. BSH uses `None`.

- **Consistency constraints** Prefix sums start at zero, increase strictly, and end at `total_tokens`. Each consecutive difference gives the corresponding request length.

- **Cross-feature constraints** The largest consecutive difference determines `max_seq_len` for Conv and replay capacity checks. TH reads prefix sums on the host and cannot be captured.

### Graph capture

Warm up compilation and workspace allocation before capturing BSH inputs. During capture the wrapper retains static checks and skips device-value checks and wrapper event waits and recording. Use valid block indices and Conv counts before replay. Manage the graph and shared workspace lifetimes and serialize their use. TH reads its prefix sums on the host and cannot be captured.

### Execution path and core quotas

| Implementation | Path | Matrix-core quotas | Vector cores per matrix core |
|---|---|---|---|
| `arch35` | Uses the specialized path for equal lengths of four or eight tokens when `B=b*TP` and `1 <= b <= 32`. Other valid inputs use the general path. | 8, 16, 24, 28, 32 | 2 |
| `arch92` | Uses the general path for BSH and TH. | 8, 16, 24, 32, 64 | 1 |

The quota comes from the active stream. Callers can specify it through stream resource limits.

Set `KDA_ARCH=arch92` to select the arch92 cases when running pytest or exporting cases. See the [case guide](../tests/ttk/README_en.md) for TTK cases and commands.

## Usage example

This BSH example uses `hidden_states` of shape `[2,4,7168]`. Prepare the weights and remaining inputs as specified in the parameter tables.

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
