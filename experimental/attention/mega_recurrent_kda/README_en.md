# mega_recurrent_kda

[简体中文](README.md)

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

`mega_recurrent_kda` performs KDA recurrence during inference. A single call runs input projections, causal Conv1D, SiLU, KDA recurrence, RMSNorm, output gating, and output projection. The operator supports equal-length BSH inputs and packed TH inputs.

### Input projections

For one request, let $x_t$ be the input column vector for token $t$ in this call, starting at $t=1$. $W_{\mathrm{qkv}}$ is the QKV weight. $W_a$ and $W_b$ are the two decay weights. $W_\beta$ and $W_z$ are the beta and output gate weights. The equations omit intermediate rounding.

$$
\begin{aligned}
p_t &= W_{\mathrm{qkv}}x_t,\\
a_t &= W_a x_t,\qquad f_t = W_b a_t,\\
b_t &= W_\beta x_t,\qquad z_t = W_z x_t.
\end{aligned}
$$

Here $x_t$ is a column vector of length $H$; $p_t$ is the QKV projection output (length $3P$), $a_t$ is the first decay projection output (length $D$), $f_t$ is the second decay projection output (length $P$), $b_t$ is the beta projection output (length $N$, one scalar per head), and $z_t$ is the output gate projection output (length $P$).

### Causal Conv1D and SiLU

The four weight vectors $w_0,\ldots,w_3$ (each of length $3P$) correspond to the four rows from the oldest history row to the current token. Let $s$ be the physical block for the request and $o=\max(\mathrm{count}-1,0)$, where $\mathrm{count}$ is `conv_num_accepted_tokens` for that request. Take $p_{-2},p_{-1},p_0$ from rows $o,o+1,o+2$ of block $s$ in `conv_state`, respectively. $\odot$ denotes elementwise multiplication and $\sigma$ denotes sigmoid.

$$
\begin{aligned}
c_t &= \sum_{j=0}^{3} w_j \odot p_{t-3+j},\\
(\hat q_t,\hat k_t,v_t)
&= \operatorname{Split}_{Q,K,V}\!\left(c_t\odot\sigma(c_t)\right).
\end{aligned}
$$

Here $p_{t-3+j}$ is the input vector (length $3P$) at window row $j$: indices $\le 0$ take the `conv_state` history rows above, and indices $\ge 1$ take the raw QKV projections of the corresponding tokens in this call, including the current token. $c_t$ is the convolution output vector (length $3P$), and $c_t\odot\sigma(c_t)$ is the SiLU activation. $\operatorname{Split}_{Q,K,V}$ divides the activated vector into three equal channel segments, giving the Q, K, and V vectors $\hat q_t$, $\hat k_t$, and $v_t$ (each of length $P$).

Split the result into Q, K, and V, then split each group into $N$ head vectors of length $D$.

### KDA recurrence

Read the initial state $M_{0,h}$ for head $h$ from the same physical block in `checkpoint_state`. $A_h$, $\delta_h$, and $\ell$ correspond to `A_log[h]`, `dt_bias[h]`, and `lower_bound`, respectively. $f_{t,h}$ is the decay projection vector for head $h$, and $b_{t,h}$ is its scalar beta projection.

$$
\begin{aligned}
q_{t,h} &= \frac{\mathrm{scale}\,\hat q_{t,h}}{\sqrt{\lVert\hat q_{t,h}\rVert_2^2+10^{-6}}},\\
k_{t,h} &= \frac{\hat k_{t,h}}{\sqrt{\lVert\hat k_{t,h}\rVert_2^2+10^{-6}}},\\
d_{t,h} &= \exp\!\left(\ell\,\sigma\!\left(e^{A_h}(f_{t,h}+\delta_h)\right)\right),\\
G_{t,h} &= M_{t-1,h}\operatorname{diag}(d_{t,h}),\\
u_{t,h} &= \sigma(b_{t,h})\left(v_{t,h}-G_{t,h}k_{t,h}\right),\\
M_{t,h} &= G_{t,h}+u_{t,h}k_{t,h}^{\mathsf T},\\
r_{t,h} &= M_{t,h}q_{t,h}.
\end{aligned}
$$

Here $\mathrm{scale}$ is the query scaling factor, corresponding to the `scale` argument; $q_{t,h}$ and $k_{t,h}$ are the normalized query and key vectors (each of length $D$), and the $10^{-6}$ in the denominator is a fixed constant unrelated to `rms_norm_eps`. $d_{t,h}$ is the per-channel decay factor vector (length $D$); $\operatorname{diag}(d_{t,h})$ is the $D\times D$ diagonal matrix with $d_{t,h}$ on its diagonal. $G_{t,h}$ is the decayed state matrix ($D\times D$); $u_{t,h}$ is the beta-gated innovation vector (length $D$); $M_{t,h}$ is the recurrent state matrix ($D\times D$), whose initial value $M_{0,h}$ comes from `checkpoint_state`; and $r_{t,h}=M_{t,h}q_{t,h}$ is the recurrent output vector of the head (length $D$).

The replay pools store $u_{t,h}$, the normalized $k_{t,h}$ in FP16, and the exponentiated decay factor $d_{t,h}$, respectively.

### Final output equation

Compute each recurrent output using the updated state $M_{t,h}$, then apply weighted RMSNorm and sigmoid gating. Concatenate the heads and apply the output projection. Let $\gamma$, $\epsilon$, and $W_o$ correspond to `output_norm_weight`, `rms_norm_eps`, and `output_projection_weight`, respectively. The value of `out` at token $t$ is

$$
\begin{aligned}
y_t &= W_o\,\operatorname{Concat}_{h=1}^{N}
\left[
  \frac{\gamma \odot (M_{t,h}q_{t,h})}
       {\sqrt{\lVert M_{t,h}q_{t,h}\rVert_2^2/D+\epsilon}}
  \odot \sigma\!\left((W_z x_t)_h\right)
\right].
\end{aligned}
$$

Here $y_t$ is the output column vector (length $H$) at token $t$, corresponding to the return value `out`; $\operatorname{Concat}_{h=1}^{N}$ concatenates the gated outputs of the $N$ heads in head order into a vector of length $P$; $\lVert M_{t,h}q_{t,h}\rVert_2^2/D$ is the sum of squared elements of the recurrent output $r_{t,h}=M_{t,h}q_{t,h}$ divided by $D$ (the mean square), and the whole denominator is the RMSNorm normalization term; $(W_z x_t)_h=z_{t,h}$ is the slice of the current token's output gate projection for head $h$.

Each logical request uses `state_indices` to select the same physical block in the Conv history, checkpoint, and three replay pools. Each pool can have its own stride along axis 0.

| State | Behavior |
|---|---|
| `checkpoint_state` | Supplies the initial recurrent state and stays read-only during the call. |
| `conv_state` | Updates convolution history and the current QKV projection in place. |
| `replay_u`, `replay_k`, `replay_decay` | Write the valid token prefix in place. Unselected blocks, unused record tails, and gaps between blocks are preserved. |

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

BSH supports graph capture after operator and workspace warm-up. Supply valid, distinct block indices and valid Conv counts before replay. The caller manages the shared workspace lifetime and serializes its use. TH reads sequence boundaries on the host and cannot be captured.

## Calling

Install the operator package and PyTorch extension built from this repository. Import the extension to register the operator, then call `torch.ops.cann_ops_transformer.mega_recurrent_kda`. The extension package name depends on the build configuration. The example uses the default package name.

### Equal-length BSH

Prepare BF16/ND `hidden_states` of shape `[2,4,7168]`, the six NZ projection weights, `conv1d_weight`, `output_norm_weight`, `A_log`, and `dt_bias`. The example allocates state pools and assigns blocks 5 and 0 to the two requests. The slices give the pools independent `stride(0)` values.

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
    rms_norm_eps=1e-5,
    cu_seqlens=None,
)
# The two requests write replay records to slots 5 and 0.
```

### Variable-length TH

Use the same weights. Pack one token from the first request and three from the second into `th_hidden_states` of shape `[4,7168]`. Pass the prefix sums on the same device. The example resets the four writable pools for an independent call.

```python
for pool in (conv_state, replay_u, replay_k, replay_decay):
    pool.zero_()
cu_seqlens = torch.tensor([0, 1, 4], dtype=torch.int32, device=device)

out, conv_state, replay_u, replay_k, replay_decay = torch.ops.cann_ops_transformer.mega_recurrent_kda(
    hidden_states=th_hidden_states,
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
    rms_norm_eps=1e-5,
    cu_seqlens=cu_seqlens,
)
```

See the [PyTorch API reference](docs/torchapi_mega_recurrent_kda_en.md) for the full interface constraints.

## Execution paths

| Source directory | Path selection |
|---|---|
| [arch35](op_dsl/arch35/) | Uses specialized S4 or S8 paths for eligible equal-length batches of four or eight tokens, and the general path for other valid inputs. |
| [arch92](op_dsl/arch92/) | Uses the general path for both BSH and TH inputs. |

TP4, TP8, and TP16 use 24, 12, and 6 heads per instance, respectively. For model batch `b`, the operator batch is `B=b*TP`. The specialized paths cover eligible equal-length inputs with `b` from 1 to 32. See [execution path and core quotas](docs/torchapi_mega_recurrent_kda_en.md#execution-path-and-core-quotas).
