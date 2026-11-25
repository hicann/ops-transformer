# MoonEP

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950DT series</term> | √ |
| <term>Ascend 950PR series</term> | √ |
| <term>Atlas A3 series</term> | × |
| <term>Atlas A2 series</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference series</term> | × |
| <term>Atlas training series</term> | × |

## Description

MoonEP is a communication library for EP (Expert Parallelism): dynamic redundant experts keep the token load on every rank perfectly balanced.

- Pipeline: `dispatch` performs planning first (producing `cu_seqlens`, `experts_to_copy`, `vtopk`, etc.) and then dispatches tokens;
  `prefetch_weight` pushes this rank's weights to every rank's slot; `combine` gathers the token outputs; `reduce_grad` accumulates replica gradients back to home.
- Plans can be reused across calls and between forward and backward; the return paths of `plan` / `prefetch_weight` / `reduce_grad` perform no host synchronization.
- All computation is integer; for fixed inputs the outputs are bit-exact (deterministic).
- `dispatch` / `combine` are carried by the `cann_ops_transformer` ElasticBuffer; the other three operators use the MoonEP private workspace.

`Buffer` wraps the four public interfaces `dispatch`, `prefetch_weight`, `combine`, and `reduce_grad`; planning happens inside `dispatch` and is not exposed separately.
A single `group` is used for both the MoonEP context and the ElasticBuffer; the backward `dispatch` reuses the forward plan by passing `plan=`.
Construction, parameters, and call examples are given below.

## Parameters

### Buffer Constructor

```python
Buffer(
    S, H, Hp, K, E, num_ep_ranks, group,
    num_sms=None, token_padding=128,
    *, weight_dtype="bf16", grad_dtype="bf16", slots=1,
)
```

| Parameter | Input/Output | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| `S` | Input | Tokens per rank | int | ND |
| `H` | Input | Hidden size, also the expert FFN input dimension | int | ND |
| `Hp` | Input | Expert FFN intermediate dimension; weight row width `W = H × Hp` | int | ND |
| `K` | Input | Top-k per token | int | ND |
| `E` | Input | Total number of experts in the EP group | int | ND |
| `num_ep_ranks` | Input | Required; number of EP ranks, range 2–128, must equal the `group` world size | int | ND |
| `group` | Input | Required; communication group shared by the MoonEP context and the ElasticBuffer | ProcessGroup | - |
| `num_sms` | Optional input | Reserved: number of AIV cores used by MoonEP kernels, range 1–the device AIV core count; currently only `None` is supported | int | ND |
| `token_padding` | Optional input | Reserved parameter, currently has no effect | int | ND |
| `weight_dtype` | Optional input | Weight data type used for specialization | str or torch.dtype | ND |
| `grad_dtype` | Optional input | Gradient data type used for specialization, defaults to `bf16` | str or torch.dtype | ND |
| `slots` | Optional input | Number of weight window slots; use 2 for layer pipelining | int | ND |

Construction is a collective call: it performs workspace registration and UbMem connection setup; the ElasticBuffer is created lazily on the first `dispatch` / `combine`.
All EP ranks must call it with the same parameters in the same order.

Interface constraints:

- `S >= 1`, `H >= 1`, `Hp >= 1`, and `W` must be a multiple of 64; `K >= 1`.
- `num_ep_ranks` must be in 2–128 and equal the `group` world size; `E <= 1024`, `E % R == 0` (R is `num_ep_ranks`).
- `group` is required and only accepts a `ProcessGroup`.
- `num_sms` is reserved and currently only accepts `None` (run on all device AIV cores); other values are rejected.
- `token_padding` is reserved and currently has no effect.
- `weight_dtype` accepts `bf16` / `fp16` / `e4m3fn` / `e5m2`; `grad_dtype` accepts `bf16` / `fp16` / `fp32`.
- `slots >= 1` and must be consistent along the whole chain.

### dispatch

```python
dispatch(hidden_sh, route_weights_sk=None, topk_experts_sk=None,
         tokens_per_expert=None, *, plan=None, do_cpu_sync=True)

# Forward: build the plan and dispatch
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
)
# Backward: reuse the forward plan
grad_expert_output_nvsh, _, _, _ = dispatch(grad_output_sh, plan=plan)
```

| Parameter | Input/Output | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| `hidden_sh` | Input | Forward token / backward output gradient, shape `(S, H)`; pass `(x_fp8, scales)` for FP8 passthrough | BFLOAT16, FLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2 | ND |
| `route_weights_sk` | Optional input | Forward routing weights, shape `(S, K)`, converted to fp32 internally | FLOAT32 | ND |
| `topk_experts_sk` | Optional input | Forward top-k expert indices, shape `(S, K)` | INT32 | ND |
| `tokens_per_expert` | Optional input | Forward per-expert counts, shape `(E,)` | INT32 | ND |
| `plan` | Optional input | Backward reuse of the forward plan; mutually exclusive with routing arguments | MoonepPlan | - |
| `do_cpu_sync` | Optional input | True (default) allocates the receive area exactly; False allocates the worst-case capacity `cap = R × S × min(K, 2·epn)` | BOOL | - |
| `hidden_nvsh` | Output | Receive area, shape `(cap, H)`; same dtype as the input; for FP8 it returns `(hidden_nvsh, scales)` with scales shape `(cap, H1)` | Same as input | ND |
| `route_weights_nvs` | Output | Routing weights of the received rows, shape `(cap,)`; None on the backward pass | FLOAT32 | ND |
| `cu_seqlens` | Output | Segment offsets of the forward plan (`plan.cu_seqlens`) | INT32 | ND |
| `plan` | Output | `MoonepPlan` created by the forward pass / returned unchanged by the backward pass | MoonepPlan | - |

Return shapes (forward and backward):

| Scenario | Return value |
| :--- | :--- |
| Forward | `hidden_nvsh (cap, H)`, `route_weights_nvs (cap,)` (`None` when routing weights are not passed), `cu_seqlens (2*epn,)`, and the new `plan` |
| Forward (FP8) | `(hidden_nvsh, scales)`: `hidden_nvsh (cap, H)`, `scales (cap, H1)`; the other return values are the same as forward |
| Backward (`plan=` passed) | `grad_expert_output_nvsh (cap, H)`, `None`, `plan.cu_seqlens (2*epn,)`, and the same `plan` |
| Backward (FP8, `plan=` passed) | `(grad_expert_output_nvsh, scales)`: `grad_expert_output_nvsh (cap, H)`, `scales (cap, H1)`; the other return values are the same as backward |

Value of `cap`:

- `do_cpu_sync=True` (default): exact allocation; `cap = S × K` (plan balance invariant: each rank receives exactly `S*K` expanded rows),
  i.e. the actual number of received rows, readable from dim 0 of the returned tensor.
- `do_cpu_sync=False`: worst-case allocation, `cap = R × S × min(K, 2·epn)` (the ElasticBuffer receive limit `A`, where `R = num_ep_ranks` and `epn = E / R`);
  valid rows are still `S*K`, given by `cu_seqlens`, while the tensor is reserved at the worst-case capacity.

The first dim of `hidden_nvsh` in the backward dispatch and in combine must equal the `cap` of the matching direction; valid rows are given by `cu_seqlens`, and trailing rows must be zero-filled as needed.

Interface constraints:

- The forward pass must pass `topk_experts_sk` and `tokens_per_expert`; `route_weights_sk` is optional. The backward pass must pass `plan=` and no routing arguments.
- A `hidden_sh` tensor must be exactly `(S, H)`; `route_weights_sk` / `topk_experts_sk` must be exactly `(S, K)`; tensors must reside on the NPU.
- FP8 passthrough: `hidden_sh=(x_fp8, scales)`, where `x_fp8` supports `float8_e4m3fn` / `float8_e5m2` and `scales` is contiguous `float32` / `float8_e8m0`, with shape `(S, H1)`; the output is also `(hidden_nvsh, scales)`.
- FP8 `H1` rules: `float32` scales use `ceil(H / 128)`; `float8_e8m0` scales use `Align2(ceil(H / 32))`. The scales returned by the receive side must keep the same dtype and `H1` mapping as the dispatch input.
- FP8 dispatch returns `(recv_x_fp8, recv_scales)`; expert computation/GMM must use `recv_scales` to dequantize. The `hidden_nvsh` passed to `combine` is the expert output in `bf16/fp16`, not a direct cast of the raw `x_fp8`.
- Internally `plan.vtopk` is used as `topk_idx`, with `num_experts = 2E` and per-rank capacity `nmtr = S`.
- `hidden_sh` must be contiguous (same rule as `combine`'s `hidden_nvsh`); both tensors in an FP8 tuple must be contiguous.

### MoonepPlan Structure

`MoonepPlan` is the planning handle returned by the forward `dispatch`. It is a `NamedTuple`: field order is fixed, it can be unpacked as a 5-tuple or accessed by attribute.

| Field | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `dst_n` | `(N,)` | INT32 | Zero placeholder; not computed on the vtopk-only fast path, always 0 |
| `cu_seqlens` | `(2*epn,)` | INT32 | Per-rank segment offsets; GMM splits expert segments by this |
| `experts_to_copy` | `(R, epn)` | INT32 | Replica list: row `r`, column `s` is the source expert id (range `[0, E)`) for rank `r`'s expert slot `s`; `-1` means the slot has no remote replica |
| `src_n` | `(N,)` | INT32 | Zero placeholder, same as `dst_n` |
| `vtopk` | `(S, K)` | INT32 | Virtual expert id (extended space `2E`), the routing basis for dispatch / combine |

Here `N = S × K`, `epn = E / R`, and `R = num_ep_ranks`.

- `experts_to_copy` is also the input of `prefetch_weight` and `reduce_grad`; `vtopk` is used directly as the `topk_idx` by `dispatch`.
- The return path performs no host synchronization: overflow and layout invariants are checked at the tail of the next MoonEP operator.
- `plan` can be reused across calls and between forward and backward; no additional same-direction `dispatch` may be inserted between a `dispatch` and its `combine`.

### prefetch_weight

```python
prefetch_weight(
    *, plan, local_gate_weight=None, local_up_weight=None,
    local_down_weight=None, local_gate_up_weight=None,
    local_gate_scales=None, local_up_scales=None, local_down_scales=None,
    local_gate_up_scales=None, async_finish=False, slot=0,
)

# 3-group form: returns the weight view tuple and the scale view tuple
(gate_pb, up_pb, down_pb), (gate_sb, up_sb, down_sb) = prefetch_weight(
    plan=plan,
    local_gate_weight=w1, local_up_weight=w3, local_down_weight=w2,
    local_gate_scales=s1, local_up_scales=s3, local_down_scales=s2,
)
# W13 interleaved form
(gate_up_pb, down_pb), (gate_up_sb, down_sb) = prefetch_weight(
    plan=plan, local_gate_up_weight=w13, local_down_weight=w2,
    local_gate_up_scales=s13, local_down_scales=s2,
)
```

| Parameter | Input/Output | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | Input | `MoonepPlan` returned by dispatch | MoonepPlan | - |
| `local_gate_weight` | Optional input | Gate home weight, shape `(epn, Wg)` or `(epn, Hp, H)`; required in the 3-group form | Constructor `weight_dtype` | ND |
| `local_up_weight` | Optional input | Up home weight, shape `(epn, Wg)` or `(epn, Hp, H)`; required in the 3-group form | Constructor `weight_dtype` | ND |
| `local_down_weight` | Input | Down home weight, shape `(epn, Wg)` or `(epn, Hp, H)`; required in both the 3-group and W13 forms | Constructor `weight_dtype` | ND |
| `local_gate_up_weight` | Optional input | W13 interleaved weight, 2W per expert row; mutually exclusive with gate/up | Constructor `weight_dtype` | ND |
| `local_gate_scales` | Optional input | Gate mxfp8 scales, shape `(epn, Sg)` or >=2-D `(epn, ..., Sg)` (leading dim epn; flattened row width a multiple of 32B); used in the 3-group form | FLOAT8_E8M0FNU | ND |
| `local_up_scales` | Optional input | Up mxfp8 scales, shape `(epn, Sg)` or >=2-D `(epn, ..., Sg)` (leading dim epn; flattened row width a multiple of 32B); used in the 3-group form | FLOAT8_E8M0FNU | ND |
| `local_down_scales` | Optional input | Down mxfp8 scales, shape `(epn, Sg)` or >=2-D `(epn, ..., Sg)` (leading dim epn; flattened row width a multiple of 32B); used in both forms | FLOAT8_E8M0FNU | ND |
| `local_gate_up_scales` | Optional input | W13 interleaved scales, shape `(epn, 2Sg)` or >=2-D, row width covers 2W; used only in the W13 form | FLOAT8_E8M0FNU | ND |
| `async_finish` | Optional input | True returns the launch stream event | BOOL | - |
| `slot` | Optional input | Window slot, `0 <= slot < slots` | int | - |
| `gate_pb` | Output | Gate weight view `[2*epn, Wg]`; returned in the 3-group form | Same as weights | ND |
| `up_pb` | Output | Up weight view `[2*epn, Wg]`; returned in the 3-group form | Same as weights | ND |
| `down_pb` | Output | Down weight view `[2*epn, Wg]`; returned in both forms | Same as weights | ND |
| `gate_up_pb` | Output | W13 interleaved weight view, row width 2W; returned only in the W13 form | Same as weights | ND |
| `scales` | Output | Scale view tuple mirroring `views`: `(gate_sb, up_sb, down_sb)` in the 3-group form, `(gate_up_sb, down_sb)` in the W13 interleaved form; empty tuple when not quantized | FLOAT8_E8M0FNU | ND |

Return shapes (`epn` experts per rank, `Wg` weight row width per group, `Sg` scale row width per group):

| Form | Synchronous return | `async_finish=True` |
| :--- | :--- | :--- |
| 3-group form | `(gate_pb[2*epn, Wg], up_pb[2*epn, Wg], down_pb[2*epn, Wg]), (gate_sb[2*epn, Sg], up_sb[2*epn, Sg], down_sb[2*epn, Sg])` | `((views, scales), event)` |
| W13 interleaved form | `(gate_up_pb[2*epn, 2Wg], down_pb[2*epn, Wg]), (gate_up_sb[2*epn, 2Sg], down_sb[2*epn, Sg])` | `((views, scales), event)` |

When not quantized (bf16/fp16), `scales` is an empty tuple `()`.

`scales` is a view tuple with the same structure as `views` (not the `local_*_scales` inputs);
element names follow the form (gate/up/down or gate_up/down), and each element matches the shape of the corresponding weight view
(`[2*epn, Sg]` in the 3-group form; `[2*epn, 2Sg]` for gate_up in the W13 form).

Interface constraints:

- The 3-group form (gate / up / down) and the W13 interleaved form (gate_up / down) are mutually exclusive; the W13 form does not accept separate gate/up arguments.
- The number of groups and each group's row width are fixed after the first call and must stay consistent; all groups share the constructor `weight_dtype`.
- mxfp8 contract in both directions: fp8 weights must be accompanied by scales, and passing scales requires fp8 weights;
  the scale type is fixed to `float8_e8m0fnu`, the leading dim is `epn`, and the flattened row width must be a multiple of 32B.
- `0 <= slot < slots`; the total width must not exceed the `3W` capacity reserved at construction.
  **Empty slots have undefined content** (equivalent to `torch.empty`: they may hold zeros or values left from a previous round);
  empty slots always have length 0 in `cu_seqlens` and are never computed.
- Return shapes are listed above; with `async_finish=True` the launch stream event is appended at the outermost level.

### combine

```python
combine(*, plan, hidden_nvsh, route_weights_nvs=None)

output_sh, gathered_route_weights_sk, _ = combine(
    plan=plan, hidden_nvsh=expert_output_nvsh,
    route_weights_nvs=route_weights_nvs,
)
grad_hidden_sh, _, _ = combine(plan=plan, hidden_nvsh=grad_hidden_nvsh)
```

| Parameter | Input/Output | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | Input | `MoonepPlan` used by the same-direction dispatch | MoonepPlan | - |
| `hidden_nvsh` | Input | Expert outputs, shape `(cap, H)`; trailing rows must be zero-filled up to cap | BFLOAT16, FLOAT16 | ND |
| `route_weights_nvs` | Optional input | Forward: pass the weights given to dispatch; backward: omit | FLOAT32 | ND |
| `output_sh` | Output | Result gathered in token-major order, shape `(S, H)` | Same as input | ND |
| `gathered_route_weights_sk` | Output | Gathered weights, shape `(S, K)`; None on the backward pass | FLOAT32 | ND |
| `_` | Output | Always None (the internal handle is not exposed) | - | - |

Return shapes (forward and backward):

| Scenario | Return value |
| :--- | :--- |
| Forward (`route_weights_nvs` passed) | `output_sh (S, H)`, `gathered_route_weights_sk (S, K)`, `None` |
| Backward (`route_weights_nvs` omitted) | `grad_hidden_sh (S, H)`, `None`, `None` |

Interface constraints:

- There is no `handle` argument: passing `route_weights_nvs` selects the forward pass; when omitted, the direction is inferred from the unconsumed dispatch handle,
  and an ambiguity error is raised if both forward and backward handles are pending.
  No additional same-direction dispatch may be inserted between a `dispatch` and its `combine`.
- the first dim of `hidden_nvsh` must equal the `cap` of the matching dispatch, and it must be a contiguous bf16/fp16 tensor;
  `route_weights_nvs` must be a contiguous 1D float32 tensor whose length equals the number of rows of `hidden_nvsh`.
- Forward and backward handles must not be mixed; when `route_weights_nvs=None`, `gathered_route_weights_sk` is None as well.

### reduce_grad

```python
reduce_grad(
    *, plan, local_gate_grad=None, local_up_grad=None,
    local_down_grad=None, local_gate_up_grad=None, async_finish=False,
)

# 3-group form
gate_rb, up_rb, down_rb = reduce_grad(
    plan=plan,
    local_gate_grad=dw1, local_up_grad=dw3, local_down_grad=dw2,
)
# W13 interleaved form
gate_up_rb, down_rb = reduce_grad(
    plan=plan, local_gate_up_grad=dw13, local_down_grad=dw2,
)
```

| Parameter | Input/Output | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | Input | `MoonepPlan` returned by dispatch | MoonepPlan | - |
| `local_gate_grad` | Optional input | Gate full-dW, `(2*epn, Wg)` or `(2*epn, Hp, H)`; required in the 3-group form | Constructor `grad_dtype` | ND |
| `local_up_grad` | Optional input | Up full-dW, `(2*epn, Wg)` or `(2*epn, Hp, H)`; required in the 3-group form | Constructor `grad_dtype` | ND |
| `local_down_grad` | Input | Down full-dW, `(2*epn, Wg)` or `(2*epn, Hp, H)`; required in both forms | Constructor `grad_dtype` | ND |
| `local_gate_up_grad` | Optional input | W13 interleaved full-dW, row width 2W; required only in the W13 form | Constructor `grad_dtype` | ND |
| `async_finish` | Optional input | True returns the launch stream event | BOOL | - |
| `gate_rb` | Output | Gate reduction result `(epn, Wg)`; returned in the 3-group form | Same as gradients | ND |
| `up_rb` | Output | Up reduction result `(epn, Wg)`; returned in the 3-group form | Same as gradients | ND |
| `down_rb` | Output | Down reduction result `(epn, Wg)`; returned in both forms | Same as gradients | ND |
| `gate_up_rb` | Output | W13 interleaved reduction result, row width 2W; returned only in the W13 form | Same as gradients | ND |

Return shapes (`epn` experts per rank, `Wg` gradient row width per group):

| Form | Synchronous return | `async_finish=True` |
| :--- | :--- | :--- |
| 3-group form | `(gate_rb[epn, Wg], up_rb[epn, Wg], down_rb[epn, Wg])` | `((gate_rb, up_rb, down_rb), event)` |
| W13 interleaved form | `(gate_up_rb[epn, 2Wg], down_rb[epn, Wg])` | `((gate_up_rb, down_rb), event)` |

Interface constraints:

- The gradients are 2 or 3 groups of full-dW: the first `epn` rows are home seeds and the last `epn` rows are replica dW; row order matches the `prefetch_weight` views.
- All groups share the constructor `grad_dtype`; `Wg` must be a multiple of 64.
- Precision bounds: replica gradients are accumulated on-chip in fp32 (16-bit gradients are converted to fp32 during the merge) and written back once in the original dtype — bf16 (7-bit mantissa) error <= 2^-8; fp16 (10-bit mantissa) error <= 2^-11; fp32 (23-bit mantissa) error <= (R-1)·2^-24, i.e. the rounding accumulation of R-1 fp32 additions. With fixed inputs the per-element addition sequence is constant, so outputs are bit-for-bit reproducible.
- Ordering: later than the last read of this layer's weights and earlier than the next layer's `prefetch_weight`; with `async_finish=True` the launch stream event is appended at the outermost level.

### Auxiliary Interfaces

- `barrier()`: group-wide symmetric-heap spin barrier (monotonically increasing epoch, flags ordered on the same channel as the data, without HCCL collectives). When it returns, all work previously launched on each rank's streams has completed. Typical uses: (1) confirming group readiness at phase switches or during debugging; (2) establishing cross-rank ordering before reusing MoonEP buffers that peers may read or write (the symmetric-heap window, `prefetch_weight` replica slots) — e.g. in multi-round weight prefetch, "this rank writes replica slots → `barrier()` → the next round's push" — and aligning the whole group before reusing or verifying windows on the `async_finish=True` path.
- `destroy(*, release_group=False, release_eb=False)`: idempotent release. The communication group is owned by the caller and is never destroyed here (`release_group` is kept for compatibility only);
  the ElasticBuffer is not destroyed by default; pass `True` to release it explicitly.
- Query properties: `rank`, `rank_size`, `epn`, `H`, `Hp`, `slots`, `weight_dtype`, `grad_dtype`, `layout_groups`,
  `heap_required_bytes`, `heap_bytes`, `destroyed`, plus the current values of the reserved parameters `num_sms` and `token_padding`.

## Constraints

- **Environment**: finish `dist.init_process_group(backend="hccl")` before construction;
  one process per card bound with `torch.npu.set_device(local_rank)`; at least 2 cards.
- **Communication domain**: `group` is required and is a single `ProcessGroup` shared by the MoonEP context and the ElasticBuffer.
  The data path does not consume the communication domain's external HCCL buffer; `hccl_buffer_size` is managed by the creator.
- **Scale and shapes**: `2 <= num_ep_ranks <= 128` and equal to the communication group world size; `E <= 1024`, `E % R == 0`, `S >= 1`, `K >= 1`; `S` must be identical on all ranks.
- **Data types**: `tokens_per_expert` / `topk_experts_sk` / `experts_to_copy` are int32;
  weights support bf16 / fp16 / e4m3 / e5m2 and must be consistent across groups;
  gradients support bf16 / fp16 / fp32 and must equal the constructor specialization;
  the tokens of `dispatch` / `combine` support bf16 / fp16.
- **Grouping and row width**: weights and gradients support either the 3-group form (commonly equal width `(W, W, W)`)
  or the W13 interleaved form (`(2W, W)`), with a total width cap of `3W` (`W = H × Hp`); `W` must be a multiple of 64.
- **Slots**: `slots >= 1`, consistent along the whole chain; `num_sms` and `token_padding` are reserved and currently only accept `None` and the default value respectively.
- **Lifetime**: tensors returned as views and by `plan` must be used before `destroy()`; `destroy` may be called repeatedly.

## Usage

### Communication Group Creation

```python
import torch.distributed as dist

ranks = list(range(dist.get_world_size()))
group = dist.new_group(
    ranks=ranks, backend="hccl", use_local_synchronization=True,
)
```

### Quick Start

```python
import torch
import torch.distributed as dist
import torch_npu
from moonep_buffer import Buffer

S, H, Hp, K, E = 4096, 7168, 2048, 8, 64
buf = Buffer(
    S, H, Hp, K, E, num_ep_ranks=dist.get_world_size(),
    group=group, slots=2,
)

hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buf.dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
)
# When not quantized the second return value is an empty tuple
(gate_pb, up_pb, down_pb), _ = buf.prefetch_weight(
    plan=plan,
    local_gate_weight=w1, local_up_weight=w3, local_down_weight=w2,
)
output_sh, gathered_route_weights_sk, _ = buf.combine(
    plan=plan, hidden_nvsh=expert_output_nvsh,
    route_weights_nvs=route_weights_nvs,
)
grad_expert_output_nvsh, _, _, _ = buf.dispatch(grad_output_sh, plan=plan)
grad_hidden_sh, _, _ = buf.combine(plan=plan, hidden_nvsh=grad_hidden_nvsh)
gate_rb, up_rb, down_rb = buf.reduce_grad(
    plan=plan,
    local_gate_grad=dw1, local_up_grad=dw3, local_down_grad=dw2,
)
buf.destroy()
```

The full set of weight forms (3-group / W13 / fp16 / FP8) is covered by the sections above.

### Testing and Performance

This repository ships the operator implementation and documentation only;
integration and performance tests are maintained in the source repository's internal suite
and must run per file in separate processes on device.
