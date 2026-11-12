# FlashKDA

[中文](FlashKDA.md)

FlashKDA uses CANNBotDSL to implement fused Kimi Delta Attention prefill for both training forward and inference.

| Public entry | Result |
|---|---|
| `flash_kda` | Three outputs: `out, final_state, chunk_states`. |
| `flash_kda_fwd` | The same inputs and compute kernels, with 16 fixed output slots for callers that consume training caches. |
| `flash_kda_metadata` | An int32 schedule tensor consumed by either forward entry. |

The forward path uses `torch.is_inference_mode_enabled()` to select its mode. Use `torch.inference_mode()` for inference. With `torch.no_grad()` alone it still takes the training-forward path. Automatic backward is currently unsupported.

## Product support

| Product | FlashKDA support |
|---|---|
| Ascend 950PR | Supported. |
| Ascend 950DT | Supported by the `ascend950` target. The metadata producer selects DT-specific scheduling parameters. |
| Atlas A2 and A3 series, Atlas 200I/500 A2 inference products, other Atlas inference and training series | Outside the documented FlashKDA target. |

## Directory structure

| Path | Contents |
|---|---|
| `op_dsl/flash_kda.py` | Forward entries, kernel exports and template dispatch |
| `op_dsl/flash_kda_metadata.py` | Host template selection, AICPU metadata generation and decoding |
| `op_dsl/flash_kda_staged.py` | Staged execution over chunk groups using a shared workspace |
| `op_dsl/flash_kda_persistent.py` | Consecutive chunks of a sequence head processed on the same core |
| `op_dsl/flash_kda_validation.py` | Input, schedule and Meta/FakeTensor validation |
| `torch_extension/` | Torch schemas and dispatch registration |
| `tests/ut/`, `tests/ttk/` | pytest tests and the TTK adapter, reference implementations and CSV suites |

## Computation

The operator processes 64 tokens per chunk. It can fuse Q/K L2 normalization, gate activation and beta sigmoid. It supports grouped-query attention (GQA) and partial final chunks. Let $Q,K$ denote the optionally normalized inputs and $g,\beta$ the gates and weights after activation. Cumulative log decay restarts at each chunk.

$$
G_t=\sum_{i=0}^{t}g_i,\quad \gamma_t=\exp(G_t),\quad
\tilde K=\gamma\odot K,\quad \bar K=\gamma^{-1}\odot K,\quad
\tilde Q=\operatorname{scale}\cdot(\gamma\odot Q).
$$

The chunk-local triangular solve gives

$$
T_c=\operatorname{stril}\!\left(\operatorname{diag}(\beta)\tilde K\bar K^\top\right),\quad
A_c=(I+T_c)^{-1},\quad
U=A_c\operatorname{diag}(\beta)V,\quad
W=A_c\operatorname{diag}(\beta)\tilde K,
$$

$$
M_{qk}=\operatorname{tril}(\tilde Q\bar K^\top),\quad
K_{\mathrm{restored},t}=\exp(G_{\mathrm{last}}-G_t)\odot K_t.
$$

Here $\operatorname{stril}$ excludes the diagonal and $\operatorname{tril}$ includes it. The incoming state $H_c$ has shape $D_v\times D_k$, with $H_0$ initialized from `initial_state`.

$$
V_{\mathrm{new}}=U-WH_c^\top,\qquad
O_c=\tilde QH_c^\top+M_{qk}V_{\mathrm{new}},
$$

$$
H_{c+1}=H_c\operatorname{diag}(\gamma_{\mathrm{last}})
       +V_{\mathrm{new}}^\top K_{\mathrm{restored}}.
$$

`Aqk`, `Akk`, `w`, `u` and `v_new` correspond to $M_{qk}, A_c, W, U$ and $V_{\mathrm{new}}$. The `qg` cache holds $\gamma\odot Q$ before scaling, `kg` holds restored keys and `g_cumsum` holds $G$. State recurrence uses FP32. The metadata producer selects the staged or persistent template from the layout, lengths and training mode.

## Quick start

Install the Torch extension wheel and native package as described below. This example uses BNSD with full-length sequences.

```python
import torch
import torch_npu

from cann_ops_transformer import flash_kda, flash_kda_metadata

B, Nqk, Nv, S, D = 1, 1, 2, 65, 128
q = torch.randn(B, Nqk, S, D, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, Nv, S, D, device="npu", dtype=torch.bfloat16)
g = torch.randn_like(v)
beta = torch.randn(B, Nv, S, device="npu", dtype=torch.float32)
initial_state = torch.zeros(B, Nv, D, D, device="npu", dtype=torch.float32)
A_log = torch.zeros(Nv, device="npu", dtype=torch.float32)
dt_bias = torch.zeros(Nv, D, device="npu", dtype=torch.float32)

# Inference: explicitly enter inference_mode and generate inference metadata.
with torch.inference_mode():
    metadata = flash_kda_metadata(
        batch_size=B, nheads_qk=Nqk, nheads_v=Nv, seqlens=S,
        enable_grad=False, layout="BNSD",
    )
    out, final_state, chunk_states = flash_kda(
        q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
        layout="BNSD", output_final_state=True,
        return_intermediate_states=True,
    )

# Training: use the same flash_kda entry outside inference_mode.
metadata = flash_kda_metadata(
    batch_size=B, nheads_qk=Nqk, nheads_v=Nv, seqlens=S,
    enable_grad=True, layout="BNSD",
)
out, final_state, chunk_states = flash_kda(
    q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
    layout="BNSD", output_final_state=True,
)
assert chunk_states is None
```

## PyTorch interface prototypes

Import `cann_ops_transformer` to register the three entries below. Both forward entries require a sixth argument, `metadata`, with no default. Supply a valid schedule tensor for execution. The full forward entry always returns 16 slots. See [Outputs](#outputs) for their order and the conditions under which each is present.

### `flash_kda`

```text
torch.ops.cann_ops_transformer.flash_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    metadata: Optional[Tensor],
    initial_state: Optional[Tensor] = None,
    A_log: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    *,
    scale: Optional[float] = 0.08838834764831845,
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
```

### `flash_kda_fwd`

```text
torch.ops.cann_ops_transformer.flash_kda_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    metadata: Optional[Tensor],
    initial_state: Optional[Tensor] = None,
    A_log: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    *,
    scale: Optional[float] = 0.08838834764831845,
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[
    Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[Tensor], Optional[Tensor], Optional[Tensor]
]
```

### `flash_kda_metadata`

```text
torch.ops.cann_ops_transformer.flash_kda_metadata(
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    *,
    batch_size: int,
    nheads_qk: int,
    nheads_v: int,
    seqlens: int,
    headdim_qk: int = 128,
    headdim_v: int = 128,
    enable_grad: bool = False,
    layout: str = "TND",
) -> Tensor
```

## Inputs and attributes

Let `B` be the logical batch, `S` the padded storage length, `T` the packed token count, and `Nqk` and `Nv` the Q/K and value head counts. Both head dimensions are 128.

| Input | BNSD shape | Type and semantics |
|---|---|---|
| `q`, `k` | `[B,Nqk,S,128]` | Required BF16 query and key. |
| `v` | `[B,Nv,S,128]` | Required BF16 value. |
| `g` | Same as `v` | BF16 or FP32 raw gate or preprocessed log decay. FP32 input is converted to BF16. |
| `beta` | `[B,Nv,S]` | BF16 or FP32 logits or preprocessed beta. The fused sigmoid uses FP32 operands. |
| `metadata` | `[M]` | The sixth parameter has annotation `Optional[torch.Tensor]` and no default. Supply valid int32 metadata for execution. Omitting it from a direct DSL call raises `TypeError`. Passing `None` raises `ValueError`. |
| `initial_state` | `[B,Nv,128,128]` | Optional FP32 state. The default is an all-zero state. |
| `A_log` | `[Nv]` | FP32, required when gate activation is fused. |
| `dt_bias` | `[Nv,128]` | Optional FP32 gate bias. The default is zero. |
| `cu_seqlens` | `[B+1]` | int32 cumulative lengths, required for TND. The first element is zero and the last is `T` in TND. |
| `seqused` | `[B]` | Optional int32 valid lengths. If both length tensors are provided, values must match adjacent differences in `cu_seqlens`. |

BSND shapes for Q/K, V/g and beta are `[B,S,Nqk,128]`, `[B,S,Nv,128]` and `[B,S,Nv]`. The corresponding TND shapes are `[T,Nqk,128]`, `[T,Nv,128]` and `[T,Nv]`.

| Keyword | Default | Meaning |
|---|---|---|
| `scale` | `128 ** -0.5` | Finite scale. Passing `None` selects the head-dimension scale. |
| `output_final_state` | `False` | Return the final state in either mode. |
| `return_intermediate_states` | `False` | Return incoming chunk states during inference. |
| `use_qk_l2norm_in_kernel` | `True` | Normalize Q/K and expose their training caches in the full forward result. |
| `use_gate_in_kernel` | `True` | Activate the raw gate. With preprocessed gates, `A_log`, `dt_bias` and `lower_bound` are ignored. |
| `use_beta_sigmoid_in_kernel` | `True` | Activate raw beta logits. |
| `allow_neg_eigval` | `False` | Double sigmoid beta. Requires fused beta sigmoid. |
| `disable_recompute` | `False` | Set to `True` to retain six recomputable forward caches in training. |
| `lower_bound` | `None` | For fused gates, `None` uses negative exponential times softplus. A value in `[-5,0)` selects bounded sigmoid. |
| `layout` | `"TND"` | BNSD, BSND or TND. |

### Generating metadata

Pass `batch_size`, `nheads_qk`, `nheads_v` and `seqlens` as keyword arguments. Here `seqlens` is the padded storage length `S` or packed total `T`. Use `enable_grad=True` for training and `False` for inference. Supply the same layout and length tensors to the producer and forward call. Omitted lengths in either padded layout mean all sequences use the full storage length. Both head dimensions default to 128.

The producer returns a one-dimensional int32 tensor on the current NPU. Reuse it for matching schedules. Regenerate it if the shape, layout, sequence lengths, mode, device or effective core count changes.

## Outputs

| Output | Type and shape | When present |
|---|---|---|
| `out` | BF16, same layout and shape as `v` | Always. Only valid token positions are defined. An unaligned tail can produce a noncontiguous view. |
| `final_state` | FP32 `[B,Nv,128,128]` | When requested. The input state is preserved. |
| `chunk_states` | FP32 `[B,J,Nv,128,128]` for BNSD/BSND or `[J,Nv,128,128]` for TND | When requested in inference. Each entry is the state entering a chunk. |

For padded layouts `J=ceil(S/64)`. For TND `J=ceil(T/64)+B`, and each sequence starts at `token_start//64 + logical_batch`. Only the slots occupied by valid chunks should be consumed.

`flash_kda_fwd` always returns the following tuple. Its first three slots match `flash_kda`.

```python
(out, final_state, chunk_states, Aqk, Akk, w, u, qg, kg, v_new,
 g_cumsum, q_norm, q_rstd, k_norm, k_rstd, beta_activated)
```

In inference the remaining 13 slots are `None`. Training returns `Aqk` and `Akk`. Setting `disable_recompute=True` also returns `w, u, qg, kg, v_new, g_cumsum`. The Q/K normalization and beta activation flags control their respective caches. Disabled slots retain their position and contain `None`.

## Constraints

Use contiguous inputs, `B >= 1`, positive sequence lengths and `1 <= Nqk <= Nv <= 96` with `Nv % Nqk == 0`. TND sequence offsets increase strictly. In padded layouts each valid length is in `[1,S]`. Supplied length tensors must be contiguous int32 tensors on the current NPU. Generate matching ABI 4 metadata with the accompanying producer. Eager execution reads back metadata and length tensors to validate the schedule. Meta/FakeTensor execution checks static properties.

## Building the Torch extension

From the repository root, activate the matching Python/CANN environment and install the base `cann_ops_transformer` package. Build and install the FlashKDA registration wheel.

```bash
(
    cd torch_extension
    TORCH_EXTENSION_EXPERIMENTAL=TRUE TORCH_EXTENSION_OPS=flash_kda TORCH_EXTENSION_VENDOR=flash_kda \
        python setup.py bdist_wheel
    python -m pip install --no-deps --force-reinstall dist/cann_ops_transformer_flash_kda-*.whl
)
```

The wheel contains the Python DSL and registration code. After `import cann_ops_transformer`, the `torch.ops.cann_ops_transformer` namespace exposes `flash_kda`, `flash_kda_fwd` and `flash_kda_metadata`. Both forward exports collect the same staged and persistent kernels.

## Building the native package

Load the CANN environment and set the shell variable `SOC_VERSION` to the `build.sh --soc` value for the target device. Run the following commands from the repository root to build and install the custom package in a separate OPP directory. On-device verification requires the matching base operator package and the Torch extension described above.

```bash
bash build.sh --pkg --experimental --ops=flash_kda --soc="${SOC_VERSION:?}"
WORK_DIR="$(mktemp -d)"
bash build/cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path="$WORK_DIR/opp"
source "$WORK_DIR/opp/vendors/custom_transformer/bin/set_env.bash"
export CANNBOTDSL_NATIVE_BINARY_MODE=require
export CANNBOTDSL_NATIVE_BINARY_REPORT="$WORK_DIR/hits.json"
```

With `CANNBOTDSL_NATIVE_BINARY_MODE=require`, run a forward call and inspect `counts` in `$WORK_DIR/hits.json`. A positive `installed_hit` and a zero or absent `installed_miss` indicate an installed binary hit.

## UT and TTK

From the repository root, run the pytest suite for the forward and metadata entries.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q \
    experimental/attention/flash_kda/tests/ut/
```

pytest checks the Python DSL return contract, Meta outputs and NPU forward values. The default forward accuracy tolerance is `atol=rtol=5e-3`. TTK uses separate functional, negative, generalized, redline and profiling suites. See the [TTK guide](../tests/ttk/README_en.md) for setup, criteria and examples.

For NPU graph capture, generate metadata outside the graph and warm up the same configuration before capture.

```python
metadata = flash_kda_metadata(**metadata_args)
flash_kda(**inputs, metadata=metadata)
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    outputs = flash_kda(**inputs, metadata=metadata)

inputs["v"].copy_(next_v)
graph.replay()
```

Keep the original metadata and length tensors alive and unchanged through the last replay. A change to sequence boundaries requires new metadata and a new capture.
