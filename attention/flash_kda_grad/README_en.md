# FlashKDAGrad

## Product Support

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Function: Computes the training backward pass of [Kimi Delta Attention (KDA)](../flash_kda/README_en.md), using forward caches, output gradient `do`, and optional final-state gradient `dht`. Returns gradients of Q, K, V, the initial state, gate, beta, and gate parameters. Supports variable-length sequences, grouped query attention (GQA), Q/K L2 normalization, and backward computation for fused activations.
- Formulas: For one value head of one sequence, let $\bar q_t$ and $\bar k_t$ be the Q/K values used by forward, $\ell_t$ the per-step log gate, and $b_t$ the activated beta. State $H_t$ uses `(Dv,Dk)` storage. The forward recurrence and loss are:

$$
\widetilde H_t=H_{t-1}\operatorname{diag}(\exp(\ell_t)),\quad
\Delta v_t=b_t(v_t-\widetilde H_t\bar k_t),\quad
H_t=\widetilde H_t+\Delta v_t\bar k_t^T,\quad
o_t=scale\cdot H_t\bar q_t
$$

$$
L=\sum_t\langle do_t,o_t\rangle+\langle dht,H_{last}\rangle
$$

Backward differentiates this loss with respect to the inputs and gate parameters, returning `(dq,dk,dv,dh0,dg,dbeta,dA,dBias)`. With Q/K normalization backward enabled, `dq/dk` correspond to Q/K before normalization. `dht=None` omits the final-state loss term. See the [PyTorch API reference](docs/torchapi_flash_kda_grad_en.md#function-description) for gate and beta activation formulas.

- Implementation: Tokens are processed in chunks of 64. The host selects a persistent or staged schedule using effective lengths, head counts, and available device cores. The default `disable_recompute=False` reconstructs seven caches inside the backward kernel; `True` consumes saved forward caches. Saved `chunk_states` use BFLOAT16; reconstructed state buffers and state gradients use FLOAT32.

## Parameters

This operator is implemented with CANNBot DSL. The table lists main backward parameters and returns in public PyTorch interface order. See the [PyTorch API reference](docs/torchapi_flash_kda_grad_en.md) for metadata-generation parameters, cache shapes, and complete conditional requirements.

| Parameter | Input/Output/Attribute | Description | Data Type | Format |
| --- | --- | --- | --- | --- |
| q | Input | Query tensor. Pass saved forward `q_norm` when Q/K normalization backward is enabled, or the original Q otherwise. | BFLOAT16 | ND |
| k | Input | Key tensor, with the same shape as q. Pass saved forward `k_norm` when Q/K normalization backward is enabled. | BFLOAT16 | ND |
| v | Input | Original value tensor consumed by forward. | BFLOAT16 | ND |
| g | Input | Gate input. With gate fusion, it is the pre-activation input and must be bfloat16. Without fusion, it is the processed per-step log gate and must be float32. | BFLOAT16, FLOAT32 | ND |
| beta | Input | Update-coefficient input. With sigmoid fusion, it is the pre-activation input and must be bfloat16. Without fusion, it is the activated coefficient and must be float32. | BFLOAT16, FLOAT32 | ND |
| do | Input | Upstream gradient of the forward output. A Tensor is required; pass a matching zero Tensor when differentiating only the final state. | BFLOAT16 | ND |
| dht | Optional input | Upstream gradient of the final state. Default: `None`, interpreted as a zero gradient. | FLOAT32 | ND |
| Aqk | Input | Forward causal QK matrix cache, including scale. Required, with no default; pass it by keyword. Neither cache mode reconstructs it. | BFLOAT16 | ND |
| Akk | Input | Block-local inverse matrix cache of the forward WY transformation. Required, with no default; pass it by keyword. | BFLOAT16 | ND |
| metadata | Optional input | Backward schedule. Default: `None`, but execution requires the result of `flash_kda_grad_metadata`. | INT32 | ND |
| initial_state | Optional input | Forward initial state. Default: `None`, meaning a zero initial state during recompute. A nonzero initial state must be supplied for reconstruction; its presence also controls whether dh0 is returned. | FLOAT32 | ND |
| chunk_states | Optional input | State at each chunk entrance. Default: `None`; required in saved-cache mode. | BFLOAT16 | ND |
| w | Optional input | WY-transformed key cache. Default: `None`; required in saved-cache mode. | BFLOAT16 | ND |
| u | Optional input | WY-transformed value cache. Default: `None`; required to preserve the full forward-cache contract in saved-cache mode. | BFLOAT16 | ND |
| v_new | Optional input | Value cache corrected using the chunk entrance state. Default: `None`; required in saved-cache mode. | BFLOAT16 | ND |
| qg | Optional input | Q cache incorporating cumulative gates, excluding scale. Default: `None`; required in saved-cache mode. | BFLOAT16 | ND |
| kg | Optional input | K cache incorporating cumulative gates. Default: `None`; required in saved-cache mode. | BFLOAT16 | ND |
| g_cumsum | Optional input | Cumulative log gates within each chunk. Default: `None`; required in saved-cache mode and cannot be replaced by raw g. | FLOAT32 | ND |
| q_rstd | Optional input | Saved inverse norm of Q. Default: `None`; required when Q/K normalization backward is enabled. | FLOAT32 | ND |
| k_rstd | Optional input | Saved inverse norm of K. Default: `None`; required when Q/K normalization backward is enabled. | FLOAT32 | ND |
| beta_sigmoid | Optional input | Saved activated beta. Default: `None`; required with beta sigmoid fusion and already includes the factor of 2 in negative-eigenvalue mode. | FLOAT32 | ND |
| A_log | Optional input | Gate parameter for each value head. Default: `None`; required with gate fusion. | FLOAT32 | ND |
| dt_bias | Optional input | Gate bias. Default: `None`, interpreted as a zero bias. | FLOAT32 | ND |
| cu_seqlens | Optional input | TND sequence boundaries. Default: `None`. Optional for the main backward call; when supplied, must match metadata. | INT32 | ND |
| cu_seqlens_cpu | Optional attribute | <ul><li>Reserved parameter.</li><li>Default: `None`; only `None` is supported.</li></ul> | INT list | - |
| seqused | Optional input | Effective BNSD/BSND lengths. Default: `None`. Optional for the main backward call; when supplied, must match metadata. | INT32 | ND |
| scale | Optional attribute | <ul><li>Q/output scale.</li><li>Default: `None`, using `1/sqrt(128)`. An explicit value must be finite and match forward.</li></ul> | FLOAT | - |
| use_qk_l2norm_in_kernel | Optional attribute | <ul><li>Whether to compute Q/K L2-normalization backward.</li><li>Default: `True`; accepts `True` or `False`.</li></ul> | BOOL | - |
| use_gate_in_kernel | Optional attribute | <ul><li>Whether to compute gate-activation backward.</li><li>Default: `True`; accepts `True` or `False`.</li></ul> | BOOL | - |
| use_beta_sigmoid_in_kernel | Optional attribute | <ul><li>Whether to compute beta-sigmoid backward.</li><li>Default: `True`; accepts `True` or `False`.</li></ul> | BOOL | - |
| allow_neg_eigval | Optional attribute | <ul><li>Whether to use `2*sigmoid(beta)`.</li><li>Default: `False`; accepts `True` or `False`. Requires beta sigmoid fusion when `True`.</li></ul> | BOOL | - |
| disable_recompute | Optional attribute | <ul><li>Whether to disable cache reconstruction.</li><li>Default: `False`. `False` rebuilds seven caches; `True` consumes saved caches.</li></ul> | BOOL | - |
| lower_bound | Optional attribute | <ul><li>Gate formula selector and lower bound.</li><li>Default: `None`. Any non-None value must be in `[-5,0)`. With gate fusion, `None` selects softplus and a non-None value selects bounded sigmoid.</li></ul> | FLOAT | - |
| layout | Optional attribute | <ul><li>Logical Q/K/V layout.</li><li>Default: `"TND"`; supports `"TND"`, `"BNSD"` and `"BSND"`. See [input layouts](docs/torchapi_flash_kda_grad_en.md#input-layouts) for g, beta and do.</li></ul> | STRING | - |
| dbeta_dtype | Optional attribute | <ul><li>Output dtype of dbeta.</li><li>Default: `None`, using `beta.dtype`. An explicit value must be `beta.dtype` or `torch.bfloat16`.</li></ul> | torch.dtype | - |
| dq | Output | Q gradient, including L2-normalization backward when enabled and the reduction over corresponding value heads for GQA. | BFLOAT16 | ND |
| dk | Output | K gradient, with the same normalization and GQA semantics as dq. | BFLOAT16 | ND |
| dv | Output | V gradient, with the shape and logical layout of v. | BFLOAT16 | ND |
| dh0 | Optional output | Initial-state gradient. Returns `None` when `initial_state=None`, and a float32 Tensor otherwise. | FLOAT32 | ND |
| dg | Output | Gate-input gradient, preserving the dtype and shape of g. Without fusion, it is the gradient of the per-step log gate. | BFLOAT16, FLOAT32 | ND |
| dbeta | Output | Beta-input gradient, with the shape of beta and the dtype selected by dbeta_dtype. Without fusion, it differentiates the activated coefficient. | BFLOAT16, FLOAT32 | ND |
| dA | Optional output | Gradient of A_log. Returns `None` when gate fusion is disabled. | FLOAT32 | ND |
| dBias | Optional output | Gradient of dt_bias, with the same shape. Returns `None` when gate fusion is disabled or `dt_bias=None`. | FLOAT32 | ND |

Gradients retain the logical layouts of their corresponding inputs. See [input layouts](docs/torchapi_flash_kda_grad_en.md#input-layouts) and [forward-cache layouts](docs/torchapi_flash_kda_grad_en.md#forward-cache-layouts) for tensor shapes and storage order.

## Constraints

- Currently, `Dk=Dv=128` and chunk size is 64. `B >= 1`, `S/T >= 0`, and head counts satisfy `1 <= Nqk <= Nv` and `Nv % Nqk == 0`. Backward supports empty sequences and tail chunks shorter than 64 tokens.
- `Aqk`, `Akk`, backward metadata, and auxiliary caches required by enabled fusions must match the forward specification. Saved-cache mode requires all seven caches. Regenerate metadata when layout, dimensions, or effective lengths change.
- For fixed-length inputs, g, beta, and do always place the sequence dimension before the head dimension. Caches for BSND inputs remain head-major. Preserve the shapes and storage order returned by forward.
- All tensors consumed by computation must be on the same NPU. Length tensors and metadata must be contiguous. See the [PyTorch API constraints](docs/torchapi_flash_kda_grad_en.md#constraints) for fusion switches, dtypes, length ranges, and output contiguity.

## Invocation

| Interface | Example | Description |
| --- | --- | --- |
| PyTorch API | [Autograd, saved-cache, and recompute examples](docs/torchapi_flash_kda_grad_en.md#examples) | Generate backward metadata with `cann_ops_transformer.flash_kda_grad_metadata`, then call [flash_kda_grad](docs/torchapi_flash_kda_grad_en.md). For regular training, [FlashKDA autograd](../flash_kda/docs/torchapi_flash_kda_en.md) manages caches and backward calls. |
| Direct DSL call | - | Call `flash_kda_grad_npu` or `flash_kda_grad_from_forward` in the [DSL implementation](op_dsl/flash_kda_grad.py). |

See the [PyTorch extension guide](../../torch_extension/README_en.md) for build and installation instructions. Compatible CANN, torch_npu, and CANNBot DSL environments are required.

## References

- [Unit tests and interface-contract tests](tests/ut/test_flash_kda_grad_ut.py)
- [TTK precision and performance guide](tests/assets/README_en.md)
- [FlashKDA forward operator](../flash_kda/README_en.md)
- [Chinese API reference](docs/torchapi_flash_kda_grad.md)
