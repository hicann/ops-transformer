# FlashKDAGrad

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/950DT series</term>|      √     |
|<term>Atlas A3 training series</term>|      ×     |
|<term>Atlas A3 inference series</term>|      ×     |
|<term>Atlas A2 training series</term>|      ×     |
|<term>Atlas A2 inference series</term>|      ×     |

## Description

- Operator function: computes the backward output of Flash Kimi Delta Attention (KDA) in training scenarios.
  It takes the forward inputs, forward caches, the output gradient `do` and the final-state gradient `dht`, and
  returns the gradients of Q, K, V, the initial state, the gate, beta and the gate parameters. Variable-length
  sequences, GQA, Q/K L2 normalization, gate activation and beta sigmoid fusion are supported.

- Formulas: for one sequence and one value head, let $\bar q_t$, $\bar k_t$ be the Q/K used by attention,
  $\ell_t$ the per-step log gate, $b_t$ the activated beta, and store the state $H_t$ as `[Dv,Dk]`. The forward is:

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

  The backward treats `do` and `dht` as the upstream gradients of $o_t$ and of the final state, and returns:

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

  With Q/K L2 normalization enabled, `dq/dk` are the gradients of the pre-normalization Q/K.

- Implementation: the backward uses chunk decomposition and a reverse scan over state gradients, with BF16 Cube
  operands and FP32 state caches and state gradients. By default it consumes the seven caches saved by the forward
  (`disable_recompute=True`); with `disable_recompute=False` the same kernel first rebuilds the required caches and
  then runs the backward. The host automatically selects persistent or staged scheduling from the effective
  sequence lengths, the head counts and the number of device cores.

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 460px">
  <col style="width: 220px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>In/Out/Attr</th>
      <th>Description</th>
      <th>dtype</th>
      <th>Format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>q</td>
      <td>input</td>
      <td>Q in the formula. The normalized forward value when L2 normalization is enabled, otherwise the raw value.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>input</td>
      <td>K in the formula, same requirements as q.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>input</td>
      <td>V in the formula.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>g</td>
      <td>input</td>
      <td>Gate input. Raw gate input when the gate is fused, per-step log gate otherwise.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>input</td>
      <td>Update coefficient per token and value head. Pre-sigmoid input when fused, activated coefficient otherwise.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>do</td>
      <td>input</td>
      <td>Upstream gradient of the forward output o. Pass a zero Tensor of the same shape when absent; None is rejected.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dht</td>
      <td>input</td>
      <td>Upstream gradient of the forward final state. Pass a zero Tensor when absent; None is rejected.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>Aqk</td>
      <td>input</td>
      <td>Forward causal QK matrix cache, already scaled. Mandatory in both modes and never recomputed.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>Akk</td>
      <td>input</td>
      <td>Inverse of the unit lower-triangular matrix of the forward WY transform. Mandatory in both modes.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>input</td>
      <td>Backward scheduling information produced by flash_kda_grad_metadata; a forward metadata cannot substitute.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>initial_state</td>
      <td>input</td>
      <td>Forward initial state. Used in recompute mode; None means a zero state; unused in saved-cache mode.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>chunk_states</td>
      <td>input</td>
      <td>Chunk entry states. Mandatory in saved-cache mode; ignored and regenerated in recompute mode.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w</td>
      <td>input</td>
      <td>Key cache after the WY transform. Mandatory in saved-cache mode; regenerated in recompute mode.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>u</td>
      <td>input</td>
      <td>Value cache after the WY transform. Still required in saved-cache mode; the backward math does not read it.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_new</td>
      <td>input</td>
      <td>Value corrected with the chunk entry state. Mandatory in saved-cache mode; regenerated in recompute mode.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qg</td>
      <td>input</td>
      <td>Q cache combined with the cumulative gate, excluding scale. Mandatory in saved-cache mode.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kg</td>
      <td>input</td>
      <td>K cache combined with the cumulative gate. Mandatory in saved-cache mode.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>g_cumsum</td>
      <td>input</td>
      <td>Cumulative log gate inside a chunk. Mandatory in saved-cache mode; the raw g cannot substitute.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_rtsd</td>
      <td>input</td>
      <td>Inverse norm of the forward Q L2Norm. Mandatory with Q/K normalization on; never recomputed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_rstd</td>
      <td>input</td>
      <td>Inverse norm of the forward K L2Norm, same requirements as q_rtsd.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta_sigmoid</td>
      <td>input</td>
      <td>Activated beta from the forward. Mandatory with beta sigmoid fusion on; includes the factor 2 for negative eigenvalues.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>A_log</td>
      <td>input</td>
      <td>Gate parameter per value head. Mandatory when use_gate_in_kernel=True.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dt_bias</td>
      <td>input</td>
      <td>Gate bias per value head and Key channel. None means zero bias when the gate is fused.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens</td>
      <td>input</td>
      <td>Token prefix sums for TND. May be omitted on the main backward; when provided it must match the metadata.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused</td>
      <td>input</td>
      <td>Effective lengths for BSND/BNSD. May be omitted; when provided it must match the metadata.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>attr</td>
      <td>Output/Query scaling factor. None means 1/sqrt(Dk); an explicit value must be finite and match the forward.</td>
      <td>float, None</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_qk_l2norm_in_kernel</td>
      <td>attr</td>
      <td>Whether to compute the Q/K normalization backward. Default True.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_gate_in_kernel</td>
      <td>attr</td>
      <td>Whether to compute the gate activation backward. Default True.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_beta_sigmoid_in_kernel</td>
      <td>attr</td>
      <td>Whether to compute the beta sigmoid backward. Default True.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>allow_neg_eigval</td>
      <td>attr</td>
      <td>Whether 2*sigmoid(beta) is used. Default False; when True, beta sigmoid fusion must be on.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>disable_recompute</td>
      <td>attr</td>
      <td>Whether the seven saved caches are used. Default True requiring full caches; False enables recompute.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lower_bound</td>
      <td>attr</td>
      <td>Lower bound of the fused gate and the formula selection. None uses softplus; otherwise a value in [-5,0) selects bounded sigmoid.</td>
      <td>float, None</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>attr</td>
      <td>Layout of inputs and corresponding gradients. Default "TND"; "TND", "BNSD", "BSND" are supported.</td>
      <td>str</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dq</td>
      <td>output</td>
      <td>Q gradient. Summed over the corresponding value heads under GQA; includes the L2Norm backward when enabled.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dk</td>
      <td>output</td>
      <td>K gradient, same requirements as dq.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dv</td>
      <td>output</td>
      <td>V gradient.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dh0</td>
      <td>output</td>
      <td>Gradient of the initial state, always an FP32 V-first state; equals dht for an empty sequence.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dg</td>
      <td>output</td>
      <td>Gate input gradient, keeping the dtype and shape of g.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbeta</td>
      <td>output</td>
      <td>Beta input gradient, keeping the dtype and shape of beta.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dA</td>
      <td>output</td>
      <td>A_log gradient. None when gate fusion is off.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dBias</td>
      <td>output</td>
      <td>dt_bias gradient. None when gate fusion is off or dt_bias was not passed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The dtypes of q, k, v and do must match (BFLOAT16); state, state-gradient and matrix-cache dtypes are fixed as listed above.
- Fixed `Dk=Dv=128` and `C=64`. Chunk cache capacity: `Nc = ceil(T/C) + B` for TND and `Cmax = ceil(S/C)` for padded layouts.
- Shape constraints, taking TND, BNSD and BSND as examples:
  - B: number of logical sequences, `B >= 1`.
  - S: physical sequence length of BSND/BNSD.
  - T: total token count of TND, equal to the sum of all effective lengths.
  - Nqk: number of Q/K heads; Nv: number of V heads.
  - Dk, Dv: head dims of Q/K and V, both 128 currently.
- `1 <= Nqk <= Nv` and `Nv % Nqk == 0` (GQA); under GQA the Q/K inputs and gradients are stored per Nqk heads while
  the gate, beta, matrix and state caches are stored per Nv heads.
- `g` has the same shape as `v`, and `beta` has shape `v.shape[:-1]`; `dg` and `dbeta` follow the input shapes and dtypes.
- `do` and `dht` are mandatory; even when only one of the output or the final state is differentiated, the other
  must receive a shape- and type-matching zero Tensor.
- `Aqk`, `Akk` and `metadata` are mandatory in both modes; `disable_recompute=True` additionally requires
  `chunk_states`, `w`, `u`, `v_new`, `qg`, `kg` and `g_cumsum`.
- Fusion switches require their caches: `use_qk_l2norm_in_kernel=True` needs `q_rtsd`/`k_rstd`;
  `use_beta_sigmoid_in_kernel=True` needs `beta_sigmoid`; `use_gate_in_kernel=True` needs `A_log` (`dt_bias` is optional).
- `allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.
- `lower_bound` may only be None or a value in `[-5, 0)`; the backward entry rejects 0.
- For TND, `cu_seqlens` has length `B+1`, starts at 0, is non-decreasing and ends at T; for BSND/BNSD, `seqused` has
  length `B` with each item in `[0, S]`. The two length Tensors are mutually exclusive and must match the metadata.
- `metadata` is a contiguous 1-D INT32 Tensor of length `8 + 7*B*Nv`; forward and backward metadata have different
  ABIs and must be generated separately, and must be regenerated after dims, layout or effective lengths change.
  First use validates magic, ABI version, status code and input shapes; an invalid snapshot raises `ValueError`.
- Caches must come from the same forward and keep the shapes and storage order returned by the current forward;
  caches for BSND inputs remain head-major. In TND, chunk `j` of sequence `s` occupies slot
  `floor(cu_seqlens[s]/64) + s + j`.
- Empty sequences and tail chunks shorter than 64 tokens are allowed; an empty sequence passes `dht` to `dh0`, and
  invalid tokens in padded layouts write zeros into `dq/dk/dv/dg/dbeta`.
- Outputs are not guaranteed contiguous (for example `dbeta` may be a transposed view and `dA` a column view);
  downstream consumers that need contiguous tensors should call `.contiguous()` explicitly.

## Calling

| Method | Sample | Description |
|:---|:---|:---|
| Torch interface | [torchapi_flash_kda_grad.md](./docs/torchapi_flash_kda_grad.md) | Call through `cann_ops_transformer.flash_kda_grad` and `cann_ops_transformer.flash_kda_grad_metadata`; the document contains the full prototypes, parameters, returns and examples. |
| Direct DSL call | [op_dsl/flash_kda_grad.py](./op_dsl/flash_kda_grad.py) | Call the DSL implementation through `flash_kda_grad_npu` (main entry) and `flash_kda_grad_from_forward` (forward-cache adapter). |
| Unit tests | [tests/ut](./tests/ut) | Operator UT, exception contracts, and the performance/precision comparison entries. |
| TTK precision and performance | [tests/ttk/README.md](./tests/ttk/README.md) | STC case execution, dump/replay modes and the `cross_check L2` criteria. |

## References

- [Torch interface document (Chinese)](./docs/torchapi_flash_kda_grad.md)
- [Torch API document (English)](./docs/torchapi_flash_kda_grad_en.md)
- [TTK precision and performance guide](./tests/ttk/README.md)
