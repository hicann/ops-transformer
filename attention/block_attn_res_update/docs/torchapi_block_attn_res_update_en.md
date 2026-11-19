# block_attn_res_update

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:05:21.745Z -->

## Product Support

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

- Accumulates `delta` into `partial_block` in place, computes the normalized score of the updated residual, and combines it with the historical online Softmax state to produce `h`. Optional output RMSNorm is applied to the BFLOAT16 mixed result.

- Output RMSNorm is provided by [aclnnBlockAttnResUpdateV2](aclnnBlockAttnResUpdateV2_en.md), using `output_norm_weight` and the independent `output_norm_eps`.

- Computation

  For each token $t$, update the partial residual and write it back in place.

  $$
  p_t=P_t+\operatorname{FP32}(\Delta_t),\qquad
  P^{\mathrm{out}}_t=p_t
  $$

  Compute the current score using root mean square normalization (RMSNorm) and a query dot product.

  $$
  C_t=\left\langle Q,\operatorname{RMSNorm}(p_t,\mathbf{1},\epsilon)\right\rangle
  $$

  Merge the current residual with the historical online Softmax state.

  $$
  m_t=\max(M_t,C_t),\qquad
  \alpha_t=\exp(M_t-m_t),\qquad
  \beta_t=\exp(C_t-m_t)
  $$

  $$
  H^{(0)}_t=\operatorname{BF16}\left(
  \frac{\alpha_t O_t+\beta_t p_t}{\alpha_t L_t+\beta_t}
  \right)
  $$

  If `output_norm_weight` is `None`, return $H=H^{(0)}$. Otherwise, apply output RMSNorm to the BFLOAT16 mixed result.

  $$
  H_t=\operatorname{BF16}\left(
  \operatorname{RMSNorm}\left(\operatorname{FP32}(H^{(0)}_t),\operatorname{FP32}(W),\epsilon_{\mathrm{out}}\right)
  \right)
  $$

  Here, $W$ is `output_norm_weight` and $\epsilon_{\mathrm{out}}=\operatorname{FP32}(\texttt{output\_norm\_eps})$.

  - $T$ and $D$ are the token count and hidden dimension, with $t\in[0,T)$. The residuals, query, numerator, and output are $D$-dimensional vectors.
  - $P$ and $P^{\mathrm{out}}$ are the values of `partial_block` before and after the call. $\Delta$ is `delta`, $Q$ is `pseudo_query`, and $p$ is the FLOAT32 residual sum.
  - $O$, $M$, and $L$ are `numerator`, `logit_max`, and `exp_sum`. $H$ is `h`.
  - $\epsilon=\operatorname{FP32}(\texttt{eps})$. RMSNorm operates along $D$ and uses an all-ones scale vector for scoring. The scoring scale can be multiplied elementwise into the original query in advance to form $Q$.

## Function Prototype

```python
cann_ops_transformer.block_attn_res_update(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    *,
    eps=1e-6,
    output_norm_weight=None,
    output_norm_eps=1e-6,
) -> Tensor
```

## Parameters

| Parameter | Type | Required/Optional | Description | Data Type | Dimensions (Shape) |
| --- | --- | --- | --- | --- | --- |
| `partial_block` | Tensor | Required | Partial residual $P$, updated in place with the FLOAT32 sum $P^{\mathrm{out}}$. | torch.float32 | [T, D] |
| `delta` | Tensor | Required | Increment $\Delta$ added to the partial residual. | torch.bfloat16 | [T, D] |
| `pseudo_query` | Tensor | Required | Effective query $Q$ used to score the updated residual. | torch.float32 | [D] |
| `numerator` | Tensor | Required | Read-only weighted numerator $O$ of the historical online Softmax state. | torch.float32 | [T, D] |
| `logit_max` | Tensor | Required | Read-only maximum historical logit $M$ for each token. | torch.float32 | [T] |
| `exp_sum` | Tensor | Required | Read-only historical exponential sum $L$ for each token. | torch.float32 | [T] |
| `eps` | float | Optional | Stability term $\epsilon$ for residual scoring. Must be finite and positive. Defaults to `1e-6`. | float64 | - |
| `output_norm_weight` | Optional[Tensor] | Optional | Output RMSNorm scale $W$. Defaults to `None`, which disables output RMSNorm. | torch.bfloat16, torch.float32 | [D] |
| `output_norm_eps` | float | Optional | Stability term $\epsilon_{\mathrm{out}}$ for output RMSNorm. Defaults to `1e-6`. Must be in `[FLT_MIN, FLT_MAX]` when a weight is supplied and is ignored otherwise. | float64 | - |

## Return Values

| Parameter | Type | Required/Optional | Description | Data Type | Dimensions (Shape) |
| --- | --- | --- | --- | --- | --- |
| `h` | Tensor | Required | Contiguous output $H$ for the current layer. When `output_norm_weight` is supplied, $H$ contains the output RMSNorm result. | torch.bfloat16 | [T, D] |

## Constraints

- This API supports inference workloads.
- Single-operator mode and aclgraph mode are supported.
- All input and output tensors must be contiguous and on the same device.
- `T >= 0` and `0 <= D <= 8192`. `partial_block`, `delta`, `numerator`, and `h` have shape `[T, D]`. `pseudo_query` and a supplied `output_norm_weight` have shape `[D]`. `logit_max` and `exp_sum` have shape `[T]`.
- When `T == 0` or `D == 0`, parameter validation still applies and an empty output of the corresponding shape is returned. A supplied weight must still satisfy its dtype, shape, layout, and epsilon requirements.
- `eps` is passed to ACLNN as a double and converted to float32 for computation. It must be finite and positive. For non-empty workloads, the converted value must also be finite and positive.
- When `output_norm_weight` is supplied, `output_norm_eps` must be in `[FLT_MIN, FLT_MAX]` before float32 conversion. With `None`, output RMSNorm is disabled and `output_norm_eps` is ignored. Here, `FLT_MIN = 2^-126 ≈ 1.1754943508222875e-38` and `FLT_MAX ≈ 3.4028234663852886e38`. Subnormal epsilon values are unsupported because the default Sqrt FTZ mode flushes them to zero.
- For each token in a non-empty workload, `numerator` and `logit_max` must contain finite values, and `exp_sum` must contain finite, strictly positive values. All three inputs must describe the same non-empty historical online Softmax state and are read-only.
- Further underlying constraints are described in [aclnnBlockAttnResUpdateV2](aclnnBlockAttnResUpdateV2_en.md).

## Deterministic Computation

Defaults to a deterministic implementation.

## Examples

- Call in single-operator mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  T = 32
  D = 7168

  # Create inputs.
  partial_block = torch.rand((T, D), dtype=torch.float32).npu()
  delta = torch.rand((T, D), dtype=torch.bfloat16).npu()
  pseudo_query = torch.rand((D,), dtype=torch.float32).npu()
  # Construct O/M/L from one historical item with a zero value and a zero logit.
  numerator = torch.zeros((T, D), dtype=torch.float32).npu()
  logit_max = torch.zeros((T,), dtype=torch.float32).npu()
  exp_sum = torch.ones((T,), dtype=torch.float32).npu()

  output_norm_weight = torch.ones((D,), dtype=torch.float32).npu()

  # Call the single-operator API with output RMSNorm.
  h = cann_ops_transformer.block_attn_res_update(
      partial_block,
      delta,
      pseudo_query,
      numerator,
      logit_max,
      exp_sum,
      eps=1e-6,
      output_norm_weight=output_norm_weight,
      output_norm_eps=1e-6,
  )
  ```

- Call in aclgraph mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer


  # Define the network to be compiled.
  class OneOp(torch.nn.Module):
      def forward(
          self,
          partial_block,
          delta,
          pseudo_query,
          numerator,
          logit_max,
          exp_sum,
      ):
          return cann_ops_transformer.block_attn_res_update(
              partial_block,
              delta,
              pseudo_query,
              numerator,
              logit_max,
              exp_sum,
              eps=1e-6,
          )


  # Compile the network with the npugraph_ex backend.
  compiled_op = torch.compile(
      OneOp(),
      backend="npugraph_ex",
      fullgraph=True,
      dynamic=False,
      options={
          "static_kernel_compile": True,
      },
  )

  T = 8
  D = 7168

  # Create inputs.
  partial_block = torch.rand((T, D), dtype=torch.float32).npu()
  delta = torch.rand((T, D), dtype=torch.bfloat16).npu()
  pseudo_query = torch.rand((D,), dtype=torch.float32).npu()
  # Construct O/M/L from one historical item with a zero value and a zero logit.
  numerator = torch.zeros((T, D), dtype=torch.float32).npu()
  logit_max = torch.zeros((T,), dtype=torch.float32).npu()
  exp_sum = torch.ones((T,), dtype=torch.float32).npu()

  # Execute the compiled network.
  h = compiled_op(
      partial_block,
      delta,
      pseudo_query,
      numerator,
      logit_max,
      exp_sum,
  )
  ```
