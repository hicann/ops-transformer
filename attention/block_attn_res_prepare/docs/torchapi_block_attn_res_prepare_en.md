# block_attn_res_prepare

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:13:35.074Z -->

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

- Computes the first stage of Attention Residuals historical residual attention. It produces the online Softmax state $(O,M,L)$ for all $S$ target slots, which the update stage combines with the current residual.

- Computation

  Let $V$ be the FLOAT32 representation of `block_res`, $Q$ be `pseudo_query`, $N_v=\min(\texttt{valid\_blocks[0]},N)$ be the number of valid historical blocks, and $\epsilon=\operatorname{FP32}(\texttt{eps})$.

  For $N_v>0$, compute root mean square normalization (RMSNorm) along the hidden dimension and the attention logits.

  $$
  \widehat V_{t,n}=\operatorname{RMSNorm}(V_{t,n},\mathbf{1},\epsilon),\qquad
  Z_{s,t,n}=\left\langle Q_s,\widehat V_{t,n}\right\rangle
  $$

  Compute the online Softmax state over the valid historical blocks.

  $$
  M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},\qquad
  E_{s,t,n}=\exp(Z_{s,t,n}-M_{s,t})
  $$

  $$
  L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},\qquad
  O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}
  $$

  - $T$, $N$, $S$, and $D$ are the token count, historical block capacity, target slot count, and hidden dimension. The corresponding indices are $t\in[0,T)$, $n\in[0,N_v)$, $s\in[0,S)$, and $d\in[0,D)$.
  - RMSNorm uses an all-ones scale vector. The scoring scale can be multiplied elementwise into the original query in advance to form $Q$.
  - $O$, $M$, and $L$ correspond to `numerator`, `logit_max`, and `exp_sum`. The weighted numerator $O$ has not been divided by $L$.

  For $N_v=0$, return the empty online Softmax state.

  $$
  O=\mathbf{0}_{S\times T\times D},\qquad
  M=m_{\min}\mathbf{1}_{S\times T},\qquad
  L=\mathbf{0}_{S\times T},\qquad
  m_{\min}=-3.4028234663852886\times10^{38}
  $$

  Here, $m_{\min}$ is the minimum finite FLOAT32 value.

## Function Prototype

```python
cann_ops_transformer.block_attn_res_prepare(
    block_res,
    valid_blocks,
    pseudo_query,
    *,
    eps=1.0e-6,
) -> Tuple[Tensor, Tensor, Tensor]
```

## Parameters

| Parameter | Type | Required/Optional | Description | Data Type | Dimensions (Shape) |
| --- | --- | --- | --- | --- | --- |
| block_res | Tensor | Required | Historical residual block, corresponding to $V$ in the formula. Data format: ND. $T$ is the number of tokens, $N$ is the number of historical residual blocks that can be stored, and $D$ is the hidden feature dimension. | torch.bfloat16, torch.float32 | [T, N, D] |
| valid_blocks | Tensor | Required | Historical residual block count in ND format. Zero denotes an empty history. Clipping the count to $N$ gives $N_v$. | torch.uint64 | [1] |
| pseudo_query | Tensor | Required | Pseudo Query, corresponding to $Q$ in the formula. Each target slot corresponds to one Query vector, and the last dimension must be the same as the last dimension of `block_res`. Data format: ND. | torch.float32 | [S, D] |
| eps | float | Optional | Stability term $\epsilon$ for RMS normalization, converted to FLOAT32 after range validation. Must be finite and in `[FLT_MIN, FLT_MAX]` (approximately `[1.1754943508222875e-38, 3.4028234663852886e+38]`); subnormal values are not supported. The default value is `1.0e-6`. | float64 | - |

## Return Values

Returns `(numerator, logit_max, exp_sum)`. All three tensors are newly allocated, contiguous, and on the input device.

| Parameter | Type | Required/Optional | Description | Data Type | Dimensions (Shape) |
| --- | --- | --- | --- | --- | --- |
| numerator | Tensor | Required | softmax weighted numerator, corresponding to $O$ in the formula. The data format is ND. | torch.float32 | [S, T, D] |
| logit_max | Tensor | Required | Maximum of the logits from the valid history, corresponding to $M$ in the formula. The data format is ND. | torch.float32 | [S, T] |
| exp_sum | Tensor | Required | Sum of exponents after max-shift, corresponding to $L$ in the formula. The data format is ND. | torch.float32 | [S, T] |

## Constraints

- Supports forward computation in training and inference.
- This API supports the single-operator call mode, but TorchAir graph mode calls are not yet supported.
- Shape requirements: `T >= 0`, `S >= 0`, `1 <= N <= 64`, `1 <= D <= 8192`.
- For non-empty computation (`T > 0` and `S > 0`), `T` and `S` must not exceed `4294967295` (`UINT32_MAX`), and `T * S` must not exceed `9223372036854775807` (`INT64_MAX`).
- All input tensors must be on the same device.
- `block_res.shape[2]` must be equal to `pseudo_query.shape[1]`.
- All input and output tensors must be contiguous.
- If `T == 0` or `S == 0`, the operator validates the arguments and infers the shapes, then returns empty outputs without launching the computation kernel. The `eps` range still applies.
- If `block_res` or `pseudo_query` contains `NaN` or `Inf`, or FP32 squaring, multiplication, or accumulation overflows, the result may contain `NaN` or `Inf`.

## Deterministic/Batch Consistency

- Deterministic computation: Deterministic computation is supported by default.

- Batch consistency is not supported by default.

## Examples

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  block_res = torch.tensor(
      [
          [
              [1.0, 2.0, 3.0, 4.0],
              [2.0, 0.0, -1.0, 1.0],
          ]
      ],
      dtype=torch.bfloat16,
      device="npu",
  )
  valid_blocks = torch.tensor([2], dtype=torch.uint64, device="npu")
  pseudo_query = torch.tensor(
      [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
      ],
      dtype=torch.float32,
      device="npu",
  )

  # eps is an optional parameter; if omitted, the default value 1.0e-6 is used.
  numerator, logit_max, exp_sum = cann_ops_transformer.block_attn_res_prepare(
      block_res,
      valid_blocks,
      pseudo_query,
  )
  torch_npu.npu.synchronize()

  print("numerator:", numerator)
  print("logit_max:", logit_max)
  print("exp_sum:", exp_sum)
  ```

- TorchAir graph mode call: currently not supported.
