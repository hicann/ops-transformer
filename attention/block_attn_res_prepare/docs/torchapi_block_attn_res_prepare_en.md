# block_attn_res_prepare

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:13:35.074Z -->

## Applicable Products

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

## Function

- **Description**: Completes the first phase of the two-phase computation of historical residual attention for Attention Residuals. Computes the inter-block attention for all $S$ target slots in parallel at a time, and returns the softmax weighted numerator and statistics $(O,M,L)$ for the next phase to fuse the current residual.

- **Formula**:

    $$
    \operatorname{block\_attn\_res\_prepare}
    \left(V,Q,\operatorname{valid\_blocks};\epsilon\right)
    \longrightarrow (O,M,L).
    $$

    For the $n$-th historical source of token $t$, first compute the RMS normalization factor:

    $$
    R_{t,n}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}V_{t,n,d}^{2}+\epsilon}.
    $$

    Compute logits for each slot, token, and valid history source:

    $$
    Z_{s,t,n}=R_{t,n}^{-1}\sum_{d=0}^{D-1}Q_{s,d}V_{t,n,d}.
    $$

    Let $N_v$ be the number of valid history sources specified by `valid_blocks`. Then:

    $$
    M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},
    $$

    $$
    E_{s,t,n}=\exp\left(Z_{s,t,n}-M_{s,t}\right),
    $$

    $$
    L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},
    $$

    $$
    O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}.
    $$

    If `valid_blocks[0] == 0`, the preceding softmax computation is not performed, and the following is returned directly:

    $$
    O=\mathbf{0}_{S\times T\times D},\qquad
    M=m_{\min}\mathbf{1}_{S\times T},\qquad
    L=\mathbf{0}_{S\times T},\qquad
    m_{\min}=-3.4028234663852886\times10^{38}.
    $$

    That is, `numerator` is an all-zero tensor of shape `[S, T, D]`, `logit_max` is a tensor of shape `[S, T]` whose elements are all the minimum finite value of float32 $m_{\min}$, and `exp_sum` is an all-zero tensor of shape `[S, T]`. This result represents the empty state of the online softmax.

    Where:

    - $V\in\mathbb{R}^{T\times N\times D}$ represents input `block_res`, which stores historical residual blocks; $V_{t,n,d}$ represents the value of the $t$-th token from the $n$-th historical source at the $d$-th hidden dimension.
    - $Q\in\mathbb{R}^{S\times D}$ represents input `pseudo_query`; $Q_{s,d}$ represents the value of the $s$-th target slot at the $d$-th hidden dimension.
    - $\operatorname{valid\_blocks}$ represents input `valid_blocks`; the minimum of its sole element and $N$ gives the number $N_v$ of valid history sources involved in the current computation.
    - $\epsilon$ represents input `eps`, a numerical stability term used to compute the RMS normalization factor.
    - $T$ represents the number of tokens, $N$ represents the capacity of the historical source buffer, $S$ represents the number of target slots, and $D$ represents the hidden dimension.
    - $t\in[0,T)$, $n\in[0,N_v)$, $s\in[0,S)$, and $d\in[0,D)$ represent the indexes of the token, valid history source, target slot, and hidden dimension, respectively.
    - $R_{t,n}$ represents the RMS normalization factor of the $n$-th valid history source of the $t$-th token, and $R_{t,n}^{-1}$ represents its reciprocal.
    - $Z_{s,t,n}$ represents the normalized attention logit between the $s$-th target slot, the $t$-th token, and the $n$-th valid history source.
    - $M_{s,t}$ represents the maximum of $Z_{s,t,n}$ over all valid history sources, used to perform the numerically stable max-shift.
    - $E_{s,t,n}$ represents the exponential value after the max-shift, that is $\exp(Z_{s,t,n}-M_{s,t})$.
    - $L_{s,t}$ represents the sum of exponentials over all valid history sources, and is the denominator used in the subsequent softmax normalization.
    - $O_{s,t,d}$ represents the softmax weighted numerator obtained by weighted summation of all valid history sources using $E_{s,t,n}$; $O$, $M$, and $L$ correspond to the return values `numerator`, `logit_max`, and `exp_sum`, respectively.
    - $\sum$, $\max$, and $\exp$ represent the summation, maximum, and natural exponential operations, respectively, and $\mathbb{R}$ represents the real number domain.

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

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| block_res | Tensor | Required | Historical residual block, corresponding to $V$ in the formula. Data format: ND. $T$ is the number of tokens, $N$ is the number of historical residual blocks that can be stored, and $D$ is the hidden feature dimension. | float32 | [T, N, D] |
| valid_blocks | Tensor | Required | Number of currently valid historical residual blocks $N_v$. Data format: ND. If the value is 0, `numerator` returns an all-zero Tensor, `logit_max` returns a Tensor in which all elements are the minimum finite value of float32, and `exp_sum` returns an all-zero Tensor. If the value is greater than $N$, it is treated as $N$. | uint64 | [1] |
| pseudo_query | Tensor | Required | Pseudo Query, corresponding to $Q$ in the formula. Each target slot corresponds to one Query vector, and the last dimension must be the same as the last dimension of `block_res`. Data format: ND. | float32 | [S, D] |
| eps | float | Optional | Numerical stability term for RMS normalization. Validated against the range below, then passed as a FLOAT32 attribute for computation. Must be finite and in `[FLT_MIN, FLT_MAX]` (approximately `[1.1754943508222875e-38, 3.4028234663852886e+38]`); subnormal values are not supported. The default value is `1.0e-6`. | float | - |

## Return Values

Returns a tuple of `numerator`, `logit_max`, and `exp_sum`. If `valid_blocks[0] == 0`, the three outputs form an empty state for online softmax: `numerator` is an all-zero Tensor, all elements of `logit_max` are the float32 minimum finite value, and `exp_sum` is an all-zero Tensor.

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| numerator | Tensor | Required | softmax weighted numerator, corresponding to $O$ in the formula. The data format is ND. | float32 | [S, T, D] |
| logit_max | Tensor | Required | Maximum of the logits from the valid history, corresponding to $M$ in the formula. The data format is ND. | float32 | [S, T] |
| exp_sum | Tensor | Required | Sum of exponents after max-shift, corresponding to $L$ in the formula. The data format is ND. | float32 | [S, T] |

## Constraints

- This API supports both training and inference use cases.
- This API supports the single-operator call mode, but TorchAir graph mode calls are not yet supported.
- Dimension constraints: `T >= 0`, `S >= 0`, `1 <= N <= 64`, and `1 <= D <= 8192`. When both `T > 0` and `S > 0`, `T` and `S` must not exceed `4294967295` (`UINT32_MAX`, the maximum unsigned 32-bit integer), and `T * S` must not exceed `9223372036854775807` (`INT64_MAX`, the maximum signed 64-bit integer).
- If `T == 0` or `S == 0`, empty outputs of the corresponding shapes are returned after parameter validation and shape inference, without launching AICore computation. The limits for non-empty computation above do not apply as checks on this path; the valid `eps` range and other API checks still apply.
- `block_res.shape[2]` must be equal to `pseudo_query.shape[1]`.
- All input and output Tensors must be contiguous. Non-contiguous Tensors are not supported.
- If `valid_blocks[0] == 0`, `numerator` returns an all-zero Tensor, `logit_max` returns a Tensor whose elements are all the float32 minimum finite value, and `exp_sum` returns an all-zero Tensor; if `valid_blocks[0] > N`, it is processed as `N`.
- If `block_res` or `pseudo_query` contains `NaN` or `Inf`, or FP32 squaring, multiplication, or accumulation overflows, the result may contain `NaN` or `Inf`.

## Deterministic/Batch Consistency

- Deterministic computation: Deterministic computation is supported by default.

- Batch consistency: Defaults to a Batch consistency implementation.

## Example

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
        dtype=torch.float32,
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
