# EngramGate

English | [简体中文](./README.md)

## Applicable Products

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR/950DT products</term> | √ |
| <term>Atlas A3 training/inference products</term> | × |
| <term>Atlas A2 training/inference products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

EngramGate computes a residual gate in an Engram network. It uses the current residual features, Key, and fused weights to calculate a gate, then adds the gated Value shared by all heads of a token to the residual stream. Intermediate results needed for backward computation can also be returned.

For `hidden_states` of shape `[T, H, D]`, token `t` and head `h` use:

- $x=hidden\_states[t,h,:]$.
- $k=kv[t,hD:(h+1)D]$.
- $v=kv[t,HD:(H+1)D]$; all heads of the same token share this Value.
- $w=weight\_fused[h,:]$.

The outputs are:

$$
rstd_x=\frac{1}{\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}x_d^2+eps}},
\qquad
rstd_k=\frac{1}{\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}k_d^2+eps}}
$$

$$
raw\_dot=\sum_{d=0}^{D-1}x_dw_dk_d,
\qquad
normalized\_dot=\frac{raw\_dot\cdot rstd_x\cdot rstd_k}{\sqrt{D}}
$$

$$
signed\_root=\operatorname{copysign}
\left(\sqrt{\max\left(\left|normalized\_dot\right|,clamp\_value\right)},normalized\_dot\right)
$$

$$
gate\_score=\frac{1}{1+\exp(-signed\_root)},
\qquad
output[t,h,d]=\operatorname{cast}_{hidden\_states.dtype}
\left(x_d+gate\_score\cdot v_d\right)
$$

The `dot` output contains `raw_dot`, before normalization.

## Parameters

`T` is the number of tokens, `H` is the number of heads, and `D` is the feature dimension per head.

| Parameter | Input/Output/Attribute | Description | Data type | Format |
| :--- | :--- | :--- | :--- | :--- |
| hidden_states | Input | Residual features, shape `[T, H, D]`. | FLOAT16, BFLOAT16 | ND |
| kv | Input | Key and shared Value, shape `[T, (H+1)×D]`; each token stores H Keys followed by one shared Value. | Same as `hidden_states` | ND |
| weight_fused | Input | Fused weights, shape `[H, D]`. | FLOAT32 | ND |
| eps | Optional attribute | Positive, finite parameter in the RMS computation; defaults to `1e-6`. | FLOAT | - |
| clamp_value | Optional attribute | Positive, finite lower bound before the signed square root; defaults to `1e-6`. | FLOAT | - |
| save_for_backward | Optional attribute | Whether to return four intermediate results for backward computation; defaults to `true`. | BOOL | - |
| output | Output | Gated residual, same shape and dtype as `hidden_states`. | FLOAT16, BFLOAT16 | ND |
| dot | Optional output | Unnormalized weighted dot product `raw_dot`; shape `[T, H]` when `save_for_backward=true`, otherwise an empty tensor. | FLOAT32 | ND |
| gate_score | Optional output | Gate value; follows the `dot` shape rule. | FLOAT32 | ND |
| rstd_x | Optional output | Reciprocal RMS of `hidden_states` over D; follows the `dot` shape rule. | FLOAT32 | ND |
| rstd_k | Optional output | Reciprocal RMS of Key over D; follows the `dot` shape rule. | FLOAT32 | ND |

## Constraints

- `hidden_states` must be three-dimensional; `kv` and `weight_fused` must be two-dimensional.
- T is nonnegative, H is fixed at 4, and D must be a positive multiple of 64. An empty tensor is supported when T is 0; other H or D values are rejected.
- `hidden_states`, `kv`, and `output` must have the same dtype, either FLOAT16 or BFLOAT16. `weight_fused` and the four intermediate outputs are FLOAT32.
- The aclnn interface accepts noncontiguous inputs and converts them to contiguous tensors before execution. The PyTorch interface requires all three inputs to be contiguous.
- Input and output cannot alias the same device memory; in-place updates are unsupported.
- Multiplication, reduction, and gating use FLOAT32 internally; `output` is cast back to the input dtype.
- `aclnnEngramGate` uses deterministic computation without cross-core reductions or atomic accumulation.

## Calling Instructions

| Method | Sample | Description |
| :--- | :--- | :--- |
| aclnn API | [test_aclnn_engram_gate.cpp](./examples/test_aclnn_engram_gate.cpp) | Calls the operator through [aclnnEngramGate](./docs/aclnnEngramGate.md). |
| GE graph | [test_geir_engram_gate.cpp](./examples/test_geir_engram_gate.cpp) | Builds and executes a graph with the EngramGate GE IR prototype. |

See [EngramGate invocation samples](./examples/README_en.md) for build and run instructions.
