# KdaInputProj Operator Design Description

## Product Support

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

KdaInputProj is the preprocessing operator of Recurrent KDA in the inference scenario. It performs the projection of qkv, beta, gate, and g. The overall computation flow is as follows:
![KdaInputProj Flow](figures/kda_input_proj_flow.png)

### Stage 1

In Stage 1, the KdaInputProj operator uses AIC and AIV to execute the two computation tasks `Matmul (beta/gate/g)` and `DynamicMxQuant` in parallel. All BlockMmad tasks of `Matmul (beta/gate/g)` are processed in parallel by all AIC units to reduce computation rounds and improve hardware utilization. The calculation formula is as follows:

$$
\begin{bmatrix}
\mathbf{beta}\\
\mathbf{gate}\\
\mathbf{g}\\
\end{bmatrix} = \mathbf{X}
\begin{bmatrix}
W_{beta}\\
W_{gate}\\
W_{g}
\end{bmatrix}
$$

### Stage 2

In Stage 2, the KdaInputProj operator uses AIC and AIV to execute the two computation tasks `QuantMatmul (qkv)` and `Sigmoid` in parallel. The activation of `QuantMatmul` is the output of the `DynamicMxQuant` task in Stage 1, and `qkv_weight` is an offline quantized weight.

## Interface Description

This operator is registered as `torch.ops.cann_ops_transformer.kda_input_proj` through `cann_ops_transformer`.

### PyTorch API Prototype

```python
cann_ops_transformer.kda_input_proj(
    x: Tensor,
    weight_qkv: Tensor,
    weight_beta: Tensor,
    weight_gate: Tensor,
    weight_g: Tensor,
    weight_qkv_scale: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]
```

## Parameter Description

Dimension symbol description:

- $T$: number of input tokens.

| Parameter Name | Input / Output | Data Type | Data Format | Dimension (shape) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `x` | Input | BFLOAT16 | ND | `[T, 7168]` | Hidden layer input. |
| `weight_qkv` | Input | FLOAT8_E4M3 | ND | `[7168, 4608]` | qkv projection weight, the matmul RHS `[K, N]`. |
| `weight_beta` | Input | BFLOAT16 | ND | `[7168, 12]` | beta projection weight, the matmul RHS `[K, N]`. |
| `weight_gate` | Input | BFLOAT16 | ND | `[7168, 1536]` | gate projection weight, the matmul RHS `[K, N]`. |
| `weight_g` | Input | BFLOAT16 | ND | `[7168, 1536]` | g projection weight, the matmul RHS `[K, N]`. |
| `weight_qkv_scale` | Input | FLOAT8_E8M0 | ND | `[112, 4608, 2]` | MX quantization scale factor of `weight_qkv`. |

## Constraints

### Single-Parameter Constraints

None.

### Existence Constraints

None.

### Consistency Constraints

None.

### Cross-Feature Constraints

None.

## Ascend 950PR&950DT Products Calling Example

```python
import torch
import torch_npu
import cann_ops_transformer

qkv, beta, gate, g = cann_ops_transformer.kda_input_proj(
    x, weight_qkv, weight_beta, weight_gate, weight_g, weight_qkv_scale=weight_qkv_scale
)
```
