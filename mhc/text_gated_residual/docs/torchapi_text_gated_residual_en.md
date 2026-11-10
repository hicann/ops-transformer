# text_gated_residual

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

## Description

- **API function**: Applies grouped RMS normalization, a low-rank transformation, and gated mixing to return `mixed`. Returns injection coefficients `inject` when `w_gate` is provided. Automatically binds first-order backward when gradient mode is enabled and any input requires gradients.

- **Formula**:

    Let the input be $x\in\mathbb{R}^{B\times S\times D}$, with $C$ channels and $H$ hidden features per channel. Here, $D=C\times H$, and $R$ is the low-rank dimension.

    $$
    \begin{aligned}
    v_{b,s,c,h} &= \frac{x_{b,s,c,h}}{\sqrt{\frac{1}{H}\sum_{j=0}^{H-1}x_{b,s,c,j}^{2}+\epsilon}},\\
    z_{b,s,c,h} &= v_{b,s,c,h}(1+\gamma_{cH+h}),\\
    a &= \operatorname{SiLU}\left((zW_{down}^{T})/C\right),\\
    u &= aW_{up}^{T},\\
    mixed_{b,s,h} &= \frac{1}{C}\sum_{c=0}^{C-1}z_{b,s,c,h}\,\sigma(u_{b,s,c,h}),\\
    inject &= 2\sigma\left((zW_{gate}^{T})/C\right).
    \end{aligned}
    $$

    Matrix multiplication flattens the last two dimensions of $z$ into $D$. $\sigma$ denotes Sigmoid, and $\operatorname{SiLU}(t)=t\sigma(t)$. The scale factor is 1 when `gamma` is omitted; `inject` is not computed when `w_gate` is omitted.

## Prototype

```python
text_gated_residual(
    x,
    w_down,
    w_up,
    w_gate=None,
    channels=4,
    gamma=None,
    eps=1e-6,
) -> Union[Tensor, Tuple[Tensor, Tensor]]
```

## Parameters

| Parameter | Parameter Type | Optional/Required | Description | Data Type | Shape |
| --- | --- | --- | --- | --- | --- |
| x | Tensor | Required | Input features in ND format, with $D=C\times H$. | float16, bfloat16 | [B, S, D] |
| w_down | Tensor | Required | Down-projection weight in ND format. Its dtype must match `x`. | Same as x | [R, D] |
| w_up | Tensor | Required | Up-projection weight in ND format. Its dtype must match `x`. | Same as x | [D, R] |
| w_gate | Tensor | Optional | Injection-branch weight in ND format. Its dtype must match `x`; the default is `None`. | Same as x | [C, D] |
| channels | int | Optional | Number of channels $C$. The default is 4. | int | - |
| gamma | Tensor | Optional | RMS scaling weight in ND format. Applies scaling with `1 + gamma`; the default is `None`. | float32 | [D] |
| eps | float | Optional | Numerical stability term for RMS normalization. The default is `1e-6`. | float | - |

## Return Value

Returns `mixed` when `w_gate` is omitted; returns `(mixed, inject)` when `w_gate` is provided.

| Parameter | Parameter Type | Optional/Required | Description | Data Type | Shape |
| --- | --- | --- | --- | --- | --- |
| mixed | Tensor | Required | Gated mixing result in ND format. | Same as x | [B, S, H] |
| inject | Tensor | Optional | Injection coefficients in ND format. Returned only when `w_gate` is provided. | Same as x | [B, S, C] |

## Constraints

- The API supports single-operator calls. TorchAir graph mode is not supported.
- `2 <= C <= 10`; `256 <= H <= 4096`, and `H` must be a multiple of 256; `64 <= R <= 896`, and `R` must be a multiple of 64.
- `x` must be a three-dimensional Tensor whose last dimension is divisible by `channels`. Input shapes must match the parameter table, and identically named dimensions must have equal sizes.
- The dtypes of `x`, `w_down`, `w_up`, and `w_gate` must match; all provided Tensors must reside on the same NPU device.
- `eps` must be finite, nonnegative, and representable as float32.
- The API supports gradients for a subset of inputs; not all inputs need `requires_grad=True`. Gradient computation also requires $R\le640$, $13H\le K$, $3H+8R\le K$, and $7H+6R\le K$. For Ascend 950PR&950DT products, use $K=62448-16\lceil C/8\rceil$; for Atlas A2 and Atlas A3 products, use $K=39664-16\lceil C/8\rceil$.
- `torch.no_grad()` executes only the forward pass. Higher-order differentiation is not supported.

## Determinism/Batch Consistency

- Determinism: Deterministic computation is supported by default.
- Batch consistency: The API does not guarantee batch consistency by default.

## Examples

- Single-operator mode:

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops.mhc.text_gated_residual import text_gated_residual

    torch_npu.npu.set_device(0)
    b, s, c, h, r = 1, 32, 4, 256, 64
    d = c * h
    x = torch.randn(b, s, d, dtype=torch.float16, device="npu", requires_grad=True)
    w_down = (torch.randn(r, d, dtype=torch.float16, device="npu") * 0.01).requires_grad_()
    w_up = (torch.randn(d, r, dtype=torch.float16, device="npu") * 0.01).requires_grad_()
    w_gate = (torch.randn(c, d, dtype=torch.float16, device="npu") * 0.01).requires_grad_()
    gamma = torch.zeros(d, dtype=torch.float32, device="npu", requires_grad=True)

    mixed, inject = text_gated_residual(
        x, w_down, w_up, w_gate=w_gate, channels=c, gamma=gamma,
    )
    (mixed.float().sum() + inject.float().sum()).backward()
    torch_npu.npu.synchronize()
    print(mixed.shape, inject.shape, x.grad.shape)
    ```

- TorchAir graph mode: Not supported.
