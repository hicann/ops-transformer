# TextGatedResidualGrad

## Supported Products

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR & 950DT Series</term> | √ |
| <term>Atlas A3 Series</term> | √ |
| <term>Atlas A2 Series</term> | √ |
| <term>Atlas 200I/500 A2 Inference Products</term> | × |
| <term>Atlas Inference Series</term> | × |
| <term>Atlas Training Series</term> | × |

## Function

`TextGatedResidualGrad` is the backward operator of `TextGatedResidual`. It computes gradients of input `x` and weights `wDown`, `wUp`, `wGate`, and `gamma` from the upstream gradients of the mixed and injection outputs and tensors saved during the forward pass.

Let $B$ denote the batch size, $S$ the sequence length, $C$ the number of channels, $H$ the number of features per channel, $R$ the down-projection dimension, and $D=C\times H$. $\odot$ denotes element-wise multiplication, $@$ denotes matrix multiplication, and $\sigma$ denotes the sigmoid function.

The corresponding forward computation is:

$$
\begin{aligned}
\mathrm{invRms}_{b,s,c} &= \left(\frac{1}{H}\sum_{j=0}^{H-1}x_{b,s,c,j}^{2}+\epsilon\right)^{-1/2} \\
\mathrm{xNorm} &= x\odot\mathrm{invRms}\odot(1+\gamma) \\
\mathrm{downPreAct} &= \mathrm{xNorm}@\mathrm{wDown}^{T} \\
\mathrm{act} &= \mathrm{SiLU}(\mathrm{downPreAct}/C) \\
\mathrm{up} &= \mathrm{act}@\mathrm{wUp}^{T} \\
\mathrm{mixed}_{b,s,j} &= \frac{1}{C}\sum_{c=0}^{C-1}\mathrm{xNorm}_{b,s,c,j}\,\sigma(\mathrm{up}_{b,s,c,j}) \\
\mathrm{inject} &= 2\sigma\!\left((\mathrm{xNorm}@\mathrm{wGate}^{T})/C\right)
\end{aligned}
$$

The last dimension $D$ is interpreted as $[C,H]$ for normalization and summation across channels, and remains $D$ for matrix multiplication. `invRms` is broadcast across the $H$ features of each channel; `gamma` is broadcast across the $B$ and $S$ dimensions. $\epsilon$ is the parameter used for forward normalization. This interface receives the corresponding forward result through `invRms`.

Given `gradMixed` and `gradInject`, define

$$
L=\sum_{b,s,j}\mathrm{gradMixed}_{b,s,j}\,\mathrm{mixed}_{b,s,j}
 +\sum_{b,s,c}\mathrm{gradInject}_{b,s,c}\,\mathrm{inject}_{b,s,c}.
$$

The outputs of this operator are:

$$
dx=\frac{\partial L}{\partial x},\quad
dwDown=\frac{\partial L}{\partial wDown},\quad
dwUp=\frac{\partial L}{\partial wUp},\quad
dwGate=\frac{\partial L}{\partial wGate},\quad
dgamma=\frac{\partial L}{\partial\gamma}.
$$

When `wGate` is omitted, the injection branch and its gradients are excluded. When `gamma` is omitted, the normalization scale is 1 and `dgamma` is not produced. `invRms`, `inject`, `downPreAct`, and `up` must come from the same forward pass and correspond to the supplied input and weights.

## Parameters

| Name | Input/Output/Attribute | Description | Data Type | Format | Shape |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gradMixed | Input | Upstream gradient of the mixed output mixed. | FLOAT16, BFLOAT16 | ND | [B,S,H] |
| x | Input | Forward input. | FLOAT16, BFLOAT16 | ND | [B,S,D] |
| wDown | Input | Down-projection weight. | FLOAT16, BFLOAT16 | ND | [R,D] |
| wUp | Input | Up-projection weight. | FLOAT16, BFLOAT16 | ND | [D,R] |
| invRms | Input | Per-channel reciprocal root mean square saved during the forward pass. | FLOAT32 | ND | [B,S,C] |
| downPreAct | Input | Down-projection result xNorm@wDownᵀ saved during the forward pass. | FLOAT16, BFLOAT16 | ND | [B,S,R] |
| up | Input | Up-projection result act@wUpᵀ saved during the forward pass. | FLOAT16, BFLOAT16 | ND | [B,S,D] |
| gradInject | Optional input | Upstream gradient of the injection output inject. | FLOAT16, BFLOAT16 | ND | [B,S,C] |
| wGate | Optional input | Injection branch weight. | FLOAT16, BFLOAT16 | ND | [C,D] |
| gamma | Optional input | Normalization scale weight; the actual scale is 1+gamma. | FLOAT32 | ND | [D] |
| inject | Optional input | Injection output 2*sigmoid((xNorm@wGateᵀ)/C) saved during the forward pass. | FLOAT16, BFLOAT16 | ND | [B,S,C] |
| channels | Attribute | Number of channels C. | INT64 | - | Scalar |
| dx | Output | Gradient of x. | Same as x | ND | [B,S,D] |
| dwDown | Output | Gradient of wDown. | FLOAT16, BFLOAT16 | ND | [R,D] |
| dwUp | Output | Gradient of wUp. | FLOAT16, BFLOAT16 | ND | [D,R] |
| dwGate | Optional output | Gradient of wGate. | FLOAT16, BFLOAT16 | ND | [C,D] |
| dgamma | Optional output | Gradient of gamma. | FLOAT32 | ND | [D] |

## Constraints

- `C`, `H`, and `R` must be positive integers; both `H` and `R` must be multiples of 16.

The number of channels must satisfy `C ≤ 20`.

The range of `H` depends on the product. Define

<!-- npu="950" id7 -->
- <term>Ascend 950PR & 950DT Series</term>:

  $$
  K = 62448-16\left\lceil C/8\right\rceil
  $$

<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A2 Series</term>, <term>Atlas A3 Series</term>:

  $$
  K = 39664-16\left\lceil C/8\right\rceil
  $$

<!-- end id8 -->

`H` and `R` must satisfy all of the following:

$$
\boxed{
\begin{aligned}
13H &\le K,\\
3H+8R &\le K,\\
7H+6R &\le K.
\end{aligned}
}
$$

Therefore, for a given `C`, the upper bound of `H` is

$$
H_{\max}=16\left\lfloor\frac{K}{208}\right\rfloor.
$$

`R` must satisfy both the inequalities above and `R ≤ 640`.

Here, $\lceil\cdot\rceil$ denotes rounding up and $\lfloor\cdot\rfloor$ denotes rounding down. These relationships apply to FLOAT16, BFLOAT16, and all valid combinations of optional parameters.

For example, `C=4, H=2560, R=320` is valid for all three supported product series. Other `C/H/R` combinations must satisfy the constraints above. Call the first-stage interface `aclnnTextGatedResidualGradGetWorkspaceSize` to validate parameters and allocate memory according to the returned `workspaceSize`.

- All FLOAT16/BFLOAT16 inputs and `dx`, `dwDown`, `dwUp`, and `dwGate` must have the same data type. Each weight gradient must have the same data type as its corresponding weight. `invRms`, `gamma`, and `dgamma` are FLOAT32.
- Non-contiguous input tensors are supported. Input and output shapes must match the parameter table, and dimensions with the same name must have the same size.
- `B=0` or `S=0` is supported: `dx` is empty, and all supplied non-empty weight gradients and `dgamma` are set to zero. Other parameters must still satisfy the constraints above.
- Output gradients are overwritten. They do not require initialization and are not accumulated into existing output tensor values.
- Computation is deterministic by default.

For ACLNN calls, optional parameters must satisfy the following requirements:

| Feature | When enabled | When omitted |
| :--- | :--- | :--- |
| Injection branch | `wGate`, `gradInject`, `inject`, and `dwGate` must all be supplied | All four must be `nullptr` |
| Normalization scaling | `gamma` and `dgamma` must both be supplied | Both must be `nullptr`; the scale is 1 |

The two groups of optional parameters are independent. `nullptr` omits the corresponding feature; a placeholder tensor with shape `[0]` cannot be used instead.

## Calling the Operator

| Method | Example | Description |
| :--- | :--- | :--- |
| aclnn call | [Interface example](./docs/aclnnTextGatedResidualGrad.md#调用示例) | Create input and output tensors, then call the two-stage interface to prepare and execute the computation. |
