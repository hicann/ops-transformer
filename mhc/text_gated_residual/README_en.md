# TextGatedResidual

## Applicable Products

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

- **Operator function**: Fuses grouped RMSNorm, low-rank down/SiLU/up projections, sigmoid mixing, and optional inject to produce mixed. Saves three backward caches when needBackward=true.

- **Formulas**:

### Grouped RMS Normalization

The input `x` has shape $(B,S,C\times H)$ and contains channel elements $x_{b,s,c,h}$. Grouped RMS normalization is defined as follows:

$$
invRms_{b,s,c}=\frac{1}{\sqrt{\frac{1}{H}\sum_{j=0}^{H-1}x_{b,s,c,j}^{2}+\epsilon}}
$$

$$
xNorm_{b,s,c,h}=x_{b,s,c,h}\cdot invRms_{b,s,c}\cdot(1+\gamma_{cH+h})
$$

The scale factor is 1 when `gamma` is absent. Do not add 1 to gamma before supplying it.

### Read Fusion Path

Flatten $xNorm$ to $(B,S,C\times H)$ and compute:

$$
downPreAct_{b,s,r}=\sum_{d=0}^{C\times H-1}xNorm_{b,s,d}\,wDown_{r,d}
$$

$$
gate_{b,s,c,h}=\operatorname{sigmoid}\left(\sum_{r=0}^{R-1}\operatorname{SiLU}\left(\frac{downPreAct_{b,s,r}}{C}\right)\,wUp_{c\times H+h,r}\right)
$$

$$
mixed_{b,s,h}=\frac{1}{C}\sum_{c=0}^{C-1}gate_{b,s,c,h}\,xNorm_{b,s,c,h}
$$

### Backward Caches and Usage

Let D=C×H. The upPreActOptional cache is defined as follows:

$$
upPreAct_{b,s,d}=\sum_{r=0}^{R-1}\operatorname{SiLU}\left(\frac{downPreAct_{b,s,r}}{C}\right)\,wUp_{d,r}
$$

downPreActOptional contains the down-projection result before division by C. upPreActOptional contains the up-projection result before sigmoid. Both have the same dtype as x.
When needBackward=true, provide invRmsOptional, downPreActOptional, and upPreActOptional, regardless of whether wGateOptional is present. When needBackward=false, these outputs accept nullptr. Their contents are invalid and must not be read.

### Injection Path

When `wGateOptional` is not null, the injection path is defined as follows:

$$
inject_{b,s,c}=2\cdot\operatorname{sigmoid}\left(\frac{1}{C}\sum_{d=0}^{C\times H-1}xNorm_{b,s,d}\,wGate_{c,d}\right)
$$

### Symbols

Here, $B,S$ denote the batch size and sequence length. $C$, $H$, and $R$ denote the channel count, per-channel hidden dimension, and bottleneck dimension. The optional `gamma` contains original weights with shape $(C\times H)$; its scale factor is $1+\gamma_{cH+h}$.

When `needBackward=true`, the forward operation also returns `invRmsOptional`, `downPreActOptional`, and `upPreActOptional` for backward computation. When `wGateOptional` is not null, it returns `injectOptional`. The backward operation recovers injection gate values from `injectOptional / 2`.

## Parameters

<table style="table-layout: auto; width: 100%">
<thead><tr><th>Parameter</th><th>Input/Output/Attribute</th><th>Description</th><th>Data Type</th><th>Format</th></tr></thead>
<tbody>
<tr><td>x</td><td>Input</td><td>Forward residual input with shape (B,S,D), where D=C×H.</td><td>FLOAT16, BFLOAT16</td><td>ND</td></tr>
<tr><td>wDown</td><td>Input</td><td>Down-projection weights with shape (R,D) and the same dtype as x.</td><td>FLOAT16, BFLOAT16</td><td>ND</td></tr>
<tr><td>wUp</td><td>Input</td><td>Up-projection weights with shape (D,R) and the same dtype as x.</td><td>FLOAT16, BFLOAT16</td><td>ND</td></tr>
<tr><td>wGateOptional</td><td>Optional input</td><td>Injection weights with shape (C,D). No inject is produced when this input is absent.</td><td>FLOAT16, BFLOAT16</td><td>ND</td></tr>
<tr><td>gammaOptional</td><td>Optional input</td><td>Shape (D). The scale factor is 1+gamma, or 1 when this input is absent.</td><td>FLOAT32</td><td>ND</td></tr>
<tr><td>channels</td><td>Required attribute</td><td>Channel count C. C must be positive and divide D exactly.</td><td>INT64</td><td>-</td></tr>
<tr><td>eps</td><td>Optional attribute</td><td>RMS numerical stability parameter. It must be finite and non-negative. The default is 1e-6; ACLNN calls require an explicit value.</td><td>DOUBLE</td><td>-</td></tr>
<tr><td>needBackward</td><td>Optional attribute</td><td>Whether to save backward caches. The default is false; ACLNN calls require an explicit value.</td><td>BOOL</td><td>-</td></tr>
<tr><td>mixed</td><td>Output</td><td>Fused result with shape (B,S,H) and the same dtype as x.</td><td>FLOAT16, BFLOAT16</td><td>ND</td></tr>
<tr><td>injectOptional</td><td>Output</td><td>Injection weights with shape (B,S,C). Valid only when wGateOptional is not null.</td><td>FLOAT16, BFLOAT16</td><td>ND</td></tr>
<tr><td>invRmsOptional</td><td>Output</td><td>Per-channel RMS normalization coefficients with shape (B,S,C). Valid only when needBackward=true.</td><td>FLOAT32</td><td>ND</td></tr>
<tr><td>downPreActOptional</td><td>Output</td><td>Shape (B,S,R). Stores the down projection before division by C. Valid only when needBackward=true.</td><td>FLOAT16, BFLOAT16</td><td>ND</td></tr>
<tr><td>upPreActOptional</td><td>Output</td><td>Shape (B,S,D). Stores the up projection before sigmoid. Valid only when needBackward=true.</td><td>FLOAT16, BFLOAT16</td><td>ND</td></tr>
</tbody>
</table>

## Constraints

- B and S must be non-negative; D and R must be positive, with D=C×H. Computational inputs and outputs use matching dtypes. gamma and invRms use FLOAT32.
- Input limits: C is an integer with 2≤C≤10. The per-channel hidden dimension satisfies 256≤H≤4096 and H%256=0. The low-rank dimension satisfies 64≤R≤896 and R%64=0.
- x has shape [B,S,D], where D=C×H and H=D/C. wDown and wUp have shapes [R,D] and [D,R], respectively. Optional wGateOptional and gammaOptional have shapes [C,D] and [D].
- Provide injectOptional when wGateOptional is not null. Provide all three cache outputs when needBackward=true.

## Calling Instructions

| Calling Method | Sample | Description |
| --- | --- | --- |
| ACLNN API call | [test_aclnn_text_gated_residual](examples/test_aclnn_text_gated_residual.cpp) | Call the operator using the two-phase [aclnnTextGatedResidual](docs/aclnnTextGatedResidual_en.md) API. |
