# KdaInputProj

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :---------------------------- | :-----------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

- Operator function: This operator is the preprocessing operator of Recurrent KDA in the inference scenario. It projects the hidden state $X$ to obtain `qkv`, `beta`, `gate`, and `g`. The overall computation flow is as follows:

  ![KdaInputProj Flow](docs/figures/kda_input_proj_flow.png)

  - Stage 1: AIC and AIV execute `Matmul (beta/gate/g)` and `DynamicMxQuant` in parallel. The BlockMmad tasks of `Matmul (beta/gate/g)` are processed in parallel by all AIC cores to reduce computation rounds and improve utilization.
  - Stage 2: AIC and AIV execute `QuantMatmul (qkv)` and `Sigmoid` in parallel. The activation of `QuantMatmul` comes from the `DynamicMxQuant` output of Stage 1, and `weight_qkv` is an offline MX quantized weight.

- **Calculation formula**:

  beta / gate / g projection (Cube, BF16 input and FP32 accumulation):

  $$
  \mathbf{beta}_{raw} = X W_{\beta}^{\mathrm{T}},\quad
  \mathbf{gate} = X W_{gate}^{\mathrm{T}},\quad
  \mathbf{g} = X W_{g}^{\mathrm{T}}
  $$

  beta passes through Sigmoid (AIV, FP32 output):

  $$
  \mathbf{beta} = \sigma(\mathbf{beta}_{raw}) = \frac{1}{1 + e^{-\mathbf{beta}_{raw}}}
  $$

  qkv quantized matrix multiplication (the activation is DynamicMxQuant($X$)):

  $$
  \mathbf{qkv} = \mathrm{QuantMatmul}\bigl(\mathrm{DynamicMxQuant}(X),\; W_{qkv},\; \mathrm{weight\_qkv\_scale}\bigr)
  $$

  where $X$ is $[T,K]$ BF16; `gate` and `g` are BF16; `beta` is FP32.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1427px"><colgroup>
<col style="width: 194px">
<col style="width: 146px">
<col style="width: 721px">
<col style="width: 230px">
<col style="width: 136px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>Hidden layer input, corresponding to $X$ in the formula. The shape is $[T,K]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightQkv</td>
    <td>Input</td>
    <td>qkv projection weight $W_{qkv}$, the right-hand side (RHS) of the matmul. The shape is $[K,N_{qkv}]$. The K dimension must be the same as the K of x</td>
    <td>FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightBeta</td>
    <td>Input</td>
    <td>beta projection weight $W_{\beta}$. The shape is $[K,N_{beta}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightGate</td>
    <td>Input</td>
    <td>gate projection weight $W_{gate}$. The shape is $[K,N_{gate}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightG</td>
    <td>Input</td>
    <td>g projection weight $W_{g}$. The shape is $[K,N_{g}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightQkvScale</td>
    <td>Input</td>
    <td>MX quantization scale factor of weightQkv. The last dimension is fixed to 2 (packed high and low bits), and the MX block size is 64. When weightQkv is a transposed view, the shape is $[N_{qkv},\lceil K/64\rceil,2]$; otherwise it is $[\lceil K/64\rceil,N_{qkv},2]$</td>
    <td>FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qkvOut</td>
    <td>Output</td>
    <td>qkv projection output. The shape is $[T,N_{qkv}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>betaOut</td>
    <td>Output</td>
    <td>Output of beta after projection and Sigmoid. The shape is $[T,N_{beta}]$. The caller pre-allocates it</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gateOut</td>
    <td>Output</td>
    <td>gate projection output. The shape is $[T,N_{gate}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gOut</td>
    <td>Output</td>
    <td>g projection output. The shape is $[T,N_{g}]$</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

Dimension symbols: $T$ is the number of tokens; $K$ is the hidden dimension; $N_{qkv}$, $N_{beta}$, $N_{gate}$, and $N_{g}$ are the output feature dimensions of each path.

Typical shape example (referring to the model configuration): $x$ is `[T, 7168]`, `weight_qkv` is `[7168, 4608]`, `weight_beta` is `[7168, 12]`, `weight_gate` / `weight_g` is `[7168, 1536]`, and `weight_qkv_scale` is `[112, 4608, 2]`.

## Constraints

- Deterministic computation: aclnnKdaInputProj is deterministic by default.
- Shape constraints
  - $T$, $K$, $N_{qkv}$, $N_{beta}$, $N_{gate}$, and $N_{g}$ are all greater than 0.
  - The K dimension of each weight matrix must be the same as the K dimension of `x`.
  - `weightQkv`, `weightBeta`, `weightGate`, `weightG`, and `weightQkvScale` support only the transposed non-contiguous layout.
- Data type constraints
  - `x`, `weightBeta`, `weightGate`, `weightG`, `gateOut`, `gOut`, and `qkvOut`: BF16.
  - `weightQkv`: FLOAT8_E4M3FN.
  - `weightQkvScale`: FLOAT8_E8M0.
  - `betaOut`: FP32.

## Calling Instructions

| Calling Method | Sample Code / Documentation | Description |
| -------- | ---------------- | ---- |
| aclnn API | [test_aclnn_kda_input_proj.cpp](./examples/test_aclnn_kda_input_proj.cpp) | Called through the [aclnnKdaInputProj](docs/aclnnKdaInputProj_en.md) two-phase API; |
