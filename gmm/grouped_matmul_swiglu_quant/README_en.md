# GroupedMatmulSwigluQuant

## Applicable Products

| Product | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |
|<term>Kirin X90 processor products</term> | √ |
|<term>Kirin 9030 processor products</term> | √ |

## Description

- API function: This operator fuses GroupedMatmul, dquant, swiglu, and quant. For details, see the computation formula.
- Computation formula:

  - **Definition**:

    * **⋅** denotes matrix multiplication.
    * **⊙** denotes element-wise multiplication.
    * $\left \lfloor x\right \rceil$ denotes rounding x to the nearest integer.
    * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
    * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
  - **Inputs**:

    * $X∈\mathbb{Z_8}^{M \times K}$: Input matrix (left matrix). M is the total number of tokens and K is the feature dimension.
    * $W∈\mathbb{Z_8}^{E \times K \times N}$: Grouped weight matrix (right matrix). E is the number of experts, K is the feature dimension, and N is the output dimension.
    * $bias∈\mathbb{R}^{E \times N}$: FLOAT32 auxiliary matrix (`weightAssistMatrix`) for the A8W4 scenario. E is the number of experts and N is the output dimension. In the A8W8 scenario, pass in a null pointer.
    * $offset∈\mathbb{R}^{E  \times N}$: Offset for per-channel asymmetric dequantization. E is the number of experts and N is the output dimension.
    * $w\_scale∈\mathbb{R}^{E \times N}$: Per-channel scale factor of the grouped weight matrix (right matrix). E is the number of experts and N is the output dimension.
    * $x\_scale∈\mathbb{R}^{M}$: Per-token scale factor of the input matrix (left matrix). M is the total number of tokens.
    * $groupList∈\mathbb{N}^{E}$: Group index list in prefix-sum form.
  - **Outputs**:

    * $Q∈\mathbb{Z_8}^{M \times N / 2}$: Quantized output matrix.
    * $Q\_scale∈\mathbb{R}^{M}$: Quantization scale factor.
  - **Computation Process**

    - 1. Determine the tokens of the current group based on groupList[i], where $i \in [0,Len(groupList)]$.

      > Example: Assume groupList=[3,4,4,6], with counting starting from 0.
      >
      > The 0th right matrix `W[0,:,:]` corresponds to the tokens `x[0:3]` at index positions [0,3) (3-0=3 tokens in total), which correspond to `x_scale[0:3]`, `w_scale[0]`, `bias[0]`, `offset[0]`, `Q[0:3]`, and `Q_scale[0:3]`;
      >
      > The 1st right matrix `W[1,:,:]` corresponds to the tokens `x[3:4]` at index positions [3,4) (4-3=1 token in total), which correspond to `x_scale[3:4]`, `w_scale[1]`, `bias[1]`, `offset[1]`, `Q[3:4]`, and `Q_scale[3:4]`;
      >
      > The 2nd right matrix `W[2,:,:]` corresponds to the tokens `x[4:4]` at index positions [4,4) (4-4=0 tokens in total), which correspond to `x_scale[4:4]`, `w_scale[2]`, `bias[2]`, `offset[2]`, `Q[4:4]`, and `Q_scale[4:4]`;
      >
      > The 3rd right matrix `W[3,:,:]` corresponds to the tokens `x[4:6]` at index positions [4,6) (6-4=2 tokens in total), which correspond to `x_scale[4:6]`, `w_scale[3]`, `bias[3]`, `offset[3]`, `Q[4:6]`, and `Q_scale[4:6]`;
      >
      > Note: Parts not specified in groupList will not participate in the update.
      > For example, if groupList=[12,14,18] and the shape of X is [30,:],
      >
      > the shape of the first output Q is [30,:], where Q[18:,:] is not updated or initialized, and the data in it is the original data of the device memory space at allocation.
      >
      > Similarly, the shape of the second output Q is [30], where Q\_scale[18:] is not updated or initialized, and the data in it is the original data of the device memory space at allocation.
      >
      > That is, Q[:groupList[-1],:] and Q\_scale[:groupList[-1]] of the outputs are the valid data parts.

    - 2. Perform the following computation with the input parameters determined by the group:

      $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

      $C_{i,act}, gate_{i} = split(C_{i})$

      $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;where $Swish(x)=\frac{x}{1+e^{-x}}$

      > Note: The role of and requirements for the `bias` input vary by scenario: in the A8W4 scenario, you must pass in the FLOAT32 auxiliary matrix (`weightAssistMatrix`); in the A8W8 scenario, pass in a null pointer. It is not an INT32 matrix multiplication bias. The current version does not support $offset_{i}$; asymmetric quantization is not yet supported.

    - 3. Determine the quantization mode

      - When the quantization mode is symmetric quantization:

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left \lfloor \frac{S_{i}}{Q\_scale_{i}}\right \rceil $

      - When the quantization mode is asymmetric quantization: (not yet supported)

        $Q\_scale_{i} = \frac{max(S_{i})-min(S_{i})}{255}$

        $Q\_offset_{i} = -128 - \left \lfloor \frac{min(S_{i})}{Q\_scale_{i}}\right \rceil$

        $Q_{i} = \left \lfloor \frac{S_{i}}{ Q\_scale_{i} } + Q\_offset_{i}\right \rceil $

## Parameters

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|:---:|------|----------|:-----------:|
| x | Input | Left matrix $X$ in the formula. | INT8 | ND |
| weight | Input | Grouped weight matrix $W$ in the formula. Each INT32 value packs eight INT4 elements. | INT8, INT4<sup>1</sup>, and INT32<sup>1</sup> | ND/NZ |
| bias | Optional input | Auxiliary matrix (`weightAssistMatrix`) with shape `[E, N]`. You must provide it for A8W4. Pass nullptr for A8W8. | FLOAT | ND |
| offset | Optional input | Reserved offset for per-channel asymmetric dequantization. Pass nullptr. | FLOAT | ND |
| weightScale | Input | Quantization scale factor $w\_scale$ for the right matrix. | UINT64<sup>1</sup>, FLOAT, FLOAT16, and BFLOAT16<sup>2</sup> | ND |
| xScale | Input | Quantization scale factor $x\_scale$ for the left matrix. | FLOAT | ND |
| <abbr title="Length equals the number of experts. Elements are cumulative token counts.">groupList</abbr> | Input | Token ranges for the experts. | INT64 | ND |
| output | Output | Quantized result $Q$. | INT8 | ND |
| outputScale | Output | Quantization scale factor $Q\_scale$ for each token. | FLOAT | ND |
| outputOffset | Output | Reserved output. The WeightNZ path on Ascend 950 ignores it. Pass nullptr in other scenarios. | - | - |

- <term>Ascend 950PR&950DT products</term>:
  - Superscript <sup>1</sup> marks data types that these products do not support.
  - weight supports only the NZ format.
  - Both bias and offset must be nullptr.
- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>:
  - Superscript <sup>2</sup> marks data types that these products do not support.

---

## Constraints

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
   - N must not exceed 10240.
   - K must not exceed 65536.
- <term>Ascend 950PR&950DT products</term>:
  - These products support only A8W8 per-token/per-channel quantization.
   - N must be positive and a multiple of 64.

> For the constraints of aclnnGroupedMatmulSwigluQuant, see [aclnnGroupedMatmulSwigluQuant Constraints](docs/aclnnGroupedMatmulSwigluQuant_en.md#constraints).
>
> For the constraints of WeightNZ, see [aclnnGroupedMatmulSwigluQuantWeightNZ Constraints](docs/aclnnGroupedMatmulSwigluQuantWeightNZ_en.md#constraints).

---

## Invocation Instructions

| Invocation Method | Invocation Sample | Description |
|----------|----------|------|
| aclnn API | [test_aclnn_grouped_matmul_swiglu_quant](examples/test_aclnn_grouped_matmul_swiglu_quant.cpp) | Call the GroupedMatmulSwigluQuant operator using the two-stage [aclnnGroupedMatmulSwigluQuant](docs/aclnnGroupedMatmulSwigluQuant_en.md) API. |
| aclnn API | [test_aclnn_grouped_matmul_swiglu_quant_weight_nz](examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_weight_nz.cpp) | Call the GroupedMatmulSwigluQuantWeightNZ operator using the two-stage [aclnnGroupedMatmulSwigluQuantWeightNZ](docs/aclnnGroupedMatmulSwigluQuantWeightNZ_en.md) API. |
