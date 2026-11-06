# GroupedMatmulSwigluQuant

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

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

## Parameter Description

<table style="table-layout: auto; width: 100%">
<thead>
<tr>
<th style="white-space: nowrap">Parameter Name</th>
<th style="white-space: nowrap">Input/Output/Attribute</th>
<th style="white-space: nowrap">Description</th>
<th style="white-space: nowrap">Data Type</th>
<th style="white-space: nowrap">Data Format</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space: nowrap">x</td>
<td style="white-space: nowrap">Input</td>
<td style="white-space: nowrap">Left matrix, which is X in the formula.</td>
<td style="white-space: nowrap">INT8</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">weight</td>
<td style="white-space: nowrap">Input</td>
<td style="white-space: nowrap">Weight matrix, which is W in the formula.</td>
<td style="white-space: nowrap">INT8</td>
<td style="white-space: nowrap">ND / NZ</td>
</tr>
<tr>
<td style="white-space: nowrap">bias</td>
<td style="white-space: nowrap">Input</td>
<td style="white-space: nowrap">Auxiliary matrix for matrix multiplication (weightAssistMatrix), with the shape [E,N]. It must be passed in only in the A8W4 scenario; in the A8W8 scenario, pass in a null pointer.</td>
<td style="white-space: nowrap">FLOAT32</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">offset</td>
<td style="white-space: nowrap">Input</td>
<td style="white-space: nowrap">Offset for per-channel asymmetric dequantization, which is offset in the formula.</td>
<td style="white-space: nowrap">FLOAT32</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">weightScale</td>
<td style="white-space: nowrap">Input</td>
<td style="white-space: nowrap">Quantization factor of the right matrix, which is w_scale in the formula.</td>
<td style="white-space: nowrap">FLOAT, FLOAT16, BFLOAT16</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">xScale</td>
<td style="white-space: nowrap">Input</td>
<td style="white-space: nowrap">Quantization factor of the left matrix, which is x_scale in the formula.</td>
<td style="white-space: nowrap">FLOAT32</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">groupList</td>
<td style="white-space: nowrap">Input</td>
<td style="white-space: nowrap">Indicates the number of tokens participating in the computation in each group, which is groupList in the formula.</td>
<td style="white-space: nowrap">INT64</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">output</td>
<td style="white-space: nowrap">Output</td>
<td style="white-space: nowrap">Quantized output result, which is Q in the formula.</td>
<td style="white-space: nowrap">INT8</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">outputScale</td>
<td style="white-space: nowrap">Output</td>
<td style="white-space: nowrap">Quantization factor of the output, which is Q_scale in the formula.</td>
<td style="white-space: nowrap">FLOAT</td>
<td style="white-space: nowrap">ND </td>
</tr>
<tr>
<td style="white-space: nowrap">outputOffset</td>
<td style="white-space: nowrap">Reserved parameter</td>
<td style="white-space: nowrap">Reserved parameter for interface compatibility. It is not supported in the current version, and you must pass in a null pointer.</td>
<td style="white-space: nowrap">-</td>
<td style="white-space: nowrap">-</td>
</tr>
</tbody>
</table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

 - The length of the N axis cannot exceed 10240.
 - The length of the K axis cannot exceed 65536.

## Invocation Instructions

| Invocation Method | Invocation Sample | Description |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_grouped_matmul_swiglu_quant](examples/test_aclnn_grouped_matmul_swiglu_quant.cpp) | Invoke the [GroupedMatmulSwigluQuant](docs/aclnnGroupedMatmulSwigluQuant_en.md) operator through the API. |
