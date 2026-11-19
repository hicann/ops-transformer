# GroupedMatmulSwigluQuantV2

English | [简体中文](./README.md)

## Product Support

| Product | Support |
| :---------------------------- | :-----------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

- Operator function: fuses grouped matrix multiplication, dequantization, SwiGLU activation, and quantization. In the MX A8W4 scenario, an expert-specific smoothing factor can be applied after activation and before quantization. For details, see the computation formula.
- Computation formula:
  - For <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
    <details>
    <summary>Quantization scenario A8W8 (A refers to the activation matrix, W refers to the weight matrix, and 8 refers to the INT8 data type):</summary>
    <a id="quantization-scenario-a8w8"></a>

      - **Definitions**:

        * **⋅** denotes matrix multiplication.
        * **⊙** denotes element-wise multiplication.
        * $\left \lfloor x\right \rceil$ denotes rounding x to the nearest integer.
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **Inputs**:

        * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        * $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        * $w\_scale∈\mathbb{R}^{E \times N}$: per-channel scaling factor of the grouped weight matrix (right matrix), where E is the number of experts and N is the output dimension.
        * $x\_scale∈\mathbb{R}^{M}$: per-token scaling factor of the activation matrix (left matrix), where M is the total number of tokens.
        * $grouplist∈\mathbb{N}^{E}$: grouped index list in cumsum or count mode.
      - **Outputs**:

        * $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        * $Q\_scale∈\mathbb{R}^{M}$: quantization scaling factor.

      - **Computation Process**

        - 1. Determine the tokens of the current group based on groupList[i], where $i \in [0,Len(groupList)]$.

          > Example: Assume groupList=[3,4,4,6] with groupListType=cumsum, or groupList=[3,1,0,2] with groupListType=count.
          >
          > Note: The two different grouping methods actually produce the same grouping result.
          >
          > The 0th right matrix `W[0,:,:]` corresponds to the tokens `x[0:3]` at index positions [0,3) (3-0=3 tokens in total), and corresponds to `x_scale[0:3]`, `w_scale[0]`, `bias[0]`, `offset[0]`, `Q[0:3]`, `Q_scale[0:3]`, and `Q_offset[0:3]`;
          >
          > The 1st right matrix `W[1,:,:]` corresponds to the tokens `x[3:4]` at index positions [3,4) (4-3=1 token in total), and corresponds to `x_scale[3:4]`, `w_scale[1]`, `bias[1]`, `offset[1]`, `Q[3:4]`, `Q_scale[3:4]`, and `Q_offset[3:4]`;
          >
          > The 2nd right matrix `W[2,:,:]` corresponds to the tokens `x[4:4]` at index positions [4,4) (4-4=0 tokens in total), and corresponds to `x_scale[4:4]`, `w_scale[2]`, `bias[2]`, `offset[2]`, `Q[4:4]`, `Q_scale[4:4]`, and `Q_offset[4:4]`;
          >
          > The 3rd right matrix `W[3,:,:]` corresponds to the tokens `x[4:6]` at index positions [4,6) (6-4=2 tokens in total), and corresponds to `x_scale[4:6]`, `w_scale[3]`, `bias[3]`, `offset[3]`, `Q[4:6]`, `Q_scale[4:6]`, and `Q_offset[4:6]`;
          >
          > Note: Parts not specified in grouplist will not participate in the update.
          > For example, when groupList=[12,14,18] with GroupListType=cumsum and the shape of X is [30, :]:
          >
          > The shape of the first output Q is [30, :], where the part Q[18:, :] is neither updated nor initialized, and the data is the original data from when the device memory was allocated.
          >
          > Similarly, the shape of the second output Q is [30], where the part Q\_scale[18:] is neither updated nor initialized, and the data is the original data from when the device memory was allocated.
          >
          > That is, Q[:grouplist[-1],:] and Q\_scale[:grouplist[-1]] of the outputs contain the valid data.

        - 2. Perform the following computation with the input parameters determined by the grouping:

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;where $Swish(x)=\frac{x}{1+e^{-x}}$

        - 3. Quantize the output result

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>
    <details>
    <summary>MSD scenario A8W4 (A refers to the activation matrix, W refers to the weight matrix, 8 refers to the INT8 data type, and 4 refers to the INT4 data type):</summary>
    <a id="msd-scenario-a8w4"></a>

      - **Definitions**:
        * **⋅** denotes matrix multiplication.
        * **⊙** denotes element-wise multiplication.
        * $\left \lfloor x\right \rceil$ denotes rounding x to the nearest integer.
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **Inputs**:
        * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        * $W∈\mathbb{Z_4}^{E \times K \times N}$: grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        * $weightAsistMatrix∈\mathbb{R}^{E \times N}$: auxiliary matrix for the matrix multiplication (for the computation process of generating the auxiliary matrix, see the following text).
        * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$: per-channel scaling factor of the grouped weight matrix (right matrix), where E is the number of experts, K\_group\_num is the number of groups along the K dimension, and N is the output dimension.
        * $x\_scale∈\mathbb{R}^{M}$: per-token scaling factor of the activation matrix (left matrix), where M is the total number of tokens.
        * $grouplist∈\mathbb{N}^{E}$: grouped index list in cumsum or count mode.
      - **Outputs**:
        * $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        * $Q\_scale∈\mathbb{R}^{M}$: quantization scaling factor.
      - **Computation Process**
        - 1. Determine the tokens of the current group based on groupList[i], where $i \in [0,Len(groupList)]$.
          - The grouping logic is the same as that in A8W8.
        - 2. Computation process of generating the auxiliary matrix (weightAsistMatrix). Note that the weightAssistMatrix is computed offline and provided as an input; the operator does not compute it internally:
          - When per-channel quantization is used ($w\_scale$ is 2D):

            $weightAsistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$

          - When per-group quantization is used ($w\_scale$ is 3D):

            $weightAsistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$

            Note: $num\_per\_group = K // K\_group\_num$

        - 3. Perform the following computation with the input parameters determined by the grouping:

          - 3.1. Convert the left matrix $\mathbb{Z}_8$ into two $\mathbb{Z}_4$ parts: the high 4 bits and the low 4 bits
            $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
            $X\_low\_4bits_{i} = (X_{i} \& 0x0f) - 8$
          - 3.2. Use per-channel or per-group quantization in the matrix multiplication
            per-channel:

            $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            per-group:

            $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

            $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

          - 3.3. Restore the high-bit and low-bit matrix multiplication results to the complete result

            $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAsistMatrix_{i}) \odot x\_scale_{i}$

            $C_{i,act}, gate_{i} = split(C_{i})$

            $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;where $Swish(x)=\frac{x}{1+e^{-x}}$

        - 3. Quantize the output result

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

  - For <term>Ascend 950PR&950DT products</term>:
    <details>
    <summary>MX quantization scenario:</summary>

      - **Definitions**:

        * **⋅** denotes matrix multiplication.
        * **⊙** denotes element-wise multiplication.
      - **Computation Process**
        - 1. Determine the tokens of the current group based on groupList[i], where $i \in [0,Len(groupList)]$

        - 2. Perform the following computation with the input parameters determined by the grouping:

          $C_i[m,n]=\sum_{k=0}^{K-1}\left(X_i[m,k]\cdot s_i^X[m,\lfloor k/32\rfloor]\right)\left(W_i[k,n]\cdot s_i^W[\lfloor k/32\rfloor,n]\right)$

          Here, $s_i^X$ and $s_i^W$ are decoded MX scales. Each scale is shared by 32 elements along K, and scaling is applied before summation along K.

          $C_{i,act}, gate_{i} = split(C_{i})$

          When swigluMode is 0, $S_i=Swish(C_{i,act})\odot gate_i$, where $Swish(x)=\frac{x}{1+e^{-x}}$.

          When swigluMode is 2, split the result into two halves along N, clamp the two branches, and compute SwiGLU:

          $A_i'=\min(C_{i,act},L),\qquad B_i'=\operatorname{clip}(gate_i,-L,L)$

          $S_i=A_i'\odot\operatorname{sigmoid}(\alpha A_i')\odot(B_i'+\beta)$

          Here, $L$, $\alpha$, and $\beta$ correspond to clampLimit, gluAlpha, and gluBias, respectively.

          In the MX A8W4 scenario, x is FLOAT8_E4M3FN and weight is FLOAT4_E2M1. The optional smoothScale has type FLOAT and shape $(E,N/2)$. Its i-th row is broadcast over the token rows of the corresponding expert; omitting it is equivalent to using all ones. SwiGLU and smoothing use FLOAT arithmetic, followed by conversion to BF16 with round-to-nearest-even:

          $V_i=\operatorname{BF16}_{\mathrm{rint}}(S_i\odot smoothScale_i)$

        - 3. Quantize the output result

          MX A8W4 uses $V_i$ as the quantization input $Z_i$ and produces FLOAT8_E4M3FN output and FLOAT8_E8M0 scales. For other scenarios, $Z_i$ in the following formulas denotes the activation result $S_i$.

          For the nonzero finite maximum absolute value $a=\max_j|Z_{i,j}|$ of each quantization group, the OCP algorithm computes the shared scale from its exponent:

          $shared\_exp = \lfloor\log_2 a\rfloor - emax$

          In MX A8W4, scaleAlg=1 selects the cuBLAS algorithm, using the maximum finite E4M3FN value of 448:

          $shared\_exp = \lceil\log_2(a/448)\rceil$

          $QScale = 2 ^ {shared\_exp}$

          $Q_{i,j} = quantize\_to\_element\_format(Z_{i,j}/QScale), \space j\space from\space 1\space to\space blocksize$
          - $emax$: the exponent of the largest normal number for the corresponding data type.

            |   DataType    | emax |
            | :-----------: | :--: |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |
            |  FLOAT4_E2M1  |  2   |

          - $blocksize$: the number of elements quantized each time. Only 32 is supported.
    </details>


    <details>
    <summary>Pertoken quantization scenario:</summary>

      - **Definitions**:

        * **⋅** denotes matrix multiplication.
        * **⊙** denotes element-wise multiplication.
      - **Inputs**:

        * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        * $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        * $w\_scale∈\mathbb{R}^{E \times N}$: per-channel scaling factor of the grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        * $x\_scale∈\mathbb{R}^{M}$: per-token scaling factor of the activation matrix (left matrix), where M is the total number of tokens.
        * $grouplist∈\mathbb{N}^{E}$: grouped index list in cumsum or count mode.
      - **Outputs**:

        * $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        * $Q\_scale∈\mathbb{R}^{M}$: quantization scaling factor.
      - **Computation Process**
        - 1. Determine the tokens of the current group based on groupList[i], where $i \in [0,Len(groupList)]$

           - 2. Perform the following computation with the input parameters determined by the grouping:

             $C_{i} = (X_{i}\cdot W_{i} )\odot xScale_{i} \odot wScale_{i}$

             $C_{i,act}, gate_{i} = split(C_{i})$

             $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$

             where $xScale_{i}$ indicates the quantization factor corresponding to the token
           - 3. Quantize the output result

             $Q\_scale_{i} = \frac{max(|S_{i}|)}{max(type)}$

             $Q_{i} = \lfloor \frac{S_{i}}{Q\_scale_{i}} \rceil$
    </details>

## Parameter Description

<table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
<col style="width: 170px">
<col style="width: 120px">
<col style="width: 300px">
<col style="width: 330px">
<col style="width: 212px">
<col style="width: 100px">
<col style="width: 190px">
<col style="width: 145px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th style="white-space: nowrap">Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x</td>
    <td rowspan="1">Input</td>
    <td>Represents the left matrix, corresponding to X in the formula.</td>
    <td>FLOAT8_E4M3FN<sup>1</sup>, FLOAT8_E5M2<sup>1</sup>, FLOAT4_E2M1<sup>1</sup>, FLOAT4_E1M2<sup>1</sup>, INT8, INT4<sup>2</sup>, INT32<sup>2</sup>, HIFLOAT8<sup>1</sup></td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td rowspan="1">Input</td>
    <td>Represents the weight matrix, corresponding to W in the formula.</td>
    <td>FLOAT8_E4M3FN<sup>1</sup>, FLOAT8_E5M2<sup>1</sup>, FLOAT4_E2M1<sup>1</sup>, FLOAT4_E1M2<sup>1</sup>, INT8, INT4<sup>2</sup>, INT32<sup>2</sup>, HIFLOAT8<sup>1</sup></td>
    <td>ND, FRACTAL_NZ</td>
  </tr>
  <tr>
    <td>weightScale</td>
    <td rowspan="1">Input</td>
    <td>Represents the quantization factor of the right matrix, corresponding to wScale in the formula.</td>
    <td>FLOAT8_E8M0<sup>1</sup>, UINT64<sup>2</sup>, FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightAssistMatrix</td>
    <td rowspan="1">Optional input</td>
    <td>Represents the auxiliary matrix for the matrix multiplication, corresponding to weightAssistMatrix in the formula.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td rowspan="1">Optional input</td>
    <td>Represents the offset value for the matrix multiplication.</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>xScale</td>
    <td rowspan="1">Input</td>
    <td>Represents the quantization factor of the left matrix, corresponding to xScale in the formula.</td>
    <td>FLOAT8_E8M0<sup>1</sup>, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>smoothScale</td>
    <td rowspan="1">Optional input</td>
    <td>Optional expert-specific, per-output-channel smoothing factors. When supplied, they are multiplied element-wise after SwiGLU and before output quantization. The shape is (E,N/2). For supported scenarios, see Constraints.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>groupList</td>
    <td rowspan="1">Input</td>
    <td>Represents the number of tokens participating in the computation for each group, corresponding to grouplist in the formula.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dequantMode</td>
    <td rowspan="1">Optional attribute</td>
    <td>Dequantization mode. The default value is 0, indicating the pertoken perchannel quantization mode.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dequantDtype</td>
    <td rowspan="1">Optional attribute</td>
    <td>Data type of the intermediate GroupedMatmul result. The default value is 0, indicating FLOAT.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>quantMode</td>
    <td rowspan="1">Optional attribute</td>
    <td>Quantization mode of the SwiGLU result. The default value is 0, indicating pertoken quantization.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>groupListType</td>
    <td rowspan="1">Optional attribute</td>
    <td>Interpretation mode of groupList. The default value is 0, indicating the cumulative number of tokens (cumsum).</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>tuningConfig</td>
    <td rowspan="1">Optional attribute</td>
    <td>Tiling tuning configuration. The default value is [0], indicating that the expected number of tokens for each expert is not specified.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>output</td>
    <td rowspan="1">Output</td>
    <td>Represents the output result. Quantized output corresponds to Q in the formula; the V3 A4W4 direct-output scenario produces BF16 results.</td>
    <td>FLOAT8_E4M3FN<sup>1</sup>, FLOAT8_E5M2<sup>1</sup>, FLOAT4_E2M1<sup>1</sup>, FLOAT4_E1M2<sup>1</sup>, INT8, BFLOAT16, HIFLOAT8<sup>1</sup></td>
    <td>ND</td>
  </tr>
  <tr>
    <td>outputScale</td>
    <td rowspan="1">Output</td>
    <td>Represents the quantization factor of the output, corresponding to QScale in the formula.</td>
    <td>FLOAT8_E8M0<sup>1</sup>, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td rowspan="1">Output</td>
    <td>Returns the size of the workspace that needs to be allocated on the device.</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor</td>
    <td rowspan="1">Output</td>
    <td>Returns the operator executor, which contains the operator computation process.</td>
    <td>-</td>
    <td>-</td>
  </tr>
</tbody>
</table>

- For <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
    - The superscript <sup>1</sup> in the Data Type column of the preceding table indicates data types not supported by this series.
    - x supports the INT8, INT4, and INT32 data types.
    - weight supports the INT8, INT4, and INT32 data types. In ND format, the shape is in the form {(E, K, N)}. In NZ format, when the data type of weight is INT8, the shape is in the form {(E, N / 32, K / 16, 16, 32)}; when the data type is INT4, the shape is in the form {(E, N / 64, K / 16, 16, 64)}; and when the data type is INT32, the shape is in the form {(E, N / 64, K / 16, 16, 8)}.
    - For weightScale, the A8W8 scenario supports the FLOAT, FLOAT16, and BFLOAT16 data types, and the shape supports only 2 dimensions in the form {(E, N)}. The A8W4 scenario supports the UINT64 data type, and the shape supports 2 or 3 dimensions: the per-channel shape is in the form {(E, N)}, and the per-group shape is in the form {(E, KGroupCount, N)}.
    - The dequantMode parameter is supported: 0 indicates that the activation matrix uses per-token and the weight matrix uses per-channel; 1 indicates that the activation matrix uses per-token and the weight matrix uses per-group.
    - The dequantDtype parameter is not supported.
    - The quantMode parameter is not supported.
    - In the A8W8/A8W4 scenarios, an N-axis length greater than 10240 is not supported.
    - In the A8W8 scenario, a tail-axis length of x greater than or equal to 65536 is not supported.
    - In the A8W4 scenario, a tail-axis length of x greater than or equal to 20000 is not supported.
    - output supports only the INT8 data type, and its shape supports 2 dimensions in the form (M, N / 2).
    - outputScale supports only the FLOAT data type, and its shape supports 1 dimension in the form (M,).
- For <term>Ascend 950PR&950DT products</term>:
    - The superscript <sup>2</sup> in the Data Type column of the preceding table indicates data types not supported by this series.
    - x and weight do not support the INT4 or INT32 data types. The transposed weight is supported.
    - weightScale does not support the UINT64 data type. In the MX single-tensor scenario, the non-transposed shape is (E, ceil(K / 64), N, 2), and the transposed shape is (E, N, ceil(K / 64), 2). For the shapes in the per-token and WeightNzV2 multi-tensor scenarios, see the corresponding API documents.
    - xScale supports the FLOAT8_E8M0 and FLOAT data types. In the MX quantization scenario, the shape is (M, ceil(K / 64), 2); in the per-token quantization scenario, the shape is (M,).
    - The dequantMode parameter is supported: the value is 2 in the MX quantization scenario and 0 in the per-token quantization scenario.
    - The dequantDtype parameter is supported: only 0 (FLOAT) is supported in the MX quantization scenario; 0 (FLOAT), 1 (FLOAT16), and 27 (BFLOAT16) are supported in the per-token quantization scenario.
    - The quantMode parameter is supported: the value is 2 in the MX quantization scenario and 0 in the per-token quantization scenario.
    - output supports the FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT8, and INT8 data types, and its shape supports 2 dimensions in the form (M, N / 2).
    - outputScale supports the FLOAT8_E8M0 and FLOAT data types. In the MX quantization scenario, the shape is (M, ceil((N / 2) / 64), 2); in the per-token quantization scenario, the shape is (M,).

## Constraints

  - For <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    - The following constraints must be met in the A8W8/A8W4 quantization scenarios:
        - The data types must comply with the following table:
        <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
        <col style="width: 319px">
        <col style="width: 144px">
        <col style="width: 671px">
        </colgroup>
        <thead>
          <tr>
            <th>Quantization scenario</th>
            <th>x</th>
            <th>weight</th>
            <th>weightScale</th>
            <th>xScale</th>
            <th>output</th>
            <th>outputScale</th>
          </tr></thead>
        <tbody>
          <tr>
            <td>A8W8</td>
            <td>INT8</td>
            <td>INT8</td>
            <td>FLOAT, FLOAT16, BFLOAT16</td>
            <td>FLOAT</td>
            <td>INT8</td>
            <td>FLOAT</td>
          </tr>
          <tr>
            <td>A8W4</td>
            <td>INT8</td>
            <td>INT4, INT32</td>
            <td>UINT64</td>
            <td>FLOAT</td>
            <td>INT8</td>
            <td>FLOAT</td>
          </tr>
        </tbody>
        </table>

      - In the A8W8 scenario, an N-axis length greater than 10240 is not supported, and a tail-axis length of x greater than or equal to 65536 is not supported.
      - In the A8W4 scenario, an N-axis length greater than 10240 is not supported, and a tail-axis length of x greater than or equal to 20000 is not supported.

  - For <term>Ascend 950PR&950DT products</term>:
    - The following constraints must be met in the MX quantization scenario:
        - The data types of aclnnGroupedMatmulSwigluQuantV2 must comply with the following table:
        <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
        <col style="width: 319px">
        <col style="width: 144px">
        <col style="width: 671px">
        </colgroup>
        <thead>
          <tr>
            <th>MX quantization scenario</th>
            <th>x</th>
            <th>weight</th>
            <th>weightScale</th>
            <th>xScale</th>
            <th>output</th>
            <th>outputScale</th>
          </tr></thead>
        <tbody>
          <tr>
            <td>MXFP8</td>
            <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
            <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
            <td>FLOAT8_E8M0</td>
          </tr>
          <tr>
            <td>MXFP4</td>
            <td>FLOAT4_E2M1</td>
            <td>FLOAT4_E2M1</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
            <td>FLOAT8_E8M0</td>
          </tr>
        </tbody>
        </table>

      - In the MXFP8 scenario of aclnnGroupedMatmulSwigluQuantWeightNzV2, x, weight, and output support only FLOAT8_E4M3FN. In the MXFP4 scenario, x and weight support FLOAT4_E2M1, FLOAT4_E1M2, and their cross combinations, and output supports FLOAT8_E4M3FN, FLOAT4_E1M2, and FLOAT4_E2M1. The scale in each scenario is FLOAT8_E8M0. For other scenarios such as MXA8W4, see [WeightNzV2 Constraint Description](docs/aclnnGroupedMatmulSwigluQuantWeightNzV2_en.md#constraints).

      - In the MX quantization scenario with weight in ND format, N must be aligned to 2 when the input is of the fp8 data type, and N must be aligned to 4 when the input is of the fp4 data type.
      - In the MX quantization scenario with weight in NZ format, N must be aligned to 64 when the input is of the fp8 data type, and N must be aligned to 128 when the input is of the fp4 data type.
      - K=2 is not supported in the MXFP4 scenario.
      - K must be an even number in the MXFP4 scenario.

    - The quantized-output scenarios of `aclnnGroupedMatmulSwigluQuantWeightNzV3` support MX A8W8 and MX A8W4, both requiring a single TensorList and FRACTAL_NZ weights. The x/weight types are FLOAT8_E4M3FN/FLOAT8_E4M3FN for A8W8 and FLOAT8_E4M3FN/FLOAT4_E2M1 for A8W4. A8W4 requires K and N to be multiples of 128 and supports only transposed weights. Both scenarios produce FLOAT8_E4M3FN output and FLOAT8_E8M0 scales; swigluMode supports only 2, and scaleAlg supports 0 or 1. Only A8W4 supports optional FLOAT smoothScale with shape (E,N/2). The existing A4W4 BF16 direct-output scenario does not produce outputScale, and its API constraints remain unchanged. For complete constraints, see [ACLNN API Documentation](docs/aclnnGroupedMatmulSwigluQuantWeightNzV3.md).

## Invocation

| Invocation Method | Invocation Example | Description |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_quant_grouped_matmul_swiglu_quant_V2](examples/test_aclnn_grouped_matmul_swiglu_quant_v2_a8w8.cpp) | Invokes the [GroupedMatmulSwigluQuantV2](docs/aclnnGroupedMatmulSwigluQuantV2_en.md) operator through the API. |
| aclnn API | [test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3](examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.cpp) | Invokes V3 WeightNZ scenarios through [aclnnGroupedMatmulSwigluQuantWeightNzV3](docs/aclnnGroupedMatmulSwigluQuantWeightNzV3.md). See that API reference for MX A8W4 parameters and constraints; this linked example covers the existing MX A8W8 scenario. |
| aclnn API | [test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3_a8w4](examples/arch35/test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v3_a8w4.cpp) | V3 MX A8W4 with transposed WeightNZ, SwiGLU mode 2, and optional smoothScale; see [Build and Run](examples/arch35/README_v3_a8w4_en.md). |
| PyTorch API | [grouped_matmul_swiglu_quant](docs/torchapi_grouped_matmul_swiglu_quant.md#调用说明) | Invokes MX A8W8 or MX A8W4 WeightNZ through [grouped_matmul_swiglu_quant](docs/torchapi_grouped_matmul_swiglu_quant.md). The Torch documentation provides an A8W4 example with transposed weights and smooth_scale. |
