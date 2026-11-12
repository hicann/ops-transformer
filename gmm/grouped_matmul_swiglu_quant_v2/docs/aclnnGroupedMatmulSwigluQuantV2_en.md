# aclnnGroupedMatmulSwigluQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:37:29.360Z -->

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

## Function

- Description: Fuses GroupedMatmul, dequant, swiglu, and quant. For details, see Formula.

  Compared with the [aclnnGroupedMatmulSwigluQuant](../../grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant_en.md) API, **this API adds:**

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>:
      - Added MXFP8, MXFP4, and Pertoken quantization scenarios.
      - The field types of the parameters weight, weightScale, and weightAssistMatrix are changed to tensorlist. Select the appropriate API based on the actual situation.
    <!-- end id7 -->

- Formula:

  <!-- npu="A3,910b" id8 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    <details>
    <summary>Quantization scenario A8W8 (A refers to the activation matrix, W refers to the weight matrix, and 8 refers to the INT8 data type):</summary>
    <a id="quantization-scenario-a8w8"></a>

      - **Definition**:

        - **⋅** denotes matrix multiplication.
        - **⊙** indicates per-element multiplication.
        - $\left \lfloor x\right \rceil$ indicates rounding x to the nearest integer.
        - $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        - $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **Input**:

        - $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $w\_scale∈\mathbb{R}^{E \times N}$: Per-channel scale factor of the grouped weight matrix (right matrix), where E is the number of experts and N is the output dimension.
        - $x\_scale∈\mathbb{R}^{M}$: Per-token scale factor of the activation matrix (left matrix), where M is the total number of tokens.
        - $grouplist∈\mathbb{N}^{E}$: group index list of cumsum or count.
      - **Output**:

        - $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        - $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.

      - **Computation Process**

        - 1. Determine the tokens of the current group based on groupList[i], where $i \in [0,Len(groupList)]$.

          >Example: Assume groupList = [3, 4, 4, 6] with groupListType = cumsum, or groupList = [3, 1, 0, 2] with groupListType = count.
          >
          > Note: The two different grouping methods above actually produce the same grouping result.
          >
          > The 0th right matrix `W[0,:,:]` corresponds to tokens `x[0:3]` at index positions [0, 3) (3 - 0 = 3 tokens in total), and corresponds to `x_scale[0:3]`, `w_scale[0]`, `bias[0]`, `offset[0]`, `Q[0:3]`, `Q_scale[0:3]`, and `Q_offset[0:3]`.
          >
          > The 1st right matrix `W[1,:,:]` corresponds to tokens `x[3:4]` at index positions [3, 4) (4 - 3 = 1 token in total), and corresponds to `x_scale[3:4]`, `w_scale[1]`, `bias[1]`, `offset[1]`, `Q[3:4]`, `Q_scale[3:4]`, and `Q_offset[3:4]`.
          >
          > The 2nd right matrix `W[2,:,:]` corresponds to tokens `x[4:4]` at index positions [4, 4) (4 - 4 = 0 tokens in total), and corresponds to `x_scale[4:4]`, `w_scale[2]`, `bias[2]`, `offset[2]`, `Q[4:4]`, `Q_scale[4:4]`, and `Q_offset[4:4]`.
          >
          > The 3rd right matrix `W[3,:,:]` corresponds to tokens `x[4:6]` at index positions [4, 6) (6 - 4 = 2 tokens in total), and corresponds to `x_scale[4:6]`, `w_scale[3]`, `bias[3]`, `offset[3]`, `Q[4:6]`, `Q_scale[4:6]`, and `Q_offset[4:6]`.
          >
          > Note: Portions not specified in groupList will not participate in the update.
          > For example, when groupList = [12, 14, 18] and GroupListType = cumsum, and the shape of X is [30, :]:
          >
          > The shape of the first output Q is [30, :], where the portion Q[18:, :] will not be updated or initialized, and the data in that portion is the original data from when the memory space was allocated.
          >
          > Similarly, the shape of the second output Q is [30], where the portion Q_scale[18:] will not be updated or initialized, and the data in that portion is the original data from when the memory space was allocated.
          >
          > That is, Q[:grouplist[-1], :] and Q_scale[:grouplist[-1]] in the output are the valid data portions.

        - Perform the following computation based on the input parameters determined by the group:

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;where $Swish(x)=\frac{x}{1+e^{-x}}$

        - Quantize the output result

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>
    <details>
    <summary>MSD scenario A8W4 (A refers to the activation matrix, W refers to the weight matrix, 8 refers to the INT8 data type, and 4 refers to the INT4 data type):</summary>
    <a id="msd-scenario-a8w4"></a>

      - **Definition**:
        - **⋅** denotes matrix multiplication.
        - **⊙** indicates per-element multiplication.
        - $\left \lfloor x\right \rceil$ indicates rounding x to the nearest integer.
        - $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        - $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
        - $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **Input**:
        - $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $W∈\mathbb{Z_4}^{E \times K \times N}$: Grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $weightAssistMatrix∈\mathbb{R}^{E \times N}$: Auxiliary matrix used in matrix multiplication (see the computation process for generating the auxiliary matrix below).
        - $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$: Per-channel scale factor of the grouped weight matrix (right matrix), where E is the number of experts, K\_group\_num is the number of groups along the K axis, and N is the output dimension.
        - $x\_scale∈\mathbb{R}^{M}$: Per-token scale factor of the activation matrix (left matrix), where M is the total number of tokens.
        - $grouplist∈\mathbb{N}^{E}$: group index list of cumsum or count.
      - **Output**:
        - $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        - $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
      - **Computation Process**
        - Determine the tokens of the current group based on groupList[i], where $i \in [0,Len(groupList)]$.
          - The grouping logic is the same as that of A8W8.
        - Computation process for generating the auxiliary matrix (weightAssistMatrix) (note that the weightAssistMatrix is partially computed offline and provided as input, not completed inside the operator):
          - For per-channel quantization ($w\_scale$ is 2-dimensional):

            $weightAssistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$

          - For per-group quantization ($w\_scale$ is 3-dimensional):

            $weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$

            NOTE: $num\_per\_group = K // K\_group\_num$

        - Perform the following computation based on the inputs determined by grouping:

          - 3.1. Convert the left matrix $\mathbb{Z_8}$ into $\mathbb{Z_4}$ split into high-bit and low-bit parts
            $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
            $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
          - 3.2. Enable per-channel or per-group quantization when performing matrix multiplication
            per-channel:

            $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            per-group:

            $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

            $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

          - 3.3. Restore the high-bit and low-bit matrix multiplication results to the overall result.

            $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$

            $C_{i,act}, gate_{i} = split(C_{i})$

            $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;where $Swish(x)=\frac{x}{1+e^{-x}}$

        - Quantize the output result

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>
    <details>
    <summary>Quantization scenario A4W4 (A refers to the activation matrix, W refers to the weight matrix, and 4 refers to the INT4 data type):</summary>
    <a id="quantization-scenario-a4w4"></a>

      - **Definition**:

        - **⋅** denotes matrix multiplication.
        - **⊙** indicates per-element multiplication.
        - $\left \lfloor x\right \rceil$ indicates rounding x to the nearest integer.
        - $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
        - $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        - $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **Input**:

        - $X∈\mathbb{Z_4}^{M \times K}$: Activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $W∈\mathbb{Z_4}^{E \times K \times N}$: Grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $w\_scale∈\mathbb{R}^{E \times N}$: Per-channel scale factor of the grouped weight matrix (right matrix), where E is the number of experts and N is the output dimension.
        - $x\_scale∈\mathbb{R}^{M}$: Per-token scale factor of the activation matrix (left matrix), where M is the total number of tokens.
        - $smoothScale∈\mathbb{R}^{E \times N/2}(per-channel) or \mathbb{R}^{E}(per-tensor)$: smooth scale factor, where E is the number of experts and N is the output dimension.
        - $grouplist∈\mathbb{N}^{E}$: group index list of cumsum or count.
      - **Output**:

        - $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        - $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.

      - **Computation Process**

        - 1. Determine the current group's tokens based on groupList[i], where $i \in [0,Len(groupList)]$.
          - The grouping logic is the same as that of A8W8.

        - Perform the following computation based on the input parameters determined by the group:

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;where $Swish(x)=\frac{x}{1+e^{-x}}$

          $S_{i} = S_{i} \odot smoothScale_{i\ Broadcast}$

        - Quantize the output result

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    <details>
    <summary>MX Quantization Scenario:</summary>

      - **Definition**:

        - **⋅** denotes matrix multiplication.
        - **⊙** indicates per-element multiplication.
      - **Input**:

        - $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $w\_scale∈\mathbb{R}^{E \times ceil(K / 64) \times N \times 2}$: per-channel scaling factor of the grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $x\_scale∈\mathbb{R}^{M \times ceil(K / 64) \times 2}$: per-token scaling factor of the activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $grouplist∈\mathbb{N}^{E}$: group index list of cumsum or count.
      - **Output**:

        - $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        - $Q\_scale∈\mathbb{R}^{M \times ceil((N / 2) / 64) \times 2}$: Quantization scale factor.
      - **Computation Process**
        - 1. Determine the tokens of the current group based on groupList[i], $i \in [0,Len(groupList)]$

        - Perform the following computation based on the input parameters determined by the group:

          $C_{i} = (X_{i}\cdot W_{i} )\odot xScale_{i\ Broadcast} \odot wScale_{i\ Broadcast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$

        - Quantize the output result

          $shared\_exp = \left\lfloor \log_2(max_i(|S_i|)) \right\rceil - emax$

          $QScale = 2 ^ {shared\_exp}$

          $Q_i = quantize\_to\_element\_format(S_i/Qscale), \space i\space from\space 1\space to\space blocksize$
          - $emax$: The exponent of the maximum normal number of the corresponding data type.

            |   DataType    | emax |
            | :-----------: | :--: |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |
            |  FLOAT4_E2M1  |  2   |

          - $blocksize$: The number of elements quantized each time. Only 32 is supported.
    </details>

    <details>
    <summary>Pertoken quantization scenario:</summary>

      - **Definition**:

        - **⋅** denotes matrix multiplication.
        - **⊙** indicates per-element multiplication.
      - **Input**:

        - $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $w\_scale∈\mathbb{R}^{E \times N}$: per-channel scaling factor of the grouped weight matrix (right matrix), E is the number of experts, N is the output dimension.
        - $x\_scale∈\mathbb{R}^{M}$: per-token scaling factor of the activation matrix (left matrix), M is the total number of tokens, K is the feature dimension.
        - $grouplist∈\mathbb{N}^{E}$: group index list of cumsum or count.
      - **Output**:

        - $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        - $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
      - **Computation Process**
        - 1. Determine the tokens in the current group based on groupList[i], $i \in [0,Len(groupList)]$

           - Perform the following computation based on the input parameters determined by the group:

             $C_{i} = (X_{i}\cdot W_{i} )\odot xScale_{i} \odot wScale_{i}$

             $C_{i,act}, gate_{i} = split(C_{i})$

             $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$

             where $xScale_{i}$ represents the quantization factor of the corresponding token
           - Quantize the output result

             $Q\_scale_{i} = \frac{max(|S_{i}|)}{max(type)}$

             $Q_{i} = \lfloor \frac{S_{i}}{Q\_scale_{i}} \rceil$
    </details>

  <!-- end id9 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that includes the operator computation flow, and then call the "aclnnGroupedMatmulSwigluQuantV2" API to perform the computation.

```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
    const aclTensor     *x,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *weightAssistMatrix,
    const aclTensor     *bias,
    const aclTensor     *xScale,
    const aclTensor     *smoothScale,
    const aclTensor     *groupList,
    int64_t              dequantMode,
    int64_t              dequantDtype,
    int64_t              quantMode,
    int64_t              groupListType,
    const aclIntArray   *tuningConfigOptional,
    aclTensor           *output,
    aclTensor           *outputScale,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize

  - **Parameters**
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
        <th>Parameter</th>
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th style="white-space: nowrap">Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td rowspan="1">Input</td>
        <td>Represents the left matrix, corresponding to X in the formula.</td>
        <td>-</td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT8, INT4, INT32, HIFLOAT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td rowspan="1">Input</td>
        <td>Represents the weight matrix, corresponding to W in the formula.</td>
        <td>-</td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT8, INT4, INT32, HIFLOAT8</td>
        <td>ND, FRACTAL_NZ</td>
        <td>2, 3, 4, 5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightScale</td>
        <td rowspan="1">Input</td>
        <td>Represents the quantization factor of the right matrix, i.e., wScale in the formula.</td>
        <td>The first axis length must be equal to the first axis dimension of weight, and the tail axis length must be the same as the tail axis of weight restored to the ND format.</td>
        <td>FLOAT8_E8M0, UINT64, FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1, 2, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightAssistMatrix</td>
        <td rowspan="1">optional input</td>
        <td>Represents the auxiliary matrix used for the matrix multiplication, which is weightAssistMatrix in the formula.</td>
        <td><ul>
          <li>Takes effect only in the A8W4 use case. A null pointer must be passed in other use cases.</li>
          <li>The first axis length must be equal to the first axis dimension of weight. The tail axis length must be the same as the tail axis of weight restored to the ND format.</li>
        </ul></td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1, 2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td rowspan="1">optional input</td>
        <td>Represents the bias value of the matrix multiplication computation.</td>
        <td>Reserved input. Currently not supported. A null pointer must be passed.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>xScale</td>
        <td rowspan="1">Input</td>
        <td>Refers to the quantization factor of the left matrix, that is, xScale in the formula.</td>
        <td>-</td>
        <td>FLOAT8_E8M0, FLOAT</td>
        <td>ND</td>
        <td>1, 3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>smoothScale</td>
        <td rowspan="1">Optional input</td>
        <td>Refers to the smooth scaling factor.</td>
        <td><ul>
        <li>Optional in the A4W4 quantization scenario. A null pointer must be passed in other scenarios.</li>
        <li>In the A4W4 quantization scenario, the first axis length must be equal to the first axis dimension of weight.</li>
        <li>In the A4W4 quantization scenario, a null pointer or two shapes are supported: (E, N / 2) or (E,).</li>
        </ul></td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1, 2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>groupList</td>
        <td rowspan="1">Input</td>
        <td>Represents the matmul size distribution of the input and output along the group axis, corresponding to grouplist in the formula. The specific meaning is determined by groupListType.</td>
        <td><ul>
          <li>The length must be equal to the first axis dimension of weight.</li>
          <li>When groupListType is 0, the last value must not be greater than the first dimension of the tensor in x; when groupListType is 1, the sum of the values must not be greater than the first dimension of the tensor in x.</li>
          <li>The values in groupList constrain the valid part of the output data. The part not specified by groupList will not participate in the update.</li>
        </ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dequantMode</td>
        <td rowspan="1">Input</td>
        <td>Represents the dequantization computation type, used to determine the dequantization method for the activation matrix and weight matrix.</td>
        <td><ul>
          <li>0 indicates the activation matrix is per-token and the weight matrix is per-channel.</li>
          <li>1 indicates the activation matrix is per-token and the weight matrix is per-group.</li>
          <li>2 indicates MX quantization.</li>
        </ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dequantDtype</td>
        <td rowspan="1">Input</td>
        <td>Represents the data type of the intermediate GroupedMatmul result.</td>
        <td><ul>
          <li>0 represents FLOAT.</li>
          <li>1 represents FLOAT16.</li>
          <li>27 represents BFLOAT16.</li>
          <li>28 represents UNDEFINED.</li>
        </ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td rowspan="1">Input</td>
        <td>Indicates the quantization computation type, used to determine the quantization mode of the swiglu result.</td>
        <td><ul>
          <li>0 indicates per-token.</li>
          <li>1 indicates per-group.</li>
          <li>2 indicates MX quantization.</li></ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupListType</td>
        <td rowspan="1">Input</td>
        <td>Indicates the grouping interpretation mode, used to determine the semantics of groupList.</td>
        <td><ul><li>0 indicates the cumsum mode. Each element in groupList represents the cumulative length of the current group.</li><li>1 indicates the count mode. Each element in groupList represents the number of elements in the group.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>tuningConfigOptional</td>
        <td rowspan="1">Optional input</td>
        <td>Used by the operator to estimate the M/E size and select different operator templates to meet the performance requirements of different use cases.</td>
        <td>Array. The first number passed in indicates the expected number of tokens processed by each expert and is used for tiling optimization. This parameter is enabled for A4W4 NZ right matrix input. For other inputs, pass in a null pointer.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>Output</td>
        <td rowspan="1">output</td>
        <td>Represents the quantized output, which is Q in the formula.</td>
        <td>-</td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT8, HIFLOAT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputScale</td>
        <td rowspan="1">Output</td>
        <td>Represents the quantization factor of the output, which is QScale in the formula.</td>
        <td>-</td>
        <td>FLOAT8_E8M0, FLOAT</td>
        <td>ND</td>
        <td>1, 3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td rowspan="1">Output</td>
        <td>Returns the workspace size to be side-requested on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td rowspan="1">Output</td>
        <td>Returns the op executor, which includes the computation process of the operator.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
    </table>

    <!-- npu="A3,910b" id10 -->
    - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
      - weight supports NZ input transpose in A4W4 scenarios, and only non-transposed input in other scenarios. INT32 is used for adaptation purposes in A8W4 and A4W4 scenarios, where one INT32 value is interpreted as eight INT4 data. The ND data format is not supported in A8W8 scenarios.
      - The dequantMode parameter is supported: values 0 and 1 are supported in A8W4 and A4W4 scenarios, and only value 0 is supported in A8W8 scenarios.
      - The dequantDtype and quantMode parameters are not supported.
      - x and weight do not support empty tensors.
      - When weight is input in NZ transpose format, only single-tensor mode is supported.
      - weight, weightScale, and weightAssistMatrix support single-tensor scenarios (tensorlist length is 1) and multi-tensor scenarios (tensorlist length is greater than 1).
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>:
      - weight supports transposition, and only the ND format is supported.
      - The dequantMode parameter is supported: in the MX quantization scenario, the value 2 is supported; in the Pertoken scenario, the value 0 is supported.
      - The dequantDtype parameter is supported: in the MX quantization scenario, the value 0 is supported; in the Pertoken scenario, the values 0, 1, and 27 are supported.
      - Supports the quantMode parameter: In the MX quantization scenario, the value 2 is supported; in the Pertoken scenario, the value 0 is supported.
      - Only the same value for dequantMode and quantMode is supported.
      - x and xScale support empty tensors with M equal to 0.
      - weight and weightScale support empty tensors with N equal to 0.
      - weight and weightScale currently support only a tensorlist length of 1.

    <!-- end id11 -->

  - **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:
    <table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
    <col style="width: 167px">
    <col style="width: 123px">
    <col style="width: 860px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x, weight, weightScale, xScale, groupList, output, or outputScale is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="9">161002</td>
        <td>The dimensions of the input x, weight, weightScale, xScale, groupList, output, or outputScale do not satisfy the constraint.</td>
      </tr>
      <tr>
        <td>The data type of a checked parameter, such as x, weight, weightScale, xScale, groupList, output, or outputScale, does not meet the requirements for the applicable scenario.</td>
      </tr>
      <tr>
        <td>The shape of the input x, weight, weightScale, xScale, groupList, output, or outputScale does not satisfy the constraint.</td>
      </tr>
      <tr>
        <td>The format of the input x, weight, weightScale, xScale, groupList, output, or outputScale does not satisfy the constraint.</td>
      </tr>
      <tr>
        <td>The length of the input weight or weightScale tensor list is not 1.</td>
      </tr>
      <tr>
        <td>The input x or xScale is an empty tensor, and the input weight or weightScale is an empty tensorList.</td>
      </tr>
      <tr>
        <td>The number of elements in groupList is greater than the first axis length of weight.</td>
      </tr>
      <tr>
        <td>The input dequantMode, quantMode, or dequantDtype does not satisfy the constraint.</td>
      </tr>
      <tr>
        <td>The input bias, weightAssistMatrix, smoothScale, or tuningConfigOptional does not satisfy the constraint.</td>
      </tr>
    </tbody>
    </table>

## aclnnGroupedMatmulSwigluQuantV2

- **Parameters**
  <table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
    <col style="width: 167px">
    <col style="width: 123px">
    <col style="width: 860px">
    </colgroup>
    <thead>
      <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>Input</td><td>Device side-requested workspace memory address.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Device side-requested workspace size, obtained by the first-phase API aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Op executor, includes the computation flow of the operator.</td></tr>
      <tr><td>stream</td><td>Input</td><td>Stream used to execute the task.</td></tr>
    </tbody>
  </table>

- **Return Value**

  Returns a status code of the aclnnStatus type. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - Deterministic computation:
      - aclnnGroupedMatmulSwigluQuantV2 defaults to a deterministic implementation.

  <!-- npu="A3,910b" id12 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - The following constraint conditions must be met in A8W8/A8W4/A4W4 quantization scenarios:
        - The data types must meet the requirements in the following table:
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 100px">
          <col style="width: 100px">
          <col style="width: 300px">
          <col style="width: 300px">
          <col style="width: 130px">
          <col style="width: 80px">
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 130px">
          </colgroup>
          <thead>
            <tr>
              <th>Quantization scenario</th>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>weightAssistMatrix</th>
              <th>xScale</th>
              <th>smoothScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>A8W8</td>
              <td>INT8</td>
              <td>INT8</td>
              <td>FLOAT, FLOAT16, BFLOAT16</td>
              <td>nullptr</td>
              <td>FLOAT</td>
              <td>nullptr</td>
              <td>INT8</td>
              <td>FLOAT</td>
            </tr>
            <tr>
              <td>A8W4</td>
              <td>INT8</td>
              <td>INT4, INT32</td>
              <td>UINT64</td>
              <td>FLOAT</td>
              <td>FLOAT</td>
              <td>nullptr</td>
              <td>INT8</td>
              <td>FLOAT</td>
            </tr>
            <tr>
              <td>A4W4</td>
              <td>INT4, INT32</td>
              <td>INT4, INT32</td>
              <td>UINT64</td>
              <td>nullptr</td>
              <td>FLOAT</td>
              <td>nullptr/FLOAT</td>
              <td>INT8</td>
              <td>FLOAT</td>
            </tr>
          </tbody>
          </table>

        - The shape constraints must meet the following table:
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 100px">
          <col style="width: 100px">
          <col style="width: 300px">
          <col style="width: 300px">
          <col style="width: 130px">
          <col style="width: 80px">
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 130px">
          </colgroup>
          <thead>
            <tr>
              <th>Quantization scenario</th>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>weightAssistMatrix</th>
              <th>xScale</th>
              <th>smoothScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>A8W8</td>
              <td>(M, K)</td>
              <td>NZ format shape is of the form {(E, N / 32, K / 16, 16, 32)}.</td>
              <td>{(E, N)}</td>
              <td>nullptr</td>
              <td>(M,)</td>
              <td>nullptr</td>
              <td>(M, N / 2)</td>
              <td>(M,)</td>
            </tr>
              <tr>
              <td>A8W4</td>
              <td>(M, K)</td>
              <td><ul>
              <li>For ND format, the shape is {(E, K, N)}.</li>
              <li>For NZ format and INT4, the shape is {(E, N / 64, K / 16, 16, 64)}.</li>
              <li>For NZ format and INT32, the shape is {(E, N / 64, K / 16, 16, 8)}.</li></ul></td>
              <td><ul>
              <li>For the per-channel scenario, the shape is {(E, N)}.</li>
              <li>For the per-group scenario, the shape is {(E, K_group_num, N)}.</li></ul></td>
              <td>{(E, N)}</td>
              <td>(M,)</td>
              <td>nullptr</td>
              <td>(M, N / 2)</td>
              <td>(M,)</td>
            </tr>
            <tr>
              <td>A4W4</td>
              <td>(M, K)</td>
              <td><ul>
              <li>The ND format shape is {(E, K, N)}.</li>
              <li>A4W4 supports non-transposed and transposed NZ.</li>
              <li>For the NZ non-transposed format and INT4, the shape is {(E, N / 64, K / 16, 16, 64)}.</li>
              <li>For the NZ non-transposed format and INT32, the shape is {(E, N / 64, K / 16, 16, 8)}.</li>
              <li>For the NZ transposed format and INT4, the original shape is {(E, K / 64, N / 16, 16, 64)}, and transpose(-1,-2) must be called before passing it in.</li>
              <li>For the NZ transposed format and INT32, the original shape is {(E, K / 64, N / 16, 16, 8)}, and transpose(-1,-2) must be called before passing it in.</li>
              <li>For NZ transposed input, the per-group K/K_group_num must be aligned to 64.</li></ul>
              </td>
              <td><ul>
              <li>In the per-channel scenario, the shape is {(E, N)}.</li>
              <li>In the per-group scenario, the shape is {(E, K_group_num, N)}.</li></ul>
              </td>
              <td>nullptr</td>
              <td>(M,)</td>
              <td><ul>
              <li>nullptr</li>
              <li>(E,)</li>
              <li>(E, N / 2)</li>
              </ul></td>
              <td>(M, N / 2)</td>
              <td>(M,)</td>
            </tr>
          </tbody>
          </table>

        - In the A8W8 scenario, the N axis length cannot exceed 10240, and the last axis length of x cannot be greater than or equal to 65536.
        - In the A4W4 scenario, the N axis length cannot exceed 10240, and the last axis length of x cannot be greater than or equal to 20000.
        - In the A4W4 scenario, the N-axis length cannot exceed 10240, and the last axis length of x cannot be greater than or equal to 20000.
        - In the multi-tensor scenario, that is, when the tensorlist length is greater than 1, the shapes of weight, weightScale, and weightAssistMatrix need to be flattened along the E dimension. For example, {(E, K, N)} needs to become {E instances of (K, N)}.

  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The first dimension of groupList supports a maximum of 1024, i.e., a maximum of 1024 groups.
    - The following constraints must be met in the MX quantization scenario:
        - The data types must meet the requirements in the following table:
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 300px">
          <col style="width: 300px">
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 130px">
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

        - The shape constraints must meet the following table:
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 130px">
          <col style="width: 250px">
          <col style="width: 320px">
          <col style="width: 180px">
          <col style="width: 250px">
          <col style="width: 160px">
          </colgroup>
          <thead>
            <tr>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>xScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>(M, K)</td>
              <td><ul>
              <li>The non-transposed shape is {(E, K, N)}.</li>
              <li>The transposed shape is {(E, N, K)}.</li></ul></td>
              <td><ul>
              <li>Non-transposed shape is {(E, ceil(K / 64), N, 2)}</li>
              <li>Transposed shape is {(E, N, ceil(K / 64), 2)}</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
          </tbody>
          </table>

        - The transpose attribute of weightScale must be consistent with that of weight.
        - In the MX quantization scenario, if the input is of the fp8 data type, N must be aligned to 2; if the input is of the fp4 data type, N must be aligned to 4.
        - The MXFP4 scenario does not support K=2.
        - In the MXFP4 scenario, K must be an even number.

    - The Pertoken quantization scenario must satisfy the following constraints:
        - The data types must meet the requirements in the following table:
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 250px">
          <col style="width: 250px">
          <col style="width: 300px">
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 130px">
          </colgroup>
          <thead>
            <tr>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>xScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
              <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
              <td>FLOAT, BFLOAT16</td>
              <td>FLOAT</td>
              <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
              <td>FLOAT</td>
            </tr>
            <tr>
              <td>INT8</td>
              <td>INT8</td>
              <td>FLOAT, BFLOAT16, FLOAT16</td>
              <td>FLOAT</td>
              <td>INT8</td>
              <td>FLOAT</td>
            </tr>
            <tr>
              <td>HIFLOAT8</td>
              <td>HIFLOAT8</td>
              <td>FLOAT, BFLOAT16</td>
              <td>FLOAT</td>
              <td>HIFLOAT8</td>
              <td>FLOAT</td>
            </tr>
          </tbody>
          </table>

        - The shape constraints must meet the following table:
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 130px">
          <col style="width: 250px">
          <col style="width: 320px">
          <col style="width: 180px">
          <col style="width: 160px">
          <col style="width: 230px">
          </colgroup>
          <thead>
            <tr>
              <th>x</th>
              <th>weight</th>
              <th>weightScale</th>
              <th>xScale</th>
              <th>output</th>
              <th>outputScale</th>
            </tr></thead>
          <tbody>
            <tr>
              <td>(M, K)</td>
              <td><ul>
              <li>Non-transposed shape: {(E, K, N)}.</li>
              <li>Transposed shape: {(E, N, K)}.</li></ul></td>
              <td>Shape: {(E, N)}.</td>
              <td>(M, )</td>
              <td>(M, N / 2)</td>
              <td>(M, )</td>
            </tr>
          </tbody>
          </table>

  <!-- end id13 -->

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  <!-- npu="A3,910b" id14 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.

    ```cpp
    #include <iostream>
    #include <vector>
    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_swiglu_quant_v2.h"

    #define CHECK_RET(cond, return_expr)                                                                                   \
        do {                                                                                                               \
            if (!(cond)) {                                                                                                 \
                return_expr;                                                                                               \
            }                                                                                                              \
        } while (0)

    #define LOG_PRINT(message, ...)                                                                                        \
        do {                                                                                                               \
            printf(message, ##__VA_ARGS__);                                                                                \
        } while (0)

    int64_t GetShapeSize(const std::vector<int64_t>& shape) {
        int64_t shapeSize = 1;
        for (auto i : shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream* stream) {
        // Boilerplate for resource initialization.
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSetDevice(deviceId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                        void** deviceAddr, aclDataType dataType, aclFormat formatType, aclTensor** tensor) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // Call aclrtMalloc to allocate device-side memory.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy host-side data to the device-side memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Calculate the strides of the contiguous tensor.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
        }

        // Call aclCreateTensor to create an aclTensor.
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                                shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    template <typename T>
    int CreateAclTensorList(const std::vector<T> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                            void **deviceAddr, aclDataType dataType, aclFormat formatType, aclTensorList **tensor) {
        int size = shapes.size();
        aclTensor* tensors[size];
        for (int i = 0; i < size; i++) {
            int ret = CreateAclTensor<T>(hostData, shapes[i], deviceAddr + i, dataType, formatType, tensors + i);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
        }
        *tensor = aclCreateTensorList(tensors, size);
        return ACL_SUCCESS;
    }

    int main() {
        // 1. (Boilerplate) Initialize device/stream. For details, see the acl API reference.
        // Set deviceId based on the actual device.
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = Init(deviceId, &stream);
        // Handle the check as needed.
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        // 2. Construct the inputs and outputs. Customize the construction according to the API definition.
        int64_t E = 4;
        int64_t M = 8192;
        int64_t N = 4096;
        int64_t K = 7168;
        std::vector<int64_t> xShape = {M, K};
        std::vector<std::vector<int64_t>> weightShape = {{E, N / 32 , K / 16, 16, 32}};
        std::vector<std::vector<int64_t>> weightScaleShape = {{E, N}};
        std::vector<int64_t> xScaleShape = {M};
        std::vector<int64_t> groupListShape = {E};
        std::vector<int64_t> outputShape = {M, N / 2};
        std::vector<int64_t> outputScaleShape = {M};

        void* xDeviceAddr = nullptr;
        void* weightDeviceAddr[1];
        void* weightScaleDeviceAddr[1];
        void* xScaleDeviceAddr = nullptr;
        void* groupListDeviceAddr = nullptr;
        void* outputDeviceAddr = nullptr;
        void* outputScaleDeviceAddr = nullptr;

        aclTensor* x = nullptr;
        aclTensorList* weight = nullptr;
        aclTensorList* weightScale = nullptr;
        aclTensor* xScale = nullptr;
        aclTensor* groupList = nullptr;
        aclTensor* output = nullptr;
        aclTensor* outputScale = nullptr;

        std::vector<int8_t> xHostData(M * K, 0);
        std::vector<int8_t> weightHostData(E * N * K, 0);
        std::vector<float> weightScaleHostData(E * N, 0);
        std::vector<float> xScaleHostData(M, 0);
        std::vector<int64_t> groupListHostData(E, 0);
        std::vector<int8_t> outputHostData(M * N / 2, 0);
        std::vector<float> outputScaleHostData(M, 0);

        // Create the x aclTensor.
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the weight aclTensorList.
        ret = CreateAclTensorList(weightHostData, weightShape, weightDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the weightScale aclTensorList.
        ret = CreateAclTensorList(weightScaleHostData, weightScaleShape, weightScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &weightScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the xScale aclTensor.
        ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &xScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the groupList aclTensor.
        ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the output aclTensor.
        ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &output);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the outputScale aclTensor.
        ret = CreateAclTensor(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &outputScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Add the V2 parameters.
        aclTensorList* weightAssistMatrix = nullptr;
        aclTensor* bias = nullptr;
        aclTensor* smoothScale = nullptr;
        int64_t dequantMode = 0;
        int64_t dequantDtype = 28;
        int64_t quantMode = 0;
        int64_t groupListType = 0;

        std::vector<int64_t> tuningConfigData;
        aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigData.data(), 0);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;

        // 3. Call the CANN operator library API.
        // Call the first-phase API of aclnnGroupedMatmulSwigluQuantV2.
        ret = aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
            x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode, dequantDtype,
            quantMode, groupListType, tuningConfig, output, outputScale, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Request device memory based on the workspaceSize calculated by the first-phase API.
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API of aclnnGroupedMatmulSwigluQuantV2.
        ret = aclnnGroupedMatmulSwigluQuantV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2 failed. ERROR: %d\n", ret); return ret);

        // 4. (boilerplate) Synchronize and wait for the task execution to complete.
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. Obtain the output values and copy the results from the Device side to the Host side. Modify the code based on the specific API definition.
        auto size = 10;
        std::vector<int8_t> out1Data(size, 0);
        ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), outputDeviceAddr,
                            size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, out1Data[j]);
        }
        std::vector<float> out2Data(size, 0);
        ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), outputScaleDeviceAddr,
                            size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, out2Data[j]);
        }
        // 6. Release aclTensor, aclTensorList, and aclScalar. Modify the code based on the specific API definition.
        aclDestroyTensor(x);
        aclDestroyTensorList(weight);
        aclDestroyTensorList(weightScale);
        aclDestroyTensor(xScale);
        aclDestroyTensor(groupList);
        aclDestroyTensor(output);
        aclDestroyTensor(outputScale);

        aclDestroyIntArray(tuningConfig);

        // 7. Release device resources. Modify the code based on the specific API definition.
        aclrtFree(xDeviceAddr);
        for (int64_t i = 0; i < 1; i++) {
            aclrtFree(weightDeviceAddr[i]);
            aclrtFree(weightScaleDeviceAddr[i]);
        }
        aclrtFree(xScaleDeviceAddr);
        aclrtFree(groupListDeviceAddr);
        aclrtFree(outputDeviceAddr);
        aclrtFree(outputScaleDeviceAddr);
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }
    ```

  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>:

    ```cpp
    #include <iostream>
    #include <memory>
    #include <vector>

    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_swiglu_quant_v2.h"

    #define CHECK_RET(cond, return_expr) \
        do {                               \
          if (!(cond)) {                   \
            return_expr;                   \
          }                                \
        } while (0)

    #define CHECK_FREE_RET(cond, return_expr) \
        do {                                  \
            if (!(cond)) {                    \
                Finalize(deviceId, stream);   \
                return_expr;                  \
            }                                 \
        } while (0)

    #define LOG_PRINT(message, ...)     \
        do {                              \
          printf(message, ##__VA_ARGS__); \
        } while (0)

    int64_t GetShapeSize(const std::vector<int64_t>& shape) {
        int64_t shapeSize = 1;
        for (auto i : shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream* stream) {
        // Boilerplate: resource initialization
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSetDevice(deviceId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclFormat FormatType, aclTensor** tensor) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // Call aclrtMalloc to allocate Device-side memory.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        // Call aclrtMemcpy to copy Host-side data to Device-side memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Calculate the strides of the contiguous tensor.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        // Call aclCreateTensor to create an aclTensor.
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, FormatType,
                                  shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    template <typename T>
    int CreateAclTensorList(const std::vector<std::vector<T>>& hostData, const std::vector<std::vector<int64_t>>& shapes,
                            void** deviceAddr, aclDataType dataType, aclTensorList** tensor) {
        int size = shapes.size();
        aclTensor* tensors[size];
        for (int i = 0; i < size; i++) {
            int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, ACL_FORMAT_ND, tensors + i);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
        }
        *tensor = aclCreateTensorList(tensors, size);
        return ACL_SUCCESS;
    }

    template <typename T1, typename T2>
    auto CeilDiv(T1 a, T2 b) -> T1
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    }

    void Finalize(int32_t deviceId, aclrtStream stream)
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }

    int aclnnGroupedMatmulSwigluQuantV2Test(int32_t deviceId, aclrtStream& stream)
    {
        auto ret = Init(deviceId, &stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        // 2. Construct the inputs and outputs, which need to be custom-constructed based on the API.
        int64_t E = 8;
        int64_t M = 2048;
        int64_t N = 4096;
        int64_t K = 7168;

        std::vector<int64_t> xShape = {M, K};
        std::vector<int64_t> weightShape = {E, K, N};
        std::vector<int64_t> weightScaleShape = {E, CeilDiv(K, 64), N, 2};
        std::vector<int64_t> xScaleShape = {M, CeilDiv(K, 64), 2};
        std::vector<int64_t> groupListShape = {E};
        std::vector<int64_t> outputShape = {M, N / 2};
        std::vector<int64_t> outputScaleShape = {M, CeilDiv((N / 2), 64), 2};

        void* xDeviceAddr = nullptr;
        void* weightDeviceAddr = nullptr;
        void* weightScaleDeviceAddr = nullptr;
        void* xScaleDeviceAddr = nullptr;
        void* groupListDeviceAddr = nullptr;
        void* outputDeviceAddr = nullptr;
        void* outputScaleDeviceAddr = nullptr;

        aclTensor* x = nullptr;
        aclTensorList* weight = nullptr;
        aclTensorList* weightScale = nullptr;
        aclTensor* xScale = nullptr;
        aclTensor* groupList = nullptr;
        aclTensor* output = nullptr;
        aclTensor* outputScale = nullptr;
        aclTensorList* weightAssistMatrix = nullptr;
        aclTensorList* smoothScale = nullptr;

        std::vector<int8_t> xHostData(M * K, 1);
        std::vector<int8_t> weightHostData(E * N * K, 1);
        std::vector<int8_t> weightScaleHostData(E * CeilDiv(K, 64) * N * 2, 1);
        std::vector<int8_t> xScaleHostData(M * CeilDiv(K, 64) * 2, 1);
        std::vector<int64_t> groupListHostData(E, 1);
        std::vector<int8_t> outputHostData(M * N / 2, 1);
        std::vector<int8_t> outputScaleHostData(M * CeilDiv((N / 2), 64) * 2, 1);
        std::vector<int64_t> tuningConfigData = {1};
        aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

        int64_t quantMode = 2;
        int64_t dequantMode = 2;
        int64_t dequantDtype = 0;
        int64_t groupListType = 1;

        // Create the x aclTensor.
        std::vector<unsigned char> xHostDataUnsigned(xHostData.begin(), xHostData.end());
        ret = CreateAclTensor<uint8_t>(xHostDataUnsigned, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, aclFormat::ACL_FORMAT_ND, &x);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Create the weight aclTensorList.
        std::vector<std::vector<int8_t>> weightHostDataList = {weightHostData};
        std::vector<std::vector<int64_t>> weightShapeList = {weightShape};
        ret = CreateAclTensorList<int8_t>(weightHostDataList, weightShapeList, &weightDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &weight);
        std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> weightTensorListPtr(weight, aclDestroyTensorList);
        std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Create the weightScale aclTensorList.
        std::vector<std::vector<int8_t>> weightScaleHostDataList = {weightScaleHostData};
        std::vector<std::vector<int64_t>> weightScaleShapeList = {weightScaleShape};
        ret = CreateAclTensorList<int8_t>(weightScaleHostDataList, weightScaleShapeList, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &weightScale);
        std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> weightScaleTensorListPtr(weightScale, aclDestroyTensorList);
        std::unique_ptr<void, aclError (*)(void*)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Create the xScale aclTensor.
        ret = CreateAclTensor<int8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &xScale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xScaleTensorPtr(xScale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Create the group_list aclTensor.
        ret = CreateAclTensor<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupListTensorPtr(groupList, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Create the y aclTensor.
        ret = CreateAclTensor<int8_t>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, aclFormat::ACL_FORMAT_ND, &output);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outputTensorPtr(output, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outputDeviceAddrPtr(outputDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Create the yScale aclTensor.
        ret = CreateAclTensor<int8_t>(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &outputScale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outputScaleTensorPtr(outputScale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void*)> outputScaleDeviceAddrPtr(outputScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        void* workspaceAddr = nullptr;

        // 3. Call the CANN operator library API.
        // Call the first-phase API of aclnnGroupedMatmulSwigluQuantV2.
        ret = aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(x, weight, weightScale, nullptr, nullptr, xScale, nullptr, groupList,
                                                            dequantMode, dequantDtype, quantMode, groupListType, nullptr, output, outputScale, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API of aclnnGroupedMatmulSwigluQuantV2.
        ret = aclnnGroupedMatmulSwigluQuantV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2 failed. ERROR: %d\n", ret); return ret);

        // 4. (boilerplate) Synchronously wait for the task execution to complete.
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. Obtain the output value and copy the result from Device memory to the Host. Modify it based on the specific API definition.
        auto size = GetShapeSize(outputShape);
        std::vector<int8_t> outputData(size, 0);
        ret = aclrtMemcpy(outputData.data(), size * sizeof(outputData[0]), outputDeviceAddr,
                          size * sizeof(outputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputData from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, outputData[j]);
        }

        size = GetShapeSize(outputScaleShape);
        std::vector<int8_t> outputScaleData(size, 0);
        ret = aclrtMemcpy(outputScaleData.data(), size * sizeof(outputScaleData[0]), outputScaleDeviceAddr,
                          size * sizeof(outputScaleData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputScaleData from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, outputScaleData[j]);
        }
        return ACL_SUCCESS;
    }

    int main()
    {
        // (boilerplate) device/stream initialization. For details, see the AscendCL external API list.
        // Set deviceId based on the actual device.
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = aclnnGroupedMatmulSwigluQuantV2Test(deviceId, stream);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2Test failed. ERROR: %d\n", ret); return ret);

        Finalize(deviceId, stream);
        return 0;
    }
    ```

  <!-- end id15 -->
