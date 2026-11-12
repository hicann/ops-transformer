# aclnnGroupedMatmulSwigluQuantWeightNzV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:40:28.403Z -->

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

- Description: Fuses GroupedMatmul, dequant, swiglu, and quant. For details, see the formula. This is a weightNZ-specific version of the [aclnnGroupedMatmulSwigluQuantV2](./aclnnGroupedMatmulSwigluQuantV2_en.md) API. The difference between this API and aclnnGroupedMatmulSwigluQuantV2 is that the weight parameter is forcibly treated as FRACTAL_NZ format in this scenario.

- Formula:

  <!-- npu="A3,910b" id7 -->
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

        - 1. Determine the tokens of the current group according to groupList[i], where $i \in [0,Len(groupList)]$.

          >Example: Assume groupList=[3,4,4,6] and groupListType=cumsum, or groupList=[3,1,0,2] and groupListType=count.
          >
          >Note: The two different grouping methods above actually produce the same grouping result.
          >
          >The 0th right matrix `W[0,:,:]` corresponds to the tokens `x[0:3]` at index positions [0,3) (3-0=3 tokens in total), and corresponds to `x_scale[0:3]`, `w_scale[0]`, `bias[0]`, `offset[0]`, `Q[0:3]`, `Q_scale[0:3]`, and `Q_offset[0:3]`;
          >
          >The 1st right matrix `W[1,:,:]` corresponds to the tokens `x[3:4]` at index positions [3,4) (4-3=1 token in total), and corresponds to `x_scale[3:4]`, `w_scale[1]`, `bias[1]`, `offset[1]`, `Q[3:4]`, `Q_scale[3:4]`, and `Q_offset[3:4]`;
          >
          >The 2nd right matrix `W[2,:,:]` corresponds to the tokens `x[4:4]` at index positions [4,4) (4-4=0 tokens in total), and corresponds to `x_scale[4:4]`, `w_scale[2]`, `bias[2]`, `offset[2]`, `Q[4:4]`, `Q_scale[4:4]`, and `Q_offset[4:4]`;
          >
          >The 3rd right matrix `W[3,:,:]` corresponds to the tokens `x[4:6]` at index positions [4,6) (6-4=2 tokens in total), and corresponds to `x_scale[4:6]`, `w_scale[3]`, `bias[3]`, `offset[3]`, `Q[4:6]`, `Q_scale[4:6]`, and `Q_offset[4:6]`;
          >
          >Note: The parts not specified in grouplist will not be updated.
          >For example, when groupList=[12,14,18], GroupListType=cumsum, and the shape of X is [30,:].
          >
          >Then the shape of the first output Q is [30,:], where the part Q[18:,:] is not updated or initialized, and the data is the original data of the device memory space upon allocation.
          >
          >Similarly, the shape of the second output Q_scale is [30], where Q\_scale[18:] is not updated or initialized, and the data is the original data of the device memory space upon allocation.
          >
          >That is, Q[:grouplist[-1],:] and Q\_scale[:grouplist[-1]] of the output are the valid data.

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

          - 3.1. Convert the left matrix $\mathbb{Z_8}$ into high-bit and low-bit parts $\mathbb{Z_4}$
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

        - 1. Determine the current group's tokens according to groupList[i], $i \in [0,Len(groupList)]$.
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
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    <details>
    <summary>A8W8 Pertoken quantization:</summary>

      - **Definition**:

        - **⋅** denotes matrix multiplication.
        - **⊙** indicates per-element multiplication.
      - **Input**:

        - $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $w\_scale∈\mathbb{R}^{E \times N}$: Per-channel scale factor of the grouped weight matrix (right matrix), where E is the number of experts and N is the output dimension.
        - $x\_scale∈\mathbb{R}^{M}$: Per-token scaling factor of the activation matrix (left matrix). M is the total number of tokens, and K is the feature dimension.
        - $grouplist∈\mathbb{N}^{E}$: group index list of cumsum or count.
      - **Output**:

        - $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        - $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
      - **Computation Process**
        - 1. Determine the current group's tokens according to groupList[i], $i \in [0,Len(groupList)]$

           - Perform the following computation based on the input parameters determined by the group:

             $C_{i} = (X_{i}\cdot W_{i} )\odot xScale_{i} \odot wScale_{i}$

             $C_{i,act}, gate_{i} = split(C_{i})$

             $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$

             where $xScale_{i}$ is the quantization factor corresponding to the token
           - Quantize the output result

             $Q\_scale_{i} = \frac{max(|S_{i}|)}{max(type)}$

             $Q_{i} = \lfloor \frac{S_{i}}{Q\_scale_{i}} \rceil$
    </details>
    <details>
    <summary>MX quantization scenario:</summary>

      - **Definition**:

        - **⋅** denotes matrix multiplication.
        - **⊙** indicates per-element multiplication.
      - **Inputs**:

        - $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $w\_scale∈\mathbb{R}^{E \times ceil(K / 64) \times N \times 2}$: per-channel scaling factor of the grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $x\_scale∈\mathbb{R}^{M \times ceil(K / 64) \times 2}$: per-token scaling factor of the activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $grouplist∈\mathbb{N}^{E}$: group index list of cumsum or count.
      - **Outputs**:

        - $Q∈\mathbb{F}_{8E4M3FN}^{M \times N / 2}$ or $Q∈\mathbb{F}_{4E1M2}^{M \times N / 2}$ or $Q∈\mathbb{F}_{4E2M1}^{M \times N / 2}$: quantized output matrix. Supported data types: FLOAT8_E4M3FN, FLOAT4_E1M2, FLOAT4_E2M1.
        - $Q\_scale∈\mathbb{R}^{M \times ceil((N / 2) / 64) \times 2}$: Quantization scale factor.
      - **Computation Process**
        - 1. Determine the tokens of the current group according to groupList[i], $i \in [0,Len(groupList)]$

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
            | FLOAT4_E1M2   |  1   |
            | FLOAT4_E2M1   |  2   |

          - $blocksize$: The number of elements quantized each time. Only 32 is supported.
    </details>

  <!-- end id8 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that includes the operator computation flow, and then call the "aclnnGroupedMatmulSwigluQuantWeightNzV2" API to perform computation.

```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
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
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize

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
        <td>INT8, INT4<sup>2</sup>, INT32<sup>2</sup>, FLOAT8_E4M3FN<sup>1</sup>, FLOAT4_E2M1<sup>1</sup>, FLOAT4_E1M2<sup>1</sup>, HIFLOAT8<sup>1</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td rowspan="1">Input</td>
        <td>Represents the weight matrix, corresponding to W in the formula.</td>
        <td>The weight of this API is forced to be treated as the FRACTAL_NZ format.</td>
        <td>INT8, INT4<sup>2</sup>, INT32<sup>2</sup>, FLOAT8_E4M3FN<sup>1</sup>, FLOAT4_E2M1<sup>1</sup>, FLOAT4_E1M2<sup>1</sup>, HIFLOAT8<sup>1</sup></td>
        <td>FRACTAL_NZ</td>
        <td>4, 5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightScale</td>
        <td rowspan="1">Input</td>
        <td>Represents the quantization factor of the right matrix, which is wScale in the formula.</td>
        <td>The first axis length must be equal to the first axis dimension of weight, and the tail axis length must be the same as the tail axis of weight restored to the ND format.</td>
        <td>UINT64<sup>2</sup>, FLOAT, FLOAT16, BFLOAT16, FLOAT8_E8M0<sup>1</sup></td>
        <td>ND</td>
        <td>1, 2, 3, 4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightAssistMatrix</td>
        <td rowspan="1">Optional input</td>
        <td>Represents the assist matrix used in matrix multiplication computation, i.e., the weightAssistMatrix in the formula.</td>
        <td><ul>
          <li>Takes effect only in the A8W4 scenario. A null pointer must be passed in other scenarios.</li>
          <li>The first axis length must be equal to the first axis dimension of weight, and the tail axis length must be the same as the tail axis of weight restored to ND format.</li>
        </ul></td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1, 2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td rowspan="1">Optional input</td>
        <td>Offset value of the matrix multiplication computation.</td>
        <td>Reserved input, not supported currently, must be a null pointer.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>xScale</td>
        <td rowspan="1">Input</td>
        <td>Represents the quantization factor of the left matrix, that is, xScale in the formula.</td>
        <td>-</td>
        <td>FLOAT, FLOAT8_E8M0<sup>1</sup></td>
        <td>ND</td>
        <td>1, 3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>smoothScale</td>
        <td rowspan="1">Optional input</td>
        <td>Represents the smooth scaling factor.</td>
        <td><ul>
        <li>Optional in the A4W4 use case; a null pointer must be passed in other use cases.</li>
        <li>In the A4W4 use case, the first axis length must be equal to the first axis dimension of weight.</li>
        <li>In the A4W4 use case, a null pointer or one of the following two shapes is supported: (E, N / 2) or (E,).</li>
        </ul></td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1, 2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>groupList</td>
        <td rowspan="1">Input</td>
        <td>Represents the distribution of matmul sizes of the input and output along the group axis, corresponding to grouplist in the formula. The specific meaning is determined by groupListType.</td>
        <td><ul>
          <li>The length must be equal to the first axis dimension of weight.</li>
          <li>When groupListType is 0, the last value must not be greater than the first dimension of the tensor in x. When groupListType is 1, the sum of the values must not be greater than the first dimension of the tensor in x.</li>
          <li>The values in groupList constrain the valid part of the output data. The part not specified by groupList will not be updated.</li>
        </ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dequantMode</td>
        <td rowspan="1">Input</td>
        <td>Represents the dequantization computation type, used to determine the dequantization method of the activation matrix and weight matrix.</td>
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
          <li>0 indicates FLOAT.</li>
          <li>1 indicates FLOAT16.</li>
          <li>27 indicates BFLOAT16.</li>
          <li>28 indicates UNDEFINED.</li>
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
          <li>2 indicates MX quantization.</li>
        </ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupListType</td>
        <td rowspan="1">Input</td>
        <td>Indicates the grouping interpretation method, used to determine the semantics of groupList.</td>
        <td><ul><li>0 indicates the cumsum mode. Each element in groupList represents the cumulative length of the current group.</li><li>1 indicates the count mode. Each element in groupList represents the number of elements in the group.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>tuningConfigOptional</td>
        <td rowspan="1">Optional input</td>
        <td>Used by the operator to estimate the size of M/E and select different operator templates to meet the performance requirements of different use cases.</td>
        <td>Array. The first number passed in indicates the expected number of tokens processed by each expert and is used to optimize tiling. This parameter is enabled for the A4W4 right matrix NZ input. Pass in a null pointer for other inputs.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>Output</td>
        <td rowspan="1">output</td>
        <td>Represents the quantized output, Q in the formula.</td>
        <td>-</td>
        <td>INT8, FLOAT8_E4M3FN<sup>1</sup>, FLOAT4_E1M2<sup>1</sup>, FLOAT4_E2M1<sup>1</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputScale</td>
        <td rowspan="1">Output</td>
        <td>Represents the quantization factor of the output, i.e., QScale in the formula.</td>
        <td>-</td>
        <td>FLOAT, FLOAT8_E8M0<sup>1</sup></td>
        <td>ND</td>
        <td>1, 3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td rowspan="1">Output</td>
        <td>Returns the size of the workspace side-requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td rowspan="1">Output</td>
        <td>Returns the op executor, which includes the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
    </table>

    <!-- npu="A3,910b" id9 -->
    - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
      - <strong>weight is forcibly treated as FRACTAL_NZ format.</strong>
      - The superscript "1" in the data type column of the table above indicates data types not supported by this series.
      - weight supports transposition in the A4W4 scenario, while other inputs support only the non-transposed format. INT32 is used for adaptation in A8W4 and A4W4 scenarios, where one INT32 is interpreted as eight INT4 data values in practice. The A8W8 scenario does not support the ND data format.
      - The dequantMode parameter is supported: values 0 and 1 are supported in A8W4 and A4W4 scenarios, while only value 0 is supported in the A8W8 scenario.
      - The dequantDtype and quantMode parameters are not supported.
      - x and weight do not support empty tensors.
      - When weight is input in NZ transpose format, only single-tensor mode is supported.
      - weight, weightScale, and weightAssistMatrix support single-tensor scenarios (tensorlist length is 1) and multi-tensor scenarios (tensorlist length is greater than 1).
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>:
      - <strong>weight is forcibly treated as FRACTAL_NZ format.</strong>
      - The superscript "2" in the data type column of the table above indicates data types not supported by this series.
      - Supports the dequantMode parameter: supported values are 0 for the A8W8 Pertoken quantization scenario and 2 for the MX quantization scenario.
      - The dequantDtype parameter is supported: the A8W8 Pertoken quantization scenario supports values 0, 1, and 27, and the MX quantization scenario supports value 0.
      - The quantMode parameter is supported: the A8W8 Pertoken quantization scenario supports value 0, and the MX quantization scenario supports value 2.
      - Only the same value for dequantMode and quantMode is supported.
      - x and xScale support empty tensors with M equal to 0.
      - weight and weightScale support empty tensors with N equal to 0.
      - In MXFP4 and MXFP8 scenarios, weight and weightScale support both the single-Tensor scenario (the tensorlist length must be 1) and the multi-Tensor scenario (the tensorlist length is greater than or equal to 1). In the multi-Tensor scenario, the tensorlist length must equal E, and E can be 1. The shapes of weight and weightScale must be flattened along the E dimension. For example, {(E, K, N)} must be changed to {E x (K, N)}.
      - In the MXA8W4 scenario, weight and weightScale support the multi-Tensor scenario (the tensorlist length is greater than or equal to 1).
      - The MXFP4 scenario supports weight in NZ format. x and weight can be FLOAT4_E2M1 or FLOAT4_E1M2 (cross combinations are supported), and output can be FLOAT8_E4M3FN, FLOAT4_E1M2, or FLOAT4_E2M1.
      - In the A8W8 Pertoken quantization scenario, weight and weightScale support only the single-Tensor scenario, that is, the tensorlist length is 1.

    <!-- end id10 -->

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
        <td>Parameters x, weight, weightScale, xScale, groupList, output, and outputScale are null pointers.</td>
      </tr>
      <tr>
        <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="9">161002</td>
        <td>The data dimensions of the input x, weight, weightScale, xScale, groupList, output, or outputScale do not satisfy the constraint.</td>
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
        <td>In use cases that do not support multiple tensors, the tensor list length of the input weight or weightScale is not 1; or in multiple-tensor use cases, the tensor list lengths of weight and weightScale are inconsistent.</td>
      </tr>
      <tr>
        <td>The input x and xScale are empty tensors, and the input weight and weightScale are empty tensorLists.</td>
      </tr>
      <tr>
        <td>The number of elements in groupList is greater than the first axis length of weight.</td>
      </tr>
      <tr>
        <td>The passed-in dequantMode, quantMode, or dequantDtype does not satisfy the constraint.</td>
      </tr>
      <tr>
        <td>The passed-in bias, weightAssistMatrix, smoothScale, or tuningConfigOptional does not satisfy the constraint.</td>
      </tr>
    </tbody>
    </table>

## aclnnGroupedMatmulSwigluQuantWeightNzV2

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
      <tr><td>workspaceSize</td><td>Input</td><td>Device side-requested workspace size, obtained from the first-phase API aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>op executor, includes the computation process of the operator.</td></tr>
      <tr><td>stream</td><td>Input</td><td>Specifies the Stream for task execution.</td></tr>
    </tbody>
  </table>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - Deterministic computation:
      - aclnnGroupedMatmulSwigluQuantWeightNzV2 defaults to a deterministic implementation.

  <!-- npu="A3,910b" id11 -->
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
              <th>quantization scenario</th>
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
              <td>NZ format shape: {(E, N / 32, K / 16, 16, 32)}</td>
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
              <li>For the ND format, the shape is {(E, K, N)}.</li>
              <li>For the NZ format with INT4, the shape is {(E, N / 64, K / 16, 16, 64)}.</li>
              <li>For the NZ format with INT32, the shape is {(E, N / 64, K / 16, 16, 8)}.</li></ul></td>
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
              <li>For ND format, the shape is {(E, K, N)}.</li>
              <li>A4W4 supports non-transposed and transposed NZ.</li>
              <li>For NZ non-transposed format with INT4, the shape is {(E, N / 64, K / 16, 16, 64)}.</li>
              <li>For NZ non-transposed format with INT32, the shape is {(E, N / 64, K / 16, 16, 8)}.</li>
              <li>For NZ transposed format with INT4, the original shape is {(E, K / 64, N / 16, 16, 64)}, and the input is passed in after calling transpose(-1,-2).</li>
              <li>For NZ transposed format with INT32, the original shape is {(E, K / 64, N / 16, 16, 8)}, and the input is passed in after calling transpose(-1,-2).</li>
             <li>For NZ transposed input, align per-group K/K_group_num to 64.</li></ul>
              </td>
              <td><ul>
              <li>For the per-channel use case, the shape is {(E, N)}.</li>
              <li>For the per-group use case, the shape is {(E, K_group_num, N)}.</li></ul>
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
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The first dimension of groupList supports a maximum of 1024, i.e., a maximum of 1024 groups.
    - The A8W8 Pertoken quantization scenario must meet the following constraints:
        - x, weight, and output support the following data type combinations; the data types of xScale and outputScale are both FLOAT:
          - INT8: x, weight, and output are all INT8; weightScale supports FLOAT, FLOAT16, or BFLOAT16.
          - HIFLOAT8: x, weight, and output are all HIFLOAT8; weightScale supports FLOAT or BFLOAT16.
          - FP8: x and weight support FLOAT8_E4M3FN; weightScale supports FLOAT or BFLOAT16; output supports FLOAT8_E4M3FN and FLOAT_E5M2.
        - The shape of x is (M, K). weight supports non-transposed and transposed input, with the entry view shape (E, K, N) for both: non-transposed input uses a contiguous stride, with FRACTAL_NZ storage shape (E, ceil(N/32), ceil(K/16), 16, 32); transposed input uses the stride (K * N, 1, K), with FRACTAL_NZ storage shape (E, ceil(K/32), ceil(N/16), 16, 32).
        - x supports only non-transposed input. When weight is transposed input, the internal `transposeWeight` attribute of the operator is true, and the ZN template is used to process FRACTAL_NZ data.
        - The shape of weightScale is (E, N), the shapes of xScale and outputScale are (M,), and the shape of output is (M, N/2).
        - Both dequantMode and quantMode must be 0; dequantDtype supports 0, 1, and 27.
        - N must be greater than 0 and an integer multiple of 64.
        - The tensorlist lengths of weight and weightScale must both be 1.
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
              <th>MX quantization</th>
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
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E8M0</td>
            </tr>
          <tr>
            <td>MXFP4</td>
            <td>FLOAT4_E2M1, FLOAT4_E1M2</td>
            <td>FLOAT4_E2M1, FLOAT4_E1M2</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT4_E1M2</td>
            <td>FLOAT8_E8M0</td>
          </tr>
            <tr>
              <td>MXA8W4</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT4_E2M1</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E8M0</td>
              <td>FLOAT8_E4M3FN</td>
              <td>FLOAT8_E8M0</td>
            </tr>
          </tbody>
          </table>

        - The shape constraints must meet the following table:
          <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
          <col style="width: 130px">
          <col style="width: 130px">
          <col style="width: 250px">
          <col style="width: 320px">
          <col style="width: 180px">
          <col style="width: 250px">
          <col style="width: 160px">
          <col style="width: 160px">
          </colgroup>
          <thead>
            <tr>
              <th>MX quantization</th>
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
              <td>(M, K)</td>
              <td><ul>
              <li>Non-transposed shape: {(E, ceil(N / 32), ceil(K / 16), 16, 32)}</li>
              <li>Multi-Tensor non-transposed shape: {E × (ceil(N / 32), ceil(K / 16), 16, 32)}, viewShape: {E × (K, N)}</li>
              <li>Transposed shape: {(E, ceil(K / 32), ceil(N / 16), 16, 32)}</li></ul></td>
              <td><ul>
              <li>Non-transposed shape: {(E, ceil(K / 64), N, 2)}</li>
              <li>Multi-tensor non-transposed shape: {E tensors of (ceil(K / 64), N, 2)}</li>
              <li>Transposed shape: {(E, N, ceil(K / 64), 2)}</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
            <tr>
              <td>MXFP4</td>
              <td>(M, K)</td>
              <td><ul>
              <li>Non-transposed shape: {(E, ceil(N / 64), ceil(K / 16), 16, 64)}</li>
              <li>Multi-tensor non-transposed shape: {E tensors of (ceil(N / 64), ceil(K / 16), 16, 64)}, viewShape: {E tensors of (K, N)}</li>
              <li>Transposed shape: {(E, ceil(K / 64), ceil(N / 16), 16, 64)}</li></ul></td>
              <td><ul>
              <li>Non-transposed shape: {(E, ceil(K / 64), N, 2)}</li>
              <li>Multi-tensor non-transposed shape: {E tensors of (ceil(K / 64), N, 2)}</li>
              <li>Transposed shape: {(E, N, ceil(K / 64), 2)}</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
            <tr>
              <td>MXA8W4</td>
              <td>(M, K)</td>
              <td><ul>
              <li>The transposed shape is in the form of {(E, K/32, N/16, 16, 32)}.</li></ul></td>
              <td><ul>
              <li>The transposed shape is in the form of {(E, N, ceil(K / 64), 2)}.</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
            <tr>
              <td>MXA8W4 (multiple tensors)</td>
              <td>(M, K)</td>
              <td><ul>
              <li>The transposed view shape is in the form of {E x (N, K)}, and the storage shape is in the form of {E x (N/32, K/16, 16, 32)}</li></ul></td>
              <td><ul>
              <li>The transposed view shape is in the form of {E x (N, ceil(K / 64), 2)}</li></ul></td>
              <td>(M, ceil(K / 64), 2)</td>
              <td>(M, N / 2)</td>
              <td>(M, ceil((N / 2) / 64), 2)</td>
            </tr>
          </tbody>
          </table>

        - The transpose attribute of weightScale must be consistent with that of weight.
        - In the MX quantization scenario, when the input is of the fp8 data type, N must be 64-aligned; when the input is of the fp4 data type, N must be 128-aligned.
        - For MXFP4 and MXFP8 multi-Tensor use cases, the tensorlist length of weight and weightScale must be equal to E.
        - For MXFP4, K must be greater than 2.
        - For MXFP4, the inner axes of both the left and right matrices must be even.
        - For MXFP4, the last two axes of weight must not be 1.
        - In the MXA8W4 scenario, the following constraints must be met:
          - K % 32 == 0, indicating that the K axis must be aligned to 32.
          - K >= 32.
          - N >= 128.
          - Only transposed weight is supported. The user needs to construct weight data in transposed NZ format and transpose it to the non-transposed state before the call.
          - In the multi-tensor case (tensorlist length is greater than or equal to 1), the number of tensors in weight and weightScale must be the same.
          - In the multi-tensor case, K and N of each tensor in weight must be equal.
          - In the multi-tensor case, the view shape of each tensor in weight is 2D, and the view shape of each tensor in weightScale is 3D; in the single-tensor case, the view shape of each tensor in weight is 3D, and the view shape of each tensor in weightScale is 4D.
          - In the multi-tensor case, the shape of each weightScale tensor must match that of the corresponding weight tensor.
          - In the multi-tensor case, the length of groupList must be equal to the number of tensors in weight/weightScale.

  <!-- end id12 -->

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  <!-- npu="A3,910b" id13 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.

    ```cpp
    #include <iostream>
    #include <vector>
    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"

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
        // Boilerplate for resource initialization
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
        // Call aclrtMalloc to allocate device-side memory
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy host-side data to device-side memory
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Compute the strides of the contiguous tensor
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
        }

        // Call aclCreateTensor to create an aclTensor
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
        // 1. (Boilerplate) device/stream initialization, see the acl API manual
        // Set deviceId based on the actual device
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = Init(deviceId, &stream);
        // Handle the check as needed
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        // 2. Construct the input and output. Custom construction is required based on the API.
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

        std::vector<int8_t> xHostData(M * K, 1);
        std::vector<int8_t> weightHostData(E * N * K, 1);
        std::vector<float> weightScaleHostData(E * N, 0.5f);
        std::vector<float> xScaleHostData(M, 0.0314f);
        std::vector<int64_t> groupListHostData = {1, 2, 2, 3};
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
        // Call the first-phase API of aclnnGroupedMatmulSwigluQuantWeightNzV2.
        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
            x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode, dequantDtype,
            quantMode, groupListType, tuningConfig, output, outputScale, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API aclnnGroupedMatmulSwigluQuantWeightNzV2.
        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2 failed. ERROR: %d\n", ret); return ret);

        // 4. (boilerplate) Wait synchronously for the task execution to complete.
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. Obtain the output value and copy the result from the device memory to the host memory. Modify the code according to the specific API definition.
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
        // 6. Release aclTensor, aclTensorList, and aclScalar. Modify the code according to the specific API definition.
        aclDestroyTensor(x);
        aclDestroyTensorList(weight);
        aclDestroyTensorList(weightScale);
        aclDestroyTensor(xScale);
        aclDestroyTensor(groupList);
        aclDestroyTensor(output);
        aclDestroyTensor(outputScale);

        aclDestroyIntArray(tuningConfig);

        // 7. Release device resources. Modify the code according to the specific API definition.
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

  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950PR&950DT products</term>:

    ```cpp
    #include <iostream>
    #include <memory>
    #include <vector>

    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"

    #define CHECK_RET(cond, return_expr) \
        do { \
            if (!(cond)) { \
                return_expr; \
            } \
        } while (0)

    #define CHECK_FREE_RET(cond, return_expr) \
        do { \
            if (!(cond)) { \
                Finalize(deviceId, stream); \
                return_expr; \
            } \
        } while (0)

    #define LOG_PRINT(message, ...) \
        do { \
            printf(message, ##__VA_ARGS__); \
        } while (0)

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shapeSize = 1;
        for (auto i : shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream *stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSetDevice(deviceId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclFormat formatType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType,
                                  shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    template <typename T>
    int CreateAclTensorList(const std::vector<std::vector<T>> &hostData,
                            const std::vector<std::vector<int64_t>> &shapes,
                            void **deviceAddr, aclDataType dataType, aclFormat formatType, aclTensorList **tensor)
    {
        int size = shapes.size();
        aclTensor *tensors[size];
        for (int i = 0; i < size; i++) {
            int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, formatType, tensors + i);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
        }
        *tensor = aclCreateTensorList(tensors, size);
        return ACL_SUCCESS;
    }

    template <typename T>
    int CreateAclTensorListNZTranspose(const std::vector<std::vector<T>> &hostData,
                                       const std::vector<std::vector<int64_t>> &viewShapes,
                                       const std::vector<std::vector<int64_t>> &storageShapes,
                                       const std::vector<std::vector<int64_t>> &transposeStrides,
                                       void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
    {
        int size = storageShapes.size();
        aclTensor *tensors[size];
        for (int i = 0; i < size; i++) {
            auto storageSize = GetShapeSize(storageShapes[i]) * sizeof(T);
            auto ret = aclrtMalloc(deviceAddr + i, storageSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

            ret = aclrtMemcpy(*(deviceAddr + i), storageSize, hostData[i].data(), storageSize, ACL_MEMCPY_HOST_TO_DEVICE);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

            tensors[i] = aclCreateTensor(viewShapes[i].data(), viewShapes[i].size(), dataType,
                                         transposeStrides[i].data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                         storageShapes[i].data(), storageShapes[i].size(), *(deviceAddr + i));
        }
        *tensor = aclCreateTensorList(tensors, size);
        return ACL_SUCCESS;
    }

    template <typename T>
    int CreateAclTensorListTranspose(const std::vector<std::vector<T>> &hostData,
                                     const std::vector<std::vector<int64_t>> &viewShapes,
                                     const std::vector<std::vector<int64_t>> &transposeStrides,
                                     void **deviceAddr, aclDataType dataType, aclFormat formatType, aclTensorList **tensor)
    {
        int size = viewShapes.size();
        aclTensor *tensors[size];
        for (int i = 0; i < size; i++) {
            auto storageSize = GetShapeSize(viewShapes[i]) * sizeof(T);
            auto ret = aclrtMalloc(deviceAddr + i, storageSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

            ret = aclrtMemcpy(*(deviceAddr + i), storageSize, hostData[i].data(), storageSize, ACL_MEMCPY_HOST_TO_DEVICE);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

            tensors[i] = aclCreateTensor(viewShapes[i].data(), viewShapes[i].size(), dataType,
                                         transposeStrides[i].data(), 0, formatType, viewShapes[i].data(),
                                         viewShapes[i].size(), *(deviceAddr + i));
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

    int aclnnGroupedMatmulSwigluQuantWeightNzV2MxA8W4Test(int32_t deviceId, aclrtStream &stream)
    {
        auto ret = Init(deviceId, &stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        int64_t E = 8;
        int64_t M = 2048;
        int64_t N = 4096;
        int64_t K = 4096;

        int64_t kScaleBlock = CeilDiv(K, 64);

        std::vector<int64_t> xShape = {M, K};
        std::vector<std::vector<int64_t>> weightViewShapes = {{E, K, N}};
        std::vector<std::vector<int64_t>> weightStorageShapes = {{E, K / 32, N / 16, 16, 32}};
        std::vector<std::vector<int64_t>> weightTransposeStrides = {{K * N, 1, K}};
        std::vector<std::vector<int64_t>> weightScaleViewShapes = {{E, kScaleBlock, N, 2}};
        std::vector<std::vector<int64_t>> weightScaleTransposeStrides = {{kScaleBlock * N * 2, 2, kScaleBlock * 2, 1}};
        std::vector<int64_t> xScaleShape = {M, CeilDiv(K, 64), 2};
        std::vector<int64_t> groupListShape = {E};
        std::vector<int64_t> outputShape = {M, N / 2};
        std::vector<int64_t> outputScaleShape = {M, CeilDiv((N / 2), 64), 2};

        void *xDeviceAddr = nullptr;
        void *weightDeviceAddr = nullptr;
        void *weightScaleDeviceAddr = nullptr;
        void *xScaleDeviceAddr = nullptr;
        void *groupListDeviceAddr = nullptr;
        void *outputDeviceAddr = nullptr;
        void *outputScaleDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensorList *weight = nullptr;
        aclTensorList *weightScale = nullptr;
        aclTensor *xScale = nullptr;
        aclTensor *groupList = nullptr;
        aclTensor *output = nullptr;
        aclTensor *outputScale = nullptr;

        std::vector<uint8_t> xHostData(M * K, 1);
        std::vector<uint8_t> weightHostData(E * (K / 32) * (N / 16) * 16 * 32, 1);
        std::vector<uint8_t> weightScaleHostData(E * kScaleBlock * N * 2, 1);
        std::vector<uint8_t> xScaleHostData(M * CeilDiv(K, 64) * 2, 1);
        std::vector<int64_t> groupListHostData(E, 256);
        std::vector<uint8_t> outputHostData(M * N / 2, 1);
        std::vector<uint8_t> outputScaleHostData(M * CeilDiv((N / 2), 64) * 2, 1);

        aclIntArray *tuningConfig = nullptr;

        int64_t quantMode = 2;
        int64_t dequantMode = 2;
        int64_t dequantDtype = 0;
        int64_t groupListType = 1;

        ret = CreateAclTensor<uint8_t>(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                                       aclFormat::ACL_FORMAT_ND, &x);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<std::vector<uint8_t>> weightHostDataList = {weightHostData};
        ret = CreateAclTensorListNZTranspose<uint8_t>(weightHostDataList, weightViewShapes, weightStorageShapes,
                                                      weightTransposeStrides, &weightDeviceAddr,
                                                      aclDataType::ACL_FLOAT4_E2M1, &weight);
        std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorListPtr(weight,
                                                                                                   aclDestroyTensorList);
        std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<std::vector<uint8_t>> weightScaleHostDataList = {weightScaleHostData};
        ret = CreateAclTensorListTranspose<uint8_t>(weightScaleHostDataList, weightScaleViewShapes,
                                                    weightScaleTransposeStrides, &weightScaleDeviceAddr,
                                                    aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &weightScale);
        std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightScaleTensorListPtr(
            weightScale, aclDestroyTensorList);
        std::unique_ptr<void, aclError (*)(void *)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor<uint8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                                       aclFormat::ACL_FORMAT_ND, &xScale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xScaleTensorPtr(xScale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                       aclFormat::ACL_FORMAT_ND, &groupList);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor<uint8_t>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                                       aclFormat::ACL_FORMAT_ND, &output);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(output, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> outputDeviceAddrPtr(outputDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor<uint8_t>(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr,
                                       aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &outputScale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputScaleTensorPtr(outputScale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> outputScaleDeviceAddrPtr(outputScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
            x, weight, weightScale, nullptr, nullptr, xScale, nullptr, groupList, dequantMode, dequantDtype, quantMode,
            groupListType, tuningConfig, output, outputScale, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2 failed. ERROR: %d\n", ret);
                  return ret);

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        auto size = GetShapeSize(outputShape);
        std::vector<uint8_t> outputData(size, 0);
        ret = aclrtMemcpy(outputData.data(), size * sizeof(outputData[0]), outputDeviceAddr, size * sizeof(outputData[0]),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputData from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, outputData[j]);
        }

        size = GetShapeSize(outputScaleShape);
        std::vector<uint8_t> outputScaleData(size, 0);
        ret = aclrtMemcpy(outputScaleData.data(), size * sizeof(outputScaleData[0]), outputScaleDeviceAddr,
                          size * sizeof(outputScaleData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputScaleData from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, outputScaleData[j]);
        }

        return ACL_SUCCESS;
    }

    int main()
    {
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = aclnnGroupedMatmulSwigluQuantWeightNzV2MxA8W4Test(deviceId, stream);
        CHECK_FREE_RET(ret == ACL_SUCCESS,
                       LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2MxA8W4Test failed. ERROR: %d\n", ret);
                       return ret);

        Finalize(deviceId, stream);
        return 0;
    }
    ```

  <!-- end id14 -->
