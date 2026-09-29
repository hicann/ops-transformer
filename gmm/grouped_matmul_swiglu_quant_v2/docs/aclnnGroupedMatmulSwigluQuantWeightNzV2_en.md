# aclnnGroupedMatmulSwigluQuantWeightNzV2

[📄 View source code](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_swiglu_quant_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Fuses `GroupedMatmul`, `dequant`, `swiglu`, and `quant`. For details, see the formulas. This API is the weightNZ specialization version of [aclnnGroupedMatmulSwigluQuantV2](./aclnnGroupedMatmulSwigluQuantV2.md). The difference between this API and aclnnGroupedMatmulSwigluQuantV2 is that the weight parameter is forcibly regarded as the FRACTAL_NZ format in this scenario.

- Formulas:
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:

    <details>
    <summary>Quantization scenario A8W8 (A: activation matrix; W: weight matrix; 8: INT8)</summary>
    <a id="quantization-scenario-a8w8"></a>

      - **Definition**

        * **⋅** indicates matrix multiplication.
        * **⊙** indicates element-wise multiplication.
        * $\left \lfloor x\right \rceil$ indicates rounding `x` to the nearest integer.
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **Inputs**

        * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where $M$ indicates the total number of tokens and $K$ indicates the feature dimension.
        * $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where $E$ indicates the number of routed experts, $K$ indicates the feature dimension, and $N$ indicates the output dimension.
        * $w\_scale∈\mathbb{R}^{E \times N}$: per-channel scale factor for the grouped weight matrix (right matrix), where `E` indicates the number of experts and `N` indicates the output dimension.
        * $x\_scale∈\mathbb{R}^{M}$: per-token scale factor for the activation matrix (left matrix), where `M` indicates the total number of tokens.
        * $grouplist∈\mathbb{N}^{E}$: grouped index list of cumsum or count.
      - **Outputs**

        * $Q∈\mathbb{Z_8}^{M \times N/2}$: quantized output matrix.
        * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.

      - **Calculation process**

        1. Determine the tokens of the current group based on `groupList[i]`, where $i \in [0,Len(groupList)]$.

            >Example: `groupList=[3,4,4,6]`, `groupListType=cumsum` or `groupList=[3,1,0,2]`, `groupListType=count`
            >
            >Note: The two grouping methods yield the same grouping result.
            >
            >Zero-th right matrix `W[0,:,:]`, corresponding to tokens `x[0:3]` (3-0=3 tokens) at index positions [0,3), corresponding to `x_scale[0:3]`, `w_scale[0]`, `bias[0]`, `offset[0]`, `Q[0:3]`, `Q_scale[0:3]`, and `Q_offset[0:3]`
            >
            >First right matrix `W[1,:,:]`, corresponding to token `x[3:4]` (4-3=1 token) at index position [3,4), corresponding to `x_scale[3:4]`, `w_scale[1]`, `bias[1]`, `offset[1]`, `Q[3:4]`, `Q_scale[3:4]`, and `Q_offset[3:4]`
            >
            >Second right matrix `W[2,:,:]`, corresponding to token `x[4:4]` (4-4=0 token) at index position [4,4), corresponding to `x_scale[4:4]`, `w_scale[2]`, `bias[2]`, `offset[2]`, `Q[4:4]`, `Q_scale[4:4]`, and `Q_offset[4:4]`
            >
            >Third right matrix `W[3,:,:]`, corresponding to tokens `x[4:6]` (6-4=2 tokens) at index positions [4,6), corresponding to `x_scale[4:6]`, `w_scale[3]`, `bias[3]`, `offset[3]`, `Q[4:6]`, `Q_scale[4:6]`, and `Q_offset[4:6]`
            >
            >Note: Any portion not specified in `groupList` will not be updated.
            >Assume `groupList=[12,14,18]`, `GroupListType=cumsum`, and the shape of `X` is [30,:].
            >
            >The shape of the first output `Q` will be [30,:]. The portion `Q[18:,:]` will not be updated or initialized, and the data therein is consistent with the original data when the device memory is allocated.
            >
            >Similarly, the second output `Q` has a shape of [30]. The portion `Q_scale[18:]` will not be updated or initialized, and the data therein is consistent with the original data when the device memory is allocated.
            >
            >In other words, only `Q[:grouplist[-1],:]` and `Q_scale[:grouplist[-1]]` constitute the valid data portions.

        2. Perform the following computation based on the input parameters determined by grouping:

            $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

            $C_{i,act}, gate_{i} = split(C_{i})$

            $S_{i}=Swish(C_{i,act})\odot gate_{i}$ &nbsp;&nbsp; where $Swish(x)=\frac{x}{1+e^{-x}}$

        3. Quantize the output.

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>
    <details>
    <summary>MSD scenario A8W4 (A: activation matrix; W: weight matrix; 8: INT8; 4: INT4)</summary>
    <a id="msd-scenario-a8w4"></a>
    
      - **Definition**
        * **⋅** indicates matrix multiplication.
        * **⊙** indicates element-wise multiplication.
        * $\left \lfloor x\right \rceil$ indicates rounding `x` to the nearest integer.
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **Inputs**
        * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where $M$ indicates the total number of tokens and $K$ indicates the feature dimension.
        * $W∈\mathbb{Z_4}^{E \times K \times N}$: grouped weight matrix (right matrix), where $E$ indicates the number of routed experts, $K$ indicates the feature dimension, and $N$ indicates the output dimension.
        * $weightAssistMatrix∈\mathbb{R}^{E \times N}$: auxiliary matrix for matrix multiplication (the computation process for generating the auxiliary matrix is described below).
        * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$: per-channel scale factor for the grouped weight matrix (right matrix), where `E` indicates the number of experts, `K_group_num` indicates the number of groups along the K-axis, and `N` indicates the output dimension.
        * $x\_scale∈\mathbb{R}^{M}$: per-token scale factor for the activation matrix (left matrix), where `M` indicates the total number of tokens.
        * $grouplist∈\mathbb{N}^{E}$: grouped index list of cumsum or count.
      - **Outputs**
        * $Q∈\mathbb{Z_8}^{M \times N/2}$: quantized output matrix.
        * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
      - **Calculation Process:**
         1. Determine the tokens of the current group based on `groupList[i]`, where $i \in [0,Len(groupList)]$.
            - The grouping logic is the same as that of A8W8.
         2. The computation process of generating the auxiliary matrix (weightAssistMatrix) is as follows. (Note that the computation of weightAssistMatrix is generated offline and used as the input, instead of being completed inside the operator.)
            - For per-channel quantization ($w\_scale$ is 2D):

              $weightAssistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$

            - For per-group quantization ($w\_scale$ is 3D):

              $weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$

              Note: $num\_per\_group = K // K\_group\_num$

         3. Perform the following computation based on the input parameters determined by grouping:

            - 3.1. Convert the left matrix $\mathbb{Z_8}$ into two $\mathbb{Z_4}$ components that represent the high and low bits.
              $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
              $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
            - 3.2. Enable per-channel or per-group quantization during matrix multiplication.
              Per-channel:

              $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

              $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

              Per-group:

              $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

              $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

            - 3.3. Restore the matrix multiplication results of the high and low bits into the overall result.

              $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$

              $C_{i,act}, gate_{i} = split(C_{i})$

              $S_{i}=Swish(C_{i,act})\odot gate_{i}$ &nbsp;&nbsp; where $Swish(x)=\frac{x}{1+e^{-x}}$

         4. Quantize the output.

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>
    <details>
    <summary>Quantization scenario A4W4 (A indicates the activation matrix, W indicates the weight matrix, and 4 indicates the INT4 data type):</summary>
    <a id="quantization-scenario-a4w4"></a>

      - **Definition**

        * **⋅** indicates matrix multiplication.
        * **⊙** indicates element-wise multiplication.
        * $\left \lfloor x\right \rceil$ indicates rounding `x` to the nearest integer.
        * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **Inputs**

        * $X∈\mathbb{Z_4}^{M \times K}$: activation matrix (left matrix), where $M$ indicates the total number of tokens and $K$ indicates the feature dimension.
        * $W∈\mathbb{Z_4}^{E \times K \times N}$: grouped weight matrix (right matrix), where $E$ indicates the number of routed experts, $K$ indicates the feature dimension, and $N$ indicates the output dimension.
        * $w\_scale∈\mathbb{R}^{E \times N}$: per-channel scale factor for the grouped weight matrix (right matrix), where `E` indicates the number of experts and `N` indicates the output dimension.
        * $x\_scale∈\mathbb{R}^{M}$: per-token scale factor for the activation matrix (left matrix), where `M` indicates the total number of tokens.
        * $smoothScale∈\mathbb{R}^{E \times N/2} (per channel) or \mathbb{R}^{E} (per tensor)$: smooth scaling factor, where E is the number of experts, and N is the output dimension.
        * $grouplist∈\mathbb{N}^{E}$: grouped index list of cumsum or count.
      - **Outputs**

        * $Q∈\mathbb{Z_8}^{M \times N/2}$: quantized output matrix.
        * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.

      - **Calculation process**

         1. Determine the tokens of the current group based on `groupList[i]`, where $i \in [0,Len(groupList)]$.
            - The grouping logic is the same as that of A8W8.

         2. Perform the following computation based on the input parameters determined by grouping:

            $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

            $C_{i,act}, gate_{i} = split(C_{i})$

            $S_{i}=Swish(C_{i,act})\odot gate_{i}$ &nbsp;&nbsp; where $Swish(x)=\frac{x}{1+e^{-x}}$

            $S_{i} = S_{i} \odot smoothScale_{i\ Broadcast}$

         3. Quantize the output.

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, `aclnnGroupedMatmulSwigluQuantWeightNzV2` is called to perform computation.

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
    const aclIntArray   *tuningConfig, 
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
      <th>Name</th>
      <th style="white-space: nowrap">Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th style="white-space: nowrap">Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td rowspan="1">Input</td>
      <td>Left matrix, corresponding to X in the formula.</td>
      <td>-</td>
      <td>INT8, INT4, INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td rowspan="1">Input</td>
      <td>Weight matrix, corresponding to W in the formula.</td>
      <td>The weight of this API is forcibly regarded as the FRACTAL_NZ format.</td>
      <td>INT8, INT4, INT32</td>
      <td>FRACTAL_NZ</td>
      <td>4, 5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightScale</td>
      <td rowspan="1">Input</td>
      <td>Quantization factor of the right matrix, wScale in the formula.</td>
      <td>The length of the first axis must be the same as that of the first axis of the weight, and the length of the last axis must be the same as that of the last axis of the weight restored to the ND format.</td>
      <td>UINT64, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1, 2, 3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightAssistMatrix</td>
      <td rowspan="1">Optional input</td>
      <td>Auxiliary matrix used for matrix multiplication, which is specified by weightAssistMatrix in the formula.</td>
      <td><ul>
        <li>This parameter is effective only in the A8W4 scenario. In other scenarios, pass a null pointer.</li>
        <li>The length of the first axis must be the same as the first axis of weight. The length of the last axis must be the same as the last axis of weight restored to the ND format.</li>
      </ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1, 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias</td>
      <td rowspan="1">Optional input</td>
      <td>Offset for matrix multiplication.</td>
      <td>This input is reserved and is not supported currently. You need to pass a null pointer.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xScale</td>
      <td rowspan="1">Input</td>
      <td>Quantization factor of the left matrix, which is specified by xScale in the formula.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScale</td>
      <td rowspan="1">Optional input</td>
      <td>Smooth scale.</td>
      <td><ul>
      <li>This parameter is optional in the A4W4 scenario. In other scenarios, a null pointer needs to be transferred.</li>
      <li>In the A4W4 scenario, the length of the first axis must be the same as that of the first axis of weight.</li>
      <li>In the A4W4 scenario, a null pointer or two shapes (E, N/2) or (E,) is supported.</li>
      </ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1, 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td rowspan="1">Input</td>
      <td>Number of tokens involved in the computation for each group, corresponding to grouplist in the formula.</td>
      <td><ul>
        <li>The length must be the same as the first axis of weight.</li>
        <li>The last value in grouplist constrains the valid portion of the output data. For details, see the computation process.</li>
      </ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dequantMode</td>
      <td rowspan="1">Input</td>
      <td>Dequantization computation type, which determines the dequantization modes for the activation matrix and weight matrix.</td>
      <td><ul>
        <li>0: per-token mode for the activation matrix and per-channel mode for the weight matrix</li>
        <li>1: per-token mode for the activation matrix and per-group mode for the weight matrix</li>
      </ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dequantDtype</td>
      <td rowspan="1">Input</td>
      <td>Data type of the intermediate GroupedMatmul result.</td>
      <td><ul>
        <li>This field is not supported currently. The default value is 28.</li>
        <li>0 indicates FLOAT.</li>
        <li>1: FLOAT16</li>
        <li>27 indicates BFLOAT16.</li>
        <li>28: UNDEFINED</li>
      </ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td rowspan="1">Input</td>
      <td>Quantization computation type, which determines the quantization mode for the swiglu result.</td>
      <td><ul>
        <li>This field is not supported currently. The default value is 0.</li>
        <li>0: per-token mode</li></ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td rowspan="1">Input</td>
      <td>Interpretation mode of the groups, used to determine the semantics of groupList.</td>
      <td><ul><li>0: cumsum mode, where each element in groupList represents the cumulative length of the current group. </li><li>1: count mode, where each element in groupList represents the number of elements contained in that group.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tuningConfig</td>
      <td rowspan="1">Optional input</td>
      <td>Used to estimate the M/E size of the operator and select different operator templates to adapt to performance requirements in different scenarios.</td>
      <td>Array. The first number passed indicates the expected number of tokens processed by each expert, which is used to optimize tiling. This parameter is enabled when the right matrix NZ input of A4W4 is used. For other inputs, pass a null pointer.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td rowspan="1">Output</td>
      <td>Quantization result, Q in the formula.</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputScale</td>
      <td rowspan="1">Output</td>
      <td>Quantization factor, QScale in the formula.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td rowspan="1">Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td rowspan="1">Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas A3 training series products/Atlas A3 inference series products</term> and <term>Atlas A2 training series products/Atlas A2 inference series products</term>:
    - <strong>The weight is forcibly regarded as the FRACTAL_NZ format.</strong>
    - The weight supports transposition in the A4W4 scenario, and supports only non-transposition in other scenarios. INT32 is used for adaptation in the A8W4 and A4W4 scenarios. In practice, one INT32 is interpreted as eight INT4 data. The ND data format is not supported in the A8W8 scenario.
    - The dequantMode parameter is supported. In the A8W4 and A4W4 scenarios, the value can be 0 or 1. In the A8W8 scenario, the value can only be 0.
    - The dequantDtype and quantMode parameters are not supported.
    - x and weight do not support empty tensors.
    - When the weight is transposed in the NZ format, only the single-tensor mode is supported.
    - weight, weightScale, and weightAssistMatrix support both single-tensor scenarios (the length of the tensor list is 1) and multi-tensor scenarios (the length of the tensor list is greater than 1).

- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed;width: 1150px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
  <col style="width: 860px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The x, weight, weightScale, xScale, groupList, output, or outputScale parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data dimensions of the input x, weight, weightScale, xScale, groupList, output, or outputScale do not comply with the constraints.</td>
    </tr>
    <tr>
      <td>The shape of the input x, weight, weightScale, xScale, groupList, output, or outputScale does not comply with the constraints.</td>
    </tr>
    <tr>
      <td>The format of the input x, weight, weightScale, xScale, groupList, output, or outputScale does not comply with the constraints.</td>
    </tr>
    <tr>
      <td>The length of the tensor list of the input weight or weightScale is not 1.</td>
    </tr>
    <tr>
      <td>The input x or xScale is an empty tensor, or the input weight or weightScale is an empty tensor list.</td>
    </tr>
    <tr>
      <td>The number of elements in groupList is greater than the length of the first axis of weight.</td>
    </tr>
    <tr>
      <td>The input dequantMode, quantMode, and dequantDtype do not meet the requirements.</td>
    </tr>
    <tr>
      <td>The input bias, weightAssistMatrix, smoothScale, and tuningConfig do not meet the requirements.</td>
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
      <tr><td>workspace</td><td>Input</td><td>Address of the workspace to be allocated on the device.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API of aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Operator executor, containing the operator computation process.</td></tr>
      <tr><td>stream</td><td>Input</td><td>Stream for executing a task.</td></tr>
    </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatmulSwigluQuantWeightNzV2` defaults to a deterministic implementation.
- <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
  - In the A8W8/A8W4/A4W4 quantization scenario, the following restrictions must be met:
    - Data type requirements
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
            <th>Quantization Scenario</th>
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

    - The shape constraints must meet the requirements listed in the following table.
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
            <th>Quantization Scenario</th>
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
            <td>The shape in NZ format is similar to {(E, N / 32, K / 16, 16, 32)}.</td>
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
            <li>The shape in ND format is similar to {(E, K, N)}.</li>
            <li>The shape in NZ format and int4 is similar to {(E, N / 64, K / 16, 16, 64)}.</li>
            <li>The shape in NZ format and int32 is similar to {(E, N / 64, K / 16, 16, 8)}.</li></ul></td>
            <td><ul>
            <li>The shape in per-channel scenario is similar to {(E, N)}.</li>
            <li>The shape in per-group scenario is similar to {(E, K_group_num, N)}.</li></ul></td>
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
            <li>The shape in ND format is similar to {(E, K, N)}.</li>
            <li>A4W4 supports non-transposed and transposed NZ.</li>
            <li>The shape in NZ non-transposed format and int4 is similar to {(E, N / 64, K / 16, 16, 64)}.</li>
            <li>The shape in NZ non-transposed format and int32 is similar to {(E, N / 64, K / 16, 16, 8)}.</li>
            <li>The original shape in NZ transposed format and int4 is similar to {(E, K / 64, N / 16, 16, 64)}, and the input is transferred after transpose(-1,-2) is called.</li>
            <li>The original shape in NZ transposed format and int32 is similar to {(E, K / 64, N / 16, 16, 8)}, and the input is transferred after transpose(-1,-2) is called.</li>
            <li>When the input is in NZ transposed format, K/K_group_num of each group must be 64-aligned.</li></ul>
            </td>
            <td><ul>
            <li>The shape in per-channel scenario is similar to {(E, N)}.</li>
            <li>The shape in per-group scenario is similar to {(E, K_group_num, N)}.</li></ul>
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

    - In A8W8 scenarios, the size of the N axis must not exceed 10240, and the size of the last axis of `x` must be less than 65536.
    - In A8W4 scenarios, the size of the N axis must not exceed 10240, and the size of the last axis of `x` must be less than 20000.
    - In A4W4 scenarios, the size of the N axis must not exceed 10240, and the size of the last axis of `x` must be less than 20000.
    - In the multi-tensor scenario, that is, when the length of the tensor list is greater than 1, the shapes of weight, weightScale, and weightAssistMatrix need to be flattened according to the dimension of E. For example, {(E, K, N)} needs to be changed to {E (K, N)}.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:

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
        // (Boilerplate) Initialize resources.
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
        // Call aclrtMalloc to allocate memory on the device.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy the data on the host to the memory on the device.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Compute the strides of the contiguous tensor.
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
        // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
        // Set the device ID in use.
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = Init(deviceId, &stream);
        // Handle the check as required.
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        // 2. Construct the inputs and outputs based on the API definition.
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

        // Create an x aclTensor.
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a weight aclTensorList.
        ret = CreateAclTensorList(weightHostData, weightShape, weightDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a weightScale aclTensorList.
        ret = CreateAclTensorList(weightScaleHostData, weightScaleShape, weightScaleDeviceAddr, aclDataType::ACL_FLOAT,  aclFormat::ACL_FORMAT_ND, &weightScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create an xScale aclTensor.
        ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &xScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a groupList aclTensor.
        ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &groupList);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create an output aclTensor.
        ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_INT8, aclFormat::ACL_FORMAT_ND, &output);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create an outputScale aclTensor.
        ret = CreateAclTensor(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &outputScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Add V2 parameters.
        aclTensorList* weightAssistMatrix = nullptr;
        aclTensor* bias = nullptr;
        aclTensor* smoothScale = nullptr;
        int64_t dequantMode = 0;
        int64_t dequantDtype = 28;
        int64_t quantMode = 0;
        int64_t groupListType = 0;

        std::vector<int64_t> tuningConfigData = {};
        aclIntArray* tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;

        // 3. Call the CANN operator library API.
        // Call the first-phase API of aclnnGroupedMatmulSwigluQuantWeightNzV2.
        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
            x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode, dequantDtype,
            quantMode, groupListType, tuningConfig, output, outputScale, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, 
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on workspaceSize computed by the first-phase API.
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API of aclnnGroupedMatmulSwigluQuantWeightNzV2.
        ret = aclnnGroupedMatmulSwigluQuantWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, 
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2 failed. ERROR: %d\n", ret); return ret);

        // 4. (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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
        // 6. Destroy aclTensors, aclTensorLists, and aclScalars. Modify the code based on the API definition.
        aclDestroyTensor(x);
        aclDestroyTensorList(weight);
        aclDestroyTensorList(weightScale);
        aclDestroyTensor(xScale);
        aclDestroyTensor(groupList);
        aclDestroyTensor(output);
        aclDestroyTensor(outputScale);

        aclDestroyIntArray(tuningConfig);

        // 7. Release device resources. Modify the code based on the API definition.
        aclrtFree(xDeviceAddr);
        for (int64_t i = 0; i < 1; i++) {
            aclrtFree(weightDeviceAddr[i]);
            aclrtFree(weightScaleDeviceAddr[i]);
        }
        aclrtFree(weightDeviceAddr);
        aclrtFree(weightScaleDeviceAddr);
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
