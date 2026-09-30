# aclnnGroupedMatmulSwigluQuantV2

[📄 View source code](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_swiglu_quant_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Fuses `GroupedMatmul`, `dequant`, `swiglu`, and `quant`. For details, see the formulas. Compared with [aclnnGroupedMatmulSwigluQuant](../../grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant_en.md), this API changes the field type of the `weight`, `weightScale`, and `weightAssistMatrix` parameters to tensor list. Select an appropriate API as required.
- Formula:
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
      - **Input**

        * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where `M` indicates the total number of tokens and `K` indicates the feature dimension.
        * $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where `E` indicates the number of experts, `K` indicates the feature dimension, and `N` indicates the output dimension.
        * $w\_scale∈\mathbb{R}^{E \times N}$: per-channel scale factor for the grouped weight matrix (right matrix), where `E` indicates the number of experts and `N` indicates the output dimension.
        * $x\_scale∈\mathbb{R}^{M}$: per-token scale factor for the activation matrix (left matrix), where `M` indicates the total number of tokens.
        * $grouplist∈\mathbb{N}^{E}$: grouped index list of cumsum or count.
      - **Output**

        * $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.

      - **Computation process**

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

            $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

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
      - **Input**
        * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where `M` indicates the total number of tokens and `K` indicates the feature dimension.
        * $W∈\mathbb{Z_4}^{E \times K \times N}$: grouped weight matrix (right matrix), where `E` indicates the number of experts, `K` indicates the feature dimension, and `N` indicates the output dimension.
        * $weightAsistMatrix∈\mathbb{R}^{E \times N}$: auxiliary matrix for matrix multiplication (the computation process for generating the auxiliary matrix is described below).
        * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$: per-channel scale factor for the grouped weight matrix (right matrix), where `E` indicates the number of experts, `K_group_num` indicates the number of groups along the K-axis, and `N` indicates the output dimension.
        * $x\_scale∈\mathbb{R}^{M}$: per-token scale factor for the activation matrix (left matrix), where `M` indicates the total number of tokens.
        * $grouplist∈\mathbb{N}^{E}$: grouped index list of cumsum or count.
      - **Output**
        * $Q∈\mathbb{Z_8}^{M \times N / 2}$: quantized output matrix.
        * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
      - **Computation process**
        1. Determine the tokens of the current group based on `groupList[i]`, where $i \in [0,Len(groupList)]$.
            - The grouping logic is the same as that of A8W8.
        2. Compute the auxiliary matrix (`weightAsistMatrix`). (Note that the computation is performed offline and provided as an input, rather than being executed within the operator.)
            - For per-channel quantization ($w\_scale$ is 2D):

              $weightAsistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$

            - For per-group quantization ($w\_scale$ is 3D):

              $weightAsistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$

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

              $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAsistMatrix_{i}) \odot x\_scale_{i}$

              $C_{i,act}, gate_{i} = split(C_{i})$

              $S_{i}=Swish(C_{i,act})\odot gate_{i}$ &nbsp;&nbsp; where $Swish(x)=\frac{x}{1+e^{-x}}$

        4. Quantize the output.

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnGroupedMatmulSwigluQuantV2` is called to perform computation.

```Cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
    const aclTensor     *x, 
    const aclTensorList *weight, 
    const aclTensorList *weightScale,
    const aclTensorList *weightAsistMatrix, 
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
        <th>Name</th>
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Usage Notes</th>
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
        <td><ul>
          <li>In the A8W8 scenario, K must be less than 65536.</li>
          <li>In the A8W4 scenario, K must be less than 20000.</li>
        </ul></td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E1M2, FLOAT4_E2M1, INT8</td>
        <td>ND</td>
        <td>2, for example, (M, K)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td rowspan="1">Input</td>
        <td>Weight matrix, corresponding to W in the formula.</td>
        <td><ul>
          <li>INT32 is used for adaptation in the A8W4 scenario. Actually, one INT32 data record is interpreted as eight INT4 data records.</li>
          <li>The ND data format is not supported in the A8W8 scenario.</li>
          <li>Currently, the tensor list length can only be 1.</li>
        </ul></td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E1M2, FLOAT4_E2M1, INT8, INT4, INT32</td>
        <td>ND, FRACTAL_NZ</td>
        <td>3, 5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightScale</td>
        <td rowspan="1">Input</td>
        <td>Quantization factor of the right matrix, wScale in the formula.</td>
        <td><ul>
          <li>The length of the first axis must be the same as the first axis of weight. The length of the last axis must be the same as the last axis of weight restored to the ND format.</li>
          <li>A8W4 scenario: The shape can be 2D or 3D, and the data type can be UINT64.</li>
          <li>A8W8 scenario: The shape can be 2D, and the data type can be FLOAT, FLOAT16, or BFLOAT16.</li>
          <li>Currently, the tensor list length can only be 1.</li>
        </ul></td>
        <td>FLOAT8_E8M0, UINT64, FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2, 3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightAssistMatrix</td>
        <td rowspan="1">Optional input</td>
        <td>Auxiliary matrix for matrix multiplication, weightAsistMatrix in the formula.</td>
        <td><ul>
          <li>This parameter is effective only in the A8W4 scenario. In other scenarios, pass a null pointer.</li>
          <li>The length of the first axis must be the same as the first axis of weight. The length of the last axis must be the same as the last axis of weight restored to the ND format.</li>
        </ul></td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
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
        <td>Quantization factor of the left matrix, xScale in the formula.</td>
        <td>FLOAT data type. The shape can be 1D, and the length must be the same as the first axis of x.</td>
        <td>FLOAT8_E8M0, FLOAT</td>
        <td>ND</td>
        <td>1, for example, (M,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>smoothScale</td>
        <td rowspan="1">Optional input</td>
        <td>Quantization factor of the left matrix.</td>
        <td>This input is reserved and is not supported currently. You need to pass a null pointer.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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
        <td>1, for example, (E,)</td>
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
          <li>0: DT_FLOAT</li>
          <li>1: FLOAT16</li>
          <li>27: BF16</li>
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
        <td>
          <li>0: per-token mode</li>
          <li>1: per-group mode</li>
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
        <td><li>0: cumsum mode, where each element in groupList represents the cumulative length of the current group. </li><li>1: count mode, where each element in groupList represents the number of elements contained in that group.</li></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>tuningConfig</td>
        <td rowspan="1">Optional input</td>
        <td>Used to estimate the M/E size and select an appropriate operator template, ensuring optimal performance across different scenarios.</td>
        <td>This input is reserved and is not supported currently. You need to pass a null pointer.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>output</td>
        <td rowspan="1">Output</td>
        <td>Quantization result, Q in the formula.</td>
        <td>-</td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E1M2, FLOAT4_E2M1, INT8</td>
        <td>ND</td>
        <td>2, for example, (M, N/2)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputScale</td>
        <td rowspan="1">Output</td>
        <td>Quantization factor, QScale in the formula.</td>
        <td>FLOAT data type. The shape supports 1D.
        </td>
        <td>FLOAT8_E8M0, FLOAT</td>
        <td>ND</td>
        <td>1, for example, (M,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td rowspan="1">Output</td>
        <td>Size of the workspace required to be allocated on the device.</td>
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

    - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
      - `X` supports only the INT8 quantization data type.
      - `weight` only supports the non-transposed mode and supports INT8, INT4, and INT32 data types. In ND format, the shape is {(E, K, N)}. In NZ format, the shape is {(E, N/32, K/16, 16, 32)} for the INT8 data type, {(E, N/64, K/16, 16, 64)} for INT4, and {(E, N/64, K/16, 16, 8)} for INT32.
      - `weightScale` supports FLOAT, FLOAT16, and BFLOAT16 data types in the A8W8 scenario, where the shape must be 2D, represented as {(E, N)}. In the A8W4 scenario, it supports the UINT64 data type, and the shape can be 2D or 3D (shape {(E, N)} for per-channel mode and shape {(E, KGroupCount, N)} for per-group mode).
      - The `dequantMode` parameter is supported. The value `0` indicates per-token mode for the activation matrix and per-channel mode for the weight matrix. The value `1` indicates per-token mode for the activation matrix and per-group mode for the weight matrix.
      - The `dequantDtype` parameter is not supported.
      - The `quantMode` parameter is not supported.
      - In the A8W8 or A8W4 scenario, the length of the N-axis cannot exceed 10240.
      - In the A8W8 scenario, the length of the last axis of `x` cannot be greater than or equal to 65536.
      - In the A8W4 scenario, the length of the last axis of `x` cannot be greater than or equal to 20000.
      - The data type of `output` must be INT8, and the shape can be 2D, for example, (M, N/2).
      - The data type of `outputScale` must be FLOAT, and the shape can be 1D, for example, (M,).
    - **Return**
  
  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following errors may be thrown:
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
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
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
      <td>In the A8W4 or A8W8 scenario, the value of the N-axis does not comply with the constraints.</td>
    </tr>
    <tr>
      <td>In the A8W4 or A8W8 scenario, the length of the last axis of x does not comply with the constraints.</td>
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
      <tr><td>workspace</td><td>Input</td><td>Address of the workspace to be allocated on the device.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Operator executor, containing the operator computation process.</td></tr>
      <tr><td>stream</td><td>Input</td><td>Stream for executing a task.</td></tr>
    </tbody>
  </table>

- **Return**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
    - In A8W8/A8W4 quantization scenarios, the following constraints must be met:
        - Data type requirements
        <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
        <col style="width: 319px">
        <col style="width: 144px">
        <col style="width: 671px">
        </colgroup>
        <thead>
          <tr>
            <th>Quantization Scenario</th>
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

      - In the A8W8 scenario, the length of the N-axis cannot exceed 10240, and the length of the last axis of `x` cannot be greater than or equal to 65536.
      - In the A8W4 scenario, the length of the N-axis cannot exceed 10240, and the length of the last axis of `x` cannot be greater than or equal to 20000.
      
  - Deterministic computation:
      - `aclnnGroupedMatmulSwigluQuantV2` defaults to a deterministic implementation.

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:

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
        // Call aclrtMalloc to allocate device memory.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy data from the host to the device memory.
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
        // Customize error handling based on your requirements.
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

        // 2. Construct inputs and outputs based on API definitions.
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
        // Call the first-phase API of aclnnGroupedMatmulSwigluQuantV2.
        ret = aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
            x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode, dequantDtype,
            quantMode, groupListType, tuningConfig, output, outputScale, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, 
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on workspaceSize computed by the first-phase API.
        void* workspaceAddr = nullptr;
        if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API of aclnnGroupedMatmulSwigluQuantV2.
        ret = aclnnGroupedMatmulSwigluQuantV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, 
        LOG_PRINT("aclnnGroupedMatmulSwigluQuantV2 failed. ERROR: %d\n", ret); return ret);

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
