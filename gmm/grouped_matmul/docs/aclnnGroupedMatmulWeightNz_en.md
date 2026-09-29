# aclnnGroupedMatmulWeightNz

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul)

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Function

- **Description**: Implements grouped matrix multiplication, supporting non-uniform matrix dimension sizes across multiple groups. The basic function is matrix multiplication, for example, $y_i[m_i,n_i]=x_i[m_i,k_i] times weight_i[k_i,n_i], i=1...g$, where $g$ indicates the number of groups and $m_i$, $k_i$, and $n_i$ define the shapes for each group. Both inputs and outputs are of the aclTensorList type, with the following functions:

  - K-axis grouping: $k_i$ varies across groups, while $m_i$ and $n_i$ remain the same for each group. In this case, $x_i$ and $weight_i$ can be concatenated along the K-axis.
  - M-axis grouping: $k_i$ remains the same for each group. In this case, $weight_i$ and $y_i$ can be concatenated along the N-axis.

  **New features compared with [GroupedMatmulV5](./aclnnGroupedMatmulV5.md)**:

  - The data format of the input weight supports the AI processor affinity data layout format (FRACTAL_NZ).
  - The `quantGroupSize` parameter is added. It is an integer that indicates the quantization group size in per-group mode. If per-group quantization is not involved, set this parameter to `0`.
  - Ascend 950PR/Ascend 950DT: The quantGroupSize parameter is not supported currently.

- **Formulas**:

  <a id="non-quantization-scenario"></a>

  - **Non-quantization scenario:**

    $$
    y_i=x_i \times weight_i + bias_i
    $$

  <a id="quantization-scenario"></a>

  - **Quantization scenario (without perTokenScaleOptional):**

    - `x` in INT8 and `bias` in INT32

      $$
      y_i=(x_i \times weight_i + bias_i) * scale_i + offset_i
      $$

    - `x` in INT8 and `bias` in BFLOAT16/FLOAT16/FLOAT32, without offset

      $$
      y_i=(x_i \times weight_i) * scale_i + bias_i
      $$

  - **Quantization scenario (with perTokenScaleOptional):**

    - `x` in INT8 and `bias` in INT32

      $$
      y_i=(x_i \times weight_i + bias_i) * scale_i * per\_token\_scale_i
      $$

    - `x` in INT8 and `bias` in BFLOAT16/FLOAT16/FLOAT32

      $$
      y_i=(x_i \times weight_i) * scale_i * per\_token\_scale_i  + bias_i
      $$

  - **Quantization scenario (MX quantization, without bias or activation layer):**

    $$
    y_i=(x_i * per\_token\_scale_i) \times (weight_i * scale_i)
    $$

  <a id="dequantization-scenario"></a>

  - **Dequantization scenario:**

    $$
    y_i=(x_i \times weight_i + bias_i) * scale_i
    $$

  <a id="fake-quantization-scenario"></a>

  - Fake-quantization (perchannel and pergroup) scenario:

    $$
    y_i=x_i \times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
    $$

  - Fake-quantization (mx) scenario:

    x is of type BFLOAT16/FLOAT16, and weight is of type FLOAT32 (indicating eight FLOAT4_E2M1 /FLOAT4_E2M1).

    $$
    y_i=x_i \times (weight_i  * antiquant\_scale_i) + bias_i
    $$

    x is of type FLOAT8_E4M3FN, and weight is of type FLOAT32 (indicating eight FLOAT4_E2M1 /FLOAT4_E2M1).

    $$
    y_i=(x_i * per\_token\_scale_i) \times (weight_i  * antiquant\_scale_i) + bias_i
    $$

  - Fake-quantization (K-CG) scenario:

    $$
    y_i=(x_i \times (weight_i * antiquant\_scale_i)) * scale_i * per\_token\_scale_i + bias_i
    $$

    In the preceding information, antiquant_scale_i is the per-group quantization parameter of the weight matrix, scale_i is the per-channel quantization parameter of the weight matrix, and per_token_scale_i is
    per-token quantization parameter.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulWeightNzGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnGroupedMatmulWeightNz` is called to perform computation.

```c++
aclnnStatus aclnnGroupedMatmulWeightNzGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *weight,
    const aclTensorList *biasOptional,
    const aclTensorList *scaleOptional,
    const aclTensorList *offsetOptional,
    const aclTensorList *antiquantScaleOptional,
    const aclTensorList *antiquantOffsetOptional,
    const aclTensorList *perTokenScaleOptional,
    const aclTensor     *groupListOptional,
    const aclTensorList *activationInputOptional,
    const aclTensorList *activationQuantScaleOptional,
    const aclTensorList *activationQuantOffsetOptional,
    int64_t              splitItem,
    int64_t              groupType,
    int64_t              groupListType,
    int64_t              actType,
    aclIntArray         *tuningConfigOptional,
    int64_t              quantGroupSize,
    aclTensorList       *out,
    aclTensorList       *activationFeatureOutOptional,
    aclTensorList       *dynQuantScaleOutOptional,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnGroupedMatmulWeightNz(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulWeightNzGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px;">
  <colgroup>
  <col style="width: 190px">
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
  <th>Input/Output</th>
  <th>Description</th>
  <th>Usage Notes</th>
  <th>Data Type</th>
  <th>Data Format</th>
  <th>Dimension (Shape)</th>
  <th>Non-contiguous Tensor</th>
  </tr>
  </thead>
  <tbody>
  <tr>
  <td>x</td>
  <td>Input</td>
  <td>Input x in the formula.</td>
  <td>The length of tensorList can be [1, 128] or [1, 1024].</td>
  <td>FLOAT16, BFLOAT16, INT8, INT4<sup>1</sup>, INT32<sup>1</sup>, FLOAT8_E4M3FN<sup>2</sup></td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>weight</td>
  <td>Input</td>
  <td>Weight in the formula.</td>
  <td>The length of tensorList can be [1, 128] or [1, 1024]. The Ascend-affinity data layout format (NZ) is supported.</td>
  <td>FLOAT16, BFLOAT16, INT8, INT4, INT32, FLOAT32, FLOAT4_E2M1<sup>2</sup></td>
  <td>FRACTAL_NZ</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>biasOptional</td>
  <td>Optional input</td>
  <td>Bias in the formula.</td>
  <td>Same length as weight.</td>
  <td>FLOAT16, FLOAT32, INT32, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>scaleOptional</td>
  <td>Optional input</td>
  <td>Scale in the formula, indicating the scale factor for quantization parameters.</td>
  <td>Generally, the length is the same as the weight length. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
  <td>UINT64<sup>1</sup>, BFLOAT16<sup>1</sup>, FLOAT32</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>offsetOptional</td>
  <td>Optional input</td>
  <td>Offset in the formula, indicating the offset for quantization parameters.</td>
  <td>Same length as weight.</td>
  <td>FLOAT32</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>antiquantScaleOptional</td>
  <td>Optional input</td>
  <td>antiquant_scale in the formula, indicating the scale factor for fake-quantization parameters.</td>
  <td>Same length as weight. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
  <td>FLOAT16, BFLOAT16<sup>1</sup>, FLOAT8_E8M0<sup>2</sup></td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>antiquantOffsetOptional</td>
  <td>Optional input</td>
  <td>antiquant_offset in the formula, indicating the offset for fake-quantization parameters.</td>
  <td>Same length as weight. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
  <td>FLOAT16, BFLOAT16<sup>1</sup></td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>perTokenScaleOptional</td>
  <td>Optional input</td>
  <td>per_token_scale in the formula, indicating the scale factor (introduced by x quantization) for quantization parameters.</td>
  <td>It is valid in scenarios where x, weight, and out are all single-tensor. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
  <td>FLOAT32, FLOAT8_E8M0<sup>2</sup></td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>groupListOptional</td>
  <td>Optional input</td>
  <td>Matmul size distribution along the grouping axis for inputs and outputs.</td>
  <td>Input data in different formats based on groupListType. Note that when the length of the output TensorList is 1, the last value of this parameter constrains the valid portion of the output.</td>
  <td>INT64</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>activationInputOptional</td>
  <td>Optional input</td>
  <td>Backward input of the activation function.</td>
  <td>Currently, only nullptr can be passed.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>activationQuantScaleOptional</td>
  <td>Optional input</td>
  <td>-</td>
  <td>Currently, only nullptr can be passed.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>activationQuantOffsetOptional</td>
  <td>Optional input</td>
  <td>-</td>
  <td>Currently, only nullptr can be passed.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>splitItem</td>
  <td>Input</td>
  <td>Specifies whether to perform tensor splitting on the output.</td>
  <td>0/1 indicates that the output is multi-tensor; 2/3 indicates that the output is single-tensor.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>groupType</td>
  <td>Input</td>
  <td>Axis to be grouped.</td>
  <td>The enumerated values are -1, 0, and 2. For example, if the matrix multiplication is C[m,n] = A[m,k] x B[k,n], the value of groupType is -1: no grouping, 0: grouping by the m axis, and 2: grouping by the k axis.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>groupListType</td>
  <td>Input</td>
  <td>Grouping mode of the groupList input.</td>
  <td>0: Cumulative sum result; 1: size of each group; 2: [groupIdx, groupSize]. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>actType</td>
  <td>Input</td>
  <td>Activation function type.</td>
  <td>The value ranges from 0 to 5.<br>
      0: GMM_ACT_TYPE_NONE;<br>
      1: GMM_ACT_TYPE_RELU;<br>
      2: GMM_ACT_TYPE_GELU_TANH;<br>
      3: GMM_ACT_TYPE_GELU_ERR_FUNC;<br>
      4: GMM_ACT_TYPE_FAST_GELU;<br>
      5: GMM_ACT_TYPE_SILU;<br>For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>tuningConfigOptional</td>
  <td>Optional input</td>
  <td>The first value indicates the expected number of tokens to be processed by each expert, which is used to optimize tiling.</td>
  <td>Compatible with earlier versions. If this parameter is not used, do not pass it (that is, pass nullptr).</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>quantGroupSize</td>
  <td>Input</td>
  <td>Group size for per-group quantization.</td>
  <td>If per-group quantization is not involved, set this parameter to `0`. The Ascend 950PR/Ascend 950DTis not supported.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>out</td>
  <td>Output</td>
  <td>Output y in the formula.</td>
  <td>The length of tensorList can be [1, 128] or [1, 1024].</td>
  <td>FLOAT16, BFLOAT16, INT8, FLOAT32, INT32</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>activationFeatureOutOptional</td>
  <td>Output</td>
  <td>Input data of the activation function.</td>
  <td>Currently, only nullptr can be passed.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>dynQuantScaleOutOptional</td>
  <td>Output</td>
  <td>-</td>
  <td>Currently, only nullptr can be passed.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>Output</td>
  <td>Size of the workspace required to be allocated on the device.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>Output</td>
  <td>Operator executor, containing the operator computation process.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  </tbody>
  </table>

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
      - The superscript "1" in the "Data Type" column of the table above indicates data types that are supported by the products, and the superscript "2" indicates data types that are not supported by the products.
      - The `weight` format can be converted from ND to NZ by calling `aclnnCalculateMatmulWeightSizeV2` and `aclnnTransMatmulWeight`. When the input data type is INT32, each INT32 value is treated as eight INT4 values inside the API.
      - The input parameters x and weight, and the output parameter out support a maximum of 128 tensors.
  - Ascend 950PR/Ascend 950DT:
      - In the preceding table, the superscript 2 in the data type column indicates the data types supported by the series.
      - x supports FLOAT16, BFLOAT16, FLOAT8_E4M3FN and INT8.
      - weight supports FLOAT16, BFLOAT16, FLOAT4_E2M1, INT8 and INT4. The FRACTAL_NZ format is supported. If either of the last two axes is 1 (that is, n = 1 or k = 1), the proprietary format is not supported and this API cannot be called. You can use the aclnnNpuFormatCast API to convert the input format from ND to the format (NZ) that is more affinity to the AI processor. If the original weight is in the transposed state and you want to use the non-transposed path with higher performance for computation, you can use the aclnnPermute API to convert the weight to the non-transposed state and then call the aclnnNpuFormatCast API. When the data type is FLOAT4_E2M1, you also need to call the aclnnCast API to convert the FLOAT4_E2M1 represented by FLOAT32 to the correct type after calling aclnnNpuFormatCast. However, when the data type is INT4, you need to use the aclnnConvertWeightToInt4Pack API to convert the data format from ND to NZ and the data type from INT32 to INT4. When FLOAT32 or INT32 is passed, each FLOAT32 or INT32 is identified as eight FLOAT4_E2M1/INT4 internally.
      - scaleOptional supports UINT64, INT64, BFLOAT16, and FLOAT32. offsetOptional and antiquantOffsetOptional are not supported.
      - groupType supports grouping by axis m. Only non-quantization supports ungrouping.
      - quantGroupSize is not supported.
      - actType supports 0, 1, 2, 4, and 5. For details about the constraints, see <a href="#constraints">Constraints</a>.
      - In the non-quantization scenario, the input parameters x and weight and the output parameter out support a maximum of 1024 tensors. In the fake-quantization and full-quantization scenarios, the input parameters x and weight and the output parameter out support a maximum of 128 tensors.

- **Return**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

   The first-phase API implements input parameter validation. The following error codes may be returned.

  <table>
  <thead>
  <tr>
  <th style="width: 250px">Return</th>
  <th style="width: 130px">Error Code</th>
  <th style="width: 850px">Description</th>
  </tr>
  </thead>
  <tbody>
  <tr>
  <td rowspan="4"> ACLNN_ERR_PARAM_NULLPTR </td>
  <td rowspan="4"> 161001 </td>
  <td>1. The required input, output, or attribute is passed as a null pointer.</td>
  </tr>
  <tr>
  <td>2. The input weight contains elements that are null pointers.</td>
  </tr>
  <tr>
  <td>3. The input x contains elements that are null pointers, while the corresponding elements in the output out are non-null pointers.</td>
  </tr>
  <tr>
  <td>4. The input x contains elements that are non-null pointers, while the corresponding elements in the output out are null pointers.</td>
  </tr>
  <tr>
  <td rowspan="7"> ACLNN_ERR_PARAM_INVALID </td>
  <td rowspan="7"> 161002 </td>
  <td>1. The data type or data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, or out is not supported.</td>
  </tr>
  <tr>
  <td>2. The length of weight is not supported.</td>
  </tr>
  <tr>
  <td>3. If bias is not null, the length of bias is not equal to that of weight.</td>
  </tr>
  <tr>
  <td>4. The groupListOptional dimension is 1.</td>
  </tr>
  <tr>
  <td>5. When splitItem is 2 or 3, the length of out is not 1.</td>
  </tr>
  <tr>
  <td>6. When splitItem is 0 or 1, the length of out is not equal to that of weight, and the length of groupListOptional is not equal to that of weight.</td>
  </tr>
  <tr>
  <td>7. The element of the input parameter tuningConfigOptional is a negative number or greater than the number of rows (m) in x.</td>
  </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulWeightNz

- **Parameters**

  |Parameter| Input/Output  |    Description|
  |-------|---------|----------------|
  |workspace|Input|Address of the workspace to be allocated on the device.|
  |workspaceSize|Input|Size of the workspace to be allocated on the device, which is obtained by calling `aclnnGroupedMatmulWeightNzGetWorkspaceSize`.|
  |executor|Input|Operator executor, containing the operator computation process.|
  |stream|Input|Stream for executing the task.|

- **Return**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatmulWeightNz` defaults to deterministic implementation.

<details>
<summary><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></summary>

  - Common constraints
    - If `groupListOptional` is passed: when `groupListType` is `0`, `groupListOptional` must be a non-negative, monotonically non-decreasing sequence; when `groupListType` is `1`, `groupListOptional` must be a non-negative sequence and its length cannot be 1; when `groupListType` is `2`, the second column of `groupListOptional` must be a non-negative sequence and its length cannot be 1.
    - The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).
    - actType (int64_t, input for computation): integer parameter, indicating the activation function type. The value ranges from 0 to 5.

  - The following input types are supported in non-quantization scenarios:

    - `x`: FLOAT16; `weight`: FLOAT16; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: FLOAT16
    - `x`: BFLOAT16; `weight`: BFLOAT16; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: BFLOAT16

  - The following input types are supported in quantization scenarios:

    - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: BFLOAT16; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: BFLOAT16
    - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: FLOAT32; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: FLOAT16
    - `x`: INT4; `weight`: INT4; `biasOptional`: null; `scaleOptional`: UINT64; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: FLOAT16 or BFLOAT16 The weight supports the transposed input in the NZ format, that is, the input is in the [E, N, K] format, but the view shape is in the [E, K, N] format to ensure that the operator can identify the transposed status. When the input is transposed, $k/G$ must be 64-byte aligned, K must be 64-byte aligned, and N must be 16-byte aligned. The ND format does not support the transposed input.

  - The following input types are supported in fake-quantization scenarios:

    - The shapes of the fake-quantization parameters `antiquantScaleOptional` and `antiquantOffsetOptional` must meet the following requirements ($g$ indicates the number of Matmul groups, $G$ indicates the number of quantization groups, and $G_i$ indicates the number of quantization groups of the *i*-th tensor).

        | Application Scenario| Sub-scenario| Shape Restriction|
        |:---------:|:-------:| :-------|
        | Fake-quantization (per-channel)| Single-tensor `weight`| $[E, N]$|
        | Fake-quantization (per-channel)| Multi-tensor `weight`| $[n_i]$|
        | Fake-quantization (per-group)| Single-tensor `weight`| $[E, G, N]$|
        | Fake-quantization (per-group)| Multi-tensor `weight`| $[G_i, N_i]$|

    - `x`: INT8; `weight`: INT4; `biasOptional`: FLOAT32; `scaleOptional`: UINT64; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: FLOAT32; `activationInputOptional`: null This scenario supports symmetric quantization and asymmetric quantization:

      - Symmetric quantization

        - The data type of the output `out` is BFLOAT16 or FLOAT16.
        - `offsetOptional` is null.
        - Only the count mode is supported (the operator does not validate `groupListType`). `k` must be an integer multiple of `quantGroupSize` and `k` ≤ 18432. `quantGroupSize` is the per-group quantization length in the K-axis. Currently, `quantGroupSize=256` is supported.
        - The scale is the result after per-group and per-channel offline fusion. The shape must be $[E, quantGroupNum, N]$, where $quantGroupNum=k \div quantGroupSize$.
        - The bias is the auxiliary result of offline computation during the computation process. Its value must be $8\times weight \times scale$ and is accumulated in the first dimension. The shape must be $[E, N]$.
        - N must be an integer multiple of 8.

      - Asymmetric quantization

        - The data type of the output `out` is FLOAT16.
        - Only the count mode is supported (the operator does not validate `groupListType`).
        - {k, n} must be {7168, 4096} or {2048, 7168}.
        - The scale is the result after per-group and per-channel offline fusion. The shape must be $[E, 1, N]$.
        - `offsetOptional` is not null. For asymmetric quantization, `offsetOptional` is the auxiliary result of offline computation during the computation process, that is, $antiquantOffset \times scale$. The shape must be $[E, 1, N]$, and the data type must be FLOAT32.
        - The bias is the auxiliary result of offline computation during the computation process. Its value must be $8\times weight \times scale$ and is accumulated in the first dimension. The shape must be $[E, N]$.
        - N must be an integer multiple of 8.

    - In fake-quantization scenarios, if the `weight` type is INT8, only the per-channel mode is supported. If the `weight` type is INT4, the per-channel and per-group modes are supported in symmetric quantization. If the per-group mode is used, the number of quantization groups ($G$ or $G_i$) must be exactly divisible by the corresponding $k_i$. For multi-tensor `weight`, the per-group length is defined as $s_i = k_i / G_i$, and all $s_i(i=1,2,...g)$ values must be the same. Asymmetric quantization supports the per-channel mode.

    - In fake-quantization scenarios, if the `weight` type is INT4, the last dimension of each group of tensors in weight must be an even number. The last dimension of $weight_i$ refers to the N-axis when `weight` is not transposed or the K-axis when `weight` is transposed. In the per-group mode, when `weight` is transposed, the per-group length $s_i$ must be an even number.

  - Supported scenarios for different `groupType` values:
    - In quantization and fake-quantization scenarios, `groupType` can be either `-1` or `0`.
    - "S" stands for single-tensor, and "M" stands for multi-tensor, expressed in the sequence of `x`, `weight`, `y`. For example, "SMS" indicates single-tensor `x`, multi-tensor `weight`, and single-tensor `y`.

      | groupType | Supported Scenario| Scenario Restrictions|
      |:---------:|:---------:| :-------|
      | -1 | MMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) The tensors in `x` must have the same dimensionality, which can be 2D to 6D. The tensors in `weight` must be 2D. The dimensionality of tensors in `y` must match that of `x`.<br>(3) `groupListOptional` must be passed as null.<br>(4) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(5) `x` cannot be transposed.|
      | 0 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensor in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.<br>(3) `groupListOptional` must be passed. When `groupListType` is `0`, the last value must be equal to the first dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be equal to the first dimension of the tensor in `x`. When `groupListType` is `2`, the sum of the values in the second column must be equal to the first dimension of the tensor in `x`.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) `weight` can be transposed.<br>(6) `x` cannot be transposed.|
      | 0 | SMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) `groupListOptional` must be passed. When `groupListType` is `0`, the last value must be equal to the first dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be equal to the first dimension of the tensor in `x` and the maximum length is 128. When `groupListType` is `2`, the sum of the values in the second column must be equal to the first dimension of the tensor in `x` and the maximum length is 128.<br>(3) Tensors in `x`, `weight`, and `y` must be 2D.<br>(4) The N-axis of each tensor in `weight` must be the same.<br>(5) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(6) `x` cannot be transposed.|
      | 0 | MMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) Tensors in `x`, `weight`, and `y` must be 2D.<br>(3) The N-axis of each tensor in `weight` must be the same.<br>(4) If `groupListOptional` is passed: when `groupListType` is `0`, the differences of `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`; when `groupListType` is `1`, the `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`, and the maximum length is 128; when `groupListType` is `2`, the values in the second column of `groupListOptional` must be in one-to-one mapping with the first dimension of the tensors in `x`, and the maximum length is 128.<br>(5) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(6) `x` cannot be transposed.|

</details>

<details>
<summary>Ascend 950PR/Ascend 950DT</summary>

  - Common constraints
    - `groupListType`: The value can be 0 or 1. When groupListType is set to 0, groupListOptional must be a non-negative monotonic non-decreasing number. When groupListType is set to 1, groupListOptional must be a non-negative number.
    - The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).
    - actType (int64_t, input for computation): integer parameter, indicating the activation function type. The value ranges from 0 to 5.
      - In fake-quantization and non-quantization scenarios, actType can only be set to 0.
      - In full quantization scenarios, when x and weight are of the INT8 type, the quantization mode is static T-C quantization or dynamic K-C quantization, and the scale data type is FLOAT32 or BFLOAT16, actType can be set to 0, 1, 2, 4, or 5. In other full quantization scenarios, actType can only be set to 0.
  - Currently, the non-quantization, fake-quantization, and full quantization scenarios are supported.
  - The following data types are supported in the non-quantization scenario:
    - The n and k axes of the input weight matrix must be 32-byte aligned.
    - The following input parameters are empty: scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional and activationFeatureOutOptional.
    - The combinations of data types supported by non-empty parameters must meet the requirements listed in the following table.

      |groupType| x       | weight  | biasOptional | out     |
      |:-------:|:-------:|:-------:| :------      |:------ |
      |-1/0   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
      |-1/0   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|

  - The data types supported in the fake-quantization scenario are as follows:
    - The following input parameters are empty: offsetOptional, antiquantOffsetOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional and activationFeatureOutOptional.
    - The combinations of data types supported by non-empty parameters must meet the requirements listed in the following table.

      |groupType| x       |perTokenScaleOptional| weight  |antiquantScaleOptional|scaleOptional|antiquantOffsetOptional| biasOptional | out     | perTokenScaleOptional Shape | weight Shape | antiquantScaleOptional Shape| scaleOptional shape|bias shape|
      |:-------:|:-------:|:-------------------:|:-------:|:--------------------:|:-----------:|:---------------------:|:------------:|:-------:|:---------------------------:|:------------:|:---------------------------:|:------------------:|:--------:|
      |0   |BFLOAT16      |null          |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null |BFLOAT16/FLOAT32/null     |BFLOAT16 |null             |(g, K, N)   |(g, K/groupSize, N) |null   |(g, N) |
      |0   |FLOAT16       |null          |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null |FLOAT16/null              |FLOAT16  |null             |(g, K, N)   |(g, K/groupSize, N) |null   |(g, N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null |FLOAT16/null              |FLOAT16  |(M, K/groupSize/2, 2) |(g, N, K)   |(g, N, K/groupSize/2, 2) |null   |(g, N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null |BFLOAT16/null             |BFLOAT16 |(M, K/groupSize/2, 2) |(g, N, K)   |(g, N, K/groupSize/2, 2) |null   |(g, N) |
      |0   |INT8          |FLOAT32       |INT4            |FLOAT16     |FLOAT32 |null |FLOAT32/null              |BFLOAT16 |(M)              |(g, K, N)   |(g, K/groupSize, N) |(g, N) |(g, N) |
      |0   |INT8          |FLOAT32       |INT4            |FLOAT16     |FLOAT32 |null |FLOAT32/null              |FLOAT16  |(M)              |(g, K, N)   |(g, K/groupSize, N) |(g, N) |(g, N) |
      |0   |BFLOAT16      |null          |FLOAT32         |FLOAT8_E8M0 |null    |null |BFLOAT16/FLOAT32/null     |BFLOAT16 |null             |(g, K, N/8) |(g, K/groupSize, N) |null   |(g, N) |
      |0   |FLOAT16       |null          |FLOAT32         |FLOAT8_E8M0 |null    |null |FLOAT16/null              |FLOAT16  |null             |(g, K, N/8) |(g, K/groupSize, N) |null   |(g, N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT32         |FLOAT8_E8M0 |null    |null |FLOAT16/null              |FLOAT16  |(M, K/groupSize/2, 2) |(g, N, K/8) |(g, N, K/groupSize/2, 2) |null   |(g, N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT32         |FLOAT8_E8M0 |null    |null |BFLOAT16/null             |BFLOAT16 |(M, K/groupSize/2, 2) |(g, N, K/8) |(g, N, K/groupSize/2, 2) |null   |(g, N) |
      |0   |INT8          |FLOAT32       |INT32           |FLOAT16     |FLOAT32 |null |FLOAT32/null              |BFLOAT16 |(M)              |(g, K, N/8) |(g, K/groupSize, N) |(g, N) |(g, N) |
      |0   |INT8          |FLOAT32       |INT32           |FLOAT16     |FLOAT32 |null |FLOAT32/null              |FLOAT16  |(M)              |(g, K, N/8) |(g, K/groupSize, N) |(g, N) |(g, N) |
      
    - Constraints:
      - When x is of type FLOAT8_E4M3FN/FLOAT16/BFLOAT16 and weight is of type FLOAT4_E2M1/FLOAT32, groupSize can only be 32.
      - When x is of type INT8 and weight is of type INT4/INT32, groupSize can only be 128, 192, 256, or 512.
      - The shape of x is fixed to (M, K), and the shape of out is fixed to (M, N).
      - When the types of x and weight are BFLOAT16/FLOAT16 and FLOAT4_E2M1/FLOAT32, or INT8 and INT4/INT32, respectively, only the case where neither x nor weight is transposed is supported. When the types of x and weight are FLOAT8_E4M3FN and FLOAT4_E2M1/FLOAT32, respectively, only the case where x is not transposed and weight is transposed is supported.
      - The transposition of antiquantScale is the same as that of weight.

  - The supported input types in static quantization scenarios are as follows:
    - The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional and activationFeatureOutOptional.
    - The following table lists the data type combinations supported by the parameters that are not empty.

      |groupType| x       | weight  | biasOptional | scaleOptional | out     |
      |:-------:|:-------:|:-------:| :------      |:-------       | :------ |
      |0|INT8     |INT8     |INT32/null    | UINT64/INT64  |BFLOAT16/FLOAT16|
      |0|INT8     |INT8     |INT32/BFLOAT16/FLOAT32/null    | BFLOAT16/FLOAT32  |BFLOAT16|
      |0|INT8     |INT8     |INT32/FLOAT16/FLOAT32/null    | FLOAT32  |FLOAT16|

    - The scaleOptional must meet the requirements listed in the following table (g indicates the number of matmul groups, that is, the number of groups).

      |groupType| Application Scenario| Shape Restriction|
      |:---------:|:---------:| :------ |
      |0|Single-tensor weight|In the perchannel scenario, each tensor is two-dimensional, and the shape is (g, N). In the pertensor scenario, each tensor is two-dimensional or one-dimensional, and the shape is (g, 1) or (g,).|

  - The supported input types in the dynamic quantization (K-T && K-C quantization) scenario are as follows:
    - The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional and activationFeatureOutOptional.
    - The combinations of data types supported by non-empty parameters must meet the requirements listed in the following table.

      |groupType| x       | weight  | biasOptional | scaleOptional | perTokenScaleOptional |out     |
      |:-------:|:-------:|:-------:| :------      |:-------    | :------   | :------ |
      |0|INT8  |INT8| INT32/BFLOAT16/FLOAT32/null     |BFLOAT16/FLOAT32    | FLOAT32   | BFLOAT16 |
      |0|INT8  |INT8| INT32/FLOAT16/FLOAT32/null     |FLOAT32    | FLOAT32   | FLOAT16 |

    - The scaleOptional must meet the requirements listed in the following table (g indicates the number of matmul groups, that is, the number of groups).

      | groupType | Application Scenario| Shape Restriction|
      |:---------:|:---------:| :------ |
      |0|Single-tensor weight|In the perchannel scenario, each tensor is two-dimensional, and the shape is (g, N). In the pertensor scenario, each tensor is two-dimensional or one-dimensional, and the shape is (g, 1) or (g,).|

    - perTokenScaleOptional must meet the following requirements:

      | groupType | Application Scenario| Shape Restriction|
      |:---------:|:---------:| :------ |
      |0|Single-tensor x|In the pertoken scenario, each tensor is one-dimensional, and the shape is (M,).|

  - Supported scenarios for different `groupType` values:

    - In the supported scenarios, single indicates a single tensor, and multiple indicates multiple tensors. The sequence is x, weight, out. For example, single-multiple-single indicates that x is a single tensor, weight is multiple tensors, and out is a single tensor.

      | groupType | Supported Scenario| Scenario Restrictions|
      |:---------:|:-------:| :------ |
      | -1 | MMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) The tensors in x and out must be 2D, with shapes of ($m_i$, $k_i$) and ($m_i$, $n_i$), respectively. The tensor in weight must be 2D, with shape of ($n_i$, $k_i$) or ($k_i$, $n_i$). The tensor in bias must be 1D, with shape of ($n_i$).<br>(3) `groupListOptional` must be passed as null.<br>(4) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(5) `x` cannot be transposed.<br>(6) Only non-quantization is supported.|
      | 0 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensor in weight must be 3D, with shape of (E, N, K) or (E, K, N). The tensors in x and out must be 2D, with shapes of (M, K) and (M, N), respectively. The tensor in bias must be 2D, with shape of (E, N).<br>(3) groupListOptional must be passed. When groupListType is 0, the last value cannot be greater than the first dimension of the tensor in x. When groupListType is 1, the sum of the values cannot be greater than the first dimension of the tensor in x.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) x can be not transposed, and weight can be transposed or not transposed.|
      | 0 | SMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) groupListOptional must be passed. When groupListType is 0, the last value is equal to the first dimension of the tensor in x. When groupListType is 1, the sum of the values is equal to the first dimension of the tensor in x. The maximum length is 1024.<br>(3) The tensors in x and out must be 2D, with shapes of (M, K) and (M, N), respectively. The tensor in weight must be 2D, with shape of (N, K) or (K, N). The tensor in bias must be 1D, with shape of (N).<br>(4) The N-axis of each tensor in `weight` must be the same.<br>(5) Weight transposition is supported, but the transposition status of each tensor in the weight tensorList must be the same.<br>(6) `x` cannot be transposed.<br>(7) Only non-quantization is supported.|
      | 0 | MMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensors in x and out must be two-dimensional, with shapes (M, K) and (M, N), respectively. The tensor in weight must be two-dimensional, with shape (N, K) or (K, N). The tensor in bias must be one-dimensional, with shape (N).<br>(3) The N-axis of each tensor in `weight` must be the same.<br>(4) If groupListOptional is passed, when groupListType is 0, the difference between groupListOptional and the first dimension of the tensor in x must be in one-to-one correspondence. When groupListType is 1, the value of groupListOptional must be in one-to-one correspondence with the first dimension of the tensor in x, and the maximum length is 1024.<br>(5) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(6) `x` cannot be transposed.<br>(7) Only non-quantization is supported.|

</details>

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

Fake-quantization calling example

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
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
int CreateAclTensor_New(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  std::vector<T> hostData(size / sizeof(T), 0);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
  int size = shapes.size();
  std::vector<aclTensor*> tensors(size);
  for (int i = 0; i < size; i++) {
    int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, &tensors[i]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *tensor = aclCreateTensorList(tensors.data(), size);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);

  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
  }

  //Check the shape dimension.
  if (shape.size() != 3) {
    LOG_PRINT("Shape must be 3D for NZ format\n");
    return -1;
  }

  int64_t E = shape[0];
  int64_t K = shape[1];
  int64_t N = shape[2];

  //Check whether the dimension can be exactly divided.
  if (N % 64 != 0 || K % 16 != 0) {
    LOG_PRINT("N must be divisible by 64 and K by 16 for NZ format\n");
    return -1;
  }

  std::vector<int64_t> shapeNz = {E, N/64, K/16, 16, 64};

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                            shapeNz.data(), shapeNz.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorListNz(const std::vector<std::vector<T>> &hostData,
                          const std::vector<std::vector<int64_t>> &shapes,
                          void **deviceAddr,
                          aclDataType dataType,
                          aclTensorList **tensor)
{
  if (hostData.size() != shapes.size()) {
    LOG_PRINT("hostData size %ld does not match shapes size %ld\n", hostData.size(), shapes.size());
    return -1;
  }

  int size = shapes.size();
  std::vector<aclTensor*> tensors(size);
  for (int i = 0; i < size; i++) {
    int ret = CreateAclTensorNz<T>(hostData[i], shapes[i], deviceAddr + i, dataType, &tensors[i]);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *tensor = aclCreateTensorList(tensors.data(), size);
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
  std::vector<std::vector<int64_t>> xShape = {{512, 256}};
  std::vector<std::vector<int64_t>> weightShape = {{2, 256, 256}};
  std::vector<std::vector<int64_t>> yShape = {{512, 256}};
  std::vector<int64_t> groupListShape = {2};
  std::vector<int64_t> groupListData = {256, 512};

  void* xDeviceAddr[1];
  void* weightDeviceAddr[1];
  void* yDeviceAddr[1];
  void* biasDeviceAddr[1] = {nullptr}; // Declare biasDeviceAddr.
  void* groupListDeviceAddr;

  aclTensorList* x = nullptr;
  aclTensorList* weight = nullptr;
  aclTensorList* bias = nullptr;
  aclTensor* groupedList = nullptr;
  aclTensorList* scale = nullptr;
  aclTensorList* offset = nullptr;
  aclTensorList* antiquantScale = nullptr;
  aclTensorList* antiquantOffset = nullptr;
  aclTensorList* perTokenScale = nullptr;
  aclTensorList* activationInput = nullptr;
  aclTensorList* activationQuantScale = nullptr;
  aclTensorList* activationQuantOffset = nullptr;
  aclTensorList* out = nullptr;
  aclTensorList* activationFeatureOut = nullptr;
  aclTensorList* dynQuantScaleOut = nullptr;

  int64_t splitItem = 3;
  int64_t groupType = 0;
  int64_t groupListType = 0;
  int64_t actType = 0;

  // Create weight data.
  int64_t weightTotalSize = 1;
  for (const auto& dim : weightShape[0]) {
    weightTotalSize *= dim;
  }
  std::vector<std::vector<int8_t>> wHostDataList(1);
  wHostDataList[0].resize(weightTotalSize * sizeof(uint16_t)); // BF16 requires 2 bytes.

  // Create a tuningconfig aclIntArray.
  std::vector<int64_t> tuningConfigData = {512};
  aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

  // Create an x aclTensorList.
  ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create weight aclTensorList in NZ format.
  ret = CreateAclTensorListNz<int8_t>(wHostDataList, weightShape, weightDeviceAddr, aclDataType::ACL_BF16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a y aclTensorList.
  ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_BF16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a group_list aclTensor.
  ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. Call the CANN operator library API.
  // Call the first-phase API of aclnnGroupedMatmulWeightNz.
  ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput, activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, tuningConfig, 0, out, activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnGroupedMatmulWeightNz.
  ret = aclnnGroupedMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  for (int i = 0; i < 1; i++) {
    auto size = GetShapeSize(yShape[i]);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < 20; j++) {
      LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
    }
    LOG_PRINT("......\n");
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensorList(x);
  aclDestroyTensorList(weight);
  if (bias) aclDestroyTensorList(bias);
  aclDestroyTensorList(out);
  if (groupedList) aclDestroyTensor(groupedList);

  // 7. Release device resources. Modify the code based on the API definition.
  for (int i = 0; i < 1; i++) {
    if (xDeviceAddr[i]) aclrtFree(xDeviceAddr[i]);
    if (weightDeviceAddr[i]) aclrtFree(weightDeviceAddr[i]);
    if (biasDeviceAddr[i]) aclrtFree(biasDeviceAddr[i]);
    if (yDeviceAddr[i]) aclrtFree(yDeviceAddr[i]);
  }
  if (groupListDeviceAddr) aclrtFree(groupListDeviceAddr);
  if (workspaceSize > 0 && workspaceAddr) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

Full Quantization Invoking Example

```c++
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"
#include "aclnnop/aclnn_npu_format_cast.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1L;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // (Fixed writing) Initialize AscendCL.
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
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
template <typename T>
int CreateAclTensorList(const std::vector<T> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}
template <typename T>
int CreateAclTensorWithFormat(const std::vector<T> &hostData, const std::vector<int64_t> &shape, int64_t **storageShape,
                              uint64_t *storageShapeSize, void **deviceAddr, aclDataType dataType, aclTensor **tensor,
                              aclFormat format)
{
    auto size = hostData.size() * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, *storageShape,
                              *storageShapeSize, *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor, aclrtStream &stream)
{
    void *srcDeviceAddr = nullptr;
    aclTensor *srcTensor = nullptr;
    auto size = hostData.size() * sizeof(T);

    auto ret = aclrtMalloc(&srcDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(srcDeviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    srcTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), srcDeviceAddr);

    int64_t *dstShape = nullptr;
    uint64_t dstShapeSize = 0;
    int actualFormat;
    ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, 29, aclFormat::ACL_FORMAT_ND, &dstShape, &dstShapeSize,
                                                   &actualFormat);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
              return ret);

    aclTensor *dstTensor = nullptr;
    void *dstDeviceAddr = nullptr;

    uint64_t tensorSize = 1;
    for (int64_t i = 0; i < dstShape[i]; i++) {
        tensorSize *= dstShape[i];
    }
    ret = aclrtMalloc(&dstDeviceAddr, tensorSize * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    int64_t weightLen = shape.size();
    for (int64_t i = 0; i < weightLen + 2; i++) {
        tensorSize = tensorSize * dstShape[i];
    }
    std::vector<uint16_t> dstTensorHostData(tensorSize, 0);

    ret = CreateAclTensorWithFormat(dstTensorHostData, shape, &dstShape, &dstShapeSize, &dstDeviceAddr, dataType,
                                    &dstTensor, static_cast<aclFormat>(actualFormat));
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);

    // Call the first-phase API of aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    *tensor = dstTensor;
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorListNz(const std::vector<T> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                          void **deviceAddr, aclDataType dataType, aclTensorList **tensor, aclrtStream &stream)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; ++i) {
        int ret = CreateAclTensorNz<T>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i, stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnGourpedMatmulTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on API definitions.
    int64_t m = 512L;
    int64_t k = 256L;
    int64_t n = 4L;
    int64_t groupnum = 2L;
    std::vector<std::vector<int64_t>> xShape = {{m, k}};
    std::vector<std::vector<int64_t>> weightShape = {{groupnum, k, n}};
    std::vector<std::vector<int64_t>> biasShape = {{groupnum, n}};
    std::vector<std::vector<int64_t>> scaleShape = {{groupnum, n}};
    std::vector<std::vector<int64_t>> pertokenShape = {{
        m,
    }};
    std::vector<std::vector<int64_t>> yShape = {{m, n}};
    std::vector<int64_t> groupListShape = {{groupnum}};
    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *pertokenDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    aclTensorList *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *bias = nullptr;
    aclTensor *groupedList = nullptr;
    aclTensorList *scale = nullptr;
    aclTensorList *offset = nullptr;
    aclTensorList *antiquantScale = nullptr;
    aclTensorList *antiquantOffset = nullptr;
    aclTensorList *perTokenScale = nullptr;
    aclTensorList *activationInput = nullptr;
    aclTensorList *activationQuantScale = nullptr;
    aclTensorList *activationQuantOffset = nullptr;
    aclTensorList *out = nullptr;
    aclTensorList *activationFeatureOut = nullptr;
    aclTensorList *dynQuantScaleOut = nullptr;
    int64_t splitItem = 3L;
    int64_t groupType = 0L;
    int64_t groupListType = 0L;
    int64_t actType = 0L;
    std::vector<int8_t> xHostData(m * k, 10);
    std::vector<int8_t> weightHostData(groupnum * k * n, 10);
    std::vector<uint16_t> yHostData(m * n, 0);
    std::vector<int64_t> groupListData = {256, 512};
    std::vector<int8_t> scaleHostData(groupnum * n, 1);
    std::vector<int8_t> biasHostData(groupnum * n, 1);
    std::vector<int8_t> pertokenHostData(m, 1);

    // Create an x aclTensorList.
    ret = CreateAclTensorList(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> xTensorPtr(x, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    // Create a weight aclTensorList.
    ret = CreateAclTensorListNz(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorPtr(weight,
                                                                                           aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    // Create a scale aclTensorList.
    ret = CreateAclTensorList(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_BF16, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> scaleTensorPtr(scale, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    // Create a per-token aclTensorList.
    ret = CreateAclTensorList(pertokenHostData, pertokenShape, &pertokenDeviceAddr, aclDataType::ACL_FLOAT,
                              &perTokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> pertokenTensorPtr(perTokenScale,
                                                                                             aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> pertokenDeviceAddrPtr(pertokenDeviceAddr, aclrtFree);
    // Create a y aclTensorList.
    ret = CreateAclTensorList(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> yTensorPtr(out, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);

    // Create group_list aclTensorList.
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupedList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);

    // 3. Call the CANN operator library API.
    ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, nullptr, groupedList, activationInput,
        activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, nullptr, 0, out,
        activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnGroupedMatmulWeightNz.
    ret = aclnnGroupedMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    for (int i = 0; i < 1; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
        }
    }
    return ACL_SUCCESS;
}
int main()
{
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGourpedMatmulTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz test failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
