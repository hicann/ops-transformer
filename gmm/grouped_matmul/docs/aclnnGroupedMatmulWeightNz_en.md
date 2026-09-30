# aclnnGroupedMatmulWeightNz

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Function

  - **Description**: Implements grouped matrix multiplication, supporting non-uniform matrix dimension sizes across multiple groups. The basic function is matrix multiplication, for example, $y_i[m_i,n_i]=x_i[m_i,k_i] times weight_i[k_i,n_i], i=1...g$, where $g$ indicates the number of groups and $m_i$, $k_i$, and $n_i$ define the shapes for each group. Both inputs and outputs are of the aclTensorList type, with the following functions:

      - K-axis grouping: $k_i$ varies across groups, while $m_i$ and $n_i$ remain the same for each group. In this case, $x_i$ and $weight_i$ can be concatenated along the K-axis.
      - M-axis grouping: $k_i$ remains the same for each group. In this case, $weight_i$ and $y_i$ can be concatenated along the N-axis.

    **New features compared with [GroupedMatmulV5](./aclnnGroupedMatmulV5_en.md)**:

      - The input `weight` data supports the AI processor-affinity layout format (FRACTAL_NZ).
      - The `quantGroupSize` parameter is added. It is an integer that indicates the quantization group size in per-group mode. If per-group quantization is not involved, set this parameter to `0`.

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
        y_i=(x_i \times per\_token\_scale_i) * (weight_i \times scale_i)
        $$

      <a id="dequantization-scenario"></a>

      - **Dequantization scenario:**

        $$
        y_i=(x_i \times weight_i + bias_i) * scale_i
        $$

      <a id="fake-quantization-scenario"></a>

      - **Fake-quantization scenario:**

        $$
        y_i=x_i \times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
        $$

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
    <td>The maximum length is 128.</td>
    <td>FLOAT16, BFLOAT16, INT8, INT4<sup>1</sup>, INT32<sup>1</sup>, FLOAT8_E4M3FN<sup>2</sup></td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weight</td>
    <td>Input</td>
    <td>Weight in the formula.</td>
    <td>The maximum length is 128. The Ascend-affinity data layout format (NZ) is supported.</td>
    <td>FLOAT16, BFLOAT16, INT8, INT4, INT32, FLOAT32, FLOAT4_E2M1<sup>2</sup></td>
    <td>ND, FRACTAL_NZ</td>
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
    <td>-1: no grouping; 0: M-axis grouping; 1: N-axis grouping; 2: K-axis grouping. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
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
    <td>The value ranges from 0 to 5. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
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
    <td>If per-group quantization is not involved, set this parameter to `0`.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>out</td>
    <td>Output</td>
    <td>Output y in the formula.</td>
    <td>The maximum length is 128.</td>
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
    - <term>Atlas inference products</term>:
        - Only FLOAT16 is supported. `weight` supports only the FRACTAL_NZ format, which needs to be converted through auxiliary APIs.
        - Quantization/asymmetric quantization parameters such as `scaleOptional` and `offsetOptional` are not supported, and need to be passed as null pointers.
        - `groupType` supports only M-axis grouping (`0`). The value of `actType` can only be `0`. `tuningConfigOptional` is not supported.
  - **Return**

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following errors may be thrown:

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
    <td rowspan="6"> ACLNN_ERR_PARAM_INVALID </td>
    <td rowspan="6"> 161002 </td>
    <td>1. The data type or data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, or out is not supported.</td>
    </tr>
    <tr>
    <td>2. The weight length is greater than 128; the bias is not null but its length is not equal to the weight length.</td>
    </tr>
    <tr>
    <td>3. The dimension of groupListOptional is 1.</td>
    </tr>
    <tr>
    <td>4. When splitItem is set to 2 or 3, the length of out is not 1.</td>
    </tr>
    <tr>
    <td>5. When splitItem is set to 0 or 1, the length of out or groupListOptional is not equal to the weight length.</td>
    </tr>
    <tr>
    <td>6. An element in the input parameter tuningConfigOptional is negative or exceeds the number of rows (M) in x.</td>
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

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatmulWeightNz` defaults to deterministic implementation.
- **Common constraints**
  - If `groupListOptional` is passed: when `groupListType` is `0`, `groupListOptional` must be a non-negative, monotonically non-decreasing sequence; when `groupListType` is `1`, `groupListOptional` must be a non-negative sequence and its length cannot be 1; when `groupListType` is `2`, the second column of `groupListOptional` must be a non-negative sequence and its length cannot be 1.
  - The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).

<details>
<summary><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></summary>

  - The following input types are supported in non-quantization scenarios:

    - `x`: FLOAT16; `weight`: FLOAT16; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: FLOAT16
    - `x`: BFLOAT16; `weight`: BFLOAT16; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: BFLOAT16

  - The following input types are supported in quantization scenarios:

    - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: BFLOAT16; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: BFLOAT16
    - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: FLOAT32; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: FLOAT16
    - `x`: INT4; `weight`: INT4; `biasOptional`: null; `scaleOptional`: UINT64; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: FLOAT16 or BFLOAT16

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
<summary><term>Atlas inference products</term></summary>

  - The input and output support only the FLOAT16 type. The N-axis size of the output `y` must be a multiple of 16.
    
    "S" stands for single-tensor and "M" stands for multi-tensor, expressed in the sequence of `x`, `weight`, `y`. For example, "SMS" indicates single-tensor `x`, multi-tensor `weight`, and single-tensor `y`.

    | groupType | Supported Scenario| Scenario Restrictions|
    |:---------:|:-------:| :------ |
    | 0 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensor in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.<br>(3) `groupListOptional` must be passed. When `groupListType` is `0`, the last value must be equal to the first dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be equal to the first dimension of the tensor in `x`.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) `weight` can be transposed, but `x` cannot.|

</details>

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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
  std::vector<T> hostData(size, 0);
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
  aclTensor* tensors[size];
  for (int i = 0; i < size; i++) {
    int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors + i);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *tensor = aclCreateTensorList(tensors, size);
  return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<std::vector<int64_t>> &shapes, void **deviceAddr,
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
int64_t E = shape[0];
int64_t K = shape[1];
int64_t N = shape[2];
std::vector<int64_t> shapeNz = {E, N/64, K/16, 16, 64};

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                            shapeNz.data(), shapeNz.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorListNz(const std::vector<std::vector> &hostData, const std::vector<std::vector<int64_t>> &shapes, void **deviceAddr,
                        aclDataType dataType, aclTensorList **tensor)
{
  int size = shapes.size();
  aclTensor * tensors[size];
  for (int i = 0; i < size; i++) {
    int ret = CreateAclTensorNz<T>(hostData[i], shapes[i], deviceAddr + i, dataType, tensors + i);
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
  std::vector<std::vector<int64_t>> xShape = {{512, 256}};
  std::vector<std::vector<int64_t>> weightShape= {{2, 256, 256}};
  std::vector<std::vector<int64_t>> yShape = {{512, 256}};
  std::vector<int64_t> groupListShape = {{2}};
  std::vector<int64_t> groupListData = {256, 512};
  void* xDeviceAddr[1];
  void* weightDeviceAddr[1];
  void* yDeviceAddr[1];
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
  std::vector<int8_t> wHostData(GetShapeSize(weightShape));
  // Create a tuningconfig aclIntArray.
  std::vector<int64_t> tuningConfigData = {512};
  aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

  // Create an x aclTensorList.
  ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensorList.
  ret = CreateAclTensorListNz(wHostData, weightShape, weightDeviceAddr, aclDataType::ACL_BF16, &weight);
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
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
    }
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensorList(x);
  aclDestroyTensorList(weight);
  aclDestroyTensorList(bias);
  aclDestroyTensorList(out);

  // 7. Release device resources. Modify the code based on the API definition.
  for (int i = 0; i < 1; i++) {
    aclrtFree(xDeviceAddr[i]);
    aclrtFree(weightDeviceAddr[i]);
    aclrtFree(biasDeviceAddr[i]);
    aclrtFree(yDeviceAddr[i]);
  }
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
