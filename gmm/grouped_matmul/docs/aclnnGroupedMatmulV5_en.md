# aclnnGroupedMatmulV5

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Implements grouped matrix multiplication. For example, $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where $g$ indicates the number of groups. Currently, M-axis grouping and K-axis grouping are supported. The corresponding functions are as follows:

  - M-axis grouping: $k_i$ and $n_i$ remain consistent for each group, while $m_i$ can vary.
  - K-axis grouping: $m_i$ and $n_i$ remain consistent for each group, while $k_i$ can vary.

- The basic computation formula is as follows (for details, see [Formulas](#formulas)):

  $$
  y_i=x_i\times weight_i + bias_i
  $$

- Version evolution:

  |Version Change     | Atlas A2 training products/Atlas A2 inference products<br>Atlas A3 training products/Atlas A3 inference products|Atlas inference products|
  |---------|---------|----------------|
  |V4 -> V5|  Added the optional parameter `tuningConfigOptional` for tuning. The first value in the array indicates the expected number of tokens to be processed by each expert. Optimal tiling is performed based on this expected value during operator tiling.  |  /  | / |
  |V1 -> V4|     Supports axis grouping, represented by `groupType`.<br>Supports the transposition of `x` and `weight` in non-quantization scenarios. Transposition refers to the case where the shape is [M, K], the stride is [1, M], and the data layout is [K, M].<br>Supports weight transposition and single-tensor weights in quantization and fake-quantization scenarios.<br>Supports FLOAT32 input for `x` and `weight` when `x`, `weight`, and `y` are all single-tensor in non-quantization scenarios.<br>Static quantization (per-tensor and per-channel), BFLOAT16 and FLOAT16 outputs, with or without activation (For details, refer to [quantization methods](../../../docs/en/context/quant_mode_introduction.md). Same below.)<br>Dynamic quantization (per-tensor and per-channel), BFLOAT16 and FLOAT16 outputs, with or without activation<br>Fake-quantization with INT4 input `weight` without activation in per-channel and per-group modes    |  / |

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulV5GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor containing the operator computation process. Then, `aclnnGroupedMatmulV5` is called to perform computation.

```c++
aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(
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
    aclTensorList       *out,
    aclTensorList       *activationFeatureOutOptional,
    aclTensorList       *dynQuantScaleOutOptional,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnGroupedMatmulV5(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulV5GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px;">
  <colgroup>
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
          <td>A maximum of 128 tensors are supported.</td>
          <td>FLOAT<sup>1</sup>, FLOAT16, INT16<sup>1</sup>, INT8, INT4<sup>1</sup>, BFLOAT16, FLOAT8_E5M2<sup>2</sup>, FLOAT8_E4M3FN<sup>2</sup>, HIFLOAT8<sup>2</sup></td>
          <td>ND</td>
          <td>2–6</td>
          <td>√</td>
      </tr>
      <tr>
          <td>weight</td>
          <td>Input</td>
          <td>Weight in the formula.</td>
          <td>A maximum of 128 tensors are supported.</td>
          <td>FLOAT<sup>1</sup>, FLOAT16, INT16<sup>1</sup>, INT8, INT4, BFLOAT16, FLOAT8_E5M2<sup>2</sup>, FLOAT8_E4M3FN<sup>2</sup>, HIFLOAT8<sup>2</sup></td>
          <td>ND/NZ</td>
          <td>2–3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>biasOptional</td>
          <td>Optional input</td>
          <td>Bias in the formula.</td>
          <td>Same length as weight.</td>
          <td>FLOAT, FLOAT16, INT32, BFLOAT16<sup>2</sup></td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>scaleOptional</td>
          <td>Optional input</td>
          <td>Scale in the formula, indicating the scale factor for quantization parameters.</td>
          <td>Generally, the length is the same as the weight length. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
          <td>FLOAT, UINT64, BFLOAT16, FLOAT8_E8M0<sup>2</sup>, INT64<sup>2</sup></td>
          <td>ND</td>
          <td>1–3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>offsetOptional</td>
          <td>Optional input</td>
          <td>Offset in the formula, indicating the offset for quantization parameters.</td>
          <td>Same length as weight.</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>antiquantScaleOptional</td>
          <td>Optional input</td>
          <td>antiquant_scale in the formula, indicating the scale factor for fake-quantization parameters.</td>
          <td>Same length as weight. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>1–3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>antiquantOffsetOptional</td>
          <td>Optional input</td>
          <td>antiquant_offset in the formula, indicating the offset for fake-quantization parameters.</td>
          <td>Same length as weight. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>1–3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>perTokenScaleOptional</td>
          <td>Optional input</td>
          <td>per_token_scale in the formula, indicating the scale factor (introduced by x quantization) for quantization parameters.</td>
          <td>Generally, only 1D is supported, and the length must match the M-axis size of x. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
          <td>FLOAT, FLOAT8_E8M0<sup>2</sup></td>
          <td>ND</td>
          <td>1–2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>groupListOptional</td>
          <td>Optional input</td>
          <td>Matmul size distribution along the grouping axis for inputs and outputs.</td>
          <td>Input data in different formats based on groupListType.</td>
          <td>INT64</td>
          <td>ND</td>
          <td>1–2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>activationInputOptional</td>
          <td>Optional input</td>
          <td>Backward input of the activation function. Currently, only nullptr can be passed.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>activationQuantScaleOptional</td>
          <td>Optional input</td>
          <td>Currently, only nullptr can be passed.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>activationQuantOffsetOptional</td>
          <td>Optional input</td>
          <td>Currently, only nullptr can be passed.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>splitItem</td>
          <td>Input</td>
          <td>Specifies whether to perform tensor splitting on the output.</td>
          <td>0/1 indicates that the output is multi-tensor; 2/3 indicates that the output is single-tensor. The aclnn interface is unaware of the difference between 0 and 1, likewise for 2 and 3.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>groupType</td>
          <td>Input</td>
          <td>Axis to be grouped.</td>
          <td>The value can be -1, 0, or 2. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>groupListType</td>
          <td>Input</td>
          <td>Grouping mode of the groupList input.</td>
          <td>The value ranges from 0 to 2. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>actType</td>
          <td>Input</td>
          <td>Activation function type.</td>
          <td>The value ranges from 0 to 5. For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>tuningConfigOptional</td>
          <td>Optional input</td>
          <td>The first value indicates the expected number of tokens to be processed by each expert, which is used to optimize tiling.<br>The second value indicates an optional enable for A8W4 weight in transposed NZ format.<br>The third value indicates the extra memory space that can be used. See <a href="#constraints">Constraints</a>.</td>
          <td>Compatible with earlier versions. If this parameter is not used, do not pass it (that is, pass nullptr).</td>
          <td>INT64</td>
          <td>-</td>
          <td>3</td>
          <td>-</td>
      </tr>
      <tr>
          <td>out</td>
          <td>Output</td>
          <td>Output y in the formula.</td>
          <td>A maximum of 128 tensors are supported.</td>
          <td>FLOAT, FLOAT16, INT32<sup>1</sup>, INT8<sup>1</sup>, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td></td>
      </tr>
      <tr>
          <td>activationFeatureOutOptional</td>
          <td>Output</td>
          <td>Input data of the activation function. Currently, only nullptr can be passed.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dynQuantScaleOutOptional</td>
          <td>Output</td>
          <td>Currently, only nullptr can be passed.</td>
          <td>-</td>
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

    - The superscript "2" in the "Data Type" column of the table above indicates data types that are not supported by the products.
    - FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, and FLOAT8_E8M0 are not supported.
    - The input parameter `biasOptional` does not support BFLOAT16.
    - The input parameter `scaleOptional` does not support INT64.

  - <term>Atlas inference products</term>: Only the scenario where the data types of `x`, `weight`, and `out` are all FLOAT16 is supported. `weight` supports only the NZ data format.

- **Return**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following errors may be thrown.

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
        <td>The required input, output, or attribute is passed as a null pointer.</td>
      </tr>
      <tr>
        <td>The input weight contains elements that are null pointers.</td>
      </tr>
      <tr>
        <td>The input x contains elements that are null pointers, while the corresponding elements in the output out are non-null pointers.</td>
      </tr>
      <tr>
        <td>The input x contains elements that are non-null pointers, while the corresponding elements in the output out are null pointers.</td>
      </tr>
      <tr>
        <td rowspan="6"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="6"> 161002 </td>
        <td>The data type or data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, or out is not supported.</td>
      </tr>
      <tr>
        <td>The weight length is greater than 128; the bias is not null but its length is not equal to the weight length.</td>
      </tr>
      <tr>
        <td>The dimension of groupListOptional is 1.</td>
      </tr>
      <tr>
        <td>When splitItem is set to 2 or 3, the length of out is not 1.</td>
      </tr>
      <tr>
        <td>When splitItem is set to 0 or 1, the length of out or groupListOptional is not equal to the weight length.</td>
      </tr>
      <tr>
        <td>An element in the input parameter tuningConfigOptional is negative or exceeds the number of rows (M) in x.</td>
      </tr>
    </tbody>
  </table>

## aclnnGroupedMatmulV5

- **Parameters**

  |Parameter| Input/Output  |    Description|
  |-------|---------|----------------|
  |workspace|Input|Address of the workspace to be allocated on the device.|
  |workspaceSize|Input|Size of the workspace to be allocated on the device, which is obtained by calling `aclnnGroupedMatmulV5GetWorkspaceSize`.|
  |executor|Input|Operator executor, containing the operator computation process.|
  |stream|Input|Stream for executing the task.|

- **Return**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Scenario Types

<a id="scenario-types"></a>

- Based on the precision processing of the input data (`x` and `weight`) and the output matrix (`out`) during computation, the GroupedMatmul operator supports three primary scenarios: non-quantization, fake quantization, and full quantization.

  - <term>Atlas inference products</term>

    |Scenario|    x    |    weight       |   out | Constraints|Formula|
    |---------|---------|----------------|--------|--------|--|
    |Non-quantization|FLOAT16|FLOAT16|FLOAT16|[Constraints](#atlas-inference-products)|[Formula](#non-quantization-scenario)|

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>

    |Scenario|    x    |    weight      |   out | Constraints|Formula|
    |---------|---------|----------------|--------|--------|--|
    |Non-quantization|FLOAT32|FLOAT32|FLOAT32|[Constraints](#constraints-for-non-quantization-scenario)|[Formula](#non-quantization-scenario)|
    |Non-quantization|BFLOAT16|BFLOAT16|BFLOAT16|[Constraints](#constraints-for-non-quantization-scenario)|[Formula](#non-quantization-scenario)|
    |Non-quantization|FLOAT16|FLOAT16|FLOAT16|[Constraints](#constraints-for-non-quantization-scenario)|[Formula](#non-quantization-scenario)|
    |Full quantization - A8W8|INT8|INT8|BFLOAT16/FLOAT16/INT32/INT8|[Constraints](#constraints-for-a8w8-scenario)|[Formula](#full-quantization-scenario)|
    |Full quantization - A4W4|INT4|INT4|BFLOAT16/FLOAT16|[Constraints](#constraints-for-a4w4-scenario)|[Formula](#full-quantization-scenario)|
    |Fake quantization - A8W4|INT8|INT4|BFLOAT16/FLOAT16|[Constraints](#constraints-for-a8w4-scenario)|[Formula](#a8w4-fake-quantization)|
    |Fake quantization - A16W8|BFLOAT16/FLOAT16|INT8|BFLOAT16/FLOAT16|[Constraints](#constraints-for-a16w8-scenario)|[Formula](#fake-quantization-scenario)|
    |Fake quantization - A16W4|BFLOAT16/FLOAT16|INT4|BFLOAT16/FLOAT16|[Constraints](#constraints-for-a16w4-scenario)|[Formula](#fake-quantization-scenario)|

<a id="formulas"></a>

- Formulas
  <a id="non-quantization-scenario"></a>

  - **Non-quantization scenario:**

    $$
    y_i=x_i\times weight_i + bias_i
    $$

  <a id="full-quantization-scenario"></a>

  - **Full-quantization scenario (without perTokenScaleOptional):**
    - `x` in INT8 and `bias` in INT32

      $$
      y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
      $$

  - **Full-quantization scenario (with perTokenScaleOptional):**
    - `x` in INT8 and `bias` in INT32

      $$
      y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i
      $$

    - `x` in INT8 and `bias` in BFLOAT16

      $$
      y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i  + bias_i
      $$

    - `x` in INT4, no `bias`

      $$
      y_i=x_i\times (weight_i * scale_i) * per\_token\_scale_i
      $$

  <a id="fake-quantization-scenario"></a>

  - **Fake-quantization scenario:**

    - `x` in FLOAT16 or BFLOAT16 and `weight` in INT4 or INT8 (supported only when `x`, `weight`, and `y` are all single-tensor)

      $$
      y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
      $$

    <a id="a8w4-fake-quantization"></a>

    - `x` in INT8 and `weight` in INT4 (supported only when `x`, `weight`, and `y` are all single-tensor) (`bias` is a required parameter. It is an auxiliary result of offline computation. It is defined as $bias_i=8\times weight_i * scale_i$, reduced along the K-axis.)
    
      $$
      y_i=((x_i - 8) \times weight_i * scale_i+bias_i ) * per\_token\_scale_i
      $$

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatmulV5` defaults to deterministic implementation.
  
<details>
<summary><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></summary>

  - **Common constraints**
  <a id="common-constraints"></a>
    - If `x` and `weight` need to be transposed, the corresponding tensors must be [non-contiguous](../../../docs/en/context/non_contiguous_tensor.md).
    - The size of the last dimension for each tensor in `x` and `weight` should be less than 65536. The last dimension of $x_i$ refers to the K-axis when `x` is not transposed or the M-axis when `x` is transposed. The last dimension of $weight_i$ refers to the N-axis when `weight` is not transposed or the K-axis when `weight` is transposed.
    - When the `weight` [data format](../../../docs/en/context/data_format.md) is FRACTAL_NZ, the shape of `weight` must meet the requirements of the FRACTAL_NZ format.
    - Generally, `perTokenScaleOptional` supports only 1D, and the length must match the M-axis size of `x`. This parameter only supports scenarios where `x`, `weight`, and `out` are all single-tensor (with a TensorList length of 1).
    - When the length of the TensorList in the output is 1, `groupListOptional` constrains the valid portion of the output data. Any portion not specified in `groupListOptional` will not be updated.
    - When `groupListType` is `0`, `groupListOptional` must be a non-negative, monotonically non-decreasing sequence, representing the cumulative sum (cumsum) results of the grouping axis sizes. When `groupListType` is `1`, it must be a non-negative sequence representing the size of each group along the grouping axis. When `groupListType` is `2`, it must be a non-negative sequence with a shape of [E, 2], where $E$ represents the group size. The data layout is `[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]`, where `groupSize` indicating the size of each group along the grouping axis. For details, see [groupListOptional configuration examples](#grouplistoptional-configuration-examples).
    - `groupType` indicates the axis to be grouped. For example, if the matrix multiplication is `C[m,n]=A[m,k]xB[k,n]`, `groupType` has the following options: `-1` means no axis grouping, `0` indicates M-axis grouping, `1` indicates N-axis grouping, and `2` indicates K-axis grouping. Currently, N-axis grouping is not supported. For details, see <a href="#groupType-constraints">groupType constraints</a>.
    - `actType` (int64\_t, computation input): integer type, indicating the activation function type. The value ranges from 0 to 5. The supported enumerated values are as follows:
      * 0: GMMActType::GMM_ACT_TYPE_NONE
      * 1: GMMActType::GMM_ACT_TYPE_RELU
      * 2: GMMActType::GMM_ACT_TYPE_GELU_TANH
      * 3: GMMActType::GMM_ACT_TYPE_GELU_ERR_FUNC (not supported)
      * 4: GMMActType::GMM_ACT_TYPE_FAST_GELU
      * 5: GMMActType::GMM_ACT_TYPE_SILU

    <a id="constraints-for-a8w8-scenario"></a>
    
    <details>
    <summary>Constraints for the A8W8 scenario</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT8 | INT8 (ND) | INT32/null | UINT64 | null | null | null | null | INT64 | null | null | null | INT8 |
      | INT8 | INT8 (ND/NZ) | INT32/null |BFLOAT16| null | null | null | FLOAT/null | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT8 (ND/NZ) | BFLOAT16/null |FLOAT/BFLOAT16| null | null | null | FLOAT/null | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT8 (ND/NZ) | INT32/null | FLOAT | null | null | null | FLOAT/null | INT64 | null | null | null | FLOAT16 |
      | INT8 | INT8 (ND/NZ) | INT32/null | null | null | null | null | null | INT64 | null | null | null | INT32 |

    - **Constraints**

      In addition to [common constraints](#common-constraints), other constraints in the A8W8 scenario are as follows:
      - Only `GroupType=0` (M-axis grouping) is supported.
      - Currently, `x`, `weight`, and `out` are only supported as TensorLists with a length of 1.
      - `x` cannot be transposed.
      - `x` supports only 2D tensors with shape (M, K).
      - `weight` supports only 3D tensors with shape (E, K, N) or (E, N, K).
      - To enable the fixed-axis algorithm for performance optimization, the following input shape and parameter configuration requirements must be met:
        * Input shape requirements (any of the following requirements must be met)

          The shape of `x` is (M, 7168), and the shape of `weight` is (7168, 4096).

          The shape of `x` is (M, 2048), and the shape of `weight` is (2048, 7168).
        * Parameter configuration requirements

          The first element of `tuningConfigOptional` must be set to a value greater than 128 and less than 512.

          The second element of `tuningConfigOptional` must be set to 0.

          The third element of `tuningConfigOptional` must be set to -1 or a value greater than or equal to M × N × 4.
          
    </details>

    <a id="constraints-for-a8w4-scenario"></a>

    <details>
    <summary>Constraints for the A8W4 scenario</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | null | null | null | FLOAT | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | null | null | FLOAT | INT64 | null | null | null | FLOAT16 |

    - **Constraints**

      In addition to [common constraints](#common-constraints), other constraints in the A8W4 scenario are as follows:
      - Only `GroupType=0` (M-axis grouping) and `actType=0` are supported.
      - Currently, `x`, `weight`, and `out` are only supported as TensorLists with a length of 1.
      - `x` and `weight` cannot be transposed.
      - `x` supports only 2D tensors with shape (M, K).
      - By default, `weight` supports 3D tensors with shape (E, K, N).
      - The bias is the auxiliary result of offline computation during the computation process. Its value must be $8\times weight \times scale$ and is accumulated in the first dimension. The shape must be $[E, N]$.
      - When the data type of `weight` is INT32, each INT32 value is treated as eight INT4 values.
      - When `offset` is null:
        - In this case, `groupListType` can only be `1` (the operator does not validate `groupListType` and treats it as `1`). `k` must be an integer multiple of `quantGroupSize` and `k` ≤ 18432. `quantGroupSize` is the per-group quantization length in the K-axis. Currently, `quantGroupSize=256` is supported.
        - In this case, `n` must be an integer multiple of 8.
        - In this case, the scale is the result after per-group and per-channel offline fusion. The shape must be $[E, quantGroupNum, N]$, where $quantGroupNum=k \div quantGroupSize$.
        - In this case, performance is typically enhanced when the expected number of tokens processed by each expert—specifically, the first value in `tuningConfigOptional`—exceeds $n/4$. This will result in an additional memory consumption of $g\times k \times n$ bytes (where $g$ represents the number of Matmul groups).
      - When `offset` is not null:
        - The scale is the result after per-group and per-channel offline fusion. The shape must be $[E, 1, N]$.
        - In this case, `offsetOptional` is not null. For asymmetric quantization, `offsetOptional` is the auxiliary result of offline computation during the computation process, that is, $antiquantOffset \times scale$. The shape must be $[E, 1, N]$, and the data type must be FLOAT32.
      - The second element of the `tuningConfigOptional` array can be set to `1` to enable a specific weight format template for the A8W4 scenario, thereby optimizing operator performance. (This provides a performance advantage when the shape meets the criteria: K >= 2048 && N >= 2048). Note that this template requires `weight` to have an initial shape of (E, N, K), which then undergoes ND2NZ conversion before being provided as the operator input.
    </details>

    <a id="constraints-for-a16w4-scenario"></a>

    <details>
    <summary>Constraints for the A16W4 scenario</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | FLOAT16 | INT4 (ND) | FLOAT16/null | null | null | FLOAT16 | FLOAT16 | null | INT64 | null | null | null | FLOAT16 |
      | BFLOAT16| INT4 (ND) | FLOAT/null | null | null | BFLOAT16 | BFLOAT16 | null | INT64 | null | null | null | BFLOAT16|

    - **Constraints**

      In addition to [common constraints](#common-constraints), other constraints in the A16W4 scenario are as follows:
      - `x` cannot be transposed.
      - Only GroupType -1 or 0, actType 0, and groupListType 0 or 1 are supported.
      - The size of the last dimension for each tensor in `weight` must be an even number. Specifically, the last dimension refers to the N-axis of $weight_i$ when `weight` is not transposed, or the K-axis of $weight_i$ when `weight` is transposed.
      - Symmetric quantization supports both per-channel and per-group modes. If the per-group mode is used, the number of quantization groups ($G$ or $G_i$) must be exactly divisible by the corresponding $k_i$.
      - Asymmetric quantization only supports the per-channel mode.
      - In the per-group mode, when `weight` is transposed, the per-group length $s_i$ must be an even number.
      - For multi-tensor `weight`, the per-group length is defined as $s_i = k_i / G_i$, and all $s_i(i=1,2,...g)$ values must be the same.
      - The shapes of the fake-quantization parameters `antiquantScaleOptional` and `antiquantOffsetOptional` must meet the following requirements ($g$ indicates the number of Matmul groups, $G$ indicates the number of quantization groups, and $G_i$ indicates the number of quantization groups of the *i*-th tensor).

        | Application Scenario| Sub-scenario| Shape Restriction|
        |:---------:|:-------:| :-------|
        | Fake-quantization (per-channel)| Single-tensor `weight`| $[E, N]$|
        | Fake-quantization (per-channel)| Multi-tensor `weight`| $[N_i]$|
        | Fake-quantization (per-group)| Single-tensor `weight`| $[E, G, N]$|
        | Fake-quantization (per-group)| Multi-tensor `weight`| $[G_i, N_i]$|

    </details>

    <a id="constraints-for-a16w8-scenario"></a>

    <details>
    <summary>Constraints for the A16W8 scenario</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | FLOAT16 | INT8 (ND) | FLOAT16/null | null | null | FLOAT16 | FLOAT16 | null | INT64 | null | null | null | FLOAT16 |
      | BFLOAT16| INT8 (ND) | FLOAT/null | null | null | BFLOAT16 | BFLOAT16 | null | INT64 | null | null | null | BFLOAT16|

    - **Constraints**

      In addition to [common constraints](#common-constraints), other constraints in the A16W8 scenario are as follows:
      - `x` cannot be transposed.
      - Only GroupType -1 or 0, actType 0, and groupListType 0 or 1 are supported.
      - Only the per-channel quantization mode is supported.
      - For multi-tensor `weight`, the per-group length is defined as $s_i = k_i / G_i$, and all $s_i(i=1,2,...g)$ values must be the same.
      - The shapes of the fake-quantization parameters `antiquantScaleOptional` and `antiquantOffsetOptional` must meet the following requirements ($g$ indicates the number of Matmul groups).

        | Application Scenario| Sub-scenario| Shape Restriction|
        |:---------:|:-------:| :-------|
        | Fake-quantization (per-channel)| Single-tensor `weight`| $[E, N]$|
        | Fake-quantization (per-channel)| Multi-tensor `weight`| $[N_i]$|

    </details>

    <a id="constraints-for-a4w4-scenario"></a>

    <details>
    <summary>Constraints for the A4W4 scenario</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT4 | INT4 (ND/NZ) | null | UINT64 | null | null | null | FLOAT/null | INT64 | null | null | null | FLOAT16/BFLOAT16|

    - **Constraints**

      In addition to [common constraints](#common-constraints), other constraints in the A4W4 scenario are as follows:
      - Only GroupType 0 (M-axis grouping), actType 0, and groupListType 0 or 1 are supported.
      - Currently, `x`, `weight`, and `out` are only supported as TensorLists with a length of 1.
      - `x` and `weight` cannot be transposed.
      - `x` supports only 2D tensors with shape (M, K).
      - `weight` supports only 3D tensors with shape (E, K, N).
      - If the data format of `weight` is ND, `n` must be an integer multiple of 8.
      - Per-channel and per-group quantization are supported. In the per-channel scenario, the shape of the scale must be $[E, N]$. In the per-group scenario, the shape must be $[E, G, N]$.
      - In the per-group scenario, $G$ must be exactly divisible by $k$, and $k/G$ must be an even number.
    </details>

    <a id="constraints-for-non-quantization-scenario"></a>

    <details>
    <summary>Constraints for the non-quantization scenario</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | FLOAT | FLOAT (ND) | FLOAT/null | null | null | null | null | null | INT64 | null | null | null | FLOAT |
      | FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | null | null | null | null | null | INT64 | null | null | null | FLOAT16 |
      | BFLOAT16| BFLOAT16(ND/NZ) | FLOAT/null | null | null | null | null | null | INT64 | null | null | null | BFLOAT16|

    - **Constraints**

      In addition to [common constraints](#common-constraints), other constraints in the non-quantization scenario are as follows:
      - GroupType -1, 0, or 2, actType 0, and groupListType 0 or 1 are supported.
    </details>

    <a id="groupType-constraints"></a>

    <details>
    <summary>groupType constraints</summary>

    - In A16W8 and A16W4 scenarios, `groupType` can be -1 or 0.
    - In A8W8, A8W4, and A4W4 scenarios, `x` can only be single -tensor when `groupType` is `0`.
    - `x`, `weight`, and `y` are of type aclTensorList, which is an array of aclTensor objects. In the following table, "S" indicates an aclTensorList consisting of one aclTensor, and "M" indicates an aclTensorList consisting of multiple aclTensors. For example, "SMS" indicates single-tensor `x`, multi-tensor `weight`, and single-tensor `y`.

      | groupType | Tensor Count in `x`| Tensor Count in `weight`| Tensor Count in `y`| splitItem| groupListOptional | Transposition| Other Constraints|
      |:---------:|:-------:|:-------:|:-------:|:--------:|:------------------|:--------| :-------|
      | -1 | M|M|M| 0/1 | `groupListOptional` must be passed as null.| (1) `x` cannot be transposed.<br> (2) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.| The tensors in `x` must have the same dimensionality, which can be 2D. The tensors in `weight` must be 2D. The dimensionality of tensors in `y` must match that of `x`.|
      | 0 | S|S|S| 2/3 | (1) `groupListOptional` must be passed.<br> (2) When `groupListType` is `0`, the last value must be less than or equal to the first dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be less than or equal to the first dimension of the tensor in `x`. When `groupListType` is `2`, the sum of the values in the second column must be less than or equal to the first dimension of the tensor in `x`.<br> (3) The first dimension of `groupListOptional` supports a maximum of 1024 groups.|(1) `x` cannot be transposed.<br> (2) `weight` can be transposed, except in A8W4 and A4W4 scenarios.|The tensor in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.|
      | 0 | S|M|S| 2/3 | (1) `groupListOptional` must be passed.<br> (2) When `groupListType` is `0`, the last value must be less than or equal to the first dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be less than or equal to the first dimension of the tensor in `x`. When `groupListType` is `2`, the sum of the values in the second column must be less than or equal to the first dimension of the tensor in `x`.<br> (3) The first dimension of `groupListOptional` supports a maximum of 128 groups.|(1) `x` cannot be transposed.<br> (2) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.|(1) Tensors in `x`, `weight`, and `y` must be 2D.<br> (2) The N-axis of each tensor in `weight` must be the same.|
      | 0 | M|M|S| 2/3 | (1) `groupListOptional` is optional.<br> (2) If `groupListOptional` is passed: when `groupListType` is `0`, the differences of `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`; when `groupListType` is `1`, the `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`; when `groupListType` is `2`, the values in the second column of `groupListOptional` must be in one-to-one mapping with the first dimension of the tensors in `x`.<br> (3) The first dimension of `groupListOptional` supports a maximum of 128 groups.|(1) `x` cannot be transposed.<br> (2) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.|(1) Tensors in `x`, `weight`, and `y` must be 2D.<br> (2) The N-axis of each tensor in `weight` must be the same.|
      | 2 | S|S|S| 2/3 | (1) `groupListOptional` must be passed.<br> (2) When `groupListType` is `0`, the last value must be less than or equal to the second dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be less than or equal to the second dimension of the tensor in `x`. When `groupListType` is `2`, the sum of the values in the second column must be less than or equal to the second dimension of the tensor in `x`.<br> (3) The first dimension of `groupListOptional` supports a maximum of 1024 groups.| (1) `x` must be transposed.<br> (2) `weight` cannot be transposed.|(1) The tensors in `x` and `weight` must be 2D, and the tensor in `y` must be 3D.<br> (2) `bias` must be passed as null.|
      | 2 | S|M|M| 0/1 | `groupListOptional` must be passed as null.| (1) `x` must be transposed.<br> (2) `weight` cannot be transposed.| (1) Tensors in `x`, `weight`, and `y` must be 2D.<br> (2) The maximum length of `weight` is 128, meaning a maximum of 128 groups are supported.<br> (3) The sum of the first dimensions of all tensors in the original `weight` shape should not exceed the first dimension of `x`.<br> (4) `bias` must be passed as null.|

    </details>

    <a id="grouplistoptional-configuration-examples"></a>

    <details>
    <summary>groupListOptional configuration examples</summary>

    - Shape information
      M = 789, K = 4096, N = 7168, E = 8 (Experts 0, 2, and 5 have tokens to process. Expert 0 processes 123 tokens, and experts 2 and 5 each process 333 tokens.)
      The shape of X is [[789, 4096]].
      The shape of W is [[9, 4096, 7168]].
      The shape of Y is [[789, 7168]].

    - When `groupListType` is set to `0`, the configuration is as follows:
      - groupListOptional: `[123, 123, 456, 456, 456, 789, 789, 789, 789]`

    - When `groupListType` is set to `1`, the configuration is as follows:
      - groupListOptional: [123, 0, 333, 0, 0, 333, 0, 0, 0]

    - When `groupListType` is set to `2`, the configuration is as follows:
      - In this mode, `groupListOptional` moves all non-zero groups to the front. This is suitable for scenarios with a large number of inactive experts.
      - groupListOptional: `[[0, 123], [2, 333], [5, 333], [1, 0], [3, 0], [4, 0], [6, 0], [7, 0], [8, 0]]`
    </details>

    <a id="tuningconfigoptional-configuration-examples"></a>

    <details>
    <summary>tuningConfigOptional configuration examples</summary>

    - The input parameter `tuningConfigOptional` is an aclIntArray on the host. The array stores INT64 elements. It is compatible with earlier versions. If this parameter is not used, do not pass it (that is, pass nullptr).
      * First element:

        Semantic: Represents the expected number of tokens processed by each expert. The operator performs optimal tiling based on the first element in this array to achieve better performance.
        
        Application scenarios: [A8W4](#constraints-for-a8w4-scenario) and [A8W8](#constraints-for-a8w8-scenario), where `x`, `weight`, and `out` are all single-tensor.
      
      * Second element:
        
        Semantic: Specifies whether to enable the core-affinity format for `weight`. This format requires the weight layout to be transposed first, followed by a conversion to the NZ format.
        
        Application scenario: [A8W4](#constraints-for-a8w4-scenario)
      
      * Third element:
        
        Description: Specifies the maximum allowable additional workspace. This operator utilizes the fixed-axis algorithm in certain scenarios to enhance performance, which requires additional memory. If there is no strict limit on workspace usage, this element can be set to `-1`.
        
        Application scenario: [A8W8](#constraints-for-a8w8-scenario)

    </details>

</details>

<a id="atlas-inference-products"></a>

<details>
<summary><term>Atlas inference products</term></summary>

- Product specifications

  - `groupType`: integer type, indicating the axis to be grouped. Currently, only M-axis grouping is supported.
  - `groupListType`: The value can be 0 or 1. `0` indicates that the values in `groupListOptional` are non-negative, monotonically non-decreasing numbers, representing the cumulative sum (cumsum) results of the grouping axis sizes. `1` indicates that the values in `groupListOptional` are non-negative numbers, representing the size of each group along the grouping axis.
  - `actType`: Currently, only `0` is supported, indicating `GMMActType::GMM_ACT_TYPE_NONE`.
  - `tuningConfigOptional`: This parameter is not supported.
  - The input and output support only the FLOAT16 type. The N-axis size of the output `y` must be a multiple of 16.

- Supported scenarios

  | groupType | Tensor Count in `x`| Tensor Count in `weight`| Tensor Count in `y`| Scenario Constraints|
  |:---------:|:-------:|:-------:|:-------:| :------ |
  | 0 | S|S|S|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensor in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.<br>(3) `groupListOptional` must be passed. When `groupListType` is `0`, the last value must be equal to the first dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be equal to the first dimension of the tensor in `x`.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) `weight` can be transposed, but `x` cannot.|

</details>

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_matmul_v5.h"

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
      std::vector<std::vector<int64_t>> biasShape = {{2, 256}};
      std::vector<std::vector<int64_t>> yShape = {{512, 256}};
      std::vector<int64_t> groupListShape = {{2}};
      std::vector<int64_t> groupListData = {256, 512};
      void* xDeviceAddr[1];
      void* weightDeviceAddr[1];
      void* biasDeviceAddr[1];
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

      // Create a tuningconfig aclIntArray.
      std::vector<int64_t> tuningConfigData = {512};
      aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

      // Create an x aclTensorList.
      ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a weight aclTensorList.
      ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a bias aclTensorList.
      ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a y aclTensorList.
      ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a group_list aclTensor.
      ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // 3. Call the CANN operator library API.
      // Call the first-phase API of aclnnGroupedMatmulV5.
      ret = aclnnGroupedMatmulV5GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput, activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, tuningConfig, out, activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnGroupedMatmulV5.
      ret = aclnnGroupedMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5 failed. ERROR: %d\n", ret); return ret);

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
