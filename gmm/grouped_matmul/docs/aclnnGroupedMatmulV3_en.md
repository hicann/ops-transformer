# aclnnGroupedMatmulV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:04:42.851Z -->

**Notice: This API will be deprecated in later versions. Use the latest aclnnGroupedMatmulV5 API instead.**

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs grouped matrix multiplication, where the dimension sizes of each group can differ. The basic function is matrix multiplication, as in $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where g is the number of groups and $m_i/k_i/n_i$ are the corresponding shapes. Both input and output data types are aclTensorList, and the corresponding functions are:

  - k-axis grouping: $k_i$ values differ, but $m_i/n_i$ are the same for each group. In this case, $x_i/weight_i$ can be concatenated along $k_i$.
  - m-axis grouping: $k_i$ is the same across groups, and $weight_i/y_i$ can be concatenated along $n_i$.

  Compared with the [aclnnGroupedMatmul](aclnnGroupedMatmul_en.md) API, **this API adds the following:**

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
      - In non-quantization scenarios, weight transpose is supported (transpose refers to the scenario where, if the shape is [M,K], the stride is [1,M] and the data layout is [K,M]).
      - Supports m-axis and k-axis grouping, indicated by groupType.
      - When x, weight, and y are all single tensors, in non-quantization scenarios, both x and weight inputs of float32 type are supported.
      - In quantization and pseudo-quantization scenarios, weight transposition is supported, and weight as a single tensor is supported.
      - For the features supported by the [aclnnGroupedMatmul](aclnnGroupedMatmul_en.md) API, this API does not support the use case where x is a single tensor and weight/y are multiple tensors.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>:
      - In pseudo-quantization scenarios, weight transposition is supported, and x, weight, and y as single tensors are all supported.
      - For features supported by the [aclnnGroupedMatmul](aclnnGroupedMatmul_en.md) API, this API does not support the use case where x is a single tensor and weight/y are multiple tensors.

    <!-- end id8 -->

**NOTE**

  - Single tensor means that all grouped tensors in a tensor list are merged into one on the grouping axis specified by groupType; otherwise, it is multi-tensor.
  - Tensor transposition: refers to the scenario where the tensor shape is [M, K], the stride is [1, M], and the data layout is [K, M], i.e., a non-contiguous tensor.

  - Formula:
    - **Non-quantized scenario:**

      $$
       y_i=x_i\times weight_i + bias_i
      $$

    - **Quantization scenario:**

      $$
       y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
      $$

    - **Dequantized scenario:**

      $$
       y_i=(x_i\times weight_i + bias_i) * scale_i
      $$

    - **Pseudo-Quantized Scenario:**

      $$
       y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnGroupedMatmulV3GetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the calculation process. Then call the "aclnnGroupedMatmulV3" API to perform the computation.

```cpp
aclnnStatus aclnnGroupedMatmulV3GetWorkspaceSize(
  const aclTensorList   *x,
  const aclTensorList   *weight,
  const aclTensorList   *biasOptional,
  const aclTensorList   *scaleOptional,
  const aclTensorList   *offsetOptional,
  const aclTensorList   *antiquantScaleOptional,
  const aclTensorList   *antiquantOffsetOptional,
  const aclTensor       *groupListOptional,
  int64_t                splitItem,
  int64_t                groupType,
  const aclTensorList   *y,
  uint64_t              *workspaceSize,
  aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedMatmulV3GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1483px"><colgroup>
  <col style="width: 240px">
  <col style="width: 90px">
  <col style="width: 370px">
  <col style="width: 232px">
  <col style="width: 339px">
  <col style="width: 86px">
  <col style="width: 92px">
  <col style="width: 64px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensorList *)</td>
      <td>Input</td>
      <td>Input x in the formula.</td>
      <td>The tensorList length supports [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight (aclTensorList *)</td>
      <td>Input</td>
      <td>Weight in the formula.</td>
      <td>The length of tensorList supports [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Bias in the formula.</td>
      <td>Same length as weight.</td>
      <td>INT32, BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the scaling factor in quantization parameters.</td>
      <td>Generally, the length is the same as that of weight.</td>
      <td>UINT64, INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the offset in the quantization parameters.</td>
      <td>Length and weight are the same.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the scale factor in pseudo-quantization parameters.</td>
      <td>The length is the same as that of weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the offset in pseudo-quantization parameters.</td>
      <td>The length is the same as weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListOptional (aclTensor *)</td>
      <td>Input</td>
      <td>Represents the size distribution of matmul along the grouping axis of input and output.</td>
      <td>When the length of TensorList in output is 1, the last value in groupListOptional constrains the valid part of the output data, and the parts not specified in groupListOptional will not participate in the update.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>splitItem (int64_t)</td>
      <td>Input</td>
      <td>An integer parameter that represents whether the output needs to be split into tensors.</td>
      <td>      0/1 represents that the output is multiple tensors
      2/3 represents that the output is a single tensor.
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType (int64_t)</td>
      <td>Input</td>
      <td>Integer parameter that represents the axis to be grouped.</td>
      <td>If the matrix multiplication is C[m,n]=A[m,k]xB[k,n], the value of groupType is -1: no grouping, 0: grouping on the m axis, 2: grouping on the k axis.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensorList *)</td>
      <td>Output</td>
      <td>Represents output y in the formula.</td>
      <td>The length of tensorList supports [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, INT8, FLOAT32, INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - x supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
    - weight supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
    - biasOptional supports FLOAT16, FLOAT32, and INT32.
    - y supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
    - The input parameters x and weight, and the output parameter y support up to 128 tensors. Except for the single-multiple-single extension described below.

    > For Atlas A2/A3 products, in the split-M single-multiple-single scenario, the weight TensorList supports up to 1024 tensors when splitItem=3; the limit remains 128 when splitItem=2.

  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - x supports FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - weight supports FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - biasOptional supports FLOAT16, BFLOAT16, FLOAT32, and INT32.
    - y supports FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - offsetOptional is not supported.
    - groupType supports grouping and non-grouping along the m-axis. Grouping along the k-axis is supported only in non-quantization scenarios.
    - The input parameters x and weight, and the output parameter y support a maximum of 1024 tensors in non-quantization scenarios, a maximum of 128 tensors in pseudo-quantization scenarios, and a maximum of 1 tensor in quantization scenarios.

  <!-- end id10 -->

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1055px"><colgroup>
  <col style="width: 242px">
  <col style="width: 78px">
  <col style="width: 735px">
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
      <td>The input parameter is a required input, output, or required attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type and data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, or out are not within the supported range.</td>
    </tr>
    <tr>
      <td>The length of weight is not within the supported range.</td>
    </tr>
    <tr>
      <td>If bias is not empty, the length of bias is not equal to the length of weight.</td>
    </tr>
    <tr>
      <td>The dimension of groupListOptional is not within the supported range.</td>
    </tr>
    <tr>
      <td>In the use cases where splitItem is 2 or 3, the length of out is not equal to 1.</td>
    </tr>
    <tr>
      <td>In the use cases where splitItem is 0 or 1, the length of out is not equal to the length of weight, or the length of groupListOptional is not equal to the length of weight.</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulV3

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 834px"><colgroup>
    <col style="width: 118px">
    <col style="width: 87px">
    <col style="width: 629px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameters</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Pointer to the workspace memory allocated on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the Device side, obtained by calling the first-phase API aclnnGroupedMatmulV3GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the computation process of the operator.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
      </tr>
    </tbody>
    </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupedMatmulV3 defaults to a deterministic implementation.
- If groupListOptional is passed in, groupListOptional must be a monotonically non-decreasing sequence of non-negative numbers.
- The size of each dimension of each group of tensors in x and weight, after 32-byte alignment, must be less than the maximum int32 value 2147483647.

<!-- npu="A3,910b" id11 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - The supported input types for the non-quantized scenario are:
    - x is FLOAT16, weight is FLOAT16, biasOptional is FLOAT16, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, y is FLOAT16.
    - x is BFLOAT16, weight is BFLOAT16, biasOptional is FLOAT32, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, y is BFLOAT16.
    - x is FLOAT32, weight is FLOAT32, biasOptional is FLOAT32, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, y is FLOAT32 (supported only when x, weight, and y are all single-tensor scenarios).
  - The input types supported in the quantized scenario are:
    - x is INT8, weight is INT8, biasOptional is INT32, scaleOptional is UINT64, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, and y is INT8.
  - The input types supported in the pseudo-quantized scenario are:
    - x is FLOAT16, weight is INT8, biasOptional is FLOAT16, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is FLOAT16, antiquantOffsetOptional is FLOAT16, and y is FLOAT16.
    - x is BFLOAT16, weight is INT8, biasOptional is FLOAT32, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is BFLOAT16, antiquantOffsetOptional is BFLOAT16, and y is BFLOAT16.
  - Supported scenarios for different groupType values:
    - The a16w8 and a16w4 scenarios only support groupType values of -1 and 0.
    - The A8W8, A8W4, and A4W4 scenarios only support the case where groupType is 0 and the number of x tensors is single.
    - The input types of x, weight, and y are aclTensorList, which represents an array object of the aclTensor type. In the following table of supported scenarios, "single" indicates an aclTensorList consisting of one aclTensor, and "multiple" indicates an aclTensorList consisting of multiple aclTensors. For example, "single-multiple-single" means that x is a single tensor, weight is multiple tensors, and y is a single tensor.

      | groupType | Number of x Tensors | Number of weight Tensors | Number of y Tensors | splitItem| groupListOptional | Transpose | Remaining Scenario Restrictions |
      |:---------:|:-------:|:-------:|:-------:|:--------:|:------------------|:--------| :-------|
      | -1 | Multiple|Multiple|Multiple | 0/1 | groupListOptional must be null. | 1) x does not support transpose.<br> 2) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must be consistent.| Tensors in x must have consistent dimensions and support 2D. Tensors in weight must be 2D. Tensors in y must have the same dimensions as x. |
      | 0 | Single|Single|Single | 2/3 | 1) groupListOptional must be passed. <br> 2) The first dimension of groupListOptional supports a maximum of 1024, that is, a maximum of 1024 groups. |1) x does not support transpose.<br> 2) Weight transpose is supported, but A8W4 and A4W4 scenarios do not support weight transpose. |Tensors in weight must be 3D. Tensors in x and y must be 2D.|
      | 0 | Single|Multiple|Single | 2/3 | 1) groupListOptional must be passed. <br> 2) The first dimension of groupListOptional supports a maximum of 1024, that is, a maximum of 1024 groups.<br>The 1024-group limit applies only to splitItem=3; splitItem=2 remains limited to 128 groups. |1) x does not support transpose.<br> 2) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must be consistent. |1) Tensors in x, weight, and y must be 2D.<br> 2) The N axis of each tensor in weight must be equal. |
      | 0 | Multiple|Multiple|Single | 2/3 | 1) groupListOptional is optional. <br> 2) The first dimension of groupListOptional supports a maximum of 128, that is, a maximum of 128 groups. |1) x does not support transpose.<br> 2) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must be consistent.|1) Tensors in x, weight, and y must be 2D.<br> 2) The N axis of each tensor in weight must be equal. |
      | 2 | Single|Single|Single | 2/3 | 1) groupListOptional must be passed. <br> 2) The first dimension of groupListOptional supports a maximum of 1024, that is, a maximum of 1024 groups. | 1) x must be transposed.<br> 2) weight cannot be transposed. |1) Tensors in x and weight must be 2D. Tensors in y must be 3D.<br> 2) bias must be null.|
      | 2 | Single|Multiple|Multiple | 0/1 | groupListOptional must be null. | 1) x must be transposed.<br> 2) weight cannot be transposed.| 1) Tensors in x, weight, and y must be 2D.<br> 2) The weight length supports a maximum of 128, i.e., up to 128 groups.<br> 3) In the original shape, the sum of the first dimension of each tensor in weight must not exceed the first dimension of x.<br> 4) bias must be null. |

  - The last dimension of each group of tensors in x and weight must be less than 65536. The last dimension of $x_i$ refers to the K-axis of $x_i$ when x is not transposed, or the M-axis of $x_i$ when x is transposed. The last dimension of $weight_i$ refers to the N-axis of $weight_i$ when weight is not transposed, or the K-axis of $weight_i$ when weight is transposed.

<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT products</term>:

  <details>
    <summary>Non-quantized and quantization scenario constraints</summary>
      <a id="non-quantized-scenario-constraints"></a>

  - The data types supported in non-quantized scenarios are:
    - The following input parameters are empty: scaleOptional, offsetOptional, antiQuantScaleOptional, antiQuantOffsetOptional
    - The supported data type combinations for non-empty parameters must satisfy the following table:

        |groupType| x       | weight  | biasOptional | y     |
        |:-------:|:-------:|:-------:| :------      |:------ |
        |-1/0/2   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
        |-1/0/2   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|
        |-1/0/2   |FLOAT32     |FLOAT32     |FLOAT32/null    | FLOAT32|

  </details>

  <details>
  <summary>Pseudo-quantized and quantization scenario constraints</summary>
  <a id="pseudo-quantized-scenario-constraints"></a>

  - The data types supported in the pseudo-quantized scenario are:
    - The following input parameters are empty: scaleOptional and offsetOptional.
    - The supported data type combinations for non-empty parameters must satisfy the following table:

      |groupType| x       | weight  | antiquantScaleOptional |   antiquantOffsetOptional | biasOptional | y     |
      |:-------:|:-------:|:-------:| :------  | :------ | :------     |:------ |
      |-1/0   |BFLOAT16     |INT8   |BFLOAT16  |BFLOAT16/null  |BFLOAT16/FLOAT32/null   | BFLOAT16|
      |-1/0   |FLOAT16     |INT8  |FLOAT16 |FLOAT16/null   |FLOAT16/null| FLOAT16|

    - antiquantScaleOptional and non-null biasOptional and antiquantOffsetOptional must satisfy the following table (where g is the number of matmul groups, i.e., the number of groups):

      |groupType| usage scenario | Shape Constraint |
      |:---------:|:---------:| :------ |
      |-1|Weight multi-tensor|Each tensor is 1D with shape ($n_i$). It is not allowed that some tensors in a tensorList have shape ($n_i$) while others are empty. |
      |0|Single weight tensor|Each tensor is 2D with shape (g, N).|

    - Only single-single-single and multi-multi-multi scenarios are supported.
  </details>

  <details>
  <summary>Quantization Scenario Constraints</summary>

  - The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional.
  - Only the single-single-single scenario is supported.

  - The supported data type and dimension combinations for non-null parameters must satisfy the following table (where g is the number of matmul groups, i.e., the number of groups):

      |groupType| x dtype     | x shape | weight dtype |weight shape| biasOptional dtype |biasOptional shape| scaleOptional dtype |scaleOptional shape| out dtype    |out shape|
      |:-------:|:-------:|:-------:|:-------:|:-------:| :------      | :------      |:-------       | :------ |:-------       | :------ |
      |0|INT8     |(M,K)|INT8     |(g,K,N)/(g,N,K)|INT32/null    | (g,N)|UINT64/INT64  |(g,N)/null|INT8|(M,N)|

  </details>

  <details>
  <summary>groupType scenario constraints</summary>
  <a id="groupType-scenario-constraints"></a>

  - Supported scenarios for different groupType values:
    - In the supported scenarios, "single" indicates a single tensor, "multiple" indicates multiple tensors, and the order is x, weight, y. For example, "single-multiple-single" indicates a scenario where x is a single tensor, weight consists of multiple tensors, and y is a single tensor.

        | groupType | Supported Scenarios | scenario restriction |
        |:---------:|:-------:| :-------|
        | -1 | multiple-multiple-multiple |1) Only splitItem 0/1 is supported.<br>2) For non-quantized x, tensors in out must be 2D with shapes ($m_i$, $k_i$) and ($m_i$, $n_i$) respectively. For pseudo-quantized scenarios, tensors in x must have the same number of dimensions, supporting 2–6 dimensions, and tensors in y must have the same number of dimensions as x. Tensors in weight must be 2D with shape ($n_i$, $k_i$) or ($k_i$, $n_i$). Tensors in bias must be 1D with shape ($n_i$).<br>3) groupListOptional must be passed as null.<br>4) Weight transposition is supported, but whether each tensor in the weight tensorList is transposed must be consistent.<br>5) x does not support transposition.<br>6) Only ND in, ND out is supported. |
        | 0 | single-single-single |1) Only splitItem 2/3 is supported.<br>2) Tensors in weight must be 3D with shape (g, N, K) or (g, K, N). Tensors in x and y must be 2D with shapes (M, K) and (M, N) respectively. Tensors in bias must be 2D with shape (g, N).<br>3) groupListOptional must be passed, and the last value must not exceed the first dimension of the tensor in x.<br>4) The first dimension of groupListOptional supports a maximum of 1024, that is, a maximum of 1024 groups.<br>5) Weight transposition is supported.<br>6) x does not support transposition.<br>7) Only ND in, ND out is supported. |
        | 0 | single-multiple-single |1) Only splitItem 2/3 is supported.<br>2) groupListOptional must be passed, the last value must equal the first dimension of the tensor in x, and the length must not exceed 1024.<br>3) Tensors in x and y must be 2D with shapes (M, K) and (M, N) respectively. Tensors in weight must be 2D with shape (N, K) or (K, N). Tensors in bias must be 1D with shape (N).<br>4) The N axis of each tensor in weight must be equal.<br>5) Weight transposition is supported, but whether each tensor in the weight tensorList is transposed must be consistent.<br>6) x does not support transposition.<br>7) Only ND in, ND out is supported.<br>8) Only non-quantized scenarios are supported.|
        | 0 | multiple-multiple-single |1) Only splitItem 2 is supported.<br>2) Tensors in x and y must be 2D with shapes (M, K) and (M, N) respectively. Tensors in weight must be 2D with shape (N, K) or (K, N). Tensors in bias must be 1D with shape (N).<br>3) The N axis of each tensor in weight must be equal.<br>4) If groupListOptional is passed, the differences of groupListOptional must correspond one-to-one with the first dimension of the tensor in x, and the length must not exceed 1024.<br>5) Weight transposition is supported, but whether each tensor in the weight tensorList is transposed must be consistent.<br>6) x does not support transposition.<br>7) Only ND in, ND out is supported.<br>8) Only non-quantized scenarios are supported.|
        | 2 | single-single-single |1) Only splitItem 2/3 is supported.<br>2) Tensors in x and weight must be 2-dimensional, with shapes (K, M) and (K, N) respectively; tensors in y must be 3-dimensional, with shape (g, M, N).<br>3) groupListOptional must be passed, and its last value must not be greater than the first dimension of the tensor in x.<br>4) The first dimension of groupListOptional supports a maximum of 1024, that is, up to 1024 groups are supported.<br>5) x must be transposed and weight cannot be transposed.<br>6) Only ND input and ND output are supported.<br>7) bias is not supported.<br>8) Only non-quantization is supported.|
        | 2 | Single Multi Multi |1) Only splitItem 0/1 is supported.<br>2) Tensors in x and weight must be 2-dimensional, with shapes (K, M) and (K, N) respectively; tensors in y must be 2-dimensional, with shape (M, N).<br>3) groupListOptional can be passed as empty. If groupListOptional is passed, its last value must not be greater than the first dimension of the tensor in x.<br>4) The first dimension of groupListOptional supports a maximum of 1024, that is, up to 1024 groups are supported.<br>5) x must be transposed and weight cannot be transposed.<br>6) Only ND input and ND output are supported.<br>7) bias is not supported.<br>8) Only non-quantization is supported.|

  </details>

<!-- end id12 -->

## Example

The sample code for the API call is as follows, for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v3.h"

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
int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the Device.
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

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate Device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy Host-side data to Device-side memory.
    std::vector<T> hostData(size, 0);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create the aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
    int size = shapes.size();
    std::vector<aclTensor*> tensors(size);
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors.data() + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}


int main() {
    // 1. (boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs, which need to be custom-constructed based on the API.
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
    aclTensorList* y = nullptr;
    int64_t splitItem = 2;
    int64_t groupType = 0;

    // Create the x aclTensorList.
    ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weight aclTensorList.
    ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the bias aclTensorList.
    ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y aclTensorList.
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the group_list aclTensor.
    ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnGroupedMatmulV3.
    ret = aclnnGroupedMatmulV3GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, groupType, y, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmulV3.
    ret = aclnnGroupedMatmulV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV3 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify it based on the specific API definition.
    for (int i = 0; i < 1; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %u\n", j, resultData[j]);
        }
    }

    // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(y);

    // 7. Release device resources. Modify it based on the specific API definition.
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
