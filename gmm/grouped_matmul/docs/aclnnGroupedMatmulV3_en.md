# aclnnGroupedMatmulV3

**Note: This API will be deprecated in later versions. Use the latest aclnnGroupedMatmulV5 API.**

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

- Description: Implements grouped matrix multiplication, supporting non-uniform matrix dimension sizes across multiple groups. The basic function is matrix multiplication, for example, $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where $g$ indicates the number of groups and $m_i$, $k_i$, and $n_i$ define the shapes for each group. Both inputs and outputs are of the aclTensorList type, with the following functions:

  - K-axis grouping: $k_i$ varies across groups, while $m_i$ and $n_i$ remain the same for each group. In this case, $x_i$ and $weight_i$ can be concatenated along the K-axis.
  - M-axis grouping: $k_i$ remains the same for each group. In this case, $weight_i$ and $y_i$ can be concatenated along the N-axis.

  Compared with [GroupedMatmul](./aclnnGroupedMatmul.md), this API provides the following new features:
    - <term>Atlas A2 training products/Atlas A2 inference products</term>:
      - Supports weight transposition in non-quantization scenarios. Transposition refers to the case where the shape is [M, K], the stride is [1, M], and the data layout is [K, M].
      - Supports M-axis and K-axis grouping, represented by `groupType`.
      - x, weight, and y are all single tensors. In non-quantization scenarios, both x and weight can be of the float32 type.
      - Supports weight transposition and single-tensor weights in quantization and fake-quantization scenarios.
      - For the features supported by [aclnnGroupedMatmulGetWorkspaceSize](./aclnnGroupedMatmul.md), this API does not support scenarios where `x` is single-tensor while `weight` and `y` are multi-tensor.
    - Ascend 950PR/Ascend 950DT:
      - In the fake-quantization scenario, weight transpose is supported, and x, weight, and y can be single tensors.
      - For the features supported by [aclnnGroupedMatmulGetWorkspaceSize](./aclnnGroupedMatmul.md), this API does not support scenarios where `x` is single-tensor while `weight` and `y` are multi-tensor.

**Notes**:

- "Single-tensor" means that tensors of all groups in a tensor list are concatenated into one tensor along the axis specified `groupType`.
- Tensor transpose: If the tensor shape is [M, K], the stride is [1, M], and the data layout is [K, M], then the tensor is a non-contiguous tensor.

- Formula:
  - **Non-quantization scenario:**

    $$
     y_i=x_i\times weight_i + bias_i
    $$

  - **Quantization scenario:**

    $$
     y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
    $$

  - **Dequantization scenario:**

    $$
     y_i=(x_i\times weight_i + bias_i) * scale_i
    $$

  - **Fake-quantization scenario:**

    $$
     y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulV3GetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnGroupedMatmulV3` is called to perform computation.

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

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1483px"><colgroup>
  <col style="width: 210px">
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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensorList)</td>
      <td>Input</td>
      <td>x in the formula.</td>
      <td>The length of tensorList can be [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight (aclTensorList)</td>
      <td>Input</td>
      <td>weight in the formula.</td>
      <td>The length of tensorList can be [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Bias in the formula.</td>
      <td>Same length as weight.</td>
      <td>INT32, BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Indicates the scaling factor in the quantization parameters.</td>
      <td>Generally, the length is the same as the weight length.</td>
      <td>UINT64, INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Indicates the offset in the quantization parameters.</td>
      <td>Same length as weight.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Indicates the scale in the fake-quantization parameters.</td>
      <td>Same length as weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Indicates the offset in the fake-quantization parameters.</td>
      <td>Same length as weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Matmul size distribution along the grouping axis for inputs and outputs.</td>
      <td>When the length of the TensorList in the output is 1, the last value in groupListOptional restricts the valid part of the output data. The part that is not specified in groupListOptional will not be updated.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>splitItem (int64_t)</td>
      <td>Input</td>
      <td>Integer, indicating whether to split the output tensor.</td>
      <td>
      0 or 1 indicates that the output is a multi-tensor.
      2 or 3 indicates that the output is a single tensor.
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType (int64_t)</td>
      <td>Input</td>
      <td>Integer, indicating the axis to be grouped.</td>
      <td>For example, if the matrix multiplication is C[m,n]=A[m,k]xB[k,n], groupType is set to -1 (no grouping), 0 (grouping by m axis), or 2 (grouping by k axis).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensorList)</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>The length of tensorList can be [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, INT8, FLOAT32, INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
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
  </tbody></table>

  - <term>Atlas A2 training products/Atlas A2 inference products</term>:
    - x supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
    - weight supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
    - biasOptional supports FLOAT16, FLOAT32, and INT32.
    - y supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
    - The input parameters x and weight, and the output parameter y support a maximum of 128 tensors.
  - Ascend 950PR/Ascend 950DT:
    - x supports FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - weight supports FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - biasOptional supports FLOAT16, BFLOAT16, FLOAT32, and INT32.
    - y supports FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - offsetOptional is not supported.
    - groupType supports grouping and non-grouping on the m axis. Only non-quantization supports grouping on the k axis.
    - In the non-quantization scenario, the input parameters x and weight, and the output parameter y support a maximum of 1024 tensors. In the fake-quantization scenario, the input parameters x and weight, and the output parameter y support a maximum of 128 tensors. In the quantization scenario, the input parameters x and weight, and the output parameter y support a maximum of 1 tensor.

- **Return**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1055px"><colgroup>
  <col style="width: 242px">
  <col style="width: 78px">
  <col style="width: 735px">
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
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, or out is not supported.</td>
    </tr>
    <tr>
      <td>The length of weight is not supported.</td>
    </tr>
    <tr>
      <td>If bias is not empty, the length of bias is not equal to that of weight.</td>
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
  </tbody>
  </table>

## aclnnGroupedMatmulV3

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 834px"><colgroup>
    <col style="width: 118px">
    <col style="width: 87px">
    <col style="width: 629px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter Description</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Memory address of the workspace to be allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API of aclnnGroupedMatmulV3GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor, containing the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
      </tr>
    </tbody>
    </table>

- **Return**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnGroupedMatmulV3` defaults to deterministic implementation.
- If `groupListOptional` is passed, it must be a non-negative ascending array, and its length cannot be 1.
- The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).
- <term>Atlas A2 training products/Atlas A2 inference products</term>:
  - The following input types are supported in non-quantization scenarios:
    - `x`: FLOAT16; `weight`: FLOAT16; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: FLOAT16
    - `x`: BFLOAT16; `weight`: BFLOAT16; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: BFLOAT16
    - `x`: FLOAT32; `weight`: FLOAT32; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: FLOAT32 (supported only when `x`, `weight`, and `y` are all single-tensor)
  - The following input types are supported in quantization scenarios:

    - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: UINT64; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `y`: INT8
  - The following input types are supported in fake-quantization scenarios:
    - `x`: FLOAT16; `weight`: INT8; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: FLOAT16; `antiquantOffsetOptional`: FLOAT16; `y`: FLOAT16
    - `x`: BFLOAT16; `weight`: INT8; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: BFLOAT16; `antiquantOffsetOptional`: BFLOAT16; `y`: BFLOAT16
  - Supported scenarios for different `groupType` values:
    - In A16W8 and A16W4 scenarios, `groupType` can be -1 or 0.
    - In A8W8, A8W4, and A4W4 scenarios, `x` can only be single -tensor when `groupType` is `0`.
    - `x`, `weight`, and `y` are of type aclTensorList, which is an array of aclTensor objects. In the following table, "S" indicates an aclTensorList consisting of one aclTensor, and "M" indicates an aclTensorList consisting of multiple aclTensors. For example, "SMS" indicates single-tensor `x`, multi-tensor `weight`, and single-tensor `y`.

      | groupType | Tensor Count in `x`| Tensor Count in `weight`| Tensor Count in `y`| splitItem| groupListOptional | Transposition| Other Constraints|
      |:---------:|:-------:|:-------:|:-------:|:--------:|:------------------|:--------| :-------|
      | -1 | M|M|M| 0/1 | `groupListOptional` must be passed as null.| (1) `x` cannot be transposed.<br> (2) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.| The tensors in `x` must have the same dimensionality, which can be 2D. The tensors in `weight` must be 2D. The dimensionality of tensors in `y` must match that of `x`.|
      | 0 | S|S|S| 2/3 | (1) `groupListOptional` must be passed.<br> (2) The maximum size of the first dimension of groupListOptional is 1024. That is, a maximum of 1024 groups are supported.|(1) `x` cannot be transposed.<br> (2) `weight` can be transposed, except in A8W4 and A4W4 scenarios.|The tensor in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.|
      | 0 | S|M|S| 2/3 | (1) `groupListOptional` must be passed.<br> (2) The maximum size of the first dimension of groupListOptional is 128. That is, a maximum of 128 groups are supported.|(1) `x` cannot be transposed.<br> (2) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.|(1) Tensors in `x`, `weight`, and `y` must be 2D.<br> (2) The N-axis of each tensor in `weight` must be the same.|
      | 0 | M|M|S| 2/3 | (1) `groupListOptional` is optional.<br> (2) The maximum size of the first dimension of groupListOptional is 128. That is, a maximum of 128 groups are supported.|(1) `x` cannot be transposed.<br> (2) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.|(1) Tensors in `x`, `weight`, and `y` must be 2D.<br> (2) The N-axis of each tensor in `weight` must be the same.|
      | 2 | S|S|S| 2/3 | (1) `groupListOptional` must be passed.<br> (2) The first dimension of groupListOptional supports a maximum of 1024 groups.| (1) `x` must be transposed.<br> (2) `weight` cannot be transposed.|(1) The tensors in `x` and `weight` must be 2D, and the tensor in `y` must be 3D.<br> (2) `bias` must be passed as null.|
      | 2 | S|M|M| 0/1 | `groupListOptional` must be passed as null.| (1) `x` must be transposed.<br> (2) `weight` cannot be transposed.| (1) Tensors in `x`, `weight`, and `y` must be 2D.<br> (2) The maximum length of `weight` is 128, meaning a maximum of 128 groups are supported.<br> (3) The sum of the first dimensions of all tensors in the original `weight` shape should not exceed the first dimension of `x`.<br> (4) `bias` must be passed as null.|

  - The size of the last dimension for each tensor in `x` and `weight` should be less than 65536. The last dimension of $x_i$ refers to the K-axis when `x` is not transposed or the M-axis when `x` is transposed. The last dimension of $weight_i$ refers to the N-axis when `weight` is not transposed or the K-axis when `weight` is transposed.

- Ascend 950PR/Ascend 950DT:

  <details>
    <summary>Constraints on non-quantization scenarios</summary>
      <a id="constraints-for-non-quantization-scenario"></a>

  - The following data types are supported in non-quantization scenarios:
    - The following input parameters are null: scaleOptional, offsetOptional, antiquantScaleOptional, and antiquantOffsetOptional.
    - The data type combinations supported by the parameters that are not null must meet the requirements in the following table.

        |groupType| x       | weight  | biasOptional | y     |
        |:-------:|:-------:|:-------:| :------      |:------ |
        |-1/0/2   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
        |-1/0/2   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|
        |-1/0/2   |FLOAT32     |FLOAT32     |FLOAT32/null    | FLOAT32|

  </details>

  <details>
  <summary>Constraints on fake-quantization scenarios</summary>
  <a id="constraints-on-fake-quantization-scenario"></a>

  - The fake-quantization scenario supports the following data types:
    - The following input parameters are empty: scaleOptional and offsetOptional.
    - The combinations of data types supported by non-empty parameters must meet the requirements listed in the following table.

      |groupType| x       | weight  | antiquantScaleOptional |   antiquantOffsetOptional | biasOptional | y     |
      |:-------:|:-------:|:-------:| :------  | :------ | :------     |:------ |
      |-1/0   |BFLOAT16     |INT8   |BFLOAT16  |BFLOAT16/null  |BFLOAT16/FLOAT32/null   | BFLOAT16|
      |-1/0   |FLOAT16     |INT8  |FLOAT16 |FLOAT16/null   |FLOAT16/null| FLOAT16|

    - The combinations of antiquantScaleOptional, non-empty biasOptional, and antiquantOffsetOptional must meet the requirements listed in the following table. Here, g indicates the number of matmul groups.

      |groupType| Application Scenario| Shape Restriction|
      |:---------:|:---------:| :------ |
      |-1|Weight (multi-tensor)|Each tensor is 1-dimensional, and the shape is ($n_i$). It is not allowed that some tensors in a tensor list have the shape ($n_i$) and some tensors are empty.|
      |0|Weight (single-tensor)|Each tensor is 2-dimensional, and the shape is (g, N).|

    - Only the SSS and MMM scenarios are supported.
  </details>

  <details>
  <summary>Constraints on the quantization scenario</summary>

  - The following input parameters are empty: offsetOptional, antiquantScaleOptional, and antiquantOffsetOptional.
  - Only the single-single-single scenario is supported.

  - The combinations of data types and dimensions supported by non-empty parameters must meet the requirements listed in the following table. Here, g indicates the number of matmul groups.

      |groupType| x dtype     | x shape | weight dtype |weight shape| biasOptional dtype |biasOptional shape| scaleOptional dtpye |scaleOptional shape| out dtpye    |out shape|
      |:-------:|:-------:|:-------:|:-------:|:-------:| :------      | :------      |:-------       | :------ |:-------       | :------ |
      |0|INT8     |(M,K)|INT8     |(g,K,N)/(g,N,K)|INT32/null    | (g,N)|UINT64/INT64  |(g,N)/null|INT8|(M,N)|

  </details>

  <details>
  <summary>Constraints on the groupType scenario</summary>
  <a id="constraints-on-groupType-scenario"></a>

  - Supported scenarios for different `groupType` values:
    - "S" stands for single-tensor, and "M" stands for multi-tensor, expressed in the sequence of `x`, `weight`, `y`. For example, "SMS" indicates single-tensor `x`, multi-tensor `weight`, and single-tensor `y`.

        | groupType | Supported Scenario| Scenario Restrictions|
        |:---------:|:-------:| :-------|
        | -1 | MMM|(1) `splitItem` can only be set to `0` or `1`.<br>2) For non-quantized x, the out tensor must be 2-dimensional, with shapes of ($m_i$, $k_i$) and ($m_i$, $n_i$). In the fake-quantization scenario, the tensor in x must have the same dimension (2-6 dimensions). The tensor in y must have the same dimension as x. The tensor in weight must have two dimensions and the shape must be ($n_i$, $k_i$) or ($k_i$, $n_i$). The tensor in bias must have one dimension and the shape must be ($n_i$).<br>(3) `groupListOptional` must be passed as null.<br>(4) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(5) `x` cannot be transposed.<br>(6) Only ND input and ND output are supported.|
        | 0 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensors in weight must be 3D, with shapes of (g, N, K) or (g, K, N). The tensors in x and y must be 2D, with shapes of (M, K) and (M, N), respectively. The tensors in bias must be 2D, with shape of (g, N).<br>(3) groupListOptional must be passed, and the last value cannot be greater than the first dimension of the tensor in x.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) `weight` can be transposed.<br>(6) `x` cannot be transposed.<br>(7) Only ND input and ND output are supported.|
        | 0 | SMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) groupListOptional must be passed, and the last value must be the same as the first dimension of the tensor in x. The maximum length is 1024.<br>(3) The tensors in x and y must be 2D, with shapes of (M, K) and (M, N), respectively. The tensors in weight must be 2D, with shapes of (N, K) or (K, N). The tensors in bias must be 1D, with shape of (N).<br>(4) The N-axis of each tensor in `weight` must be the same.<br>(5) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(6) `x` cannot be transposed.<br>(7) Only ND input and ND output are supported.<br>(8) Only non-quantization is supported.|
        | 0 | MMS|(1) `splitItem` can only be set to `2`.<br>(2) The tensors in x and y must be two-dimensional, with shapes (M, K) and (M, N), respectively. The tensor in weight must be two-dimensional, with shape (N, K) or (K, N). The tensor in bias must be one-dimensional, with shape (N).<br>(3) The N-axis of each tensor in `weight` must be the same.<br>(4) If groupListOptional is passed, the difference between groupListOptional and the first dimension of the tensor in x must be in one-to-one correspondence, and the maximum length is 1024.<br>(5) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(6) `x` cannot be transposed.<br>(7) Only ND input and ND output are supported.<br>(8) Only non-quantization is supported.|
        | 2 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensors in x and weight must be two-dimensional, with shapes (K, M) and (K, N), respectively. The tensor in y must be three-dimensional, with shape (g, M, N).<br>(3) groupListOptional must be passed, and the last value cannot be greater than the first dimension of the tensor in x.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) x must be transposed and weight cannot be transposed.<br>(6) Only ND input and ND output are supported.<br>(7) Only non-quantization is supported.|

  </details>

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
    std::vector<aclTensor*> tensors(size);
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors.data() + i);
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
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a group_list aclTensor.
    ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnGroupedMatmulV3.
    ret = aclnnGroupedMatmulV3GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, groupType, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmulV3.
    ret = aclnnGroupedMatmulV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

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
    aclDestroyTensorList(y);

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
