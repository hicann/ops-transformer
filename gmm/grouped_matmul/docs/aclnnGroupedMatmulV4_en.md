# aclnnGroupedMatmulV4

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Function

- Description: Implements grouped matrix multiplication, supporting non-uniform matrix dimension sizes across multiple groups. The basic function is matrix multiplication, for example, $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where $g$ indicates the number of groups, and $m_i$, $k_i$, and $n_i$ indicate the corresponding dimension sizes. Both input and output parameters are of the aclTensorList type, with the following functions:

    - K-axis grouping: $k_i$ varies across groups, while $m_i$ and $n_i$ remain the same for each group. In this case, $x_i$ and $weight_i$ can be concatenated along the K-axis.
    - M-axis grouping: $k_i$ remains the same for each group. In this case, $weight_i$ and $y_i$ can be concatenated along the N-axis.

    Compared with [GroupedMatmulV3](./aclnnGroupedMatmulV3_en.md), this API has the following new features:
    - The values in `groupListOptional` can be the sizes of groups along the grouping axis.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
      - Static quantization (per-tensor and per-channel), BFLOAT16 and FLOAT16 outputs, with or without activation (For details, refer to [quantization methods](../../../docs/en/context/quant_mode_introduction.md). Same below.)
      - Dynamic quantization (per-tensor and per-channel), BFLOAT16 and FLOAT16 outputs, with or without activation
      - Fake-quantization with INT4 input `weight` without activation in per-channel and per-group modes

    **Notes**:
    - "Single-tensor" means that tensors of all groups in a tensor list are concatenated into one tensor along the axis specified `groupType`.
    - Tensor transpose: If the tensor shape is [M, K], the stride is [1, M], and the data layout is [K, M], then the tensor is a non-contiguous tensor.

- Formula:

    - **Non-quantization scenario:**

    $$
     y_i=x_i\times weight_i + bias_i
    $$

    - **Quantization scenario (static quantization, T-C && T-T, without perTokenScaleOptional):**

      $$
        y_i=(x_i\times weight_i) * scale_i + offset_i
      $$

      - `x` in INT8 and `bias` in INT32

      $$
        y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
      $$

      - `x` in INT8 and `bias` in BFLOAT16/FLOAT16/FLOAT32, without offset

      $$
        y_i=(x_i\times weight_i) * scale_i + bias_i
      $$

    - **Quantization scenario (dynamic quantization, T-T && T-C && K-T && K-C):**

      $$
      y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i
      $$

      - `x` in INT8 and `bias` in INT32

      $$
        y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i
      $$

      - `x` in INT8 and `bias` in BFLOAT16/FLOAT16/FLOAT32

      $$
        y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i  + bias_i
      $$

    - **Quantization scenario (dynamic quantization, MX && G-B):**

      $$
      y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (xSlice_i * weightSlice_i)) * (per\_token\_scale_i[m/gsM, j] * scale_i[j, n/gsN])) + bias_i[n]
      $$
      
      $gsM$, $gsN$, and $gsK$ represent the quantization block sizes for the M, N, and K axes respectively. $xSlice_i$ denotes a vector of length $gsK$ from the $m$-th row of $x_i$, and $weightSlice_i$ denotes a vector of length $gsK$ from the $n$-th column of $weight_i$. Both tensors are sliced along the K-axis starting from $j \times gsK$, where $j \in [0, kLoops)$ and $kLoops = \lceil K_i / gsK \rceil$. Additionally, a final slice length less than $gsK$ is supported.

    - **Fake-quantization scenario:**

    $$
     y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulV4GetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnGroupedMatmulV4` is called to perform computation.

```Cpp
aclnnStatus aclnnGroupedMatmulV4GetWorkspaceSize(
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
  aclTensorList       *out, 
  aclTensorList       *activationFeatureOutOptional, 
  aclTensorList       *dynQuantScaleOutOptional, 
  uint64_t            *workspaceSize, 
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnGroupedMatmulV4(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnGroupedMatmulV4GetWorkspaceSize

- **Parameters**
  - x (aclTensorList *, computation input): aclTensorList on the device, $x$ in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND, and the maximum length is 128.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, BFLOAT16, FLOAT32, INT8, or INT4.
      - <term>Atlas inference products</term>: The data type can be FLOAT16.
  - weight (aclTensorList *, computation input): aclTensorList on the device, $weight$ in the formula. The maximum length is 128.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, BFLOAT16, FLOAT32, INT8, or INT4, and the [data format](../../../docs/en/context/data_format.md) can be ND or FRACTAL_NZ.
      - <term>Atlas inference products</term>: The data type can be FLOAT16, and the [data format](../../../docs/en/context/data_format.md) can only be FRACTAL_NZ.
  - biasOptional (aclTensorList *, computation input): optional parameter, aclTensorList on the device, $bias$ in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND, and the length is the same as that of `weight`.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, FLOAT32, or INT32.
      - <term>Atlas inference products</term>: The data type can be FLOAT16.
  - scaleOptional (aclTensorList *, computation input): optional parameter, aclTensorList on the device, indicating the scale factor for quantization parameters. The [data format](../../../docs/en/context/data_format.md) can be ND. Generally, the length is the same as that of `weight`. For details about the constraints, see [Constraints](#constraints).
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be UINT64, BFLOAT16, or FLOAT32.
      - <term>Atlas inference products</term>: This parameter is not supported currently and needs to be passed as a null pointer.
  - offsetOptional (aclTensorList *, computation input): optional parameter, aclTensorList on the device, indicating the offset for quantization parameters. The [data format](../../../docs/en/context/data_format.md) can be ND, and the length is the same as that of `weight`.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT32.
      - <term>Atlas inference products</term>: This parameter is not supported currently and needs to be passed as a null pointer.
  - antiquantScaleOptional (aclTensorList *, computation input): optional parameter, aclTensorList on the device, indicating the scale factor for fake-quantization parameters. The [data format](../../../docs/en/context/data_format.md) can be ND, and the length is the same as that of `weight`. For details about the constraints, see [Constraints](#constraints).
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16 or BFLOAT16.
      - <term>Atlas inference products</term>: This parameter is not supported currently and needs to be passed as a null pointer.
  - antiquantOffsetOptional (aclTensorList *, computation input): optional parameter, aclTensorList on the device, indicating the offset for fake-quantization parameters. The [data format](../../../docs/en/context/data_format.md) can be ND, and the length is the same as that of `weight`. For details about the constraints, see [Constraints](#constraints).
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16 or BFLOAT16.
      - <term>Atlas inference products</term>: This parameter is not supported currently and needs to be passed as a null pointer.
  - perTokenScaleOptional (aclTensorList *, computation input): optional parameter, aclTensorList on the device, indicating the scale factor (introduced by `x` quantization) for quantization parameters. The [data format](../../../docs/en/context/data_format.md) can be ND. This parameter only supports scenarios where `x`, `weight`, and `out` are all single-tensor (with a TensorList length of 1). For details about the constraints, see [Constraints](#constraints).
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT32.
      - <term>Atlas inference products</term>: This parameter is not supported and needs to be passed as a null pointer.
  - groupListOptional (aclTensor *, computation input): optional parameter, aclTensor type on the device, indicating the Matmul size distribution along the grouping axis for inputs and outputs. The data type can be INT64, and the [data format](../../../docs/en/context/data_format.md) can be ND. Note that when the length of the TensorList in the output is 1, the last value in `groupListOptional` constrains the valid portion of the output data. Any portion not specified in `groupListOptional` will not be updated.
  - activationInputOptional (aclTensorList *, computation input): optional parameter, aclTensorList on the device, indicating the backward input of the activation function. Currently, only `nullptr` is supported.
  - activationQuantScaleOptional (aclTensorList *, computation input): optional parameter, aclTensorList on the device. Currently, only `nullptr` is supported.
  - activationQuantOffsetOptional (aclTensorList *, computation input): optional parameter, aclTensorList on the device. Currently, only `nullptr` is supported.
  - splitItem (int64\_t, computation input): integer type, indicating whether tensor splitting is required for the output. `0` or `1` indicates multi-tensor, and `2` or `3` indicates single-tensor.
  - groupType (int64\_t, computation input): integer type, indicating the axis to be grouped. For example, if the matrix multiplication is `C[m,n]=A[m,k]xB[k,n]`, `groupType` has the following options: `-1` means no axis grouping, `0` indicates M-axis grouping, `1` indicates N-axis grouping, and `2` indicates K-axis grouping.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Currently, N-axis grouping is not supported.
      - <term>Atlas inference products</term>: Currently, only M-axis grouping is supported.
  - groupListType (int64\_t, computation input): integer type. The value can be:
        *0: Values in `groupListOptional` are the cumulative sum (cumsum) results of the grouping axis sizes.
        *1: Values in `groupListOptional` are the sizes of groups along the grouping axis.
        *2: The shape of `groupListOptional` is [e, 2], where `e` indicates the group size. The data layout is `[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]`, where `groupSize` is the size of each group along the grouping axis.
        * <term>Atlas inference products</term>: The value `2` is not supported.
        * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value `2` is supported only when the input type of `x` and `weight` is INT8 and `groupType` is set to `0` (M-axis grouping).
  - actType (int64\_t, computation input): integer type, indicating the activation function type.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The value ranges from 0 to 5, corresponding to the following enumerated values:
          * 0: GMMActType::GMM_ACT_TYPE_NONE
          * 1: GMMActType::GMM_ACT_TYPE_RELU
          * 2: GMMActType::GMM_ACT_TYPE_GELU_TANH
          * 3: GMMActType::GMM_ACT_TYPE_GELU_ERR_FUNC (not supported)
          * 4: GMMActType::GMM_ACT_TYPE_FAST_GELU
          * 5: GMMActType::GMM_ACT_TYPE_SILU
      - <term>Atlas inference products</term>: Currently, only `0` is supported, indicating `GMMActType::GMM_ACT_TYPE_NONE`.
  - out (aclTensorList *, computation output): aclTensorList on the device, $y$ in the formula. The [data format](../../../docs/en/context/data_format.md) can be ND, and the maximum length is 128.
      - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be FLOAT16, BFLOAT16, INT8, FLOAT32, or INT32.
      - <term>Atlas inference products</term>: The data type can be FLOAT16.
  - activationFeatureOutOptional (aclTensorList *, computation output): aclTensorList on the device, the input data of the activation function. Currently, only `nullptr` is supported.
  - dynQuantScaleOutOptional (aclTensorList *, computation output): aclTensorList on the device. Currently, only `nullptr` is supported.
  - workspaceSize (uint64\_t *, output): size of the workspace to be allocated on the device.
  - executor (aclOpExecutor **, output): operator executor, containing the operator computation process.

- **Return**

  `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 283px">
  <col style="width: 126px">
  <col style="width: 745px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="4">161001</td>
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
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
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
  </tbody>
  </table>

## aclnnGroupedMatmulV4

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 180px">
  <col style="width: 130px">
  <col style="width: 839px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnGroupedMatmulV4GetWorkspaceSize.</td>
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

    `aclnnStatus` status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - Deterministic computation:
    - `aclnnGroupedMatmulV4` defaults to deterministic implementation.
  - If `groupListOptional` is passed: when `groupListType` is `0`, `groupListOptional` must be a non-negative, monotonically non-decreasing sequence; when `groupListType` is `1`, `groupListOptional` must be a non-negative sequence; when `groupListType` is `2`, the second column of `groupListOptional` must be a non-negative sequence and its length cannot be 1.
  - The size of each dimension for every tensor in `x` and `weight`, after 32-byte alignment, should be less than the maximum value of INT32 (2147483647).
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
    - The following input types are supported in non-quantization scenarios:
      - `x`: FLOAT16; `weight`: FLOAT16; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: FLOAT16
      - `x`: BFLOAT16; `weight`: BFLOAT16; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: BFLOAT16
      - `x`: FLOAT32; `weight`: FLOAT32; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: FLOAT32 (supported only when `x`, `weight`, and `y` are all single-tensor)
    - The following input types are supported in quantization scenarios:
      - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: UINT64; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: INT8
      - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: BFLOAT16; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: BFLOAT16
      - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: FLOAT32; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: FLOAT16
      - `x`: INT8; `weight`: INT8; `biasOptional`: INT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: INT32
      - `x`: INT4; `weight`: INT4; `biasOptional`: null; `scaleOptional`: UINT64; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: FLOAT16
      - `x`: INT4; `weight`: INT4; `biasOptional`: null; `scaleOptional`: UINT64; `offsetOptional`: null; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: null or FLOAT32; `activationInputOptional`: null; `out`: BFLOAT16
    - The following input types are supported in fake-quantization scenarios:
      - `x`: FLOAT16; `weight`: INT8 or INT4; `biasOptional`: FLOAT16; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: FLOAT16; `antiquantOffsetOptional`: FLOAT16; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: FLOAT16
      - The shapes of the fake-quantization parameters `antiquantScaleOptional` and `antiquantOffsetOptional` must meet the following requirements ($g$ indicates the number of Matmul groups, $G$ indicates the number of quantization groups, and $G_i$ indicates the number of quantization groups of the *i*-th tensor).

          | Application Scenario| Sub-scenario| Shape Restriction|
          |:---------:|:-------:| :-------|
          | Fake-quantization (per-channel)| Single-tensor `weight`| $[g, n]$|
          | Fake-quantization (per-channel)| Multi-tensor `weight`| $[n_i]$|
          | Fake-quantization (per-group)| Single-tensor `weight`| $[g, G, n]$|
          | Fake-quantization (per-group)| Multi-tensor `weight`| $[G_i, n_i]$|

      - `x`: BFLOAT16; `weight`: INT8 or INT4; `biasOptional`: FLOAT32; `scaleOptional`: null; `offsetOptional`: null; `antiquantScaleOptional`: BFLOAT16; `antiquantOffsetOptional`: BFLOAT16; `perTokenScaleOptional`: null; `activationInputOptional`: null; `out`: BFLOAT16
      - `x`: INT8; `weight`: INT4; `biasOptional`: FLOAT32; `scaleOptional`: UINT64; `antiquantScaleOptional`: null; `antiquantOffsetOptional`: null; `perTokenScaleOptional`: FLOAT32; `activationInputOptional`: null This scenario supports symmetric quantization and asymmetric quantization:
        - Symmetric quantization
          - In this case, the data type of the output `out` is BFLOAT16 or FLOAT16.
          - In this case, `offsetOptional` is null.
          - In this case, only the count mode is supported (the operator does not validate `groupListType`). `k` must be an integer multiple of `quantGroupSize` and `k` ≤ 18432. `quantGroupSize` is the per-group quantization length in the K-axis. Currently, `quantGroupSize=256` is supported.
          - In this case, the scale is the result after per-group and per-channel offline fusion. The shape must be $[e, quantGroupNum, n]$, where $quantGroupNum=k \div quantGroupSize$.
          - The bias is the auxiliary result of offline computation during the computation process. Its value must be $8\times weight \times scale$ and is accumulated in the first dimension. The shape must be $[e, n]$.
          - In this case, `n` must be an integer multiple of 8.
        - Asymmetric quantization
          - In this case, the data type of the output `out` is FLOAT16.
          - In this case, only the count mode is supported (the operator does not validate `groupListType`).
          - In this case, {k, n} must be {7168, 4096} or {2048, 7168}.
          - The scale is the result after per-group and per-channel offline fusion. The shape must be $[e, 1, n]$.
          - In this case, `offsetOptional` is not null. For asymmetric quantization, `offsetOptional` is the auxiliary result of offline computation during the computation process, that is, $antiquantOffset \times scale$. The shape must be $[e, 1, n]$, and the data type must be FLOAT32.
          - The bias is the auxiliary result of offline computation during the computation process. Its value must be $8\times weight \times scale$ and is accumulated in the first dimension. The shape must be $[e, n]$.
          - In this case, `n` must be an integer multiple of 8.
    - In quantization scenarios, if the `weight` type is INT4, the following constraints must be met ($g$ indicates the number of Matmul groups and $G$ indicates the number of groups partitioned along the K-axis for per-group quantization):
      - If the data format of `weight` is ND, `n` must be an integer multiple of 8.
      - Per-channel and per-group quantization are supported. In the per-channel scenario, the shape of the scale must be $[g, n]$. In the per-group scenario, the shape must be $[g, G, n]$.
      - In the per-group scenario, $G$ must be exactly divisible by $k$, and $k/G$ must be an even number.
      - In this scenario, only `groupType=0` (`x`, `weight`, and `y` are all single-tensor), `actType=0`, and `groupListType=0/1` are supported.
      - Weight transposition is not supported in this scenario.
    - In fake-quantization scenarios, if the `weight` type is INT8, only the per-channel mode is supported. If the `weight` type is INT4, the per-channel and per-group modes are supported in symmetric quantization. If the per-group mode is used, the number of quantization groups ($G$ or $G_i$) must be exactly divisible by the corresponding $k_i$. For multi-tensor `weight`, the per-group length is defined as $s_i = k_i / G_i$, and all $s_i(i=1,2,...g)$ values must be the same. Asymmetric quantization supports the per-channel mode.
    - In fake-quantization scenarios, if the `weight` type is INT4, the last dimension of each group of tensors in weight must be an even number. The last dimension of $weight_i$ refers to the N-axis when `weight` is not transposed or the K-axis when `weight` is transposed. In the per-group mode, when `weight` is transposed, the per-group length $s_i$ must be an even number.

    - Supported scenarios for different `groupType` values:
      - In quantization and fake-quantization scenarios, `groupType` can be either `-1` or `0`.
      - "S" stands for single-tensor, and "M" stands for multi-tensor, expressed in the sequence of `x`, `weight`, `y`. For example, "SMS" indicates single-tensor `x`, multi-tensor `weight`, and single-tensor `y`.

        | groupType | Supported Scenario| Scenario Restrictions|
        |:---------:|:-------:| :-------|
        | -1 | MMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) The tensors in `x` must have the same dimensionality, which can be 2D to 6D. The tensors in `weight` must be 2D. The dimensionality of tensors in `y` must match that of `x`.<br>(3) `groupListOptional` must be passed as null.<br>(4) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(5) `x` cannot be transposed.|
        | 0 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensor in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.<br>(3) `groupListOptional` must be passed. When `groupListType` is `0`, the last value must be equal to the first dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be equal to the first dimension of the tensor in `x`. When `groupListType` is `2`, the sum of the values in the second column must be equal to the first dimension of the tensor in `x`.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) `weight` can be transposed.<br>(6) `x` cannot be transposed.|
        | 0 | SMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) `groupListOptional` must be passed. When `groupListType` is `0`, the last value must be equal to the first dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be equal to the first dimension of the tensor in `x` and the maximum length is 128. When `groupListType` is `2`, the sum of the values in the second column must be equal to the first dimension of the tensor in `x` and the maximum length is 128.<br>(3) Tensors in `x`, `weight`, and `y` must be 2D.<br>(4) The N-axis of each tensor in `weight` must be the same.<br>(5) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(6) `x` cannot be transposed.|
        | 0 | MMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) Tensors in `x`, `weight`, and `y` must be 2D.<br>(3) The N-axis of each tensor in `weight` must be the same.<br>(4) If `groupListOptional` is passed: when `groupListType` is `0`, the differences of `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`; when `groupListType` is `1`, the `groupListOptional` values must be in one-to-one mapping with the first dimension of the tensors in `x`, and the maximum length is 128; when `groupListType` is `2`, the values in the second column of `groupListOptional` must be in one-to-one mapping with the first dimension of the tensors in `x`, and the maximum length is 128.<br>(5) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(6) `x` cannot be transposed.|
        | 2 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensors in `x` and `weight` must be 2D, and the tensor in `y` must be 3D.<br>(3) `groupListOptional` must be passed. When `groupListType` is `0`, the last value must be equal to the second dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be equal to the second dimension of the tensor in `x`. When `groupListType` is `2`, the sum of the values in the second column must be equal to the second dimension of the tensor in `x`.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) `x` must be transposed, and `weight` cannot be transposed.<br>(6) `bias` must be passed as null.|
        | 2 | SMM|(1) `splitItem` can only be set to `0` or `1`.<br>(2) Tensors in `x`, `weight`, and `y` must be 2D.<br>(3) `groupListOptional` must be passed as null.<br>(4) The maximum length of `weight` is 128, meaning a maximum of 128 groups are supported.<br>(5) `x` must be transposed, and `weight` cannot be transposed.<br>(6) The sum of the first dimensions of all tensors in the original `weight` shape should not exceed the first dimension of `x`.<br>(7) `bias` must be passed as null.|

    - The size of the last dimension for each tensor in `x` and `weight` should be less than 65536. The last dimension of $x_i$ refers to the K-axis when `x` is not transposed or the M-axis when `x` is transposed. The last dimension of $weight_i$ refers to the N-axis when `weight` is not transposed or the K-axis when `weight` is transposed.
    - Activation function computation is supported only in quantization (per-token) and dequantization scenarios.

  - <term>Atlas inference products</term>:
    - The input and output support only the FLOAT16 type. The N-axis size of the output `y` must be a multiple of 16.

      | groupType | Supported Scenario| Scenario Restrictions|
      |:---------:|:-------:| :------ |
      | 0 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensor in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.<br>(3) `groupListOptional` must be passed. When `groupListType` is `0`, the last value must be equal to the first dimension of the tensor in `x`. When `groupListType` is `1`, the sum of the values must be equal to the first dimension of the tensor in `x`.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) `weight` can be transposed, but `x` cannot.|

## Calling Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v4.h"

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
    // Call the first-phase API of aclnnGroupedMatmulV4.
    ret = aclnnGroupedMatmulV4GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput, activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, out, activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmulV4.
    ret = aclnnGroupedMatmulV4(workspaceAddr, workspaceSize, executor, stream);
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
