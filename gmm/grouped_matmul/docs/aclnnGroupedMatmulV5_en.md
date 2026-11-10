# aclnnGroupedMatmulV5

<style>
abbr {
  border-bottom: 1px dotted #666;
  cursor: help;
}
</style>

---

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

---

## Term

| Term| Brief Description|
|------|----------|
| Non-quantized| The left and right matrices are of the floating-point type and are directly calculated.|
| Full quantization| The left and right matrices are low-precision computations and are restored by scale. For details, see [Quantization Introduction](../../../docs/en/context/quant_mode_introduction.md).|
| fake-quantization| Quantizes the weight of the right matrix, including the perchannel quantization mode.|
| pertensor | T quantization for short. Each tensor shares the same quantization parameter.|
| perchannel | C quantization for short. The quantization object is the right matrix. Each channel uses independent quantization parameters.|
| pertoken | K quantization for short. The quantization object is the left matrix. Each token uses independent quantization parameters.|
| pergroup | Group quantization (G quantization for short). Data is grouped on the reduce axis, and each group uses independent quantization parameters.|
| perblock | B quantization for short. Data is divided into blocks on all axes, and each block uses independent quantization parameters.|
| T-C | Left matrix pertensor, right matrix perchannel|
| T-T | Left matrix pertensor, right matrix pertensor|
| K-C | Left matrix pertoken, right matrix perchannel|
| K-T | Left matrix pertoken, right matrix pertensor|
| K-G | Left matrix pertoken, right matrix pergroup|
| G-B | Left matrix pergroup, right matrix perblock|
| MX | In pergroup-pergroup (G-G) quantization mode, the quantization parameter type is FLOAT8_E8M0, and the group size is 32.|
| ND | Common continuous layout|
| NZ | [FRACTAL_NZ](../../../docs/en/context/data_format.md) format (affinity)|

---

<a id="3-Description"></a>

## Description

### Group By

- Description: Implements grouped matrix multiplication. For example, $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where $g$ indicates the number of groups. Currently, m-axis and k-axis grouping are supported:

  - M-axis grouping: $k_i$ and $n_i$ remain consistent for each group, while $m_i$ can vary.
  - K-axis grouping: $m_i$ and $n_i$ remain consistent for each group, while $k_i$ can vary.

### Formula

<details>
<summary>Non-quantization scenario</summary>;

$$
y_i=x_i\times weight_i + bias_i
$$

</details>

<details>
<summary>Full quantization scenario</summary>

- Quantization scenario (static quantization, without perTokenScaleOptional):
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
- Quantization scenario (dynamic quantization, T-T && T-C && K-T && K-C):
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
- Quantization scenario (dynamic quantization, MX && G-B):
  $$
  y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (xSlice_i * weightSlice_i)) * (per\_token\_scale_i[m/gsM, j] * scale_i[j, n/gsN])) + bias_i[n]
  $$
  $gsM$, $gsN$, and $gsK$ represent the quantization block sizes for the M, N, and K axes respectively. $xSlice_i$ denotes a vector of length $gsK$ from the $m$-th row of $x_i$, and $weightSlice_i$ denotes a vector of length $gsK$ from the $n$-th column of $weight_i$. Both tensors are sliced along the K-axis starting from $j \times gsK$, where $j \in [0, kLoops)$ and $kLoops = \lceil K_i / gsK \rceil$. Additionally, a final slice length less than $gsK$ is supported.

</details>

<details>
<summary><abbr title="Quantizes the weight of the right matrix, including the perchannel quantization mode">Fake Quantization</abbr> scenario</summary>.

- `x` in FLOAT16 or BFLOAT16 and `weight` in INT4 or INT8 (supported only when `x`, `weight`, and `y` are all single-tensor)

  $$
  y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
  $$

- `x` in INT8 and `weight` in INT4 (supported only when `x`, `weight`, and `y` are all single-tensor) (`bias` is a required parameter. It is an auxiliary result of offline computation. It is defined as $bias_i=8\times weight_i * scale_i$, reduced along the K-axis.)

  $$
  y_i=((x_i - 8) \times weight_i * scale_i+bias_i ) * per\_token\_scale_i
  $$

</details>

### Version evolution

- V4 → V5:
<!-- npu="A3" id13 -->
  - <term>Atlas A3 products</term>: Added the optional parameter tuningConfigOptional. The first value in the array indicates the expected number of tokens processed by each expert. Operator tiling is performed based on the expected value.
<!-- end id13 -->

- V1 → V4:

  <!-- npu="950" id7 -->
  - Ascend 950PR&950DT products: Different group axes are supported, which are represented by groupType. Non-quantization supports x/weight transposition. Static quantization (T-C/T-T) BFLOAT16/FLOAT16/FLOAT32 output + bias is supported. Dynamic quantization (K-C/K-T/T-T/T-C/MX/G-B) BFLOAT16/FLOAT16/FLOAT32 output + bias is supported. The fake-quantization weight can be INT4, FLOAT8_E5M2, FLOAT8_E4M3FN or HIFLOAT8. (INT4 supports perchannel and pergroup, and others support only perchannel.)
  <!-- end id7 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>;: Different group axes are supported, which are represented by groupType. For non-quantization, x/weight transpose is supported. The x, weight, and y are single tensor non-quantized FLOAT32 inputs. The fake-quantization weight can be INT4 (perchannel/pergroup mode).
  <!-- end id10 -->

## Prototype

Each operator has two-phase APIs. You must call the aclnnGroupedMatmulV5GetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the aclnnGroupedMatmulV5 API to perform computation.

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

---

## aclnnGroupedMatmulV5GetWorkspaceSize

- Parameters:

  | Parameter| Input/Output| Description| Instructions| Data Type| Data Format| Dimension (shape)| Non-contiguous Tensor|
  |--------|:---:|------|------|----------|:---:|:---|:---:|
  | x (aclTensorList *)| Input| Input $x$ in the formula| TensorList length [1,128] or [1,1024]| FLOAT, FLOAT16, and INT16<span title="Ascend 950PR&950DT products: Not supported"><sup>1</sup></span>; INT8, INT4, BFLOAT16, and FLOAT8_E5M2<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>; FLOAT8_E4M3FN<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>. HIFLOAT8<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>; FLOAT4_E2M1<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>| <abbr title="Common continuous layout ">ND</abbr>| 2~6 | √ |
  | weight (aclTensorList *)| Input| $weight$ in the formula| TensorList length [1,128] or [1,1024]| FLOAT, FLOAT16, and INT16<span title="Ascend 950PR&950DT products: Not supported"><sup>1</sup></span>; INT8, INT4, BFLOAT16, and FLOAT8_E5M2<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>; FLOAT8_E4M3FN<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>. HIFLOAT8<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>; FLOAT4_E2M1<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>| ND/<abbr title="FRACTAL_NZ format (affinity)">NZ</abbr>| 2~3 | √ |
  | biasOptional (aclTensorList *)| Optional input| $bias$ in the formula| The length is the same as the weight.| FLOAT, FLOAT16, INT32, and BFLOAT16<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>"| ND | 1~2 | √ |
  | scaleOptional (aclTensorList *)| Optional input| $scale$ in the formula indicates the scaling factor in the quantization parameter.| Generally, the length is the same as the weight. For details about the constraints, see [Constraints](#7-constraints).| FLOAT, UINT64, BFLOAT16, and FLOAT8_E8M0<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>"; INT64<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>"| ND | 1~4 | √ |
  | offsetOptional (aclTensorList *)| Optional input| $offset$ in the formula indicates the offset in the quantization parameter.| The length is the same as the weight.| FLOAT | ND | 3 | √ |
  | antiquantScaleOptional (aclTensorList *)| Optional input| $antiquant\_scale$ in the formula indicates the scaling factor in the fake-quantization parameter.| The length is the same as that of weight. For details about the constraints, see [Constraints](#7-constraints).| FLOAT16, BFLOAT16| ND | 1~3 | √ |
  | antiquantOffsetOptional (aclTensorList *)| Optional input| $antiquant\_offset$ in the formula indicates the offset in the fake-quantization parameter.| The length is the same as that of weight. For details about the constraints, see [Constraints](#7-constraints).| FLOAT16, BFLOAT16| ND | 1~3 | √ |
  | perTokenScaleOptional (aclTensorList *)| Optional input| $per\_token\_scale$ in the formula indicates the scaling factor introduced by x quantization in the quantization parameter.| Generally, only one dimension is supported and the length is the same as M of x. For details about the constraints, see [Constraints](#7-constraints).| FLOAT and FLOAT8_E8M0<span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span>"| ND | 1~3 | √ |
  | groupListOptional (aclTensor *)| Optional input| Matmul size distribution in the input and output grouping axes| Enter data in different formats based on groupListType.| INT64 | ND | 1~2 | √ |
  | activationInputOptional (aclTensorList *)| Optional input| Reverse input of the activation function. Currently, only nullptr is supported.| - | - | - | - | - |
  | activationQuantScaleOptional (aclTensorList *)| Optional input| Quantization scaling factor of the activation function (reserved parameter). Currently, only nullptr is supported.| - | - | - | - | - |
  | activationQuantOffsetOptional (aclTensorList *)| Optional input| Quantization offset of the activation function (reserved parameter). Currently, only nullptr is supported.| - | - | - | - | - |
  | splitItem (int64_t)| Input| Whether tensor splitting is required for the output| 0/1 = multi-tensor output; 2/3 = single-tensor output. The APIs do not distinguish the internal differences between 0/1 and 2/3.| - | - | - | - |
  | groupType (int64_t)| Input| Axis to be grouped| For example, if a matrix is multiplied by $C[m,n]=A[m,k]\times B[k,n]$, -1 indicates no grouping, 0 indicates M-axis grouping, and 2 indicates K-axis grouping.| - | - | - | - |
  | groupListType (int64_t)| Input| Indicates the grouping mode of the groupList input.| 0 = cumsum; 1 = size of each group; 2 = [group index, group size] pair. For details, see [Constraints](#7-constraints).| - | - | - | - |
  | actType (int64_t)| Input| Activation function type| 0=GMM_ACT_TYPE_NONE / 1=GMM_ACT_TYPE_RELU / 2=GMM_ACT_TYPE_GELU_TANH / 3=GMM_ACT_TYPE_GELU_ERR_FUNC (not supported) / 4=GMM_ACT_TYPE_FAST_GELU / 5=GMM_ACT_TYPE_SILU| - | - | - | - |
  | tuningConfigOptional (aclIntArray *)| Optional input| The first number indicates the expected number of tokens processed by each expert. The second number indicates <abbr title="A8, which indicates that 8-bit quantization is used for activation. W4 indicates that 4-bit quantization is used for weight. Similarly, A8W8 and A4W4">A8W4</abbr> can be used to enable the NZ format in which weight is transposed first. The third number indicates the extra memory space that can be used. For details, see [Constraints](#7-constraints).| This parameter is compatible with earlier versions. If this parameter is not used, do not transfer it (nullptr).| INT64 | - | 3 | - |
  | out (aclTensorList *)| Output| Output $y$ in the formula| TensorList length [1,128] or [1,1024]| FLOAT, FLOAT16, INT32, INT8, BFLOAT16| ND | 2~3 | - |
  | activationFeatureOutOptional (aclTensorList *)| Output| Input data of the activation function. Currently, only nullptr is supported.| - | - | - | - | - |
  | dynQuantScaleOutOptional (aclTensorList *)| Output| Dynamic quantization scaling factor. Currently, only nullptr is supported.| - | - | - | - | - |
  | workspaceSize (uint64_t *)| Output| Returns the size (in bytes) of the workspace to be applied for on the device.| - | - | - | - | - |
  | executor (aclOpExecutor **)| Output| Returns the operator executor, including the operator computation process.| - | - | - | - | - |

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In the preceding table, <span title="Ascend 950PR&950DT products: Not supported"><sup>1</sup></span> indicates that the data type is not supported by the series.
    - The input parameters x and weight do not support the INT16 type.
  <!-- end id8 -->
  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
    - In the preceding table, <span title="Atlas A2 products and Atlas A3 products: Not supported"><sup>2</sup></span> indicates the data type not supported by the series.
    - The FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8 and FLOAT8_E8M0 types are not supported.
    - The input parameter biasOptional does not support BFLOAT16.
    - The input parameter scaleOptional does not support the INT64 type.
  <!-- end id11 -->

- Returns

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following errors may be thrown.

  <!--The following table uses the HTML format to support rowspan cell combination.-->;

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
        <td>The element of the input parameter weight contains a null pointer.</td>
      </tr>
      <tr>
        <td>The element of the input parameter x is a null pointer, and the element of the output parameter out is not a null pointer.</td>
      </tr>
      <tr>
        <td>The element of the input parameter x is not a null pointer, and the element of the output parameter out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="7"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="7"> 161002 </td>
        <td>The data type or data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, or out is not supported.</td>
      </tr>
      <tr>
        <td>The length of weight is out of the supported range.</td>
      </tr>
      <tr>
        <td>If bias is not null, the length of bias is not equal to the length of weight.</td>
      </tr>
      <tr>
        <td>The groupListOptional dimension does not meet the requirements.</td>
      </tr>
      <tr>
        <td>When splitItem is set to 2 or 3, the length of out is not 1.</td>
      </tr>
      <tr>
        <td>When splitItem is set to 0 or 1, the length of out is not equal to the length of weight, and the length of groupListOptional is not equal to the length of weight.</td>
      </tr>
      <tr>
        <td>The element of the input parameter tuningConfigOptional is a negative number or greater than the number of rows m of x.</td>
      </tr>
    </tbody>
  </table>

## aclnnGroupedMatmulV5

- Parameters:

  | Parameter| Input/Output| Description|
  |--------|:---:|------|
  | workspace | Input| Address of the workspace memory allocated on the device|
  | workspaceSize | Input| Size of the workspace applied for on the device, which is obtained by calling aclnnGroupedMatmulV5GetWorkspaceSize in the first segment.|
  | executor | Input| OP executor, including the operator computation process|
  | stream | Input| Stream for executing a task.|

- Returns

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

---

<a id="7-constraints"></a>

## Constraints

### Deterministic computation

`aclnnGroupedMatmulV5` defaults to deterministic implementation.

<!-- npu="950" id9 -->
### Ascend 950PR&950DT products

#### Platform Constraints

- groupType: M-axis grouping (0) and non-grouping (-1) are supported. K-axis grouping is additionally supported for non-quantization and full quantization (2).
- `weight` supports only the ND format through this interface. For NZ weights, use [aclnnGroupedMatmulWeightNz](aclnnGroupedMatmulWeightNz_en.md).
- groupListType: The value can be 0, 1, or 2.
  - groupListType=0: The groupList must be a non-negative monotonically non-decreasing number sequence (cumulative sum). The last value cannot be greater than the first dimension of the tensor in x. For example, M = 256 and E = 4 (the sizes of each group are 64, 0, 128, and 64 in sequence): [64, 64, 192, 256]
  - groupListType=1: The groupList must be a non-negative number sequence (size of each group). The total value cannot be greater than the first dimension of the tensor in x. Example: [64, 0, 128, 64]
  - groupListType=2: supported only in full quantization and groupType=0 scenarios. groupList must be a non-negative number sequence, shape is [E, 2], E indicates the group size, and data layout is [[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]. Non-zero groups are placed in front. The sum of values in the second column must be less than or equal to the first dimension of the tensor in x. Example: [[0, 64], [2, 128], [3, 64], [1, 0]]
- tuningConfigOptional: optional parameter. The first value in the array indicates the expected number of tokens processed by each expert. This value is currently unused. This parameter can be omitted if it is not used.
- actType (0-5) :
  - Only 0 is supported for non-quantization and fake-quantization.
  - In full quantization, the data type of x and weight is INT8, and the data type of out is BFLOAT16/FLOAT16. The value can be 0, 1, 2, 4, or 5 when the static T-C, dynamic K-C, or scale data type is FLOAT32/BFLOAT16. (Note that 3 is not supported.) In other scenarios, only 0 is supported.
- The input parameters x and weight and the output parameter out support a maximum of 1024 tensors in non-quantization scenarios, a maximum of 128 tensors in fake-quantization scenarios, and only one tensor in full quantization scenarios.

#### Scenarios supported by groupType

> x. The inputs x, weight, and y are of the aclTensorList type. "single" = single-element TensorList, "multiple" = multi-element TensorList.

| groupType | x | weight | y | splitItem | groupListOptional | Transposition| Other Constraints|
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | Many| Many| Many| 0/1 | Must be empty.| x is not transposed, and weight can be transposed (unified).| 1) For non-quantized x, the out tensor must be 2-dimensional, with shapes of ($m_i$, $k_i$) and ($m_i$, $n_i$). In the fake-quantization scenario, the tensor in x must have the same dimension (2-6 dimensions). The tensor in y must have the same dimension as x. The tensor in weight must have two dimensions and the shape must be ($n_i$, $k_i$) or ($k_i$, $n_i$). The tensor in bias must have one dimension and the shape must be ($n_i$).<br>2) Only non-quantization and fake-quantization are supported.<br>3) Only ND-in and ND-out are supported.|
| 0 | Single| Single| Single| 2/3 | Mandatory. groupListType=0 Last value ≤ x First dimension; groupListType=1 Sum ≤ x First dimension; groupListType=2 Sum of the second column ≤ x First dimension. A maximum of 1024 groups are supported.| x is not transposed, and weight can be transposed.| (1) The tensor in the weight must be 3D, and the shape must be ($g$, $N$, $K$) or ($g$, $K$, $N$). The tensor in x and out must be 2-dimensional, with shapes of ($M$, $K$) and ($M$, $N$), respectively. The tensor in bias must be 2-dimensional, with shape of ($g$, $N$).<br>2) Only ND-in and ND-out are supported.|
| 0 | Single| Many| Single| 2/3 | Mandatory. groupListType=0 Last value ≤ x First dimension; groupListType=1 Sum ≤ x First dimension; groupListType=2 Sum of the second column ≤ x First dimension; A maximum of 128 groups are supported.| x is not transposed, and weight can be transposed (unified).| (1) The tensor in x and out must be 2-dimensional, with shapes of ($M$, $K$) and ($M$, $N$), respectively. The tensor in weight must be 2-dimensional, with shape of ($N$, $K$) or ($K$, $N$); The tensor in the bias must be 1-dimensional, and the shape must be ($N$).<br>(2) The N axis of each tensor in the weight must be the same.<br>3) Only non-quantization is supported.<br>4) Only ND-in and ND-out are supported.|
| 0 | Many| Many| Single| 2 | Optional. If this parameter is transferred, the following conditions must be met: groupListType=0 Last value ≤ x First dimension; groupListType=1 Sum ≤ x First dimension; A maximum of 128 groups are supported.| x is not transposed, and weight can be transposed (unified).| (1) The tensor in x and out must be 2-dimensional, with shapes of ($M$, $K$) and ($M$, $N$), respectively. The tensor in weight must be 2-dimensional, with shape of ($N$, $K$) or ($K$, $N$); The tensor in the bias must be 1-dimensional, and the shape must be ($N$).<br>(2) The N axis of each tensor in the weight must be the same.<br>3) Only non-quantization is supported.<br>4) Only ND-in and ND-out are supported.|
| 2 | Single| Single| Single| 2/3 | Mandatory; groupListType=0; last value ≤ x first dimension; groupListType=1; sum ≤ x first dimension; a maximum of 1024 groups| x must be transposed, and weight is not transposed.| (1) The tensor in x and weight must be 2-dimensional, with shapes of ($K$, $M$) and ($K$, $N$), respectively. The tensor in out must be 3-dimensional, with shape of ($g$, $M$, $N$).<br>2) Only non-quantization and quantization are supported.<br>3) Bias is not supported.<br>4) Only ND-in and ND-out are supported.|
| 2 | Single| Many| Many| 0/1 | Optional. If this parameter is transferred, the following conditions must be met: groupListType=0 Last value ≤ x First dimension; groupListType=1 Sum ≤ x First dimension; A maximum of 128 groups are supported.| x must be transposed, and weight is not transposed.| (1) The tensor in x and weight must be 2-dimensional, with shapes of ($K$, $M$) and ($K$, $N$), respectively. The tensor in y must be 2-dimensional, with shape of ($M$, $N$).<br>2) Only ND-in and ND-out are supported.<br>3) Bias is not supported.<br>4) Only non-quantization is supported.|

#### Scenario quick reference table

| Use Case| x | weight | out | Constraint|
|--------|:---:|:---:|:---|------|
| Non-quantized| FLOAT32/FLOAT16/BFLOAT16 | FLOAT32/FLOAT16/BFLOAT16 | FLOAT32/FLOAT16/BFLOAT16 | [Constraints in Non-Quantization Scenarios](#ascend950-non-quantization-scenario-constraints)|
| Static quantization| INT8 | INT8 | BFLOAT16/FLOAT16/INT32/INT8 | [Constraints in Static Quantization Scenarios](#ascend950-static-quantization-scenario-constraints)|
| Static quantization| HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | BFLOAT16/FLOAT16/FLOAT32 | [Constraints in Static Quantization Scenarios](#ascend950-static-quantization-scenario-constraints)|
| Dynamic quantization (<abbr title="left-matrix pertensor, right-matrix pertensor">T-T</abbr>/T-C/<abbr title="left-matrix pertoken, right-matrix pertensor">K-T</abbr>/K-C)| INT8 | INT8 | BFLOAT16/FLOAT16 | [Constraints in the dynamic quantization (T-T/T-C/K-T/K-C) scenario](#ascend950-dynamic-quantization-ttck)|
| Dynamic quantization (T-T/T-C/K-T/K-C)| HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN / INT4 | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN / INT4 | BFLOAT16/FLOAT16/FLOAT32 | [Dynamic Quantization (T-T/T-C/K-T/K-C) Scenario Constraints](#ascend950-dynamic-quantization-ttck)|
| Dynamic quantization (<abbr title="pergroup-pergroup (G-G) quantization mode, which is a special case where the quantization parameter type is FLOAT8_E8M0 and the group size is 32.">MX</abbr>)| FLOAT8_E5M2/FLOAT8_E4M3FN / FLOAT4_E2M1 | FLOAT8_E5M2/FLOAT8_E4M3FN / FLOAT4_E2M1 | BFLOAT16/FLOAT16/FLOAT32 | [Dynamic Quantization (MX) Scenario Constraints](#ascend950-dynamic-quantization-mx)|
| Dynamic quantization (<abbr title="left matrix pergroup, right matrix perblock">G-B</abbr>)| HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | BFLOAT16/FLOAT16/FLOAT32 | [Dynamic Quantization (G-B) Scenario Constraints](#ascend950-dynamic-quantization-gb)|
| K-G quantization| INT4 | INT4 | BFLOAT16/FLOAT16 | [K-G Quantization Constraints](#ascend950-kg-quantization-constraints)|
| fake-quantization - S8S4| INT8 | INT4 | BFLOAT16/FLOAT16 | [Constraints in Fake-Quantization Scenarios](#ascend950-fake-quantization-scenario-constraints)|
| <abbr title="Mode for quantizing the right matrix weight, including the per-channel quantization mode.">fake-quantization</abbr>| FLOAT16 / BFLOAT16 | INT8/INT4 | FLOAT16 / BFLOAT16 | [Constraints on fake-quantization scenarios](#ascend950-fake-quantization-scenario-constraints)|
| <abbr title="Mode for quantizing the right matrix weight, including the per-channel quantization mode.">fake-quantization</abbr>| FLOAT16 / BFLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | FLOAT16 / BFLOAT16 | [Constraints on fake-quantization scenarios](#ascend950-fake-quantization-scenario-constraints)|

<a id="ascend950-non-quantization-scenario-constraints"></a>
<summary>Constraints for the non-quantization scenario</summary>
<details>

  - The following input parameters are empty: scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional and activationFeatureOutOptional.
  - The data type combinations supported by non-empty parameters must meet the following requirements:

  | groupType | x | weight | biasOptional | out |
  |:---:|:---:|:---:|:---|:---|
  | -1/0/2 | BFLOAT16 | BFLOAT16 | BFLOAT16/FLOAT32/null | BFLOAT16 |
  | -1/0/2 | FLOAT16 | FLOAT16 | FLOAT16/FLOAT32/null | FLOAT16 |
  | -1/0/2 | FLOAT32 | FLOAT32 | FLOAT32/null | FLOAT32 |

</details>

<a id="ascend950-static-quantization-scenario-constraints"></a>

<details>
<summary>Constraints on the static quantization scenario</summary>

- The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional and activationInputOptional.
- The data type combinations supported by non-empty parameters must meet the following requirements:

| groupType | x | weight | biasOptional | scaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|
| 0 | INT8 | INT8 | INT32/null | UINT64/INT64 | BFLOAT16/FLOAT16/INT8 |
| 0 | INT8 | INT8 | INT32/null | null/UINT64/INT64 | INT32 |
| 0 | INT8 | INT8 | INT32/BFLOAT16/FLOAT32/null | BFLOAT16/FLOAT32 | BFLOAT16 |
| 0 | INT8 | INT8 | INT32/FLOAT16/FLOAT32/null | FLOAT32 | FLOAT16 |
| 0 | HIFLOAT8 | HIFLOAT8 | null | UINT64/INT64 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | HIFLOAT8 | HIFLOAT8 | null | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | UINT64/INT64 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- scaleOptional shape ($g$=Number of groups):

| groupType | Sub-scenario| shape |
|:---:|:---|:---|
| 0/2 | <abbr title="C quantization for short: The quantization object is the right matrix. Each channel uses separate quantization parameters.">perchannel</abbr>| (g, N) |
| 0/2 | <abbr title="Tensor quantization (T-quantization for short). Each tensor shares the same quantization parameters.">pertensor</abbr>| (g, 1) or (g,). When the output is INT8, the pertensor scenario is not supported.|

</details>

<a id="ascend950-dynamic-quantization-ttck"></a>

<details>
<summary>Dynamic quantization (T-T/T-C/K-T/K-C) scene constraint</summary>

- The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional and activationInputOptional.
- The data type combinations supported by non-empty parameters must meet the following requirements:

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|:---|
| 0 | INT8 | INT8 | INT32/BFLOAT16/FLOAT32/null | BFLOAT16/FLOAT32 | FLOAT32 | BFLOAT16 |
| 0 | INT8 | INT8 | INT32/FLOAT16/FLOAT32/null | FLOAT32 | FLOAT32 | FLOAT16 |
| 0 | INT4 | INT4 | null | UINT64 | FLOAT32 | BFLOAT16/FLOAT16 |
| 0/2 | HIFLOAT8 | HIFLOAT8 | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- scaleOptional shape ($g$ = number of groups). It is recommended that (g,) be used for the shape in the pertensor scenario to avoid confusion with the G-B quantization mode.

  | groupType | Sub-scenario| shape |
  |:---:|:---|:---|
  | 0/2 | <abbr title="C quantization for short. The quantization object is the right matrix. Each channel uses separate quantization parameters">perchannel. When the input is INT4, groupType can only be 0.</abbr>| (g, N) |
  | 0/2 | <abbr title="Tensor quantization (T-quantization for short). Each tensor shares the same quantization parameters.">pertensor</abbr>| (g, 1) or (g,)|

- perTokenScaleOptional shape:

| groupType | Sub-scenario| shape |
|:---:|:---|:---|
| 0 | <abbr title="Per-token quantization (K quantization for short): The quantization object is the left matrix. Each token uses separate quantization parameters.">pertoken</abbr>| (M,) |
| 0 | pertensor | (g, 1) or (g,). When the input is INT4 or INT8, the pertensor scenario is not supported.|
| 2 | pertoken | (g, M). When the input is INT4, groupType 2 is not supported.|
| 2 | pertensor | (g, 1) or (g,)|

</details>

<a id="ascend950-dynamic-quantization-mx"></a>

<details>
<summary>Dynamic quantization (MX) scene constraint</summary>

- The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional and activationInputOptional.
- In the formula, the block size is quantized as follows: gsM=gsN=1, gsK=32. MX quantization is a special pergroup quantization.
- The data type combinations supported by non-empty parameters must meet the following requirements:

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |
| 0 | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32/null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |
| 2 | FLOAT4_E2M1 | FLOAT4_E2M1 | null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |

- scaleOptional shape:

| groupType | shape |
|:---:|:---|
| 0 | (g, N, ceil(K/64), 2) (weight transposed) / (g, ceil(K/64), N, 2) (weight not transposed)|
| 2 | (K/64 + g, N, 2), scale_i start address offset is ((K_0 + K_1 +... + K_{i-1}) / 64 + g_i) * N * 2 ($g_i$ indicates the ith group, and the subscript starts from 0). That is, scale_0 offset is 0, scale_1 offset is (K_0 / 64 + 1) * N * 2, scale_2 offset is ((K_0 + K_1) / 64 + 2) * N * 2, and so on.|

- perTokenScaleOptional shape:

| groupType | shape |
|:---:|:---|
| 0 | (M, ceil(K/64), 2) |
| 2 | (K/64 + g, M, 2). The start address offset is the same as that of scale.|

- When x is FLOAT4_E2M1: for groupType=0, K must be an even number and K ≠ 2; for groupType=2, the M axis of x must be an even number, and when K=1, neither M nor N can be 2. When weight is not transposed, N must be an even number.
- When x is FLOAT4_E2M1 and groupType=2, only the ND format is supported for weight.

</details>

<a id="ascend950-dynamic-quantization-gb"></a>

<details>
<summary>Dynamic quantization (G-B) scene constraint</summary>

- The following input parameters are empty: biasOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional and activationInputOptional.
- The block size is calculated as follows: Currently, only gsM=1 and gsN=gsK=128 are supported.
- The data type combinations supported by non-empty parameters must meet the following requirements:

| groupType | x | weight | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|
| 0/2 | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- scaleOptional shape ($g$=Number of groups):

| groupType | shape |
|:---:|:---|
| 0 | (g, ceil(N/gsN), ceil(K/gsK)) (weight transposed) / (g, ceil(K/gsK), ceil(N/gsN)) (weight not transposed)|
| 2 | (K/gsK + g, ceil(N/gsN)), scale_i address offset is ((K_0 + K_1 +... + K_{i-1}) / gsK + g_i) * ceil(N / gsN) ($g_i$ is the ith group, and the subscript starts from 0). That is, the scale_0 offset is 0, the scale_1 offset is (K_0 / gsK + 1) * ceil(N / gsN), the scale_2 offset is ((K_0 + K_1) / gsK + 2) * ceil(N / gsN), and so on.|

- perTokenScaleOptional shape:

| groupType | shape |
|:---:|:---|
| 0 | (M, ceil(K/gsK)) |
| 2 | (K/gsK + g, M), per_token_scale_i address offset is ((K_0 + K_1 +... + K_{i-1}) / gsK + g_i) * M ($g_i$ indicates the ith group, and the subscript starts from 0). That is, the offset of per_token_scale_0 is 0, the offset of per_token_scale_1 is (K_0 / gsK + 1) * M, the offset of per_token_scale_2 is ((K_0 + K_1) / gsK + 2) * M, and so on.|

- Special scenarios of dynamic quantization::
  - In the M-group or K-group dynamic quantization scenario, when N is equal to 1 and the shape of scaleOptional is (g, 1), the weight can be pertensor quantization or perchannel quantization. In this case, the pertensor quantization mode is preferred.
  - In the dynamic quantization scenario where M is grouped, if g = M and the shape of perTokenScaleOptional is (g,), the pertoken quantization mode is selected for x. If g = M, K ≤ 128, and the shape of perTokenScaleOptional is (g, 1), the quantization mode of x is selected based on the quantization mode of weight. If weight is quantized in perchannel or pertensor mode, x is quantized in pertensor mode. If weight is quantized in <abbr title="Quantization B, which divides data into blocks on all axes and uses independent quantization parameters">perblock</abbr> for each block, x is quantized in <abbr title="Quantization G, which groups data on the reduce axis and uses independent quantization parameters">pergroup</abbr> for each group.
  - In the dynamic quantization scenario, if K is less than 128, N is less than or equal to 128, and the shape of scaleOptional is (g, 1), the quantization mode can be non-pergroup quantization or G-B quantization according to the existing quantization mode differentiation rule, in this scenario, G-B quantization is performed.
  - In the dynamic quantization scenario, if M is 1, shape of perTokenScaleOptional is (g, 1), and x can be quantized by pertoken or pertensor, the pertensor quantization mode is preferred.
  - In the dynamic quantization scenario, if K is less than 128, M is 1, shape of perTokenScaleOptional is (g, 1), and N is less than or equal to 128, pergroup quantization is selected for x. If N is greater than 128, select the x quantization mode based on the weight quantization mode. (If the weight is perchannel or pertensor quantization, select pertensor quantization for x. If the weight is perblock quantization, select pergroup quantization for x.)
  - In the dynamic quantization scenario, if K is less than 128, M is not 1, and N is less than or equal to 128, pergroup quantization is selected for x. If N is greater than 128, select the quantization mode of x based on the quantization mode of the weight. (If the weight is perchannel or pertensor quantization, select pertoken quantization for x. If the weight is perblock quantization, select pergroup quantization for x.)

</details>

<a id="ascend950-kg-quantization-constraints"></a>

<details>
<summary>K-G scenario constraints</summary>;

- The weight supports only the non-transposed ND format, and N must be an integer multiple of 8.
- The following input parameters are empty: biasOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, activationInputOptional, activationQuantScaleOptional and activationQuantOffsetOptional.
- The data type combinations supported by non-empty parameters must meet the following requirements:

| groupType | x | weight | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|
| 0 | INT4 | INT4 | UINT64 | FLOAT/null | FLOAT16/BFLOAT16 |

- scaleOptional shape ($g$=Number of groups):

| groupType | Sub-scenario| shape | Constraint|
|:---:|:---|:---|:---|
| 0 | <abbr title="Group quantization (G-quantization for short). Data is grouped along the reduce axis, and each group uses independent quantization parameters.">pergroup</abbr>| [E, G, N] | $G$ must be exactly divided by $K$, and $K/G$ must be an even number.|

- perTokenScaleOptional shape ($g$=Number of groups):

| groupType | Sub-scenario| shape |
|:---:|:---|:---|
| 0 | <abbr title="K quantization for short: The quantization object is the left matrix. Each token uses separate quantization parameters.">pertoken</abbr>| (M,) |

</details>

<a id="ascend950-fake-quantization-scenario-constraints"></a>

<details>
<summary>Constraints on the fake-quantization scenario</summary>

<a id="ascend950-s8s4-scenario-constraints"></a>

S8/S4 scenario:

- The following input parameters are empty: antiquantScaleOptional, antiquantOffsetOptional, activationInputOptional, activationQuantScaleOptional and activationQuantOffsetOptional.
- The data type combinations supported by non-empty parameters must meet the following requirements:

| groupType | x | weight | biasOptional | scaleOptional | offsetOptional | perTokenScaleOptional | groupListOptional | out |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | INT8 | INT4 | FLOAT | UINT64 | null | FLOAT | INT64 | BFLOAT16 |
| 0 | INT8 | INT4 | FLOAT | UINT64 | FLOAT/null | FLOAT | INT64 | FLOAT16 |

- scaleOptional and offsetOptional shape ($E$=Number of groups):

| offsetOptional | Sub-scenario| scaleOptional shape | offsetOptional shape |
|:---:|:---|:---|:---|
| null | Offline convergence of per-group and per-channel| [E, K/256, N] | - |
| Not empty| <abbr title="C quantization for short: The quantization object is the right matrix. Each channel uses separate quantization parameters.">perchannel</abbr>| [E, 1, N] | [E, 1, N] (FLOAT32)|

- Restrictions

  In addition to platform restrictions, other restrictions in the S8/S4 scenario are as follows:

  | groupType | splitItem | actType | groupListType |
  |:---:|:---:|:---:|:---|
  | 0 (M-axis grouping)| 2/3 | 0 | 1 (count)|

  - Currently, only TensorList whose x, weight, biasOptional, scaleOptional, offsetOptional, perTokenScaleOptional, and out are all 1s is supported. (This requirement is omitted in the following table.)

  | Input/Output| Sub-scenario| Shape Restriction|
  |:---:|:---|:---|
  | x | Single tensor| 2D, shape = [M,K], transpose not supported|
  | weight | Single tensor| 3D, shape = [E,K,N], transpose not supported|
  | biasOptional | Mandatory input| 2D, with shape [E,N]|
  | perTokenScaleOptional | Single tensor| 1D, with shape [M]|
  | out | Single tensor| - |

  - biasOptional is mandatory. It is the correction amount for offline conversion of the INT4 weight and is obtained based on the 8 × weight × scale along the K axis.
  - When the data type of `weight` is INT32, each INT32 value is treated as eight INT4 values.
  - If offsetOptional is empty, K cannot be greater than 18432 and must be an integer multiple of 256.

Other fake-quantization scenarios:

- The following input parameters are empty: scaleOptional, offsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional and activationQuantOffsetOptional.
- The data type combinations supported by non-empty parameters must meet the following requirements:

| groupType | x | weight | biasOptional | antiquantScaleOptional | antiquantOffsetOptional | out |
|:---:|:---:|:---:|:---|:---|:---|:---|
| -1/0 | BFLOAT16 | INT8/INT4 | BFLOAT16/FLOAT32/null | BFLOAT16 | BFLOAT16/null | BFLOAT16 |
| -1/0 | FLOAT16 | INT8/INT4 | FLOAT16/null | FLOAT16 | FLOAT16/null | FLOAT16 |
| 0 | BFLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | BFLOAT16/FLOAT32/null | BFLOAT16 | null | BFLOAT16 |
| 0 | FLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | FLOAT16/null | FLOAT16 | null | FLOAT16 |

- When the weight is of the FLOAT8 series, antiquantOffsetOptional supports only a null pointer or a null TensorList, and the weight supports only transposition.
- If the weight type is INT4, the last dimension of each group of tensors in the weight must be an even number. The last dimension of $weight_i$ refers to the N axis of $weight_i$ when the weight is not transposed or the K axis of $weight_i$ when the weight is transposed.
- When weight is set to INT4, the perchannel and pergroup fake-quantization modes are supported. The number of dimensions of antiquantScaleOptional is automatically determined. If the number of dimensions of a single scenario is 2, the mode is perchannel. If the number of dimensions of a multi-scenario is 3, the mode is pergroup. If the number of dimensions of a multi-scenario is 1, the mode is perchannel. If the number of dimensions of a multi-scenario is 2, the mode is pergroup.
- In the pergroup scenario, $G$ or $G_i$ must be exactly divided by $k_i$. When there are multiple tensors, the pergroup length $s_i = k_i / G_i$ is the same for each group.
- In the pergroup scenario, the value of groupSize can only be 32, 64, 128, or 256. During weight transposition, the pergroup length $s_i$ must be an even number.
- antiquantScaleOptional and non-empty biasOptional and antiquantOffsetOptional must meet the following requirements ($g$ indicates the number of matmul groups, that is, the number of groups, $G$ indicates the number of pergroups, and $G_i$ indicates the number of pergroups of the ith tensor):

  | groupType | Scenario| Shape restriction|
  |:---:|:---|:---|
  | -1 | Weight multi-tensor (perchannel)| Each tensor has one dimension and the shape is ($n_i$). Some tensors with the shape of ($n_i$) in a tensorList cannot be empty.|
  | -1 | Weight multi-tensor (pergroup)| 2-dimensional tensor with shape ($G_i$, $n_i$)|
  | 0 | Weight single tensor (per channel)| 2-dimensional tensor with shape ($g$, N)|
  | 0 | Weight single tensor (pergroup)| 3D tensor with shape ($g$, $G$, N)|

</details>
<!-- end id9 -->

<!-- npu="A3,910b" id12 -->
### Atlas A2 products and Atlas A3 products

#### Platform Constraints

- Last dimension of each tensor group in x/weight < 65536.
  - Last dimension of $x_i$: K axis when x is not transposed, or M axis when x is transposed.
  - Last dimension of $weight_i$: N axis when the weight is not transposed, and K axis when the weight is transposed.
- When x/weight is transposed, the corresponding tensor must be [non-contiguous](../../../docs/en/context/non_contiguous_tensor.md).
- If the weight is in [FRACTAL_NZ](../../../docs/en/context/data_format.md) format, the shape must meet the NZ format requirements.
- perTokenScaleOptional: Generally, there is only one dimension. The length is the same as M of x. Only the scenario where x, weight, and out are all single tensors is supported.
- groupListOptional: If the length of TensorList in out is 1, groupList restricts the valid range of the output data. The unspecified part is not updated.
- groupListType:
  - groupListType=0: The value must be a non-negative monotonically non-decreasing number sequence (cumulative sum). For example, M = 256 and E = 4 (the sizes of each group are 64, 0, 128, and 64 in sequence): [64, 64, 192, 256]
  - groupListType=1: The value must be a non-negative number sequence (size of each group). Example: [64, 0, 128, 64]
  - groupListType=2: The value must be a non-negative number sequence. shape [E, 2] ([group index, group size]). Non-zero groups are prefixed. Example: [[0, 64], [2, 128], [3, 64], [1, 0]]
- groupType: M-axis grouping (0) and non-grouping (-1) are supported. K-axis grouping (2) is supported in non-quantization scenarios. In full quantization mode, groupType can only be set to 0 (M-axis grouping).
- tuningConfigOptional: supported. The value is INT64 aclIntArray on the host side.
  - [0]: expected number of tokens processed by each expert. Tiling is optimized based on this parameter. Application scenarios: A8W4/A8W8/A4W4, x/weight/out single tensor.
  - [1]: whether to enable the weight affinity format (transpose before NZ). The application scenario is A8W4.
  - [2]: upper limit of the allowed extra workspace (-1 = unlimited). Application scenario: The A8W8 fixed-axis algorithm is optimized.
- actType: The activation function type can be 0, 1, 2, 4, or 5 (the value 3 is not supported) only when the output data type is FLOAT16 or BFLOAT16 and the output data type is A8W8 (the data type of x and weight is INT8). In other scenarios, the value can only be 0.
- The input parameters x and weight and the output parameter out support a maximum of 128 tensors.

#### Scenarios supported by groupType

> x. The inputs x, weight, and y are of the aclTensorList type. "single" = single-element TensorList, "multiple" = multi-element TensorList.

The following table lists the group types supported by each quantization type.

- A16W8 and A16W4: Only groupType=-1 and 0 are supported.
- A8W8, A8W4, and A4W4: groupType=0 and x=single tensor.

| groupType | x | weight | y | splitItem | groupListOptional | Transposition| Other Constraints|
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | Many| Many| Many| 0/1 | Must be empty.| x is not transposed, and weight can be transposed (unified).| The tensors in `x` must have the same dimensionality, which can be 2D. The tensors in `weight` must be 2D. The dimensionality of tensors in `y` must match that of `x`.|
| 0 | Single| Single| Single| 2/3 | Mandatory. groupListType=0 Last value ≤ x First dimension; groupListType=1 Sum ≤ x First dimension; groupListType=2 Sum of the second column ≤ x First dimension. A maximum of 1024 groups are supported.| x: not transposed; weight: transposed (not supported by A8W4/A4W4)| The tensor in weight must be 3D, and the tensors in x and y must be 2D.|
| 0 | Single| Many| Single| 2/3 | Mandatory. groupListType=0 Last value ≤ x First dimension; groupListType=1 Sum ≤ x First dimension; groupListType=2 Sum of the second column ≤ x First dimension; A maximum of 128 groups are supported.| x is not transposed, and weight can be transposed (unified).| (1) Tensors in `x`, `weight`, and `y` must be 2D.<br>(2) The N-axis of each tensor in `weight` must be the same.|
| 0 | Many| Many| Single| 2/3 | Optional. If groupListType = 0, difference = x first dimension, groupListType = 1, value = x first dimension, groupListType = 2, second column = x first dimension. A maximum of 128 groups are supported.| x is not transposed, and weight can be transposed (unified).| (1) Tensors in `x`, `weight`, and `y` must be 2D.<br>(2) The N-axis of each tensor in `weight` must be the same.|
| 2 | Single| Single| Single| 2/3 | Mandatory; groupListType=0 (last value ≤ x second dimension); groupListType=1 (sum ≤ x second dimension); groupListType=2 (sum of the second column ≤ x second dimension); a maximum of 1024 groups| x must be transposed, and weight is not transposed.| (1) The tensors in `x` and `weight` must be 2D, and the tensor in `y` must be 3D.<br>(2) The bias must be empty.|
| 2 | Single| Many| Many| 0/1 | Must be empty.| x must be transposed, and weight is not transposed.| (1) Tensors in `x`, `weight`, and `y` must be 2D.<br>(2) The maximum length of `weight` is 128, meaning a maximum of 128 groups are supported.<br>(3) The sum of the first dimensions of all tensors in the original `weight` shape should not exceed the first dimension of `x`.<br>(4) `bias` must be passed as null.|

#### Scenario quick reference table

| Scene Name| x | weight | out | Constraint|
|--------|:---:|:---:|:---:|------|
| Non-quantized| FLOAT32 | FLOAT32 | FLOAT32 | [Constraints](#constraints-for-non-quantization-scenario)|
| Non-quantized| BFLOAT16 | BFLOAT16 | BFLOAT16 | [Constraints](#constraints-for-non-quantization-scenario)|
| Non-quantized| FLOAT16 | FLOAT16 | FLOAT16 | [Constraints](#constraints-for-non-quantization-scenario)|
| Full quantization -<abbr title="A8 indicates that x uses 8-bit quantization, W8 indicates that weight uses 8-bit quantization, A8W4, and A4W4">A8W8</abbr>.| INT8 | INT8 | BFLOAT16/FLOAT16/INT32/INT8 | [Constraints](#constraints-for-a8w8-scenario)|
| Full quantization - A4W4| INT4 | INT4 | BFLOAT16/FLOAT16 | [Constraints](#constraints-for-a4w4-scenario)|
| fake-quantization - A8W4| INT8 | INT4 | BFLOAT16/FLOAT16 | [Constraints](#constraints-for-a8w4-scenario)|
| fake-quantization-<abbr title="A16 indicates that x is 16-bit non-quantized, and W8 indicates that the weight is 8-bit quantized. Similarly, A16W4">A16W8</abbr> is also used.| BFLOAT16/FLOAT16 | INT8 | BFLOAT16/FLOAT16 | [Constraints](#constraints-for-a16w8-scenario)|
| fake-quantization - A16W4| BFLOAT16/FLOAT16 | INT4 | BFLOAT16/FLOAT16 | [Constraints](#constraints-for-a16w4-scenario)|

<details>
<summary>Constraints for the non-quantization scenario</summary><a id="constraints-for-non-quantization-scenario"></a>

Data type requirements:

| x | weight | bias | groupList | out |
|:---|:---|:---|:---|:---|
| FLOAT | FLOAT (ND) | FLOAT/null | INT64 | FLOAT |
| FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | INT64 | FLOAT16 |
| BFLOAT16 | BFLOAT16(ND/NZ) | FLOAT/null | INT64 | BFLOAT16 |

> The following parameters must be left empty: scale, offset, antiquantScale, antiquantOffset, perTokenScale, activationInput, activationQuantScale, activationQuantOffset and activationFeatureOut.

- Description

  In addition to the platform constraints, other constraints in non-quantization scenarios are as follows:
  - The following values are supported: groupType=-1, 0, 2, actType=0, groupListType=0/1/2.

</details>

<a id="constraints-for-a8w8-scenario"></a>

<details>
<summary>A8W8 scenario constraints</summary>;

Data type requirements:

| x | weight | bias | scale | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|
| INT8 | INT8 (ND) | INT32/null | UINT64 | null | INT8 |
| INT8 | INT8 (ND/NZ) | INT32/null | BFLOAT16 | FLOAT/null | BFLOAT16 |
| INT8 | INT8 (ND/NZ) | INT32/null | FLOAT | FLOAT/null | FLOAT16 |
| INT8 | INT8 (ND/NZ) | INT32/null | null | null | INT32 |

> The following parameters must be left empty: offset, antiquantScale, antiquantOffset, activationInput, activationQuantScale and activationQuantOffset.

- Description

  In addition to the platform restrictions, other restrictions in the A8W8 scenario are as follows:
  - Only groupType=0 (M-axis grouping) is supported.
  - Currently, `x`, `weight`, and `out` are only supported as TensorLists with a length of 1.
  - `x` cannot be transposed.
  - `x` supports only 2D tensors with shape (M, K).
  - `weight` supports only 3D tensors with shape (E, K, N) or (E, N, K).
  - To enable the fixed-axis algorithm for performance optimization, the following input shape and parameter configuration requirements must be met:
    - Input shape requirements (any of the following requirements must be met)

      The shape of x is (M, 7168), and the shape of weight is (7168, 4096).

      The shape of x is (M, 2048), and the shape of weight is (2048, 7168).
    - Parameter configuration requirements

      First element of tuningConfigOptional: Set it to a value greater than 128 and less than 512.

      The second element of tuningConfigOptional is set to 0.

      Third element of tuningConfigOptional: Set it to -1 or a value greater than or equal to the value of M × N × 4.

</details>

<a id="constraints-for-a4w4-scenario"></a>

<details>
<summary>A4W4 scenario constraints</summary>;

Data type requirements:

| x | weight | bias | scale | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|
| INT4 | INT4 (ND/NZ) | null | UINT64 | FLOAT/null | FLOAT16/BFLOAT16 |

> The following parameters must be left empty: offset, antiquantScale, antiquantOffset, activationInput, activationQuantScale and activationQuantOffset.

- Description

  In addition to platform restrictions, other restrictions in the A4W4 scenario are as follows:
  - Only groupType=0 (M-axis grouping), actType=0, and groupListType=0/1/2 are supported.
  - Currently, `x`, `weight`, and `out` are only supported as TensorLists with a length of 1.
  - x does not support transpose. When the weight is in NZ format, transpose is supported. The ND format supports only non-transpose.
  - `x` supports only 2D tensors with shape (M, K).
  - `weight` supports only 3D tensors with shape (E, K, N).
  - If the data format of `weight` is ND, `n` must be an integer multiple of 8.
  - Per-channel and per-group quantization are supported. In the perchannel scenario, the scale shape must be $[E, N]$. In the pergroup scenario, the scale shape must be $[E, G, N]$.
  - In the pergroup scenario, $G$ must be an integer multiple of $K$, and $k/G$ must be an even number.
  - After the NZ transposition of the right matrix is enabled, $K/G$ must be 64-pixel aligned, K must be 64-pixel aligned, and N must be 16-pixel aligned.

</details>

<a id="constraints-for-a8w4-scenario"></a>

<details>
<summary>A8W4 scenario restriction</summary>;

Data type requirements::

| x | weight | bias | scale | offset | perTokenScale | out |
|---------|----------------|--------------|--------|------------|---------------|---------|
| INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | FLOAT | BFLOAT16 |
| INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | FLOAT | FLOAT16 |

- Description

  In addition to the platform restrictions, other restrictions in the A8W4 scenario are as follows:
  - Only groupType=0 (M-axis grouping) and actType=0 are supported.
  - Currently, a TensorList whose x and out lengths are both 1 is supported.
  - weight, scaleOptional, biasOptional, and offsetOptional support the single-tensor scenario (the tensor list length is 1) and multi-tensor scenario (the tensor list length is greater than 1).
  - `x` and `weight` cannot be transposed.
  - `x` supports only 2D tensors with shape (M, K).
  - By default, `weight` supports 3D tensors with shape (E, K, N).
  - Bias is the auxiliary result of offline computation. The value must be $8 \times weight \times scale$ and accumulated in the first dimension. The shape must be $[E, N]$.
  - When the data type of the input weight is INT32, each INT32 is regarded as eight INT4s.
  - When `offset` is null:
    - In this case, `groupListType` can only be `1` (the operator does not validate `groupListType` and treats it as `1`). `k` must be an integer multiple of `quantGroupSize` and `k` ≤ 18432. quantGroupSize indicates the pergroup quantization length in the k direction. Currently, quantGroupSize=256 is supported.
    - In this scenario, n must be an integer multiple of 8.
    - In this scenario, scale is the result of pergroup and perchannel offline fusion. The shape must be $[E, quantGroupNum, N]$, where $quantGroupNum = k \div quantGroupSize$
    - In this scenario, if the expected number of tokens processed by each expert is greater than n/4, that is, the first value in tuningConfigOptional is greater than n/4, better performance is achieved. In this case, the $g \times k \times n$ bytes are added to the video memory usage (g indicates the number of matmul groups, that is, the number of groups).
  - When `offset` is not null:
    - scale is the result of pergroup and perchannel offline fusion. The shape must be $[E, quantGroupNum, N]$, where $quantGroupNum = k \div quantGroupSize$. Currently, quantGroupSize values of 64, 128, and 256 are supported.
    - In this case, `offsetOptional` is not null. Asymmetric quantization offsetOptional is the offline computation auxiliary result during computation, that is, $antiquantOffset \times scale$. The shape must be $[E, quantGroupNum, N]$, and the dtype must be FLOAT32.
    - In this scenario, the second element of tuningConfigOptional cannot be enabled.
  - The second element in the tuningConfigOptional array can be set to 1 to enable the special weight template in the A8W4 scenario (only per-channel is supported) to optimize the operator performance. For details about the shape range with performance advantages, see K >= 2048 && N >= 2048. Note that the template requires that the shape of the weight be (E, N, K). Then, the weight is converted to ND2NZ and then used as the operator input.

</details>

<a id="constraints-for-a16w8-scenario"></a>

<details>
<summary>Constraints for the A16W8 scenario</summary>

Data type requirements:

| x | weight | bias | antiquantScale | antiquantOffset | out |
|:---|:---|:---|:---|:---|:---|
| FLOAT16 | INT8 (ND) | FLOAT16/null | FLOAT16 | FLOAT16 | FLOAT16 |
| BFLOAT16 | INT8 (ND) | FLOAT/null | BFLOAT16 | BFLOAT16 | BFLOAT16 |

> The following parameters must be left empty: scale, offset, perTokenScale, activationInput, activationQuantScale and activationQuantOffset.

- Description

  In addition to the platform restrictions, other restrictions in the A16W8 scenario are as follows:
  - `x` cannot be transposed.
  - Only groupType=-1, 0, actType=0, and groupListType=0/1/2 are supported.
  - Only the per-channel quantization mode is supported.
  - If the weight is a multi-tensor, define the pergroup length $s_i = k_i / G_i$. All $s_i (i = 1,2, ...,g) $s must be equal.
  - The shapes of the fake-quantization parameters `antiquantScaleOptional` and `antiquantOffsetOptional` must meet the following requirements ($g$ indicates the number of Matmul groups).

    | Scenario| Sub-scenario| Shape Restriction|
    |:---------:|:-------:| :-------|
    | Fake-quantization (per-channel)| Single-tensor `weight`| $[E, N]$|
    | Fake-quantization (per-channel)| Multi-tensor `weight`| $[N_i]$|

</details>

<a id="constraints-for-a16w4-scenario"></a>

<details>
<summary>Constraints for the A16W4 scenario</summary>

Data type requirements:

| x | weight | bias | antiquantScale | antiquantOffset | out |
|:---|:---|:---|:---|:---|:---|
| FLOAT16 | INT4 (ND) | FLOAT16/null | FLOAT16 | FLOAT16 | FLOAT16 |
| BFLOAT16 | INT4 (ND) | FLOAT/null | BFLOAT16 | BFLOAT16 | BFLOAT16 |

> The following parameters must be left empty: scale, offset, perTokenScale, activationInput, activationQuantScale and activationQuantOffset.

- Description

  In addition to the platform restrictions, other restrictions in the A16W4 scenario are as follows:
  - `x` cannot be transposed.
  - Only groupType=-1, 0, actType=0, and groupListType=0/1/2 are supported.
  - The last dimension of each group of tensors in the weight must be an even number. The last dimension refers to the N axis of $weight_i$ when the weight is not transposed or the K axis of $weight_i$ when the weight is transposed.
  - Symmetric quantization supports the per-channel and per-group quantization modes. If the per-group quantization mode is used, the per-group number G or $G_i$ must be exactly divided by the corresponding $k_i$.
  - Asymmetric quantization only supports the per-channel mode.
  - In the pergroup scenario, when the weight is transposed, the pergroup length $s_i$ must be an even number.
  - If the weight is a multi-tensor, define the pergroup length $s_i = k_i / G_i$. All $s_i (i = 1,2, ...,g) $s must be equal.
  - The shapes of the fake-quantization parameters `antiquantScaleOptional` and `antiquantOffsetOptional` must meet the following requirements ($g$ indicates the number of Matmul groups, $G$ indicates the number of quantization groups, and $G_i$ indicates the number of quantization groups of the *i*-th tensor).

    | Scenario| Sub-scenario| Shape Restriction|
    |:---------:|:-------:| :-------|
    | Fake-quantization (per-channel)| Single-tensor `weight`| $[E, N]$|
    | Fake-quantization (per-channel)| Multi-tensor `weight`| $[N_i]$|
    | Fake-quantization (per-group)| Single-tensor `weight`| $[E, G, N]$|
    | Fake-quantization (per-group)| Multi-tensor `weight`| $[G_i, N_i]$|

</details>

<!-- end id12 -->
---

## Examples

The following table provides links to sample code files in different scenarios. For details about the scenarios, see the comments in the files. For details, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

| Scenario| Product Model| Example File| Description|
|------|:---:|------|------|
| Non-quantized| ALL | [test_aclnn_grouped_matmul.cpp](../examples/test_aclnn_grouped_matmul.cpp) | Non-quantized basic example (covering the general call sequence)|
| Fake-quantization-A16W8| ALL | [test_aclnn_grouped_matmul_a16w8.cpp](../examples/test_aclnn_grouped_matmul_a16w8.cpp) | x=BF16, weight=INT8, antiquantScale=BF16, antiquantOffset=BF16 |
| fake-quantization - A16W4| ALL | [test_aclnn_grouped_matmul_a16w4.cpp](../examples/test_aclnn_grouped_matmul_a16w4.cpp) | x=BF16, weight=INT4, antiquantScale=BF16, antiquantOffset=BF16 |
| MX Quantization| Ascend 950PR&950DT products | [arch35/test_aclnn_grouped_matmul_mx_quant.cpp](../examples/arch35/test_aclnn_grouped_matmul_mx_quant.cpp) | Ascend 950PR&950DT products MX Quantization Example|
| MX Quantization (K-axis grouping)| Ascend 950PR&950DT products | [arch35/test_aclnn_grouped_matmul_v5_mxfp4_splitk.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_mxfp4_splitk.cpp) | FLOAT4_E2M1 on both sides + E8M0 on both sides, groupType=2|
| Full quantization (dynamic K-C)| Ascend 950PR&950DT products | [arch35/test_aclnn_grouped_matmul_quant_dynamic.cpp](../examples/arch35/test_aclnn_grouped_matmul_quant_dynamic.cpp) | x=INT8, weight=INT8, scale=FLOAT32, perTokenScale=FLOAT |
| G-B quantization| Ascend 950PR&950DT products | [arch35/test_aclnn_grouped_matmul_quant_gb.cpp](../examples/arch35/test_aclnn_grouped_matmul_quant_gb.cpp) | x/w=FLOAT8_E5M2, scale=FLOAT32 (3 dimensions), perTokenScale=FLOAT32 (2 dimensions)|
| Fully quantized S8S4| Ascend 950PR&950DT products | [arch35/test_aclnn_grouped_matmul_v5_s8s4.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_s8s4.cpp) | Per-channel quantization with x=INT8, weight=INT4, and non-empty offset|
| Full quantization A8W8| Atlas A2 products and Atlas A3 products  | [arch22/test_aclnn_grouped_matmul_a8w8.cpp](../examples/arch22/test_aclnn_grouped_matmul_a8w8.cpp) | x=INT8, weight=INT8, scale=UINT64, out=BF16 |
