# aclnnGroupedMatmulV4

Note: This API will be deprecated in later versions. Use the latest aclnnGroupedMatmulV5 API.

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

- Description: Implements grouped matrix multiplication, supporting non-uniform matrix dimension sizes across multiple groups. The basic function is matrix multiplication, for example, $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where $g$ indicates the number of groups, and $m_i$, $k_i$, and $n_i$ indicate the corresponding dimension sizes. Both input and output parameters are of the aclTensorList type, with the following functions:

  - K-axis grouping: $k_i$ varies across groups, while $m_i$ and $n_i$ remain the same for each group. In this case, $x_i$ and $weight_i$ can be concatenated along the K-axis.
  - M-axis grouping: $k_i$ remains the same for each group. In this case, $weight_i$ and $y_i$ can be concatenated along the N-axis.

  Compared with [GroupedMatmulV3](./aclnnGroupedMatmulV3.md), this API has the following new features:
  - The values in `groupListOptional` can be the sizes of groups along the grouping axis.
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
    - Static quantization (per-tensor and per-channel), BFLOAT16 and FLOAT16 outputs, with or without activation (For details, refer to [quantization methods](../../../docs/en/context/quant_mode_introduction.md). Same below.)
    - Dynamic quantization (per-token + per-channel), BFLOAT16 and FLOAT16 outputs, with or without activation
    - Fake-quantization with INT4 input `weight` without activation in per-channel and per-group modes
  - Ascend 950PR/Ascend 950DT:
    - Static quantization (1. pertensor-perchannel (T-C); 2. pertensor-pertensor (T-T)) with BFLOAT16, FLOAT16, and FLOAT32 outputs, with bias.
    - Dynamic quantization (1. pertoken-perchannel (K-C); 2. pertoken-pertensor (K-T); 3. pertensor-pertensor (T-T); 4. pertensor-perchannel (T-C); 5. mx quantization; 6. pergroup-perblock (G-B)) with BFLOAT16, FLOAT16, and FLOAT32 outputs, with bias.
    - Fake-quantization with INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, or HIFLOAT8 input `weight` without activation in per-channel mode only.

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

```cpp
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
  aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulV4(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedMatmulV4GetWorkspaceSize

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
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight (aclTensorList)</td>
      <td>Input</td>
      <td>weight in the formula.</td>
      <td>The length of tensorList can be [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1</td>
      <td>ND, FRACTAL_NZ</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Bias in the formula.</td>
      <td>Same length as weight.</td>
      <td>INT32, BFLOAT16, FLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Scaling factor in the quantization parameters.</td>
      <td>Generally, the length is the same as the weight length.</td>
      <td>UINT64, INT64, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Offset in the quantization parameters.</td>
      <td>Same length as weight.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Indicates the scale in the fake-quantization parameter.</td>
      <td>Same length as weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Indicates the offset in the fake-quantization parameter.</td>
      <td>Same length as weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>perTokenScaleOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Indicates the scale introduced by x quantization in the quantization parameter.</td>
      <td>Only x, weight, and out can be single tensors.</td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Matmul size distribution along the grouping axis for inputs and outputs.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationInputOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Currently, only nullptr can be passed.</td>
      <td>Currently, only nullptr can be passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationQuantScaleOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Currently, only nullptr can be passed.</td>
      <td>Currently, only nullptr can be passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationQuantOffsetOptional (aclTensorList)</td>
      <td>Input</td>
      <td>Currently, only nullptr can be passed.</td>
      <td>Currently, only nullptr can be passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>splitItem (int64_t)</td>
      <td>Input</td>
      <td>Integer, indicating whether to split the output tensor.</td>
      <td>0/1 indicates that the output is multi-tensor; 2/3 indicates that the output is single-tensor.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType (int64_t)</td>
      <td>Input</td>
      <td>Integer, indicating the axis to be grouped.</td>
      <td>The enumerated values are -1, 0, and 2. For example, if the matrix multiplication is C[m,n] = A[m,k] x B[k,n], groupType is set to -1 (no grouping), 0 (grouping by the m axis), or 2 (grouping by the k axis).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType (int64_t)</td>
      <td>Input</td>
      <td>-</td>
      <td>The enumerated values are 0, 1, and 2. For details about the constraints, see <a href="#constraints">Constraints.</a></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actType (int64_t)</td>
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
      <td>out (aclTensorList)</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>The length of tensorList can be [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, INT8, FLOAT32, INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationFeatureOutOptional (aclTensorList)</td>
      <td>Output</td>
      <td>Input data of the activation function.</td>
      <td>Currently, only nullptr can be passed.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dynQuantScaleOutOptional (aclTensorList)</td>
      <td>Output</td>
      <td>Currently, only nullptr can be passed.</td>
      <td>Currently, only nullptr can be passed.</td>
      <td>-</td>
      <td>-</td>
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
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
    - x: FLOAT16, BFLOAT16, FLOAT32, INT8, and INT4
    - weight: FLOAT16, BFLOAT16, FLOAT32, INT8, and INT4; format: ND and FRACTAL_NZ
    - biasOptional: FLOAT16, FLOAT32, and INT32
    - scaleOptional: UINT64, BFLOAT16, and FLOAT32
    - perTokenScaleOptional: FLOAT32
    - out: FLOAT16, BFLOAT16, INT8, FLOAT32, and INT32
    - The input parameters x and weight, and the output parameter out support a maximum of 128 tensors.
  - Ascend 950PR/Ascend 950DT:
    - x supports FLOAT8_E4M3FN, FLOAT8_E5M2, INT8, HIFLOAT8, FLOAT16, BFLOAT16, FLOAT32 and FLOAT4_E2M1.
    - weight supports FLOAT8_E4M3FN, FLOAT8_E5M2, INT8, INT4, HIFLOAT8, FLOAT16, BFLOAT16, FLOAT32 and FLOAT4_E2M1. Only the ND format is supported.
    - biasOptional supports INT32, BFLOAT16, FLOAT16, and FLOAT32. When the input x is of type INT8, FLOAT16, BFLOAT16, or FLOAT32, biasOptional supports INT32, BFLOAT16, FLOAT16, and FLOAT32. When the input x is of type FLOAT4_E2M1, biasOptional supports only FLOAT32. For other types of inputs, a null pointer needs to be passed.
    - scaleOptional supports UINT64, INT64, BFLOAT16, FLOAT32, and FLOAT8_E8M0.
    - perTokenScaleOptional supports FLOAT32 and FLOAT8_E8M0.
    - groupListType cannot be set to 2.
    - actType supports 0, 1, 2, 4, and 5. For details about the constraints, see <a href="#constraints">Constraints</a>.
    - out supports BFLOAT16, FLOAT16, and FLOAT32.
    - offsetOptional is not supported.
    - groupType supports grouping by m axis. Only non-quantization and quantization support grouping by k axis. Only non-quantization and fake-quantization support no grouping.
    - In the non-quantization scenario, the input parameters x and weight and the output parameter out support a maximum of 1024 tensors. In the fake-quantization scenario, the input parameters x and weight and the output parameter out support a maximum of 128 tensors. In the full quantization scenario, only a single tensor is supported.

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
      <td>The length of the weight is not supported.</td>
    </tr>
    <tr>
      <td>If bias is not null, the length of bias is not equal to that of weight.</td>
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

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Use Cases

<a id="scenario-types"></a>

- Based on the precision processing of the input data (`x` and `weight`) and the output matrix (`out`) during computation, the GroupedMatmul operator supports three primary scenarios: non-quantization, fake quantization, and full quantization.

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    |Scenario|    x    |    weight      |   out | Constraints|Formula|
    |---------|---------|----------------|--------|--------|--|
    |Non-quantization|FLOAT32|FLOAT32|FLOAT32|[Constraints](#constraints-for-non-quantization-scenario)|[Formula](#non-quantization-scenario)|
    |Non-quantization|BFLOAT16|BFLOAT16|BFLOAT16|[Constraints](#constraints-for-non-quantization-scenario)|[Formula](#non-quantization-scenario)|
    |Non-quantization|FLOAT16|FLOAT16|FLOAT16|[Constraints](#constraints-for-non-quantization-scenario)|[Formula](#non-quantization-scenario)|
    |Full quantization - A8W8|INT8|INT8|BFLOAT16/FLOAT16/INT32/INT8|[Constraints](#constraints-for-a8w8-scenario)|[Formula](#full-quantization-scenario)|
    |Full quantization - A4W4|INT4|INT4|BFLOAT16/FLOAT16|[Constraints](#constraints-for-a4w4-scenario)|[Formula](#full-quantization-scenario)|
    |Fake quantization - A8W4|INT8|INT4|BFLOAT16/FLOAT16|[Constraints](#constraints-for-a8w4-scenario)|[Formula](#a8w4-fake-quantization)|
    |Fake quantization - A16W8|BFLOAT16/FLOAT16|INT8|BFLOAT16/FLOAT16|[Constraints](#constraints-for-a16w4-scenario)|[Formula](#fake-quantization-scenario)|
    |Fake quantization - A16W4|BFLOAT16/FLOAT16|INT4|BFLOAT16/FLOAT16|[Constraints](#constraints-for-a16w4-scenario)|[Formula](#fake-quantization-scenario)|

  - Ascend 950PR/Ascend 950DT:

    For details, see [Ascend 950PR/Ascend 950DT](#ascend_950pr_ascend950dt).
<a id="formulas"></a>
- Formula
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
  - `aclnnGroupedMatmulV4` defaults to deterministic implementation.

<details>
<summary><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></summary>

  - **Common constraints**
  <a id="common-constraints"></a>
    - If `x` and `weight` need to be transposed, the corresponding tensors must be [non-contiguous](../../../docs/en/context/non_contiguous_tensor.md).
    - The size of the last dimension for each tensor in `x` and `weight` should be less than 65536. The last dimension of $x_i$ refers to the K-axis when `x` is not transposed or the M-axis when `x` is transposed. The last dimension of $weight_i$ refers to the N-axis when `weight` is not transposed or the K-axis when `weight` is transposed.
    - When the `weight` [Data Format](../../../docs/en/context/data_format.md) is FRACTAL_NZ, the shape of `weight` must meet the requirements of the FRACTAL_NZ format.
    - Generally, `perTokenScaleOptional` supports only 1D, and the length must match the M-axis size of `x`. This parameter only supports scenarios where `x`, `weight`, and `out` are all single-tensor (with a TensorList length of 1).
    - When the length of the TensorList in the output is 1, `groupListOptional` constrains the valid portion of the output data. Any portion not specified in `groupListOptional` will not be updated.
    - When `groupListType` is `0`, `groupListOptional` must be a non-negative, monotonically non-decreasing sequence, representing the cumulative sum (cumsum) results of the grouping axis sizes. When `groupListType` is `1`, it must be a non-negative sequence representing the size of each group along the grouping axis. When `groupListType` is `2`, it must be a non-negative sequence with a shape of [E, 2], where $E$ represents the group size. The data layout is `[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]`, where `groupSize` indicating the size of each group along the grouping axis. For details, see [groupListOptional configuration examples](#grouplistoptional-configuration-examples).
    - groupType indicates the axis to be grouped. For example, in matrix multiplication C[m,n] = A[m,k] x B[k,n], groupType can be set to -1 (no grouping), 0 (grouping by the M axis), or 2 (grouping by the K axis). For details, see the constraints in <a href="#groupType-constraints">Supported Scenarios for groupType</a>.
    - actType (int64_t, input for computation): integer, indicating the activation function type. The value ranges from 0 to 5.

    <a id="constraints-for-a8w8-scenario"></a>

    <details>
    <summary>Constraints for the A8W8 scenario</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT8 | INT8 (ND) | INT32/null | UINT64 | null | null | null | null | INT64 | null | null | null | INT8 |
      | INT8 | INT8 (ND/NZ) | INT32/null |BFLOAT16| null | null | null | FLOAT/null | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT8 (NZ) | BFLOAT16/null |FLOAT/BFLOAT16| null | null | null | FLOAT/null | INT64 | null | null | null | BFLOAT16|
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
      M = 789, K = 4096, N = 7168, E = 9 (0, 2, and 5 experts have tokens to be processed. Expert 0 processes 123 tokens, and experts 2 and 5 process 333 tokens respectively.)
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

</details>

<a id="ascend_950pr_ascend950dt"></a>

<details>
<summary>Ascend 950PR/Ascend 950DT</summary>

  - Common constraints:

    - `groupListType`: The value can be 0 or 1. When groupListType is set to 0, groupListOptional must be a non-negative monotonic non-decreasing sequence. When groupListType is set to 1, groupListOptional must be a non-negative sequence.
    - actType (int64_t, input for computation): integer, indicating the activation function type. The value ranges from 0 to 5.
      - In fake-quantization and non-quantization scenarios, only 0 is supported for actType.
      - In full quantization scenarios, when x and weight are of the INT8 type, the quantization mode is static T-C quantization or dynamic K-C quantization, and the scale data type is FLOAT32 or BFLOAT16, the value of actType can be 0, 1, 2, 4, or 5. In other full quantization scenarios, only 0 is supported for actType.
    <a id="constraints-on-static-quantization-scenario"></a>
    <details>
    <summary>Constraints on the static quantization scenario</summary>

    - The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, and activationInputOptional.

    - The combinations of data types supported by the parameters that are not empty must meet the requirements in the following table.

      |groupType| x       | weight  | biasOptional | scaleOptional | out     |
      |:-------:|:-------:|:-------:| :------      |:-------       | :------ |
      |0|INT8     |INT8     |INT32/null    | UINT64/INT64  |BFLOAT16/FLOAT16/INT8|
      |0|INT8     |INT8     |INT32/null    | null/UINT64/INT64  |INT32|
      |0|INT8     |INT8     |INT32/BFLOAT16/FLOAT32/null    | BFLOAT16/FLOAT32  | BFLOAT16|
      |0|INT8     |INT8     |INT32/FLOAT16/FLOAT32/null    | FLOAT32  |FLOAT16|
      |0|HIFLOAT8     |HIFLOAT8    |null    | UINT64/INT64  |BFLOAT16/FLOAT16/  FLOAT32|
      |0/2|HIFLOAT8     |HIFLOAT8    |null    | FLOAT32  |BFLOAT16/FLOAT16/FLOAT32|
      |0|FLOAT8_E5M2/FLOAT8_E4M3FN   |FLOAT8_E5M2/FLOAT8_E4M3FN   |null    |  UINT64/INT64  |BFLOAT16/FLOAT16/FLOAT32|
      |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN   |FLOAT8_E5M2/FLOAT8_E4M3FN   |null    |  FLOAT32  |BFLOAT16/FLOAT16/FLOAT32|

    - The scaleOptional must meet the requirements in the following table (g indicates the number of matmul groups, that is, the number of groups).

      |groupType| Application Scenario| Shape Restriction|
      |:---------:|:---------:| :------ |
      |0/2|Single-tensor weight|In perchannel mode, each tensor is two-dimensional, and the shape is (g, N). In pertensor mode, each tensor is two-dimensional or one-dimensional, and the shape is (g, 1) or (g,).|

    </details>

    <details>
      <summary>Constraints on dynamic quantization (T-T && T-C && K-T && K-C quantization) scenarios</summary>
        <a id="constraints-on-dynamic-quantization-(T-T-&&-T-C-&&-K-T-&&-K-C-quantization)"></a>

    - The supported input types in the dynamic quantization (T-T && T-C && K-T && K-C quantization) scenario are as follows:
      - The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, and activationInputOptional.
      - The data type combinations supported by the parameters that are not empty must meet the requirements in the following table:

          |groupType| x       | weight  | biasOptional | scaleOptional |  perTokenScaleOptional |out     |
          |:-------:|:-------:|:-------:| :------      |:-------    | :------   |   :------ |
          |0|INT8  |INT8| INT32/BFLOAT16/FLOAT32/null     |BFLOAT16/FLOAT32    |  FLOAT32   | BFLOAT16 |
          |0|INT8  |INT8| INT32/FLOAT16/FLOAT32/null     |FLOAT32    | FLOAT32   |  FLOAT16 |
          |0/2|HIFLOAT8  |HIFLOAT8| null     |FLOAT32    | FLOAT32   | BFLOAT16/  FLOAT16/FLOAT32 |
          |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| null     |  FLOAT32    | FLOAT32   | BFLOAT16/  FLOAT16/FLOAT32 |

      - The scaleOptional must meet the requirements listed in the following table (g indicates the number of matmul groups, that is, the number of groups). It is recommended that the shape of scaleOptional be (g,) in the pertensor scenario to avoid confusion with the G-B quantization mode.

          | groupType | Application Scenario| Shape Restriction|
          |:---------:|:---------:| :------ |
          |0/2|Single-tensor weight|Perchannel scenario: Each tensor is two-dimensional, and the shape is (g, N). Pertensor scenario: Each tensor is two-dimensional or one-dimensional, and the shape is (g, 1) or (g,).|

      - perTokenScaleOptional must meet the following requirements:

          | groupType | Application Scenario| Shape Restriction|
          |:---------:|:---------:| :------ |
          |0|Single-tensor x|Pertoken scenario: Each tensor is one-dimensional, and the shape is (M,). Pertensor scenario: Each tensor is two-dimensional or one-dimensional, and the shape is (g, 1) or (g,). The pertensor scenario is not supported when the input is of the INT8 type.|
          |2|Single-tensor x|Pertoken scenario: Each tensor is two-dimensional, and the shape is (g, M). Pertensor scenario: Each tensor is two-dimensional or one-dimensional, and the shape is (g, 1) or (g,).|

    </details>

    <details>
      <summary>Constraints on dynamic quantization (mx quantization) scenario</summary>
        <a id="constraints-on-dynamic-quantization-(mx-quantization)-scenario"></a>

    - The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, and activationInputOptional.
    - In the calculation formula, the quantization block size is as follows: gsM = gsN = 1, gsK = 32. mx quantization is a special pergroup quantization.
    - The combinations of data types supported by the parameters that are not empty must meet the requirements listed in the following table.

        |groupType| x       | weight  | biasOptional | scaleOptional |  perTokenScaleOptional |out     |
        |:-------:|:-------:|:-------:|:-------:| :-------    | :------   | :------ |
        |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| null|   FLOAT8_E8M0    | FLOAT8_E8M0    | BFLOAT16/FLOAT16/FLOAT32 |
        |0|FLOAT4_E2M1 |FLOAT4_E2M1| FLOAT32/null |   FLOAT8_E8M0    | FLOAT8_E8M0    |   BFLOAT16/FLOAT16/FLOAT32 |

    - The scaleOptional must meet the requirements listed in the following table. Here, g indicates the number of matmul groups, and g_i indicates the ith group (starting from 0).

        |groupType| Application Scenario| Shape Restriction|
        |:---------:|:---------:| :------ |
        |0|Single-tensor weight|Each tensor has four dimensions. When the weight is transposed, the shape is (g, N, ceil(K/64), 2). When the weight is not transposed, the shape is (g, ceil(K/64), N, 2).|
        |2|Single-tensor weight|Each tensor has three dimensions. The shape is ((K/64) + g, N, 2). The start address offset of scale_i is ((K_0 + K_1 +... + K_{i-1})/64 + g_i) *N* 2. That is, the start address offset of scale_0 is 0, the start address offset of scale_1 is (K_0/64 + 1) *N* 2, and the start address offset of scale_2 is ((K_0 + K_1)/64 + 2) *N* 2.|

    - perTokenScaleOptional must meet the following requirements:

        |groupType| Application Scenario| Shape Restriction|
        |:---------:|:---------:| :------ |
        |0|Single-tensor x|Each tensor has three dimensions. The shape is (M, ceil(K/64), 2).|
        |2|Single-tensor x|Each tensor has three dimensions. The shape is ((K/64) + g, M, 2). The start address offset is the same as that of scale.|

    - If the input x in the FLOAT4_E2M1 quantization is x, K must be an even number and cannot be 2. When the weight is not transposed, N must be an even number.
    </details>

    <details>
      <summary>Constraints on dynamic quantization (G-B quantization) scenario</summary>
        <a id="constraints-on-dynamic-quantization-(g-b-quantization)-scenario"></a>

    - The following data types are supported in the dynamic quantization (G-B quantization) scenario:
    - The following input parameters are empty: biasOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, and activationInputOptional.
    - The quantization block size is calculated as follows: Currently, only gsM = 1 and gsN = gsK = 128 are supported.
    - The data type combinations supported by non-empty parameters must meet the requirements listed in the following table.

        |groupType| x       | weight  |  scaleOptional | perTokenScaleOptional |  out     |
        |:-------:|:-------:|:-------:| :-------    | :------   | :------ |
        |0/2|HIFLOAT8  |HIFLOAT8| FLOAT32    | FLOAT32    | BFLOAT16/FLOAT16/ FLOAT32 |
        |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT32    |  FLOAT32    | BFLOAT16/FLOAT16/FLOAT32 |

    - The scaleOptional parameter must meet the requirements listed in the following table. (g indicates the number of matmul groups, that is, the number of groups, and g_i indicates the ith group (the subscript starts from 0).)

        |groupType| Application Scenario| Shape Restriction|
        |:---------:|:---------:| :------ |
        |0|Single-tensor weight|Each tensor has three dimensions. When the weight is transposed, the shape is (g, ceil(N/gsN), ceil(K/gsK)). When the weight is not transposed, the shape is (g, ceil(K/gsK), ceil(N/gsN)).|
        |2|Single-tensor weight|Each tensor has two dimensions, and the shape is (K/gsK + g, ceil(N/gsN)). The address offset of scale_i is ((K_0 + K_1 +... + K_{i-1})/gsK + g_i) *ceil(N/gsN). That is, the start address offset of scale_0 is 0, the start address offset of scale_1 is (K_0/gsK + 1)* ceil(N/gsN), and the start address offset of scale_2 is ((K_0 + K_1)/gsK + 2) * ceil(N/gsN), and so on.|

    - The perTokenScaleOptional parameter must meet the following requirements:

        |groupType| Application Scenario| Shape Restriction|
        |:---------:|:---------:| :------ |
        |0|Single-tensor x|Each tensor has two dimensions, and the shape is (M, ceil(K/gsK)).|
        |2|Single-tensor x|Each tensor is two-dimensional, with the shape of (K/gsK + g, M). The address offset of per_token_scale_i is ((K_0 + K_1 +... + K_{i-1})/gsK + g_i) *M. That is, the address offset of per_token_scale_0 is 0, the address offset of per_token_scale_1 is (K_0/gsK + 1)* M, and the address offset of per_token_scale_2 is ((K_0 + K_1)/gsK + 2) * M. The rest can be deduced by analogy.|

    - Special processing in dynamic quantization scenarios:
      - In the dynamic quantization scenario where M or K is grouped, if N is 1 and the shape of scaleOptional is (g, 1), and the weight can be quantized in either per-tensor or per-channel mode, the per-tensor quantization mode is preferred.
      - In the dynamic quantization scenario where M is grouped, if g = M and the shape of perTokenScaleOptional is (g,), the pertoken quantization mode is selected for x. If g = M, K <= 128, and the shape of perTokenScaleOptional is (g, 1), the quantization mode of x is selected based on the quantization mode of the weight. If the weight is quantized in per-channel or per-tensor mode, the pertensor quantization mode is selected for x. If the weight is quantized in per-block mode, the pergroup quantization mode is selected for x.
      - In the dynamic quantization scenario where K is grouped, if K is less than 128, N is less than or equal to 128, and the shape of scaleOptional is (g, 1), the quantization mode can be either non-pergroup or G-B quantization according to the existing quantization mode differentiation rules. In this scenario, the G-B quantization mode is used.
      - In the dynamic quantization scenario with K groups, when M is equal to 1 and the shape of perTokenScaleOptional is (g, 1), if x can be either pertoken or pertensor quantized, the pertensor quantization mode is preferred.
      - In the dynamic quantization scenario with K groups, if K is less than 128, M is equal to 1, and the shape of perTokenScaleOptional is (g, 1), then: If N is less than or equal to 128, pergroup quantization is selected for x. If N is greater than 128, the quantization mode of x is selected based on the quantization mode of the weight. If the weight is perchannel or pertensor quantization, pertensor quantization is selected for x. If the weight is perblock quantization, pergroup quantization is selected for x.
      - In the dynamic quantization scenario with K groups, if K is less than 128 and M is not equal to 1, then: If N is less than or equal to 128, pergroup quantization is selected for x. If N is greater than 128, the quantization mode of x is selected based on the quantization mode of the weight. If the weight is perchannel or pertensor quantization, perToken quantization is selected for x. If the weight is perblock quantization, pergroup quantization is selected for x.
    </details>

    <details>
      <summary>Constraints on non-quantization scenarios</summary>
        <a id="constraints-for-non-quantization-scenario"></a>

    - The following data types are supported in non-quantization scenarios:
      - The following input parameters are empty: scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional and activationFeatureOutOptional.
      - The data type combinations supported by the parameters that are not empty must meet the requirements in the following table.

          |groupType| x       | weight  | biasOptional | out     |
          |:-------:|:-------:|:-------:| :------      |:------ |
          |-1/0/2   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
          |-1/0/2   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|
          |-1/0/2   |FLOAT32     |FLOAT32     |FLOAT32/null    | FLOAT32|

    </details>

    <details>
      <summary>Constraints on fake-quantization scenarios</summary>
        <a id="constraints-on-fake-quantization-scenario"></a>

    - The fake-quantization scenario supports the following data types:
      - The following input parameters are empty: scaleOptional, offsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional and activationQuantOffsetOptional.
      - The combinations of data types supported by non-empty parameters must meet the requirements in the following table.

          |groupType| x       | weight  | biasOptional |antiquantScaleOptional| antiquantOffsetOptional| out     |
          |:-------:|:-------:|:-------:| :------      |:------|:------|:------|
          |-1/0   |BFLOAT16     |INT8/INT4 |BFLOAT16/FLOAT32/null| BFLOAT16 | BFLOAT16/  null | BFLOAT16 |
          |-1/0   |FLOAT16     |INT8/INT4     |FLOAT16/null    | FLOAT16 | FLOAT16/null |  FLOAT16 |
          |0   |BFLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 |BFLOAT16/FLOAT32/ null| BFLOAT16 | null | BFLOAT16 |
          |0   |FLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 |FLOAT16/null    |   FLOAT16 | null | FLOAT16 |

      - When the data type of weight is FLOAT8_E5M2, FLOAT8_E4M3FN or HIFLOAT8, antiquantOffsetOptional can only be set to a null pointer or null tensor list, and weight can only be transposed.
      - If the weight type is INT4, the last dimension of each group of tensors in weight must be an even number. The last dimension of $weight_i$ refers to the N-axis when `weight` is not transposed or the K-axis when `weight` is transposed.
      - The following table describes the requirements for antiquantScaleOptional and non-empty biasOptional and antiquantOffsetOptional (g indicates the number of matmul groups, that is, the number of groups).

          |groupType| Application Scenario| Shape Restriction|
          |:---------:|:---------:| :------ |
          |-1|Multi-tensor weight|Each tensor is 1-dimensional, and the shape is ($n_i$). It is not allowed that some tensors in a tensor list have the shape ($n_i$) and some tensors are empty.|
          |0|Single-tensor weight|Each tensor is 2-dimensional, and the shape is (g, N).|

    </details>

    <details>
      <summary>Constraints on different groupTypes</summary>
        <a id="constraints-on-different-grouptypes"></a>

    - Supported scenarios for different `groupType` values:
      - "S" stands for single-tensor, and "M" stands for multi-tensor, expressed in the sequence of `x`, `weight`, `out`. For example, "SMS" indicates single-tensor `x`, multi-tensor `weight`, and single-tensor `out`.

          | groupType | Supported Scenario| Scenario Restrictions|
          |:---------:|:-------:| :------ |
          | -1 | MMM|(1) `splitItem` can only be set to `0` or `1`.<br>2) For non-quantized x, the out tensor must be 2-dimensional, with shapes of ($m_i$, $k_i$) and ($m_i$, $n_i$). In the fake-quantization scenario, the tensor in x must have the same dimension (2-6 dimensions). The tensor in y must have the same dimension as x. The tensor in weight must have two dimensions and the shape must be ($n_i$, $k_i$) or ($k_i$, $n_i$). The tensor in bias must have one dimension and the shape must be ($n_i$).<br>(3) `groupListOptional` must be passed as null.<br>(4) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(5) `x` cannot be transposed.<br>(6) Only non-quantization and fake-quantization are supported.<br>(7) Only ND-to-ND input and output are supported.<br>|
          | 0 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensors in weight must be 3-dimensional, with shapes of (g, N, K) or (g, K, N). The tensors in x and out must be 2-dimensional, with shapes of (M, K) and (M, N), respectively. The tensors in bias must be 2-dimensional, with shape of (g, N).<br>(3) groupListOptional must be passed. When groupListType is 0, the last value cannot be greater than the first dimension of the tensor in x. When groupListType is 1, the sum of the values cannot be greater than the first dimension of the tensor in x.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) x can be not transposed, and weight can be transposed or not transposed.<br>(6) Only ND input and ND output are supported.<br>|
          | 0 | SMS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) groupListOptional must be passed. When groupListType is set to 0, the last value must be equal to the first dimension of the tensor in x. When groupListType is set to 1, the sum of the values must be equal to the first dimension of the tensor in x. The maximum length is 1024.<br>(3) The tensor in x and out must be 2-dimensional, with the shape being (M, K) and (M, N), respectively. The tensor in weight must be 2-dimensional, with the shape being (N, K) or (K, N). The tensor in bias must be 1-dimensional, with the shape being (N).<br>(4) The N-axis of each tensor in `weight` must be the same.<br>(5) Weight transposition is supported, but the transposition status of each tensor in tensorList of weight must be the same.<br>(6) `x` cannot be transposed.<br>(7) Only non-quantization is supported.<br>(8) Only ND input and ND output are supported.<br>|
          | 0 | MMS|(1) `splitItem` can only be set to `2`.<br>(2) The tensor in x and out must be 2-dimensional, with the shape being (M, K) and (M, N), respectively. The tensor in weight must be 2-dimensional, with the shape being (N, K) or (K, N). The tensor in bias must be 1-dimensional, with the shape being (N).<br>(3) The N-axis of each tensor in `weight` must be the same.<br>(4) If groupListOptional is passed, when groupListType is set to 0, the difference between groupListOptional and the first dimension of the tensor in x must be in one-to-one correspondence. When groupListType is set to 1, the value of groupListOptional must be in one-to-one correspondence with the first dimension of the tensor in x. The maximum length is 1024.<br>(5) `weight` can be transposed. The transposition status of each tensor within the `weight` tensorList must be consistent.<br>(6) `x` cannot be transposed.<br>(7) Only non-quantization is supported.<br>(8) Only ND input and ND output are supported.<br>|
          | 2 | SSS|(1) `splitItem` can only be set to `2` or `3`.<br>(2) The tensor in x and weight must be 2-dimensional, with the shape being (K, M) and (K, N), respectively. The tensor in out must be 3-dimensional, with the shape being (g, M, N).<br>(3) groupListOptional must be passed. When groupListType is set to 0, the last value must be less than or equal to the first dimension of the tensor in x. When groupListType is set to 1, the sum of the values must be less than or equal to the first dimension of the tensor in x.<br>(4) The first dimension of `groupListOptional` supports a maximum of 1024 groups.<br>(5) `x` must be transposed, and `weight` cannot be transposed.<br>(6) Only non-quantization and quantization are supported.<br>(7) Only ND input and ND output are supported.|

    </details>

</details>

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
