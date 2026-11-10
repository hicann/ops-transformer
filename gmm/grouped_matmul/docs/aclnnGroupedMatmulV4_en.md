# aclnnGroupedMatmulV4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:18:55.141Z -->

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

- Description: Implements grouped matrix multiplication, where the dimension sizes of each group can differ. The basic function is matrix multiplication, as in $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where g is the number of groups and $m_i/k_i/n_i$ are the corresponding dimensions. Both input and output parameter types are aclTensorList, with the following functions:

  - k-axis grouping: $k_i$ values differ, but $m_i/n_i$ are the same for each group. In this case, $x_i/weight_i$ can be concatenated along $k_i$.
  - m-axis grouping: $k_i$ is the same across groups, and $weight_i/y_i$ can be concatenated along $n_i$.

  Compared with the [GroupedMatmulV3](aclnnGroupedMatmulV3_en.md) API, **this API adds the following:**
    - Supports using the value in groupListOptional as the size of each group on the grouping axis.

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
      - Supports static quantization (pertensor+perchannel) (for quantization methods, see [Quantization Introduction](../../../docs/en/context/quant_mode_introduction.md); the same below), BFLOAT16 and FLOAT16 outputs, and use cases with and without activation
      - Supports dynamic quantization (pertoken+perchannel) with BFLOAT16 and FLOAT16 output, in scenarios with and without activation.
      - Supports pseudo-quantization with INT4 weight input, without activation, supporting both per-channel and per-group modes.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>:
      - Supports static quantization. Quantization methods: 1. pertensor-perchannel (T-C); 2. pertensor-pertensor (T-T). Supports BFLOAT16, FLOAT16, and FLOAT32 outputs, and use cases with bias.
      - Supports dynamic quantization. Quantization methods: 1. pertoken-perchannel (K-C); 2. pertoken-pertensor (K-T); 3. pertensor-pertensor (T-T); 4. pertensor-perchannel (T-C); 5. MX quantization; 6. pergroup-perblock (G-B). Supports BFLOAT16, FLOAT16, and FLOAT32 outputs, and use cases with bias.
      - Supports inputs with fake-quantized weights of INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, or HIFLOAT8 in use cases without activation. The perchannel and pergroup modes are supported (INT4 supports perchannel and pergroup, and other weight types support only perchannel).

    <!-- end id8 -->

**NOTE**

  - Single tensor means that all grouped tensors in a tensor list are merged into one on the grouping axis specified by groupType; otherwise, it is multi-tensor.
  - Tensor transposition: refers to the scenario where the tensor shape is [M, K], the stride is [1, M], and the data layout is [K, M], i.e., a non-contiguous tensor.

  - Formula:
    - **Non-quantized scenario:**

      $$
       y_i=x_i\times weight_i + bias_i
      $$

    - **Quantization scenario (static quantization, T-C && T-T quantization, without perTokenScaleOptional):**
      $$
        y_i=(x_i\times weight_i) * scale_i + offset_i
      $$
      - x is INT8, bias is INT32
        $$
          y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
        $$
      - x is INT8, bias is BFLOAT16/FLOAT16/FLOAT32, no offset
        $$
          y_i=(x_i\times weight_i) * scale_i + bias_i
        $$
    - **Quantization scenario (dynamic quantization, T-T && T-C && K-T && K-C quantization):**
      $$
       y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i
      $$
      - x is INT8, bias is INT32
        $$
          y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i
        $$
      - x is INT8, bias is BFLOAT16/FLOAT16/FLOAT32
        $$
          y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i  + bias_i
        $$
    - **Quantization scenario (dynamic quantization, MX && G-B quantization):**
      $$
      y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (xSlice_i * weightSlice_i)) * (per\_token\_scale_i[m/gsM, j] * scale_i[j, n/gsN])) + bias_i[n]
      $$
      where gsM, gsN, and gsK represent the quantization block sizes of the M/N/K axes respectively, $xSlice_i$ represents the vector of length gsK in row m of $x_i$, and $weightSlice_i$ represents the vector of length gsK in column n of $weight_i$. All slices along the K axis start from j * gsK. The value range of j is [0, kLoops), where kLoops=ceil($K_i$ / gsK). A final slice shorter than gsK is supported.

    - **Pseudo-Quantized Scenario:**

      $$
       y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnGroupedMatmulV4GetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the computing process. Then call "aclnnGroupedMatmulV4" to execute the computation.

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
      <td>tensorList length support: [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight (aclTensorList *)</td>
      <td>Input</td>
      <td>Input weight in the formula.</td>
      <td>tensorList length support: [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1</td>
      <td>ND, FRACTAL_NZ</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>The bias in the formula.</td>
      <td>The length is the same as that of weight.</td>
      <td>INT32, BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the scale factor in the quantization parameters.</td>
      <td>Generally, the length is the same as that of weight.</td>
      <td>UINT64, INT64, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional(aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the offset in quantization parameters.</td>
      <td>Same length as weight.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional(aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the scale factor in fake-quantization parameters.</td>
      <td>Length is the same as weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the offset in fake-quantization parameters.</td>
      <td>Same length as weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>perTokenScaleOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the scaling factor introduced by x quantization in the quantization parameters.</td>
      <td>Supported only when x, weight, and out are all single tensors.</td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListOptional (aclTensor *)</td>
      <td>Input</td>
      <td>Represents the distribution of matmul sizes along the grouping axis of inputs and outputs.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationInputOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Currently only nullptr is supported.</td>
      <td>Currently only nullptr is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationQuantScaleOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Currently only supports passing nullptr.</td>
      <td>Currently only supports passing nullptr.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationQuantOffsetOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Currently only supports passing nullptr.</td>
      <td>Currently only supports nullptr.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>splitItem (int64_t)</td>
      <td>Input</td>
      <td>Integer parameter that represents whether the output needs to be split into multiple tensors.</td>
      <td>0/1 represents that the output is multiple tensors; 2/3 represents that the output is a single tensor.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType (int64_t)</td>
      <td>Input</td>
      <td>Integer parameter. Represents the axis for grouping.</td>
      <td>Enumeration values: -1, 0, 2. For matrix multiplication C[m,n]=A[m,k]xB[k,n], groupType values are: -1 (no grouping), 0 (grouping on the m axis), and 2 (grouping on the k axis).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType (int64_t)</td>
      <td>Input</td>
      <td>-</td>
      <td>Enum values: 0, 1, and 2. For comprehensive constraints, see <a href="#constraints">Constraints.</a></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actType (int64_t)</td>
      <td>Input</td>
      <td>Represents the activation function type.</td>
      <td>The value range is 0-5.<br>
          0: GMM_ACT_TYPE_NONE;<br>
          1: GMM_ACT_TYPE_RELU;<br>
          2: GMM_ACT_TYPE_GELU_TANH;<br>
          3: GMM_ACT_TYPE_GELU_ERR_FUNC;<br>
          4: GMM_ACT_TYPE_FAST_GELU;<br>
          5: GMM_ACT_TYPE_SILU;<br>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensorList *)</td>
      <td>Output</td>
      <td>The output y in the formula.</td>
      <td>The tensorList length supports [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, INT8, FLOAT32, INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationFeatureOutOptional(aclTensorList *)</td>
      <td>Output</td>
      <td>Input data of the activation function.</td>
      <td>Currently only supports passing nullptr.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dynQuantScaleOutOptional (aclTensorList *)</td>
      <td>Output</td>
      <td>Currently only supports passing in nullptr.</td>
      <td>Currently only supports passing in nullptr.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied for on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Returns the op executor containing the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - x supports FLOAT16, BFLOAT16, FLOAT32, INT8, and INT4.
    - weight supports FLOAT16, BFLOAT16, FLOAT32, INT8, and INT4, and supports the ND and FRACTAL_NZ formats.
    - biasOptional supports FLOAT16, FLOAT32, and INT32.
    - scaleOptional supports UINT64, BFLOAT16, and FLOAT32.
    - perTokenScaleOptional supports FLOAT32.
    - The input parameters x and weight, and the output parameter out support a maximum of 128 tensors.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - x supports FLOAT8_E4M3FN, FLOAT8_E5M2, INT8, HIFLOAT8, FLOAT16, BFLOAT16, FLOAT32, and FLOAT4_E2M1.
    - weight supports FLOAT8_E4M3FN, FLOAT8_E5M2, INT8, INT4, HIFLOAT8, FLOAT16, BFLOAT16, FLOAT32, and FLOAT4_E2M1, and only the ND format is supported.
    - biasOptional supports INT32, BFLOAT16, FLOAT16, and FLOAT32. When input x is INT8, FLOAT16, BFLOAT16, or FLOAT32, INT32, BFLOAT16, FLOAT16, and FLOAT32 are supported. When input x is FLOAT4_E2M1, only FLOAT32 is supported. For other input types, a null pointer must be passed.
    - scaleOptional supports UINT64, INT64, BFLOAT16, FLOAT32, and FLOAT8_E8M0.
    - perTokenScaleOptional supports FLOAT32 and FLOAT8_E8M0.
    - In the fully quantized field, groupListType supports the value 2; other scenarios do not support this value
    - actType supports 0, 1, 2, 4, and 5. For comprehensive constraints, see <a href="#constraints">Constraints</a>.
    - offsetOptional is not supported.
    - groupType supports m-axis grouping. Only non-quantization and quantization support k-axis grouping. Only non-quantization and pseudo-quantization support no grouping.
    - In the non-quantization scenario, the input parameters x and weight and the output parameter out support up to 1024 tensors; in the pseudo-quantization scenario, they support up to 128 tensors; in the full quantization scenario, only a single tensor is supported.

  <!-- end id10 -->

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1055px"><colgroup>
  <col style="width: 242px">
  <col style="width: 78px">
  <col style="width: 735px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, or out is out of the supported range.</td>
    </tr>
    <tr>
      <td>The length of weight is out of the supported range.</td>
    </tr>
    <tr>
      <td>bias is not empty and the length of bias is not equal to the length of weight.</td>
    </tr>
    <tr>
      <td>The dimension of groupListOptional is out of the supported range.</td>
    </tr>
    <tr>
      <td>In use cases where splitItem is 2 or 3, the length of out is not equal to 1.</td>
    </tr>
    <tr>
      <td>In use cases where splitItem is 0 or 1, the length of out is not equal to the length of weight, or the length of groupListOptional is not equal to the length of weight.</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulV4

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
        <td>The workspace memory address applied on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The workspace size applied on the Device side, obtained by the first-phase API aclnnGroupedMatmulV4GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
      </tr>
    </tbody>
    </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Scenario Classification

<a id="scenario-classification"></a>

- Based on the precision handling of input data (x, weight) and output matrix (out) during computation, the GroupedMatmul operator supports the following scenarios: non-quantized, pseudo-quantized, and fully quantized.

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    |Scenario Name|    x    |    weight      |   out | Constraints|Formula|
    |---------|---------|----------------|--------|--------|--|
    |Non-quantized|FLOAT32|FLOAT32|FLOAT32|[Non-quantized Scenario Constraints](#non-quantized-scenario-constraints)|[Formula](#non-quantized-scenario)|
    |Non-quantized|BFLOAT16|BFLOAT16|BFLOAT16|[Non-quantized Scenario Constraints](#non-quantized-scenario-constraints)|[Formula](#non-quantized-scenario)|
    |Non-quantized|FLOAT16|FLOAT16|FLOAT16|[Non-quantized Scenario Constraints](#non-quantized-scenario-constraints)|[Formula](#non-quantized-scenario)|
    |Fully quantized-A8W8|INT8|INT8|BFLOAT16/FLOAT16/INT32/INT8|[A8W8 Scenario Constraints](#a8w8-scenario-constraints)|[Formula](#fully-quantized-field)|
    |Fully quantized-A4W4|INT4|INT4|BFLOAT16/FLOAT16|[A4W4 Scenario Constraints](#a4w4-scenario-constraints)|[Formula](#fully-quantized-field)|
    |Pseudo-quantized-A8W4|INT8|INT4|BFLOAT16/FLOAT16|[A8W4 Scenario Constraints](#a8w4-scenario-constraints)|[Formula](#a8w4-pseudo-quantized-scenario)|
    |Pseudo-quantized-A16W8|BFLOAT16/FLOAT16|INT8|BFLOAT16/FLOAT16|[A16W8 scenario constraints](#a16w8-scenario-constraints)|[Formula](#pseudo-quantized-scenario)|
    |Pseudo-quantized-A16W4|BFLOAT16/FLOAT16|INT4|BFLOAT16/FLOAT16|[A16W4 Scenario Constraints](#a16w4-scenario-constraints)|[Formula](#pseudo-quantized-scenario)|

  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>:

    For details, see [Ascend 950PR&950DT products](#ascend_950pr_ascend950dt)
  <!-- end id12 -->

<a id="formula"></a>

- Formula
  <a id="non-quantized-scenario"></a>

  - **Non-quantized scenario:**

    $$
    y_i=x_i\times weight_i + bias_i
    $$

  <a id="fully-quantized-field"></a>

  - **Full quantization scenario (without perTokenScaleOptional):**
    - x is INT8, bias is INT32

      $$
      y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
      $$

  - **Full quantization scenario (with perTokenScaleOptional):**
    - x is INT8, bias is INT32

      $$
      y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i
      $$

    - x is INT8, bias is BFLOAT16

      $$
      y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i  + bias_i
      $$

    - x is INT4, without bias

      $$
      y_i=x_i\times (weight_i * scale_i) * per\_token\_scale_i
      $$

  <a id="pseudo-quantized-scenario"></a>

  - **Pseudo-Quantized Scenario:**

    - x is Float16 or BFloat16, and weight is INT4 or INT8 (only supported when x, weight, and y are all single tensors).

      $$
      y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
      $$

    <a id="a8w4-pseudo-quantized-scenario"></a>

    - x is INT8, and weight is INT4 (only supported when x, weight, and y are all single tensors). bias is a required parameter, which is an auxiliary result from offline computation, where $bias_i=8\times weight_i  * scale_i$, reduced along the K axis.

      $$
      y_i=((x_i - 8) \times weight_i * scale_i+bias_i ) * per\_token\_scale_i
      $$

## Constraints

- Deterministic computation:

  - aclnnGroupedMatmulV4 defaults to a deterministic implementation.

<!-- npu="A3,910b" id13 -->
<details>
<summary><term>Atlas A2 products</term>, <term>Atlas A3 products</term></summary>

  - **General Constraints**
  <a id="general-constraints"></a>
    - If x and weight need to be transposed, the tensors to be transposed must be [non-contiguous](../../../docs/en/context/non_contiguous_tensor.md).
    - The last dimension of each group of tensors in x and weight must be less than 65536. The last dimension of $x_i$ refers to the K-axis of $x_i$ when x is not transposed, or the M-axis of $x_i$ when x is transposed. The last dimension of $weight_i$ refers to the N-axis of $weight_i$ when weight is not transposed, or the K-axis of $weight_i$ when weight is transposed.
    - If the [data format](../../../docs/en/context/data_format.md) of weight is FRACTAL_NZ, the shape of weight must meet FRACTAL_NZ format requirements.
    - perTokenScaleOptional: In the general case, only 1D is supported and the length must be the same as the M of x. Only the scenario where x, weight, and out are all single tensors (TensorList length is 1) is supported.
    - groupListOptional: When the length of the TensorList in the output is 1, groupListOptional constrains the valid portion of the output data. The parts not specified in groupListOptional will not participate in the update.
    - When groupListType is 0, the values in groupListOptional must be a non-negative monotonically non-decreasing sequence, representing the cumsum result (cumulative sum) of the group axis size. When groupListType is 1, the values in groupListOptional must be a non-negative sequence, representing the size of each group on the group axis. When groupListType is 2, the values in groupListOptional must be a non-negative sequence with the shape [E, 2], where E represents the group size, and the data layout is [[groupIdx0, groupSize0], [groupIdx1, groupSize1]...], where groupSize is the size of each group on the group axis. For details, see [groupListOptional Configuration Example](#grouplistoptional-configuration-example).
    - groupType represents the axis to be grouped. For example, if the matrix multiplication is C[m,n]=A[m,k]xB[k,n], then groupType takes the value -1: no grouping, 0: grouping on the m axis, 2: grouping on the k axis. For details, see the <a href="#groupType-constraints">groupType supported scenarios</a> constraints.
    - actType (int64_t, computation input): An integer parameter that represents the activation function type, with a value range of 0 to 5.

    <a id="a8w8-scenario-constraints"></a>

    <details>
    <summary>A8W8 Scenario Constraints</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT8 | INT8 (ND) | INT32/null | UINT64 | null | null | null | null | INT64 | null | null | null | INT8 |
      | INT8 | INT8 (ND/NZ) | INT32/null |BFLOAT16| null | null | null | FLOAT/null | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT8 (NZ) | BFLOAT16/null |FLOAT/BFLOAT16| null | null | null | FLOAT/null | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT8 (ND/NZ) | INT32/null | FLOAT | null | null | null | FLOAT/null | INT64 | null | null | null | FLOAT16 |
      | INT8 | INT8 (ND/NZ) | INT32/null | null | null | null | null | null | INT64 | null | null | null | INT32 |

    - **Constraints**

      In addition to the [General Constraints](#general-constraints), the remaining constraints for the A8W8 scenario are as follows:
      - Only GroupType=0 (grouping along the M axis) is supported.
      - Currently, only TensorLists with a length of 1 are supported for x, weight, and out.
      - x does not support transposition.
      - x only supports 2D tensors with shape (M, K).
      - weight only supports 3D tensors with shape (E, K, N) or (E, N, K).
      - To enable the fixed-axis algorithm for performance optimization, the following input shape and parameter configuration conditions must be met simultaneously:
        - Input shape conditions (any one of the following groups must be satisfied):

          The shape of x is (M, 7168), and the shape of weight is (7168, 4096).

          The shape of x is (M, 2048), and the shape of weight is (2048, 7168).
        - Parameter configuration conditions

          The first element of tuningConfigOptional: set it to a value greater than 128 and less than 512.

          The second element of tuningConfigOptional: set it to 0.

          The third element of tuningConfigOptional: set it to -1, or set it to a value greater than or equal to M × N × 4.

    </details>

    <a id="a8w4-scenario-constraints"></a>

    <details>
    <summary>A8W4 Scenario Constraints</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | null | null | FLOAT | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | null | null | FLOAT | INT64 | null | null | null | FLOAT16 |

    - **Constraints**

      In addition to the [General Constraints](#general-constraints), the remaining constraints for the A8W4 scenario are as follows:
      - Only GroupType=0 (M-axis grouping) and actType=0 are supported.
      - Currently, only TensorLists with a length of 1 are supported for x, weight, and out.
      - x does not support transposition, and weight does not support transposition.
      - x only supports 2D tensors with shape (M, K).
      - weight defaults to supporting 3D tensors with shape (E, K, N).
      - Bias is an auxiliary result computed offline during the computation process, with the value required to be $8\times weight \times scale$ and accumulated along the first dimension, and the shape required to be $[E, N]$.
      - When the data type of weight is INT32, each INT32 is treated as eight INT4 values.
      - When offset is empty:
        - In this scenario, only groupListType of 1 is supported (the operator does not check the value of groupListType and assumes groupListType is 1). k must be an integer multiple of quantGroupSize, and k <= 18432 is required. quantGroupSize is the pergroup quantization length in the k direction, and currently quantGroupSize=256 is supported.
        - In this scenario, n must be an integer multiple of 8.
        - In this scenario, scale is the result of offline fusion of pergroup and perchannel, with the shape requirement of $[E, quantGroupNum, N]$, where $quantGroupNum=k \div quantGroupSize$.
        - In this scenario, when the expected number of tokens processed by each expert is greater than n/4, that is, when the first value in tuningConfigOptional is greater than n/4, better performance is usually achieved, and the memory usage increases by $g\times k \times n$ bytes (where g is the number of matmul groups, i.e., the number of groups).
      - When offset is not empty
        - scale is the result of offline fusion of pergroup and perchannel, with the shape required to be $[E, quantGroupNum, N]$, where $quantGroupNum = k \div quantGroupSize$. Currently, quantGroupSize values of 64, 128, and 256 are supported.
        - In this scenario, offsetOptional is not empty. For asymmetric quantization, offsetOptional is an auxiliary result computed offline during the calculation process, i.e., $antiquantOffset \times scale$, with its shape required to be $[E, quantGroupNum, N]$ and dtype being FLOAT32.
        - In this scenario, the second element of tuningConfigOptional cannot be enabled.
      - The second element of the tuningConfigOptional array can be set to 1 to enable the special format template of weight in the A8W4 use case for operator performance optimization (shape range with performance gains: K >= 2048 && N >= 2048). Note that this template requires the shape of weight to be (E, N, K), which is then converted using ND2NZ before being used as the operator input.
    </details>

    <a id="a16w4-scenario-constraints"></a>

    <details>
    <summary>A16W4 Scenario Constraints</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | FLOAT16 | INT4 (ND) | FLOAT16/null | null | null | FLOAT16 | FLOAT16 | null | INT64 | null | null | null | FLOAT16 |
      | BFLOAT16| INT4 (ND) | FLOAT/null | null | null | BFLOAT16 | BFLOAT16 | null | INT64 | null | null | null | BFLOAT16|

    - **Constraints**

      In addition to the [General Constraints](#general-constraints), the remaining constraints for the A16W4 scenario are as follows:
      - x does not support transposition.
      - Only GroupType=-1, 0, actType=0, and groupListType=0/1 are supported.
      - The last dimension size of each group of tensors in weight must be an even number. The last dimension refers to the N axis of $weight_i$ when weight is not transposed, or the K axis of $weight_i$ when weight is transposed.
      - Symmetric quantization supports perchannel and pergroup quantization modes. For pergroup, the pergroup number G or $G_i$ must evenly divide the corresponding $k_i$.
      - Asymmetric quantization supports only the perchannel mode.
      - In the pergroup scenario, when weight is transposed, the pergroup length $s_i$ must be even.
      - If weight is a multi-tensor, define the per-group length $s_i = k_i / G_i$, and all $s_i(i=1,2,...g)$ must be equal.
      - The shapes of the pseudo-quantization parameters antiQuantScaleOptional and antiQuantOffsetOptional must satisfy the following table (where g is the number of matmul groups, G is the number of pergroup, and $G_i$ is the number of pergroup of the i-th tensor):

        | usage scenario | Sub-Scenario | Shape Constraint |
        |:---------:|:-------:| :-------|
        | Pseudo-quantized perchannel | Single weight | $[E, N]$|
        | Pseudo-quantized perchannel | multiple weights | $[N_i]$|
        | Pseudo-quantized pergroup | Single weight | $[E, G, N]$|
        | Pseudo-quantized pergroup | multiple weights | $[G_i, N_i]$|

    </details>

    <a id="a16w8-scenario-constraints"></a>

    <details>
    <summary>A16W8 Scenario Constraints</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | FLOAT16 | INT8 (ND) | FLOAT16/null | null | null | FLOAT16 | FLOAT16 | null | INT64 | null | null | null | FLOAT16 |
      | BFLOAT16| INT8 (ND) | FLOAT/null | null | null | BFLOAT16 | BFLOAT16 | null | INT64 | null | null | null | BFLOAT16|

    - **Constraints**

      In addition to the [General Constraints](#general-constraints), the remaining constraints for the a16w8 scenario are as follows:
      - x does not support transposition.
      - Only GroupType=-1, 0, actType=0, and groupListType=0/1 are supported.
      - Only the per-channel quantization mode is supported.
      - If weight is a multi-tensor, define the per-group length $s_i = k_i / G_i$, and all $s_i(i=1,2,...g)$ must be equal.
      - The shapes of the pseudo-quantization parameters antiQuantScaleOptional and antiQuantOffsetOptional must satisfy the following table (where g is the number of matmul groups):

        | usage scenario | Sub-Scenario | Shape Constraint |
        |:---------:|:-------:| :-------|
        | Pseudo-quantized perchannel | Single weight | $[E, N]$|
        | Pseudo-quantized perchannel | multiple weights | $[N_i]$|

    </details>

    <a id="a4w4-scenario-constraints"></a>

    <details>
    <summary>A4W4 Scenario Constraints</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT4 | INT4 (ND/NZ) | null | UINT64 | null | null | null | FLOAT/null | INT64 | null | null | null | FLOAT16/BFLOAT16|

    - **Constraints**

      In addition to the [General Constraints](#general-constraints), the remaining constraints for the A4W4 scenario are as follows:
      - Only GroupType=0 (M-axis grouping), actType=0, and groupListType=0/1 are supported.
      - Currently, only TensorLists with a length of 1 are supported for x, weight, and out.
      - x does not support transposition, and weight does not support transposition.
      - x only supports 2D tensors with shape (M, K).
      - weight only supports 3D tensors with shape (E, K, N).
      - When the data format of weight is ND, n must be an integer multiple of 8.
      - Per-channel and per-group quantization are supported. For the per-channel scenario, the shape of scale must be $[E, N]$; for the per-group scenario, it must be $[E, G, N]$.
      - In the per-group scenario, $G$ must be divisible by $k$, and $k/G$ must be an even number.
    </details>

    <a id="non-quantized-scenario-constraints"></a>

    <details>
    <summary>Non-quantized Scenario Constraints</summary>

    - **Data type requirements**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | FLOAT | FLOAT (ND) | FLOAT/null | null | null | null | null | null | INT64 | null | null | null | FLOAT |
      | FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | null | null | null | null | null | INT64 | null | null | null | FLOAT16 |
      | BFLOAT16| BFLOAT16(ND/NZ) | FLOAT/null | null | null | null | null | null | INT64 | null | null | null | BFLOAT16|

    - **Constraints**

      In addition to the [General Constraints](#general-constraints), the remaining constraints for the non-quantized scenario are as follows:
      - Supports GroupType=-1, 0, 2, actType=0, groupListType=0/1.
    </details>

    <a id="groupType-constraints"></a>

    <details>
    <summary>Supported Scenarios for groupType</summary>

    - The a16w8 and a16w4 scenarios only support groupType values of -1 and 0.
    - The A8W8, A8W4, and A4W4 scenarios only support the case where groupType is 0 and the number of x tensors is single.
    - The input types of x, weight, and y are aclTensorList, which represents an array object of the aclTensor type. In the following table of supported scenarios, "single" indicates an aclTensorList consisting of one aclTensor, and "multiple" indicates an aclTensorList consisting of multiple aclTensors. For example, "single-multiple-single" means that x is a single tensor, weight is multiple tensors, and y is a single tensor.

      | groupType | Number of x Tensors | Number of weight Tensors | Number of y Tensors | splitItem| groupListOptional | Transpose | Remaining Scenario Restrictions |
      |:---------:|:-------:|:-------:|:-------:|:--------:|:------------------|:--------| :-------|
      | -1 | Multiple|Multiple|Multiple | 0/1 | groupListOptional must be null. | 1) x does not support transpose.<br> 2) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must be consistent. | Tensors in x must have consistent dimensions and support 2D. Tensors in weight must be 2D. Tensors in y must have the same dimensions as x. |
      | 0 | Single|Single|Single | 2/3 | 1) groupListOptional must be passed.<br> 2) When groupListType is 0, the last value must be less than or equal to the first dimension of the tensor in x. When groupListType is 1, the sum of values must be less than or equal to the first dimension of the tensor in x. When groupListType is 2, the sum of values in the second column must be less than or equal to the first dimension of the tensor in x.<br> 3) The first dimension of groupListOptional supports a maximum of 1024, i.e., up to 1024 groups. |1) x does not support transpose.<br> 2) Weight transpose is supported, but A8W4 and A4W4 scenarios do not support weight transpose. |Tensors in weight must be 3D. Tensors in x and y must be 2D.|
      | 0 | Single|Multiple|Single | 2/3 | 1) groupListOptional must be passed.<br> 2) When groupListType is 0, the last value must be less than or equal to the first dimension of the tensor in x. When groupListType is 1, the sum of values must be less than or equal to the first dimension of the tensor in x. When groupListType is 2, the sum of values in the second column must be less than or equal to the first dimension of the tensor in x.<br> 3) The first dimension of groupListOptional supports a maximum of 128, i.e., up to 128 groups.|1) x does not support transpose.<br> 2) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must be consistent. |1) Tensors in x, weight, and y must be 2D.<br> 2) The N axis of each tensor in weight must be equal. |
      | 0 | Multiple|Multiple|Single | 2/3 | 1) groupListOptional is optional.<br> 2) If groupListOptional is passed, when groupListType is 0, the differences of groupListOptional must correspond one-to-one with the first dimension of the tensors in x. When groupListType is 1, the values of groupListOptional must correspond one-to-one with the first dimension of the tensors in x. When groupListType is 2, the values in the second column of groupListOptional must correspond one-to-one with the first dimension of the tensors in x.<br> 3) The first dimension of groupListOptional supports a maximum of 128, i.e., up to 128 groups. |1) x does not support transpose.<br> 2) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must be consistent.|1) Tensors in x, weight, and y must be 2D.<br> 2) The N axis of each tensor in weight must be equal. |
      | 2 | Single|Single|Single | 2/3 | 1) groupListOptional must be passed;<br> 2) When groupListType is 0, the last value must be less than or equal to the second dimension of the tensor in x; when groupListType is 1, the sum of the values must be less than or equal to the second dimension of the tensor in x; when groupListType is 2, the sum of the values in the second column must be less than or equal to the second dimension of the tensor in x;<br> 3) The first dimension of groupListOptional supports a maximum of 1024, that is, up to 1024 groups. | 1) x must be transposed.<br> 2) weight cannot be transposed. |1) Tensors in x and weight must be 2D. Tensors in y must be 3D.<br> 2) bias must be null.|
      | 2 | Single|Multiple|Multiple | 0/1 | groupListOptional must be null. | 1) x must be transposed.<br> 2) weight cannot be transposed.| 1) Tensors in x, weight, and y must be 2D.<br> 2) The weight length supports a maximum of 128, i.e., up to 128 groups.<br> 3) In the original shape, the sum of the first dimension of each tensor in weight must not exceed the first dimension of x.<br> 4) bias must be null. |

    </details>

    <a id="grouplistoptional-configuration-example"></a>

    <details>
    <summary>groupListOptional Configuration Example</summary>

    - Shape information
      M = 789, K = 4096, N = 7168, E = 9 (experts 0, 2, and 5 have tokens to process; expert 0 processes 123 tokens, experts 2 and 5 each process 333 tokens).
      The shape of X is [[789, 4096]].
      The shape of W is [[9, 4096, 7168]].
      The shape of Y is [[789, 7168]].

    - When groupListType is 0, groupList is configured as follows:
      - groupListOptional: `[123, 123, 456, 456, 456, 789, 789, 789, 789]`

    - When groupListType is 1, groupList is configured as follows:
      - groupListOptional: `[123, 0, 333, 0, 0, 333, 0, 0, 0]`

    - When groupListType is 2, groupList is configured as follows:
      - In this mode, groupListOptional moves all non-zero groups to the front, which is suitable for scenarios with many inactive experts.
      - groupListOptional: `[[0, 123], [2, 333], [5, 333], [1, 0], [3, 0], [4, 0], [6, 0], [7, 0], [8, 0]]`
    </details>

</details>
<!-- end id13 -->

<a id="ascend_950pr_ascend950dt"></a>

<!-- npu="950" id14 -->
<details>
<summary><term>Ascend 950PR&950DT products</term></summary>

  - Common constraints:

    - groupListType: supports values 0, 1, and 2.
      - If groupListType is 0, groupListOptional must be a non-negative monotonic non-decreasing sequence;
      - If groupListType is 1, groupListOptional must be a non-negative sequence.
      - groupListType 2 is supported only in the fully quantized scenario when groupType is 0. In this case, values in groupListOptional must be a non-negative sequence, and the shape is [E, 2], where E indicates the Group size. The data layout is [[groupIdx0, groupSize0], [groupIdx1, groupSize1]...], where groupSize indicates the size of each group along the grouping axis. Groups with groupSize 0 are placed at the end of groupList, and non-zero groups are placed at the front. For details, see [groupListOptional Configuration Example](#grouplistoptional-configuration-example).
    - actType (int64_t, computation input): An integer parameter that represents the activation function type, with a value range of 0 to 5.
      - In pseudo-quantized and non-quantized scenarios, actType only supports 0.
      - In fully quantized scenarios, when x and weight are INT8, the quantization mode is static T-C quantization or dynamic K-C quantization, and the scale data type is FLOAT32 or BFLOAT16, actType supports 0, 1, 2, 4, and 5. In all other fully quantized scenarios, actType only supports 0.
    <a id="static-quantization-scenario-constraints"></a>
    <details>
    <summary>Static Quantization Scenario Constraints</summary>

    - The following input parameters are null: offsetOptional, antiQuantScaleOptional, antiQuantOffsetOptional, perTokenScaleOptional, activationInputOptional.

    - The supported data type combinations for non-empty parameters must satisfy the following table:

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

    - scaleOptional must meet the following table (where g is the number of matmul groups, i.e., the number of groups):

      |groupType| usage scenario | Shape Constraint |
      |:---------:|:---------:| :------ |
      |0/2|Single weight tensor|perchannel scenario: each tensor is 2-dimensional, with shape (g, N);  pertensor scenario: each tensor is 2- or 1-dimensional, with shape (g, 1) or (g,). The pertensor scenario is not supported when the output is INT8.|

    </details>

    <details>
      <summary>Dynamic quantization (T-T && T-C && K-T && K-C quantization) scenario constraints</summary>
        <a id="dynamic-quantization-t-t-t-c-k-t-k-c-quantization-scenario-constraints"></a>

      - The following input parameters are empty: offsetOptional, antiQuantScaleOptional, antiQuantOffsetOptional, activationInputOptional
      - The supported data type combinations for non-empty parameters must satisfy the following table:

          |groupType| x       | weight  | biasOptional | scaleOptional |  perTokenScaleOptional |out     |
          |:-------:|:-------:|:-------:| :------      |:-------    | :------   |   :------ |
          |0|INT8  |INT8| INT32/BFLOAT16/FLOAT32/null     |BFLOAT16/FLOAT32    |  FLOAT32   | BFLOAT16 |
          |0|INT8  |INT8| INT32/FLOAT16/FLOAT32/null     |FLOAT32    | FLOAT32   |  FLOAT16 |
          |0/2|HIFLOAT8  |HIFLOAT8| null     |FLOAT32    | FLOAT32   | BFLOAT16/  FLOAT16/FLOAT32 |
          |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| null     |  FLOAT32    | FLOAT32   | BFLOAT16/  FLOAT16/FLOAT32 |

      - scaleOptional must satisfy the following table (where g is the number of matmul groups, i.e., the group count). It is recommended to use the shape (g,) for scaleOptional in the pertensor scenario to avoid confusion with the G-B quantization mode:

          | groupType | usage scenario | Shape Constraint |
          |:---------:|:---------:| :------ |
          |0/2|Single weight tensor|Perchannel scenario: each tensor is 2-dimensional, with shape (g, N); pertensor scenario: each tensor is 2-dimensional or 1-dimensional, with shape (g, 1) or (g,).|

      - perTokenScaleOptional must meet the following table:

          | groupType | usage scenario | Shape Constraint |
          |:---------:|:---------:| :------ |
          |0|x single tensor|Pertoken use case: each tensor is 1-dimensional with shape (M,); pertensor use case: each tensor is 2- or 1-dimensional with shape (g, 1) or (g,). The pertensor use case is not supported when the input is INT8.|
          |2|x single tensor|Pertoken use case: each tensor is 2-dimensional with shape (g, M); pertensor use case: each tensor is 2- or 1-dimensional with shape (g, 1) or (g,).|

    </details>

    <details>
      <summary>Dynamic quantization (mx quantization) scenario constraints</summary>
        <a id="dynamic-quantization-mx-quantization-scenario-constraints"></a>

    - The following input parameters are empty: offsetOptional, antiQuantScaleOptional, antiQuantOffsetOptional, activationInputOptional
    - In the formula, the quantization block size is: gsM = gsN = 1, gsK = 32. MX quantization is a special per-group quantization.
    - The supported data type combinations for non-empty parameters must satisfy the following table:

        |groupType| x       | weight  | biasOptional | scaleOptional |  perTokenScaleOptional |out     |
        |:-------:|:-------:|:-------:|:-------:| :-------    | :------   | :------ |
        |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| null|   FLOAT8_E8M0    | FLOAT8_E8M0    | BFLOAT16/FLOAT16/FLOAT32 |
        |0|FLOAT4_E2M1 |FLOAT4_E2M1| FLOAT32/null |   FLOAT8_E8M0    | FLOAT8_E8M0    |   BFLOAT16/FLOAT16/FLOAT32 |

    - scaleOptional must comply with the following table (where g is the number of matmul groups, that is, the number of groups, and g\_i is the i-th group (subscripts start from 0)):

        |groupType| usage scenario | Shape Constraint |
        |:---------:|:---------:| :------ |
        |0|Single weight tensor|Each tensor is 4-dimensional. When weight is transposed, the shape is (g, N, ceil(K / 64), 2); when weight is not transposed, the shape is (g, ceil(K / 64), N, 2).|
        |2|Single weight tensor|Each tensor is 3-dimensional with shape ((K / 64) + g, N, 2). The start address offset of scale\_i is ((K\_0 + K\_1 + ...+ K\_{i-1})/ 64 + g\_i) *N* 2. That is, the start address offset of scale_0 is 0, the start address offset of scale_1 is (K\_0 / 64 + 1) *N* 2, and the start address offset of scale_2 is ((K\_0 + K\_1) / 64 + 2) *N* 2, and so on.|

    - perTokenScaleOptional must meet the following table:

        |groupType| usage scenario | Shape Constraint |
        |:---------:|:---------:| :------ |
        |0|x single tensor|Each tensor is 3-dimensional with shape (M, ceil(K / 64), 2).|
        |2|x single tensor|Each tensor is 3-dimensional with shape ((K / 64) + g, M, 2). The start address offset is the same as that of scale.|

    - For MX quantization where input x is FLOAT4_E2M1, K must be an even number and K must not be 2. When weight is not transposed, N must also be an even number.
    </details>

    <details>
      <summary>Dynamic quantization (G-B quantization) scenario constraints</summary>
        <a id="dynamic-quantization-g-b-quantization-scenario-constraints"></a>

    - The following input parameters are empty: biasOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, activationInputOptional
    - The quantization block size in the formula is: currently only gsM = 1, gsN = gsK = 128 are supported.
    - The supported data type combinations for non-empty parameters must satisfy the following table:

        |groupType| x       | weight  |  scaleOptional | perTokenScaleOptional |  out     |
        |:-------:|:-------:|:-------:| :-------    | :------   | :------ |
        |0/2|HIFLOAT8  |HIFLOAT8| FLOAT32    | FLOAT32    | BFLOAT16/FLOAT16/ FLOAT32 |
        |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT32    |  FLOAT32    | BFLOAT16/FLOAT16/FLOAT32 |

    - scaleOptional must meet the requirements in the following table (where g is the number of matmul groups, that is, the number of groups, and g\_i is the i-th group (index starting from 0)):

        |groupType| usage scenario | Shape Constraint |
        |:---------:|:---------:| :------ |
        |0|Single weight tensor|Each tensor is 3D. When weight is transposed, the shape is (g, ceil(N / gsN), ceil(K / gsK)). When weight is not transposed, the shape is (g, ceil(K / gsK), ceil(N / gsN)).|
        |2|Single weight tensor|Each tensor is 2-dimensional, with shape (K / gsK + g, ceil(N / gsN)). The address offset of scale\_i is ((K\_0 + K\_1 + ...+   K\_{i-1})/ gsK + g\_i)*ceil(N /  gsN). That is, the start address offset of scale\_0 is 0, the start address offset of scale\_1 is (K\_0 / gsK + 1) * ceil(N / gsN), and the start address offset of scale_2 is ((K\_0 + K\_1) / gsK + 2) * ceil(N / gsN), and so on.|

    - perTokenScaleOptional must meet the following table:

        |groupType| usage scenario | Shape Constraint |
        |:---------:|:---------:| :------ |
        |0|x single tensor|Each tensor is 2D with shape (M, ceil(K / gsK)).|
        |2|x single tensor|Each tensor is 2-dimensional, with shape (K / gsK + g, M). The address offset of per\_token\_scale\_i is ((K\_0 + K\_1 + ...+ K\_{i-1}) / gsK + g\_i) \* M. That is, the start address offset of per\_token\_scale\_0 is 0, the start address offset of per\_token\_scale\_1 is (K\_0 / gsK + 1) \* M, and the start address offset of per\_token\_scale\_2 is ((K\_0 + K\_1) / gsK + 2) * M, and so on.|

    - Special handling for dynamic quantization scenarios:
      - In the M-grouping or K-grouping case of the dynamic quantization scenario, when N is 1 and the shape of scaleOptional is (g, 1), if weight supports both pertensor and perchannel quantization, the pertensor quantization mode is preferred.
      - In the dynamic quantization scenario with M-grouping, when g = M and the shape of perTokenScaleOptional is (g,), x uses the per-token quantization mode; when g = M, K <= 128, and the shape of perTokenScaleOptional is (g, 1), the quantization mode of x is selected based on the quantization mode of weight (if weight uses per-channel or per-tensor quantization, x uses per-tensor quantization; if weight uses per-block quantization, x uses per-group quantization).
      - In the dynamic quantization scenario with K-grouping, when K is less than 128, N is less than or equal to 128, and the shape of scaleOptional is (g, 1), according to the existing quantization mode differentiation rules, this scenario could be either non-per-group quantization or G-B quantization. In such cases, it is uniformly treated as G-B quantization.
      - In the K-grouping case of the dynamic quantization scenario, when M is 1 and the shape of perTokenScaleOptional is (g, 1), if x supports both pertoken and pertensor quantization, the pertensor quantization mode is preferred.
      - In the K-grouping case of the dynamic quantization scenario, when K is less than 128, M is 1, and the shape of perTokenScaleOptional is (g, 1), if N is less than or equal to 128, x uses pergroup quantization; if N is greater than 128, the quantization mode of x is selected based on the quantization mode of weight (if weight uses perchannel or pertensor quantization, x uses pertensor quantization; if weight uses perblock quantization, x uses pergroup quantization).
      - In the K-grouping case of the dynamic quantization scenario, when K is less than 128 and M is not 1, if N is less than or equal to 128, x uses pergroup quantization; if N is greater than 128, the quantization mode of x is selected based on the quantization mode of weight (if weight uses perchannel or pertensor quantization, x uses pertoken quantization; if weight uses perblock quantization, x uses pergroup quantization).
    </details>

    <details>
      <summary>Non-quantized scenario constraints</summary>
        <a id="non-quantized-scenario-constraints"></a>

    - The data types supported in non-quantized scenarios are:
      - The following input parameters are empty: scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional, activationFeatureOutOptional.
      - The supported data type combinations for non-empty parameters must satisfy the following table.

          |groupType| x       | weight  | biasOptional | out     |
          |:-------:|:-------:|:-------:| :------      |:------ |
          |-1/0/2   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
          |-1/0/2   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|
          |-1/0/2   |FLOAT32     |FLOAT32     |FLOAT32/null    | FLOAT32|

    </details>

    <details>
      <summary>Fake-quantized scenario constraints</summary>
        <a id="fake-quantized-scenario-constraints"></a>

    - The data types supported in the pseudo-quantized scenario are:
      - The following inputs are empty: scaleOptional, offsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional.
      - The supported data type combinations for non-empty parameters must satisfy the following table.

          |groupType| x       | weight  | biasOptional |antiquantScaleOptional| antiquantOffsetOptional| out     |
          |:-------:|:-------:|:-------:| :------      |:------|:------|:------|
          |-1/0   |BFLOAT16     |INT8/INT4 |BFLOAT16/FLOAT32/null| BFLOAT16 | BFLOAT16/  null | BFLOAT16 |
          |-1/0   |FLOAT16     |INT8/INT4     |FLOAT16/null    | FLOAT16 | FLOAT16/null |  FLOAT16 |
          |0   |BFLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 |BFLOAT16/FLOAT32/ null| BFLOAT16 | null | BFLOAT16 |
          |0   |FLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 |FLOAT16/null    |   FLOAT16 | null | FLOAT16 |

      - When the data type of weight is FLOAT8_E5M2, FLOAT8_E4M3FN, or HIFLOAT8, antiquantOffsetOptional only supports passing a null pointer or an empty tensorList, and weight only supports being transposed.
      - If the type of weight is INT4, the size of the last dimension of each tensor in weight must be an even number. The last dimension of $weight_i$ refers to the N axis of $weight_i$ when weight is not transposed, or the K axis of $weight_i$ when weight is transposed.
      - When the data type of weight is INT4, the perchannel and pergroup modes are supported. The mode is automatically determined by the number of dimensions of antiquantScaleOptional: in the single-single-single use case, antiquantScaleOptional with 2 dimensions indicates perchannel, and antiquantScaleOptional with 3 dimensions indicates pergroup; in the multi-multi-multi use case, antiquantScaleOptional with 1 dimension indicates perchannel, and antiquantScaleOptional with 2 dimensions indicates pergroup.
      - In the pergroup use case, the corresponding $k_i$ must be divisible by the pergroup count G or $G_i$. If weight consists of multiple tensors, the pergroup length is defined as $s_i = k_i / G_i$, and all $s_i(i=1,2,...g)$ must be equal.
      - In the pergroup use case, groupSize only supports 32, 64, 128, and 256. If weight is transposed, the pergroup length $s_i$ must be an even number.
      - antiquantScaleOptional, non-empty biasOptional, and antiquantOffsetOptional must comply with the following table (where g is the number of matmul groups, that is, the number of groups; G is the pergroup count; and $G_i$ is the pergroup count of the i-th tensor):

          |groupType| usage scenario | Shape Constraint |
          |:---------:|:---------:| :------ |
          |-1|weight multi-tensor (perchannel)|Each tensor is 1D with shape ($n_i$). It is not allowed that some tensors in a tensorList have shape ($n_i$) while others are empty. |
          |-1|weight multi-tensor (pergroup)|Each tensor is 2-dimensional, with shape ($G_i$, $n_i$). |
          |0|weight single-tensor (perchannel)|Each tensor is 2D with shape (g, N).|
          |0|weight single-tensor (pergroup)|Each tensor is 3-dimensional, with shape (g, G, N).|

    </details>

    <details>
      <summary>Constraints for different groupTypes</summary>
        <a id="constraints-for-different-grouptypes"></a>

    - Supported scenarios for different groupType values:
      - In the supported scenarios, "single" indicates a single tensor and "multiple" indicates multiple tensors. The order is x, weight, out. For example, "single-multiple-single" indicates that x is a single tensor, weight uses multiple tensors, and out is a single tensor.

          | groupType | supported scenarios | scenario restriction |
          |:---------:|:-------:| :------ |
          | -1 | multiple-multiple-multiple |1) Only splitItem 0/1 is supported.<br>2) In non-quantized scenarios, tensors in x and out must be 2D with shapes ($m_i$, $k_i$) and ($m_i$, $n_i$) respectively. In pseudo-quantized scenarios, tensors in x must have consistent dimensions, supporting 2–6 dimensions; tensors in y must have the same dimensions as x. Tensors in weight must be 2D with shape ($n_i$, $k_i$) or ($k_i$, $n_i$). Tensors in bias must be 1D with shape ($n_i$).<br>3) groupListOptional must be passed as empty.<br>4) Weight transposition is supported, but whether each tensor in the weight tensorList is transposed must be consistent.<br>5) x does not support transposition.<br>6) Only non-quantized and pseudo-quantized scenarios are supported.<br>7) Only ND-in ND-out is supported.<br>|
          | 0 | single-single-single |1) Only splitItem 2/3 is supported<br>2) Tensors in weight must be 3D, with shapes (g, N, K) or (g, K, N); tensors in x and out must be 2D, with shapes (M, K) and (M, N) respectively; tensors in bias must be 2D, with shape (g, N)<br>3) groupListOptional must be passed in. If groupListType is 0, the last value must not be greater than the first dimension of the tensor in x; if groupListType is 1, the sum of the values must not be greater than the first dimension of the tensor in x; if groupListType is 2, the sum of the values in the second column must not be greater than the first dimension of the tensor in x<br>4) The first dimension of groupListOptional supports up to 1024, which means a maximum of 1024 groups<br>5) x without transpose is supported, and weight can be transposed or not<br>6) Only ND input and ND output are supported<br>|
          | 0 | single-multiple-single |1) Only splitItem 2/3 is supported<br>2) groupListOptional must be passed in. If groupListType is 0, the last value must be equal to the first dimension of the tensor in x; if groupListType is 1, the sum of the values must be equal to the first dimension of the tensor in x, and the maximum length is 1024<br>3) Tensors in x and out must be 2D, with shapes (M, K) and (M, N) respectively; tensors in weight must be 2D, with shapes (N, K) or (K, N); tensors in bias must be 1D, with shape (N)<br>4) The N-dimension size of each tensor in weight must be equal<br>5) weight transpose is supported, but whether each tensor in the tensorList of weight is transposed must be consistent<br>6) x transpose is not supported<br>7) Only non-quantization is supported<br>8) Only ND input and ND output are supported<br> |
          | 0 | multiple-multiple-single |1) Only splitItem 2 is supported.<br>2) Tensors in x and out must be 2D with shapes (M, K) and (M, N) respectively. Tensors in weight must be 2D with shape (N, K) or (K, N). Tensors in bias must be 1D with shape (N).<br>3) The N axis of each tensor in weight must be equal.<br>4) If groupListOptional is passed, when groupListType is 0, the differences of groupListOptional must correspond one-to-one with the first dimension of the tensors in x. When groupListType is 1, the values of groupListOptional must correspond one-to-one with the first dimension of the tensors in x, with a maximum length of 1024.<br>5) Weight transposition is supported, but whether each tensor in the weight tensorList is transposed must be consistent.<br>6) x does not support transposition.<br>7) Only non-quantized scenarios are supported.<br>8) Only ND-in ND-out is supported.<br> |
          | 2 | single-single-single |1) Only splitItem 2/3 is supported<br>2) Tensors in x and weight must be 2D, with shapes (K, M) and (K, N) respectively; tensors in out must be 3D, with shape (g, M, N)<br>3) groupListOptional must be passed in. If groupListType is 0, the last value must not be greater than the first dimension of the tensor in x; if groupListType is 1, the sum of the values must not be greater than the first dimension of the tensor in x<br>4) The first dimension of groupListOptional supports up to 1024, which means a maximum of 1024 groups<br>5) x must be transposed, and weight must not be transposed<br>6) Only non-quantization and quantization are supported<br>7) Only ND input and ND output are supported<br>8) bias is not supported|
          | 2 | Single Multi Multi |1) Only splitItem 0/1 is supported<br>2) Tensors in x and weight must be 2D, with shapes (K, M) and (K, N) respectively; tensors in y must be 2D, with shape (M, N)<br>3) groupListOptional can be empty. If groupListOptional is passed in and groupListType is 0, the last value must not be greater than the first dimension of the tensor in x; if groupListType is 1, the sum of the values must not be greater than the first dimension of the tensor in x<br>4) The first dimension of groupListOptional supports up to 1024, which means a maximum of 1024 groups<br>5) x must be transposed, and weight must not be transposed<br>6) Only ND input and ND output are supported<br>7) bias is not supported<br>8) Only non-quantization is supported|

    </details>

</details>
<!-- end id14 -->

## Example

The sample code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
    // Boilerplate, resource initialization.
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

    // Call aclrtMemcpy to copy data from Host to memory on the Device.
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
    // Call aclrtMalloc to allocate memory on the Device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host to the device memory.
    std::vector<T> hostData(size, 0);
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API reference.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.
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
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the group_list aclTensor.
    ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library APIs.
    // Call the first-phase API of aclnnGroupedMatmulV4.
    ret = aclnnGroupedMatmulV4GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput, activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, out, activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmulV4.
    ret = aclnnGroupedMatmulV4(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV4 failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the results from device memory to the host. Modify the code according to the specific API definition.
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

    // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(out);

    // 7. Release device resources. Modify the code according to the specific API definition.
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
