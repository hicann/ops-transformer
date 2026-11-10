# aclnnGroupedMatmulWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:02:08.875Z -->

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

  - **Description**: Implements grouped matrix multiplication. The dimension sizes of matrix multiplication in each group can be different. The basic operation is matrix multiplication, for example, $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where g is the number of groups and $m_i/k_i/n_i$ are the corresponding shapes. The input and output data types are both aclTensorList. The corresponding functions are:

      - k-axis grouping: $k_i$ values differ, but $m_i/n_i$ are the same for each group. In this case, $x_i/weight_i$ can be concatenated along $k_i$.
      - m-axis grouping: $k_i$ is the same across groups, and $weight_i/y_i$ can be concatenated along $n_i$.

    **New features compared with [GroupedMatmulV5](aclnnGroupedMatmulV5_en.md)**:

      - The input weight is parsed by the API in the data layout format preferred by the AI processor (FRACTAL_NZ).
      - A new parameter quantGroupSize, an integer parameter, represents the group size for per-group quantization. Set it to 0 when per-group quantization is not involved.
      <!-- npu="950" id7 -->
      - <term>Ascend 950PR&950DT products</term>: Supports the quantGroupSize parameter. For comprehensive constraints, see <a href="#constraints">Constraints</a>.
      <!-- end id7 -->

  - **Formula**:

      <a id="non-quantized-scenario"></a>

      - **Non-quantized scenario:**

        $$
        y_i=x_i \times weight_i + bias_i
        $$

      <a id="quantized-scenario"></a>

      - **Quantized scenario (without perTokenScaleOptional):**

        - x is INT8, bias is INT32

          $$
          y_i=(x_i \times weight_i + bias_i) * scale_i + offset_i
          $$

        - x is INT8, bias is BFLOAT16/FLOAT16/FLOAT32, no offset

          $$
          y_i=(x_i \times weight_i) * scale_i + bias_i
          $$

      - **Quantized scenario (with perTokenScaleOptional):**

        - x is INT8, bias is INT32

          $$
          y_i=(x_i \times weight_i + bias_i) * scale_i * per\_token\_scale_i
          $$

        - x is INT8, bias is BFLOAT16/FLOAT16/FLOAT32

          $$
          y_i=(x_i \times weight_i) * scale_i * per\_token\_scale_i  + bias_i
          $$

      - **Quantization scenario (mx quantization):**
        $$
        y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (xSlice_i * weightSlice_i)) * (per\_token\_scale_i[m/gsM, j] * scale_i[j, n/gsN])) + bias_i[n]
        $$
        where gsM, gsN, and gsK represent the quantization block sizes along the M/N/K axes, respectively. $xSlice_i$ represents the gsK-length vector of the m-th row of $x_i$. $weightSlice_i$ represents the gsK-length vector of the n-th column of $weight_i$. Slices along the K axis all start at j * gsK. The value range of j is [0, kLoops), and kLoops=ceil($K_i$ / gsK). The last slice can be shorter than gsK.

      <a id="dequantized-scenario"></a>

      - **Dequantized scenario:**

        $$
        y_i=(x_i \times weight_i + bias_i) * scale_i
        $$

      <a id="pseudo-quantized-scenario"></a>

      - **Pseudo-quantized (per-channel, per-group) scenario:**

        $$
        y_i=x_i \times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
        $$

      - **Pseudo-quantized (mx) scenario:**

        x is a BFLOAT16/FLOAT16 input, and weight is a FLOAT32 (representing eight FLOAT4_E2M1 values)/FLOAT4_E2M1 input.

        $$
        y_i=x_i \times (weight_i  * antiquant\_scale_i) + bias_i
        $$

        x is FLOAT8_E4M3FN input, and weight is FLOAT32 (representing 8 FLOAT4_E2M1 values)/FLOAT4_E2M1/FLOAT4_E1M2 input

        $$
        y_i=(x_i * per\_token\_scale_i) \times (weight_i  * antiquant\_scale_i) + bias_i
        $$

      - **Pseudo-quantized (K-CG) scenario:**

        $$
        y_i=(x_i \times (weight_i * antiquant\_scale_i)) * scale_i * per\_token\_scale_i + bias_i
        $$

        where antiquant\_scale_i is the pergroup quantization parameter of the weight matrix, scale_i is the perchannel quantization parameter of the weight matrix, and per\_token\_scale_i is the
        pertoken quantization parameter.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the aclnnGroupedMatmulWeightNzGetWorkspaceSize API to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call the aclnnGroupedMatmulWeightNz API to perform the computation.

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

  - **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1483px;">
    <colgroup>
    <col style="width: 240px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x (aclTensorList *)</td>
    <td>Input</td>
    <td>The input <code>x</code> in the formula.</td>
    <td>The tensorList length supports [1, 128] or [1, 1024].</td>
    <td>FLOAT16, BFLOAT16, INT8, INT4, INT32<sup>1</sup>, FLOAT8_E4M3FN<sup>2</sup>, FLOAT4_E2M1<sup>2</sup>, FLOAT4_E1M2<sup>2</sup></td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weight(aclTensorList *)</td>
    <td>Input</td>
    <td>The <code>weight</code> in the formula.</td>
    <td>The length of tensorList supports [1, 128] or [1, 1024]. The API parses this parameter in the Ascend-affine data layout format (FRACTAL_NZ).</td>
    <td>FLOAT16, BFLOAT16, INT8, INT4, INT32, FLOAT32, FLOAT4_E2M1<sup>2</sup>, FLOAT4_E1M2<sup>2</sup>,
    FLOAT8_E4M3FN<sup>2</sup></td>
    <td>FRACTAL_NZ</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>biasOptional (aclTensorList *)</td>
    <td>Optional input</td>
    <td><code>bias</code> in the formula.</td>
    <td>The length is the same as that of weight.</td>
    <td>FLOAT16, FLOAT32, INT32, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleOptional (aclTensorList *)</td>
    <td>Optional input.</td>
    <td><code>scale</code> in the formula, which represents the scaling factor in quantization parameters.</td>
    <td>Generally, the length is the same as that of weight. In the mx quantization scenario, scaleOptional supports the FRACTAL_NZ format. For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
    <td>UINT64, INT64, BFLOAT16, FLOAT32, FLOAT8_E8M0<sup>2</sup></td>
    <td>ND/FRACTAL_NZ</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>offsetOptional (aclTensorList *)</td>
    <td>Optional input</td>
    <td><code>offset</code> in the formula, which represents the offset in quantization parameters.</td>
    <td>Same length as weight.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>antiquantScaleOptional (aclTensorList *)</td>
    <td>Optional input</td>
    <td><code>antiquant_scale</code> in the formula, representing the scaling factor in the pseudo-quantization parameters.</td>
    <td>The length is the same as that of weight. For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16<sup>1</sup>, FLOAT8_E8M0<sup>2</sup></td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>antiquantOffsetOptional (aclTensorList *)</td>
    <td>Optional input</td>
    <td><code>antiquant_offset</code> in the formula, representing the offset in the pseudo-quantization parameters.</td>
    <td>The length is the same as weight. For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>perTokenScaleOptional (aclTensorList *)</td>
    <td>Optional input</td>
    <td><code>per_token_scale</code> in the formula, which represents the scaling factor introduced by x quantization in the quantization parameters.</td>
    <td>Only supports the use case where x and out are single Tensors; weight can be multiple Tensors in the dynamic quantization (mx quantization) scenario and the MxA8W4 pseudo-quantization scenario. For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT32, FLOAT8_E8M0<sup>2</sup></td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListOptional (aclTensor *)</td>
    <td>Optional input</td>
    <td>Represents the matmul size distribution of the input and output along the grouping axis.</td>
    <td>Input data in different formats based on groupListType. Note: When the output TensorList length is 1, the last value constrains the valid part of the output.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>activationInputOptional(aclTensorList *)</td>
    <td>Optional input</td>
    <td>Represents the backward input of the activation function.</td>
    <td>Currently only supports passing nullptr.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>activationQuantScaleOptional(aclTensorList *)</td>
    <td>Optional input</td>
    <td>-</td>
    <td>Currently only supports passing in nullptr.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>activationQuantOffsetOptional (aclTensorList *)</td>
    <td>Optional input</td>
    <td>-</td>
    <td>Currently only supports passing in nullptr.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>-</td>
    <td>Input</td>
    <td>Represents whether the output needs to be split into tensors.</td>
    <td>0/1 represents that the output is multiple tensors; 2/3 represents that the output is a single Tensor.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupType</td>
    <td>Input</td>
    <td>Represents the axis to be grouped.</td>
    <td>Enumeration values: -1, 0, 2. For matrix multiplication C[m,n]=A[m,k]xB[k,n], the values of groupType are -1: no grouping, 0: grouping on the m axis, 2: grouping on the k axis.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListType</td>
    <td>Input</td>
    <td>Represents the grouping mode of the groupList input.</td>
    <td>0: cumsum result; 1: size of each group; 2: [groupIdx, groupSize]. For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actType</td>
    <td>Input</td>
    <td>Represents the activation function type.</td>
    <td>Value range: 0-5.<br>
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
    <td>tuningConfigOptional(aclIntArray *)</td>
    <td>Optional input</td>
    <td>The first number represents the expected number of tokens processed by each expert; the second number indicates whether to enable the weight affinity format (transpose first, then NZ). Currently, only the <a href="#ascend950-weightnz-s8s4-scenario-constraints">S8S4</a> use case is supported.</td>
    <td>If this parameter is not used, do not pass it (that is, pass nullptr).</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantGroupSize</td>
    <td>Input</td>
    <td>Represents the group size for grouped quantization (per-group).</td>
    <td>Set this parameter to 0 if grouped quantization is not involved. For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>out (aclTensorList *)</td>
    <td>Output</td>
    <td>Output <code>y</code> in the formula.</td>
    <td>The tensorList length support is [1, 128] or [1, 1024].</td>
    <td>FLOAT16, BFLOAT16, FLOAT32, INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>activationFeatureOutOptional (aclTensorList *)</td>
    <td>Output</td>
    <td>Input data of the activation function.</td>
    <td>Currently only supports nullptr as the input.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dynQuantScaleOutOptional (aclTensorList *)</td>
    <td>Output</td>
    <td>-</td>
    <td>Currently only supports passing nullptr.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t *)</td>
    <td>Output</td>
    <td>Returns the workspace size to be applied on the Device side.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor (aclOpExecutor **)</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the operator computation process.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>:
      - The superscript "1" in the data type column of the preceding table represents data types not supported by the series.
      - `weight` supports the FRACTAL_NZ format. When either of the last two axes is 1 (that is, n=1 or k=1), the private format is not supported and this API cannot be called. You can call the aclnnNpuFormatCast API to convert the input format from ND to the AI processor-preferred data layout format (NZ). If the original weight is transposed and you want to use the higher-performance non-transposed path for computation, you can call the aclnnPermute API to convert weight to the non-transposed state before calling the aclnnNpuFormatCast API. In non-full quantization scenarios, when the data type is FLOAT4_E2M1, you also need to call the aclnnCast API after calling aclnnNpuFormatCast to convert FLOAT4_E2M1 represented by FLOAT32 to the correct type. However, if the type is INT4, you need to use the aclnnConvertWeightToInt4Pack API to convert the data format from ND to NZ and the data type from INT32 to INT4. When FLOAT32 or INT32 is passed in, the API internally identifies each FLOAT32/INT32 as 8 FLOAT4_E2M1/INT4 values.
      - `offsetOptional` is supported, while `antiquantOffsetOptional` is not supported yet. For comprehensive constraints, see <a href="#constraints">Constraints</a>.
      - `groupType` supports m-axis grouping. No grouping is supported only in the non-quantized scenario.
      - `quantGroupSize` is supported. For comprehensive constraints, see <a href="#constraints">Constraints</a>.
      - `actType` supports 0, 1, 2, 4, and 5. For comprehensive constraints, see <a href="#constraints">Constraints</a>.
      - In non-quantized scenarios, input parameters `x` and `weight` and output parameter `out` support up to 1024 tensors; in pseudo-quantization scenarios, up to 128 tensors. In full quantization scenarios, input parameter `x` and output parameter `out` support up to 1 tensor. In dynamic quantization (mx quantization) scenarios, input parameter `weight` supports multiple tensors, up to 1024 tensors.
    <!-- end id8 -->

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
      - The superscript "2" in the data type column of the preceding table represents data types not supported by the series.
      - `weight` is parsed by the API in FRACTAL_NZ format. When INT32 is passed in, the API internally recognizes each INT32 as 8 INT4 values.
      - The input parameters `x` and `weight`, and the output parameter `out` support up to 128 tensors.
    <!-- end id9 -->

  - **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API validates input parameters. If the following error codes occur, the corresponding causes are:

    <table>
    <thead>
    <tr>
    <th style="width: 250px">Return Value</th>
    <th style="width: 130px">Error Code</th>
    <th style="width: 850px">Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td rowspan="4"> ACLNN_ERR_PARAM_NULLPTR </td>
    <td rowspan="4"> 161001 </td>
    <td>1. The input parameter is a required input, output, or required attribute, and is a null pointer.</td>
    </tr>
    <tr>
    <td>2. An element of the input parameter weight is a null pointer.</td>
    </tr>
    <tr>
    <td>3. The element of input parameter x is a null pointer, and the element of output parameter out is not a null pointer.</td>
    </tr>
    <tr>
    <td>4. The element of input parameter x is not a null pointer, and the element of output parameter out is a null pointer.</td>
    </tr>
    <tr>
    <td rowspan="7"> ACLNN_ERR_PARAM_INVALID </td>
    <td rowspan="7"> 161002 </td>
    <td>1. The data types and data formats of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, and out are not in the supported range.</td>
    </tr>
    <tr>
    <td>2. The length of weight is not in the supported range.</td>
    </tr>
    <tr>
    <td>3. If bias is not empty, the length of bias is not equal to the length of weight.</td>
    </tr>
    <tr>
    <td>4. The dimension of groupListOptional is 1.</td>
    </tr>
    <tr>
    <td>5. In use cases where splitItem is 2 or 3, the length of out is not equal to 1.</td>
    </tr>
    <tr>
    <td>6. In use cases where splitItem is 0 or 1, the length of out is not equal to the length of weight, or the length of groupListOptional is not equal to the length of weight.</td>
    </tr>
    <tr>
    <td>7. The elements of input parameter tuningConfigOptional are negative, or greater than the number of rows m of x.</td>
    </tr>
    </tbody>
    </table>

## aclnnGroupedMatmulWeightNz

  - **Parameters:**

    |Name| Input/Output   |    Description|
    |-------|---------|----------------|
    |workspace|Input|Device-side allocated workspace memory address.|
    |workspaceSize|Input|Size of the workspace applied on the Device side, obtained from the first-phase API aclnnGroupedMatmulWeightNzGetWorkspaceSize.|
    |executor|Input|Op executor that contains the operator computation flow.|
    |stream|Input|Stream on which the task is executed.|

  - **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupedMatmulWeightNz defaults to a deterministic implementation.

<!-- npu="A3,910b" id10 -->
<details>
<summary><term>Atlas A2 products</term>, <term>Atlas A3 products</term></summary>

  - General Constraints
    - If groupListOptional is passed, when groupListType is 0, groupListOptional must be a non-negative monotonically non-decreasing sequence; when groupListType is 1, groupListOptional must be a non-negative sequence and its length cannot be 1; when groupListType is 2, the second column of groupListOptional must be a non-negative sequence and its length cannot be 1.
    - The size of each dimension of each group of tensors in x and weight, after 32-byte alignment, must be less than the maximum int32 value 2147483647.
    - actType (int64_t, computation input): An integer parameter that represents the activation function type, with a value range of 0 to 5.
      - In the quantization scenario, if the current quantization combination does not support the activation function, values 1, 2, 4, and 5 only trigger a warning and are not intercepted. Passing 0 is recommended.

  - The supported input types for the non-quantized scenario are:

    - x is FLOAT16, weight is FLOAT16, biasOptional is FLOAT16, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, perTokenScaleOptional is empty, activationInputOptional is empty, and out is FLOAT16.
    - x is BFLOAT16, weight is BFLOAT16, biasOptional is FLOAT32, scaleOptional is empty, offsetOptional is empty, antiQuantScaleOptional is empty, antiQuantOffsetOptional is empty, perTokenScaleOptional is empty, activationInputOptional is empty, and out is BFLOAT16.

  - The input types supported in the quantized scenario are:

    - x is INT8, weight is INT8, biasOptional is INT32, scaleOptional is BFLOAT16, offsetOptional is empty, antiQuantScaleOptional is empty, antiQuantOffsetOptional is empty, perTokenScaleOptional is empty or FLOAT32, activationInputOptional is empty, and out is BFLOAT16.
    - x is INT8, weight is INT8, biasOptional is INT32, scaleOptional is FLOAT32, offsetOptional is empty, antiQuantScaleOptional is empty, antiQuantOffsetOptional is empty, perTokenScaleOptional is empty or FLOAT32, activationInputOptional is empty, and out is FLOAT16.
    - x is INT4, weight is INT4, biasOptional is empty, scaleOptional is UINT64, offsetOptional is empty, antiQuantScaleOptional is empty, antiQuantOffsetOptional is empty, perTokenScaleOptional is empty or FLOAT32, activationInputOptional is empty, and out is FLOAT16 or BFLOAT16. Weight supports NZ transpose input, that is, the input is [E,N,K] but the view shape is [E,K,N] to ensure that the operator recognizes the transpose state. For transpose input, $k/G$ must be aligned to 64, K must be aligned to 64, and N must be aligned to 16. ND does not support transpose input.
    - weight supports the single Tensor use case (tensorlist length is 1) and the multi-Tensor use case (tensorlist length greater than 1). The multi-Tensor use case is supported only for MX quantization.

  - The input types supported in the pseudo-quantized scenario are:

    - The shapes of the pseudo-quantization parameters antiQuantScaleOptional and antiQuantOffsetOptional must satisfy the following table (where g is the number of matmul groups, G is the number of pergroup, and $G_i$ is the number of pergroup of the i-th tensor):

        | usage scenario | Sub-Scenario | Shape Constraint |
        |:---------:|:-------:| :-------|
        | Pseudo-quantized perchannel | Single weight | $[E, N]$|
        | Pseudo-quantized perchannel | Multiple weights | $[n_i]$|
        | Pseudo-quantized pergroup | Single weight | $[E, G, N]$|
        | Pseudo-quantized pergroup | Multiple weights | $[G_i, N_i]$|

    - x is INT8, weight is INT4, biasOptional is FLOAT32, scaleOptional is UINT64, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, perTokenScaleOptional is FLOAT32, and activationInputOptional is empty.

      - weight, scaleOptional, biasOptional, and offsetOptional support the single Tensor use case (tensorlist length is 1) and the multi-Tensor use case (tensorlist length greater than 1).

      - This use case supports symmetric quantization and asymmetric quantization:

      - Symmetric quantization scenario:

        - The dtype of the output out is BFLOAT16 or FLOAT16.
        - offsetOptional is empty.
        - Only the count mode is supported (the operator does not check the value of groupListType). k must be an integer multiple of quantGroupSize, and k <= 18432. quantGroupSize is the per-group quantization length in the k direction. Currently, quantGroupSize=256 is supported.
        - scale is the result of offline fusion of per-group and per-channel quantization. The shape must be $[E, quantGroupNum, N]$, where $quantGroupNum=k \div quantGroupSize$.
        - Bias is an auxiliary result computed offline during the computation process, with the value required to be $8\times weight \times scale$ and accumulated along the first dimension, and the shape required to be $[E, N]$.
        - N must be an integer multiple of 8.

      - Asymmetric quantization scenario:

        - The dtype of the output out is FLOAT16 or BFLOAT16.
        - Only the count mode is supported (the operator does not check the value of groupListType).
        - {k, n} must be {7168, 4096} or {2048, 7168}.
        - scale is the result of offline fusion of pergroup and perchannel, with the shape required to be $[E, quantGroupNum, N]$, where $quantGroupNum = k \div quantGroupSize$. Currently, quantGroupSize values of 64, 128, and 256 are supported.
        - offsetOptional is not empty. In asymmetric quantization, offsetOptional is an auxiliary result computed offline during the computation process, i.e., $antiquantOffset \times scale$, with the shape required to be $[E, quantGroupNum, N]$ and dtype being FLOAT32.
        - Bias is an auxiliary result computed offline during the computation process, with the value required to be $8\times weight \times scale$ and accumulated along the first dimension, and the shape required to be $[E, N]$.
        - N must be an integer multiple of 8.
        - The second element of tuningConfigOptional cannot be enabled.

    - In the pseudo-quantized scenario, if the weight type is INT8, only perchannel mode is supported; if the weight type is INT4, symmetric quantization supports both perchannel and pergroup modes. For pergroup, the number of pergroups G or $G_i$ must be divisible by the corresponding $k_i$. If the weight is multi-tensor, define the pergroup length $s_i = k_i / G_i$, and all $s_i(i=1,2,...g)$ must be equal. Asymmetric quantization supports perchannel mode.

    - In the pseudo-quantized scenario, if the weight type is INT4, the size of the last dimension of each tensor group in weight must be an even number. The last dimension of $weight_i$ refers to the N axis of $weight_i$ when weight is not transposed, or the K axis of $weight_i$ when weight is transposed. In the pergroup scenario, when weight is transposed, the pergroup length $s_i$ must be an even number.

  - Supported scenarios for different groupType values:
    - Quantized and pseudo-quantized scenarios only support groupType values of -1 and 0.
    - In the supported scenarios, "single" indicates a single tensor, "multiple" indicates multiple tensors, and the order is x, weight, y. For example, "single-multiple-single" indicates a scenario where x is a single tensor, weight consists of multiple tensors, and y is a single tensor.

      | groupType | Supported Scenarios | scenario restriction |
      |:---------:|:---------:| :-------|
      | -1 | multiple-multiple-multiple |1) Only splitItem 0/1 is supported.<br>2) Tensors in x must have the same number of dimensions, supporting 2 to 6 dimensions. Tensors in weight must be 2-dimensional. Tensors in y must have the same number of dimensions as those in x.<br>3) groupListOptional must be passed as null.<br>4) Weight transposition is supported, but whether each tensor in the weight tensorList is transposed must be consistent.<br>5) x does not support transposition. |
      | 0 | single-single-single |1) Only splitItem 2/3 is supported.<br>2) Tensors in weight must be 3-dimensional. Tensors in x and y must be 2-dimensional.<br>3) groupListOptional must be passed. When groupListType is 0, the last value must equal the first dimension of the tensor in x. When groupListType is 1, the sum of all values must equal the first dimension of the tensor in x. When groupListType is 2, the sum of the values in the second column must equal the first dimension of the tensor in x.<br>4) The first dimension of groupListOptional supports a maximum of 1024, meaning up to 1024 groups are supported.<br>5) Weight transposition is supported.<br>6) x does not support transposition. |
      | 0 | single-multiple-single |1) Only splitItem 2/3 is supported.<br>2) groupListOptional must be passed. When groupListType is 0, the last value must equal the first dimension of the tensor in x. When groupListType is 1, the sum of all values must equal the first dimension of the tensor in x, and the maximum length is 128. When groupListType is 2, the sum of the values in the second column must equal the first dimension of the tensor in x, and the maximum length is 128.<br>3) Tensors in x, weight, and y must be 2-dimensional.<br>4) The N axis of each tensor in weight must be equal.<br>5) Weight transposition is supported, but whether each tensor in the weight tensorList is transposed must be consistent.<br>6) x does not support transposition. |
      | 0 | multiple-multiple-single |1) Only splitItem 2/3 is supported.<br>2) Tensors in x, weight, and y must be 2-dimensional.<br>3) The N axis of each tensor in weight must be equal.<br>4) If groupListOptional is passed: when groupListType is 0, the differences of groupListOptional must correspond one-to-one with the first dimension of the tensors in x. When groupListType is 1, the values of groupListOptional must correspond one-to-one with the first dimension of the tensors in x, and the maximum length is 128. When groupListType is 2, the values in the second column of groupListOptional must correspond one-to-one with the first dimension of the tensors in x, and the maximum length is 128.<br>5) Weight transposition is supported, but whether each tensor in the weight tensorList is transposed must be consistent.<br>6) x does not support transposition. |

</details>
<!-- end id10 -->

<!-- npu="950" id11 -->
<details>
<summary><term>Ascend 950PR&950DT products</term></summary>

  - General Constraints
    - groupListType: The value can be 0, 1, or 2.
      - If groupListType is 0, groupListOptional must be a monotonically non-decreasing sequence of non-negative numbers;
      - If groupListType is 1, groupListOptional must be a sequence of non-negative numbers.
      - groupListType 2 is supported only when full quantization is used and groupType is 0. In this case, groupListOptional must be a sequence of non-negative numbers with a shape of [E, 2], where E indicates the Group size. The data layout is [[groupIdx0, groupSize0], [groupIdx1, groupSize1]...], where groupSize indicates the size of each group on the grouping axis. Groups with a groupSize of 0 are placed at the end of groupList, and non-zero groups are placed at the front. For details, see the groupListOptional configuration example.
    - The size of each dimension of each group of tensors in x and weight, after 32-byte alignment, must be less than the maximum int32 value 2147483647.
    - actType (int64_t, computation input): An integer parameter that represents the activation function type, with a value range of 0 to 5.
      - In pseudo-quantized and non-quantized scenarios, actType only supports 0.
      - In fully quantized scenarios, when x and weight are INT8, the quantization mode is static T-C quantization or dynamic K-C quantization, and the scale data type is FLOAT32 or BFLOAT16, actType supports 0, 1, 2, 4, and 5. In all other fully quantized scenarios, actType only supports 0.
  - Currently supports non-quantized, pseudo-quantized, and fully quantized scenarios.
  - The data types supported in non-quantized scenarios are:
    - The n and k dimensions of the input weight matrix must satisfy 32-byte alignment.
    - The following input parameters are empty: scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional, activationFeatureOutOptional.
    - The supported data type combinations for non-empty parameters must satisfy the following table.

      |groupType| x       | weight  | biasOptional | out     |
      |:-------:|:-------:|:-------:| :------      |:------ |
      |-1/0   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
      |-1/0   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|

  - The data types supported in the pseudo-quantized scenario are:
    <a id="ascend950-weightnz-s8s4-scenario-constraints"></a>

    **S8S4 use case:**

    - The following input parameters are empty: antiquantScaleOptional, antiquantOffsetOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional, and activationFeatureOutOptional
    - The supported data type combinations for non-empty parameters must satisfy the following table:

      |groupType| x | weight | biasOptional | scaleOptional | offsetOptional | perTokenScaleOptional | groupListOptional | out |
      |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
      |0|INT8|INT4|FLOAT|UINT64|null|FLOAT|INT64|BFLOAT16|
      |0|INT8|INT4|FLOAT|UINT64|FLOAT/null|FLOAT|INT64|FLOAT16|

    - scaleOptional and offsetOptional must satisfy the following table (where E is the number of groups):

      | Sub-Scenario | offsetOptional | scaleOptional shape | offsetOptional shape | quantGroupSize | out |
      |:---:|:---:|:---:|:---:|:---:|:---:|
      | Symmetric per-group | null | `[E, K/256, N]` | - | 256 | FLOAT16/BFLOAT16 |
      | Asymmetric per-channel | FLOAT32 | `[E, 1, N]` | `[E, 1, N]` | 0 | FLOAT16 |

    - Constraints:

      In addition to the common constraints, the other constraints of the S8S4 use case are as follows:

      |groupType| splitItem | actType | groupListType |
      |:---:|:---:|:---:|:---|
      |0|2/3|0|1 (count)|

      | Input/Output | Sub-Scenario | Shape Constraint |
      |:---:|:---|:---|
      | x | single Tensor | 2D, shape `[M,K]`; does not support transposition. |
      | weight | single Tensor | 3D, shape `[E,K,N]`; does not support transposition. |
      | biasOptional | Required input | 2D, shape `[E,N]`, data type FLOAT32. |
      | perTokenScaleOptional | single Tensor | 1D, shape `[M]`. |
      | out | single Tensor | - |

      - x, weight, scaleOptional, biasOptional, perTokenScaleOptional, and out are all TensorLists of length 1.
      - biasOptional is a required input and is the correction value for offline conversion of INT4 weights. It is obtained by reducing `8 × weight × scale` along the K axis.
      - In symmetric per-group scenarios, K must not be greater than 18432 and must be an integer multiple of 256.
      - In the S8S4 per-channel scenario where offsetOptional is not empty, the second element of the tuningConfigOptional array can be set to 1. In this case, weight must be laid out in the `[E,N,K]` layout and converted to NZ. Only a weight TensorList of length 1 is supported.

    **Other pseudo-quantization scenarios:**
    - The following input parameters are empty: offsetOptional, antiquantOffsetOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional, activationFeatureOutOptional
    - The supported data type combinations for non-empty parameters must satisfy the following table.

      |groupType| x       |perTokenScaleOptional| weight  |antiquantScaleOptional|scaleOptional|antiquantOffsetOptional| biasOptional | out     | perTokenScaleOptional Shape | weight Shape | antiquantScaleOptional Shape| scaleOptional shape|bias shape|
      |:-------:|:-------:|:-------------------:|:-------:|:--------------------:|:-----------:|:---------------------:|:------------:|:-------:|:---------------------------:|:------------:|:---------------------------:|:------------------:|:--------:|
      |0   |BFLOAT16      |null          |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null |BFLOAT16/FLOAT32/null     |BFLOAT16 |null             |(g, K, N)   |(g, K/groupSize, N) |null   |(g, N) |
      |0   |FLOAT16       |null          |FLOAT4_E2M1     |FLOAT8_E8M0 |null    |null |FLOAT16/null              |FLOAT16  |null             |(g, K, N)   |(g, K/groupSize, N) |null   |(g, N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT4_E2M1/FLOAT4_E1M2 |FLOAT8_E8M0 |null |null |FLOAT16/null  |FLOAT16  |(M, K/groupSize/2, 2) |(g, N, K) |(g, N, K/groupSize/2, 2) |null |(g, N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT4_E2M1/FLOAT4_E1M2 |FLOAT8_E8M0 |null |null |BFLOAT16/null |BFLOAT16 |(M, K/groupSize/2, 2) |(g, N, K) |(g, N, K/groupSize/2, 2) |null |(g, N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT4_E2M1/FLOAT4_E1M2 |FLOAT8_E8M0 |null |null |FLOAT16/null  |FLOAT16  |(M, K/groupSize/2, 2) |(N, K) |(N, K/groupSize/2, 2) |null |(N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT4_E2M1/FLOAT4_E1M2 |FLOAT8_E8M0 |null |null |BFLOAT16/null |BFLOAT16 |(M, K/groupSize/2, 2) |(N, K) |(N, K/groupSize/2, 2) |null |(N) |
      |0   |INT8          |FLOAT32       |INT4            |FLOAT16     |FLOAT32 |null |FLOAT32/null              |BFLOAT16 |(M)              |(g, K, N)   |(g, K/groupSize, N) |(g, N) |(g, N) |
      |0   |INT8          |FLOAT32       |INT4            |FLOAT16     |FLOAT32 |null |FLOAT32/null              |FLOAT16  |(M)              |(g, K, N)   |(g, K/groupSize, N) |(g, N) |(g, N) |
      |0   |BFLOAT16      |null          |FLOAT32         |FLOAT8_E8M0 |null    |null |BFLOAT16/FLOAT32/null     |BFLOAT16 |null             |(g, K, N/8) |(g, K/groupSize, N) |null   |(g, N) |
      |0   |FLOAT16       |null          |FLOAT32         |FLOAT8_E8M0 |null    |null |FLOAT16/null              |FLOAT16  |null             |(g, K, N/8) |(g, K/groupSize, N) |null   |(g, N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT32         |FLOAT8_E8M0 |null    |null |FLOAT16/null              |FLOAT16  |(M, K/groupSize/2, 2) |(g, N, K/8) |(g, N, K/groupSize/2, 2) |null   |(g, N) |
      |0   |FLOAT8_E4M3FN |FLOAT8_E8M0   |FLOAT32         |FLOAT8_E8M0 |null    |null |BFLOAT16/null             |BFLOAT16 |(M, K/groupSize/2, 2) |(g, N, K/8) |(g, N, K/groupSize/2, 2) |null   |(g, N) |
      |0   |INT8          |FLOAT32       |INT32           |FLOAT16     |FLOAT32 |null |FLOAT32/null              |BFLOAT16 |(M)              |(g, K, N/8) |(g, K/groupSize, N) |(g, N) |(g, N) |
      |0   |INT8          |FLOAT32       |INT32           |FLOAT16     |FLOAT32 |null |FLOAT32/null              |FLOAT16  |(M)              |(g, K, N/8) |(g, K/groupSize, N) |(g, N) |(g, N) |

    - Constraints:
      - In the use case where x is FLOAT16/BFLOAT16 and weight is FLOAT4_E2M1/FLOAT32, only 32 is supported for groupSize.
      - In the use case where x is FLOAT8_E4M3FN and weight is FLOAT4_E2M1/FLOAT4_E1M2/FLOAT32, only 32 is supported for groupSize.
      - When x is INT8 and weight is INT4/INT32, groupSize supports only 128, 192, 256, and 512.
      - The shape of x is fixed to (M, K), and the shape of out is fixed to (M, N).
      - In the mx pseudo-quantization and K-CG pseudo-quantization scenarios, both the K and N dimensions of weight must be 32B-aligned.
      - When x and weight are of types BFLOAT16/FLOAT16 and FLOAT4_E2M1/FLOAT32, or INT8 and INT4/INT32, only non-transposed x and weight are supported. When they are of types FLOAT8_E4M3FN and FLOAT4_E2M1/FLOAT4_E1M2/FLOAT32, only non-transposed x and transposed weight are supported.
      - Whether antiquantScale is transposed is consistent with weight.
      - In the use case where x is FLOAT8_E4M3FN, weight is FLOAT4_E2M1/FLOAT4_E1M2, and the weight shape is (N, K), it is a single-multi-single use case. weight supports multiple tensors (the number is greater than or equal to 1). The tensor numbers of antiquantScaleOptional, antiquantOffsetOptional, and biasOptional must be the same as that of weight.

  - The input data types and shapes supported by the static quantization scenario are as follows:
    - The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional, activationFeatureOutOptional
    - The supported data type combinations for non-empty parameters must satisfy the following table:

      |groupType| x       | weight  | biasOptional | scaleOptional | out     |
      |:-------:|:-------:|:-------:| :------      |:-------       | :------ |
      |0|INT8     |INT8     |INT32/null    | UINT64/INT64  |BFLOAT16/FLOAT16|
      |0|INT8     |INT8     |INT32/BFLOAT16/FLOAT32/null    | BFLOAT16/FLOAT32  |BFLOAT16|
      |0|INT8     |INT8     |INT32/FLOAT16/FLOAT32/null    | FLOAT32  |FLOAT16|
      |0|INT8     |INT8     |INT32/null    | UINT64/INT64/null  |INT32|

    - scaleOptional must meet the following table (where g is the number of matmul groups, i.e., the number of groups):

      |groupType| usage scenario | Shape Constraint |
      |:---------:|:---------:| :------ |
      |0|Single weight tensor|Per-channel scenario: each tensor is 2D with shape (g, N); per-tensor scenario: each tensor is 2D or 1D with shape (g, 1) or (g,)|

  - The dynamic quantization (K-T && K-C quantization) use case supports the following input data types and shapes:
    - The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional, activationFeatureOutOptional.
    - The supported data type combinations for non-empty parameters must satisfy the following table:

      |groupType| x       | weight  | biasOptional | scaleOptional | perTokenScaleOptional |out     |
      |:-------:|:-------:|:-------:| :------      |:-------    | :------   | :------ |
      |0|INT8  |INT8| INT32/BFLOAT16/FLOAT32/null     |BFLOAT16/FLOAT32    | FLOAT32   | BFLOAT16 |
      |0|INT8  |INT8| INT32/FLOAT16/FLOAT32/null     |FLOAT32    | FLOAT32   | FLOAT16 |
      |0|INT4  |INT4| null     |UINT64    | FLOAT32/null   | FLOAT16/BFLOAT16 |

    - scaleOptional must satisfy the following table (where g is the number of matmul groups, i.e., the group count):

      | groupType | usage scenario | Shape Constraint |
      |:---------:|:---------:| :------ |
      |0|Single weight tensor|Per-channel scenario: each tensor is 2-dimensional, with shape (g, N). Per-tensor scenario: each tensor is 2-dimensional or 1-dimensional, with shape (g, 1) or (g,).|

    - perTokenScaleOptional must meet the following table:

      | groupType | usage scenario | Shape Constraint |
      |:---------:|:---------:| :------ |
      |0|x single tensor|pertoken scenario: each tensor is 1D, with shape (M,)|

  - The K-G (pertoken-pergroup) quantization use case supports the following input data types and shapes:
    - The following input parameters must be null: biasOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional, activationFeatureOutOptional
    - The supported data type combinations for non-empty parameters must satisfy the following table:

      |groupType| x       | weight  | scaleOptional | perTokenScaleOptional |out     |
      |:-------:|:-------:|:-------:| :------      |:-------    | :------   |
      |0|INT4  |INT4     |UINT64    | FLOAT32/null   | FLOAT16/BFLOAT16 |

    - scaleOptional must satisfy the following table (where g is the number of matmul groups, i.e., the group count):

      | groupType | usage scenario | Shape Constraint |
      |:---------:|:---------:| :------ |
      |0|Single weight tensor|Each tensor is 3D with shape (E, G, N). $G$ must divide $K$, and $k/G$ must be an even number.|

    - perTokenScaleOptional must meet the following table:

      | groupType | usage scenario | Shape Constraint |
      |:---------:|:---------:| :------ |
      |0|x single tensor|pertoken scenario: each tensor is 1D, with shape (M,)|

    - Constraints:

      - For int4 weights in full quantization, NZ transposed input is supported. That is, the input shape is [E,N,K], but the view shape is [E,K,N] so that the operator can recognize the transposition state. For transposed input, $k/G$ and K must be aligned to 64, and N must be aligned to 16.

  - The dynamic quantization (mx quantization) use case supports the following input data types and shapes:
    - The following input parameters are empty: offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional, activationFeatureOutOptional.
    - The supported data type combinations for non-empty parameters must satisfy the following table:

        |groupType| x       | weight  | biasOptional | scaleOptional |  perTokenScaleOptional |out     |
        |:-------:|:-------:|:-------:|:-------:|:-------:| :-------    | :------ |
        |0|FLOAT8_E4M3FN  |FLOAT8_E4M3FN| null|   FLOAT8_E8M0    | FLOAT8_E8M0    | BFLOAT16/FLOAT16/FLOAT32 |
        |0|FLOAT4_E2M1/FLOAT4_E1M2 |FLOAT4_E2M1/FLOAT4_E1M2 | FLOAT32/null |   FLOAT8_E8M0    | FLOAT8_E8M0    | BFLOAT16/FLOAT16/FLOAT32 |

    - In the dynamic quantization (mx quantization) use case, x is a single tensor, weight is multiple tensors, and out is a single tensor. Currently, only the use cases where both x and weight are of the FLOAT8_E4M3FN type and where both x and weight are of the FLOAT4_E2M1/FLOAT4_E1M2 type are supported. In this case, passing in bias is not supported.
    - scaleOptional must satisfy the following table (where g is the number of matmul groups, that is, the number of groups, and g\_i is the i-th group (the subscript starts from 0)):

        |groupType| usage scenario | Shape Constraint |
        |:---------:|:---------:| :------ |
        |0|Single weight tensor|Each tensor is 4-dimensional. When weight is transposed, the shape is (g, N, ceil(K / 64), 2); when weight is not transposed, the shape is (g, ceil(K / 64), N, 2).|
        |0|Weight multi-tensor|Each tensor is 3-dimensional. When weight is transposed, the shape is (N, ceil(K / 64), 2); when not transposed, the shape is (ceil(K / 64), N, 2).|

    - scaleOptional FRACTAL_NZ format constraints:
      - Supported only in the dynamic quantization (mx quantization) use case. The data type of scaleOptional must be FLOAT8_E8M0.
      - Only the FLOAT8_E4M3FN data type is supported for x and weight.
      - In this version, the scaleOptional FRACTAL_NZ format does not yet support the use case where x or weight is of the FLOAT4_E2M1/FLOAT4_E1M2 type.
      - Only the use case where transpose_weight is false is supported.
      - Only the use case where groupType is 0 (that is, grouping along the M axis) is supported; the use case where groupType is 2 (grouping along the K axis) is not supported.
      - Only the use case where x is a single tensor and out is a single tensor is supported. weight can be a single tensor or multiple tensors.
      - In the weight single tensor case, scaleOptional is a single tensor, and its storage shape is (g, ceil(N / 16), ceil(K / 64), 16, 2).
      - In the weight multiple tensor case, scaleOptional contains multiple tensors of the same length as weight, and the storage shape of each tensor is (ceil(N / 16), ceil(K / 64), 16, 2).
      - For formats other than FRACTAL_NZ, scaleOptional continues to use the original ND shape constraints in the preceding table.

    - perTokenScaleOptional must meet the following table:

        |groupType| usage scenario | Shape Constraint |
        |:---------:|:---------:| :------ |
        |0|x single tensor|Each tensor is 3-dimensional with shape (M, ceil(K / 64), 2).|

  - Supported scenarios for different groupType values:

    - In the supported scenarios, "single" indicates a single tensor and "multiple" indicates multiple tensors. The order is x, weight, out. For example, "single-multiple-single" indicates that x is a single tensor, weight uses multiple tensors, and out is a single tensor.

      | groupType | Supported Scenarios | scenario restriction |
      |:---------:|:-------:| :------ |
      | -1 | multiple-multiple-multiple |1) Only splitItem 0/1 is supported.<br>2) Tensors in x and out must be 2D, with shapes ($m_i$, $k_i$) and ($m_i$, $n_i$) respectively; tensors in weight must be 2D, with shape ($n_i$, $k_i$) or ($k_i$, $n_i$); tensors in bias must be 1D, with shape ($n_i$).<br>3) groupListOptional must be passed as empty.<br>4) Weight transpose is supported, but whether to transpose must be consistent across all tensors in the weight tensorList.<br>5) x does not support transpose.<br>6) Only the non-quantized scenario is supported.|
      | 0 | single-single-single |1) Only splitItem 2/3 is supported.<br>2) Tensors in weight must be 3D with shape (E, N, K) or (E, K, N); tensors in x and out must be 2D, with shapes (M, K) and (M, N), respectively; tensors in bias must be 2D with shape (E, N).<br>3) groupListOptional is mandatory. When groupListType is 0, the last value must not be greater than the first dimension of tensors in x; when groupListType is 1, the sum of the values must not be greater than the first dimension of tensors in x; when groupListType is 2, the sum of the values in the second column must not be greater than the first dimension of tensors in x.<br>4) The first dimension of groupListOptional supports up to 1024, that is, at most 1024 groups.<br>5) Non-transposed x is supported; weight can be transposed or non-transposed.|

</details>
<!-- end id11 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT products</term>:

  In the MX_A8W4 scenario, call `aclnnWeightQuantPreprocess` first to preprocess weight and weightScale synchronously, and then call this API with the preprocessing output directly as weight and antiquantScaleOptional to perform grouped matrix multiplication.

  ```cpp
  #include <algorithm>
  #include <cstring>
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnn/opdev/float4_e2m1.h"
  #include "aclnnop/aclnn_grouped_matmul_weight_nz.h"
  #include "aclnnop/aclnn_weight_quant_preprocess.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  #define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto dim : shape) {
          shapeSize *= dim;
      }
      return shapeSize;
  }

  class AclRuntimeGuard {
  public:
      explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

      ~AclRuntimeGuard()
      {
          if (stream_ != nullptr) {
              aclrtDestroyStream(stream_);
          }
          if (deviceSet_) {
              aclrtResetDevice(deviceId_);
          }
          if (aclInited_) {
              aclFinalize();
          }
      }

      int Init(aclrtStream* stream)
      {
          auto ret = aclInit(nullptr);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
          aclInited_ = true;

          ret = aclrtSetDevice(deviceId_);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
          deviceSet_ = true;

          ret = aclrtCreateStream(stream);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
          stream_ = *stream;
          return ACL_SUCCESS;
      }

  private:
      int32_t deviceId_;
      aclrtStream stream_ = nullptr;
      bool deviceSet_ = false;
      bool aclInited_ = false;
  };

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& viewShape,
                      const std::vector<int64_t>& viewStrides, const std::vector<int64_t>& storageShape,
                      aclDataType dataType, aclFormat format, void** deviceAddr, aclTensor** tensor)
  {
      auto size = hostData.size() * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      const int64_t* strides = viewStrides.empty() ? nullptr : viewStrides.data();
      *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides, 0, format, storageShape.data(),
                                storageShape.size(), *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return ACL_SUCCESS;
  }

  int CreateSingleTensorList(aclTensor* tensor, aclTensorList** tensorList)
  {
      aclTensor* tensors[] = {tensor};
      *tensorList = aclCreateTensorList(tensors, 1);
      CHECK_RET(*tensorList != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return ACL_SUCCESS;
  }

  std::vector<uint8_t> PackFp4(const std::vector<float>& data)
  {
      std::vector<uint8_t> packedData((data.size() + 1) / 2, 0);
      for (size_t i = 0; i < data.size(); i += 2) {
          uint8_t low = op::Float4E2M1(data[i]).value;
          uint8_t high = i + 1 < data.size() ? op::Float4E2M1(data[i + 1]).value : 0;
          packedData[i / 2] = static_cast<uint8_t>((high << 4) | low);
      }
      return packedData;
  }

  float Bf16ToFloat(uint16_t value)
  {
      uint32_t bits = static_cast<uint32_t>(value) << 16;
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
  }

  int AclnnGroupedMatmulWeightNzMxA8W4PreprocessTest(int32_t deviceId)
  {
      aclrtStream stream = nullptr;
      AclRuntimeGuard runtime(deviceId);
      auto ret = runtime.Init(&stream);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      const int64_t groupNum = 2;
      const int64_t m = 64;
      const int64_t k = 128;
      const int64_t n = 64;
      const int64_t kGroupSize = 32;
      const int64_t scaleK = CEIL_DIV(k, 64);

      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> xStrides = {k, 1};
      std::vector<int64_t> weightViewShape = {groupNum, k, n};
      std::vector<int64_t> weightStorageShape = {groupNum, n, k};
      std::vector<int64_t> weightStrides = {k * n, 1, k};
      std::vector<int64_t> xScaleShape = {m, scaleK, 2};
      std::vector<int64_t> xScaleStrides = {scaleK * 2, 2, 1};
      std::vector<int64_t> weightScaleViewShape = {groupNum, scaleK, n, 2};
      std::vector<int64_t> weightScaleStorageShape = {groupNum, n, scaleK, 2};
      std::vector<int64_t> weightScaleStrides = {n * scaleK * 2, 2, scaleK * 2, 1};
      std::vector<int64_t> outWeightStorageShape = {
          groupNum, CEIL_DIV(k, 32), CEIL_DIV(n, 16), 16, 32};
      std::vector<int64_t> outShape = {m, n};
      std::vector<int64_t> outStrides = {n, 1};
      std::vector<int64_t> groupListShape = {groupNum};
      std::vector<int64_t> groupListStrides = {1};

      std::vector<uint8_t> xHostData(GetShapeSize(xShape), 0b00111000); // FLOAT8_E4M3FN 1.0
      std::vector<float> weightFloatData(GetShapeSize(weightStorageShape), 1.0f);
      std::vector<uint8_t> weightHostData = PackFp4(weightFloatData);
      std::vector<uint8_t> xScaleHostData(GetShapeSize(xScaleShape), 0b01111111); // FLOAT8_E8M0 1.0
      std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleStorageShape),
                                               0b10000101); // FLOAT8_E8M0 64.0
      std::vector<uint8_t> outWeightHostData(GetShapeSize(outWeightStorageShape) / 2, 0);
      std::vector<uint8_t> outWeightScaleHostData(GetShapeSize(weightScaleStorageShape), 0);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);
      std::vector<int64_t> groupListHostData = {m / groupNum, m};

      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* xScaleDeviceAddr = nullptr;
      void* weightScaleDeviceAddr = nullptr;
      void* outWeightDeviceAddr = nullptr;
      void* outWeightScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      void* groupListDeviceAddr = nullptr;

      aclTensor* xTensor = nullptr;
      aclTensor* weightTensor = nullptr;
      aclTensor* xScaleTensor = nullptr;
      aclTensor* weightScaleTensor = nullptr;
      aclTensor* outWeightTensor = nullptr;
      aclTensor* outWeightScaleTensor = nullptr;
      aclTensor* outTensor = nullptr;
      aclTensor* groupListTensor = nullptr;
      aclTensorList* x = nullptr;
      aclTensorList* weight = nullptr;
      aclTensorList* perTokenScale = nullptr;
      aclTensorList* antiquantScale = nullptr;
      aclTensorList* out = nullptr;

      ret = CreateAclTensor(xHostData, xShape, xStrides, xShape, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, &xDeviceAddr,
                            &xTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorGuard(xTensor, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateSingleTensorList(xTensor, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> xPtr(x, aclDestroyTensorList);
      xTensorGuard.release();

      ret = CreateAclTensor(weightHostData, weightViewShape, weightStrides, weightStorageShape, ACL_FLOAT4_E2M1,
                            ACL_FORMAT_ND, &weightDeviceAddr, &weightTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weightTensor, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(xScaleHostData, xScaleShape, xScaleStrides, xScaleShape, ACL_FLOAT8_E8M0, ACL_FORMAT_ND,
                            &xScaleDeviceAddr, &xScaleTensor);
      std::unique_ptr<void, aclError (*)(void*)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xScaleTensorGuard(xScaleTensor, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateSingleTensorList(xScaleTensor, &perTokenScale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> perTokenScalePtr(perTokenScale,
                                                                                             aclDestroyTensorList);
      xScaleTensorGuard.release();

      ret = CreateAclTensor(weightScaleHostData, weightScaleViewShape, weightScaleStrides, weightScaleStorageShape,
                            ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &weightScaleDeviceAddr, &weightScaleTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightScaleTensorPtr(weightScaleTensor,
                                                                                          aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Keep the transposed stride of weight so that GroupedMatmul can identify the transposed layout from tensor metadata.
      ret = CreateAclTensor(outWeightHostData, weightViewShape, weightStrides, outWeightStorageShape,
                            ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ_C0_32, &outWeightDeviceAddr, &outWeightTensor);
      std::unique_ptr<void, aclError (*)(void*)> outWeightDeviceAddrPtr(outWeightDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightTensorGuard(outWeightTensor,
                                                                                         aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateSingleTensorList(outWeightTensor, &weight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> outWeightPtr(weight, aclDestroyTensorList);
      outWeightTensorGuard.release();

      ret = CreateAclTensor(outWeightScaleHostData, weightScaleViewShape, weightScaleStrides, weightScaleStorageShape,
                            ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &outWeightScaleDeviceAddr, &outWeightScaleTensor);
      std::unique_ptr<void, aclError (*)(void*)> outWeightScaleDeviceAddrPtr(outWeightScaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightScaleTensorGuard(outWeightScaleTensor,
                                                                                              aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateSingleTensorList(outWeightScaleTensor, &antiquantScale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> outWeightScalePtr(antiquantScale,
                                                                                              aclDestroyTensorList);
      outWeightScaleTensorGuard.release();

      ret = CreateAclTensor(outHostData, outShape, outStrides, outShape, ACL_BF16, ACL_FORMAT_ND, &outDeviceAddr,
                            &outTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorGuard(outTensor, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateSingleTensorList(outTensor, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList*)> outPtr(out, aclDestroyTensorList);
      outTensorGuard.release();

      ret = CreateAclTensor(groupListHostData, groupListShape, groupListStrides, groupListShape, ACL_INT64,
                            ACL_FORMAT_ND, &groupListDeviceAddr, &groupListTensor);
      std::unique_ptr<void, aclError (*)(void*)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupListPtr(groupListTensor, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      ret = aclnnWeightQuantPreprocessGetWorkspaceSize(
          weightTensor, weightScaleTensor, nullptr, nullptr, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, kGroupSize,
          outWeightTensor, outWeightScaleTensor, nullptr, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantPreprocessGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      void* preprocessWorkspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> preprocessWorkspacePtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&preprocessWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate preprocess workspace failed. ERROR: %d\n", ret);
                    return ret);
          preprocessWorkspacePtr.reset(preprocessWorkspaceAddr);
      }

      ret = aclnnWeightQuantPreprocess(preprocessWorkspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantPreprocess failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after preprocess failed. ERROR: %d\n", ret);
                return ret);

      workspaceSize = 0;
      executor = nullptr;
      // The output of WeightQuantPreprocess is directly used as weight and antiquantScaleOptional.
      ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
          x, weight, nullptr, nullptr, nullptr, antiquantScale, nullptr, perTokenScale, groupListTensor, nullptr,
          nullptr, nullptr, 3, 0, 0, 0, nullptr, 0, out, nullptr, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnGroupedMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      void* groupedMatmulWorkspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> groupedMatmulWorkspacePtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&groupedMatmulWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate grouped matmul workspace failed. ERROR: %d\n", ret);
                    return ret);
          groupedMatmulWorkspacePtr.reset(groupedMatmulWorkspaceAddr);
      }

      ret = aclnnGroupedMatmulWeightNz(groupedMatmulWorkspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after grouped matmul failed. ERROR: %d\n", ret);
                return ret);

      std::vector<uint16_t> resultData(GetShapeSize(outShape), 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (size_t i = 0; i < std::min<size_t>(resultData.size(), 10); ++i) {
          LOG_PRINT("result[%zu] is: %.1f\n", i, Bf16ToFloat(resultData[i]));
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // WeightQuantPreprocess currently only supports the Ascend 950PR&950DT products.
      int32_t deviceId = 0;
      auto ret = AclnnGroupedMatmulWeightNzMxA8W4PreprocessTest(deviceId);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("AclnnGroupedMatmulWeightNzMxA8W4PreprocessTest failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  ```

<!-- end id12 -->

Pseudo-Quantized Scenario Example

```c++
#include <cstdint>
#include <cstring>
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

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate for resource initialization
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
  // Call aclrtMalloc to allocate memory on the Device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy Host-side data to Device-side memory.
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
  // Call aclrtMalloc to allocate memory on the Device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy Host-side data to Device-side memory.
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

  // Call aclrtMalloc to allocate memory on the device.
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

  // Check the shape dimensions.
  if (shape.size() != 3) {
    LOG_PRINT("Shape must be 3D for NZ format\n");
    return -1;
  }

  int64_t E = shape[0];
  int64_t K = shape[1];
  int64_t N = shape[2];

  // Check whether the dimensions are divisible.
  if (N % 64 != 0 || K % 16 != 0) {
    LOG_PRINT("N must be divisible by 64 and K by 16 for NZ format\n");
    return -1;
  }

  std::vector<int64_t> shapeNz = {E, N/64, K/16, 16, 64};

  // Call the aclCreateTensor API to create an aclTensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<std::vector<int64_t>> xShape = {{512, 256}};
  std::vector<std::vector<int64_t>> weightShape = {{2, 256, 256}};
  std::vector<std::vector<int64_t>> yShape = {{512, 256}};
  std::vector<int64_t> groupListShape = {2};
  std::vector<int64_t> groupListData = {256, 512};

  void* xDeviceAddr[1];
  void* weightDeviceAddr[1];
  void* yDeviceAddr[1];
  void* biasDeviceAddr[1] = {nullptr};  // Declare biasDeviceAddr.
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

  // Create tuningconfig aclIntArray.
  std::vector<int64_t> tuningConfigData = {512};
  aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

  // Create x aclTensorList.
  ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create weight aclTensorList - NZ format.
  ret = CreateAclTensorListNz<int8_t>(wHostDataList, weightShape, weightDeviceAddr, aclDataType::ACL_BF16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create y aclTensorList.
  ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_BF16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create group_list aclTensor.
  ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. Call the CANN operator library API.
  // Call the first-phase API of aclnnGroupedMatmulWeightNz.
  ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput, activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, tuningConfig, 0, out, activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnGroupedMatmulWeightNz.
  ret = aclnnGroupedMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from Device-side memory to the Host side. Modify it based on the specific API definition.
  for (int i = 0; i < 1; i++) {
    auto size = GetShapeSize(yShape[i]);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < 20; j++) {
      LOG_PRINT("result[%ld] is: %g\n", j, Bf16ToFloat(resultData[j]));
    }
    LOG_PRINT("......\n");
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensorList(x);
  aclDestroyTensorList(weight);
  if (bias) aclDestroyTensorList(bias);
  aclDestroyTensorList(out);
  if (groupedList) aclDestroyTensor(groupedList);

  // 7. Release device resources. Modify it based on the specific API definition.
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

Fully Quantized Example

```c++
#include <cstdint>
#include <cstring>
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

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

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
    // Boilerplate: AscendCL initialization
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
    // Call aclrtMalloc to allocate Device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the Host side to Device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
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
    for (uint64_t i = 0; i < dstShapeSize; ++i) {
        tensorSize *= dstShape[i];
    }
    std::vector<T> dstTensorHostData(tensorSize, 0);

    ret = CreateAclTensorWithFormat(dstTensorHostData, shape, &dstShape, &dstShapeSize, &dstDeviceAddr, dataType,
                                    &dstTensor, static_cast<aclFormat>(actualFormat));
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);

    // Call the first-phase API aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    *deviceAddr = dstDeviceAddr;
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

    // 2. Construct the inputs and outputs. Customize the construction according to the API.
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

    // Create the x aclTensorList.
    ret = CreateAclTensorList(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> xTensorPtr(x, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    // Create the weight aclTensorList.
    ret = CreateAclTensorListNz(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorPtr(weight,
                                                                                           aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    // Create the scale aclTensorList.
    ret = CreateAclTensorList(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_BF16, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> scaleTensorPtr(scale, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    // Create the pertoken aclTensorList.
    ret = CreateAclTensorList(pertokenHostData, pertokenShape, &pertokenDeviceAddr, aclDataType::ACL_FLOAT,
                              &perTokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> pertokenTensorPtr(perTokenScale,
                                                                                             aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> pertokenDeviceAddrPtr(pertokenDeviceAddr, aclrtFree);
    // Create the y aclTensorList.
    ret = CreateAclTensorList(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> yTensorPtr(out, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);

    // Create the group_list aclTensorList.
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
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnGroupedMatmulWeightNz.
    ret = aclnnGroupedMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, copy the results from the Device memory to the Host, and modify them according to the specific API definition.
    for (int i = 0; i < 1; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %g\n", j, Bf16ToFloat(resultData[j]));
        }
    }
    return ACL_SUCCESS;
}
int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the list of AscendCL external APIs.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGourpedMatmulTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz test failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```

- <term>Ascend 950PR&950DT products</term>:

  MxA8W4 pseudo-quantization (single-multiple-single use case) example

```c++
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            Finalize(deviceId, stream);                                                                                \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

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
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostDataList,
                        const std::vector<std::vector<int64_t>> &shapes, std::vector<void *> &deviceAddrList,
                        aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    std::vector<aclTensor *> tensors(size, nullptr);
    deviceAddrList.resize(size, nullptr);
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor(hostDataList[i], shapes[i], &deviceAddrList[i], dataType, &tensors[i]);
        if (ret != ACL_SUCCESS) {
            for (int j = 0; j < i; j++) {
                aclDestroyTensor(tensors[j]);
                aclrtFree(deviceAddrList[j]);
                deviceAddrList[j] = nullptr;
            }
            return ret;
        }
    }
    *tensor = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorNzTransposed(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                                aclDataType dataType, aclTensor **tensor)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    int64_t k = shape[0];
    int64_t n = shape[1];
    std::vector<int64_t> transStrides = {1, k};
    std::vector<int64_t> storageShape = {k / 32, n / 16, 16, 32};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, transStrides.data(), 0,
                              aclFormat::ACL_FORMAT_FRACTAL_NZ, storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorListNzTransposed(const std::vector<std::vector<T>> &hostDataList,
                                    const std::vector<std::vector<int64_t>> &shapes,
                                    std::vector<void *> &deviceAddrList, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    std::vector<aclTensor *> tensors(size, nullptr);
    deviceAddrList.resize(size, nullptr);
    for (int i = 0; i < size; ++i) {
        int ret = CreateAclTensorNzTransposed<T>(hostDataList[i], shapes[i], &deviceAddrList[i], dataType, &tensors[i]);
        if (ret != ACL_SUCCESS) {
            for (int j = 0; j < i; j++) {
                aclDestroyTensor(tensors[j]);
                aclrtFree(deviceAddrList[j]);
                deviceAddrList[j] = nullptr;
            }
            return ret;
        }
    }
    *tensor = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

int aclnnGroupedMatmulWeightNzMxA8W4MultiTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Parameters for the single-multiple-single MxA8W4 use case
    int64_t m = 64L;
    int64_t k = 128L;
    int64_t n = 64L;
    int64_t groupSize = 32L;
    int64_t groupNum = 2L;

    std::vector<std::vector<int64_t>> xShape = {{m, k}};
    std::vector<std::vector<int64_t>> weightShapes;
    std::vector<std::vector<int64_t>> biasShapes;
    std::vector<std::vector<int64_t>> antiquantScaleShapes;
    for (int64_t i = 0; i < groupNum; i++) {
        weightShapes.push_back({k, n});
        biasShapes.push_back({n});
        antiquantScaleShapes.push_back({k / groupSize / 2, n, 2});
    }
    std::vector<std::vector<int64_t>> perTokenScaleShape = {{m, k / groupSize / 2, 2}};
    std::vector<std::vector<int64_t>> yShape = {{m, n}};
    std::vector<int64_t> groupListShape = {groupNum};
    std::vector<int64_t> groupListData;
    for (int64_t i = 0; i < groupNum; i++) {
        groupListData.push_back(m / groupNum * (i + 1));
    }

    std::vector<void *> xDeviceAddrList;
    std::vector<void *> weightDeviceAddrList;
    std::vector<void *> biasDeviceAddrList;
    std::vector<void *> antiquantScaleDeviceAddrList;
    std::vector<void *> perTokenScaleDeviceAddrList;
    std::vector<void *> yDeviceAddrList;
    void *groupListDeviceAddr = nullptr;

    aclTensorList *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *bias = nullptr;
    aclTensorList *antiquantScale = nullptr;
    aclTensorList *perTokenScale = nullptr;
    aclTensor *groupedList = nullptr;
    aclTensorList *out = nullptr;
    aclTensorList *scale = nullptr;
    aclTensorList *offset = nullptr;
    aclTensorList *antiquantOffset = nullptr;
    aclTensorList *activationInput = nullptr;
    aclTensorList *activationQuantScale = nullptr;
    aclTensorList *activationQuantOffset = nullptr;
    aclTensorList *activationFeatureOut = nullptr;
    aclTensorList *dynQuantScaleOut = nullptr;

    int64_t splitItem = 3L;
    int64_t groupType = 0L;
    int64_t groupListType = 0L;
    int64_t actType = 0L;

    // x: FLOAT8_E4M3FN, 1 byte per element
    std::vector<std::vector<int8_t>> xHostData = {std::vector<int8_t>(m * k)};
    for (int64_t j = 0; j < m * k; j++) {
        xHostData[0][j] = static_cast<int8_t>(j % 0x7F);
    }
    ret = CreateAclTensorList(xHostData, xShape, xDeviceAddrList, aclDataType::ACL_FLOAT8_E4M3FN, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> xPtr(x, aclDestroyTensorList);

    // weight: FLOAT4_E2M1, 4 bits per element, 2 elements per byte
    std::vector<std::vector<int8_t>> weightHostDataList;
    for (int64_t i = 0; i < groupNum; i++) {
        std::vector<int8_t> data(k * n / 2);
        for (int64_t j = 0; j < k * n / 2; j++) {
            data[j] = static_cast<int8_t>((j + i * 37) % 256);
        }
        weightHostDataList.push_back(data);
    }
    ret = CreateAclTensorListNzTransposed(weightHostDataList, weightShapes, weightDeviceAddrList,
                                          aclDataType::ACL_FLOAT4_E2M1, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightPtr(weight, aclDestroyTensorList);

    // bias: BFLOAT16
    std::vector<std::vector<uint16_t>> biasHostDataList;
    for (int64_t i = 0; i < groupNum; i++) {
        std::vector<uint16_t> data(n);
        for (int64_t j = 0; j < n; j++) {
            data[j] = static_cast<uint16_t>((j + i * 50) % 100 + 1);
        }
        biasHostDataList.push_back(data);
    }
    ret = CreateAclTensorList(biasHostDataList, biasShapes, biasDeviceAddrList, aclDataType::ACL_BF16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> biasPtr(bias, aclDestroyTensorList);

    // antiquantScale: FLOAT8_E8M0, 1 byte per element
    std::vector<std::vector<int8_t>> antiquantScaleHostDataList;
    for (int64_t i = 0; i < groupNum; i++) {
        int64_t scaleSize = k / groupSize / 2 * n * 2;
        std::vector<int8_t> data(scaleSize);
        for (int64_t j = 0; j < scaleSize; j++) {
            data[j] = static_cast<int8_t>((j + i * 5) % 16 + 120);
        }
        antiquantScaleHostDataList.push_back(data);
    }
    ret = CreateAclTensorList(antiquantScaleHostDataList, antiquantScaleShapes, antiquantScaleDeviceAddrList,
                              aclDataType::ACL_FLOAT8_E8M0, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> antiquantScalePtr(antiquantScale,
                                                                                             aclDestroyTensorList);

    // perTokenScale: FLOAT8_E8M0, single Tensor
    std::vector<std::vector<int8_t>> perTokenScaleHostData = {std::vector<int8_t>(m * k / groupSize / 2 * 2)};
    for (int64_t j = 0; j < m * k / groupSize / 2 * 2; j++) {
        perTokenScaleHostData[0][j] = static_cast<int8_t>(j % 16 + 120);
    }
    ret = CreateAclTensorList(perTokenScaleHostData, perTokenScaleShape, perTokenScaleDeviceAddrList,
                              aclDataType::ACL_FLOAT8_E8M0, &perTokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> perTokenScalePtr(perTokenScale,
                                                                                            aclDestroyTensorList);

    // y: BFLOAT16
    std::vector<std::vector<uint16_t>> yHostData = {std::vector<uint16_t>(m * n, 0)};
    ret = CreateAclTensorList(yHostData, yShape, yDeviceAddrList, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> outPtr(out, aclDestroyTensorList);

    // groupList: INT64
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListPtr(groupedList, aclDestroyTensor);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);

    // Call the aclnnGroupedMatmulWeightNz first-phase API.
    ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput,
        activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, nullptr, 0, out,
        activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // Call the aclnnGroupedMatmulWeightNz second-phase API.
    ret = aclnnGroupedMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the output result.
    auto size = GetShapeSize(yShape[0]);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddrList[0], size * sizeof(resultData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < std::min<int64_t>(size, 10); j++) {
        LOG_PRINT("result[%ld] is: %g\n", j, Bf16ToFloat(resultData[j]));
    }

    // Release device resources.
    for (auto addr : xDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : weightDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : biasDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : antiquantScaleDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : perTokenScaleDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    for (auto addr : yDeviceAddrList) {
        if (addr)
            aclrtFree(addr);
    }
    if (groupListDeviceAddr)
        aclrtFree(groupListDeviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedMatmulWeightNzMxA8W4MultiTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnGroupedMatmulWeightNz MxA8W4 multi test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
