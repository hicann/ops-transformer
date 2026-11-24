# aclnnGroupedMatmulActivationQuantWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:09:59.725Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Fuses GroupedMatmul, activation, and quant. For details, see Formula. This API is a WeightNz-specific version. The caller must pass weight in FRACTAL_NZ format, which the API parses accordingly. The current version supports only the MX quantization scenario, and the activation function supports only gelu_tanh.

- Formula:
  <details>
  <summary>MX quantization scenario:</summary>
  <a id="mx-quantization-scenario"></a>

    - **Definition**:

      - **⋅** denotes matrix multiplication.
      - **⊙** indicates per-element multiplication.
      - The mathematical semantics of the gelu_tanh activation function is as follows:

        $$
        \text{GELU}_{\text{tanh}}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)
        $$

      - The current underlying kernel implementation uses the gelu_sigmoid function to approximate gelu_tanh, defined as:

        $$
        \operatorname{GeluTanh}(x) =
        \frac{x}{1 + \exp\left(-1.595769121 \times \left(x + 0.044715 \times x^3\right)\right)}
        $$
      - $E$ denotes the number of experts, $M$ denotes the total number of tokens, $K$ denotes the input feature dimension, and $N$ denotes the output feature dimension.
      - $blocksize$ denotes the storage block size of the MX quantization result. Currently, only 64 is supported, corresponding to 2 MX quantization groups, each containing 32 elements.

    - **Inputs**:

      - $X$: activation matrix.
      - $groupList$: Grouping index list, interpreted as cumsum or count based on groupListType.
      - $weight$: Grouped weight matrix.
      - $weightScale$: quantization factor of the weight matrix.
      - $bias$: must be empty in the MX scenario. Supports nullptr, an empty tensorlist, or an empty tensorlist with a length of 1 and an element shape of (0).
      - $xScaleOptional$: quantization factor of the left matrix, corresponding to $xScale$ in the formula.

    - **Outputs**:

      - $Y$: output matrix after activation and quantization. Its data type is specified by the output tensor y.
      - $YScale$: output quantization factor.

    - **Computation Process**

      - 1.Determine the token range of the current group based on groupList[i], where $i \in [0, Len(groupList))$.

      - 2.Perform GroupedMatmul and dequantization computation based on the input parameters determined by grouping. The intermediate GroupedMatmul result defaults to FLOAT32:

        $C_i = (X_i \cdot weight_i) \odot xScale_{i\ Broadcast} \odot weightScale_{i\ Broadcast}$

      - 3.Perform gelu_tanh activation:

        $S_i = GeluTanh(C_i)$

        Note: The underlying kernel implementation uses the gelu_sigmoid function to approximately compute gelu_tanh. For the approximate formula, see the definition.

      - 4. Perform MX quantization on the activation result. The target data type DType is specified by the data type of output Tensor y:

        - Case 1: When scaleAlg is 0, the OCP implementation is used. The activation result $S_i$ is grouped along the N axis by $group\_size=32$. Each group of $group\_size$ elements $\{V_j\}_{j=1}^{group\_size}$ is dynamically quantized to $\{YScale, \{P_j\}_{j=1}^{group\_size}\}$.

          $$
          shared\_exp = floor(log_2(max_j(|V_j|))) - emax
          $$

          $$
          YScale = 2 ^ {shared\_exp}
          $$

          $$
          P_j = cast\_to\_dst\_type(V_j / YScale, roundMode), \space j \space from \space 1 \space to \space group\_size
          $$

          The quantized $P_j$ values are placed at the positions of the corresponding $V_j$ to form output $Y$. The $YScale$ values of the quantization groups along the N axis form output $YScale$.

          - $emax$: Exponent of the largest normal number for the corresponding data type.

            | DataType | emax |
            | :---: | :---: |
            | FLOAT4_E2M1 | 2 |
            | FLOAT4_E1M2 | 0 |
            | FLOAT8_E4M3FN | 8 |
            | FLOAT8_E5M2 | 15 |

        - Use case 2: when scaleAlg is 1, the cuBLAS implementation is used, and only FP8 types are involved. Group the activation result $S_i$ along the N axis by the storage $blocksize=64$, and compute a block scaling factor $S_{fp32}^b$ for each block independently. Then map all elements in the block to the target FP8 type using the same $S_{fp32}^b$. If the last block contains fewer than $blocksize$ elements, treat the missing values as 0 and process the block as a complete one.

          Find the maximum absolute value of the elements in the block:

          $$
          Amax(D_{fp32}^b)=max(\{|d_j|\}_{j=1}^{k})
          $$

          Map FP32 to the range representable by the target data type, where $Amax(DType)$ is the maximum value representable by the target precision:

          $$
          S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
          $$

          Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format. First, extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from $S_{fp32}^b$. Then round up the exponent to avoid quantization overflow:

          $$
          E_{int}^b = \begin{cases}
          E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\
          E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a subnormal number, and} M_{fixp}^b > 0.5 \\
          E_{int}^b, & \text{ otherwise}
          \end{cases}
          $$

          $$
          S_{ue8m0}^b = 2 ^ {E_{int}^b}
          $$

          $$
          R_{fp32}^b = \frac{1}{fp32(S_{ue8m0}^b)}
          $$

          Quantize each element in the block:

          $$
          d^j = DType(d_{fp32}^j \cdot R_{fp32}^b), \space j \space from \space 1 \space to \space blocksize
          $$

          The final quantized output is $\left(S^b, [d^j]_{j=1}^{k}\right)$, where $S^b$ denotes the block scaling factor $S_{ue8m0}^b$, and $[d^j]_{j=1}^{k}$ denotes the quantized data in the block.

        - Scenario 3: when scaleAlg is 2, only the FLOAT4_E2M1 output is involved, and `dstTypeMax` supports 0.0 or [6.0, 12.0]. Group the activation result $S_i$ along the N axis by $group\_size=32$:

          When `dstTypeMax` is 0.0, 6.0, or 7.0, determine whether to carry the shared exponent based on the exponent and high-order mantissa bits of the maximum absolute value in the block. Specifically, 0.0 and 6.0 use one high-order mantissa bit for the determination, and 7.0 uses two:

          $$
          shared\_exp = \begin{cases}
          ceil(log_2(max_j(|V_j|))) - emax, & \text{ if the high-order mantissa bits meet the carry condition and the mantissa is not all zeros} \\
          floor(log_2(max_j(|V_j|))) - emax, & \text{ otherwise}
          \end{cases}
          $$

          where the $emax$ of FLOAT4_E2M1 is 2. Then, compute $YScale=2^{shared\_exp}$ and perform the following:

          $$
          P_j = cast\_to\_FLOAT4\_E2M1(V_j / YScale, roundMode)
          $$

          When `dstTypeMax` is any other value in [6.0, 12.0], first compute $scale=max_j(|V_j|)/dstTypeMax$, round the result up to a power of 2 representable in E8M0 as $YScale$, and obtain $P_j$ using the preceding formula.

          The quantized $P_j$ values form the output $Y$. Each 32-element group corresponds to one $YScale$ of the FLOAT8_E8M0 type.

  </details>

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnGroupedMatmulActivationQuantWeightNz" API to perform computation.

```cpp
aclnnStatus aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
  const aclTensor     *x,
  const aclTensor     *groupList,
  const aclTensorList *weight,
  const aclTensorList *weightScale,
  const aclTensorList *bias,
  const aclTensor     *xScaleOptional,
  const char          *activationType,
  int64_t              groupListType,
  const aclIntArray   *tuningConfig,
  const char          *quantMode,
  const char          *roundMode,
  int64_t              scaleAlg,
  float                dstTypeMax,
  aclTensor           *y,
  aclTensor           *yScale,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulActivationQuantWeightNz(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
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
      <th>Parameter</th>
      <th style="white-space: nowrap">Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th style="white-space: nowrap">Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x (const aclTensor *)</td>
      <td rowspan="1">Input</td>
      <td>Represents the left matrix, corresponding to X in the formula.</td>
      <td>Mandatory parameter.</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList (const aclTensor *)</td>
      <td rowspan="1">Input</td>
      <td>Represents grouping information, corresponding to groupList in the formula.</td>
      <td><ul>
      <li>Mandatory parameter. The input data format varies with groupListType. When groupListType is 0, the last value must not be greater than the first dimension of the tensor in x. When groupListType is 1, the sum of values must not be greater than the first dimension of the tensor in x.</li>
      <li>When groupListType=0, elements in groupList must be non-negative and monotonically non-decreasing, and the last element must not be greater than M. When groupListType=1, elements must be non-negative, and the sum must not be greater than M.</li>
      <li>The first phase of GetWorkspaceSize only checks the pointer, dtype, and shape of groupList, and does not read the element values of the device-side tensor. The caller must ensure the preceding value range constraints.</li>
      <li>Values in groupList define the valid part of the output data. The part not specified by groupList will not be updated.</li>
      </ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (const aclTensorList *)</td>
      <td rowspan="1">Input</td>
      <td>Represents the weight matrix, corresponding to weight in the formula.</td>
      <td>Currently, only tensorList length 1 is supported. The caller must pass weight in FRACTAL_NZ format, which the API parses accordingly. viewShape must have 3 dimensions, and storageShape must have 5 dimensions.</td>
      <td>FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>FRACTAL_NZ</td>
      <td>3 (storageShape is 5)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightScale (const aclTensorList *)</td>
      <td rowspan="1">Input</td>
      <td>Represents the quantization factor of the weight matrix, corresponding to weightScale in the formula.</td>
      <td>Currently, only a tensorList length of 1 is supported.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias (const aclTensorList *)</td>
      <td rowspan="1">Input</td>
      <td>Bias.</td>
      <td>Must be empty in the MX quantization scenario. Supports nullptr, an empty tensorlist, or an empty tensorlist of length 1 with an element shape of (0).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xScaleOptional (const aclTensor *)</td>
      <td rowspan="1">Input</td>
      <td>Represents the quantization factor of the left matrix, corresponding to xScale in the formula.</td>
      <td>In the MX quantization scenario, nullptr is not supported.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>activationType (const char *)</td>
      <td>Input</td>
      <td>Specifies the activation function type.</td>
      <td>Mandatory attribute. Currently, only "gelu_tanh" is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the interpretation mode of groupList.</td>
      <td>Supported values: 0 and 1. When groupListType is 0, groupList must be a monotonically non-decreasing sequence of non-negative numbers. When groupListType is 1, groupList must be a sequence of non-negative numbers.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tuningConfig (const aclIntArray *)</td>
      <td>Input</td>
      <td>Represents the tuning parameter. Reserved parameter.</td>
      <td>Supports nullptr. Currently not used.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (const char *)</td>
      <td>Input</td>
      <td>Indicates the quantization mode.</td>
      <td>Supports nullptr or an empty string. Currently, only "mx" is supported. When nullptr or an empty string is passed, the mode is deduced as "mx" if the data type of x is a supported FLOAT8 or FLOAT4 type and the data type of xScaleOptional is FLOAT8_E8M0. If a value other than "mx" is explicitly passed, an error is reported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode (const char *)</td>
      <td>Input</td>
      <td>Indicates the quantization rounding mode.</td>
      <td>Passing nullptr or an empty string is supported. If nullptr or an empty string is passed, the value is processed as "rint". Currently, only "rint" is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Indicates the quantization factor calculation algorithm.</td>
      <td>Supports 0, 1, and 2: 0 indicates the OCP implementation; 1 indicates the cuBLAS implementation, which supports only FLOAT8 output; 2 indicates the FLOAT4 dynamic range implementation, which supports only FLOAT4_E2M1 output. FLOAT4_E1M2 output supports only 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (float)</td>
      <td>Input</td>
      <td>Specifies the value of maxType, corresponding to Amax(DType) in the formula.</td>
      <td>Supports 0.0 or [6.0, 12.0] when scaleAlg is 2; 0.0 indicates using the maximum value 6.0 of FLOAT4_E2M1. Other scaleAlg values support only 0.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor *)</td>
      <td>Output</td>
      <td>Output matrix after activation and quantization.</td>
      <td>Mandatory parameter. FLOAT4 output is supported only in the FLOAT4 input use case.</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yScale (aclTensor *)</td>
      <td>Output</td>
      <td>Represents the output quantization factor.</td>
      <td>Mandatory parameter.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t *)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor **)</td>
      <td>Output</td>
      <td>Returns the op executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - x supports non-transposed input only, and weight supports non-transposed and transposed input. The API deduces whether weight is transposed based on the shapes and transposition relationship of weight and weightScale.
    - The transpose attribute of weightScale must be consistent with that of weight.
    - Empty tensor handling rules:
      - weight and weightScale are mandatory inputs. Empty tensorlist is not supported, and elements in the tensorlist cannot be nullptr.
      - Empty tensors with M of 0 or N of 0 are supported: x and xScaleOptional support M of 0, and weight and weightScale support N of 0. In this case, K of 0 is allowed. The first-phase API returns ACLNN_SUCCESS, and workspaceSize is 0. When both M and N are not 0, K of 0 is not supported.
      - Supports the empty tensor use case where groupList is empty: when groupList is empty and output y or yScale is empty, the first-phase API returns ACLNN_SUCCESS and workspaceSize is 0.

  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A mandatory parameter is a null pointer. For example, x, groupList, weight, weightScale, activationType, y, yScale, workspaceSize, or executor is null.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>activationType is not "gelu_tanh".</td>
    </tr>
    <tr>
      <td>bias is not empty.</td>
    </tr>
    <tr>
      <td>xScaleOptional is a null pointer in the MX quantization scenario.</td>
    </tr>
    <tr>
      <td>The input data type or data format is not supported.</td>
    </tr>
    <tr>
      <td>The shape dimensions do not match.</td>
    </tr>
    <tr>
      <td>The attribute value is out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulActivationQuantWeightNz

- **Parameters**

  <table style="undefined;table-layout: fixed;width: 795px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 505px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace address applied for on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied for on the device, obtained by the first-phase API aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>ACL stream.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: Returned status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
    - aclnnGroupedMatmulActivationQuantWeightNz defaults to a deterministic implementation.
- In non-empty tensor use cases, the first dimension of groupList is in the range 1-1024.
- The following constraints must be met in the MX quantization scenario:
    - The data types must meet the requirements in the following table:
      <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 180px">
      <col style="width: 160px">
      <col style="width: 160px">
      <col style="width: 160px">
      <col style="width: 180px">
      <col style="width: 160px">
      </colgroup>
      <thead>
        <tr>
          <th>MX quantization scenario</th>
          <th>activation function</th>
          <th>x</th>
          <th>weight</th>
          <th>weightScale</th>
          <th>xScaleOptional</th>
          <th>y</th>
          <th>yScale</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>MXFP8</td>
          <td>gelu_tanh</td>
          <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
          <td>FLOAT8_E4M3FN</td>
          <td>FLOAT8_E8M0</td>
          <td>FLOAT8_E8M0</td>
          <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
          <td>FLOAT8_E8M0</td>
        </tr>
        <tr>
          <td>MXFP4</td>
          <td>gelu_tanh</td>
          <td>FLOAT4_E2M1, FLOAT4_E1M2</td>
          <td>FLOAT4_E2M1, FLOAT4_E1M2</td>
          <td>FLOAT8_E8M0</td>
          <td>FLOAT8_E8M0</td>
          <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2</td>
          <td>FLOAT8_E8M0</td>
        </tr>
      </tbody>
      </table>

    - The shape constraints must meet the following table:
      <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 280px">
      <col style="width: 320px">
      <col style="width: 180px">
      <col style="width: 110px">
      <col style="width: 160px">
      </colgroup>
      <thead>
        <tr>
          <th>MX quantization scenario</th>
          <th>activation function</th>
          <th>x</th>
          <th>weight (storageShape)</th>
          <th>weightScale</th>
          <th>xScale</th>
          <th>output</th>
          <th>outputScale</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>MXFP8</td>
          <td>gelu_tanh</td>
          <td>(M, K)</td>
          <td><ul>
          <li>Non-transposed shape: {(E, ceil(N / 32), ceil(K / 16), 16, 32)}</li>
          <li>Transposed shape: {(E, ceil(K / 32), ceil(N / 16), 16, 32)}</li></ul></td>
          <td><ul>
          <li>Non-transposed shape: {(E, ceil(K / 64), N, 2)}</li>
          <li>Transposed shape: {(E, N, ceil(K / 64), 2)}</li></ul></td>
          <td>(M, ceil(K / 64), 2)</td>
          <td>(M, N)</td>
          <td>(M, ceil(N / 64), 2)</td>
        </tr>
        <tr>
          <td>MXFP4</td>
          <td>gelu_tanh</td>
          <td>(M, K)</td>
          <td><ul>
          <li>Non-transposed shape: {(E, ceil(N / 64), ceil(K / 16), 16, 64)}.</li>
          <li>Transposed shape: {(E, ceil(K / 64), ceil(N / 16), 16, 64)}.</li></ul></td>
          <td><ul>
          <li>Non-transposed shape: {(E, ceil(K / 64), N, 2)}.</li>
          <li>Transposed shape: {(E, N, ceil(K / 64), 2)}.</li></ul></td>
          <td>(M, ceil(K / 64), 2)</td>
          <td>(M, N)</td>
          <td>(M, ceil(N / 64), 2)</td>
        </tr>
       </tbody>
       </table>

     - In the table, the last dimension of the shapes of xScale, weightScale, and outputScale is 2, indicating that each 64-element storage block contains 2 MX quantization groups, and each group covers 32 elements.

     - In non-empty use cases, N must be a multiple of 64 and at least 64.
     - In non-empty use cases, K must be greater than 0. In MXFP4 use cases, x and weight must both be FLOAT4, and each can be FLOAT4_E2M1 or FLOAT4_E1M2. K must be an even number and cannot be 2.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdint>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_activation_quant_weight_nz.h"

/*!
 * \file test_aclnn_grouped_matmul_activation_quant_weight_nz.cpp
 * \brief GroupedMatmulActivationQuant WeightNz MXFP8 aclnn example for Ascend 950PR&950DT products.
 */

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
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

template <typename T1, typename T2>
auto CeilDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclFormat formatType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType, shape.data(),
        shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostData,
    const std::vector<std::vector<int64_t>> &shapes, void **deviceAddr, aclDataType dataType, aclFormat formatType,
    aclTensorList **tensorList)
{
    int size = static_cast<int>(shapes.size());
    std::vector<aclTensor *> tensors(size, nullptr);
    for (int i = 0; i < size; ++i) {
        int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, formatType, &tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensorList = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorListWithStorageShape(const std::vector<std::vector<T>> &hostData,
    const std::vector<std::vector<int64_t>> &viewShapes, const std::vector<std::vector<int64_t>> &storageShapes,
    const std::vector<std::vector<int64_t>> &strides, void **deviceAddr, aclDataType dataType, aclFormat formatType,
    aclTensorList **tensorList)
{
    int size = static_cast<int>(storageShapes.size());
    std::vector<aclTensor *> tensors(size, nullptr);
    for (int i = 0; i < size; ++i) {
        auto storageSize = GetShapeSize(storageShapes[i]) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr + i, storageSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        ret = aclrtMemcpy(*(deviceAddr + i), storageSize, hostData[i].data(), storageSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        tensors[i] = aclCreateTensor(viewShapes[i].data(), viewShapes[i].size(), dataType, strides[i].data(), 0,
            formatType, storageShapes[i].data(), storageShapes[i].size(), *(deviceAddr + i));
    }
    *tensorList = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}

int RunGroupedMatmulActivationQuantWeightNz(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t e = 2;
    int64_t m = 4;
    int64_t k = 64;
    int64_t n = 128;
    int64_t kScaleBlock = CeilDiv(k, 64);
    int64_t nScaleBlock = CeilDiv(n, 64);

    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> groupListShape = {e};
    std::vector<std::vector<int64_t>> weightViewShapes = {{e, k, n}};
    std::vector<std::vector<int64_t>> weightStorageShapes = {{e, CeilDiv(n, 32), CeilDiv(k, 16), 16, 32}};
    std::vector<std::vector<int64_t>> weightStrides = {{k * n, n, 1}};
    std::vector<std::vector<int64_t>> weightScaleShapes = {{e, kScaleBlock, n, 2}};
    std::vector<int64_t> xScaleShape = {m, kScaleBlock, 2};
    std::vector<int64_t> yShape = {m, n};
    std::vector<int64_t> yScaleShape = {m, nScaleBlock, 2};

    std::vector<uint8_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<int64_t> groupListHostData = {2, 2};
    std::vector<uint8_t> weightHostData(GetShapeSize(weightStorageShapes[0]), 1);
    std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleShapes[0]), 1);
    std::vector<uint8_t> xScaleHostData(GetShapeSize(xScaleShape), 1);
    std::vector<uint8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<uint8_t> yScaleHostData(GetShapeSize(yScaleShape), 0);

    void *xDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *xScaleDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *yScaleDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *groupList = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *weightScale = nullptr;
    aclTensorList *bias = nullptr;
    aclTensor *xScale = nullptr;
    aclTensor *y = nullptr;
    aclTensor *yScale = nullptr;

    ret = CreateAclTensor<uint8_t>(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
        aclFormat::ACL_FORMAT_ND, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
        aclFormat::ACL_FORMAT_ND, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<std::vector<uint8_t>> weightHostDataList = {weightHostData};
    ret = CreateAclTensorListWithStorageShape<uint8_t>(weightHostDataList, weightViewShapes, weightStorageShapes,
        weightStrides, &weightDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorListPtr(
        weight, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<std::vector<uint8_t>> weightScaleHostDataList = {weightScaleHostData};
    ret = CreateAclTensorList<uint8_t>(weightScaleHostDataList, weightScaleShapes, &weightScaleDeviceAddr,
        aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &weightScale);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightScaleTensorListPtr(
        weightScale, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
        aclFormat::ACL_FORMAT_ND, &xScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xScaleTensorPtr(xScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
        aclFormat::ACL_FORMAT_ND, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
        aclFormat::ACL_FORMAT_ND, &yScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yScaleTensorPtr(yScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> yScaleDeviceAddrPtr(yScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const char *activationType = "gelu_tanh";
    int64_t groupListType = 1;
    aclIntArray *tuningConfig = nullptr;
    const char *quantMode = "mx";
    const char *roundMode = "rint";
    int64_t scaleAlg = 0;
    float dstTypeMax = 0.0F;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    ret = aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(x, groupList, weight, weightScale, bias, xScale,
        activationType, groupListType, tuningConfig, quantMode, roundMode, scaleAlg, dstTypeMax, y, yScale,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnGroupedMatmulActivationQuantWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnGroupedMatmulActivationQuantWeightNz failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("y shape is: [%ld, %ld]\n", yShape[0], yShape[1]);
    LOG_PRINT("yScale shape is: [%ld, %ld, %ld]\n", yScaleShape[0], yScaleShape[1], yScaleShape[2]);
    LOG_PRINT("GroupedMatmulActivationQuantWeightNz example execute success.\n");
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = RunGroupedMatmulActivationQuantWeightNz(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
        LOG_PRINT("RunGroupedMatmulActivationQuantWeightNz failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
