# aclnnAlltoAllQuantMatmulV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:59:17.887Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

- Description: Fuses the AlltoAll communication, Permute (ensuring address contiguity after communication), Quant, Matmul, and Dequant computations. **Communication first, computation later**. Supports K-C quantization, K-C dynamic quantization, and the mx [quantization mode](../../../docs/en/context/quant_mode_introduction.md).
- Formula: Assume that the input shape of x1 is (BS, H). In the mx quantization scenario, the input shape of x1ScaleOptional is (BS, ceil(H/64), 2), and rankSize is the number of NPU devices.

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>:

    - K-C Quantization Scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      output_{quant} = x1 @ x2 \\
      output = output_{quant} \times x1_{scale} \times x2_{scale} \\
      output = output + bias
      $$

    - K-C Dynamic Quantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      x1_{quant}, x1_{scale} = Quant(permutedOut) \\
      output_{quant} = x1_{quant} @ x2 \\
      output = output_{quant} \times x1_{scale} \times x2_{scale} \\
      output = output + bias
      $$

  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:

    - K-C Dynamic Quantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      dynQuantX1, dynQuantX1Scale = dynamicQuant(permutedOut) \\
      output = (dynQuantX1@x2 + bias) \times dynQuantX1Scale \times x2Scale
      $$

    - mx quantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      commScale = AlltoAll(x1Scale.view(rankSize, BS/rankSize, ceil(H/64), 2)) \\
      permutedScale = commScale.permute(1, 0, 2, 3).view(BS/rankSize, ceil(H/64)*rankSize, 2) \\
      output = \sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (permutedOut @ x2 * (permutedScale * x2Scale)) + bias
      $$

  <!-- end id8 -->

Compared with the `aclnnAlltoAllQuantMatmul` API, this API has the following changes:

- The `commMode` parameter is added. Users can use this parameter to specify the communication engine used by the chip.

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: `commMode` supports `ai_cpu` and `ccu`.
  <!-- end id9 -->
  <!-- npu="910b" id10 -->
  - <term>Atlas A2 products</term>: No new features are added. `commMode` supports only `aiv`.

  <!-- end id10 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAlltoAllQuantMatmulV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the "aclnnAlltoAllQuantMatmulV2" API to execute the computation.

```cpp
aclnnStatus aclnnAlltoAllQuantMatmulV2GetWorkspaceSize(
  const aclTensor*   x1,
  const aclTensor*   x2,
  const aclTensor*   biasOptional,
  const aclTensor*   x1ScaleOptional,
  const aclTensor*   x2Scale,
  const aclTensor*   commScaleOptional,
  const aclTensor*   x1OffsetOptional,
  const aclTensor*   x2OffsetOptional,
  const char*        group,
  const char*        commMode,
  const aclIntArray* alltoAllAxesOptional,
  int64_t            x1QuantMode,
  int64_t            x2QuantMode,
  int64_t            commQuantMode,
  int64_t            commQuantDtype,
  int64_t            x1QuantDtype,
  int64_t            groupSize,
  bool               transposeX1,
  bool               transposeX2,
  const aclTensor*   output,
  const aclTensor*   alltoAllOutOptional,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllQuantMatmulV2(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAlltoAllQuantMatmulV2GetWorkspaceSize

- **Parameters**:

    <table style="undefined;table-layout: fixed; width: 1556px"> <colgroup>
    <col style="width: 154px">
    <col style="width: 123px">
    <col style="width: 270px">
    <col style="width: 325px">
    <col style="width: 245px">
    <col style="width: 120px">
    <col style="width: 203px">
    <col style="width: 116px">
    </colgroup>
    <thead>
    <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Instruction</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Shape</th>
    <th>non-contiguous tensor</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>x1</td>
    <td>Input</td>
    <td>Left Matrix input of the fusion operator, corresponding to x1 in the formula.</td>
    <td>The result of AlltoAll communication and Permute operations on this input is used as the Left Matrix input for MatMul computation. The data type restrictions vary by product model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT4</td>
    <td>ND</td>
    <td>2 dimensions, shape (BS, H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right Matrix input of the fusion operator, also the Right Matrix of MatMul computation, corresponding to x2 in the formula.</td>
    <td>Right Matrix input for the MatMul computation. Restrictions on data types and non-contiguous tensors vary by product. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT8, INT4</td>
    <td>ND</td>
    <td>2D, shape (H*rankSize, N)</td>
    <td>Support varies by product. For details, see <a href="#constraints">Constraints</a>.</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>Input</td>
    <td>Optional Input, the bias accumulated after the matrix multiplication, corresponding to bias in the formula.</td>
    <td>Restrictions on data types vary by product. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>1D, shape (N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x1ScaleOptional</td>
    <td>Input</td>
    <td>Optional input. Quantization scale of the left matrix.</td>
    <td>Required in K-C quantization and mx quantization scenarios. In K-C dynamic quantization scenarios, x1ScaleOptional can be passed in as smoothScale. In this case, its data type must be the same as that of x1.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1D/3D. In K-C quantization scenarios, the shape is (BS). In K-C dynamic quantization scenarios, the shape is (H*rankSize). In mx quantization scenarios, the shape is (BS, ceil(H/64), 2).</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2Scale</td>
    <td>Input</td>
    <td>Quantization coefficients of the right matrix.</td>
    <td>Corresponding to x2Scale in the formula.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1D/3D. For K-C quantization and K-C dynamic quantization, the shape is (N). For the mx quantization scenario, the shape is (N, ceil(H*rankSize/64), 2).</td>
    <td>x</td>
    </tr>
    <tr>
    <td>commScaleOptional</td>
    <td>Input</td>
    <td>Optional input, quantization coefficients for low-bit communication.</td>
    <td>Reserved parameter. Low-bit communication is not supported yet.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1OffsetOptional</td>
    <td>Input</td>
    <td>Optional input, quantization offset of the left matrix.</td>
    <td>Reserved parameter, not supported yet.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2OffsetOptional</td>
    <td>Input</td>
    <td>Optional input, quantization offset of the right matrix.</td>
    <td>Reserved parameter, not supported yet.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>alltoAllAxesOptional</td>
    <td>Input</td>
    <td>Optional input. Direction of data exchange for AlltoAll and Permute.</td>
    <td>Only empty or [-2, -1] is supported. If empty is passed, it defaults to [-2, -1], indicating that the input is converted from (BS, H) to (BS/rankSize, rankSize*H).</td>
    <td>aclIntArray* (element type INT64)</td>
    <td>-</td>
    <td>1 dimension, shape (2)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Input</td>
    <td>Host-side string that identifies the group, that is, the communicator name. Obtain commName through the Hccl API HcclGetCommName and use it as this parameter.</td>
    <td>String length: (0, 128).</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commMode</td>
    <td>Input</td>
    <td>Specifies the current communication type.</td>
    <td>Supports "", "ai_cpu", or "ccu".</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantMode</td>
    <td>Input</td>
    <td>Quantization mode of the left matrix.</td>
    <td>The value is subject to different restrictions depending on the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2QuantMode</td>
    <td>Input</td>
    <td>Quantization mode of the right matrix.</td>
    <td>The value is subject to different restrictions depending on the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>Input</td>
    <td>Quantization mode for low-bit communication.</td>
    <td>Reserved parameter. Currently, only 0 is supported, indicating no quantization.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantDtype</td>
    <td>Input</td>
    <td>Quantization type for low-bit communication.</td>
    <td>Reserved Parameter. Currently, only -1 can be configured, indicating ACL_DT_UNDEFINED.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantDtype</td>
    <td>Input</td>
    <td>Quantization type of the left matrix for Matmul computation.</td>
    <td>The result after AlltoAll communication and the Permute operation. It is quantized based on this parameter and used as the left matrix input for MatMul computation. Value restrictions vary by device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupSize</td>
    <td>Input</td>
    <td>Quantization group sizes in three directions for the Matmul computation, composed of three values: groupSizeM, groupSizeN, and groupSizeK. Each value occupies 16 bits, and the three values occupy the lower 48 bits of the int64_t groupSize in total (the upper 16 bits of groupSize are invalid).</td>
    <td><ul><li>In the mx quantization scenario, only [groupSizeM, groupSizeN, groupSizeK] = [1, 1, 32] is supported. For details about the corresponding groupSize values, see <a href="#constraints">Constraints</a>. In other quantization scenarios, the default configuration is 0 and the values do not take effect.</li><li>Parameter auto-deduction is supported. When any one or more of groupSizeM, groupSizeN, and groupSizeK decomposed according to the formula is 0, the operator automatically deduces these parameter values. For details about the rules, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX1</td>
    <td>Input</td>
    <td>Indicates whether the left matrix is transposed.</td>
    <td>Setting it to True is not supported yet.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX2</td>
    <td>Input</td>
    <td>Indicates whether the right matrix is transposed.</td>
    <td>If set to True, the right matrix shape is (N, rankSize*H). Must be set to True in mx quantization mode.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>Output</td>
    <td>Output</td>
    <td>Final computation result.</td>
    <td>Data type restrictions vary by device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>2 dimensions, shape (BS/rankSize, N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllOutOptional</td>
    <td>Output</td>
    <td>Receives the content after AlltoAll and Permute. The data type is the same as that of input x1.</td>
    <td>If nullptr is passed, no communication output is returned.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT4</td>
    <td>ND</td>
    <td>2D, shape (BS/rankSize, rankSize*H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size to be requested on the device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the computation flow of the operator.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

  The mapping between the enumeration values of x1QuantMode, x2QuantMode, and commQuantMode and [Quantization Mode](../../../docs/en/context/quant_mode_introduction.md) is as follows:
  - 0: no quantization
  - 1: pertensor
  - 2: perchannel
  - 3: pertoken
  - 4: pergroup
  - 5: perblock
  - 6: mx quantization
  - 7: pertoken dynamic quantization

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:650px">
    </colgroup>
    <thead>
     <tr>
       <th>Return Value</th>
       <th>Error Code</th>
       <th>Description</th>
     </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A required input or output parameter tensor is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data type of the input or output is out of the supported range.</td>
    </tr>
    <tr>
        <td>The input tensor is an empty tensor.</td>
    </tr>
    <tr>
        <td>alltoAllAxesOptional is invalid.</td>
    </tr>
    <tr>
        <td>transposeX1 is true.</td>
    </tr>
    <tr>
        <td>The communicator length is invalid.</td>
    </tr>
    <tr>
        <td>The input/output tensor dimensions are invalid.</td>
    </tr>
    <tr>
        <td>The input/output format is a private format.</td>
    </tr>
      </tbody>
  </table>

## aclnnAlltoAllQuantMatmulV2

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
    <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size requested on the device, obtained by calling the first-phase API aclnnAlltoAllQuantMatmulV2GetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor that contains the operator computation process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
    </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- aclnnAlltoAllQuantMatmulV2 supports deterministic computation by default.
- Number of NPU cards (rankSize), with different restrictions based on the device model:

  <!-- npu="910b" id11 -->
  - <term>Atlas A2 products</term>: Supports 2, 4, or 8 cards.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: Supports 2, 4, 8, or 16 cards.
  <!-- end id12 -->

- The variable BS used in the shape in Parameters must be divisible by rankSize.
- The values of BS and N must not exceed 2147483647 (INT32_MAX). The value of BS must not be less than 2, and the value of N must not be less than 1.
- Empty tensors are not supported.
- Support for non-contiguous tensors varies by device model:

  <!-- npu="910b" id13 -->
  - <term>Atlas A2 products</term>: No non-contiguous tensors are supported.
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950PR&950DT products</term>: Only x2 is supported as a non-contiguous tensor. Other non-contiguous tensors are not supported.
  <!-- end id14 -->

- The input x1, x2, x2Scale, and output must not be null pointers, and

  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>: In the per-token dynamic quantization scenario where x1QuantMode is set, x1ScaleOptional is not supported.
  <!-- end id15 -->

- Constraints on groupSize:
  - groupSize is valid only when both the x1ScaleOptional and x2Scale inputs are two-dimensional or higher; otherwise, 0 must be passed in.
  - The passed groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. If one or more of them is 0, groupSizeM, groupSizeN, and groupSizeK are reset for computation based on the shapes of the x1/x2/x1ScaleOptional/x2Scale inputs. Principle: If groupSizeM=0, the quantization grouping value in the m direction is inferred by the API. The inference formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the same as m in the x1 shape, and scaleM is the same as m in the x1ScaleOptional shape. The k and n directions follow the same rule.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
  - If the input x and scale are divisible in each dimension, and the automatically derived groupSizeM, groupSizeN, and groupSizeK satisfy [1,1,32], groupSize supports the following values in the mx quantization scenario:

    | groupSize | Based on the Formula [gsM,gsN,gsK] | Based on Automatic Derivation [gsM,gsN,gsK] |
    | :------: | :------: | :------: |
    | 4295032864 | [1,1,32] | - |
    | 0 | [0,0,0] | [1,1,32] |
    | 32 | [0,0,32] | [1,1,32] |
    | 65536 | [0,1,0] | [1,1,32] |
    | 65568 | [0,1,32] | [1,1,32] |
    | 4294967296 | [1,0,0] | [1,1,32] |
    | 4294967328 | [1,0,32] | [1,1,32] |
    | 4295032832 | [1,1,0] | [1,1,32] |

- The data types, data dimensions, and quantization modes of the input and output of this operator have different restrictions depending on the device model:

  <!-- npu="910b" id16 -->
  - <term>Atlas A2 products</term>:
    - Quantization mode:
      - Currently supports per-token quantization and per-token dynamic quantization for the left matrix, with x1QuantMode=3 or 7; and per-channel quantization for the right matrix, with x2QuantMode=2.
    - Type constraints:
      - The data types of x1 and alltoAllOutOptional must be consistent.
      - If x1, x2, or alltoallout is of the int32 type, it is treated as eight packed int4 values and will be reinterpreted as int4.
      - In A16W8 and A16W4 modes, for the smoothQuant scenario, the data types of x1ScaleOptional and x1 must be consistent.
      - For A16W8, the supported data type combinations of x1, x2, biasOptional, and output are:

        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | FLOAT16 | INT8 | FLOAT16 | FLOAT16 |
        | FLOAT16 | INT8 | FLOAT32 | FLOAT16 |
        | BFLOAT16 | INT8 | BFLOAT16 | BFLOAT16 |
        | BFLOAT16 | INT8 | FLOAT32 | BFLOAT16 |

      - For A16W4, the supported data type combinations of x1, x2, biasOptional, and output are:

        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | FLOAT16 | INT4 | FLOAT16 | FLOAT16 |
        | FLOAT16 | INT4 | FLOAT32 | FLOAT16 |
        | BFLOAT16 | INT4 | BFLOAT16 | BFLOAT16 |
        | BFLOAT16 | INT4 | FLOAT32 | BFLOAT16 |

      - For A4W4, x1ScaleOptional supports only FLOAT32. The supported data type combinations of x1, x2, biasOptional, and output are:

        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | INT4 | INT4 | FLOAT16 | FLOAT16 |
        | INT4 | INT4 | FLOAT32 | FLOAT16 |
        | INT4 | INT4 | BFLOAT16 | BFLOAT16 |
        | INT4 | INT4 | FLOAT32 | BFLOAT16 |

    - Dimension Constraints:
      - For A16W8, rankSize * H must be divisible by 16; value range of rankSize * H: [1, 35000].
      - For A16W4, rankSize * H must be divisible by 16; N must be even; value range of rankSize * H: [1, 35000].
      - For A4W4, H and N must be even numbers. Value range of rankSize * H: [1, 35000].
  <!-- end id16 -->
  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Quantization Mode:
      - Currently supported: K-C dynamic quantization, left matrix perToken dynamic quantization, x1QuantMode=7, right matrix perChannel quantization, x2QuantMode=2; mx quantization, left matrix mx quantization, x1QuantMode=6, right matrix mx quantization, x2QuantMode=6.
    - Type constraints:
      - The data types of x1 and alltoAllOutOptional must be consistent.
      - x1QuantDtype takes effect in the K-C Dynamic Quantization scenario, supporting 35 (indicating aclDataType.ACL_FLOAT8_E5M2) and 36 (indicating aclDataType.ACL_FLOAT8_E4M3FN). It does not take effect in other quantization scenarios.
      - biasOptional can be empty.
      - The supported input/output data type combinations are:
        - K-C Dynamic Quantization:

          | x1 | x2 | biasOptional | output | x1QuantMode | x2QuantMode | x1ScaleOptional | x2Scale |
          | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
          | FLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |

        - mx Quantization:

          | x1 | x2 | biasOptional | output | x1QuantMode | x2QuantMode | x1ScaleOptional | x2Scale |
          | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |

    - Dimension Constraints:
      - The range of rankSize * H only supports [1, 65535].
      - In the mx quantization scenario, H must be divisible by 64.
      - In the mx quantization scenario, x2 must be transposed, with shape (H*rankSize, N) and transposeX2 set to True.
  <!-- end id17 -->

- The communication-computation fusion operator does not support concurrent calls, and different communication-computation fusion operators also do not support concurrent calls.
- Cross-supernode communication is not supported. Only intra-supernode communication is supported.
- The communication engine parameter commMode has different restrictions depending on the device model:

   <!-- npu="910b" id18 -->
   - <term>Atlas A2 products</term>: Only input `aiv` is supported.
   <!-- end id18 -->
   <!-- npu="950" id19 -->
   - <term>Ascend 950PR&950DT products</term>: Inputs `ai_cpu` and `ccu` are supported.
   <!-- end id19 -->

- Communication engine constraints:

   <!-- npu="910b" id20 -->
   - <term>Atlas A2 products</term>: MTE communication is supported.
   <!-- end id20 -->
   <!-- npu="950" id21 -->
   - <term>Ascend 950PR&950DT products</term>:
      - Supports CCU communication and AI_CPU communication.
      - CCU communication supports only interconnection within a single-server UB domain, while AI_CPU supports interconnection within a UB domain across servers.
      - Communicator constraint: only one communication method can be used in the same communicator.
      - When the CCU communication engine is used, the total alltoall(x1) communication volume within a communication domain must not exceed 63 \* 256MB. The total communication volume is calculated as (BS / rankSize) \* H \* sizeof(x1_dtype). Due to shape differences, the actual supported total volume may be slightly smaller.
      - When the CCU communication engine is used, x1Scale performs an additional AllToAll in the MX quantization scenario. One communication task must be reserved for x1Scale, so the total alltoall(x1) communication volume must not exceed 62 \* 256MB. The x1Scale AllToAll volume must not exceed 256MB and is calculated as (BS / rankSize) \* ceil(H / 64) \* 2 bytes.

   <!-- end id21 -->

## Example

The following is sample code for reference only. For details about compilation and execution, see Compile and Run Samples.

NOTE: The sample code calls some APIs of the HCCL collective communication library: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="910b" id22 -->
- <term>Atlas A2 products</term>:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_allto_all_quant_matmul_v2.h"

    int ndev = 2;

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

    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // Call aclrtMalloc to allocate device-side memory.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy host-side data to device-side memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        // Calculate the strides of a contiguous tensor.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        // Call aclCreateTensor to create an aclTensor.
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    int launchOneThreadAlltoAllQuantMatmul(Args &args)
    {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128}; // ndev = 2; the shape of x2Shape remains unchanged before and after transposition
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> allToAllOutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *allToAllOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1ScaleOptional = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor* commScaleOptional = nullptr;
        aclTensor* x1OffsetOptional = nullptr;
        aclTensor* x2OffsetOptional = nullptr;
        aclTensor *out = nullptr;
        aclTensor *allToAllOut = nullptr;

        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        int64_t x1QuantMode = 3;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t x1QuantDtype = 2;
        int64_t groupSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long allToAllOutShapeSize = GetShapeSize(allToAllOutShape);
        std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<op::fp16_t> biasHostData(biasShapeSize, 1);
        std::vector<float> x2ScaleHostData(x2ScaleShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        std::vector<op::fp16_t> allToAllOutHostData(allToAllOutShapeSize, 0);

        // Create tensors.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(allToAllOutHostData, allToAllOutShape, &allToAllOutDeviceAddr, aclDataType::ACL_FLOAT16, &allToAllOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API.
        ret = aclnnAlltoAllQuantMatmulV2GetWorkspaceSize(x1, x2, bias, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
                                                hcom_name, "aiv", alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
                                                groupSize, false, true,
                                                out, allToAllOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllQuantMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAlltoAllQuantMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmulV2 failed. ERROR: %d\n", ret); return ret);
        // (boilerplate) Wait synchronously for the task to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllQuantMatmulV2 execute success \n", args.rankId);
        // Release device resources. Modify them according to the specific API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (allToAllOut != nullptr) {
            aclDestroyTensor(allToAllOut);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        // This sample is implemented based on Atlas A2 and must run on Atlas A2.
        int ret = aclInit(nullptr);
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // Initialize the collective communication communicator.
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // Start multiple threads.
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllQuantMatmul, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id22 -->
<!-- npu="950" id23 -->
- <term>Ascend 950PR&950DT products</term>:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_allto_all_quant_matmul_v2.h"

    int ndev = 2;

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

    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i: shape) {
    shapeSize *= i;
    }
    return shapeSize;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
    shape.data(), shape.size(), *deviceAddr);
    return 0;
    }

    struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    };

    int launchOneThreadAlltoAllQuantMatmul(Args &args)
    {
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcom_name[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcom_name);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
    args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> allToAllOutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *allToAllOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1ScaleOptional = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor* commScaleOptional = nullptr;
        aclTensor* x1OffsetOptional = nullptr;
        aclTensor* x2OffsetOptional = nullptr;
        aclTensor *out = nullptr;
        aclTensor *allToAllOut = nullptr;

        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        int64_t x1QuantMode = 7;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t x1QuantDtype = 35;
        int64_t groupSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long allToAllOutShapeSize = GetShapeSize(allToAllOutShape);
        std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<float> biasHostData(biasShapeSize, 1);
        std::vector<float> x2ScaleHostData(x2ScaleShapeSize, 1);
        std::vector<float> outHostData(outShapeSize, 0);
        std::vector<op::fp16_t> allToAllOutHostData(allToAllOutShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(allToAllOutHostData, allToAllOutShape, &allToAllOutDeviceAddr, aclDataType::ACL_FLOAT16, &allToAllOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API.
        ret = aclnnAlltoAllQuantMatmulV2GetWorkspaceSize(x1, x2, bias, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
                                                hcom_name, "ai_cpu", alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
                                                groupSize, false, false,
                                                out, allToAllOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllQuantMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAlltoAllQuantMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmulV2 failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Synchronously wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllQuantMatmulV2 execute success \n", args.rankId);
        // Release device resources, which need to be modified based on the specific API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (allToAllOut != nullptr) {
            aclDestroyTensor(allToAllOut);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
    // This sample is implemented based on Ascend 950PR&950DT products and must be run on Ascend 950PR&950DT products.
    int ret = aclInit(nullptr);
    int32_t devices[ndev];
    for (int i = 0; i < ndev; i++) {
    devices[i] = i;
    }
    HcclComm comms[128];
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    // Initialize the communicator.
    for (int i = 0; i < ndev; i++) {
    ret = aclrtSetDevice(devices[i]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    Args args[ndev];
    aclrtStream stream[ndev];
    aclrtContext context[ndev];
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(&context[rankId], rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream[rankId]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    }
    // Start multiple threads.
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
    args[rankId].rankId = rankId;
    args[rankId].hcclComm = comms[rankId];
    args[rankId].stream = stream[rankId];
    args[rankId].context = context[rankId];
    threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllQuantMatmul, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
    threads[rankId]->join();
    }
    aclFinalize();
    return 0;
    }
    ```

<!-- end id23 -->
