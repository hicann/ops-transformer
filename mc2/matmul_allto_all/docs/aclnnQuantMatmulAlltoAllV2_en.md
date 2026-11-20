# aclnnQuantMatmulAlltoAllV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:46:54.486Z -->

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

- Description: Fuses quantized Matmul computation, Permute (ensuring contiguous addresses after communication), and AlltoAll communication, **computing first and then communicating**. Supports K-C quantization and the mx [Quantization Mode](../../../docs/en/context/quant_mode_introduction.md).

- Formula: Assume the shape of x1 is (BS, H1), the shape of x2 is (H1, H2), and rankSize is the number of NPUs.

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>:

    - K-C Quantization Scenario:

      $$
      computeOut = (x1 @ x2) * x1Scale * x2Scale  + bias \\
      permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
      $$

  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:

    - K-C Quantization Scenario:

      $$
      computeOut = (x1 @ x2 + bias) * x1Scale * x2Scale \\
      permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
      $$

    - mx quantization scenario:

      $$
      computeOut = \sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (x1 @ x2 * (x1Scale * x2Scale)) + bias \\
      permutedOut = computeOut.view(BS, rankSize, H2 / rankSize).permute(1, 0, 2) \\
      output = AlltoAll(permutedOut).view(rankSize * BS, H2 / rankSize)
      $$

  <!-- end id8 -->

Compared with the `aclnnQuantMatmulAlltoAll` API, this API includes the following changes:

- Adds the `commMode` parameter. Users can use this parameter to specify the communication engine used by the chip.

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: `commMode` supports `ai_cpu` and `ccu`.
  <!-- end id9 -->
  <!-- npu="910b" id10 -->
  - <term>Atlas A2 products</term>: No new features are added. `commMode` supports only `aiv`.

  <!-- end id10 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnQuantMatmulAlltoAllV2GetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then call the "aclnnQuantMatmulAlltoAllV2" API to execute the computation.

```cpp
aclnnStatus aclnnQuantMatmulAlltoAllV2GetWorkspaceSize(
const aclTensor*   x1,
const aclTensor*   x2,
const aclTensor*   biasOptional,
const aclTensor*   x1Scale,
const aclTensor*   x2Scale,
const aclTensor*   commScaleOptional,
const aclTensor*   x1OffsetOptional,
const aclTensor*   x2OffsetOptional,
const aclIntArray* alltoAllAxesOptional,
const char*        group,
const char*        commMode,
int64_t            x1QuantMode,
int64_t            x2QuantMode,
int64_t            commQuantMode,
int64_t            commQuantDtype,
int64_t            groupSize,
bool               transposeX1,
bool               transposeX2,
const aclTensor*   output,
uint64_t*          workspaceSize,
aclOpExecutor**    executor);
```

```cpp
aclnnStatus aclnnQuantMatmulAlltoAllV2(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulAlltoAllV2GetWorkspaceSize

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
    <th>Dimension (shape)</th>
    <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>x1</td>
    <td>Input</td>
    <td>Left matrix input of the fused operator, corresponding to x1 in the formula.</td>
    <td>This input serves as the left matrix input for the MatMul computation; the supported data types vary by device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT8</td>
    <td>ND</td>
    <td>2 dimensions, shape (BS, H1)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right matrix input of the fused operator, corresponding to x2 in the formula.</td>
    <td>Directly serves as the Right Matrix input for the MatMul computation; restrictions on the data type and non-contiguous tensors vary by device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT8</td>
    <td>ND</td>
    <td>2 dimensions, shape (H1, H2)</td>
    <td>Support varies by device model. See <a href="#constraints">Constraints</a>.</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>Input</td>
    <td>Bias added after the matrix multiplication, corresponding to bias in the formula.</td>
    <td>Restrictions on the data type vary by device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>1D, shape (H2).</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x1Scale</td>
    <td>Input</td>
    <td>Quantization coefficient of the left matrix.</td>
    <td>x1Scale in the corresponding formula.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1D/3D. For the K-C quantization use case, the shape is (BS). For the mx quantization use case, the shape is (BS, ceil(H1/64), 2).</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2Scale</td>
    <td>Input</td>
    <td>Quantization coefficients of the right matrix.</td>
    <td>x2Scale in the corresponding formula.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1D/3D. The shape is (H2) for the K-C quantization use case. The shape is (H2, ceil(H1/64), 2) for the mx quantization use case.</td>
    <td>x</td>
    </tr>
    <tr>
    <td>commScaleOptional</td>
    <td>Input</td>
    <td>Quantization coefficient for low-bit communication.</td>
    <td>Reserved Parameter. Low-bit communication is not supported currently.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1OffsetOptional</td>
    <td>Input</td>
    <td>Quantization offset of the left matrix.</td>
    <td>Reserved parameter, not supported yet.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2OffsetOptional</td>
    <td>Input</td>
    <td>Quantization offset of the right matrix.</td>
    <td>Reserved parameter, not supported yet.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>alltoAllAxesOptional</td>
    <td>Input</td>
    <td>Direction of data exchange in AlltoAll and Permute.</td>
    <td>Supports configuring empty or [-1, -2]. If empty is passed, it defaults to [-1, -2], which means converting the input from (BS, H2) to (BS*rankSize, H2/rankSize).</td>
    <td>aclIntArray* (element type INT64)</td>
    <td>-</td>
    <td>1 dimension, shape (2)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Input</td>
    <td>String identifying the column group, that is, the communicator name. Obtain commName through the Hccl API HcclGetCommName and use it as this parameter.</td>
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
    <td>The value can be "aiv", "ai_cpu", or "ccu".</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantMode</td>
    <td>Input</td>
    <td>Quantization mode of the left matrix.</td>
    <td>The allowed values vary by device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2QuantMode</td>
    <td>Input</td>
    <td>Quantization mode of the right matrix.</td>
    <td>The allowed values vary by device model. For details, see <a href="#constraints">Constraints</a>.</td>
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
    <td>Reserved Parameter. Currently, only -1 is supported, indicating ACL_DT_UNDEFINED.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupSize</td>
    <td>Input</td>
    <td>Size of quantization groups in three directions for the Matmul computation. Composed of three values: groupSizeM, groupSizeN, and groupSizeK, each occupying 16 bits and together occupying the lower 48 bits of groupSize of type int64_t (the value of the upper 16 bits in groupSize is invalid).</td>
    <td><ul><li>In the mx quantization scenario, only [groupSizeM, groupSizeN, groupSizeK] = [1, 1, 32] is supported. For the specific value of the corresponding groupSize, see <a href="#constraints">Constraints</a>. For other quantization scenarios, the default value is 0 and does not take effect.</li><li>Automatic parameter deduction is supported. When any one or more of groupSizeM, groupSizeN, and groupSizeK derived from the formula is 0, the operator automatically deduces these parameter values. For details, see <a href="#constraints">Constraints</a></li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX1</td>
    <td>Input</td>
    <td>Indicates whether the left matrix is transposed.</td>
    <td>Setting to True is not supported yet.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX2</td>
    <td>Input</td>
    <td>Specifies whether the right matrix is transposed.</td>
    <td>When set to True, the shape of the right matrix is (H2, H1). Must be set to True in mx quantization mode.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>Output</td>
    <td>Output</td>
    <td>Final computation result.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>2 dimensions, shape (BS*rankSize, H2/rankSize).</td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size to be allocated on the Device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor that contains the operator computation flow.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

  The relationship between the enumeration values of x1QuantMode, x2QuantMode, and commQuantMode and [Quantization Mode](../../../docs/en/context/quant_mode_introduction.md) is as follows:
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

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
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
      <td>The required input or output tensor parameter is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data types of input and output are not in the supported range.</td>
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
        <td>The dimensions of input and output tensors are invalid.</td>
    </tr>
    <tr>
        <td>The data format of input and output is a private format.</td>
    </tr>
      </tbody>
  </table>

## aclnnQuantMatmulAlltoAllV2

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
        <td>The workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The workspace size allocated on the Device side, obtained by calling the first-phase API aclnnQuantMatmulAlltoAllV2GetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream on which the task is executed.</td>
    </tr>
    </tbody>
    </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- aclnnQuantMatmulAlltoAllV2 defaults to deterministic computation.
- Number of NPU cards (rankSize) varies by device model:

  <!-- npu="910b" id11 -->
  - <term>Atlas A2 products</term>: Supports 2, 4, or 8 cards.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: Supports 2, 4, 8, or 16 cards.
  <!-- end id12 -->

- The variable H2 used in the shape in Parameters must be divisible by the number of NPU cards.
- The values of BS * rankSize and H2 must not exceed 2147483647 (INT32_MAX). The value of BS must not be less than 1, and the value of H2 must not be less than 2.
- Empty tensors are not supported.
- Support for non-contiguous tensors varies by device model:

  <!-- npu="910b" id13 -->
  - <term>Atlas A2 products</term>: No non-contiguous tensors are supported.
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950PR&950DT products</term>: Only x2 is supported as a non-contiguous tensor. Other non-contiguous tensors are not supported.
  <!-- end id14 -->

- The input x1, x2, x1Scale, x2Scale, and output must not be null pointers, and

  <!-- npu="910b" id15 -->
  - <term>Atlas A2 products</term>: biasOptional does not support null pointers.
  <!-- end id15 -->

- Constraints on groupSize:
  - The groupSize value is valid only when both x1Scale and x2Scale inputs are 2-dimensional or higher. In other scenarios, pass 0.
  - The passed-in groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. If one or more of them is 0, groupSizeM, groupSizeN, and groupSizeK are reset for computation based on the input shapes of x1/x2/x1Scale/x2Scale. Principle: If groupSizeM=0, the quantization group value along the m dimension is inferred by the API. The inference formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the same as m in the x1 shape, scaleM is the same as m in the x1Scale shape, and the same applies to the k and n dimensions.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
  - Assume that the input x and scale are divisible along all dimensions, and the automatically derived groupSizeM, groupSizeN, and groupSizeK satisfy [1,1,32]. The supported groupSize values in the mx quantization scenario are as follows:

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
    - Quantization Mode:
      - Currently supported: K-C quantization, left matrix perToken quantization with x1QuantMode=3, right matrix perChannel quantization with x2QuantMode=2.
      - The bias is added after quantization.
    - Type constraints:
      - The supported input/output data type combinations are:
        - K-C Quantization:

          | x1 | x2 | biasOptional | output |
          | :------: | :------: | :------: | :------: |
          | INT8 | INT8 | FLOAT16 | FLOAT16 |
          | INT8 | INT8 | FLOAT32 | FLOAT16 |
          | INT8 | INT8 | BFLOAT16 | BFLOAT16 |
          | INT8 | INT8 | FLOAT32 | BFLOAT16 |

    - Dimension Constraints:
      - The H1 range supports only [1, 65535].
  <!-- end id16 -->
  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Quantization Mode:
      - Currently supported: K-C quantization, left matrix perToken quantization, x1QuantMode=3, right matrix perChannel quantization, x2QuantMode=2; mx quantization, left matrix mx quantization, x1QuantMode=6, right matrix mx quantization, x2QuantMode=6.
    - Type constraints:
      - biasOptional can be empty.
      - The supported input/output data type combinations are:
        - K-C Quantization:

          | x1 | x2 | biasOptional | output | x1QuantMode | x2QuantMode | x1ScaleOptional | x2Scale |
          | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 3 | 2 | FLOAT32 | FLOAT32 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 3 | 2 | FLOAT32 | FLOAT32 |

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
      - The H1 range supports only [1, 65535].
      - In mx quantization scenarios, x2 must be transposed, with shape (H2, H1) and transposeX2 set to True.
      - In the mx quantization scenario, when the inputs x1 and x2 are FLOAT4_E2M1, H1 must be an even number, and ceil(H1/32) must be an even number.
  <!-- end id17 -->

- The communication-computation fusion operator does not support concurrent calls, and different communication-computation fusion operators also do not support concurrent calls.
- Cross-supernode communication is not supported. Only intra-supernode communication is supported.
- The communication engine parameter commMode has different restrictions depending on the device model:

   <!-- npu="910b" id18 -->
   - <term>Atlas A2 products</term>: Only `aiv` input is supported.
   <!-- end id18 -->
   <!-- npu="950" id19 -->
   - <term>Ascend 950PR&950DT products</term>: `ai_cpu` and `ccu` inputs are supported.
   <!-- end id19 -->

- Communication engine constraints:

   <!-- npu="910b" id20 -->
   - <term>Atlas A2 products</term>: MTE communication is supported.
   <!-- end id20 -->
   <!-- npu="950" id21 -->
   - <term>Ascend 950PR&950DT products</term>:
      - CCU communication and AI\_CPU communication are supported.
      - CCU supports only interconnection within the UB domain of a single device, while AI_CPU supports interconnection across the UB domains of multiple devices.
      - Communicator constraint: Only one communication method can be used within the same communicator.
      - When the CCU communication engine is used, the total alltoall(y) communication volume within a communication domain must not exceed 63 \* 256MB. The total communication volume is calculated as BS \* (H2 / rankSize) \* sizeof(y_dtype). Due to shape differences, the actual supported total volume may be slightly smaller.

   <!-- end id21 -->

## Example

The sample code is as follows and is for reference only. For details about how to compile and run the sample, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

Note: This sample code calls some HCCL collective communication library APIs: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. See [*HCCL API (C)*](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md).

<!-- npu="910b" id22 -->
- <term>Atlas A2 products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_quant_matmul_allto_all_v2.h"

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
        // Call aclrtMalloc to allocate memory on the device
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy data from the host to the device memory
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        // Calculate the strides of the contiguous tensor
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        // Call the aclCreateTensor API to create an aclTensor
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

    int launchOneThreadQuantMatmulAlltoAllV2(Args &args) {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128];
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x1ScaleShape = {32};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32 * ndev, 128 / ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x1ScaleDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1Scale = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor *out = nullptr;

        int64_t x1QuantMode = 3;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t groupSize = 0;

        int64_t a2aAxes[2] = {-1, -2};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        std::vector<int8_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<op::fp16_t> biasHostData(biasShapeSize, 1);
        std::vector<float> x1ScaleHostData(x1ScaleShapeSize, 1);
        std::vector<float> x2ScaleHostData(x2ScaleShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        // Create a tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API
        ret = aclnnQuantMatmulAlltoAllV2GetWorkspaceSize(x1, x2, bias, x1Scale, x2Scale, nullptr, nullptr, nullptr,
                                                      alltoAllAxesOptional, hcom_name, "aiv", x1QuantMode, x2QuantMode,
                                                      commQuantMode, commQuantDtype, groupSize, false, false,
                                                      out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulAlltoAllV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize computed by the first-phase API
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API
        ret = aclnnQuantMatmulAlltoAllV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAlltoAllV2 failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Synchronously wait for the task execution to complete
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnQuantMatmulAlltoAllV2 execute success \n", args.rankId);
        // Release device resources. Modify it based on the interface definition of the specific API.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (x1Scale != nullptr) {
            aclDestroyTensor(x1Scale);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
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

    int main(int argc, char *argv[]) {
        // This sample is implemented based on Atlas A2 and must run on Atlas A2.
        int ret;
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        ret = aclInit(nullptr);
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
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAlltoAllV2, std::ref(args[rankId])));
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

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnnop/aclnn_quant_matmul_allto_all_v2.h"

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
        // Call aclrtMalloc to allocate device memory.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy host data to device memory.
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

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    int launchOneThreadQuantMatmulAlltoAllV2(Args &args) {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128];
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x1ScaleShape = {32};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32 * ndev, 128 / ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x1ScaleDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1Scale = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor *out = nullptr;

        int64_t x1QuantMode = 3;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t groupSize = 0;

        int64_t a2aAxes[2] = {-1, -2};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x1ScaleShapeSize = GetShapeSize(x1ScaleShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        std::vector<int8_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<float> biasHostData(biasShapeSize, 1);
        std::vector<float> x1ScaleHostData(x1ScaleShapeSize, 1);
        std::vector<float> x2ScaleHostData(x2ScaleShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API.
        ret = aclnnQuantMatmulAlltoAllV2GetWorkspaceSize(x1, x2, bias, x1Scale, x2Scale, nullptr, nullptr, nullptr,
                                                       alltoAllAxesOptional, hcom_name, "ai_cpu", x1QuantMode, x2QuantMode,
                                                       commQuantMode, commQuantDtype, groupSize, false, false,
                                                       out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulAlltoAllV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnQuantMatmulAlltoAllV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAlltoAllV2 failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Synchronize and wait for the task to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnQuantMatmulAlltoAllV2 execute success \n", args.rankId);
        // Release device resources. Modify this part based on the specific API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (x1Scale != nullptr) {
            aclDestroyTensor(x1Scale);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
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

    int main(int argc, char *argv[]) {
        // This sample is implemented based on <term>Ascend 950PR&950DT products</term> and must run on <term>Ascend 950PR&950DT products</term>.
        int ret;
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        ret = aclInit(nullptr);
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
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAlltoAllV2, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id23 -->
