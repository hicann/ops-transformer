# aclnnAlltoAllQuantMatmul

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Fuses `AlltoAll` communication, `Permute` (to ensure contiguous memory addresses after communication), `Quant`, `Matmul`, and `Dequant` computation using a **communication-before-computation** sequence. It supports K-C quantization, K-C dynamic quantization, and mx [quantization mode](../../../docs/en/context/quant_mode_introduction.md).
- The calculation formula is as follows: Assume that the input shape of x1 is (BS, H), the input shape of x1Scale in the mx quantization scenario is (BS, ceil(H/64), 2), and rankSize is the number of NPUs.

  - <term>Atlas A2 training products/Atlas A2 inference products</term>:

    - K-C quantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      output_{quant} = x1 @ x2 \\
      output = output_{quant} \times x1_{scale} \times x2_{scale} \\
      output = output + bias
      $$

    - K-C dynamic quantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      x1_{quant}, x1_{scale} = Quant(permutedOut) \\
      output_{quant} = x1_{quant} @ x2 \\
      output = output_{quant} \times x1_{scale} \times x2_{scale} \\
      output = output + bias
      $$

  - Ascend 950PR/Ascend 950DT:

    - K-C dynamic quantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      dynQuantX1, dynQuantX1Scale = dynamicQuant(permutedOut) \\
      output = (dynQuantX1@x2 + bias) \times dynQuantX1Scale \times x2Scale
      $$

    - mxQuantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      commX1Scale = AlltoAll(x1Scale.view(rankSize, BS/rankSize, ceil(H/64), 2)) \\
      permuteX1Scale = commX1Scale.permute(1, 0, 2, 3) \\
      permutedX1Scale = permuteX1Scale.view(BS/rankSize, ceil(H/64)*rankSize, 2) \\
      output = (permutedOut* permutedX1Scale)@(x2* x2Scale) + bias
      $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAlltoAllQuantMatmulGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnAlltoAllQuantMatmul` is called to perform computation.

```cpp
aclnnStatus aclnnAlltoAllQuantMatmulGetWorkspaceSize(
  const aclTensor*   x1, 
  const aclTensor*   x2,
  const aclTensor*   biasOptional,
  const aclTensor*   x1ScaleOptional,
  const aclTensor*   x2Scale,
  const aclTensor*   commScaleOptional,
  const aclTensor*   x1OffsetOptional,
  const aclTensor*   x2OffsetOptional,
  const char*        group,
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
aclnnStatus aclnnAlltoAllQuantMatmul(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAlltoAllQuantMatmulGetWorkspaceSize

- ​**Parameters:**

    <table style="undefined;table-layout: fixed; width: 1543px"><colgroup>
    <col style="width: 186px">
    <col style="width: 123px">
    <col style="width: 283px">
    <col style="width: 295px">
    <col style="width: 181px">
    <col style="width: 122px">
    <col style="width: 206px">
    <col style="width: 147px">
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
    <td>x1</td>
    <td>Input</td>
    <td>Left matrix input of the fusion operator, corresponding to x1 in the formula.</td>
    <td>The result of AlltoAll communication and Permute operations on this input is used as the left matrix input for MatMul computation.<br>The constraints on the data type vary depending on the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, INT4</td>
    <td>ND</td>
    <td>2D, with shape of (BS, H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right matrix input of the fusion operator, which is also the right matrix for MatMul computation, corresponding to x2 in the formula.</td>
    <td>Serves as the right matrix input for MatMul computation.<br>The restrictions on the data type and discontinuity vary according to the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT8_E4M3FN, FLOAT8_E5M2, INT8, INT4</td>
    <td>ND</td>
    <td>2D, with shape of (H*rankSize, N)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>Optional input</td>
    <td>Bias to be accumulated after matrix multiplication, corresponding to bias in the formula.</td>
    <td>The constraints on the data type vary depending on the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>1D, with shape of (N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x1ScaleOptional</td>
    <td>Optional input</td>
    <td>Quantization coefficient of the left matrix.</td>
    <td>This parameter is required in the K-C quantization and mx quantization scenarios.<br>In the K-C dynamic quantization scenario, x1ScaleOptional can be passed as smoothScale. In this case, the type must be the same as that of x1.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1D/3D.<br>In the K-C quantization scenario, the shape is (BS).<br>In the K-C dynamic quantization scenario, the shape is (H x rankSize).<br>In the mx quantization scenario, the shape is (BS, ceil(H/64), 2)</td>.
    <td>x</td>
    </tr>
    <tr>
    <td>x2Scale</td>
    <td>Input</td>
    <td>Quantization coefficient of the right matrix.</td>
    <td>Corresponds to x2Scale in the formula.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1D/3D.<br>In the K-C quantization and K-C dynamic quantization scenarios, the shape is (N).<br>In the mx quantization scenario, the shape is (N, ceil(H*rankSize/64), 2)</td>.
    <td>x</td>
    </tr>
    <tr>
    <td>commScaleOptional</td>
    <td>Optional input</td>
    <td>Quantization coefficient for low-bit communication.</td>
    <td>Reserved. Low-bit communication is not supported currently.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1OffsetOptional</td>
    <td>Optional input</td>
    <td>Quantization bias of the left matrix.</td>
    <td>Reserved. This parameter is not supported currently.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2OffsetOptional</td>
    <td>Optional input</td>
    <td>Quantization bias of the right matrix.</td>
    <td>Reserved parameter, which is not supported currently.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>alltoAllAxesOptional</td>
    <td>Optional input</td>
    <td>Direction of AlltoAll and Permute data exchange.</td>
    <td>Only null or [-2, -1] is supported. If null is passed, [-2, -1] is used by default, indicating that the input is converted from (BS, H) to (BS/rankSize, rankSize*H).</td>
    <td>aclIntArray* (element type: INT64)</td>
    <td>-</td>
    <td>1D, shape: (2)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Input</td>
    <td>A string on the host identifying the communication domain name. The `commName` obtained via the HcclGetCommName API is used as the value for this parameter.</td>
    <td>The string length must be in the range (0, 128).</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantMode</td>
    <td>Input</td>
    <td>Quantization mode of the left matrix</td>
    <td>The constraints on the value vary depending on the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2QuantMode</td>
    <td>Input</td>
    <td>Quantization mode of the right matrix</td>
    <td>The constraints on the value vary depending on the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>Input</td>
    <td>Quantization mode of low-bit communication.</td>
    <td>Reserved. Currently, this parameter can only be set to 0, indicating that quantization is not performed.</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantDtype</td>
    <td>Input</td>
    <td>Quantization type of low-bit communication.</td>
    <td>Reserved. Currently, this parameter can only be set to -1, indicating ACL_DT_UNDEFINED.</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantDtype</td>
    <td>Input</td>
    <td>Quantization type of the left matrix of Matmul.</td>
    <td>Result after AlltoAll communication and Permute operations. The quantized result is used as the input of the left matrix for MatMul computation based on the configuration of this parameter. The value range varies depending on the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupSize</td>
    <td>Input</td>
    <td>Quantization group size in three directions for Matmul computation.</td>
    <td>The group size input consists of three values: groupSizeM, groupSizeN, and groupSizeK in three directions. Each value occupies 16 bits, and the three values together occupy the lower 48 bits of the int64_t group size (the upper 16 bits of the group size are invalid). The calculation formula is as follows: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.<br>In the mx quantization scenario, this parameter is set to 4295032864, which corresponds to [1, 1, 32] and is calculated using the preceding formula. In other quantization scenarios, the default value is 0, and the value does not take effect.</td>
    <td>INT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX1</td>
    <td>Input</td>
    <td>Whether the left matrix has been transposed.</td>
    <td>Currently, this parameter cannot be set to True.</td>
    <td>bool</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX2</td>
    <td>Input</td>
    <td>Whether the right matrix has been transposed.</td>
    <td>If this parameter is set to True, the shape of the right matrix is (N, rankSize*H). In mx mode, this parameter must be set to True.</td>
    <td>bool</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>output</td>
    <td>Input</td>
    <td>Final computation result.</td>
    <td>The constraints on the data type vary depending on the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>2D, with the shape of (BS/rankSize, N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllOutOptional</td>
    <td>Optional output</td>
    <td>The data type is the same as that of the input x1 after AlltoAll and Permute.</td>
    <td>If nullptr is passed, no communication output is generated.</td>
    <td>FLOAT16, BFLOAT16, INT4</td>
    <td>ND</td>
    <td>2D, with the shape of (BS/rankSize, rankSize*H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace to be allocated on the device.</td>
    <td></td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Operator executor, containing the operator computation process.</td>
    <td></td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

  The mapping between the enumerated values of x1QuantMode, x2QuantMode, and commQuantMode and the [quantization modes](../../../docs/en/context/quant_mode_introduction.md) is as follows:
  * 0: no quantization
  * 1: pertensor
  * 2: perchannel
  * 3: pertoken
  * 4: pergroup
  * 5: perblock
  * 6: mxQuant
  * 7: per-token dynamic quantization

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:650px">
    </colgroup>
    <thead>
     <tr>
       <th>Return</th>
       <th>Error Code</th>
       <th>Description</th>
     </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Mandatory input and output tensors are null pointers.</td>
    </tr>
    <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>The input and output data types are not supported.</td>
    </tr>
    <tr>
        <td>The input Tensor is empty.</td>
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
        <td>The input and output Tensor dimensions are invalid.</td>
    </tr>
    <tr>
        <td>The input and output formats are private formats.</td>
    </tr>
      </tbody>
  </table>

## aclnnAlltoAllQuantMatmul

* **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
        <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnAlltoAllMatmulGetWorkspaceSize API.</td>
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

* **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

* Deterministic computing is supported by default.
* The number of NPUs (rankSize) varies depending on the device model:
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: 2, 4, or 8 NPUs are supported.
  - Ascend 950PR/Ascend 950DT: 2, 4, 8, or 16 NPUs are supported.
* The variable BS used in the shape in the parameter description must be exactly divided by rankSize.
* The values of BS and N cannot exceed 2147483647 (INT32_MAX). The value of BS cannot be less than 2, and the value of N cannot be less than 1.
* Empty tensors are not supported.
* The support for non-contiguous tensors varies depending on the device model:
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: Non-contiguous tensors are not supported.
  - Ascend 950PR/Ascend 950DT: Only x2 can be a non-contiguous tensor. Other non-contiguous tensors are not supported.
* The input x1, x2, x2Scale, and output are not null pointers, and
  - Ascend 950PR/Ascend 950DT: In the case of x1QuantMode being pertoken dynamic quantization, x1ScaleOptional cannot be passed.
* The data types, dimensions, and quantization modes of the operator's inputs and outputs vary depending on the device model:
  - <term>Atlas A2 training products/Atlas A2 inference products</term>:
    * Quantization mode:
      * Currently, the left matrix supports perToken quantization and perToken dynamic quantization (x1QuantMode = 3 or 7), and the right matrix supports perChannel quantization (x2QuantMode = 2).
    * Type constraints:
      * The data types of x1 and alltoAllOutOptional must be the same.
      * If the int32 type is used for x1, x2, and alltoallout, the data is considered as eight packed int4s and will be reinterpreted as int4.
      * For A16W8 and A16W4, in the smoothQuant scenario, the data type of x1ScaleOptional must be the same as that of x1.
      * For A16W8, the supported data type combinations of x1, x2, biasOptional, and output are as follows:

        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | FLOAT16 | INT8 | FLOAT16 | FLOAT16 |
        | FLOAT16 | INT8 | FLOAT32 | FLOAT16 |
        | BFLOAT16 | INT8 | BFLOAT16 | BFLOAT16 |
        | BFLOAT16 | INT8 | FLOAT32 | BFLOAT16 |

      For * A16W4, the supported data type combinations of x1, x2, biasOptional, and output are as follows:

        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | FLOAT16 | INT4 | FLOAT16 | FLOAT16 |
        | FLOAT16 | INT4 | FLOAT32 | FLOAT16 |
        | BFLOAT16 | INT4 | BFLOAT16 | BFLOAT16 |
        | BFLOAT16 | INT4 | FLOAT32 | BFLOAT16 |

      For * A4W4, x1ScaleOptional supports only FLOAT32. The supported data type combinations of x1, x2, biasOptional, and output are as follows:

        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | INT4 | INT4 | FLOAT16 | FLOAT16 |
        | INT4 | INT4 | FLOAT32 | FLOAT16 |
        | INT4 | INT4 | BFLOAT16 | BFLOAT16 |
        | INT4 | INT4 | FLOAT32 | BFLOAT16 |

    * Dimension constraints:
      * For A16W8, rankSize x H must be exactly divided by 16. The value range of rankSize x H is [1, 35000].
      * For A16W4, rankSize x H must be exactly divided by 16. N must be an even number. The value range of rankSize x H is [1, 35000].
      * For A4W4, both H and N must be even numbers. The value range of rankSize x H is [1, 35000].
  - Ascend 950PR/Ascend 950DT:
    * Quantization mode:
      * Currently, the following modes are supported: K-C dynamic quantization, left matrix perToken dynamic quantization (x1QuantMode=7), right matrix perChannel quantization (x2QuantMode=2), and mx quantization. In mx quantization, the left matrix is quantized (x1QuantMode=6) and the right matrix is quantized (x2QuantMode=6).
    * Type constraints:
      * The data types of x1 and alltoAllOutOptional must be the same.
      * x1QuantDtype takes effect in the K-C dynamic quantization scenario. The value 35 (aclDataType.ACL_FLOAT8_E5M2) or 36 (aclDataType.ACL_FLOAT8_E4M3FN) can be configured. In other quantization scenarios, the configuration does not take effect.
      * biasOptional can be empty.
      * The supported data type combinations of the input and output are as follows:
        * K-C dynamic quantization:

          | x1 | x2 | biasOptional | output | x1QuantDtype | x2QuantDtype | x1ScaleOptional | x2Scale |
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

        * mxQuantize:

          | x1 | x2 | biasOptional | output | x1QuantDtype | x2QuantDtype | x1ScaleOptional | x2Scale |
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

    * Dimension constraints:
      * The value range of rankSize * H is [1, 65535].
      * In the mx quantization scenario, H must be exactly divided by 64.
      * In the mx quantization scenario, x2 must be transposed, the shape is (H*rankSize, N), and transposeX2 is True.
* MC2 operators cannot be called concurrently, nor can different MC2 operators.
* Inter-super node communication is not supported. Only intra-super node communication is supported.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

Note: This sample code calls some HCCL collective communication library APIs, including HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/hcclapiref/hcclcpp_07_0001.html>.

- <term>Atlas A2 training products/Atlas A2 inference products</term>:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_allto_all_quant_matmul.h"
    
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
        std::vector<int64_t> x2Shape = {64 * ndev, 128}; // ndev = 2, x2Shape: The shape remains unchanged before and after transposition.
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

        // Create a tensor.
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
        ret = aclnnAlltoAllQuantMatmulGetWorkspaceSize(x1, x2, bias, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
                                                hcom_name, alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
                                                groupSize, false, true,
                                                out, allToAllOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllQuantMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAlltoAllQuantMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmul failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnMatmulAlltoAll execute success \n", args.rankId);
        // Release device resources. Modify the code based on the API definition.
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
        // This example is implemented based on Atlas A2 and can only run on Atlas A2.
        int ret = aclInit(nullptr);
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // Initialize the collective communication domain.
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
        // Enable multi-threading.
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

- Ascend 950PR/Ascend 950DT:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnnop/aclnn_allto_all_quant_matmul.h"
  
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
        std::vector<int16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int16_t> x2HostData(x2ShapeSize, 1);
        std::vector<int16_t> biasHostData(biasShapeSize, 1);
        std::vector<int16_t> x2ScaleHostData(x2ScaleShapeSize, 1);
        std::vector<int16_t> outHostData(outShapeSize, 0);
        std::vector<int16_t> allToAllOutHostData(allToAllOutShapeSize, 0);
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
        ret = aclnnAlltoAllQuantMatmulGetWorkspaceSize(x1, x2, bias, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
                                                hcom_name, alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
                                                groupSize, false, false,
                                                out, allToAllOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllQuantMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAlltoAllQuantMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmul failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllQuantMatmul execute success \n", args.rankId);
        // Release device resources. Modify the code based on the API definition.
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
    //This sample is implemented based on Ascend 950PR/Ascend 950DT and must run on Ascend 950PR/Ascend 950DT.
    int ret = aclInit(nullptr);
    int32_t devices[ndev];
    for (int i = 0; i < ndev; i++) {
    devices[i] = i;
    }
    HcclComm comms[128];
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    // Initialize the collective communication domain.
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
    // Enable multi-threading.
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
