# aclnnGroupedMatmulFinalizeRoutingWeightNzV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:31:00.986Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

A fused operator of GroupedMatmul and MoeFinalizeRouting, where the output after GroupedMatmul computation is combined according to indices. Supports w in the AI processor-friendly data layout format (NZ).

Compared with aclnnGroupedMatmulFinalizeRoutingWeightNz, this API adds the following:

- Adds new input parameters offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, and tuningConfigOptional. For the INT4 weight use case on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, offsetOptional can be used to pass in the asymmetric quantization offset; antiquantScaleOptional and antiquantOffsetOptional are reserved parameters and do not take effect currently. Pass in a null pointer.

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Added support for INT4 weight matrices and the tuningConfigOptional tuning parameter. The first value in the array indicates the expected number of tokens processed by each expert. During operator tiling, the tiling split is performed reasonably based on this expected value, achieving better performance. Select the appropriate API based on the actual situation.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: Adds the Pertoken-perchannel, static pertensor-perchannel, and MxA8W4 quantization use cases. For details, see [Quantization Introduction](../../../docs/en/context/quant_mode_introduction.md).

<!-- end id8 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnGroupedMatmulFinalizeRoutingWeightNzV2" to perform the computation.

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
    const aclTensor   *x1,
    const aclTensor   *x2,
    const aclTensor   *scale,
    const aclTensor   *bias,
    const aclTensor   *offsetOptional,
    const aclTensor   *antiquantScaleOptional,
    const aclTensor   *antiquantOffsetOptional,
    const aclTensor   *pertokenScaleOptional,
    const aclTensor   *groupList,
    const aclTensor   *sharedInput,
    const aclTensor   *logit,
    const aclTensor   *rowIndex,
    int64_t            dtype,
    float              sharedInputWeight,
    int64_t            sharedInputOffset,
    bool               transposeX1,
    bool               transposeX2,
    int64_t            groupListType,
    const aclIntArray *tuningConfigOptional,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 400px">
  <col style="width: 230px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 190px">
  <col style="width: 145px">
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
      <td>Input x (left matrix).</td>
      <td>-</td>
      <td>INT8, FLOAT8_E4M3FN<sup>1</sup>, HIFLOAT8<sup>1</sup></td>
      <td>ND</td>
      <td>(m, k)</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input weight (right matrix).</td>
      <td>Supports the Ascend-affine data layout format (NZ). For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
      <td>INT8, INT4<sup>2</sup>, INT32<sup>2</sup>, FLOAT8_E4M3FN<sup>1</sup>, HIFLOAT8<sup>1</sup>, FLOAT4_E2M1<sup>1</sup></td>
      <td>NZ</td>
      <td>Support 3D</td>
      <td>Non-contiguous tensors are supported only in the transpose use case</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scale factor in quantization parameters, that is, the per-channel or Mx quantization parameter.</td>
      <td>-</td>
      <td>FLOAT, BF16, INT64<sup>2</sup>, FLOAT8_E8M0<sup>1</sup></td>
      <td>ND</td>
      <td>Shape supports 3-4 dimensions.</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Matrix bias.</td>
      <td>-</td>
      <td>BF16, FLOAT</td>
      <td>ND</td>
      <td>Shape supports 2D, dimension is (e, n), and e and n are consistent with e and n of x2.</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>Input</td>
      <td>Asymmetric quantization offset.</td>
      <td>For Atlas A2 products and Atlas A3 products, a value can be passed in the INT4 weight asymmetric quantization use case; pass nullptr for the symmetric quantization use case. Ascend 950PR&950DT products do not support this parameter yet; nullptr must be passed.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>Atlas A2 products and Atlas A3 products support 3D; the dimension is (e, 1, n)</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>Input</td>
      <td>Scale factor for pseudo-quantization.</td>
      <td>Currently not enabled.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>Input</td>
      <td>Offset for pseudo-quantization.</td>
      <td>Not currently enabled.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>pertokenScaleOptional</td>
      <td>Input</td>
      <td>Dequantization parameter for matrix computation.</td>
      <td>-</td>
      <td>FLOAT, FLOAT8_E8M0<sup>1</sup></td>
      <td>ND</td>
      <td>Supports 1D and 3D shapes. Dimension is (m), and m is consistent with m of x1; 3D (m, ceil(k/64), 2) in the MxA8W4 use case.</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>Input</td>
      <td>Distribution of matmul sizes along the group axis for input and output.</td>
      <td>Input data in different formats according to groupListType. See <a href="#Constraints">Constraints</a> for comprehensive constraints.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>Supports 1D shape. Dimension is (e), and e is consistent with e of x2.</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>sharedInput</td>
      <td>Input</td>
      <td>Output of the shared expert in MoE computation, which needs to be combined with the output of MoE experts.</td>
      <td>-</td>
      <td>BF16</td>
      <td>ND</td>
      <td>Support 2D, dimension is (bsdp, n). bsdp must be less than or equal to batchSize/e, and n must be consistent with the n of x2.</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>logit</td>
      <td>Input</td>
      <td>Logit size of moe experts for each token.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>Supports 1D shape, dimension is (m), m is consistent with the m of x1</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>rowIndex</td>
      <td>Input</td>
      <td>The moe expert outputs are combined according to this rowIndex, whose values serve as the indices for scatter add in combine.</td>
      <td>-</td>
      <td>INT64, INT32</td>
      <td>ND</td>
      <td>Supports 1D shape, dimension is (m), and m is consistent with the m of x1.</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Input</td>
      <td>Output data type of the computation: 0: FLOAT; 1: FLOAT16; 2: BFLOAT16. Currently only 0 is supported.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>Input</td>
      <td>Coefficient for the combine operation between the shared expert and MoE experts. sharedInput is first multiplied by this parameter and then accumulated with the MoE expert result.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputOffset</td>
      <td>Input</td>
      <td>Offset of the shared expert output in the total output.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>Input</td>
      <td>Whether the left matrix is transposed. Only false is supported.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Whether the right matrix is transposed. For comprehensive constraints, see also <a href="#constraints">Constraints</a>.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>Input</td>
      <td>Grouping mode: 0 indicates the cumsum mode (prefix sum); 1 indicates the count mode.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tuningConfigOptional</td>
      <td>Input</td>
      <td>The first element in the array indicates the expected number of tokens processed by each expert. During operator tiling, tiling is split appropriately based on the first element of the array, achieving better performance. If the second element in the array is set to 1, the operator attempts to use a more suitable algorithm based on the actual input during tiling, which may deliver better performance when k<=2048. Elements from the third onward are reserved and need not be filled in by users. Extensions will be supported in the future. For backward compatibility, if you do not use this parameter, simply do not pass it in (that is, set it to nullptr).</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch, n)</td>
      <td>✗</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Return the workspace size that needs to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Return the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The superscript "1" in the data type column in the table above indicates data types not supported by the series.
    - x1 only supports INT8. The value range of dimension m is [1,16\*1024\*8], and k supports 2048.
    - x2 supports INT8, INT4, and INT32. When the input is INT32, the dimension is (e, k, n / 8); when converted to INT4, the dimension is (e, k, n). The value range of e is [1,256], k supports 2048, and n supports 7168. Only the transpose attribute false is supported.
    - offsetOptional supports a three-dimensional shape with dimensions (e, 1, n), where e and n are consistent with those of weight.
    - scale supports INT64, FLOAT, and BF16.
    - rowIndex supports INT64 and INT32.
    - x1, x2, scale, groupList, logit, and rowIndex are required parameters, and bias and sharedInput are optional parameters. When x2 is INT4 or INT32, pertokenScaleOptional is a required parameter; when x2 is INT8, pertokenScaleOptional is an optional parameter.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - The superscript "2" in the data type column of the preceding table indicates the data types not supported by this series.
    - When the data types of x1 and x2 are INT8, rowIndex supports the INT64 and INT32 data types; when the data types of x1 and x2 are FLOAT8_E4M3FN, HIFLOAT8, or the MxA8W4 quantization mode, rowIndex supports only the INT64 data type.
    - x1, x2, scale, groupList, logit, and rowIndex are required parameters; pertokenScaleOptional, sharedInput, and bias are optional parameters. In the MxA8W4 use case, pertokenScaleOptional is a required parameter. Currently, the offsetOptional parameter is not supported and must be nullptr.
    - When groupListType is 0, groupList must be a non-negative monotonically non-decreasing sequence (cumulative sum), and the last value must not be greater than the first dimension of the tensor in x1; when groupListType is 1, groupList must be a non-negative array (size of each group), and the sum of the values must not be greater than the first dimension of the tensor in x1.
    - The first dimension batch of out and sharedInputOffset must be greater than or equal to 0 and less than or equal to m.
    - When sharedInput is not empty, its first dimension is sharedInputLen. sharedInputOffset, sharedInputLen, and their sum must be less than or equal to the first dimension batch of out.
    - x1 supports M = 0 and x2 supports N = 0; other empty tensor use cases are not supported.

  <!-- end id10 -->

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 700px">
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
      <td>The input parameter is a required input, output, or required attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or data format of x1, x2, scale, bias, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupList, sharedInput, logit, rowIndex, sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, or out is not in the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x1, x2, scale, bias, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupList, sharedInput, logit, rowIndex, or out does not meet the validation conditions.</td>
    </tr>
    <tr>
      <td>Except when M of x1 is 0 or N of x2 is 0, other empty tensors are passed in.</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmulFinalizeRoutingWeightNzV2

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
    <col style="width: 173px">
    <col style="width: 112px">
    <col style="width: 668px">
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
        <td>Workspace memory address applied on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size applied on the Device side, obtained by calling the first-phase API aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor. Contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
      </tr>
    </tbody>
    </table>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: aclnnGroupedMatmulFinalizeRoutingWeightNzV2 defaults to a non-deterministic implementation and supports enabling deterministic computation through aclrtSetSysParamOpt.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnGroupedMatmulFinalizeRoutingWeightNzV2 defaults to a non-deterministic implementation. Enabling deterministic computation through aclrtSetSysParamOpt is supported only when both inputs x1 and x2 are of the int8 data type.

  <!-- end id12 -->

<!-- npu="A3,910b" id13 -->
<details>
<summary><term>Atlas A2 products</term>, <term>Atlas A3 products</term></summary>

  - General Constraints
    - groupList: When groupListType is 0, groupList must be a non-negative monotonically non-decreasing sequence; when groupListType is 1, groupList must be a non-negative sequence.
    - The value range of dimension m of x1 is $[1, 16*1024*8]$, and k supports 2048.
    - The value range of e of x2 is $[1, 256]$, k supports 2048, and n supports 7168. Only the transpose attribute false is supported.
    - The shape of offsetOptional supports 3D, and the dimension is (e, 1, n), where e and n are consistent with e and n of x2.

  - The input and output support the following data type combinations:

    | x1    | x2    | scale   | bias    | offsetOptional  | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupList | sharedInput | logit   |   rowIndex | out   | tuningConfigOptional |
    |------|------|---------|---------|---------|----------------|-----------------|---------------|-----------|-------------|---------|----------|-------| ----------------------|
    | INT8 | INT8 | FLOAT | BFLOAT16 / null | null | null | null | FLOAT / null | INT64 | BFLOAT16 / null | FLOAT | INT64 / INT32 | FLOAT | IntArray |
    | INT8 | INT4 | INT64 | FLOAT / null | FLOAT | null | null | FLOAT | INT64 | BFLOAT16 / null | FLOAT | INT64 | FLOAT | IntArray |
    | INT8 | INT4 | INT64 | FLOAT / null | null | null | null | FLOAT | INT64 | BFLOAT16 / null | FLOAT | INT64 | FLOAT | IntArray |

</details>
<!-- end id13 -->

<!-- npu="950" id14 -->
<details>
<summary><term>Ascend 950PR&950DT products</term></summary>

  - General Constraints
    - groupList: When groupListType is 0, groupList must be a non-negative monotonically non-decreasing sequence; when groupListType is 1, groupList must be a non-negative sequence.
    - The value range of e of x2 is [1, 1024].
    - The n of sharedInput is consistent with the n of x2.
    - x1 supports M being 0, and x2 supports N being 0; other empty tensor use cases are not supported.

  - The supported data types in the full quantification use case are:

    | x1    | x2    | scale   | bias    | offsetOptional | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupList | sharedInput | logit | rowIndex | out   | tuningConfigOptional |
    |------|------|---------|---------|---------|----------|-------|------|------|------|-------|--------|-------|------|
    | INT8 | INT8 | FLOAT/BFLOAT16 | BFLOAT16/null | null | null | null | FLOAT/null | INT64 | BFLOAT16 | FLOAT | INT64/INT32 | FLOAT | null |
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT/BFLOAT16 | BFLOAT16/null | null | null | null | FLOAT/null | INT64 | BFLOAT16 | FLOAT | INT64 | FLOAT | null |
    | HIFLOAT8 | HIFLOAT8 | FLOAT/BFLOAT16 | BFLOAT16/null | null | null | null | FLOAT/null | INT64 | BFLOAT16 | FLOAT | INT64 | FLOAT | null |

    - x2 supports a transpose attribute of true or false. The dimension is (e, k, n) for non-transpose and (e, n, k) for transpose.
    - The shape of scale is (e, 1, n). e and n are consistent with e and n of x2.
    - The shape of pertokenScaleOptional is (m). m is consistent with m of x1.

  - The data types supported in the pseudo-quantized scenario are:

    | x1    | x2    | scale   | bias    | offsetOptional | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupList | sharedInput | logit | rowIndex | out   | tuningConfigOptional |
    |------|------|---------|---------|---------|----------|-------|------|------|------|-------|--------|-------|------|
    | FLOAT8_E4M3FN | FLOAT4_E2M1 | FLOAT8_E8M0 | BFLOAT16/null | null | null | null | FLOAT8_E8M0 | INT64 | BFLOAT16 | FLOAT | INT64 | FLOAT | null |

    - The format of x2 is NZ_C0_32, and the shape is (e, n, k). The shape corresponding to the NZ format is (e, ceil(k/32), ceil(n/16), 16, 32).
    - transposeX2 is fixed to true.
    - pertokenScaleOptional is a required parameter, and the shape is (m, ceil(k/64), 2).
    - The shape of scale is (e, n, ceil(k/64), 2).
    - k must satisfy k % 32 == 0.
    - n must satisfy n % 32 == 0.

</details>
<!-- end id14 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="A3,910b" id15 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    ```Cpp
    #include <iostream>
    #include <memory>
    #include <vector>

    #include "acl/acl.h"
    #include "aclnnop/aclnn_permute.h"
    #include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
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

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shapeSize = 1;
        for (auto i : shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream *stream)
    {
        // Boilerplate code for resource initialization
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
        // Call aclrtMalloc to allocate memory on the device.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy data from the host to device memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Calculate the strides of the contiguous tensor.
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
    int CreateAclTensorWeight(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                          aclDataType dataType, aclTensor **tensor)
    {
        auto size = static_cast<uint64_t>(GetShapeSize(shape));

        const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
        auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                  return ret);
        size *= sizeof(T);

        // Call aclrtMalloc to allocate memory on the device.
        ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy host-side data to device-side memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Calculate the strides of the contiguous tensor.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        std::vector<int64_t> storageShape;
        storageShape.push_back(GetShapeSize(shape));

        // Call the aclCreateTensor API to create an aclTensor.
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  storageShape.data(), storageShape.size(), *deviceAddr);
        return 0;
    }

      int main() {
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = Init(deviceId, &stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

        // 2. Construct inputs and outputs, which need to be custom built based on the API interface.
        int64_t m = 192;
        int64_t k = 2048;
        int64_t n = 7168;
        int64_t e = 4;
        int64_t batch = 24;
        int64_t bsdp = 8;
        int64_t dtype = 0;
        float sharedInputWeight = 1.0;
        int64_t sharedInputOffset = 0;
        bool transposeX = false;
        bool transposeW = false;
        int64_t groupListType = 1;

        std::vector<int64_t> xShape = {m, k};
        std::vector<int64_t> wShape = {e, k, n};
        std::vector<int64_t> scaleShape = {e, n};
        std::vector<int64_t> pertokenScaleShape = {m};
        std::vector<int64_t> groupListShape = {e};
        std::vector<int64_t> sharedInputShape = {bsdp, n};
        std::vector<int64_t> logitShape = {m};
        std::vector<int64_t> rowIndexShape = {m};
        std::vector<int64_t> outShape = {batch, n};
        std::vector<int64_t> tuningConfigVal = {1};

        void *xDeviceAddr = nullptr;
        void *wDeviceAddr = nullptr;
        void *scaleDeviceAddr = nullptr;
        void *pertokenScaleDeviceAddr = nullptr;
        void *groupListDeviceAddr = nullptr;
        void *sharedInputDeviceAddr = nullptr;
        void *logitDeviceAddr = nullptr;
        void *rowIndexDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;

        aclTensor* x = nullptr;
        aclTensor* w = nullptr;
        aclTensor* bias = nullptr;
        aclTensor* groupList = nullptr;
        aclTensor* scale = nullptr;
        aclTensor* pertokenScale = nullptr;
        aclTensor* sharedInput = nullptr;
        aclTensor* logit = nullptr;
        aclTensor* rowIndex = nullptr;
        aclTensor* out = nullptr;

        std::vector<int8_t> xHostData(GetShapeSize(xShape));
        std::vector<int8_t> wHostData(GetShapeSize(wShape));
        std::vector<float> scaleHostData(GetShapeSize(scaleShape));
        std::vector<float> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
        std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
        groupListHostData[0] = 7;
        groupListHostData[1] = 32;
        groupListHostData[2] = 40;
        groupListHostData[3] = 64;

        std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
        std::vector<float> logitHostData(GetShapeSize(logitShape));
        std::vector<int64_t> rowIndexHostData(GetShapeSize(rowIndexShape));
        std::vector<float> outHostData(GetShapeSize(outShape));

        // Create the x aclTensor.
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the w aclTensor in the AI processor-affine data layout format.
        ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_INT8, &w);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the scale aclTensor.
        ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the pertokenScale aclTensor.
        ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &pertokenScale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the groupList aclTensor.
        ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the sharedInput aclTensor.
        ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16, &sharedInput);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the logit aclTensor.
        ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the rowIndex aclTensor.
        ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the out aclTensor.
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the tuningConfig aclIntArray.
        aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
        std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> tuningConfigIntArrayPtr(tuningConfig, aclDestroyIntArray);
        CHECK_RET(tuningConfig != nullptr, return -1);
        // 3. Call the CANN operator library API. Change it to the specific API name.
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        // x2 must be converted from the ND format to the NZ format required by the operator. Therefore, call aclnnTransMatmulWeight first to complete the weight format conversion.
        // Call the first-phase API of aclnnTransMatmulWeight.
        ret = aclnnTransMatmulWeightGetWorkspaceSize(w, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API of aclnnTransMatmulWeight.
        ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

        // Call the first-phase API of aclnnGroupedMatmulFinalizeRoutingWeightNzV2.
        workspaceSize = 0;
        ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, sharedInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, tuningConfig, out, &workspaceSize, &executor);

        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);
        // Release the workspace allocated in the weight format conversion phase to prevent memory leaks caused by the second allocation overwriting the pointer.
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API of aclnnGroupedMatmulFinalizeRoutingWeightNzV2.
        ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret); return ret);

        // 4. (Boilerplate) Synchronously wait for task execution to complete.
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. Obtain the output value, copy the result on the device side to the host side, and modify the code as required by the specific API definition.
        auto size = GetShapeSize(outShape);
        std::vector<float> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("result[%lld] is: %f\n", i, resultData[i]);
        }

        // 6. Release the aclTensor resources. Modify the code as required by the specific API definition.
        aclDestroyTensor(x);
        aclDestroyTensor(w);
        aclDestroyTensor(scale);
        aclDestroyTensor(pertokenScale);
        aclDestroyTensor(groupList);
        aclDestroyTensor(sharedInput);
        aclDestroyTensor(logit);
        aclDestroyTensor(rowIndex);
        aclDestroyTensor(out);

        // 7. Release the device resources. Modify the code as required by the specific API definition.
        aclrtFree(xDeviceAddr);
        aclrtFree(wDeviceAddr);
        aclrtFree(scaleDeviceAddr);
        aclrtFree(pertokenScaleDeviceAddr);
        aclrtFree(groupListDeviceAddr);
        aclrtFree(sharedInputDeviceAddr);
        aclrtFree(logitDeviceAddr);
        aclrtFree(rowIndexDeviceAddr);
        aclrtFree(outDeviceAddr);
        aclDestroyIntArray(tuningConfig);

        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }
    ```

<!-- end id15 -->
<!-- npu="950" id16 -->
- <term>Ascend 950PR&950DT products</term>:

  - Pertoken quantization data flow example:

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      // Boilerplate, resource initialization
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
      // Call aclrtMalloc to allocate device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to the device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
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
  int CreateAclTensorWeightNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                              aclDataType dataType, aclTensor **tensor)
  {
      // Calculate the storageShape in NZ format.
      int64_t e = shape[0];
      int64_t k = shape[1];
      int64_t n = shape[2];
      std::vector<int64_t> shapeNz = {e, n / 64, k / 16, 16L, 64L};

      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to the device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor in NZ format.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, shapeNz.data(), shapeNz.size(), *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize the device/stream. For details, see the AscendCL external API list.
      // Fill in deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs. Custom construction is required based on the API.
      int64_t m = 192;
      int64_t k = 2048;
      int64_t n = 7168;
      int64_t e = 4;
      int64_t batch = 24;
      int64_t bsdp = 8;
      int64_t dtype = 0;
      float sharedInputWeight = 1.0;
      int64_t sharedInputOffset = 0;
      bool transposeX = false;
      bool transposeW = false;
      int64_t groupListType = 1;

      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> wShape = {e, k, n};
      std::vector<int64_t> scaleShape = {e, 1, n};
      std::vector<int64_t> pertokenScaleShape = {m};
      std::vector<int64_t> groupListShape = {e};
      std::vector<int64_t> sharedInputShape = {bsdp, n};
      std::vector<int64_t> logitShape = {m};
      std::vector<int64_t> rowIndexShape = {m};
      std::vector<int64_t> outShape = {batch, n};

      void *xDeviceAddr = nullptr;
      void *wDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *groupListDeviceAddr = nullptr;
      void *sharedInputDeviceAddr = nullptr;
      void *logitDeviceAddr = nullptr;
      void *rowIndexDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;

      aclTensor *x = nullptr;
      aclTensor *w = nullptr;
      aclTensor *groupList = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *pertokenScale = nullptr;
      aclTensor *sharedInput = nullptr;
      aclTensor *logit = nullptr;
      aclTensor *rowIndex = nullptr;
      aclTensor *out = nullptr;

      std::vector<int8_t> xHostData(GetShapeSize(xShape));
      std::vector<int8_t> wHostData(GetShapeSize(wShape));
      std::vector<float> scaleHostData(GetShapeSize(scaleShape));
      std::vector<float> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
      std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
      // Assign values to groupList.
      groupListHostData[0] = 64;

      std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
      std::vector<float> logitHostData(GetShapeSize(logitShape));
      std::vector<int64_t> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));

      // Create the x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_HIFLOAT8, &x);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // Create the w aclTensor in the AI processor-affine data layout format (Nz format).
      ret = CreateAclTensorWeightNz(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_HIFLOAT8, &w);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // Create the pertokenScale aclTensor.
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT,
                            &pertokenScale);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // Create the groupList aclTensor.
      ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // Create the sharedInput aclTensor.
      ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16,
                            &sharedInput);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // Create the logit aclTensor.
      ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // Create the rowIndex aclTensor.
      ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Change it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // Call the first-phase API of aclnnGroupedMatmulFinalizeRoutingWeightNzV2.
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
          x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype,
          sharedInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, nullptr, out, &workspaceSize,
          &executor);
      CHECK_FREE_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                    return ret);

      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API of aclnnGroupedMatmulFinalizeRoutingWeightNzV2.
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, and copy the result from the device to the host. Modify it according to the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                    return ret);
      for (int64_t i = 0; i < 10 && i < size; i++) {
          LOG_PRINT("result[%lld] is: %f\n", static_cast<long long>(i), resultData[i]);
      }

      // 6. Release the aclTensor resources. Modify it according to the specific API definition.
      aclDestroyTensor(x);
      aclDestroyTensor(w);
      aclDestroyTensor(scale);
      aclDestroyTensor(pertokenScale);
      aclDestroyTensor(groupList);
      aclDestroyTensor(sharedInput);
      aclDestroyTensor(logit);
      aclDestroyTensor(rowIndex);
      aclDestroyTensor(out);

      // 7. Release the device resources. Modify it according to the specific API definition.
      aclrtFree(xDeviceAddr);
      aclrtFree(wDeviceAddr);
      aclrtFree(scaleDeviceAddr);
      aclrtFree(pertokenScaleDeviceAddr);
      aclrtFree(groupListDeviceAddr);
      aclrtFree(sharedInputDeviceAddr);
      aclrtFree(logitDeviceAddr);
      aclrtFree(rowIndexDeviceAddr);
      aclrtFree(outDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```

  - MxA8W4 data flow example:

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include <cmath>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int64_t CeilDiv(int64_t a, int64_t b)
  {
      if (b == 0) return 0;
      if (a <= 0) return 0;
      return (a - 1) / b + 1;
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
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1L);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T> &hostData, const std::vector<int64_t> &shape, int64_t *storageShape,
                                uint64_t storageShapeSize, void **deviceAddr, aclDataType dataType, aclTensor **tensor,
                                aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, storageShape,
                                storageShapeSize, *deviceAddr);
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
      // Private format code of aclnnNpuFormatCast: 29 indicates FRACTAL_NZ.
      constexpr int64_t kNpuFormatFractalNz = 29;
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, kNpuFormatFractalNz, aclFormat::ACL_FORMAT_ND,
                                                     &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
                return ret);

      aclTensor *dstTensor = nullptr;
      void *dstDeviceAddr = nullptr;

      uint64_t tensorSize = 1;
      for (uint64_t i = 0; i < dstShapeSize; i++) {
          tensorSize *= dstShape[i];
      }
      std::vector<T> dstTensorHostData(tensorSize, 0);

      ret = CreateAclTensorWithFormat(dstTensorHostData, shape, dstShape, dstShapeSize, &dstDeviceAddr, dataType,
                                      &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      *tensor = dstTensor;
      *deviceAddr = dstDeviceAddr;

      aclDestroyTensor(srcTensor);
      aclrtFree(srcDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }

      return ACL_SUCCESS;
  }

  int RunExample(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      int64_t m = 192;
      int64_t k = 2048;
      int64_t n = 7168;
      int64_t e = 4;
      int64_t batch = 24;
      int64_t bsdp = 8;
      int64_t dtype = 0;
      float sharedInputWeight = 1.0;
      int64_t sharedInputOffset = 0;
      bool transposeX = false;
      bool transposeW = true;
      int64_t groupListType = 1;
      int64_t ceil_k_64 = CeilDiv(k, 64);

      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> wShape = {e, n, k};
      std::vector<int64_t> scaleShape = {e, n, ceil_k_64, 2};
      std::vector<int64_t> pertokenScaleShape = {m, ceil_k_64, 2};
      std::vector<int64_t> groupListShape = {e};
      std::vector<int64_t> sharedInputShape = {bsdp, n};
      std::vector<int64_t> logitShape = {m};
      std::vector<int64_t> rowIndexShape = {m};
      std::vector<int64_t> outShape = {batch, n};

      void *xDeviceAddr = nullptr;
      void *wDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *groupListDeviceAddr = nullptr;
      void *sharedInputDeviceAddr = nullptr;
      void *logitDeviceAddr = nullptr;
      void *rowIndexDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;

      aclTensor *x = nullptr;
      aclTensor *w = nullptr;
      aclTensor *groupList = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *pertokenScale = nullptr;
      aclTensor *sharedInput = nullptr;
      aclTensor *logit = nullptr;
      aclTensor *rowIndex = nullptr;
      aclTensor *out = nullptr;

      std::vector<uint8_t> xHostData(GetShapeSize(xShape));
      std::vector<uint8_t> wHostData(GetShapeSize(wShape));
      std::vector<uint8_t> scaleHostData(GetShapeSize(scaleShape));
      std::vector<uint8_t> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
      std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape), m/e);

      std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
      std::vector<float> logitHostData(GetShapeSize(logitShape));
      std::vector<int64_t> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));

      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);

      ret = CreateAclTensorNz(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &w, stream);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);

      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);

      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                            aclDataType::ACL_FLOAT8_E8M0, &pertokenScale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale,
                                                                                            aclDestroyTensor);

      ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);

      ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16,
                            &sharedInput);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);

      ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);

      ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);

      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);

      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);

      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
          x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype,
          sharedInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, nullptr, out, &workspaceSize,
          &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }

      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret);
                return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < std::min((int64_t)10, size); i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }

      return ACL_SUCCESS;
  }

  int main()
  {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = RunExample(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunExample failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id16 -->
