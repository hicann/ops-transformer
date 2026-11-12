# aclnnGroupedMatmulFinalizeRoutingV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:24:20.964Z -->

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

- Description:
  A fused operator of GroupedMatmul and MoeFinalizeRouting, where the output after GroupedMatmul computation is combined according to indices.

  Compared with aclnnGroupedMatmulFinalizeRoutingV2, **this API adds the following:**

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The input parameter tuningConfigOptional is added as a tuning parameter. The first value in the array indicates the expected number of tokens processed by each expert. The operator tiling is performed based on this expected value for better performance.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: Adds the MX quantization scenario. For details, see [Quant Mode Introduction](../../../docs/en/context/quant_mode_introduction.md).
    <!-- end id8 -->

- Formula:

  - Grouped matrix multiplication (GMM):

    $$
    y_i=(x_i\times weight_i) * scale_i * perTokenScale_i
    $$

  - Routing expert and expert output assignment:

    For each token j, perform routing and output expert assignment:

    $$
    y[ rowIndex[i] , : ] = y[rowIndex[i], :] +
    y_{i(j)}[ j - start_{i(j)}]
    $$

    Here, $i(j)$ is the index of the expert to which token j is assigned. $y_{i(j)}[ j - start_{i(j)}]$ is the computation result of the token under the corresponding expert.

  - Shared expert output fusion:

    $$
    y [rowIndex[i],:] = y[rowIndex[i],:] + sharedInputWeight \times sharedInput[j, :]
    $$

  - Shared expert output fusion: The final output is the result of merging all expert outputs with the shared expert output according to rowIndex. The computation process is as follows:

    $$
    y[rowIndex[i],:] = \sum_{i \in \mathcal{E}[j]} y_i [j - start_i] + sharedInputWeight \times sharedInput[j, :]
    $$

    where $\mathcal{E}[j]$ denotes the set of experts assigned to token j.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnGroupedMatmulFinalizeRoutingV3" API to execute the computation.

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(
    const aclTensor   *x1,
    aclTensor         *x2,
    const aclTensor   *scaleOptional,
    const aclTensor   *biasOptional,
    const aclTensor   *offsetOptional,
    const aclTensor   *antiquantScaleOptional,
    const aclTensor   *antiquantOffsetOptional,
    const aclTensor   *pertokenScaleOptional,
    const aclTensor   *groupListOptional,
    const aclTensor   *sharedInputOptional,
    const aclTensor   *logitOptional,
    const aclTensor   *rowIndexOptional,
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
aclnnStatus aclnnGroupedMatmulFinalizeRoutingV3(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize

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
      <td>INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>(m, k)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input weight (right matrix).</td>
      <td>-</td>
      <td>INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>Supports three dimensions</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>Input</td>
      <td>Scale factor in quantization parameters, per-channel quantization parameter.</td>
      <td>-</td>
      <td>INT64, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Bias of the matrix.</td>
      <td>-</td>
      <td>FLOAT32, BF16</td>
      <td>ND</td>
      <td>Supports two dimensions, and the dimension is (e, n)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>Input</td>
      <td>Offset for asymmetric quantization.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>Input</td>
      <td>Scale factor for pseudo quantization.</td>
      <td>Currently not enabled</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>Input</td>
      <td>Offset for pseudo quantization.</td>
      <td>Currently not enabled</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>pertokenScaleOptional</td>
      <td>Input</td>
      <td>Dequantization parameter for matrix computation.</td>
      <td></td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupListOptional</td>
      <td>Input</td>
      <td>Distribution of matmul sizes along the grouping axis direction for input and output.</td>
      <td></td>
      <td>INT64</td>
      <td>ND</td>
      <td>Supports one dimension, where the dimension is (e)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sharedInputOptional</td>
      <td>Input</td>
      <td>Output of the shared expert in moe computation, which needs to be combined with the output of moe experts.</td>
      <td></td>
      <td>BF16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>logitOptional</td>
      <td>Input</td>
      <td>Logit size of each token computed by MoE experts.</td>
      <td></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>shape supports one dimension, the dimension is (m)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rowIndexOptional</td>
      <td>Input</td>
      <td>The MoE expert output is combined according to this rowIndex. The values serve as the indices for scatter add in the combine operation.</td>
      <td></td>
      <td>INT64, INT32</td>
      <td>ND</td>
      <td>Shape supports one dimension, and the dimension is (m)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Input</td>
      <td>Output data type: 0: FLOAT32; 1: FLOAT16; 2: BFLOAT16. Currently, only 0 is supported.</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>Input</td>
      <td>Coefficient for combining the shared expert and MoE experts. sharedInput is multiplied by this parameter first, and then accumulated with the MoE expert result.</td>
      <td></td>
      <td>FLOAT32</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputOffset</td>
      <td>Input</td>
      <td>Offset of the shared expert output in the total output.</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>Input</td>
      <td>Whether the left matrix is transposed. Only false is supported.</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Whether the right matrix is transposed.</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>Input</td>
      <td>Grouping mode: 0: cumsum mode, that is, prefix sum; 1: count mode.</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>tuningConfigOptional</td>
      <td>Input</td>
      <td>The first element in the array indicates the expected number of tokens processed by each expert. During operator tiling, the first element is used to properly split tiling, achieving better performance. Elements starting from the second are reserved and need not be filled in by users. They will be extended in the future. For backward compatibility, if this parameter is not used, simply do not pass it in (that is, nullptr).</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(batch, n)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be side-applied on the Device.</td>
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
  </tbody>
  </table>

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    - x1 only supports INT8. Its dimension is (m, k), where m ranges from [1, 16\*1024\*8] and k supports 2048.
    - x2 only supports INT4. When the input is INT32, the dimension is (e, k, n / 8); when the input is converted to INT4, the dimension is (e, k, n). The value range of e is [1,256], k supports 2048, and n supports 7168. Only non-transposition is supported, that is, transposeX2 must be false.
    - scaleOptional supports INT64. Its shape supports 3D, with a dimension of (e, 1, n). e and n are consistent with those of w.
    - biasOptional supports FLOAT32. e and n are consistent with e and n of w.
    - offsetOptional supports FLOAT32. It supports 3D, and its dimension is (e, 1, n), where e and n are consistent with e and n of w.
    - perTokenScaleOptional supports FLOAT32. It supports 1D, and its dimension is (m), where m is consistent with m of x.
    - groupListOptional supports that e is consistent with e of w.
    - sharedInputOptional supports 2D, and its dimension is (bsdp, n), where bsdp must be less than or equal to batchSize/e, and n is consistent with n of w.
    - logitOptional supports m, which must be consistent with the m of x.
    - rowIndexOptional supports INT64 and INT32, and m must be consistent with the m of x.
    - x1, x2, scaleOptional, biasOptional, pertokenScaleOptional, groupListOptional, logitOptional, and rowIndexOptional are mandatory parameters, and sharedInputOptional is an optional parameter.

  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - x1 does not support INT8.
    - x2 does not support INT4. The dimension is (e, k, n), or (e, n, k) in the transposed case, where e ranges from [1, 1024].
    - scaleOptional supports FLOAT8_E8M0. The shape supports four dimensions. When x2 is not transposed, the dimension is (e,Ceil(k/64),n,2); when x2 is transposed, the dimension is (e,n,Ceil(k/64),2). Only the FLOAT8_E8M0 data type is supported, and the transpose attribute must be consistent with that of x2.
    - biasOptional supports BF16.
    - sharedInputOptional supports 2D with a dimension of (bsdp, n), where bsdp represents batchSize / dataParallelSize.
    - perTokenScaleOptional supports FLOAT8_E8M0. The shape supports 3D with a dimension of (m, Ceil(k/64), 2).
    - rowIndexOptional only supports INT64.
    - x1, x2, scaleOptional, pertokenScaleOptional, groupListOptional, logitOptional, and rowIndexOptional are mandatory parameters; biasOptional and sharedInputOptional are optional parameters. The offsetOptional parameter is currently not supported.
    - When groupListType is 0, groupListOptional must be a non-negative, monotonically non-decreasing sequence (cumulative sum), and the last value must not be greater than the first dimension of the tensor in x1; when groupListType is 1, groupListOptional must be a non-negative array (size of each group), and the sum of the values must not be greater than the first dimension of the tensor in x1.
    - batch (the first dimension of out) and sharedInputOffset must be greater than or equal to 0 and less than or equal to m.
    - When sharedInputOptional is not empty, the length of its first dimension is sharedInputLen. sharedInputOffset, sharedInputLen, and their sum must be less than or equal to batch, the first dimension of out.
    - x1 supports M being 0, and x2 supports N being 0. Other empty tensor use cases are not supported.

  <!-- end id10 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
      <td>The parameter passed in is a null pointer for a required input, output, or attribute.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of a checked tensor does not meet the requirements for the applicable scenario, or an unsupported storage format is detected (for example, out is not ND on some platforms).</td>
    </tr>
    <tr>
      <td>The combination of transposeX1 and transposeX2 is not supported on the current platform.</td>
    </tr>
    <tr>
      <td>The shape of x1, x2, scaleOptional, biasOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupListOptional, sharedInputOptional, logitOptional, rowIndexOptional, or out does not meet the validation conditions.</td>
    </tr>
    <tr>
      <td>An empty tensor is passed in, except when M of x1 is 0 or N of x2 is 0.</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmulFinalizeRoutingV3

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
        <td>Workspace memory address side-applied on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size side-applied on the Device, obtained by the first-phase API aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for executing the task.</td>
      </tr>
    </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: aclnnGroupedMatmulFinalizeRoutingV3 defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnGroupedMatmulFinalizeRoutingV3 defaults to a non-deterministic implementation. Deterministic computation cannot be enabled through aclrtSetSysParamOpt.

  <!-- end id12 -->

<!-- npu="A3,910b" id13 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only pseudo-quantization scenarios are supported.

  - The input and output support the following data type combinations:

    | x1   | x2   | scaleOptional | biasOptional | offsetOptional | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupListOptional | sharedInputOptional | logitOptional | rowIndexOptional | out     |
    | ---- | ---- | ------------- | ------------ | -------------- | ---------------------- | ----------------------- | --------------------- | ----------------- | ------------------- | ------------- | ---------------- | ------- |
    | INT8 | INT4 | INT64         | FLOAT32      | FLOAT32        | null                   | null                    | FLOAT32               | INT64             | BFLOAT16 / null     | FLOAT32       | INT64 / INT32    | FLOAT32 |
    | INT8 | INT4 | INT64         | FLOAT32      | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16 / null     | FLOAT32       | INT64 / INT32    | FLOAT32 |

  - In this scenario, scaleOptional represents the result of offline fusion of per-channel and per-group computations.
  - In this scenario, biasOptional represents an auxiliary result of offline computation. The value must be $8 \times w \times scaleOptional$ and accumulated along the first dimension.
  - This scenario supports both symmetric quantization and asymmetric quantization. For symmetric quantization, offsetOptional must be set to null. For asymmetric quantization, offsetOptional represents an auxiliary result of offline computation, which is the result of $antiquantOffsetOptional \times scaleOptional$.
  - In this scenario, antiquantScaleOptional and antiquantOffsetOptional must be set to null.

<!-- end id13 -->
<!-- npu="950" id14 -->
- <term>Ascend 950PR&950DT products</term>: Only supports the MX full quantization scenario.

  - The input and output support the following data type combinations:

    | MX Quantization Scenario | x1                        | x2                         | scaleOptional | biasOptional  | pertokenScaleOptional | groupListOptional | sharedInputOptional | logitOptional | rowIndexOptional | out     |
    | ---------- | ------------------------- | -------------------------- | ------------- | ------------- | --------------------- | ----------------- | ------------------- | ------------- | ---------------- | ------- |
    | MXFP8      | FLOAT8_E4M3FN / FLOAT8_E5M2 | FLOAT8_E4M3FN / FLOAT8_E5M2 | FLOAT8_E8M0   | BFLOAT16 / null | FLOAT8_E8M0           | INT64             | BFLOAT16  / null    | FLOAT32       | INT64            | FLOAT32 |
    | MXFP4      | FLOAT4_E2M1  | FLOAT4_E2M1 | FLOAT8_E8M0   | BFLOAT16 / null | FLOAT8_E8M0           | INT64             | BFLOAT16 / null     | FLOAT32       | INT64            | FLOAT32 |

  - In the MXFP4/MXFP8 scenario, offsetOptional, antiQuantScaleOptional, and antiQuantOffsetOptional must be set to null.
  - In the MXFP4 scenario, the constraint that k must be an even number must be satisfied. When x2 is not transposed, n must be an even number.
  - In the MXFP4/MXFP8 scenario, x2 transpose or non-transpose is supported. The transpose attributes of x2 and scale must be consistent.
  - e must be less than or equal to 1024.
  - In the MXFP4 scenario, k cannot be 2.

<!-- end id14 -->

## Example

The following is sample code for reference only. For details about how to compile and run the sample, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="A3,910b" id15 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_grouped_matmul_finalize_routing_v3.h"
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
      // Boilerplate code for resource initialization.
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

      // Call the aclCreateTensor interface to create an aclTensor
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
      std::unique_ptr<const aclIntArray, decltype(&aclDestroyIntArray)> mat2SizePtr(mat2Size, aclDestroyIntArray);
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
              return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to allocate device-side memory
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call the aclCreateTensor interface to create an aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

    int main() {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction based on the API.
      int64_t m = 1;
      int64_t k = 2048;
      int64_t n = 1024;
      int64_t e = 1;
      int64_t batch = 1;
      int64_t bsdp = 1;
      int64_t dtype = 0;
      float sharedInputWeight = 1.0;
      int64_t sharedInputOffset = 0;
      bool transposeX = false;
      bool transposeW = false;
      int64_t groupListType = 1;

      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> wShape = {e, k, n / 8};
      std::vector<int64_t> scaleShape = {e, 1, n};
      std::vector<int64_t> biasShape = {e, n};
      std::vector<int64_t> offsetShape = {e, 1, n};
      std::vector<int64_t> pertokenScaleShape = {m};
      std::vector<int64_t> groupListShape = {e};
      std::vector<int64_t> sharedInputShape = {bsdp, n};
      std::vector<int64_t> logitShape = {m};
      std::vector<int64_t> rowIndexShape = {m};
      std::vector<int64_t> outShape = {batch, n};
      std::vector<int64_t> tuningConfigVal = { 1 };

      void *xDeviceAddr = nullptr;
      void *wDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
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
      aclTensor* offset = nullptr;
      aclTensor* pertokenScale = nullptr;
      aclTensor* sharedInput = nullptr;
      aclTensor* logit = nullptr;
      aclTensor* rowIndex = nullptr;
      aclTensor* out = nullptr;

      std::vector<int8_t> xHostData(GetShapeSize(xShape));
      std::vector<int32_t> wHostData(GetShapeSize(wShape));
      std::vector<int64_t> scaleHostData(GetShapeSize(scaleShape));
      std::vector<float> biasHostData(GetShapeSize(biasShape));
      std::vector<float> offsetHostData(GetShapeSize(offsetShape));
      std::vector<float> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
      std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
      std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
      std::vector<float> logitHostData(GetShapeSize(logitShape));
      std::vector<int64_t> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));
      // Assign values to groupList.
      groupListHostData[0] = 1;
      // Create the x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the w aclTensor of int32_t, which is later converted to int_4.
      ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_INT32, &w);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_INT64, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
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

      aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
      CHECK_RET(tuningConfig != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return -1);
      // 3. Call the CANN operator library API. Change it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // Call the first-phase API of aclnnGroupedMatmulFinalizeRoutingV3.
      workspaceSize = 0;
      ret = aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(x, w, scale, bias, offset, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, sharedInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, tuningConfig, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnGroupedMatmulFinalizeRoutingV3.
      ret = aclnnGroupedMatmulFinalizeRoutingV3(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3 failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device side to the host side. Change it based on the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return ret);

      // 6. Release resources.
      xTensorPtr.reset();
      xDeviceAddrPtr.reset();
      wTensorPtr.reset();
      wDeviceAddrPtr.reset();
      scaleTensorPtr.reset();
      scaleDeviceAddrPtr.reset();
      biasTensorPtr.reset();
      biasDeviceAddrPtr.reset();
      offsetTensorPtr.reset();
      offsetDeviceAddrPtr.reset();
      pertokenScaleTensorPtr.reset();
      pertokenScaleDeviceAddrPtr.reset();
      groupListTensorPtr.reset();
      groupListDeviceAddrPtr.reset();
      sharedInputTensorPtr.reset();
      sharedInputDeviceAddrPtr.reset();
      logitTensorPtr.reset();
      logitDeviceAddrPtr.reset();
      rowIndexTensorPtr.reset();
      rowIndexDeviceAddrPtr.reset();
      outTensorPtr.reset();
      outDeviceAddrPtr.reset();

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

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_matmul_finalize_routing_v3.h"

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
      // Boilerplate, resource initialization
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return ACL_SUCCESS;
  }

  template <typename T>
  aclnnStatus CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device-side memory
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory
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
      return ACL_SUCCESS;
  }

  template <typename T>
  aclnnStatus CreateAclTensorWeight(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));
      size *= sizeof(T);
      // Call aclrtMalloc to allocate device-side memory
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of the contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call the aclCreateTensor API to create an aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return ACL_SUCCESS;
  }

  template <typename T1, typename T2>
  auto Ceil(T1 a, T2 b) -> T1
  {
      if (b == 0) {
          return a;
      }
      return (a + b - 1) / b;
  }

    int main() {
      // 1. (Boilerplate) Initialize the device/stream. Refer to the AscendCL external API list.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output as required based on the API.
      int64_t m = 8;
      int64_t k = 2048;
      int64_t n = 7168;
      int64_t e = 1;
      int64_t g = 1;
      int64_t batch = 8;
      int64_t bsdp = 1;
      int64_t dtype = 0;
      float sharedInputWeight = 1.0;
      int64_t sharedInputOffset = 0;
      bool transposeX = false;
      bool transposeW = false;
      int64_t groupListType = 1;

      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> wShape = {e, k, n};
      std::vector<int64_t> scaleShape = {g, Ceil(k,64),n,2};
      std::vector<int64_t> biasShape = {e, n};
      std::vector<int64_t> offsetShape = {e, 1, n};
      std::vector<int64_t> pertokenScaleShape = {m,Ceil(k,64),2};
      std::vector<int64_t> groupListShape = {e};
      std::vector<int64_t> sharedInputShape = {bsdp, n};
      std::vector<int64_t> logitShape = {m};
      std::vector<int64_t> rowIndexShape = {m};
      std::vector<int64_t> outShape = {batch, n};

      void *xDeviceAddr = nullptr;
      void *wDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
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
      aclTensor* offset = nullptr;
      aclTensor* pertokenScale = nullptr;
      aclTensor* sharedInput = nullptr;
      aclTensor* logit = nullptr;
      aclTensor* rowIndex = nullptr;
      aclTensor* out = nullptr;

      std::vector<int8_t> xHostData(GetShapeSize(xShape));
      std::vector<int32_t> wHostData(GetShapeSize(wShape));
      std::vector<int64_t> scaleHostData(GetShapeSize(scaleShape));
      std::vector<float> biasHostData(GetShapeSize(biasShape));
      std::vector<float> offsetHostData(GetShapeSize(offsetShape));
      std::vector<float> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
      std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
      std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
      std::vector<float> logitHostData(GetShapeSize(logitShape));
      std::vector<int64_t> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));
      // Assign values to groupList
      groupListHostData[0] = 8;

      // Create the x aclTensor
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the w aclTensor
      ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &w);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the scale aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the bias aclTensor
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_BF16, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the offset aclTensor
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the pertokenScale aclTensor
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, ACL_FLOAT8_E8M0, &pertokenScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the groupList aclTensor
      ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create sharedInput aclTensor.
      ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16, &sharedInput);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create logit aclTensor.
      ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create rowIndex aclTensor.
      ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // Call the first-phase API of aclnnGroupedMatmulFinalizeRoutingV3.
      ret = aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(x, w, scale, bias, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, sharedInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, nullptr, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnGroupedMatmulFinalizeRoutingV3.
      ret = aclnnGroupedMatmulFinalizeRoutingV3(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3 failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronously wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from device-side memory to the host side. Modify it according to the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < 10 && i < size; i++) {
          LOG_PRINT("result[%lld] is: %f\n", i, resultData[i]);
      }

      // 6. Release tensors managed by unique_ptr before shutting down the ACL runtime.
      outTensorPtr.reset();
      rowIndexTensorPtr.reset();
      logitTensorPtr.reset();
      sharedInputTensorPtr.reset();
      groupListTensorPtr.reset();
      pertokenScaleTensorPtr.reset();
      offsetTensorPtr.reset();
      biasTensorPtr.reset();
      scaleTensorPtr.reset();
      wTensorPtr.reset();
      xTensorPtr.reset();

      // 7. Release device buffers managed by unique_ptr before shutting down the ACL runtime.
      outDeviceAddrPtr.reset();
      rowIndexDeviceAddrPtr.reset();
      logitDeviceAddrPtr.reset();
      sharedInputDeviceAddrPtr.reset();
      groupListDeviceAddrPtr.reset();
      pertokenScaleDeviceAddrPtr.reset();
      offsetDeviceAddrPtr.reset();
      biasDeviceAddrPtr.reset();
      scaleDeviceAddrPtr.reset();
      wDeviceAddrPtr.reset();
      xDeviceAddrPtr.reset();
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```

<!-- end id16 -->
