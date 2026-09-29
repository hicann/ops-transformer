# aclnnGroupedMatmulFinalizeRoutingV3

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_finalize_routing)

## Supported Products

| Product                                                            | Supported|
| :--------------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                 |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>|    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                 |    ×    |
| <term>Atlas inference products</term>                         |    ×    |
| <term>Atlas training products</term>                         |    ×    |

## Function

- API function:
  A fused operator of `GroupedMatmul` and `MoeFinalizeRouting`. It performs a "combine" operation on the output of the `GroupedMatmul` computation based on specified indices.
  
  Compared with aclnnGroupedMatmulFinalizeRoutingV2, this API has the following new features:
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The input parameter tuningConfigOptional is added, which is used for tuning. The first value in the array indicates the expected number of tokens to be processed by each expert. Operator tiling is performed based on this expected value to achieve higher performance.
    - Ascend 950PR/Ascend 950DT: The MX quantization scenario is added. For details, see [quantization methods](../../../docs/en/context/quant_mode_introduction.md).
- Formulas:

  1. Group matrix multiplication (GMM):

      $$
      y_i=(x_i\times weight_i) * scale_i * perTokenScale_i
      $$

  2. Routing expert and expert output allocation:

        For each token j, routing and output expert allocation are performed as follows:

        $$
        y[ rowIndex[i] , : ] = y[rowIndex[i], :] + 
        y_{i(j)}[ j - start_{i(j)}]
        $$

        $i(j)$ is the index of the expert to which token j is allocated. $y_{i(j)}[ j - start_{i(j)}]$ is the computation result of the token under the corresponding expert.

  3. Output fusion of shared experts:

        $$
        y [rowIndex[i],:] = y[rowIndex[i],:] + sharedInputWeight \times sharedInput[j, :]
        $$

  4. Output fusion of shared experts: The final output is the result of merging all expert outputs and shared expert outputs based on rowIndex. The calculation process is as follows:

        $$
        y[rowIndex[i],:] = \sum_{i \in \mathcal{E}[j]} y_i [j - start_i] + sharedInputWeight \times sharedInput[j, :]
        $$

        $\mathcal{E}[j]$ is the set of experts assigned to token j.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnGroupedMatmulFinalizeRoutingV3` is called to perform computation.

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
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
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input weight (right matrix).</td>
      <td>-</td>
      <td>INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>3D supported</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>Input</td>
      <td>Scale factor in the quantization parameters, per-channel quantization parameter.</td>
      <td>-</td>
      <td>INT64, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Matrix offset.</td>
      <td>-</td>
      <td>FLOAT32, BF16</td>
      <td>ND</td>
      <td>2D, with the dimension of (e, n)</td>
      <td>-</td>
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
      <td>Fake-quantization scale factor.</td>
      <td>Not available currently</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>Input</td>
      <td>Fake-quantization offset.</td>
      <td>Not available currently</td>
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
      <td>-</td>
    </tr>
    <tr>
      <td>groupListOptional</td>
      <td>Input</td>
      <td>Size distribution of the input and output in the group axis direction of matmul.</td>
      <td></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1D tensor of shape (e)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputOptional</td>
      <td>Input</td>
      <td>Output of the shared expert in MOE computation, which needs to be combined with the output of the MOE expert.</td>
      <td></td>
      <td>BF16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>logitOptional</td>
      <td>Input</td>
      <td>Logit size of each token by the MOE expert.</td>
      <td></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1D tensor of shape (m) is supported.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowIndexOptional</td>
      <td>Input</td>
      <td>The output of the MOE expert is combined based on the rowIndex. The value is the index for scatter add in the combination.</td>
      <td></td>
      <td>INT64</td>
      <td>ND</td>
      <td>The shape supports one dimension, and the dimension is (m).</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Input</td>
      <td>Computation output type. 0: FLOAT32; 1: FLOAT16; 2: BFLOAT16 Currently, only 0 is supported.</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>Input</td>
      <td>Combination coefficient. sharedInput is multiplied by this parameter first, and then the result is accumulated with the MoE experts' results.</td>
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
      <td>Whether to transpose the left matrix. The value can only be false.</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Whether to transpose the right matrix. The value can only be false.</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>Input</td>
      <td>Grouping mode. 0: cumsum mode (prefix sum); 1: count mode</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <td>tuningConfigOptional</td>
      <td>Input</td>
      <td>The first element in the array indicates the expected number of tokens to be processed by each expert. Operator tiling is performed based on the first element to achieve higher performance. The second and subsequent elements are reserved. You do not need to set them. It will be extended in the future. It is compatible with earlier versions. If this parameter is not used, do not pass it (that is, pass nullptr).</td>
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
      <td>Its shape is the same as that of self.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(batch, n)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

  - x1 supports only INT8. The shape is (m, k), where m is in the range of [1, 16 x 1024 x 8] and k is 2048.
  - x2 supports only INT4. When the input is of type INT32, the dimension is (e, k, n/8). When the input is converted to type INT4, the dimension is (e, k, n). The value range of e is [1, 256], k is 2048, and n is 7168.
  - scaleOptional supports INT64. The shape supports three dimensions, and the dimension is (e, 1, n). The value of e and n is the same as that of e and n in w.
  - biasOptional supports FLOAT32. The value of e and n is the same as that of e and n in w.
  - offsetOptional supports FLOAT32. The shape supports three dimensions, and the dimension is (e, 1, n). The value of e and n is the same as that of e and n in w.
  - perTokenScaleOptional supports FLOAT32. The shape supports one dimension, and the dimension is (m). The value of m is the same as that of m in x.
  - groupListOptional supports the same value of e in e and w.
  - sharedInputOptional supports two dimensions, and the dimension is (bsdp, n). The value of bsdp must be less than or equal to batchSize/e, and the value of n is the same as that of n in w.
  - logitOptional supports the same value of m in m and x.
  - rowIndexOptional supports the same value of m in m and x.
  - x1, x2, and groupListOptional are mandatory. scaleOptional, perTokenScaleOptional, logitOptional, rowIndexOptional, biasOptional, and sharedInputOptional are optional.

- Ascend 950PR/Ascend 950DT:
  - x1 does not support INT8.
  - x2 does not support INT4. The dimension is (e, k, n). In the case of transposition, the dimension is (e, n, k). The value range of e is [1, 1024].
  - scaleOptional supports FLOAT8_E8M0. The shape supports four dimensions, and the dimension is (e, n, ceil(k/64), 2). The data type can only be FLOAT8_E8M0. The transpose attribute must be the same as that of x2.
  - biasOptional supports BF16.
  - sharedInputOptional supports two dimensions, and the dimension is (bsdp, n). The value of bsdp is batchSize/dataParallelSize.
  - perTokenScaleOptional supports FLOAT8_E8M0. The shape supports three dimensions, and the dimension is (m, ceil(k/64), 2).
  - x1, x2, scaleOptional, pertokenScaleOptional, groupListOptional, logitOptional, and rowIndexOptional are mandatory parameters. biasOptional and sharedInputOptional are optional parameters. Currently, the offsetOptional parameter is not supported. None of the parameters supports empty tensors.
  - The first dimension (batch) and sharedInputOffset of out must be greater than or equal to 0.
  - x1 supports empty tensors with M being 0.
  - x2 supports empty tensors with N being 0.
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following errors may be thrown:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 700px">
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
      <td>The input parameter is mandatory for input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type or data format of x1, x2, scaleOptional, biasOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupListOptional, sharedInputOptional, logitOptional, rowIndexOptional, sharedInputWeight, sharedInputOffset, transposeX1, transposeX2, or out is not supported.</td>
    </tr>
    <tr>
      <td>The shape of x1, x2, scaleOptional, biasOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupListOptional, sharedInputOptional, logitOptional, rowIndexOptional, or out does not meet the validation conditions.</td>
    </tr>
    <tr>
      <td>x1, x2, scaleOptional, biasOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, pertokenScaleOptional, groupListOptional, sharedInputOptional, logitOptional, rowIndexOptional, or out is an empty tensor.</td>
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
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Address of the workspace to be allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize.</td>
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
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: aclnnGroupedMatmulFinalizeRoutingV3 is implemented in non-deterministic mode by default. Deterministic implementation can be enabled by calling aclrtCtxSetSysParamOpt.
  - Ascend 950PR/Ascend 950DT: aclnnGroupedMatmulFinalizeRoutingV3 is implemented in non-deterministic mode by default. Deterministic implementation cannot be enabled by calling aclrtCtxSetSysParamOpt.

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Only the fake-quantization scenario is supported.
  
  - The following table describes the supported input and output data type combinations.

    | x1   | x2   | scaleOptional | biasOptional | offsetOptional | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional | groupListOptional | sharedInputOptional | logitOptional | rowIndexOptional | out     |
    | ---- | ---- | ------------- | ------------ | -------------- | ---------------------- | ----------------------- | --------------------- | ----------------- | ------------------- | ------------- | ---------------- | ------- |
    | INT8 | INT4 | INT64         | FLOAT32      | FLOAT32        | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT32 |
    | INT8 | INT4 | INT64         | FLOAT32      | null           | null                   | null                    | FLOAT32               | INT64             | BFLOAT16            | FLOAT32       | INT64            | FLOAT32 |

  - In this scenario, `scaleOptional` represents the result of per-channel and per-group offline fusion.
  - In this scenario, `biasOptional` represents the auxiliary result of offline computation. Its value must be $8 \times w \times scaleOptional$ and is accumulated in the first dimension.
  - This scenario supports symmetric quantization and asymmetric quantization. During symmetric quantization, `offsetOptional` must be null. During asymmetric quantization, `offsetOptional` represents the auxiliary result of offline computation, which is the result of $antiquantOffsetOptional \times scaleOptional$.
  - In this scenario, `antiquantScaleOptional` and `antiquantOffsetOptional` must be null.

- Ascend 950PR/Ascend 950DT: Only the full quantization scenario of the MX is supported.
  
  - The following table describes the supported input and output data type combinations.

    | MX quantization scenario| x1                        | x2                         | scaleOptional | biasOptional  | pertokenScaleOptional | groupListOptional | sharedInputOptional | logitOptional | rowIndexOptional | out     |
    | ---------- | ------------------------- | -------------------------- | ------------- | ------------- | --------------------- | ----------------- | ------------------- | ------------- | ---------------- | ------- |
    | MXFP8      | FLOAT8_E4M3FN / FLOAT8_E5M2 | FLOAT8_E4M3FN / FLOAT8_E5M2 | FLOAT8_E8M0   | BFLOAT16 / null | FLOAT8_E8M0           | INT64             | BFLOAT16  / null    | FLOAT32       | INT64            | FLOAT32 |
    | MXFP4      | FLOAT4_E2M1  | FLOAT4_E2M1 | FLOAT8_E8M0   | BFLOAT16 / null | FLOAT8_E8M0           | INT64             | BFLOAT16 / null     | FLOAT32       | INT64            | FLOAT32 |

  - In the MXFP4/MXFP8 scenario, offsetOptional, antiquantScaleOptional, and antiquantOffsetOptional must be left empty.
  - In the MXFP4 scenario, k must be an even number. In the case of x2 non-transposition, n must be an even number.
  - In the MXFP4/MXFP8 scenario, x2 transposition or non-transposition is supported. The transposition attributes of x2 and scale must be the same.
  - e must be less than or equal to 1024.
  - In the MXFP4 scenario, k cannot be 2.

## Calling Example

The following is a call example, which is for reference only. For details about the compilation and execution processes, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

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
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        // Call aclrtMalloc to allocate device memory.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy host data to the device memory.
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
    int CreateAclTensorWeight(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                          aclDataType dataType, aclTensor **tensor)
    {
        auto size = static_cast<uint64_t>(GetShapeSize(shape));

        const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
        auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                  return ret);
        size *= sizeof(T);

        // Call aclrtMalloc to allocate device memory.
        ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy host data to the device memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Compute the strides of the contiguous tensor.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        std::vector<int64_t> storageShape;
        storageShape.push_back(GetShapeSize(shape));

        // Call aclCreateTensor to create an aclTensor.
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  storageShape.data(), storageShape.size(), *deviceAddr);
        return 0;
    }

      int main() {
        int32_t deviceId = 0;
        aclrtStream stream;
        auto ret = Init(deviceId, &stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

        // 2. Construct inputs and outputs based on API definitions.
        int64_t m = 8;
        int64_t k = 2048;
        int64_t n = 7168;
        int64_t e = 1;
        int64_t batch = 8;
        int64_t bsdp = 1;
        int64_t dtype = 0;
        float shareInputWeight = 1.0;
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
        std::vector<int64_t> logitHostData(GetShapeSize(logitShape));
        std::vector<float> rowIndexHostData(GetShapeSize(rowIndexShape));
        std::vector<float> outHostData(GetShapeSize(outShape));
        //Assign a value to groupList.
        groupListHostData[0] = 8;
        // Create an x aclTensor.
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create an int32_t w aclTensor, which will be converted to int_4.
        ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_INT32, &w);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a scale aclTensor.
        ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_INT64, &scale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a bias aclTensor.
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create an offset aclTensor.
        ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a pertokenScale aclTensor.
        ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &pertokenScale);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a groupList aclTensor.
        ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a sharedInput aclTensor.
        ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16, &sharedInput);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a logit aclTensor.
        ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create a rowIndex aclTensor.
        ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create an out aclTensor.
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
        std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
        CHECK_RET(tuningConfig == nullptr, -1);
        // 3. Call the CANN operator library API. Change the API name to the actual one.
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        // Call the first-phase API of aclnnGroupedMatmulFinalizeRoutingV3.
        workspaceSize = 0;
        ret = aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(x, w, scale, bias, offset, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, shareInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, tuningConfig, out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);
        // Allocate device memory based on workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API of aclnnGroupedMatmulFinalizeRoutingV3.
        ret = aclnnGroupedMatmulFinalizeRoutingV3(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3 failed. ERROR: %d\n", ret); return ret);

        // 4. (Boilerplate) Wait until the task execution is complete.
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
        auto size = GetShapeSize(outShape);
        std::vector<float> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("result[%lld] is: %f\n", i, resultData[i]);
        }

        // 6. Free aclTensor resources. Modify the code based on the API definition.
        aclDestroyTensor(x);
        aclDestroyTensor(w);
        aclDestroyTensor(scale);
        aclDestroyTensor(bias);
        aclDestroyTensor(offset);
        aclDestroyTensor(pertokenScale);
        aclDestroyTensor(groupList);
        aclDestroyTensor(sharedInput);
        aclDestroyTensor(logit);
        aclDestroyTensor(rowIndex);
        aclDestroyTensor(out);

        // 7. Free device resources. Modify the code based on the API definition.
        aclrtFree(xDeviceAddr);
        aclrtFree(wDeviceAddr);
        aclrtFree(scaleDeviceAddr);
        aclrtFree(biasDeviceAddr);
        aclrtFree(offsetDeviceAddr);
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

- Ascend 950PR/Ascend 950DT:

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
      // (Boilerplate) Initialize resources.
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
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to the device memory.
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
      return ACL_SUCCESS;
  }

  template <typename T>
  aclnnStatus CreateAclTensorWeight(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));
      size *= sizeof(T);
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call aclCreateTensor to create an aclTensor.
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
      // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs based on API definitions.
      int64_t m = 8;
      int64_t k = 2048;
      int64_t n = 7168;
      int64_t e = 1;
      int64_t g = 1;
      int64_t batch = 8;
      int64_t bsdp = 1;
      int64_t dtype = 0;
      float shareInputWeight = 1.0;
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
      std::vector<int64_t> logitHostData(GetShapeSize(logitShape));
      std::vector<float> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));
    
      groupListHostData[0] = 8;
    
      // Create an x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create a w aclTensor.
      ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &w);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create a scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create a bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_BF16, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create an offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create a pertokenScale aclTensor.
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, ACL_FLOAT8_E8M0, &pertokenScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create a groupList aclTensor.
      ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create a sharedInput aclTensor.
      ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16, &sharedInput);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create a logit aclTensor.
      ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create a rowIndex aclTensor.
      ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Change the API name to the actual one.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // Call the first-phase API of aclnnGroupedMatmulFinalizeRoutingV3.
      ret = aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(x, w, scale, bias, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, shareInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, nullptr, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnGroupedMatmulFinalizeRoutingV3.
      ret = aclnnGroupedMatmulFinalizeRoutingV3(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingV3 failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%lld] is: %f\n", i, resultData[i]);
      }

      // 6. Release aclTensors. Modify the code based on the API definition.
      aclDestroyTensor(x);
      aclDestroyTensor(w);
      aclDestroyTensor(scale);
      aclDestroyTensor(bias);
      aclDestroyTensor(offset);
      aclDestroyTensor(pertokenScale);
      aclDestroyTensor(groupList);
      aclDestroyTensor(sharedInput);
      aclDestroyTensor(logit);
      aclDestroyTensor(rowIndex);
      aclDestroyTensor(out);

      // 7. Free device resources. Modify the code based on the API definition.
      aclrtFree(xDeviceAddr);
      aclrtFree(wDeviceAddr);
      aclrtFree(scaleDeviceAddr);
      aclrtFree(biasDeviceAddr);
      aclrtFree(offsetDeviceAddr);
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
