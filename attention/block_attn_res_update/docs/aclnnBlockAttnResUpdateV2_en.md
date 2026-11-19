# aclnnBlockAttnResUpdateV2

## Product Support

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

## Function Description

- Function:

  Performs the second stage of Attention Residuals computation: accumulates `delta` into `partialBlockRef` in place, computes the RMS score of the updated residual, and merges it with the historical online Softmax state to produce the current layer output. Optionally applies fused root mean square normalization (RMSNorm) to this output. The historical state can be taken from a selected slot of [aclnnBlockAttnResPrepare](../../block_attn_res_prepare/docs/aclnnBlockAttnResPrepare_en.md).

- API evolution:

  Extends [aclnnBlockAttnResUpdate](aclnnBlockAttnResUpdate_en.md) with the optional input `outputNormWeightOptional` and the independent parameter `outputNormEps` to control output RMSNorm.

- Computation:

  For each token $t$, first update the partial residual and write it back in place:

  $$
  p_t=P_t+\operatorname{FP32}(\Delta_t),\qquad
  P^{\mathrm{out}}_t=p_t
  $$

  Compute the score of the current residual using RMSNorm and a query dot product:

  $$
  C_t=\left\langle Q,\operatorname{RMSNorm}(p_t,\mathbf{1},\epsilon)\right\rangle
  $$

  Merge the current residual with the historical online Softmax state:

  $$
  m_t=\max(M_t,C_t),\qquad
  \alpha_t=\exp(M_t-m_t),\qquad
  \beta_t=\exp(C_t-m_t)
  $$

  $$
  H^{(0)}_t=\operatorname{BF16}\left(
  \frac{\alpha_t O_t+\beta_t p_t}{\alpha_t L_t+\beta_t}
  \right)
  $$

  When `outputNormWeightOptional` is `nullptr`, the output is $H=H^{(0)}$. Otherwise, apply output RMSNorm to the mixture after it has been rounded to BFLOAT16:

  $$
  H_t=\operatorname{BF16}\left(
  \operatorname{RMSNorm}\left(\operatorname{FP32}(H^{(0)}_t),\operatorname{FP32}(W),\epsilon_{\mathrm{out}}\right)
  \right)
  $$

  Where:

  - $T$ and $D$ are the token count and hidden dimension, with $t\in[0,T)$. $P_t$, $p_t$, $\Delta_t$, $O_t$, $H_t$, and $Q$ are all $D$-dimensional vectors.
  - $P$ and $P^{\mathrm{out}}$ are the values of `partialBlockRef` before and after the call. $\Delta$ is `delta`, $Q$ is `pseudoQuery`, and $p$ is the FLOAT32 residual sum.
  - $O$, $M$, and $L$ are the historical state inputs `numerator`, `logitMax`, and `expSum`. $H$ is the output `h`, and $W$ is `outputNormWeightOptional`.
  - $\epsilon=\operatorname{FP32}(\texttt{eps})$ and $\epsilon_{\mathrm{out}}=\operatorname{FP32}(\texttt{outputNormEps})$ are used for scoring and output normalization, respectively.
  - RMSNorm operates along $D$ and uses an all-ones scale vector for scoring. The scoring scale can be multiplied elementwise into the original query in advance to form $Q$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnBlockAttnResUpdateV2GetWorkspaceSize` first to obtain the workspace size and the executor containing the computation flow, and then call `aclnnBlockAttnResUpdateV2` to perform the computation. Both `eps` and `outputNormEps` must be supplied by the caller.

```cpp
aclnnStatus aclnnBlockAttnResUpdateV2GetWorkspaceSize(
  aclTensor        *partialBlockRef,
  const aclTensor  *delta,
  const aclTensor  *pseudoQuery,
  const aclTensor  *numerator,
  const aclTensor  *logitMax,
  const aclTensor  *expSum,
  const aclTensor  *outputNormWeightOptional,
  double            eps,
  double            outputNormEps,
  aclTensor        *h,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnBlockAttnResUpdateV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBlockAttnResUpdateV2GetWorkspaceSize

- **Parameters**

  All tensor arguments must use ND for both their original and storage formats and must be contiguous.

  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 100px">
      <col style="width: 230px">
      <col style="width: 330px">
      <col style="width: 160px">
      <col style="width: 90px">
      <col style="width: 150px">
      <col style="width: 140px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimensions (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap;">partialBlockRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>The accumulated partial residual, updated in place with the residual sum; P and p in the formulas.</td>
        <td>
          <ul>
            <li>T &gt;= 0, 0 &lt;= D &lt;= 8192.</li>
            <li>Supports empty tensors, which must still satisfy all input/output shape relationships.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">delta (const aclTensor*)</td>
        <td>Input</td>
        <td>The increment to be added to the partial residual; Δ in the formulas.</td>
        <td>
          <ul>
            <li>The shape must match partialBlockRef.</li>
            <li>Supports empty tensors.</li>
          </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">pseudoQuery (const aclTensor*)</td>
        <td>Input</td>
        <td>The effective query used to compute the normalized score of the current residual; Q in the formulas.</td>
        <td>
          <ul>
            <li>The length must equal D of partialBlockRef.</li>
            <li>Supports an empty tensor when D is 0.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">numerator (const aclTensor*)</td>
        <td>Input</td>
        <td>The historical online Softmax weighted numerator; O in the formulas. This API does not modify it.</td>
        <td>
          <ul>
            <li>The shape must match partialBlockRef.</li>
            <li>Supports empty tensors. For the numerical requirements of a non-empty historical state, see <a href="#constraints">Constraints</a>.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">logitMax (const aclTensor*)</td>
        <td>Input</td>
        <td>The historical maximum logit for each token; M in the formulas. This API does not modify it.</td>
        <td>
          <ul>
            <li>The length must equal T of partialBlockRef.</li>
            <li>Supports an empty tensor when T is 0.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[T]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">expSum (const aclTensor*)</td>
        <td>Input</td>
        <td>The historical Softmax exponential sum for each token; L in the formulas. This API does not modify it.</td>
        <td>
          <ul>
            <li>The length must equal T of partialBlockRef.</li>
            <li>Supports an empty tensor when T is 0.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[T]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>outputNormWeightOptional (const aclTensor*)</td>
        <td>Optional Input</td>
        <td>The RMSNorm gamma for the next sublayer; W in the formulas.</td>
        <td>
          <ul>
            <li>Pass nullptr to disable output RMSNorm.</li>
            <li>A non-null pointer enables output RMSNorm. The length must equal D.</li>
            <li>When D is 0, an empty tensor of shape [0] is allowed. When D is greater than 0, an empty tensor cannot be used in place of nullptr.</li>
          </ul>
        </td>
        <td>BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">eps (double)</td>
        <td>Input</td>
        <td>The numerical stability term for the RMS score of the current residual; ε in the formulas.</td>
        <td>Must be finite and positive. For non-empty workloads, the value after conversion to FLOAT32 must also be finite and greater than 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">outputNormEps (double)</td>
        <td>Input</td>
        <td>The independent numerical stability term for output RMSNorm; ε_out in the formulas.</td>
        <td>When outputNormWeightOptional is non-null, the value before conversion to FLOAT32 must be in <code>[FLT_MIN, FLT_MAX]</code>. When outputNormWeightOptional is nullptr, this parameter is ignored and its value is not validated.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">h (aclTensor*)</td>
        <td>Output</td>
        <td>The current layer output; H in the formulas. Contains the normalized result when output RMSNorm is enabled.</td>
        <td>
          <ul>
            <li>The shape must match partialBlockRef.</li>
            <li>Returns an empty tensor of the corresponding shape when T or D is 0.</li>
          </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to allocate on the device, in bytes.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the executor containing the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API validates the arguments and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value/Status Code</th>
        <th>Code</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap;">ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, h, workspaceSize, or executor is null. outputNormWeightOptional may be null.</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>
          <ul>
            <li>The data type, original format, or storage format of a required tensor or a non-null outputNormWeightOptional does not meet the parameter requirements.</li>
            <li>A tensor rank or a shape relationship does not meet the parameter requirements, including the requirement that a non-null outputNormWeightOptional have shape [D].</li>
            <li>T is less than 0, D is outside [0, 8192], or T * D exceeds the range of int64_t for a non-empty workload.</li>
            <li>A required tensor or a non-null outputNormWeightOptional is non-contiguous.</li>
            <li>eps is not a finite positive number.</li>
            <li>outputNormWeightOptional is non-null and outputNormEps is outside <code>[FLT_MIN, FLT_MAX]</code> before conversion to FLOAT32.</li>
          </ul>
        </td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>The current runtime environment is outside the supported product range.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>The operator executor could not be created.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>Creating a tensor view, the computation node, or the output-copy node failed.</td>
      </tr>
    </tbody>
  </table>

## aclnnBlockAttnResUpdateV2

- **Parameters**

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 700px">
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
        <td style="white-space: nowrap;">workspace (void*)</td>
        <td>Input</td>
        <td>Device workspace address. May be <code>nullptr</code> when <code>workspaceSize</code> is zero.</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">workspaceSize (uint64_t)</td>
        <td>Input</td>
        <td>The workspace size allocated on the device, obtained from the first-phase API aclnnBlockAttnResUpdateV2GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">executor (aclOpExecutor*)</td>
        <td>Input</td>
        <td>The executor containing the operator computation flow.</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">stream (aclrtStream)</td>
        <td>Input</td>
        <td>The stream on which to execute the task.</td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Shape relationships**: `partialBlockRef`, `delta`, `numerator`, and `h` must have shape `[T, D]`; `pseudoQuery` and a non-null `outputNormWeightOptional` must have shape `[D]`; `logitMax` and `expSum` must have shape `[T]`.
- **Empty tensors**: When `T == 0` or `D == 0`, the API returns an empty output of the corresponding shape after first-phase argument validation, without launching the computation kernel. Required tensors cannot be replaced with `nullptr`. If a weight is supplied, it must still satisfy the `[D]` shape, data type, ND format, and contiguity requirements, and `outputNormEps` is still validated.
- **Non-empty workloads**: `T > 0`, `1 <= D <= 8192`, and `T * D` must fit in `int64_t`. The per-core T extent must fit in `uint32_t`. The full-D data and optional weight must fit within the unified buffer (UB) budget; otherwise, tiling fails.
- **Historical state**: For non-empty workloads, the caller must provide a non-empty historical set for each token. Elements of `numerator` and `logitMax` must be finite, and elements of `expSum` must be finite and strictly positive. All three inputs must describe the same historical online Softmax state. This API does not update these inputs or read their elements for host validation. An empty historical state represented by `logitMax=-inf`, `expSum=0`, and a zero `numerator` is not supported.
- **Output normalization epsilon**: When output RMSNorm is enabled, validate `outputNormEps` against `[FLT_MIN, FLT_MAX]` before FLOAT32 conversion, where `FLT_MIN = 2^-126 ≈ 1.1754943508222875e-38` and `FLT_MAX ≈ 3.4028234663852886e38`. The default Sqrt uses FTZ, so subnormal epsilon values cannot ensure a nonzero denominator for an all-zero mixture.
- **Precision and numerical behavior**: `eps` and, when enabled, `outputNormEps` are converted to FLOAT32 for computation and serve independent purposes. Output RMSNorm consumes the mixture after it has been rounded to BFLOAT16; this rounding step is retained. Input elements containing zero, extreme values, NaN, or Inf receive no additional handling. Overflow in squaring, multiplication, or accumulation may produce non-finite results.
- **Deterministic computation**:

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnBlockAttnResUpdateV2 uses a deterministic implementation by default.
  <!-- end id7 -->

## Examples

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

The complete sample source is [test_aclnn_block_attn_res_update_v2.cpp](../examples/arch35/test_aclnn_block_attn_res_update_v2.cpp). The sample uses FLOAT32 `outputNormWeightOptional` and enables output RMSNorm by default, with `eps=1e-6` and `outputNormEps=1e-5`. Run the executable with `--no-output-norm` to disable output RMSNorm by passing `nullptr`.

Compile and run the sample from the repository root:

```bash
bash build.sh --run_example block_attn_res_update eager --soc=ascend950 \
    --example_name=test_aclnn_block_attn_res_update_v2
```

```cpp
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_update_v2.h"

#define CHECK_RET(cond, returnExpr) \
    do { \
        if (!(cond)) { \
            returnExpr; \
        } \
    } while (0)

#define CHECK_FREE_RET(cond, returnExpr) \
    do { \
        if (!(cond)) { \
            Finalize(deviceId, stream); \
            returnExpr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr size_t FIRST_ELEMENT_INDEX = 0UL;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16U);
}

float Bfloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int Init(int32_t deviceId, aclrtStream* stream)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat formatType, aclTensor** tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
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
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int RunBlockAttnResUpdateV2(int32_t deviceId, aclrtStream& stream, bool applyOutputNorm)
{
    // 1. Initialize the device and stream.
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Prepare the inputs, output, and RMSNorm weights.
    constexpr int64_t tokenNum = 2;
    constexpr int64_t hiddenSize = 64;
    const std::vector<int64_t> matrixShape = {tokenNum, hiddenSize};
    const std::vector<int64_t> queryShape = {hiddenSize};
    const std::vector<int64_t> statsShape = {tokenNum};

    std::vector<float> partialBlockRefHostData(GetShapeSize(matrixShape), 0.25F);
    std::vector<uint16_t> deltaHostData(GetShapeSize(matrixShape), FloatToBfloat16(0.125F));
    std::vector<float> pseudoQueryHostData(GetShapeSize(queryShape), 1.0F / static_cast<float>(hiddenSize));
    std::vector<float> numeratorHostData(GetShapeSize(matrixShape), 0.5F);
    std::vector<float> logitMaxHostData(GetShapeSize(statsShape), 0.0F);
    std::vector<float> expSumHostData(GetShapeSize(statsShape), 1.0F);
    std::vector<uint16_t> hHostData(GetShapeSize(matrixShape), 0U);
    std::vector<float> outputNormWeightHostData(hiddenSize);
    for (int64_t d = 0; d < hiddenSize; ++d) {
        outputNormWeightHostData[d] = 0.5F + static_cast<float>(d % 8) * 0.125F;
    }

    void* partialBlockRefDeviceAddr = nullptr;
    void* deltaDeviceAddr = nullptr;
    void* pseudoQueryDeviceAddr = nullptr;
    void* numeratorDeviceAddr = nullptr;
    void* logitMaxDeviceAddr = nullptr;
    void* expSumDeviceAddr = nullptr;
    void* hDeviceAddr = nullptr;

    aclTensor* partialBlockRef = nullptr;
    aclTensor* delta = nullptr;
    aclTensor* pseudoQuery = nullptr;
    aclTensor* numerator = nullptr;
    aclTensor* logitMax = nullptr;
    aclTensor* expSum = nullptr;
    aclTensor* h = nullptr;

    // 3. Create contiguous ND tensors and manage resources with RAII.
    ret = CreateAclTensor<float>(partialBlockRefHostData, matrixShape, &partialBlockRefDeviceAddr,
                                 aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &partialBlockRef);
    std::unique_ptr<void, aclError (*)(void*)> partialBlockRefDeviceAddrPtr(partialBlockRefDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> partialBlockRefTensorPtr(partialBlockRef,
                                                                                           aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint16_t>(deltaHostData, matrixShape, &deltaDeviceAddr, aclDataType::ACL_BF16,
                                    aclFormat::ACL_FORMAT_ND, &delta);
    std::unique_ptr<void, aclError (*)(void*)> deltaDeviceAddrPtr(deltaDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> deltaTensorPtr(delta, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(pseudoQueryHostData, queryShape, &pseudoQueryDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &pseudoQuery);
    std::unique_ptr<void, aclError (*)(void*)> pseudoQueryDeviceAddrPtr(pseudoQueryDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> pseudoQueryTensorPtr(pseudoQuery, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(numeratorHostData, matrixShape, &numeratorDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &numerator);
    std::unique_ptr<void, aclError (*)(void*)> numeratorDeviceAddrPtr(numeratorDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> numeratorTensorPtr(numerator, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(logitMaxHostData, statsShape, &logitMaxDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &logitMax);
    std::unique_ptr<void, aclError (*)(void*)> logitMaxDeviceAddrPtr(logitMaxDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> logitMaxTensorPtr(logitMax, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(expSumHostData, statsShape, &expSumDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &expSum);
    std::unique_ptr<void, aclError (*)(void*)> expSumDeviceAddrPtr(expSumDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> expSumTensorPtr(expSum, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint16_t>(hHostData, matrixShape, &hDeviceAddr, aclDataType::ACL_BF16,
                                    aclFormat::ACL_FORMAT_ND, &h);
    std::unique_ptr<void, aclError (*)(void*)> hDeviceAddrPtr(hDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> hTensorPtr(h, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 4. Set the two epsilons independently; keep the weight null when output RMSNorm is disabled.
    constexpr double eps = 1.0e-6;
    constexpr double outputNormEps = 1.0e-5;
    void* outputNormWeightDeviceAddr = nullptr;
    aclTensor* outputNormWeightOptional = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> outputNormWeightDevicePtr(nullptr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outputNormWeightTensorPtr(nullptr, aclDestroyTensor);
    if (applyOutputNorm) {
        ret = CreateAclTensor<float>(outputNormWeightHostData, queryShape, &outputNormWeightDeviceAddr,
                                     aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &outputNormWeightOptional);
        outputNormWeightDevicePtr.reset(outputNormWeightDeviceAddr);
        outputNormWeightTensorPtr.reset(outputNormWeightOptional);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // 5. Call the first-phase API to obtain the workspace size and executor.
    ret = aclnnBlockAttnResUpdateV2GetWorkspaceSize(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum,
                                                    outputNormWeightOptional, eps, outputNormEps, h, &workspaceSize,
                                                    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResUpdateV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 6. Allocate the workspace using the size returned by the first-phase API.
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 7. Call the second-phase API and wait for completion.
    ret = aclnnBlockAttnResUpdateV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResUpdateV2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 8. Copy the updated residual and BF16 output back to the host.
    const size_t partialBlockRefSize = partialBlockRefHostData.size() * sizeof(float);
    ret = aclrtMemcpy(partialBlockRefHostData.data(), partialBlockRefSize, partialBlockRefDeviceAddr,
                      partialBlockRefSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy partialBlockRef to host failed. ERROR: %d\n", ret); return ret);

    const size_t hSize = hHostData.size() * sizeof(uint16_t);
    ret = aclrtMemcpy(hHostData.data(), hSize, hDeviceAddr, hSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy h to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("partialBlockRef[0] after in-place update: %.6f\n", partialBlockRefHostData[FIRST_ELEMENT_INDEX]);
    LOG_PRINT("h[0]: %.6f\n", Bfloat16ToFloat(hHostData[FIRST_ELEMENT_INDEX]));
    LOG_PRINT("Output RMSNorm: %s\n", applyOutputNorm ? "enabled" : "disabled");
    LOG_PRINT("block_attn_res_update_v2 example execute success.\n");
    return ACL_SUCCESS;
}

int main(int argc, char** argv)
{
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--no-output-norm") != 0)) {
        LOG_PRINT("Usage: %s [--no-output-norm]\n", argv[0]);
        return 1;
    }
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = RunBlockAttnResUpdateV2(deviceId, stream, argc == 1);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBlockAttnResUpdateV2 failed. ERROR: %d\n", ret); return ret);

    // Tensors and device memory were released when the Run function returned; release the stream and device.
    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
