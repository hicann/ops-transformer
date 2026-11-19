# aclnnBlockAttnResUpdate

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

- Accumulates `delta` into `partialBlockRef` in place, computes the normalized score of the updated residual, and combines it with the historical online Softmax state to produce `h`.

- The [V2 API](aclnnBlockAttnResUpdateV2_en.md) adds optional output RMSNorm. Passing `nullptr` as `outputNormWeightOptional` to V2 retains this API's computation. `outputNormEps` is ignored in that case.

- Computation

  For each token $t$, update the partial residual and write it back in place.

  $$
  p_t=P_t+\operatorname{FP32}(\Delta_t),\qquad
  P^{\mathrm{out}}_t=p_t
  $$

  Compute the current score using root mean square normalization (RMSNorm) and a query dot product.

  $$
  C_t=\left\langle Q,\operatorname{RMSNorm}(p_t,\mathbf{1},\epsilon)\right\rangle
  $$

  Merge the current residual with the historical online Softmax state.

  $$
  m_t=\max(M_t,C_t),\qquad
  \alpha_t=\exp(M_t-m_t),\qquad
  \beta_t=\exp(C_t-m_t)
  $$

  $$
  H_t=\operatorname{BF16}\left(
  \frac{\alpha_t O_t+\beta_t p_t}{\alpha_t L_t+\beta_t}
  \right)
  $$

  - $T$ and $D$ are the token count and hidden dimension, with $t\in[0,T)$. The residuals, query, numerator, and output are $D$-dimensional vectors.
  - $P$ and $P^{\mathrm{out}}$ are the values of `partialBlockRef` before and after the call. $\Delta$ is `delta`, $Q$ is `pseudoQuery`, and $p$ is the FLOAT32 residual sum.
  - $O$, $M$, and $L$ are `numerator`, `logitMax`, and `expSum`. $H$ is `h`.
  - $\epsilon=\operatorname{FP32}(\texttt{eps})$. RMSNorm operates along $D$ and uses an all-ones scale vector for scoring. The scoring scale can be multiplied elementwise into the original query in advance to form $Q$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call
`aclnnBlockAttnResUpdateGetWorkspaceSize` first to obtain the workspace size and executor, and then call
`aclnnBlockAttnResUpdate` to perform the computation.

```cpp
aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(
  aclTensor        *partialBlockRef,
  const aclTensor  *delta,
  const aclTensor  *pseudoQuery,
  const aclTensor  *numerator,
  const aclTensor  *logitMax,
  const aclTensor  *expSum,
  double            eps,
  aclTensor        *h,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnBlockAttnResUpdate(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBlockAttnResUpdateGetWorkspaceSize

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
        <td>partialBlockRef (aclTensor*)</td>
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
        <td>delta (const aclTensor*)</td>
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
        <td>pseudoQuery (const aclTensor*)</td>
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
        <td>numerator (const aclTensor*)</td>
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
        <td>logitMax (const aclTensor*)</td>
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
        <td>expSum (const aclTensor*)</td>
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
        <td>eps (double)</td>
        <td>Input</td>
        <td>The numerical stability term for the RMS score of the current residual; ε in the formulas.</td>
        <td>Must be finite and positive. For non-empty workloads, the value after conversion to FLOAT32 must also be finite and greater than 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>h (aclTensor*)</td>
        <td>Output</td>
        <td>Attention output for the current layer, corresponding to H in the formula.</td>
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
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to allocate on the device, in bytes.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
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

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
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
        <td><code>partialBlockRef</code>, <code>delta</code>, <code>pseudoQuery</code>, <code>numerator</code>, <code>logitMax</code>, <code>expSum</code>, <code>h</code>, <code>workspaceSize</code>, or <code>executor</code> is null.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>
          <ul>
            <li>The dtype, original or storage format, dimension, or shape relationship of the input or output does not meet the constraints.</li>
            <li><code>T &lt; 0</code>, or <code>D</code> is not in the range <code>[0, 8192]</code>.</li>
            <li>In non-empty cases, <code>T * D</code> exceeds the range representable by <code>int64_t</code>.</li>
            <li>Any input or output Tensor is a non-contiguous Tensor.</li>
            <li><code>eps</code> is not a finite positive number.</li>
          </ul>
        </td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>The product model is not in the supported range.</td>
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

## aclnnBlockAttnResUpdate

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
        <td>workspace (void*)</td>
        <td>Input</td>
        <td>Device workspace address. May be <code>nullptr</code> when <code>workspaceSize</code> is zero.</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t)</td>
        <td>Input</td>
        <td>Workspace size applied for on the Device side, obtained by calling the first-phase API aclnnBlockAttnResUpdateGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor*)</td>
        <td>Input</td>
        <td>Executor of the op, containing the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream (aclrtStream)</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Empty tensors must satisfy the shape relationships in the parameter table. Required tensors must be supplied as tensor objects.
- For non-empty workloads, `T * D` must fit in `int64_t`, and the FLOAT32 conversion of `eps` must be finite and positive. The per-core token count must fit in `uint32_t`, and the data for the full hidden dimension must fit in the unified buffer (UB).
- For each token in a non-empty workload, `numerator` and `logitMax` must contain finite values, and `expSum` must contain finite, strictly positive values. All three inputs must describe the same non-empty historical online Softmax state. Host validation does not read their elements.
- Deterministic computation

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnBlockAttnResUpdate uses a deterministic implementation by default.
  <!-- end id7 -->

## Examples

The following sample code is for reference only. For details about the compilation and execution process, see
[Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md). The complete sample source code is in
[test_aclnn_block_attn_res_update.cpp](../examples/arch35/test_aclnn_block_attn_res_update.cpp).

```cpp
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_update.h"

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

int RunBlockAttnResUpdate(int32_t deviceId, aclrtStream& stream)
{
    // 1. Initialize the Device and Stream.
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output data.
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

    // 3. Create the input and output aclTensors.
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

    constexpr double eps = 1.0e-6;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // 4. Call the first-phase API to obtain the workspace size and executor.
    ret = aclnnBlockAttnResUpdateGetWorkspaceSize(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, eps,
                                                  h, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResUpdateGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        // Allocate the device-side workspace based on the size returned by the first-phase API.
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 5. Call the second-phase API to execute the operator.
    ret = aclnnBlockAttnResUpdate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResUpdate failed. ERROR: %d\n", ret); return ret);

    // 6. Synchronously wait for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. Copy the in-place update result and output result back to the host side.
    const size_t partialBlockRefSize = partialBlockRefHostData.size() * sizeof(float);
    ret = aclrtMemcpy(partialBlockRefHostData.data(), partialBlockRefSize, partialBlockRefDeviceAddr,
                      partialBlockRefSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy partialBlockRef to host failed. ERROR: %d\n", ret); return ret);

    const size_t hSize = hHostData.size() * sizeof(uint16_t);
    ret = aclrtMemcpy(hHostData.data(), hSize, hDeviceAddr, hSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy h to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("partialBlockRef[0] after in-place update: %.6f\n", partialBlockRefHostData[FIRST_ELEMENT_INDEX]);
    LOG_PRINT("h[0]: %.6f\n", Bfloat16ToFloat(hHostData[FIRST_ELEMENT_INDEX]));
    LOG_PRINT("block_attn_res_update example execute success.\n");
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = RunBlockAttnResUpdate(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBlockAttnResUpdate failed. ERROR: %d\n", ret); return ret);

    // 8. Release Device and Stream resources.
    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
