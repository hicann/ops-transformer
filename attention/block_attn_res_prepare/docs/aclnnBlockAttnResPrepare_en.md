# aclnnBlockAttnResPrepare

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:00:45.318Z -->

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

- Description: Phase one of the two-phase computation of historical residual attention for Attention Residuals: completes the inter-block attention computation for all S layers in parallel at one time; returns the output and softmax statistics (O, M, L) for phase two to fuse the current residual and form the input of the next sub-layer.
- Formula:

    $$
    \operatorname{block\_attn\_res\_prepare}
    \left(V,Q,\operatorname{valid\_blocks};\epsilon\right)
    \longrightarrow (O,M,L).
    $$

     For the $n$-th historical source of token $t$, first compute the RMS normalization factor:

    $$
    R_{t,n}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}V_{t,n,d}^{2}+\epsilon}.
    $$

     Compute logits for each slot, token, and valid historical source:

    $$
    Z_{s,t,n}=R_{t,n}^{-1}\sum_{d=0}^{D-1}Q_{s,d}V_{t,n,d}.
    $$

     Let $N_v$ be the number of valid historical sources specified by `valid_blocks`. Then:

    $$
    M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},
    $$

    $$
    E_{s,t,n}=\exp\left(Z_{s,t,n}-M_{s,t}\right),
    $$

    $$
    L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},
    $$

    $$
    O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}.
    $$

     When `validBlocks[0] == 0`, the preceding softmax computation is not performed, and the result is returned directly:

    $$
    O=\mathbf{0}_{S\times T\times D},\qquad
    M=m_{\min}\mathbf{1}_{S\times T},\qquad
    L=\mathbf{0}_{S\times T},\qquad
    m_{\min}=-3.4028234663852886\times10^{38}.
    $$

      That is, `numerator` is an all-zero tensor of shape `[S, T, D]`, `logitMax` is a tensor of shape `[S, T]` with all elements set to the minimum finite value of FLOAT32, and `expSum` is an all-zero tensor of shape `[S, T]`.

  Where:
    - $V\in\mathbb{R}^{T\times N\times D}$ denotes the input `blockRes`; $V_{t,n,d}$ denotes the value of the $t$-th token and the $n$-th historical residual block at the $d$-th hidden feature.
    - $Q\in\mathbb{R}^{S\times D}$ denotes the input `pseudoQuery` (pseudo query); each target slot corresponds to a query vector, and $Q_{s,d}$ denotes the value of the $s$-th target slot at the $d$-th hidden feature.
    - $N_v=\min(\texttt{validBlocks[0]},N)$, which denotes the number of historical residual blocks actually involved in the computation.
    - $\epsilon$ denotes the input `eps`, a numerical stability term used for computing the RMS normalization factor.
    - $T$ denotes the number of tokens, $N$ denotes the number of historical residual blocks that can be accommodated in dimension 1 of `blockRes`, $S$ denotes the number of target slots, and $D$ denotes HiddenSize.
    - $t\in[0,T)$, $n\in[0,N_v)$, $s\in[0,S)$, and $d\in[0,D)$ denote the indices of the token, valid historical residual block, target slot, and hidden feature, respectively.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnBlockAttnResPrepareGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnBlockAttnResPrepare" API to execute the computation.

```cpp
aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(
    const aclTensor *blockRes,
    const aclTensor *validBlocks,
    const aclTensor *pseudoQuery,
    aclTensor       *numerator,
    aclTensor       *logitMax,
    aclTensor       *expSum,
    double           eps,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnBlockAttnResPrepare(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnBlockAttnResPrepareGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1503px"><colgroup>
  <col style="width: 130px">
  <col style="width: 97px">
  <col style="width: 308px">
  <col style="width: 488px">
  <col style="width: 197px">
  <col style="width: 77px">
  <col style="width: 115px">
  <col style="width: 95px">
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
      <td>blockRes (const aclTensor*)</td>
      <td>Input</td>
      <td>Represents the residual block, corresponding to input V in the formula.</td>
      <td><li>Empty tensors are supported, and T can be 0.</li><li>The shape is [T, N, D].</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>validBlocks (const aclTensor*)</td>
      <td>Input</td>
      <td>Represents the current number of valid history blocks.</td>
      <td><li>Empty tensors are not supported; shape is [1].</li><li>When the value in the tensor is N_v, only the first N_v values along the N dimension of blockRes participate in the computation. If the value is 0, numerator returns an all-zero tensor, logitMax returns a tensor with all elements set to the minimum finite value of FLOAT32, and expSum returns an all-zero tensor; if the value is greater than N, it is treated as N.</li></td>
      <td>UINT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>pseudoQuery (const aclTensor*)</td>
      <td>Input</td>
      <td>Pseudo query, corresponding to input Q in the formula. Each target slot corresponds to one query vector.</td>
      <td><li>Empty tensors are supported; S can be 0.</li><li>shape is [S, D], and the last dimension must be the same as that of blockRes.</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numerator (aclTensor*)</td>
      <td>Output</td>
      <td>Denotes the softmax weighted numerator, corresponding to output O in the formula.</td>
      <td><li>Supports empty tensor.</li><li>The shape is [S, T, D].</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>logitMax (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the softmax maximum value, corresponding to output M in the formula.</td>
      <td><li>Empty tensors are supported.</li><li>The shape is [S, T].</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expSum (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the sum of softmax exponents, corresponding to output L in the formula.</td>
      <td><li>Supports empty tensors.</li><li>shape is [S, T].</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Represents the stabilization term for RMS normalization, typically 1e-6.</td>
      <td>Computed internally in FLOAT32. Must be finite and in `[FLT_MIN, FLT_MAX]` (approximately `[1.1754943508222875e-38, 3.4028234663852886e+38]`). Subnormal values and values that narrow to zero or Inf are not supported.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
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
- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs parameter validation and executor creation, and may return the following status codes:

  | Return Value                       | Error Code | Description                                                                                                                           |
  | ----------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
  | `ACLNN_SUCCESS`                    | 0      | The API call succeeds.                                                                                                                        |
  | `ACLNN_ERR_PARAM_NULLPTR`          | 161001 | One of `blockRes`, `validBlocks`, `pseudoQuery`, `numerator`, `logitMax`, `expSum`, `workspaceSize`, or `executor` is a null pointer. |
  | `ACLNN_ERR_PARAM_INVALID`          | 161002 | The data type, dimension, shape, or contiguity of the input or output does not meet the constraints, or `eps` is non-finite or outside the positive normal FLOAT32 range above.                                                      |
  | `ACLNN_ERR_RUNTIME_ERROR`          | 361001 | The API runs on an unsupported product model.                                                                                                      |
  | `ACLNN_ERR_INNER_CREATE_EXECUTOR`  | 561101 | The API fails to create `aclOpExecutor` internally.                                                                                                  |
  | `ACLNN_ERR_INNER_NULLPTR`          | 561103 | A null pointer is returned when the API internally performs contiguity processing, creates the operator output, or creates the output copy node.                                                                  |

## aclnnBlockAttnResPrepare

- **Parameters**

  <table style="table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnBlockAttnResPrepareGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor*)</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream (aclrtStream)</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>
- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The second-phase API calls `CommonOpExecutorRun` to dispatch the operator. On success, it returns `ACLNN_SUCCESS` (error code 0); on failure, it returns the corresponding ACLNN Runtime or internal error code.

## Constraints

- Deterministic computation: aclnnBlockAttnResPrepare defaults to a deterministic implementation.
- Batch consistency: aclnnBlockAttnResPrepare defaults to a batch consistency implementation.
- `T >= 0`, `S >= 0`, `1 <= N <= 64`, `1 <= D <= 8192`.
- The last dimension of `blockRes` must be the same as that of `pseudoQuery`.
- If `validBlocks[0] == 0`, the empty state of online softmax is returned; if `validBlocks[0] > N`, it is processed as `N`.
- All input and output tensors must be contiguous tensors. Non-contiguous tensors are not supported.
- If `T == 0` or `S == 0`, an empty output of the corresponding shape is returned.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_prepare.h"

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
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

constexpr int64_t ROW_MAJOR_STRIDE_START_OFFSET = 2;
constexpr float DEFAULT_EPS = 1.0e-6F;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (const auto dim : shape) {
        shapeSize *= dim;
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
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - ROW_MAJOR_STRIDE_START_OFFSET; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnBlockAttnResPrepareTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
              return ret);

    // Construct the inputs and outputs: 1 token, 2 history sources, 2 target slots, and a hidden size of 4.
    const std::vector<int64_t> blockResShape = {1, 2, 4};
    const std::vector<int64_t> validBlocksShape = {1};
    const std::vector<int64_t> pseudoQueryShape = {2, 4};
    const std::vector<int64_t> numeratorShape = {2, 1, 4};
    const std::vector<int64_t> statsShape = {2, 1};

    const std::vector<float> blockResHostData = {
        1.0F, 2.0F, 3.0F, 4.0F, 2.0F, 0.0F, -1.0F, 1.0F,
    };
    const std::vector<uint64_t> validBlocksHostData = {2U};
    const std::vector<float> pseudoQueryHostData = {
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
    };
    std::vector<float> numeratorHostData(GetShapeSize(numeratorShape), 0.0F);
    std::vector<float> logitMaxHostData(GetShapeSize(statsShape), 0.0F);
    std::vector<float> expSumHostData(GetShapeSize(statsShape), 0.0F);

    void *blockResDeviceAddr = nullptr;
    void *validBlocksDeviceAddr = nullptr;
    void *pseudoQueryDeviceAddr = nullptr;
    void *numeratorDeviceAddr = nullptr;
    void *logitMaxDeviceAddr = nullptr;
    void *expSumDeviceAddr = nullptr;
    aclTensor *blockRes = nullptr;
    aclTensor *validBlocks = nullptr;
    aclTensor *pseudoQuery = nullptr;
    aclTensor *numerator = nullptr;
    aclTensor *logitMax = nullptr;
    aclTensor *expSum = nullptr;

    ret = CreateAclTensor(blockResHostData, blockResShape, &blockResDeviceAddr, aclDataType::ACL_FLOAT, &blockRes);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> blockResTensorPtr(blockRes, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> blockResDeviceAddrPtr(blockResDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(validBlocksHostData, validBlocksShape, &validBlocksDeviceAddr, aclDataType::ACL_UINT64,
                          &validBlocks);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> validBlocksTensorPtr(validBlocks, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> validBlocksDeviceAddrPtr(validBlocksDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(pseudoQueryHostData, pseudoQueryShape, &pseudoQueryDeviceAddr, aclDataType::ACL_FLOAT,
                          &pseudoQuery);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pseudoQueryTensorPtr(pseudoQuery, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> pseudoQueryDeviceAddrPtr(pseudoQueryDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(numeratorHostData, numeratorShape, &numeratorDeviceAddr, aclDataType::ACL_FLOAT, &numerator);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> numeratorTensorPtr(numerator, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> numeratorDeviceAddrPtr(numeratorDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(logitMaxHostData, statsShape, &logitMaxDeviceAddr, aclDataType::ACL_FLOAT, &logitMax);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitMaxTensorPtr(logitMax, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> logitMaxDeviceAddrPtr(logitMaxDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(expSumHostData, statsShape, &expSumDeviceAddr, aclDataType::ACL_FLOAT, &expSum);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> expSumTensorPtr(expSum, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> expSumDeviceAddrPtr(expSumDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBlockAttnResPrepareGetWorkspaceSize(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum,
                                                   DEFAULT_EPS, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBlockAttnResPrepareGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnBlockAttnResPrepare(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBlockAttnResPrepare failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtMemcpy(numeratorHostData.data(), numeratorHostData.size() * sizeof(numeratorHostData[0]),
                      numeratorDeviceAddr, numeratorHostData.size() * sizeof(numeratorHostData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy numerator from device to host failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(logitMaxHostData.data(), logitMaxHostData.size() * sizeof(logitMaxHostData[0]),
                      logitMaxDeviceAddr, logitMaxHostData.size() * sizeof(logitMaxHostData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy logitMax from device to host failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(expSumHostData.data(), expSumHostData.size() * sizeof(expSumHostData[0]), expSumDeviceAddr,
                      expSumHostData.size() * sizeof(expSumHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy expSum from device to host failed. ERROR: %d\n", ret);
              return ret);

    for (size_t i = 0; i < numeratorHostData.size(); ++i) {
        LOG_PRINT("numerator[%zu] = %.6f\n", i, numeratorHostData[i]);
    }
    for (size_t i = 0; i < logitMaxHostData.size(); ++i) {
        LOG_PRINT("logit_max[%zu] = %.6f, exp_sum[%zu] = %.6f\n", i, logitMaxHostData[i], i, expSumHostData[i]);
    }
    return ACL_SUCCESS;
}

} // namespace

int main()
{
    // Set the Device ID based on the actual running environment.
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    const auto ret = aclnnBlockAttnResPrepareTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnBlockAttnResPrepareTest failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}

```
