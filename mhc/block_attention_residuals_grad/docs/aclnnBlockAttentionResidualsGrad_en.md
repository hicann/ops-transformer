# aclnnBlockAttentionResidualsGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:44:28.274Z -->

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

- Description: `BlockAttentionResidualsGrad` is the backward propagation operator of the forward operator `BlockAttentionResiduals` (attention residual). Based on the invNorm and probs saved during forward propagation, the forward parameters projWeight and normWeight, and gradHiddenStates, this API computes the gradients of partialBlock, blockRes, projWeight, and normWeight.

- Formula:

  $$
  g_{t,i} = \sum_{h=0}^{H-1} grad\_output_{t,h} \cdot v_{t,i,h}
  $$

  $$
  grad\_score_{t,i} = probs_{t,i} \cdot \left(g_{t,i} - \sum_{j=0}^{N} probs_{t,j} \cdot g_{t,j}\right)
  $$

  $$
  grad\_k_{t,i,h} = grad\_score_{t,i} \cdot score\_weight_{h}
  $$

  $$
  grad\_score\_weight_{h} = \sum_{t=0}^{T-1} \sum_{i=0}^{N} grad\_score_{t,i} \cdot k_{t,i,h}
  $$

  $$
  grad\_inv\_rms_{t,i} = \sum_{h=0}^{H-1} grad\_k_{t,i,h} \cdot v_{t,i,h}
  $$

  $$
  grad\_v_{t,i,h} = grad\_output_{t,h} \cdot probs_{t,i} + grad\_k_{t,i,h} \cdot inv\_rms_{t,i} - \frac{grad\_inv\_rms_{t,i} \cdot inv\_rms_{t,i}^{3}}{H} \cdot v_{t,i,h}
  $$

  Where:

  $$
  grad\_block\_res_{t,i,h} = grad\_v_{t,i,h}, \quad i < N
  $$

  $$
  grad\_partial\_block_{t,h} = grad\_v_{t,N,h}
  $$

  $$
  grad\_norm\_weight_{h} = grad\_score\_weight_{h} \cdot proj\_weight_{0,h}
  $$

  $$
  grad\_proj\_weight_{0,h} = grad\_score\_weight_{h} \cdot norm\_weight_{h}
  $$

  The current version directly uses the probs saved in the forward pass for backward computation and does not reconstruct the mask based on validBlockNum.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnBlockAttentionResidualsGradGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnBlockAttentionResidualsGrad" API to perform the computation.

```c++
aclnnStatus aclnnBlockAttentionResidualsGradGetWorkspaceSize(
    const aclTensor *partialBlock,
    const aclTensor *blockRes,
    const aclTensor *projWeight,
    const aclTensor *normWeight,
    const aclTensor *gradHiddenStates,
    const aclTensor *invNorm,
    const aclTensor *probs,
    int64_t          validBlockNum,
    const aclTensor *gradPartialBlock,
    const aclTensor *gradBlockRes,
    const aclTensor *gradProjWeight,
    const aclTensor *gradNormWeight,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor);
```

```c++
aclnnStatus aclnnBlockAttentionResidualsGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnBlockAttentionResidualsGradGetWorkspaceSize

- **Parameters**
  <table style="table-layout: fixed; width: 1500px">
          <colgroup>
              <col style="width: 220px">
              <col style="width: 120px">
              <col style="width: 300px">
              <col style="width: 350px">
              <col style="width: 210px">
              <col style="width: 100px">
              <col style="width: 100px">
              <col style="width: 100px">
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
              <td>partialBlock (aclTensor*)</td>
              <td>Input</td>
              <td>Forward input prefix sum, corresponding to the (N+1)th value in the formula.</td>
              <td><li>The data type must remain consistent with that of the other inputs.</li></td>
              <td>FLOAT16, BFLOAT16, FLOAT32</td>
              <td>ND</td>
              <td>(T,H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>blockRes (aclTensor*)</td>
              <td>Input</td>
              <td>Forward input block residual, corresponding to the first N values in the formula.</td>
              <td><li>The data type must remain consistent with that of partialBlock.</li></td>
              <td>FLOAT16, BFLOAT16, FLOAT32</td>
              <td>ND</td>
              <td>(T,N,H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>projWeight (aclTensor*)</td>
              <td>Input</td>
              <td>Forward linear projection weight. Together with normWeight, it forms score_weight.</td>
              <td><li>The data type must remain consistent with that of partialBlock.</li></td>
              <td>FLOAT16, BFLOAT16, FLOAT32</td>
              <td>ND</td>
              <td>(1,H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>normWeight(aclTensor*)</td>
              <td>Input</td>
              <td>Forward normalization weight, forming score_weight together with projWeight.</td>
              <td><li>Data type must remain consistent with partialBlock.</li></td>
              <td>FLOAT16, BFLOAT16, FLOAT32</td>
              <td>ND</td>
              <td>(H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>validBlockNum (int64_t)</td>
              <td>Attribute</td>
              <td>Reserved attribute, currently unused in computation.</td>
              <td><li>Supports -1 (all N blocks) or N (blockRes.shape[1]).</li></td>
              <td>INT64</td>
              <td>-</td>
              <td>Scalar</td>
              <td>-</td>
          </tr>
          <tr>
              <td>gradHiddenStates (aclTensor*)</td>
              <td>Input</td>
              <td>Gradient of the forward output out.</td>
              <td><li>Data type must remain consistent with partialBlock.</li></td>
              <td>FLOAT16, BFLOAT16, FLOAT32</td>
              <td>ND</td>
              <td>(T,H)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>invNorm (aclTensor*)</td>
              <td>Input</td>
              <td>Row-wise normalization coefficients saved in the forward pass.</td>
              <td><li>Only FLOAT32 is supported.</li></td>
              <td>FLOAT32</td>
              <td>ND</td>
              <td>(T,N+1)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>probs (aclTensor*)</td>
              <td>Input</td>
              <td>Forward softmax output probabilities.</td>
              <td><li>Only FLOAT32 is supported.</li></td>
              <td>FLOAT32</td>
              <td>ND</td>
              <td>(T,N+1)</td>
              <td>√</td>
          </tr>
          <tr>
              <td>gradPartialBlock (aclTensor*)</td>
              <td>Output</td>
              <td>Gradient of partialBlock.</td>
              <td><li>Data type and shape remain consistent with partialBlock.</li></td>
              <td>FLOAT16, BFLOAT16, FLOAT32</td>
              <td>ND</td>
              <td>(T,H)</td>
              <td>-</td>
          </tr>
          <tr>
              <td>gradBlockRes (aclTensor*)</td>
              <td>Output</td>
              <td>Gradient of blockRes.</td>
              <td><li>Data type and shape must remain consistent with blockRes.</li></td>
              <td>FLOAT16, BFLOAT16, FLOAT32</td>
              <td>ND</td>
              <td>(T,N,H)</td>
              <td>-</td>
          </tr>
          <tr>
              <td>gradProjWeight (aclTensor*)</td>
              <td>Output</td>
              <td>Gradient of projWeight.</td>
              <td><li>The data type and shape remain consistent with projWeight.</li></td>
              <td>FLOAT16, BFLOAT16, FLOAT32</td>
              <td>ND</td>
              <td>(1,H)</td>
              <td>-</td>
          </tr>
          <tr>
              <td>gradNormWeight(aclTensor*)</td>
              <td>Output</td>
              <td>Gradient of normWeight.</td>
              <td><li>The data type and shape remain consistent with normWeight.</li></td>
              <td>FLOAT16, BFLOAT16, FLOAT32</td>
              <td>ND</td>
              <td>(H)</td>
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
              <td>Returns the op executor that contains the operator computation flow.</td>
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

  <table style="table-layout: fixed; width: 1000px"><colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
      </colgroup>
          <thead>
              <th>Return Value</th>
              <th>Error Code</th>
              <th>Description</th>
          </thead>
          <tbody>
              <tr>
                  <td>ACLNN_ERR_PARAM_NULLPTR</td>
                  <td>161001</td>
                  <td>partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs, or an output tensor is a null pointer.</td>
              </tr>
              <tr>
                  <td>ACLNN_ERR_PARAM_INVALID</td>
                  <td>161002</td>
                  <td>The data type, data format, or shape of the input tensor is out of the supported range.</td>
              </tr>
              <tr>
                  <td>ACLNN_ERR_RUNTIME_ERROR</td>
                  <td>361001</td>
                  <td>An exception occurs when the API calls the NPU runtime API for memory operations.</td>
              </tr>
          </tbody>
      </table>

## aclnnBlockAttentionResidualsGrad

- **Parameters**
  <table style="table-layout: fixed; width: 1000px"><colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 700px">
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
          <td>Workspace memory address allocated on the Device.</td>
          </tr>
          <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnBlockAttentionResidualsGradGetWorkspaceSize.</td>
          </tr>
          <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Op executor, containing the computation flow of the operator.</td>
          </tr>
          <tr>
          <td>stream</td>
          <td>Input</td>
          <td>Specifies the stream for task execution.</td>
          </tr>
      </tbody>
      </table>

- **Return Value**
  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnBlockAttentionResidualsGrad defaults to a deterministic implementation.
- The data types of partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, and their corresponding outputs must remain consistent. FLOAT16, BFLOAT16, and FLOAT32 are supported.
- validBlockNum is a reserved INT64 attribute with a default value of -1. Only -1 or N (blockRes.shape[1]) is supported; other values return a parameter error.
- Only the ND format is supported for partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, and probs.
- The shapes must meet the following requirements: partialBlock is (T,H), blockRes is (T,N,H), projWeight is (1,H), normWeight is (H), gradHiddenStates is (T,H), invNorm is (T,N+1), and probs is (T,N+1). T and H are non-negative integers, and N is in the range [0,128]. T and H must remain consistent across all tensors, and N+1 must remain consistent between invNorm and probs.
- Non-contiguous input tensors are supported. The API internally converts them to the Contiguous format before computation.
- probs must be the forward softmax output, and invNorm must be the forward row-wise normalization coefficient. Only FLOAT32 is supported for invNorm and probs. The current implementation relies entirely on the saved probs and invNorm.
- When T=0 or H=0, the aclnn layer skips the main operator. Non-empty weight gradient outputs are zeroed, while empty outputs retain the shapes of the corresponding inputs. When H=0, all outputs are empty and no zeroing tasks are scheduled. For T=0 and H>0, the zeroing tasks must still be executed by calling the second-phase API.
- Only when N=0 and both T and H are non-zero, the API does not return early; partialBlock and the weight gradients are still computed.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_block_attention_residuals_grad.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t T  = 2;
    const int64_t N  = 4;
    const int64_t H  = 64;
    const int64_t N1 = N + 1;

    std::vector<int64_t> partialBlockShape      = {T, H};
    std::vector<int64_t> blockResShape  = {T, N, H};
    std::vector<int64_t> projWeightShape     = {1, H};
    std::vector<int64_t> normWeightShape     = {H};
    std::vector<int64_t> gradHiddenStatesShape     = {T, H};
    std::vector<int64_t> invNormShape         = {T, N1};
    std::vector<int64_t> probsShape          = {T, N1};

    std::vector<uint16_t> partialBlockData     (GetShapeSize(partialBlockShape),     0x3C00);
    std::vector<uint16_t> blockResData (GetShapeSize(blockResShape),  0x3C00);
    std::vector<uint16_t> projWeightData    (GetShapeSize(projWeightShape),     0x3C00);
    std::vector<uint16_t> normWeightData    (GetShapeSize(normWeightShape),     0x3C00);
    std::vector<uint16_t> gradHiddenStatesData    (GetShapeSize(gradHiddenStatesShape),     0x3C00);
    std::vector<float> invNormData           (GetShapeSize(invNormShape),         1.0f);
    std::vector<float> probsData            (GetShapeSize(probsShape),          1.0f);
    int64_t validBlockNum = -1; // Reserved attribute: -1 or N (blockRes.shape[1]).

    aclTensor *partialBlock = nullptr, *blockRes = nullptr, *projWeight = nullptr, *normWeight = nullptr;
    aclTensor *gradHiddenStates = nullptr, *invNorm = nullptr, *probs = nullptr;
    void *d_partialBlock = nullptr, *d_blockRes = nullptr, *d_projWeight = nullptr, *d_normWeight = nullptr;
    void *d_gradHiddenStates = nullptr, *d_invNorm = nullptr, *d_probs = nullptr;

    ret = CreateAclTensor(partialBlockData, partialBlockShape, &d_partialBlock, aclDataType::ACL_FLOAT16, &partialBlock); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockResData, blockResShape, &d_blockRes, aclDataType::ACL_FLOAT16, &blockRes); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(projWeightData, projWeightShape, &d_projWeight, aclDataType::ACL_FLOAT16, &projWeight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normWeightData, normWeightShape, &d_normWeight, aclDataType::ACL_FLOAT16, &normWeight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHiddenStatesData, gradHiddenStatesShape, &d_gradHiddenStates, aclDataType::ACL_FLOAT16, &gradHiddenStates); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(invNormData, invNormShape, &d_invNorm, aclDataType::ACL_FLOAT, &invNorm); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsData, probsShape, &d_probs, aclDataType::ACL_FLOAT, &probs); CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *gradPartialBlock = nullptr, *gradBlockRes = nullptr, *gradProjWeight = nullptr, *gradNormWeight = nullptr;
    void *d_gradPartialBlock = nullptr, *d_gradBlockRes = nullptr, *d_gradProjWeight = nullptr, *d_gradNormWeight = nullptr;

    std::vector<uint16_t> gradPartialBlockData    (GetShapeSize(partialBlockShape),     0);
    std::vector<uint16_t> gradBlockResData(GetShapeSize(blockResShape), 0);
    std::vector<uint16_t> gradProjWeightData   (GetShapeSize(projWeightShape),    0);
    std::vector<uint16_t> gradNormWeightData   (GetShapeSize(normWeightShape),    0);

    ret = CreateAclTensor(gradPartialBlockData, partialBlockShape, &d_gradPartialBlock, aclDataType::ACL_FLOAT16, &gradPartialBlock); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradBlockResData, blockResShape, &d_gradBlockRes, aclDataType::ACL_FLOAT16, &gradBlockRes); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradProjWeightData, projWeightShape, &d_gradProjWeight, aclDataType::ACL_FLOAT16, &gradProjWeight); CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradNormWeightData, normWeightShape, &d_gradNormWeight, aclDataType::ACL_FLOAT16, &gradNormWeight); CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnBlockAttentionResidualsGradGetWorkspaceSize(
        partialBlock, blockRes, projWeight, normWeight,
        gradHiddenStates, invNorm, probs, validBlockNum,
        gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnBlockAttentionResidualsGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("BlockAttentionResidualsGrad example ran successfully!\n");

    aclDestroyTensor(partialBlock); aclDestroyTensor(blockRes); aclDestroyTensor(projWeight);
    aclDestroyTensor(normWeight); aclDestroyTensor(gradHiddenStates); aclDestroyTensor(invNorm);
    aclDestroyTensor(probs);
    aclrtFree(d_partialBlock); aclrtFree(d_blockRes); aclrtFree(d_projWeight);
    aclrtFree(d_normWeight); aclrtFree(d_gradHiddenStates); aclrtFree(d_invNorm);
    aclrtFree(d_probs);

    aclDestroyTensor(gradPartialBlock); aclDestroyTensor(gradBlockRes); aclDestroyTensor(gradProjWeight);
    aclDestroyTensor(gradNormWeight);
    aclrtFree(d_gradPartialBlock); aclrtFree(d_gradBlockRes); aclrtFree(d_gradProjWeight);
    aclrtFree(d_gradNormWeight);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
