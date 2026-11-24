# aclnnBlockAttentionResiduals

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:41:47.565Z -->

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

- Description: Concatenates `partialBlock` and `blockRes`, then applies RMS normalization, linear projection, and softmax weighted fusion to output `hiddenStates`. When `needBackward` is true, `invNorm` and `probs` required for backward computation are also output.

- Formula:

  The forward computation can be expressed as:

  $$
  v_{t,i,h} =
  \begin{cases}
  block\_res_{t,i,h}, & i < N \\
  partial\_block_{t,h}, & i = N
  \end{cases}
  $$

  $$
  variance_{t,i} = \frac{1}{H}\sum_{h=0}^{H-1} v_{t,i,h}^{2}
  $$

  $$
  inv\_rms_{t,i} = \frac{1}{\sqrt{variance_{t,i} + norm\_eps}}
  $$

  $$
  k_{t,i,h} = v_{t,i,h} \cdot inv\_rms_{t,i}
  $$

  $$
  score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
  $$

  $$
  s_{t,i} = \sum_{h=0}^{H-1} k_{t,i,h} \cdot score\_weight_{h}
  $$

  $$
  probs_{t,i} = \frac{e^{s_{t,i}}}{\sum_{j=0}^{N} e^{s_{t,j}}}
  $$

  $$
  hidden\_states_{t,h} = \sum_{i=0}^{N} probs_{t,i} \cdot v_{t,i,h}
  $$

  Where:

  - $T$ indicates the number of tokens, $N$ indicates the number of blocks in `blockRes`, and $H$ indicates the hidden size.
  - $v_{t,i,h}$ indicates the value after concatenating `blockRes` and `partialBlock`, with the shape of $(T,N+1,H)$.
  - $inv\_rms_{t,i}$ denotes the forward row-wise normalization coefficient.
  - $score\_weight_{h}$ denotes the element-wise product of `normWeight` and `projWeight`.
  - $probs_{t,i}$ denotes the forward softmax output probability.
  - $hidden\_states_{t,h}$ denotes the forward weighted fusion output.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnBlockAttentionResidualsGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process. Then call the "aclnnBlockAttentionResiduals" API to perform the computation.

```Cpp
aclnnStatus aclnnBlockAttentionResidualsGetWorkspaceSize(
    const aclTensor *partialBlock,
    const aclTensor *blockRes,
    const aclTensor *projWeight,
    const aclTensor *normWeight,
    int64_t          validBlockNum,
    double           normEps,
    bool             needBackward,
    aclTensor       *hiddenStates,
    aclTensor       *invNorm,
    aclTensor       *probs,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBlockAttentionResiduals(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnBlockAttentionResidualsGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 387px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 146px">
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
      <td>Forward input partialBlock, the (N+1)th value after concatenation in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors, and T can be 0.</li><li>The shape is (T, H).</li><li>T and H must keep consistent with blockRes.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
  </tr>
  <tr>
      <td>blockRes (aclTensor*)</td>
      <td>Input</td>
      <td>Forward input block residual, the first N values in the corresponding formula.</td>
      <td><ul><li>Empty tensors are supported, and T can be 0.</li><li>The shape is (T, N, H).</li><li>The data type must keep consistent with partialBlock.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
  </tr>
  <tr>
      <td>projWeight (aclTensor*)</td>
      <td>Input</td>
      <td>Forward linear projection weight. It forms score_weight together with normWeight.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is (H) or (1, H).</li><li>The data type must keep consistent with partialBlock.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
  </tr>
  <tr>
      <td>normWeight (aclTensor*)</td>
      <td>Input</td>
      <td>Forward normalization weight. Together with projWeight, it forms score_weight.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is (H).</li><li>The data type must keep consistent with that of partialBlock.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
  </tr>
  <tr>
      <td>validBlockNum (int64_t)</td>
      <td>Input</td>
      <td>Reserved attribute, which does not participate in computation currently.</td>
      <td><ul><li>Default value: -1.</li><li>Only -1 is supported.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>normEps (double)</td>
      <td>Optional input</td>
      <td>Numerical stability parameter of RMS normalization, corresponding to norm_eps in the corresponding formula.</td>
      <td><ul><li>Must be a finite positive number.</li><li>Default value: 1e-6.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>needBackward (bool)</td>
      <td>Optional input</td>
      <td>Whether to save invNorm and probs for backward propagation.</td>
      <td><ul><li>Default value: false.</li><li>If true, invNorm and probs cannot be null pointers.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hiddenStates (aclTensor*)</td>
      <td>Output</td>
      <td>Weighted fusion output of the forward computation, corresponding to hidden_states in the formula.</td>
      <td><ul><li>Empty tensors are supported, and T can be 0.</li><li>Shape: (T, H).</li><li>The data type must keep consistent with partialBlock.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
  </tr>
  <tr>
      <td>invNorm (aclTensor*)</td>
      <td>Optional output</td>
      <td>Per-row normalization coefficient saved in forward, which is inv_rms in the corresponding formula.</td>
      <td><ul><li>A null pointer can be passed when needBackward is false.</li><li>A null pointer is not supported when needBackward is true. The shape is (T, N+1).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
  </tr>
  <tr>
      <td>probs (aclTensor*)</td>
      <td>Optional output.</td>
      <td>Forward softmax output probability, which is probs in the corresponding formula.</td>
      <td><ul><li>A null pointer can be passed when needBackward is false.</li><li>A null pointer is not supported when needBackward is true. The shape is (T, N+1).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
  </tr>
  <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be side-requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>executor(aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor. It contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 330px">
  <col style="width: 140px">
  <col style="width: 762px">
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
    <td> ACLNN_ERR_PARAM_NULLPTR </td>
    <td> 161001 </td>
    <td>partialBlock, blockRes, projWeight, normWeight, or hiddenStates is a null pointer; invNorm or probs is a null pointer when needBackward is true.</td>
    </tr>
    <tr>
    <td rowspan="3"> ACLNN_ERR_PARAM_INVALID </td>
    <td rowspan="3"> 161002 </td>
    <td>The data type of partialBlock, blockRes, projWeight, normWeight, hiddenStates, invNorm, or probs is not in the supported range, or the data types of computation tensors are inconsistent.</td>
    </tr>
    <tr>
    <td>The shape dimension of partialBlock, blockRes, projWeight, normWeight, hiddenStates, invNorm, or probs is not in the supported range.</td>
    </tr>
    <tr>
    <td>validBlockNum is not -1, or normEps is less than or equal to 0.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_INNER_CREATE_EXECUTOR </td>
    <td> 561101 </td>
    <td>Interface internal creation of aclOpExecutor failed.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_INNER_NULLPTR </td>
    <td> 561103 </td>
    <td>A null pointer is returned when interface internal performs contiguous processing, creates operator output, or creates an output copy node.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_RUNTIME_ERROR </td>
    <td> 361001 </td>
    <td>An exception occurs when the API internally calls npu runtime APIs.</td>
    </tr>
  </tbody></table>

## aclnnBlockAttentionResiduals

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 134px">
  <col style="width: 844px">
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
    <td>workspace</td>
    <td>Input</td>
    <td>Memory address of the workspace side-requested on the Device.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace side-requested on the Device, obtained by calling the first-phase API aclnnBlockAttentionResidualsGetWorkspaceSize.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Op executor, containing the operator computation flow.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the stream where the task is executed.</td>
  </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnBlockAttentionResiduals defaults to a deterministic implementation.

- Batch consistency:
  - aclnnBlockAttentionResiduals defaults to a Batch-consistent implementation.

- Specification constraints:
  - T is greater than or equal to 0, H is greater than or equal to 1, and N is in the range of 1-100.
  - partialBlock, blockRes, projWeight, normWeight, and hiddenStates must have the same data type. FLOAT16, BFLOAT16, and FLOAT32 are supported. invNorm and probs support only FLOAT32.
  - The shape must meet the following requirements: partialBlock is (T, H), blockRes is (T, N, H), projWeight is (H) or (1, H), normWeight is (H), and hiddenStates is (T, H). When needBackward is true, invNorm and probs are (T, N+1).
  - validBlockNum is a reserved attribute. The default value is -1, and it does not participate in the computation currently. Only -1 can be passed in.
  - normEps must be a finite positive number. The default value is 1e-6. If normEps is NaN, hiddenStates is NaN, and invNorm is also NaN when needBackward is true. If normEps is positive infinity, invNorm is 0 when needBackward is true.
  - Input tensors support non-contiguous tensors; the computation is performed after the interface-internal conversion to Contiguous. Output tensors do not support non-contiguous tensors.
  - Only the platform system-reserved workspace is used; no additional user-side workspace is requested.

## Example

The following sample code is for reference only. For the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_block_attention_residuals.h"

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
    for (auto dim : shape) {
        shapeSize *= dim;
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

int CreateAclTensor(
    const std::vector<uint16_t>& hostData, const std::vector<int64_t>& shape,
    void** deviceAddr, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(uint16_t);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), aclDataType::ACL_BF16, nullptr, 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t T = 2;
    const int64_t N = 4;
    const int64_t H = 64;

    std::vector<int64_t> partialBlockShape     = {T, H};
    std::vector<int64_t> blockShape      = {T, N, H};
    std::vector<int64_t> projShape       = {1, H};
    std::vector<int64_t> normShape       = {H};
    std::vector<int64_t> hiddenShape     = {T, H};

    std::vector<uint16_t> partialBlockData(GetShapeSize(partialBlockShape), 0x3C00);
    std::vector<uint16_t> blockData(GetShapeSize(blockShape), 0x3C00);
    std::vector<uint16_t> projData(GetShapeSize(projShape), 0x3C00);
    std::vector<uint16_t> normData(GetShapeSize(normShape), 0x3C00);
    std::vector<uint16_t> hiddenData(GetShapeSize(hiddenShape), 0);

    void* partialBlockDev = nullptr;
    void* blockDev = nullptr;
    void* projDev = nullptr;
    void* normDev = nullptr;
    void* hiddenDev = nullptr;

    aclTensor* partialBlockTensor = nullptr;
    aclTensor* blockTensor = nullptr;
    aclTensor* projTensor = nullptr;
    aclTensor* normTensor = nullptr;
    aclTensor* hiddenTensor = nullptr;

    ret = CreateAclTensor(partialBlockData, partialBlockShape, &partialBlockDev, &partialBlockTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockData, blockShape, &blockDev, &blockTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(projData, projShape, &projDev, &projTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normData, normShape, &normDev, &normTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hiddenData, hiddenShape, &hiddenDev, &hiddenTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnBlockAttentionResidualsGetWorkspaceSize(
        partialBlockTensor, blockTensor, projTensor, normTensor, -1, 1e-6, false,
        hiddenTensor, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnBlockAttentionResiduals(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResiduals failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("BlockAttentionResiduals example ran successfully!\n");

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyTensor(partialBlockTensor);
    aclDestroyTensor(blockTensor);
    aclDestroyTensor(projTensor);
    aclDestroyTensor(normTensor);
    aclDestroyTensor(hiddenTensor);
    aclrtFree(partialBlockDev);
    aclrtFree(blockDev);
    aclrtFree(projDev);
    aclrtFree(normDev);
    aclrtFree(hiddenDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
