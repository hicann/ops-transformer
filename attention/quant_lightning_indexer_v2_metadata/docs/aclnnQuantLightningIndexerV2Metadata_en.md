# aclnnQuantLightningIndexerV2Metadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:46:09.521Z -->

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

- Description: `aclnnQuantLightningIndexerV2Metadata` is the preceding operator of `aclnnQuantLightningIndexerV2`, used to generate a load-balanced task partitioning scheme. This operator does not perform actual LightningIndexer computation. Instead, it computes on the AI CPU the start and end range of the computation to be processed by each AI Core based on the input parameters, to maximize compute resource utilization and avoid load imbalance across Cores.

  **This operator is not recommended for standalone use. It is recommended to use it together with the aclnnQuantLightningIndexerV2 operator to form a complete workflow.**
    1. Accepts the input data shape information of the aclnnQuantLightningIndexerV2 operator API, including batchSize, qSeqlen, kSeqlen, and mask. By splitting the input into blocks and simulating the computation time, blocks are evenly distributed across available cores, reducing the overall computation time of the aclnnQuantLightningIndexerV2 operator and improving hardware utilization.
    2. After the allocation result is output, it serves as input for the aclnnQuantLightningIndexerV2 operator.
    3. The allocation result contains the start and end points of the basic block on each AIC core, and the FD task information of each AIV core. For details, see [Example](#example).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize" to obtain the workspace size, and then call "aclnnQuantLightningIndexerV2Metadata" to perform the computation.

``` cpp
aclnnStatus aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensKOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedKOptional,
    const aclTensor   *cmpResidualKOptional,
    int64_t            numHeadsQ,
    int64_t            numHeadsK,
    int64_t            headDim,
    int64_t            topk,
    int64_t            quantMode,
    int64_t            batchSize,
    int64_t            maxSeqlenQ,
    int64_t            maxSeqlenK,
    char              *layoutQOptional,
    char              *layoutKOptional,
    int64_t            maskMode,
    int64_t            cmpRatio,
    const aclTensor   *metadata,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

``` cpp
aclnnStatus aclnnQuantLightningIndexerV2Metadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 240px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 160px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 170px">
  <col style="width: 140px">
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
      <td>cuSeqlensQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates different valid Sequence Length of q in each Batch.</td>
      <td><ul><li>Supports empty tensor</li><li>shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensKOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates different valid Sequence Length of k in each Batch.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates different Sequence Lengths of q actually participating in computation in different Batches.</td>
      <td><ul><li>Supports empty tensor.</li><li>Shape fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedKOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates different Sequence Lengths of k actually participating in computation in different Batches.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the remainder of the compressed Sequence Length of k in different Batches. Used together with cmpRatio to implement mask and load calculation.</td>
      <td><ul><li>Supports empty tensors.</li><li>Must be passed when cmpRatio is not 1 and mask is 3.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of heads of q.</td>
      <td>Currently supports [1, 64].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of heads of k.</td>
      <td>Currently only supports 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim (int64_t)</td>
      <td>Input</td>
      <td>Dimension of the attention head.</td>
      <td>Currently only supports 128.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key sparse tokens filtered from q.</td>
      <td>Currently only supports [1, 8192].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the quantization mode.</td>
      <td><ul><li>Currently supports 1/2/3/4/5.</li><li>1: qk: fp8(e4m3) per-token-head; scale: fp32.</li><li>2: qk: int8 per-token-head; scale: fp16 w: fp16.</li><li>3: qk: mxfp8(e4m3), scale: fp8(e8m0).</li><li>4: qk: hif8 per-tensor; scale: fp32.</li><li>5: mxfp4(e2m1), scale: fp8(e8m0).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of batches.</td>
      <td><ul><li>Supports non-negative values.</li><li>Recommended value: 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of q.</td>
      <td><ul><li>Value range: ≥-1. -1 indicates any possible length.</li><li>Recommended value: -1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenK (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of k.</td>
      <td><ul><li>Value range: ≥-1. -1 indicates any possible length.</li><li>Recommended value: -1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format of q.</td>
      <td><ul><li>Supports BSND and TND.</li><li>Recommended value: BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format of k.</td>
      <td><ul><li>Supports BSND, TND, and PA_BBND.</li><li>Recommended value: BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the sparse mode.</td>
      <td><ul><li>0: No mask.</li><li>3: Mask of rightDownCausal mode, corresponding to the lower-triangular use case divided by the right vertex.</li><li>Recommended value: 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Indicates the compression ratio of k.</td>
      <td><ul><li>Value range: [1, 128].</li><li>Recommended value: 1, indicating no compression.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata (aclTensor*)</td>
      <td>Output</td>
      <td>Load balancing result output.</td>
      <td>The shape is (8 * (1 + aic_num + aiv_num) + 256,), where aic_num and aiv_num are obtained from the current device. The final 256 int32 elements form a shared extension area.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <ul>
    <!-- npu="A3" id7 -->
    <li><term>Atlas A3 products</term>: numHeadsQ supports 64, 32, and 24 (when numHeadsQ is 24/32, layoutQOptional must be TND, maskMode must be 3, and cmpRatio must be 1), quantMode = 1/3/4/5 is not supported, topk only supports [1, 2048], layoutKOptional = BSND is not supported, cmpRatio does not support any value in [1, 128], and only cmpRatio = 1/2/4/8/16/32/64/128 is supported.</li>
    <!-- end id7 -->
    <!-- npu="910b" id8 -->
    <li><term>Atlas A2 products</term>: numHeadsQ supports 64, 32, and 24 (when numHeadsQ is 24/32, layoutQOptional must be TND, maskMode must be 3, and cmpRatio must be 1), quantMode = 1/3/4/5 is not supported, topk only supports [1, 2048], layoutKOptional = BSND is not supported, cmpRatio does not support any value in [1, 128], and only cmpRatio = 1/2/4/8/16/32/64/128 is supported.</li>
    <!-- end id8 -->
  </ul>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
      <tr>
        <th>Return value</th>
        <th>Error code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>Failed to create aclOpExecutor.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>The workspaceSize or executor parameter is a null pointer, or parameters cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, and cmpResidualKOptional are null pointers after Contiguous processing.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>The specifications of parameters cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional, numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional, maskMode, and cmpRatio are not within the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnQuantLightningIndexerV2Metadata

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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>workspace</td>
            <td>Input</td>
            <td>Workspace memory address applied by the Device side.</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>Input</td>
            <td>Workspace size applied by the Device side, obtained by the first-phase API aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize.</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>Input</td>
            <td>Executor of the op, containing the computation process of the operator.</td>
        </tr>
        <tr>
            <td>stream</td>
            <td>Input</td>
            <td>Stream for task execution.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - aclnnQuantLightningIndexerV2Metadata defaults to a deterministic implementation.
  - B (Batch) indicates the batch size of the input samples. q and k are input parameters of the paired aclnnQuantLightningIndexerV2 operator. S1 indicates the size of the S axis in the q shape when layoutQOptional=BSND, and S2 indicates the size of the S axis in the k shape when layoutKOptional=BSND.
  - The value of parameters cuSeqlensQOptional and cuSeqlensKOptional must be the cumulative number of valid tokens in the current batch and preceding batches. The first element is fixed to 0, and the value of each subsequent element must be greater than or equal to that of the preceding element.
  - The values of sequsedQOptional and sequsedKOptional must indicate the number of valid tokens in each Batch.
  - For non-PA use cases, layoutQOptional and layoutKOptional must be the same.
  - cmpResidualKOptional must satisfy cmpResidualKOptional[i] < cmpRatio.
  - For layoutQOptional=BSND use case
    - The value of S1 must be passed to maxSeqlenQ.
  - For layoutKOptional=BSND use case
    - The value of S2 must be passed to maxSeqlenK.
  - For layoutQOptional=TND use case
    - cuSeqlensQOptional must be passed.
  - For layoutKOptional=TND use case
    - cuSeqlensKOptional must be passed.
  - layoutKOptional=PA_BBND use case
    - sequsedKOptional must be passed.
  - Batch value rule
    - For layoutQOptional=BSND, prefer deriving batch through the shape of sequsedQOptional. If not passed, obtain the batch count from batchSize.
    - For layoutQOptional=TND, prefer deriving batch through the shape of sequsedQOptional. If not passed, derive batch through the shape of cuSeqlensQOptional.
  - q Seqlen value rule
    - For layoutQOptional=BSND, prefer obtaining seqlen through elements in sequsedQOptional. If not passed, obtain seqlen from maxSeqlenQ.
    - For layoutQOptional=TND, prefer obtaining seqlen through elements in sequsedQOptional. If not passed, obtain seqlen through elements in cuSeqlensQOptional.
  - k Seqlen value rule
    - For layoutKOptional=BSND, prefer through elements in sequsedKOptional to obtain seqlen. If sequsedKOptional is not passed, obtain seqlen through maxSeqlenK.
    - For layoutKOptional=TND, prefer through elements in sequsedKOptional to obtain seqlen. If sequsedKOptional is not passed, obtain seqlen through elements in cuSeqlensKOptional.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

``` cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

// Refer to quant_lightning_indexer_v2_metadata.h.
uint32_t aicCoreNum = 0;
uint32_t aivCoreNum = 0;
uint32_t metadataSize = 0;
constexpr uint32_t QLI_V2_METADATA_SIZE = 8;
constexpr uint32_t QLD_V2_METADATA_SIZE = 8;

// QLI Metadata Index Definitions
constexpr uint32_t QLI_V2_CORE_ENABLE_INDEX = 0;
constexpr uint32_t QLI_V2_BN2_START_INDEX = 1;
constexpr uint32_t QLI_V2_M_START_INDEX = 2;
constexpr uint32_t QLI_V2_S2_START_INDEX = 3;
constexpr uint32_t QLI_V2_BN2_END_INDEX = 4;
constexpr uint32_t QLI_V2_M_END_INDEX = 5;
constexpr uint32_t QLI_V2_S2_END_INDEX = 6;
constexpr uint32_t QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX = 7;

// QLD Metadata Index Definitions
constexpr uint32_t QLD_V2_CORE_ENABLE_INDEX = 0;
constexpr uint32_t QLD_V2_BN2_IDX_INDEX = 1;
constexpr uint32_t QLD_V2_M_IDX_INDEX = 2;
constexpr uint32_t QLD_V2_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t QLD_V2_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t QLD_V2_M_START_INDEX = 5;
constexpr uint32_t QLD_V2_M_NUM_INDEX = 6;

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    // Do not copy.
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr { nullptr };
    void *deviceAddr { nullptr };
    aclTensor *data { nullptr };
};

struct ArgScenario {
    bool hasCuSeq { false };
    bool hasSeqused { false };
};

struct ArgContext {
    // required input
    int64_t numHeadsQ { 0 };
    int64_t numHeadsK { 0 };
    int64_t headDim { 0 };
    int64_t topk { 0 };
    int64_t quantMode { 2 };
    // optional input
    Tensor cuSeqlensQOptional {};
    Tensor cuSeqlensKOptional {};
    Tensor sequsedQOptional {};
    Tensor sequsedKOptional {};
    Tensor cmpResidualKOptional {};
    int64_t batchSize { 0 };
    int64_t maxSeqlenQ { 0 };
    int64_t maxSeqlenK { 0 };
    char *layoutQOptional { nullptr };
    char *layoutKOptional { nullptr };
    int64_t maskMode { 0 };
    int64_t cmpRatio { 0 };
    // output
    Tensor metadata {};
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate for initialization.
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // Call aclrtMallocHost to allocate memory on the host side.
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // Call aclrtMalloc to allocate memory on the device side.
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // Call the aclCreateTensor API to create an aclTensor.
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);

    // Call aclrtMemcpy to copy data from the host side to the device side.
    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqlensQOptional);
    DestroyTensor(context.cuSeqlensKOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedKOptional);
    DestroyTensor(context.cmpResidualKOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKOptional != nullptr) {
        free(context.layoutKOptional);
        context.layoutKOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;
    int64_t batchSize = 4;

    context.numHeadsQ = 1;
    context.numHeadsK = 1;
    context.headDim = 128;
    context.topk = 0;
    context.quantMode = 2; // 2: per-token-head / 3: group-scaling
    ret = aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &aicCoreNum);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Failed to query AIC count\n");
    ret = aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &aivCoreNum);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Failed to query AIV count\n");
    metadataSize = 8 * (1 + aicCoreNum + aivCoreNum) + 256;
    ret = CreateTensor(aclDataType::ACL_INT32, {metadataSize}, context.metadata);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create meta failed. Error: %d", ret);

    context.maskMode = 0;                   // 0: no mask, 3: causal
    context.cmpRatio = 1;                   // [1, 128], 1: no compress
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKOptional = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQOptional, "BSND");                // BSND,TND
    strcpy(context.layoutKOptional, "BSND");                // BSND,TND,PA_BBND

    if (!scenario.hasCuSeq && !scenario.hasSeqused) {
        context.batchSize = batchSize;
        context.maxSeqlenK = 1024;
        context.maxSeqlenQ = 1024;
        argsGuard.Dismiss();
        return ACL_SUCCESS;
    }

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqlensKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensKOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.sequsedKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedKOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main() {
    // 1. (Boilerplate) Device/stream initialization. Refer to the external API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. Construct the input and output based on the API definition.
    ArgScenario scenario {};
    scenario.hasCuSeq = true;
    scenario.hasSeqused = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. Call the CANN operator library API. Replace it with the specific API.
    // Call the first-phase API of aclnnQuantLightningIndexerV2Metadata.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensKOptional.data, context.sequsedQOptional.data,
        context.sequsedKOptional.data, context.cmpResidualKOptional.data,
        context.numHeadsQ, context.numHeadsK, context.headDim, context.topk, context.quantMode,
        context.batchSize, context.maxSeqlenQ, context.maxSeqlenK, context.layoutQOptional,
        context.layoutKOptional, context.maskMode, context.cmpRatio,
        context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d\n", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    // Call the second-phase API of aclnnQuantLightningIndexerV2Metadata.
    ret = aclnnQuantLightningIndexerV2Metadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnQuantLightningIndexerV2Metadata failed. ERROR: %d\n", ret);

    // 4. (Boilerplate) Wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. Print the output.
    std::vector<uint32_t> result(metadataSize);
    ret = aclrtMemcpy(result.data(), result.size() * sizeof(uint32_t), context.metadata.deviceAddr,
                      result.size() * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < result[0]; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Core Enable : %u\n", result[8 + i * QLI_V2_METADATA_SIZE + QLI_V2_CORE_ENABLE_INDEX]);
        printf("    Start BN2   : %u\n", result[8 + i * QLI_V2_METADATA_SIZE + QLI_V2_BN2_START_INDEX]);
        printf("    Start M     : %u\n", result[8 + i * QLI_V2_METADATA_SIZE + QLI_V2_M_START_INDEX]);
        printf("    Start S2    : %u\n", result[8 + i * QLI_V2_METADATA_SIZE + QLI_V2_S2_START_INDEX]);
        printf("    End BN2     : %u\n", result[8 + i * QLI_V2_METADATA_SIZE + QLI_V2_BN2_END_INDEX]);
        printf("    End M       : %u\n", result[8 + i * QLI_V2_METADATA_SIZE + QLI_V2_M_END_INDEX]);
        printf("    End S2      : %u\n", result[8 + i * QLI_V2_METADATA_SIZE + QLI_V2_S2_END_INDEX]);
        printf("    First Workspace Index : %u\n", result[8 + i * QLI_V2_METADATA_SIZE + QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX]);
    }
    for (uint32_t i = 0; i < result[1]; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result[8 + result[0] * QLI_V2_METADATA_SIZE + i * QLD_V2_METADATA_SIZE + QLD_V2_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result[8 + result[0] * QLI_V2_METADATA_SIZE + i * QLD_V2_METADATA_SIZE + QLD_V2_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result[8 + result[0] * QLI_V2_METADATA_SIZE + i * QLD_V2_METADATA_SIZE + QLD_V2_M_IDX_INDEX]);
        printf("    FD Task Workspace Idx   : %u\n", result[8 + result[0] * QLI_V2_METADATA_SIZE + i * QLD_V2_METADATA_SIZE + QLD_V2_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result[8 + result[0] * QLI_V2_METADATA_SIZE + i * QLD_V2_METADATA_SIZE + QLD_V2_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result[8 + result[0] * QLI_V2_METADATA_SIZE + i * QLD_V2_METADATA_SIZE + QLD_V2_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result[8 + result[0] * QLI_V2_METADATA_SIZE + i * QLD_V2_METADATA_SIZE + QLD_V2_M_NUM_INDEX]);
    }

    return 0;
}
```
