# aclnnLightningIndexerV2Metadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:13:23.057Z -->

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

- Description: `aclnnLightningIndexerV2Metadata` is the preceding operator of the `aclnnLightningIndexerV2` operator. It generates a load-balanced task partitioning scheme. This operator does not perform actual LightningIndexer computation. Instead, it computes the start and end of the computation range for each AI Core on the AI CPU based on the input parameters. This maximizes the utilization of computing resources and avoids load imbalance among Cores.

  **This operator is not recommended for standalone use. Use it together with the aclnnLightningIndexerV2 operator to form a complete workflow.**
    1. Accepts the input data shape information of the aclnnLightningIndexerV2 API, including batchSize, qSeqlen, kSeqlen, and mask. It partitions the input and simulates the computation time, then evenly allocates the partitions to the available cores. This reduces the overall computation time of the aclnnLightningIndexerV2 operator and improves hardware utilization.
    2. The allocation result is output and then used as input for the aclnnLightningIndexerV2 operator.
    3. The allocation result contains the start and end points of the basic block of each AIC core, and the FD task information of each AIV core. For details, see [Example](#example).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnLightningIndexerV2MetadataGetWorkspaceSize" first to obtain the workspace size, and then call "aclnnLightningIndexerV2Metadata" to perform the computation.

``` cpp
aclnnStatus aclnnLightningIndexerV2MetadataGetWorkspaceSize(
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensKOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedKOptional,
    const aclTensor   *cmpResidualKOptional,
    int64_t            numHeadsQ,
    int64_t            numHeadsK,
    int64_t            headDim,
    int64_t            topk,
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
aclnnStatus aclnnLightningIndexerV2Metadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnLightningIndexerV2MetadataGetWorkspaceSize

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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>cuSeqlensQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the effective Sequence Length of q in different Batches.</td>
      <td><ul><li>Supports empty Tensor</li><li>Shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-dim</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensKOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the effective Sequence Length of k in different Batches.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the actual sequence length of q participating in computation in each batch.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedKOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the actual Sequence Length of k participating in the computation in different Batches.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the remainder of the compressed Sequence Length of k in different batches. Works with cmpRatio to implement mask and load calculation.</td>
      <td><ul><li>Supports empty tensors.</li><li>Mandatory when cmpRatio is not 1 and mask is 3.</li><li>The shape is fixed to (B, ).</li><li>Each element value must be less than the input compression ratio cmpRatio.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ (int64_t)</td>
      <td>Input</td>
      <td>Represents the number of q heads.</td>
      <td>Currently supports [1, 64].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK (int64_t)</td>
      <td>Input</td>
      <td>Represents the number of k heads.</td>
      <td>Only 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim (int64_t)</td>
      <td>Input</td>
      <td>Dimension of the attention head.</td>
      <td>Only 128 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk (int64_t)</td>
      <td>Input</td>
      <td>Number of key sparse tokens selected from q.</td>
      <td>Currently only [1, 8192] is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of batches.</td>
      <td><ul><li>Supports non-negative values.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of q.</td>
      <td><ul><li>Value Range: ≥-1. A value of -1 indicates any possible length.</li><li>The recommended value is -1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenK (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum Sequence Length of k.</td>
      <td><ul><li>Value Range ≥ -1. -1 indicates any possible length.</li><li>The recommended value is -1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional (char*)</td>
      <td>Input</td>
      <td>Specifies the layout of q.</td>
      <td><ul><li>Supported values: BSND and TND.</li><li>The recommended value is BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional(char*)</td>
      <td>Input</td>
      <td>Represents the layout format of k.</td>
      <td><ul><li>Supported values: BSND, TND, and PA_BBND.</li><li>Recommended value: BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the sparse mode.</td>
      <td><ul><li>0: No mask.</li><li>3: mask in rightDownCausal mode, corresponding to the lower-triangular use case divided by the right vertex.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Compression rate of k.</td>
      <td><ul><li>Value Range: [1, 128].</li><li>Recommended value: 1, indicating no compression.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata (aclTensor*)</td>
      <td>Output</td>
      <td>Load balancing result output.</td>
      <td>The shape is fixed to (1024, ).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
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

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
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
        <td>The workspaceSize or executor parameter is a null pointer, or the cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, or cmpResidualKOptional parameter is a null pointer after Contiguous processing.</td>
      </tr>
      <tr>
        <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="1">161002</td>
        <td>The specifications of the cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional, numHeadsQ, numHeadsK, headDim, topk, batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional, maskMode, or cmpRatio parameter are not within the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnLightningIndexerV2Metadata

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
            <td>Workspace memory address allocated on the Device side.</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>Input</td>
            <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnLightningIndexerV2MetadataGetWorkspaceSize.</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>Input</td>
            <td>Op executor containing the operator computation process.</td>
        </tr>
        <tr>
            <td>stream</td>
            <td>Input</td>
            <td>Stream for task execution.</td>
        </tr>
    </tbody>
    </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - aclnnLightningIndexerV2Metadata defaults to a deterministic implementation.
  - B (Batch) indicates the batch size of input samples. q and k are input parameters of the matching aclnnLightningIndexerV2 operator. S1 indicates the size of the S axis in the q shape when layoutQOptional=BSND. S2 indicates the size of the S axis in the k shape when layoutKOptional=BSND.
  - Parameters cuSeqlensQOptional and cuSeqlensKOptional require their values to be cumulative sums of the number of valid tokens in the current and preceding batches. The first element is fixed to 0. Each subsequent element must be greater than or equal to the preceding element.
  - The values of sequsedQOptional and sequsedKOptional must indicate the number of valid tokens in each batch.
  - In non-PA scenarios, layoutQOptional and layoutKOptional must be the same.
  - cmpResidualKOptional must satisfy cmpResidualKOptional[i] < cmpRatio.
  - layoutQOptional=BSND scenario
    - maxSeqlenQ must be set to the value of S1.
  - layoutKOptional=BSND scenario
    - maxSeqlenK must be set to the value of S2.
  - layoutQOptional=TND scenario
    - cuSeqlensQOptional must be passed in.
  - layoutKOptional=TND scenario
    - cuSeqlensKOptional must be passed in.
  - layoutKOptional=PA_BBND scenario
    - sequsedKOptional must be passed in.
  - Batch Value Rules
    - If layoutQOptional is BSND, batch is preferentially inferred from the shape of sequsedQOptional. If sequsedQOptional is not passed in, batch is obtained from batchSize.
    - If layoutQOptional is TND, batch is preferentially inferred from the shape of sequsedQOptional. If sequsedQOptional is not passed in, batch is inferred from the shape of cuSeqlensQOptional.
  - q Seqlen Value Rules
    - If layoutQOptional is BSND, seqlen is preferentially obtained from the elements in sequsedQOptional. If sequsedQOptional is not passed in, seqlen is obtained from maxSeqlenQ.
    - If layoutQOptional is TND, seqlen is preferentially obtained from the elements in sequsedQOptional. If sequsedQOptional is not passed in, seqlen is obtained from the elements in cuSeqlensQOptional.
  - k Seqlen Value Rules
    - If layoutKOptional is BSND, obtain seqlen preferentially from the elements in sequsedKOptional. If sequsedKOptional is not passed in, obtain seqlen from maxSeqlenK.
    - If layoutKOptional is TND, obtain seqlen preferentially from the elements in sequsedKOptional. If sequsedKOptional is not passed in, obtain seqlen from the elements in cuSeqlensKOptional.

## Example

The following sample code is for reference only. For the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

``` cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_v2_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

// Refer to lightning_indexer_v2_metadata.h.
constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t LI_V2_METADATA_TOTAL_SIZE = 1024;
constexpr uint32_t LI_V2_METADATA_SIZE = 8;
constexpr uint32_t LD_V2_METADATA_SIZE = 8;

// LI Metadata Index Definitions
constexpr uint32_t LI_V2_CORE_ENABLE_INDEX = 0;
constexpr uint32_t LI_V2_BN2_START_INDEX = 1;
constexpr uint32_t LI_V2_M_START_INDEX = 2;
constexpr uint32_t LI_V2_S2_START_INDEX = 3;
constexpr uint32_t LI_V2_BN2_END_INDEX = 4;
constexpr uint32_t LI_V2_M_END_INDEX = 5;
constexpr uint32_t LI_V2_S2_END_INDEX = 6;
constexpr uint32_t LI_V2_FIRST_LD_V2_DATA_WORKSPACE_IDX_INDEX = 7;

// LD Metadata Index Definitions
constexpr uint32_t LD_V2_CORE_ENABLE_INDEX = 0;
constexpr uint32_t LD_V2_BN2_IDX_INDEX = 1;
constexpr uint32_t LD_V2_M_IDX_INDEX = 2;
constexpr uint32_t LD_V2_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t LD_V2_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t LD_V2_M_START_INDEX = 5;
constexpr uint32_t LD_V2_M_NUM_INDEX = 6;

struct LiV2Metadata {
    uint32_t faData[AIC_CORE_MAX_NUM][LI_V2_METADATA_SIZE];
    uint32_t fdData[AIV_CORE_MAX_NUM][LD_V2_METADATA_SIZE];
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    // Copying is prohibited.
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
    // Call aclrtMallocHost to allocate host memory.
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // Call aclrtMalloc to allocate device memory.
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // Call aclCreateTensor to create an aclTensor.
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);

    // Call aclrtMemcpy to copy data from the host to the device memory.
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
    ret = CreateTensor(aclDataType::ACL_INT32, { LI_V2_METADATA_TOTAL_SIZE }, context.metadata);     // 1024: Fix size
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
    // 1. (Boilerplate) Initialize device/stream. Refer to the external API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. Construct inputs and outputs based on the API definition.
    ArgScenario scenario {};
    scenario.hasCuSeq = true;
    scenario.hasSeqused = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. Call the CANN operator library API. Replace it with the specific API.
    // Call the first-phase API of aclnnLightningIndexerV2Metadata.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnLightningIndexerV2MetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensKOptional.data, context.sequsedQOptional.data,
        context.sequsedKOptional.data, context.cmpResidualKOptional.data,
        context.numHeadsQ, context.numHeadsK, context.headDim, context.topk,
        context.batchSize, context.maxSeqlenQ, context.maxSeqlenK, context.layoutQOptional,
        context.layoutKOptional, context.maskMode, context.cmpRatio,
        context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnLightningIndexerV2MetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

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

    // Call the second-phase API of aclnnLightningIndexerV2Metadata.
    ret = aclnnLightningIndexerV2Metadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnLightningIndexerV2Metadata failed. ERROR: %d\n", ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. Print the output.
    LiV2Metadata result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < AIC_CORE_MAX_NUM; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Core Enable : %u\n", result.faData[i][LI_V2_CORE_ENABLE_INDEX]);
        printf("    Start BN2   : %u\n", result.faData[i][LI_V2_BN2_START_INDEX]);
        printf("    Start M     : %u\n", result.faData[i][LI_V2_M_START_INDEX]);
        printf("    Start S2    : %u\n", result.faData[i][LI_V2_S2_START_INDEX]);
        printf("    End BN2     : %u\n", result.faData[i][LI_V2_BN2_END_INDEX]);
        printf("    End M       : %u\n", result.faData[i][LI_V2_M_END_INDEX]);
        printf("    End S2      : %u\n", result.faData[i][LI_V2_S2_END_INDEX]);
        printf("    First Workspace Index : %u\n", result.faData[i][LI_V2_FIRST_LD_V2_DATA_WORKSPACE_IDX_INDEX]);
    }
    for (uint32_t i = 0; i < AIV_CORE_MAX_NUM; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result.fdData[i][LD_V2_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result.fdData[i][LD_V2_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result.fdData[i][LD_V2_M_IDX_INDEX]);
        printf("    FD Task Workspace Idx   : %u\n", result.fdData[i][LD_V2_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdData[i][LD_V2_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdData[i][LD_V2_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdData[i][LD_V2_M_NUM_INDEX]);
    }

    return 0;
}
```
