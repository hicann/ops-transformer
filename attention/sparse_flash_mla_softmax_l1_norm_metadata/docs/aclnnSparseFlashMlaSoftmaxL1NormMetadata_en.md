# aclnnSparseFlashMlaSoftmaxL1NormMetadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:31:57.291Z -->

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

- Description: This operator is an AICPU operator and serves as the preceding operator of aclnnSparseFlashMlaSoftmaxL1Norm. It calculates and outputs core-partitioned metadata based on the input shape, layout, mask, and compression ratio of aclnnSparseFlashMlaSoftmaxL1Norm. The output can be used as the metadataOptional input of aclnnSparseFlashMlaSoftmaxL1Norm, reducing host array access in the tiling phase of the main operator.

  **This operator is not recommended for standalone use. Use it with aclnnSparseFlashMlaSoftmaxL1Norm to form a complete workflow.**
    1. Receives the shape information of the main operator, including batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK, headDim, topk, layout, and mask.
    2. Calculates the total number of seqs involved in load balancing based on layout and the optional input seqUsedQOptional, and evenly partitions tasks across available AIC cores in a strided manner.
    3. Outputs metadata, which is then used as the metadataOptional input of the aclnnSparseFlashMlaSoftmaxL1Norm operator.

## Function Prototype

Each operator uses a two-phase API. You must first call the first-phase API "aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize" to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the second-phase API "aclnnSparseFlashMlaSoftmaxL1NormMetadata" to execute the computation.

``` cpp
aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize(
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *topkLengthOptional,
    int64_t batchSize,
    int64_t maxSeqLenQ,
    int64_t maxSeqLenK,
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    int64_t topk,
    int64_t cmpRatio,
    int64_t maskMode,
    char *layoutQ,
    char *layoutK,
    const aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

``` cpp
aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 350px">
  <col style="width: 150px">
  <col style="width: 70px">
  <col style="width: 70px">
  <col style="width: 190px">
  <col style="width: 80px">
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
      <td>cuSeqLensQOptional</td>
      <td>Input</td>
      <td>Represents the cumulative sequence length of query in different batches.<br>Required in the TND use case. The batch count can be derived from the shape of this parameter.<br>The first value is fixed to 0.</td>
      <td>Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional</td>
      <td>Input</td>
      <td>Represents the cumulative sequence length of key in different batches.<br>Required in the TND use case.<br>The first value is fixed to 0.</td>
      <td>Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D, shape (B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>seqUsedQOptional</td>
      <td>Input</td>
      <td>Indicates the actual sequence length of query participating in computation in different batches.</td>
      <td>Supports empty tensors. If this parameter exists, the total number of seqs is the sum of seqs actually used in each batch.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>seqUsedKOptional</td>
      <td>Input</td>
      <td>Sequence length of keys that actually participate in the computation in different batches.</td>
      <td>Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional</td>
      <td>Input</td>
      <td>Represents the residual of key sequence lengths in different batches related to cmpRatio.</td>
      <td>Supports empty tensors. Required when maskMode=3 and cmpRatio!=1.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>topkLengthOptional</td>
      <td>Input</td>
      <td>Indicates the actual selectable topk length of keys corresponding to each row of query.</td>
      <td>Empty tensors are supported. Required when maskMode=0 and sparse indices exist, and the value must be accurate.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2D or 3D, shape (B,S1,N2) or (T1,N2)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>batchSize</td>
      <td>Input</td>
      <td>Indicates the batch count.</td>
      <td>Non-negative values are supported. In the TND use case, this parameter can be set to 0 and derived through cuSeqLensQOptional.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenQ</td>
      <td>Input</td>
      <td>Represents the maximum sequence length of query.</td>
      <td>Non-negative values are supported. In the BSND use case, this parameter must be a positive number.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenK</td>
      <td>Input</td>
      <td>Indicates the maximum sequence length of key.</td>
      <td>Supports non-negative values. Must be a positive number in the BSND use case.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsQ</td>
      <td>Input</td>
      <td>Head count of query.</td>
      <td>Value range [1, 128], and must be divisible by numHeadsK.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK</td>
      <td>Input</td>
      <td>Indicates the head count of key.</td>
      <td>Currently, only 1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>Input</td>
      <td>Indicates the head dimension of q/k.</td>
      <td>256 or 512 is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>Input</td>
      <td>Count of key tokens filtered from key.</td>
      <td>Non-negative numbers are supported. 0 indicates no sparsity.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio</td>
      <td>Input</td>
      <td>Represents the compression ratio of key.</td>
      <td>Value range [1, 128].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode</td>
      <td>Input</td>
      <td>Represents the sparse mask mode.</td>
      <td>0: No mask<br>3: Causal</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>Input</td>
      <td>Represents the layout format on the query side.</td>
      <td>BSND/TND are supported. The default value is BSND.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutK</td>
      <td>Input</td>
      <td>Represents the layout format on the key side.</td>
      <td>Supports BSND/TND. The default value is BSND.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>Output</td>
      <td>Indicates the load balancing result output.</td>
      <td>The output is used as the metadataOptional input of aclnnSparseFlashMlaSoftmaxL1Norm.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, fixed shape (64,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>The current implementation returns 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor. It contains the operator computation flow.</td>
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

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
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
      <td>A required parameter or the output is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or value of the input parameter is out of the supported range.</td>
    </tr>
    </tbody></table>

## aclnnSparseFlashMlaSoftmaxL1NormMetadata

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
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Workspace memory address allocated on the Device side.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device side. The value is obtained by calling the first-phase API aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation process.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
        </tr>
    </tbody>
    </table>

- **Return Value**

   aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnSparseFlashMlaSoftmaxL1NormMetadata defaults to a deterministic implementation.

  - BSND use case
    - batchSize, maxSeqLenQ, and maxSeqLenK must be passed to obtain shape information.
  - TND use case
    - Mandatory to pass cuSeqLensQOptional and cuSeqLensKOptional parameters to obtain correct shape information.
    - When batchSize is 0, derive the batch from the shape of cuSeqLensQOptional.

<details>
<summary><a id="Mask"></a>Mask</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>mask_mode</th>
                <th>Meaning</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>No mask.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Mask in rightDownCausal mode, corresponding to the lower-triangular use case partitioned by the right vertex.</td>
            <td>Supported</td>
        </tr>
        </tbody>
    </table>
</details>

<details>
<summary><a id="SpecialConstraints"></a>Special Constraints</summary>

  - Batch value rules
    - If batchSize is greater than 0, batchSize takes priority.
    - If batchSize is less than or equal to 0 and layoutQ is TND, derive the batch from the shape of cuSeqLensQOptional.
    - If batchSize is less than or equal to 0 and layoutQ is BSND, an error is reported.
  - Seqlen value rules
    - In the TND use case, the actual q/k length of each batch is computed through cuSeqLensQOptional and cuSeqLensKOptional.
    - In the BSND use case, the q/k length is obtained through maxSeqLenQ and maxSeqLenK.
  - layout constraints
    - layoutQ must be BSND or TND.
    - layoutK supports BSND and TND. Consistency with layoutQ is recommended.
  - head constraints
    - numHeadsQ: range [1, 128].
    - Currently, only numHeadsK = 1 is supported.
    - headDim supports 256 or 512.
    - numHeadsQ must be divisible by numHeadsK.
  - Load balancing constraint
    - topk supports non-negative values. 0 indicates no sparsity.
    - cmpRatio: range [1, 128].
    - maskMode supports only 0 and 3.

</details>

<details>
<summary><a id="Metadata"></a>Metadata output layout</summary>

  The metadata output is an INT32 tensor. The shape is currently fixed to (64,). The field layout is as follows.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>Field</th>
        <th>index</th>
        <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>totalNum</td>
        <td>0</td>
        <td>Total number of seq values participating in load balancing. In the TND use case, this value is the cumulative sum of seqQ across batches; in the BSND use case, this value is B*S; when seqUsedQOptional exists, this value is the cumulative sum of the seq values actually used in each batch.</td>
        </tr>
        <tr>
        <td>formerCoreProcessNum</td>
        <td>1</td>
        <td>Number of seq values processed by regular cores, that is, ceil(totalNum / totalCoreNum).</td>
        </tr>
        <tr>
        <td>remainCoreProcessNum</td>
        <td>2</td>
        <td>Number of seq values processed by tail cores, that is, floor(totalNum / totalCoreNum).</td>
        </tr>
        <tr>
        <td>remainCoreNum</td>
        <td>3</td>
        <td>Number of tail cores. 0 if totalNum is divisible by totalCoreNum; otherwise, totalCoreNum - (totalNum % totalCoreNum).</td>
        </tr>
        <tr>
        <td>totalCoreNum</td>
        <td>4</td>
        <td>Number of AIC cores actually used, min(totalNum, aicCoreNum, 36).</td>
        </tr>
    </tbody>
  </table>

  Strided partitioning is used. The seq indexes processed by a core step by totalCoreNum across cores. For example, if totalNum=72 and totalCoreNum=36, core 0 processes seq[0] and seq[36], core 1 processes seq[1] and seq[37], and so on.

</details>

## Example

The following sample code is for reference only. For details about compilation and execution, see [compile_and_run_sample](../../../docs/en/context/compile_and_run_sample.md).

``` cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_softmax_l1_norm_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

constexpr uint32_t SMLA_METADATA_SIZE = 64;

struct SmlaSoftmaxL1NormMetaData {
    int32_t totalNum;
    int32_t formerCoreProcessNum;
    int32_t remainCoreProcessNum;
    int32_t remainCoreNum;
    int32_t totalCoreNum;
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
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
    bool hasCuSeq { true };
};

struct ArgContext {
    Tensor cuSeqLensQOptional {};
    Tensor cuSeqLensKOptional {};
    Tensor seqUsedQOptional {};
    Tensor seqUsedKOptional {};
    Tensor cmpResidualKOptional {};
    Tensor topkLengthOptional {};
    Tensor metadata {};
    int64_t batchSize { 0 };
    int64_t maxSeqLenQ { 0 };
    int64_t maxSeqLenK { 0 };
    int64_t numHeadsQ { 8 };
    int64_t numHeadsK { 1 };
    int64_t headDim { 512 };
    int64_t topk { 64 };
    char *layoutQ { nullptr };
    char *layoutK { nullptr };
    int64_t maskMode { 3 };
    int64_t cmpRatio { 1 };
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
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);

    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);

    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void SetInt32TensorData(Tensor &tensor, const std::vector<int32_t> &hostData)
{
    auto size = hostData.size() * sizeof(int32_t);
    memcpy(tensor.hostAddr, hostData.data(), size);
    aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
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
    DestroyTensor(context.cuSeqLensQOptional);
    DestroyTensor(context.cuSeqLensKOptional);
    DestroyTensor(context.seqUsedQOptional);
    DestroyTensor(context.seqUsedKOptional);
    DestroyTensor(context.cmpResidualKOptional);
    DestroyTensor(context.topkLengthOptional);

    if (context.layoutQ != nullptr) {
        free(context.layoutQ);
        context.layoutQ = nullptr;
    }
    if (context.layoutK != nullptr) {
        free(context.layoutK);
        context.layoutK = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    int64_t batchSize = 1;
    context.maxSeqLenQ = 16;
    context.maxSeqLenK = 4;
    context.layoutQ = (char *)malloc(sizeof(char) * 16);
    context.layoutK = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQ, scenario.hasCuSeq ? "TND" : "BSND");
    strcpy(context.layoutK, scenario.hasCuSeq ? "TND" : "BSND");

    ret = CreateTensor(aclDataType::ACL_INT32, { SMLA_METADATA_SIZE }, context.metadata);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);

    if (scenario.hasCuSeq) {
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensKOptional failed. Error: %d", ret);
        SetInt32TensorData(context.cuSeqLensQOptional, { 0, static_cast<int32_t>(context.maxSeqLenQ) });
        SetInt32TensorData(context.cuSeqLensKOptional, { 0, static_cast<int32_t>(context.maxSeqLenK) });
        context.batchSize = 0;
    } else {
        context.batchSize = batchSize;
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

void PrintMetadata(const SmlaSoftmaxL1NormMetaData &metadata)
{
    printf("totalNum             : %d\n", metadata.totalNum);
    printf("formerCoreProcessNum: %d\n", metadata.formerCoreProcessNum);
    printf("remainCoreProcessNum: %d\n", metadata.remainCoreProcessNum);
    printf("remainCoreNum       : %d\n", metadata.remainCoreNum);
    printf("totalCoreNum        : %d\n", metadata.totalCoreNum);
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    ArgScenario scenario {};
    scenario.hasCuSeq = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize(
        context.cuSeqLensQOptional.data, context.cuSeqLensKOptional.data, context.seqUsedQOptional.data,
        context.seqUsedKOptional.data, context.cmpResidualKOptional.data, context.topkLengthOptional.data,
        context.batchSize, context.maxSeqLenQ, context.maxSeqLenK, context.numHeadsQ, context.numHeadsK,
        context.headDim, context.topk, context.cmpRatio, context.maskMode, context.layoutQ, context.layoutK,
        context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize failed. ERROR: %d", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    ret = aclnnSparseFlashMlaSoftmaxL1NormMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseFlashMlaSoftmaxL1NormMetadata failed. ERROR: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d", ret);

    SmlaSoftmaxL1NormMetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    PrintMetadata(result);

    return 0;
}
```

## Troubleshooting

- For common execution issues or exceptions of AI CPU operator kernels, see "[Fault Cases > Operator Execution Issues > AI CPU Operator Kernel Execution Errors](https://www.hiascend.com/document/detail/en/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)" in *Troubleshooting*.
