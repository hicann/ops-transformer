# aclnnSparseLightningIndexerKLLossGradMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                                   |    x     |
| Atlas A3训练系列产品/Atlas A3推理系列产品                 |    √     |
| Atlas A2训练系列产品/Atlas A2推理系列产品                 |    √     |
| Atlas 200I/500 A2推理产品                                  |    x     |
| Atlas 推理系列产品                                          |    x     |
| Atlas 训练系列产品                                          |    x     |

## 功能说明

- 算子功能：该算子为AICPU算子，是aclnnSparseLightningIndexerKLLossGrad算子的前置算子。根据aclnnSparseLightningIndexerKLLossGrad算子的输入shape、layout、mask和压缩比例信息，计算并输出分核切分metadata。输出结果可作为aclnnSparseLightningIndexerKLLossGrad算子的metadataOptional输入，减少主算子tiling阶段对host array的访问。

  **该算子不建议单独使用，建议与aclnnSparseLightningIndexerKLLossGrad算子配合使用，形成完整的工作流。**
    1. 接收主算子的shape信息，包括batchSize、maxSeqLenQ、maxSeqLenK、numHeadsQ、numHeadsK、headDim、topk、layout和mask信息。
    2. 根据每个query对应的有效sparse长度估算负载，并将B/S1合轴后的任务均衡切分到可用AIC核上。
    3. 输出metadata后，后续作为aclnnSparseLightningIndexerKLLossGrad算子的metadataOptional输入使用。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize"获取workspace大小，再调用"aclnnSparseLightningIndexerKLLossGradMetadata"执行计算。

``` cpp
aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional,
    const aclTensor *cmpResidualKOptional,
    int64_t batchSize,
    int64_t maxSeqLenQ,
    int64_t maxSeqLenK,
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    int64_t topk,
    char *layoutQ,
    char *layoutK,
    int64_t maskMode,
    int64_t cmpRatio,
    const aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

``` cpp
aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize

## 参数说明

- **参数说明**

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
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>cuSeqLensQOptional</td>
      <td>输入</td>
      <td>表示不同batch中query的累积sequence length。<br>TND场景下必传，并可通过该入参shape推导batch。<br>第一个值固定为0。</td>
      <td>支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional</td>
      <td>输入</td>
      <td>表示不同batch中key的累积sequence length。<br>TND场景下必传。<br>第一个值固定为0。</td>
      <td>支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>seqUsedQOptional</td>
      <td>输入</td>
      <td>预留接口参数，表示不同batch中query实际参与运算的sequence length。</td>
      <td>支持空Tensor。当前kernel路径暂不使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>seqUsedKOptional</td>
      <td>输入</td>
      <td>预留接口参数，表示不同batch中key实际参与运算的sequence length。</td>
      <td>支持空Tensor。当前kernel路径暂不使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional</td>
      <td>输入</td>
      <td>预留接口参数，表示不同batch中key的sequence length与cmpRatio相关的残差。</td>
      <td>支持空Tensor。当前kernel路径暂不使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>batchSize</td>
      <td>输入</td>
      <td>表示batch数量。</td>
      <td>支持非负数。TND场景可填0，并通过cuSeqLensQOptional推导。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenQ</td>
      <td>输入</td>
      <td>表示query的最大sequence length。</td>
      <td>支持非负数。BSND场景必须为正数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenK</td>
      <td>输入</td>
      <td>表示key的最大sequence length。</td>
      <td>支持非负数。BSND场景必须为正数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsQ</td>
      <td>输入</td>
      <td>表示query的head个数。</td>
      <td>必须为正数，并且能被numHeadsK整除。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK</td>
      <td>输入</td>
      <td>表示key的head个数。</td>
      <td>必须为正数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>输入</td>
      <td>表示q/k的head dimension。</td>
      <td>必须为正数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>输入</td>
      <td>表示从key中筛选出的关键token个数。</td>
      <td>必须为正数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>输入</td>
      <td>表示query侧的排列格式。</td>
      <td>支持BSND/TND，默认值为BSND。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutK</td>
      <td>输入</td>
      <td>表示key侧的排列格式。</td>
      <td>支持BSND/TND，默认值为BSND。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode</td>
      <td>输入</td>
      <td>表示sparse mask模式。</td>
      <td>0: No mask<br>3: Causal</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio</td>
      <td>输入</td>
      <td>表示key的压缩率。</td>
      <td>取值范围[0, 128]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>表示负载均衡结果输出。</td>
      <td>输出结果作为aclnnSparseLightningIndexerKLLossGrad的metadataOptional输入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(64,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>当前实现返回0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnSparseLightningIndexerKLLossGradMetadata

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
        <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
        </tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

  - aclnnSparseLightningIndexerKLLossGradMetadata为确定性实现，确定性计算配置不会改变其输出规则。

  - BSND场景
    - 必传batchSize、maxSeqLenQ、maxSeqLenK和topk参数，以获取shape信息。
  - TND场景
    - 必传cuSeqLensQOptional、cuSeqLensKOptional和topk参数，以获取正确shape信息。
    - 当batchSize为0时，通过cuSeqLensQOptional的shape推导batch。

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
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>无mask。</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
            <td>支持</td>
        </tr>
        </tbody>
    </table>
</details>

<details>
<summary><a id="特殊约束"></a>特殊约束</summary>

  - Batch取值规则
    - 如果batchSize大于0，优先使用batchSize。
    - 如果batchSize小于等于0，且layoutQ为TND，则通过cuSeqLensQOptional的shape推导batch。
    - 如果batchSize小于等于0，且layoutQ为BSND，则报错。
  - Seqlen取值规则
    - TND场景下，通过cuSeqLensQOptional和cuSeqLensKOptional计算每个batch的实际q/k长度。
    - BSND场景下，通过maxSeqLenQ和maxSeqLenK获取q/k长度。
  - layout约束
    - layoutQ必须为BSND或TND。
    - layoutK支持BSND和TND，建议与layoutQ保持一致。
  - head约束
    - numHeadsQ、numHeadsK和headDim必须为正数。
    - numHeadsQ必须能被numHeadsK整除。
  - sparse约束
    - topk必须为正数。
    - cmpRatio取值范围为[0, 128]。
    - maskMode当前仅支持0和3。

</details>

<details>
<summary><a id="Metadata"></a>Metadata输出布局</summary>

  metadata输出为INT32 Tensor，当前shape固定为(64,)，字段布局如下。

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>字段</th>
        <th>index</th>
        <th>说明</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>coreNum</td>
        <td>0</td>
        <td>主kernel实际使用的cube core数量。</td>
        </tr>
        <tr>
        <td>totalSize</td>
        <td>1</td>
        <td>B/S1合轴后的任务总行数。</td>
        </tr>
        <tr>
        <td>splitFactorSize</td>
        <td>2</td>
        <td>均分场景下的初始分核粒度。</td>
        </tr>
        <tr>
        <td>reserved</td>
        <td>3-7</td>
        <td>预留字段，当前置0。</td>
        </tr>
        <tr>
        <td>bS1Index</td>
        <td>8-32</td>
        <td>每个cube core的起始B/S1合轴索引，非活跃core填充totalSize。</td>
        </tr>
    </tbody>
  </table>

</details>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参见[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

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
#include "aclnnop/aclnn_sparse_lightning_indexer_kl_loss_grad_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

constexpr uint32_t SLI_METADATA_MAX_CORE_NUM = 25;
constexpr uint32_t SLI_METADATA_HEADER_SIZE = 8;
constexpr uint32_t SLI_METADATA_SIZE = 64;

struct SliGradKLLossMetaData {
    int32_t coreNum;
    int32_t totalSize;
    int32_t splitFactorSize;
    int32_t reserved[SLI_METADATA_HEADER_SIZE - 3];
    int32_t bS1Index[SLI_METADATA_MAX_CORE_NUM];
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
    Tensor metadata {};
    int64_t batchSize { 0 };
    int64_t maxSeqLenQ { 0 };
    int64_t maxSeqLenK { 0 };
    int64_t numHeadsQ { 8 };
    int64_t numHeadsK { 1 };
    int64_t headDim { 128 };
    int64_t topk { 512 };
    char *layoutQ { nullptr };
    char *layoutK { nullptr };
    int64_t maskMode { 3 };
    int64_t cmpRatio { 4 };
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

    ret = CreateTensor(aclDataType::ACL_INT32, { SLI_METADATA_SIZE }, context.metadata);
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

void PrintMetadata(const SliGradKLLossMetaData &metadata)
{
    printf("coreNum          : %d\n", metadata.coreNum);
    printf("totalSize        : %d\n", metadata.totalSize);
    printf("splitFactorSize  : %d\n", metadata.splitFactorSize);
    for (uint32_t i = 0; i < std::min<uint32_t>(metadata.coreNum, SLI_METADATA_MAX_CORE_NUM); ++i) {
        printf("bS1Index[%u]      : %d\n", i, metadata.bS1Index[i]);
    }
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
    ret = aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
        context.cuSeqLensQOptional.data, context.cuSeqLensKOptional.data, context.seqUsedQOptional.data,
        context.seqUsedKOptional.data, context.cmpResidualKOptional.data, context.batchSize, context.maxSeqLenQ,
        context.maxSeqLenK, context.numHeadsQ, context.numHeadsK, context.headDim, context.topk, context.layoutQ,
        context.layoutK, context.maskMode, context.cmpRatio, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d", ret);

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

    ret = aclnnSparseLightningIndexerKLLossGradMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseLightningIndexerKLLossGradMetadata failed. ERROR: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d", ret);

    SliGradKLLossMetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    PrintMetadata(result);

    return 0;
}
```

## 调用说明

| 调用方式  | 样例代码 | 说明 |
| ----------- | -------- | ---- |
| aclnn接口 | [调用示例](#调用示例) | 通过`aclnnSparseLightningIndexerKLLossGradMetadata`接口方式调用，输出metadata供`aclnnSparseLightningIndexerKLLossGrad`使用。 |
