# aclnnLightningIndexerV2Metadata

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer_v2_metadata)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    x     |
| <term>Atlas 推理系列产品</term>                              |    x     |
| <term>Atlas 训练系列产品</term>                              |    x     |

## 功能说明

- 接口功能：该接口为AI CPU算子接口，是aclnnLightningIndexerV2算子的前置算子接口。根据aclnnLightningIndexerV2算子接口的输入信息，计算并输出负载均衡结果。输出结果可以作为aclnnLightningIndexerV2算子接口的输入，减少aclnnLightningIndexerV2算子接口的执行耗时。

  **该算子不建议单独使用，建议与aclnnLightningIndexerV2算子配合使用，形成完整的工作流。**
    1. 接受aclnnLightningIndexerV2算子接口输入数据shape信息，包含batchSize、qSeqlen、kSeqlen、mask。通过对输入分块并模拟计算耗时，均匀分配分块到可用核上，以降低aclnnLightningIndexerV2算子的整体计算耗时，并提高硬件利用率。
    2. 分配结果输出后，后续作为输入供aclnnLightningIndexerV2算子使用。
    3. 分配结果包含每个AIC核基本块的起始点和终止点，已经每个AIV核的FD任务信息。详细内容可以参考[调用示例](#调用示例)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnLightningIndexerV2MetadataGetWorkspaceSize"获取workspace大小，在调用"aclnnLightningIndexerV2Metadata"执行计算

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

- **参数说明**

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
      <td>cuSeqlensQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中Query的有效Sequence Length。</td>
      <td><ul><li>支持空Tensor</li><li>layoutQOptional为TND场景下必传。</li><li>第一个值为额外值并固定为0。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensKOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中Key的有效Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>layoutKOptional为TND场景下必传。</li><li>第一个值为额外值并固定为0。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中Query实际参与运算的Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedKOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中Key实际参与运算的Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmpRatio实现cmp_kv部分的mask和负载计算。</td>
      <td><ul><li>支持空Tensor。</li><li>cmpRatio不为1，且mask为3场景下必传。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>表示Query的head个数。</td>
      <td>当前仅支持32/64。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK（int64_t）</td>
      <td>输入</td>
      <td>表示Key的head个数。</td>
      <td>当前仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim（int64_t）</td>
      <td>输入</td>
      <td>注意力头的维度。</td>
      <td>当前仅支持128。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk（int64_t）</td>
      <td>输入</td>
      <td>表示从Query中筛选出的关键稀疏token的个数。</td>
      <td>当前仅支持[1, 2048]</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>表示Batch数量。</td>
      <td><ul><li>支持非负数。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ（int64_t）</td>
      <td>输入</td>
      <td>表示Query的最长Sequence Length。</td>
      <td><ul><li>取值范围≥-1，-1表示任意可能长度。</li><li>建议值为-1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenK（int64_t）</td>
      <td>输入</td>
      <td>表示Key的最长Sequence Length。</td>
      <td><ul><li>取值范围≥-1，-1表示任意可能长度。</li><li>建议值为-1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional（char*）</td>
      <td>输入</td>
      <td>表示Query的排列格式。</td>
      <td><ul><li>支持 BSND、TND。</li><li>建议值为BSND。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional（char*）</td>
      <td>输入</td>
      <td>表示Key的排列格式。</td>
      <td><ul><li>支持 BSND、TND、PA_BBND。</li><li>建议值为BSND。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode（int64_t）</td>
      <td>输入</td>
      <td>表示sparse模式。</td>
      <td><ul><li>0: No mask。</li><li>3: rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li><li>建议值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>表示Key的压缩率。</td>
      <td><ul><li>取值范围[1，128]</li><li>建议值1，表示无压缩。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata（aclTensor*）</td>
      <td>输出</td>
      <td>表示负载均衡结果输出。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建aclOpExecutor失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>参数workspaceSize、executor是空指针，或参数cuSeqlensQOptional、cuSeqlensKOptional、sequsedQOptional、sequsedKOptional、cmpResidualKOptional进行Contiguous处理后为空指针。</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>参数cuSeqlensQOptional、cuSeqlensKOptional、sequsedQOptional、sequsedKOptional、cmpResidualKOptional、numHeadsQ、numHeadsK、headDim、topk、batchSize、maxSeqlenQ、maxSeqlenK、layoutQOptional、layoutKOptional、maskMode、cmpRatio的规格不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

## aclnnLightningIndexerV2Metadata

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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>workspace</td>
            <td>输入</td>
            <td>在Device侧申请的workspace内存地址</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>输入</td>
            <td>在Device侧申请的workspace大小，由第一段接口aclnnLightningIndexerV2MetadataGetWorkspaceSize获取</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>输入</td>
            <td>op执行器，包含了算子计算流程</td>
        </tr>
        <tr>
            <td>stream</td>
            <td>输入</td>
            <td>指定执行任务的Stream</td>
        </tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  - aclnnLightningIndexerV2Metadata默认确定性实现。
  - B（Batch）表示输入样本批量大小。
  - Batch取值规则
    - 优先获取sequsedQOptional中的Batch信息。
    - 如果未传入sequsedQOptional，优先获取cuSeqlensQOptional中的Batch信息。
    - 如果未传入sequsedQOptional，且layoutQOptional为TND，则必获取cuSeqlensQOptional中的Batch信息。
    - 除上所述，使用batchSize。
  - Sequence Length取值规则
    - 优先获取sequsedQOptional中的Sequence Length信息。
    - 如果未传入sequsedQOptional，且layoutQOptional为TND，则必获取cuSeqlensQOptional中的Sequence Length信息。
    - 除上所述，使用maxSeqlenQ。
    - Key与Query的获取规则一致。
  - layout约束
    - 当layoutKOptional为PA_BBND时，layoutQOptional可以任意取值。
    - 除上所述，layoutQOptional必须与layoutKOptional保持一致。
  - BSND场景
    - 当传入的layoutQOptional为"BSND"时，视为使用BSND场景。
    - 在未传入cuSeqlensQOptional和sequsedQOptional的情况下，必传batchSize、maxSeqlenQ、maxSeqlenK参数。
  - TND场景
    - 当传入的layoutQOptional为"TND"时，视为使用TND场景。
    - 必传cuSeqlensQOptional、cuSeqlensKOptional参数。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

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

// 参考 lightning_indexer_v2_metadata.h
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
    // 禁止拷贝
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
    // 固定写法，初始化
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
    // 调用aclrtMallocHost申请host侧内存
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // 调用aclCreateTensor接口创建aclTensor
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
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
    // 1.（固定写法）device/stream初始化，参考对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. 构造输入与输出，需要根据API的接口定义构造
    ArgScenario scenario {};
    scenario.hasCuSeq = true;
    scenario.hasSeqused = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. 调用CANN算子库API，需要修改为具体的API
    // 调用aclnnLightningIndexerV2Metadata第一段接口
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

    // 调用aclnnLightningIndexerV2Metadata第二段接口
    ret = aclnnLightningIndexerV2Metadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnLightningIndexerV2Metadata failed. ERROR: %d\n", ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. 打印输出
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
        printf("    First Worksapce Index : %u\n", result.faData[i][LI_V2_FIRST_LD_V2_DATA_WORKSPACE_IDX_INDEX]);
    }
    for (uint32_t i = 0; i < AIV_CORE_MAX_NUM; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result.fdData[i][LD_V2_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result.fdData[i][LD_V2_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result.fdData[i][LD_V2_M_IDX_INDEX]);
        printf("    FD Task S2 Idx          : %u\n", result.fdData[i][LD_V2_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdData[i][LD_V2_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdData[i][LD_V2_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdData[i][LD_V2_M_NUM_INDEX]);
    }

    return 0;
}
```
