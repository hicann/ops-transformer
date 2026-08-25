# aclnnDenseLightningIndexerSoftmaxLseV2Metadata

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：该算子为AICPU算子，是aclnnDenseLightningIndexerSoftmaxLseV2算子的前置算子。根据aclnnDenseLightningIndexerSoftmaxLseV2算子的输入shape、layout、mask和压缩比例信息，计算并输出分核切分metadata。输出结果可作为aclnnDenseLightningIndexerSoftmaxLseV2算子的metadataOptional输入，减少主算子tiling阶段对host array的访问。

  **该算子不建议单独使用，建议与aclnnDenseLightningIndexerSoftmaxLseV2算子配合使用，形成完整的工作流。**
    1. 接收主算子的shape信息，包括batchSize、maxSeqLenQ、maxSeqLenK、numHeadsQ、numHeadsK、headDim、layout和mask信息。
    2. 根据layout和可选输入seqUsedQOptional计算参与负载均衡的seq总数，采用strided方式将任务均衡切分到可用AIC核上。
    3. 输出metadata后，后续作为aclnnDenseLightningIndexerSoftmaxLseV2算子的metadataOptional输入使用。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize"获取workspace大小，再调用"aclnnDenseLightningIndexerSoftmaxLseV2Metadata"执行计算。

``` cpp
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize(
    const aclTensor   *cuSeqLensQOptional,
    const aclTensor   *cuSeqLensKOptional,
    const aclTensor   *seqUsedQOptional,
    const aclTensor   *seqUsedKOptional,
    const aclTensor   *cmpResidualKOptional,
    int64_t            batchSize,
    int64_t            maxSeqLenQ,
    int64_t            maxSeqLenK,
    int64_t            numHeadsQ,
    int64_t            numHeadsK,
    int64_t            headDim,
    char              *layoutQ,
    char              *layoutK,
    int64_t            maskMode,
    int64_t            cmpRatio,
    const aclTensor   *metadata,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

``` cpp
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2Metadata(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize

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
      <td>cuSeqLensQOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中query的累积sequence length。<br>TND场景下必传，并可通过该入参shape推导batch。<br>第一个值固定为0。</td>
      <td>支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中key的累积sequence length。<br>TND场景下必传。<br>第一个值固定为0。</td>
      <td>支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seqUsedQOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中query实际参与运算的sequence length。</td>
      <td>支持空Tensor。当该入参存在时，seq总数按每个batch实际用到的seq累加计算。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seqUsedKOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中key实际参与运算的sequence length。</td>
      <td>支持空Tensor。当该入参存在时，seqused_k的值不大于各batch的实际seqlen_k。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional（const aclTensor*）</td>
      <td>输入</td>
      <td>表示不同batch中key的sequence length与cmpRatio相关的残差。</td>
      <td>支持空Tensor。当maskMode=3且cmpRatio>1时必须传入，且每个值必须小于cmpRatio。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape为(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>表示batch数量。</td>
      <td>支持非负数。TND场景可填0，并通过cuSeqLensQOptional推导。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenQ（int64_t）</td>
      <td>输入</td>
      <td>表示query的最大sequence length。</td>
      <td>支持非负数。BSND场景必须为正数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenK（int64_t）</td>
      <td>输入</td>
      <td>表示key的最大sequence length。</td>
      <td>支持非负数。BSND场景必须为正数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>表示query的head个数。</td>
      <td>必须为正数，取值范围[1, 128]，并且能被numHeadsK整除。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK（int64_t）</td>
      <td>输入</td>
      <td>表示key的head个数。</td>
      <td>必须为正数。当前仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim（int64_t）</td>
      <td>输入</td>
      <td>表示q/k的head dimension。</td>
      <td>必须为正数。当前仅支持128。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ（char*）</td>
      <td>输入</td>
      <td>表示query侧的排列格式。</td>
      <td>支持BSND/TND，传空指针时为BSND。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutK（char*）</td>
      <td>输入</td>
      <td>表示key侧的排列格式。</td>
      <td>支持BSND/TND，传空指针时为BSND。必须与layoutQ保持一致。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode（int64_t）</td>
      <td>输入</td>
      <td>表示sparse mask模式。</td>
      <td>0: No mask<br>3: Causal</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>表示key的压缩率。</td>
      <td>取值范围[1, 128]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata（const aclTensor*）</td>
      <td>输出</td>
      <td>表示负载均衡结果输出。</td>
      <td>输出结果作为aclnnDenseLightningIndexerSoftmaxLseV2的metadataOptional输入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(64,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>当前实现返回0。</td>
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

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>metadata、workspaceSize、executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>metadata的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>numHeadsQ、numHeadsK、headDim不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>layoutQ与layoutK不一致，或不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>batchSize、maxSeqLenQ、maxSeqLenK不满足取值约束。</td>
    </tr>
    <tr>
      <td>maskMode不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>cmpRatio不在支持的范围之内。</td>
    </tr>
    </tbody></table>

## aclnnDenseLightningIndexerSoftmaxLseV2Metadata

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  - aclnnDenseLightningIndexerSoftmaxLseV2Metadata为确定性实现，确定性计算配置不会改变其输出规则。

  - BSND场景
    - 必传batchSize、maxSeqLenQ和maxSeqLenK参数，以获取shape信息。
  - TND场景
    - 必传cuSeqLensQOptional和cuSeqLensKOptional参数，以获取正确shape信息。
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
    - layoutK支持BSND和TND，必须与layoutQ保持一致。
  - head约束
    - numHeadsQ取值范围为[1, 128]，必须能被numHeadsK整除。
    - numHeadsK当前仅支持1。
    - headDim当前仅支持128。
  - 负载均衡约束
    - cmpRatio取值范围为[1, 128]。
    - maskMode当前仅支持0和3。
  - cmp_residual_k约束
    - 当maskMode=3且cmpRatio>1时，必须传入cmpResidualKOptional。
    - cmpResidualKOptional的每个值必须小于cmpRatio。
  - seqused约束
    - seqUsedQOptional的每个值不大于各batch的实际seqlen_q。
    - seqUsedKOptional的每个值不大于各batch的实际seqlen_k。

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
        <td>forecore_num</td>
        <td>0</td>
        <td>前置核数，即处理较多任务的核数。当totalNum不能被aicCoreNum整除时，forecore_num = totalNum % aicCoreNum，每个前置核处理 ceil(totalNum / aicCoreNum) 个任务；当能整除时，forecore_num = aicCoreNum。</td>
        </tr>
        <tr>
        <td>tail_core_num</td>
        <td>1</td>
        <td>尾核数，即处理较少任务的核数。tail_core_num = aicCoreNum - forecore_num，每个尾核处理 floor(totalNum / aicCoreNum) 个任务。当totalNum能被aicCoreNum整除时为0。</td>
        </tr>
        <tr>
        <td>b_s1_per_core</td>
        <td>2</td>
        <td>前置核每核处理的 B*S1 任务数，即 ceil(totalNum / aicCoreNum)。</td>
        </tr>
        <tr>
        <td>b_s1_per_tail_core</td>
        <td>3</td>
        <td>尾核每核处理的 B*S1 任务数，即 floor(totalNum / aicCoreNum)。当totalNum能被aicCoreNum整除时为0。</td>
        </tr>
    </tbody>
  </table>

  采用前后核差异化分块切分方式。设totalNum为参与负载均衡的seq总数，N为aicCoreNum。当totalNum能被N整除时，forecore_num=N，tail_core_num=0，每核处理totalNum/N个任务。当不能整除时，设base=floor(totalNum/N)，rem=totalNum%N，则forecore_num=rem，每个前置核处理base+1个任务，tail_core_num=N-rem，每个尾核处理base个任务。例如totalNum=16，N=36时，16<36，forecore_num=16，tail_core_num=0，b_s1_per_core=1，前16个核各处理1个任务，其余核不工作。

</details>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

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
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse_v2_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

constexpr uint32_t DLI_METADATA_SIZE = 64;

struct DenseLISoftmaxLseV2MetaData {
    int32_t forecore_num;
    int32_t tail_core_num;
    int32_t b_s1_per_core;
    int32_t b_s1_per_tail_core;
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

    ret = CreateTensor(aclDataType::ACL_INT32, { DLI_METADATA_SIZE }, context.metadata);
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

    // cmp_residual_k is required when maskMode=3 and cmpRatio>1
    if (context.maskMode == 3 && context.cmpRatio > 1) {
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.cmpResidualKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKOptional failed. Error: %d", ret);
        SetInt32TensorData(context.cmpResidualKOptional, std::vector<int32_t>(batchSize, 0));
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

void PrintMetadata(const DenseLISoftmaxLseV2MetaData &metadata)
{
    printf("forecore_num      : %d\n", metadata.forecore_num);
    printf("tail_core_num     : %d\n", metadata.tail_core_num);
    printf("b_s1_per_core     : %d\n", metadata.b_s1_per_core);
    printf("b_s1_per_tail_core: %d\n", metadata.b_s1_per_tail_core);
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
    ret = aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize(
        context.cuSeqLensQOptional.data, context.cuSeqLensKOptional.data, context.seqUsedQOptional.data,
        context.seqUsedKOptional.data, context.cmpResidualKOptional.data, context.batchSize, context.maxSeqLenQ,
        context.maxSeqLenK, context.numHeadsQ, context.numHeadsK, context.headDim, context.layoutQ,
        context.layoutK, context.maskMode, context.cmpRatio, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize failed. ERROR: %d", ret);

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

    ret = aclnnDenseLightningIndexerSoftmaxLseV2Metadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnDenseLightningIndexerSoftmaxLseV2Metadata failed. ERROR: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d", ret);

    DenseLISoftmaxLseV2MetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    PrintMetadata(result);
    printf("pass\n");

    return 0;
}
```

## 问题定位说明

- 关于AI CPU算子Kernel常见执行问题或异常错误，问题定位方法请参考《故障处理》中“[故障案例集>算子执行问题>AI CPU算子Kernel执行报错](https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)”。
