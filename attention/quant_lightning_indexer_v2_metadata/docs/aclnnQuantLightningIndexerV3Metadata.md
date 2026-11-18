# aclnnQuantLightningIndexerV3Metadata

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：`aclnnQuantLightningIndexerV3Metadata`是`aclnnQuantLightningIndexerV3`的前置算子，用于生成负载均衡的任务划分方案。本算子根据序列长度、head配置和计算属性，在AI CPU上计算各AI Core的任务起止范围，不执行实际的QuantLightningIndexer计算。

  本接口需与aclnnQuantLightningIndexerV3配套使用，主要过程如下：

  1. 根据输入shape、序列信息及计算属性划分任务，并估算各任务的计算量。
  2. 将任务分配到可用计算核，生成各核的任务起止信息。
  3. 输出各AIC核基本块的起止位置及AIV核的FD任务信息，并将metadata传入aclnnQuantLightningIndexerV3，用于后续计算。

- 与V2Metadata的关系：在[aclnnQuantLightningIndexerV2Metadata](aclnnQuantLightningIndexerV2Metadata.md)的参数基础上，新增candidateTopkBlocks、candidateBlockSize两个属性。两项仅支持(-1,-1)和(2048,8)，必须与V3计算接口一致。即使关闭候选选择，也必须使用本接口生成的metadata。

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize”接口获取计算所需workspace大小及执行器，再调用“aclnnQuantLightningIndexerV3Metadata”接口执行计算。

``` cpp
aclnnStatus aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize(
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
    int64_t            candidateTopkBlocks,
    int64_t            candidateBlockSize,
    const aclTensor   *metadata,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

``` cpp
aclnnStatus aclnnQuantLightningIndexerV3Metadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize

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
      <td>表示不同Batch中q的Sequence Length累加前缀和。</td>
      <td><ul><li>layoutQOptional为TND时必须传入，为BSND时不允许传入。</li><li>shape固定为(B+1, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqlensKOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中k的Sequence Length累加前缀和。</td>
      <td><ul><li>layoutKOptional为PA_BBND时，不允许传入。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedQOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中q实际参与运算的Sequence Length。</td>
      <td><ul><li>支持空Tensor。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedKOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中k实际参与运算的Sequence Length。</td>
      <td><ul><li>layoutKOptional为PA_BBND时必须传入。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示不同Batch中k压缩前Sequence Length除以cmpRatio的余数，配合cmpRatio还原压缩前长度。</td>
      <td><ul><li>支持空Tensor。</li><li>cmpRatio不为1，且mask为3场景下必传。</li><li>shape固定为(B, )。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numHeadsQ（int64_t）</td>
      <td>输入</td>
      <td>表示q的head个数。</td>
      <td>当前支持32、64。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK（int64_t）</td>
      <td>输入</td>
      <td>表示k的head个数。</td>
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
      <td>表示为每个q token保留的k token索引数量。</td>
      <td>当前仅支持[1, 2048]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>表示量化模式。</td>
      <td>当前仅支持2（Per-Token-Head量化），q、k的数据类型为INT8，w、qDescale、kDescale的数据类型为FLOAT16。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize（int64_t）</td>
      <td>输入</td>
      <td>表示Batch数量。</td>
      <td><ul><li>支持非负数。</li><li>须与序列输入的Batch数量一致。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ（int64_t）</td>
      <td>输入</td>
      <td>表示q的最长Sequence Length。</td>
      <td><ul><li>取值范围≥-1，-1表示任意可能长度。</li><li>建议值为-1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenK（int64_t）</td>
      <td>输入</td>
      <td>表示k的最长Sequence Length。</td>
      <td><ul><li>取值范围≥-1，-1表示任意可能长度。</li><li>建议值为-1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional（char*）</td>
      <td>输入</td>
      <td>表示q的排列格式。</td>
      <td><ul><li>支持 BSND、TND。</li><li>建议值为BSND。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional（char*）</td>
      <td>输入</td>
      <td>表示k的排列格式。</td>
      <td><ul><li>仅支持PA_BBND。</li></ul></td>
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
      <td>表示k的压缩率。</td>
      <td><ul><li>仅支持2的幂次方且范围为[1, 128]，即1/2/4/8/16/32/64/128。</li><li>建议值1，表示无压缩。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>candidateTopkBlocks（int64_t）</td>
      <td>输入</td>
      <td>两级TopK候选源(source)模式的候选块数量，即块级Top-k需要保留的候选块个数。</td>
      <td><ul><li>-1表示关闭candidate模式。</li><li>开启时仅支持2048。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>candidateBlockSize（int64_t）</td>
      <td>输入</td>
      <td>候选块大小，即压缩后K空间中每个候选块包含的位置数。</td>
      <td><ul><li>candidateTopkBlocks为-1时，必须为-1。</li><li>candidateTopkBlocks为2048时，必须为8。</li><li>需与aclnnQuantLightningIndexerV3的配置保持一致。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata（aclTensor*）</td>
      <td>输出</td>
      <td>表示负载均衡结果输出。</td>
      <td>shape固定为(1024, )。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维</td>
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
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建aclOpExecutor失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>创建metadata计算任务失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>参数cuSeqlensQOptional、cuSeqlensKOptional、sequsedQOptional、sequsedKOptional、cmpResidualKOptional、numHeadsQ、numHeadsK、headDim、topk、quantMode、batchSize、maxSeqlenQ、maxSeqlenK、layoutQOptional、layoutKOptional、maskMode、cmpRatio、candidateTopkBlocks、candidateBlockSize、metadata的规格不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

## aclnnQuantLightningIndexerV3Metadata

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
            <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize获取</td>
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

  - aclnnQuantLightningIndexerV3Metadata默认确定性实现。
  - Batch一致性说明：本接口生成当前批次的分核信息，不涉及计算结果的Batch一致性；批次的大小、顺序或序列信息变化后需重新生成metadata，不要求不同批次的metadata内容相同。
  - B（Batch）表示输入样本批量大小，q、k为配套的aclnnQuantLightningIndexerV3算子的入参，S1表示layoutQOptional=BSND时，q shape中的S轴的大小。
  - 参数cuSeqlensQOptional要求其值为当前Batch与前序Batch有效token数的累加值，第一个元素固定为0，后一个元素的值必须大于等于前一个元素的值。
  - 参数sequsedQOptional、sequsedKOptional要求其值表示每个Batch中的有效token数。
  - 参数cmpResidualKOptional需满足0 <= cmpResidualKOptional[i] < cmpRatio。
  - layoutQOptional=BSND场景
    - maxSeqlenQ必须传入S1的值。
  - layoutQOptional=TND场景
    - cuSeqlensQOptional必须传入。
  - layoutKOptional=PA_BBND场景
    - sequsedKOptional必须传入。
  - batchSize须与所有序列输入的Batch数量一致。
  - q Seqlen取值规则
    - layoutQOptional为BSND时，优先通过sequsedQOptional中的元素获取seqlen，sequsedQOptional未传入则通过maxSeqlenQ获取seqlen。
    - layoutQOptional为TND时，优先通过sequsedQOptional中的元素获取seqlen，sequsedQOptional未传入则通过cuSeqlensQOptional中相邻元素的差值获取seqlen。
  - candidateTopkBlocks、candidateBlockSize只支持(-1, -1)或(2048, 8)，需与aclnnQuantLightningIndexerV3接口的配置保持一致。
  - 序列信息、head配置、topk、quantMode、布局、maskMode、cmpRatio或candidate参数变化后，需重新生成metadata。candidate关闭时也需使用本接口生成的metadata。

  - 所有输入、输出Tensor须位于同一NPU。

## 调用示例

Q使用TND布局，K使用PA_BBND布局，quantMode=2，candidateTopkBlocks=2048，candidateBlockSize=8。生成的metadata可直接用于具有相同序列信息和计算属性的QLI V3调用。

```cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v3_metadata.h"


#define CHECK_RET(expr)                                      \
    do {                                                     \
        int ret = (expr);                                     \
        if (ret != ACL_SUCCESS) {                            \
            std::printf("%s failed: %d\n", #expr, ret);        \
            return ret;                                      \
        }                                                    \
    } while (0)

struct Resources {
    aclrtStream stream;
    std::vector<aclTensor *> tensors;
    std::vector<void *> allocations;

    explicit Resources(aclrtStream inputStream) : stream(inputStream) {}
    ~Resources()
    {
        // 异常返回时也先等待已提交任务，随后释放Tensor描述和Device内存。
        aclrtSynchronizeStream(stream);
        for (auto tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto address : allocations) {
            aclrtFree(address);
        }
    }

    int Allocate(uint64_t bytes, void **address)
    {
        CHECK_RET(aclrtMalloc(address, std::max<uint64_t>(bytes, 1), ACL_MEM_MALLOC_HUGE_FIRST));
        allocations.push_back(*address);
        return ACL_SUCCESS;
    }

    template <typename T>
    int CreateTensor(const std::vector<T> &host, const std::vector<int64_t> &shape,
                     aclDataType dtype, aclTensor **tensor, void **address)
    {
        uint64_t bytes = host.size() * sizeof(T);
        CHECK_RET(Allocate(bytes, address));
        if (bytes != 0) {
            CHECK_RET(aclrtMemcpy(*address, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        }
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), *address);
        if (*tensor == nullptr) {
            std::printf("aclCreateTensor failed\n");
            return 1;
        }
        tensors.push_back(*tensor);
        return ACL_SUCCESS;
    }
};

int RunExample(aclrtStream stream)
{
    Resources resources(stream);
    constexpr int64_t batch = 2;
    constexpr int64_t queryHeads = 32;
    constexpr int64_t keyHeads = 1;
    constexpr int64_t headDim = 128;
    constexpr int64_t keyLength = 2048;
    constexpr int64_t topk = 2048;
    constexpr int64_t candidateCapacity = 2048;
    constexpr int64_t candidateBlockSize = 8;
    char layoutQ[] = "TND";
    char layoutK[] = "PA_BBND";

    // 每个batch一个Query，Key有效长度均为2048；metadata使用固定1024个INT32元素。
    aclTensor *cuQ = nullptr;
    aclTensor *usedK = nullptr;
    aclTensor *metadata = nullptr;
    void *cuQAddress = nullptr;
    void *usedKAddress = nullptr;
    void *metadataAddress = nullptr;
    CHECK_RET(resources.CreateTensor<int32_t>({0, 1, 2}, {batch + 1}, ACL_INT32, &cuQ, &cuQAddress));
    CHECK_RET(resources.CreateTensor<int32_t>({keyLength, keyLength}, {batch}, ACL_INT32,
                                            &usedK, &usedKAddress));
    CHECK_RET(resources.CreateTensor<int32_t>(std::vector<int32_t>(1024), {1024}, ACL_INT32,
                                            &metadata, &metadataAddress));

    // 先生成metadata；后续计算使用完全相同的序列信息及属性。
    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;
    CHECK_RET(aclnnQuantLightningIndexerV3MetadataGetWorkspaceSize(
        cuQ, nullptr, nullptr, usedK, nullptr, queryHeads, keyHeads, headDim,
        topk, 2, batch, 1, keyLength, layoutQ, layoutK, 0, 1,
        candidateCapacity, candidateBlockSize, metadata, &metadataWorkspaceSize, &metadataExecutor));
    void *metadataWorkspace = nullptr;
    if (metadataWorkspaceSize != 0) {
        CHECK_RET(resources.Allocate(metadataWorkspaceSize, &metadataWorkspace));
    }
    CHECK_RET(aclnnQuantLightningIndexerV3Metadata(
        metadataWorkspace, metadataWorkspaceSize, metadataExecutor, stream));

    CHECK_RET(aclrtSynchronizeStream(stream));
    std::printf("metadata generated: INT32[1024]\n");
    return ACL_SUCCESS;
}

int main()
{
    constexpr int32_t deviceId = 0;
    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        aclFinalize();
        return ret;
    }
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret == ACL_SUCCESS) {
        ret = RunExample(stream);
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
```
