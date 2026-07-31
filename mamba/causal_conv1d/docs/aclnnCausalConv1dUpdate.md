# aclnnCausalConv1dUpdate

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mamba/causal_conv1d)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：x
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

- 接口功能：完成因果一维卷积（Causal Conv1d）的状态更新（decode / update）计算。算子内部支持 SiLU 激活、缓存索引（cacheIndices）、投机解码（numAcceptedTokens）等特性。

- 支持以下场景：
  - 场景一（decode/update场景 — 固定batch）：

    ```Cpp
    x: [batch, 1, dim]
    weight: [K, dim]，其中K∈{2,3,4}
    convStatesRef: [numCacheLines, stateLen, dim]
    biasOptional: [dim] 或 nullptr
    queryStartLocOptional: [batch+1] 或 nullptr
    cacheIndicesOptional: [batch] 或 nullptr（默认使用恒等映射）
    numAcceptedTokensOptional: [batch] 或 nullptr（投机解码场景，仅K=4支持）
    blockIdxLastScheduledTokenOptional: [batch] 或 nullptr
    initialStateIdxOptional: [batch] 或 nullptr
    activation: "silu" or "none"
    nullBlockId: int64_t
    maxQueryLen: int64_t
    y: [batch, 1, dim]
    ```

  - 场景二（decode/update场景 — 投机解码变长序列）：

    ```Cpp
    x: [cuSeqLen, dim]
    weight: [K, dim]，其中K∈{2,3,4}
    convStatesRef: [numCacheLines, stateLen, dim]
    biasOptional: [dim] 或 nullptr
    queryStartLocOptional: [batch+1] （必须提供）
    cacheIndicesOptional: [batch] 或 nullptr
    numAcceptedTokensOptional: [batch] （必须提供，仅K=4支持）
    blockIdxLastScheduledTokenOptional: [batch] 或 nullptr
    initialStateIdxOptional: [batch] 或 nullptr
    activation: "silu" or "none"
    nullBlockId: int64_t
    maxQueryLen: int64_t
    y: [cuSeqLen, dim]
    ```

    其中cuSeqLen为batch内所有变长序列拼接后的总长度。2D变长输入仅在提供numAcceptedTokens（投机解码）时支持，非投机解码的decode场景请使用3D输入[batch, 1, dim]。

- 计算公式：

  Causal Conv1d 是一种因果一维卷积算子，常用于序列建模中。在每个时间步 $t$，根据当前输入 $x_t$、卷积权重 $w$ 和历史状态，计算卷积输出 $y_t$。

  $$
  y_t = \text{Activation}\left(\sum_{j=0}^{W-1} w_j \cdot x_{t-j} + b\right)
  $$

  其中，$W$ 为卷积核宽度（支持2、3、4），$w_j$ 为卷积权重，$b$ 为偏置（可选），$\text{Activation}$ 为激活函数（可选，SiLU）。当 `activation="none"` 时不使用激活函数，`activation="silu"` 时使用 SiLU 激活函数。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnCausalConv1dUpdateGetWorkspaceSize` 接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用 `aclnnCausalConv1dUpdate` 接口执行计算。

```cpp
aclnnStatus aclnnCausalConv1dUpdateGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    aclTensor       *convStatesRef,
    const aclTensor *biasOptional,
    const aclTensor *queryStartLocOptional,
    const aclTensor *cacheIndicesOptional,
    const aclTensor *numAcceptedTokensOptional,
    const aclTensor *blockIdxLastScheduledTokenOptional,
    const aclTensor *initialStateIdxOptional,
    const char      *activation,
    int64_t         nullBlockId,
    int64_t         maxQueryLen,
    aclTensor       *y,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnCausalConv1dUpdate(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnCausalConv1dUpdateGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 230px">
  <col style="width: 100px">
  <col style="width: 320px">
  <col style="width: 400px">
  <col style="width: 180px">
  <col style="width: 80px">
  <col style="width: 130px">
  <col style="width: 110px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>计算公式中的x，代表输入序列。</td>
      <td><ul><li>不支持空tensor。</li><li>decode固定batch场景：shape为[batch, 1, dim]（3D）。</li><li>投机解码变长场景：shape为[cuSeqLen, dim]（2D），需同时提供numAcceptedTokens。非投机解码的decode场景请使用3D输入[batch, 1, dim]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>计算公式中的w，代表因果1维卷积核。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[K, dim]，K∈{2,3,4}。</li></ul></td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>convStatesRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>缓存状态张量，存储各序列的历史token数据，各序列计算完成后原地更新为最新状态。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[numCacheLines, stateLen, dim]。</li><li>stateLen ≥ K-1。</li><li>numCacheLines ≥ batch（未提供cacheIndices时）。</li><li>每次调用后，算子将最新状态写回此张量。</li></ul></td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>卷积的偏置。</td>
      <td>shape为[dim]。</td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>queryStartLocOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>序列起始位置索引，记录各序列在拼接张量x中的起始位置。变长序列场景必须提供。提供时，batch大小由queryStartLoc.size(0)-1决定，而非由x的shape推导。</td>
      <td>shape为[batch+1]。queryStartLoc[0]必须为0，queryStartLoc[-1]必须等于cuSeqLen，值必须非递减。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cacheIndicesOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>缓存索引，指定每个序列对应的缓存状态在convStates中的索引。不传时使用恒等映射。</td>
      <td>shape为[batch]。值∈[0, numCacheLines)，或等于nullBlockId表示跳过该序列。nullBlockId可被多个序列重复使用（padding场景）；除nullBlockId外的值不应重复，否则多个序列读写同一缓存槽位会导致历史状态错误和写入冲突。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numAcceptedTokensOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>投机解码中每个batch已接受的token数，用于确定从convStates中读取历史状态的起始位置。仅K=4支持，不传时从convStates的起始位置开始读取历史。</td>
      <td>shape为[batch]，dtype为INT32。取值范围[0, stateLen - (K - 1)]，其中stateLen为convStates.shape[1]，K为weight.shape[0]。算子内部计算offset = max(numAcceptedTokens[i] - 1, 0)，从convStates的第offset个位置开始读取历史状态进行卷积计算。超出此范围会导致convStates越界访问。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockIdxLastScheduledTokenOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>最后调度token块索引，指向cacheIndices中最后一个待填充缓存块的位置。</td>
      <td>shape为[batch]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>initialStateIdxOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>初始状态索引，指向cacheIndices中初始状态所在位置。</td>
      <td>shape为[batch]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>activation（const char*）</td>
      <td>输入</td>
      <td>激活函数类型。</td>
      <td>"silu"：SiLU；"none"：无激活。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nullBlockId（int64_t）</td>
      <td>输入</td>
      <td>用于跳过不需要参与计算的batch。</td>
      <td>当cacheIndices[i]==nullBlockId时跳过该batch，对应输出填零。用户应设置为一个不会出现在cacheIndices中的值（如-1），以避免误跳过有效batch。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxQueryLen（int64_t）</td>
      <td>输入</td>
      <td>最大查询长度。</td>
      <td>-1表示不限制。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>计算公式中的y，代表卷积输出序列。</td>
      <td><ul><li>decode固定batch：shape为[batch, 1, dim]。</li><li>投机解码变长序列：shape为[cuSeqLen, dim]。</li></ul></td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>传入的x、weight、convStatesRef、y是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>
      输入和输出的数据类型不在支持的范围内。<br>
      x、weight、convStatesRef、biasOptional、y的数据类型不一致。<br>
      cacheIndicesOptional、queryStartLocOptional、numAcceptedTokensOptional等的数据类型不是INT32。<br>
      输入、输出Tensor的shape不在支持的范围内。<br>
      输入的属性不在支持的范围内。<br>
      dim不满足对齐要求（需满足 (dim * dtypeSize) % 32 == 0）。<br>
      K不在[2,4]范围内。<br>
      变长场景未提供queryStartLocOptional。<br>
      K≠4时提供了numAcceptedTokensOptional。<br>
      </td>
    </tr>
  </tbody></table>

## aclnnCausalConv1dUpdate

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCausalConv1dUpdateGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnCausalConv1dUpdate默认确定性实现。

- 输入shape限制：
  - decode场景（固定batch）：
    - x为3维[batch, 1, dim]。
    - weight为2维[K, dim]，K∈{2,3,4}。
    - convStatesRef为3维[numCacheLines, stateLen, dim]，stateLen ≥ K-1，numCacheLines ≥ batch。
    - dim范围[64, 16384]且满足 (dim * dtypeSize) % 32 == 0，batch范围[1, 1024]。
  - decode场景（投机解码变长序列）：
    - x为2维[cuSeqLen, dim]，必须同时提供numAcceptedTokens。
    - queryStartLocOptional为1维[batch+1]，必须提供。
    - convStatesRef为3维[numCacheLines, stateLen, dim]，numCacheLines ≥ batch。
    - cuSeqLen范围[batch, batch×8]。
  - 非投机解码的decode场景不支持2D变长输入，请使用3D输入[batch, 1, dim]。

- 输入值域限制：
  - queryStartLocOptional[0]必须为0，queryStartLocOptional[-1]必须等于cuSeqLen，值必须非递减。
  - cacheIndicesOptional中的值∈[0, numCacheLines)，或等于nullBlockId表示跳过。nullBlockId可重复；除nullBlockId外的值不应重复，否则多个序列读写同一缓存槽位会导致历史状态错误和写入冲突。
  - numAcceptedTokensOptional仅在K=4时支持，值∈[0, stateLen - (K - 1)]，其中stateLen为convStates.shape[1]。

- 卷积核宽度K仅支持2、3、4。
- K为运行时参数。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d_update.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(const char *name, std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<aclFloat16> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    LOG_PRINT("%s (first 5 elements):\n", name);
    for (int64_t i = 0; i < 5 && i < size; i++) {
        LOG_PRINT("  [%ld] = %f\n", i, (double)aclFloat16ToFloat(resultData[i]));
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *convStatesDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *convStates = nullptr;
    aclTensor *y = nullptr;

    int32_t batchSize = 4;
    int32_t dim = 512;
    int32_t width = 2;
    int32_t stateLen = 4;

    std::vector<int64_t> xShape = {batchSize, 1, dim};
    std::vector<int64_t> weightShape = {width, dim};
    std::vector<int64_t> biasShape = {dim};
    std::vector<int64_t> convStatesShape = {batchSize, stateLen, dim};
    std::vector<int64_t> yShape = {batchSize, 1, dim};

    std::vector<int16_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<int16_t> weightHostData(GetShapeSize(weightShape), 1);
    std::vector<int16_t> biasHostData(GetShapeSize(biasShape), 0);
    std::vector<int16_t> convStatesHostData(GetShapeSize(convStatesShape), 0);
    std::vector<int16_t> yHostData(GetShapeSize(yShape), 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_BF16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(convStatesHostData, convStatesShape, &convStatesDeviceAddr, aclDataType::ACL_BF16, &convStates);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    const char *activation = "silu";
    int64_t nullBlockId = 0;
    int64_t maxQueryLen = -1;

    ret = aclnnCausalConv1dUpdateGetWorkspaceSize(x, weight, convStates, bias, nullptr, nullptr, nullptr,
                                                  nullptr, nullptr,
                                                  activation, nullBlockId, maxQueryLen,
                                                  y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dUpdateGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnCausalConv1dUpdate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dUpdate failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResult("convStates (in-place updated)", convStatesShape, &convStatesDeviceAddr);
    PrintOutResult("y", yShape, &yDeviceAddr);

    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(convStates);
    aclDestroyTensor(y);

    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(convStatesDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
