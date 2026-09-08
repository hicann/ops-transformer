# aclnnKeyPool

[📄查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/key_pool)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

- 接口功能：KeyPool是推理场景下的Key压缩算子。算子对每个输入token分别执行K
投影和Gate投影，将连续的`cmpRatio`个token分为一组，按Gate与位置偏置
确定的权重将组内K合并为一个Key，并将未完成压缩组所需的中间状态写入分页
`stateCache`。

- 计算公式：

  对当前输入的每个token：

  ```text
  K = hiddenStates @ wk.T
  G = hiddenStates @ gateWeight.T
  K_norm = LayerNorm(K, dim=-1)  # 提供normWeightOptional/normBiasOptional时
  ```

  对第`q`个全局压缩组：

  ```text
  logits[q, r] = G[q * cmpRatio + r] + ape[r]
  weight = softmax(logits, dim=0)
  pooledKeyOut[b, q - startPool[b]] = sum(weight[r] * K_norm[q * cmpRatio + r], dim=0)
  ```

  历史token通过`cacheBlockTable`从`stateCache`读取，当前输入token的投影
结果优先使用当前调用产生的数据。对于`startPos`未按`cmpRatio`对齐的场景，
首个压缩组可以由历史token和当前输入token共同组成。`ape`只参与池化计算，
不写入`stateCache`。RoPE参数入口保留用于后续扩展，但当前版本不实现RoPE。

  LayerNorm沿每个token的最后一维`D`独立计算，不跨token归约。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnKeyPoolGetWorkspaceSize`接口获取入参并计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnKeyPool`接口执行计算。

```cpp
aclnnStatus aclnnKeyPoolGetWorkspaceSize(
    const aclTensor *hiddenStates,
    const aclTensor *wk,
    const aclTensor *gateWeight,
    const aclTensor *ape,
    aclTensor       *stateCacheRef,
    const aclTensor *cacheBlockTable,
    const aclTensor *startPos,
    const aclTensor *normWeightOptional,
    const aclTensor *normBiasOptional,
    const aclTensor *cosOptional,
    const aclTensor *sinOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    int64_t          cmpRatio,
    double           normEps,
    int64_t          rotaryMode,
    int64_t          stateCacheStrideDim0,
    aclTensor       *pooledKeyOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```cpp
aclnnStatus aclnnKeyPool(
    void          *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

`stateCacheRef`为原地更新输入，`pooledKeyOut`是唯一输出。

## aclnnKeyPoolGetWorkspaceSize

### 参数说明

<table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 260px">
  <col style="width: 380px">
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 230px">
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
  </tr>
</thead>
<tbody>
  <tr>
    <td>hiddenStates（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的hiddenStates，表示当前调用的hidden states。</td>
    <td><ul><li>支持BSH和TH两种输入布局。</li><li>支持B=0、S=0或T=0的空Tensor。</li><li>BSH场景shape为：[B,S,H]，TH场景shape为：[T,H]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2-3</td>
    <td>×</td>
  </tr>
  <tr>
    <td>wk（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的wk，表示K投影权重。</td>
    <td><ul><li>不支持空Tensor，数据类型必须与hiddenStates一致。</li><li>shape为[D,H]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
  </tr>
  <tr>
    <td>gateWeight（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的gateWeight，表示Gate投影权重。</td>
    <td><ul><li>不支持空Tensor，数据类型必须与hiddenStates一致。</li><li>shape为[D,H]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
  </tr>
  <tr>
    <td>ape（aclTensor*）</td>
    <td>输入</td>
    <td>公式中的ape，表示位置偏置，对应公式中的<span class="math-inline">Ape</span>。</td>
    <td><ul><li>不支持空Tensor，第一维必须与cmpRatio相等。</li><li>shape为[cmpRatio,D]。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
  </tr>
  <tr>
    <td>stateCacheRef（aclTensor*）</td>
    <td>输入/输出</td>
    <td>表示K和Gate的历史状态，并由算子原地更新。</td>
    <td><ul><li>不支持空Tensor，要求block_num&gt;0。</li><li>前D列保存LayerNorm后且未执行RoPE的K，后D列保存原始Gate。</li><li>支持第0轴非连续，stateCacheStrideDim0由调用方提供实际stride。</li><li>shape为[block_num,block_size,2*D]。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>3</td>
    <td>支持0轴非连续</td>
  </tr>
  <tr>
    <td>cacheBlockTable（aclTensor*）</td>
    <td>输入</td>
    <td>表示逻辑block到物理cache block的映射表。</td>
    <td><ul><li>不支持空Tensor，每个Batch对应一行逻辑block映射。</li><li>shape为[B,L]。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
  </tr>
  <tr>
    <td>startPos（aclTensor*）</td>
    <td>输入</td>
    <td>表示每个Batch当前输入在逻辑序列中的起始位置。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为[B]。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
  </tr>
  <tr>
    <td>normWeightOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示LayerNorm的权重参数。</td>
    <td><ul><li>不支持空Tensor。</li><li>与normBiasOptional必须同时提供或同时为空；提供时沿每个token的D维执行LayerNorm。</li><li>shape为[D]。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
  </tr>
  <tr>
    <td>normBiasOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示LayerNorm的偏置参数。</td>
    <td><ul><li>不支持空Tensor。</li><li>与normBiasOptional必须同时提供或同时为空；提供时沿每个token的D维执行LayerNorm。</li><li>shape为[D]。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
  </tr>
  <tr>
    <td>cosOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示RoPE使用的余弦参数。</td>
    <td><ul><li>不支持空Tensor。</li><li>保留接口，当前版本必须传空指针。</li></ul></td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>当前版本不适用</td>
    <td>×</td>
  </tr>
  <tr>
    <td>sinOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示RoPE使用的正弦参数。</td>
    <td><ul><li>不支持空Tensor。</li><li>保留接口，当前版本必须传空指针。</li></ul></td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>当前版本不适用</td>
    <td>×</td>
  </tr>
  <tr>
    <td>cuSeqlensOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>表示TH场景下各Batch在hiddenStates首轴上的累计token边界。</td>
    <td><ul><li>不支持空Tensor。</li><li>TH场景必须提供，首元素为0，末元素为T，且单调不减。</li><li>BSH场景必须传空指针。</li><li>TH场景shape为[B+1]。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
  </tr>
  <tr>
    <td>sequsedOptional（aclTensor*）</td>
    <td>可选输入</td>
    <td>预留的有效序列长度参数。</td>
    <td><ul><li>不支持空Tensor。</li><li>当前版本必须传空指针，不支持通过该参数选择参与计算的token。</li><li>shape为[B]。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1（预留）</td>
    <td>×</td>
  </tr>
  <tr>
    <td>cmpRatio（int64_t）</td>
    <td>输入</td>
    <td>公式中的cmpRatio，表示压缩率，即每组参与池化的token数。</td>
    <td>取值范围为{2,4,8,16,32,64,128}。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>normEps（double）</td>
    <td>输入</td>
    <td>表示LayerNorm的epsilon。</td>
    <td>默认值为1e-6，取值必须大于0。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>rotaryMode（int64_t）</td>
    <td>输入</td>
    <td>表示RoPE模式。</td>
    <td>取值范围为{0,1}；当前版本不执行RoPE，cosOptional和sinOptional必须为空。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stateCacheStrideDim0（int64_t）</td>
    <td>输入</td>
    <td>表示stateCache第0轴的stride。</td>
    <td>由调用方根据stateCache的实际stride传入。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pooledKeyOut（aclTensor*）</td>
    <td>输出</td>
    <td>表示按cmpRatio个token池化后的K。</td>
    <td><ul><li>支持B=0、S=0或T=0的空Tensor；输出为固定容量的BSH Tensor。</li><li>shape为[B,Sr,D]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
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
    <td>返回包含算子计算流程的执行器。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</tbody>
</table>

其中：

- `D`为K和Gate的head dimension，由`wk`的第0维决定；
- `R=cmpRatio`；
- `N`为物理block数；
- `BS`为block size；
- `L`为每个Batch的最大逻辑block数；
- `Sr=ceil(L*BS/cmpRatio)`，表示pooledKeyOut的容量。
- `BSH`：Batch-Sequence-Hidden的输入布局，`hiddenStates`的shape为
  `[B,S,H]`。其中`B`表示BatchSize，`S`表示每个Batch当前输入的序列
  长度，`H`表示hiddensize。
- `TH`：Token-Hidden的输入布局，`hiddenStates`的shape为`[T,H]`。
  其中`T`表示所有Batch当前输入token数的总和；`cuSeqlensOptional`
  是shape为`[B+1]`的前缀和数组，用于描述各Batch在首轴上的token边界。
- `RoPE`：Rotary Position Embedding，一种位置编码方式，通常使用`cos`
  和`sin`对K的最后一维中的成对通道执行旋转。本算子当前仅保留RoPE
  参数入口，不执行RoPE，`cosOptional`和`sinOptional`必须传空指针。

输入tensor使用ND格式。`stateCacheRef`是原地输入，用于保存跨算子调用
的未完成压缩组状态。

### 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
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
    <td>必选参数中存在空指针。</td>
  </tr>
  <tr>
    <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="4">161002</td>
    <td>输入输出Tensor的数据类型、shape或数据格式不在支持范围内。</td>
  </tr>
  <tr>
    <td>cmpRatio、normEps或rotaryMode的取值不在支持范围内。</td>
  </tr>
  <tr>
    <td>BSH/TH场景与cuSeqlensOptional、sequsedOptional的组合不满足约束。</td>
  </tr>
  <tr>
    <td>normWeightOptional与normBiasOptional未同时提供，或cosOptional与sinOptional未按当前版本要求传空。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_INNER_TILING_ERROR</td>
    <td>561002</td>
    <td>tiling发生异常，入参dtype、shape或属性组合错误。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>调用NPURuntime接口时发生异常，例如Runtime服务未启动或内存申请失败。</td>
  </tr>
</tbody>
</table>

## aclnnKeyPool

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
    <td>在 Device 侧申请的 workspace 内存地址。</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnKeyPoolGetWorkspaceSize 获取。</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>输入</td>
    <td>op 执行器，包含算子计算流程。</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>输入</td>
    <td>指定执行任务的 Stream。</td>
  </tr>
</tbody>
</table>

## 约束说明

- 支持BSH和TH两种场景：
  - BSH场景`cuSeqlensOptional`必须为空；
  - TH场景必须提供合法的`cuSeqlensOptional`前缀和数组，首元素为0，
    末元素为`T`，且单调不减。
- `cmpRatio`仅支持`2/4/8/16/32/64/128`；`normEps`必须大于0；
  `rotaryMode`仅支持`0/1`。
- `normWeightOptional`和`normBiasOptional`必须同时提供或同时为空。
- `cosOptional`和`sinOptional`必须同时为空，当前不支持RoPE；
  `sequsedOptional`必须为空，当前不支持通过`attention_mask`选择token。
- 支持BSH场景的`B=0`或`S=0`，以及TH场景的`T=0`。
- `cmpRatio`的默认值为4，取值范围是`2/4/8/16/32/64/128`。
- `normEps`的默认值为`1e-6`，取值范围是大于0。
- `rotaryMode`的默认值为1，取值范围是`0/1`，当前不执行RoPE。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

下面给出一个可直接编译运行的BSH调用示例。示例使用BF16的
`hiddenStates`、`wk`、`gateWeight`和`pooledKeyOut`，使用FP32的
`ape`、`stateCacheRef`、LayerNorm权重和偏置，并通过空指针关闭当前尚未
实现的RoPE。示例完整代码如下：

```cpp
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_key_pool.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

using BFloat16 = uint16_t;

BFloat16 FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<BFloat16>(bits >> 16);
}

float BFloat16ToFloat(BFloat16 value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (const auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续selfOrResult的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

void PrintBFloat16Tensor(const char *name, const std::vector<int64_t> &shape, void *deviceAddr)
{
    const auto elementCount = GetShapeSize(shape);
    std::vector<BFloat16> hostData(elementCount);
    const auto ret = aclrtMemcpy(hostData.data(), elementCount * sizeof(BFloat16), deviceAddr,
                                 elementCount * sizeof(BFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret); return);
    const auto printCount = std::min<int64_t>(elementCount, 8);
    for (int64_t i = 0; i < printCount; ++i) {
        LOG_PRINT("%s[%ld] = %f\n", name, i, BFloat16ToFloat(hostData[i]));
    }
}

} // namespace

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API
    // 根据自己的实际device填写deviceId
    constexpr int32_t deviceId = 0;
    constexpr int64_t batchSize = 1;
    constexpr int64_t seqLength = 8;
    constexpr int64_t hiddenSize = 4096;
    constexpr int64_t headDim = 128;
    constexpr int64_t cmpRatio = 4;
    constexpr int64_t blockSize = 4;
    constexpr int64_t maxBlockNumPerBatch = (seqLength + blockSize - 1) / blockSize;
    constexpr int64_t blockNum = batchSize * maxBlockNumPerBatch + 1; // block 0 is reserved.
    constexpr int64_t pooledSeqLength = (maxBlockNumPerBatch * blockSize + cmpRatio - 1) / cmpRatio;
    constexpr double normEps = 1e-6;
    constexpr int64_t rotaryMode = 1;
    constexpr int64_t stateCacheStrideDim0 = blockSize * 2 * headDim;

    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    const std::vector<int64_t> hiddenStatesShape = {batchSize, seqLength, hiddenSize};
    const std::vector<int64_t> wkShape = {headDim, hiddenSize};
    const std::vector<int64_t> gateWeightShape = {headDim, hiddenSize};
    const std::vector<int64_t> apeShape = {cmpRatio, headDim};
    const std::vector<int64_t> stateCacheShape = {blockNum, blockSize, 2 * headDim};
    const std::vector<int64_t> cacheBlockTableShape = {batchSize, maxBlockNumPerBatch};
    const std::vector<int64_t> startPosShape = {batchSize};
    const std::vector<int64_t> normShape = {headDim};
    const std::vector<int64_t> pooledKeyShape = {batchSize, pooledSeqLength, headDim};

    const auto hiddenStatesSize = GetShapeSize(hiddenStatesShape);
    const auto wkSize = GetShapeSize(wkShape);
    const auto gateWeightSize = GetShapeSize(gateWeightShape);
    const auto apeSize = GetShapeSize(apeShape);
    const auto stateCacheSize = GetShapeSize(stateCacheShape);
    const auto pooledKeySize = GetShapeSize(pooledKeyShape);

    std::vector<BFloat16> hiddenStatesHost(hiddenStatesSize);
    std::vector<BFloat16> wkHost(wkSize);
    std::vector<BFloat16> gateWeightHost(gateWeightSize);
    std::vector<float> apeHost(apeSize);
    std::vector<float> stateCacheHost(stateCacheSize, 0.0f);
    std::vector<int32_t> cacheBlockTableHost = {1, 2};
    std::vector<int32_t> startPosHost = {0};
    std::vector<float> normWeightHost(headDim, 1.0f);
    std::vector<float> normBiasHost(headDim, 0.0f);
    std::vector<BFloat16> pooledKeyHost(pooledKeySize, FloatToBFloat16(0.0f));

    for (int64_t i = 0; i < hiddenStatesSize; ++i) {
        hiddenStatesHost[i] = FloatToBFloat16(0.01f * std::sin(static_cast<float>(i % 97)));
    }
    for (int64_t i = 0; i < wkSize; ++i) {
        wkHost[i] = FloatToBFloat16(0.02f * std::cos(static_cast<float>(i % 53)));
        gateWeightHost[i] = FloatToBFloat16(0.02f * std::sin(static_cast<float>(i % 47)));
    }
    for (int64_t i = 0; i < apeSize; ++i) {
        apeHost[i] = 0.01f * static_cast<float>((i % cmpRatio) - 1);
    }

    void *hiddenStatesDeviceAddr = nullptr;
    void *wkDeviceAddr = nullptr;
    void *gateWeightDeviceAddr = nullptr;
    void *apeDeviceAddr = nullptr;
    void *stateCacheDeviceAddr = nullptr;
    void *cacheBlockTableDeviceAddr = nullptr;
    void *startPosDeviceAddr = nullptr;
    void *normWeightDeviceAddr = nullptr;
    void *normBiasDeviceAddr = nullptr;
    void *pooledKeyDeviceAddr = nullptr;

    aclTensor *hiddenStates = nullptr;
    aclTensor *wk = nullptr;
    aclTensor *gateWeight = nullptr;
    aclTensor *ape = nullptr;
    aclTensor *stateCacheRef = nullptr;
    aclTensor *cacheBlockTable = nullptr;
    aclTensor *startPos = nullptr;
    aclTensor *normWeight = nullptr;
    aclTensor *normBias = nullptr;
    aclTensor *pooledKeyOut = nullptr;

    ret = CreateAclTensor(hiddenStatesHost, hiddenStatesShape, &hiddenStatesDeviceAddr, ACL_BF16, &hiddenStates);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkHost, wkShape, &wkDeviceAddr, ACL_BF16, &wk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gateWeightHost, gateWeightShape, &gateWeightDeviceAddr, ACL_BF16, &gateWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(apeHost, apeShape, &apeDeviceAddr, ACL_FLOAT, &ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateCacheHost, stateCacheShape, &stateCacheDeviceAddr, ACL_FLOAT, &stateCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cacheBlockTableHost, cacheBlockTableShape, &cacheBlockTableDeviceAddr, ACL_INT32,
                          &cacheBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHost, startPosShape, &startPosDeviceAddr, ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normWeightHost, normShape, &normWeightDeviceAddr, ACL_FLOAT, &normWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normBiasHost, normShape, &normBiasDeviceAddr, ACL_FLOAT, &normBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pooledKeyHost, pooledKeyShape, &pooledKeyDeviceAddr, ACL_BF16, &pooledKeyOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // 调用aclnnKeyPool第一段接口
    ret = aclnnKeyPoolGetWorkspaceSize(
        hiddenStates, wk, gateWeight, ape, stateCacheRef, cacheBlockTable, startPos, normWeight, normBias, nullptr,
        nullptr, nullptr, nullptr, cmpRatio, normEps, rotaryMode, stateCacheStrideDim0, pooledKeyOut, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnKeyPoolGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnKeyPool第二段接口
    ret = aclnnKeyPool(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKeyPool failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    LOG_PRINT("KeyPool execution succeeded. pooled_key shape=[%ld,%ld,%ld]\n", batchSize, pooledSeqLength, headDim);
    PrintBFloat16Tensor("pooled_key", pooledKeyShape, pooledKeyDeviceAddr);

    aclDestroyTensor(hiddenStates);
    aclDestroyTensor(wk);
    aclDestroyTensor(gateWeight);
    aclDestroyTensor(ape);
    aclDestroyTensor(stateCacheRef);
    aclDestroyTensor(cacheBlockTable);
    aclDestroyTensor(startPos);
    aclDestroyTensor(normWeight);
    aclDestroyTensor(normBias);
    aclDestroyTensor(pooledKeyOut);

    aclrtFree(hiddenStatesDeviceAddr);
    aclrtFree(wkDeviceAddr);
    aclrtFree(gateWeightDeviceAddr);
    aclrtFree(apeDeviceAddr);
    aclrtFree(stateCacheDeviceAddr);
    aclrtFree(cacheBlockTableDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(normWeightDeviceAddr);
    aclrtFree(normBiasDeviceAddr);
    aclrtFree(pooledKeyDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
```

当前 BSH 示例中 `cosOptional`、`sinOptional`、`cuSeqlensOptional`和
`sequsedOptional`均传空指针；TH场景需要按输入约束创建并传入
`cuSeqlensOptional`。示例执行命令如下：

```bash
bash build.sh --run_example key_pool eager cust \
  --vendor_name=custom --soc=ascend910b --example_name=test_aclnn_key_pool

# A5
bash build.sh --run_example key_pool eager cust \
  --vendor_name=custom --soc=ascend950 --example_name=test_aclnn_key_pool
```

- [A2/A3完整调用示例](../examples/test_aclnn_key_pool.cpp)
- [A5完整调用示例](../examples/arch35/test_aclnn_key_pool.cpp)
