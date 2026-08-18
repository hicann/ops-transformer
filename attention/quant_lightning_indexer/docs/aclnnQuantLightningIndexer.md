# aclnnQuantLightningIndexer

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/quant_lightning_indexer)

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

- 接口功能：QuantLightningIndexer在LightningIndexer的基础上支持了Per-Token-Head量化输入。

- 计算公式：
    $$
    out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnQuantLightningIndexerGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantLightningIndexer”接口执行计算。

```Cpp
aclnnStatus aclnnQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    int64_t          queryQuantMode,
    int64_t          keyQuantMode,
    char            *layoutQueryOptional,
    char            *layoutKeyOptional,
    int64_t          sparseCount,
    int64_t          sparseMode,
    int64_t          preTokens,
    int64_t          nextTokens,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnQuantLightningIndexer(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnQuantLightningIndexerGetWorkspaceSize

- **参数说明：**

> [!NOTE]
>
>- query、key、weights、queryDequantScale、keyDequantScale参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- 使用S1和S2分别表示query和key的输入样本序列长度，N1和N2分别表示query和key对应的多头数，k表示最后选取的索引个数。参数query中的D和参数key中的D值相等为128。T1和T2分别表示query和key的输入样本序列长度的累加和。

<table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
<col style="width: 240px">
<col style="width: 132px">
<col style="width: 232px">
<col style="width: 330px">
<col style="width: 233px">
<col style="width: 119px">
<col style="width: 215px">
<col style="width: 100px">
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
    <td>query</td>
    <td>输入</td>
    <td>公式中的输入Q。</td>
    <td>不支持空tensor。</td>
    <td>INT8、FLOAT8_E4M3、HIFLOAT8。</td>
    <td>ND</td>
    <td>
        <ul>
                <li>layoutQuery为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layoutQuery为TND时，shape为(T1,N1,D)。</li>
        </ul>
    </td>
    <td>x</td>
    </tr>
    <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td>
        <ul>
                <li>不支持空tensor。</li>
                <li>blockNum为PageAttention时block总数，blockSize为一个block的token数。</li>
        </ul>
    </td>
    <td>INT8、FLOAT8_E4M3、HIFLOAT8。</td>
    <td>ND</td>
    <td>
        <ul>
                <li>layoutKey为PA_BSND时，shape为(blockNum, blockSize, N2, D)。</li>
                <li>layoutKey为BSND时，shape为(B, S2, N2, D)。</li>
                <li>layoutKey为TND时，shape为(T2, N2, D)。</li>
        </ul>
    </td>
    <td>✓</td>
    </tr>
    <tr>
    <td>weights</td>
    <td>输入</td>
    <td>公式中的输入W。</td>
    <td>不支持空tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>
        <ul>
                <li>layoutQuery为BSND时，shape为(B,S1,N1)。</li>
                <li>layoutQuery为TND时，shape为(T1,N1)。</li>
        </ul>
    </td>
    <td>x</td>
    </tr>
    <tr>
    <td>queryDequantScale</td>
    <td>输入</td>
    <td>表示Index Query的反量化系数Scale_Q。</td>
    <td>不支持空tensor。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
    <td>
        <ul>
                <li>layoutQuery为BSND时，shape为(B,S1,N1)。</li>
                <li>layoutQuery为TND时，shape为(T1,N1)。</li>
        </ul>
    </td>
    <td>x</td>
    </tr>
    <tr>
    <td>keyDequantScale</td>
    <td>输入</td>
    <td>表示Index Key的反量化系数Scale_K。</td>
    <td>不支持空tensor。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
    <td>
        <ul>
                <li>layoutKey为BSND时，shape为(B,S2,N2)。</li>
                <li>layoutKey为TND时，shape为(T2,N2)。</li>
                <li>layoutKey为PA_BSND时，shape为(blockNum, blockSize, N2)。</li>
                <li>blockNum为PageAttention时block总数，blockSize为一个block的token数。</li>
        </ul>
    </td>
    <td>✓</td>
    </tr>
    <tr>
    <td>actualSeqLengthsQueryOptional</td>
    <td>输入</td>
    <td>每个Batch中，Query的有效token数。</td>
    <td>
        <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和query的shape的S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过query中的维度S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当layoutQuery为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
                <li>不能出现负值。</li>
        </ul>
    </td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>actualSeqLengthsKeyOptional</td>
    <td>输入</td>
    <td>每个Batch中，Key的有效token数。</td>
    <td>
        <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和key的shape的S长度相同。</li>
                <li> 该参数中每个Batch的有效token数不超过key/value中的维度S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当layoutKey为TND或PA_BSND时，该入参必须传入，layoutKey为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
                <li>不能出现负值。</li>
        </ul>
    </td>
    <td>INT32</td>
    <td>ND</td>
    <td>(B,)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>blockTableOptional</td>
    <td>输入</td>
    <td>表示PageAttention中KV存储使用的block映射表。</td>
    <td>
        <ul>
                <li>不支持空tensor。</li>
                <li>PageAttention场景下，blockTable必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKey对应的block数量）</li>
                <li>blockSize取值为16的整数倍，最大支持到1024。</li>
        </ul>
    </td>
    <td>INT32</td>
    <td>ND</td>
    <td>shape支持(B,S2_max/blockSize)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>queryQuantMode</td>
    <td>输入</td>
    <td>用于标识输入query的量化模式。</td>
    <td>
        <ul>
            <li>当前支持Per-Token-Head量化模式。</li>
            <li>当前仅支持传入0。</li>
        </ul>
    </td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>keyQuantMode</td>
    <td>输入</td>
    <td>用于标识输入key的量化模式。</td>
    <td>
        <ul>
            <li>当前支持Per-Token-Head量化模式。</li>
            <li>当前仅支持传入0。</li>
        </ul>
    </td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutQueryOptional</td>
    <td>输入</td>
    <td>用于标识输入Query的数据排布格式。</td>
    <td>当前支持BSND、TND。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutKeyOptional</td>
    <td>输入</td>
    <td>用于标识输入Key的数据排布格式。</td>
    <td>当前支持PA_BSND、BSND、TND。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparseCount</td>
    <td>输入</td>
    <td>topK阶段需要保留的block数量。</td>
    <td>当前支持[1, 2048]。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparseMode</td>
    <td>输入</td>
    <td>表示sparse的模式。</td>
    <td>
        <ul>
                <li>sparseMode为0时，代表defaultMask模式。</li>
                <li>sparseMode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
        </ul>
    </td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>preTokens</td>
    <td>输入</td>
    <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
    <td>建议值2^63-1。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>nextTokens</td>
    <td>输入</td>
    <td>用于稀疏计算，表示attention需要和后几个Token计算关联。</td>
    <td>建议值2^63-1。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>out</td>
    <td>输出</td>
    <td>公式中的Indices输出。</td>
    <td>不支持空tensor。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>
        <ul>
                <li>layoutQuery为BSND时输出shape为[B, S1, N2, sparseCount]。</li>
                <li>layoutQuery为TND时输出shape为[T1, N2, sparseCount]。</li>
        </ul>
    </td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
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

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口会完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>query、key、weights、queryDequantScale、keyDequantScale、actualSeqLengthsQueryOptional、actualSeqLengthsKeyOptional、queryQuantMode、keyQuantMode、layoutQueryOptional、layoutKeyOptional、sparseCount、sparseMode、out的数据类型和数据格式不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnQuantLightningIndexer

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantLightningIndexerGetWorkspaceSize获取。</td>
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

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnQuantLightningIndexer默认确定性实现。
- 参数query中的N1支持小于等于64/32/24/16，key的N2支持1。
- D支持128。
- blockSize取值为16的倍数，最大支持1024。
- 参数query、key的数据类型应保持一致。

<!-- npu="A3" id8 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - query和key的数据类型支持`INT8`。
  - 仅支持weights、queryDequantScale、keyDequantScale数据类型为`FLOAT16、FLOAT16、FLOAT16`。
  - key和keyDequantScale在layout_key为PA_BSND时支持0轴非连续，其余轴必须连续。

<!-- npu="950" id7 -->
- Ascend 950PR/Ascend 950DT：
  - query N1仅支持8、16、24、32、64。
  - query和key的数据类型支持`FLOAT8_E4M3、HIFLOAT8、INT8`。
  - 当query和key的数据类型为`FLOAT8_E4M3`时，支持weights、queryDequantScale、keyDequantScale的数据类型为`BFLOAT16、FLOAT、FLOAT`或`FLOAT16、FLOAT16、FLOAT16`；
  - 当query和key的数据类型为`HIFLOAT8`时，仅支持weights、queryDequantScale、keyDequantScale数据类型为`BFLOAT16、FLOAT、FLOAT`；
  - 当query和key的数据类型为`INT8`时，仅支持weights、queryDequantScale、keyDequantScale数据类型为`FLOAT16、FLOAT16、FLOAT16`。

<!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_lightning_indexer.cpp
 * \brief
 */
//testci
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* queryDequantScaleDeviceAddr = nullptr;
    void* keyDequantScaleDeviceAddr = nullptr;
    void* actualSeqLengthsKeyDeviceAddr = nullptr;
    void* blockTableDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* queryDequantScaleTensor = nullptr;
    aclTensor* keyDequantScaleTensor = nullptr;
    aclTensor* actualSeqLengthsKeyTensor = nullptr;
    aclTensor* blockTableTensor = nullptr;
    aclTensor* outTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {2, 64, 16, 128};
    std::vector<int64_t> keyShape = {32, 16, 1, 128};
    std::vector<int64_t> weightsShape = {2, 64, 16};
    std::vector<int64_t> queryDequantScaleShape = {2, 64, 16};
    std::vector<int64_t> keyDequantScaleShape = {32, 16, 1};
    std::vector<int64_t> actualSeqLengthsKeyShape = {2};
    std::vector<int64_t> blockTableShape = {2, 32};
    std::vector<int64_t> outShape = {2, 64, 1, 2048};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t queryDequantScaleShapeSize = GetShapeSize(queryDequantScaleShape);
    int64_t keyDequantScaleShapeSize = GetShapeSize(keyDequantScaleShape);
    int64_t actualSeqLengthsKeyShapeSize = GetShapeSize(actualSeqLengthsKeyShape);
    int64_t blockTableShapeSize = GetShapeSize(blockTableShape);
    int64_t outShapeSize = GetShapeSize(outShape);

    std::vector<int8_t> queryHostData(queryShapeSize, 1);
    std::vector<int8_t> keyHostData(keyShapeSize, 1);
    std::vector<uint16_t> weightsHostData(weightsShapeSize, 0x3F00);
    std::vector<float> queryDequantScaleHostData(queryDequantScaleShapeSize, 1.0f);
    std::vector<float> keyDequantScaleHostData(keyDequantScaleShapeSize, 1.0f);
    std::vector<int32_t> actualSeqLengthsKeyHostData = {256, 512};
    std::vector<int32_t> blockTableHostData(blockTableShapeSize, 0);
    for (int32_t i = 0; i < 32; i++) {
        blockTableHostData[i] = i;
        blockTableHostData[32 + i] = i;
    }
    std::vector<int32_t> outHostData(outShapeSize, 0);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT8_E4M3FN, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_FLOAT8_E4M3FN, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr,
                          aclDataType::ACL_BF16, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(queryDequantScaleHostData, queryDequantScaleShape, &resources.queryDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.queryDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyDequantScaleHostData, keyDequantScaleShape, &resources.keyDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.keyDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(actualSeqLengthsKeyHostData, actualSeqLengthsKeyShape, &resources.actualSeqLengthsKeyDeviceAddr,
                          aclDataType::ACL_INT32, &resources.actualSeqLengthsKeyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(blockTableHostData, blockTableShape, &resources.blockTableDeviceAddr,
                          aclDataType::ACL_INT32, &resources.blockTableTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr,
                          aclDataType::ACL_INT32, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }
    return ACL_SUCCESS;
}

int ExecuteQuantLightningIndexer(TensorResources& resources, aclrtStream stream,
                                 void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t queryQuantMode = 0;
    int64_t keyQuantMode = 0;
    int64_t sparseCount = 2048;
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    constexpr const char layoutQueryStr[] = "BSND";
    constexpr const char layoutKeyStr[] = "PA_BSND";
    constexpr size_t layoutQueryLen = sizeof(layoutQueryStr);
    constexpr size_t layoutKeyLen = sizeof(layoutKeyStr);
    char layoutQuery[layoutQueryLen];
    char layoutKey[layoutKeyLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layoutQueryStr, layoutQueryLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layoutKeyStr, layoutKeyLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnQuantLightningIndexerGetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor,
        resources.queryDequantScaleTensor, resources.keyDequantScaleTensor,
        nullptr, resources.actualSeqLengthsKeyTensor, resources.blockTableTensor,
        queryQuantMode, keyQuantMode, layoutQuery, layoutKey,
        sparseCount, sparseMode, preTokens, nextTokens,
        resources.outTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexer failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("sparse_indices result[%ld] is: %d\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
      aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.queryDequantScaleTensor) {
      aclDestroyTensor(resources.queryDequantScaleTensor);
    }
    if (resources.keyDequantScaleTensor) {
      aclDestroyTensor(resources.keyDequantScaleTensor);
    }
    if (resources.actualSeqLengthsKeyTensor) {
      aclDestroyTensor(resources.actualSeqLengthsKeyTensor);
    }
    if (resources.blockTableTensor) {
      aclDestroyTensor(resources.blockTableTensor);
    }
    if (resources.outTensor) {
      aclDestroyTensor(resources.outTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
      aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.queryDequantScaleDeviceAddr) {
      aclrtFree(resources.queryDequantScaleDeviceAddr);
    }
    if (resources.keyDequantScaleDeviceAddr) {
      aclrtFree(resources.keyDequantScaleDeviceAddr);
    }
    if (resources.actualSeqLengthsKeyDeviceAddr) {
      aclrtFree(resources.actualSeqLengthsKeyDeviceAddr);
    }
    if (resources.blockTableDeviceAddr) {
      aclrtFree(resources.blockTableDeviceAddr);
    }
    if (resources.outDeviceAddr) {
      aclrtFree(resources.outDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outShape = {2, 64, 1, 2048};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteQuantLightningIndexer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    PrintOutResult(outShape, &resources.outDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}

```
