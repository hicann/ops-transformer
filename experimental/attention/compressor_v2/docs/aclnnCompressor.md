# aclnnCompressorV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/compressor_v2)

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

- 接口功能：CompressorV2是推理场景下SMLA和QLI的前处理算子，用于将每cmp_ratio个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。在长序列的情况下，可以有效减少计算开销。主要计算过程为：
    1. 将输入$X$与$W^{KV}$做Matmul运算得到$kv\_state$，将输入$X$与$W^{Gate}$做Matmul运算得到$score\_state$，$kv\_state$与$score\_state$根据输入的start_pos及cu_seqlens写入state_cache（循环buffer，写本次序列末尾min(seqused, block_size)个token）。
    2. 对$score\_state$进行softmax运算，将softmax结果与$kv\_state$做Mul计算，后进行ReduceSum运算得到压缩结果。

- 计算公式：

    1. 计算矩阵乘法：

        $$
        \left[kv\_state, score\_state\right] = \left[X @ W^{KV}, X @ W^{Gate}\right]
        $$

    2. 计算分组Softmax：

        $$
        S_i^\prime = softmax(score\_state_i),~i=1,2,\cdots, \frac{s}{cmp\_ratio}
        $$

    3. 计算Hadamard乘积：

        $$
        S_H = S_i^\prime \odot kv\_state
        $$

    4. 沿着压缩轴分组求和：

        $$
        C_{i}^{\text{Comp}} = \left[1\right]_{1\times cmp\_ratio} @ (S_H)_i, ~i=1,2,\cdots, \frac{s}{cmp\_ratio}
        $$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnCompressorV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCompressorV2”接口执行计算。

```cpp
aclnnStatus aclnnCompressorV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *wkv,
    const aclTensor *wgate,
    aclTensor       *stateCacheRef,
    const aclTensor *stateBlockTableOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional,
    int64_t          cmpRatio,
    int64_t          stateCacheStrideDim0,
    const aclTensor *cmpKvOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnCompressorV2(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnCompressorV2GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 230px">
    <col style="width: 360px">
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 220px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的<span class="math-inline">X</span>，表示原始不经压缩的数据。</td>
      <td><ul><li>支持B=0、S=0、T=0的空Tensor。</li><li>B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、T表示所有Batch输入样本序列长度的累加和。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>BS合轴：[T,H]<br>BS非合轴：[B,S,H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>wkv（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的<span class="math-inline">W<sup>KV</sup></span>，表示kv压缩权重。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[D,H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>wgate（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的<span class="math-inline">W<sup>Gate</sup></span>，表示gate压缩权重。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[D,H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>stateCacheRef（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的<span class="math-inline">[kv_state, score_state]</span>，表示kv_state和score_state的历史数据。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[block_num,block_size,2* D]，要求block_num>0；需满足block_size >= cmp_ratio + S - 1。</td>
      <td>支持0轴非连续</td>
    </tr>
    <tr>
      <td>stateBlockTableOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示state_cache存储使用的block映射表。当其中元素的值为0时，表示当前位置无需进行更新state_cache操作。</td>
      <td>支持S=0、T=0的空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqlensOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中的有效token数。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值，且第一位必须为0。</td>
      <td><ul><li>支持B=0、S=0、T=0的空Tensor。</li><li>当x的shape为[B,S,H]时，该参数必须传入空指针。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>当x的shape为[T,H]时，输入shape为[B+1,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不同Batch中实际参与压缩的token数。传入空指针时，表示与每个Batch上的Sequence Length长度相同。</td>
      <td><ul><li>支持B=0的空Tensor。</li><li>当x的shape为[B,S,H]时，要求0 ≤ seqused[n] ≤ S。</li><li>当x的shape为[T,H]时，要求0 ≤ seqused[n] ≤ cu_seqlens[n+1] - cu_seqlens[n]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>startPosOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示计算起始位置。</td>
      <td><ul><li>支持B=0、T=0的空Tensor。</li><li>传入空指针时，表示从0开始进行计算。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>[B,]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cmpRatio（int64_t）</td>
      <td>输入</td>
      <td>用于稀疏计算，表示数据压缩率。</td>
      <td>取值范围为[2, 128]内的整数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stateCacheStrideDim0（int64_t）</td>
      <td>输入</td>
      <td>表示state_cache的0轴stride。</td>
      <td>由框架根据state_cache的实际stride传入。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpKvOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示压缩后的数据，对应公式中的<span class="math-inline">C<sub>i</sub><sup>Comp</sup></span>。</td>
      <td><ul><li>支持B=0、S=0、T=0的空Tensor。</li><li>BS合轴场景下输出shape为[min(T, T/cmp_ratio+B), D]。</li><li>BS非合轴场景下输出shape为[B, ceil(S/cmp_ratio), D]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>BS合轴：[min(T, T/cmp_ratio+B), D]<br>BS非合轴：[B, ceil(S/cmp_ratio), D]</td>
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

  - D（Head Dim）表示hidden层的最小单元大小，D取值为wkv的第一维大小。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>x、wkv、wgate、stateCacheRef、cmpKvOut等必选参数中存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>x、wkv、wgate、stateCacheRef、cmpKvOut的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、wkv、wgate、stateCacheRef的shape维度不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>cmpRatio的取值不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>cuSeqlensOptional、sequsedOptional、startPosOptional与x的shape组合不满足约束。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>tiling发生异常，入参的dtype类型或者shape错误。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>API内存调用NPU Runtime接口时发生异常（如Runtime服务未启动、内存申请失败等）。</td>
    </tr>
  </tbody>
  </table>

## aclnnCompressorV2

- **参数说明**

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
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCompressorV2GetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnCompressorV2默认确定性实现。
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：batch一致性：通过aclrtSetSysParamOpt()配置ACL_OPT_DETERMINISTIC为3来开启batch一致性，开启后可以满足计算结果和所在批次大小、位置无关。
  <!-- end id8 -->
- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 输入shape限制：
  - stateCacheRef支持输入shape为[block_num, block_size, 2* D]，要求block_num>0；需满足block_size >= cmp_ratio + S - 1。
  - 当x采用BS合轴，即x的输入shape为[T,H]时：
    - cuSeqlensOptional输入shape必须为[B+1,]，且该参数为前缀和数组，后一个元素的值必须大于等于前一个元素的值，第一位必须为0。
    - sequsedOptional支持输入shape为[B,]，要求0 ≤ seqused[n] ≤ cu_seqlens[n+1] - cu_seqlens[n]。
    - stateBlockTableOptional支持输入shape为[B]。
    - cmpKvOut输出shape为[min(T, T/cmp_ratio+B), D]。
  - 当x不采用BS合轴，即x的输入shape为[B,S,H]时：
    - cuSeqlensOptional必须传入空指针。
    - sequsedOptional支持输入shape为[B,]，要求0 ≤ seqused[n] ≤ S。
    - stateBlockTableOptional支持输入shape为[B]。
    - cmpKvOut输出shape为[B, ceil(S/cmp_ratio), D]。
- 输入值域限制：
  - 该接口支持B、S泛化，且存在如下场景限制：
    - 只支持B、S为0。
    - 部分长序列场景下，如果计算量过大可能会导致出现超过NPU内存的报错，注：这里计算量会受x输入shape的影响，值越大计算量越大。典型的长序列（即B、S的乘积或T较大）场景包括但不限于：

      <table style="undefined;table-layout: fixed; width: 400px"><colgroup>
        <col style="width: 100px">
        <col style="width: 100px">
        <col style="width: 100px">
      </colgroup>
      <thead>
        <tr>
          <th>B</th>
          <th>S</th>
          <th>H</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>100</td><td>65525</td><td>4096</td></tr>
        <tr><td>25</td><td>261120</td><td>4096</td></tr>
        <tr><td>100</td><td>131072</td><td>4096</td></tr>
        <tr><td>100</td><td>261120</td><td>4096</td></tr>
      </tbody>
      </table>
  - 该接口支持B、S、T取0，即shape与B、S、T值相关的入参允许传入空tensor，其余入参不支持传入空tensor。该场景下stateCacheRef不做更新，输出cmpKvOut为空tensor。
- 输入属性限制：
  - 支持D为128/512。
  - 支持H为1K~10K，512对齐。
  - 支持cmpRatio为2/4/8/16/32/64/128。
  - 支持blockSize为1~1024。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
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
 * \file test_aclnn_compressor_v2.cpp
 * \brief CompressorV2 算子 aclnn 调用示例（A5 / ascend950）
 *        场景：D=512, cmp_ratio=4, 循环buffer（cache_mode=2）, BSH layout, BF16
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_compressor_v2.h"
#include "opdev/bfloat16.h"

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

using op::bfloat16;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    } else {
        *deviceAddr = nullptr;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void PrintBf16Result(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<bfloat16> resultData(size, bfloat16(0.0f));
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) { // 10: max print
        LOG_PRINT("cmp_kv[%ld] is: %f\n", i, static_cast<float>(resultData[i]));
    }
}

} // namespace

int main()
{
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 场景参数: D=512, cmp_ratio=4, 循环buffer（cache_mode=2）, BSH layout, BF16
    int64_t B = 1;
    int64_t S = 4;
    int64_t hiddenSize = 4096;
    int64_t headDim = 512;
    int64_t cmpRatio = 4;
    int64_t blockSize = 128;

    // 固定为循环buffer（cache_mode=2）：每个 batch 使用一个 block，blockSize 需满足 blockSize >= cmpRatio + S - 1
    int64_t blockNum = B;
    int64_t stateCacheStrideDim0 = blockSize * 2 * headDim; // state_cache 0轴 stride = dim1 * dim2

    // 2. 构造输入与输出 shape
    std::vector<int64_t> xShape = {B, S, hiddenSize};
    std::vector<int64_t> wkvShape = {headDim, hiddenSize};
    std::vector<int64_t> wgateShape = {headDim, hiddenSize};
    std::vector<int64_t> stateCacheShape = {blockNum, blockSize, 2 * headDim};
    std::vector<int64_t> stateBlockTableShape = {B};
    std::vector<int64_t> startPosShape = {B};
    int64_t Sr = (S + cmpRatio - 1) / cmpRatio;
    std::vector<int64_t> cmpKvShape = {B, Sr, headDim};

    // 3. 构造 host 数据
    int64_t xSize = GetShapeSize(xShape);
    int64_t wkvSize = GetShapeSize(wkvShape);
    int64_t wgateSize = GetShapeSize(wgateShape);
    int64_t stateCacheSize = GetShapeSize(stateCacheShape);
    int64_t cmpKvSize = GetShapeSize(cmpKvShape);

    std::vector<bfloat16> xHostData(xSize, bfloat16(0.1f));
    std::vector<bfloat16> wkvHostData(wkvSize, bfloat16(0.1f));
    std::vector<bfloat16> wgateHostData(wgateSize, bfloat16(0.1f));
    std::vector<float_t> stateCacheHostData(stateCacheSize, 0.1f);
    std::vector<int32_t> stateBlockTableHostData(B, 1);
    std::vector<int32_t> startPosHostData(B, 0);
    std::vector<bfloat16> cmpKvHostData(cmpKvSize, bfloat16(0.0f));

    // 4. 创建 aclTensor
    void *xDeviceAddr = nullptr;
    void *wkvDeviceAddr = nullptr;
    void *wgateDeviceAddr = nullptr;
    void *stateCacheDeviceAddr = nullptr;
    void *stateBlockTableDeviceAddr = nullptr;
    void *startPosDeviceAddr = nullptr;
    void *cmpKvDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *wkv = nullptr;
    aclTensor *wgate = nullptr;
    aclTensor *stateCacheRef = nullptr;
    aclTensor *stateBlockTable = nullptr;
    aclTensor *startPos = nullptr;
    aclTensor *cmpKvOut = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvHostData, wkvShape, &wkvDeviceAddr, aclDataType::ACL_BF16, &wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateHostData, wgateShape, &wgateDeviceAddr, aclDataType::ACL_BF16, &wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateCacheHostData, stateCacheShape, &stateCacheDeviceAddr, aclDataType::ACL_FLOAT,
                          &stateCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateBlockTableHostData, stateBlockTableShape, &stateBlockTableDeviceAddr,
                          aclDataType::ACL_INT32, &stateBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHostData, startPosShape, &startPosDeviceAddr, aclDataType::ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_BF16, &cmpKvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. 调用 aclnnCompressorV2GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnCompressorV2GetWorkspaceSize(x, wkv, wgate, stateCacheRef, stateBlockTable, nullptr, nullptr, startPos,
                                            cmpRatio, stateCacheStrideDim0, cmpKvOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCompressorV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 6. 申请 workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 7. 调用 aclnnCompressorV2
    ret = aclnnCompressorV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCompressorV2 failed. ERROR: %d\n", ret); return ret);

    // 8. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 9. 获取输出
    LOG_PRINT("CompressorV2 execution succeeded.\n");
    PrintBf16Result(cmpKvShape, &cmpKvDeviceAddr);

    // 10. 释放资源
    aclDestroyTensor(x);
    aclDestroyTensor(wkv);
    aclDestroyTensor(wgate);
    aclDestroyTensor(stateCacheRef);
    aclDestroyTensor(stateBlockTable);
    aclDestroyTensor(startPos);
    aclDestroyTensor(cmpKvOut);

    aclrtFree(xDeviceAddr);
    aclrtFree(wkvDeviceAddr);
    aclrtFree(wgateDeviceAddr);
    aclrtFree(stateCacheDeviceAddr);
    aclrtFree(stateBlockTableDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(cmpKvDeviceAddr);
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
