# aclnnBlockAttnResUpdate

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/block_attn_res_update)

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

- **接口功能**：

  Attention Residuals的历史残差注意力两段式计算的阶段二：将`delta`原地累加到`partialBlockRef`，计算更新后残差的RMSNorm score；随后与阶段一`block_attn_res_prepare`返回的softmax统计量`numerator`、`logitMax`、`expSum`（分别对应O、M、L）计算得到输出`h`，同时将更新后的残差原地写回`partialBlockRef`。
- **计算公式**：

  设`T`表示token数，`D`表示hidden size。主要计算过程如下：

  1. 更新当前`partialBlockRef`：

     $$
     p[t,d] = partialBlockRef[t,d] + \operatorname{FP32}(delta[t,d])
     $$
  2. 计算RMSNorm分母和当前score：

     $$
     r[t] = \sqrt{\frac{1}{D}\sum_{d=0}^{D-1}p[t,d]^2 + eps}
     $$

     $$
     score[t] = \frac{\sum_{d=0}^{D-1}p[t,d] \times pseudoQuery[d]}{r[t]}
     $$
  3. 将当前score与历史online softmax状态合并：

     $$
     m[t] = \max(logitMax[t], score[t])
     $$

     $$
     alpha[t] = \exp(logitMax[t] - m[t]), \qquad beta[t] = \exp(score[t] - m[t])
     $$

     $$
     ell[t] = expSum[t] \times alpha[t] + beta[t]
     $$

     $$
     updatedNumerator[t,d] = numerator[t,d] \times alpha[t] + p[t,d] \times beta[t]
     $$
  4. 生成输出：

     $$
     partialBlockRef[t,d] \leftarrow p[t,d]
     $$

     $$
     h[t,d] = \operatorname{BF16}\left(\frac{updatedNumerator[t,d]}{ell[t]}\right)
     $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)。必须先调用
`aclnnBlockAttnResUpdateGetWorkspaceSize` 获取 workspace 大小和执行器，再调用
`aclnnBlockAttnResUpdate` 执行计算。

```cpp
aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(
    aclTensor       *partialBlockRef,
    const aclTensor *delta,
    const aclTensor *pseudoQuery,
    const aclTensor *numerator,
    const aclTensor *logitMax,
    const aclTensor *expSum,
    float            eps,
    aclTensor       *h,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnBlockAttnResUpdate(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnBlockAttnResUpdateGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 248px">
  <col style="width: 121px">
  <col style="width: 210px">
  <col style="width: 327px">
  <col style="width: 160px">
  <col style="width: 115px">
  <col style="width: 138px">
  <col style="width: 145px">
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
        <td>partialBlockRef(aclTensor*)</td>
        <td>输入/输出</td>
        <td>当前已累计的partialBlockRef。</td>
        <td>
          <ul>
            <li>必选。</li>
            <li>数据类型支持FLOAT。</li>
            <li>原始格式和存储格式均仅支持ND。</li>
            <li>必须为连续Tensor。</li>
            <li><code>T >= 0</code>，<code>0 <= D <= 8192</code>。</li>
            <li>支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>delta(const aclTensor*)</td>
        <td>输入</td>
        <td>本次需要累加的delta。</td>
        <td>
          <ul>
            <li>必选。</li>
            <li>数据类型支持BFLOAT16。</li>
            <li>原始格式和存储格式均仅支持ND。</li>
            <li>必须为连续Tensor。</li>
            <li>shape必须与partialBlockRef相同。</li>
            <li>支持空Tensor。</li>
          </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseudoQuery(const aclTensor*)</td>
        <td>输入</td>
        <td>用于计算当前partial block logit的pseudoQuery。</td>
        <td>
          <ul>
            <li>必选。</li>
            <li>数据类型支持FLOAT。</li>
            <li>原始格式和存储格式均仅支持ND。</li>
            <li>必须为连续Tensor。</li>
            <li>长度必须等于partialBlockRef的D。</li>
            <li>支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>numerator(const aclTensor*)</td>
        <td>输入</td>
        <td>block_attn_res_prepare输出的历史online softmax加权和。</td>
        <td>
          <ul>
            <li>必选。</li>
            <li>数据类型支持FLOAT。</li>
            <li>原始格式和存储格式均仅支持ND。</li>
            <li>必须为连续Tensor。</li>
            <li>shape必须与partialBlockRef相同。</li>
            <li>支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>logitMax(const aclTensor*)</td>
        <td>输入</td>
        <td>block_attn_res_prepare输出的历史最大logit。</td>
        <td>
          <ul>
            <li>必选。</li>
            <li>数据类型支持FLOAT。</li>
            <li>原始格式和存储格式均仅支持ND。</li>
            <li>必须为连续Tensor。</li>
            <li>长度必须等于partialBlockRef的T。</li>
            <li>支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>expSum(const aclTensor*)</td>
        <td>输入</td>
        <td>block_attn_res_prepare输出的历史softmax分母累积值。</td>
        <td>
          <ul>
            <li>必选。</li>
            <li>数据类型支持FLOAT。</li>
            <li>原始格式和存储格式均仅支持ND。</li>
            <li>必须为连续Tensor。</li>
            <li>长度必须等于partialBlockRef的T。</li>
            <li>支持空Tensor。</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>eps(float)</td>
        <td>输入</td>
        <td>RMSNorm数值稳定项。</td>
        <td>
          <ul>
            <li>必选。</li>
            <li>必须为有限正数。</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>h(aclTensor*)</td>
        <td>输出</td>
        <td>当前层结果。</td>
        <td>
          <ul>
            <li>必选。</li>
            <li>数据类型支持BFLOAT16。</li>
            <li>原始格式和存储格式均仅支持ND。</li>
            <li>必须为连续Tensor。</li>
            <li>shape必须与partialBlockRef相同。</li>
            <li>空Tensor场景下返回空Tensor。</li>
          </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t*)</td>
        <td>输出</td>
        <td>返回Device侧需要申请的workspace大小。</td>
        <td>
          <ul>
            <li>必选。</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor(aclOpExecutor**)</td>
        <td>输出</td>
        <td>返回包含算子计算流程的执行器。</td>
        <td>
          <ul>
            <li>必选。</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <div style="overflow-x: auto;">
  <table style="table-layout: fixed; width: 1100px">
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 720px">
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
        <td><code>partialBlockRef</code>、<code>delta</code>、<code>pseudoQuery</code>、<code>numerator</code>、<code>logitMax</code>、<code>expSum</code>、<code>h</code>、<code>workspaceSize</code> 或 <code>executor</code> 为空。</td>
      </tr>
      <tr>
        <td class="merged-cell" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td class="merged-cell" rowspan="5">161002</td>
        <td>输入或输出的 dtype、原始或存储 format、dimension 或 shape 关系不满足约束。</td>
      </tr>
      <tr>
        <td><code>T < 0</code>，或者 <code>D</code> 不在 <code>[0, 8192]</code> 范围内。</td>
      </tr>
      <tr>
        <td>非空场景下 <code>T * D</code> 超出 <code>int64_t</code> 可表示范围。</td>
      </tr>
      <tr>
        <td>任一 Tensor 输入或输出为非连续 Tensor。</td>
      </tr>
      <tr>
        <td><code>eps</code> 不是有限正数。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>产品型号不在支持的范围内。</td>
      </tr>
    </tbody>
  </table>
  </div>

## aclnnBlockAttnResUpdate

- **参数说明**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
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
    <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockAttnResUpdateGetWorkspaceSize获取。</td>
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
  </tbody></table>

- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnBlockAttnResUpdate默认确定性实现。
- 所有 Tensor 输入都支持空 Tensor，但必须继续满足彼此的 shape 关系，不能单独任意置空。
- 非空场景下，`T * D` 不能超出`int64_t`可表示范围。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考
[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。对应的完整示例源码见
[test_aclnn_block_attn_res_update.cpp](../examples/arch35/test_aclnn_block_attn_res_update.cpp)。

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

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_update.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            Finalize(deviceId, stream);                                                                                \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

constexpr size_t FIRST_ELEMENT_INDEX = 0UL;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16U);
}

float Bfloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclFormat formatType, aclTensor **tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n");
              return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int RunBlockAttnResUpdate(int32_t deviceId, aclrtStream &stream)
{
    // 1. 初始化Device和Stream。
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入和输出数据。
    constexpr int64_t tokenNum = 2;
    constexpr int64_t hiddenSize = 64;
    const std::vector<int64_t> matrixShape = {tokenNum, hiddenSize};
    const std::vector<int64_t> queryShape = {hiddenSize};
    const std::vector<int64_t> statsShape = {tokenNum};

    std::vector<float> partialBlockRefHostData(GetShapeSize(matrixShape), 0.25F);
    std::vector<uint16_t> deltaHostData(GetShapeSize(matrixShape), FloatToBfloat16(0.125F));
    std::vector<float> pseudoQueryHostData(GetShapeSize(queryShape), 1.0F / static_cast<float>(hiddenSize));
    std::vector<float> numeratorHostData(GetShapeSize(matrixShape), 0.5F);
    std::vector<float> logitMaxHostData(GetShapeSize(statsShape), 0.0F);
    std::vector<float> expSumHostData(GetShapeSize(statsShape), 1.0F);
    std::vector<uint16_t> hHostData(GetShapeSize(matrixShape), 0U);

    void *partialBlockRefDeviceAddr = nullptr;
    void *deltaDeviceAddr = nullptr;
    void *pseudoQueryDeviceAddr = nullptr;
    void *numeratorDeviceAddr = nullptr;
    void *logitMaxDeviceAddr = nullptr;
    void *expSumDeviceAddr = nullptr;
    void *hDeviceAddr = nullptr;

    aclTensor *partialBlockRef = nullptr;
    aclTensor *delta = nullptr;
    aclTensor *pseudoQuery = nullptr;
    aclTensor *numerator = nullptr;
    aclTensor *logitMax = nullptr;
    aclTensor *expSum = nullptr;
    aclTensor *h = nullptr;

    // 3. 创建输入和输出aclTensor。
    ret = CreateAclTensor<float>(partialBlockRefHostData, matrixShape, &partialBlockRefDeviceAddr,
                                 aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &partialBlockRef);
    std::unique_ptr<void, aclError (*)(void *)> partialBlockRefDeviceAddrPtr(partialBlockRefDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> partialBlockRefTensorPtr(partialBlockRef,
                                                                                           aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint16_t>(deltaHostData, matrixShape, &deltaDeviceAddr, aclDataType::ACL_BF16,
                                    aclFormat::ACL_FORMAT_ND, &delta);
    std::unique_ptr<void, aclError (*)(void *)> deltaDeviceAddrPtr(deltaDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> deltaTensorPtr(delta, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(pseudoQueryHostData, queryShape, &pseudoQueryDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &pseudoQuery);
    std::unique_ptr<void, aclError (*)(void *)> pseudoQueryDeviceAddrPtr(pseudoQueryDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pseudoQueryTensorPtr(pseudoQuery, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(numeratorHostData, matrixShape, &numeratorDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &numerator);
    std::unique_ptr<void, aclError (*)(void *)> numeratorDeviceAddrPtr(numeratorDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> numeratorTensorPtr(numerator, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(logitMaxHostData, statsShape, &logitMaxDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &logitMax);
    std::unique_ptr<void, aclError (*)(void *)> logitMaxDeviceAddrPtr(logitMaxDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitMaxTensorPtr(logitMax, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(expSumHostData, statsShape, &expSumDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &expSum);
    std::unique_ptr<void, aclError (*)(void *)> expSumDeviceAddrPtr(expSumDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> expSumTensorPtr(expSum, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint16_t>(hHostData, matrixShape, &hDeviceAddr, aclDataType::ACL_BF16,
                                    aclFormat::ACL_FORMAT_ND, &h);
    std::unique_ptr<void, aclError (*)(void *)> hDeviceAddrPtr(hDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> hTensorPtr(h, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    constexpr float eps = 1.0e-6F;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    // 4. 调用第一段接口获取workspace大小和执行器。
    ret = aclnnBlockAttnResUpdateGetWorkspaceSize(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum,
                                                   eps, h, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBlockAttnResUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        // 根据第一段接口返回的大小申请Device侧workspace。
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 5. 调用第二段接口执行算子。
    ret = aclnnBlockAttnResUpdate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResUpdate failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待任务执行完成。
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 将原地更新结果和输出结果拷贝回Host侧。
    const size_t partialBlockRefSize = partialBlockRefHostData.size() * sizeof(float);
    ret = aclrtMemcpy(partialBlockRefHostData.data(), partialBlockRefSize, partialBlockRefDeviceAddr,
                      partialBlockRefSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy partialBlockRef to host failed. ERROR: %d\n", ret); return ret);

    const size_t hSize = hHostData.size() * sizeof(uint16_t);
    ret = aclrtMemcpy(hHostData.data(), hSize, hDeviceAddr, hSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy h to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("partialBlockRef[0] after in-place update: %.6f\n", partialBlockRefHostData[FIRST_ELEMENT_INDEX]);
    LOG_PRINT("h[0]: %.6f\n", Bfloat16ToFloat(hHostData[FIRST_ELEMENT_INDEX]));
    LOG_PRINT("block_attn_res_update example execute success.\n");
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = RunBlockAttnResUpdate(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBlockAttnResUpdate failed. ERROR: %d\n", ret); return ret);

    // 8. 释放Device和Stream资源。
    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
