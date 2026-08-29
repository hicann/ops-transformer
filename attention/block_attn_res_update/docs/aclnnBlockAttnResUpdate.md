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

- 接口功能：Attention Residuals 的历史残差注意力两段式计算的阶段二：将 `delta` 原地累加到
  `partialBlockRef`，计算更新后残差的 RMSNorm score；随后与阶段一 `block_attn_res_prepare` 返回的
  softmax 统计量 `numerator`、`logitMax`、`expSum`（分别对应 O、M、L）计算得到输出 `h`，
  同时将更新后的残差原地写回 `partialBlockRef`。

### 计算公式

设 `T` 表示 token 数，`D` 表示 hidden size，计算流程如下。

#### 1. 累加当前增量

将 BFLOAT16 类型的 `delta` 转换为 FP32，并逐元素累加到 FP32 类型的 `partialBlockRef`：

$$
p[t,d] = partialBlockRef[t,d] + \operatorname{FP32}(delta[t,d])
$$

`p` 表示包含本次 `delta` 后的最新累计结果。该步骤使用 FP32 计算，最后将 `p` 原地写回
`partialBlockRef`。

#### 2. 计算 RMSNorm 分母

对每个 token 沿 hidden 维进行 FP32 平方和归约，并计算 RMSNorm 分母：

$$
r[t] = \sqrt{\frac{1}{D}\sum_{d=0}^{D-1}p[t,d]^2 + eps}
$$

后续计算只使用每个 token 对应的标量 `r[t]`，不需要生成完整的 RMSNorm 输出 Tensor。

#### 3. 计算当前累计结果的 score

`pseudoQuery` 是用于计算当前 partial block logit 的 FP32 向量。对每个 token，将 `p` 与 `pseudoQuery` 做 FP32
点乘归约，再除以 RMSNorm 分母，得到当前累计结果的 `score`：

$$
score[t] = \frac{\sum_{d=0}^{D-1}p[t,d] \times pseudoQuery[d]}{r[t]}
$$

#### 4. 合并 online softmax 历史状态

`logitMax`、`expSum` 和 `numerator` 分别表示 `block_attn_res_prepare` 输出的历史最大值、历史
softmax 分母累积值和历史加权和。首先计算合并后的最大值：

$$
m[t] = \max(logitMax[t], score[t])
$$

以新的最大值 `m[t]` 为基准，分别计算历史项和当前项的缩放因子，避免指数计算发生数值溢出：

$$
alpha[t] = \exp(logitMax[t] - m[t]), \qquad beta[t] = \exp(score[t] - m[t])
$$

使用 `alpha` 和 `beta` 更新 softmax 分母：

$$
ell[t] = expSum[t] \times alpha[t] + beta[t]
$$

同时更新 online softmax 加权和：

$$
updatedNumerator[t,d] = numerator[t,d] \times alpha[t] + p[t,d] \times beta[t]
$$

上述 online softmax 合并过程均使用 FP32 计算。

#### 5. 写回累计结果并生成输出

将最新累计结果 `p` 原地写回 `partialBlockRef`：

$$
partialBlockRef[t,d] = p[t,d]
$$

将更新后的加权和除以更新后的 softmax 分母，再转换为 BFLOAT16，得到当前层结果 `h`：

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

  | 参数名                                   | 输入/输出 | 描述                                                          | 数据类型 | 数据格式 | 维度(shape) | 非连续的Tensor |
  | ---------------------------------------- | --------- | ------------------------------------------------------------- | -------- | -------- | ----------- | -------------- |
  | `partialBlockRef`（`aclTensor *`）   | 输入/输出 | 当前已累计的`partialBlockRef`。                             | FLOAT    | ND       | `[T, D]`  | ×             |
  | `delta`（`const aclTensor *`）       | 输入      | 本次需要累加的`delta`。                                     | BFLOAT16 | ND       | `[T, D]`  | ×             |
  | `pseudoQuery`（`const aclTensor *`） | 输入      | 用于计算当前 partial block logit 的`pseudoQuery`。          | FLOAT    | ND       | `[D]`     | ×             |
  | `numerator`（`const aclTensor *`）   | 输入      | `block_attn_res_prepare` 输出的历史 online softmax 加权和。 | FLOAT    | ND       | `[T, D]`  | ×             |
  | `logitMax`（`const aclTensor *`）    | 输入      | `block_attn_res_prepare` 输出的历史最大 logit。             | FLOAT    | ND       | `[T]`     | ×             |
  | `expSum`（`const aclTensor *`）      | 输入      | `block_attn_res_prepare` 输出的历史 softmax 分母累积值。    | FLOAT    | ND       | `[T]`     | ×             |
  | `eps`（`float`）                     | 输入      | RMSNorm 数值稳定项。                                          | -        | -        | -           | -              |
  | `h`（`aclTensor *`）                 | 输出      | 当前层结果。                                                  | BFLOAT16 | ND       | `[T, D]`  | ×             |
  | `workspaceSize`（`uint64_t *`）      | 输出      | 返回 Device 侧需要申请的 workspace 大小。                     | -        | -        | -           | -              |
  | `executor`（`aclOpExecutor **`）     | 输出      | 返回包含算子计算流程的执行器。                                | -        | -        | -           | -              |
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
        <th>返回码</th>
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
        <td>当前平台不是 Ascend 950PR/Ascend 950DT。</td>
      </tr>
    </tbody>
  </table>
  </div>

## aclnnBlockAttnResUpdate

- **参数说明**

  | 参数名            | 输入/输出 | 描述                                         |
  | ----------------- | --------- | -------------------------------------------- |
  | `workspace`     | 输入      | Device 侧 workspace 地址。                   |
  | `workspaceSize` | 输入      | Device 侧 workspace 大小，由第一段接口获取。 |
  | `executor`      | 输入      | 包含算子计算流程的执行器。                   |
  | `stream`        | 输入      | ACL stream。                                 |
- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- `partialBlockRef`、`delta`、`pseudoQuery`、`numerator`、`logitMax`、`expSum`、`h`、`workspaceSize` 和
  `executor` 均不能为空。
- 所有 Tensor 输入和输出的原始格式及存储格式均仅支持 ND。
- 所有 Tensor 输入和输出必须连续。
- `partialBlockRef`、`delta`、`numerator` 和 `h` 的 shape 必须均为 `[T, D]`。
- `logitMax` 和 `expSum` 的 shape 必须为 `[T]`；`pseudoQuery` 的 shape 必须为 `[D]`。
- `T >= 0`、`0 <= D <= 8192`；当 `T == 0 || D == 0` 时支持空 Tensor 返回。
- 非空场景下，`T * D` 不能超出 `int64_t` 可表示范围。
- `partialBlockRef`、`pseudoQuery`、`numerator`、`logitMax`、`expSum` 为 FLOAT；`delta` 和 `h` 为 BFLOAT16。
- `eps` 为必选属性，ACLNN C 接口要求调用方显式传入，且必须为有限正数。

<!-- npu="950" id7 -->

<!-- end id7 -->

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
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

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

    ret = aclnnBlockAttnResUpdateGetWorkspaceSize(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum,
                                                   eps, h, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBlockAttnResUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnBlockAttnResUpdate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResUpdate failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

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

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
