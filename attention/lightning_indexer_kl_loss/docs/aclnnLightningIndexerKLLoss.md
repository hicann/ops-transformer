# aclnnLightningIndexerKLLoss

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
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

- 接口功能：`lightning_indexer_kl_loss` 计算 Lightning Indexer 中 teacher 分布与 student 分布之间的 KL 散度损失函数。

  - **teacher 侧**（target_score）：压缩段未归一化的原始主注意力分数（sum ≠ 1），用 `clamp_min` 防止 y=0 处 log(0) 导致 NaN。
  - **student 侧**（index_probs）：indexer softmax 后的概率分布，用 `+eps` 保住 Y→0 处的梯度。

- 计算公式：

  $$
  y = \text{target\_score}, \quad Y = \text{index\_probs}
  $$

  $$
  P = \frac{y}{\text{sum}(y, \text{dim}=-1, \text{keepdim=True}) + \varepsilon}
  $$

  $$
  \log\_P = \log(\text{clamp\_min}(\tilde{y}, \varepsilon))
  $$

  $$
  \log\_Y = \log(Y + \varepsilon)
  $$

  $$
  \text{loss} = \sum((\log\_P - \log\_Y) \cdot \text{weight})
  $$

  其中 $\varepsilon$ 为 `eps` 参数$。

  weight 的选择由 `weight_type` 控制：

  - `'logits'`：weight = y，即原始未归一化分数
  - `'probs'`：weight = P，即归一化概率

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnLightningIndexerKLLossGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnLightningIndexerKLLoss"接口执行计算。

```c++
aclnnStatus aclnnLightningIndexerKLLossGetWorkspaceSize(
  const aclTensor   *targetScore,
  const aclTensor   *indexProbs,
  double             eps,
  const char        *weightType,
  const aclTensor   *loss,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnLightningIndexerKLLoss(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnLightningIndexerKLLossGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; max-width: 1550px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
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
      <td>targetScore</td>
      <td>输入</td>
      <td>teacher 未归一化的原始主注意力分数。</td>
      <td>shape 支持 (T, K) 或 (B, S, K)。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indexProbs</td>
      <td>输入</td>
      <td>student softmax 后的概率分布。</td>
      <td>shape 与 targetScore 保持一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>数值稳定常数。</td>
      <td>用于防止 log(0) 导致 NaN。</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightType</td>
      <td>属性</td>
      <td>外层权重选择，'logits' 或 'probs'。</td>
      <td>'logits'用原始 target_score 作为外层权重，'probs' 用归一化概率 P = target_score / sum(target_score) 作为外层权重。</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>输出</td>
      <td>损失函数值，标量。</td>
      <td>数据类型与 targetScore 一致。shape 为 (1,)。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
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
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## aclnnLightningIndexerKLLoss

- **参数说明**

  <table style="undefined;table-layout: fixed; max-width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnLightningIndexerKLLossGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnLightningIndexerKLLoss默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 输入shape限制：
  - 支持 shape 为 (B, S, K) 或 (T, K)，B的取值范围为1\~512，最后一维 K 的取值范围为 1\~8192。
- `eps` 必须大于 0。
- `weight_type` 必须为 `'logits'` 或 `'probs'`。

## 调用示例

通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_kl_loss.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. device/context/stream 初始化
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  // target_score: (B, S, K) = (2, 4, 128), float32, 非归一化原始注意力分数
  // index_probs:  (B, S, K) = (2, 4, 128), float32, softmax 后的概率分布
  // loss:         标量 (1,), float32
  int64_t b = 2;
  int64_t s = 4;
  int64_t k = 128;
  std::vector<int64_t> inputShape = {b, s, k};
  std::vector<int64_t> lossShape = {1};

  std::vector<float> targetScoreHostData(b * s * k, 0);
  for (int64_t i = 0; i < b * s * k; i++) {
    targetScoreHostData[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f;
  }

  std::vector<float> indexProbsHostData(b * s * k, 0);
  for (int64_t bs = 0; bs < b * s; bs++) {
    float sum = 0.0f;
    for (int64_t j = 0; j < k; j++) {
      indexProbsHostData[bs * k + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      sum += indexProbsHostData[bs * k + j];
    }
    for (int64_t j = 0; j < k; j++) {
      indexProbsHostData[bs * k + j] /= sum;
    }
  }

  std::vector<float> lossHostData(1, 0.0f);

  void* targetScoreDeviceAddr = nullptr;
  void* indexProbsDeviceAddr = nullptr;
  void* lossDeviceAddr = nullptr;

  aclTensor* targetScore = nullptr;
  aclTensor* indexProbs = nullptr;
  aclTensor* loss = nullptr;

  ret = CreateAclTensor(targetScoreHostData, inputShape, &targetScoreDeviceAddr,
                        aclDataType::ACL_FLOAT, &targetScore);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexProbsHostData, inputShape, &indexProbsDeviceAddr,
                        aclDataType::ACL_FLOAT, &indexProbs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr,
                        aclDataType::ACL_FLOAT, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // eps 参数
  double eps = 1e-9;

  // 3. 调用 C(A)NN 算子库 API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnLightningIndexerKLLossGetWorkspaceSize第一段接口
  ret = aclnnLightningIndexerKLLossGetWorkspaceSize(
      targetScore, indexProbs, eps, "logits", loss, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnLightningIndexerKLLossGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // 根据workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnLightningIndexerKLLoss第二段接口
  ret = aclnnLightningIndexerKLLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerKLLoss failed. ERROR: %d\n", ret); return ret);

  // 4. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值
  PrintOutResult(lossShape, &lossDeviceAddr);

  // 6. 释放aclTensor
  aclDestroyTensor(targetScore);
  aclDestroyTensor(indexProbs);
  aclDestroyTensor(loss);

  // 7. 释放device资源
  aclrtFree(targetScoreDeviceAddr);
  aclrtFree(indexProbsDeviceAddr);
  aclrtFree(lossDeviceAddr);
  if (workspaceSize > static_cast<uint64_t>(0)) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
