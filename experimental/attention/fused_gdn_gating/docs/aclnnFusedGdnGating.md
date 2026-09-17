# aclnnFusedGdnGating

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      √     |
|<term>Atlas 推理系列产品</term>|      √     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：FusedGdnGating是GDN（Gated Delta Network）模型中的融合门控算子，用于在一次kernel调用中完成softplus门控计算和sigmoid门控计算，减少kernel launch开销。

- 计算公式：

  门控输出$g$的计算：

  $$
  g = -e^{a\_log} \cdot \text{softplus}(a + dt\_bias)
  $$

  其中$\text{softplus}(x) = \frac{1}{\beta}\ln(1 + e^{\beta x})$，当$\beta \cdot (a + dt\_bias) > \text{threshold}$时，采用线性近似$\text{softplus}(x) \approx x$以避免数值溢出。

  门控输出$\text{beta\_output}$的计算：

  $$
  \text{beta\_output} = \text{sigmoid}(b) = \frac{1}{1 + e^{-b}}
  $$

## 接口说明

该算子提供底层ACLNN接口`aclnnFusedGdnGating`，并通过PyTorch扩展注册为`torch.ops.custom.npu_fused_gdn_gating`。PyTorch入口负责创建输出Tensor并调用底层ACLNN接口。

### PyTorch接口原型

```python
torch.ops.custom.npu_fused_gdn_gating(
    A_log: Tensor,      # [num_heads]
    a: Tensor,          # [batch, num_heads]
    b: Tensor,          # [batch, num_heads]
    dt_bias: Tensor,    # [num_heads]
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[Tensor, Tensor]
    # g:           [1, batch, num_heads], FLOAT32
    # beta_output: [1, batch, num_heads], 与a同dtype
```

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnFusedGdnGatingGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnFusedGdnGating"接口执行计算。

```cpp
aclnnStatus aclnnFusedGdnGatingGetWorkspaceSize(
    const aclTensor *aLog,
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *dtBias,
    float beta,
    float threshold,
    aclTensor *g,
    aclTensor *betaOutput,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnFusedGdnGating(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnFusedGdnGatingGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1200px">
  <colgroup>
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 300px">
    <col style="width: 200px">
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 100px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>aLog</td>
      <td>输入</td>
      <td>公式中的$a\_log$，log域参数，用于指数衰减。</td>
      <td>数据类型须与dtBias一致。</td>
      <td>FLOAT32、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>a</td>
      <td>输入</td>
      <td>公式中的$a$，门控输入数据。</td>
      <td>支持BFLOAT16或FLOAT16。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>公式中的$b$，sigmoid门控输入数据。</td>
      <td>数据类型和shape与a一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dtBias</td>
      <td>输入</td>
      <td>公式中的$dt\_bias$，偏置参数。</td>
      <td>数据类型须与aLog一致，shape与aLog一致。</td>
      <td>FLOAT32、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>公式中的$\beta$，softplus缩放系数。</td>
      <td>默认值为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>输入</td>
      <td>softplus阈值，超过该值时采用线性近似。</td>
      <td>默认值为20.0。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>g</td>
      <td>输出</td>
      <td>公式中的$g$，门控输出。</td>
      <td>数据类型固定为FLOAT32。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>betaOutput</td>
      <td>输出</td>
      <td>公式中的$\text{beta\_output}$，sigmoid输出。</td>
      <td>数据类型与a一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回Device侧需要申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回算子执行器，包含计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 650px">
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
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>aLog、a、b、dtBias、g、betaOutput的数据类型和shape不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedGdnGating

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 650px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedGdnGatingGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 输入aLog和dtBias的数据类型必须一致。
- 输入a和b的数据类型和shape必须一致。
- 输入a的shape为[batch, num_heads]，aLog和dtBias的shape为[num_heads]，且num_heads与a的第二维一致。
- 输出g的shape为[1, batch, num_heads]，数据类型固定为FLOAT32。
- 输出betaOutput的shape为[1, batch, num_heads]，数据类型与a一致。
- beta取值不能为0，否则会触发除零保护，inv_beta将被设为无穷大。
- <term>Atlas A2训练系列产品</term>：支持如下6种数据类型组合：

  <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th>序号</th>
      <th>aLog / dtBias</th>
      <th>a</th>
      <th>b</th>
      <th>g</th>
      <th>betaOutput</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>FLOAT32</td><td>BFLOAT16</td><td>BFLOAT16</td><td>FLOAT32</td><td>BFLOAT16</td></tr>
    <tr><td>2</td><td>FLOAT32</td><td>FLOAT16</td><td>FLOAT16</td><td>FLOAT32</td><td>FLOAT16</td></tr>
    <tr><td>3</td><td>BFLOAT16</td><td>BFLOAT16</td><td>BFLOAT16</td><td>FLOAT32</td><td>BFLOAT16</td></tr>
    <tr><td>4</td><td>BFLOAT16</td><td>FLOAT16</td><td>FLOAT16</td><td>FLOAT32</td><td>FLOAT16</td></tr>
    <tr><td>5</td><td>FLOAT16</td><td>BFLOAT16</td><td>BFLOAT16</td><td>FLOAT32</td><td>BFLOAT16</td></tr>
    <tr><td>6</td><td>FLOAT16</td><td>FLOAT16</td><td>FLOAT16</td><td>FLOAT32</td><td>FLOAT16</td></tr>
  </tbody>
  </table>

- <term>Atlas 推理系列产品</term>和<term>Atlas 200I/500 A2 推理产品</term>：仅支持全FLOAT16数据类型组合（即aLog、a、b、dtBias均为FLOAT16，g为FLOAT32，betaOutput为FLOAT16）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

### PyTorch调用示例

```python
import torch
import torch_npu
import custom_ops  # noqa: F401  注册torch.ops.custom.npu_fused_gdn_gating

torch_npu.npu.set_device(0)

num_heads, batch = 8, 4
# 注意：Atlas 推理系列产品（310P）上 aLog/dtBias 须为 FLOAT16，
# 即所有输入均为全FLOAT16组合（见约束说明）。
A_log = torch.randn(num_heads, dtype=torch.float16).npu()
a = torch.randn(batch, num_heads, dtype=torch.float16).npu()
b = torch.randn(batch, num_heads, dtype=torch.float16).npu()
dt_bias = torch.randn(num_heads, dtype=torch.float16).npu()

g, beta_output = torch.ops.custom.npu_fused_gdn_gating(
    A_log, a, b, dt_bias, beta=1.0, threshold=20.0,
)
# g:           [1, batch, num_heads], FLOAT32
# beta_output: [1, batch, num_heads], FLOAT16
```

### aclnn调用示例

```C++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_gdn_gating.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
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
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int64_t batch = 4;
  int64_t numHeads = 8;

  std::vector<int64_t> aLogShape = {numHeads};
  std::vector<int64_t> aShape = {batch, numHeads};
  std::vector<int64_t> bShape = {batch, numHeads};
  std::vector<int64_t> dtBiasShape = {numHeads};
  std::vector<int64_t> gShape = {1, batch, numHeads};
  std::vector<int64_t> betaOutputShape = {1, batch, numHeads};

  void* aLogDeviceAddr = nullptr;
  void* aDeviceAddr = nullptr;
  void* bDeviceAddr = nullptr;
  void* dtBiasDeviceAddr = nullptr;
  void* gDeviceAddr = nullptr;
  void* betaOutputDeviceAddr = nullptr;

  aclTensor* aLog = nullptr;
  aclTensor* a = nullptr;
  aclTensor* b = nullptr;
  aclTensor* dtBias = nullptr;
  aclTensor* g = nullptr;
  aclTensor* betaOutput = nullptr;

  std::vector<float> aLogHostData(numHeads, 0.5);
  std::vector<uint16_t> aHostData(batch * numHeads, 0);
  std::vector<uint16_t> bHostData(batch * numHeads, 0);
  std::vector<float> dtBiasHostData(numHeads, 0.1);
  std::vector<float> gHostData(batch * numHeads, 0);
  std::vector<uint16_t> betaOutputHostData(batch * numHeads, 0);

  ret = CreateAclTensor(aLogHostData, aLogShape, &aLogDeviceAddr, aclDataType::ACL_FLOAT, &aLog);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_BF16, &a);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(bHostData, bShape, &bDeviceAddr, aclDataType::ACL_BF16, &b);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dtBiasHostData, dtBiasShape, &dtBiasDeviceAddr, aclDataType::ACL_FLOAT, &dtBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gHostData, gShape, &gDeviceAddr, aclDataType::ACL_FLOAT, &g);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaOutputHostData, betaOutputShape, &betaOutputDeviceAddr, aclDataType::ACL_BF16, &betaOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  float beta = 1.0f;
  float threshold = 20.0f;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnFusedGdnGatingGetWorkspaceSize(aLog, a, b, dtBias, beta, threshold,
                                             g, betaOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedGdnGatingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnFusedGdnGating(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedGdnGating failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  PrintOutResult(gShape, &gDeviceAddr);

  aclDestroyTensor(aLog);
  aclDestroyTensor(a);
  aclDestroyTensor(b);
  aclDestroyTensor(dtBias);
  aclDestroyTensor(g);
  aclDestroyTensor(betaOutput);

  aclrtFree(aLogDeviceAddr);
  aclrtFree(aDeviceAddr);
  aclrtFree(bDeviceAddr);
  aclrtFree(dtBiasDeviceAddr);
  aclrtFree(gDeviceAddr);
  aclrtFree(betaOutputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```

## 相关资源

- PyTorch扩展构建与安装见[torch_ops_extension说明](../torch_ops_extension/README.md)。
- 更多测试与调用示例见[pytest说明](../tests/pytest/README.md)。
