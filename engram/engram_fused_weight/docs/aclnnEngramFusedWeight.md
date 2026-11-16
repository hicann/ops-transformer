# aclnnEngramFusedWeight

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

- 接口功能：Engram 门控的权重融合。将两个BF16/FP16权重逐元素相乘并升精度输出 FP32：`weight_fused = fp32(weight_hidden) * fp32(weight_embed)`。

- 计算公式（对展平后的每个元素 i）：

  $$
  y_i = \mathrm{float}(x1_i) \times \mathrm{float}(x2_i)
  $$

- 示例：

  ```python
  weight_hidden shape: [4, 5120]   # hcMult=4, hiddenSize=5120
  weight_embed shape:  [4, 5120]
  weight_fused shape:  [4, 5120]   # FP32 输出, 与输入同 shape
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnEngramFusedWeightGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEngramFusedWeight”接口执行计算。

```cpp
aclnnStatus aclnnEngramFusedWeightGetWorkspaceSize(
  const aclTensor *weightHidden,
  const aclTensor *weightEmbed,
  aclTensor       *weightFused,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnEngramFusedWeight(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnEngramFusedWeightGetWorkspaceSize

- **参数说明**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  |---|---|---|---|---|---|---|---|
  | weightHidden（const aclTensor*） | 输入 | Engram 门控的第一个权重（`q_weight`）。 | <li>支持空Tensor</li><li>必须为二维。</li> | BFLOAT16、FLOAT16| ND | [hcMult, hiddenSize] | √ |
  | weightEmbed（const aclTensor*） | 输入 | Engram 门控的第二个权重（`k_weight`）。 | <li>shape 和数据类型必须与 weightHidden 相同。</li> | BFLOAT16、FLOAT16| ND | [hcMult, hiddenSize] | √ |
  | weightFused（aclTensor*） | 输出 | 融合权重。 | <li>shape 与输入相同，dtype 为 FLOAT32。</li> | FLOAT32 | ND | [hcMult, hiddenSize] | × |
  | workspaceSize（uint64_t*） | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
  | executor（aclOpExecutor**） | 输出 | 返回op执行器，包含了算子计算流程。 | - | - | - | - | - |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的weightHidden、weightEmbed、weightFused是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>输入或输出的shape不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnEngramFusedWeight

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  |---|---|---|
  | workspace | 输入 | 在Device侧申请的workspace内存地址。 |
  | workspaceSize | 输入 | 在Device侧申请的workspace大小，由第一段接口aclnnEngramFusedWeightGetWorkspaceSize获取。 |
  | executor | 输入 | op执行器，包含了算子计算流程。 |
  | stream | 输入 | 指定执行任务的Stream。 |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnEngramFusedWeight为确定性实现。
- 两个输入必须均为二维 `[hcMult, hiddenSize]` 且 shape 和数据类型必须相同。
- 输入 dtype 仅支持 BF16、FP16，输出固定为 FP32。
- 总元素数（hcMult × hiddenSize）必须能被 256 整除。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)，完整代码参见 [test_aclnn_engram_fused_weight.cpp](../examples/test_aclnn_engram_fused_weight.cpp)。

```cpp
#include <cmath>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_engram_fused_weight.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法, 资源初始化
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.(固定写法)device/stream初始化, 参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出: 两个 BF16 权重 [hcMult, hiddenSize] = [4, 5120], 输出 FP32 同 shape
    std::vector<int64_t> weightShape = {4, 5120};
    void* weightHiddenDeviceAddr = nullptr;
    void* weightEmbedDeviceAddr = nullptr;
    void* weightFusedDeviceAddr = nullptr;
    aclTensor* weightHidden = nullptr;
    aclTensor* weightEmbed = nullptr;
    aclTensor* weightFused = nullptr;
    int64_t weightElemNum = GetShapeSize(weightShape);
    std::vector<uint16_t> weightHiddenHostData(weightElemNum, 0x3F80); // bf16 1.0
    std::vector<uint16_t> weightEmbedHostData(weightElemNum, 0xBF00);  // bf16 -0.5
    std::vector<float> weightFusedHostData(weightElemNum, 0.0f);
    ret = CreateAclTensor(weightHiddenHostData, weightShape, &weightHiddenDeviceAddr, aclDataType::ACL_BF16,
                          &weightHidden);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(weightEmbedHostData, weightShape, &weightEmbedDeviceAddr, aclDataType::ACL_BF16, &weightEmbed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(weightFusedHostData, weightShape, &weightFusedDeviceAddr, aclDataType::ACL_FLOAT, &weightFused);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API: aclnn 两段式接口 (无属性)
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnEngramFusedWeight第一段接口
    ret = aclnnEngramFusedWeightGetWorkspaceSize(weightHidden, weightEmbed, weightFused, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEngramFusedWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnEngramFusedWeight第二段接口
    ret = aclnnEngramFusedWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEngramFusedWeight failed. ERROR: %d\n", ret); return ret);

    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.拷回 host 校验: 期望 1.0 * -0.5 = -0.5
    std::vector<float> result(weightElemNum, 0.0f);
    ret = aclrtMemcpy(result.data(), result.size() * sizeof(float), weightFusedDeviceAddr,
                      weightElemNum * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    float maxErr = 0.0f;
    for (int64_t i = 0; i < weightElemNum; i++) {
        maxErr = std::max(maxErr, std::fabs(result[i] - (-0.5f)));
    }
    LOG_PRINT("aclnnEngramFusedWeight done, max error = %f\n", maxErr);

    // 6.释放aclTensor
    aclDestroyTensor(weightHidden);
    aclDestroyTensor(weightEmbed);
    aclDestroyTensor(weightFused);

    // 7.释放device资源
    aclrtFree(weightHiddenDeviceAddr);
    aclrtFree(weightEmbedDeviceAddr);
    aclrtFree(weightFusedDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
