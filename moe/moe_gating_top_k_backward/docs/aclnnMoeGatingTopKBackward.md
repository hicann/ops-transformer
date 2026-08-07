# aclnnMoeGatingTopKBackward

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_gating_top_k_backward)

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

- 接口功能：完成MoE（Mixture of Experts）门控Top-K选择的反向梯度计算。该算子是aclnnMoeGatingTopK的反向算子，根据前向算子输出的归一化得分（xNorm）、上游梯度（gradY）和专家索引（expertIdx），计算输入得分矩阵的梯度（gradX）。支持sigmoid模式（normType=1）。
- 计算公式（sigmoid模式，normType=1）：

  1. 缩放梯度

      $$
      gradYScaled_{ip} = routedScalingFactor \cdot gradY_{ip}
      $$

  2. 正向renorm的反向传播

      $$
      wPrime_{ip} = xNorm_{i,\ expertIdx_{ip}}
      $$

      $$
      D_i = \sum_{p} wPrime_{ip} + eps
      $$

      $$
      w_{ip} = \frac{wPrime_{ip}}{D_i}
      $$

      $$
      beta_i = \sum_{p} w_{ip} \cdot gradYScaled_{ip}
      $$

      $$
      gradWPrime_{ip} = \frac{gradYScaled_{ip} - beta_i}{D_i}
      $$

  3. 散射到完整维度

      $$
      gradNormX_{ij} = \sum_{p:\ expertIdx_{ip}=j} gradWPrime_{ip}
      $$

  4. Sigmoid反向传播

      $$
      gradX_{ij} = xNorm_{ij} \cdot (1 - xNorm_{ij}) \cdot gradNormX_{ij}
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnMoeGatingTopKBackwardGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMoeGatingTopKBackward"接口执行计算。

```cpp
aclnnStatus aclnnMoeGatingTopKBackwardGetWorkspaceSize(
  const aclTensor  *xNorm,
  const aclTensor  *gradY,
  const aclTensor  *expertIdx,
  int64_t           renorm,
  int64_t           normType,
  double            routedScalingFactor,
  double            eps,
  const aclTensor  *out,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeGatingTopKBackward(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnMoeGatingTopKBackwardGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
   <col style="width: 230px">
  <col style="width: 120px">
  <col style="width: 230px">
  <col style="width: 330px">
  <col style="width: 210px">
  <col style="width: 100px">
  <col style="width: 140px">
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
      <td>xNorm（aclTensor*）</td>
      <td>输入</td>
      <td>计算的输入，对应公式中的xNorm。</td>
      <td><ul><li>不支持空Tensor。</li><li>要求是一个2D的Tensor，维度为[M,N]。</li><li>最后一维（专家数N）要求大于等于2，并小于等于2048。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradY（aclTensor*）</td>
      <td>输入</td>
      <td>表示前向算子输出yOut的上游梯度，对应公式中的gradY。</td>
      <td><ul><li>不支持空Tensor。</li><li>要求是一个2D的Tensor，维度为[M,K]，K的范围要求大于等于1，并小于等于N。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdx（aclTensor*）</td>
      <td>输入</td>
      <td>表示前向算子输出的top-k专家索引，对应公式中的expertIdx。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape要求与gradY一致，维度为[M,K]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>renorm（int64_t）</td>
      <td>输入</td>
      <td>表示前向算子在softmax模式下renorm标记。</td>
      <td>0：不做renorm；1：需要做renorm；预留参数，当前仅支持sigmoid模式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normType（int64_t）</td>
      <td>输入</td>
      <td>表示norm函数类型。</td>
      <td>1表示使用Sigmoid函数，0表示Softmax函数。当前仅支持1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>routedScalingFactor（double）</td>
      <td>输入</td>
      <td>表示前向算子使用的routed_scaling_factor系数，对应公式中的routedScalingFactor。</td>
      <td></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps（double）</td>
      <td>输入</td>
      <td>表示前向计算使用的防除零常数，对应公式中的eps。</td>
      <td></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示前向算子输入参数x的梯度，对应公式中的gradX。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与gradY需要保持一致。</li><li>shape与xNorm需要一致，维度为[M,N]。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
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
  </tbody></table>

- **返回值**

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
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>xNorm、gradY、expertIdx、out存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="5"> ACLNN_ERR_PARAM_INVALID </td>
      <td rowspan="5"> 161002 </td>
      <td>xNorm、gradY或expertIdx的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>out的数据类型与xNorm不一致。</td>
    </tr>
    <tr>
      <td>xNorm、gradY或expertIdx的shape维度不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradY与expertIdx的shape不一致。</td>
    </tr>
    <tr>
      <td>out与xNorm的shape不一致。</td>
    </tr>
  </tbody></table>

## aclnnMoeGatingTopKBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeGatingTopKBackwardGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnMoeGatingTopKBackward默认确定性实现。

<!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：
  - xNorm最后一维的大小（即专家数N）取值范围为[2, 2048]。
  - gradY最后一维的大小（即K）取值范围为[1, N]。

<!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_backward.h"

#define CHECK_RET(cond, return_expr)                                                                                  \
    do {                                                                                                              \
        if (!(cond)) {                                                                                                \
            return_expr;                                                                                              \
        }                                                                                                             \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                       \
    do {                                                                                                              \
        printf(message, ##__VA_ARGS__);                                                                               \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
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
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出
    // xNorm: [M, N] = [4, 8], float32, sigmoid输出值在(0,1)之间
    std::vector<int64_t> xNormShape = {4, 8};
    std::vector<float> xNormHostData = {0.5f, 0.7f, 0.3f, 0.8f, 0.2f, 0.6f, 0.9f, 0.4f, 0.6f, 0.4f, 0.8f,
                                        0.1f, 0.7f, 0.5f, 0.3f, 0.9f, 0.2f, 0.9f, 0.5f, 0.6f, 0.8f, 0.3f,
                                        0.7f, 0.4f, 0.8f, 0.3f, 0.6f, 0.5f, 0.4f, 0.7f, 0.2f, 0.9f};
    // gradY: [M, K] = [4, 2], float32
    std::vector<int64_t> gradYShape = {4, 2};
    std::vector<float> gradYHostData = {1.0f, 0.5f, 0.8f, 0.3f, 0.6f, 0.9f, 0.4f, 0.7f};
    // expertIdx: [M, K] = [4, 2], int32
    std::vector<int64_t> expertIdxShape = {4, 2};
    std::vector<int32_t> expertIdxHostData = {3, 6, 2, 7, 1, 4, 0, 7};
    // gradX: [M, N] = [4, 8], float32
    std::vector<int64_t> gradXShape = {4, 8};
    std::vector<float> gradXHostData(32, 0);

    void *xNormDeviceAddr = nullptr;
    void *gradYDeviceAddr = nullptr;
    void *expertIdxDeviceAddr = nullptr;
    void *gradXDeviceAddr = nullptr;
    aclTensor *xNorm = nullptr;
    aclTensor *gradY = nullptr;
    aclTensor *expertIdx = nullptr;
    aclTensor *gradX = nullptr;

    ret = CreateAclTensor(xNormHostData, xNormShape, &xNormDeviceAddr, aclDataType::ACL_FLOAT, &xNorm);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用aclnnMoeGatingTopKBackward
    int64_t renorm = 0;
    int64_t normType = 1; // sigmoid模式
    double routedScalingFactor = 2.5;
    double eps = 1e-20;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    ret = aclnnMoeGatingTopKBackwardGetWorkspaceSize(xNorm, gradY, expertIdx, renorm, normType, routedScalingFactor,
                                                     eps, gradX, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    ret = aclnnMoeGatingTopKBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKBackward failed. ERROR: %d\n", ret); return ret);

    // 4.同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值
    auto size = GetShapeSize(gradXShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradX[%ld] is: %f\n", i, resultData[i]);
    }

    // 6.释放aclTensor
    aclDestroyTensor(xNorm);
    aclDestroyTensor(gradY);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(gradX);

    // 7.释放device资源
    aclrtFree(xNormDeviceAddr);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
