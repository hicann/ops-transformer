# aclnnInplacePartialRotaryMul

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/inplace_partial_rotary_mul)

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
- <term>Atlas 推理系列产品</term>：x
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：执行单路旋转位置编码的Inplace计算，直接修改输入张量xRef，不产生新的输出张量。
- 计算公式：

    （1）interleave模式（rotary_mode等于1）：

    $$
    x1 = x[..., ::2].view(-1, 1)
    $$

    $$
    x2 = x[..., 1::2].view(-1, 1)
    $$

    $$
    x\_rotate = torch.cat((-x2, x1), dim=-1).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    $$

    $$
    x = x * cos + x\_rotate * sin
    $$

- partial_slice参数说明：支持对输入张量的最后一维进行部分旋转位置编码计算。通过`partial_slice`指定[start, end)范围，仅对该范围内的数据进行旋转位置编码。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnInplacePartialRotaryMulGetWorkspaceSize"接口获取入参并根据流程计算所需workspace大小，再调用"aclnnInplacePartialRotaryMul"接口执行计算。

```c++
aclnnStatus aclnnInplacePartialRotaryMulGetWorkspaceSize(
  const aclTensor   *xRef,
  const aclTensor   *cos,
  const aclTensor   *sin,
  int64_t           rotary_mode,
  const aclIntArray *partialSlice,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnInplacePartialRotaryMul(
  void          *workspace,
  uint64_t      workspaceSize,
  aclOpExecutor *executor,
  aclrtStream   stream)
```

## aclnnInplacePartialRotaryMulGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1461px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 332px">
  <col style="width: 169px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
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
      <td>xRef</td>
      <td>输入</td>
      <td>待执行旋转位置编码的张量，公式中的x。Inplace模式，xRef同时作为输出写入结果。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>位置编码张量，公式中的cos。</td>
      <td>与xRef数据类型一致，或者为FLOAT32。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>位置编码张量，公式中的sin。</td>
      <td>与xRef数据类型一致，或者为FLOAT32</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rotary_mode</td>
      <td>输入</td>
      <td>旋转模式</td>
      <td>0为half模式，1为interleave模式，当前仅支持interleave模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>partialSlice</td>
      <td>输入</td>
      <td>部分旋转的切片范围[start, end)，作用于最后一维。</td>
      <td>不传值则默认整D轴做旋转编码，start和end相等时则不做旋转编码。</td>
      <td>INT64数组</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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

  第一段接口完成入参校验，出现以下场景时报错：
  <table>
  <tr>
  <td align="center" style="width:169px;">返回值</td>
  <td align="center" style="width:125px;">错误码</td>
  <td align="center" style="width:855px;">描述</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">传入的xRef、cos或sin是空指针。</td>
  </tr>
  <tr>
  <td rowspan="4" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="4" align="left">161002</td>
  <td align="left">传入的xRef、cos、sin的数据类型不在支持范围内，或cos与sin的数据类型不一致，或精度组合不满足要求。</td>
  </tr>
  <tr><td align="left">传入的rotary_mode不为1（仅支持interleave模式）。</td></tr>
  <tr><td align="left">传入的partialSlice长度不为2，或取值范围不合法。</td></tr>
  <tr><td align="left">传入的xRef、cos、sin的形状不满足约束（维度不为4，或cos与sin形状不一致，或xRef最后一维大小超过1024，或xRef最后一维不是2的倍数，或partialSlice切片长度不是2的倍数）。</td></tr>
  </table>

## aclnnInplacePartialRotaryMul

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplacePartialRotaryMulGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream流。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnInplacePartialRotaryMul默认确定性实现。
- 不支持非连续。
- 仅支持interleave模式（rotary_mode = 1）。
- Inplace执行：输入xRef和输出共享同一个Tensor，计算结果直接写回输入xRef。
- 输入张量xRef的shape为BSND排布。各参数的shape约束可以描述如下：
  - 输入张量xRef的最后一维（D）大小不超过1024。
  - interleave模式（rotary_mode = 1）下，xRef最后一维（D）大小必须为2的倍数。partialSlice切片长度（即partialSlice[1] - partialSlice[0]）必须为2的倍数。
  - 输入张量cos、sin最后一维大小必须相同，且必须等于partialSlice的切片长度（即partialSlice[1] - partialSlice[0]）。
  - cos/sin的shape必须与xRef满足[broadcast关系](../../../docs/zh/context/broadcast_relationship.md)，且存在如下约束：

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：cos/sin的shape当前只支持BSND、B1ND、B11D、111D排布。
    <!-- end id7 -->
    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：cos/sin的shape当前只支持BS1D、B11D排布，即要求B轴保持相等。
    <!-- end id8 -->

  - partialSlice约束：sliceStart ≥ 0，sliceEnd ≥ 0，sliceEnd ≤ xRef最后一维（D）大小，sliceLength = sliceEnd - sliceStart >= 0，当sliceEnd和sliceStart相同时，不做旋转位置编码，直接返回。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_inplace_partial_rotary_mul.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // 1. 固定写法，device/stream初始化，参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {96, 1, 1, 512};
    std::vector<int64_t> cosShape = {96, 1, 1, 64};
    std::vector<int64_t> sinShape = {96, 1, 1, 64};
    int64_t rotary_mode = 1;
    std::vector<int64_t> partialSlice = {448, 512};

    void* xDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;

    // Create host data (example data)
    std::vector<float> xHostData = { /* fill with your data */ };
    std::vector<float> cosHostData = { /* fill with your data */ };
    std::vector<float> sinHostData = { /* fill with your data */ };

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray* partialSliceArray = aclCreateIntArray(partialSlice.data(), partialSlice.size());

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnInplacePartialRotaryMul第一段接口
    ret = aclnnInplacePartialRotaryMulGetWorkspaceSize(x, cos, sin, rotary_mode, partialSliceArray,
                                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    // 调用aclnnInplacePartialRotaryMul第二段接口
    ret = aclnnInplacePartialRotaryMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InplacePartialRotaryMul failed. ERROR: %d\n", ret); return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Cleanup
    aclDestroyIntArray(partialSliceArray);
    aclDestroyTensor(x);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclrtFree(xDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
