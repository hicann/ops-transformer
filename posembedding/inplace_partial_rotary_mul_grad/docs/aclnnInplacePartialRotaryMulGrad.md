# aclnnInplacePartialRotaryMulGrad

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/inplace_partial_rotary_mul_grad)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：执行局部旋转位置编码InplacePartialRotaryMul的反向计算。该算子对输入dy的D维度上切片[start, end)区域执行旋转位置编码梯度计算，计算结果inplace写回dy。
- 计算公式：

    取旋转位置编码的正向计算中，broadcast的轴列表为`dims`，在D维度上的切片范围为`[start, end)`，令参与计算的切片数据为：
    $$
    dy' = dy[..., start:end]
    $$
    $$
    cos' = cos[..., start:end]
    $$
    $$
    sin' = sin[..., start:end]
    $$
    则梯度计算公式可表达如下：

    （1）half模式（rotary_mode等于0）：

    $$
    dy1', dy2' = chunk(dy', chunks=2, dim=-1)
    $$

    $$
    cos1', cos2' = chunk(cos', chunks=2, dim=-1)
    $$

    $$
    sin1', sin2' = chunk(sin', chunks=2, dim=-1)
    $$

    $$
    dx' = cat((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1)
    $$

    dx'的结果inplace写回dy的[start, end)区间。

    （2）interleave模式（rotary_mode等于1）：

    $$
    dy1', dy2' = dy'[..., :: 2], dy'[..., 1 :: 2]
    $$

    $$
    cos1', cos2' = cos'[..., :: 2], cos'[..., 1 :: 2]
    $$

    $$
    sin1', sin2' = sin'[..., :: 2], sin'[..., 1 :: 2]
    $$

    $$
    dx' = stack((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1).reshape(dy'.shape)
    $$

    dx'的结果inplace写回dy的[start, end)区间。

    （3）quarter模式（rotary_mode等于2）：

    $$
    dy1', dy2', dy3', dy4' = chunk(dy', chunks=4, dim=-1)
    $$

    $$
    cos1', cos2', cos3', cos4' = chunk(cos', chunks=4, dim=-1)
    $$

    $$
    sin1', sin2', sin3', sin4' = chunk(sin', chunks=4, dim=-1)
    $$

    $$
    dx' = cat((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1', cos3' * dy3' + sin4' * dy4', cos4' * dy4' - sin3' * dy3'), dim=-1)
    $$

    dx'的结果inplace写回dy的[start, end)区间。

    （4）interleave-half模式（rotary_mode等于3）：

    $$
    dy1', dy2' = chunk(dy', chunks=2, dim=-1)
    $$

    $$
    cos1', cos2' = chunk(cos', chunks=2, dim=-1)
    $$

    $$
    sin1', sin2' = chunk(sin', chunks=2, dim=-1)
    $$

    $$
    dx' = stack((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1).reshape(dy'.shape)
    $$

    dx'的结果inplace写回dy的[start, end)区间。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnInplacePartialRotaryMulGradGetWorkspaceSize"接口获取入参并根据流程计算所需workspace大小，再调用"aclnnInplacePartialRotaryMulGrad"接口执行计算。

```c++
aclnnStatus aclnnInplacePartialRotaryMulGradGetWorkspaceSize(
    aclTensor  *dyRef,
    const aclTensor  *cos,
    const aclTensor  *sin,
    int64_t           rotaryMode,
    const aclIntArray *partialSlice,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnInplacePartialRotaryMulGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnInplacePartialRotaryMulGradGetWorkspaceSize

- **参数说明**

  <table style="table-layout: auto; width: 100%"><colgroup>
  <col style="width: 10%">
  <col style="width: 8%">
  <col style="width: 28%">
  <col style="width: 11%">
  <col style="width: 18%">
  <col style="width: 8%">
  <col style="width: 9%">
  <col style="width: 8%">
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
      <td>dyRef</td>
      <td>输入</td>
      <td>正向输出y的导数，inplace更新为正向输入x的导数。Inplace模式，dyRef同时作为输出写入结果。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>正向计算输入cos。</td>
      <td>与sin数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>正向计算输入sin。</td>
      <td>与cos数据类型一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rotaryMode</td>
      <td>输入</td>
      <td>旋转模式，0=half，1=interleave，2=quarter，3=interleave-half。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>partialSlice</td>
      <td>输入</td>
      <td>D维度上的切片范围。</td>
      <td>-</td>
      <td>IntArray</td>
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

  <table style="table-layout: auto; width: 100%"><colgroup>
  <col style="width: 25%">
  <col style="width: 11%">
  <col style="width: 64%">
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
      <td>传入的必选输入dyRef、cos、sin是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>传入的输入dyRef、cos、sin的数据类型和格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>传入的参数shape不满足约束说明章节中的条件。</td>
    </tr>
    <tr>
      <td>传入的rotaryMode参数不在0、1、2、3范围内。 </td>
    </tr>
  </tbody>
  </table>

## aclnnInplacePartialRotaryMulGrad

- **参数说明**

  <table style="table-layout: auto; width: 100%"><colgroup>
  <col style="width: 15%">
  <col style="width: 12%">
  <col style="width: 73%">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplacePartialRotaryMulGradGetWorkspaceSize获取。</td>
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

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 该算子仅支持Ascend 950 AI Processor。
- 该算子仅支持连续Tensor，不支持非连续Tensor。
- **该算子当前版本仅支持 interleave 模式（`rotary_mode=1`）**。half（0）、quarter（2）、interleave-half（3）模式暂未实现。
- 该算子不支持输入空Tensor（任意维度大小不能为0），不支持 slice 长度为零（即 `partial_slice[0] == partial_slice[1]`）的场景。

- 确定性计算：
  - aclnnInplacePartialRotaryMulGrad默认确定性实现。

- 输入张量dyRef支持BSND排布以及其B/S/N维度的广播变体（如111D、1SND、B1ND、BS1D、11ND、B11D、1S1D等）。各参数的shape约束可以描述如下：
  - 输入张量dyRef的最后一维大小D必须小于等于1024。
  - 输入张量cos、sin的最后一维大小必须等于切片长度(end - start)。
  - 输入张量cos和sin的shape必须完全相同，cos和sin的B、S、N维度需要与dyRef满足[broadcast关系](../../../docs/zh/context/broadcast_relationship.md)，且广播后的B、S、N必须等于dyRef的B、S、N。
  - half、interleave和interleave-half模式下，切片长度(end - start)必须能被2整除。
  - quarter模式下，切片长度(end - start)必须能被4整除。
  - 输入张量cos和sin的数据类型必须相同。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_inplace_partial_rotary_mul_grad.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> dyShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    int64_t rotaryMode = 1;
    std::vector<int64_t> partialSliceData = {0, 128};

    void* dyRefDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    aclTensor* dyRef = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;

    std::vector<float> dyHostData = {
        74,  54, 84, 125, 23,  78,  37,  72,  27, 98,  34,  107, 29,  23,  54,  60, 70,  49,  119, 54,  29,  54,
        41,  99, 27, 62,  5,   46,  108, 39,  24, 123, 33,  82,  6,   40,  88,  24, 6,   116, 38,  119, 110, 5,
        30,  79, 87, 18,  29,  100, 90,  24,  21, 93,  63,  68,  34,  112, 119, 48, 74,  43,  85,  64,  14,  49,
        128, 59, 18, 37,  123, 76,  14,  63,  10, 39,  107, 124, 79,  16,  17,  76, 80,  47,  90,  41,  58,  82,
        75,  80, 69, 37,  74,  36,  54,  26,  32, 54,  13,  100, 105, 15,  13,  69, 122, 26,  94,  59,  29,  14,
        60,  8,  24, 17,  45,  33,  107, 122, 63, 111, 75,  128, 68,  31,  105, 6,  82,  99};
    std::vector<float> cosHostData = {
        41, 37,  17, 25, 49, 25,  22,  24,  110, 120, 107, 3,   82, 66,  75,  86,  85,  115, 110, 56,  52,  39,
        86, 23,  36, 71, 20, 73,  113, 25,  114, 56,  125, 80,  95, 82,  31,  63,  99,  62,  23,  55,  30,  99,
        42, 121, 15, 24, 97, 87,  81,  67,  43,  21,  13,  9,   33, 29,  117, 10,  114, 61,  98,  15,  78,  108,
        48, 97,  1,  3,  78, 109, 57,  46,  47,  56,  50,  66,  81, 77,  17,  128, 68,  121, 47,  91,  114, 125,
        51, 108, 31, 15, 47, 78,  109, 115, 113, 26,  53,  97,  1,  111, 103, 58,  106, 68,  11,  104, 22,  79,
        61, 127, 86, 39, 33, 123, 102, 39,  64,  41,  119, 120, 61, 29,  94,  68,  36,  12};
    std::vector<float> sinHostData = {
        46, 56,  56,  101, 66,  10,  96,  16, 86,  57,  102, 66,  12,  105, 76, 58,  90,  6,   79, 128, 126, 82,
        41, 3,   45,  7,   66,  4,   46,  22, 31,  26,  37,  63,  97,  84,  91, 90,  47,  77,  90, 34,  41,  83,
        91, 108, 120, 13,  90,  32,  85,  37, 119, 31,  51,  82,  122, 125, 7,  116, 121, 108, 38, 56,  100, 20,
        97, 119, 10,  4,   53,  13,  46,  82, 103, 119, 124, 80,  23,  67,  78, 56,  119, 122, 40, 58,  128, 27,
        30, 52,  71,  42,  123, 69,  4,   5,  116, 97,  38,  107, 8,   4,   65, 120, 40,  22,  60, 44,  48,  66,
        68, 125, 4,   93,  112, 112, 113, 90, 94,  23,  104, 39,  85,  84,  64, 128, 96,  119};

    // 创建dyRef aclTensor（inplace操作，dyRef同时作为输入和输出）
    ret = CreateAclTensor(dyHostData, dyShape, &dyRefDeviceAddr, aclDataType::ACL_FLOAT, &dyRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cos aclTensor
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sin aclTensor
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建partialSlice aclIntArray
    aclIntArray* partialSlice = aclCreateIntArray(partialSliceData.data(), partialSliceData.size());

    // 调用aclnnInplacePartialRotaryMulGrad第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnInplacePartialRotaryMulGradGetWorkspaceSize(dyRef, cos, sin, rotaryMode,partialSlice,&workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnInplacePartialRotaryMulGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnInplacePartialRotaryMulGrad第二段接口
    ret = aclnnInplacePartialRotaryMulGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePartialRotaryMulGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取结果（inplace，直接从dyRefDeviceAddr读取即可）
    auto size = GetShapeSize(dyShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dyRefDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 释放资源
    aclDestroyIntArray(partialSlice);
    aclDestroyTensor(dyRef);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);

    aclrtFree(dyRefDeviceAddr);
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
