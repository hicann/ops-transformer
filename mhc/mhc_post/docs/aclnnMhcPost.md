# aclnnMhcPost

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_post)

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

- 接口功能：MhcPost基于一系列计算对mHC架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_l$进行Res Mapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。

- 计算公式：

  当 hRes 提供时：

  $$
  x_{l+1} = (H_{l}^{res})^{T} \times x_l + h_{l}^{out} \otimes H_{t}^{post}
  $$

  当 hRes 缺省时（仅 Ascend 950 支持）：

  $$
  x_{l+1} = x_l + h_{l}^{out} \otimes H_{t}^{post}
  $$

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnMhcPostGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMhcPost"接口执行计算。

```c++
aclnnStatus aclnnMhcPostGetWorkspaceSize(
    const aclTensor  *x,
    const aclTensor  *hRes,
    const aclTensor  *hOut,
    const aclTensor  *hPost,
    aclTensor        *out,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnMhcPost(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnMhcPostGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1470px"><colgroup>
  <col style="width: 145px">
  <col style="width: 120px">
  <col style="width: 441px">
  <col style="width: 158px">
  <col style="width: 186px">
  <col style="width: 120px">
  <col style="width: 155px">
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
      <td>x</td>
      <td>输入</td>
      <td>待计算的张量，表示网络中mHC层的输入数据。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,N,D]、[T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hRes</td>
      <td>输入（可选）</td>
      <td>mHC的hRes变换矩阵，是做完sinkhorn变换后的双随机矩阵。传nullptr时退化为直接残差连接（仅 Ascend 950 支持）。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,N,N]、[T,N,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hOut</td>
      <td>输入</td>
      <td>Atten/MLP层的输出。</td>
      <td>数据类型与x相同。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,D]、[T,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hPost</td>
      <td>输入</td>
      <td>mHC的hPost变换矩阵。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,N]、[T,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>网络中mHC层的输出数据，作为下一层的输入。</td>
      <td>数据类型与x相同。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,N,D]、[T,N,D]</td>
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
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
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
              <td>x、hOut、hPost、out存在空指针。</td>
          </tr>
          <tr>
              <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
              <td rowspan="3">161002</td>
              <td>x、hRes（非空时）、hOut、hPost、out的数据类型不在支持的范围内。</td>
          </tr>
            <tr>
              <td>x、hRes（非空时）、hOut、hPost、out的shape维度不在支持的范围内。</td>
          </tr>
          <tr>
              <td>x、hRes（非空时）、hOut、hPost、out的数据类型或shape不匹配。</td>
          </tr>
            <tr>
              <td>ACLNN_ERR_INNER_NULLPTR</td>
              <td>561103</td>
              <td>n不等于4; d不在[1, 100000]范围内；d不能被128整除。</td>
          </tr>
      </tbody>
  </table>

## aclnnMhcPost

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 969px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPostGetWorkspaceSize获取。</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的stream流。</td>
      </tr>
  </tbody>
  </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- aclnnMhcPost默认确定性实现。

<!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：hRes 支持传入 nullptr，此时退化为直接残差连接。
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：hRes 为必传参数，不支持传入 nullptr。

  | 规格项   | 规格               | 规格说明                                |
  | :------- | :----------------- | :------------------------------------- |
  | n        | 4                  | 固定为4                              |
  | d        | 范围1到100000       | 128的倍数                             |

<!-- end id8 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_post.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    // Fixed writing method, resource initialization.
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to the API's external interface list.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    // Example: BSND format (B, S, n, D)
    std::vector<int64_t> xShape = {1, 1024, 4, 5120};   // BSND
    std::vector<int64_t> hResShape = {1, 1024, 4, 4};   // BSND
    std::vector<int64_t> hOutShape = {1, 1024, 5120};   // BSD
    std::vector<int64_t> hPostShape = {1, 1024, 4};     // BSn
    std::vector<int64_t> outShape = {1, 1024, 4, 5120};  // BSND

    void *xDeviceAddr = nullptr;
    void *hResDeviceAddr = nullptr;
    void *hOutDeviceAddr = nullptr;
    void *hPostDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *xTensor = nullptr;
    aclTensor *hResTensor = nullptr;
    aclTensor *hOutTensor = nullptr;
    aclTensor *hPostTensor = nullptr;
    aclTensor *outTensor = nullptr;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t hResShapeSize = GetShapeSize(hResShape);
    int64_t hOutShapeSize = GetShapeSize(hOutShape);
    int64_t hPostShapeSize = GetShapeSize(hPostShape);
    int64_t outShapeSize = GetShapeSize(outShape);

    std::vector<aclFloat16> xHostData(xShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hResHostData(hResShapeSize, 1.0f);
    std::vector<aclFloat16> hOutHostData(hOutShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hPostHostData(hPostShapeSize, 1.0f);
    std::vector<aclFloat16> outHostData(outShapeSize, aclFloatToFloat16(0.0f));

    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hRes aclTensor.
    ret = CreateAclTensor(hResHostData, hResShape, &hResDeviceAddr, aclDataType::ACL_FLOAT, &hResTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hOut aclTensor.
    ret = CreateAclTensor(hOutHostData, hOutShape, &hOutDeviceAddr, aclDataType::ACL_FLOAT16, &hOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hPost aclTensor.
    ret = CreateAclTensor(hPostHostData, hPostShape, &hPostDeviceAddr, aclDataType::ACL_FLOAT, &hPostTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnMhcPostGetWorkspaceSize(
        xTensor, hResTensor, hOutTensor, hPostTensor, outTensor,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnMhcPost(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPost failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. Retrieve the output value, copy the result from the device side memory to the host side.
    auto size = GetShapeSize(outShape);
    std::vector<aclFloat16> resultData(size, aclFloatToFloat16(0.0f));
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    // 6. Release resources.
    aclDestroyTensor(xTensor);
    aclDestroyTensor(hResTensor);
    aclDestroyTensor(hOutTensor);
    aclDestroyTensor(hPostTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(xDeviceAddr);
    aclrtFree(hResDeviceAddr);
    aclrtFree(hOutDeviceAddr);
    aclrtFree(hPostDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
