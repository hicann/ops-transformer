# aclnnMhcPostBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：mhc_post基于一系列计算对MHC（Manifold-Constrained Hyper-Connection）架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_j$进行ResMapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。该算子实现前述过程的反向功能。

- 计算公式：

  当传入hRes（hRes不为nullptr）时：
  $$
  grad\_x = H_{l}^{res} \times grad\_output\\
  grad\_h\_res = x_{l} \times {grad\_output}^{T}
  $$
  $$
  grad\_h\_out=({grad\_output} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
  grad\_h\_post=({grad\_output} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
  $$

  当不传hRes（hRes为nullptr）时：
  $$
  grad\_x = grad\_output
  $$
  $$
  grad\_h\_res = x_{l} \times {grad\_output}^{T}
  $$
  $$
  grad\_h\_out=({grad\_output} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
  grad\_h\_post=({grad\_output} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
  $$

  > 注：hRes作为可选参数（支持置为nullptr）仅在 <term>Ascend 950PR/Ascend 950DT</term> 上支持。

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMhcPostBackwardGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnMhcPostBackward”接口执行计算。

```c++
aclnnStatus aclnnMhcPostBackwardGetWorkspaceSize(
    const aclTensor     *gradOutput,
    const aclTensor     *x,
    const aclTensor     *hRes,
    const aclTensor     *hOut,
    const aclTensor     *hPost,
    aclTensor           *gradX,
    aclTensor           *gradHRes,
    aclTensor           *gradHOut,
    aclTensor           *gradHPost,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnMhcPostBackward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnMhcPostBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width:  1400px"><colgroup>
  <col style="width: 145px">
  <col style="width: 90px">
  <col style="width: 441px">
  <col style="width: 158px">
  <col style="width: 186px">
  <col style="width: 80px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>待计算的数据，表示网络中MHC层的输入数据</td>
      <td>
        <ul>
          <li>不支持空Tensor</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[B,S,N,D]、[T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待计算的数据，表示网络中MHC层的输入数据</td>
      <td>
        <ul>
          <li>不支持空Tensor</li>
        </ul>
      </td>
      <td>数据类型与gradOutput一致</td>
      <td>ND</td>
      <td>[B,S,N,D]、[T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hRes</td>
      <td>输入</td>
      <td>MHC的hRes变换矩阵</td>
      <td>
        <ul>
          <li>可选参数，不传时置为nullptr</li>
          <li>不支持空Tensor</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[B,S,N,N]、[T,N,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hOut</td>
      <td>输入</td>
      <td>Atten/MLP层的输出</td>
      <td>
        <ul>
          <li>不支持空Tensor</li>
        </ul>
      </td>
      <td>数据类型与gradOutput一致</td>
      <td>ND</td>
      <td>[B,S,D]、[T,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hPost</td>
      <td>输入</td>
      <td>MHC的hPost变换矩阵</td>
      <td>
        <ul>
          <li>不支持空Tensor</li>
        </ul>
      </td>
      <td>数据类型与hRes一致</td>
      <td>ND</td>
      <td>[B,S,N]、[T,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradX</td>
      <td>输出</td>
      <td>网络中MHC层的输入数据x的梯度</td>
      <td>-</td>
      <td>数据类型与gradOutput一致</td>
      <td>ND</td>
      <td>[B,S,N,D]、[T,N,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradHRes</td>
      <td>输出</td>
      <td>网络中MHC层的输入数据hRes的梯度</td>
      <td>-</td>
      <td>数据类型与hRes一致</td>
      <td>ND</td>
      <td>[B,S,N,N]、[T,N,N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradHOut</td>
      <td>输出</td>
      <td>网络中MHC层的输入数据hOut的梯度</td>
      <td>-</td>
      <td>数据类型与hOut一致</td>
      <td>ND</td>
      <td>[B,S,D]、[T,D]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradHPost</td>
      <td>输出</td>
      <td>网络中MHC层的输入数据h_post的梯度</td>
      <td>-</td>
      <td>数据类型与hPost一致</td>
      <td>ND</td>
      <td>[B,S,N]、[T,N]</td>
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
          <td>gradOutput、x、hRes、hOut、hPost存在空指针。</td>
        </tr>
        <tr>
          <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
          <td rowspan="3">161002</td>
          <td>gradOutput、x、hRes、hOut、hPost的数据类型不在支持的范围内。</td>
        </tr>
          <tr>
          <td>gradOutput、x、hRes、hOut、hPost的shape维度或取值不在支持的范围内。</td>
        </tr>
        <tr>
          <td>gradOutput、x、hRes、hOut、hPost的数据类型或shape不匹配。</td>
        </tr>
  </tbody></table>

## aclnnMhcPostBackward

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPostBackwardGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的AscendCL stream流。</td>
        </tr>
    </tbody>
  </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：参数说明中维度N的取值目前仅支持4，维度D的取值需要128对齐并且取值范围为[1,100000]。
- <term>Ascend 950PR/Ascend 950DT</term>：参数说明中维度N的取值目前仅支持4、6和8。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_post_backward.h"
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
    // Fixed writing method, AscendCL initialization.
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
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    LOG_PRINT("ACL environment initialized successfully!\n");

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    // Example: BSND format (B, S, n, D)
    std::vector<int64_t> gradYShape = {1, 1024, 4, 5120};   // BSND
    std::vector<int64_t> xShape = {1, 1024, 4, 5120};   // BSND
    std::vector<int64_t> hResShape = {1, 1024, 4, 4};   // BSND
    std::vector<int64_t> hOutShape = {1, 1024, 5120};   // BSD
    std::vector<int64_t> hPostShape = {1, 1024, 4};     // BSn
    std::vector<int64_t> gradXShape = {1, 1024, 4, 5120};   // BSND
    std::vector<int64_t> gradHResShape = {1, 1024, 4, 4};   // BSND
    std::vector<int64_t> gradHOutShape = {1, 1024, 5120};   // BSD
    std::vector<int64_t> gradHPostShape = {1, 1024, 4};     // BSn

    void *gradYDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;
    void *hResDeviceAddr = nullptr;
    void *hOutDeviceAddr = nullptr;
    void *hPostDeviceAddr = nullptr;
    void *gradXDeviceAddr = nullptr;
    void *gradHResDeviceAddr = nullptr;
    void *gradHOutDeviceAddr = nullptr;
    void *gradHPostDeviceAddr = nullptr;

    aclTensor *gradYTensor = nullptr;
    aclTensor *xTensor = nullptr;
    aclTensor *hResTensor = nullptr;
    aclTensor *hOutTensor = nullptr;
    aclTensor *hPostTensor = nullptr;
    aclTensor *gradXTensor = nullptr;
    aclTensor *gradHResTensor = nullptr;
    aclTensor *gradHOutTensor = nullptr;
    aclTensor *gradHPostTensor = nullptr;

    int64_t gradYShapeSize = GetShapeSize(gradYShape);
    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t hResShapeSize = GetShapeSize(hResShape);
    int64_t hOutShapeSize = GetShapeSize(hOutShape);
    int64_t hPostShapeSize = GetShapeSize(hPostShape);
    int64_t gradXShapeSize = GetShapeSize(gradXShape);
    int64_t gradHResShapeSize = GetShapeSize(gradHResShape);
    int64_t gradHOutShapeSize = GetShapeSize(gradHOutShape);
    int64_t gradHPostShapeSize = GetShapeSize(gradHPostShape);
    std::vector<aclFloat16> gradYHostData(gradYShapeSize, aclFloatToFloat16(1.0f));
    std::vector<aclFloat16> xHostData(xShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hResHostData(hResShapeSize, 1.0f);
    std::vector<aclFloat16> hOutHostData(hOutShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hPostHostData(hPostShapeSize, 1.0f);
    std::vector<aclFloat16> gradXHostData(gradXShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> gradHResHostData(gradHResShapeSize, 1.0f);
    std::vector<aclFloat16> gradHOutHostData(gradHOutShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> gradHPostHostData(gradHPostShapeSize, 1.0f);

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT16, &gradYTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
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

    // Create gradX aclTensor.
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT16, &gradXTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create gradHRes aclTensor.
    ret = CreateAclTensor(gradHResHostData, gradHResShape, &gradHResDeviceAddr, aclDataType::ACL_FLOAT, &gradHResTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create gradHOut aclTensor.
    ret = CreateAclTensor(gradHOutHostData, gradHOutShape, &gradHOutDeviceAddr, aclDataType::ACL_FLOAT16, &gradHOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create gradHPost aclTensor.
    ret = CreateAclTensor(gradHPostHostData, gradHPostShape, &gradHPostDeviceAddr, aclDataType::ACL_FLOAT, &gradHPostTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnMhcPostBackwardGetWorkspaceSize(
        gradYTensor, xTensor, hResTensor, hOutTensor, hPostTensor, gradXTensor, gradHResTensor, gradHOutTensor,
        gradHPostTensor, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
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
    ret = aclnnMhcPostBackward(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackward failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }
    //gradX, gradHRes, gradHOut, gradHPost
    // 5. Retrieve the output value, copy the gradX from the device side memory to the host side.
    auto gradXsize = GetShapeSize(gradXShape);
    std::vector<aclFloat16> gradXData(gradXsize, aclFloatToFloat16(0.0f));
    ret = aclrtMemcpy(gradXData.data(), gradXData.size() * sizeof(gradXData[0]), gradXDeviceAddr,
                      gradXsize * sizeof(gradXData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradX from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    auto gradHRessize = GetShapeSize(gradHResShape);
    std::vector<float> gradHResData(gradHRessize, 0.0f);
    ret = aclrtMemcpy(gradHResData.data(), gradHResData.size() * sizeof(gradHResData[0]), gradHResDeviceAddr,
                      gradHRessize * sizeof(gradHResData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradHRes from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    auto gradHOutsize = GetShapeSize(gradHOutShape);
    std::vector<aclFloat16> gradHOutData(gradHOutsize, aclFloatToFloat16(0.0f));
    ret = aclrtMemcpy(gradHOutData.data(), gradHOutData.size() * sizeof(gradHOutData[0]), gradHOutDeviceAddr,
                      gradHOutsize * sizeof(gradHOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradHOut from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    auto gradHPostsize = GetShapeSize(gradHPostShape);
    std::vector<float> gradHPostData(gradHPostsize, 0.0f);
    ret = aclrtMemcpy(gradHPostData.data(), gradHPostData.size() * sizeof(gradHPostData[0]), gradHPostDeviceAddr,
                      gradHPostsize * sizeof(gradHPostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradHPost from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    // 6. Release resources.
    aclDestroyTensor(gradYTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(hResTensor);
    aclDestroyTensor(hOutTensor);
    aclDestroyTensor(hPostTensor);
    aclDestroyTensor(gradXTensor);
    aclDestroyTensor(gradHResTensor);
    aclDestroyTensor(gradHOutTensor);
    aclDestroyTensor(gradHPostTensor);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(hResDeviceAddr);
    aclrtFree(hOutDeviceAddr);
    aclrtFree(hPostDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradHResDeviceAddr);
    aclrtFree(gradHOutDeviceAddr);
    aclrtFree(gradHPostDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
