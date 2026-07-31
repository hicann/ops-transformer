# aclnnMhcSinkhornBackward

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

- 接口功能：MhcSinkhornBackward是MhcSinkhorn的反向算子。mHC（Manifold-Constrained Hyper-Connections）架构中的MhcSinkhorn算子对输入矩阵做sinkhorn变换得到双随机矩阵$\mathbf{H}_{\text{res}}$，输出的双随机矩阵的所有元素≥0、每一行之和为1且每一列之和为1 (具有范数保持、组合封闭性和凸组合几何解释三大特性)。对mHC架构中双随机矩阵$\mathbf{H}_{\text{res}}$矩阵的梯度进行sinkhorn变换的反向计算得到输入$\mathbf{H}'_{\text{res}}$的梯度。

- 计算公式

  - Sinkhorn-Knopp算法在正向计算中通过 $\mathbf{num\_iters}$ 次迭代归一化实现双随机投影，在反向传播的迭代计算中：

  - 前 $\mathbf{num\_iters}-1$ 次迭代:

  $$
  \begin{aligned}
      \mathbf{dot\_prod}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i+1}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i+1}}{\mathbf{sum\_out}_{2i+1}}, \\
      \mathbf{dot\_prod}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i}}{\mathbf{sum\_out}_{2i}}, \\
  \end{aligned}
  $$

  - 最后一次迭代：

  $$
  \begin{aligned}
      \mathbf{dot\_prod}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{1}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{1}}{\mathbf{sum\_out}_{1}}, \\
      \mathbf{dot\_prod}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\  {⋅}\  \mathbf{norm\_out}_{0}),  \\
      \mathbf{grad}_{input} &← ({\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{0}})\  {⋅}\  \mathbf{norm\_out}_{0} \\
  \end{aligned}
  $$

  - 其中：$\mathbf{grad}_\text{curr}$ 为初始梯度，$\mathbf{grad}_\text{input}$ 为输出梯度，$\mathbf{norm\_out}_\text{k}$为第$k$次归一化方向向量，$\mathbf{sum\_out}_\text{k}$ 为对应的缩放系数。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMhcSinkhornBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMhcSinkhornBackward”接口执行计算。

```c++
aclnnStatus aclnnMhcSinkhornBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* normOut,
    const aclTensor* sumOut,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```c++
aclnnStatus aclnnMhcSinkhornBackward(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnMhcSinkhornBackwardGetWorkspaceSize

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  |:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
  | gradOutput | 输入 | 表示Sinkhorn变换输出的初始梯度 | 不支持空Tensor | FLOAT32 | ND | (B,S,n,n)、(T,n,n) | √ |
  | normOut| 输入 | 表示Sinkhorn变换正向计算保存的中间结果norm| 不支持空Tensor | FLOAT32 | ND | (2\*num_iters\*n\*align_n\*B\*S)、(2\*num_iters\*n\*align_n\*T) | √ |
  | sumOut| 输入 | 表示Sinkhorn变换正向计算保存的中间结果sum | 不支持空Tensor | FLOAT32 | ND | (2\*num_iters\*align_n\*B\*S)、(2\*num_iters\*align_n\*T) | √ |
  | out| 输出 | 表示对Sinkhorn变换的输入$\mathbf{H}_{\text{res}}$的梯度 | 不支持空Tensor | FLOAT32 | ND | (B,S,n,n)、(T,n,n) | √ |
  | workspaceSize | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
  | executor | 输出 | 返回op执行器，包含了算子计算流程。 | - | - | - | - | - |

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一阶段接口完成入参校验，出现以下场景时报错。

  <table>
    <thead>
      <tr>
        <th style="width: 250px">返回值</th>
        <th style="width: 130px">错误码</th>
        <th style="width: 850px">描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="1"> 161001 </td>
        <td> gradOutput、normOut、sumOut或out存在空指针。 </td>
      </tr>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="1"> 161002 </td>
        <td> gradOutput、normOut或sumOut等输入变量的数据类型和数据格式不在支持的范围内。 </td>
      </tr>
    </tbody>
  </table>

## aclnnMhcSinkhornBackward

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 |
  |:--- |:--- |:--- |
  | workspace | 输入 | 在Device侧申请的workspace内存地址。 |
  | workspaceSize | 输入 | 在Device侧申请的workspace大小，由第一段接口aclnnMhcSinkhornBackwardGetWorkspaceSize获取。 |
  | executor | 输入 | op执行器，包含了算子计算流程。 |
  | stream | 输入 | 指定执行任务的Stream。|

- **返回值：**

  返回`aclnnStatus`状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- **确定性计算：**

  aclnnMhcSinkhornBackward默认确定性实现。

- **规格约束：**

  - num_iters：取值范围1~100，超出则报参数无效。
  - n：输入矩阵最后两维尺寸，仅支持4、6或8。
  - align_n：固定为8，是n按FP32块大小32字节对齐后的值。
  - 维度数：输入gradOutput仅支持3维(T,n,n)或4维(B,S,n,n)。

## 调用示例

  示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

  ```c++
  #include <iostream>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_mhc_sinkhorn_backward.h"

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

  // 计算Tensor形状对应的总元素数
  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (int64_t dim : shape) {
      size *= dim;
    }
    return size;
  }

  // 向上对齐到align的倍数
  int64_t CeilAlign(int64_t value, int64_t align) {
    return (value + align - 1) / align * align;
  }

  // 将Device侧Tensor数据拷贝到Host侧并打印
  void PrintTensorData(const std::vector<int64_t>& shape, void* device_addr, const char* name) {
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    // Device -> Host数据拷贝
    aclError ret = aclrtMemcpy(
        host_data.data(), size * sizeof(float),
        device_addr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("Memcpy device to host failed, error: %d\n", ret);
              return);

    // 打印前10个元素
    LOG_PRINT("%s (first 10 elements): ", name);
    for (int i = 0; i < std::min((int64_t)10, size); ++i) {
      LOG_PRINT("%.6f ", host_data[i]);
    }
    LOG_PRINT("\n");
  }

  // 初始化AscendCL环境（Device/Context/Stream）
  int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream) {
    // 1. 初始化ACL
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclInit failed, error: %d\n", ret);
              return -1);

    // 2. 设置Device
    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret);
              return -1);

    // 3. 创建Context
    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret);
              return -1);

    // 4. 设置当前Context
    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret);
              return -1);

    // 5. 创建Stream
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret);
              return -1);

    return 0;
  }

  // 创建Device侧aclTensor（含数据拷贝）
  int CreateAclTensor(
      const std::vector<float>& host_data,
      const std::vector<int64_t>& shape,
      void*& device_addr,
      aclTensor*& tensor) {
    // 1. 计算内存大小
    int64_t size = GetShapeSize(shape) * sizeof(float);

    // 2. 申请Device侧内存
    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed, error: %d\n", ret);
              return -1);

    // 3. Host -> Device数据拷贝
    ret = aclrtMemcpy(
        device_addr, size,
        host_data.data(), size,
        ACL_MEMCPY_HOST_TO_DEVICE
    );
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret);
              return -1);

    // 4. 计算Tensor的strides（连续Tensor）
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    // 5. 创建aclTensor
    tensor = aclCreateTensor(
        shape.data(), shape.size(),
        ACL_FLOAT, strides.data(), 0,
        ACL_FORMAT_ND, shape.data(), shape.size(),
        device_addr
    );
    CHECK_RET(tensor != nullptr,
              LOG_PRINT("aclCreateTensor failed\n");
              return -1);

    return 0;
  }

  // 创建输出Tensor（仅申请内存，无初始数据）
  int CreateOutputTensor(
      const std::vector<int64_t>& shape,
      void*& device_addr,
      aclTensor*& tensor) {
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc output failed, error: %d\n", ret);
              return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(
        shape.data(), shape.size(),
        ACL_FLOAT, strides.data(), 0,
        ACL_FORMAT_ND, shape.data(), shape.size(),
        device_addr
    );
    CHECK_RET(tensor != nullptr,
              LOG_PRINT("aclCreateTensor output failed\n");
              return -1);

    return 0;
  }

  int main() {
    // ========== 1. 初始化环境 ==========
    int32_t device_id = 0;  // 根据实际Device ID调整
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0,
              LOG_PRINT("InitAcl failed, error: %d\n", ret);
              return -1);

    LOG_PRINT("ACL environment initialized successfully!\n");

    // ========== 2. 定义算子参数 ==========
    // grad_y形状可以是(T, N, N)或(B, S, N, N)
    // 这里使用T形状作为示例
    int64_t T = 128;       // total length
    int64_t N = 4;         // n size (N x N矩阵)
    int64_t num_iters = 20; // Sinkhorn迭代次数

    // 计算对齐后的N大小（对齐到8）
    int64_t align_unit = 8;
    int64_t n_align_size = CeilAlign(N, align_unit);

    LOG_PRINT("Parameters: T=%lld, N=%lld, num_iters=%lld, n_align_size=%lld\n",
              T, N, num_iters, n_align_size);

    // ========== 3. 构造输入/输出张量 ==========
    // 输入grad_y形状：(T, N, N)
    std::vector<int64_t> grad_y_shape = {T, N, N};
    int64_t grad_y_size = GetShapeSize(grad_y_shape);
    std::vector<float> grad_y_host_data(grad_y_size, 1.0f);
    for (int64_t i = 0; i < grad_y_size; ++i) {
      grad_y_host_data[i] = 0.1f * static_cast<float>(i % 100);  // 填充测试数据
    }

    // 输入norm形状：[2 * num_iters * T * N * n_align_size]
    int64_t norm_size = 2 * num_iters * T * N * n_align_size;
    std::vector<int64_t> norm_shape = {norm_size};
    std::vector<float> norm_host_data(norm_size, 0.5f);
    for (int64_t i = 0; i < norm_size; ++i) {
      norm_host_data[i] = 1.0f / (1.0f + static_cast<float>(i % 1000) * 0.001f);  // 填充测试数据
    }

    // 输入sum形状：[2 * num_iters * T * n_align_size]
    int64_t sum_size = 2 * num_iters * T * n_align_size;
    std::vector<int64_t> sum_shape = {sum_size};
    std::vector<float> sum_host_data(sum_size, 1.0f);
    for (int64_t i = 0; i < sum_size; ++i) {
      sum_host_data[i] = 1.0f + static_cast<float>(i % 500) * 0.01f;  // 填充测试数据
    }

    // 输出grad_input形状与grad_y相同：(T, N, N)
    std::vector<int64_t> grad_input_shape = grad_y_shape;

    // 创建Device侧Tensor
    void* grad_y_device_addr = nullptr;
    aclTensor* grad_y_tensor = nullptr;
    ret = CreateAclTensor(grad_y_host_data, grad_y_shape, grad_y_device_addr, grad_y_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_y_tensor failed\n"); return -1);

    void* norm_device_addr = nullptr;
    aclTensor* norm_tensor = nullptr;
    ret = CreateAclTensor(norm_host_data, norm_shape, norm_device_addr, norm_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_tensor failed\n"); return -1);

    void* sum_device_addr = nullptr;
    aclTensor* sum_tensor = nullptr;
    ret = CreateAclTensor(sum_host_data, sum_shape, sum_device_addr, sum_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_tensor failed\n"); return -1);

    void* grad_input_device_addr = nullptr;
    aclTensor* grad_input_tensor = nullptr;
    ret = CreateOutputTensor(grad_input_shape, grad_input_device_addr, grad_input_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_input_tensor failed\n"); return -1);

    LOG_PRINT("All tensors created successfully!\n");
    LOG_PRINT("grad_y shape: [%lld, %lld, %lld]\n", grad_y_shape[0], grad_y_shape[1], grad_y_shape[2]);
    LOG_PRINT("norm shape: [%lld]\n", norm_shape[0]);
    LOG_PRINT("sum shape: [%lld]\n", sum_shape[0]);
    LOG_PRINT("grad_input shape: [%lld, %lld, %lld]\n", grad_input_shape[0], grad_input_shape[1], grad_input_shape[2]);

    // ========== 4. 调用第一段接口：获取Workspace大小 ==========
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclnn_ret = aclnnMhcSinkhornBackwardGetWorkspaceSize(
        grad_y_tensor,
        norm_tensor,
        sum_tensor,
        grad_input_tensor,
        &workspace_size,
        &executor
    );
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcSinkhornBackwardGetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);

    LOG_PRINT("Workspace size: %lu bytes\n", workspace_size);

    // ========== 5. 申请Workspace内存 ==========
    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
      ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret);
                return -1);
    }

    // ========== 6. 调用第二段接口：执行MhcSinkhornBackward计算 ==========
    aclnn_ret = aclnnMhcSinkhornBackward(
        workspace_addr,
        workspace_size,
        executor,
        stream
    );
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcSinkhornBackward failed, error: %d\n", aclnn_ret);
              return -1);

    // ========== 7. 同步Stream ==========
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret);
              return -1);

    LOG_PRINT("MhcSinkhornBackward compute success!\n");

    // ========== 8. 打印结果 ==========
    PrintTensorData(grad_input_shape, grad_input_device_addr, "grad_input");

    // ========== 9. 释放资源 ==========
    // 销毁Tensor
    aclDestroyTensor(grad_y_tensor);
    aclDestroyTensor(norm_tensor);
    aclDestroyTensor(sum_tensor);
    aclDestroyTensor(grad_input_tensor);

    // 释放Device内存
    aclrtFree(grad_y_device_addr);
    aclrtFree(norm_device_addr);
    aclrtFree(sum_device_addr);
    aclrtFree(grad_input_device_addr);
    if (workspace_size > 0) {
      aclrtFree(workspace_addr);
    }

    // 销毁Stream/Context，重置Device
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();

    LOG_PRINT("All resources released successfully!\n");
    return 0;
  }
  ```
