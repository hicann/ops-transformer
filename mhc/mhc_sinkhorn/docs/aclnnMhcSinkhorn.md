# aclnnMhcSinkhorn

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

- 接口功能：对mHC架构中的$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出迭代过程中的中间归一化结果（norm_out）和求和结果（sum_out），用于反向梯度计算。

## 计算公式

Sinkhorn变换共执行$\mathbf{num\_iters}$次迭代，迭代过程中生成中间归一化结果$\mathbf{norm\_out}[k]$和求和结果$\mathbf{sum\_out}[k]$，最终输出最后一次迭代的$\mathbf{norm\_out}$作为变换结果。

第一次迭代（初始化）：

$$
\begin{aligned}
    \mathbf{norm\_out}[0] &= \text{softmax}(\mathbf{x}, \dim=-1) + \epsilon, \\
    \mathbf{sum\_out}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{norm\_out}[0] + \epsilon, \\
    \mathbf{norm\_out}[1] &= \frac{\mathbf{norm\_out}[0]}{\mathbf{sum\_out}[1]}, \\
\end{aligned}
$$

第$i$次迭代（$i = 1, 2, \dots, \mathbf({num\_iters}-1)$）：

$$
\begin{aligned}
    \mathbf{sum\_out}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \mathbf{norm\_out}[2i-1] + \epsilon, \\
    \mathbf{norm\_out}[2i] &= \frac{\mathbf{norm\_out}[2i-1]}{\mathbf{sum\_out}[2i]}, \\
    \mathbf{sum\_out}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{norm\_out}[2i] + \epsilon, \\
    \mathbf{norm\_out}[2i+1] &= \frac{\mathbf{norm\_out}[2i]}{\mathbf{sum\_out}[2i+1]}, \\
\end{aligned}
$$

### 最终输出

$$
\text{output} = \mathbf{norm\_out}[2 \times \mathbf{num\_iters} - 1]
$$

---

### 🔍 符号说明

| 符号 | 含义 |
|------|------|
| $\mathbf{x}$ | 输入张量（mHC层的$\mathbf{H}'_{\text{res}}$矩阵） |
| $\epsilon$ | 防除零参数（对应入参`eps`） |
| $\text{softmax}(\cdot, \dim=-1)$ | 在最后一维执行softmax归一化 |
| $\sum_{\dim=d,\text{keepdim}=\text{True}}$ | 在指定维度$d$上求和并保持维度 |
| $\mathbf{norm\_out}[k]$ | 第$k$步归一化中间结果 |
| $\mathbf{sum\_out}[k]$ | 第$k$步求和中间结果 |
| $\mathbf{num\_iters}$ | 迭代次数（入参） |

## 函数原型

算子采用两段式接口调用：需先调用`aclnnMhcSinkhornGetWorkspaceSize`获取计算所需的Device侧内存大小，再调用`aclnnMhcSinkhorn`执行实际计算。

```c++
aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(
    const aclTensor *x,
    float eps,
    int64_t numIters,
    aclTensor *output,
    aclTensor *normOut,
    aclTensor *sumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnMhcSinkhorn(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnMhcSinkhornGetWorkspaceSize

- **参数说明**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  |:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
  | x | 输入 | 待计算数据，mHC层的$\mathbf{H}'_{\text{res}}$矩阵 | 必选参数，不能为空Tensor | FLOAT32 | ND | (B,S,n,n)、(T,n,n) | √ |
  | eps | 输入 | 归一化防除零参数 | 建议值：1e-6 | FLOAT32 | - | - | - |
  | numIters| 输入 | 迭代次数 | 建议值：20；范围：1~100 | INT64 | - | - | - |
  | output | 输出 | MhcSinkhorn变换最终结果（双随机矩阵$\mathbf{H}_{\text{res}}$） | 必选输出，维度与输入x一致 | FLOAT32 | ND | (B,S,n,n)、(T,n,n) | √ |
  | normOut | 输出 | 迭代过程中的归一化中间结果，用于反向 | 可选参数，传空指针时不输出该结果 | FLOAT32 | ND | [2\*num_iters\*n\*n_align\*B\*S]、[2\*num_iters\*n\*n_align\*T]，n_align为n按8对齐 | √ |
  | sumOut | 输出 | 迭代过程中的求和中间结果，用于反向 | 可选参数，传空指针时不输出该结果 | FLOAT32 | ND | [2\*num_iters\*n_align\*B\*S]、[2\*num_iters\*n_align\*T]，n_align为n按8对齐 | √ |
  | workspaceSize | 输出 | 计算所需的Device侧workspace内存大小（字节） | 由算子内部计算得出，用于后续申请内存 | UINT64 | - | - | - |
  | executor | 输出 | 算子执行器，包含计算流程和参数信息 | 需传递给第二段接口使用 | aclOpExecutor* | - | - | - |

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  | 返回值 | 错误码 | 描述 |
  |:--- |:--- |:--- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选参数（x/output）或输出参数（workspaceSize/executor）为空指针。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | 1. x的数据类型/格式非FLOAT32/ND；<br>2. numIters超出1~100范围；<br>3. n值非4/6/8；<br>4. normOut/sumOut不为空时，其元素总数或格式不符合要求。 |
  | ACLNN_ERR_RUNTIME_ERROR | 361001 | 调用NPU Runtime接口申请内存/创建Tensor失败。 |

## aclnnMhcSinkhorn

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  |:--- |:--- |:--- |
  | workspace | 输入 | Device侧申请的workspace内存地址，需与第一段接口返回的workspaceSize匹配。 |
  | workspaceSize | 输入 | Device侧workspace内存大小，由`aclnnMhcSinkhornGetWorkspaceSize`接口返回。 |
  | executor | 输入 | 算子执行器，由第一段接口创建，包含计算流程和参数信息。 |
  | stream | 输入 | 指定执行计算任务的Stream，需提前创建并绑定Device。 |

- **返回值**

  返回`aclnnStatus`状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算

  - aclnnMhcSinkhorn默认采用确定性实现，相同输入多次调用结果一致。

- 公共约束

  1. 输入约束：
     - 输入Tensor `x`为空，报错`ACLNN_ERR_PARAM_NULLPTR`；
     - 所有输入/输出Tensor的数据格式仅支持`ACL_FORMAT_ND`；
     - 仅支持`FLOAT32`数据类型，不支持其他精度（如FLOAT16/DOUBLE）。
     - normOut和sumOut为可选参数，可传空指针；传空指针时不输出对应结果。不为空时，normOut的元素总数须为`2*numIters*T*n*8`，sumOut的元素总数须为`2*numIters*T*8`（其中`T`为输入合并后的batch数：3维时`T=x.shape[0]`，4维时`T=x.shape[0]*x.shape[1]`；`n`为矩阵维度，取值4/6/8；`8`为n按8对齐后的值），且格式须为`ACL_FORMAT_ND`，否则报错`ACLNN_ERR_PARAM_INVALID`。
     - 输入-inf/inf/nan/，输出nan/nan/nan。
  2. 内存约束：
     - Workspace内存需在Device侧申请，且大小需严格匹配第一段接口返回值；

- 规格约束

  | 规格项   | 规格    | 规格说明                                              |
  | :------- | :------ | :---------------------------------------------------- |
  | numIters | 1~100   | 迭代次数超出该范围会返回参数无效错误。                |
  | n        | 4、6、8 | 输入Tensor最后两维的大小（矩阵维度）仅支持这三个值。  |
  | 维度数   | 3/4     | 输入x支持3维(T,n,n)或4维(B,S,n,n)，其他维度数不支持。 |

## 调用示例

以下为C++调用示例，需结合AscendCL环境编译运行，具体流程参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_sinkhorn.h"

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

// 将Device侧Tensor数据拷贝到Host侧并打印
void PrintTensorData(const std::vector<int64_t>& shape, void* device_addr) {
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

  // 打印前10个元素（示例）
  LOG_PRINT("Tensor data (first 10 elements): ");
  for (int i = 0; i < std::min((int64_t)10, size); ++i) {
    LOG_PRINT("%f ", host_data[i]);
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

int main() {
  // ========== 1. 初始化环境 ==========
  int32_t device_id = 0;  // 根据实际Device ID调整
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;

  int ret = InitAcl(device_id, context, stream);
  CHECK_RET(ret == 0,
            LOG_PRINT("InitAcl failed, error: %d\n", ret);
            return -1);

  // ========== 2. 构造输入/输出参数 ==========
  // 输入x的形状：B=1, S=1024, n=4 → (1024,4,4)（合并B*S为T=1024）
  std::vector<int64_t> x_shape = {1024, 4, 4};
  int64_t x_size = GetShapeSize(x_shape);
  std::vector<float> x_host_data(x_size, 1.0f);  // 初始化输入数据为1.0

  // 输出output的形状与x一致
  std::vector<int64_t> output_shape = x_shape;
  void* output_device_addr = nullptr;
  aclTensor* output_tensor = nullptr;

  // 可选输出：norm_out（out_flag=1时有效）
  std::vector<int64_t> norm_out_shape = {40, 4, 8, 1024};  // 2*num_iters=40, n=4, n_align=8, T=1024
  void* norm_out_device_addr = nullptr;
  aclTensor* norm_out_tensor = nullptr;

  // 可选输出：sum_out（out_flag=1时有效）
  std::vector<int64_t> sum_out_shape = {40, 8, 1024};  // 2*num_iters=40, n_align=8, T=1024
  void* sum_out_device_addr = nullptr;
  aclTensor* sum_out_tensor = nullptr;

  // 输入x的Device Tensor
  void* x_device_addr = nullptr;
  aclTensor* x_tensor = nullptr;
  ret = CreateAclTensor(x_host_data, x_shape, x_device_addr, x_tensor);
  CHECK_RET(ret == 0,
            LOG_PRINT("Create x_tensor failed\n");
            return -1);

  // 输出output的Device Tensor（仅申请内存，无初始数据）
  ret = aclrtMalloc(&output_device_addr, GetShapeSize(output_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("Malloc output failed, error: %d\n", ret);
            return -1);
  output_tensor = aclCreateTensor(
      output_shape.data(), output_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      output_shape.data(), output_shape.size(),
      output_device_addr
  );

  // 可选输出norm_out/sum_out的Tensor（out_flag=1）
  ret = aclrtMalloc(&norm_out_device_addr, GetShapeSize(norm_out_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("Malloc norm_out failed, error: %d\n", ret);
            return -1);
  norm_out_tensor = aclCreateTensor(
      norm_out_shape.data(), norm_out_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      norm_out_shape.data(), norm_out_shape.size(),
      norm_out_device_addr
  );

  ret = aclrtMalloc(&sum_out_device_addr, GetShapeSize(sum_out_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("Malloc sum_out failed, error: %d\n", ret);
            return -1);
  sum_out_tensor = aclCreateTensor(
      sum_out_shape.data(), sum_out_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      sum_out_shape.data(), sum_out_shape.size(),
      sum_out_device_addr
  );

  // MhcSinkhorn算子参数
  float eps = 1e-6f;       // 防除零参数
  int64_t num_iters = 20;  // 迭代次数

  // ========== 3. 调用第一段接口：获取Workspace大小 ==========
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  aclnnStatus aclnn_ret = aclnnMhcSinkhornGetWorkspaceSize(
      x_tensor,
      eps,
      num_iters,
      output_tensor,
      norm_out_tensor,
      sum_out_tensor,
      &workspace_size,
      &executor
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMhcSinkhornGetWorkspaceSize failed, error: %d\n", aclnn_ret);
            return -1);

  // ========== 4. 申请Workspace内存 ==========
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret);
              return -1);
  }

  // ========== 5. 调用第二段接口：执行MhcSinkhorn计算 ==========
  aclnn_ret = aclnnMhcSinkhorn(
      workspace_addr,
      workspace_size,
      executor,
      stream
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMhcSinkhorn failed, error: %d\n", aclnn_ret);
            return -1);

  // ========== 6. 同步Stream并打印结果 ==========
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret);
            return -1);

  LOG_PRINT("MhcSinkhorn compute success!\n");
  LOG_PRINT("Output tensor data: ");
  PrintTensorData(output_shape, output_device_addr);

  // ========== 7. 释放资源 ==========
  // 销毁Tensor
  aclDestroyTensor(x_tensor);
  aclDestroyTensor(output_tensor);
  aclDestroyTensor(norm_out_tensor);
  aclDestroyTensor(sum_out_tensor);

  // 释放Device内存
  aclrtFree(x_device_addr);
  aclrtFree(output_device_addr);
  aclrtFree(norm_out_device_addr);
  aclrtFree(sum_out_device_addr);
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
