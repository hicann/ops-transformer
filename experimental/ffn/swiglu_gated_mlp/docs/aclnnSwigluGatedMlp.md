# aclnnSwigluGatedMlp

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| <term>昇腾910_95 AI处理器</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |
| <term>Atlas 200/300/500 推理产品</term> | × |

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：完成融合SwiGLU门控MLP计算。该算子将第一个矩阵乘、SwiGLU激活以及第二个矩阵乘融合为一个两段式aclnn接口。
- 计算公式：

  $$
  gate\_up = x * gate\_up\_weight
  $$

  $$
  hidden = swiglu(gate\_up)
  $$

  $$
  y = hidden * down\_weight
  $$

  其中，`gate_up`会在最后一维均分为两部分，SwiGLU激活计算为：

  $$
  swiglu(gate\_up)=silu(gate) * up
  $$

  `gate`表示`gate_up`前半部分，`up`表示`gate_up`后半部分。

## 实现原理

SwigluGatedMlp主要由两个MatMul和一个SwiGLU激活组成，计算过程分为3步：

1. `gate_up = MatMul(x, gate_up_weight)`，执行第一个矩阵乘。若`x`的shape为`[..., hiddenSize]`，则内部按二维矩阵`[rows, hiddenSize]`参与计算，其中`rows`为`x`除最后一维外所有维度的乘积。
2. `hidden = SwiGLU(gate_up)`，将`gate_up`最后一维按2等分，计算`silu(gate) * up`，输出中间结果`hidden`。
3. `y = MatMul(hidden, down_weight)`，执行第二个矩阵乘，并将结果reshape为用户输出`y`的shape。

如下代码示例给出小算子和SwigluGatedMlp融合算子的对应关系：

```python
# 小算子
gate_up = MatMul(x, gate_up_weight)
gate, up = Split(gate_up, split_num=2, axis=-1)
hidden = Silu(gate) * up
y = MatMul(hidden, down_weight)

# 融合算子
y = SwigluGatedMlp(x, gate_up_weight, down_weight, cube_math_type)
```

## 算子执行接口

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSwigluGatedMlpGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwigluGatedMlp”接口执行计算。

* `aclnnStatus aclnnSwigluGatedMlpGetWorkspaceSize(const aclTensor* x, const aclTensor* gateUpWeight, const aclTensor* downWeight, int64_t cubeMathType, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnSwigluGatedMlp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnSwigluGatedMlpGetWorkspaceSize

- **参数说明：**

  - x（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，公式中的输入`x`。数据类型支持FLOAT16、FLOAT、BFLOAT16，[数据格式](../../../../docs/zh/context/data_format.md)支持ND，shape为`[..., hiddenSize]`，维度个数需大于等于2。
  - gateUpWeight（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，第一个MatMul的权重，公式中的`gate_up_weight`。数据类型需与`x`一致，支持FLOAT16、FLOAT、BFLOAT16，[数据格式](../../../../docs/zh/context/data_format.md)支持ND，shape为`[hiddenSize, 2 * intermediateSize]`。
  - downWeight（aclTensor\*，计算输入）：必选参数，Device侧的aclTensor，第二个MatMul的权重，公式中的`down_weight`。数据类型需与`x`一致，支持FLOAT16、FLOAT、BFLOAT16，[数据格式](../../../../docs/zh/context/data_format.md)支持ND，shape为`[intermediateSize, outSize]`。
  - cubeMathType（int64\_t，计算输入）：Host侧属性值，表示MatMul计算模式。当前支持取值0和1，默认值为1。
  - y（aclTensor\*，计算输出）：必选参数，Device侧的aclTensor，公式中的输出`y`。数据类型需与`x`一致，支持FLOAT16、FLOAT、BFLOAT16，[数据格式](../../../../docs/zh/context/data_format.md)支持ND，shape为`[..., outSize]`。
  - workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

  > **说明：**
  > `hiddenSize`表示输入`x`的最后一维大小；`intermediateSize`表示SwiGLU激活后中间结果的最后一维大小；`outSize`表示输出`y`的最后一维大小。`gateUpWeight`的第二维必须等于`2 * intermediateSize`，`downWeight`的第一维必须等于`intermediateSize`。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，若出现以下错误码，则对应原因为：

  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入、输出或者出参，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：输入或输出的数据类型、shape关系、`cubeMathType`取值不在支持范围内。

### aclnnSwigluGatedMlp

- **参数说明：**

  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnSwigluGatedMlpGetWorkspaceSize获取。
  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnSwigluGatedMlp默认为确定性实现，确定性计算配置不会影响该算子执行结果。
- `x`、`gateUpWeight`、`downWeight`、`y`均不支持空指针。
- `x`、`gateUpWeight`、`downWeight`、`y`的数据类型必须一致，当前支持FLOAT16、FLOAT、BFLOAT16。
- `x`维度个数需大于等于2，`gateUpWeight`和`downWeight`必须为2维。
- `gateUpWeight`第0维必须等于`x`最后一维。
- `gateUpWeight`第1维必须为正数且能被2整除。
- `downWeight`第0维必须等于`gateUpWeight`第1维的一半。
- `y`维度个数必须与`x`一致；`y`除最后一维外的各维度必须与`x`一致；`y`最后一维必须等于`downWeight`第1维。
- `cubeMathType`当前仅支持取值0和1。

## 算子原型

```c++
REG_OP(SwigluGatedMlp)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gate_up_weight, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(down_weight, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(cube_math_type, Int, 1)
    .OP_END_FACTORY_REG(SwigluGatedMlp)
```

参数解释请参见**算子执行接口**。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_gated_mlp.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    std::printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
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
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
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

int main() {
  // 1.（固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {2, 3};
  std::vector<int64_t> gateUpWeightShape = {3, 8};
  std::vector<int64_t> downWeightShape = {4, 2};
  std::vector<int64_t> yShape = {2, 2};

  void* xDeviceAddr = nullptr;
  void* gateUpWeightDeviceAddr = nullptr;
  void* downWeightDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* gateUpWeight = nullptr;
  aclTensor* downWeight = nullptr;
  aclTensor* y = nullptr;

  std::vector<float> xHostData = {
    0.1F, -0.2F, 0.3F,
    0.4F, 0.5F, -0.6F
  };
  std::vector<float> gateUpWeightHostData = {
    0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, 0.5F, 0.6F,
    0.2F, 0.1F, -0.4F, 0.3F, 0.7F, -0.3F, 0.2F, 0.1F,
    -0.5F, 0.4F, 0.2F, -0.1F, 0.3F, 0.8F, -0.2F, 0.5F
  };
  std::vector<float> downWeightHostData = {
    0.5F, -0.1F,
    0.2F, 0.4F,
    -0.3F, 0.6F,
    0.7F, 0.1F
  };
  std::vector<float> yHostData(GetShapeSize(yShape), 0.0F);

  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gateUpWeight aclTensor
  ret = CreateAclTensor(gateUpWeightHostData, gateUpWeightShape, &gateUpWeightDeviceAddr,
                        aclDataType::ACL_FLOAT, &gateUpWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建downWeight aclTensor
  ret = CreateAclTensor(downWeightHostData, downWeightShape, &downWeightDeviceAddr, aclDataType::ACL_FLOAT,
                        &downWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnSwigluGatedMlp接口调用示例
  LOG_PRINT("test aclnnSwigluGatedMlp\n");

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // 调用aclnnSwigluGatedMlp第一段接口
  int64_t cubeMathType = 1;
  ret = aclnnSwigluGatedMlpGetWorkspaceSize(x, gateUpWeight, downWeight, cubeMathType, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSwigluGatedMlpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSwigluGatedMlp第二段接口
  ret = aclnnSwigluGatedMlp(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGatedMlp failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto outputSize = GetShapeSize(yShape);
  std::vector<float> resultData(outputSize, 0.0F);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    outputSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < outputSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(gateUpWeight);
  aclDestroyTensor(downWeight);
  aclDestroyTensor(y);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(gateUpWeightDeviceAddr);
  aclrtFree(downWeightDeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
