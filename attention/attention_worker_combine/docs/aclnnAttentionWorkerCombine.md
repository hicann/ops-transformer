# aclnnAttentionWorkerCombine

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/attention_worker_combine)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
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

- 接口功能：Attention与FFN分离部署场景下，Attention侧的数据融合算子。该算子将多个计算单元处理的注意力token数据进行融合，结合专家权重对结果进行加权，输出最终的注意力融合结果，并更新层ID。

  **该算子不建议单独使用，建议与AttentionWorkerScheduler等算子配合使用，形成完整的工作流。**

    1. 接收FFN侧回传的数据。该数据以ScheduleContext结构体内存排布方式存储。其具体定义参见[调用示例](#调用示例)。该结构体包含CommonArea、ControlArea、AttentionArea、FfnArea域。本接口从AttentionArea的token_data_buf中读取token数据。

    2. 结合expertScales对token数据进行加权融合，输出最终注意力结果y。

    3. 根据输入layerId计算并输出nextLayerId，指示下一个要处理的层ID。

- 计算公式：

  $$
  y[i] = \sum_{k=0}^{K-1} \text{expertScales}[i][k] \times \text{token\_data}[i][k]
  $$

  $$
  \text{nextLayerId} = \text{layerId} + 1
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnAttentionWorkerCombineGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAttentionWorkerCombine”接口执行计算。

```Cpp
aclnnStatus aclnnAttentionWorkerCombineGetWorkspaceSize(
    const aclTensor *scheduleContext,
    const aclTensor *expertScales,
    const aclTensor *layerId,
    int64_t          hiddenSize,
    int64_t          tokenDtype,
    int64_t          needSchedule,
    const aclTensor *y,
    const aclTensor *nextLayerId,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAttentionWorkerCombine(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream)
```

## aclnnAttentionWorkerCombineGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1540px"><colgroup>
  <col style="width: 210px">
  <col style="width: 135px">
  <col style="width: 360px">
  <col style="width: 280px">
  <col style="width: 130px">
  <col style="width: 110px">
  <col style="width: 170px">
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
      <td>scheduleContext</td>
      <td>输入</td>
      <td>Attention侧接收的调度上下文，内含CommonArea、ControlArea、AttentionArea、FfnArea。算子从AttentionArea的token_data_buf中读取token数据。</td>
      <td>不支持空tensor。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertScales</td>
      <td>输入</td>
      <td>专家权重，表示每个token对应的各专家权重。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2维，(BS, K)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>layerId</td>
      <td>输入</td>
      <td>当前的模型层ID。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，(1)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>hiddenSize</td>
      <td>输入</td>
      <td>token_data的隐藏维度大小，用于确定输出y的第二维大小。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tokenDtype</td>
      <td>输入</td>
      <td>指定scheduleContext中token数据的原始精度类型。0表示FLOAT16；1表示BFLOAT16。</td>
      <td>取值为0或1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>needSchedule</td>
      <td>输入</td>
      <td>指定是否等待token数据填充完成后再执行。0表示不等待；1表示等待。</td>
      <td>取值为0或1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>最终的注意力合并结果。</td>
      <td>-</td>
      <td>FP16、BF16</td>
      <td>ND</td>
      <td>2维，(BS, hiddenSize)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>nextLayerId</td>
      <td>输出</td>
      <td>下一个要处理的层ID。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，(1)</td>
      <td>×</td>
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
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
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
                <td>输入是空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入数据类型不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnAttentionWorkerCombine

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAttentionWorkerCombineGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- expertScales的第二维K ≤ 64。
- 确定性计算：
  - aclnnAttentionWorkerCombine默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_worker_combine.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream) {
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int CreateAclTensorNoData(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                          aclTensor** tensor) {
  uint64_t elemSize = sizeof(int8_t);
  if (dataType == ACL_INT32) { elemSize = sizeof(int32_t); }
  if (dataType == ACL_FLOAT16) { elemSize = sizeof(int16_t); }
  if (dataType == ACL_BF16) { elemSize = sizeof(int16_t); }
  auto size = GetShapeSize(shape) * elemSize;
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

#pragma pack(push, 1)
struct AttentionDataDesc {
  int32_t flag[0];
};

struct ScheduleContext {
  struct CommonArea {
    uint32_t session_num;           // Number of attention nodes
    uint32_t micro_batch_num;
    uint32_t micro_batch_size;
    uint32_t selected_expert_num;   // topK + 1
    uint32_t expert_num;            // experts per layer
    uint32_t attn_to_ffn_token_size;
    uint32_t ffn_to_attn_token_size;
    int32_t  schedule_mode;         // 0: Ffn only 1: Attention only
    int8_t   reserve0[96];
  };
  struct ControlArea {
    int32_t run_flag;               // 0 : exited  1 : running
    int8_t reserve2[124];
  };
  struct AttentionArea {
    uint64_t token_info_buf;        // Points to device memory.
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;        // Points to device memory.
    uint64_t token_data_buf_size;
    uint32_t micro_batch_id;        // Records the latest ready micro batch id.
    int8_t   reserve5[92];
  };
  struct FfnArea {
    uint64_t token_info_buf;
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;
    uint64_t token_data_buf_size;
    uint64_t polling_index;
    int8_t   reserve3[88];
    uint64_t layer_ids_buf;
    uint64_t layer_ids_buf_size;
    uint64_t session_ids_buf;
    uint64_t session_ids_buf_size;
    uint64_t micro_batch_ids_buf;
    uint64_t micro_batch_ids_buf_size;
    uint64_t expert_ids_buf;
    uint64_t expert_ids_buf_size;
    uint32_t out_num;
    int8_t   reserve4[60];
  };
  CommonArea    common;
  ControlArea   control;
  AttentionArea attention;
  FfnArea       ffn;
  int8_t        reserve6[384];      // Padding to 1024 bytes.
};
static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1024 bytes");
#pragma pack(pop)

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int64_t BS = 48;       // batch size
  int64_t K = 8;         // expert num per token
  int64_t hiddenSize = 20480;
  int64_t tokenDtype = 1;    // BF16
  int64_t needSchedule = 0;

  // 初始化ScheduleContext
  ScheduleContext scheduleContext = {};
  scheduleContext.common.session_num = 1;
  scheduleContext.common.micro_batch_num = 1;
  scheduleContext.common.micro_batch_size = BS;
  scheduleContext.common.selected_expert_num = K;
  scheduleContext.common.expert_num = 16;
  scheduleContext.common.attn_to_ffn_token_size = 512;
  scheduleContext.common.ffn_to_attn_token_size = 512;
  scheduleContext.common.schedule_mode = 1;   // Attention only
  scheduleContext.control.run_flag = 1;       // running
  scheduleContext.attention.micro_batch_id = 0;

  // 初始化Attention token_info_buf（flag置1表示数据就绪）
  size_t perDataDescSize = sizeof(AttentionDataDesc) + sizeof(int32_t) * BS * K;
  size_t tokenInfoBufSize = static_cast<size_t>(scheduleContext.common.micro_batch_num) * perDataDescSize;
  void* tokenInfoBuf = nullptr;
  ret = aclrtMalloc(&tokenInfoBuf, tokenInfoBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token info buf failed. ERROR: %d\n", ret); return ret);
  scheduleContext.attention.token_info_buf = reinterpret_cast<uint64_t>(tokenInfoBuf);
  scheduleContext.attention.token_info_buf_size = tokenInfoBufSize;
  std::vector<int32_t> hostFlags(static_cast<size_t>(BS) * K, 1);
  ret = aclrtMemcpy(tokenInfoBuf, tokenInfoBufSize, hostFlags.data(),
                    static_cast<size_t>(BS) * K * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token info buf failed. ERROR: %d\n", ret); return ret);

  // 初始化Attention token_data_buf
  uint64_t tokenDataSize = static_cast<uint64_t>(BS) * K * hiddenSize * sizeof(int16_t);
  void* tokenDataBuf = nullptr;
  ret = aclrtMalloc(&tokenDataBuf, tokenDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token data buf failed. ERROR: %d\n", ret); return ret);
  scheduleContext.attention.token_data_buf = reinterpret_cast<uint64_t>(tokenDataBuf);
  scheduleContext.attention.token_data_buf_size = tokenDataSize;
  std::vector<int16_t> hostTokenData(static_cast<size_t>(BS) * K * hiddenSize, 1);
  ret = aclrtMemcpy(tokenDataBuf, tokenDataSize, hostTokenData.data(), tokenDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token data buf failed. ERROR: %d\n", ret); return ret);

  // 创建scheduleContext aclTensor
  std::vector<int64_t> scheduleContextShape = {1024};
  void* scheduleContextDeviceAddr = nullptr;
  aclTensor* scheduleContextRef = nullptr;
  std::vector<int8_t> hostCtxData(1024, 0);
  std::memcpy(hostCtxData.data(), &scheduleContext, sizeof(ScheduleContext));
  ret = CreateAclTensor(hostCtxData, scheduleContextShape, &scheduleContextDeviceAddr, aclDataType::ACL_INT8,
                        &scheduleContextRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建expert_scales aclTensor, shape (BS, K)
  std::vector<int64_t> expertScalesShape = {BS, K};
  std::vector<float> hostExpertScales(static_cast<size_t>(BS) * K, 0.125f);
  void* expertScalesDeviceAddr = nullptr;
  aclTensor* expertScalesRef = nullptr;
  ret = CreateAclTensor(hostExpertScales, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                        &expertScalesRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建layer_id aclTensor, shape (1)
  std::vector<int64_t> layerIdShape = {1};
  std::vector<int32_t> hostLayerId = {0};
  void* layerIdDeviceAddr = nullptr;
  aclTensor* layerIdRef = nullptr;
  ret = CreateAclTensor(hostLayerId, layerIdShape, &layerIdDeviceAddr, aclDataType::ACL_INT32, &layerIdRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建输出y aclTensor, shape (BS, hiddenSize)
  std::vector<int64_t> yShape = {BS, hiddenSize};
  void* yDeviceAddr = nullptr;
  aclTensor* yRef = nullptr;
  ret = CreateAclTensorNoData(yShape, &yDeviceAddr, aclDataType::ACL_BF16, &yRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建输出next_layer_id aclTensor, shape (1)
  std::vector<int64_t> nextLayerIdShape = {1};
  void* nextLayerIdDeviceAddr = nullptr;
  aclTensor* nextLayerIdRef = nullptr;
  ret = CreateAclTensorNoData(nextLayerIdShape, &nextLayerIdDeviceAddr, aclDataType::ACL_INT32, &nextLayerIdRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnAttentionWorkerCombineGetWorkspaceSize(scheduleContextRef, expertScalesRef, layerIdRef, hiddenSize,
                                                    tokenDtype, needSchedule, yRef, nextLayerIdRef, &workspaceSize,
                                                    &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnAttentionWorkerCombineGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnAttentionWorkerCombine(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionWorkerCombine failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出结果，将device侧内存上的结果拷贝至host侧
  int32_t nextLayerId = 0;
  ret = aclrtMemcpy(&nextLayerId, sizeof(int32_t), nextLayerIdDeviceAddr, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy next_layer_id failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("next_layer_id = %d.\n", nextLayerId);

  // 6. 释放aclTensor
  aclDestroyTensor(scheduleContextRef);
  aclDestroyTensor(expertScalesRef);
  aclDestroyTensor(layerIdRef);
  aclDestroyTensor(yRef);
  aclDestroyTensor(nextLayerIdRef);

  // 7. 释放device资源
  aclrtFree(scheduleContextDeviceAddr);
  aclrtFree(expertScalesDeviceAddr);
  aclrtFree(layerIdDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(nextLayerIdDeviceAddr);
  aclrtFree(tokenInfoBuf);
  aclrtFree(tokenDataBuf);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  Finalize(deviceId, stream);
  return 0;
}
```
