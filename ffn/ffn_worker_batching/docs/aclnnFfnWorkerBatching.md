# aclnnFfnWorkerBatching

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/ffn/ffn_worker_batching)

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

- 接口功能：Attention与FFN分离部署场景下，FFN worker侧的token重排算子。Attention将token按专家路由发送到对应FFN worker的预分配数据区，本接口从该数据区中扫描调度信息，按专家维度聚合并重排token，产出各专家对应的连续token数据块。

  **该算子不建议单独使用，建议与FfnWorkerScheduler等算子配合使用，形成完整的工作流。**

    1. 接收Attention侧发送的数据。该数据以ScheduleContext结构体内存排布方式存储。其具体定义参见[调用示例](#调用示例)。该结构体包含CommonArea、ControlArea、AttentionArea、FfnArea域。本接口从FfnArea中读取token_info_buf和token_data_buf获取待重排的token数据与描述信息，并获取layer_id、session_id、micro_batch_id、expert_ids等路由信息。

    2. 对`expert_ids_in`中所有token的专家ID进行排序（被mask的token初始化为大值），生成gather索引。

    3. 多核并行按gather索引从`token_data`中提取token的hidden states和dynamic scale，同时查表得到对应的session_id、micro_batch_id、token_id。

    4. 单核扫描排序后的专家ID序列，查找跳变点，生成`groupList`（每个专家处理的token起止偏移）。

    其中 $Y = A \times BS \times (K+1)$，$A$ 为Attention worker数量，$BS$ 为micro batch size，$K+1$ 为topK加共享专家数。

- 计算公式：

  $$
  \text{gather\_idx}, \_ = \text{sort}(\text{expert\_ids\_in})
  $$

  $$
  y[i] = \text{token\_data}[\text{gather\_idx}[i]]
  $$

  $$
  \text{groupList}[e] = [\text{expert\_id}_e, \text{expert\_token\_num}_e]
  $$

  $$
  \text{actualTokenNum} = \sum_{e} \text{expert\_token\_num}_e
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnFfnWorkerBatchingGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFfnWorkerBatching”接口执行计算。

```Cpp
aclnnStatus aclnnFfnWorkerBatchingGetWorkspaceSize(
    const aclTensor   *scheduleContext,
    int64_t            expertNum,
    const aclIntArray *maxOutShape,
    int64_t            tokenDtype,
    int64_t            needSchedule,
    int64_t            layerNum,
    const aclTensor   *y,
    const aclTensor   *groupList,
    const aclTensor   *sessionIds,
    const aclTensor   *microBatchIds,
    const aclTensor   *tokenIds,
    const aclTensor   *expertOffsets,
    const aclTensor   *dynamicScale,
    const aclTensor   *actualTokenNum,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnFfnWorkerBatching(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream)
```

## aclnnFfnWorkerBatchingGetWorkspaceSize

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
      <td>FFN侧接收的调度上下文，内含CommonArea、ControlArea、AttentionArea、FfnArea。算子从FfnArea中读取token_info_buf和token_data_buf获取待重排的token数据与描述信息。</td>
      <td>不支持空tensor。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>输入</td>
      <td>本卡专家总数，等于每层本卡专家数 × layerNum。用于推导groupList输出大小。</td>
      <td>取值范围为(0, 8192]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxOutShape</td>
      <td>输入</td>
      <td>输出shape上限，格式为 {A, BS, topK+1, H}。用于推导y输出的shape上限 Y = A × BS × (topK+1)，以及H值。</td>
      <td>数组长度必须为4。其中A取值范围为(0, 1024]，BS大于0，topK+1取值范围为(0, 64]，H大于0。</td>
      <td>LIST_INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tokenDtype</td>
      <td>输入</td>
      <td>输入token的数据类型。0表示FP16；1表示BF16；2表示INT8动态量化（INT8数据与FP32 dynamic scale连续排布）。取值为2时需输出dynamic_scale。</td>
      <td>取值为0、1或2。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>needSchedule</td>
      <td>输入</td>
      <td>调度模式。0表示仅做batching不扫描数据；1表示先扫描数据再做batching。</td>
      <td>取值为0或1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layerNum</td>
      <td>输入</td>
      <td>层数，每层专家独立索引。</td>
      <td>取值范围为[0, expertNum]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>重排后的token hidden states，按专家ID排序后连续存放。</td>
      <td>-</td>
      <td>FP16、BF16、INT8</td>
      <td>ND</td>
      <td>2维，(Y, H)，其中Y = A × BS × (topK+1)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>输出</td>
      <td>每个专家处理的token范围。每行格式为 [expert_id, expert_token_num]，未使用的专家填 [0, 0]。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>2维，(expertNum, 2)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sessionIds</td>
      <td>输出</td>
      <td>每个输出token对应的Attention session ID。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，(Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>microBatchIds</td>
      <td>输出</td>
      <td>每个输出token对应的micro batch ID。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，(Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>tokenIds</td>
      <td>输出</td>
      <td>每个输出token在原始输入中的位置索引。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，(Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertOffsets</td>
      <td>输出</td>
      <td>每个输出token在其所属专家分组内的偏移。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，(Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dynamicScale</td>
      <td>输出</td>
      <td>动态量化的scale值，仅在tokenDtype=2时有效。tokenDtype为0或1时为空tensor。</td>
      <td>-</td>
      <td>FP32</td>
      <td>ND</td>
      <td>1维，(Y)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>actualTokenNum</td>
      <td>输出</td>
      <td>所有专家有效token数之和。</td>
      <td>-</td>
      <td>INT64</td>
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

## aclnnFfnWorkerBatching

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFfnWorkerBatchingGetWorkspaceSize获取。</td>
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

- 参数A（Attention worker数量）支持 ≤ 1024。
- 参数M（micro batch数量）支持 ≤ 64。
- 参数K+1（topK加共享专家数）支持 ≤ 64。
- 参数BS（micro batch size）和Y支持泛化，无硬上限（受内存限制）。
- 参数H（hidden size）支持泛化。
- 确定性计算：
  - aclnnFfnWorkerBatching默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_worker_batching.h"

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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
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

    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensorNoData(const std::vector<int64_t> &shape, void **deviceAddr, aclDataType dataType,
                          aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(int8_t);
    if (dataType == ACL_INT32) {
        size = GetShapeSize(shape) * sizeof(int32_t);
    }
    if (dataType == ACL_INT64) {
        size = GetShapeSize(shape) * sizeof(int64_t);
    }
    if (dataType == ACL_FLOAT) {
        size = GetShapeSize(shape) * sizeof(float);
    }
    if (dataType == ACL_FLOAT16) {
        size = GetShapeSize(shape) * sizeof(int16_t);
    }
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

constexpr uint64_t kBufAlignSize = 512;

inline uint64_t AlignUp(uint64_t num, uint64_t align)
{
    return ((num + align - 1) / align) * align;
}

#pragma pack(push, 1)
struct FfnDataDesc {
    volatile int32_t flag;
    volatile int32_t layer_id;
    volatile int32_t expert_ids[0];
};

struct ScheduleContext {
    struct CommonArea {
        uint32_t session_num; // Number of attention nodes
        uint32_t micro_batch_num;
        uint32_t micro_batch_size;
        uint32_t selected_expert_num; // topK + 1
        uint32_t expert_num;          // experts per layer
        uint32_t attn_to_ffn_token_size;
        uint32_t ffn_to_attn_token_size;
        int32_t schedule_mode; // 0: Ffn only 1: Attention only
        int8_t reserve0[96];
    };
    struct ControlArea {
        int32_t run_flag; // 0 : exited  1 : running
        int8_t reserve2[124];
    };
    struct FfnArea {
        uint64_t token_info_buf; // Points to device memory.
        uint64_t token_info_buf_size;
        uint64_t token_data_buf; // Points to device memory.
        uint64_t token_data_buf_size;
        uint64_t polling_index;
        int8_t reserve3[88];
        uint64_t layer_ids_buf;
        uint64_t layer_ids_buf_size;
        uint64_t session_ids_buf;
        uint64_t session_ids_buf_size;
        uint64_t micro_batch_ids_buf;
        uint64_t micro_batch_ids_buf_size;
        uint64_t expert_ids_buf;
        uint64_t expert_ids_buf_size;
        uint32_t out_num;
        int8_t reserve4[60];
    };
    struct AttentionArea {
        uint64_t token_info_buf;
        uint64_t token_info_buf_size;
        uint64_t token_data_buf;
        uint64_t token_data_buf_size;
        uint32_t micro_batch_id;
        int8_t reserve5[92];
    };
    CommonArea common;
    ControlArea control;
    AttentionArea attention;
    FfnArea ffn;
    int8_t reserve6[384]; // Padding to 1024 bytes.
};
static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1024 bytes");
#pragma pack(pop)

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    ScheduleContext scheduleContext = {};
    scheduleContext.common.session_num = 1;
    scheduleContext.common.micro_batch_num = 1;
    scheduleContext.common.micro_batch_size = 8;
    scheduleContext.common.selected_expert_num = 9; // topK + 1
    scheduleContext.common.expert_num = 8;
    scheduleContext.common.attn_to_ffn_token_size = 512;
    scheduleContext.common.ffn_to_attn_token_size = 512;
    scheduleContext.common.schedule_mode = 0; // Ffn only
    scheduleContext.control.run_flag = 1;     // running
    scheduleContext.ffn.polling_index = 0;

    // 属性参数
    int64_t expertNum = 8;                                  // 每层专家数 × layer_num
    int64_t A = scheduleContext.common.session_num;         // Attention worker数量
    int64_t BS = scheduleContext.common.micro_batch_size;   // micro batch size
    int64_t K = scheduleContext.common.selected_expert_num; // topK + 1
    int64_t H = 4096;                                       // hidden size
    int64_t Y = A * BS * K;
    std::vector<int64_t> maxOutShapeValue = {A, BS, K, H};
    int64_t tokenDtype = 0;   // FP16
    int64_t needSchedule = 0; // 仅做batching
    int64_t layerNum = 1;

    // 初始化Ffn token_info_buf
    uint64_t flagAndLayerIdSize = sizeof(FfnDataDesc);
    uint64_t tokenInfoSize = (sizeof(int32_t) * static_cast<uint64_t>(K) * BS + flagAndLayerIdSize) *
                             static_cast<uint64_t>(scheduleContext.common.micro_batch_num) *
                             static_cast<uint64_t>(scheduleContext.common.session_num);
    scheduleContext.ffn.token_info_buf_size = tokenInfoSize;
    void *tokenInfoBuf = nullptr;
    ret = aclrtMalloc(&tokenInfoBuf, tokenInfoSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token info buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.token_info_buf = reinterpret_cast<uint64_t>(tokenInfoBuf);

    // 填充token_info_buf：flag=1，layer_id，expert_ids
    std::vector<int32_t> hostTokenInfo(tokenInfoSize / sizeof(int32_t), 0);
    int32_t *pInt = hostTokenInfo.data();
    for (uint32_t s = 0; s < scheduleContext.common.session_num; ++s) {
        for (uint32_t m = 0; m < scheduleContext.common.micro_batch_num; ++m) {
            *pInt++ = 1; // flag
            *pInt++ = 0; // layer_id
            for (uint32_t i = 0; i < BS * K; ++i) {
                *pInt++ = static_cast<int32_t>(i % scheduleContext.common.expert_num); // expert_id
            }
        }
    }
    ret = aclrtMemcpy(tokenInfoBuf, tokenInfoSize, hostTokenInfo.data(), tokenInfoSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token info buf failed. ERROR: %d\n", ret); return ret);

    // 初始化Ffn token_data_buf
    uint64_t tokenDataSize = static_cast<uint64_t>(Y) * H * sizeof(int16_t);
    scheduleContext.ffn.token_data_buf_size = tokenDataSize;
    void *tokenDataBuf = nullptr;
    ret = aclrtMalloc(&tokenDataBuf, tokenDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token data buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.token_data_buf = reinterpret_cast<uint64_t>(tokenDataBuf);
    std::vector<int16_t> hostTokenData(Y * H, 1);
    ret = aclrtMemcpy(tokenDataBuf, tokenDataSize, hostTokenData.data(), tokenDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token data buf failed. ERROR: %d\n", ret); return ret);

    // 创建scheduleContext aclTensor
    std::vector<int64_t> scheduleContextShape = {1024};
    void *scheduleContextDeviceAddr = nullptr;
    aclTensor *scheduleContextRef = nullptr;
    // ==== FfnArea 中的扁平 id 缓冲 ====
    // 这四个字段是二级指针:context 里存的是 device 地址,由 kernel 自行解引用。框架看不到内层缓冲,
    // 不填也能通过 tiling 校验,只在 kernel 访存时暴露为 errcode:(95) MTE 访问 DDR 越界。
    // NORM 路径(needSchedule=0)两代实现都会读它们,必须提供。
    std::vector<int32_t> hostExpertIds(Y);
    for (int64_t i = 0; i < Y; ++i) {
        hostExpertIds[i] = static_cast<int32_t>(i % expertNum);
    }
    void *expertIdsBuf = nullptr;
    ret = aclrtMalloc(&expertIdsBuf, hostExpertIds.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc expert ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(expertIdsBuf, hostExpertIds.size() * sizeof(int32_t), hostExpertIds.data(),
                      hostExpertIds.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy expert ids buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.expert_ids_buf = reinterpret_cast<uint64_t>(expertIdsBuf);
    scheduleContext.ffn.expert_ids_buf_size = hostExpertIds.size() * sizeof(int32_t);

    std::vector<int32_t> hostSessionIds(A, 0);
    std::vector<int32_t> hostMicroBatchIds(A, 0);
    std::vector<int32_t> hostLayerIds(A, 0);
    const uint64_t idsBufSize = static_cast<uint64_t>(A) * sizeof(int32_t);
    void *sessionIdsBuf = nullptr;
    void *microBatchIdsBuf = nullptr;
    void *layerIdsBuf = nullptr;
    ret = aclrtMalloc(&sessionIdsBuf, idsBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc session ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&microBatchIdsBuf, idsBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc micro batch ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&layerIdsBuf, idsBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc layer ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(sessionIdsBuf, idsBufSize, hostSessionIds.data(), idsBufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy session ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(microBatchIdsBuf, idsBufSize, hostMicroBatchIds.data(), idsBufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy micro batch ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(layerIdsBuf, idsBufSize, hostLayerIds.data(), idsBufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy layer ids buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.session_ids_buf = reinterpret_cast<uint64_t>(sessionIdsBuf);
    scheduleContext.ffn.session_ids_buf_size = idsBufSize;
    scheduleContext.ffn.micro_batch_ids_buf = reinterpret_cast<uint64_t>(microBatchIdsBuf);
    scheduleContext.ffn.micro_batch_ids_buf_size = idsBufSize;
    scheduleContext.ffn.layer_ids_buf = reinterpret_cast<uint64_t>(layerIdsBuf);
    scheduleContext.ffn.layer_ids_buf_size = idsBufSize;

    std::vector<int8_t> hostCtxData(1024, 0);
    std::memcpy(hostCtxData.data(), &scheduleContext, sizeof(ScheduleContext));
    ret = CreateAclTensor(hostCtxData, scheduleContextShape, &scheduleContextDeviceAddr, aclDataType::ACL_INT8,
                          &scheduleContextRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 aclTensor
    std::vector<int64_t> yShape = {Y, H};
    void *yDeviceAddr = nullptr;
    aclTensor *yRef = nullptr;
    ret = CreateAclTensorNoData(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &yRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> groupListShape = {expertNum, 2};
    void *groupListDeviceAddr = nullptr;
    aclTensor *groupListRef = nullptr;
    ret = CreateAclTensorNoData(groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupListRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> oneDimShape = {Y};
    void *sessionIdsAddr = nullptr;
    aclTensor *sessionIdsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &sessionIdsAddr, aclDataType::ACL_INT32, &sessionIdsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *microBatchIdsAddr = nullptr;
    aclTensor *microBatchIdsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &microBatchIdsAddr, aclDataType::ACL_INT32, &microBatchIdsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *tokenIdsAddr = nullptr;
    aclTensor *tokenIdsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &tokenIdsAddr, aclDataType::ACL_INT32, &tokenIdsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *expertOffsetsAddr = nullptr;
    aclTensor *expertOffsetsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &expertOffsetsAddr, aclDataType::ACL_INT32, &expertOffsetsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *dynamicScaleAddr = nullptr;
    aclTensor *dynamicScaleRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &dynamicScaleAddr, aclDataType::ACL_FLOAT, &dynamicScaleRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualTokenNumShape = {1};
    void *actualTokenNumAddr = nullptr;
    aclTensor *actualTokenNumRef = nullptr;
    ret = CreateAclTensorNoData(actualTokenNumShape, &actualTokenNumAddr, aclDataType::ACL_INT64, &actualTokenNumRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建maxOutShape aclIntArray
    aclIntArray *maxOutShapeArray = aclCreateIntArray(maxOutShapeValue.data(), maxOutShapeValue.size());

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnFfnWorkerBatchingGetWorkspaceSize(scheduleContextRef, expertNum, maxOutShapeArray, tokenDtype,
                                                 needSchedule, layerNum, yRef, groupListRef, sessionIdsRef,
                                                 microBatchIdsRef, tokenIdsRef, expertOffsetsRef, dynamicScaleRef,
                                                 actualTokenNumRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFfnWorkerBatchingGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnFfnWorkerBatching(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFfnWorkerBatching failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出结果，将device侧内存上的结果拷贝至host侧
    int64_t actualTokenNum = 0;
    ret = aclrtMemcpy(&actualTokenNum, sizeof(int64_t), actualTokenNumAddr, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy actual_token_num failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("actual_token_num = %ld.\n", actualTokenNum);

    std::vector<int64_t> groupListHost(expertNum * 2, 0);
    ret = aclrtMemcpy(groupListHost.data(), expertNum * 2 * sizeof(int64_t), groupListDeviceAddr,
                      expertNum * 2 * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy group_list failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertNum; i++) {
        LOG_PRINT("group_list[%ld] = [expert_id=%ld, token_num=%ld]\n", i, groupListHost[i * 2],
                  groupListHost[i * 2 + 1]);
    }

    // 6. 释放aclTensor与aclIntArray
    aclDestroyTensor(scheduleContextRef);
    aclDestroyTensor(yRef);
    aclDestroyTensor(groupListRef);
    aclDestroyTensor(sessionIdsRef);
    aclDestroyTensor(microBatchIdsRef);
    aclDestroyTensor(tokenIdsRef);
    aclDestroyTensor(expertOffsetsRef);
    aclDestroyTensor(dynamicScaleRef);
    aclDestroyTensor(actualTokenNumRef);
    aclDestroyIntArray(maxOutShapeArray);

    // 7. 释放device资源
    aclrtFree(scheduleContextDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(sessionIdsAddr);
    aclrtFree(microBatchIdsAddr);
    aclrtFree(tokenIdsAddr);
    aclrtFree(expertOffsetsAddr);
    aclrtFree(dynamicScaleAddr);
    aclrtFree(actualTokenNumAddr);
    aclrtFree(tokenInfoBuf);
    aclrtFree(tokenDataBuf);
    aclrtFree(expertIdsBuf);
    aclrtFree(sessionIdsBuf);
    aclrtFree(microBatchIdsBuf);
    aclrtFree(layerIdsBuf);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    Finalize(deviceId, stream);
    return 0;
}
```
