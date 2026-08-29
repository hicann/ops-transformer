# aclnnApplyRotaryPosEmbGrad

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/apply_rotary_pos_emb_grad)

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

- 接口功能：执行双路旋转位置编码[aclnnApplyRotaryPosEmb](../../apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmb.md)的反向计算。同时计算 query 和 key 的 rope 反向梯度，融合为一次 kernel 调用。

- 计算公式：

    取旋转位置编码的正向计算中，broadcast的轴列表为`dims`，即`cos`/`sin`中取值为1、而`gradQueryEmbed`/`gradKeyEmbed`中对应维度大于1的轴（包含N轴，以及BSND、SBND布局下可选的B轴），half模式计算公式如下：

    $$
    grad\_q_1, grad\_q_2 = chunk(grad\_query\_embed, chunks=2, dim=-1)
    $$

    $$
    grad\_k_1, grad\_k_2 = chunk(grad\_key\_embed, chunks=2, dim=-1)
    $$

    $$
    cos_1, cos_2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin_1, sin_2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    query\_rotate = cat((-query_2, query_1), dim=-1)
    $$

    $$
    key\_rotate = cat((-key_2, key_1), dim=-1)
    $$

    $$
    grad\_query = cat(cos_1 * grad\_q_1 + sin_2 * grad\_q_2, cos_2 * grad\_q_2 - sin_1 * grad\_q_1, dim=-1)
    $$

    $$
    grad\_key = cat(cos_1 * grad\_k_1 + sin_2 * grad\_k_2, cos_2 * grad\_k_2 - sin_1 * grad\_k_1, dim=-1)
    $$

    $$
    grad\_cos = sum(grad\_query\_embed * query + grad\_key\_embed * key, dims)
    $$

    $$
    grad\_sin = sum(grad\_query\_embed * query\_rotate + grad\_key\_embed * key\_rotate, dims)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnApplyRotaryPosEmbGradGetWorkspaceSize"接口获取入参并根据流程计算所需workspace大小，再调用"aclnnApplyRotaryPosEmbGrad"接口执行计算。

```c++
aclnnStatus aclnnApplyRotaryPosEmbGradGetWorkspaceSize(
    const aclTensor *gradQueryEmbed,
    const aclTensor *gradKeyEmbed,
    const aclTensor *cos,
    const aclTensor *sin,
    const aclTensor *queryOptional,
    const aclTensor *keyOptional,
    char            *rotaryModeOptional,
    int64_t          layout,
    const aclTensor *gradQueryOut,
    const aclTensor *gradKeyOut,
    const aclTensor *gradCosOut,
    const aclTensor *gradSinOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnApplyRotaryPosEmbGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnApplyRotaryPosEmbGradGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 403px">
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
      <td>gradQueryEmbed</td>
      <td>输入</td>
      <td>正向输出query的导数，对应公式中grad_q_embed。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradKeyEmbed</td>
      <td>输入</td>
      <td>正向输出key的导数，对应公式中grad_k_embed。</td>
      <td>与gradQueryEmbed的数据类型和维度一致，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>正向计算输入cos。</td>
      <td>与gradQueryEmbed的数据类型和维度一致，N维度必须等于1，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>正向计算输入sin。</td>
      <td>与gradQueryEmbed的数据类型和维度一致，N维度必须等于1，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>queryOptional</td>
      <td>可选输入</td>
      <td>正向计算输入query，空指针时不计算gradCosOutOptional和gradSinOutOptional。</td>
      <td>与gradQueryEmbed的数据类型和维度一致，必须与keyOptional同时传入或同时不传入。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>keyOptional</td>
      <td>可选输入</td>
      <td>正向计算输入key，空指针时不计算gradCosOutOptional和gradSinOutOptional。</td>
      <td>与gradQueryEmbed的数据类型和维度一致，必须与queryOptional同时传入或同时不传入。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rotaryModeOptional</td>
      <td>输入</td>
      <td>旋转模式。</td>
      <td>仅支持"half"。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>输入</td>
      <td>输入Tensor的布局格式。</td>
      <td>取值范围：1-BSND，2-SBND，4-TND，3-BNSD（预留）。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradQueryOut</td>
      <td>输出</td>
      <td>正向计算输入query的导数。</td>
      <td>与gradQueryEmbed的数据类型和维度一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradKeyOut</td>
      <td>输出</td>
      <td>正向计算输入key的导数。</td>
      <td>与gradQueryEmbed的数据类型和维度一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradCosOut</td>
      <td>输出</td>
      <td>正向计算输入cos的导数，queryOptional和keyOptional非空时有效。</td>
      <td>与gradQueryEmbed的数据类型和维度一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradSinOut</td>
      <td>输出</td>
      <td>正向计算输入sin的导数，queryOptional和keyOptional非空时有效。</td>
      <td>与gradQueryEmbed的数据类型和维度一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4(layout为1或2)或3(layout为4)</td>
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

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 288px">
  <col style="width: 125px">
  <col style="width: 742px">
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
      <td>传入的必选输入gradQueryEmbed、gradKeyEmbed、cos、sin和必选输出gradQueryOut、gradKeyOut是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>传入的输入输出数据类型和格式不在支持的范围内、shape不满足校验条件、维度不在支持的范围、queryOptional与keyOptional未同时传入或同时不传、或rotaryMode/layout不符合支持的值。</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyRotaryPosEmbGrad

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnApplyRotaryPosEmbGradGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnApplyRotaryPosEmbGrad默认确定性实现。
- 各参数的约束描述如下：
  - 输入输出Tensor只支持3维或4维：layout为1或2时为4维，layout为4时为3维。
  - 输入输出Tensor的dtype必须相同。
  - 输入输出Tensor不支持空Tensor（各维度必须大于0）。
  - 输入输出Tensor的layout必须相同。
  - 输入输出Tensor的D轴必须相同，在half模式下必须≤1024且能被2整除。
  - `gradQueryEmbed`、`gradQueryOut`的shape必须相同，`gradKeyEmbed`、`gradKeyOut`的shape必须相同。
  - 对于任意`layout`，`gradQueryEmbed`和`gradKeyEmbed`除N维度外其它维度必须相同。
  - `cos`、`sin`的N维度必须等于1；`layout`为1（BSND）或2（SBND）时，`cos`、`sin`的B维度可以等于1，也可以和`gradQueryEmbed`的B维度一致；`layout`为4（TND）时，`cos`、`sin`的T维度必须和`gradQueryEmbed`的T维度一致；除N（及BSND、SBND布局下可选广播的B）维度外，其余维度需与`gradQueryEmbed`一致。
  - `cos`、`sin`、`gradCosOutOptional`、`gradSinOutOptional`的shape必须相同。
  - `query`维度需与`gradQueryEmbed`一致，`key`维度需与`gradKeyEmbed`一致，且`query`和`key`必须同时传入或同时不传入。
  - `rotaryModeOptional`仅支持"half"。
  - `layout`仅支持{1, 2, 4}，对应{BSND, SBND, TND}。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb_grad.h"
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
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

int main()
{
    // 1. 固定写法，device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> gradQEmbedShape = {1, 1, 1, 128};
    std::vector<int64_t> gradKEmbedShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    std::vector<int64_t> queryShape = {1, 1, 1, 128};
    std::vector<int64_t> keyShape = {1, 1, 1, 128};
    std::vector<int64_t> gradQueryOutShape = {1, 1, 1, 128};
    std::vector<int64_t> gradKeyOutShape = {1, 1, 1, 128};
    std::vector<int64_t> gradCosOutShape = {1, 1, 1, 128};
    std::vector<int64_t> gradSinOutShape = {1, 1, 1, 128};
    int64_t layout = 1;
    const char* rotaryModeOptional = "half";

    void* gradQEmbedDeviceAddr = nullptr;
    void* gradKEmbedDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* gradQueryOutDeviceAddr = nullptr;
    void* gradKeyOutDeviceAddr = nullptr;
    void* gradCosOutDeviceAddr = nullptr;
    void* gradSinOutDeviceAddr = nullptr;
    aclTensor* gradQueryEmbed = nullptr;
    aclTensor* gradKeyEmbed = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* query = nullptr;
    aclTensor* key = nullptr;
    aclTensor* gradQueryOut = nullptr;
    aclTensor* gradKeyOut = nullptr;
    aclTensor* gradCosOut = nullptr;
    aclTensor* gradSinOut = nullptr;

    std::vector<float> gradQEmbedHostData = {
        74,  54, 84, 125, 23,  78,  37,  72,  27, 98,  34,  107, 29,  23,  54,  60, 70,  49,  119, 54,  29,  54,
        41,  99, 27, 62,  5,   46,  108, 39,  24, 123, 33,  82,  6,   40,  88,  24, 6,   116, 38,  119, 110, 5,
        30,  79, 87, 18,  29,  100, 90,  24,  21, 93,  63,  68,  34,  112, 119, 48, 74,  43,  85,  64,  14,  49,
        128, 59, 18, 37,  123, 76,  14,  63,  10, 39,  107, 124, 79,  16,  17,  76, 80,  47,  90,  41,  58,  82,
        75,  80, 69, 37,  74,  36,  54,  26,  32, 54,  13,  100, 105, 15,  13,  69, 122, 26,  94,  59,  29,  14,
        60,  8,  24, 17,  45,  33,  107, 122, 63, 111, 75,  128, 68,  31,  105, 6,  82,  99};
    std::vector<float> gradKEmbedHostData = {
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
    std::vector<float> queryHostData = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128};
    std::vector<float> keyHostData = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
        89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
        109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128};
    std::vector<float> gradQueryOutHostData(128, 0.0f);
    std::vector<float> gradKeyOutHostData(128, 0.0f);

    // 创建gradQueryEmbed aclTensor
    ret = CreateAclTensor(gradQEmbedHostData, gradQEmbedShape, &gradQEmbedDeviceAddr, aclDataType::ACL_FLOAT, &gradQueryEmbed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradKeyEmbed aclTensor
    ret = CreateAclTensor(gradKEmbedHostData, gradKEmbedShape, &gradKEmbedDeviceAddr, aclDataType::ACL_FLOAT, &gradKeyEmbed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cos aclTensor
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sin aclTensor
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建query aclTensor (用于计算grad_cos/grad_sin)
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建key aclTensor (用于计算grad_cos/grad_sin)
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradQueryOut aclTensor
    ret = CreateAclTensor(gradQueryOutHostData, gradQueryOutShape, &gradQueryOutDeviceAddr, aclDataType::ACL_FLOAT, &gradQueryOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradKeyOut aclTensor
    ret = CreateAclTensor(gradKeyOutHostData, gradKeyOutShape, &gradKeyOutDeviceAddr, aclDataType::ACL_FLOAT, &gradKeyOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradCosOut aclTensor
    ret = CreateAclTensor(std::vector<float>(128, 0.0f), gradCosOutShape, &gradCosOutDeviceAddr, aclDataType::ACL_FLOAT, &gradCosOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradSinOut aclTensor
    ret = CreateAclTensor(std::vector<float>(128, 0.0f), gradSinOutShape, &gradSinOutDeviceAddr, aclDataType::ACL_FLOAT, &gradSinOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnApplyRotaryPosEmbGrad第一段接口
    ret = aclnnApplyRotaryPosEmbGradGetWorkspaceSize(gradQueryEmbed, gradKeyEmbed, cos, sin, query, key,
                                                      const_cast<char*>(rotaryModeOptional), layout,
                                                      gradQueryOut, gradKeyOut, gradCosOut, gradSinOut,
                                                      &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnApplyRotaryPosEmbGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnApplyRotaryPosEmbGrad第二段接口
    ret = aclnnApplyRotaryPosEmbGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyRotaryPosEmbGrad failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(gradQueryOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradQueryOutDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar
    aclDestroyTensor(gradQueryEmbed);
    aclDestroyTensor(gradKeyEmbed);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(gradQueryOut);
    aclDestroyTensor(gradKeyOut);
    aclDestroyTensor(gradCosOut);
    aclDestroyTensor(gradSinOut);

    // 7. 释放device资源
    aclrtFree(gradQEmbedDeviceAddr);
    aclrtFree(gradKEmbedDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(gradQueryOutDeviceAddr);
    aclrtFree(gradKeyOutDeviceAddr);
    aclrtFree(gradCosOutDeviceAddr);
    aclrtFree(gradSinOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}

```
