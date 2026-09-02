# aclnnChunkGatedDeltaRule

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_gated_delta_rule)

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

- 接口功能：完成chunk版的Gated Delta Rule计算。

- 计算公式：

  Gated Delta Rule（门控Delta规则，GDR）是一种应用于循环神经网络的算子，也被应用于一种线性注意力机制中。在每个时间步 $t$，GDR根据当前的输入 $q_t$、$k_t$、$v_t$、上一个隐藏状态 $S_{t-1}$、衰减系数 $\alpha_t$ 以及更新强度 $\beta_t$，计算当前的注意力输出 $o_t$ 和新的隐藏状态 $S_t$，其计算公式如下：
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := S_t (q_t \cdot scale)
  $$

  其中，$S_{t-1},S_t \in \mathbb{R}^{D_v \times D_k}$，$q_t, k_t \in \mathbb{R}^{D_k}$，$v_t \in \mathbb{R}^{D_v}$，$\alpha_t \in \mathbb{R}$，$\beta_t \in \mathbb{R}$，$o_t \in \mathbb{R}^{D_v}$。

  Chunked Gated Delta Rule是GDR的chunk版实现（[参考论文](https://arxiv.org/abs/2412.06464)），它通过将输入序列切块，实现了一定的并行效果，在长上下文场景其计算效率相对Recurrent Gated Delta Rule更高，适用于prefill阶段。输入一个长度为L的序列，该算子可以计算出每一步的输出 $o_t, t \in \{1, .., L\}$ 以及最终的状态矩阵 $S_L$。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnChunkGatedDeltaRuleGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnChunkGatedDeltaRule”接口执行计算。

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *beta,
    const aclTensor *initialState,
    const aclTensor *actualSeqLengths,
    const aclTensor *gOptional,
    float           scaleValue,
    const aclTensor *out,
    const aclTensor *finalState,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnChunkGatedDeltaRule(
    void          *workspace,
    uint64_t      workspaceSize,
    aclOpExecutor *executor,
    aclrtStream   stream)
```

## aclnnChunkGatedDeltaRuleGetWorkspaceSize

- 参数说明

  <table style="undefined; table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 165px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的q。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的k。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nk, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的v。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv, Dv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>公式中的β。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>initialState</td>
      <td>输入</td>
      <td>初始状态矩阵S_0。</td>
      <td>
      <ul>
          <li>不支持空Tensor。</li>
          <li>支持0轴和1轴非连续。</li>
      </ul>
      </td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(B, Nv, Dv, Dk)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengths</td>
      <td>输入</td>
      <td>不同batch的有效序列长度。</td>
      <td>不支持空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gOptional</td>
      <td>输入</td>
      <td>衰减系数的指数，公式中的α=e^g。</td>
      <td>可选输入。<ul><li>不支持空Tensor。</li><li>如果传入nullptr，则表示全0的tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(T, Nv)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>输入</td>
      <td>query的缩放因子，对应公式中的scale。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的o。</td>
      <td>-</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(T, Nv, Dv)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>finalState</td>
      <td>输出</td>
      <td>最终状态矩阵S_t。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(B, Nv, Dv, Dk)</td>
      <td>x</td>
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
  </tbody>
  </table>

  其中 $B$ 表示batch size，令 $L_i$ 表示第i个序列的长度，则 $T=\sum_i^B L_i$ 表示累积序列长度。$N_k$ 表示key的头数，$N_v$ 表示value的头数，$D_k$ 表示key的隐藏层维度，$D_v$ 表示value的隐藏层维度。

- 返回值

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 670px">
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
      <td>query、key、value、beta、initialState、actualSeqLengths、out、finalState存在空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>传入的query、key、value、beta、initialState、actualSeqLengths、out、finalState的数据类型和格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>传入的query、key、value、beta、initialState、actualSeqLengths、out、finalState的shape不匹配。</td>
    </tr>
  </tbody>
  </table>

## aclnnChunkGatedDeltaRule

- 参数说明

  <table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 670px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace (void*)</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnChunkGatedDeltaRuleGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor*)</td>
      <td>输入</td>
      <td>op执行器，包含算子计算流程。</td>
    </tr>
    <tr>
      <td>stream (aclrtStream)</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- 返回值
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnChunkGatedDeltaRule默认确定性实现。
- 维度约束：
  - $0 \lt Nv \le 64，0 \lt Nk \le 64$，且 $Nv \bmod Nk = 0$
  - $0 \lt Dv \le 128$, $0 \lt Dk \le 128$
  - $B \gt 0$, $T \gt 0$
- 由于算法特性，用户需保障以下数值约束，否则计算结果可能出现溢出：
  - $-1 \le query[i][j][k] \le 1$，且query需要归一化处理
  - $-1 \le key[i][j][k] \le 1$，且key需要归一化处理
  - $-1 \le g[i][j] \le 0$
  - $0 < beta[i][j] < 1$
- initialState、finalState float32数据类型仅在Ascend 950PR/Ascend 950DT支持。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_gated_delta_rule.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr, const char* name)
{
    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        if (i >= 5) { // print the first five data
            break;
        }
        float result = 0.0f;
        uint32_t val = static_cast<uint32_t>(resultData[i]) << 16U;
        std::memcpy(&result, &val, sizeof(result));
        LOG_PRINT("%s result[%ld] is: %f\n", name, i, result);
    }
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    // AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1.device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *gamaDeviceAddr = nullptr;
    void *betaDeviceAddr = nullptr;
    void *initStateDeviceAddr = nullptr;
    void *actSeqLenDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;
    void *finalStateDeviceAddr = nullptr;

    aclTensor *query = nullptr;
    aclTensor *key = nullptr;
    aclTensor *value = nullptr;
    aclTensor *gama = nullptr;
    aclTensor *beta = nullptr;
    aclTensor *initState = nullptr;
    aclTensor *actSeqLen = nullptr;
    aclTensor *attnOut = nullptr;
    aclTensor *finalState = nullptr;

    // 自定义输入与属性
    int32_t batchSize = 2;
    int32_t seqLength = 32;
    int32_t headKNum = 4;
    int32_t headVNum = 8;
    int32_t dimV = 32;
    int32_t dimK = 32;


    std::vector<int64_t> stateShape = {batchSize, headVNum, dimV, dimK};
    std::vector<int64_t> qkShape = {batchSize * seqLength, headKNum, dimK};
    std::vector<int64_t> vShape = {batchSize * seqLength, headVNum, dimV};
    std::vector<int64_t> gamaShape = {batchSize * seqLength, headVNum};
    std::vector<int64_t> actSeqLenShape = {batchSize};
    std::vector<int16_t> initStateHostData(GetShapeSize(stateShape));
    std::vector<int16_t> queryHostData(GetShapeSize(qkShape));
    std::vector<int16_t> keyHostData(GetShapeSize(qkShape));
    std::vector<int16_t> valueHostData(GetShapeSize(vShape));
    std::vector<float> gamaHostData(GetShapeSize(gamaShape));
    std::vector<int16_t> betaHostData(GetShapeSize(gamaShape));
    std::vector<int32_t> actSeqLenHostData(batchSize, seqLength);
    int16_t bfloatHalf = 16128;   // int16_t的16128的二进制对应bfloat16的0.5
    int16_t bfloatQuarter = 15936; // int16_t的15936的二进制对应bfloat16的0.25
    for (int i = 0; i < initStateHostData.size(); i++) {
        initStateHostData[i] = bfloatQuarter;
    }
    for (int i = 0; i < queryHostData.size(); i++) {
        queryHostData[i] = bfloatQuarter;
    }
    for (int i = 0; i < keyHostData.size(); i++) {
        keyHostData[i] = bfloatQuarter;
    }
    for (int i = 0; i < valueHostData.size(); i++) {
        valueHostData[i] = bfloatHalf;
    }
    for (int i = 0; i < betaHostData.size(); i++) {
        betaHostData[i] = bfloatHalf;
    }
    for (int i = 0; i < gamaHostData.size(); i++) {
        gamaHostData[i] = -0.5f;
    }

    std::vector<int16_t> attnOutHostData(valueHostData);
    std::vector<int16_t> finalStateHostData(GetShapeSize(stateShape));

    ret = CreateAclTensor(initStateHostData, stateShape, &initStateDeviceAddr, aclDataType::ACL_BF16, &initState);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(queryHostData, qkShape, &queryDeviceAddr, aclDataType::ACL_BF16, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHostData, qkShape, &keyDeviceAddr, aclDataType::ACL_BF16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(valueHostData, vShape, &valueDeviceAddr, aclDataType::ACL_BF16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gamaHostData, gamaShape, &gamaDeviceAddr, aclDataType::ACL_FLOAT, &gama);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, gamaShape, &betaDeviceAddr, aclDataType::ACL_BF16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(actSeqLenHostData, actSeqLenShape, &actSeqLenDeviceAddr, aclDataType::ACL_INT32, &actSeqLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, vShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(finalStateHostData, stateShape, &finalStateDeviceAddr, aclDataType::ACL_BF16, &finalState);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    float scale = 1.0;
    aclOpExecutor *executor;
    // 调用aclnnChunkGatedDeltaRuleGetWorkspaceSize第一段接口
    ret = aclnnChunkGatedDeltaRuleGetWorkspaceSize(query, key, value, beta, initState, actSeqLen, gama,
                                                       scale, attnOut, finalState, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkGatedDeltaRuleGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnChunkGatedDeltaRule第二段接口
    ret = aclnnChunkGatedDeltaRule(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkGatedDeltaRule failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(vShape, &attnOutDeviceAddr, "out");
    PrintOutResult(stateShape, &finalStateDeviceAddr, "finalState");

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(query);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(gama);
    aclDestroyTensor(beta);
    aclDestroyTensor(initState);
    aclDestroyTensor(actSeqLen);
    aclDestroyTensor(attnOut);
    aclDestroyTensor(finalState);

    // 7. 释放device资源
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(gamaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(initStateDeviceAddr);
    aclrtFree(actSeqLenDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
    aclrtFree(finalStateDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
