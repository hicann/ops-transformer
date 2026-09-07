# aclnnBlockAttentionResiduals

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mhc/block_attention_residuals)

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

- 接口功能：将`partialBlock`与`blockRes`拼接后，经RMS归一化、线性投影和softmax加权融合，输出`hiddenStates`；当`needBackward`为true时，同时输出反向计算所需的`invNorm`和`probs`。

- 计算公式：

  正向计算可表示为：

  $$
  v_{t,i,h} =
  \begin{cases}
  block\_res_{t,i,h}, & i < N \\
  partial\_block_{t,h}, & i = N
  \end{cases}
  $$

  $$
  variance_{t,i} = \frac{1}{H}\sum_{h=0}^{H-1} v_{t,i,h}^{2}
  $$

  $$
  inv\_rms_{t,i} = \frac{1}{\sqrt{variance_{t,i} + norm\_eps}}
  $$

  $$
  k_{t,i,h} = v_{t,i,h} \cdot inv\_rms_{t,i}
  $$

  $$
  score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
  $$

  $$
  s_{t,i} = \sum_{h=0}^{H-1} k_{t,i,h} \cdot score\_weight_{h}
  $$

  $$
  probs_{t,i} = \frac{e^{s_{t,i}}}{\sum_{j=0}^{N} e^{s_{t,j}}}
  $$

  $$
  hidden\_states_{t,h} = \sum_{i=0}^{N} probs_{t,i} \cdot v_{t,i,h}
  $$

  其中：
  - $T$ 表示 token 数量，$N$ 表示 `blockRes` 的 block 数量，$H$ 表示 hidden size。
  - $v_{t,i,h}$ 表示 `blockRes` 与 `partialBlock` 拼接后的 value，维度为 $(T,N+1,H)$。
  - $inv\_rms_{t,i}$ 表示前向逐行归一化系数。
  - $score\_weight_{h}$ 表示 `normWeight` 与 `projWeight` 逐元素相乘的结果。
  - $probs_{t,i}$ 表示前向 softmax 输出概率。
  - $hidden\_states_{t,h}$ 表示前向加权融合输出。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnBlockAttentionResidualsGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBlockAttentionResiduals"接口执行计算。

```Cpp
aclnnStatus aclnnBlockAttentionResidualsGetWorkspaceSize(
    const aclTensor *partialBlock,
    const aclTensor *blockRes,
    const aclTensor *projWeight,
    const aclTensor *normWeight,
    int64_t          validBlockNum,
    double           normEps,
    bool             needBackward,
    aclTensor       *hiddenStates,
    aclTensor       *invNorm,
    aclTensor       *probs,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBlockAttentionResiduals(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnBlockAttentionResidualsGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 387px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 187px">
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
  </tr></thead>
  <tbody>
  <tr>
      <td>partialBlock（aclTensor*）</td>
      <td>输入</td>
      <td>前向输入partialBlock，对应公式中拼接后的第N+1个value。</td>
      <td><ul><li>支持空Tensor，T可以为0。</li><li>shape为(T, H)。</li><li>T、H须与blockRes保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
  </tr>
  <tr>
      <td>blockRes（aclTensor*）</td>
      <td>输入</td>
      <td>前向输入分块残差，对应公式中前N个value。</td>
      <td><ul><li>支持空Tensor，T可以为0。</li><li>shape为(T, N, H)。</li><li>数据类型须与partialBlock保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
  </tr>
  <tr>
      <td>projWeight（aclTensor*）</td>
      <td>输入</td>
      <td>前向线性投影权重，与normWeight共同构成score_weight。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(H)或(1, H)。</li><li>数据类型须与partialBlock保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
  </tr>
  <tr>
      <td>normWeight（aclTensor*）</td>
      <td>输入</td>
      <td>前向归一化权重，与projWeight共同构成score_weight。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(H)。</li><li>数据类型须与partialBlock保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
  </tr>
  <tr>
      <td>validBlockNum（int64_t）</td>
      <td>输入</td>
      <td>当前有效的block数量。</td>
      <td><ul><li>默认值：-1，表示使用blockRes的N。</li><li>当前须为-1或等于blockRes的N。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>normEps（double）</td>
      <td>可选输入</td>
      <td>RMS归一化数值稳定性参数，对应公式中的norm_eps。</td>
      <td><ul><li>须大于0。</li><li>默认值：1e-6。</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>needBackward（bool）</td>
      <td>可选输入</td>
      <td>是否保存invNorm和probs供反向计算使用。</td>
      <td><ul><li>默认值：false。</li><li>为true时invNorm、probs不能为空指针。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>hiddenStates（aclTensor*）</td>
      <td>输出</td>
      <td>前向加权融合输出，对应公式中的hidden_states。</td>
      <td><ul><li>支持空Tensor，T可以为0。</li><li>shape为(T, H)。</li><li>数据类型须与partialBlock保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
  </tr>
  <tr>
      <td>invNorm（aclTensor*）</td>
      <td>可选输出</td>
      <td>前向保存的逐行归一化系数，对应公式中的inv_rms。</td>
      <td><ul><li>needBackward为false时可传空指针。</li><li>needBackward为true时不支持空指针，shape为(T, N+1)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
  </tr>
  <tr>
      <td>probs（aclTensor*）</td>
      <td>可选输出</td>
      <td>前向softmax输出概率，对应公式中的probs。</td>
      <td><ul><li>needBackward为false时可传空指针。</li><li>needBackward为true时不支持空指针，shape为(T, N+1)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
  </tr>
  <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 330px">
  <col style="width: 140px">
  <col style="width: 762px">
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
    <td> ACLNN_ERR_PARAM_NULLPTR </td>
    <td> 161001 </td>
    <td>partialBlock、blockRes、projWeight、normWeight、hiddenStates存在空指针；needBackward为true时invNorm、probs存在空指针。</td>
    </tr>
    <tr>
    <td rowspan="3"> ACLNN_ERR_PARAM_INVALID </td>
    <td rowspan="3"> 161002 </td>
    <td>partialBlock、blockRes、projWeight、normWeight、hiddenStates、invNorm、probs的数据类型不在支持的范围内，或计算张量之间的数据类型不一致。</td>
    </tr>
    <tr>
    <td>partialBlock、blockRes、projWeight、normWeight、hiddenStates、invNorm、probs的shape维度不在支持的范围内。</td>
    </tr>
    <tr>
    <td>validBlockNum既不是-1也不等于blockRes的N，或normEps小于等于0。</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_INNER_CREATE_EXECUTOR </td>
    <td> 561101 </td>
    <td>接口内部创建aclOpExecutor失败。</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_INNER_NULLPTR </td>
    <td> 561103 </td>
    <td>接口内部执行连续化处理、创建算子输出或创建输出拷贝节点时返回空指针。</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_RUNTIME_ERROR </td>
    <td> 361001 </td>
    <td>API内部调用npu runtime的接口异常。</td>
    </tr>
  </tbody></table>

## aclnnBlockAttentionResiduals

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 134px">
  <col style="width: 844px">
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
    <td>workspace</td>
    <td>输入</td>
    <td>在Device侧申请的workspace内存地址。</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockAttentionResidualsGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnBlockAttentionResiduals默认确定性实现。

- Batch一致性说明：
  - aclnnBlockAttentionResiduals默认Batch一致性实现。

- 规格约束：
  - T大于等于0，H大于等于1，N取值范围为1~100。
  - partialBlock、blockRes、projWeight、normWeight、hiddenStates的数据类型须一致，支持FLOAT16、BFLOAT16、FLOAT32；invNorm、probs仅支持FLOAT32。
  - shape须满足：partialBlock为(T, H)，blockRes为(T, N, H)，projWeight为(H)或(1, H)，normWeight为(H)，hiddenStates为(T, H)；needBackward为true时invNorm、probs为(T, N+1)。
  - validBlockNum须为-1（默认，使用N）或等于blockRes的N。
  - normEps须大于0，默认值为1e-6。
  - 输入张量支持非连续Tensor，接口内部转为Contiguous后计算；输出不支持非连续Tensor。
  - workspace仅使用平台系统预留，不追加用户侧workspace。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_block_attention_residuals.h"

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
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
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

int CreateAclTensor(
    const std::vector<uint16_t>& hostData, const std::vector<int64_t>& shape,
    void** deviceAddr, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(uint16_t);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), aclDataType::ACL_BF16, nullptr, 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t T = 2;
    const int64_t N = 4;
    const int64_t H = 64;

    std::vector<int64_t> partialBlockShape     = {T, H};
    std::vector<int64_t> blockShape      = {T, N, H};
    std::vector<int64_t> projShape       = {1, H};
    std::vector<int64_t> normShape       = {H};
    std::vector<int64_t> hiddenShape     = {T, H};

    std::vector<uint16_t> partialBlockData(GetShapeSize(partialBlockShape), 0x3C00);
    std::vector<uint16_t> blockData(GetShapeSize(blockShape), 0x3C00);
    std::vector<uint16_t> projData(GetShapeSize(projShape), 0x3C00);
    std::vector<uint16_t> normData(GetShapeSize(normShape), 0x3C00);
    std::vector<uint16_t> hiddenData(GetShapeSize(hiddenShape), 0);

    void* partialBlockDev = nullptr;
    void* blockDev = nullptr;
    void* projDev = nullptr;
    void* normDev = nullptr;
    void* hiddenDev = nullptr;

    aclTensor* partialBlockTensor = nullptr;
    aclTensor* blockTensor = nullptr;
    aclTensor* projTensor = nullptr;
    aclTensor* normTensor = nullptr;
    aclTensor* hiddenTensor = nullptr;

    ret = CreateAclTensor(partialBlockData, partialBlockShape, &partialBlockDev, &partialBlockTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockData, blockShape, &blockDev, &blockTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(projData, projShape, &projDev, &projTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normData, normShape, &normDev, &normTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hiddenData, hiddenShape, &hiddenDev, &hiddenTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnBlockAttentionResidualsGetWorkspaceSize(
        partialBlockTensor, blockTensor, projTensor, normTensor, -1, 1e-6, false,
        hiddenTensor, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResidualsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnBlockAttentionResiduals(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttentionResiduals failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("BlockAttentionResiduals example ran successfully!\n");

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyTensor(partialBlockTensor);
    aclDestroyTensor(blockTensor);
    aclDestroyTensor(projTensor);
    aclDestroyTensor(normTensor);
    aclDestroyTensor(hiddenTensor);
    aclrtFree(partialBlockDev);
    aclrtFree(blockDev);
    aclrtFree(projDev);
    aclrtFree(normDev);
    aclrtFree(hiddenDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
