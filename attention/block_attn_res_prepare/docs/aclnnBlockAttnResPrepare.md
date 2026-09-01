# aclnnBlockAttnResPrepare

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/block_attn_res_prepare)

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

* 接口功能：Attention Residuals的历史残差注意力两段式计算的阶段一：一次性并行完成全部 S 层的块间注意力计算；同时返回输出结果与 softmax 统计量(O, M, L)，供阶段二融合当前残差并形成下一子层输入。
* 计算公式：

    $$
    \operatorname{block\_attn\_res\_prepare}
    \left(V,Q,\operatorname{valid\_blocks};\epsilon\right)
    \longrightarrow (O,M,L).
    $$

     对 token $t$ 的第 $n$ 个历史来源，首先计算 RMS 归一化因子：

    $$
    R_{t,n}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}V_{t,n,d}^{2}+\epsilon}.
    $$

     对每个 slot、token 和有效历史来源计算 logits：

    $$
    Z_{s,t,n}=R_{t,n}^{-1}\sum_{d=0}^{D-1}Q_{s,d}V_{t,n,d}.
    $$

     设 $N_v$ 为 `valid_blocks` 指定的有效历史来源数，则：

    $$
    M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},
    $$

    $$
    E_{s,t,n}=\exp\left(Z_{s,t,n}-M_{s,t}\right),
    $$

    $$
    L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},
    $$

    $$
    O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}.
    $$

     当 `validBlocks[0] == 0` 时，不执行上述 softmax 计算，直接返回：

    $$
    O=\mathbf{0}_{S\times T\times D},\qquad
    M=m_{\min}\mathbf{1}_{S\times T},\qquad
    L=\mathbf{0}_{S\times T},\qquad
    m_{\min}=-3.4028234663852886\times10^{38}.
    $$

      即 `numerator` 为 shape `[S, T, D]` 的全 0 Tensor，`logitMax` 为 shape `[S, T]` 且所有元素均为 FLOAT32 最小有限值的 Tensor，`expSum` 为 shape `[S, T]` 的全 0 Tensor。

  其中：
    - $V\in\mathbb{R}^{T\times N\times D}$ 表示输入 `blockRes`；$V_{t,n,d}$ 表示第 $t$ 个 token、第 $n$ 个历史残差块在第 $d$ 个隐藏特征上的值。
    - $Q\in\mathbb{R}^{S\times D}$ 表示输入 `pseudoQuery`（伪 Query）；每个目标 slot 对应一个 Query 向量，$Q_{s,d}$ 表示第 $s$ 个目标 slot 在第 $d$ 个隐藏特征上的值。
    - $N_v=\min(\texttt{validBlocks[0]},N)$，表示实际参与计算的历史残差块数。
    - $\epsilon$ 表示输入 `eps`，是计算 RMS 归一化因子时使用的数值稳定项。
    - $T$ 表示 token 数，$N$ 表示 `blockRes` 第 1 维可容纳的历史残差块数，$S$ 表示目标 slot 数，$D$ 表示HiddenSize。
    - $t\in[0,T)$、$n\in[0,N_v)$、$s\in[0,S)$、$d\in[0,D)$ 分别表示 token、有效历史残差块、目标 slot 和隐藏特征的索引。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBlockAttnResPrepareGetWorkspaceSize”接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用“aclnnBlockAttnResPrepare”接口执行计算。

```cpp
aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(
    const aclTensor *blockRes,
    const aclTensor *validBlocks,
    const aclTensor *pseudoQuery,
    aclTensor       *numerator,
    aclTensor       *logitMax,
    aclTensor       *expSum,
    float            eps,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```
```cpp
aclnnStatus aclnnBlockAttnResPrepare(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```


## aclnnBlockAttnResPrepareGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1503px"><colgroup>
  <col style="width: 130px">
  <col style="width: 97px">
  <col style="width: 308px">
  <col style="width: 488px">
  <col style="width: 197px">
  <col style="width: 77px">
  <col style="width: 115px">
  <col style="width: 95px">
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
      <td>blockRes（const aclTensor*）</td>
      <td>输入</td>
      <td>表示残差块，对应公式中的输入V。</td>
      <td><li>支持空Tensor，T可以为0。</li><li>shape为[T, N, D]。</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>validBlocks（const aclTensor*）</td>
      <td>输入</td>
      <td>表示当前有效历史块数。</td>
      <td><li>不支持空Tensor，shape为[1]。</li><li>Tensor中的值为N_v时，计算时blockRes的N维仅前N_v参与运算。值为0时，numerator返回全0 Tensor、logitMax返回所有元素均为FLOAT32最小有限值的Tensor、expSum返回全0 Tensor；大于N时按N处理。</li></td>
      <td>UINT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>pseudoQuery（const aclTensor*）</td>
      <td>输入</td>
      <td>表示伪 Query，对应公式中的输入Q；每个目标slot对应一个Query向量。</td>
      <td><li>支持空Tensor，S可以为0。</li><li>shape为[S, D]，最后一维必须与blockRes的最后一维相同。</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numerator（aclTensor*）</td>
      <td>输出</td>
      <td>表示softmax加权分子，对应公式中的输出O。</td>
      <td><li>支持空Tensor。</li><li>shape为[S, T, D]。</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>logitMax（aclTensor*）</td>
      <td>输出</td>
      <td>表示softmax最大值，对应公式中的输出M。</td>
      <td><li>支持空Tensor。</li><li>shape为[S, T]。</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expSum（aclTensor*）</td>
      <td>输出</td>
      <td>表示softmax指数和，对应公式中的输出L。</td>
      <td><li>支持空Tensor。</li><li>shape为[S, T]。</li></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>eps（float）</td>
      <td>输入</td>
      <td>表示RMS归一化的稳定项，通常为1e-6。</td>
      <td>必须为有限正数。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成参数校验和执行器创建，可能返回以下状态码：

  | 返回值 | 错误码 | 说明 |
  | --- | --- | --- |
  | `ACLNN_SUCCESS` | 0 | 接口调用成功。 |
  | `ACLNN_ERR_PARAM_NULLPTR` | 161001 | `blockRes`、`validBlocks`、`pseudoQuery`、`numerator`、`logitMax`、`expSum`、`workspaceSize` 或 `executor` 为空指针。 |
  | `ACLNN_ERR_PARAM_INVALID` | 161002 | 输入或输出的数据类型、维数、shape 或连续性不满足约束，或者 `eps` 不是有限正数。 |
  | `ACLNN_ERR_RUNTIME_ERROR` | 361001 | 当前运行平台不是 Ascend 950。 |
  | `ACLNN_ERR_INNER_CREATE_EXECUTOR` | 561101 | 接口内部创建 `aclOpExecutor` 失败。 |
  | `ACLNN_ERR_INNER_NULLPTR` | 561103 | 接口内部执行连续化处理、创建算子输出或创建输出拷贝节点时返回空指针。 |

## aclnnBlockAttnResPrepare

- **参数说明**

  <table style="table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>workspace（void*）</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t）</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockAttnResPrepareGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor*）</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream（aclrtStream）</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第二段接口调用 `CommonOpExecutorRun` 下发算子，成功时返回 `ACLNN_SUCCESS`（错误码为0）；执行失败时返回相应的 ACLNN Runtime 或内部错误码。

## 约束说明

- 确定性说明：aclnnBlockAttnResPrepare默认确定性实现。
- Batch一致性说明：aclnnBlockAttnResPrepare默认Batch一致性实现。
- `T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- `blockRes` 与 `pseudoQuery` 的最后一维必须相同。
- `validBlocks[0] == 0` 时，返回 online softmax 的空状态；`validBlocks[0] > N` 时按 `N` 处理。
- 所有输入和输出 Tensor 均必须为连续 Tensor，不支持非连续 Tensor。
- `T == 0` 或 `S == 0` 时返回对应形状的空输出。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_prepare.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

constexpr int64_t ROW_MAJOR_STRIDE_START_OFFSET = 2;
constexpr float DEFAULT_EPS = 1.0e-6F;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (const auto dim : shape) {
        shapeSize *= dim;
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
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - ROW_MAJOR_STRIDE_START_OFFSET; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnBlockAttnResPrepareTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
              return ret);

    // 构造输入与输出：1个token、2个历史来源、2个目标slot，hidden size为4。
    const std::vector<int64_t> blockResShape = {1, 2, 4};
    const std::vector<int64_t> validBlocksShape = {1};
    const std::vector<int64_t> pseudoQueryShape = {2, 4};
    const std::vector<int64_t> numeratorShape = {2, 1, 4};
    const std::vector<int64_t> statsShape = {2, 1};

    const std::vector<float> blockResHostData = {
        1.0F, 2.0F, 3.0F, 4.0F, 2.0F, 0.0F, -1.0F, 1.0F,
    };
    const std::vector<uint64_t> validBlocksHostData = {2U};
    const std::vector<float> pseudoQueryHostData = {
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
    };
    std::vector<float> numeratorHostData(GetShapeSize(numeratorShape), 0.0F);
    std::vector<float> logitMaxHostData(GetShapeSize(statsShape), 0.0F);
    std::vector<float> expSumHostData(GetShapeSize(statsShape), 0.0F);

    void *blockResDeviceAddr = nullptr;
    void *validBlocksDeviceAddr = nullptr;
    void *pseudoQueryDeviceAddr = nullptr;
    void *numeratorDeviceAddr = nullptr;
    void *logitMaxDeviceAddr = nullptr;
    void *expSumDeviceAddr = nullptr;
    aclTensor *blockRes = nullptr;
    aclTensor *validBlocks = nullptr;
    aclTensor *pseudoQuery = nullptr;
    aclTensor *numerator = nullptr;
    aclTensor *logitMax = nullptr;
    aclTensor *expSum = nullptr;

    ret = CreateAclTensor(blockResHostData, blockResShape, &blockResDeviceAddr, aclDataType::ACL_FLOAT, &blockRes);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> blockResTensorPtr(blockRes, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> blockResDeviceAddrPtr(blockResDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(validBlocksHostData, validBlocksShape, &validBlocksDeviceAddr, aclDataType::ACL_UINT64,
                          &validBlocks);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> validBlocksTensorPtr(validBlocks, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> validBlocksDeviceAddrPtr(validBlocksDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(pseudoQueryHostData, pseudoQueryShape, &pseudoQueryDeviceAddr, aclDataType::ACL_FLOAT,
                          &pseudoQuery);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pseudoQueryTensorPtr(pseudoQuery, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> pseudoQueryDeviceAddrPtr(pseudoQueryDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(numeratorHostData, numeratorShape, &numeratorDeviceAddr, aclDataType::ACL_FLOAT, &numerator);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> numeratorTensorPtr(numerator, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> numeratorDeviceAddrPtr(numeratorDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(logitMaxHostData, statsShape, &logitMaxDeviceAddr, aclDataType::ACL_FLOAT, &logitMax);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitMaxTensorPtr(logitMax, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> logitMaxDeviceAddrPtr(logitMaxDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(expSumHostData, statsShape, &expSumDeviceAddr, aclDataType::ACL_FLOAT, &expSum);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> expSumTensorPtr(expSum, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> expSumDeviceAddrPtr(expSumDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBlockAttnResPrepareGetWorkspaceSize(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum,
                                                   DEFAULT_EPS, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBlockAttnResPrepareGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnBlockAttnResPrepare(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBlockAttnResPrepare failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtMemcpy(numeratorHostData.data(), numeratorHostData.size() * sizeof(numeratorHostData[0]),
                      numeratorDeviceAddr, numeratorHostData.size() * sizeof(numeratorHostData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy numerator from device to host failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(logitMaxHostData.data(), logitMaxHostData.size() * sizeof(logitMaxHostData[0]),
                      logitMaxDeviceAddr, logitMaxHostData.size() * sizeof(logitMaxHostData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy logitMax from device to host failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(expSumHostData.data(), expSumHostData.size() * sizeof(expSumHostData[0]), expSumDeviceAddr,
                      expSumHostData.size() * sizeof(expSumHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy expSum from device to host failed. ERROR: %d\n", ret);
              return ret);

    for (size_t i = 0; i < numeratorHostData.size(); ++i) {
        LOG_PRINT("numerator[%zu] = %.6f\n", i, numeratorHostData[i]);
    }
    for (size_t i = 0; i < logitMaxHostData.size(); ++i) {
        LOG_PRINT("logit_max[%zu] = %.6f, exp_sum[%zu] = %.6f\n", i, logitMaxHostData[i], i, expSumHostData[i]);
    }
    return ACL_SUCCESS;
}

} // namespace

int main()
{
    // 根据实际运行环境设置Device ID。
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    const auto ret = aclnnBlockAttnResPrepareTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnBlockAttnResPrepareTest failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}

```
