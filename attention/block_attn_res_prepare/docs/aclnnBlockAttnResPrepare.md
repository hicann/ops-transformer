# aclnnBlockAttnResPrepare

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：完成Attention Residuals历史残差注意力两阶段计算的第一阶段，批量计算全部 $S$ 个目标slot的历史在线Softmax状态 $(O,M,L)$，供下一阶段与当前残差合并。

- 计算公式：

  设 $V$ 为历史残差`blockRes`的FP32表示，$Q$ 为有效Query`pseudoQuery`，$N_v=\min(\texttt{validBlocks[0]},N)$ 为有效历史残差块数，$\epsilon=\operatorname{FP32}(\texttt{eps})$。

  当 $N_v>0$ 时，先计算历史残差的均方根归一化（Root Mean Square Normalization，RMSNorm）和注意力logit：

  $$
  \widehat V_{t,n}=\operatorname{RMSNorm}(V_{t,n},\mathbf{1},\epsilon),\qquad
  Z_{s,t,n}=\left\langle Q_s,\widehat V_{t,n}\right\rangle
  $$

  再沿有效历史残差块维度计算在线Softmax状态：

  $$
  M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n},\qquad
  E_{s,t,n}=\exp(Z_{s,t,n}-M_{s,t})
  $$

  $$
  L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n},\qquad
  O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}
  $$

  其中：

  - $T$、$N$、$S$、$D$ 分别表示token数、历史残差块容量、目标slot数和隐藏维度；$t\in[0,T)$、$n\in[0,N_v)$、$s\in[0,S)$、$d\in[0,D)$ 为对应索引，$V_{t,n}$ 和 $Q_s$ 为 $D$ 维向量。
  - RMSNorm沿 $D$ 维计算，缩放参数为全1向量；打分所需的缩放参数可预先与原始Query逐元素相乘，合并到 $Q$ 中。
  - $O$、$M$、$L$ 分别对应输出`numerator`、`logitMax`和`expSum`。$O$ 是未除以 $L$ 的加权分子。

  当 $N_v=1$ 且logit有限时，公式可化简为 $O_{s,t,d}=V_{t,0,d}$、$M_{s,t}=Z_{s,t,0}$、$L_{s,t}=1$。logit为NaN或Inf时，输出可能包含NaN。

  当 $N_v=0$ 时，返回在线Softmax的空状态：

  $$
  O=\mathbf{0}_{S\times T\times D},\qquad
  M=m_{\min}\mathbf{1}_{S\times T},\qquad
  L=\mathbf{0}_{S\times T},\qquad
  m_{\min}=-3.4028234663852886\times10^{38}
  $$

  $m_{\min}$ 为FLOAT32最小有限值。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBlockAttnResPrepareGetWorkspaceSize”接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用“aclnnBlockAttnResPrepare”接口执行计算。

```cpp
aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(
  const aclTensor  *blockRes,
  const aclTensor  *validBlocks,
  const aclTensor  *pseudoQuery,
  aclTensor        *numerator,
  aclTensor        *logitMax,
  aclTensor        *expSum,
  double            eps,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnBlockAttnResPrepare(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBlockAttnResPrepareGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 350px">
      <col style="width: 250px">
      <col style="width: 100px">
      <col style="width: 100px">
      <col style="width: 100px">
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
        <td>历史残差块，对应公式中的V。</td>
        <td>
          <ul>
            <li>支持空Tensor，T可以为0。</li>
            <li>shape为[T, N, D]。</li>
            <li>BFLOAT16输入转换为FLOAT32进行计算。</li>
          </ul>
        </td>
        <td>BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>validBlocks（const aclTensor*）</td>
        <td>输入</td>
        <td>当前有效历史残差块数。</td>
        <td>
          <ul>
            <li>不支持空Tensor，shape为[1]。</li>
            <li>仅前min(validBlocks[0], N)个历史残差块参与计算。</li>
            <li>值为0时，numerator和expSum返回全0Tensor，logitMax的所有元素均为FLOAT32最小有限值。</li>
          </ul>
        </td>
        <td>UINT64</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseudoQuery（const aclTensor*）</td>
        <td>输入</td>
        <td>伪Query，对应公式中的Q；每个目标slot对应一个Query向量。</td>
        <td>
          <ul>
            <li>支持空Tensor，S可以为0。</li>
            <li>shape为[S, D]，最后一维必须与blockRes的最后一维相同。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>numerator（aclTensor*）</td>
        <td>输出</td>
        <td>softmax加权分子，对应公式中的O。</td>
        <td>
          <ul>
            <li>支持空Tensor。</li>
            <li>shape为[S, T, D]。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>logitMax（aclTensor*）</td>
        <td>输出</td>
        <td>有效历史来源logit的最大值，对应公式中的M。</td>
        <td>
          <ul>
            <li>支持空Tensor。</li>
            <li>shape为[S, T]。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>expSum（aclTensor*）</td>
        <td>输出</td>
        <td>max-shift后的指数和，对应公式中的L。</td>
        <td>
          <ul>
            <li>支持空Tensor。</li>
            <li>shape为[S, T]。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>eps（double）</td>
        <td>输入</td>
        <td>RMS归一化的数值稳定项，对应公式中的ε。</td>
        <td>eps经范围校验后转换为FLOAT32属性参与计算。必须为有限值，且位于FLOAT32正规正数范围<code>[FLT_MIN, FLT_MAX]</code>（约<code>[1.1754943508222875e-38, 3.4028234663852886e+38]</code>）。不支持次正规数或收窄后为0、Inf的值。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t*）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小，单位为字节。</td>
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

  该接口针对参数异常返回以下错误码：

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>返回码/状态码</th>
        <th>码值</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>blockRes、validBlocks、pseudoQuery、numerator、logitMax、expSum、workspaceSize或executor为空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>输入或输出的数据类型、维数、shape或连续性不满足约束，或者eps非有限值或不在上述FLOAT32正规正数范围内。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>当前运行环境不在接口支持的产品范围内。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建算子执行器失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>连续化处理、算子执行节点创建或输出拷贝节点创建失败。</td>
      </tr>
    </tbody>
  </table>

## aclnnBlockAttnResPrepare

- **参数说明**

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 700px">
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
        <td>Device侧workspace内存地址。workspaceSize为0时可传入<code>nullptr</code>。</td>
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

## 约束说明

- 确定性计算：

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：aclnnBlockAttnResPrepare默认确定性实现。
  <!-- end id7 -->

- Batch一致性说明：默认非Batch一致性实现，不支持通过aclrtSetSysParamOpt开启Batch一致性。
- `T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- 非空计算（`T > 0`且`S > 0`）时，`T`和`S`不能超过`4294967295`（`UINT32_MAX`），`T * S`不能超过`9223372036854775807`（`INT64_MAX`）。
- `blockRes`与`pseudoQuery`的最后一维必须相同。
- `validBlocks[0] == 0`时，返回online softmax的空状态；`validBlocks[0] > N`时按`N`处理。
- 所有输入和输出Tensor均必须为连续Tensor，不支持非连续Tensor。
- `T == 0`或`S == 0`时，在参数校验和形状推导通过后返回对应shape的空输出，不发起AI Core计算；上述非空计算的上限不适用于该路径，`eps`有效范围及其他接口校验仍适用。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_prepare.h"

#define CHECK_RET(cond, returnExpr) \
    do { \
        if (!(cond)) { \
            returnExpr; \
        } \
    } while (0)

#define CHECK_FREE_RET(cond, returnExpr) \
    do { \
        if (!(cond)) { \
            Finalize(deviceId, stream); \
            returnExpr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

constexpr int64_t ROW_MAJOR_STRIDE_START_OFFSET = 2;
constexpr double DEFAULT_EPS = 1.0e-6;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (const auto dim : shape) {
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
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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

int RunBlockAttnResPrepare(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

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

    void* blockResDeviceAddr = nullptr;
    void* validBlocksDeviceAddr = nullptr;
    void* pseudoQueryDeviceAddr = nullptr;
    void* numeratorDeviceAddr = nullptr;
    void* logitMaxDeviceAddr = nullptr;
    void* expSumDeviceAddr = nullptr;
    aclTensor* blockRes = nullptr;
    aclTensor* validBlocks = nullptr;
    aclTensor* pseudoQuery = nullptr;
    aclTensor* numerator = nullptr;
    aclTensor* logitMax = nullptr;
    aclTensor* expSum = nullptr;

    ret = CreateAclTensor(blockResHostData, blockResShape, &blockResDeviceAddr, aclDataType::ACL_FLOAT, &blockRes);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> blockResTensorPtr(blockRes, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> blockResDeviceAddrPtr(blockResDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(validBlocksHostData, validBlocksShape, &validBlocksDeviceAddr, aclDataType::ACL_UINT64,
                          &validBlocks);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> validBlocksTensorPtr(validBlocks, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> validBlocksDeviceAddrPtr(validBlocksDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(pseudoQueryHostData, pseudoQueryShape, &pseudoQueryDeviceAddr, aclDataType::ACL_FLOAT,
                          &pseudoQuery);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> pseudoQueryTensorPtr(pseudoQuery, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> pseudoQueryDeviceAddrPtr(pseudoQueryDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(numeratorHostData, numeratorShape, &numeratorDeviceAddr, aclDataType::ACL_FLOAT, &numerator);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> numeratorTensorPtr(numerator, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> numeratorDeviceAddrPtr(numeratorDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(logitMaxHostData, statsShape, &logitMaxDeviceAddr, aclDataType::ACL_FLOAT, &logitMax);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> logitMaxTensorPtr(logitMax, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> logitMaxDeviceAddrPtr(logitMaxDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(expSumHostData, statsShape, &expSumDeviceAddr, aclDataType::ACL_FLOAT, &expSum);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> expSumTensorPtr(expSum, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> expSumDeviceAddrPtr(expSumDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnBlockAttnResPrepareGetWorkspaceSize(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum,
                                                   DEFAULT_EPS, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResPrepareGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    ret = aclnnBlockAttnResPrepare(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResPrepare failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(numeratorHostData.data(), numeratorHostData.size() * sizeof(numeratorHostData[0]),
                      numeratorDeviceAddr, numeratorHostData.size() * sizeof(numeratorHostData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy numerator from device to host failed. ERROR: %d\n", ret); return ret);
    ret =
        aclrtMemcpy(logitMaxHostData.data(), logitMaxHostData.size() * sizeof(logitMaxHostData[0]), logitMaxDeviceAddr,
                    logitMaxHostData.size() * sizeof(logitMaxHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy logitMax from device to host failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(expSumHostData.data(), expSumHostData.size() * sizeof(expSumHostData[0]), expSumDeviceAddr,
                      expSumHostData.size() * sizeof(expSumHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy expSum from device to host failed. ERROR: %d\n", ret); return ret);

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
    const auto ret = RunBlockAttnResPrepare(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBlockAttnResPrepare failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
