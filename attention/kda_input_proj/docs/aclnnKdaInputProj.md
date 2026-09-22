# aclnnKdaInputProj

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/9.2.0/attention/kda_input_proj)

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

- **接口功能**：推理场景下 Recurrent KDA 的前处理计算。对输入 $x$ 分别投影得到 `qkv`、`beta`、`gate`、`g`：
    - Stage1：AIC 并行计算 `X @ W_beta`、`X @ W_gate`、`X @ W_g`（BF16 Cube，FP32 累加）；AIV 并行执行 DynamicMxQuant。
    - Stage2：AIV 对 `beta` 做逐元素 Sigmoid 并写回 FP32；AIC 执行 QuantMatmul(`qkv`) 。
- **计算公式**：

    $$
    \mathbf{beta}_{raw} = X W_{\beta}^{\mathrm{T}},\quad
    \mathbf{gate} = X W_{gate}^{\mathrm{T}},\quad
    \mathbf{g} = X W_{g}^{\mathrm{T}}
    $$

    $$
    \mathbf{beta} = \sigma(\mathbf{beta}_{raw}) = \frac{1}{1 + e^{-\mathbf{beta}_{raw}}}
    $$

    $$
    \mathbf{qkv} = \mathrm{QuantMatmul}\bigl(\mathrm{DynamicMxQuant}(X),\; W_{qkv},\; \mathrm{weight\_qkv\_scale}\bigr)
    $$

    其中 $X$ 为 $[T,K]$ BF16；`gate`/`g` 为 BF16；`beta` 保持 FP32 后做 Sigmoid。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnKdaInputProjGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnKdaInputProj”接口执行计算。

```cpp
aclnnStatus aclnnKdaInputProjGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightQkv,
  const aclTensor *weightBeta,
  const aclTensor *weightGate,
  const aclTensor *weightG,
  const aclTensor *weightQkvScale,
  const aclTensor *qkvOut,
  const aclTensor *betaOut,
  const aclTensor *gateOut,
  const aclTensor *gOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnKdaInputProj(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnKdaInputProjGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1700px"><colgroup>
  <col style="width: 181px">
  <col style="width: 121px">
  <col style="width: 301px">
  <col style="width: 331px">
  <col style="width: 217px">
  <col style="width: 111px">
  <col style="width: 240px">
  <col style="width: 150px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>隐藏层输入，对应公式中的 $X$。</td>
      <td>必选。不支持空Tensor。输入侧接口内部做AutoContiguous。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[T, K]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightQkv</td>
      <td>输入</td>
      <td>qkv投影权重，对应公式中的 $W_{qkv}$。</td>
      <td>必选。不支持空Tensor。K维必须与x的K一致。</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>[K, N_qkv]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightBeta</td>
      <td>输入</td>
      <td>beta投影权重，对应公式中的 $W_{\beta}$。</td>
      <td>必选。不支持空Tensor。K维必须与x的K一致。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[K, N_beta]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightGate</td>
      <td>输入</td>
      <td>gate投影权重，对应公式中的 $W_{gate}$。</td>
      <td>必选。不支持空Tensor。K维必须与x的K一致。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[K, N_gate]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightG</td>
      <td>输入</td>
      <td>g投影权重，对应公式中的 $W_{g}$。</td>
      <td>必选。不支持空Tensor。K维必须与x的K一致。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[K, N_g]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightQkvScale</td>
      <td>输入</td>
      <td>weightQkv 的 MX 量化缩放因子，最后一维为高低位打包。</td>
      <td>必选。不支持空Tensor。最后一维固定为2。MX block大小为64。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>weightQkv 为转置 view 时：[N_qkv, Ceil(K/64), 2]；否则：[Ceil(K/64), N_qkv, 2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qkvOut</td>
      <td>输出</td>
      <td>qkv投影输出。</td>
      <td></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[T, N_qkv]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>betaOut</td>
      <td>输出</td>
      <td>beta投影后再做Sigmoid的输出，对应公式中的 $\mathbf{beta}$。</td>
      <td>调用方预分配。输出为FP32。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[T, N_beta]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gateOut</td>
      <td>输出</td>
      <td>gate投影输出，对应公式中的 $\mathbf{gate}$。</td>
      <td>调用方预分配。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[T, N_gate]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gOut</td>
      <td>输出</td>
      <td>g投影输出，对应公式中的 $\mathbf{g}$。</td>
      <td>调用方预分配。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[T, N_g]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出参数</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出参数</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 275px">
  <col style="width: 125px">
  <col style="width: 755px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必须传入的参数中存在空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入参数的shape、dtype不在支持的范围之内，或权重K维与x不一致。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>API内存调用npu runtime的接口异常。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>tiling发生异常，入参的dtype类型或者shape错误。</td>
    </tr>
  </tbody>
  </table>

## aclnnKdaInputProj

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
      <td>在Device侧申请的workspace内存地址。workspaceSize为0时可以传入空指针。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnKdaInputProjGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnKdaInputProj默认确定性实现。
- shape字段含义及约束
    - T：token个数，取值范围大于0。
    - K：隐藏层大小，取值范围大于0。
    - N_qkv / N_beta / N_gate / N_g：各路投影输出特征数，取值范围大于0。
- 数据类型约束
    - x、weightBeta、weightGate、weightG、gateOut、gOut、qkvOut：BF16。
    - weightQkv：FLOAT8_E4M3FN。
    - weightQkvScale：FLOAT8_E8M0。
    - betaOut：FLOAT。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。完整源码见[test_aclnn_kda_input_proj.cpp](../examples/test_aclnn_kda_input_proj.cpp)。

```Cpp
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_kda_input_proj.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
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
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()), dataType, strides.data(), 0,
                              ACL_FORMAT_ND, shape.data(), static_cast<int64_t>(shape.size()), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t T = 1;
    const int64_t K = 256;
    const int64_t Nqkv = 128;
    const int64_t Nbeta = 16;
    const int64_t Ngate = 32;
    const int64_t Ng = 32;
    const int64_t mxK = (K + 63) / 64;

    std::vector<int64_t> xShape = {T, K};
    std::vector<int64_t> wQkvShape = {K, Nqkv};
    std::vector<int64_t> wBetaShape = {K, Nbeta};
    std::vector<int64_t> wGateShape = {K, Ngate};
    std::vector<int64_t> wGShape = {K, Ng};
    std::vector<int64_t> weightQkvScaleShape = {mxK, Nqkv, 2};
    std::vector<int64_t> qkvShape = {T, Nqkv};
    std::vector<int64_t> betaShape = {T, Nbeta};
    std::vector<int64_t> gateShape = {T, Ngate};
    std::vector<int64_t> gShape = {T, Ng};

    std::vector<uint16_t> xHost(T * K, 0x3F80);
    std::vector<uint8_t> wQkvHost(Nqkv * K, 0);
    std::vector<uint16_t> wBetaHost(Nbeta * K, 0);
    std::vector<uint16_t> wGateHost(Ngate * K, 0);
    std::vector<uint16_t> wGHost(Ng * K, 0);
    std::vector<uint8_t> weightQkvScaleHost(Nqkv * mxK * 2, 0);
    std::vector<uint16_t> qkvHost(T * Nqkv, 0);
    std::vector<float> betaHost(T * Nbeta, 0.0f);
    std::vector<uint16_t> gateHost(T * Ngate, 0);
    std::vector<uint16_t> gHost(T * Ng, 0);

    void *xDev = nullptr, *wQkvDev = nullptr, *wBetaDev = nullptr, *wGateDev = nullptr, *wGDev = nullptr;
    void *weightQkvScaleDev = nullptr, *qkvDev = nullptr, *betaDev = nullptr, *gateDev = nullptr, *gDev = nullptr;
    aclTensor *x = nullptr, *wQkv = nullptr, *wBeta = nullptr, *wGate = nullptr, *wG = nullptr;
    aclTensor *weightQkvScale = nullptr, *qkv = nullptr, *beta = nullptr, *gate = nullptr, *g = nullptr;

    CreateAclTensor(xHost, xShape, &xDev, ACL_BF16, &x);
    CreateAclTensor(wQkvHost, wQkvShape, &wQkvDev, ACL_FLOAT8_E4M3FN, &wQkv);
    CreateAclTensor(wBetaHost, wBetaShape, &wBetaDev, ACL_BF16, &wBeta);
    CreateAclTensor(wGateHost, wGateShape, &wGateDev, ACL_BF16, &wGate);
    CreateAclTensor(wGHost, wGShape, &wGDev, ACL_BF16, &wG);
    CreateAclTensor(weightQkvScaleHost, weightQkvScaleShape, &weightQkvScaleDev, ACL_FLOAT8_E8M0, &weightQkvScale);
    CreateAclTensor(qkvHost, qkvShape, &qkvDev, ACL_BF16, &qkv);
    CreateAclTensor(betaHost, betaShape, &betaDev, ACL_FLOAT, &beta);
    CreateAclTensor(gateHost, gateShape, &gateDev, ACL_BF16, &gate);
    CreateAclTensor(gHost, gShape, &gDev, ACL_BF16, &g);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnKdaInputProjGetWorkspaceSize(x, wQkv, wBeta, wGate, wG, weightQkvScale, qkv, beta, gate, g, &workspaceSize,
                                         &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnKdaInputProj(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKdaInputProj failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclDestroyTensor(x);
    aclDestroyTensor(wQkv);
    aclDestroyTensor(wBeta);
    aclDestroyTensor(wGate);
    aclDestroyTensor(wG);
    aclDestroyTensor(weightQkvScale);
    aclDestroyTensor(qkv);
    aclDestroyTensor(beta);
    aclDestroyTensor(gate);
    aclDestroyTensor(g);
    aclrtFree(xDev);
    aclrtFree(wQkvDev);
    aclrtFree(wBetaDev);
    aclrtFree(wGateDev);
    aclrtFree(wGDev);
    aclrtFree(weightQkvScaleDev);
    aclrtFree(qkvDev);
    aclrtFree(betaDev);
    aclrtFree(gateDev);
    aclrtFree(gDev);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
