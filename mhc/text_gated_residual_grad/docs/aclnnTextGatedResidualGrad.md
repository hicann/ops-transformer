# aclnnTextGatedResidualGrad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

`TextGatedResidualGrad` 是 `TextGatedResidual` 的反向算子。根据混合输出和注入输出的上游梯度，以及前向保存的张量，计算输入 `x` 和权重 `wDown`、`wUp`、`wGate`、`gamma` 的梯度。

记 $B$ 为批大小，$S$ 为序列长度，$C$ 为通道数，$H$ 为每通道的特征数，$R$ 为下投影维度，$D=C\times H$。$\odot$ 表示逐元素乘法，$@$ 表示矩阵乘法，$\sigma$ 表示 Sigmoid 函数。

对应的前向计算为：

$$
\begin{aligned}
\mathrm{invRms}_{b,s,c} &= \left(\frac{1}{H}\sum_{j=0}^{H-1}x_{b,s,c,j}^{2}+\epsilon\right)^{-1/2} \\
\mathrm{xNorm} &= x\odot\mathrm{invRms}\odot(1+\gamma) \\
\mathrm{downPreAct} &= \mathrm{xNorm}@\mathrm{wDown}^{T} \\
\mathrm{act} &= \mathrm{SiLU}(\mathrm{downPreAct}/C) \\
\mathrm{up} &= \mathrm{act}@\mathrm{wUp}^{T} \\
\mathrm{mixed}_{b,s,j} &= \frac{1}{C}\sum_{c=0}^{C-1}\mathrm{xNorm}_{b,s,c,j}\,\sigma(\mathrm{up}_{b,s,c,j}) \\
\mathrm{inject} &= 2\sigma\!\left((\mathrm{xNorm}@\mathrm{wGate}^{T})/C\right)
\end{aligned}
$$

其中，归一化及通道求和时将最后一维 $D$ 视为 $[C,H]$，矩阵乘法时仍使用最后一维 $D$。`invRms` 在每个通道的 $H$ 个特征上广播，`gamma` 在 $B$、$S$ 维上广播。$\epsilon$ 是前向归一化使用的参数，本接口通过 `invRms` 接收对应的前向结果。

给定 `gradMixed` 和 `gradInject`，定义

$$
L=\sum_{b,s,j}\mathrm{gradMixed}_{b,s,j}\,\mathrm{mixed}_{b,s,j}
 +\sum_{b,s,c}\mathrm{gradInject}_{b,s,c}\,\mathrm{inject}_{b,s,c}.
$$

本算子的输出为：

$$
dx=\frac{\partial L}{\partial x},\quad
dwDown=\frac{\partial L}{\partial wDown},\quad
dwUp=\frac{\partial L}{\partial wUp},\quad
dwGate=\frac{\partial L}{\partial wGate},\quad
dgamma=\frac{\partial L}{\partial\gamma}.
$$

`wGate` 缺省时不包含注入分支及其梯度；`gamma` 缺省时归一化缩放系数为 1，不输出 `dgamma`。`invRms`、`inject`、`downPreAct`、`up` 必须来自同一次前向计算，并与本次传入的输入、权重对应。

## 函数原型

本接口采用[两段式调用](../../../docs/zh/context/two_phase_api.md)：先调用 `aclnnTextGatedResidualGradGetWorkspaceSize` 获取工作空间大小和执行器，再调用 `aclnnTextGatedResidualGrad` 执行计算。

```cpp
aclnnStatus aclnnTextGatedResidualGradGetWorkspaceSize(
    const aclTensor* gradMixed, const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp,
    const aclTensor* invRms, const aclTensor* downPreAct, const aclTensor* up, const aclTensor* gradInject,
    const aclTensor* wGate, const aclTensor* gamma, const aclTensor* inject, int64_t channels, aclTensor* dx,
    aclTensor* dwDown, aclTensor* dwUp, aclTensor* dwGate, aclTensor* dgamma, uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnTextGatedResidualGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);
```

## aclnnTextGatedResidualGradGetWorkspaceSize

- **参数说明**

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | shape | 非连续 Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| gradMixed（const aclTensor*） | 输入 | 混合输出 mixed 的上游梯度。 | 与 x 数据类型一致。 | FLOAT16、BFLOAT16 | ND | [B,S,H] | √ |
| x（const aclTensor*） | 输入 | 前向输入。 | D=C*H。 | FLOAT16、BFLOAT16 | ND | [B,S,D] | √ |
| wDown（const aclTensor*） | 输入 | 下投影权重。 | 与 x 数据类型一致。 | FLOAT16、BFLOAT16 | ND | [R,D] | √ |
| wUp（const aclTensor*） | 输入 | 上投影权重。 | 与 x 数据类型一致。 | FLOAT16、BFLOAT16 | ND | [D,R] | √ |
| invRms（const aclTensor*） | 输入 | 前向保存的各通道均方根倒数。 | 对应前向的归一化结果，包含前向 epsilon 的作用。 | FLOAT32 | ND | [B,S,C] | √ |
| downPreAct（const aclTensor*） | 输入 | 前向保存的下投影结果 xNorm@wDownᵀ。 | 保存未除以 C、未经过 SiLU 的结果；与 x 数据类型一致。 | FLOAT16、BFLOAT16 | ND | [B,S,R] | √ |
| up（const aclTensor*） | 输入 | 前向保存的上投影结果 act@wUpᵀ。 | 保存 Sigmoid 之前的值；与 x 数据类型一致。 | FLOAT16、BFLOAT16 | ND | [B,S,D] | √ |
| gradInject（const aclTensor*） | 可选输入 | 注入输出 inject 的上游梯度。 | 与 x 数据类型一致；与 wGate、inject、dwGate 同时提供或同时缺省。 | FLOAT16、BFLOAT16 | ND | [B,S,C] | √ |
| wGate（const aclTensor*） | 可选输入 | 注入分支权重。 | 与 x 数据类型一致；与 gradInject、inject、dwGate 同时提供或同时缺省。 | FLOAT16、BFLOAT16 | ND | [C,D] | √ |
| gamma（const aclTensor*） | 可选输入 | 归一化缩放权重，实际缩放系数为 1+gamma。 | 传入原始 gamma；与 dgamma 同时提供或同时缺省。 | FLOAT32 | ND | [D] | √ |
| inject（const aclTensor*） | 可选输入 | 前向保存的注入输出 2*sigmoid((xNorm@wGateᵀ)/C)。 | 与 x 数据类型一致；与 wGate、gradInject、dwGate 同时提供或同时缺省。 | FLOAT16、BFLOAT16 | ND | [B,S,C] | √ |
| channels（int64_t） | 输入 | 通道数 C。 | C 为正整数，C ≤ 20。 | INT64 | - | 标量 | - |
| dx（aclTensor*） | 输出 | x 的梯度。 | shape、数据类型与 x 一致。 | 同 x | ND | [B,S,D] | √ |
| dwDown（aclTensor*） | 输出 | wDown 的梯度。 | shape 和数据类型与 wDown 一致。 | FLOAT16、BFLOAT16 | ND | [R,D] | √ |
| dwUp（aclTensor*） | 输出 | wUp 的梯度。 | shape 和数据类型与 wUp 一致。 | FLOAT16、BFLOAT16 | ND | [D,R] | √ |
| dwGate（aclTensor*） | 可选输出 | wGate 的梯度。 | shape 和数据类型与 wGate 一致；与 wGate、gradInject、inject 同时提供或同时缺省。 | FLOAT16、BFLOAT16 | ND | [C,D] | √ |
| dgamma（aclTensor*） | 可选输出 | gamma 的梯度。 | shape 与 gamma 一致；与 gamma 同时提供或同时缺省。 | FLOAT32 | ND | [D] | √ |
| workspaceSize（uint64_t*） | 输出 | 所需工作空间的字节数。 | 不能为空指针；按返回值申请 Device 内存。 | - | - | - | - |
| executor（aclOpExecutor**） | 输出 | 本次调用的执行器。 | 不能为空指针；用于第二段接口。 | - | - | - | - |

- **返回值**

返回 `aclnnStatus`，参见 [aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。

| 返回值 | 错误码 | 描述 |
| :--- | ---: | :--- |
| ACLNN_SUCCESS | 0 | 参数校验及执行准备成功。 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选 Tensor、workspaceSize 或 executor 为空指针，或可选参数未按要求成组提供。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | Tensor 的数据类型、维度或 shape 不符合参数表；channels 不在 [1,20] 范围内或不能整除 D；R 不在 [1,640] 范围内；B、S 为负数或 B*S 超出 INT64 表示范围。 |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 561101 | 执行器创建失败。 |
| ACLNN_ERR_INNER_NULLPTR | 561103 | 请求的形状组合不满足规格约束，或计算所需 Tensor 创建失败。 |

## aclnnTextGatedResidualGrad

- **参数说明**

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| workspace（void*） | 输入 | Device 侧工作空间地址；workspaceSize 为 0 时可传入 nullptr。 |
| workspaceSize（uint64_t） | 输入 | 第一段接口返回的工作空间字节数。 |
| executor（aclOpExecutor*） | 输入 | 第一段接口返回的执行器。 |
| stream（aclrtStream） | 输入 | 执行计算的 Stream。 |

- **返回值**

返回 `aclnnStatus`，参见 [aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。

| 返回值 | 错误码 | 描述 |
| :--- | ---: | :--- |
| ACLNN_SUCCESS | 0 | 计算任务提交成功。 |
| ACLNN_ERR_INNER | 561000 | 计算任务提交失败。 |

第二段接口异步提交计算。读取输出前，应调用 `aclrtSynchronizeStream` 等待任务完成并检查其返回值。

## 约束说明

- `C`、`H`、`R` 为正整数；`H`、`R` 均须为 16 的整数倍。

通道数须满足 `C ≤ 20`。

`H` 的取值范围按产品确定。令

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  $$
  K = 62448-16\left\lceil C/8\right\rceil
  $$

<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

  $$
  K = 39664-16\left\lceil C/8\right\rceil
  $$

<!-- end id8 -->

`H`、`R` 须同时满足：

$$
\boxed{
\begin{aligned}
13H &\le K,\\
3H+8R &\le K,\\
7H+6R &\le K.
\end{aligned}
}
$$

因此，给定 `C` 后，`H` 的上限为

$$
H_{\max}=16\left\lfloor\frac{K}{208}\right\rfloor.
$$

`R` 须同时满足上述关系式和 `R ≤ 640`。

其中，$\lceil\cdot\rceil$ 表示向上取整，$\lfloor\cdot\rfloor$ 表示向下取整。上述关系适用于 FLOAT16、BFLOAT16，以及所有合法的可选参数组合。

例如，`C=4、H=2560、R=320` 适用于上述三类产品。其他 `C/H/R` 组合须满足上述约束。调用时仍需通过第一段接口 `aclnnTextGatedResidualGradGetWorkspaceSize` 完成参数校验，并按返回的 `workspaceSize` 申请内存。

- 所有 FLOAT16/BFLOAT16 输入以及 `dx`、`dwDown`、`dwUp`、`dwGate` 的数据类型必须一致；各权重梯度与对应权重的数据类型相同。`invRms`、`gamma` 和 `dgamma` 为 FLOAT32。
- 支持非连续输入 Tensor。输入输出的 shape 必须符合参数表，同名维度的大小必须一致。
- 支持 `B=0` 或 `S=0`：`dx` 为空，已提供的非空权重梯度和 `dgamma` 全部置零，其余参数仍须满足上述约束。
- 输出梯度覆盖写入，调用前无需初始化，也不会累加到输出 Tensor 的原有数据中。
- 默认执行确定性计算。

可选参数的使用方式如下：

| 功能 | 提供时 | 缺省时 |
| :--- | :--- | :--- |
| 注入分支 | `wGate`、`gradInject`、`inject`、`dwGate` 四者必须同时提供 | 四者必须全部传入 `nullptr` |
| 归一化缩放 | `gamma`、`dgamma` 必须同时提供 | 两者必须全部传入 `nullptr`，缩放系数取 1 |

两组可选参数相互独立。`nullptr` 表示省略该功能，不能用 shape 为 `[0]` 的占位 Tensor 代替。

## 调用示例

以下示例使用 BFLOAT16，`B=1、S=2、C=4、H=16、R=16`，同时计算注入分支和 gamma 梯度。
输入 `x`、全部权重及 `gamma` 为 0，前向取 `epsilon=1e-6`，因此 `invRms=1000`、`downPreAct=0`、`up=0`、`inject=1`；两个上游梯度均为 1。
预期 `dx` 的所有元素为 125，`dwDown`、`dwUp`、`dwGate`、`dgamma` 的所有元素为 0。

编译与运行方式参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。请根据安装方式设置本算子的头文件和库搜索路径。

```cpp
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_text_gated_residual_grad.h"

namespace {
constexpr int32_t DEVICE_ID = 0;
constexpr int64_t B = 1;
constexpr int64_t S = 2;
constexpr int64_t C = 4;
constexpr int64_t H = 16;
constexpr int64_t R = 16;
constexpr int64_t D = C * H;
constexpr uint16_t BF16_ZERO = 0x0000;
constexpr uint16_t BF16_ONE = 0x3F80;
constexpr uint32_t BF16_SHIFT = 16;
constexpr float SAVED_INV_RMS = 1000.0F;

bool Check(int status, const char* action)
{
    if (status != ACL_SUCCESS) {
        std::printf("%s failed: %d\n", action, status);
        return false;
    }
    return true;
}

struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    std::vector<aclTensor*> tensors;
    std::vector<void*> buffers;

    ~Resources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (auto* tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto* buffer : buffers) {
            aclrtFree(buffer);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(DEVICE_ID);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};

template <typename T>
aclTensor* CreateTensor(Resources& resources, const std::vector<int64_t>& shape, aclDataType dtype, T value,
                        bool isInput, void** outputAddress = nullptr)
{
    int64_t count = 1;
    std::vector<int64_t> strides(shape.size());
    for (size_t axis = shape.size(); axis > 0; --axis) {
        strides[axis - 1] = count;
        count *= shape[axis - 1];
    }
    const size_t bytes = static_cast<size_t>(count) * sizeof(T);
    void* address = nullptr;
    if (!Check(aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) {
        return nullptr;
    }
    resources.buffers.push_back(address);
    if (isInput) {
        std::vector<T> host(static_cast<size_t>(count), value);
        if (!Check(aclrtMemcpy(address, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy")) {
            return nullptr;
        }
    }
    auto* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                   shape.size(), address);
    if (tensor == nullptr) {
        std::printf("aclCreateTensor failed\n");
        return nullptr;
    }
    resources.tensors.push_back(tensor);
    if (outputAddress != nullptr) {
        *outputAddress = address;
    }
    return tensor;
}
} // namespace

int main()
{
    Resources resources;
    if (!Check(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.initialized = true;
    if (!Check(aclrtSetDevice(DEVICE_ID), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    auto* gradMixed = CreateTensor(resources, {B, S, H}, ACL_BF16, BF16_ONE, true);
    auto* gradInject = CreateTensor(resources, {B, S, C}, ACL_BF16, BF16_ONE, true);
    auto* x = CreateTensor(resources, {B, S, D}, ACL_BF16, BF16_ZERO, true);
    auto* wDown = CreateTensor(resources, {R, D}, ACL_BF16, BF16_ZERO, true);
    auto* wUp = CreateTensor(resources, {D, R}, ACL_BF16, BF16_ZERO, true);
    auto* wGate = CreateTensor(resources, {C, D}, ACL_BF16, BF16_ZERO, true);
    auto* gamma = CreateTensor(resources, {D}, ACL_FLOAT, 0.0F, true);
    auto* invRms = CreateTensor(resources, {B, S, C}, ACL_FLOAT, SAVED_INV_RMS, true);
    auto* inject = CreateTensor(resources, {B, S, C}, ACL_BF16, BF16_ONE, true);
    auto* downPreAct = CreateTensor(resources, {B, S, R}, ACL_BF16, BF16_ZERO, true);
    auto* up = CreateTensor(resources, {B, S, D}, ACL_BF16, BF16_ZERO, true);

    void* dxAddress = nullptr;
    auto* dx = CreateTensor(resources, {B, S, D}, ACL_BF16, BF16_ZERO, false, &dxAddress);
    auto* dwDown = CreateTensor(resources, {R, D}, ACL_BF16, BF16_ZERO, false);
    auto* dwUp = CreateTensor(resources, {D, R}, ACL_BF16, BF16_ZERO, false);
    auto* dwGate = CreateTensor(resources, {C, D}, ACL_BF16, BF16_ZERO, false);
    auto* dgamma = CreateTensor(resources, {D}, ACL_FLOAT, 0.0F, false);
    for (auto* tensor : {gradMixed, gradInject, x, wDown, wUp, wGate, gamma, invRms, inject, downPreAct, up, dx, dwDown,
                         dwUp, dwGate, dgamma}) {
        if (tensor == nullptr) {
            return 1;
        }
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnTextGatedResidualGradGetWorkspaceSize(gradMixed, x, wDown, wUp, invRms, downPreAct, up,
                                                             gradInject, wGate, gamma, inject, C, dx, dwDown, dwUp,
                                                             dwGate, dgamma, &workspaceSize, &executor);
    if (!Check(status, "aclnnTextGatedResidualGradGetWorkspaceSize")) {
        return 1;
    }
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        if (!Check(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace")) {
            return 1;
        }
        resources.buffers.push_back(workspace);
    }
    if (!Check(aclnnTextGatedResidualGrad(workspace, workspaceSize, executor, resources.stream),
               "aclnnTextGatedResidualGrad") ||
        !Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
        return 1;
    }

    uint16_t first = 0;
    if (!Check(aclrtMemcpy(&first, sizeof(first), dxAddress, sizeof(first), ACL_MEMCPY_DEVICE_TO_HOST),
               "copy dx to host")) {
        return 1;
    }
    const uint32_t bits = static_cast<uint32_t>(first) << BF16_SHIFT;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    std::printf("dx[0] = %.1f (expected 125.0)\n", value);
    return 0;
}
```
