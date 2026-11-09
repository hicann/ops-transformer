# aclnnTextGatedResidual

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

- 接口功能：融合分组 RMSNorm、低秩 down/SiLU/up、sigmoid 混合及可选 inject，输出 mixed；needBackward=true 时同时输出 invRmsOptional、downPreActOptional、upPreActOptional。

- 计算公式：

### 分组 RMS 归一化

设输入 `x` 的 shape 为 $(B,S,C\times H)$，按通道拆分为 $x_{b,s,c,h}$。分组 RMS 归一化定义为：

$$
invRms_{b,s,c}=\frac{1}{\sqrt{\frac{1}{H}\sum_{j=0}^{H-1}x_{b,s,c,j}^{2}+\epsilon}}
$$

$$
xNorm_{b,s,c,h}=x_{b,s,c,h}\cdot invRms_{b,s,c}\cdot(1+\gamma_{cH+h})
$$

未提供 `gamma` 时缩放因子为 1；传入的 gamma 不应提前加 1。

### 读融合路径

将 $xNorm$ 展平为 $(B,S,C\times H)$ 后计算：

$$
downPreAct_{b,s,r}=\sum_{d=0}^{C\times H-1}xNorm_{b,s,d}\,wDown_{r,d}
$$

$$
gate_{b,s,c,h}=\operatorname{sigmoid}\left(\sum_{r=0}^{R-1}\operatorname{SiLU}\left(\frac{downPreAct_{b,s,r}}{C}\right)\,wUp_{c\times H+h,r}\right)
$$

$$
mixed_{b,s,h}=\frac{1}{C}\sum_{c=0}^{C-1}gate_{b,s,c,h}\,xNorm_{b,s,c,h}
$$

### 反向缓存和使用方式

定义 D=C×H，upPreActOptional 缓存为：

$$
upPreAct_{b,s,d}=\sum_{r=0}^{R-1}\operatorname{SiLU}\left(\frac{downPreAct_{b,s,r}}{C}\right)\,wUp_{d,r}
$$

downPreActOptional 为未除以 C 的下投影结果，upPreActOptional 为 sigmoid 之前的上投影结果，二者 dtype 均与 x 一致。
needBackward=true 时必须提供 invRmsOptional、downPreActOptional 和 upPreActOptional；该要求不受 wGateOptional 是否传入影响。needBackward=false 时三个缓存输出可传入 nullptr，其内容无效，不应读取。

### 注入路径

当 `wGateOptional` 非空 时，注入路径为：

$$
inject_{b,s,c}=2\cdot\operatorname{sigmoid}\left(\frac{1}{C}\sum_{d=0}^{C\times H-1}xNorm_{b,s,d}\,wGate_{c,d}\right)
$$

### 符号说明

其中 $B,S$ 为 batch 和序列长度，$C$ 为通道数，$H$ 为每通道隐藏维度，$R$ 为瓶颈维度；`gamma` 为 shape $(C\times H)$ 的可选原始权重，缩放因子为 $1+\gamma_{cH+h}$。

`needBackward=true` 时，前向另输出 `invRmsOptional`、`downPreActOptional` 和 `upPreActOptional` 供反向计算；`wGateOptional` 非空 时输出 `injectOptional`，反向由 `injectOptional / 2` 还原注入门控值。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnTextGatedResidualGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnTextGatedResidual"接口执行计算。

```cpp
aclnnStatus aclnnTextGatedResidualGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *wDown,
    const aclTensor *wUp,
    const aclTensor *wGateOptional,
    const aclTensor *gammaOptional,
    int64_t          channels,
    double           eps,
    bool             needBackward,
    aclTensor       *mixed,
    aclTensor       *injectOptional,
    aclTensor       *invRmsOptional,
    aclTensor       *downPreActOptional,
    aclTensor       *upPreActOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnTextGatedResidual(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnTextGatedResidualGetWorkspaceSize

### 参数说明

<table style="undefined;table-layout: fixed; width: 1550px">
<colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 140px">
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
  <td>x（aclTensor*）</td>
  <td>输入</td>
  <td>前向输入张量。</td>
  <td>shape 为 [B,S,C×H]。</td>
  <td>FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>wDown（aclTensor*）</td>
  <td>输入</td>
  <td>读融合路径降维投影权重。</td>
  <td>shape 为 [R,C×H]。</td>
  <td>FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>(R,C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>wUp（aclTensor*）</td>
  <td>输入</td>
  <td>读融合路径升维投影权重。</td>
  <td>shape 为 [C×H,R]。</td>
  <td>FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>(C×H,R)</td>
  <td>√</td>
</tr>
<tr>
  <td>wGateOptional（aclTensor*）</td>
  <td>可选输入</td>
  <td>Injection 路径投影权重。</td>
  <td>传入时计算 inject，且 injectOptional 必须提供；省略时跳过；shape 为 [C,C×H]。</td>
  <td>FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>(C,C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>gammaOptional（aclTensor*）</td>
  <td>可选输入</td>
  <td>Grouped RMS 权重。</td>
  <td>使用逐元素 gamma[C×H] 原始权重，缩放因子为 1+gamma，沿 batch 和 sequence 维广播。</td>
  <td>FLOAT32</td>
  <td>ND</td>
  <td>(C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>channels（int64_t）</td>
  <td>输入</td>
  <td>通道数 C。</td>
  <td>指定输入末维拆分出的通道数 C。</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
<tr>
  <td>eps（double）</td>
  <td>可选属性</td>
  <td>RMS 数值稳定参数。</td>
  <td>可选属性，默认 1e-6；须为非负有限值且可表示为 FLOAT。</td>
  <td>DOUBLE</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
<tr>
  <td>needBackward（bool）</td>
  <td>可选属性</td>
  <td>是否返回反向计算需要保存的前向中间量。</td>
  <td>可选属性，默认 false；为 true 时 invRms、downPreAct、upPreAct 必须提供，为 false 时均可为空指针。</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
<tr>
  <td>mixed（aclTensor*）</td>
  <td>输出</td>
  <td>channel 维度加权平均后的混合结果。</td>
  <td>shape 为 [B,S,H]；B 或 S 为 0 时支持空输出，dtype 与 x 相同。</td>
  <td>FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,H)</td>
  <td>√</td>
</tr>
<tr>
  <td>injectOptional（aclTensor*）</td>
  <td>可选输出</td>
  <td>由 injection gate 生成的注入权重。</td>
  <td>wGateOptional 非空时 shape 为 [B,S,C]；wGateOptional 为空时可为空指针。</td>
  <td>FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,C)</td>
  <td>√</td>
</tr>
<tr>
  <td>invRmsOptional（aclTensor*）</td>
  <td>可选输出</td>
  <td>RMS 反向需要的逐组 inverse RMS。</td>
  <td>needBackward=true 时保存；shape 为 [B,S,C]。</td>
  <td>FLOAT32</td>
  <td>ND</td>
  <td>(B,S,C)</td>
  <td>√</td>
</tr>
<tr>
  <td>downPreActOptional（aclTensor*）</td>
  <td>可选输出</td>
  <td>down projection 的 SiLU 输入，用于重建 SiLU 导数。</td>
  <td>needBackward=true 时保存；shape 为 [B,S,R]。</td>
  <td>FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,R)</td>
  <td>√</td>
</tr>
<tr>
  <td>upPreActOptional（aclTensor*）</td>
  <td>可选输出</td>
  <td>mix 分支 up matmul 的输出，即 sigmoid 输入，用于反向重建 mix gate。</td>
  <td>needBackward=true 时必须提供；shape 为 [B,S,C×H]，dtype 与 x 相同，支持 FLOAT16、BFLOAT16；false 时可为空指针。</td>
  <td>FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>workspaceSize（uint64_t*）</td>
  <td>输出</td>
  <td>返回 Device 侧所需 workspace 大小。</td>
  <td>由第一段接口返回 workspace 大小与 executor；调用侧据返回值申请 workspace。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
<tr>
  <td>executor（aclOpExecutor**）</td>
  <td>输出</td>
  <td>返回包含算子执行流程的 executor。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
</tbody>
</table>

### 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1232px">
<colgroup>
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
  <td>ACLNN_ERR_PARAM_NULLPTR</td>
  <td>161001</td>
  <td>x、wDown、wUp 或 mixed 为空指针；wGateOptional 非空时 injectOptional 为空指针；needBackward=true 时 invRmsOptional、downPreActOptional 或 upPreActOptional 为空指针。</td>
</tr>
<tr>
  <td>ACLNN_ERR_PARAM_INVALID</td>
  <td>161002</td>
  <td>x、wDown、wUp、wGate、mixed、inject、downPreAct 支持 FLOAT16/BFLOAT16 且计算张量 dtype 一致；gamma、invRms 为 FLOAT32；upPreAct 与 x dtype 相同；shape/rank 不匹配或数据格式不为 ND。</td>
</tr>
<tr>
  <td>ACLNN_ERR_PARAM_INVALID</td>
  <td>161002</td>
  <td>eps 非有限、小于 0 或超出 FLOAT 可表示范围，或 channels 与 x/wDown/wUp/wGate shape 不匹配。</td>
</tr>
<tr>
  <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
  <td>561101</td>
  <td>接口内部创建aclOpExecutor失败。</td>
</tr>
<tr>
  <td>ACLNN_ERR_INNER_NULLPTR</td>
  <td>561103</td>
  <td>接口内部创建节点或输出时返回空指针。</td>
</tr>
<tr>
  <td>ACLNN_ERR_RUNTIME_ERROR</td>
  <td>361001</td>
  <td>接口内部调用NPU Runtime接口异常。</td>
</tr>
</tbody>
</table>

## aclnnTextGatedResidual

### 参数说明

<table style="undefined;table-layout: fixed; width: 1148px">
<colgroup>
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
  <td>Device 侧 workspace 内存地址。</td>
</tr>
<tr>
  <td>workspaceSize</td>
  <td>输入</td>
  <td>workspace 大小，由第一段接口获取。</td>
</tr>
<tr>
  <td>executor</td>
  <td>输入</td>
  <td>第一段接口返回的算子执行器。</td>
</tr>
<tr>
  <td>stream</td>
  <td>输入</td>
  <td>指定执行任务的 Stream。</td>
</tr>
</tbody>
</table>

### 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 确定性计算

- aclnnTextGatedResidual采用默认确定性实现。

### Batch一致性说明

- aclnnTextGatedResidual采用默认Batch一致性实现。

### 规格约束

- 输入限制：通道数 C 为整数，满足 2≤C≤10；每通道隐藏维度 H 满足 256≤H≤4096 且 H%256=0；低秩维度 R 满足 64≤R≤896 且 R%64=0。
- 输入 x 的 shape 为 [B,S,D]，D=C×H，H=D/C；wDown、wUp 的 shape 分别为 [R,D]、[D,R]，可选 wGateOptional 的 shape 为 [C,D]，可选 gammaOptional 的 shape 为 [D]。
- 输入和输出计算张量支持FLOAT16、BFLOAT16；gamma和invRms为FLOAT32；upPreAct与x相同。
- `needBackward`为false时反向缓存输出可为空指针；为true时必须提供`invRmsOptional`、`downPreActOptional`和`upPreActOptional`。

## 调用示例

以下示例使用 FLOAT16 输入，开启 inject，关闭反向缓存。

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_text_gated_residual.h"

// FP16 zero inputs: mixed and inject must both be zero. No backward cache requested.
namespace {
constexpr int64_t EXAMPLE_SEQUENCE = 32;
constexpr int64_t EXAMPLE_CHANNELS = 4;
constexpr int64_t EXAMPLE_HIDDEN = 512;
constexpr size_t MIXED_ALLOCATION_INDEX = 4;
} // namespace

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS)
        return 1;
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS)
        return 1;
    std::vector<aclTensor*> tensors;
    std::vector<void*> allocations;
    auto tensor = [&](std::vector<int64_t> shape) -> aclTensor* {
        std::vector<int64_t> strides(shape.size());
        int64_t elements = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = elements;
            elements *= shape[i];
        }
        void* data = nullptr;
        if (aclrtMalloc(&data, elements * 2, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
            return nullptr;
        allocations.push_back(data);
        std::vector<uint16_t> zeros(elements, 0);
        if (aclrtMemcpy(data, elements * 2, zeros.data(), elements * 2, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
            return nullptr;
        auto* t = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(), 0, ACL_FORMAT_ND,
                                  shape.data(), shape.size(), data);
        if (t)
            tensors.push_back(t);
        return t;
    };
    auto* x = tensor({1, 32, 2048});
    auto* down = tensor({64, 2048});
    auto* up = tensor({2048, 64});
    auto* gate = tensor({4, 2048});
    auto* mixed = tensor({1, 32, 512});
    auto* inject = tensor({1, 32, 4});
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    int result = 1;
    void* workspace = nullptr;
    if (x && down && up && gate && mixed && inject &&
        aclnnTextGatedResidualGetWorkspaceSize(x, down, up, gate, nullptr, EXAMPLE_CHANNELS, 1e-6, false, mixed, inject,
                                               nullptr, nullptr, nullptr, &bytes, &executor) == ACL_SUCCESS &&
        (bytes == 0 || aclrtMalloc(&workspace, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS) &&
        aclnnTextGatedResidual(workspace, bytes, executor, stream) == ACL_SUCCESS &&
        aclrtSynchronizeStream(stream) == ACL_SUCCESS) {
        std::vector<uint16_t> output(EXAMPLE_SEQUENCE * EXAMPLE_HIDDEN);
        if (aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), allocations[MIXED_ALLOCATION_INDEX],
                        output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
            result = 0;
            for (auto value : output)
                if ((value & 0x7fff) != 0)
                    result = 1;
        }
    }
    if (workspace)
        aclrtFree(workspace);
    for (auto* t : tensors)
        aclDestroyTensor(t);
    for (auto* p : allocations)
        aclrtFree(p);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    std::printf("TextGatedResidual example: %s\n", result == 0 ? "PASS" : "FAIL");
    return result;
}
```
