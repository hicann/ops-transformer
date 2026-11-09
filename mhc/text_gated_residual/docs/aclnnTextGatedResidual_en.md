# aclnnTextGatedResidual

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Description

- API function: Fuses grouped RMSNorm, low-rank down/SiLU/up projections, sigmoid mixing, and optional inject to produce mixed. Also returns invRmsOptional, downPreActOptional, and upPreActOptional when needBackward=true.

- Formulas:

### Grouped RMS Normalization

The input `x` has shape $(B,S,C\times H)$ and contains channel elements $x_{b,s,c,h}$. Grouped RMS normalization is defined as follows:

$$
invRms_{b,s,c}=\frac{1}{\sqrt{\frac{1}{H}\sum_{j=0}^{H-1}x_{b,s,c,j}^{2}+\epsilon}}
$$

$$
xNorm_{b,s,c,h}=x_{b,s,c,h}\cdot invRms_{b,s,c}\cdot(1+\gamma_{cH+h})
$$

The scale factor is 1 when `gamma` is absent. Do not add 1 to gamma before supplying it.

### Read Fusion Path

Flatten $xNorm$ to $(B,S,C\times H)$ and compute:

$$
downPreAct_{b,s,r}=\sum_{d=0}^{C\times H-1}xNorm_{b,s,d}\,wDown_{r,d}
$$

$$
gate_{b,s,c,h}=\operatorname{sigmoid}\left(\sum_{r=0}^{R-1}\operatorname{SiLU}\left(\frac{downPreAct_{b,s,r}}{C}\right)\,wUp_{c\times H+h,r}\right)
$$

$$
mixed_{b,s,h}=\frac{1}{C}\sum_{c=0}^{C-1}gate_{b,s,c,h}\,xNorm_{b,s,c,h}
$$

### Backward Caches and Usage

Let D=C×H. The upPreActOptional cache is defined as follows:

$$
upPreAct_{b,s,d}=\sum_{r=0}^{R-1}\operatorname{SiLU}\left(\frac{downPreAct_{b,s,r}}{C}\right)\,wUp_{d,r}
$$

downPreActOptional contains the down-projection result before division by C. upPreActOptional contains the up-projection result before sigmoid. Both have the same dtype as x.
When needBackward=true, provide invRmsOptional, downPreActOptional, and upPreActOptional, regardless of whether wGateOptional is present. When needBackward=false, these outputs accept nullptr. Their contents are invalid and must not be read.

### Injection Path

When `wGateOptional` is not null, the injection path is defined as follows:

$$
inject_{b,s,c}=2\cdot\operatorname{sigmoid}\left(\frac{1}{C}\sum_{d=0}^{C\times H-1}xNorm_{b,s,d}\,wGate_{c,d}\right)
$$

### Symbols

Here, $B,S$ denote the batch size and sequence length. $C$, $H$, and $R$ denote the channel count, per-channel hidden dimension, and bottleneck dimension. The optional `gamma` contains original weights with shape $(C\times H)$; its scale factor is $1+\gamma_{cH+h}$.

When `needBackward=true`, the forward operation also returns `invRmsOptional`, `downPreActOptional`, and `upPreActOptional` for backward computation. When `wGateOptional` is not null, it returns `injectOptional`. The backward operation recovers injection gate values from `injectOptional / 2`.

## Prototype

The operator uses a [two-phase API](../../../docs/zh/context/two_phase_api.md). First call "aclnnTextGatedResidualGetWorkspaceSize" to obtain the workspace size and executor. Then call "aclnnTextGatedResidual" to execute the operator.

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

### Parameters

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
  <th>Parameter</th>
  <th>Input/Output</th>
  <th>Description</th>
  <th>Precautions</th>
  <th>Data Type</th>
  <th>Format</th>
  <th>Shape</th>
  <th>Non-contiguous Tensor</th>
</tr>
</thead>
<tbody>
<tr>
  <td>x（aclTensor*）</td>
  <td>Input</td>
  <td>Forward input tensor.</td>
  <td>Shape [B,S,C×H].</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>wDown（aclTensor*）</td>
  <td>Input</td>
  <td>Down-projection weights for the read fusion path.</td>
  <td>Shape [R,C×H].</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>(R,C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>wUp（aclTensor*）</td>
  <td>Input</td>
  <td>Up-projection weights for the read fusion path.</td>
  <td>Shape [C×H,R].</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>(C×H,R)</td>
  <td>√</td>
</tr>
<tr>
  <td>wGateOptional（aclTensor*）</td>
  <td>Optional input</td>
  <td>Projection weights for the injection path.</td>
  <td>Shape [C,C×H]. When supplied, this input enables inject and requires injectOptional. The injection path is skipped when this input is absent.</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>(C,C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>gammaOptional（aclTensor*）</td>
  <td>Optional input</td>
  <td>Grouped RMS weights.</td>
  <td>Original element-wise gamma[C×H] weights. The scale factor is 1+gamma, broadcast over the batch and sequence dimensions.</td>
  <td>FLOAT32</td>
  <td>ND</td>
  <td>(C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>channels（int64_t）</td>
  <td>Input</td>
  <td>Channel count C.</td>
  <td>Number of channels C in the last input dimension.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
<tr>
  <td>eps（double）</td>
  <td>Optional attribute</td>
  <td>RMS numerical stability parameter.</td>
  <td>Optional attribute with a default of 1e-6. The value must be finite, non-negative, and representable as FLOAT.</td>
  <td>DOUBLE</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
<tr>
  <td>needBackward（bool）</td>
  <td>Optional attribute</td>
  <td>Whether to return forward intermediate results required for backward computation.</td>
  <td>Optional attribute with a default of false. When true, invRms, downPreAct, and upPreAct are required. When false, all three may be null.</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
<tr>
  <td>mixed（aclTensor*）</td>
  <td>Output</td>
  <td>Mixed result after weighted averaging over the channel dimension.</td>
  <td>Shape [B,S,H], with the same dtype as x. Empty outputs are supported when B or S is 0.</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,H)</td>
  <td>√</td>
</tr>
<tr>
  <td>injectOptional（aclTensor*）</td>
  <td>Optional output</td>
  <td>Injection weights generated by the injection gate.</td>
  <td>Shape [B,S,C] when wGateOptional is not null. May be null when wGateOptional is absent.</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,C)</td>
  <td>√</td>
</tr>
<tr>
  <td>invRmsOptional（aclTensor*）</td>
  <td>Optional output</td>
  <td>Per-group inverse RMS values required for RMS backward computation.</td>
  <td>Saved when needBackward=true. Shape [B,S,C].</td>
  <td>FLOAT32</td>
  <td>ND</td>
  <td>(B,S,C)</td>
  <td>√</td>
</tr>
<tr>
  <td>downPreActOptional（aclTensor*）</td>
  <td>Optional output</td>
  <td>SiLU input from the down projection, used to reconstruct the SiLU derivative.</td>
  <td>Saved when needBackward=true. Shape [B,S,R].</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,R)</td>
  <td>√</td>
</tr>
<tr>
  <td>upPreActOptional（aclTensor*）</td>
  <td>Optional output</td>
  <td>Up-matmul output of the mix branch, which is the sigmoid input. Used to reconstruct the mix gate during backward computation.</td>
  <td>Required when needBackward=true; otherwise may be null. Shape [B,S,C×H], with the same dtype as x: FLOAT16 or BFLOAT16.</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>(B,S,C×H)</td>
  <td>√</td>
</tr>
<tr>
  <td>workspaceSize（uint64_t*）</td>
  <td>Output</td>
  <td>Returns the required device workspace size.</td>
  <td>The first-phase API returns the workspace size and executor. Allocate the workspace according to the returned size.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
<tr>
  <td>executor（aclOpExecutor**）</td>
  <td>Output</td>
  <td>Returns the executor containing the operator execution procedure.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
</tr>
</tbody>
</table>

### Return Value

aclnnStatus: Returns a status code. For details, see [aclnn return codes](../../../docs/zh/context/aclnn_return_code.md).

The first-phase API validates arguments and reports errors in the following cases:

<table style="undefined;table-layout: fixed; width: 1232px">
<colgroup>
  <col style="width: 330px">
  <col style="width: 140px">
  <col style="width: 762px">
</colgroup>
<thead>
<tr>
  <th>Return Value</th>
  <th>Error Code</th>
  <th>Description</th>
</tr>
</thead>
<tbody>
<tr>
  <td>ACLNN_ERR_PARAM_NULLPTR</td>
  <td>161001</td>
  <td>x, wDown, wUp, or mixed is null; injectOptional is null when wGateOptional is supplied; or a required cache is null when needBackward=true.</td>
</tr>
<tr>
  <td>ACLNN_ERR_PARAM_INVALID</td>
  <td>161002</td>
  <td>x, wDown, wUp, wGate, mixed, inject, and downPreAct support FLOAT16/BFLOAT16 with matching computational dtypes. gamma and invRms use FLOAT32; upPreAct matches x. Shapes or ranks do not match, or the format is not ND.</td>
</tr>
<tr>
  <td>ACLNN_ERR_PARAM_INVALID</td>
  <td>161002</td>
  <td>eps is non-finite, negative, or outside the FLOAT range, or channels does not match the x/wDown/wUp/wGate shapes.</td>
</tr>
<tr>
  <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
  <td>561101</td>
  <td>The API fails to create an aclOpExecutor.</td>
</tr>
<tr>
  <td>ACLNN_ERR_INNER_NULLPTR</td>
  <td>561103</td>
  <td>The API returns a null pointer when creating a node or output.</td>
</tr>
<tr>
  <td>ACLNN_ERR_RUNTIME_ERROR</td>
  <td>361001</td>
  <td>An NPU Runtime API call fails.</td>
</tr>
</tbody>
</table>

## aclnnTextGatedResidual

### Parameters

<table style="undefined;table-layout: fixed; width: 1148px">
<colgroup>
  <col style="width: 170px">
  <col style="width: 134px">
  <col style="width: 844px">
</colgroup>
<thead>
<tr>
  <th>Parameter</th>
  <th>Input/Output</th>
  <th>Description</th>
</tr>
</thead>
<tbody>
<tr>
  <td>workspace</td>
  <td>Input</td>
  <td>Device workspace address.</td>
</tr>
<tr>
  <td>workspaceSize</td>
  <td>Input</td>
  <td>Workspace size obtained from the first-phase API.</td>
</tr>
<tr>
  <td>executor</td>
  <td>Input</td>
  <td>Operator executor returned by the first-phase API.</td>
</tr>
<tr>
  <td>stream</td>
  <td>Input</td>
  <td>Stream used to execute the task.</td>
</tr>
</tbody>
</table>

### Return Value

aclnnStatus: Returns a status code. For details, see [aclnn return codes](../../../docs/zh/context/aclnn_return_code.md).

## Constraints

### Deterministic Computation

- aclnnTextGatedResidual uses the default deterministic implementation.

### Batch Consistency

- aclnnTextGatedResidual uses the default batch consistency implementation.

### Specifications

- Input limits: C is an integer with 2≤C≤10. The per-channel hidden dimension satisfies 256≤H≤4096 and H%256=0. The low-rank dimension satisfies 64≤R≤896 and R%64=0.
- x has shape [B,S,D], where D=C×H and H=D/C. wDown and wUp have shapes [R,D] and [D,R], respectively. Optional wGateOptional and gammaOptional have shapes [C,D] and [D].
- Computational input and output tensors support FLOAT16 and BFLOAT16. gamma and invRms use FLOAT32; upPreAct has the same dtype as x.
- When `needBackward` is false, backward cache outputs may be null. When true, provide `invRmsOptional`, `downPreActOptional`, and `upPreActOptional`.

## Examples

The following sample uses FLOAT16 inputs, enables inject, and disables backward caches.

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
