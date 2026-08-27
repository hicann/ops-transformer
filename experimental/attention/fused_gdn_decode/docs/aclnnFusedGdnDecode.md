# aclnnFusedGdnDecode

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

aclnnFusedGdnDecode完成门控Delta网络单token解码计算，融合QKV拆分、Q/K归一化、门控计算、循环状态更新和输出投影。

$$
q_h = scale \times \frac{q_h}{\sqrt{\sum_i q_{h,i}^2+\epsilon}}, \quad
k_h = \frac{k_h}{\sqrt{\sum_i k_{h,i}^2+\epsilon}}
$$

$$
g_j = -\exp(A_j)\times softplus(a_j+dtBias_j), \quad
\beta_j = sigmoid(b_j)
$$

$$
S_j = \exp(g_j)S_j+\beta_j(v_j-\exp(g_j)S_jk_h)k_h^T, \quad
out_j = S_jq_h
$$

其中，$h=\lfloor j/(H_v/H)\rfloor$，`stateRef`原地保存更新后的$S_j$。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnFusedGdnDecodeGetWorkspaceSize`获取workspace大小和执行器，再调用`aclnnFusedGdnDecode`执行计算。

```cpp
aclnnStatus aclnnFusedGdnDecodeGetWorkspaceSize(
    const aclTensor *mixedQkv,
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *aLog,
    const aclTensor *dtBias,
    aclTensor       *stateRef,
    const aclTensor *ssmStateIndices,
    float            scale,
    float            softplusThreshold,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor);
```

```cpp
aclnnStatus aclnnFusedGdnDecode(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnFusedGdnDecodeGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px"><col style="width: 120px"><col style="width: 260px">
  <col style="width: 380px"><col style="width: 180px"><col style="width: 100px">
  <col style="width: 180px"><col style="width: 100px">
  </colgroup><thead><tr>
  <th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th>
  <th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th>
  </tr></thead><tbody>
  <tr><td>mixedQkv（aclTensor*）</td><td>输入</td><td>按Q、K、V顺序拼接的输入。</td><td>不支持空Tensor。</td><td>BFLOAT16、FLOAT16</td><td>ND</td><td>(B,2*H*K+Hv*V)</td><td>√</td></tr>
  <tr><td>a（aclTensor*）</td><td>输入</td><td>门控输入a。</td><td>不支持空Tensor，数据类型与mixedQkv一致。</td><td>BFLOAT16、FLOAT16</td><td>ND</td><td>(B,Hv)</td><td>√</td></tr>
  <tr><td>b（aclTensor*）</td><td>输入</td><td>门控输入b。</td><td>不支持空Tensor，数据类型与mixedQkv一致。</td><td>BFLOAT16、FLOAT16</td><td>ND</td><td>(B,Hv)</td><td>√</td></tr>
  <tr><td>aLog（aclTensor*）</td><td>输入</td><td>门控参数A。</td><td>不支持空Tensor。</td><td>FLOAT32</td><td>ND</td><td>(Hv,)</td><td>√</td></tr>
  <tr><td>dtBias（aclTensor*）</td><td>输入</td><td>softplus偏置。</td><td>不支持空Tensor，数据类型与mixedQkv一致。</td><td>BFLOAT16、FLOAT16</td><td>ND</td><td>(Hv,)</td><td>√</td></tr>
  <tr><td>stateRef（aclTensor*）</td><td>输入&输出</td><td>循环状态矩阵。</td><td>不支持空Tensor，必须为连续Tensor。</td><td>FLOAT32、BFLOAT16、FLOAT16</td><td>ND</td><td>(BlockNum,Hv,V,K)</td><td>×</td></tr>
  <tr><td>ssmStateIndices（aclTensor*）</td><td>输入</td><td>batch到stateRef槽位的映射。</td><td>不支持空Tensor。</td><td>INT32</td><td>ND</td><td>(B,)</td><td>√</td></tr>
  <tr><td>scale（float）</td><td>输入</td><td>Q归一化后的缩放系数。</td><td>必须为有限值。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>softplusThreshold（float）</td><td>输入</td><td>softplus阈值。</td><td>必须为有限值。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>out（aclTensor*）</td><td>输出</td><td>GDN输出。</td><td>不支持空Tensor，数据类型与mixedQkv一致。</td><td>BFLOAT16、FLOAT16</td><td>ND</td><td>(B,1,Hv,V)</td><td>√</td></tr>
  <tr><td>workspaceSize（uint64_t*）</td><td>输出</td><td>返回Device侧workspace大小。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>executor（aclOpExecutor**）</td><td>输出</td><td>返回op执行器。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  | 返回值 | 错误码 | 描述 |
  | :--- | :---: | :--- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选Tensor、workspaceSize或executor为空指针。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | 输入数据类型、shape、连续性或属性不满足约束。 |

## aclnnFusedGdnDecode

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  | :--- | :--- | :--- |
  | workspace | 输入 | Device侧workspace内存地址。 |
  | workspaceSize | 输入 | 第一段接口返回的workspace大小。 |
  | executor | 输入 | 第一段接口返回的op执行器。 |
  | stream | 输入 | 执行任务的Stream。 |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- `64 <= K <= 2032`，`Hv % H == 0`，所有维度均为正数。
- stateRef为FLOAT32时，需满足`ceil(K/16)*16-K <= 8`。
- stateRef支持FLOAT32，或与mixedQkv相同的数据类型。
- `ssmStateIndices[i] <= 0`表示无效槽位，对应输出为0且stateRef不更新。
- `ssmStateIndices[i] > 0`时，用户需保证`ssmStateIndices[i] < BlockNum`，且同一批次内的正索引互不重复。
- aclnnFusedGdnDecode默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include "aclnnop/aclnn_fused_gdn_decode.h"

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnFusedGdnDecodeGetWorkspaceSize(
    mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices,
    scale, softplusThreshold, out, &workspaceSize, &executor);
if (ret == ACLNN_SUCCESS) {
    ret = aclnnFusedGdnDecode(workspace, workspaceSize, executor, stream);
}
```
