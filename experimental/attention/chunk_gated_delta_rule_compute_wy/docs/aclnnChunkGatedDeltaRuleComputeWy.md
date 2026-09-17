# aclnnChunkGatedDeltaRuleComputeWy

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

aclnnChunkGatedDeltaRuleComputeWy完成门控Delta网络prefill阶段chunk内的WY/UT前处理变换，一次性输出后续chunk扫描所需的q、k、w、u、g。记chunk长度为$C$（固定64），对每个$(b, h_v, chunk)$：

$$
a_i = \sum_{j \le i} g_j, \quad \gamma_i = \exp(a_i), \quad K^\beta_i = \beta_i K_i
$$

$$
\Lambda_{ij} = \exp(a_i - a_j) \ (i > j), \quad
A = -\mathrm{strictlower}\left((K^\beta K^\top) \odot \Lambda\right)
$$

$$
T = (I - A)^{-1}, \quad U = T (\beta V), \quad W = T (\gamma K^\beta)
$$

$A$严格下三角故幂零，$T$由倍增恒等式$(I-A)^{-1} = (I+A)(I+A^2)\cdots(I+A^{C/2})$在$\log_2 C$轮内求出。qKernelOut、kKernelOut为q、k由BTHD到BHTD的重排，不参与数学计算。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnChunkGatedDeltaRuleComputeWyGetWorkspaceSize`获取workspace大小和执行器，再调用`aclnnChunkGatedDeltaRuleComputeWy`执行计算。

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleComputeWyGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    int64_t          chunkSize,
    const aclTensor *qKernelOut,
    const aclTensor *kKernelOut,
    const aclTensor *wKernelOut,
    const aclTensor *uKernelOut,
    const aclTensor *gKernelOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleComputeWy(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnChunkGatedDeltaRuleComputeWyGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px"><col style="width: 120px"><col style="width: 260px">
  <col style="width: 380px"><col style="width: 180px"><col style="width: 100px">
  <col style="width: 180px"><col style="width: 100px">
  </colgroup><thead><tr>
  <th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th>
  <th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th>
  </tr></thead><tbody>
  <tr><td>q（aclTensor*）</td><td>输入</td><td>query。</td><td>不支持空Tensor。</td><td>FLOAT16</td><td>ND</td><td>(B,T,Hk,K)</td><td>√</td></tr>
  <tr><td>k（aclTensor*）</td><td>输入</td><td>key。</td><td>不支持空Tensor，shape与q一致。</td><td>FLOAT16</td><td>ND</td><td>(B,T,Hk,K)</td><td>√</td></tr>
  <tr><td>v（aclTensor*）</td><td>输入</td><td>value。</td><td>不支持空Tensor。</td><td>FLOAT16</td><td>ND</td><td>(B,T,Hv,V)</td><td>√</td></tr>
  <tr><td>g（aclTensor*）</td><td>输入</td><td>门控对数衰减。</td><td>不支持空Tensor，各元素须非正。</td><td>FLOAT32</td><td>ND</td><td>(B,T,Hv)</td><td>√</td></tr>
  <tr><td>beta（aclTensor*）</td><td>输入</td><td>delta规则更新步长。</td><td>不支持空Tensor。</td><td>FLOAT16</td><td>ND</td><td>(B,T,Hv)</td><td>√</td></tr>
  <tr><td>chunkSize（int64_t）</td><td>输入</td><td>chunk长度。</td><td>当前仅支持64。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>qKernelOut（aclTensor*）</td><td>输出</td><td>q的BHTD重排结果。</td><td>不支持空Tensor。</td><td>FLOAT16</td><td>ND</td><td>(B,Hk,T,K)</td><td>√</td></tr>
  <tr><td>kKernelOut（aclTensor*）</td><td>输出</td><td>k的BHTD重排结果。</td><td>不支持空Tensor。</td><td>FLOAT16</td><td>ND</td><td>(B,Hk,T,K)</td><td>√</td></tr>
  <tr><td>wKernelOut（aclTensor*）</td><td>输出</td><td>WY变换的W。</td><td>不支持空Tensor。</td><td>FLOAT16</td><td>ND</td><td>(B,Hv,T,K)</td><td>√</td></tr>
  <tr><td>uKernelOut（aclTensor*）</td><td>输出</td><td>WY变换的U。</td><td>不支持空Tensor。</td><td>FLOAT16</td><td>ND</td><td>(B,Hv,T,V)</td><td>√</td></tr>
  <tr><td>gKernelOut（aclTensor*）</td><td>输出</td><td>chunk内g的前缀和。</td><td>不支持空Tensor。</td><td>FLOAT32</td><td>ND</td><td>(B,Hv,T)</td><td>√</td></tr>
  <tr><td>workspaceSize（uint64_t*）</td><td>输出</td><td>返回Device侧workspace大小。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>executor（aclOpExecutor**）</td><td>输出</td><td>返回op执行器。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  | 返回值 | 错误码 | 描述 |
  | :--- | :---: | :--- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选Tensor、workspaceSize或executor为空指针。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | chunkSize不为64，或输入数据类型、shape不满足约束。 |

## aclnnChunkGatedDeltaRuleComputeWy

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

- `chunkSize`仅支持64，且要求`T % 64 == 0`。
- `Hv % Hk == 0`，`B <= 32`，`Hv <= 64`。
- `K`、`V`需为16的整数倍且不大于128，上界由<term>Atlas 推理系列产品</term>的192KB UB容量决定。
- g为对数衰减，须非正；否则`exp`上溢，结果无意义。
- aclnnChunkGatedDeltaRuleComputeWy默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include "aclnnop/aclnn_chunk_gated_delta_rule_compute_wy.h"

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnChunkGatedDeltaRuleComputeWyGetWorkspaceSize(
    q, k, v, g, beta, chunkSize,
    qKernelOut, kKernelOut, wKernelOut, uKernelOut, gKernelOut,
    &workspaceSize, &executor);
if (ret == ACLNN_SUCCESS) {
    ret = aclnnChunkGatedDeltaRuleComputeWy(workspace, workspaceSize, executor, stream);
}
```

完整样例参见[test_aclnn_chunk_gated_delta_rule_compute_wy.cpp](../examples/test_aclnn_chunk_gated_delta_rule_compute_wy.cpp)。
