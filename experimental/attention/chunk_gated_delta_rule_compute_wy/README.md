# ChunkGatedDeltaRuleComputeWy

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

- 算子功能：门控Delta网络（Gated Delta Network，GDN）prefill阶段chunk内的WY/UT前处理变换，一次性输出后续chunk扫描所需的全部kernel输入（q、k、w、u、g）。该变换在Qwen3.5等GDN模型上原本由框架侧的torch算子序列完成，是prefill流程的主要瓶颈。
- 计算公式：记chunk长度为$C$（固定64），对每个$(b, h_v, chunk)$，

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

  其中$A$为严格下三角阵，故幂零（$A^C = 0$），$T$通过倍增恒等式
  $(I-A)^{-1} = (I+A)(I+A^2)(I+A^4)\cdots(I+A^{C/2})$在$\log_2 C$轮内求出，无需逐行前代。
  $q\_kernel$、$k\_kernel$为$q$、$k$的BTHD到BHTD重排，不参与数学计算。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| q | 输入 | query，shape为`(B, T, Hk, K)`。 | FLOAT16 | ND |
| k | 输入 | key，shape为`(B, T, Hk, K)`，与q一致。 | FLOAT16 | ND |
| v | 输入 | value，shape为`(B, T, Hv, V)`。 | FLOAT16 | ND |
| g | 输入 | 门控对数衰减，shape为`(B, T, Hv)`，要求各元素非正。 | FLOAT32 | ND |
| beta | 输入 | delta规则更新步长，shape为`(B, T, Hv)`。 | FLOAT16 | ND |
| chunk_size | 属性 | chunk长度，当前仅支持64。 | INT64 | - |
| q_kernel | 输出 | q重排结果，shape为`(B, Hk, T, K)`。 | FLOAT16 | ND |
| k_kernel | 输出 | k重排结果，shape为`(B, Hk, T, K)`。 | FLOAT16 | ND |
| w_kernel | 输出 | WY变换的$W$，shape为`(B, Hv, T, K)`。 | FLOAT16 | ND |
| u_kernel | 输出 | WY变换的$U$，shape为`(B, Hv, T, V)`。 | FLOAT16 | ND |
| g_kernel | 输出 | chunk内g的前缀和，shape为`(B, Hv, T)`。 | FLOAT32 | ND |

## 约束说明

- `chunk_size`仅支持64，且要求`T % 64 == 0`（序列需由调用方补齐）。
- `Hv % Hk == 0`（GQA分组），`B <= 32`，`Hv <= 64`。
- `K`、`V`需为16的整数倍且不大于128；上界由<term>Atlas 推理系列产品</term>的192KB UB容量决定。
- 输入dtype固定：q/k/v/beta为FLOAT16，g为FLOAT32；g为对数衰减，须非正。
- 所有输入支持非连续Tensor，aclnn接口内部会先做连续化。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn接口 | [test_aclnn_chunk_gated_delta_rule_compute_wy.cpp](./examples/test_aclnn_chunk_gated_delta_rule_compute_wy.cpp) | 通过[aclnnChunkGatedDeltaRuleComputeWy](./docs/aclnnChunkGatedDeltaRuleComputeWy.md)调用ChunkGatedDeltaRuleComputeWy算子。 |
| torch接口 | [torch_ops_extension](./torch_ops_extension/README.md) | 安装torch_ops_extension后，通过`torch.ops.custom.npu_chunk_gated_delta_rule_compute_wy`调用ChunkGatedDeltaRuleComputeWy算子。 |
