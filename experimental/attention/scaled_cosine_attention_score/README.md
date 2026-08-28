# ScaledCosineAttentionScore

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

`ScaledCosineAttentionScore` 将 query/key 沿 head dimension 的 L2 统计、批量
余弦打分以及 per-head logit scale 融合为一个算子，输出 softmax 前的注意力分数：

```text
inv_q[b,h,i] = 1 / sqrt(sum_t(query[b,h,i,t]^2) + eps)
inv_k[b,h,j] = 1 / sqrt(sum_t(key[b,h,j,t]^2) + eps)
score[b,h,i,j] = sum_t(query[b,h,i,t] * key[b,h,j,t])
                   * inv_q[b,h,i] * inv_k[b,h,j]
                   * exp(min(scale[h], clamp_max))
```

平方和、点积、开方、除法及 scale 均按 FLOAT32 计算，结果转换回 query dtype。
实现不生成归一化中间张量，不申请用户 workspace，也不显式转置 key。

## 参数说明

| 参数名 | 输入/输出/属性 | shape | 数据类型 | 格式 |
|---|---|---|---|---|
| query | 输入 | `[B,H,N,d]` | FLOAT16、BFLOAT16、FLOAT32 | ND |
| key | 输入 | `[B,H,N,d]`，shape/dtype 与 query 相同 | 同 query | ND |
| scale | 输入 | `[H]` 或 `[H,1,1]` | FLOAT32 | ND |
| clamp_max | 属性 | 标量，默认 `4.6052` | FLOAT | - |
| eps | 属性 | 正标量，默认 `1e-12` | FLOAT | - |
| attn_score | 输出 | `[B,H,N,N]` | 同 query | ND |

## 约束说明

- query/key rank 必须为 4，且 shape 完全一致；各运行时维度必须为正数。
- scale 必须为 FLOAT32，shape 为 `[H]` 或 `[H,1,1]`。
- `clamp_max` 必须为有限值；`eps` 必须为有限正数。
- 当前只注册 `ascend910b` 和 `ascend910_93`。
- 当前 kernel 为 Vector Core 融合基线；最终吞吐需要在目标 NPU 上与组合算子实测。
- 本算子采用 `sqrt(sum(x^2) + eps)`，不要与 `max(norm(x), eps)` 的参考式混用。

## ops-transformer 集成

本目录是 ops-transformer 仓库 experimental/attention 下的单算子模块，不是独立 CMake 工程。提交时将整个目录放到
ops-transformer 对应算子分类目录，由仓库顶层 CMake 提供 `add_modules_sources`、
`add_graph_plugin_sources`、`AddOpTestCase` 等构建函数。

主要文件：

```text
├── CMakeLists.txt
├── docs/aclnnScaledCosineAttentionScore.md
├── examples/test_aclnn_scaled_cosine_attention_score.cpp
├── op_graph
├── op_host
│   ├── scaled_cosine_attention_score_def.cpp
│   ├── scaled_cosine_attention_score_infershape.cpp
│   ├── scaled_cosine_attention_score_tiling.cpp
│   └── scaled_cosine_attention_score_tiling.h
├── op_kernel
│   ├── scaled_cosine_attention_score.cpp
│   ├── scaled_cosine_attention_score_impl.hpp
│   └── scaled_cosine_attention_score_tiling_def.h
└── tests/ut
```

调用方式参见
[`docs/aclnnScaledCosineAttentionScore.md`](docs/aclnnScaledCosineAttentionScore.md)。
算法与 tiling 细节参见 [`docs/design.md`](docs/design.md)。
