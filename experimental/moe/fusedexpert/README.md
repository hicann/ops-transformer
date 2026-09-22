# 算子名称：FusedExpert

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2训练系列产品</term>                                         | 是       |

## 功能说明

- 算子功能：在MoE (Mixture of Experts)架构的分布式推理/训练中，将普通路由专家与共享专家的ID及其对应的TopK权重进行融合拼接。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| blockDim | 输入 | AI CORE的数量，Ascend910B是40 | int64_t | - |
| expertId | 输入 | 原始路由专家ID，shape为(token_num, k) | int32_t | ND |
| fused_expertId | 输出 | 融合共享专家后的专家ID，shape为(token_num, k + shared_expert_num) | int32_t | ND |
| topk_logit | 输入 | 原始路由专家的TopK权重，shape为(token_num, k) | BFLOAT16 | ND |
| fused_topk_logit | 输出 | 融合后的权重，shape为(token_num, k + shared_expert_num) | BFLOAT16 | ND |
| pos | 输入 | 位移参数，用于计算当前专家所属的Device ID | int64_t | - |
| expertNum | 输入 | 增加共享专家后，每张卡实际分配到的专家总数 | int64_t | - |
| shared_expert | 输入 | 共享专家数量 | int64_t | - |
| sharedId_dp | 输入 | 当前DP域内共享专家的起始基础ID | int64_t | - |
| thread_count | 输入 | DP域内的卡数 | int64_t | - |
| ROWS | 输入 | 输入的token_num | int64_t | - |
| per_count | 输入 | 等于普通专家的topk: k | int64_t | - |
| start_tokenidx | 输入 | 当前ep上原始输入的起始索引 | int64_t | - |
| end_tokenidx | 输入 | 当前ep上原始输入的结束索引的后一位 | int64_t | - |
| dtype | 输入 | 数据类型标识 | int64_t | - |

## 约束说明

- 当前k支持7，8，16
- shared_expert_num当前支持1
- 位移参数pos = log2(local_simple_num), local_simple_num >= 8
- dtype取值说明（1=half, 27=bfloat16）

## 调用说明

```python
torch.ops.ascend_ops.fused_expert(
    block_dim, expertId, fused_expertId, topk_logit, fused_topk_logit,
    pos, expertNum, shared_expert, sharedId_dp, thread_count,
    ROWS, per_count, start_tokenidx, end_tokenidx, dtype)
```
