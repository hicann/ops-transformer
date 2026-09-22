# 算子名称：MoeGateGeluClamp

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2训练系列产品</term>                                         | 是       |

## 功能说明

- 算子功能：在MoE架构的Expert MLP阶段，实现GeGLU激活函数与Clamp值截断操作的深度融合，并支持动态Shape下的真实行数读取。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| block_dim | 输入 | AI CORE的数量，Ascend910B是40 | int64_t | - |
| x | 输入 | Expert MLP的Gate和Up投影融合输入，shape为(ROWS, Hidden) | BFLOAT16 | ND |
| total_rows | 输入 | 动态Shape场景下的真实有效行数，shape为(local_expert_num) | int64_t | ND |
| out | 输出 | GeGLU激活及Clamp后的输出，shape为(ROWS, Hidden/2) | BFLOAT16 | ND |
| NUM_ROWS | 输入 | 静态总行数ROWS，若use_total_rows为true则被覆盖 | int64_t | - |
| per_count | 输入 | 输入张量的隐藏层维度Hidden | int64_t | - |
| per_total_count | 输入 | 每个ep上专家数 | int64_t | - |
| is_clamp | 输入 | 是否启用Clamp截断操作 | bool | - |
| clamp_value | 输入 | Clamp的截断边界值 | double | - |
| dtype | 输入 | 数据类型标识 | int64_t | - |
| use_total_rows | 输入 | 是否启用动态行数读取 | bool | - |

## 约束说明

- Hidden必须小于等于8192

## 调用说明

```python
torch.ops.ascend_ops.moe_gate_gelu_clamp(
    block_dim, x, total_rows, out,
    NUM_ROWS, per_count, per_total_count,
    is_clamp, clamp_value, dtype, use_total_rows)
```
