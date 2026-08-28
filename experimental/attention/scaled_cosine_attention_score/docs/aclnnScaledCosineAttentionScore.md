# aclnnScaledCosineAttentionScore

## 功能说明

融合 query/key L2 统计、余弦相似度矩阵和 per-head logit scale，返回 softmax
之前的注意力分数。

## 接口原型

```cpp
aclnnStatus aclnnScaledCosineAttentionScoreGetWorkspaceSize(
    const aclTensor* query,
    const aclTensor* key,
    const aclTensor* scale,
    double clampMax,
    double eps,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnScaledCosineAttentionScore(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

参数顺序为输入 → 属性 → 输出。当前实现的用户 workspace 为 0，但调用方仍应
使用第一段接口返回的 `workspaceSize`，不可硬编码。

## 参数

| 参数 | 说明 |
|---|---|
| query | `[B,H,N,d]`，FLOAT16/BFLOAT16/FLOAT32 |
| key | 与 query shape/dtype 相同 |
| scale | `[H]` 或 `[H,1,1]`，FLOAT32 |
| clampMax | `exp` 前的单边上限，默认 4.6052 |
| eps | 加到平方和中的正数，默认 1e-12 |
| out | `[B,H,N,N]`，dtype 同 query |

## 调用样例

参见
[`examples/test_aclnn_scaled_cosine_attention_score.cpp`](../examples/test_aclnn_scaled_cosine_attention_score.cpp)。
