# ScaledCosineAttentionScore 设计说明

## 1. 数学契约

输入 `query,key:[B,H,N,d]`、`scale:[H]` 或 `[H,1,1]`：

```text
inv_q[b,h,i] = 1 / sqrt(sum_t(query[b,h,i,t]^2) + eps)
inv_k[b,h,j] = 1 / sqrt(sum_t(key[b,h,j,t]^2) + eps)
score[b,h,i,j] = exp(min(scale[h], clamp_max))
                   * inv_q[b,h,i] * inv_k[b,h,j]
                   * sum_t(query[b,h,i,t] * key[b,h,j,t])
```

所有平方和、点积、开方、除法、clamp 和 exp 均使用 fp32；最终转换回
`query/key` 的 dtype。这里严格采用 `sqrt(sum(x^2) + eps)`，它与
`max(sqrt(sum(x^2)), eps)` 并非完全相同，不能在测试中混用。

## 2. 当前 kernel 数据流

当前版本是无 workspace 的融合 Vector Core 基线：

1. 将 `B*H*N` 个 query 行以循环分片方式分配给 AIV core。
2. 每个 query 行只搬入 UB 一次，计算并保留 fp32 的逆范数。
3. 同一 `(b,h)` 下的 key 按 `keyTileRows` 分块搬入 UB；每行尾部补零，
   因而 `d=80/88` 等非 32-byte 对齐 shape 不会越界。
4. 在 UB 中计算 key 逆范数、点积和 per-head scale，结果按 key tile
   一次写回 GM。
5. `N` 的尾块使用实际行数，输出搬运也使用实际字节数。

该路径不生成 `q_norm`、`k_norm` 或转置张量，不申请用户 workspace，整项
计算只有一次 kernel launch。其代价是当前点积走 Vector reduction，尚未使用
Cube Matmul，因此它是功能正确、融合边界清晰的第一版，不应在实测前宣称其
吞吐一定超过高度优化的 BMM 组合。

## 3. Tiling

Host 侧读取 AIV core 数和 UB 容量：

- `blockDim = min(B*H*N, aivCoreNum)`；
- `alignedHeadDim` 按输入 dtype 的 32-byte 粒度向上对齐；
- UB 预算保留 20% 给 API 内部开销；
- `keyTileRows` 根据 query 常驻区、fp32 临时区、key 输入/fp32 区和输出区
  反推，最大为 128；
- 若最小 tile 也放不进 UB，tiling 直接失败，不让 kernel 越界运行。

## 4. 校验与边界

Host 侧拒绝以下输入：

- query/key 不是相同的四维正整数 shape；
- dtype 不同或不是 fp16/bf16/fp32；
- scale 不是 fp32，或 shape 不是 `[H]`/`[H,1,1]`；
- `eps <= 0`，或属性不是有限值；
- 维度超过 tiling 字段范围，或 UB 容量不足。

零向量的结果定义良好：点积为零，分母由 `eps` 保证非零。

## 5. 下一阶段性能路线

若 910B 实测显示 Vector reduction 成为瓶颈，建议保留相同对外接口，增加
Vector/Cube 混合 tiling key：Vector 侧生成片上归一化 tile，Cube 侧消费 A1/B1
tile 并以 fp32 累加，copy-out 阶段融合 scale。实现时必须验证片上生产消费同步、
L1/L0 容量和尾 K padding；若退化为把完整归一化张量写入 GM workspace，就失去
本算子的核心收益，不应作为默认优化路径。
