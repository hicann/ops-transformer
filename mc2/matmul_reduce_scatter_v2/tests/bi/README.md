# Batch-Invariance (BI) 测试 — `matmul_reduce_scatter_v2`

本目录提供 `aclnnQuantMatmulReduceScatterV2` 算子的 batch-invariance 测试套件，作为 issue #2956 提出的「MC2 算子官方测试套件增加 BI 测试维度」需求的 **参考实现**。

## 1. 为什么需要 BI 测试

现有 UT/ST 测试覆盖：

- `acc`：与 reference 实现的数值精度对比（tolerance 内）
- `perf`：性能阈值（吞吐 / 延迟）

但不覆盖 **literature batch-invariance**：

```text
y[i] = f(x[i], W)

输出第 i 行仅依赖于输入第 i 行 + 共享权重 W，
与同一 batch 中其他行的内容无关。
```

文献定义参考：<https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference>

一个 op 可以满足 `acc` 与 `perf`、且满足 cross-rank determinism，但仍然违反 literature-BI（行间相互影响）。PR #6373 修复的 `AllGatherMatmulV2` 即属此类。

## 2. Methodology Note —— 三步流程缺一不可

> **本节为重点**。前一版方法学（每个 cell 只跑 variant A + B 各一次）有隐藏漏洞：无法区分「真 BI bug」和「trial 非确定性」。下述三步流程是修正后的版本。

每个 BI cell 走以下三个 phase：

### Phase 0：HCCL warm-up

Worker 启动后 HCCL state 在前几次调用容易处于 cold-start 状态，第一次 op 调用经常 `HcclAllocComResourceByTiling failed`（error code 561000），即便成功也可能选择跑非确定的 ring 拓扑。

**做法**：测试矩阵正式开始前，先做 `N_WARMUP=2` 次 dummy 调用（用任意合法 input），让 HCCL state 稳定。warm-up 中的 error 不计入测试结果。

### Phase 1：Multi-trial determinism baseline

每个 cell 的 variant A 用同一个 input 跑 `N_TRIALS=3` 次，所有 SHA 必须完全一致；variant B 同样。

```python
for _ in range(N_TRIALS):
    sha_list_a.append(run(seed_a))
assert len(set(sha_list_a)) == 1  # variant A 必须 deterministic
```

若 `sha_list_a` 不一致 → 标记为 `NON_DETERMINISTIC`，该 cell **不进入 Phase 2**（因为 A vs B 比较已无意义）。

### Phase 2：Variant A vs Variant B BI 检查

只有 Phase 1 通过，才把 `sha_a` vs `sha_b` 比较：

- 同 → `BI_PASS`（行间独立）
- 不同 → `BI_FAIL`（真 BI 违例）

### 三态输出

| 状态 | 含义 | 含义解读 |
|---|---|---|
| `BI_PASS` | 确定性基线通过 + A == B | row 内容跨 batch 独立 ✓ |
| `BI_FAIL` | 确定性基线通过 + A ≠ B | **真 BI 违例**，需修复 |
| `NON_DETERMINISTIC` | 同 input 多次跑 SHA 不一致 | 测试系统问题（HCCL state / 共享 NPU / 硬件），需排查；**不能简单判 BI fail** |
| `ERROR` | op 调用抛异常 | op-plugin shape 拒绝或基础设施故障 |

**关键教训**：把 `BI_FAIL` 与 `NON_DETERMINISTIC` 混淆会产生大量假阳性，淹没真 bug。这是 2026-06-06 调研中通过 multi-trial 重测纠正的方法学漏洞。

## 3. 算子状态

`matmul_reduce_scatter_v2` 是 **natively BI-clean**：`op_kernel/arch35` 使用 `ExecuteAicMatMulPipeline` 通过单一 `Mc2QuantBatchMatmulASWKernel<isGather=false>` wrapper 处理所有 input 行，无 LOCAL / GATHER 两路径分裂。本测试套件在严格方法学下确认零失败，作为 positive baseline 验证测试方法学本身可工作。

`AllGatherMatmulV2`（pre-PR-#6373）则会触发同一测试套件的 `BI_FAIL`，证明方法学的灵敏度。

## 4. 矩阵参数

| 维度 | 取值 |
|---|---|
| `WorldSize` | {2, 4, 8} |
| M_total | {8, 16, 32}（必须能被 WS 整除） |
| K | {256, 512, 1024}（多次跨 cube `baseK`=128 边界） |
| N | {64, 128}（含 `baseN`=64 边界） |
| 量化 scale | (`x1`=ID, `x2`=VAR) — K-axis varying，已知 BI 触发条件 |
| 目标 row 位置 | 0 |
| N_WARMUP | 2 |
| N_TRIALS | 3 |

约 24 cells × WS={2,4,8} = 72 cells，每个 cell 6 次 op 调用（3 trials × 2 variants），CI wall-clock 估算 < 15 min。

## 5. 如何运行

需要 V300 (Ascend 950PR/DT) 硬件 + CANN 9.1+ + `torch_npu` 2.9+。

```bash
cd mc2/matmul_reduce_scatter_v2/tests/bi/
bash run_bi_test.sh
```

`run_bi_test.sh` 会以 WS=2/4/8 三次启动 `bi_test_driver.py`，每次打印：

```text
=== matmul_reduce_scatter_v2 BI rigorous WS=2 N_WARMUP=2 N_TRIALS=3 ===
  Phase 0: HCCL warm-up...
  result: BI_PASS=24 BI_FAIL=0 NON_DET=0 ERROR=0 SKIP=0
```

任何 `BI_FAIL > 0` 即真 BI 违反；`NON_DET > 0` 需排查测试系统而非算子；`ERROR` 通常是 op-plugin shape rejection。

## 6. 与 ATK 框架的集成（待讨论）

当前实现作为 standalone Python driver，不依赖 ATK。issue #2956 中讨论的 ATK 集成方案需要在 `standard` 字段新增 `'bi'` key，executor 增加 `_run_bi_check` 钩子。集成时需保留三个关键点：

1. `n_warmup` 和 `n_trials` 必须可配置（这是 BI 方法学的核心，不是 boilerplate）
2. 三态结果 `BI_PASS / BI_FAIL / NON_DETERMINISTIC` 需独立汇总，不能合并
3. `NON_DETERMINISTIC > 0` 不应该自动判 op 失败，但要写入测试报告引起关注

待 maintainer 评估后再统一改造。本 PR 优先保证 BI 测试可独立运行，作为方法学的可执行规范。
