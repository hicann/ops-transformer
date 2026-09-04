# `all_gather_matmul_v2` Batch-Invariance (BI) 测试

`aclnnAllGatherMatmulV2` / `npu_all_gather_quant_mm`（MXFP8, `block_size=32`）的
batch-invariance 测试套件。是 `matmul_reduce_scatter_v2` BI 套件（**PR #6593**）的姊妹实现，
沿用同一套方法学，覆盖本算子的两种 `x2` 朝向（**非转置** 与 `transB`）——正是 **PR #4705**
（「支持 `transB` 和校验 MX 量化 BI 一致性」）改动的面。

## 1. 为什么需要 BI 测试

现有 UT/ST 覆盖 `acc`（数值精度 tolerance）与 `perf`（性能阈值），但不覆盖
literature batch-invariance：

```text
y[i] = f(x[i], W)   —— 输出第 i 行只依赖输入第 i 行 + 共享权重 W，与同 batch 中其他行无关。
```

一个算子可以同时满足 `acc`、`perf`、cross-rank determinism，却仍违反 literature-BI
（行间相互影响）。定义参考：<https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference>

## 2. 方法学（三步，缺一不可，与 #6593 README §2 一致）

- **Phase 0 HCCL warm-up**：`N_WARMUP=2` 次 dummy 调用稳定 HCCL state（首调易 cold-start
  报 `HcclAllocComResourceByTiling` / 561000 或选到非确定 ring 拓扑）。warm-up 的 error 不计入。
- **Phase 1 多次确定性基线**：同 input 跑 `N_TRIALS=3` 次，SHA 必须全一致；否则记 `NON_DET`，
  该 cell 不进入 Phase 2。
- **Phase 2 组合-BI（variant A vs B）**：同一 pinned 探针行、**不同邻居行**（`seed_a` vs `seed_b`），
  探针行输出须 byte 一致。同 → `BI_PASS`；不同 → `BI_FAIL`。

三态输出：`BI_PASS` / `BI_FAIL` / `NON_DET`（另 `ERROR` = op 拒绝，`SKIP` = shape 不适用）。

## 3. 覆盖

每个 shape × `M ∈ {8,16,32}`，`(K,N)`：

| 组 | (K, N) |
|---|---|
| #6593 范围（小/中）| (256,64) (256,128) (512,64) (512,128) (1024,64) (1024,128) |
| 大 + 非方形 | (1024,1024) (2048,2048) (4096,4096) (4096,2048) (2048,4096) |

朝向：`BI_ORIENT=both`（默认，非转置 + `transB`）/ `notrans` / `trans`。
WS：`run_bi_test.sh` 默认跑 2/4/8。

## 4. 运行

```bash
cd mc2/all_gather_matmul_v2/tests/bi/
bash run_bi_test.sh "2 4 8"            # 全 WS
bash run_bi_test.sh "2" 0,1            # 单 WS + 指定卡
BI_ORIENT=notrans bash run_bi_test.sh  # 只非转置
```

每次打印：

```text
=== AllGatherMatmulV2 BI rigorous WS=2 N_WARMUP=2 N_TRIALS=3 DATA=random orient=both ===
  [NON-TRANS] SUMMARY BI_PASS=33 BI_FAIL=0 NON_DET=0 ERROR=0 SKIP=0
  [TRANS_B  ] SUMMARY ...
```

任何 `BI_FAIL > 0` 即真 BI 违反（退出码 1）；`NON_DET > 0` 需排查测试系统/硬件；
`ERROR` 通常是 `block_size` gate 未打（EZ0002）或 shape 拒绝。

## 5. 前置条件

- **`block_size` gate fix**：MXFP8 需 `block_size=32`；未打 patch 的 tiling 会 `EZ0002 blockSize
  should be 0` 拒绝（cann/ops-transformer issue #2778 / PR #6137）。需在含该修复的 build/vendor 上跑。
- **空闲 NPU 卡**：`HcclAllocComResourceByTiling ret=4` 表示卡被别的 HCCL 作业占用（端口 16666），
  换空闲卡（`ASCEND_RT_VISIBLE_DEVICES`）。
- **torchrun**（非手动 multiprocessing），否则 MC2 op 的 HCCL 资源分配失败。

## 6. 数据生成

- **非转置**：`BI_DATA_MODE=random`（默认）随机 `uint8` → `fp8_e4m3` / `e8m0`，自包含无外部依赖，已充分验证。
- **`transB`**：driver 内部**已接** `script/generate_mx_data.mx_quantize`（合法 MX 值，方形 K==N），
  `x2`/scale 以**非连续转置 view** 传入（`x2.t()`、`torch.transpose(x2s,0,1)`，不加 `.contiguous()`——
  加了报 `561002`）。随机字节在 `transB` 下会触发 aicpu 异常，故 `transB` **必须**用合法 MX 数据。
  依赖 `en_dtypes` / `ml_dtypes`（ST 树内自带），在 ops-transformer ST 树里跑即可。

## 7. 已验证状态

- **非转置**：WS=2/4/8 × 11 shape × `M ∈ {8,16,32}` = **99/99 `BI_PASS`**。
- **`transB`**：WS=2/4 **`BI_PASS` + 数值正确**（组合-BI + batch-size-BI，probe SHA 跨 M/seed/WS 一致，
  正确性 vs `fp8` 反量化 `fp32` 参考 median_rel 0.001）。在含合入版 #4705（master `03b5c20`）的 vendor 上验证。

## 8. 与 ATK 框架集成（待讨论）

同 #6593：作为 standalone driver，不依赖 ATK；集成时需保留 `n_warmup` / `n_trials` 可配、
三态结果独立汇总、`NON_DET` 不自动判失败但写入报告。
