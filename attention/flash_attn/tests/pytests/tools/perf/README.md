# 性能测试脚本

本目录集中放置 flash_attn 的性能用例生成、执行、结果收集和线性度绘图脚本。
所有命令默认在 `attention/flash_attn/tests/pytests` 下执行。

## 脚本分类

| 分类 | 脚本 | 用途 |
|---|---|---|
| 1. 用例生成 | `gen_perf_base_cases.py` | 按 `CASE_GROUPS` 生成基础性能用例 |
| 1. 用例生成 | `gen_perf_linearity_cases.py` | 生成 B/S1/S2 线性度用例 |
| 2. 执行与结果收集 | `run_perf_raw.py` | 直接跑 msprof，导出逐条原始表和最终性能表 |
| 2. 执行与结果收集 | `export_perf_raw.py` | 从已有 perf 目录单独导出逐条原始表 |
| 2. 执行与结果收集 | `summarize_perf_raw.py` | 把逐条原始表汇总成每 case 一行 |
| 3. 线性度测试 | `plot_perf_linearity.py` | 从逐条原始表画 B/S1/S2 线性度图 |

## 前置条件

1. 已编译并安装 custom 包，且 `flash_attn` 与 `flash_attn_metadata` 一起编译。
2. 跑全量基础/线性度用例前使用全量 tiling key 编译，否则可能缺 kernel。
3. 执行前自行 source CANN 与 custom 包环境（脚本不再代为 source，也不感知安装路径）。

---

## 1. 用例生成

### 1.1 基础性能用例

```bash
python3 tools/perf/gen_perf_base_cases.py
python3 tools/perf/gen_perf_base_cases.py --check
```

生成基础性能用例。

输出：

- `test_cases/perf/base_generated.py`
- `test_cases/perf/manifest.json`

case 名格式（`QN`/`KVN` 为真实 head 数，`G = QN / KVN`）：

```text
Perf_Base_D{D}_DV{DV}_M{mask}_QN{q_heads}_KVN{kv_heads}_{layout}_B{batch}_QS{s1}_KVS{s2}[_W{left}_{right}]_{dtype}
```

用例参数键仍为 `N1`/`N2`（框架契约），仅展示名使用 `QN`/`KVN`，避免 `N116_N216` 这类写法被误读。

`win_left_right` 只对 `mask_mode=4` 生效；非 mask4 用例不带窗口字段。

### 1.2 线性度用例

```bash
python3 tools/perf/gen_perf_linearity_cases.py
```

生成 B/S1/S2 线性度用例。

输出：

- `test_cases/perf/linearity_generated.py`
- `test_cases/perf/linearity_manifest.json`

两个生成器都支持 `--dry-run` 预览；`gen_perf_base_cases.py` 额外支持 `--check`（校验生成结果与 manifest 条数/字段一致）。

## 2. 性能用例执行与结果收集

### 2.1 执行

先 source CANN 与 custom 包环境，再在 `attention/flash_attn/tests/pytests` 下直接调用：

```bash
python3 tools/perf/run_perf_raw.py --case-files perf.base_generated --runs 5 --out-dir /abs/path/run_base
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--case-files` | 用例模块，如 `perf.base_generated`；可逗号分隔多个 |
| `--runs` | 每个 case 执行次数，默认 5，包含 warmup |
| `--out-dir` | 绝对/相对路径（相对当前工作目录），目录必须不存在；不指定时在 `tools/perf/` 下创建 `perf_<时间戳>/` |

### 2.2 输出目录

```text
<out_dir>/
├── profiling_raw.csv        # 逐条原始表，一行一次运行
├── profiling_summary.csv    # 每 case 一行的最终性能表
├── _perf_sweep.py           # 本次 msprof 的被测脚本
├── _perf_log.jsonl          # 逐 case 运行状态
├── run.log                  # 精简进度
├── run_full.log             # msprof/子进程完整输出
└── msprof_output/           # 原始 profiling
```

`run.log` 只保留 `[batch]` 和 `[CASE_END]`，两者使用同一个全局编号：

```text
start: 166 cases x 5 runs -> /path/to/out
  [batch] 1/166  Perf_Base_prefill_...
[CASE_END] 1/166  OK
  [batch] 2/166  Perf_Base_decode_...
[CASE_END] 2/166  OK
```

终端和 `run.log` 都只显示精简进度；完整输出看 `run_full.log`。

### 2.3 只导出已有 perf 目录

```bash
python3 tools/perf/export_perf_raw.py --perf-dir <perf目录> --case-files perf.base_generated --runs 5
```

默认写 `<perf目录>/profiling_raw.csv`，可用 `--out` 指定路径。
优先读顶层 `op_summary.csv`，没有则读 `msprof_output/PROF_*/mindstudio_profiler_output/op_summary_*.csv`。

### 2.4 profiling_raw.csv

一行对应一次运行，同一个 case 会出现 `--runs` 行。

前五列固定：

```text
case, case_id, phase, mode, run
```

随后是 case 参数：

```text
B, N2, G, D, DV, S1, S2, mask, dtype,
layout_q, layout_kv, layout_out,
max_seqlen_q, max_seqlen_kv,
cu_seqlens_q, cu_seqlens_kv,
seqused_q, seqused_kv
```

之后是 `op_summary` 原始列，最后一列 `kernel_name` 来自 `task_time_*.csv`
（按 Device/Stream/Task ID 关联）。
`cu_seqlens_*`、`seqused_*` 为 varlen 字段，普通用例为空。

### 2.5 profiling_summary.csv

每个 case 一行，列结构与 `profiling_raw.csv` 类似，但 profiling 部分只保留最终性能和 kernel 名：

```text
case, case_id, phase, mode,
B, N2, G, D, DV, S1, S2, mask, dtype,
layout_q, layout_kv, layout_out,
max_seqlen_q, max_seqlen_kv,
cu_seqlens_q, cu_seqlens_kv,
seqused_q, seqused_kv,
avg_us, theory_us, mfu_pct, kernel_name
```

`avg_us` 是 hot（`S1 > --cold-thr`，默认 16）丢首轮、cold 去 min/max 后的值；
`kernel_name` 取该 case 首次非空值。

`theory_us` / `mfu_pct` 按模型 FLOPs 计算，峰值算力通过 `--peak-tflops` 指定（默认 0 = 不计算这两列）：

```text
theory_us = 2·B·N1·可见对·(D+DV) / 峰值算力      N1 = N2·G
mfu_pct   = theory_us / avg_us × 100
可见对: mask=0 → S1·S2；mask=3 → 右对齐 causal(第 i 行 min(S2, max(0, i+S2-S1+1)))；mask=4 留空
```

单独生成：

```bash
python3 tools/perf/summarize_perf_raw.py --raw <dir>/profiling_raw.csv --out <dir>/profiling_summary.csv
```

### 2.6 分段 + 间隔（避免持续满载影响测量）

```bash
python3 tools/perf/run_perf_raw.py --case-files perf.base_generated \
  --runs 5 --cold-thr 16 \
  --chunk-size 30 --gap-ms 20000 \
  --big-case-regex "KVS(32768|131072)" --big-gap-ms 30000 \
  --out-dir ./run_chunked
```

- `--chunk-size N`：普通用例每 N 条一段，段间 `sleep --gap-ms`；
- `--big-case-regex`：匹配到的用例**单独成段、逐条跑**，段后 `sleep --big-gap-ms`（0 则用 `--gap-ms`）；
- 每段独立 `chunk_XX/`（含自己的 msprof 输出与 raw/summary），跑完自动合并成顶层
  `profiling_raw.csv` / `profiling_summary.csv`（`ab_min.py` 直接可用）；
- 自动写 `chunk_meta.csv`：每段的 `起止时间 / 是否大 case`。

---

## 3. 线性度测试

### 3.1 生成线性度用例

参考 1.2，先生成 `linearity_generated.py`：

```bash
python3 tools/perf/gen_perf_linearity_cases.py
```

### 3.2 执行

```bash
python3 tools/perf/run_perf_raw.py --case-files perf.linearity_generated --runs 5 --out-dir /abs/path/run_linearity
```

输出目录同样包含 `profiling_raw.csv`、`profiling_summary.csv` 和原始 profiling。

### 3.3 绘图

```bash
python3 tools/perf/plot_perf_linearity.py --raw <run_linearity>/profiling_raw.csv --out <run_linearity>
```

输出：

- `linearity_B.png`
- `linearity_S1.png`
- `linearity_S2.png`
- `linearity_all.png`
- `linearity_summary.csv`

图顶部标注公共固定参数（D、DV、mask、layout、G、N2、dtype），每个子图标注该组固定的 B/S1/S2。

`--perf-dir` 是旧入口（读 `perf.csv` + 每 case `op_summary`），新流程统一使用 `--raw`。

---

## 自定义用例

在 `test_cases/perf/` 下新建模块，按现有 `TestCases` 格式写参数，然后：

```bash
python3 tools/perf/run_perf_raw.py --case-files perf.<模块名> --runs 5 --out-dir ./run_custom
```

用例模块需要能被 `test_cases.<模块名>` 导入；`--case-files` 可逗号分隔多个模块。
只支持批量模式，不要用 `test_flash_attn.py --one_by_one`，否则导出无法按 case 对应行。

## 常见问题

| 现象 | 处理 |
|---|---|
| `out dir already exists` | 换一个新目录，脚本不允许复用已有目录 |
| 行数不等于 `runs × case 数` | 有 case 崩溃或误用了 `--one_by_one`；看 `run_full.log` 和 `_perf_log.jsonl` |
| `Cannot find tilingKey` | 用全量 tiling key 重新编译安装 |
| `kernel_name` 为空 | 检查 `task_time_*.csv` 是否存在，及 Device/Stream/Task ID 是否匹配 |
| 导入不到用例模块 | 确认模块在 `test_cases/perf/` 下，`--case-files` 传 `perf.<模块名>` |

## 数据存放

原始 profiling、`profiling_raw.csv` 和 `profiling_summary.csv` 都放在 out-dir；新运行建议显式指定 `--out-dir`，避免散落。
