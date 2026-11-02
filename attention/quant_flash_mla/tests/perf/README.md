# quant_flash_mla NPU 性能测试工具

从 ops-transformer-testkit（非量化 FA 算子测试框架）抽取的最小独立性能测试脚本集，
用于测试**全量化 MLA FlashAttention**（`QuantFlashMla`）在 NPU 上的性能。

## 目录结构

| 文件 | 作用 |
|---|---|
| `run_perf.py` | CLI 入口：逐 case 独立 msprof 进程，汇总 `perf.csv` |
| `perf_cases.py` | 用例集：运行时从 `../pytest/qfmla_paramset_func_rdv.CASES` 加载（跳过 `SKIP_CASES`） |
| `perf_codegen.py` | 生成单 case 自包含性能脚本（数据构造 + 热循环调用），支持 FP8/HIF8 与 TND/BSND/BNSD |
| `perf_profiler.py` | `msprof --aic-mode=task-based` 采集 + `msprof op --aic-metrics=PipeTimeline` pipe 时间线采集 |
| `perf_parser.py` | 按 `OP Type=QuantFlashMla` 过滤，热启动统计（丢首条取均值） |

## 快速开始

```bash
cd tests/perf

# 1. 列出全部用例（与 pytest 参数集同源，自动排除 SKIP_CASES；当前 89 条）
python3 run_perf.py --list

# 2. 单用例冒烟
python3 run_perf.py --runs 5 --case-filter _000000
python3 run_perf.py --runs 1 --case-filter _000000        # 只测一次（含首次运行开销）
python3 run_perf.py --runs 5 --discard-first --case-filter _000000  # 丢弃预热取平稳均值

# 3. 全量跑（建议放后台）
nohup python3 run_perf.py --runs 5 > full_run.log 2>&1 &

# 4. 按子串过滤（逗号分隔，任一子串命中即选中）
python3 run_perf.py --runs 5 --case-filter Decode,NZ
python3 run_perf.py --runs 5 --case-filter PA_BnNBsD

# 5. 性能 + PipeTimeline（对 --case-filter 命中的全部用例额外采集 pipe 时间线）
python3 run_perf.py --runs 5 --case-filter _000000 --pipe-timeline
python3 run_perf.py --runs 5 --case-filter Decode --pipe-timeline
```

## CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--runs` | 5 | 每 case 迭代次数（≥ 1）；默认对全部条目取均值 |
| `--discard-first` | 关闭 | 丢弃第一条（预热）后取均值，取平稳值；需 `--runs >= 2` |
| `--case-filter` | 无 | 逗号分隔的用例名子串过滤 |
| `--output` | `results/<时间戳>` | 输出目录 |
| `--pipe-timeline` | 关闭 | 布尔开关；对 `--case-filter` 命中的全部用例，在性能采集后额外跑一次 `msprof op --aic-metrics=PipeTimeline` |
| `--list` | - | 只列出用例名，不运行 |

## 输出结构

```
results/<时间戳>/          （或 --output 指定目录）
├── perf.csv               # 汇总: case_name, duration_us, mfu, samples, status, message, pipe_visualize_bin
└── <case_name>/
    ├── <case_name>.py     # 生成的性能脚本（可直接单独调试）
    ├── <case_name>.msprof.log
    ├── op_summary.csv     # msprof 采集的原始算子条目
    └── pipe_timeline/     # 启用 --pipe-timeline 时才有
        ├── <case_name>_pipe.py
        ├── <case_name>.msopprof.log
        └── OPPROF_*/      # PipeUtilization.csv + visualize_data.bin 等
```

终端同时打印摘要：

```
[SUMMARY] 89 cases, 0 failed, results -> .../perf.csv
  [HOT] Decode_..._TND_PA_lSE_000000    117.83 us  MFU=12.34%
```

## MFU 计算

每个 case 在采集到热启动耗时后按下列公式估算 MFU（以一个 batch 为单位，
多 batch 逐条累加）：

```
sp0:            compute_num = 2*kvs*qs*(qkd+vd)*qn
sp3 (qs<kvs):   compute_num = 2*(kvs-qs+1+kvs)*qs/2*(qkd+vd)*qn
sp3 (qs>=kvs):  compute_num = 2*kvs*kvs/2*(qkd+vd)*qn
fp8_peak_tflops = 2*16*16*32*1.6*core_num*0.001
mfu = (compute_num / (duration*1e-6)) * 1e-12 / fp8_peak_tflops
```

- `qs` = `seqused_q[b]`，`kvs` = `cache_seqlens[b]`，`qn` = `N_q`
- `qkd` = 576（nope 512 + rope 64，常量 `QK_HEAD_DIM`），`vd` = 512（V head dim，
  常量 `V_HEAD_DIM`，取自 `perf_codegen.D_V`）；QK 与 PV 的 head dim 不同，需分开计入
- `core_num` 运行时取设备 `cube_core_num`（纯 CPU 环境回退 32），
  32 核对应 `fp8_peak_tflops = 838.86`
- 结果以**百分比**写入 `perf.csv` 的 `mfu` 列，终端同时打印 `MFU=xx.xx%`
- `mask_mode != 3` 的用例按 sp0 公式计算

## 用例来源与字段

用例不再在 perf 目录内维护，而是由 `perf_cases.load_cases()` 在运行时从
`../pytest/qfmla_paramset_func_rdv.py` 的 `CASES` 加载，并跳过其 `SKIP_CASES`，
因此性能用例与功能用例始终同源。新增/删除用例请改 pytest 参数集。

`load_cases()` 将参数集条目归一为以下字段：

| 字段 | 说明 |
|---|---|
| `B / N_q / N_kv` | batch 数 / Q 头数 / KV 头数 |
| `seqused_q` | 逐 batch 实际 Q 序列长 |
| `cache_seqlens` | 逐 batch KV cache 序列长 |
| `enable_pa` | 仅标注；新接口 `block_table` 必填，统一按 PA 顺序映射构造，kernel 数据量与 noPA 语义等价 |
| `kv_cache_layout` | `BnNBsD` / `BnBsH` / `NZ`（映射到 `layout_kv`：`PA_BNBD` / `PA_BBND` / `PA_NZ`） |
| `sparse_mode` | 即参数集的 `mask_mode`：0=无掩码，3=causal（mask 以 int8 传入） |
| `enable_lse` | 是否返回 softmax lse |
| `num_blocks` | 物理 cache 块复用：0=不重用（顺序分配）；>0 时按块数随机映射，模拟生产环境块复用 |
| `quant_mode` | 0=hifp8（uint8 承载 hif8 编码，`32768/row_max`）；1=fp8_e4m3（`448/row_max`） |
| `layout` | Q 输入布局：`TND` / `BSND` / `BNSD`，同时决定 `q_descale` 形状与 `layout_out` |
| `graph_path` | 仅标注，perf 统一走单算子调用 |

`quant_mode` 与 `layout` 的构造差异：

| 场景 | Q / q_descale | cu_seqlens_q |
|---|---|---|
| FP8（1） | 量化到 `float8_e4m3fn`，`QK_MAX=448` | 按布局 |
| HIF8（0） | 复用 `pytest/common/generate_hifloat8_data.py` 编码为 `uint8`，`QK_MAX=32768` | 按布局 |
| TND | `q (T,N,576)` / `descale (T,N)` | 传入 |
| BSND | `q (B,S,N,576)` / `descale (B,S,N)` | `None` |
| BNSD | `q (B,N,S,576)` / `descale (B,N,S)` | `None` |

说明：当前参数集全部为 `quant_mode=1 + TND`，perf 默认只跑 FP8/TND；
HIF8 或 BSND/BNSD 性能用例需先在 pytest 参数集中补充对应字段。

## 统计口径

- **过滤**：只统计 `OP Type=QuantFlashMla` 的 AI Core kernel；
  Metadata（AI_CPU）与 ZerosLike 等干扰算子自动排除。
- **统计口径**：默认对全部采样取均值；加 `--discard-first` 时丢弃每次独立进程的第 1 次调用
  （含 tiling 缓存预热），其余取均值，`--runs` 需 ≥ 2。`samples` 列记录实际参与均值的条数。
- **Metadata 开销**：`quant_flash_mla_metadata` 在热循环外只调一次，
  不计入 kernel 时长。若要测含 metadata 的端到端开销，把生成脚本里的 metadata 调用
  挪进循环即可（脚本在输出目录 `<case_name>.py` 中，可手动改后单跑）。

## PipeTimeline

`--pipe-timeline`（布尔开关）对 `--case-filter` 命中的**全部**用例，在性能采集完成后
**单独再跑一次** `msprof op --aic-metrics=PipeTimeline --launch-count=1 --kernel-name=QuantFlashMla`。
需要缩小 pipe 采集范围时，直接用 `--case-filter` 收窄，不再提供第二套过滤条件。

- 采集的是 PMU 硬件计数级的 Cube/Vector/MTE pipe 时间线（与非量化框架
  `core/msopprof.py` 的 Phase 2 相同的采集方式）。
- 产物在 `<case>/pipe_timeline/OPPROF_*/`：
  - `visualize_data.bin`：用 **MindStudio Insight** 打开看 pipe 时间线可视化；
  - `PipeUtilization.csv` 等：各 pipe 占比的 CSV。
- `perf.csv` 新增 `pipe_visualize_bin` 列记录 bin 路径，未采集为空。
- 与性能采集分开跑的原因：PipeTimeline 的 visualize_data.bin 较大，且 msprof 采集
  模式互斥。

## 已知说明

- **torch 扩展缓存隔离**：`run_perf.py` 自动设置 `TORCH_EXTENSIONS_DIR=perf/.torch_ext/`。
  共享的 `~/.cache/torch_extensions/` 可能被其它 conda 环境的旧版
  `cann_ops_transformer` 污染（`.so` 与 wrapper 参数签名不一致，metadata 报
  TypeError），独立缓存目录保证 `.so` 始终从当前环境的 csrc 编译。
- 数据构造对齐 `tests/pytest/common/qfmla_golden.py` 的 NPU 路径：
  Q per-token-head 量化（按 `layout` 排成 TND/BSND/BNSD）、K per-tensor 量化 → PA cache；
  V 由算子内部从 K 前 512 维复用。
- HIF8 数据由 `pytest/common/generate_hifloat8_data.py` 在 CPU 编码成 `uint8` 后
  直接传入算子（不做 dtype 转换），与功能用例的数据链路一致。
- `mask_mode=3` 的 `attn_mask` 需以 **int8** 传入（bool 会被算子校验拒绝）。
- `results/`、`.torch_ext/`、`*.log` 已在 `.gitignore` 中忽略，请勿提交。
