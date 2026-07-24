# FIA FullQuant MXFP8 测试用例执行指南

## 1. 环境准备

### 1.1 安装 CANN / PyTorch + torch-npu

参考根目录 README.md，或使用项目根目录下的安装脚本：

```bash
bash install_cann_950.sh    # 安装 CANN 9.1.0 + 950 ops
bash install_torch_npu.sh   # 安装 PyTorch + torch-npu
```

### 1.2 每次执行前加载环境

```bash
source /home/user/Ascend/cann/set_env.sh
conda activate your-env-name
pip install pytest
cd attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test
```

---

## 2. 文件结构

```
fia_fullquant_mxfp8_test/
├── pytest.ini                                  # pytest 配置（自定义 marker）
├── conftest.py                                 # pytest 命令行选项（--golden-mode, --cache-dir, --msprof, --parse-prof, --perf-baseline）
├── common/
│   ├── __init__.py
│   ├── fia_fullquant_mxfp8_golden.py           # CPU golden 参考实现 + NPU 算子调用
│   ├── golden_cache.py                         # .pt 缓存工具模块
│   ├── result_compare_method.py                # 精度对比工具
│   ├── perf_parser.py                          # msprof op_summary.csv 解析 + baseline 比较
│   └── test_runner.py                          # 共享测试执行逻辑（apply_params / execute_test / check_results）
├── fia_fullquant_mxfp8_paramset_common.py      # 参数展开公共逻辑 + 默认值
├── fia_fullquant_mxfp8_paramset_debug.py       # debug 参数集（少量用例，快速验证）
├── fia_fullquant_mxfp8_paramset_func_rdv.py    # 功能正确性参数集（~150 条）
├── fia_fullquant_mxfp8_paramset_perf_rdv.py    # 性能/压力参数集（~100 条）
├── test_fia_fullquant_mxfp8_debug.py           # debug 测试入口
├── test_fia_fullquant_mxfp8_func_rdv.py        # 功能正确性测试入口
└── test_fia_fullquant_mxfp8_perf_rdv.py        # 性能/压力测试入口
```

---

## 3. 执行测试

### 3.1 三个测试入口

| 测试文件 | Marker | 参数集 | 用途 |
|----------|--------|--------|------|
| `test_fia_fullquant_mxfp8_debug.py` | `@pytest.mark.debug` | debug（2 条） | 快速验证基本功能 |
| `test_fia_fullquant_mxfp8_func_rdv.py` | `@pytest.mark.func_rdv` `@pytest.mark.ci` | func_rdv（~150 条） | 功能正确性全覆盖 |
| `test_fia_fullquant_mxfp8_perf_rdv.py` | `@pytest.mark.perf_rdv` | perf_rdv（~100 条） | 性能/压力验证 |

### 3.2 基本执行

```bash
# 运行 debug 用例
pytest test_fia_fullquant_mxfp8_debug.py -v

# 运行功能正确性用例
pytest test_fia_fullquant_mxfp8_func_rdv.py -v

# 运行性能/压力用例
pytest test_fia_fullquant_mxfp8_perf_rdv.py -v

# 按 marker 运行
pytest -m debug -v
pytest -m func_rdv -v
pytest -m perf_rdv -v
pytest -m ci -v
```

### 3.3 过滤用例（-k）

```bash
# 精确指定单个用例
pytest test_fia_fullquant_mxfp8_func_rdv.py -v -k "PA_NZ_B1_QS1_KVS512_Nq1_Nkv1_D128_SP3"

# 按 Prefill / Decode 模式
pytest test_fia_fullquant_mxfp8_perf_rdv.py -v -k "Prefill"
pytest test_fia_fullquant_mxfp8_perf_rdv.py -v -k "Decode"

# 按 layout 类型
pytest -v -k "PA_NZ"
pytest -v -k "BnNBsD"

# 组合过滤
pytest -v -k "PA_NZ and Prefill"

# 排除某些用例
pytest -v -k "not Decode"
```

### 3.4 常用选项

| 选项 | 作用 |
|------|------|
| `-v` | 详细输出，显示每个用例名 |
| `-s` | 不截断 stdout/stderr |
| `-x` | 遇到失败立即停止 |
| `--tb=long` | 失败时显示完整堆栈 |
| `-m ci` | 只运行 CI 标记的用例 |

---

## 4. Golden 缓存模式（--golden-mode）

测试支持将输入数据、CPU golden 输出、NPU 输出保存为 `.pt` 文件，下次运行时可以跳过数据生成，直接加载缓存执行指定步骤。

### 4.1 模式说明

| 模式 | 作用 |
|------|------|
| `all` | 全流程：生成数据 → CPU → NPU → 精度对比（**默认值**） |
| `gen` | 仅生成并保存输入数据 |
| `cpu` | 加载输入缓存 → 跑 CPU 并保存输出 |
| `npu` | 加载输入缓存 → 跑 NPU 并保存输出 |
| `compare` | 加载 CPU/NPU 缓存输出 → 精度对比 |

### 4.2 组合模式

模式支持逗号分隔组合，按 gen → cpu → npu → compare 顺序执行：

```bash
# 仅生成并保存输入数据（不跑 CPU/NPU）
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=gen

# 仅跑 NPU + 精度对比（CPU 输出从缓存加载）
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=npu,compare

# 跑 CPU + NPU（不对比）
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=cpu,npu

# 全流程（等同于 --golden-mode=all）
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=gen,cpu,npu,compare
```

### 4.3 典型工作流

**方式一：全流程一次跑完（默认）**

```bash
pytest test_fia_fullquant_mxfp8_func_rdv.py -v
```

**方式二：分步执行（适合跨机器调试）**

```bash
# 第一步：生成输入数据（不跑 CPU/NPU）
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=gen

# 第二步：跑 CPU golden
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=cpu

# 第三步：在 NPU 机器上跑 NPU + 精度对比
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=npu,compare
```

### 4.4 自定义缓存目录

```bash
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --cache-dir=/tmp/my_cache
```

默认缓存目录为 `common/golden_cache/`。

### 4.5 缓存文件命名

每个 case 生成 3 个 `.pt` 文件：

```
golden_cache/
├── {case_name}_input.pt        # 输入数据（Q/K/V fp8 + scale + block_table）
├── {case_name}_cpu_output.pt   # CPU golden 输出（atten_out + lse）
└── {case_name}_npu_output.pt   # NPU 算子输出（atten_out + lse）
```

---

## 5. 独立脚本运行

`common/fia_fullquant_mxfp8_golden.py` 也支持独立运行，同样支持 `--mode` 组合：

```bash
cd common/
python fia_fullquant_mxfp8_golden.py --mode all --case-name my_case
python fia_fullquant_mxfp8_golden.py --mode gen --case-name my_case
python fia_fullquant_mxfp8_golden.py --mode npu,compare --case-name my_case
python fia_fullquant_mxfp8_golden.py --mode cpu --case-name my_case --cache-dir=/tmp/cache
```

---

## 6. 用例命名规则

格式：`{LAYOUT}_B{batch}_QS{Q长度}_KVS{KV长度}_Nq{Q头数}_Nkv{KV头数}_D{维度}_SP{稀疏模式}[_{模式}]`

| 字段 | 含义 | 示例 |
|------|------|------|
| `PA_NZ` | kv_cache_layout = PA_NZ | `PA_NZ_B1_QS64_KVS64_...` |
| `PA_BnNBsD` | kv_cache_layout = BnNBsD | `PA_BnNBsD_B2_QS48_KVS64_...` |
| `TND` | enable_pa = False | `TND_B1_QS128_KVS128_...` |
| `B` | batch size | `B1`, `B8` |
| `QS` | Q 序列长度 | `QS128` |
| `KVS` | KV 序列长度 | `KVS512` |
| `Nq` | Q 头数 | `Nq80` |
| `Nkv` | KV 头数 | `Nkv8` |
| `D` | Head 维度 | `D128`, `D64` |
| `SP` | sparse_mode | `SP3` |
| `Prefill` | q_scale_layout = TND | 长序列 prefill 阶段 |
| `Decode` | q_scale_layout = N2TGD | 短序列 decode 阶段 |

---

## 7. 参数体系

### 7.1 全部参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `B` | int | 必填 | batch size |
| `N_q` | int | 必填 | Q 头数 |
| `N_kv` | int | 必填 | KV 头数（N_q 需能被 N_kv 整除） |
| `D` | int | 必填 | Head 维度（64 / 128） |
| `actual_seq_q` | list | 必填 | 每个 batch 的 Q 序列长度 |
| `actual_seq_kv` | list | 必填 | 每个 batch 的 KV 序列长度 |
| `enable_pa` | bool | 必填 | 是否启用 PagedAttention |
| `kv_cache_layout` | str | 必填 | KV Cache 布局（`PA_NZ` / `BnNBsD`） |
| `block_size` | int | 必填 | PA block 大小（512 / 1024） |
| `sparse_mode` | int | 必填 | 稀疏模式（0=无 mask, 3=causal+padding） |
| `q_scale_layout` | str | 必填 | Q scale 布局（`TND` / `N2TGD`） |
| `p_scale` | float | 必填 | P 量化 scale（1.0 / 15.0 / 100.0 / 128.0 / 256.0） |
| `data_range_q` | float | 1.0 | Q 数据范围 |
| `data_range_k` | float | 1.0 | K 数据范围 |
| `data_range_v` | float | 1.0 | V 数据范围 |
| `enable_lse` | bool | 必填 | 是否返回 Log-Sum-Exp |
| `enable_rope` | bool | 必填 | 是否启用 RoPE |
| `D_rope` | int | 64 | RoPE 维度 |
| `graph_path` | int | 0 | 图模式（0=单算子, 3=静态图, 5=动态图, 6=tiling下沉, 7=aclgraph） |
| `device_id` | int | 0 | NPU 设备 ID |
| `is_contiguous` | bool | True | 输入是否 contiguous |

### 7.2 默认值机制

`TEST_PARAMS_DEFAULTS` 中的参数如果在 paramset 中未显式指定，会自动使用默认值。例如 `D_rope`、`graph_path`、`device_id`、`is_contiguous` 等通常不需要每个 case 都写。

如需某个 case 使用非默认值，只需在 paramset 中显式指定：

```python
"MY_CASE": {
    "B": [1],
    ...
    "graph_path": [7],       # 使用 aclgraph 模式
    "D_rope": [128],         # 使用 128 维 RoPE
},
```

### 7.3 参数展开

`expand_paramset_to_cases()` 对每个配置中的所有维度做笛卡尔积展开。例如：

```python
"MY_CASE": {
    "B": [1, 2],
    "D": [64, 128],
    ...
}
```

会展开为 4 个独立用例（B=1/D=64, B=1/D=128, B=2/D=64, B=2/D=128）。

---

## 8. 新增用例

在对应的 paramset 文件中添加新条目。以 `fia_fullquant_mxfp8_paramset_func_rdv.py` 为例：

```python
"PA_NZ_B1_QS256_KVS512_Nq4_Nkv1_D128_SP3": {
    "B": [1],
    "N_q": [4],
    "N_kv": [1],
    "D": [128],
    "actual_seq_q": [[256]],
    "actual_seq_kv": [[512]],
    "enable_pa": [True],
    "kv_cache_layout": ["PA_NZ"],
    "block_size": [512],
    "sparse_mode": [3],
    "q_scale_layout": ["TND"],
    "p_scale": [1.0],
    "enable_lse": [False],
    "enable_rope": [False],
},
```

### B>1 不等长序列

当 B>1 时，`actual_seq_q` 和 `actual_seq_kv` 应为不等长列表：

```python
"PA_NZ_B4_QS128_KVS1024_Nq8_Nkv1_D128_SP3": {
    "B": [4],
    ...
    "actual_seq_q": [[96, 128, 160, 112]],
    "actual_seq_kv": [[800, 1024, 1280, 960]],
},
```

### SKIP_CASES 机制

对于执行时间过长的用例，可在 paramset 文件中定义 `SKIP_CASES` 集合：

```python
SKIP_CASES = {
    "PA_NZ_B128_QS1_KVS10240_Nq1_Nkv1_D128_SP0",
    "PA_NZ_B128_QS4_KVS16384_Nq1_Nkv1_D128_SP0",
}
```

被标记的用例会自动添加 `pytest.mark.skip`，在 pytest 输出中显示为 `SKIPPED`。

---

## 9. 精度对比标准

使用"双千分之五"标准：

| 指标 | FP16 阈值 | BF16 阈值 |
|------|-----------|-----------|
| rtol（相对容差） | 0.005 | 0.0078125 |
| atol（绝对容差） | 0.000025 | 0.0001 |
| pct_thd（通过率阈值） | 99.5% | 99.5% |
| max_diff_hd（最大相对误差上限） | 10 | 10 |

判定逻辑：通过率 >= 99.5% **且** 最大相对误差 < 10 时为 Pass。

---

## 10. pytest Marker 说明

| Marker | 用途 |
|--------|------|
| `ci` | CI 流水线测试 |
| `func_rdv` | 功能正确性 RDV 测试 |
| `perf_rdv` | 性能/压力 RDV 测试 |
| `debug` | 日常调试测试 |
| `graph` | 图模式编译测试 |
| `npu_only` | 仅 NPU 执行（无 CPU golden 对比） |

---

## 11. 常见问题

| 报错 | 原因 | 解决 |
|------|------|------|
| `Unsupported Q scale layout` | q_scale_layout 值不合法 | 只支持 `"TND"` / `"N2TGD"` |
| `libhccl.so not found` | 未 source CANN 环境变量 | `source /home/user/Ascend/cann/set_env.sh` |
| `No module named 'torch_npu'` | 未安装 torch-npu | `bash install_torch_npu.sh` |
| `No cached input: xxx_input.pt` | 缓存模式下缺少输入数据 | 先运行 `--golden-mode=gen` 生成数据 |
| `No cached CPU output` | npu/compare 模式下缺少 CPU 缓存 | 先运行 `--golden-mode=cpu` 生成 CPU 输出 |

---

## 12. 性能 Profiling（--msprof / --parse-prof / --perf-baseline）

### 12.1 概述

通过 `msprof` 工具包裹 pytest 运行，自动收集 `FusedInferAttentionScore` 算子的 profiling 数据（Duration、AI Core 时间、Cube 利用率），并支持与基线 log 进行性能回归比较。

### 12.2 一键运行 + Profiling + 报告

```bash
# 运行测试并自动收集 profiling，结束后输出报告
pytest --msprof -v -m debug

# 运行 perf_rdv 用例并收集 profiling
pytest --msprof -v -m perf_rdv
```

`--msprof` 的工作流程：
1. 快照当前目录已有的 `PROF_*` 目录
2. 用 `msprof python -m pytest ...` 包裹运行内层测试
3. 测试完成后找到新生成的 `PROF_*` 目录
4. 解析 `op_summary_*.csv`，提取 `FusedInferAttentionScore` 条目
5. 输出性能报告并归档到 `perf_output/` 目录

### 12.3 事后解析已有 PROF 目录

```bash
# 解析指定的 PROF 目录
pytest --parse-prof=./PROF_000001_20260615113759304_xxx
```

### 12.4 性能基线比较

```bash
# 运行 + profiling + 与 baseline 比较（默认 8% 阈值）
pytest --msprof --perf-baseline=./perf_baseline/perf_report_20260615113640.log -v -m debug

# 自定义阈值（5%）
pytest --msprof --perf-baseline=./perf_baseline/perf_report_xxx.log --perf-threshold=5.0 -v -m debug

# 事后解析 + 比较
pytest --parse-prof=./PROF_xxx --perf-baseline=./perf_baseline/perf_report_xxx.log
```

### 12.5 命令行选项

| 选项 | 作用 |
|------|------|
| `--msprof` | 自动用 msprof 包裹 pytest 运行，测试完成后解析 PROF 并输出报告 |
| `--parse-prof=PROF_DIR` | 解析指定的 PROF 目录（事后分析） |
| `--perf-baseline=LOG_FILE` | 性能基线 log 文件路径，与当前结果比较 Duration |
| `--perf-threshold=PERCENT` | 性能劣化阈值百分比（默认 8.0） |

### 12.6 报告输出格式

```
========================================================================================================================
  #    Duration    AI Core   Cube%  Input Shapes
------------------------------------------------------------------------------------------------------------------------
  1      21.3us     20.5us   22.4%  Q=[4, 64, 128]  K=[2, 8, 4, 512, 32]  V=[2, 8, 4, 512, 32]
  2      27.8us     27.2us   92.4%  Q=[128, 64, 128]  K=[2, 8, 512, 128]  V=[2, 8, 512, 128]
========================================================================================================================

Total: 2 entries
```

### 12.7 Baseline 比较报告

```
============================================================================================================================================
  #     Current   Baseline     Diff   Status  Input Shapes
--------------------------------------------------------------------------------------------------------------------------------------------
  1      21.3us     21.7us    -1.9%     PASS  Q=[4, 64, 128]  K=[2, 8, 4, 512, 32]  V=[2, 8, 4, 512, 32]
  2      27.8us     27.2us    +2.2%     PASS  Q=[128, 64, 128]  K=[2, 8, 512, 128]  V=[2, 8, 512, 128]
============================================================================================================================================

Total: 2 entries | PASS: 2 | FAILED: 0 | IMPROVED: 0 | NEW: 0
Threshold: 8.0%
```

| Status | 含义 |
|--------|------|
| `PASS` | Duration 变化在阈值内 |
| `FAILED` | Duration 劣化超过阈值 → pytest 退出码 1 |
| `IMPROVED` | Duration 提升超过阈值 |
| `NEW` | 当前 case 在 baseline 中不存在 |

匹配方式：按 Input Shapes（Q/K/V 的 shape 字符串）精确匹配。

### 12.8 文件归档

每次运行后，报告 log 和原始 CSV 自动归档到 `perf_output/` 目录，时间戳保持一致：

```
perf_output/
├── perf_report_20260615113640.log    # 格式化报告
└── op_summary_20260615113640.csv     # 原始 profiling 数据
```

### 12.9 建立 Baseline

1. 运行一次 profiling 生成报告：
   ```bash
   pytest --msprof -v -m debug
   ```
2. 将生成的报告拷贝到 `perf_baseline/` 目录作为基线：
   ```bash
   cp perf_output/perf_report_*.log perf_baseline/
   ```
3. 后续运行时指定该基线进行比较：
   ```bash
   pytest --msprof --perf-baseline=./perf_baseline/perf_report_20260615113640.log -v -m debug
   ```