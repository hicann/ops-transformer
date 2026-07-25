# FIA FullQuant GQA 测试用例执行指南

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
cd attention/fused_infer_attention_score/tests/pytest/fia_fullquant_gqa_test
```

---

## 2. 文件结构

```
fia_fullquant_gqa_test/
├── pytest.ini                                  # pytest 配置（自定义 marker）
├── conftest.py                                 # pytest 命令行选项（--golden-mode, --cache-dir, --use-external-input, --load-pt-dir, --msprof, --parse-prof, --perf-baseline）
├── common/
│   ├── __init__.py
│   ├── fia_fullquant_gqa_golden.py             # CPU golden 参考实现 + NPU 算子调用
│   ├── golden_cache.py                         # .pt 缓存工具模块
│   ├── result_compare_method.py                # 精度对比工具
│   ├── perf_parser.py                          # msprof op_summary.csv 解析 + baseline 比较
│   ├── load_external_data.py                   # 外部 NPU 排布 pt 文件加载
│   └── test_runner.py                          # 共享测试执行逻辑（apply_params / execute_test / check_results）
├── fia_fullquant_gqa_paramset_common.py        # 参数展开公共逻辑 + 默认值
├── fia_fullquant_gqa_paramset_debug.py         # debug 参数集（少量用例，快速验证）
├── fia_fullquant_gqa_paramset_func_rdv.py      # 功能正确性参数集
├── fia_fullquant_gqa_paramset_perf_rdv.py      # 性能/压力参数集
├── test_fia_fullquant_gqa_debug.py             # debug 测试入口
├── test_fia_fullquant_gqa_func_rdv.py          # 功能正确性测试入口
└── test_fia_fullquant_gqa_perf_rdv.py          # 性能/压力测试入口
```

---

## 3. 执行测试

### 3.1 三个测试入口

| 测试文件 | Marker | 参数集 | 用途 |
|----------|--------|--------|------|
| `test_fia_fullquant_gqa_debug.py` | `@pytest.mark.debug` | debug（2 条） | 快速验证基本功能 |
| `test_fia_fullquant_gqa_func_rdv.py` | `@pytest.mark.func_rdv` `@pytest.mark.ci` | func_rdv | 功能正确性全覆盖 |
| `test_fia_fullquant_gqa_perf_rdv.py` | `@pytest.mark.perf_rdv` | perf_rdv | 性能/压力验证 |

### 3.2 基本执行

```bash
# 运行 debug 用例
pytest test_fia_fullquant_gqa_debug.py -v

# 运行功能正确性用例
pytest test_fia_fullquant_gqa_func_rdv.py -v

# 运行性能/压力用例
pytest test_fia_fullquant_gqa_perf_rdv.py -v

# 按 marker 运行
pytest -m debug -v
pytest -m func_rdv -v
pytest -m perf_rdv -v
pytest -m ci -v
```

### 3.3 过滤用例（-k）

```bash
# 精确指定单个用例
pytest test_fia_fullquant_gqa_func_rdv.py -v -k "B1_G8_Nq16_Nkv2_D128_SM3_LSE1_Q2048_KV2048"

# 按 Prefill / Decode 模式
pytest test_fia_fullquant_gqa_perf_rdv.py -v -k "Prefill"
pytest test_fia_fullquant_gqa_perf_rdv.py -v -k "Decode"

# 按 GQA 比例
pytest -v -k "G64"
pytest -v -k "G128"

# 组合过滤
pytest -v -k "B1 and G8"

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
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=gen

# 仅跑 NPU + 精度对比（CPU 输出从缓存加载）
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=npu,compare

# 跑 CPU + NPU（不对比）
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=cpu,npu

# 全流程（等同于 --golden-mode=all）
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=gen,cpu,npu,compare
```

### 4.3 典型工作流

**方式一：全流程一次跑完（默认）**

```bash
pytest test_fia_fullquant_gqa_func_rdv.py -v
```

**方式二：分步执行（适合跨机器调试）**

```bash
# 第一步：生成输入数据（不跑 CPU/NPU）
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=gen

# 第二步：跑 CPU golden
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=cpu

# 第三步：在 NPU 机器上跑 NPU + 精度对比
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=npu,compare
```

### 4.4 自定义缓存目录

```bash
pytest test_fia_fullquant_gqa_func_rdv.py -v --cache-dir=/tmp/my_cache
```

默认缓存目录为 `common/golden_cache/`。

### 4.5 缓存文件命名

每个 case 生成 3 个 `.pt` 文件：

```
golden_cache/
├── {case_name}_input.pt        # 输入数据（Q/K/V fp8 + deq_scale + p_scale + block_table）
├── {case_name}_cpu_output.pt   # CPU golden 输出（atten_out + lse）
└── {case_name}_npu_output.pt   # NPU 算子输出（atten_out + lse）
```

---

## 5. 外部 NPU 排布数据加载（--use-external-input / --load-pt-dir）

支持从外部 pt 文件加载 NPU 排布的数据（替代 `generate_data`），适用于从真实模型 dump 数据复现问题。

### 5.1 目录结构要求

```
<pt_dir>/
├── rank0_query_ntd.pt        # NTD (N_q, T, D)
├── rank0_key_cache.pt        # BNBD (block_num, N_kv, block_size, D)
├── rank0_value_cache.pt      # BNBD (block_num, N_kv, block_size, D)
├── rank0_q_scale.pt          # NT (N_q, T)
├── rank0_k_scale_cache.pt    # BNB (block_num, N_kv, block_size)
├── rank0_v_scale_cache.pt    # N (N_kv,)
├── rank0_block_table.pt      # (B, max_blocks) int32
├── rank0_seq_lens.pt         # list of int
└── rank0_attn_mask.pt        # [可选]
```

### 5.2 使用方式

```bash
# 仅生成阶段使用外部数据，之后跑 CPU + NPU + 对比
pytest test_fia_fullquant_gqa_debug.py -v \
    --use-external-input \
    --load-pt-dir=/path/to/external_pt \
    --golden-mode=gen,cpu,npu,compare

# 分步：先用外部数据生成缓存
pytest test_fia_fullquant_gqa_debug.py -v \
    --use-external-input \
    --load-pt-dir=/path/to/external_pt \
    --golden-mode=gen

# 后续步骤不需要再指定 --use-external-input（从缓存加载）
pytest test_fia_fullquant_gqa_debug.py -v --golden-mode=cpu,npu,compare
```

---

## 6. 独立脚本运行

`common/fia_fullquant_gqa_golden.py` 也支持独立运行，同样支持 `--mode` 组合：

```bash
cd common/
python fia_fullquant_gqa_golden.py --mode all --case-name my_case
python fia_fullquant_gqa_golden.py --mode gen --case-name my_case
python fia_fullquant_gqa_golden.py --mode npu,compare --case-name my_case
python fia_fullquant_gqa_golden.py --mode cpu --case-name my_case --cache-dir=/tmp/cache

# 使用外部数据
python fia_fullquant_gqa_golden.py --mode gen --case-name my_case \
    --use-external-input --load-pt-dir=/path/to/external_pt
```

---

## 7. 用例命名规则

格式：`B{batch}_G{gqa_ratio}_Nq{Q头数}_Nkv{KV头数}_D{维度}_SM{稀疏模式}_LSE{0/1}_Q{Q长度}_KV{KV长度}[_{标签}]`

| 字段 | 含义 | 示例 |
|------|------|------|
| `B` | batch size | `B1`, `B8` |
| `G` | GQA 比例 (N_q / N_kv) | `G1`, `G8`, `G128` |
| `Nq` | Q 头数 | `Nq16` |
| `Nkv` | KV 头数 | `Nkv2` |
| `D` | Head 维度 | `D128`, `D64` |
| `SM` | sparse_mode | `SM0`, `SM3` |
| `LSE` | 是否返回 LSE | `LSE0`, `LSE1` |
| `Q` | 最大 Q 序列长度 | `Q128`, `Q2048` |
| `KV` | 最大 KV 序列长度 | `KV256`, `KVS8192` |
| `Prefill` | 长序列 prefill 阶段 | `_Prefill` |
| `Decode` | 短 Q 长 KV decode 阶段 | `_Decode` |
| `noncontig` | IS_CONTIGUOUS=False | `_noncontig` |
| `numblocks{n}` | NUM_BLOCKS 物理块复用 | `_numblocks4` |
| `pscale{n}` | 自定义 p_scale（1.0/15.0/100.0/128.0/256.0） | `_pscale15`, `_pscale128` |
| `block{n}` | 自定义 block_size | `_block64` |

---

## 8. 参数体系

### 8.1 全部参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `B` | int | 必填 | batch size |
| `N_q` | int | 必填 | Q 头数 |
| `N_kv` | int | 必填 | KV 头数（N_q 需能被 N_kv 整除） |
| `D` | int | 128 | Head 维度（64 / 128） |
| `actual_seq_q` | list | 必填 | 每个 batch 的 Q 序列长度 |
| `actual_seq_kv` | list | 必填 | 每个 batch 的 KV 序列长度 |
| `enable_pa` | bool | True | 是否启用 PagedAttention |
| `enable_lse` | bool | True | 是否返回 Log-Sum-Exp |
| `golden_mode` | bool | True | 是否执行 CPU golden 对比 |
| `block_size` | int | 128 | PA block 大小 |
| `sparse_mode` | int | 3 | 稀疏模式（0=无 mask, 3=causal+padding） |
| `input_layout` | str | NTD_TND | Q 输入布局 |
| `output_layout` | str | TND | 输出布局 |
| `q_scale_layout` | str | NT | Q scale 布局 |
| `kv_cache_layout` | str | BnNBsD | KV Cache 布局 |
| `p_scale` | float | 1.0 | P 量化 scale |
| `scale_value` | float | None | softmax_scale（None 时用 1/sqrt(D)） |
| `is_contiguous` | bool | True | 输入是否 contiguous |
| `num_blocks` | int | 0 | 物理 block 数量（0=默认=total_blocks） |
| `graph_path` | int | 0 | 图模式（0=单算子, 5=动态图, 7=aclgraph） |
| `device_id` | int | 0 | NPU 设备 ID |
| `q_data_range` | tuple | (-1.0, 1.0) | Q 数据范围 |
| `k_data_range` | tuple | (-1.0, 1.0) | K 数据范围 |
| `v_data_range` | tuple | (-1.0, 1.0) | V 数据范围 |
| `seed_q` | int | 54 | Q 随机种子 |
| `seed_k` | int | 3 | K 随机种子 |
| `seed_v` | int | 20 | V 随机种子 |
| `seed_block_table` | int | 1234 | block_table 随机种子 |

### 8.2 默认值机制

`TEST_PARAMS_DEFAULTS` 中的参数如果在 paramset 中未显式指定，会自动使用默认值。例如 `D`、`block_size`、`graph_path`、`device_id` 等通常不需要每个 case 都写。

如需某个 case 使用非默认值，只需在 paramset 中显式指定：

```python
"MY_CASE": {
    "B": [1],
    ...
    "graph_path": [7],       # 使用 aclgraph 模式
    "D": [64],               # 使用 D=64
},
```

### 8.3 参数展开

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

## 9. 新增用例

在对应的 paramset 文件中添加新条目。以 `fia_fullquant_gqa_paramset_func_rdv.py` 为例：

```python
"B1_G4_Nq8_Nkv2_D128_SM3_LSE1_Q256_KV1024": {
    "B": [1],
    "N_q": [8],
    "N_kv": [2],
    "actual_seq_q": [[256]],
    "actual_seq_kv": [[1024]],
    "sparse_mode": [3],
    "enable_lse": [True],
    "p_scale": [1.0],
},
```

### B>1 不等长序列

当 B>1 时，`actual_seq_q` 和 `actual_seq_kv` 应为不等长列表：

```python
"B4_G4_Nq16_Nkv4_D128_SM3_LSE1_Q256_KV512_unequal": {
    "B": [4],
    ...
    "actual_seq_q": [[128, 256, 192, 256]],
    "actual_seq_kv": [[512, 384, 640, 512]],
},
```

### SKIP_CASES 机制

对于执行时间过长的用例，可在 paramset 文件中定义 `SKIP_CASES` 集合：

```python
SKIP_CASES = {
    "B128_G8_Nq16_Nkv2_D128_SM3_LSE0_Q1_KVS16384_Decode",
}
```

被标记的用例会自动添加 `pytest.mark.skip`，在 pytest 输出中显示为 `SKIPPED`。

---

## 10. NUM_BLOCKS 物理块复用

默认 `num_blocks=0` 表示物理 block 数量等于实际需要的 block 总数（无复用）。

设置 `num_blocks=N`（N < total_blocks）时，会触发物理 block 复用：cache 中部分数据会被覆盖。此时 test_runner 会在 NPU 执行后从 cache 还原 BNSD 数据，重新跑 CPU golden 进行对比，确保对比的是 NPU 实际看到的数据。

```python
"B1_G4_Nq4_Nkv1_D128_SM3_LSE1_Q128_KV1024_numblocks4": {
    ...
    "num_blocks": [4],
},
```

---

## 11. 精度对比标准

使用"双千分之五"标准：

| 指标 | FP16 阈值 | BF16 阈值 |
|------|-----------|-----------|
| rtol（相对容差） | 0.005 | 0.0078125 |
| atol（绝对容差） | 0.000025 | 0.0001 |
| pct_thd（通过率阈值） | 99.5% | 99.5% |
| max_diff_hd（最大相对误差上限） | 10 | 10 |

判定逻辑：通过率 >= 99.5% **且** 最大相对误差 < 10 时为 Pass。

---

## 12. pytest Marker 说明

| Marker | 用途 |
|--------|------|
| `ci` | CI 流水线测试 |
| `func_rdv` | 功能正确性 RDV 测试 |
| `perf_rdv` | 性能/压力 RDV 测试 |
| `debug` | 日常调试测试 |
| `graph` | 图模式编译测试 |
| `npu_only` | 仅 NPU 执行（无 CPU golden 对比） |

---

## 13. 常见问题

| 报错 | 原因 | 解决 |
|------|------|------|
| `libhccl.so not found` | 未 source CANN 环境变量 | `source /home/user/Ascend/cann/set_env.sh` |
| `No module named 'torch_npu'` | 未安装 torch-npu | `bash install_torch_npu.sh` |
| `No cached input: xxx_input.pt` | 缓存模式下缺少输入数据 | 先运行 `--golden-mode=gen` 生成数据 |
| `No cached CPU output` | npu/compare 模式下缺少 CPU 缓存 | 先运行 `--golden-mode=cpu` 生成 CPU 输出 |
| `--use-external-input requires --load-pt-dir` | 使用外部数据但未指定目录 | 添加 `--load-pt-dir=/path/to/pt` |

---

## 14. 性能 Profiling（--msprof / --parse-prof / --perf-baseline）

### 14.1 概述

通过 `msprof` 工具包裹 pytest 运行，自动收集 `FusedInferAttentionScore` 算子的 profiling 数据（Duration、AI Core 时间、Cube 利用率），并支持与基线 log 进行性能回归比较。

### 14.2 一键运行 + Profiling + 报告

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

### 14.3 事后解析已有 PROF 目录

```bash
# 解析指定的 PROF 目录
pytest --parse-prof=./PROF_000001_20260615113759304_xxx
```

### 14.4 性能基线比较

```bash
# 运行 + profiling + 与 baseline 比较（默认 8% 阈值）
pytest --msprof --perf-baseline=./perf_baseline/perf_report_20260615113640.log -v -m debug

# 自定义阈值（5%）
pytest --msprof --perf-baseline=./perf_baseline/perf_report_xxx.log --perf-threshold=5.0 -v -m debug

# 事后解析 + 比较
pytest --parse-prof=./PROF_xxx --perf-baseline=./perf_baseline/perf_report_xxx.log
```

### 14.5 命令行选项

| 选项 | 作用 |
|------|------|
| `--msprof` | 自动用 msprof 包裹 pytest 运行，测试完成后解析 PROF 并输出报告 |
| `--parse-prof=PROF_DIR` | 解析指定的 PROF 目录（事后分析） |
| `--perf-baseline=LOG_FILE` | 性能基线 log 文件路径，与当前结果比较 Duration |
| `--perf-threshold=PERCENT` | 性能劣化阈值百分比（默认 8.0） |

### 14.6 建立 Baseline

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
