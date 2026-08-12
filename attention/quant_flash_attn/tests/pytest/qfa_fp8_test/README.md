# QFA FullQuant GQA 测试用例执行指南

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
cd attention/quant_flash_attn/tests/pytest/quant_flash_attn_fp8_test
```

---

## 2. 文件结构

```
quant_flash_attn_fp8_test/
├── pytest.ini                                  # pytest 配置（自定义 marker）
├── conftest.py                                 # pytest 命令行选项（--golden-mode, --cache-dir, --msprof, --parse-prof, --perf-baseline）
├── common/
│   ├── __init__.py
│   ├── quant_flash_attn_fp8_golden.py         # CPU golden 参考实现 + QFA NPU 双算子调用
│   ├── golden_cache.py                         # .pt 缓存工具模块
│   ├── result_compare_method.py                # 精度对比工具
│   ├── perf_parser.py                          # msprof op_summary.csv 解析 + baseline 比较
│   └── test_runner.py                          # 共享测试执行逻辑（apply_params / execute_test / check_results）
├── quant_flash_attn_fp8_paramset_common.py        # 参数展开公共逻辑 + 默认值
├── quant_flash_attn_fp8_paramset_func_rdv.py      # 功能正确性参数集（12 条代表性用例）
├── test_quant_flash_attn_fp8_func_rdv.py          # 功能正确性测试入口
└── README.md                                   # 本文件
```

---

## 3. 执行测试

### 3.1 测试入口

| 测试文件 | Marker | 参数集 | 用途 |
|----------|--------|--------|------|
| `test_quant_flash_attn_fp8_func_rdv.py` | `@pytest.mark.func_rdv` `@pytest.mark.ci` | func_rdv（12 条） | 功能正确性验证 |

### 3.2 基本执行

```bash
# 运行全部功能正确性用例
pytest test_quant_flash_attn_fp8_func_rdv.py -v

# 按 marker 运行
pytest -m func_rdv -v
pytest -m ci -v
```

### 3.3 过滤用例（-k）

```bash
# 精确指定单个用例
pytest test_quant_flash_attn_fp8_func_rdv.py -v -k "B1_G2_Nq32_Nkv16_D128_SM3_LSE1_Q2048_KV2048"

# 按 GQA 比例
pytest -v -k "G2"
pytest -v -k "G28"

# 按 batch
pytest -v -k "B4"

# 组合过滤
pytest -v -k "B1 and LSE1"
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
# 仅生成并保存输入数据
pytest test_quant_flash_attn_fp8_func_rdv.py -v --golden-mode=gen

# 仅跑 NPU + 精度对比
pytest test_quant_flash_attn_fp8_func_rdv.py -v --golden-mode=npu,compare

# 全流程
pytest test_quant_flash_attn_fp8_func_rdv.py -v --golden-mode=gen,cpu,npu,compare
```

### 4.3 自定义缓存目录

```bash
pytest test_quant_flash_attn_fp8_func_rdv.py -v --cache-dir=/tmp/my_cache
```

默认缓存目录为 `common/golden_cache/`。

### 4.4 缓存文件命名

每个 case 生成 3 个 `.pt` 文件：

```
golden_cache/
├── {case_name}_input.pt        # 输入数据（Q/K/V fp8 + deq_scale + p_scale + block_table）
├── {case_name}_cpu_output.pt   # CPU golden 输出（atten_out + lse）
└── {case_name}_npu_output.pt   # NPU 算子输出（atten_out + lse）
```

---

## 5. 独立脚本运行

`common/quant_flash_attn_fp8_golden.py` 也支持独立运行：

```bash
cd common/
python quant_flash_attn_fp8_golden.py --mode all --case-name my_case
python quant_flash_attn_fp8_golden.py --mode gen --case-name my_case
python quant_flash_attn_fp8_golden.py --mode npu,compare --case-name my_case
python quant_flash_attn_fp8_golden.py --mode cpu --case-name my_case --cache-dir=/tmp/cache
```

---

## 6. 用例命名规则

格式：`B{batch}_G{gqa_ratio}_Nq{Q头数}_Nkv{KV头数}_D{维度}_SM{mask_mode}_LSE{0/1}_Q{Q长度}_KV{KV长度}[_P{p_scale}]`

| 字段 | 含义 | 示例 |
|------|------|------|
| `B` | batch size | `B1`, `B4` |
| `G` | GQA 比例 (N_q / N_kv) | `G2`, `G8`, `G28` |
| `Nq` | Q 头数 | `Nq32`, `Nq80` |
| `Nkv` | KV 头数 | `Nkv2`, `Nkv16` |
| `D` | Head 维度（固定 128） | `D128` |
| `SM` | mask_mode | `SM0`（NO_MASK）, `SM3`（CAUSAL） |
| `LSE` | 是否返回 LSE | `LSE0`, `LSE1` |
| `Q` | 最大 Q 序列长度 | `Q64`, `Q2048` |
| `KV` | 最大 KV 序列长度 | `KV257`, `KV5119` |
| `P` | p_scale 值 | `_P15`, `_P256` |

---

## 7. 参数体系

### 7.1 GQA 固定参数

以下参数在 QFA GQA 场景中固定，不支持修改：

| 参数 | 固定值 | 说明 |
|------|--------|------|
| `quant_mode` | 6 | GQA_FP8_FULLQUANT |
| `layout_q` | `"NTD"` | Q 输入布局 |
| `layout_q_descale` | `"NT"` | Q descale 布局（2D [N1,T]） |
| `layout_kv` | `"PA_BNBD"` | KV Cache 布局 |
| `layout_out` | `"TND"` | 输出布局 |
| `kv_cache_layout` | `"BnNBsD"` | KV Cache 数据排布 |
| `block_size` | 128 | PA block 大小 |
| `D` | 128 | Head 维度 |
| `enable_pa` | True | 强制 PagedAttention |

### 7.2 可变参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `B` | int | 必填 | batch size |
| `N_q` | int | 必填 | Q 头数 |
| `N_kv` | int | 必填 | KV 头数（N_q % N_kv == 0） |
| `actual_seq_q` | list | 必填 | 每个 batch 的 Q 序列长度 |
| `actual_seq_kv` | list | 必填 | 每个 batch 的 KV 序列长度 |
| `enable_lse` | bool | True | 是否返回 Log-Sum-Exp |
| `mask_mode` | int | 3 | 0=NO_MASK, 3=CAUSAL |
| `p_scale` | float | 1.0 | P 量化 scale |
| `scale_value` | float | None | softmax_scale（None 时用 1/sqrt(D)） |
| `is_contiguous` | bool | True | 输入是否 contiguous |
| `num_blocks` | int | 0 | 物理 block 数量（0=无复用） |
| `graph_path` | int | 0 | 图模式（0=单算子） |
| `device_id` | int | 0 | NPU 设备 ID |
| `q_data_range` | tuple | (-1.0, 1.0) | Q 数据范围 |
| `k_data_range` | tuple | (-1.0, 1.0) | K 数据范围 |
| `v_data_range` | tuple | (-1.0, 1.0) | V 数据范围 |
| `seed_q` | int | 54 | Q 随机种子 |
| `seed_k` | int | 3 | K 随机种子 |
| `seed_v` | int | 20 | V 随机种子 |
| `seed_block_table` | int | 1234 | block_table 随机种子 |

### 7.3 参数展开

`expand_paramset_to_cases()` 对每个配置中的所有维度做笛卡尔积展开。固定参数由 `TEST_PARAMS_DEFAULTS` 自动填充。

---

## 8. 新增用例

在 `quant_flash_attn_fp8_paramset_func_rdv.py` 中添加新条目：

```python
"B1_G4_Nq8_Nkv2_D128_SM0_LSE1_Q256_KV1024": {
    "B": [1],
    "N_q": [8],
    "N_kv": [2],
    "actual_seq_q": [[256]],
    "actual_seq_kv": [[1024]],
    "mask_mode": [0],
    "enable_lse": [True],
    "p_scale": [1.0],
},
```

### SKIP_CASES 机制

```python
SKIP_CASES = {
    "B4_G52_Nq52_Nkv1_D128_SM0_LSE0_Q257_4b_KV127_4b",
}
```

被标记的用例会自动添加 `pytest.mark.skip`。

---

## 9. NUM_BLOCKS 物理块复用

默认 `num_blocks=0` 表示物理 block 数量等于实际需要的 block 总数（无复用）。

设置 `num_blocks=N`（N < total_blocks）时，会触发物理 block 复用。test_runner 会在 NPU 执行后从 cache 还原 BNSD 数据，重新跑 CPU golden 进行对比。

---

## 10. 精度对比标准

使用"双千分之五"标准：

| 指标 | BF16 阈值 |
|------|-----------|
| rtol（相对容差） | 0.0078125 |
| atol（绝对容差） | 0.0001 |
| pct_thd（通过率阈值） | 99.5% |
| max_diff_hd（最大相对误差上限） | 10 |

判定逻辑：通过率 >= 99.5% **且** 最大相对误差 < 10 时为 Pass。

---

## 11. pytest Marker 说明

| Marker | 用途 |
|--------|------|
| `ci` | CI 流水线测试 |
| `func_rdv` | 功能正确性 RDV 测试 |
| `perf_rdv` | 性能/压力 RDV 测试 |
| `debug` | 日常调试测试 |
| `graph` | 图模式编译测试 |
| `npu_only` | 仅 NPU 执行（无 CPU golden 对比） |

---

## 12. 性能 Profiling（--msprof / --parse-prof / --perf-baseline）

### 12.1 一键运行 + Profiling + 报告

```bash
# 运行测试并自动收集 profiling
pytest --msprof -v -m func_rdv
```

### 12.2 事后解析已有 PROF 目录

```bash
pytest --parse-prof=./PROF_000001_20260615113759304_xxx
```

### 12.3 性能基线比较

```bash
pytest --msprof --perf-baseline=./perf_baseline/perf_report_xxx.log -v -m func_rdv
pytest --msprof --perf-baseline=./perf_baseline/perf_report_xxx.log --perf-threshold=5.0 -v -m func_rdv
```

| 选项 | 作用 |
|------|------|
| `--msprof` | 自动用 msprof 包裹 pytest 运行 |
| `--parse-prof=PROF_DIR` | 解析指定的 PROF 目录 |
| `--perf-baseline=LOG_FILE` | 性能基线 log 文件路径 |
| `--perf-threshold=PERCENT` | 性能劣化阈值百分比（默认 8.0） |
