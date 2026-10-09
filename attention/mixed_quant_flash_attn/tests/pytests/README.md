# Mixed Quant Flash Attention 测试框架

本目录提供 `mixed_quant_flash_attn` 算子的功能与性能测试，基于 pytest 实现 CPU golden 与 NPU 结果的精度对比。测试以「数据生成 → CPU 计算 → NPU 计算 → 精度对比」四阶段流水线方式组织，每个阶段独立运行、产物落盘，便于调试与复现。

## 目录结构

```
pytest/
├── README.md
├── conftest.py                          # pytest 插件：注册命令行选项与 markers
├── test_mixed_quant_flash_attn.py       # 测试入口：参数化、四阶段调度、结果判定
├── core/
│   ├── case_loader.py                   # 用例加载、参数归一化、用例 ID 过滤
│   ├── gen_data.py                      # 输入张量生成（含 fp4/fp8 量化数据）
│   ├── cpu.py                           # CPU golden 参考实现
│   ├── npu.py                           # NPU 算子调用（eager / aclgraph 两种模式）
│   ├── compare.py                       # 精度对比与报告打印
│   ├── postprocess.py                   # 用例结果汇总、CSV 导出
│   └── meta_cpu/
│       ├── mixed_quant_flash_attn_metadata_op.py   # metadata 子进程调用封装
│       └── mixed_quant_flash_attn_metadata_test   # metadata 二进制（C++）
├── testcase/
│   ├── functional_stc.py                # STC 功能用例
│   └── functional_rdv.py                # RDV 功能用例
│
├── data/
│   └── <testset>/                       # 默认 functional_stc；按用例名分子目录
│       └── <case_safe>/                 # 例 functional_stc_fp4e2m1_stc_bnsd_bf16
│           ├── <case_safe>_input.pt     # gen 阶段产物
│           ├── <case_safe>_cpu.pt       # cpu 阶段产物
│           └── <case_safe>_npu.pt       # npu 阶段产物
└── tools/
    └── precision_analysis.py            # 离线精度分析工具
```

## 核心流程

### 1. 用例发现与参数化

- `conftest.py:14` 注册命令行选项：`--mode`、`--testset`、`--case_id`、`--device_id`、`--seed`。
- `conftest.py:37` 注册 markers：`ci`、`func_rdv`、`func_redline`、`func_stc`、`perf_rdv`、`perf_redline`、`debug`。
- `test_mixed_quant_flash_attn.py:19` 维护 marker → 用例模块映射 `_MARKER_TO_MODULE`。
- `pytest_generate_tests`（`test_mixed_quant_flash_attn.py:67`）根据测试函数上的 marker 选定用例模块，再通过 `_load_cases_for_modules` 加载并校验用例（要求 `N1 >= N2` 且 `N1 % N2 == 0`），最后对 `case_name` 参数化。
- 用例定义见 `testcase/*.py` 中的 `TestCases` 字典：key 为用例名，value 为参数字典；任一参数为列表时按笛卡尔积展开为多条用例（`case_loader.py:24` `expand_cases`）。
- `normalize_params`（`case_loader.py:41`）补齐默认值：`layout_kv` 默认同 `layout_q`，`softmax_scale` 默认 `1/sqrt(D)`，`mask_mode` 默认 0，`kv_dtype` 默认 `fp4_e2m1`，并对 `TND`/`PA` layout 自动构造 `cu_seqlens_*`、`seqused_*`、`block_table` 等辅助张量。

### 2. 四阶段执行

入口为 `_execute_step`（`test_mixed_quant_flash_attn.py:106`），按 `--mode` 列表依次执行：

| Mode  | 行为                                                                                                  | 产物                              |
| ----- | --------------------------------------------------------------------------------------------------- | -------------------------------- |
| `gen` | 调 `generate_inputs(params, seed)` 生成 q/k/v/k_descale/v_descale/block_table 等，连同标量参数一起保存 | `<case>_input.pt`                |
| `cpu` | 加载 `_input.pt`，调 `run_cpu_golden`，执行分块 softmax + flash attention 参考实现                         | `<case>_cpu.pt`                  |
| `npu` | 加载 `_input.pt`，先经 `MixedQuantFlashAttnMetadataOp` 生成 metadata，再调 `torch.ops.npu_ops_transformer.mixed_quant_flash_attn`，支持 `eager`/`aclgraph` | `<case>_npu.pt` |
| `compare` | 加载 `_cpu.pt` 与 `_npu.pt`，调 `compare_results` 做精度对比并返回统计字典                          | 控制台报告 + `result.csv`         |

产物路径由 `get_data_dir`（`case_loader.py:132`）决定：`data/<testset>/<case_name.replace("/", "_")>/`。

### 3. CPU Golden 实现

`core/cpu.py` 中的 `tforward`（`cpu.py:373`）实现分块参考计算：

1. `get_axis_from_tensor` 解析 B/N1/N2/S1/S2/D/block_num/block_size，支持 `BNSD`/`BSND`/`TND` 与 `PA_NZ`/`PA_BNBD`/`PA_BBND`。
2. `quant_compute_mode == 1` 时先 `unpack_int4` 把 `uint8` 拆成两个 fp4。
3. 若 `layout_kv` 含 `PA`，`trans_pa_to_no_pa` 按 `block_table` 把 paged KV 重排为非分页布局。
4. `antiquant_data` 将 fp4 反量化为 bf16/fp16，按 `quant_compute_mode` 与 `gs_flag` 决定 group_size（32 或 4）与缩放尺度对齐方式。
5. 按 `souter`/`sinner` 分块迭代，`softmax_flash` 维护在线 softmax 的 `block_max`/`block_sum`/`block_exp`，最终 `attn_out = Σ p_block @ v_block` 并除以 `block_sum`。
6. `run_cpu_golden` 额外计算 `softmax_lse = log(x_sum) + x_max`，全掩码行置为 `inf`。

### 4. NPU 调用

`core/npu.py` 中 `run_npu`（`npu.py:138`）：

1. 把张量拷贝到 `npu:<device_id>`，标量参数原样传递。
2. 调 `MixedQuantFlashAttnMetadataOp.npu_mixed_quant_flash_attn_metadata`（`meta_cpu/mixed_quant_flash_attn_metadata_op.py:184`）：构造输入 JSON → 子进程执行 `mixed_quant_flash_attn_metadata_test` 二进制 → 解析输出 JSON 组装 `metadata` int32 列表，描述 AIC/AIV 核的 task 划分。
3. 调 `torch.ops.npu_ops_transformer.mixed_quant_flash_attn`，返回 `attn_out` 与可选 `softmax_lse`。
4. `mode == "aclgraph"` 时通过 `torchair` + `CompilerConfig` 走图编译路径（`reduce-overhead`、静态 shape、tiling schedule 优化等）。

### 5. 精度对比

`core/compare.py` 提供两个对比函数：

- `check_result(expect, result, data_type, pct_thd=0.005)`（`compare.py:175`）：numpy 实现。通过条件为 `fulfill_percent >= 99.5%` **且** `max_rel_err < 10.0`；相对误差分母取 `max(|real|, |expect|, (1/16384)/0.005) + 10e-10`，避免小数值放大。对 `bfloat16`/`float8_e4m3fn`/`float8_e5m2` 有专门处理，`np.isclose(..., equal_nan=True)`。
- `check_result_(...)`（`compare.py:18`）：与 `check_result` 同等判定逻辑，但输出带框线的中文精度报告，并返回结构化 `stats` 字典（含 `passed`/`max_abs`/`mean_abs`/`max_rel`/`mean_rel`/`fail_cnt`/`total`/`fail_ratio`/`fulfill_percent`）。

`compare_results(cpu_result, npu_result, case_name)`（`compare.py:297`）对 `attn_out` 与 `softmax_lse` 分别调用上述对比函数；tolerance 按 dtype 自动选取：

| output dtype | atol      | rtol       |
| ------------ | --------- | ---------- |
| bf16         | 0.0001    | 0.0078125  |
| fp16（默认）  | 0.000025  | 0.005      |

### 6. 结果汇总

`PostProcessor`（`core/postprocess.py`）在 session 级 fixture 中累积每条用例的判定结果：

- `record`：记录 attn/lse 是否通过及关键指标。
- `print_summary`：打印汇总表（用例名、PASS/FAIL、MaxAbsErr、FailRatio、LSE 状态）。
- `save_csv`：导出 `result.csv` 到 pytest 目录。

`test_summary`（`test_mixed_quant_flash_attn.py:246`）在 `--mode compare` 时触发汇总与 CSV 导出。

## 用例参数

用例定义在 `testcase/*.py` 的 `TestCases` 字典中，常用字段：

| 字段          | 含义                                    | 取值示例                          |
| ------------- | --------------------------------------- | --------------------------------- |
| `B`           | batch size                              | 1, 2, 4, 8                        |
| `N1` / `N2`   | Q / KV 头数，要求 `N1 % N2 == 0`        | N1=8 N2=1（GQA）, N1=8 N2=8（MHA）|
| `S1` / `S2`   | Q / KV 序列长度                         | 1, 64, 1024, 2048                 |
| `D`           | 头维度                                  | 128                               |
| `layout_q`    | Q 布局                                  | `BNSD`, `BSND`, `TND`             |
| `layout_kv`   | KV 布局                                 | `PA_NZ`, `PA_BNBD`, `PA_BBND`     |
| `q_dtype`     | Q dtype                                 | `bf16`, `fp16`                    |
| `quant_compute_mode`  | 量化模式（决定 KV dtype 与 descale）    | 1 → `fp4_e2m1` + `fp8_e8m0`       |
| `mask_mode`   | 掩码模式                                | 0 无, 3 causal, 4 band            |
| `win_left` / `win_right` | band 窗口（mask_mode=4）      | 32                                |
| `block_size`  | PA 分页块大小                           | 128, 512                          |
| `block_table` | 分页表（不填则自动随机生成）            | [[0,1,2], [3,4]]                  |
| `seqused_kv`  | 每批实际 KV 长度                        | [0,0,1024,1024]                   |
| `q_range` / `kv_range` | 数据取值范围                       | (-5.0, 5.0)                       |

支持的 KV 量化 dtype：`fp4_e2m1`、`fp4_e1m2`、`fp8_e4m3`（通过 `quant_compute_mode` 控制）。

## 命令行用法

```bash
cd pytest/

# 完整四阶段流水线（默认 functional_stc 用例集）
pytest test_mixed_quant_flash_attn.py -m func_stc --mode gen --testset functional_stc
pytest test_mixed_quant_flash_attn.py -m func_stc --mode cpu  --testset functional_stc
pytest test_mixed_quant_flash_attn.py -m func_stc --mode npu  --testset functional_stc --device_id 0
pytest test_mixed_quant_flash_attn.py -m func_stc --mode compare --testset functional_stc

# 跑指定用例（按用例名后缀过滤，支持逗号分隔多个）
pytest test_mixed_quant_flash_attn.py -m func_redline --mode gen --case_id "fp4e2m1_gqa_bf16_bnsd_causal"

# 跑 RDV 功能用例
pytest test_mixed_quant_flash_attn.py -m func_rdv --mode gen cpu npu compare --testset functional_rdv

# 跑性能用例
pytest test_mixed_quant_flash_attn.py -m perf_redline --mode npu --testset perf_redline

# 多进程并行（需 pytest-xdist）
pytest test_mixed_quant_flash_attn.py -m func_stc --mode gen -n 4

# CPU golden 单独跑（无需 pytest）
python core/cpu.py --input_dir data/functional_stc/functional_stc_fp4e2m1_stc_bnsd_bf16

# 离线精度分析（独立工具）
python tools/precision_analysis.py \
    --cpu_pt data/functional_stc/functional_stc_fp4e2m1_stc_bnsd_bf16/functional_stc_fp4e2m1_stc_bnsd_bf16_cpu.pt \
    --npu_pt data/functional_stc/functional_stc_fp4e2m1_stc_bnsd_bf16/functional_stc_fp4e2m1_stc_bnsd_bf16_npu.pt \
    --output_dir ./analysis
```

### 命令行参数

| 参数           | 默认值            | 说明                                                  |
| -------------- | ----------------- | ----------------------------------------------------- |
| `--mode`       | `gen`             | 可多选：`gen`、`cpu`、`npu`、`compare`                |
| `--testset`    | `functional_stc`  | 数据目录名，对应 `data/<testset>/`                    |
| `--case_id`    | `all`             | 用例过滤：`all` 或逗号分隔的用例名                    |
| `--device_id`  | `0`               | NPU 设备号                                            |
| `--seed`       | `21`              | 数据生成随机种子                                      |
| `-m`           | (pytest 内置)     | marker 过滤，如 `func_stc`、`perf_rdv`、`ci`、`debug`|
| `-n`           | (pytest-xdist)    | 并行进程数                                            |

## Markers

| Marker          | 对应用例模块            | 说明                       |
| --------------- | ----------------------- | -------------------------- |
| `func_stc`      | `functional_stc`        | STC 功能正确性             |
| `func_redline`  | `functional_redline`    | 功能红线                   |
| `ci`            | (叠加在 func_rdv 上)    | CI 门禁用                  |
| `debug`         | (无对应模块，全量跑)    | 日常调试                   |

## 注意事项

1. `--mode` 可一次传多个值，例如 `--mode gen cpu npu compare` 会按顺序在同一用例上跑完整流水线；但产物需前序阶段已落盘，跨 pytest 进程调度时建议分阶段执行。
2. `compare` 阶段若 `_cpu.pt` 或 `_npu.pt` 缺失会 `pytest.skip`。
3. NPU 阶段依赖 `torch_npu`、`torchair`、`npu_ops_transformer`，以及 `core/meta_cpu/mixed_quant_flash_attn_metadata_test` 二进制；CPU 阶段仅依赖 `torch`、`numpy`、`ml_dtypes`。
4. `aclgraph` 模式会在当前目录生成静态 shape 编译产物，调试时优先用 `eager`。
5. fp4 数据以 `uint8` 打包存储（两个 fp4 共用一个 byte），CPU/NPU 侧均需先 `unpack_int4`。
