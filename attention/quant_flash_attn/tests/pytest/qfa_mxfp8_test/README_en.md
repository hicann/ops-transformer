<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# QFA MXFP8 Test Case Execution Guide

English | [简体中文](./README.md)

## 1. Environment Preparation

### 1.1 Installing CANN / PyTorch + TorchNPU

Refer to the installation instructions in the README.md in the root directory.

### 1.2 Loading the Environment Before Each Execution

```bash
source /home/user/Ascend/cann/set_env.sh
conda activate your-env-name
pip install pytest
cd attention/quant_flash_attn/tests/pytest/qfa_mxfp8_test
```

---

## 2. File Structure

```
qfa_mxfp8_test/
├── pytest.ini                                  # pytest configuration (custom markers)
├── conftest.py                                 # pytest command-line options (--golden-mode, --cache-dir, --msprof, --parse-prof, --perf-baseline)
├── common/
│   ├── __init__.py
│   ├── quant_flash_attn_golden.py              # CPU golden reference implementation + NPU operator invocation
│   ├── golden_cache.py                         # .pt cache utility module
│   ├── result_compare_method.py                # Accuracy comparison tool
│   ├── precision_compare_v2.py                 # Three-way comparison tool (method=12 CV dual benchmark, see section 5.1)
│   ├── perf_parser.py                          # msprof op_summary.csv parsing + baseline comparison
│   └── test_runner.py                          # Shared test execution logic (apply_params / execute_test / check_results)
├── quant_flash_attn_paramset_common.py         # Common parameter expansion logic + default values
├── quant_flash_attn_paramset_debug.py          # debug parameter set (a few cases for quick verification)
├── quant_flash_attn_paramset_func_rdv.py       # Functional correctness parameter set (~150 cases)
├── quant_flash_attn_paramset_perf_rdv.py       # Performance/stress parameter set (~100 cases)
├── test_quant_flash_attn_debug.py              # debug test entry
├── test_quant_flash_attn_func_rdv.py           # Functional correctness test entry
└── test_quant_flash_attn_perf_rdv.py           # Performance/stress test entry
```

---

## 3. Running Tests

### 3.1 Three Test Entries

| Test File | Marker | Parameter Set | Purpose |
|----------|--------|--------|------|
| `test_quant_flash_attn_debug.py` | `@pytest.mark.debug` | debug (2 cases) | Quick verification of basic functions |
| `test_quant_flash_attn_func_rdv.py` | `@pytest.mark.func_rdv` `@pytest.mark.ci` | func_rdv (~150 cases) | Full functional correctness coverage |
| `test_quant_flash_attn_perf_rdv.py` | `@pytest.mark.perf_rdv` | perf_rdv (~100 cases) | Performance/stress verification |

### 3.2 Basic Execution

```bash
# Run debug cases
pytest test_quant_flash_attn_debug.py -v

# Run functional correctness cases
pytest test_quant_flash_attn_func_rdv.py -v

# Run performance/stress cases
pytest test_quant_flash_attn_perf_rdv.py -v

# Run by marker
pytest -m debug -v
pytest -m func_rdv -v
pytest -m perf_rdv -v
pytest -m ci -v
```

### 3.3 Filtering Cases (-k)

```bash
# Specify a single case exactly
pytest test_quant_flash_attn_func_rdv.py -v -k "PA_NZ_B1_QS1_KVS512_Nq1_Nkv1_D128_SP3"

# By Prefill / Decode mode
pytest test_quant_flash_attn_perf_rdv.py -v -k "Prefill"
pytest test_quant_flash_attn_perf_rdv.py -v -k "Decode"

# By layout type
pytest -v -k "PA_NZ"
pytest -v -k "BnNBsD"

# Combined filtering
pytest -v -k "PA_NZ and Prefill"

# Exclude certain cases
pytest -v -k "not Decode"
```

### 3.4 Common Options

| Option | Description |
|------|------|
| `-v` | Verbose output, displaying each case name |
| `-s` | Does not truncate stdout/stderr |
| `-x` | Stops immediately when a failure occurs |
| `--tb=long` | Displays the full stack trace on failure |
| `-m ci` | Runs only cases marked with CI |

---

## 4. Golden Cache Mode (--golden-mode)

The tests support saving input data, CPU golden output, and NPU output as `.pt` files. In the next run, data generation can be skipped, and the cache can be loaded directly to execute the specified steps.

### 4.1 Mode Description

| Mode | Description |
|------|------|
| `all` | Full flow: data generation → CPU → NPU → accuracy comparison (**default**) |
| `gen` | Generates and saves input data only |
| `cpu` | Loads the input cache → runs CPU and saves the output |
| `npu` | Loads the input cache → runs NPU and saves the output |
| `compare` | Loads the cached CPU/NPU output → performs accuracy comparison |

### 4.2 Combined Modes

Modes support comma-separated combinations and are executed in the order gen → cpu → npu → compare:

```bash
# Generate and save input data only (CPU/NPU is not run)
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=gen

# Run NPU + accuracy comparison only (the CPU output is loaded from the cache)
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=npu,compare

# Run CPU + NPU (no comparison)
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=cpu,npu

# Full flow (equivalent to --golden-mode=all)
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=gen,cpu,npu,compare
```

### 4.3 Typical Workflows

**Method 1: Run the full flow in one go (default)**

```bash
pytest test_quant_flash_attn_func_rdv.py -v
```

**Method 2: Step-by-step execution (suitable for cross-machine debugging)**

```bash
# Step 1: Generate input data (CPU/NPU is not run)
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=gen

# Step 2: Run the CPU golden
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=cpu

# Step 3: Run NPU + accuracy comparison on the NPU machine
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=npu,compare
```

### 4.4 Custom Cache Directory

```bash
pytest test_quant_flash_attn_func_rdv.py -v --cache-dir=/tmp/my_cache
```

The default cache directory is `common/golden_cache/`.

### 4.5 Cache File Naming

Each case generates 3 `.pt` files:

```
golden_cache/
├── {case_name}_input.pt        # Input data (Q/K/V fp8 + scale + block_table)
├── {case_name}_cpu_output.pt   # CPU golden output (atten_out + lse)
└── {case_name}_npu_output.pt   # NPU operator output (atten_out + lse)
```

---

## 5. Standalone Script Execution

`common/quant_flash_attn_golden.py` also supports standalone execution and likewise supports `--mode` combinations:

```bash
cd common/
python quant_flash_attn_golden.py --mode all --case-name my_case
python quant_flash_attn_golden.py --mode gen --case-name my_case
python quant_flash_attn_golden.py --mode npu,compare --case-name my_case
python quant_flash_attn_golden.py --mode cpu --case-name my_case --cache-dir=/tmp/cache
```

### 5.1 Three-Way Accuracy Comparison (bench golden | third_party sectional reference | NPU)

When a case fails the accuracy check (especially in scenarios where the QKV input range is not [-1,1] (`DATA_RANGE_Q/K/V > 1.0`) and FD is enabled), you can use the three-way comparison to determine whether the NPU output deviation is within a reasonable range. The three roles are exactly the same as those of TTK `--compare cross_check`:

- golden (ground truth/error denominator): `cpu_mxfp8_bench`, FP64 arithmetic with FP8 quantization that preserves P;
- third_party (upper bound of competitor noise): `cpu_mxfp8_sectioned_golden`, sectional quantization reference;
- actual: NPU output.

The decision follows the TTK `CrossCheckComparison` criteria (`precision_compare_v2.cross_check_compare`): in the large domain, `mare/mere/rmse <= 10/2/2`; in the small domain, `err(actual)/err(third_party) <= 2`; NaN/Inf must be consistent with either golden or third_party.

Usage: Edit the configuration area in `common/quant_flash_attn_golden.py`, align the case parameters (B/N_q/N_kv/D, `CU_SEQLENS_*`/`SEQUSED_*`, `ENABLE_PA`, `KV_CACHE_LAYOUT`, `BLOCK_SIZE`, `SPARSE_MODE`, `Q_SCALE_LAYOUT`, `P_SCALE`, `DATA_RANGE_Q/K/V`, `ENABLE_LSE`, and so on) with the failed case, and set:

```python
USE_FP64_GOLDEN = True   # Enable three-way cross_check
DATA_RANGE_Q = 8.0       # Example: input range other than [-1,1]
DATA_RANGE_K = 4.0
DATA_RANGE_V = 2.0
```

Then run:

```bash
cd common/
# Full flow: data generation → FP32 golden cache → NPU → three-way cross_check
python quant_flash_attn_golden.py --mode all --case-name my_case
# Step-by-step run (the NPU output and golden are both cached as .pt files, so the comparison can be rerun separately)
python quant_flash_attn_golden.py --mode gen,cpu --case-name my_case
python quant_flash_attn_golden.py --mode npu,compare --case-name my_case
```

Output description:
- `[Compare] Atten OUT / LSE`: three-way cross_check result (status + ratio metrics)
- LSE (when `ENABLE_LSE=True`): the same three-way comparison, aligned in N-major arrangement
- When `USE_FP64_GOLDEN=False`, the original two-way simple comparison is used (`check_result`, FP32 golden vs NPU)

Cache files: `{case_name}_input.pt`, `{case_name}_cpu_output.pt` (FP32 two-way baseline), and `{case_name}_npu_output.pt`. The bench and sectional reference used by the three-way comparison are computed on the fly in the compare stage and are not written to disk.

### 5.2 Running the Three-Way Comparison through pytest debug

The pytest path also supports the three-way comparison: directly add `use_fp64_golden: [True]` to the case parameters in `quant_flash_attn_paramset_debug.py`. The compare stage performs `cross_check_compare` with bench as golden and the sectional reference as third_party (including attention and LSE). For the case template, see the `PA_BnNBsD_B1_QS128_KVS1024_Nq64_Nkv8_D128_SP3_FP64CMP` example in that file (`data_range_q/k/v` already demonstrates the non-[-1,1] usage).

```bash
cd attention/quant_flash_attn/tests/pytest/qfa_mxfp8_test/
# Run only the three-way comparison cases
pytest test_quant_flash_attn_debug.py -k FP64CMP -v
# Step-by-step rerun after failure (cached .pt files avoid rerunning golden/NPU)
pytest test_quant_flash_attn_debug.py -k FP64CMP -v --golden-mode=npu,compare
```

When `use_fp64_golden=True`, the three-way `cross_check_compare` prints `[CROSS_CHECK]` metrics; the decision criteria use the **same ruler** as the two-way gating: the proportion of elements with `|x-golden| > atol + rtol*|golden|` is <= 0.5%, and the result is not worse than twice that of the independent reference (when the reference is cleaner, the 0.5% floor serves as the fallback). The out-of-tolerance rate, mare/mere/rmse, and small_err are also printed for attribution. Pass/fail is still determined by the two-way `check_result` — consistent with the semantics of the CV dual benchmark on master (`compute_cv_report` is display-only). Only `--compare cross_check` on the TTK side is an explicit gate.

The tolerance tier is selected based on the **NPU output dtype** (consistent with the three-way criteria of `quant_flash_mla` / `flash_attn`): bf16 → `small_value=2^-8` and `small_value_atol=2^-16`; fp16/fp32 → `2^-10` / `2^-16`.

Criteria description: the old criteria made the decision based on "the worst single-element ratio + zero tolerance" (mare<=10/mere<=2/rmse<=2 and `err_target/max(err_third,1)<=2`), while the measured number of out-of-tolerance elements in the small value range on the reference side is mostly 0, the denominator is raised to 1, and 3~5 near-zero outlier elements would flag the case as failed. 44 of the 252 full cases were once flagged as failed for this reason (39 of them had 0 out-of-tolerance elements on the reference side), which contradicts the two-way gating. Under the current criteria, the out-of-tolerance rate, mare/mere/rmse, and small_err are all retained as attribution metrics.

---

## 6. Case Naming Rules

Format: `{LAYOUT}_B{batch}_QS{Q length}_KVS{KV length}_Nq{Q heads}_Nkv{KV heads}_D{dimension}_SP{sparse mode}[_{mode}]`

| Field | Meaning | Example |
|------|------|------|
| `PA_NZ` | kv_cache_layout = PA_NZ | `PA_NZ_B1_QS64_KVS64_...` |
| `PA_BnNBsD` | kv_cache_layout = BnNBsD | `PA_BnNBsD_B2_QS48_KVS64_...` |
| `TND` | enable_pa = False | `TND_B1_QS128_KVS128_...` |
| `B` | batch size | `B1`, `B8` |
| `QS` | Q sequence length | `QS128` |
| `KVS` | KV sequence length | `KVS512` |
| `Nq` | Number of Q heads | `Nq80` |
| `Nkv` | Number of KV heads | `Nkv8` |
| `D` | Head dimension | `D128`, `D64` |
| `SP` | sparse_mode | `SP3` |
| `Prefill` | q_scale_layout = TND | Long-sequence prefill stage |
| `Decode` | q_scale_layout = N2TGD | Short-sequence decode stage |

---

## 7. Parameter System

### 7.1 All Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `B` | int | Required | batch size |
| `N_q` | int | Required | Number of Q heads |
| `N_kv` | int | Required | Number of KV heads (N_q must be divisible by N_kv) |
| `D` | int | Required | Head dimension (64 / 128) |
| `actual_seq_q` | list | Required | Q sequence length of each batch |
| `actual_seq_kv` | list | Required | KV sequence length of each batch |
| `enable_pa` | bool | Required | Whether to enable PagedAttention |
| `kv_cache_layout` | str | Required | KV Cache layout (`PA_NZ` / `BnNBsD`) |
| `block_size` | int | Required | PA block size (512 / 1024) |
| `sparse_mode` | int | Required | Sparse mode (0 = no mask, 3 = causal+padding) |
| `q_scale_layout` | str | Required | Q scale layout (`TND` / `N2TGD`) |
| `p_scale` | float | Required | P quantization scale (1.0 / 15.0 / 100.0 / 128.0 / 256.0) |
| `data_range_q` | float | 1.0 | Q data range |
| `data_range_k` | float | 1.0 | K data range |
| `data_range_v` | float | 1.0 | V data range |
| `enable_lse` | bool | Required | Whether to return Log-Sum-Exp |
| `enable_rope` | bool | Required | Whether to enable RoPE |
| `D_rope` | int | 64 | RoPE dimension |
| `graph_path` | int | 0 | Graph mode (0 = single operator, 3 = static graph, 5 = dynamic graph, 6 = tiling sinking, 7 = aclgraph) |
| `device_id` | int | 0 | NPU device ID |
| `is_contiguous` | bool | True | Whether the input is contiguous |

### 7.2 Default Value Mechanism

If parameters in `TEST_PARAMS_DEFAULTS` are not explicitly specified in the paramset, the default values are automatically used. For example, `D_rope`, `graph_path`, `device_id`, and `is_contiguous` usually do not need to be written for every case.

If a case needs to use non-default values, explicitly specify them in the paramset:

```python
"MY_CASE": {
    "B": [1],
    ...
    "graph_path": [7],       # Use the aclgraph mode
    "D_rope": [128],         # Use 128-dimensional RoPE
},
```

### 7.3 Parameter Expansion

`expand_paramset_to_cases()` performs a Cartesian product expansion over all dimensions in each configuration. For example:

```python
"MY_CASE": {
    "B": [1, 2],
    "D": [64, 128],
    ...
}
```

It expands into 4 independent cases (B=1/D=64, B=1/D=128, B=2/D=64, B=2/D=128).

---

## 8. Adding New Cases

Add a new entry in the corresponding paramset file. Taking `quant_flash_attn_paramset_func_rdv.py` as an example:

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

### B>1 with Unequal Sequence Lengths

When B>1, `actual_seq_q` and `actual_seq_kv` should be lists of unequal lengths:

```python
"PA_NZ_B4_QS128_KVS1024_Nq8_Nkv1_D128_SP3": {
    "B": [4],
    ...
    "actual_seq_q": [[96, 128, 160, 112]],
    "actual_seq_kv": [[800, 1024, 1280, 960]],
},
```

### SKIP_CASES Mechanism

For cases with excessively long execution time, you can define a `SKIP_CASES` set in the paramset file:

```python
SKIP_CASES = {
    "PA_NZ_B128_QS1_KVS10240_Nq1_Nkv1_D128_SP0",
    "PA_NZ_B128_QS4_KVS16384_Nq1_Nkv1_D128_SP0",
}
```

The marked cases automatically receive `pytest.mark.skip` and are displayed as `SKIPPED` in the pytest output.

---

## 9. Accuracy Comparison Criteria

The dual 0.5% criterion is used:

| Metric | FP16 Threshold | BF16 Threshold |
|------|-----------|-----------|
| rtol (relative tolerance) | 0.005 | 0.0078125 |
| atol (absolute tolerance) | 0.000025 | 0.0001 |
| pct_thd (pass rate threshold) | 99.5% | 99.5% |
| max_diff_hd (maximum relative error limit) | 10 | 10 |

Decision logic: the result is Pass when the pass rate >= 99.5% **and** the maximum relative error < 10.

---

## 10. pytest Marker Description

| Marker | Purpose |
|--------|------|
| `ci` | CI pipeline test |
| `func_rdv` | Functional correctness RDV test |
| `perf_rdv` | Performance/stress RDV test |
| `debug` | Daily debugging test |
| `graph` | Graph mode compilation test |
| `npu_only` | NPU execution only (no CPU golden comparison) |

---

## 11. FAQ

| Error | Cause | Solution |
|------|------|------|
| `Unsupported Q scale layout` | Invalid q_scale_layout value | Only `"TND"` / `"N2TGD"` are supported |
| `libhccl.so not found` | CANN environment variables are not sourced | `source /home/user/Ascend/cann/set_env.sh` |
| `No module named 'torch_npu'` | TorchNPU is not installed | Refer to the README.md in the root directory to install PyTorch + TorchNPU |
| `No cached input: xxx_input.pt` | Input data is missing in cache mode | Run `--golden-mode=gen` first to generate data |
| `No cached CPU output` | The CPU cache is missing in npu/compare mode | Run `--golden-mode=cpu` first to generate the CPU output |

---

## 12. Performance Profiling (--msprof / --parse-prof / --perf-baseline)

### 12.1 Overview

The pytest run is wrapped with the `msprof` tool to automatically collect profiling data of the `FusedInferAttentionScore` operator (Duration, AI Core time, Cube utilization), and supports performance regression comparison with the baseline log.

### 12.2 One-Command Run + Profiling + Report

```bash
# Run tests, automatically collect profiling, and output the report after the run
pytest --msprof -v -m debug

# Run perf_rdv cases and collect profiling
pytest --msprof -v -m perf_rdv
```

The workflow of `--msprof`:

1. Snapshots existing `PROF_*` directories in the current directory
2. Wraps and runs the inner test with `msprof python -m pytest ...`
3. After the tests complete, finds the newly generated `PROF_*` directories
4. Parses `op_summary_*.csv` and extracts the `FusedInferAttentionScore` entries
5. Outputs the performance report and archives it to the `perf_output/` directory

### 12.3 Parsing an Existing PROF Directory Afterwards

```bash
# Parse the specified PROF directory
pytest --parse-prof=./PROF_000001_20260615113759304_xxx
```

### 12.4 Performance Baseline Comparison

```bash
# Run + profiling + compare with the baseline (default 8% threshold)
pytest --msprof --perf-baseline=./perf_baseline/perf_report_20260615113640.log -v -m debug

# Custom threshold (5%)
pytest --msprof --perf-baseline=./perf_baseline/perf_report_xxx.log --perf-threshold=5.0 -v -m debug

# Post-hoc parsing + comparison
pytest --parse-prof=./PROF_xxx --perf-baseline=./perf_baseline/perf_report_xxx.log
```

### 12.5 Command-Line Options

| Option | Description |
|------|------|
| `--msprof` | Automatically wraps the pytest run with msprof, parses PROF after the tests complete, and outputs the report |
| `--parse-prof=PROF_DIR` | Parses the specified PROF directory (post-hoc analysis) |
| `--perf-baseline=LOG_FILE` | Path of the performance baseline log file; compares Duration with the current result |
| `--perf-threshold=PERCENT` | Performance degradation threshold percentage (default 8.0) |

### 12.6 Report Output Format

```
========================================================================================================================
  #    Duration    AI Core   Cube%  Input Shapes
------------------------------------------------------------------------------------------------------------------------
  1      21.3us     20.5us   22.4%  Q=[4, 64, 128]  K=[2, 8, 4, 512, 32]  V=[2, 8, 4, 512, 32]
  2      27.8us     27.2us   92.4%  Q=[128, 64, 128]  K=[2, 8, 512, 128]  V=[2, 8, 512, 128]
========================================================================================================================

Total: 2 entries
```

### 12.7 Baseline Comparison Report

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

| Status | Meaning |
|--------|------|
| `PASS` | The Duration change is within the threshold |
| `FAILED` | The Duration degradation exceeds the threshold → pytest exit code 1 |
| `IMPROVED` | The Duration improvement exceeds the threshold |
| `NEW` | The current case does not exist in the baseline |

Matching method: exact match by Input Shapes (the shape strings of Q/K/V).

### 12.8 File Archiving

After each run, the report log and the raw CSV are automatically archived to the `perf_output/` directory with consistent timestamps:

```
perf_output/
├── perf_report_20260615113640.log    # Formatted report
└── op_summary_20260615113640.csv     # Raw profiling data
```

### 12.9 Establishing a Baseline

1. Run profiling once to generate a report:

   ```bash
   pytest --msprof -v -m debug
   ```

2. Copy the generated report to the `perf_baseline/` directory as the baseline:

   ```bash
   cp perf_output/perf_report_*.log perf_baseline/
   ```

3. Specify this baseline for comparison in subsequent runs:

   ```bash
   pytest --msprof --perf-baseline=./perf_baseline/perf_report_20260615113640.log -v -m debug
   ```
