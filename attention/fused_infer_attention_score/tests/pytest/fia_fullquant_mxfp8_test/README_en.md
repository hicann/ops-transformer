# FIA FullQuant MXFP8 Test Case Execution Guide

English | [简体中文](./README.md)

## 1. Environment Preparation

### 1.1 Install CANN / PyTorch + TorchNPU

Refer to the installation instructions in the root README.md.

### 1.2 Load the Environment Before Each Execution

```bash
source /home/user/Ascend/cann/set_env.sh
conda activate your-env-name
pip install pytest
cd attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test
```

---

## 2. File Structure

```
fia_fullquant_mxfp8_test/
├── pytest.ini                                  # pytest configuration (custom markers)
├── conftest.py                                 # pytest command line options (--golden-mode, --cache-dir, --msprof, --parse-prof, --perf-baseline)
├── common/
│   ├── __init__.py
│   ├── fia_fullquant_mxfp8_golden.py           # CPU golden reference implementation + NPU operator invocation
│   ├── golden_cache.py                         # .pt cache utility module
│   ├── result_compare_method.py                # Accuracy comparison tool
│   ├── perf_parser.py                          # msprof op_summary.csv parsing + baseline comparison
│   └── test_runner.py                          # Shared test execution logic (apply_params / execute_test / check_results)
├── fia_fullquant_mxfp8_paramset_common.py      # Common parameter expansion logic + default values
├── fia_fullquant_mxfp8_paramset_debug.py       # Debug paramset (a few cases, quick verification)
├── fia_fullquant_mxfp8_paramset_func_rdv.py    # Functional correctness paramset (~150 cases)
├── fia_fullquant_mxfp8_paramset_perf_rdv.py    # Performance/stress paramset (~100 cases)
├── test_fia_fullquant_mxfp8_debug.py           # Debug test entry
├── test_fia_fullquant_mxfp8_func_rdv.py        # Functional correctness test entry
└── test_fia_fullquant_mxfp8_perf_rdv.py        # Performance/stress test entry
```

---

## 3. Running Tests

### 3.1 Three Test Entries

| Test File | Marker | Paramset | Purpose |
|----------|--------|--------|------|
| `test_fia_fullquant_mxfp8_debug.py` | `@pytest.mark.debug` | debug (2 cases) | Quickly verifies basic functions |
| `test_fia_fullquant_mxfp8_func_rdv.py` | `@pytest.mark.func_rdv` `@pytest.mark.ci` | func_rdv (~150 cases) | Full coverage of functional correctness |
| `test_fia_fullquant_mxfp8_perf_rdv.py` | `@pytest.mark.perf_rdv` | perf_rdv (~100 cases) | Performance/stress verification |

### 3.2 Basic Execution

```bash
# Run debug cases
pytest test_fia_fullquant_mxfp8_debug.py -v

# Run functional correctness cases
pytest test_fia_fullquant_mxfp8_func_rdv.py -v

# Run performance/stress cases
pytest test_fia_fullquant_mxfp8_perf_rdv.py -v

# Run by marker
pytest -m debug -v
pytest -m func_rdv -v
pytest -m perf_rdv -v
pytest -m ci -v
```

### 3.3 Filtering Cases (-k)

```bash
# Precisely specify a single case
pytest test_fia_fullquant_mxfp8_func_rdv.py -v -k "PA_NZ_B1_QS1_KVS512_Nq1_Nkv1_D128_SP3"

# By Prefill / Decode mode
pytest test_fia_fullquant_mxfp8_perf_rdv.py -v -k "Prefill"
pytest test_fia_fullquant_mxfp8_perf_rdv.py -v -k "Decode"

# By layout type
pytest -v -k "PA_NZ"
pytest -v -k "BnNBsD"

# Combined filtering
pytest -v -k "PA_NZ and Prefill"

# Exclude some cases
pytest -v -k "not Decode"
```

### 3.4 Common Options

| Option | Description |
|------|------|
| `-v` | Verbose output; displays each case name |
| `-s` | Does not truncate stdout/stderr |
| `-x` | Stops immediately on the first failure |
| `--tb=long` | Displays the full stack trace on failure |
| `-m ci` | Runs only cases marked with CI |

---

## 4. Golden Cache Mode (--golden-mode)

The tests support saving input data, CPU golden outputs, and NPU outputs as `.pt` files. In the next run, data generation can be skipped and the cache can be loaded directly to execute specified steps.

### 4.1 Mode Description

| Mode | Description |
|------|------|
| `all` | Full flow: generate data → CPU → NPU → accuracy comparison (**default**) |
| `gen` | Only generates and saves input data |
| `cpu` | Loads the input cache → runs CPU and saves the output |
| `npu` | Loads the input cache → runs NPU and saves the output |
| `compare` | Loads cached CPU/NPU outputs → performs accuracy comparison |

### 4.2 Combined Modes

Modes support comma-separated combinations and run in the order gen → cpu → npu → compare:

```bash
# Only generate and save input data (without running CPU/NPU)
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=gen

# Run NPU + accuracy comparison only (CPU output is loaded from the cache)
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=npu,compare

# Run CPU + NPU (no comparison)
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=cpu,npu

# Full flow (equivalent to --golden-mode=all)
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=gen,cpu,npu,compare
```

### 4.3 Typical Workflows

**Method 1: Run the full flow at once (default)**

```bash
pytest test_fia_fullquant_mxfp8_func_rdv.py -v
```

**Method 2: Run step by step (suitable for debugging across machines)**

```bash
# Step 1: Generate input data (without running CPU/NPU)
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=gen

# Step 2: Run the CPU golden
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=cpu

# Step 3: Run NPU + accuracy comparison on the NPU machine
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --golden-mode=npu,compare
```

### 4.4 Custom Cache Directory

```bash
pytest test_fia_fullquant_mxfp8_func_rdv.py -v --cache-dir=/tmp/my_cache
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

`common/fia_fullquant_mxfp8_golden.py` also supports standalone execution and likewise supports `--mode` combinations:

```bash
cd common/
python fia_fullquant_mxfp8_golden.py --mode all --case-name my_case
python fia_fullquant_mxfp8_golden.py --mode gen --case-name my_case
python fia_fullquant_mxfp8_golden.py --mode npu,compare --case-name my_case
python fia_fullquant_mxfp8_golden.py --mode cpu --case-name my_case --cache-dir=/tmp/cache
```

---

## 6. Case Naming Rules

Format: `{LAYOUT}_B{batch}_QS{q_max_len}_KVS{kv_max_len}_Nq{num_q_heads}_Nkv{num_kv_heads}_D{dim}_SP{sparse_mode}[_{mode}]`

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

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `B` | int | Required | batch size |
| `N_q` | int | Required | Number of Q heads |
| `N_kv` | int | Required | Number of KV heads (N_q must be divisible by N_kv) |
| `D` | int | Required | Head dimension (64 / 128) |
| `actual_seq_q` | list | Required | Q sequence length of each batch |
| `actual_seq_kv` | list | Required | KV sequence length of each batch |
| `enable_pa` | bool | Required | Whether to enable PagedAttention |
| `kv_cache_layout` | str | Required | KV cache layout (`PA_NZ` / `BnNBsD`) |
| `block_size` | int | Required | PA block size (512 / 1024) |
| `sparse_mode` | int | Required | Sparse mode (0 = no mask, 3 = causal + padding) |
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

If a parameter in `TEST_PARAMS_DEFAULTS` is not explicitly specified in a paramset, the default value is used automatically. For example, `D_rope`, `graph_path`, `device_id`, and `is_contiguous` usually do not need to be written for every case.

If a case needs a non-default value, explicitly specify it in the paramset:

```python
"MY_CASE": {
    "B": [1],
    ...
    "graph_path": [7],       # Use aclgraph mode
    "D_rope": [128],         # Use 128-dim RoPE
},
```

### 7.3 Parameter Expansion

`expand_paramset_to_cases()` expands the Cartesian product of all dimensions in each configuration. For example:

```python
"MY_CASE": {
    "B": [1, 2],
    "D": [64, 128],
    ...
}
```

This expands into 4 independent cases (B=1/D=64, B=1/D=128, B=2/D=64, B=2/D=128).

---

## 8. Adding Cases

Add a new entry to the corresponding paramset file. Taking `fia_fullquant_mxfp8_paramset_func_rdv.py` as an example:

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

### B>1 With Unequal Sequence Lengths

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

For cases with excessively long execution time, define a `SKIP_CASES` set in the paramset file:

```python
SKIP_CASES = {
    "PA_NZ_B128_QS1_KVS10240_Nq1_Nkv1_D128_SP0",
    "PA_NZ_B128_QS4_KVS16384_Nq1_Nkv1_D128_SP0",
}
```

Marked cases automatically get `pytest.mark.skip` added and are displayed as `SKIPPED` in the pytest output.

---

## 9. Accuracy Comparison Criteria

The "double five-per-mille" standard is used:

| Metric | FP16 Threshold | BF16 Threshold |
|------|-----------|-----------|
| rtol (relative tolerance) | 0.005 | 0.0078125 |
| atol (absolute tolerance) | 0.000025 | 0.0001 |
| pct_thd (pass rate threshold) | 99.5% | 99.5% |
| max_diff_hd (maximum relative error upper bound) | 10 | 10 |

Pass criteria: the pass rate >= 99.5% **and** the maximum relative error < 10.

---

## 10. pytest Marker Description

| Marker | Purpose |
|--------|------|
| `ci` | CI pipeline test |
| `func_rdv` | Functional correctness RDV test |
| `perf_rdv` | Performance/stress RDV test |
| `debug` | Daily debugging test |
| `graph` | Graph-mode compilation test |
| `npu_only` | NPU execution only (no CPU golden comparison) |

---

## 11. Common Problems

| Error | Cause | Solution |
|------|------|------|
| `Unsupported Q scale layout` | The q_scale_layout value is invalid | Only `"TND"` / `"N2TGD"` are supported |
| `libhccl.so not found` | CANN environment variables are not sourced | `source /home/user/Ascend/cann/set_env.sh` |
| `No module named 'TorchNPU'` | TorchNPU is not installed | Refer to the root README.md to install PyTorch + TorchNPU |
| `No cached input: xxx_input.pt` | Input data is missing in cache mode | First run `--golden-mode=gen` to generate data |
| `No cached CPU output` | The CPU cache is missing in npu/compare mode | First run `--golden-mode=cpu` to generate the CPU output |

---

## 12. Performance Profiling (--msprof / --parse-prof / --perf-baseline)

### 12.1 Overview

This wraps the pytest run with the `msprof` tool, automatically collects profiling data of the `FusedInferAttentionScore` operator (Duration, AI Core time, and Cube utilization), and supports performance regression comparison against a baseline log.

### 12.2 One-Step Run + Profiling + Report

```bash
# Run tests, automatically collect profiling, and output a report afterwards
pytest --msprof -v -m debug

# Run perf_rdv cases and collect profiling
pytest --msprof -v -m perf_rdv
```

Workflow of `--msprof`:

1. Take a snapshot of the `PROF_*` directories that already exist in the current directory
2. Wrap and run the inner test with `msprof python -m pytest ...`
3. After the tests finish, locate the newly generated `PROF_*` directory
4. Parse `op_summary_*.csv` and extract the `FusedInferAttentionScore` entries
5. Output the performance report and archive it to the `perf_output/` directory

### 12.3 Parsing an Existing PROF Directory Afterwards

```bash
# Parse the specified PROF directory
pytest --parse-prof=./PROF_000001_20260615113759304_xxx
```

### 12.4 Performance Baseline Comparison

```bash
# Run + profiling + comparison against the baseline (default threshold of 8%)
pytest --msprof --perf-baseline=./perf_baseline/perf_report_20260615113640.log -v -m debug

# Custom threshold (5%)
pytest --msprof --perf-baseline=./perf_baseline/perf_report_xxx.log --perf-threshold=5.0 -v -m debug

# Post-analysis + comparison
pytest --parse-prof=./PROF_xxx --perf-baseline=./perf_baseline/perf_report_xxx.log
```

### 12.5 Command Line Options

| Option | Description |
|------|------|
| `--msprof` | Automatically wraps the pytest run with msprof, parses PROF after the tests finish, and outputs a report |
| `--parse-prof=PROF_DIR` | Parses the specified PROF directory (post-analysis) |
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
| `FAILED` | The Duration degrades beyond the threshold → pytest exit code 1 |
| `IMPROVED` | The Duration improves beyond the threshold |
| `NEW` | The current case does not exist in the baseline |

Matching method: exact matching by Input Shapes (shape strings of Q/K/V).

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
