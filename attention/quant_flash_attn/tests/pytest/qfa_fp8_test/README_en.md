# QFA FullQuant GQA Test Case Execution Guide

English | [简体中文](./README.md)

## 1. Environment Preparation

### 1.1 Installing CANN / PyTorch + torch-npu

Refer to the installation instructions in the README.md in the root directory.

### 1.2 Loading the Environment Before Each Execution

```bash
source /home/user/Ascend/cann/set_env.sh
conda activate your-env-name
pip install pytest
cd attention/quant_flash_attn/tests/pytest/quant_flash_attn_fp8_test
```

---

## 2. File Structure

```
quant_flash_attn_fp8_test/
├── pytest.ini                                  # pytest configuration (custom markers)
├── conftest.py                                 # pytest command-line options (--golden-mode, --cache-dir, --msprof, --parse-prof, --perf-baseline)
├── common/
│   ├── __init__.py
│   ├── quant_flash_attn_fp8_golden.py         # CPU golden reference implementation + QFA NPU dual-operator invocation
│   ├── golden_cache.py                         # .pt cache utility module
│   ├── result_compare_method.py                # Accuracy comparison tool
│   ├── perf_parser.py                          # msprof op_summary.csv parsing + baseline comparison
│   └── test_runner.py                          # Shared test execution logic (apply_params / execute_test / check_results)
├── quant_flash_attn_fp8_paramset_common.py        # Common parameter expansion logic + default values
├── quant_flash_attn_fp8_paramset_func_rdv.py      # Functional correctness parameter set (12 representative cases)
├── test_quant_flash_attn_fp8_func_rdv.py          # Functional correctness test entry
└── README.md                                   # This file
```

---

## 3. Running Tests

### 3.1 Test Entries

| Test File | Marker | Parameter Set | Purpose |
|----------|--------|--------|------|
| `test_quant_flash_attn_fp8_func_rdv.py` | `@pytest.mark.func_rdv` `@pytest.mark.ci` | func_rdv (12 cases) | Functional correctness verification |

### 3.2 Basic Execution

```bash
# Run all functional correctness cases
pytest test_quant_flash_attn_fp8_func_rdv.py -v

# Run by marker
pytest -m func_rdv -v
pytest -m ci -v
```

### 3.3 Filtering Cases (-k)

```bash
# Specify a single case exactly
pytest test_quant_flash_attn_fp8_func_rdv.py -v -k "B1_G2_Nq32_Nkv16_D128_SM3_LSE1_Q2048_KV2048"

# By GQA ratio
pytest -v -k "G2"
pytest -v -k "G28"

# By batch
pytest -v -k "B4"

# Combined filtering
pytest -v -k "B1 and LSE1"
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
# Generate and save input data only
pytest test_quant_flash_attn_fp8_func_rdv.py -v --golden-mode=gen

# Run NPU + accuracy comparison only
pytest test_quant_flash_attn_fp8_func_rdv.py -v --golden-mode=npu,compare

# Full flow
pytest test_quant_flash_attn_fp8_func_rdv.py -v --golden-mode=gen,cpu,npu,compare
```

### 4.3 Custom Cache Directory

```bash
pytest test_quant_flash_attn_fp8_func_rdv.py -v --cache-dir=/tmp/my_cache
```

The default cache directory is `common/golden_cache/`.

### 4.4 Cache File Naming

Each case generates 3 `.pt` files:

```
golden_cache/
├── {case_name}_input.pt        # Input data (Q/K/V fp8 + deq_scale + p_scale + block_table)
├── {case_name}_cpu_output.pt   # CPU golden output (atten_out + lse)
└── {case_name}_npu_output.pt   # NPU operator output (atten_out + lse)
```

---

## 5. Standalone Script Execution

`common/quant_flash_attn_fp8_golden.py` also supports standalone execution:

```bash
cd common/
python quant_flash_attn_fp8_golden.py --mode all --case-name my_case
python quant_flash_attn_fp8_golden.py --mode gen --case-name my_case
python quant_flash_attn_fp8_golden.py --mode npu,compare --case-name my_case
python quant_flash_attn_fp8_golden.py --mode cpu --case-name my_case --cache-dir=/tmp/cache
```

---

## 6. Case Naming Rules

Format: `B{batch}_G{gqa_ratio}_Nq{Q heads}_Nkv{KV heads}_D{dimension}_SM{mask_mode}_LSE{0/1}_Q{Q length}_KV{KV length}[_P{p_scale}]`

| Field | Meaning | Example |
|------|------|------|
| `B` | batch size | `B1`, `B4` |
| `G` | GQA ratio (N_q / N_kv) | `G2`, `G8`, `G28` |
| `Nq` | Number of Q heads | `Nq32`, `Nq80` |
| `Nkv` | Number of KV heads | `Nkv2`, `Nkv16` |
| `D` | Head dimension (fixed 128) | `D128` |
| `SM` | mask_mode | `SM0` (NO_MASK), `SM3` (CAUSAL) |
| `LSE` | Whether to return LSE | `LSE0`, `LSE1` |
| `Q` | Maximum Q sequence length | `Q64`, `Q2048` |
| `KV` | Maximum KV sequence length | `KV257`, `KV5119` |
| `P` | p_scale value | `_P15`, `_P256` |

---

## 7. Parameter System

### 7.1 GQA Fixed Parameters

The following parameters are fixed in the QFA GQA scenario and cannot be modified:

| Parameter | Fixed Value | Description |
|------|--------|------|
| `quant_mode` | 6 | GQA_FP8_FULLQUANT |
| `layout_q` | `"NTD"` | Q input layout |
| `layout_q_descale` | `"NT"` | Q descale layout (2D [N1,T]) |
| `layout_kv` | `"PA_BNBD"` | KV Cache layout |
| `layout_out` | `"TND"` | Output layout |
| `kv_cache_layout` | `"BnNBsD"` | KV Cache data arrangement |
| `block_size` | 128 | PA block size |
| `D` | 128 | Head dimension |
| `enable_pa` | True | Forces PagedAttention |

### 7.2 Variable Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `B` | int | Required | batch size |
| `N_q` | int | Required | Number of Q heads |
| `N_kv` | int | Required | Number of KV heads (N_q % N_kv == 0) |
| `actual_seq_q` | list | Required | Q sequence length of each batch |
| `actual_seq_kv` | list | Required | KV sequence length of each batch |
| `enable_lse` | bool | True | Whether to return Log-Sum-Exp |
| `mask_mode` | int | 3 | 0=NO_MASK, 3=CAUSAL |
| `p_scale` | float | 1.0 | P quantization scale |
| `scale_value` | float | None | softmax_scale (1/sqrt(D) when None) |
| `is_contiguous` | bool | True | Whether the input is contiguous |
| `num_blocks` | int | 0 | Number of physical blocks (0 = no reuse) |
| `graph_path` | int | 0 | Graph mode (0 = single operator) |
| `device_id` | int | 0 | NPU device ID |
| `q_data_range` | tuple | (-1.0, 1.0) | Q data range |
| `k_data_range` | tuple | (-1.0, 1.0) | K data range |
| `v_data_range` | tuple | (-1.0, 1.0) | V data range |
| `seed_q` | int | 54 | Q random seed |
| `seed_k` | int | 3 | K random seed |
| `seed_v` | int | 20 | V random seed |
| `seed_block_table` | int | 1234 | block_table random seed |

### 7.3 Parameter Expansion

`expand_paramset_to_cases()` performs a Cartesian product expansion over all dimensions in each configuration. Fixed parameters are automatically filled by `TEST_PARAMS_DEFAULTS`.

---

## 8. Adding New Cases

Add a new entry in `quant_flash_attn_fp8_paramset_func_rdv.py`:

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

### SKIP_CASES Mechanism

```python
SKIP_CASES = {
    "B4_G52_Nq52_Nkv1_D128_SM0_LSE0_Q257_4b_KV127_4b",
}
```

The marked cases automatically receive `pytest.mark.skip`.

---

## 9. NUM_BLOCKS Physical Block Reuse

The default `num_blocks=0` indicates that the number of physical blocks equals the total number of required blocks (no reuse).

When `num_blocks=N` (N < total_blocks) is set, physical block reuse is triggered. After NPU execution, test_runner restores the BNSD data from the cache and reruns the CPU golden for comparison.

---

## 10. Accuracy Comparison Criteria

The dual 0.5% criterion is used:

| Metric | BF16 Threshold |
|------|-----------|
| rtol (relative tolerance) | 0.0078125 |
| atol (absolute tolerance) | 0.0001 |
| pct_thd (pass rate threshold) | 99.5% |
| max_diff_hd (maximum relative error limit) | 10 |

Decision logic: the result is Pass when the pass rate >= 99.5% **and** the maximum relative error < 10.

---

## 11. pytest Marker Description

| Marker | Purpose |
|--------|------|
| `ci` | CI pipeline test |
| `func_rdv` | Functional correctness RDV test |
| `perf_rdv` | Performance/stress RDV test |
| `debug` | Daily debugging test |
| `graph` | Graph mode compilation test |
| `npu_only` | NPU execution only (no CPU golden comparison) |

---

## 12. Performance Profiling (--msprof / --parse-prof / --perf-baseline)

### 12.1 One-Command Run + Profiling + Report

```bash
# Run tests and automatically collect profiling
pytest --msprof -v -m func_rdv
```

### 12.2 Parsing an Existing PROF Directory Afterwards

```bash
pytest --parse-prof=./PROF_000001_20260615113759304_xxx
```

### 12.3 Performance Baseline Comparison

```bash
pytest --msprof --perf-baseline=./perf_baseline/perf_report_xxx.log -v -m func_rdv
pytest --msprof --perf-baseline=./perf_baseline/perf_report_xxx.log --perf-threshold=5.0 -v -m func_rdv
```

| Option | Description |
|------|------|
| `--msprof` | Automatically wraps the pytest run with msprof |
| `--parse-prof=PROF_DIR` | Parses the specified PROF directory |
| `--perf-baseline=LOG_FILE` | Path of the performance baseline log file |
| `--perf-threshold=PERCENT` | Performance degradation threshold percentage (default 8.0) |
