# FIA FullQuant GQA Test Case Execution Guide

English | [简体中文](./README.md)

## 1. Environment Preparation

### 1.1 Install CANN / PyTorch + TorchNPU

Refer to the installation instructions in the root README.md.

### 1.2 Load the Environment Before Each Execution

```bash
source /home/user/Ascend/cann/set_env.sh
conda activate your-env-name
pip install pytest
cd attention/fused_infer_attention_score/tests/pytest/fia_fullquant_gqa_test
```

---

## 2. File Structure

```
fia_fullquant_gqa_test/
├── pytest.ini                                  # pytest configuration (custom markers)
├── conftest.py                                 # pytest command line options (--golden-mode, --cache-dir, --use-external-input, --load-pt-dir, --msprof, --parse-prof, --perf-baseline)
├── common/
│   ├── __init__.py
│   ├── fia_fullquant_gqa_golden.py             # CPU golden reference implementation + NPU operator invocation
│   ├── golden_cache.py                         # .pt cache utility module
│   ├── result_compare_method.py                # Accuracy comparison tool
│   ├── perf_parser.py                          # msprof op_summary.csv parsing + baseline comparison
│   ├── load_external_data.py                   # Loading of external pt files in NPU layout
│   └── test_runner.py                          # Shared test execution logic (apply_params / execute_test / check_results)
├── fia_fullquant_gqa_paramset_common.py        # Common parameter expansion logic + default values
├── fia_fullquant_gqa_paramset_debug.py         # Debug paramset (a few cases, quick verification)
├── fia_fullquant_gqa_paramset_func_rdv.py      # Functional correctness paramset
├── fia_fullquant_gqa_paramset_perf_rdv.py      # Performance/stress paramset
├── test_fia_fullquant_gqa_debug.py             # Debug test entry
├── test_fia_fullquant_gqa_func_rdv.py          # Functional correctness test entry
└── test_fia_fullquant_gqa_perf_rdv.py          # Performance/stress test entry
```

---

## 3. Running Tests

### 3.1 Three Test Entries

| Test File | Marker | Paramset | Purpose |
|----------|--------|--------|------|
| `test_fia_fullquant_gqa_debug.py` | `@pytest.mark.debug` | debug (2 cases) | Quickly verifies basic functions |
| `test_fia_fullquant_gqa_func_rdv.py` | `@pytest.mark.func_rdv` `@pytest.mark.ci` | func_rdv | Full coverage of functional correctness |
| `test_fia_fullquant_gqa_perf_rdv.py` | `@pytest.mark.perf_rdv` | perf_rdv | Performance/stress verification |

### 3.2 Basic Execution

```bash
# Run debug cases
pytest test_fia_fullquant_gqa_debug.py -v

# Run functional correctness cases
pytest test_fia_fullquant_gqa_func_rdv.py -v

# Run performance/stress cases
pytest test_fia_fullquant_gqa_perf_rdv.py -v

# Run by marker
pytest -m debug -v
pytest -m func_rdv -v
pytest -m perf_rdv -v
pytest -m ci -v
```

### 3.3 Filtering Cases (-k)

```bash
# Precisely specify a single case
pytest test_fia_fullquant_gqa_func_rdv.py -v -k "B1_G8_Nq16_Nkv2_D128_SM3_LSE1_Q2048_KV2048"

# By Prefill / Decode mode
pytest test_fia_fullquant_gqa_perf_rdv.py -v -k "Prefill"
pytest test_fia_fullquant_gqa_perf_rdv.py -v -k "Decode"

# By GQA ratio
pytest -v -k "G64"
pytest -v -k "G128"

# Combined filtering
pytest -v -k "B1 and G8"

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
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=gen

# Run NPU + accuracy comparison only (CPU output is loaded from the cache)
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=npu,compare

# Run CPU + NPU (no comparison)
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=cpu,npu

# Full flow (equivalent to --golden-mode=all)
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=gen,cpu,npu,compare
```

### 4.3 Typical Workflows

**Method 1: Run the full flow at once (default)**

```bash
pytest test_fia_fullquant_gqa_func_rdv.py -v
```

**Method 2: Run step by step (suitable for debugging across machines)**

```bash
# Step 1: Generate input data (without running CPU/NPU)
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=gen

# Step 2: Run the CPU golden
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=cpu

# Step 3: Run NPU + accuracy comparison on the NPU machine
pytest test_fia_fullquant_gqa_func_rdv.py -v --golden-mode=npu,compare
```

### 4.4 Custom Cache Directory

```bash
pytest test_fia_fullquant_gqa_func_rdv.py -v --cache-dir=/tmp/my_cache
```

The default cache directory is `common/golden_cache/`.

### 4.5 Cache File Naming

Each case generates 3 `.pt` files:

```
golden_cache/
├── {case_name}_input.pt        # Input data (Q/K/V fp8 + deq_scale + p_scale + block_table)
├── {case_name}_cpu_output.pt   # CPU golden output (atten_out + lse)
└── {case_name}_npu_output.pt   # NPU operator output (atten_out + lse)
```

---

## 5. Loading External NPU-Layout Data (--use-external-input / --load-pt-dir)

Loading NPU-layout data from external pt files is supported (replacing `generate_data`). This is suitable for reproducing problems from data dumped by real models.

### 5.1 Directory Structure Requirements

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
└── rank0_attn_mask.pt        # [Optional]
```

### 5.2 Usage

```bash
# Use external data only in the generation phase, then run CPU + NPU + comparison
pytest test_fia_fullquant_gqa_debug.py -v \
    --use-external-input \
    --load-pt-dir=/path/to/external_pt \
    --golden-mode=gen,cpu,npu,compare

# Step by step: first generate the cache from external data
pytest test_fia_fullquant_gqa_debug.py -v \
    --use-external-input \
    --load-pt-dir=/path/to/external_pt \
    --golden-mode=gen

# Subsequent steps do not need --use-external-input again (loaded from the cache)
pytest test_fia_fullquant_gqa_debug.py -v --golden-mode=cpu,npu,compare
```

---

## 6. Standalone Script Execution

`common/fia_fullquant_gqa_golden.py` also supports standalone execution and likewise supports `--mode` combinations:

```bash
cd common/
python fia_fullquant_gqa_golden.py --mode all --case-name my_case
python fia_fullquant_gqa_golden.py --mode gen --case-name my_case
python fia_fullquant_gqa_golden.py --mode npu,compare --case-name my_case
python fia_fullquant_gqa_golden.py --mode cpu --case-name my_case --cache-dir=/tmp/cache

# Use external data
python fia_fullquant_gqa_golden.py --mode gen --case-name my_case \
    --use-external-input --load-pt-dir=/path/to/external_pt
```

---

## 7. Case Naming Rules

Format: `B{batch}_G{gqa_ratio}_Nq{num_q_heads}_Nkv{num_kv_heads}_D{dim}_SM{sparse_mode}_LSE{0/1}_Q{q_max_len}_KV{kv_max_len}[_{tag}]`

| Field | Meaning | Example |
|------|------|------|
| `B` | batch size | `B1`, `B8` |
| `G` | GQA ratio (N_q / N_kv) | `G1`, `G8`, `G128` |
| `Nq` | Number of Q heads | `Nq16` |
| `Nkv` | Number of KV heads | `Nkv2` |
| `D` | Head dimension | `D128`, `D64` |
| `SM` | sparse_mode | `SM0`, `SM3` |
| `LSE` | Whether to return LSE | `LSE0`, `LSE1` |
| `Q` | Maximum Q sequence length | `Q128`, `Q2048` |
| `KV` | Maximum KV sequence length | `KV256`, `KVS8192` |
| `Prefill` | Long-sequence prefill stage | `_Prefill` |
| `Decode` | Decode stage with short Q and long KV | `_Decode` |
| `noncontig` | IS_CONTIGUOUS=False | `_noncontig` |
| `numblocks{n}` | NUM_BLOCKS physical block reuse | `_numblocks4` |
| `pscale{n}` | Custom p_scale (1.0/15.0/100.0/128.0/256.0) | `_pscale15`, `_pscale128` |
| `block{n}` | Custom block_size | `_block64` |

---

## 8. Parameter System

### 8.1 All Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `B` | int | Required | batch size |
| `N_q` | int | Required | Number of Q heads |
| `N_kv` | int | Required | Number of KV heads (N_q must be divisible by N_kv) |
| `D` | int | 128 | Head dimension (64 / 128) |
| `actual_seq_q` | list | Required | Q sequence length of each batch |
| `actual_seq_kv` | list | Required | KV sequence length of each batch |
| `enable_pa` | bool | True | Whether to enable PagedAttention |
| `enable_lse` | bool | True | Whether to return Log-Sum-Exp |
| `golden_mode` | bool | True | Whether to perform the CPU golden comparison |
| `block_size` | int | 128 | PA block size |
| `sparse_mode` | int | 3 | Sparse mode (0 = no mask, 3 = causal + padding) |
| `input_layout` | str | NTD_TND | Q input layout |
| `output_layout` | str | TND | Output layout |
| `q_scale_layout` | str | NT | Q scale layout |
| `kv_cache_layout` | str | BnNBsD | KV cache layout |
| `p_scale` | float | 1.0 | P quantization scale |
| `scale_value` | float | None | softmax_scale (1/sqrt(D) when None) |
| `is_contiguous` | bool | True | Whether the input is contiguous |
| `num_blocks` | int | 0 | Number of physical blocks (0 = default = total_blocks) |
| `graph_path` | int | 0 | Graph mode (0 = single operator, 5 = dynamic graph, 7 = aclgraph) |
| `device_id` | int | 0 | NPU device ID |
| `q_data_range` | tuple | (-1.0, 1.0) | Q data range |
| `k_data_range` | tuple | (-1.0, 1.0) | K data range |
| `v_data_range` | tuple | (-1.0, 1.0) | V data range |
| `seed_q` | int | 54 | Q random seed |
| `seed_k` | int | 3 | K random seed |
| `seed_v` | int | 20 | V random seed |
| `seed_block_table` | int | 1234 | block_table random seed |

### 8.2 Default Value Mechanism

If a parameter in `TEST_PARAMS_DEFAULTS` is not explicitly specified in a paramset, the default value is used automatically. For example, `D`, `block_size`, `graph_path`, and `device_id` usually do not need to be written for every case.

If a case needs a non-default value, explicitly specify it in the paramset:

```python
"MY_CASE": {
    "B": [1],
    ...
    "graph_path": [7],       # Use aclgraph mode
    "D": [64],               # Use D=64
},
```

### 8.3 Parameter Expansion

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

## 9. Adding Cases

Add a new entry to the corresponding paramset file. Taking `fia_fullquant_gqa_paramset_func_rdv.py` as an example:

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

### B>1 With Unequal Sequence Lengths

When B>1, `actual_seq_q` and `actual_seq_kv` should be lists of unequal lengths:

```python
"B4_G4_Nq16_Nkv4_D128_SM3_LSE1_Q256_KV512_unequal": {
    "B": [4],
    ...
    "actual_seq_q": [[128, 256, 192, 256]],
    "actual_seq_kv": [[512, 384, 640, 512]],
},
```

### SKIP_CASES Mechanism

For cases with excessively long execution time, define a `SKIP_CASES` set in the paramset file:

```python
SKIP_CASES = {
    "B128_G8_Nq16_Nkv2_D128_SM3_LSE0_Q1_KVS16384_Decode",
}
```

Marked cases automatically get `pytest.mark.skip` added and are displayed as `SKIPPED` in the pytest output.

---

## 10. NUM_BLOCKS Physical Block Reuse

By default, `num_blocks=0` means that the number of physical blocks equals the total number of blocks actually needed (no reuse).

When `num_blocks=N` (N < total_blocks) is set, physical block reuse is triggered: some data in the cache is overwritten. In this case, test_runner restores the BNSD data from the cache after NPU execution and runs the CPU golden again for comparison, ensuring that the compared data is what the NPU actually sees.

```python
"B1_G4_Nq4_Nkv1_D128_SM3_LSE1_Q128_KV1024_numblocks4": {
    ...
    "num_blocks": [4],
},
```

---

## 11. Accuracy Comparison Criteria

The "double five-per-mille" standard is used:

| Metric | FP16 Threshold | BF16 Threshold |
|------|-----------|-----------|
| rtol (relative tolerance) | 0.005 | 0.0078125 |
| atol (absolute tolerance) | 0.000025 | 0.0001 |
| pct_thd (pass rate threshold) | 99.5% | 99.5% |
| max_diff_hd (maximum relative error upper bound) | 10 | 10 |

Pass criteria: the pass rate >= 99.5% **and** the maximum relative error < 10.

---

## 12. pytest Marker Description

| Marker | Purpose |
|--------|------|
| `ci` | CI pipeline test |
| `func_rdv` | Functional correctness RDV test |
| `perf_rdv` | Performance/stress RDV test |
| `debug` | Daily debugging test |
| `graph` | Graph-mode compilation test |
| `npu_only` | NPU execution only (no CPU golden comparison) |

---

## 13. Common Problems

| Error | Cause | Solution |
|------|------|------|
| `libhccl.so not found` | CANN environment variables are not sourced | `source /home/user/Ascend/cann/set_env.sh` |
| `No module named 'torch_npu'` | TorchNPU is not installed | Refer to the root README.md to install PyTorch + TorchNPU |
| `No cached input: xxx_input.pt` | Input data is missing in cache mode | First run `--golden-mode=gen` to generate data |
| `No cached CPU output` | The CPU cache is missing in npu/compare mode | First run `--golden-mode=cpu` to generate the CPU output |
| `--use-external-input requires --load-pt-dir` | External data is used but no directory is specified | Add `--load-pt-dir=/path/to/pt` |

---

## 14. Performance Profiling (--msprof / --parse-prof / --perf-baseline)

### 14.1 Overview

This wraps the pytest run with the `msprof` tool, automatically collects profiling data of the `FusedInferAttentionScore` operator (Duration, AI Core time, and Cube utilization), and supports performance regression comparison against a baseline log.

### 14.2 One-Step Run + Profiling + Report

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

### 14.3 Parsing an Existing PROF Directory Afterwards

```bash
# Parse the specified PROF directory
pytest --parse-prof=./PROF_000001_20260615113759304_xxx
```

### 14.4 Performance Baseline Comparison

```bash
# Run + profiling + comparison against the baseline (default threshold of 8%)
pytest --msprof --perf-baseline=./perf_baseline/perf_report_20260615113640.log -v -m debug

# Custom threshold (5%)
pytest --msprof --perf-baseline=./perf_baseline/perf_report_xxx.log --perf-threshold=5.0 -v -m debug

# Post-analysis + comparison
pytest --parse-prof=./PROF_xxx --perf-baseline=./perf_baseline/perf_report_xxx.log
```

### 14.5 Command Line Options

| Option | Description |
|------|------|
| `--msprof` | Automatically wraps the pytest run with msprof, parses PROF after the tests finish, and outputs a report |
| `--parse-prof=PROF_DIR` | Parses the specified PROF directory (post-analysis) |
| `--perf-baseline=LOG_FILE` | Path of the performance baseline log file; compares Duration with the current result |
| `--perf-threshold=PERCENT` | Performance degradation threshold percentage (default 8.0) |

### 14.6 Establishing a Baseline

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
