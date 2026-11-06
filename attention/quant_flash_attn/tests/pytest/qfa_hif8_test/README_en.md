<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# QFA HIF8 Test Case Execution Guide

English | [简体中文](./README.md)

Functional tests for the quant_flash_attn HIF8 full quantization (`quant_mode=0`, A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32).

## 1. Scenario Features

| Feature | Value | Description |
|------|------|------|
| quant_mode | 0 | HIF8 full quantization |
| q/k/v dtype | HIFLOAT8 | uint8 encoding, converted to ACL_HIFLOAT8 on the csrc side |
| descale | FP32 scalar `(1,)` | One per-tensor scale for each of q/k/v |
| p_scale | FP32 `(1,)` | P quantization scale |
| head_dim | **128 only** | The tiling config locks DAligned128 |
| layout | TND / BSND / BNSD | PA is not supported (no block_table) |
| layout_q_descale | BSND | Fixed value constrained by HIF8 |
| attn_out | BF16 | — |
| softmax_lse | FP32 | TND output `(N, T)`; non-TND output `(B, N, S)` |

## 2. Environment Preparation

```bash
source /home/user/Ascend/cann/set_env.sh
conda activate your-env-name
pip install pytest
cd attention/quant_flash_attn/tests/pytest/qfa_hif8_test
```

## 3. File Structure

```
qfa_hif8_test/
├── pytest.ini                                  # pytest configuration (custom markers)
├── conftest.py                                 # pytest command-line options (--golden-mode, --cache-dir, --msprof, --parse-prof, --perf-baseline)
├── common/
│   ├── __init__.py
│   ├── quant_flash_attn_golden.py              # CPU golden reference implementation + NPU operator invocation
│   ├── golden_cache.py                         # .pt cache utility module
│   ├── generate_hifloat8_data.py               # HIF8 encoding/decoding (hifloat8 <-> float)
│   ├── result_compare_method.py                # Accuracy comparison tool
│   ├── perf_parser.py                          # msprof op_summary.csv parsing + baseline comparison
│   └── test_runner.py                          # Shared test execution logic (apply_params / execute_test / check_results)
├── quant_flash_attn_paramset_common.py         # Common parameter expansion logic + default values
├── quant_flash_attn_paramset_debug.py          # debug parameter set (a few cases for quick verification)
├── quant_flash_attn_paramset_func_rdv.py       # Functional correctness parameter set (100 cases)
├── quant_flash_attn_paramset_perf_rdv.py       # Performance/stress parameter set (8 cases)
├── test_quant_flash_attn_debug.py              # debug test entry
├── test_quant_flash_attn_func_rdv.py           # Functional correctness test entry
└── test_quant_flash_attn_perf_rdv.py           # Performance/stress test entry
```

## 4. Running Tests

### 4.1 Three Test Entries

| Test File | Marker | Parameter Set | Purpose |
|----------|--------|--------|------|
| `test_quant_flash_attn_debug.py` | `@pytest.mark.debug` | debug (2 cases) | Quick verification of basic functions |
| `test_quant_flash_attn_func_rdv.py` | `@pytest.mark.func_rdv` `@pytest.mark.ci` | func_rdv (100 cases) | Full functional correctness coverage |
| `test_quant_flash_attn_perf_rdv.py` | `@pytest.mark.perf_rdv` | perf_rdv (8 cases) | Performance/stress verification |

### 4.2 Basic Execution

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
pytest -m ci -v
```

### Performance Fast Mode and Accuracy Switch

By default, `perf_rdv` cases run performance only: inputs in `--cache-dir` are reused first; if missing, only the inputs are generated and cached. Then NPU execution is performed and the run synchronously waits for completion. This mode does not generate or load the CPU golden, does not perform accuracy comparison, and does not move the NPU output back to the CPU or save it to disk.
The first-time input preparation, extension loading, and profiler startup still take time; operator performance should be measured by the kernel Duration in msprof.

```bash
# Quickly collect performance: no golden needs to be generated in advance; an empty cache directory is supported.
pytest test_quant_flash_attn_perf_rdv.py -v --msprof

# Final accuracy acceptance: generate inputs and the CPU golden, execute NPU, and compare.
pytest test_quant_flash_attn_perf_rdv.py -v --check-accuracy

# Explicitly use the fast mode for debug/functional cases.
pytest test_quant_flash_attn_debug.py -v --golden-mode=perf --msprof

# When matching input and golden caches exist, rerun only NPU and compare.
pytest test_quant_flash_attn_perf_rdv.py -v --golden-mode=npu,compare
```

Functional/debug cases still perform full accuracy verification by default. An explicit `--golden-mode` retains the original manual stage selection; `perf` must be used alone, and `--check-accuracy` and `--golden-mode` cannot be specified at the same time.
In performance mode, passed only indicates that NPU execution succeeded, not that accuracy passed. pytest summarizes a notice that accuracy comparison was not performed and records `accuracy_checked=False` in the report attributes.
Reuse the cache only when the parameters and input generation logic are consistent; otherwise, specify a new cache directory.

There are 8 performance RDV cases in total. The following 5 long-sequence cases use the same parameters: B=1, Nq=20, Nkv=2, D=128, TND, SP0, LSE=false, p_scale=1.0; the Q/KV lengths are identical.

| Scenario | Q Length | KV Length |
| --- | ---: | ---: |
| 10K | 10240 | 10240 |
| 16K | 16384 | 16384 |
| 20K | 20480 | 20480 |
| 32K | 32768 | 32768 |
| 64K | 65536 | 65536 |

```bash
# Collect only these 5 long-sequence cases.
pytest test_quant_flash_attn_perf_rdv.py -k "Nq20_Nkv2_D128_SP0" --msprof -v

# Collect only the 64K case.
pytest test_quant_flash_attn_perf_rdv.py -k QS65536 --msprof -v
```

### 4.3 Filtering Cases (-k)

```bash
# By layout type
pytest test_quant_flash_attn_func_rdv.py -v -k "TND"
pytest test_quant_flash_attn_func_rdv.py -v -k "BSND"
pytest test_quant_flash_attn_func_rdv.py -v -k "BNSD"

# Specify a single case exactly
pytest test_quant_flash_attn_func_rdv.py -v -k "TND_B1_QS4_KVS1024_Nq1_Nkv1_D128_SP3"

# Exclude certain cases
pytest -v -k "not BSND"
```

### 4.4 Common Options

| Option | Description |
|------|------|
| `-v` | Verbose output, displaying each case name |
| `-s` | Does not truncate stdout/stderr |
| `-x` | Stops immediately when a failure occurs |
| `--tb=long` | Displays the full stack trace on failure |
| `-m ci` | Runs only cases marked with CI |

## 5. Golden Cache Mode (--golden-mode)

| Mode | Description |
|------|------|
| `all` | Full flow: data generation → CPU → NPU → accuracy comparison (default for functional/debug cases) |
| `perf` | Default for performance cases: reuse/generate inputs → NPU → synchronize; no golden, comparison, or output saving to disk |
| `gen` | Generates and saves input data only |
| `cpu` | Loads the input cache → runs CPU and saves the output |
| `npu` | Loads the input cache → runs NPU and saves the output |
| `compare` | Loads the cached CPU/NPU output → performs accuracy comparison |

Modes support comma-separated combinations and are executed in the order gen → cpu → npu → compare:

```bash
# Step-by-step execution (suitable for cross-machine debugging)
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=gen
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=cpu
pytest test_quant_flash_attn_func_rdv.py -v --golden-mode=npu,compare

# Custom cache directory (default common/golden_cache/)
pytest test_quant_flash_attn_func_rdv.py -v --cache-dir=/tmp/my_cache
```

The full flow generates the following 3 `.pt` files for each case; the `perf` mode only generates or reuses input files:

```
golden_cache/
├── {case_name}_input.pt        # Input data (Q/K/V hif8 + descale + p_scale)
├── {case_name}_cpu_output.pt   # CPU golden output (atten_out + lse)
└── {case_name}_npu_output.pt   # NPU operator output (atten_out + lse)
```

## 6. Standalone Script Execution

`common/quant_flash_attn_golden.py` also supports standalone execution and likewise supports `--mode` combinations:

```bash
cd common/
python quant_flash_attn_golden.py --mode all --case-name my_case
python quant_flash_attn_golden.py --mode gen --case-name my_case
python quant_flash_attn_golden.py --mode npu,compare --case-name my_case
```

### 6.1 Three-Way Accuracy Comparison (bench golden | third_party sectional reference | NPU)

When a case fails the accuracy check (especially in scenarios where the QKV input range is not [-1,1] (`DATA_RANGE_Q/K/V > 1.0`)), you can use the three-way comparison to determine whether the NPU output deviation is within a reasonable range. The three roles are exactly the same as those of TTK `--compare cross_check`:

- golden (ground truth/error denominator): `cpu_hif8_bench`, FP64 arithmetic with HIF8 quantization that preserves P;
- third_party (upper bound of competitor noise): `cpu_hif8_sectioned_golden`, sectional quantization reference;
- actual: NPU output.

The decision follows the TTK `CrossCheckComparison` criteria (`precision_compare_v2.cross_check_compare`, reusing the same implementation under `qfa_mxfp8_test/common`): in the large domain, `mare/mere/rmse <= 10/2/2`; in the small domain, `err(actual)/err(third_party) <= 2`; NaN/Inf must be consistent with either golden or third_party.

Usage: Edit the configuration area in `common/quant_flash_attn_golden.py`, align the case parameters (B/N_q/N_kv/D, `CU_SEQLENS_*`/`SEQUSED_*`, `SPARSE_MODE`, `Q_SCALE_LAYOUT`, `P_SCALE`, `ENABLE_ROPE`/`D_rope`, `DATA_RANGE_Q/K/V`, `ENABLE_LSE`, and so on) with the failed case, and set:

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

### 6.2 Running the Three-Way Comparison through pytest debug

The pytest path also supports the three-way comparison: directly add `use_fp64_golden: [True]` to the case parameters in `quant_flash_attn_paramset_debug.py`. The compare stage performs `cross_check_compare` with bench as golden and the sectional reference as third_party (including attention and LSE).

```bash
cd attention/quant_flash_attn/tests/pytest/qfa_hif8_test/
# Run only the three-way comparison cases (replace -k with the actual case name)
pytest test_quant_flash_attn_debug.py -k <case_name> -v
# Step-by-step rerun after failure (cached .pt files avoid rerunning golden/NPU)
pytest test_quant_flash_attn_debug.py -k <case_name> -v --golden-mode=npu,compare
```

When `use_fp64_golden=True`, pass/fail is determined directly by `cross_check_compare`, consistent with the TTK criteria; when the default `False` is used, the original two-way `check_result` path is still followed.

The tolerance tier is selected based on the output dtype of the golden on the TTK side (TTK `CrossCheckComparison` looks up the table by the golden output dtype): the hif8 golden on the TTK side returns FP32, so the FP32 tier is used (`small_value=2^-14`).

The K sectional blocks of the sectional reference are adaptively subdivided based on the KV length (when the KV is less than one sectional block, it is split by 32 / `KV//4`), preventing the denominator of the ratio from degenerating after the reference and bench become numerically equivalent. Meanwhile, the bench uses float64 and the sectional reference uses float32 for accumulation (consistent with the three-way criteria of `quant_flash_mla`), so that the absolute error noise floor of the reference is of the same order of magnitude as that of the target.

## 7. Case Naming Rules

Format: `{LAYOUT}_B{batch}_QS{Q length}_KVS{KV length}_Nq{Q heads}_Nkv{KV heads}_D{dimension}_SP{sparse mode}`

| Field | Meaning | Example |
|------|------|------|
| `LAYOUT` | input_layout (TND / BSND / BNSD) | `BSND_B1_QS128_...` |
| `B` | batch size | `B1`, `B5` |
| `QS` | Q sequence length (the sum of batches when B>1) | `QS128` |
| `KVS` | KV sequence length (the sum of batches when B>1) | `KVS2048` |
| `Nq` | Number of Q heads | `Nq80` |
| `Nkv` | Number of KV heads (N_q must be divisible by N_kv) | `Nkv8` |
| `D` | Head dimension (**fixed 128**) | `D128` |
| `SP` | sparse_mode (0 = no mask, 3 = causal+padding) | `SP3` |

## 8. Parameter System

### 8.1 HIF8-Specific Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `input_layout` | str | `TND` | q/k/v/out layout (`TND` / `BSND` / `BNSD`) |
| `q_scale_layout` | str | `BSND` | layout_q_descale (fixed to BSND for HIF8) |
| `p_scale` | float | Required | P quantization scale (1.0 / 15.0 / 128.0 / 256.0) |

### 8.2 Common Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `B` | int | Required | batch size |
| `N_q` | int | Required | Number of Q heads |
| `N_kv` | int | Required | Number of KV heads |
| `D` | int | Required | Head dimension (128 only) |
| `cu_seqlens_q/kv` | list | Required | Cumulative sequence lengths (used for TND) |
| `seqused_q/kv` | list | Required | Actual sequence length of each batch |
| `max_seqlen_q/kv` | int | -1 | Maximum sequence length (-1 automatically uses max(seqused)) |
| `mask_mode` | int | Required | Sparse mode (0 / 3) |
| `enable_lse` | bool | Required | Whether to return Log-Sum-Exp |
| `enable_pa` | bool | False | PA is not supported by HIF8, fixed to False |
| `softmax_scale` | float | None | 1/sqrt(D) when None |
| `data_range_q/k/v` | float | 1.0 | Data range |
| `device_id` | int | 0 | NPU device ID |
| `is_contiguous` | bool | True | Whether the input is contiguous |

### 8.3 Default Value Mechanism

Parameters in `TEST_PARAMS_DEFAULTS` automatically take effect when the paramset does not explicitly specify them; `expand_paramset_to_cases()` performs a Cartesian product expansion over all dimensions in each configuration.

## 9. Adding New Cases

Add an entry in the corresponding paramset file:

```python
"BSND_B1_QS256_KVS1024_Nq8_Nkv2_D128_SP3": {
    "B": [1],
    "N_q": [8],
    "N_kv": [2],
    "D": [128],
    "cu_seqlens_q": [[0, 256]],
    "cu_seqlens_kv": [[0, 1024]],
    "seqused_q": [[256]],
    "seqused_kv": [[1024]],
    "max_seqlen_q": [256],
    "max_seqlen_kv": [1024],
    "mask_mode": [3],
    "q_scale_layout": ["BSND"],
    "p_scale": [1.0],
    "enable_lse": [True],
    "input_layout": ["BSND"],
},
```

Precautions:

- **D can only be 128** (the op-side HIF8 check verifies `head_dim = 128`, and the UT case `QFA_hif8_head_dim_64_unsupported` verifies illegal values)
- **BSND/BNSD cases are recommended to use equal lengths for all batches** (seqused == max_seqlen) to avoid padding behavior differences in non-TND layouts
- **B>1 with unequal sequence lengths** is recommended only for TND cases (the cumulative sum of cu_seqlens is used as the QS/KVS naming value)
- The LAYOUT prefix in the case name must be consistent with `input_layout`

## 10. Related UT (op_host)

The host-side UT files are located in `tests/ut/op_host/`. In the CSVs, `quant_compute_mode=0` indicates the HIF8 scenario:

| File | HIF8 Coverage |
|------|-----------|
| `arch35/test_quant_flash_attn_tiling.csv` | 14 cases: TND/BSND/BNSD SUCCESS (including LSE, GQA, QS=1); FAILED (D=64, descale 2D/E8M0, q dtype E4M3, PA_NZ, layout inconsistency, layout_q_descale=TND) |
| `test_quant_flash_attn_shape_infershape.csv` | 3 cases: TND/BSND/BNSD infershape (including LSE `(N,T)` / `(B,N,S)`) |
| `test_quant_flash_attn_dtype_infershape.csv` | 1 case: HIFLOAT8/FLOAT/BF16 dtype inference |
