# Mixed Quant Flash Attention Test Framework

English | [简体中文](./README.md)

This directory provides functional and performance tests for the `mixed_quant_flash_attn` operator. It implements precision comparison between the CPU golden results and the NPU results based on pytest. Tests are organized as a four-stage pipeline of data generation → CPU computation → NPU computation → precision comparison. Each stage runs independently and writes its artifacts to disk, which facilitates debugging and reproduction.

## Directory Structure

```
pytest/
├── README.md
├── conftest.py                          # pytest plugin: registers command line options and markers
├── test_mixed_quant_flash_attn.py       # Test entry: parametrization, four-stage scheduling, and result decision
├── core/
│   ├── case_loader.py                   # Case loading, parameter normalization, and case ID filtering
│   ├── gen_data.py                      # Input tensor generation (including fp4/fp8 quantized data)
│   ├── cpu.py                           # CPU golden reference implementation
│   ├── npu.py                           # NPU operator invocation (eager / aclgraph modes)
│   ├── compare.py                       # Precision comparison and report printing
│   ├── postprocess.py                   # Case result summary and CSV export
│   └── meta_cpu/
│       ├── mixed_quant_flash_attn_metadata_op.py   # metadata subprocess invocation wrapper
│       └── mixed_quant_flash_attn_metadata_test   # metadata binary (C++)
├── testcase/
│   ├── functional_stc.py                # STC functional cases
│   └── functional_rdv.py                # RDV functional cases
│
└── data/
    └── <testset>/                       # functional_stc by default; subdirectories named by case
        └── <case_safe>/                 # for example, functional_stc_fp4e2m1_stc_bnsd_bf16
            ├── <case_safe>_input.pt     # artifact of the gen stage
            ├── <case_safe>_cpu.pt       # artifact of the cpu stage
            └── <case_safe>_npu.pt       # artifact of the npu stage
```

## Core Workflow

### 1. Case Discovery and Parametrization

- `conftest.py:14` registers command line options: `--mode`, `--testset`, `--case_id`, `--device_id`, and `--seed`.
- `conftest.py:37` registers markers: `ci`, `func_rdv`, `func_redline`, `func_stc`, `perf_rdv`, `perf_redline`, and `debug`.
- `test_mixed_quant_flash_attn.py:19` maintains the `_MARKER_TO_MODULE` mapping from markers to case modules.
- `pytest_generate_tests` (`test_mixed_quant_flash_attn.py:67`) selects case modules based on the markers on the test function, then loads and validates cases through `_load_cases_for_modules` (requiring `N1 >= N2` and `N1 % N2 == 0`), and finally parametrizes `case_name`.
- Cases are defined in the `TestCases` dictionary in `testcase/*.py`: the key is the case name, and the value is a parameter dictionary. When any parameter is a list, it is expanded into multiple cases by Cartesian product (`expand_cases` in `case_loader.py:24`).
- `normalize_params` (`case_loader.py:41`) fills in default values: `layout_kv` defaults to `layout_q`, `softmax_scale` defaults to `1/sqrt(D)`, `mask_mode` defaults to 0, and `kv_dtype` defaults to `fp4_e2m1`. For `TND`/`PA` layouts, it automatically constructs auxiliary tensors such as `cu_seqlens_*`, `seqused_*`, and `block_table`.

### 2. Four-Stage Execution

The entry point is `_execute_step` (`test_mixed_quant_flash_attn.py:106`), which executes the stages in sequence according to the `--mode` list:

| Mode  | Behavior                                                                                                  | Artifact                              |
| ----- | --------------------------------------------------------------------------------------------------- | -------------------------------- |
| `gen` | Calls `generate_inputs(params, seed)` to generate q/k/v/k_descale/v_descale/block_table, and saves them together with the scalar parameters | `<case>_input.pt`                |
| `cpu` | Loads `_input.pt` and calls `run_cpu_golden` to run the blocked softmax + flash attention reference implementation                         | `<case>_cpu.pt`                  |
| `npu` | Loads `_input.pt`, first generates metadata through `MixedQuantFlashAttnMetadataOp`, and then calls `torch.ops.npu_ops_transformer.mixed_quant_flash_attn`. Supports `eager`/`aclgraph` | `<case>_npu.pt` |
| `compare` | Loads `_cpu.pt` and `_npu.pt`, and calls `compare_results` to perform the precision comparison and return a statistics dictionary                          | Console report + `result.csv`         |

The artifact path is determined by `get_data_dir` (`case_loader.py:132`): `data/<testset>/<case_name.replace("/", "_")>/`.

### 3. CPU Golden Implementation

`tforward` in `core/cpu.py` (`cpu.py:373`) implements the blocked reference computation:

1. `get_axis_from_tensor` parses B/N1/N2/S1/S2/D/block_num/block_size, and supports `BNSD`/`BSND`/`TND` and `PA_NZ`/`PA_BNBD`/`PA_BBND`.
2. When `quant_compute_mode == 1`, `unpack_int4` first splits each `uint8` into two fp4 values.
3. If `layout_kv` contains `PA`, `trans_pa_to_no_pa` rearranges the paged KV into a non-paged layout according to `block_table`.
4. `antiquant_data` dequantizes fp4 to bf16/fp16. It determines the group_size (32 or 4) and the scale alignment method according to `quant_compute_mode` and `gs_flag`.
5. Iterates block by block according to `souter`/`sinner`. `softmax_flash` maintains the online softmax `block_max`/`block_sum`/`block_exp`, and finally computes `attn_out = Σ p_block @ v_block` divided by `block_sum`.
6. `run_cpu_golden` additionally computes `softmax_lse = log(x_sum) + x_max`, and sets fully masked rows to `inf`.

### 4. NPU Invocation

`run_npu` in `core/npu.py` (`npu.py:138`):

1. Copies the tensors to `npu:<device_id>` and passes the scalar parameters as is.
2. Calls `MixedQuantFlashAttnMetadataOp.npu_mixed_quant_flash_attn_metadata` (`meta_cpu/mixed_quant_flash_attn_metadata_op.py:184`): constructs the input JSON → executes the `mixed_quant_flash_attn_metadata_test` binary in a subprocess → parses the output JSON and assembles the `metadata` int32 list, which describes the task partitioning of the AIC/AIV cores.
3. Calls `torch.ops.npu_ops_transformer.mixed_quant_flash_attn`, which returns `attn_out` and the optional `softmax_lse`.
4. When `mode == "aclgraph"`, the graph compilation path is used through `torchair` + `CompilerConfig`, with optimizations such as `reduce-overhead`, static shape, and tiling schedule optimization.

### 5. Precision Comparison

`core/compare.py` provides two comparison functions:

- `check_result(expect, result, data_type, pct_thd=0.005)` (`compare.py:175`): implemented with numpy. The pass condition is `fulfill_percent >= 99.5%` **and** `max_rel_err < 10.0`. The relative error denominator is `max(|real|, |expect|, (1/16384)/0.005) + 10e-10` to prevent small values from being amplified. Dedicated handling exists for `bfloat16`/`float8_e4m3fn`/`float8_e5m2`, using `np.isclose(..., equal_nan=True)`.
- `check_result_(...)` (`compare.py:18`): uses the same decision logic as `check_result`, but outputs a boxed precision report in Chinese and returns a structured `stats` dictionary (containing `passed`/`max_abs`/`mean_abs`/`max_rel`/`mean_rel`/`fail_cnt`/`total`/`fail_ratio`/`fulfill_percent`).

`compare_results(cpu_result, npu_result, case_name)` (`compare.py:297`) calls the preceding comparison functions for `attn_out` and `softmax_lse` respectively. The tolerance is selected automatically based on the dtype:

| output dtype | atol      | rtol       |
| ------------ | --------- | ---------- |
| bf16         | 0.0001    | 0.0078125  |
| fp16 (default)  | 0.000025  | 0.005      |

### 6. Result Summary

`PostProcessor` (`core/postprocess.py`) accumulates the decision result of each case in a session-level fixture:

- `record`: records whether attn/lse passes and the key metrics.
- `print_summary`: prints the summary table (case name, PASS/FAIL, MaxAbsErr, FailRatio, and LSE status).
- `save_csv`: exports `result.csv` to the pytest directory.

`test_summary` (`test_mixed_quant_flash_attn.py:246`) triggers the summary and CSV export when `--mode compare` is used.

## Case Parameters

Cases are defined in the `TestCases` dictionary in `testcase/*.py`. Common fields:

| Field          | Description                                    | Example Values                          |
| ------------- | --------------------------------------- | --------------------------------- |
| `B`           | Batch size                              | 1, 2, 4, 8                        |
| `N1` / `N2`   | Number of Q/KV heads; requires `N1 % N2 == 0`        | N1=8 N2=1 (GQA), N1=8 N2=8 (MHA)|
| `S1` / `S2`   | Q/KV sequence length                         | 1, 64, 1024, 2048                 |
| `D`           | Head dimension                                  | 128                               |
| `layout_q`    | Q layout                                  | `BNSD`, `BSND`, `TND`             |
| `layout_kv`   | KV layout                                 | `PA_NZ`, `PA_BNBD`, `PA_BBND`     |
| `q_dtype`     | Q dtype                                 | `bf16`, `fp16`                    |
| `quant_compute_mode`  | Quantization mode (determines the KV dtype and descale)    | 1 → `fp4_e2m1` + `fp8_e8m0`       |
| `mask_mode`   | Mask mode                                | 0 none, 3 causal, 4 band            |
| `win_left` / `win_right` | Band window (mask_mode=4)      | 32                                |
| `block_size`  | PA paging block size                           | 128, 512                          |
| `block_table` | Paging table (automatically generated at random if not specified)            | [[0,1,2], [3,4]]                  |
| `seqused_kv`  | Actual KV length of each batch                        | [0,0,1024,1024]                   |
| `q_range` / `kv_range` | Data value range                       | (-5.0, 5.0)                       |

Supported KV quantization dtypes: `fp4_e2m1`, `fp4_e1m2`, and `fp8_e4m3` (controlled by `quant_compute_mode`).

## Command Line Usage

```bash
cd pytest/

# Complete four-stage pipeline (functional_stc case set by default)
pytest test_mixed_quant_flash_attn.py -m func_stc --mode gen --testset functional_stc
pytest test_mixed_quant_flash_attn.py -m func_stc --mode cpu  --testset functional_stc
pytest test_mixed_quant_flash_attn.py -m func_stc --mode npu  --testset functional_stc --device_id 0
pytest test_mixed_quant_flash_attn.py -m func_stc --mode compare --testset functional_stc

# Run specified cases (filtered by case name suffix; multiple names separated by commas are supported)
pytest test_mixed_quant_flash_attn.py -m func_redline --mode gen --case_id "fp4e2m1_gqa_bf16_bnsd_causal"

# Run RDV functional cases
pytest test_mixed_quant_flash_attn.py -m func_rdv --mode gen cpu npu compare --testset functional_rdv

# Run performance cases
pytest test_mixed_quant_flash_attn.py -m perf_redline --mode npu --testset perf_redline

# Multi-process parallelism (requires pytest-xdist)
pytest test_mixed_quant_flash_attn.py -m func_stc --mode gen -n 4

# Run CPU golden separately (no pytest required)
python core/cpu.py --input_dir data/functional_stc/functional_stc_fp4e2m1_stc_bnsd_bf16
```

### Command Line Parameters

| Parameter           | Default Value            | Description                                                  |
| -------------- | ----------------- | ----------------------------------------------------- |
| `--mode`       | `gen`             | Multiple values allowed: `gen`, `cpu`, `npu`, `compare`                |
| `--testset`    | `functional_stc`  | Data directory name, corresponding to `data/<testset>/`                    |
| `--case_id`    | `all`             | Case filter: `all` or case names separated by commas                    |
| `--device_id`  | `0`               | NPU device ID                                            |
| `--seed`       | `21`              | Random seed for data generation                                      |
| `-m`           | (built into pytest)     | Marker filter, such as `func_stc`, `perf_rdv`, `ci`, `debug`|
| `-n`           | (pytest-xdist)    | Number of parallel processes                                            |

## Markers

| Marker          | Corresponding Case Module            | Description                       |
| --------------- | ----------------------- | -------------------------- |
| `func_stc`      | `functional_stc`        | STC functional correctness             |
| `func_redline`  | `functional_redline`    | Functional redline                   |
| `ci`            | (stacked on top of func_rdv)    | For CI gate checks                  |
| `debug`         | (no corresponding module; runs all cases)    | Daily debugging                   |

## Precautions

1. `--mode` accepts multiple values at a time. For example, `--mode gen cpu npu compare` runs the complete pipeline in order on the same case. However, the artifacts of the preceding stages must already exist on disk. When scheduling across pytest processes, it is recommended that you run the stages separately.
2. In the `compare` stage, `pytest.skip` is called if `_cpu.pt` or `_npu.pt` is missing.
3. The NPU stage depends on `torch_npu`, `torchair`, `npu_ops_transformer`, and the `core/meta_cpu/mixed_quant_flash_attn_metadata_test` binary. The CPU stage depends only on `torch`, `numpy`, and `ml_dtypes`.
4. The `aclgraph` mode generates static shape compilation artifacts in the current directory. Prefer `eager` for debugging.
5. fp4 data is stored packed in `uint8` (two fp4 values share one byte). Both the CPU side and the NPU side must call `unpack_int4` first.
