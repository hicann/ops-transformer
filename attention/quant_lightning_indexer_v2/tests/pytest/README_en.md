# quant_lightning_indexer_v2 Operator Test Framework

English | [简体中文](./README.md)

## Description

The framework implements functional verification of the quant_lightning_indexer_v2 operator based on the pytest test framework:

- **CPU side**: Replicates the operator logic to generate golden data
- **NPU side**: Calls the operator directly through TorchNPU to obtain actual output data
- **Accuracy comparison**: Compares CPU and NPU results to verify operator functions
- **Dual-mode execution isolation**: Supports two batch modes: direct pytest multi-process execution and process isolation at the shell layer
- **PT rerun mode**: `batch -P <directory>` directly runs existing PT files; add `-E` to generate PT files first
- **Controllable batch running**: `-P` uniformly specifies the directory for PT generation and reading, and supports selection by result path, case name, and 1-based index
- **Performance profiling**: Supports mounting msprof to collect operator performance data and aggregate the output
- **Run mode switching**: Supports two operator invocation modes: direct eager calls and graph (torch.compile + torchair)

## Current Implementation Scope

### Parameter Restrictions

- **Data formats**:
  - **query_layout**: BSND, TND
  - **key_layout**: PA_BBND, BSND, TND

- **Data types**:
  - **qk_dtype**: FLOAT8_E4M3FN, INT8, HIFLOAT8, FLOAT4_E2M1
  - **dequant_dtype**: FP32 (Ascend950), FP16 (Ascend910_93), FLOAT8_E8M0 (MXFP8/MXFP4)
  - **actual_seq_dtype**: INT32

### PyTorch MX Type Conventions

- When `quant_mode` is 3 or 5, `query_dequant_scale` and `key_dequant_scale` must be `torch.float8_e8m0fnu`.
- When `quant_mode` is 5, the logical data type of `query` and `key` in the operator is `FLOAT4_E2M1` (abbreviated as `float4_e2m1` in this document):
  - If PyTorch provides `torch.float4_e2m1fn_x2`, prefer this native packed type; `x2` in the name indicates that two E2M1 logical elements are packed into each physical byte.
  - If PyTorch does not provide this type, use `torch.uint8` to carry the packed data; the wrapper layer passes it to the operator as `ACL_FLOAT4_E2M1`.

- **Run modes**:
  - **eager**: Directly calls `torch.ops.cann_ops_transformer.quant_lightning_indexer`
  - **graph**: Compiles and executes through the `torch.compile` + `torchair` backend (torchair support is required)

### Environment Configuration

#### Prerequisites

1. Confirm that TorchNPU is the latest version.
2. Activate the CANN package and the custom operator package.
3. The graph mode requires the torchair compiler backend to be installed.

#### Custom Package Invocation

Invocation through the custom package is supported.

## File Structure

#### pytest File Structure Description

- test_run.sh                                  # Execution script; supports the single/batch/batch_exec commands
- batch_isolated_run.sh                        # Batch isolated execution script (shell-layer process isolation + msprof performance profiling)
- quant_lightning_indexer_v2_golden.py         # CPU-side golden implementation of the operator
- quant_lightning_indexer_v2_acl_graph.py      # torchair backend implementation for the graph mode
- result_compare_method.py                     # Accuracy comparison between CPU golden data and NPU output
- qliv2_test_utils.py                          # Common logic for case selection, stable naming, and the result table
- collect_perf_data.py                         # msprof performance data collection and aggregation
- pytest.ini                                   # Defines test markers

Single-case testing:

- test_quant_lightning_indexer_v2_single.py    # Main program for running a single pytest test case
- test_quant_lightning_indexer_v2_paramset.py  # Input parameter configuration for single cases; automatically selects cases by chip model
- test_quant_lightning_indexer_v2_stc.py       # STC white-box case configuration

Batch testing:

- test_quant_lightning_indexer_v2_batch.py     # Main program for batch case testing; generates an Excel file to save results
- ./batch/quant_lightning_indexer_v2_pt_loadprocess.py   # Reads PT files and calls the operator to obtain NPU output
- ./batch/quant_lightning_indexer_v2_pt_save.py          # Reads the Excel table and generates case PT files in batch
- ./batch/list_pt_from_excel.py                           # Extracts Testcase_Name from the Excel file and matches PT files by name (used in batch_exec mode)

## Architecture

- **Data generation entry**: `generate_qliv2_test_data` reuses the existing batch data pipeline to generate inputs and CPU golden data; it does not call metadata or the main operator. Parameter preparation can still query device information.
- **single mode**: Directly executes the configured cases. When `--save-pt` is specified, it saves and executes the same actual input to avoid a second random generation.
- **batch mode**: `-P` is the only PT directory. With `-E`, the script generates PT files before execution; without `-E`, it directly executes existing PT files.
- **batch_exec mode**: Filters existing PT files by `Testcase_Name` in the Excel file and executes only the NPU test and accuracy comparison.
- Both paths share `_qliv2_prepare_tensors_and_metadata` and `_qliv2_run_compiled_graph`, and consistently use `fullgraph=False`.
- The result table is written to disk first. When the accuracy result of index or return value is `Failed`, pytest then exits with a non-zero status.

## Usage

Run the commands in the pytest folder:

### Running Test Cases

#### Single-Case Debugging

The default parameter set suits temporary debugging. Configure `test_quant_lightning_indexer_v2_paramset.py` first, then run:

``` bash
bash test_run.sh single
bash test_run.sh single --save-pt ./single_pt -O ./result/single.xlsx
bash test_run.sh single -M graph --save-pt ./single_pt
```

The STC parameter set is located in `test_quant_lightning_indexer_v2_stc.py` and runs white-box cases by code path:

``` bash
bash test_run.sh single --paramset stc
```

You can specify one or more cases with `-C` or `--cases`; separate multiple names with commas:

``` bash
# Specify a single-combination case
bash test_run.sh single --paramset stc -C MXFP8_S1_21

# Specify a multi-combination case and run all of its expanded nodes
bash test_run.sh single --paramset stc -C MXFP8_META_70

# Specify one of the expanded nodes; _002 corresponds to return_value=1
bash test_run.sh single --paramset stc -C MXFP8_META_70_002

# Run multiple specified cases at a time
bash test_run.sh single --paramset stc -C MXFP8_S1_21,MXFP8_META_73

# Specify the graph mode; the eager mode is used by default
bash test_run.sh single --paramset stc -C MXFP4_META_70_002 -M graph
```

#### Batch Case Generation and Testing

##### Method A: Batch Execution with test_run.sh

1. Place the case Excel table under the Excel path.

`-P` specifies both the PT save directory and the PT read directory; the default is `pt_path` under the current pytest directory.

##### Directly Executing Existing PT Files

Without `-E`, the script does not read the Excel file or regenerate PT files; it only executes the NPU test and comparison:

3. Run the commands:

``` bash
bash test_run.sh batch -P ./pt_path
bash test_run.sh batch -P ./pt_path -O ./result/rerun.xlsx
bash test_run.sh batch -P ./pt_path -C case_b,case_a       # By name in the given order
bash test_run.sh batch -P ./pt_path -I 3,1,5-7             # By index after natural sorting
bash test_run.sh batch -P ./pt_path -M graph
```

4. Default values in the configuration section:

| Variable | Default Value | Command-Line Option | Description |
|---|---|---|---|
| DEFAULT_EXCEL | `./excel/test_cases.xlsx` | `-E` | Path to the Excel case table (**a specific file name is required**; wildcards such as `./excel/*` are not supported) |
| DEFAULT_PT_PATH | `./pt_path` | `-P` | Directory for storing PT files |
| (None) | `Sheet1` | `-S` | Excel sheet name |
| (None) | `eager` | `-M` | Run mode (eager/graph) |

#### Selecting Existing PT Files from the Excel Table for Batch Execution (batch_exec Mode)
>
> The script re-runs only the NPU test and accuracy comparison and does not regenerate PT files. This mode applies when PT files already exist and only accuracy results need updating.

After `-E` is added, the script first generates the Excel cases into `-P` and then executes them from the same directory:

2. Run the commands:

``` bash
bash test_run.sh batch -E ./excel/test_cases.xlsx -P ./pt_path
bash test_run.sh batch -E ./excel/test_cases.xlsx -S Sheet1 -P ./pt_path
bash test_run.sh batch -E ./excel/test_cases.xlsx -P ./pt_path -O ./result/batch.xlsx
```

3. Execution flow:

- Reads the `Testcase_Name` column from the Excel table
- Matches the corresponding .pt files under pt_path by `<Testcase_Name>.pt`
- Executes the NPU test and accuracy comparison only for the matched .pt files
- Generates the `result.xlsx` test result table
- If a case in the Excel file has no corresponding .pt file, the script outputs a warning and skips the case

4. Differences from the `batch` mode:

|  | `batch` | `batch_exec` |
|---|---|---|
| PT generation | Regenerated each time | Skipped |
| Execution speed | Slower (includes PT generation) | Faster |
| Applicable scenario | First run / parameter change | Accuracy retest / NPU result update only |

##### Method B: Manual Step-by-Step Execution

1. Generate PT files:

``` bash
python3 batch/quant_lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path
python3 batch/quant_lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path --sheet Sheet1  # Specify the sheet
```

2. Replace the test script paths:

``` bash
QLIV2_TESTCASE_DIR=pt_path QLIV2_RESULT_PATH=result.xlsx \
python3 -m pytest -rA -s test_quant_lightning_indexer_v2_batch.py -v -m ci
```

3. Run the tests:

``` bash
python3 -m pytest -rA -s test_quant_lightning_indexer_v2_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

4. Restore the test script:

``` bash
cp test_quant_lightning_indexer_v2_batch.py.bak test_quant_lightning_indexer_v2_batch.py
```

##### Method C: Batch Isolated Execution (Recommended for Performance Profiling)

The script starts a separate pytest process for each case, achieving complete inter-process isolation and preventing a crashed case from affecting other cases.

``` bash
bash batch_isolated_run.sh ./pt_path 0         # Without performance profiling
bash batch_isolated_run.sh ./pt_path 1         # With performance profiling (mounts msprof)
bash batch_isolated_run.sh ./pt_path 0 graph   # Graph mode + no performance profiling
bash batch_isolated_run.sh ./pt_path 1 graph   # Graph mode + performance profiling
```

## Excel Case Table Format

`excel/test_cases.xlsx` must contain the following columns (Sheet1):

| Column Name | Type | Example |
|---|---|---|
| Testcase_Name | str | `test_case_01` |
| batch_size | int | `8` |
| q_seq | int | `15` |
| k_seq | int | `111` |
| q_t_size | int | `8` |
| k_t_size | int | `15` |
| q_head_num | int | `64` |
| k_head_num | int | `1` |
| head_dim | int | `128` |
| block_size | int | `512` |
| block_num | int | `8` |
| qk_dtype | str | `FLOAT8_E4M3FN` / `INT8` / `HIFLOAT8` / `FLOAT4_E2M1` |
| dequant_dtype | str | `FP32` / `FP16` / `FLOAT8_E8M0` |
| actual_seq_dtype | str | `INT32` |
| cu_seqlens_q | None/str | `None` or `"[0, 1]"` |
| cu_seqlens_k | None/str | `None` or `"[0, 1]"` |
| seqused_q | None/str | `None` or `"[3,3,3,3,3,3,3,3]"` |
| seqused_k | str | `"[28,24,80,96,47,76,0,111]"` |
| cmp_residual_k | None/str | `None` or `"[0,0,0,0,0,0,0,0]"` (required when cmp_ratio>1)|
| max_seqlen_q | int | `-1` |
| quant_mode | int | `1` / `2` / `4` |
| layout_query | str | `BSND` / `TND` |
| layout_key | str | `PA_BBND` |
| sparse_count | int | `512` |
| sparse_mode | int | `0` / `3` |
| query_datarange | str | `"[-448,448]"` |
| key_datarange | str | `"[-20,20]"` |
| weights_datarange | str | `"[-123,123]"` |
| q_scale_datarange | str | `"[0,255]"` |
| k_scale_datarange | str | `"[0,65504]"` |
| cmp_ratio | int | `1` / `4` |
| return_value | int | `0` / `1` |
| output_idx_offset | None/str | `None` or a list string |

**Precautions**:

- `dequant_dtype`: For Ascend950 with `quant_mode=3/5`, only `FLOAT8_E8M0` is supported; other quantization modes support `FP32`. Ascend910_93 supports `FP16`.
- When `cmp_ratio > 1` and `sparse_mode != 0`, `cmp_residual_k` is required (a list with a length equal to batch_size).
- When `return_value=1`, `output_idx_offset` must be set to a valid value.
- Ascend910_93 requires `quant_mode=2`.

## Output Files

| File | Description |
|---|---|
| `result.xlsx` | Test results (accuracy, parameters, and so on) |
| `result_perf.xlsx` | Test results + performance data (msprof mode only) |
| `batch_summary.log` | Detailed log of batch execution |
| `batch_fail_list.log` | List of failed cases |
| `PROF_*/` | Directory of raw msprof performance data |
