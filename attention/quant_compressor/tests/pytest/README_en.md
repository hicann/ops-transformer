# QuantCompressor Operator Test Framework

English | [简体中文](./README.md)

## Description

Based on the pytest test framework, this framework implements function verification of the QuantCompressor operator:

- **CPU side**: reproduces the operator function to generate golden data
- **NPU side**: directly invokes the operator through torch_npu to obtain actual data
- **Accuracy comparison**: compares the accuracy of CPU and NPU results to verify the operator function
- **Dual-mode execution isolation**: supports two batch modes, direct pytest multi-process execution and shell-layer process isolation
- **Performance collection**: supports mounting msprof to collect operator performance data and summarize the output

## Current Implementation Scope

### Parameter Restrictions

- D of 128/512 is supported.
- H from 1K to 10K, aligned to 512, is supported.
- cmp_ratio of 2/4/8/16/32/64/128 is supported. The following three typical scenarios are supported:
    - C4A: D=512, coff=2, cmp_ratio=4;
    - C4Li: D=128, coff=2, cmp_ratio=4;
    - C128A: D=512, coff=1, cmp_ratio=128.

### Environment Configuration

#### Prerequisites

1. Confirm that torch_npu is the latest version.

#### Custom Package Invocation

Custom package invocation is supported.

## File Structure

### pytest File Structure Description

- test_run.sh                                  # Execution script, supporting the single/batch commands

- batch_isolated_run.sh                        # Batch isolated execution script (shell-layer process isolation + msprof performance collection)

- quant_compressor_golden.py                   # CPU-side operator golden implementation and accuracy comparison

- collect_perf_data.py                         # msprof performance data collection and summarization

- pytest.ini                                   # Creates test markers

Single-case test:

- test_quant_compressor_single.py              # Main program for running a single test case

- quant_compressor_operator_single.py          # CPU-side operator logic implementation for obtaining golden data and direct NPU operator invocation

- test_quant_compressor_paramset.py            # Input parameter configuration of a single case

Batch test:

- test_quant_compressor_batch.py               # Main program for batch case testing, generating an Excel file to save results (supports the isolated mode)

- ./batch/quant_compressor_pt_loadprocess.py   # Reads pt files and invokes the operator to obtain NPU output

- ./batch/quant_compressor_pt_save.py          # Reads the Excel sheet and generates case pt files in batches

- ./batch/replace_path.py                      # Placeholder replacement for test_quant_compressor_batch.py

## Usage

Run the commands in the pytest directory:

### Running Test Cases

#### Single-Case Debugging

1. Manually configure the ENABLED_PARAMS parameter in test_quant_compressor_paramset.py.

2. Run the command:

``` bash
bash test_run.sh single
```

#### Batch Generation and Testing of Cases

##### Method A: Batch Execution Using test_run.sh

1. Store the case Excel sheet in the excel directory.

2. Set the case Excel sheet path (PATH1) and the pt file storage path (PATH2) in test_run.sh.

3. Run the command:

``` bash
bash test_run.sh batch
```

##### Method B: Manual Step-by-Step Execution

1. Generate pt files:

``` bash
python3 batch/quant_compressor_pt_save.py "excel/*.xlsx" pt_path
```

2. Replace the test script path:

``` bash
python3 batch/replace_path.py test_quant_compressor_batch.py pt_path
```

3. Run the test:

``` bash
python3 -m pytest -rA -s test_quant_compressor_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

4. Restore the test script:

``` bash
cp test_quant_compressor_batch.py.bak test_quant_compressor_batch.py
```

##### Method C: Batch Isolated Execution (recommended for performance collection)

This method launches a separate pytest process for each case, achieving complete inter-process isolation and preventing a crashed case from affecting other cases.
Case acquisition logic: The `Testcase_Name` column of the Excel file in the `./excel` directory is read first and mapped to the `.pt` file paths. If no Excel file exists or the `Testcase_Name` column is missing, the method automatically falls back to scanning all `.pt` files in the case directory.

``` bash
bash batch_isolated_run.sh ./pt_path 0         # Do not collect performance
bash batch_isolated_run.sh ./pt_path 1         # Collect performance (mount msprof)
```

## Excel Case Sheet Format

Sheet1 of `excel/test_cases.xlsx` must contain the following columns:

| Column Name | Type | Example | Description |
|---|---|---|---|
| Testcase_Name | str | `Prefill0` | Case name, also used as the generated .pt file name |
| batch_size | int | `1` | |
| hidden_size | int | `4096` | |
| Seq_len | int | `8192` | |
| head_dim | int | `128` / `512` | |
| block_size | int | `128` | |
| cmp_ratio | int | `4` / `128` | |
| coff | int | `1` / `2` | |
| start_p | int | `0` / `8192` | |
| cache_mode | int | `1` / `2` | 1: continuous buffer, 2: circular buffer |
| layout_x | str | `TH` / `BSH` | |
| data_type | str | `BF16` / `FP16` | |
| cu_seqlens | None/str | `None` or `0,8192` | Required for the TH layout, separated by commas |
| seqused | None/str | `None` or `8192` | Separated by commas |
| start_pos | None/str | `None` or `0` | Separated by commas |
| x_datarange | str | `-10,10` | Separated by commas |
| wkv_datarange | str | `-10,10` | Separated by commas |
| wgate_datarange | str | `-10,10` | Separated by commas |
| ape_datarange | str | `-10,10` | Separated by commas |
| kv_state_datarange | str | `-10,10` | Separated by commas |
| score_state_datarange | str | `-10,10` | Separated by commas |
| x_descale_datarange | str | `0.001,1.0` | Separated by commas |
| wkv_descale_datarange | str | `0.001,1.0` | Separated by commas |
| wgate_descale_datarange | str | `0.001,1.0` | Separated by commas |
| quant_mode | int | `1` | 1: A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL |

**Precautions**:

- When `layout_x=TH`, `cu_seqlens` is required (a list with a length of batch_size+1)

## Output Files

| File | Description |
|---|---|
| `result.xlsx` | Test results (accuracy, parameters, and so on) |
| `result_perf.xlsx` | Test results + performance data (msprof mode only) |
| `batch_summary.log` | Detailed log of batch isolated execution |
| `batch_fail_list.log` | List of failed cases |
| `PROF_*/` | Directory of raw msprof performance data |
