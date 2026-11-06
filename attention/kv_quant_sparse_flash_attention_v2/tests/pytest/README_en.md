# kv_quant_sparse_flash_attention_v2 pytest Test Framework

English | [简体中文](./README.md)

## Description

This framework verifies the functions of the kv_quant_sparse_flash_attention_v2 operator based on the pytest test framework:

- **CPU side**: reproduces the operator function to generate golden data
- **NPU side**: directly invokes the operator through TorchNPU to obtain actual data
- **Accuracy comparison**: compares the accuracy between CPU and NPU results to verify the operator function

The following main workflows are supported:

- `single`: constructs inputs directly from the fixed parameters in `kv_quant_sparse_flash_attention_v2_paramset.py` and launches NPU single-operator execution.
- `batch_save`: reads parameters from Excel and generates `.pt` case files containing CPU golden data.
- `gen_excel_from_paramset`: generates Excel files from a paramset.
- `batch_exec`: replays and executes the NPU operator in batches from existing `.pt` files and compares accuracy.

## Current Implementation Scope

### Parameter Description

The following parameter constraints have been moved into framework validation and this document for centralized maintenance:

- `layout_query`: supports `BSND` and `TND`
- `layout_kv`: supports `BSND`, `TND`, and `PA_BSND`
- In non-PA scenarios, `layout_query == layout_kv` is required
- `q_type`: supports only `torch.float16` and `torch.bfloat16`
- `kv_dtype`: supports `hifloat8` and `float8_e4m3fn`, and `None` is also compatible as the default generation path of `float8_e4m3fn`
- `N1`: supports only `1/2/4/8/16/32/48/64`
- `N2`: supports only `1`
- `sparse_mode`: supports only `0` and `3`
- `sparse_block_size`: currently supports only `1`
- `key_quant_mode` / `value_quant_mode`: support only `2`
- `tile_size`: currently supports only `128`
- `quant_scale_repo_mode`: currently supports only `1`
- `attention_mode`: supports `0` and `2`. When the value is `2`, `rope_head_dim` must be `64`
- `block_size`: takes effect only in `PA_BSND` and must be a multiple of 16
- `actual_seq_q` / `actual_seq_kv`: if passed, the length must equal `B`

For more complete operator definitions and input constraints, also refer to the following:

- `attention/kv_quant_sparse_flash_attention_v2/README.md`

### Environment Configuration

#### Prerequisites

1. Complete the environment installation and environment variable configuration. For details, see [Environment Deployment](../../../../docs/zh/install/quick_install.md).
2. TorchNPU installation package download path (replace it with the latest version in time): [Attention Fusion Operators Experimental Usage Instructions](../../../../experimental/attention/Attention融合算子Experimental使用说明.md)

#### Custom Package Invocation

Custom package invocation is supported.

## File Structure

```text
pytest/
├── README.md
├── pytest.ini   # Create test markers
├── test_run.sh   # Execution script
├── check_valid_param.py   # Parameter constraint interception
├── kv_quant_sparse_flash_attention_v2_golden.py  # tensor conversion/CPU-side operator golden implementation
├── kv_quant_sparse_flash_attention_v2_paramset.py  # Single-case input parameter configuration
├── result_compare_method.py  # Output accuracy comparison
├── utils.py   # Parameter parsing/CPU NPU execution entry
├── test_kv_quant_sparse_flash_attention_v2_single.py # Main program for single-case execution
├── test_kv_quant_sparse_flash_attention_v2_batch.py # Batch NPU tests executed from pt files
└── batch/
    ├── kv_quant_sparse_flash_attention_v2_process.py # NPU interfaces
    ├── test_kv_quant_sparse_flash_attention_v2_pt_save.py  # Generate pt files in batches from Excel
    ├── gen_excel_from_paramset.py # Generate Excel files from a paramset
    └── excel/
        ├── example.xlsx  # Sample Excel case file
        └── gen_example_xlsx.py  # Script for generating the sample Excel file
```

## Usage

Run the following in the `attention/kv_quant_sparse_flash_attention_v2/test/pytest` directory:

### Command Format

```bash
bash test_run.sh <mode> [-E excel_path] [-S sheet] [-P path] [-O output_path]
```

### Parameter Options

| Option | Description | Applicable Modes |
| --- | --- | --- |
| `-E excel_path` | Specifies the Excel file path. Default: `./excel/example.xlsx` | batch_save |
| `-S sheet` | Specifies the Excel sheet name. Default: `Sheet1` | batch_save/gen_excel_from_paramset |
| `-P path` | Specifies a path (the meaning differs across modes; see the table below for details) | single/batch_save/batch_exec/gen_excel_from_paramset |

| Mode | Meaning of `-P` | Default Value |
| --- | --- | --- |
| single | paramset file name | `kv_quant_sparse_flash_attention_v2_paramset` |
| batch_save | pt file saving path | `./pt_files/` |
| batch_exec | pt file execution path (directory or single file) | `./pt_files/` |
| gen_excel_from_paramset | paramset file name | `kv_quant_sparse_flash_attention_v2_paramset` |

**Additional parameters in gen_excel_from_paramset mode:**

| Option | Description | Default Value |
| --- | --- | --- |
| `-E excel_output` | Output Excel file path | `./excel/example.xlsx` |
| `-S sheet` | Excel sheet name | `Sheet1` |

### single

Manually configure the parameters in `kv_quant_sparse_flash_attention_v2_paramset.py`, or use `-P` to specify another paramset file.

```bash
bash test_run.sh single                              # Use the default paramset
bash test_run.sh single -P my_paramset                # Use the specified paramset file
```

### batch_save

Reads parameters from Excel and generates `.pt` case files containing CPU golden data.

```bash
bash test_run.sh batch_save                           # Use the default Excel and sheet
bash test_run.sh batch_save -E ./test.xlsx            # Specify the Excel file
bash test_run.sh batch_save -E ./test.xlsx -S Sheet1  # Specify the Excel file and sheet
bash test_run.sh batch_save -E ./test.xlsx -S Sheet1 -P ./output_pt/  # Specify all parameters
bash test_run.sh batch_save -S Sheet1 -E ./test.xlsx  # Parameters can be in any order
```

### gen_excel_from_paramset

Generates Excel files from a paramset.

```bash
bash test_run.sh gen_excel_from_paramset                           # Use the default paramset
bash test_run.sh gen_excel_from_paramset -P my_paramset            # Specify the paramset file
bash test_run.sh gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx  # Specify the output path
bash test_run.sh gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx -S decode  # Specify the sheet name
```

### batch_exec

Replays and executes the NPU operator in batches from `.pt` files and compares accuracy.

```bash
bash test_run.sh batch_exec                           # Execute all pt files in the default directory
bash test_run.sh batch_exec -P ./pt_files/test.pt     # Execute a single pt file
bash test_run.sh batch_exec -P ./custom_pt_dir/       # Execute all pt files in the specified directory
```

The following Excel cases can be used directly for reference. Column names must be consistent with the fields read by the batch framework:

| Testcase_Prefix | Testcase_Number | layout_query | layout_kv | q_type | kv_type | B | S1 | S2 | N1 | N2 | D | K | scale_value | key_quant_mode | value_quant_mode | sparse_block_size | tile_size | rope_head_dim | sparse_mode | attention_mode | quant_scale_repo_mode | block_size | block_num | actual_seq_q | actual_seq_kv | enable_sinks | range_sinks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tnd_sample | 1 | TND | TND | torch.bfloat16 | hifloat8 | 2 | 8 | 8 | 16 | 1 | 512 | 2048 | 0.04166666666666666 | 2 | 2 | 1 | 128 | 64 | 3 | 2 | 1 | 256 |  | [5,8] | [6,8] | true | [-1,1] |

## Result Files

- `result.xlsx`: records the key information, execution status, and `fulfill_percent` of each case
- `./pt_files/*.pt`: intermediate test cases generated by the batch workflow
