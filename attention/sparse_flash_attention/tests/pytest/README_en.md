# sparse_flash_attention pytest Test Framework

English | [简体中文](./README.md)

## Description

Based on the pytest test framework, this directory implements the function verification of the sparse_flash_attention operator:

- **CPU side**: Reproduces the operator function to generate golden data
- **NPU side**: Directly calls the operator through `torch_npu.npu_sparse_flash_attention` to obtain actual data. When `sinks` is passed in, the upper-layer API calls `aclnnSparseFlashAttentionV2`; when `sinks` is not passed in, it calls `aclnnSparseFlashAttention`
- **Precision comparison**: Compares the precision between CPU and NPU results to verify the operator function

Supports the following main flows:

- `single`: Builds inputs directly from the fixed parameters in `sparse_flash_attention_paramset.py` and launches NPU single-operator execution.
- `batch_save`: Reads parameters from Excel and generates `.pt` case files containing CPU golden data.
- `gen_excel_from_paramset`: Generates an Excel file from a paramset.
- `batch_exec`: Replays and executes the NPU operator in batches from existing `.pt` files and compares precision.

## Current Implementation Scope

### Parameters

The following parameter constraints have been moved into framework validation and this document for unified maintenance:

- `layout_query`: supports `BSND` and `TND`
- `layout_kv`: supports `BSND`, `TND`, and `PA_BSND`
- The non-PA scenario requires `layout_query == layout_kv`
- `q_type`: supports only `torch.float16` and `torch.bfloat16`
- `kv_type`: supports only `torch.float16` and `torch.bfloat16`
- `N2`: supports only `1`
- `g = N1 / N2`: supports only `1/2/4/8/16/32/64/128`
- `D`: currently supports only `512`
- `rope_head_dim`: currently supports only `64`
- `attention_mode`: currently supports only `2`
- `sparse_mode`: supports only `0` and `3`
- `sparse_block_size`: currently supports only `1`; it must be in `[1, 128]` and be a power of 2
- `K`: must not exceed `ceil(S2 / sparse_block_size)`
- `block_size`: takes effect only in `PA_BSND`; it must be a positive integer and aligned to 16
- `block_num`: must cover the actual KV length
- `actual_seq_q` / `actual_seq_kv`: if passed in, the length must equal `B`

For more complete operator definitions and input constraints, also refer to:

- `attention/sparse_flash_attention/README.md`

### Environment Configuration

#### Prerequisites

1. Complete the environment installation and environment variable configuration. For details, refer to [Environment Deployment](../../../../docs/zh/install/quick_install.md).
2. TorchNPU installation package download path (replace it with the latest version in time): [Attention Fused Operator Experimental Usage Instructions](../../../../experimental/attention/Attention融合算子Experimental使用说明.md)

#### Custom Package Invocation

Custom package invocation is supported.

## File Structure

```text
pytest/
├── README.md
├── pytest.ini   # Creates test markers
├── test_run.sh   # Execution script
├── check_valid_param.py   # Parameter constraint interception
├── sparse_flash_attention_golden.py  # Tensor conversion/CPU-side operator golden implementation
├── sparse_flash_attention_paramset.py  # Single-case input parameter configuration
├── result_compare_method.py  # Output precision comparison
├── utils.py   # Parameter parsing/CPU NPU execution entry
├── test_sparse_flash_attention_single.py # Main program for single-case execution
├── test_sparse_flash_attention_batch.py # Batch NPU test execution from pt files
└── batch/
    ├── sparse_flash_attention_process.py # NPU API
    ├── test_sparse_flash_attention_pt_save.py  # Batch generation of pt files from Excel
    ├── gen_excel_from_paramset.py # Generates Excel files from paramsets
    └── excel/
        ├── example.xlsx  # Sample Excel case file
        └── .gitkeep   # Directory placeholder
```

## Usage

Run the commands in the `attention/sparse_flash_attention/tests/pytest` directory:

### Command Format

```bash
bash test_run.sh <mode> [-E excel_path] [-S sheet] [-P path] [-O output_path]
```

### Parameter Options

| Option | Description | Applicable Modes |
| --- | --- | --- |
| `-E excel_path` | Specifies the Excel file path. Default: `./excel/example.xlsx` | batch_save |
| `-S sheet` | Specifies the Excel sheet name. Default: `Sheet1` | batch_save/gen_excel_from_paramset |
| `-P path` | Specifies a path (the meaning differs by mode; see the following table for details) | single/batch_save/batch_exec/gen_excel_from_paramset |

| Mode | Meaning of `-P` | Default Value |
| --- | --- | --- |
| single | paramset file name | `sparse_flash_attention_paramset` |
| batch_save | pt file save path | `./pt_files/` |
| batch_exec | pt file execution path (directory or single file) | `./pt_files/` |
| gen_excel_from_paramset | paramset file name | `sparse_flash_attention_paramset` |

**Additional parameters in gen_excel_from_paramset mode:**

| Option | Description | Default Value |
| --- | --- | --- |
| `-E excel_output` | Output Excel file path | `./excel/example.xlsx` |
| `-S sheet` | Excel sheet name | `Sheet1` |

### single

Manually configure the parameters in `sparse_flash_attention_paramset.py`, or use `-P` to specify another paramset file.

```bash
bash test_run.sh single                              # Use the default paramset
bash test_run.sh single -P my_paramset                # Use the specified paramset file
```

### batch_save

Reads parameters from Excel and generates `.pt` case files containing CPU golden data.

```bash
bash test_run.sh batch_save                           # Use the default Excel file and sheet
bash test_run.sh batch_save -E ./test.xlsx            # Specify the Excel file
bash test_run.sh batch_save -E ./test.xlsx -S Sheet1  # Specify the Excel file and sheet
bash test_run.sh batch_save -E ./test.xlsx -S Sheet1 -P ./output_pt/  # Specify all parameters
bash test_run.sh batch_save -S Sheet1 -E ./test.xlsx  # Parameters can be in any order
```

### gen_excel_from_paramset

Generates an Excel file from a paramset.

```bash
bash test_run.sh gen_excel_from_paramset                           # Use the default paramset
bash test_run.sh gen_excel_from_paramset -P my_paramset            # Specify the paramset file
bash test_run.sh gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx  # Specify the output path
bash test_run.sh gen_excel_from_paramset -P my_paramset -E ./output/example.xlsx -S decode  # Specify the sheet name
```

### batch_exec

Replays and executes the NPU operator in batches from `.pt` files and compares precision.

```bash
bash test_run.sh batch_exec                           # Execute all pt files in the default directory
bash test_run.sh batch_exec -P ./pt_files/test.pt     # Execute a single pt file
bash test_run.sh batch_exec -P ./custom_pt_dir/       # Execute all pt files in the specified directory
```

The following is an Excel case for direct reference. The column names must be consistent with the fields read by the batch framework:

| Testcase_Prefix | layout_query | layout_kv | q_type | kv_type | B | T1 | T2 | S1 | S2 | N1 | N2 | D | K | scale_value | sparse_block_size | rope_head_dim | sparse_mode | attention_mode | return_softmax_lse | use_sinks | block_size | block_num | actual_seq_q | actual_seq_kv | range_query | range_key | range_query_rope | range_key_rope | range_sinks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bsnd_basic | BSND | BSND | torch.float16 | torch.float16 | 1 |  |  | 5 | 128 | 8 | 1 | 512 | 16 | 0.041666666666666664 | 1 | 64 | 0 | 2 | True | True |  |  | [4] | [4] | [-10.0, 100.0] | [5.0, 100.0] |  |  | [-0.2, 0.2] |

## Result Files

- `result.xlsx`: Records the key information, execution status, and `fulfill_percent` of each case
- `./pt_files/*.pt`: Intermediate test cases generated by the batch flow
