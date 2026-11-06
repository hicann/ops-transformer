# quant_lightning_indexer Operator Test Framework

English | [简体中文](./README.md)

## Function Description

Based on the pytest test framework, this implements the functional verification of the quant_lightning_indexer operator:

- **CPU side**: reproduces the operator function to generate golden data
- **NPU side**: directly invokes the operator through TorchNPU to obtain actual data
- **Accuracy comparison**: performs an accuracy comparison between CPU and NPU results to verify the operator function

## Current Implementation Scope

### Parameter Restrictions

- **Data formats**:
  - **query_layout**: BSND, TND
  - **key_layout**: PA_BSND

### Environment Configuration

#### Prerequisites

1. Complete the environment installation and environment variable configuration. For details, see [Environment Deployment](../../../../docs/zh/install/quick_install.md).
2. Download path of the TorchNPU installation package (replace it with the latest version in time): [Attention Fusion Operator Experimental Usage Instructions](../../../../experimental/attention/Attention融合算子Experimental使用说明.md)

#### custom Package Invocation

custom package invocation is supported.

## File Structure

### pytest File Structure Description

- test_run_sh                                  # Execution script
- quant_lightning_indexer_golden.py            # CPU-side operator golden implementation
- result_compare_method.py                     # Accuracy comparison between CPU golden and NPU output
- pytest.ini                                   # Creates test markers

Single-case testing:

- test_quant_lightning_indexer_single.py       # Main program for running a single case in pytest
- test_quant_lightning_indexer_paramset.py     # Input parameter configuration for a single case

Batch testing:

- test_quant_lightning_indexer_batch.py        # Main program for batch case testing and generates an Excel file to save results
- ./batch/quant_lightning_indexer_pt_loadprocess.py    # Reads pt files and invokes the operator to obtain NPU output
- ./batch/quant_lightning_indexer_pt_save.py           # Reads the Excel table to batch generate case pt files
- ./batch/replace_path.py                              # Placeholder replacement for test_quant_lightning_indexer_batch.py

## Usage

Execute under the pytest folder path:

### Running Test Cases

#### Single-Case Debugging

1. Manually configure the parameters of test_quant_lightning_indexer_paramset.py

2. Execute the command:

``` bash
bash test_run.sh single
```

#### Batch Case Generation and Testing

1. Place the case Excel tables under the excel path

##### Batch Test Case Table

Batch testing configures test case parameters through Excel files. `tests/pytest/test_run.sh batch` automatically reads the Excel files under the `tests/pytest/excel/` directory and executes the batch test. You need to create the Excel file in that directory yourself, with the format shown in the following table (28 columns in total):

| Column Name | Description |
|:-----|:-----|
| Testcase_Name | Case name |
| batch_size | Batch size |
| q_seq | Sequence Length of Query |
| k_seq | Sequence Length of Key |
| q_t_size | Total Tokens of Query (used in the TND layout) |
| k_t_size | Total Tokens of Key (used in the TND layout) |
| q_head_num | Head Num of Query |
| k_head_num | Head Num of Key (only 1 is supported) |
| head_dim | Head Dim (only 128 is supported) |
| block_size | Number of tokens per block in the PageAttention scenario (must be a multiple of 16, with a maximum of 1024) |
| block_num | Total number of blocks in the PageAttention scenario |
| qk_dtype | Data types of Query and Key |
| weight_dtype | Data type of Weights |
| dequant_dtype | Data type of Dequant Scale |
| actual_seq_dtype | Data type of Actual Seq Lengths |
| act_seq_q | Number of valid tokens of Query (comma-separated; prefix-sum format for TND) |
| act_seq_k | Number of valid tokens of Key (comma-separated; prefix-sum format for TND) |
| query_quant_mode | Query quantization mode (currently only 0 is supported) |
| key_quant_mode | Key quantization mode (currently only 0 is supported) |
| layout_query | Data layout format of Query |
| layout_key | Data layout format of Key |
| sparse_count | Number of indices retained by TopK |
| sparse_mode | Sparse mode (0: defaultMask, 3: rightDownCausal) |
| query_datarange | Query data range (minimum,maximum separated by a comma) |
| key_datarange | Key data range (minimum,maximum separated by a comma) |
| weights_datarange | Weights data range (minimum,maximum separated by a comma) |
| q_scale_datarange | Query Dequant Scale data range (minimum,maximum separated by a comma) |
| k_scale_datarange | Key Dequant Scale data range (minimum,maximum separated by a comma) |

An example of the batch test case table format is as follows:

| Testcase_Name | batch_size | q_seq | k_seq | q_t_size | k_t_size | q_head_num | k_head_num | head_dim | block_size | block_num | qk_dtype | weight_dtype | dequant_dtype | actual_seq_dtype | act_seq_q | act_seq_k | query_quant_mode | key_quant_mode | layout_query | layout_key | sparse_count | sparse_mode | query_datarange | key_datarange | weights_datarange | q_scale_datarange | k_scale_datarange |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|:---|:---|:---:|:---:|:---|:---|:---:|:---:|:---|:---|:---|:---|:---|
| test001 | 4 | 2 | 115 | 4 | 2 | 64 | 1 | 128 | 128 | 10 | INT8 | FP16 | FP16 | INT32 | 1,2,1,1 | 0,352,14,461 | 0 | 0 | BSND | PA_BSND | 512 | 3 | 2,10 | 0,1 | -255,255 | -171,171 | 0,65504 |

2. Set the case Excel table path to read and the pt file storage path in test_run.sh

3. Execute the command:

``` bash
bash test_run.sh batch
```
