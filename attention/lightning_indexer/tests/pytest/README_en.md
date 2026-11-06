# lightning_indexer Operator Test Framework

English | [简体中文](./README.md)

## Description

This framework verifies the functions of the lightning_indexer operator based on the pytest test framework:

- **CPU side**: reproduces the operator function to generate golden data
- **NPU side**: directly invokes the operator through TorchNPU to obtain actual data
- **Accuracy comparison**: compares the accuracy between CPU and NPU results to verify the operator function

## Current Implementation Scope

### Parameter Restrictions

- **Data formats**:
  - **query_layout**: BSND, TND
  - **key_layout**: PA_BSND

### Environment Configuration

#### Prerequisites

1. Complete the environment installation and environment variable configuration. For details, see [Environment Deployment](../../../../docs/zh/install/quick_install.md).
2. TorchNPU installation package download path (replace it with the latest version in time): [Attention Fusion Operators Experimental Usage Instructions](../../../../experimental/attention/Attention融合算子Experimental使用说明.md)

#### Custom Package Invocation

Custom package invocation is supported.

## File Structure

### pytest File Structure Description

- test_run_sh                                  # Execution script
- lightning_indexer_golden.py            # CPU-side operator golden implementation
- result_compare_method.py                     # Accuracy comparison between CPU golden and NPU output
- pytest.ini                                   # Create test markers

Single-case test:

- test_lightning_indexer_single.py       # Main program for running a single case with pytest
- test_lightning_indexer_paramset.py     # Single-case input parameter configuration

Batch test:

- test_lightning_indexer_batch.py        # Main program for batch case tests that generates an Excel file to save results
- ./batch/lightning_indexer_pt_loadprocess.py    # Reads pt files and invokes the operator to obtain NPU output
- ./batch/lightning_indexer_pt_save.py           # Reads Excel tables and generates case pt files in batches
- ./batch/replace_path.py                              # Placeholder replacement for test_lightning_indexer_batch.py

## Usage

Run the following in the pytest folder path:

### Running Test Cases

#### Single-Case Debugging

1. Manually configure the parameters in test_lightning_indexer_paramset.py

2. Run the command:

``` bash
bash test_run.sh single
```

#### Batch Case Generation and Testing

1. Place the case Excel tables in the excel path

2. Set the case Excel table path to read and the pt file storage path in test_run.sh

3. Run the command:

``` bash
bash test_run.sh batch
```
