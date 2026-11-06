# Bandwidth Test Usage Instructions

English | [简体中文](./README.md)

## Overview

`bandwidth_test.py` is a script for testing the NPU bandwidth test operator (`npu_bandwidth_test`). It supports two modes, precision verification and performance testing, and can run single-configuration or multi-configuration batch tests.

## Features

- **Precision test**: Verifies the consistency between the operator output and the Golden data
- **Bandwidth test**: Measures communication bandwidth performance and supports profiling data collection
- **Multi-configuration test**: Supports batch tests with multiple data sizes and automatically generates a performance summary report
- **Communicator reuse**: Shares the communicator during multi-configuration tests, reducing creation/destruction overhead

## Environment Requirements

- Python 3.8+
- PyTorch
- TorchNPU
- npu_ops_transformer
- pandas, numpy

## Quick Start

```bash
python bandwidth_test.py
```

## Configuration Description

### Test Mode Configuration

```python
# The bandwidth test and the precision test cannot be enabled at the same time
is_precision_test = True   # Whether to perform the precision test
is_bandwidth_test = False # Whether to perform the bandwidth test
is_single_test = True     # True: single-configuration test; False: multi-configuration batch test
```

### Core Configuration

```python
graph_type = 0            # 0: single-operator mode; 3: aclgraph mode
is_full_core = 1         # 1: use all cores; 0: use some cores
core_num = 16            # Number of cores when is_full_core=0
aiv_num = 32             # Number of AIV cores (core_num * 2). In later versions, core_num and aiv_num may have a ratio of 1:1
```

### Profiling Configuration

```python
loop = 100               # Total number of iterations
repeat = 1               # Number of repeats
warmup = 5               # Number of warmup iterations
profiling_path = './Profiling_Result_bandwidth_test'  # Profiling result path
```

### Communicator Configuration

```python
server_num = 1           # Number of servers
is_cloud = 0             # 0: local environment; 1: cloud environment
rank_per_dev = 2         # Number of ranks per device
world_size = server_num * rank_per_dev  # Total world size
```

**Cloud environment configuration**: When `is_cloud = 1`, the following environment variables must be set:

- `VC_TASK_INDEX`: Index of the current task
- `VC_WORKER_HOSTS`: List of worker hosts

### Data Configuration

```python
x_dtype = 'fp16'         # Data type: 'fp16'
random_seed = 90         # Random seed
```

### Operator Configuration

```python
mode = 0                 # Operator mode
comm_alg = "mte"         # Communication algorithm
```

### Test Data Configuration

**Single-configuration test** (`is_single_test = True`):

```python
bs = 2                   # Batch size
data_size = 8192         # Data dimension H
max_bs = 2               # Maximum batch size
```

**Multi-configuration test** (`is_single_test = False`):

```python
test_configs = [
    {'bs': 1, 'data_size': 1024, 'max_bs': 1},     # 2KB
    {'bs': 2, 'data_size': 1024, 'max_bs': 2},     # 4KB
    {'bs': 2, 'data_size': 2048, 'max_bs': 2},     # 8KB
    # ... more configurations
]
```

### Other Configuration

```python
is_save_tensor_to_bin = False  # Whether to save the tensor to a bin file
is_check_setting = True        # Whether to check the validity of the configuration
```

## Usage Examples

### Example 1: Single-Configuration Precision Test

```python
is_precision_test = True
is_bandwidth_test = False
is_single_test = True
bs = 4
data_size = 4096
max_bs = 4
```

### Example 2: Single-Configuration Bandwidth Test

```python
is_precision_test = False
is_bandwidth_test = True
is_single_test = True
bs = 64
data_size = 8192
max_bs = 64
```

### Example 3: Multi-Configuration Batch Bandwidth Test

```python
is_precision_test = False
is_bandwidth_test = True
is_single_test = False
# Use the preset multiple configurations
```

### Example 4: Combined Precision and Bandwidth Test

```python
is_precision_test = True
is_bandwidth_test = True
is_single_test = True
```

## Output Description

### Precision Test Output

```
[PASS] rank_0_y 精度对比通过
[PASS] rank_0_receive_cnt 精度对比通过
```

### Bandwidth Test Output

```
============================================================
Bandwidth Test Performance Results
============================================================
Single rank data: 128.0000 KB
Total data per iteration: 256.00 MB
World size: 2
BS per rank: 64
Data size (H): 4096
------------------------------------------------------------
Mean execution time: 123.56 us
Min execution time:  60.00 us
Max execution time:  633.05 us
------------------------------------------------------------
Mean bandwidth: 77.44 GB/s
Max bandwidth:  113.33 GB/s
============================================================
```

### Multi-Configuration Test Summary Table

```
==========================================================================================
Bandwidth Test Summary - All Configurations
==========================================================================================
No.   BS    H          Total Data(KB)   Mean BW(GB/s)    Max BW(GB/s)     Mean Time(us)
------------------------------------------------------------------------------------------
1     1     1024       2.0000           45.23            48.12            88.56
2     2     1024       4.0000           78.45            82.31            102.34
...
```

## Core Function Description

| Function Name | Description |
|--------|------|
| `run_precision_test` | Runs the precision test |
| `run_bandwidth_test` | Runs the single-configuration bandwidth test |
| `run_multi_config_bandwidth_test` | Runs the multi-configuration bandwidth test (shares the communicator) |
| `generate_inputs` | Generates test input data |
| `gen_golden_data` | Generates the Golden reference data |
| `compare_result` | Compares the NPU output with the Golden data |
| `get_bandwidth_performance` | Extracts performance data from the profiling result |

## Precautions

1. **Communicator creation**: Multi-configuration tests reuse the communicator, avoiding the overhead of repeated creation and destruction
2. **HCCL_BUFFSIZE**: The script automatically calculates and sets the `HCCL_BUFFSIZE` environment variable based on the test configuration
3. **Profiling path**: In multi-configuration tests, each configuration has an independent profiling subdirectory
4. **Memory usage**: Pay attention to the NPU memory limit when running tests with large amounts of data
