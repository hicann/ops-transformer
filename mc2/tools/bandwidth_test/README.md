# Bandwidth Test 使用说明

## 概述

`bandwidth_test.py` 是一个用于测试 NPU 带宽测试算子 (`npu_bandwidth_test`) 的脚本，支持精度验证和性能测试两种模式，可进行单配置或多配置批量测试。

## 功能特性

- **精度测试**：验证算子输出与 Golden 数据的一致性
- **带宽测试**：测量通信带宽性能，支持 profiling 性能采集
- **多配置测试**：支持多组数据大小的批量测试，自动生成性能汇总报告
- **通信域复用**：多配置测试时共享通信域，减少创建/销毁开销

## 环境要求

- Python 3.8+
- PyTorch
- TorchNPU
- npu_ops_transformer
- pandas, numpy

## 快速开始

```bash
python bandwidth_test.py
```

## 配置说明

### 测试模式配置

```python
# 带宽测试与精度测试不能同时使能
is_precision_test = True   # 是否进行精度测试
is_bandwidth_test = False # 是否进行带宽测试
is_single_test = True     # True: 单配置测试; False: 多配置批量测试
```

### 核心配置

```python
graph_type = 0            # 0: 单算子模式; 3: aclgraph模式
is_full_core = 1         # 1: 使用全部核心; 0: 使用部分核心
core_num = 16            # is_full_core=0 时的核心数
aiv_num = 32             # AIV 核心数 (core_num * 2)，后续存在core_num与aiv_num = 1:1
```

### 性能采集配置

```python
loop = 100               # 总迭代次数
repeat = 1               # 重复次数
warmup = 5               # 预热迭代次数
profiling_path = './Profiling_Result_bandwidth_test'  # Profiling 结果路径
```

### 通信域配置

```python
server_num = 1           # 服务器数量
is_cloud = 0             # 0: 本地环境; 1: 云环境
rank_per_dev = 2         # 每个设备的 rank 数量
world_size = server_num * rank_per_dev  # 总 world size
```

**云环境配置**：当 `is_cloud = 1` 时，需要设置环境变量：

- `VC_TASK_INDEX`: 当前任务索引
- `VC_WORKER_HOSTS`: Worker 主机列表

### 数据配置

```python
x_dtype = 'fp16'         # 数据类型: 'fp16'
random_seed = 90         # 随机种子
```

### 算子配置

```python
mode = 0                 # 算子模式
comm_alg = "mte"         # 通信算法
```

### 测试数据配置

**单配置测试** (`is_single_test = True`)：

```python
bs = 2                   # Batch size
data_size = 8192         # 数据维度 H
max_bs = 2               # 最大 batch size
```

**多配置测试** (`is_single_test = False`)：

```python
test_configs = [
    {'bs': 1, 'data_size': 1024, 'max_bs': 1},     # 2KB
    {'bs': 2, 'data_size': 1024, 'max_bs': 2},     # 4KB
    {'bs': 2, 'data_size': 2048, 'max_bs': 2},     # 8KB
    # ... 更多配置
]
```

### 其他配置

```python
is_save_tensor_to_bin = False  # 是否保存 tensor 到 bin 文件
is_check_setting = True        # 是否检查配置有效性
```

## 使用示例

### 示例 1：单配置精度测试

```python
is_precision_test = True
is_bandwidth_test = False
is_single_test = True
bs = 4
data_size = 4096
max_bs = 4
```

### 示例 2：单配置带宽测试

```python
is_precision_test = False
is_bandwidth_test = True
is_single_test = True
bs = 64
data_size = 8192
max_bs = 64
```

### 示例 3：多配置批量带宽测试

```python
is_precision_test = False
is_bandwidth_test = True
is_single_test = False
# 使用预设的多组配置
```

### 示例 4：精度+带宽联合测试

```python
is_precision_test = True
is_bandwidth_test = True
is_single_test = True
```

## 输出说明

### 精度测试输出

```
[PASS] rank_0_y 精度对比通过
[PASS] rank_0_receive_cnt 精度对比通过
```

### 带宽测试输出

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

### 多配置测试汇总表

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

## 核心函数说明

| 函数名 | 功能 |
|--------|------|
| `run_precision_test` | 运行精度测试 |
| `run_bandwidth_test` | 运行单配置带宽测试 |
| `run_multi_config_bandwidth_test` | 运行多配置带宽测试（共享通信域） |
| `generate_inputs` | 生成测试输入数据 |
| `gen_golden_data` | 生成 Golden 参考数据 |
| `compare_result` | 对比 NPU 输出与 Golden 数据 |
| `get_bandwidth_performance` | 从 profiling 结果提取性能数据 |

## 注意事项

1. **通信域创建**：多配置测试时会复用通信域，避免重复创建销毁的开销
2. **HCCL_BUFFSIZE**：脚本会根据测试配置自动计算并设置 `HCCL_BUFFSIZE` 环境变量
3. **Profiling 路径**：多配置测试时，每个配置有独立的 profiling 子目录
4. **内存占用**：大数据量测试时注意 NPU 内存限制
