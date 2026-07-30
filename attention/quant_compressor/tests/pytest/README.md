# QuantCompressor算子测试框架

## 功能说明

基于pytest测试框架，实现QuantCompressor算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能
- **双模式执行隔离**：支持直接pytest多进程执行和shell层进程隔离两种批量模式
- **性能采集**：支持挂载msprof采集算子性能数据并汇总输出

## 当前实现范围

### 参数限制

- 支持D为128/512。
- 支持H为1K~10K，512对齐。
- 支持cmp_ratio为2/4/8/16/32/64/128。支持如下三种典型场景：
    - C4A: D=512, coff=2, cmp_ratio=4;
    - C4Li: D=128, coff=2, cmp_ratio=4;
    - C128A: D=512, coff=1, cmp_ratio=128。

### 环境配置

#### 前置要求

1、 确认torch_npu为最新版本

#### custom包调用

支持custom包调用

## 文件结构

### pytest文件结构说明

- test_run.sh                                  # 执行脚本，支持single/batch两种命令

- batch_isolated_run.sh                        # 批量隔离执行脚本（shell层进程隔离+msprof性能采集）

- quant_compressor_golden.py                   # cpu侧算子golden实现以及精度对比

- collect_perf_data.py                         # msprof性能数据收集与汇总

- pytest.ini                                   # 创建测试标记

单用例测试：

- test_quant_compressor_single.py              # 测试单用例运行主程序

- quant_compressor_operator_single.py          # CPU侧算子逻辑实现获取golden与npu算子直调

- test_quant_compressor_paramset.py            # 单用例入参配置

批量测试：

- test_quant_compressor_batch.py               # 用例批量测试主程序并生成excel文件保存结果（支持隔离模式）

- ./batch/quant_compressor_pt_loadprocess.py   # 读取pt文件并调用算子获取npu输出

- ./batch/quant_compressor_pt_save.py          # 读取excel表格批量生成用例pt文件

- ./batch/replace_path.py                      # test_quant_compressor_batch.py占位符替换

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_quant_compressor_paramset.py的ENABLED_PARAMS参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### 用例的批量生成与测试

##### 方式A：test_run.sh 批量执行

1、excel路径下存放用例excel表格

2、test_run.sh中设置读取的用例excel表格路径（PATH1）和pt文件存放路径（PATH2）

3、执行指令：

``` bash
bash test_run.sh batch
```

##### 方式B：手工分步执行

1、生成pt文件：

``` bash
python3 batch/quant_compressor_pt_save.py "excel/*.xlsx" pt_path
```

2、替换测试脚本路径：

``` bash
python3 batch/replace_path.py test_quant_compressor_batch.py pt_path
```

3、执行测试：

``` bash
python3 -m pytest -rA -s test_quant_compressor_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

4、恢复测试脚本：

``` bash
cp test_quant_compressor_batch.py.bak test_quant_compressor_batch.py
```

##### 方式C：批量隔离执行（推荐用于性能采集）

对每条用例单独拉起一个pytest进程，实现进程间完全隔离，避免单条用例崩溃影响其他用例。
用例获取逻辑：优先从 `./excel` 目录读取Excel文件的 `Testcase_Name` 列映射为 `.pt` 文件路径；若无Excel文件或缺少 `Testcase_Name` 列，则自动回退为扫描用例目录下所有 `.pt` 文件。

``` bash
bash batch_isolated_run.sh ./pt_path 0         # 不采集性能
bash batch_isolated_run.sh ./pt_path 1         # 采集性能（挂载msprof）
```

## Excel 用例表格式

`excel/test_cases.xlsx` 的 Sheet1 需包含以下列：

| 列名 | 类型 | 示例 | 说明 |
|---|---|---|---|
| Testcase_Name | str | `Prefill0` | 用例名，也作为生成的.pt文件名 |
| batch_size | int | `1` | |
| hidden_size | int | `4096` | |
| Seq_len | int | `8192` | |
| head_dim | int | `128` / `512` | |
| block_size | int | `128` | |
| cmp_ratio | int | `4` / `128` | |
| coff | int | `1` / `2` | |
| start_p | int | `0` / `8192` | |
| cache_mode | int | `1` / `2` | 1:连续buffer, 2:循环buffer |
| layout_x | str | `TH` / `BSH` | |
| data_type | str | `BF16` / `FP16` | |
| cu_seqlens | None/str | `None` 或 `0,8192` | TH布局必填，逗号分隔 |
| seqused | None/str | `None` 或 `8192` | 逗号分隔 |
| start_pos | None/str | `None` 或 `0` | 逗号分隔 |
| x_datarange | str | `-10,10` | 逗号分隔 |
| wkv_datarange | str | `-10,10` | 逗号分隔 |
| wgate_datarange | str | `-10,10` | 逗号分隔 |
| ape_datarange | str | `-10,10` | 逗号分隔 |
| kv_state_datarange | str | `-10,10` | 逗号分隔 |
| score_state_datarange | str | `-10,10` | 逗号分隔 |
| x_descale_datarange | str | `0.001,1.0` | 逗号分隔 |
| wkv_descale_datarange | str | `0.001,1.0` | 逗号分隔 |
| wgate_descale_datarange | str | `0.001,1.0` | 逗号分隔 |
| quant_mode | int | `1` | 1: A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL |

**注意事项**：

- `layout_x=TH` 时，`cu_seqlens` 为必填项（长度=batch_size+1的列表）

## 输出文件

| 文件 | 说明 |
|---|---|
| `result.xlsx` | 测试结果（精度、参数等） |
| `result_perf.xlsx` | 测试结果 + 性能数据（仅msprof模式） |
| `batch_summary.log` | 批量隔离执行详细日志 |
| `batch_fail_list.log` | 失败用例清单 |
| `PROF_*/` | msprof性能原始数据目录 |
