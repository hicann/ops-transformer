# quant_lightning_indexer算子测试框架

## 功能说明

基于pytest测试框架，实现quant_lightning_indexer算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制

- **数据格式**:
  - **query_layout**：BSND、TND
  - **key_layout**: PA_BSND

### 环境配置

#### 前置要求

1. 完成环境安装和环境变量配置，具体操作请参考：[环境部署](../../../../docs/zh/install/quick_install.md)
2. TorchNPU安装包下载路径（需及时更换为最新版本）：[Attention融合算子Experimental使用说明](../../../../experimental/attention/Attention融合算子Experimental使用说明.md)

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run_sh                                  # 执行脚本
- quant_lightning_indexer_golden.py            # cpu侧算子golden实现
- result_compare_method.py                     # cpu golden与npu输出精度对比
- pytest.ini                                   # 创建测试标记

单用例测试：

- test_quant_lightning_indexer_single.py       # pytest测试单用例运行主程序 
- test_quant_lightning_indexer_paramset.py     # 单用例入参配置

批量测试：

- test_quant_lightning_indexer_batch.py        # 用例批量测试主程序并生成excel文件保存结果
- ./batch/quant_lightning_indexer_pt_loadprocess.py    # 读取pt文件并调用算子获取npu输出
- ./batch/quant_lightning_indexer_pt_save.py           # 读取excel表格批量生成用例pt文件
- ./batch/replace_path.py                              # test_quant_lightning_indexer_batch.py占位符替换
 
## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_quant_lightning_indexer_paramset.py的参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### 用例的批量生成与测试

1、excel路径下存放用例excel表格

##### 批跑用例表格

批跑测试通过Excel文件配置测试用例参数，由`tests/pytest/test_run.sh batch`自动读取`tests/pytest/excel/`目录下的Excel文件并执行批量测试。用户需自行在该目录下创建Excel文件，格式如下表所示（共28列）：

| 列名 | 说明 |
|:-----|:-----|
| Testcase_Name | 用例名称 |
| batch_size | Batch大小 |
| q_seq | Query的Sequence Length |
| k_seq | Key的Sequence Length |
| q_t_size | Query的Total Tokens（TND布局使用） |
| k_t_size | Key的Total Tokens（TND布局使用） |
| q_head_num | Query的Head Num |
| k_head_num | Key的Head Num（仅支持1） |
| head_dim | Head Dim（仅支持128） |
| block_size | PageAttention场景下每个block的token数（需为16的整数倍，最大1024） |
| block_num | PageAttention场景下的block总数 |
| qk_dtype | Query和Key的数据类型 |
| weight_dtype | Weights的数据类型 |
| dequant_dtype | Dequant Scale的数据类型 |
| actual_seq_dtype | Actual Seq Lengths的数据类型 |
| act_seq_q | Query的有效token数（逗号分隔；TND时为前缀和格式） |
| act_seq_k | Key的有效token数（逗号分隔；TND时为前缀和格式） |
| query_quant_mode | Query量化模式（当前仅支持0） |
| key_quant_mode | Key量化模式（当前仅支持0） |
| layout_query | Query数据排布格式 |
| layout_key | Key数据排布格式 |
| sparse_count | TopK保留的索引数量 |
| sparse_mode | Sparse模式（0: defaultMask, 3: rightDownCausal） |
| query_datarange | Query数据范围（逗号分隔的最小值,最大值） |
| key_datarange | Key数据范围（逗号分隔的最小值,最大值） |
| weights_datarange | Weights数据范围（逗号分隔的最小值,最大值） |
| q_scale_datarange | Query Dequant Scale数据范围（逗号分隔的最小值,最大值） |
| k_scale_datarange | Key Dequant Scale数据范围（逗号分隔的最小值,最大值） |

批跑测试用例表格格式示例如下：

| Testcase_Name | batch_size | q_seq | k_seq | q_t_size | k_t_size | q_head_num | k_head_num | head_dim | block_size | block_num | qk_dtype | weight_dtype | dequant_dtype | actual_seq_dtype | act_seq_q | act_seq_k | query_quant_mode | key_quant_mode | layout_query | layout_key | sparse_count | sparse_mode | query_datarange | key_datarange | weights_datarange | q_scale_datarange | k_scale_datarange |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|:---|:---|:---:|:---:|:---|:---|:---:|:---:|:---|:---|:---|:---|:---|
| test001 | 4 | 2 | 115 | 4 | 2 | 64 | 1 | 128 | 128 | 10 | INT8 | FP16 | FP16 | INT32 | 1,2,1,1 | 0,352,14,461 | 0 | 0 | BSND | PA_BSND | 512 | 3 | 2,10 | 0,1 | -255,255 | -171,171 | 0,65504 |

2、test_run.sh中设置读取的用例excel表格路径和pt文件存放路径

3、执行指令：

``` bash
bash test_run.sh batch
```
