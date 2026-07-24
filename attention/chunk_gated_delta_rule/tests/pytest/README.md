# Chunk_gated_delta_rule算子测试框架

## 功能说明

基于pytest测试框架，实现Chunk_gated_delta_rule算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制

- 支持batch_size大于0。
- 支持seqlen序列长度。
- 支持NK、NV head数，NV需要为NK倍数。
- 支持DK、DV不超过128。
- 支持data_type为BF16。

### 环境配置

#### 前置要求

1. TorchNPU安装包下载路径（需及时更换为最新版本）：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. 完成环境安装和环境变量配置，具体操作请参考：[ops-transformer](../../../../README.md)

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run.sh                               # 执行脚本
- chunk_gated_delta_rule_golden.py          # cpu侧算子golden实现
- chunk_gated_delta_rule_main.py            # cpu golden与npu结果精度对比及主调测逻辑
- pytest.ini                                # 创建ci单算子和graph图模式的测试标记

单用例测试:

- test_chunk_gated_delta_rule_single.py     # 测试单用例运行主程序
- chunk_gated_delta_rule_operator_single.py # CPU侧算子逻辑实现获取golden与npu算子直调
- test_chunk_gated_delta_rule_paramset.py   # 单用例入参配置
- test_chunk_gated_delta_rule_paramset_rdv.py   # RDV测试入参配置

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_chunk_gated_delta_rule_paramset.py的参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### RDV测试

1、手动配置test_chunk_gated_delta_rule_paramset_rdv.py的参数

2、执行指令：

``` bash
bash test_run.sh rdv
```
