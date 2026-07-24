# Recurrent_gated_delta_rule算子测试框架

## 功能说明

基于pytest测试框架，实现Recurrent_gated_delta_rule算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制

- 支持batch_size大于0。
- 支持mtp为1~8。
- 支持NK、Nv小于等于256，Nv大于等于Nk且Nv需整除Nk。
- 支持Dk、Dv小于等于512。
- 支持actual_seq_lengths输入，长度为batch_size。
    - actual_seq_lengths中数值需大于等于0且小于等于mtp。
    - 不指定输入时，默认传入长度为batch_size，数值为mtp的数组。
    - T等于actual_seq_lengths所有元素之和。
- 支持ssm_state_indices输入，长度为T。
    - ssm_state_indices中数值需小于block_num。
    - 不指定输入时默认长度为[0,1,...,T-1]。
- 支持block_num手动传入，需大于等于T。
- 支持data_type为BF16。
- 支持query_datarange左区间最小为-1，右区间最大为1。
- 支持key_datarange左区间最小为-1，右区间最大为1。
- 支持gamma_datarange右区间最大为0。
- 支持gamma_k_datarange右区间最大为0。
- 支持beta_datarange左区间最小为0，右区间最大为1。

### 环境配置

#### 前置要求

1. TorchNPU安装包下载路径（需及时更换为最新版本）：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. 完成环境安装和环境变量配置，具体操作请参考：[ops-transformer](../../../../README.md)

#### custom包调用

支持custom包调用

## 文件结构

#### pytest文件结构说明

- test_run.sh                               # 执行脚本
- recurrent_gated_delta_rule_golden.py      # cpu侧算子golden实现以及cpu golden与npu结果精度对比
- pytest.ini                                # 创建ci单算子和graph图模式的测试标记

单用例测试:

- test_recurrent_gated_delta_rule_single.py                 # 测试单用例运行主程序
- recurrent_gated_delta_rule_operator_single.py             # CPU侧算子逻辑实现获取golden与npu算子直调
- test_recurrent_gated_delta_rule_paramset.py               # 单用例入参配置
- test_recurrent_gated_delta_rule_paramset_rdv.py           # RDV测试入参配置

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

1、手动配置test_recurrent_gated_delta_rule_paramset.py的参数

2、执行指令：

``` bash
bash test_run.sh single
```

#### RDV测试

1、手动配置test_recurrent_gated_delta_rule_paramset_rdv.py的参数

2、执行指令：

``` bash
bash test_run.sh rdv
```
