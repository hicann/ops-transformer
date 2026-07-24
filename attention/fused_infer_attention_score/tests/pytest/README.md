# GQA算子测试框架

## 文件结构

```text
pytest/
├── test.py                          # pytest泛化测试用例运行主程序
├── testcases.py                     # 泛化测试用例入参配置
├── check_valid_param.py             # 入参检查及精度对比
├── gqa_no_quant_bnsd_bsnd.py        # CPU侧算子逻辑实现获取Golden，npu算子直调
├── gqa_no_quant_bnsd_bsnd_ge.py     # 图模式编译调用，与单算子调用结果精度对比
├── pytest.ini                       # 创建ci单算子和graph图模式的测试标记
```

## 概述

pytest框架作为一个轻量化精度对比的测试框架，提供了简单化的验证方式及流程：

- 测试用例只需要执行一个test.py脚本，简化执行流程，并能够对测试用例进行泛化
- 支持自定义生成的custom包适配，可通过直接调用custom包验证，无需交付件
- 脚本内各功能分离，易于特性验证及自定义修改
- Golden生成逻辑完善，为先构建输入，后进行运算
- 环境配置较简单，无依赖冲突，仅需python环境及torch&TorchNPU安装

## 功能说明

基于pytest测试框架，实现GQA算子的功能验证：

- **CPU侧**：复现算子功能用以生成Golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据，通过torchair组件入图
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

### 主要特性

- **实现逻辑**：在cpu测复现算子逻辑生成Golden脚本，与调用算子的结果进行精度对比，输出精度准确率及失败数据的下标和数据信息
- **用例泛化**：通过pytest进行测试用例的泛化，可以通过输入不同的参数在运行时自动交叉组合，无需手动单个配置，可同时验证大量用例
- **场景支持**：支持GQA非量化PA场景，Query输入格式支持BNSD/BSND，PA场景下的KV cache支持BBH和BNBD格式
- **模式支持**：支持单算子模式及图模式调用，实现tiling下沉功能
- **参数配置支持**：支持在配置参数脚本testcases.py内修改配置，自定义开发
- **随机性测试**：基于测试随机性目标，QKV矩阵通过torch.randn随机生成，输入格式由参数定义，KV的actual sequence length也为随机生成的列表，长度为[batch size, 1~Skv]，无需手动配置每一个用例

### 参数限制

- **数据格式**：FP16 / BF16
- **in_layout**：BSND /BNSD
- **kvcache_layout**: BNBD / BBH
  - BSND格式仅支持BBH

## 环境配置

### 前置要求

1. TorchNPU安装包下载路径（需及时更换为最新版本）：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. 完成环境安装和环境变量配置，具体操作请参考：[ops-transformer](../../../../README.md)

### Custom包调用

支持custom包调用

## 运行方式

在pytest文件夹路径下执行：

单算子直调+图模式

```bash
python3 -m pytest -rA -s test.py
```

单算子直调

```bash
python3 -m pytest -rA -s test.py -v -m ci
```

图模式

```bash
python3 -m pytest -rA -s test.py -v -m graph
```
