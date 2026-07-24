# MlaPrologV2算子测试框架

## 文件结构

pytest/

- test.py                      # pytest泛化测试用例运行主程序
- testcases.py                    # 泛化测试用例入参配置
- check_valid_param.py            # 入参检查及精度对比
- prologv2_no_quant_pa_bsnd.py       # CPU侧算子逻辑实现获取golden，npu算子直调
- pytest.ini                      # 创建ci单算子和graph图模式的测试标记

## 功能说明

基于pytest测试框架，实现MlaPrologV2算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据，通过torchair入图暂不支持
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

### 当前实现范围

✅**已实现**：基础MlaPrologV2非量化PA场景
❌**未实现**：MlaPrologV2部分量化、全量化场景

### 参数限制

- **数据格式**：BF16
- **B**：Batch表示输入样本批量大小，取值范围为1~65536。
- **S**：Seq-Length表示输入样本序列长度，取值范围为1~16。
- **He**：Head-Size表示隐藏层的大小，取值为7168。
- **Hcq**：q低秩矩阵维度，取值为1536。
- **N**：Head-Num表示多头数，取值范围为8、16、32、64、128。
- **Hckv**：kv低秩矩阵维度，取值为512。
- **D**：qk不含位置编码维度，取值为128。
- **Dr**：qk位置编码维度，取值为64。
- **Nkv**：kv的head数，取值为1。
- **BlockNum**：PagedAttention场景下的块数，取值为计算B*Skv/BlockSize的值后再向上取整，其中Skv表示kv的序列长度。
- **BlockSize**：PagedAttention场景下的块大小，取值范围为16、128。
- **T**：BS合轴后的大小，取值范围：1~1048576。
- **cache_mode**: PA_BSND / PA_NZ，其用户不特意指定时可传入默认值“PA_BSND”。

## 环境配置

### 前置要求

1. 确认TorchNPU为最新版本
2. source CANN包

### Custom包调用

支持custom包调用

## 使用方法

在pytest文件夹路径下执行：

### 运行方式：运行泛化测试用例

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
