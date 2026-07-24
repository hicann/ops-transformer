# FlashAttentionScoreGrad算子测试框架

## 文件结构

pytest/

- test_case.py              # 测试用例集
- test_main.py              # 执行主程序
- common.py                 # 公共定义文件
- cpu_impl.py               # cpu实现
- npu_impl.py               # npu实现

## 功能说明

基于pytest测试框架，实现FA算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 环境配置

### 前置要求

1. 确认TorchNPU为最新版本
2. source CANN包环境变量

### Custom包调用

支持custom包调用

## 使用方法

在pytest文件夹路径下执行：

pytest -s
