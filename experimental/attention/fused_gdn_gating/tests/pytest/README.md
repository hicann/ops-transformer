# FusedGdnGating算子测试框架

## 功能说明

基于pytest测试框架，实现FusedGdnGating算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过`torch.ops.custom.npu_fused_gdn_gating`进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围

### 参数限制

- 支持batch_size大于0。
- 支持num_heads为任意正整数。
- 支持dtype为BF16（910B）和FP16（310P/910B）。
- 支持param_dtype（A_log/dt_bias）为FP32、BF16（910B）和FP16。
- 支持beta为正浮点数。
- 支持threshold为正浮点数。

### 测试覆盖

- **核心正确性**：num_heads × batch × dtype 组合全覆盖。
- **非默认参数**：beta=0.5, threshold=1.0, 注入softplus阈值边界值。
- **dtype矩阵**：(a/b dtype) × (A_log/dt_bias dtype) 交叉组合。
- **大batch多行处理**：Bulk DMA对齐与非对齐路径。
- **小batch优化**：batch < rows_per_iter的自适应UB预算。
- **极端大batch**：65536 batch压力测试。
- **310P专用**：FP16 only, 多种beta/threshold组合。

### SOC适配

- BF16测试用例标记 `soc="910b"`，在310P上自动跳过。
- FP16测试用例标记 `soc="all"`，在310P和910B上均可运行。
- 310P专用测试用例（FP16 + 多beta/threshold）标记 `soc="all"`，同时覆盖310P和910B内核路径。

## 环境配置

### 前置要求

1. TorchNPU安装包下载路径（需及时更换为最新版本）：[TorchNPU安装教程](https://gitcode.com/Ascend/pytorch)
2. 完成环境安装和环境变量配置，具体操作请参考：[ops-transformer](../../../../../README.md)

### 编译安装torch_ops_extension

```bash
cd experimental/attention/fused_gdn_gating/torch_ops_extension
bash build_and_install.sh
```

### 编译安装算子包

```bash
bash build.sh --pkg --experimental --ops=fused_gdn_gating --soc=ascend310p -j16
bash output/custom_opp_*.run
```

## 文件结构

- test_run.sh                               # 执行脚本
- pytest.ini                                # 创建ci单算子和graph图模式的测试标记
- conftest.py                               # pytest环境配置（libstdc++预加载、ASCEND_CUSTOM_OPP_PATH）
- custom_ops.py                             # torch_ops_extension .so 加载器
- result_compare_method.py                  # 精度对比工具
- fused_gdn_gating_golden.py                # CPU侧算子golden实现与NPU算子直调
- fused_gdn_gating_paramset.py              # 测试入参配置
- test_fused_gdn_gating_single.py           # 测试单用例运行主程序

## 使用方法

在pytest文件夹路径下执行：

### 运行测试用例

#### 单用例调测

```bash
bash test_run.sh single
```
