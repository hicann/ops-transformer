# Mega TTK 用例

正式 CSV 的 `api_name` 统一为 `mega_recurrent_kda`。直接运行 TTK 前，使用 `--for-ttk` 导出临时运行表，根据预期结果选择插件入口。默认导出保留算子名。

[English](README_en.md)

TTK 使用各架构的 CSV 清单。下表列出默认 [arch35](arch35/) 的五份清单，各清单的用例 ID 互不重叠。

| 清单 | 条数 |
|---|---:|
| ttk_cases_interception.csv | 126 |
| ttk_cases_redline.csv | 7 |
| ttk_cases_generalization.csv | 960 |
| ttk_cases_stc.csv | 46 |
| ttk_cases_performance.csv | 102 |

设计配置保留 S ≤ 8 且排除 28 AIC。拦截仍覆盖非法 AIC 配额。
泛化清单沿用原「其他」清单的范围。性能清单固定 BSH、32 AIC，覆盖 S4/S8、TP4/8/16、
模型 batch 1～16 和 32，算子实际 B 为 TP × 模型 batch。
性能采集需单独配置 profiling，清单校验与精度执行均不代表性能结果。

`testcase_name` 是唯一用例 ID。`tensor_view_shapes`、`tensor_dtypes`、
`input_data_ranges`、`tensor_storage_shapes` 和 `tensor_view_strides` 使用 TTK 标准列。
张量声明顺序与 `kda_ttk_ops.verify_csv` 的签名一致。BSH 的 `cu_seqlens` 槽填 `None`。
各张量的 `tensor_formats` 明确标注 ND 或 FRACTAL_NZ，TTK 在上板前转换权重格式。
`input_data_ranges` 声明真实数值输入的允许上下界，输入钩子读取 `attributes.case_config.input_specs` 中的分布和参数生成输入，并校验范围。
`attributes.case_config` 保存该行的 seed、标量、PA 映射、核数配额及设计信息，
其中 `aliases` 仅用于通过历史编号选择用例。用例形状变化时应同步更新张量声明和配置。
数值入口使用 TTK 按表分配的真实张量，输入钩子按配置填入原 seed 的数据并验证视图。
拦截和图捕获使用一个标准控制张量，专用入口按同一行配置构造异常或捕获场景。
PASS_NUMERIC 表示预期通过精度比较，PASS_CAPTURE 验证捕获及重放契约，
REJECT_ATOMIC 要求异常匹配且状态和记录保持不变。执行结果以日志为准。

## 接口覆盖

- BSH/TH 输出形状、可选 cu_seqlens（int32/int64）、正长度与分块边界，S4/S8 优化与变长分支。
- checkpoint、Conv 和三个 replay 池共用 PA 块数，各自支持独立的第 0 轴步长，并统一使用 `state_indices[b]` 寻址。
- replay shape 为 `[block_num,N,S,D]`。覆盖槽 0、末槽、稀疏映射、Conv 历史接受计数、未选块、PA 间隙和记录尾部。
- Mega 的 checkpoint 只读，Conv 与记录原位返回。
- FP32 U、FP16 K、FP32 decay；PA 块数量与记录容量边界；三种 Conv 权重形状。
- 张量 dtype/shape/连续性、ND/NZ、控制值、标量、别名、Parameter、可写梯度与图捕获契约。
- 旧 `max_batch` 仅出现在移除参数的拒绝用例中。

有限设计集覆盖代表值和边界，不穷举所有正长度、张量数值和地址排列。
无法合理分配的极大 token 总数等资源上限不作为已覆盖项。配置校验不等同于设备编译、STC 或精度通过。

### arch92 清单

`KDA_ARCH=arch92` 选择 [arch92](arch92/) 的 Mega-only 清单。共有 7 条红线、126 条拦截、65 条 STC、1,163 条泛化和 102 条性能形状。arch92 使用 AIC:AIV 1:1，支持 8、16、24、32、64 AIC，等长 S4/S8 也走通用路径。

## 执行

先构建并安装当前算子的 custom 包，再构建并安装 `torch_extension` wheel。
测试入口只加载已安装的扩展，通过 `MKDA_TORCH_PACKAGE` 指定实际包名（例如单算子 vendor 为 custom 时使用 `cann_ops_transformer_custom`），不回退到工作区 DSL。
运行前设置 `CANNBOTDSL_NATIVE_BINARY_MODE=require`，并加载本次 custom 包的 `bin/set_env.bash`；Native 未命中时直接报错。

五份 arch35 TTK 清单合计 1235 条。可用核数不足时单独记录未执行。

TTK 使用框架标准 e2e 入口、`kda_ttk_ops.py` 插件和所选 CSV：

```bash
export PYTHONPATH="$PWD/experimental/attention/mega_recurrent_kda/tests:${PYTHONPATH:-}"
export MKDA_RUN_DIR=/tmp/mega_kda_ttk_run
export MKDA_TORCH_PACKAGE=cann_ops_transformer_custom
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --suite stc --for-ttk --output /tmp/mega_kda_stc_runtime.csv
python -B -m ttk e2e -i /tmp/mega_kda_stc_runtime.csv --plugin experimental/attention/mega_recurrent_kda/tests/kda_ttk_ops.py --validate
```

数值精度只保留三方：安装包的算子输出、`assets/golden.py` 的 CPU FP32 golden、`assets/npu_peer.py` 的 NPU `torch_npu` 小算子参考。
第三方服务必须配置 NPU 后端和可用设备；参考结果显式检查 NPU 设备，不允许 CPU 替代。
比较沿用 `cross_check/L2` 原有阈值，覆盖五个输出；FP32 累加、Fa/Fb 的 BF16 舍入和 replay 的原有 dtype 均保留。
算子不执行额外的补偿点积重算或 Conv/SiLU 残差修正。精度运行使用 `--seed 3 --warmup=false --task-prof=false`，不采集性能。
UT 入口保留在 `tests/ut/test_mega_recurrent_kda_ut.py`，按 CSV 的 `pytest_smoke` 标签选择全部原有用例，并复用 CPU FP32 golden。

```bash
python -B -m pytest -q experimental/attention/mega_recurrent_kda/tests/ut/test_mega_recurrent_kda_ut.py
```

执行子集时从清单导出，不手写新参数或动态扩展：

```bash
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --suite interception --output /tmp/mega_kda_interception.csv
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --ids I-015 F-012 --output /tmp/mega_kda_selected.csv
```

`--ids` 能解析旧编号，导出其对应唯一新 ID。测试按 `KDA_ARCH` 选择实现，默认使用 `arch35`。
日志、结果、输入清单、dump 和导出子集均写到仓库外，不提交验证证据。

## CSV 输入范围与存储

CSV 同时声明数值范围和分布参数，输入辅助代码读取这些配置。`clipped_normal` 表示先按 `mean/std` 生成正态样本，再限制到 `low/high`；它不是均匀分布，也不是重新抽样的截断正态。

| 输入 | 标准差 | 允许范围 |
|---|---:|---|
| hidden_states、投影权重、Conv 权重和 Conv 状态 | 0.02 | [-0.25, 0.25] |
| checkpoint_state | 0.002 | [-0.03125, 0.03125] |
| output_norm_weight | 0.2 | [-2, 2] |
| A_log、dt_bias | 0.1 | [-1, 1] |

零输入用例使用各自声明的常量范围；replay 三池按 `records_fill` 填充，索引和接受长度由该行显式配置确定。TH 的 cu_seqlens 由 lengths 的前缀和构造。控制张量用例的 `(0,0)` 仅描述控制张量，实际待测数据的范围仍完整保存在该行 `input_specs` 中，非法输入在基准构造后按 mutation 生成。

例如 PERF-S4-TP4-MB01 的三池逻辑形状均为 `[6,24,4,128]`，storage 分别为 `[12,24,4,128]`、`[18,24,4,128]` 和 `[24,24,4,128]`。第0轴步长分别跨2、3、4个连续槽位。CPU 和 NPU 都分配声明的完整 backing，再建立非连续逻辑视图；未选块、块间空隙及最后一个逻辑块之后的 backing 尾部均纳入状态保护。

CPU FP32 和 torch_npu 参考按 `KDA_ARCH` 选择独立的计算步骤，不读取算子输出或中间值。Fa 使用完整 K 归约，不使用四段部分和合并或精度补偿。

- arch35 保留原有对齐路径：QKV/Fa/Fb/Beta 的输出 tile、padding、BF16 转换及显式 Conv/激活步骤与其调度对应。这些 NPU 投影 tile 使用1 AIC/2 AIV的完整 K mm，其他小算子使用该行核配额。
- arch92 对齐实际入口统一使用的变长阶段：所有投影使用128×128输出 tile，全局 M 维补齐到128，Beta 的输出列也按128补齐；各 tile 完整累加 K 维后才执行该阶段的存储类型转换。NPU 参考每个 tile 使用1 AIC/1 AIV。
- arch92 的 Vector 步骤对齐两组 Conv 乘加及显式 SiLU/门控、两个64元素半段的归约、归约前的 Q 缩放、`decay*K` 后的状态点积与状态更新乘加，以及输出 RMSNorm 的计算顺序。输出门控和输出投影同样按全局 M tile 计算，不按每条序列重新起一个矩阵乘。

上述串行参考只用于精度比较，不作为并行性能基线。CPU 与 NPU 仍各自执行浮点归约和超越函数；步骤对齐不保证逐位一致，不能替代对应架构上的实际精度验证。

数值行的上下界以 `input_data_ranges` 为唯一来源，`input_specs` 保存分布参数。导出运行表时将范围注入 case_config，确保输入钩子实际消费同一份声明；控制张量行的 payload 范围单独保存在 input_specs 中。
