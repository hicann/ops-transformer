# Mega TTK 用例

正式 CSV 的 `api_name` 统一为 `mega_recurrent_kda`。直接运行 TTK 前，使用 `--for-ttk` 导出临时运行表，根据预期结果选择插件入口。默认导出保留算子名。

[English](README_en.md)

pytest 与 TTK 共用各架构的 CSV 清单。下表列出默认 [arch35](arch35/) 的五份清单，各清单的用例 ID 互不重叠。

| 清单 | 条数 |
|---|---:|
| ttk_cases_interception.csv | 126 |
| ttk_cases_redline.csv | 7 |
| ttk_cases_generalization.csv | 960 |
| ttk_cases_stc.csv | 46 |
| ttk_cases_performance.csv | 96 |

设计配置保留 S ≤ 8 且排除 28 AIC。拦截仍覆盖非法 AIC 配额。
泛化清单沿用原「其他」清单的范围。性能清单固定 BSH、32 AIC，覆盖 S4/S8、TP4/8/16、
模型 batch 1～16，算子实际 B 为 TP × 模型 batch。
性能采集需单独配置 profiling，清单校验与精度执行均不代表性能结果。

`testcase_name` 是唯一用例 ID。`tensor_view_shapes`、`tensor_dtypes`、
`input_data_ranges`、`tensor_storage_shapes` 和 `tensor_view_strides` 使用 TTK 标准列。
张量声明顺序与 `kda_ttk_ops.verify_csv` 的签名一致。BSH 的 `cu_seqlens` 槽填 `None`。
各张量的 `tensor_formats` 明确标注 ND 或 FRACTAL_NZ，TTK 在上板前转换权重格式。
`input_data_ranges` 的 `(0, 0)` 是分配初值，真实输入由输入钩子按原规则填充。
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

`KDA_ARCH=arch92` 选择 [arch92](arch92/) 的 Mega-only 清单。共有 7 条红线、126 条拦截、65 条 STC、1,163 条泛化和 96 条性能形状。arch92 使用 AIC:AIV 1:1，支持 8、16、24、32、64 AIC，等长 S4/S8 也走通用路径。

## 执行

从仓库根目录执行：

```bash
python -B -m pytest -q experimental/attention/mega_recurrent_kda/tests/ut --basetemp=/tmp/mega_kda_pytest
```

pytest 从清单中读取标记为 `pytest_smoke` 的 11 条代表用例。
`-m interception`、`-m functional` 和 `MEGA_KDA_SUITE` 可进一步筛选。五份 TTK 清单合计 1235 条。
可用计算核数不足时记录为未执行。

TTK 使用框架标准 e2e 入口、`kda_ttk_ops.py` 插件和所选 CSV：

```bash
export PYTHONPATH="$PWD/experimental/attention/mega_recurrent_kda/tests:${PYTHONPATH:-}"
export MKDA_RUN_DIR=/tmp/mega_kda_ttk_run
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --suite stc --for-ttk --output /tmp/mega_kda_stc_runtime.csv
python -B -m ttk e2e -i /tmp/mega_kda_stc_runtime.csv --plugin experimental/attention/mega_recurrent_kda/tests/kda_ttk_ops.py --validate
```

TTK 精度运行需要 CPU third-party 服务。`assets/golden.py` 提供 FP64 参考和原 CPU 小算子参考，Verify 使用二者与 NPU 输出进行 `cross_check/L2` 三方比较。
全量精度固定 `--seed 3 --warmup=false --task-prof=false`，不启用性能任务。
pytest 保留原 CPU FP32 数学参考与原容差；两种框架的判定标准不能混为一谈。

执行子集时从清单导出，不手写新参数或动态扩展：

```bash
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --suite interception --output /tmp/mega_kda_interception.csv
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --ids I-015 F-012 --output /tmp/mega_kda_selected.csv
```

`--ids` 能解析旧编号，导出其对应唯一新 ID。测试按 `KDA_ARCH` 选择实现，默认使用 `arch35`。
日志、结果、输入清单、dump 和导出子集均写到仓库外，不提交验证证据。
