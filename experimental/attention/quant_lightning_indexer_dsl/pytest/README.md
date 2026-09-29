# Quant Lightning Indexer pytest

## 接口与环境

测试通过本仓库的 Torch 接口执行 metadata→QLI 完整调用。公共调用函数生成 metadata 后显式传入主算子，CPU golden 和比较器保持独立。

主算子和 metadata 支持 TND / PA_BBND。metadata 输出为 int32 `[1024]`；Sparse 和 Candidate 输出沿用统一比较规则。

运行前激活匹配的 Torch、NPU、DSL 环境并加载 CANN。按当前环境配置 `CANNBOTDSL_TEST_OPS_DIR` 和 `OPS_TRANSFORMER_TORCH_EXTENSION_DIR`，分别指定待验证算子源码和 Torch 基础接口。测试会记录实际加载的实现及 SHA-256，不依赖文档中的开发机路径。

用 `QLIV2_DEVICE_ID` 指定设备，使用 `QLIV2_CASE_NAMES` 或 `QLIV2_CASE_INDEXES` 筛选用例。图测试捕获 metadata→主算子链路。

## 单用例与批跑

在当前测试目录运行：

```bash
bash test_run.sh single
python -m pytest -v -s test_quant_lightning_indexer_v2_batch.py
```

通过 `QLIV2_SINGLE_SAVE_PT_DIR` 指定单用例生成的 PT 输出位置，通过 `QLIV2_TESTCASE_DIR` 指定批跑输入。批跑使用完整 CPU/NPU 输出比较 Sparse 和 Candidate 的索引与数值。

## 外部用例数据

算子目录不附带用例表。外部表格需包含 `Testcase_Name` 及 `PARAM_NAMES` 定义的参数列；调用数据生成器的 `load_excel_test_cases()` 读取表格，`save_test_case()` 为每行生成独立 PT，再交给批跑测试。生成位置由调用者指定。
