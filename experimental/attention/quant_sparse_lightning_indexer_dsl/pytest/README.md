# Quant Sparse Lightning Indexer pytest

## 接口与环境

测试通过本仓库的 Torch 接口执行 metadata→QSLI 完整调用。`run_qsli_case()` 先根据当前输入长度和参数生成 metadata，再显式传给主算子；single 和 batch 共用此逻辑。

主算子和 metadata 均使用 `layout_k`，支持 TND / PA_BBND。metadata 输出为 int32 `[1024]`，`return_value=False` 时 values 为空张量。

运行前激活匹配的 Torch、NPU、DSL 环境并加载 CANN。按当前环境配置 `CANNBOTDSL_TEST_OPS_DIR` 和 `OPS_TRANSFORMER_TORCH_EXTENSION_DIR`，分别指定待验证算子源码和 Torch 基础接口。测试记录实际实现及 SHA-256，不依赖文档中的开发机路径。用 `QSLI_DEVICE_ID` 指定设备，图测试捕获 metadata→主算子链路。

## 单用例与批跑

在当前测试目录运行：

```bash
bash test_run.sh single
python -m pytest -v -s test_quant_sparse_lightning_indexer_batch.py
```

使用 `QSLI_CASE_NAMES` 或 `QSLI_CASE_INDEXES` 筛选用例，使用 `QSLI_TESTCASE_DIR` 指定 PT 批跑输入，使用 `QSLI_ARTIFACT_DIR` 指定生成产物的位置。CPU golden、数据生成和比较均为独立测试实现。

## 输入与比较

PA 融合 K 的 shape 为 `(P, block_size/8, 544)`。测试适配器在 CPU 上打包旧格式输入，计算接口不执行额外打包 kernel。TND 使用独立 K/scale，`descale_k` 为关键字参数，并提供对应序列的累计长度。

批跑从 PT 读取输入并执行原始比较器。延迟生成 golden 的 PT 会分块比较所有输出行，分块计算不抽样、不改变比较阈值。

## 外部用例数据

算子目录不附带用例表。表格生成器读取调用者提供的表格，默认 sheet 为 `TestCases`；`--paired-layouts` 生成等价的 PA_BBND 和 TND 输入，`--defer-golden` 将完整 golden 计算延后到批跑比较阶段。PT 输出位置由调用者指定。
