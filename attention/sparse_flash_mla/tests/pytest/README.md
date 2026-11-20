# sparse_flash_mla pytest 测试

先激活 CANN 及所需的 custom 包，并准备匹配的 PyTorch、`torch_npu` 和 `torchair`。
以下命令均在本目录执行；完整选项见 `bash test_run.sh --help`。

## 常用命令

### 单例测试

单例参数在 `sparse_flash_mla_paramset.py` 中配置：

```bash
bash test_run.sh single
```

### 批量生成 PT

从 Excel 读取用例，生成 CPU golden 和 PT 文件：

```bash
bash test_run.sh save \
  -E ./excel/example.xlsx -S CSA -P ./data
```

`-E`、`-S`、`-P` 分别指定 Excel 文件、sheet 和 PT 输出目录；不指定时使用脚本默认值。

### 批量执行

`load` 执行 eager 测试，`load_graph` 执行 ACLGraph 测试。两者都从已有 PT 文件读取输入：

```bash
bash test_run.sh load \
  -P ./data -R ./result/smla_eager.xlsx
bash test_run.sh load_graph \
  -P ./data -R ./result/smla_graph.xlsx
```

不指定 `-E` 时执行目录中的所有 PT；增加 `-E` 和 `-S` 可按 Excel 的 `testcase_name` 列筛选，
不会重新生成 PT：

```bash
bash test_run.sh load_graph \
  -P ./data -E ./excel/example.xlsx -S CSA
```

### SuperKernel

在 `single`、`load` 或 `load_graph` 命令末尾增加 `--super-kernel`：

```bash
bash test_run.sh single --super-kernel
bash test_run.sh load_graph -P ./data --super-kernel
```

该选项默认关闭，开启后使用 ACLGraph SuperKernel；metadata 仍在图外生成，
`debug_per_op_max_core_num=1` 表示单算子满核调试，不是限制为一个核。

### Batch 一致性

`mixed_quant_sparse_flash_mla`（MQSMLA）和 `quant_sparse_flash_mla`（QSMLA）的 pytest batch一致性测试都依赖 SMLA 在此目录提供的 [batch_consistency 模块](./batch_consistency/)。
运行 MQSMLA 或 QSMLA 测试时，将对应的算子目录与 `sparse_flash_mla/` 放在同一父目录下，或将此处的整个 `batch_consistency/` 目录复制到对应算子的 `tests/pytest/` 下。

batch一致性策略通过 `--batch-consistency` 控制：

```bash
bash test_run.sh load_graph -P ./data --batch-consistency on
```

可选值为 `auto`（默认）、`on` 和 `off`。`auto` 根据参数集或 PT 中的配置决定是否启用。
