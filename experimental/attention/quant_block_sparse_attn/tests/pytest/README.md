# QuantBlockSparseAttn pytest 使用手册

## 用途

本目录为 `QuantBlockSparseAttn`（QBSA）提供 pytest 测试：先在 CPU 上构造输入并生成 Golden，再在 NPU eager 或 aclgraph 路径执行算子并比较输出。测试同时覆盖单参数集调试、CSV 驱动的批量用例，以及图模式批量对比。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `test_run.sh` | 统一入口，提供 `single`、`batch`、`graph` 和 `help` 四种模式。 |
| `test_quant_block_sparse_attn_single.py` | single pytest 入口：遍历 `quant_block_sparse_attn_paramset.py` 中启用的参数集，生成 CPU Golden 并进行 eager NPU 对比。 |
| `test_quant_block_sparse_attn_batch.py` | batch pytest 入口：读取已生成的 PT 用例，执行 eager NPU 对比。 |
| `test_quant_block_sparse_attn_batch_graph.py` | graph pytest 入口：读取 PT 用例，通过 aclgraph 的 `reduce-overhead` 路径进行批量对比。 |
| `quant_block_sparse_attn_golden.py` | 生成 FP8 输入、逻辑稀疏信息、组合 PA KV Cache 和 CPU Golden；也负责保存 PT 用例。 |
| `quant_block_sparse_attn_paramset.py` | single 模式的参数集，使用 `ENABLED_PARAMS` 选择要执行的用例。 |
| `batch/quant_block_sparse_attn_paramset_batch.py` | batch CSV 加载器：解析用例字段、筛选 `enable` 用例，并生成批量参数集。 |
| `custom_ops.py` | 自定义算子加载器：检查算子注册状态，并按约定位置尝试加载扩展。 |
| `result_compare_method.py` | 比较 CPU Golden 与 NPU 输出，并给出通过状态和满足比例。 |

`batch/test_quant_block_sparse_attn_pt_save.py` 是 batch 和 graph 的 PT 生成入口；`batch/quant_block_sparse_attn_process.py` 封装 eager 与图模式的 NPU 调用。

## 前置条件

- 从本目录运行命令：`experimental/attention/quant_block_sparse_attn/tests/pytest`。
- Python 环境应具备测试所需的 `pytest`、`torch`、`torch_npu` 和 `torchair`；CPU Golden 需要 PyTorch 提供 `torch.float8_e4m3fn`。
- 进行 NPU 精度对比时，需要可用的 NPU 设备，以及已注册的 `torch.ops.custom.npu_quant_block_sparse_attn` 和 `torch.ops.custom.npu_quant_block_sparse_attn_metadata`。
- 若仓内扩展尚未构建，可在仓库根目录执行：

```bash
cd experimental/attention/quant_block_sparse_attn/torch_ops_extension
bash build_and_install.sh
```

构建完成后，回到 pytest 目录运行测试。

## 测试范围

当前测试围绕 QBSA 的全量化和稀疏数据流，覆盖以下内容：

- FP8 格式的 Q、K、V；
- Q、K 反量化，以及 P 的静态量化和 V 缩放；
- 由 `sparse_indices` 描述的逻辑稀疏 KV block 与 `block_table` 的物理映射；
- K、V、K 反量化尺度共用存储的组合 PA KV Cache；
- 稀疏选择为空时的输出；
- `mask_mode=0`（无掩码）和 `mask_mode=3`（因果下三角掩码）；
- TND、NTD 查询布局，以及可选的 `softmax_lse` 输出。

## Golden 语义

`quant_block_sparse_attn_golden.py` 在 CPU 上生成参考结果。Q、K、V 均以 FP8 数据构造；Q/K 反量化尺度在 `QK^T` 之后以行、列尺度作用于分数。参考实现对稳定式 Softmax 的指数权重，对指数权重量化为 FP8，用于 BMM2 累加；最终以同样含该因子的累积和归一化输出，因此该尺度在分子、分母中相消。V 的反量化尺度在累加过程中参与缩放。

`sparse_indices` 保存逻辑 KV block 编号，`block_table` 将逻辑 block 映射到 PA KV Cache 中的物理 block。K、V 与 K 反量化尺度被打包在同一段 `uint8` 存储内，并分别以视图访问。若某个查询行没有有效的稀疏 KV 位置，Golden 保持 `attention_out` 为零，并将该行的 LSE 设为 `EMPTY_LSE`。当 `return_softmax_lse=True` 时，比较同时包含 LSE。

## 扩展加载

导入 `custom_ops.py` 时，加载顺序如下：

1. 优先检查两个 QBSA 自定义算子是否已经注册在 `torch.ops.custom` 中。
2. 若尚未注册，检查环境变量 `QBSA_CUSTOM_OPS_PATH` 指向的位置；可用冒号分隔多个路径。
3. 若仍未注册，检查仓内 `experimental/attention/quant_block_sparse_attn/torch_ops_extension`。

加载器会尝试加载匹配的共享库或 Python 扩展，并在适用时加载转换器。算子未注册，或 `torch_npu` / NPU 设备不可用时，NPU 精度对比会被 pytest 跳过；CPU Golden 生成不依赖 NPU 执行。

## 运行方式

进入 pytest 目录后，使用以下四个命令：

```bash
bash test_run.sh single
bash test_run.sh batch
bash test_run.sh graph
bash test_run.sh help
```

`single` 使用 `quant_block_sparse_attn_paramset.py` 的 `ENABLED_PARAMS`，对每个参数组合生成 CPU Golden，并立即执行 eager NPU 对比。该模式不保存 PT 用例。

`batch` 先读取 `quant_block_sparse_attn_paramset_batch.py` 中已启用的用例，生成 `bsa_testcase/*.pt`，随后读取这些 PT 执行 eager NPU 对比。已存在的同名 PT 用例会在生成阶段跳过。

`graph` 同样先生成 PT，再读取 `bsa_testcase/` 内的 PT，用 aclgraph 的 `reduce-overhead` 编译路径批量对比。图模式要求输入使用组合 PA KV Cache；不满足该输入形式的用例会被跳过。

## 用例维护

single 用例在 `quant_block_sparse_attn_paramset.py` 中维护，并通过 `ENABLED_PARAMS` 控制执行范围。batch 用例在三个 CSV 中维护；将一行的 `enable` 设为 `TRUE` 才会被加载，设为 `FALSE` 则不会被 batch 加载或生成。graph 会枚举 `bsa_testcase/*.pt` 中所有已有文件，因此已禁用或已删除 CSV 行的遗留 PT 仍会执行，直到手动删除该 PT。

修改用例时，应保持关键参数关系有效：`cu_seqlens_q_value` 的长度为 `B + 1` 且首项为 0，`seqused_kv_value` 的长度为 `B`；`N1` 应能被 `N2` 整除；当前参数校验要求 Q/KV 稀疏 block 大小均为 128、`quant_mode=1`、`mask_mode` 为 0 或 3，并要求布局和 block 映射与输入形状一致。`block_num`、`max_block_per_batch`、逻辑稀疏 block 与 `block_table` 还需要能容纳实际 KV 长度和物理 block 映射。

新增或修改 batch CSV 后运行 `batch` 或 `graph` 生成 PT。若希望使用同名更新后的数据重新生成 PT，请先处理对应的旧 PT 文件，再运行生成入口。

## 输出文件

- `bsa_testcase/`：batch 和 graph 的 PT 用例目录。每个 PT 包含输入、参数、CPU Golden 及相关元数据。
- `result.xlsx`：single、batch 和 graph 会追加用例参数、比较结果及满足比例。若 Excel 写入所需的可选依赖不可用，结果会写入同名的 CSV 文件。

这些文件为测试运行产物，不应作为参数集或 Golden 源码的替代品。

## 常见问题

### 扩展未注册

确认已构建并安装扩展，或将扩展位置设置到 `QBSA_CUSTOM_OPS_PATH`，然后重新运行测试。若两个 `torch.ops.custom` 算子仍未注册，pytest 会跳过 NPU 精度对比并输出相关原因。

### NPU 不可用

确认 `torch_npu` 可导入且 NPU 设备可用。设备不可用时，执行路径会跳过 NPU 精度对比；这不表示已完成 NPU 结果验证。

### PT 文件缺失

先执行 `bash test_run.sh batch` 或 `bash test_run.sh graph` 触发 PT 生成。batch 对比仅选取 CSV 中启用且已存在的 PT；graph 需要 `bsa_testcase/` 中存在 PT 文件。

### 参数校验跳过

在 single 入口中，`check_valid_param.py` 抛出的 `ValueError` 会被转换为 pytest 跳过。通过该入口校验后，Golden 或数据生成阶段抛出的错误不会被这一跳过逻辑捕获：例如部分序列长度或 block 映射不一致会以 `ValueError` 使本次运行失败。遇到此类错误时，检查布局、序列长度、head 数关系、128 block 大小、量化模式、掩码模式和 block 映射字段。

### graph 输入要求

graph 模式只能处理组合 PA KV Cache 存储形式。请通过本目录的 Golden/PT 生成流程创建输入，确保 PT 中包含 `kv_cache_storage` 和 `kv_cache_meta`。
