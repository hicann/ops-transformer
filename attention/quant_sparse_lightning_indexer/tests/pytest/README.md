# QuantSparseLightningIndexer 测试（两级TopK 第二级）

本目录是第二级 consumer 算子（`quant_sparse_lightning_indexer`）的独立 pytest 用例。
单算子测试自行生成候选块，本算子在候选块集合内做 token 级 TopK。

## 文件

| 文件 | 说明 |
|---|---|
| `test_quant_sparse_lightning_indexer_consumer.py` | 第二级设备测试（38 个用例）。由主算子测试 `qli_cand_orig.py`（上溯 `/opt/tjj/qli_cand_test/test_qli_cand.py`）按 `cand_mode` 切片迁移而来，并在 2026-10-08 并入原 `qli_consumer_two_level.py`（算子级精度矩阵）中形状不重复的 13 个用例。用例形态：大 shape（16K/128K）、TND、0 轴非连续（pa_gap）、output_idx_offset、g 32/64、B=2/4 变长、topk=2048，以及两个锚点 —— `pin_overflow`（位置块数 2049 = 候选容量 2048 + 1，部分覆盖 + pin 尾块）与 `decode_long_m1`（s1=1 × s2=131072，生产 pool/table 规格） |
| `test_run.sh` | 用例驱动脚本：`all` 跑全量、`single -k <用例 id>` 跑指定用例 |
| `pytest.ini` | pytest 配置（`ci` / `graph` 标记） |

全部 38 个用例使用本算子参考实现生成候选，独立验证 QSLI 精度。

第一级与现网（候选关闭）路径的用例见
[../quant_lightning_indexer_v2/tests/pytest/](../../../quant_lightning_indexer_v2/tests/pytest/)。

## 运行

```bash
source <CANN>/set_env.sh          # 需要能 import 到含本算子的 cann_ops_transformer 包
export ASCEND_CUSTOM_OPP_PATH=<算子包 vendors 目录>   # 如用的是共享 CANN，可省略
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> pytest test_quant_sparse_lightning_indexer_consumer.py [-k <用例 id>]
# 或用驱动脚本：
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> ./test_run.sh all
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> ./test_run.sh single -k pin_overflow
```

文件名以 `test_` 开头，pytest 默认即可收集。
共享 NPU 上单进程长跑可能出现 507034（向量核超时）级联失败，建议逐用例进程隔离运行
（驱动脚本见工作区 `scripts/qliv2_debug/run_adapted_pytest_isolated.py`，支持 `--test-dir/--test-file`）。

比对规则位于本算子的 `result_compare_method.py`，与 QLI v2 使用相同的边界容忍判据。

## 独立测试与 TTK

QSLI 的输入生成、golden、metadata 参数准备和精度比较均位于本算子目录。
`assets/spec.py` 复用 `pytest/quant_sparse_lightning_indexer_golden.py`，支持
INT8、MXFP4、BSND/TND、分页 K、候选有效长度与输出偏移；回放使用实际输入。
`pytest/result_compare_method.py` 与 QLI v2 保持相同判定规则和阈值，独立维护，
不在运行时导入 QLI v2 的测试文件。

测试中不调用 QLI v2。需要检查两级串联时，可手动保存 QLI v2 输出的候选块索引和有效长度，
连同匹配的 Q/K、权重、scale、序列参数和分页表作为 QSLI 输入，按同一候选计算 golden 后比较。
