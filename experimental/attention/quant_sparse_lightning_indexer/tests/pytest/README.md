# QuantSparseLightningIndexer 测试（两级TopK 第二级）

本目录是第二级 consumer 算子（`quant_sparse_lightning_indexer`）的 pytest 用例。它必须与
第一级算子 [QuantLightningIndexerV2](../../../quant_lightning_indexer_v2/README.md) 配合使用：
第一级按块粒度挑出候选块并输出块级索引，本算子在候选块集合内做 token 级 TopK。

## 文件

| 文件 | 说明 |
|---|---|
| `qli_cand_orig_consumer.py` | 第二级设备测试（38 个用例）。由主算子测试 `qli_cand_orig.py`（上溯 `/opt/tjj/qli_cand_test/test_qli_cand.py`）按 `cand_mode` 切片迁移而来，并在 2026-10-08 并入原 `qli_consumer_two_level.py`（算子级精度矩阵）中形状不重复的 13 个用例与其端到端检查。用例形态：大 shape（16K/128K）、TND、0 轴非连续（pa_gap）、output_idx_offset、g 32/64、B=2/4 变长、topk=2048，以及两个锚点 —— `pin_overflow`（位置块数 2049 = 候选容量 2048 + 1，部分覆盖 + pin 尾块）与 `decode_long_m1`（s1=1 × s2=131072，生产 pool/table 规格） |

**两条比对路径**（同一文件内）：
1. **吃 golden 候选**（全部 38 个用例默认执行）：候选索引由参考实现生成，隔离 consumer 自身精度；
2. **吃第一级 NPU 候选**（`E2E_IDS` 内 8 个用例额外执行）：先跑第一级 `quant_lightning_indexer_candidate`
   取候选，再喂给 consumer，与「同一候选下」的 golden 比对 —— 属链路一致性验证（第一级自身精度由
   主算子目录的 `qli_cand_orig.py` 覆盖）。覆盖 decode/prefill/TND/0 轴非连续/offset + 两个锚点。

第一级与现网（候选关闭）路径的用例见
[../quant_lightning_indexer_v2/tests/pytest/qli_cand_orig.py](../../../quant_lightning_indexer_v2/tests/pytest/qli_cand_orig.py)。

## 运行

```bash
source <CANN>/set_env.sh          # 需要能 import 到含本算子的 cann_ops_transformer 包
export ASCEND_CUSTOM_OPP_PATH=<算子包 vendors 目录>   # 如用的是共享 CANN，可省略
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> pytest qli_cand_orig_consumer.py [-k <用例 id>]
```

注意：文件名不是 `test_*.py`，pytest 默认收集不到，需显式指定文件路径（或改名）。
共享 NPU 上单进程长跑可能出现 507034（向量核超时）级联失败，建议逐用例进程隔离运行
（驱动脚本见工作区 `scripts/qliv2_debug/run_adapted_pytest_isolated.py`，支持 `--test-dir/--test-file`）。

比对规则取自第一级算子的测试目录（`result_compare_method.py`，与本算子共用一套边界容忍判据）。
