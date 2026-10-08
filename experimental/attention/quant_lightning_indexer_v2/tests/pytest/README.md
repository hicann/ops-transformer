# QuantLightningIndexerV2 pytest 用例（两级TopK 第一级 / 现网路径）

本目录只保留按本分支契约改造后的设备测试用例；官方原有的 single/batch/paramset 那套
（`test_run.sh`、`test_quant_lightning_indexer_v2_{single,batch,paramset}.py`、
`quant_lightning_indexer_v2_golden.py`、`quant_lightning_indexer_v2_acl_graph.py`、
`qliv2_test_utils.py`、`qliv2_parameter_normalization.py`、`collect_perf_data.py`、
`batch_isolated_run.sh`、`batch/`）已于 2026-10-08 按约定删除（本分支后续只用下面两个文件跑验证）。

## 文件

| 文件 | 说明 |
|---|---|
| `qli_cand_orig.py` | 两级TopK **第一级（candidate/source）与现网路径**的设备测试，由 `/opt/tjj/qli_cand_test/test_qli_cand.py` 改造而来：`candidate_mode` 属性已删除，开关由 `candidate_topk_blocks` 承载（-1=关闭/现网，2048=开启第一级）；用例按 `cand_mode` 路由到两个入口（3 -> 现网基础入口，1 -> 第一级 `quant_lightning_indexer_candidate`），统一返回 3 元组。含 cmp_ratio 1/2、mask 0/3、g 32/64、BSND/TND、B=4 变长、pin 边界、0 轴非连续（pa_gap/offset）等形态 |
| `result_compare_method.py` | CPU golden 与 NPU 输出的精度比对判据（官方边界容忍规则），被本目录用例与第二级用例共用 |
| `pytest.ini` | pytest 配置（本目录 rootdir 锚点；声明 `ci` / `graph` 标记） |

第二级（consumer）算子的用例在
[../../quant_sparse_lightning_indexer/tests/pytest/](../../../quant_sparse_lightning_indexer/tests/pytest/)：
`qli_cand_orig_consumer.py`（原 `cand_mode=2` 用例迁移而来）与 `qli_consumer_two_level.py`
（吃 golden 候选 / 吃第一级 NPU 候选的算子级精度用例）。

## 运行

```bash
source <CANN>/set_env.sh        # 需能 import 到含这两个算子的 cann_ops_transformer 包
export ASCEND_CUSTOM_OPP_PATH=<算子包 vendors 目录>   # 用共享 CANN 且已装本分支包时可省略
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> pytest qli_cand_orig.py [-k <用例 id>]
```

注意：`qli_cand_orig.py` 文件名不是 `test_*.py`，pytest 默认收集不到，需显式指定文件路径（或改名）。
共享 NPU 上单进程长跑到约 40 个用例后可能出现 507034（向量核超时）级联失败，建议逐用例进程隔离
运行（驱动脚本见工作区 `scripts/qliv2_debug/run_adapted_pytest_isolated.py`，支持 `--test-dir/--test-file`）。
