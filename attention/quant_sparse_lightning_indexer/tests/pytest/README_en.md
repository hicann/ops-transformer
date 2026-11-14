# QuantSparseLightningIndexer Tests (second stage of two-level TopK)

This directory contains the pytest cases for the second-stage consumer operator (`quant_sparse_lightning_indexer`). It must be used together with the first-stage operator [QuantLightningIndexerV2](../../../quant_lightning_indexer_v2/README.md): the first stage selects candidate blocks at block granularity and outputs the block-level index, and this operator performs token-level TopK within the set of candidate blocks.

## Files

| File | Description |
|---|---|
| `test_quant_sparse_lightning_indexer_consumer.py` | Second-stage device tests (38 cases). Migrated by slicing the `cand_mode` of the main-operator test `qli_cand_orig.py` (originally `/opt/tjj/qli_cand_test/test_qli_cand.py`), and merged on 2026-10-08 with the 13 shape-distinct cases and their end-to-end checks from `qli_consumer_two_level.py` (the operator-level accuracy matrix). Case shapes: large shapes (16K/128K), TND, axis-0 non-contiguous (pa_gap), output_idx_offset, g 32/64, B=2/4 variable length, topk=2048, plus two anchors — `pin_overflow` (2049 position blocks = candidate capacity 2048 + 1, partial coverage + pinned tail block) and `decode_long_m1` (s1=1 × s2=131072, production pool/table configuration) |
| `test_run.sh` | Case driver: `all` runs everything, `single -k <case id>` runs the specified cases |
| `pytest.ini` | pytest configuration (`ci` / `graph` markers) |

**Two comparison paths** (in the same file):
1. **Consuming golden candidates** (default for all 38 cases): candidate indices are produced by the reference implementation, which isolates the accuracy of the consumer itself;
2. **Consuming first-stage NPU candidates** (run additionally by the 8 cases in `E2E_IDS`): run the first stage first (the unified entry `quant_lightning_indexer` with the two candidate parameters explicitly passed) to get the candidates, feed them to the consumer, and compare against the golden under the same candidates — this is a link-consistency check (the accuracy of the first stage itself is covered by `qli_cand_orig.py` in the main-operator directory). Covers decode/prefill/TND/axis-0 non-contiguous/offset plus the two anchors.

For the first stage and the production path (candidates disabled), see
[../quant_lightning_indexer_v2/tests/pytest/](../../../quant_lightning_indexer_v2/tests/pytest/).

## Running

```bash
source <CANN>/set_env.sh          # must be able to import the cann_ops_transformer package that contains this operator
export ASCEND_CUSTOM_OPP_PATH=<operator package vendors directory>   # may be omitted when using the shared CANN
ASCEND_RT_VISIBLE_DEVICES=<idle device> pytest test_quant_sparse_lightning_indexer_consumer.py [-k <case id>]
# or use the driver script:
ASCEND_RT_VISIBLE_DEVICES=<idle device> ./test_run.sh all
ASCEND_RT_VISIBLE_DEVICES=<idle device> ./test_run.sh single -k pin_overflow
```

The file name starts with `test_`, so pytest collects it by default.
Long single-process runs on a shared NPU may hit cascading 507034 (vector core timeout) failures; running one case per process is recommended
(the driver is `scripts/qliv2_debug/run_adapted_pytest_isolated.py` in the workspace, which supports `--test-dir/--test-file`).

The comparison rules come from the test directory of the first-stage operator (`result_compare_method.py`, which shares one set of boundary-tolerance criteria with this operator).
