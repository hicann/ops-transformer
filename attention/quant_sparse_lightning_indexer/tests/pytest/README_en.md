# QuantSparseLightningIndexer Tests (second stage of two-level TopK)

This directory contains independent pytest cases for the second-stage consumer operator (`quant_sparse_lightning_indexer`). Single-operator tests generate their own candidate blocks.

## Files

| File | Description |
|---|---|
| `test_quant_sparse_lightning_indexer_consumer.py` | Second-stage device tests (38 cases). Migrated by slicing the `cand_mode` of the main-operator test `qli_cand_orig.py` (originally `/opt/tjj/qli_cand_test/test_qli_cand.py`), and merged on 2026-10-08 with the 13 shape-distinct cases from `qli_consumer_two_level.py` (the operator-level accuracy matrix). Case shapes: large shapes (16K/128K), TND, axis-0 non-contiguous (pa_gap), output_idx_offset, g 32/64, B=2/4 variable length, topk=2048, plus two anchors — `pin_overflow` (2049 position blocks = candidate capacity 2048 + 1, partial coverage + pinned tail block) and `decode_long_m1` (s1=1 × s2=131072, production pool/table configuration) |
| `test_run.sh` | Case driver: `all` runs everything, `single -k <case id>` runs the specified cases |
| `pytest.ini` | pytest configuration (`ci` / `graph` markers) |

All 38 cases generate candidates with the local reference to test QSLI independently.

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

The local `result_compare_method.py` uses the same boundary-tolerance criteria as QLI v2.

## Independent tests and TTK

QSLI owns its input generation, golden, metadata adapter, and comparator.
`assets/spec.py` reuses `pytest/quant_sparse_lightning_indexer_golden.py` for
INT8/MXFP4, BSND/TND, paged keys, candidate lengths, and output offsets.
Replay evaluates the actual inputs. The local `pytest/result_compare_method.py`
keeps the same comparison rules and thresholds as QLI v2 without importing it.

Tests do not call QLI v2. To check the two-stage flow manually, save its candidate
block indices and valid lengths together with the matching Q/K, weights, scales,
sequence parameters, and page table. Feed those inputs to QSLI and compare against
the golden evaluated with the same candidates.
