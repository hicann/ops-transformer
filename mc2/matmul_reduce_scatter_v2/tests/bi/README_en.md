# Batch-Invariance (BI) Tests — `matmul_reduce_scatter_v2`

English | [简体中文](./README.md)

This directory provides the batch-invariance test suite for the `aclnnQuantMatmulReduceScatterV2` operator, serving as the **reference implementation** for the requirement proposed in issue #2956 to add a BI test dimension to the official MC2 operator test suite.

## 1. Why BI Tests Are Needed

Existing UT/ST tests cover:

- `acc`: Numerical accuracy comparison against the reference implementation (within tolerance)
- `perf`: Performance thresholds (throughput / latency)

However, they do not cover **literature batch-invariance**:

```text
y[i] = f(x[i], W)

The output row i depends only on the input row i and the shared weights W,
and is independent of the content of other rows in the same batch.
```

For the literature definition, see <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference>

An operator can satisfy `acc` and `perf` and satisfy cross-rank determinism, but still violate literature-BI (rows affect each other). `AllGatherMatmulV2` fixed in PR #6373 is of this type.

## 2. Methodology Note — All Three Steps Are Required

> **This section is the focus**. The previous methodology (running variant A and variant B only once per cell) had a hidden flaw: it could not distinguish a real BI bug from trial non-determinism. The three-step procedure below is the corrected version.

Each BI cell goes through the following three phases:

### Phase 0: HCCL Warm-up

After workers start, the HCCL state is prone to a cold-start state in the first few calls. The first operator call often reports `HcclAllocComResourceByTiling failed` (error code 561000), and even if the call succeeds, a non-deterministic ring topology may be selected.

**Approach**: Before the test matrix officially starts, make `N_WARMUP=2` dummy calls (with any valid input) to stabilize the HCCL state. Errors during warm-up are not counted in the test results.

### Phase 1: Multi-Trial Determinism Baseline

For each cell, variant A runs `N_TRIALS=3` times with the same input, and all SHA values must be exactly identical; the same applies to variant B.

```python
for _ in range(N_TRIALS):
    sha_list_a.append(run(seed_a))
assert len(set(sha_list_a)) == 1  # variant A must be deterministic
```

If `sha_list_a` is inconsistent, mark the cell as `NON_DETERMINISTIC`; the cell **does not enter Phase 2** (because the A vs B comparison is meaningless).

### Phase 2: Variant A vs Variant B BI Check

Only when Phase 1 passes is `sha_a` compared with `sha_b`:

- Identical → `BI_PASS` (rows are independent)
- Different → `BI_FAIL` (a real BI violation)

### Three-State Output

| Status | Meaning | Interpretation |
|---|---|---|
| `BI_PASS` | Determinism baseline passed + A == B | Row content is independent across batches ✓ |
| `BI_FAIL` | Determinism baseline passed + A ≠ B | **A real BI violation**; fix required |
| `NON_DETERMINISTIC` | Multiple runs with the same input produce inconsistent SHA values | A test system issue (HCCL state / shared NPU / hardware); investigate it; **do not simply judge it as a BI fail** |
| `ERROR` | The operator call throws an exception | Operator-plugin shape rejection or an infrastructure failure |

**Key lesson**: Confusing `BI_FAIL` with `NON_DETERMINISTIC` produces many false positives that bury real bugs. This methodology flaw was corrected through multi-trial re-testing during the 2026-06-06 investigation.

## 3. Operator Status

`matmul_reduce_scatter_v2` is **natively BI-clean**: `op_kernel/arch35` uses `ExecuteAicMatMulPipeline` to process all input rows through a single `Mc2QuantBatchMatmulASWKernel<isGather=false>` wrapper, without the LOCAL / GATHER two-path split. This test suite confirms zero failures under the rigorous methodology, serving as a positive baseline to verify that the test methodology itself works.

`AllGatherMatmulV2` (pre-PR-#6373) triggers `BI_FAIL` in the same test suite, proving the sensitivity of the methodology.

## 4. Matrix Parameters

| Dimension | Value |
|---|---|
| `WorldSize` | {2, 4, 8} |
| M_total | {8, 16, 32} (must be divisible by WS) |
| K | {256, 512, 1024} (crosses the cube `baseK`=128 boundary multiple times) |
| N | {64, 128} (including the `baseN`=64 boundary) |
| Quantization scale | (`x1`=ID, `x2`=VAR) — K-axis varying, a known BI trigger condition |
| Target row position | 0 |
| N_WARMUP | 2 |
| N_TRIALS | 3 |

About 24 cells × WS={2,4,8} = 72 cells, with 6 operator calls per cell (3 trials × 2 variants). The estimated CI wall-clock time is < 15 min.

## 5. How to Run

Requires V300 (Ascend 950PR/DT) hardware + CANN 9.1+ + `torch_npu` 2.9+.

```bash
cd mc2/matmul_reduce_scatter_v2/tests/bi/
bash run_bi_test.sh
```

`run_bi_test.sh` launches `bi_test_driver.py` three times with WS=2/4/8, and each run prints:

```text
=== matmul_reduce_scatter_v2 BI rigorous WS=2 N_WARMUP=2 N_TRIALS=3 ===
  Phase 0: HCCL warm-up...
  result: BI_PASS=24 BI_FAIL=0 NON_DET=0 ERROR=0 SKIP=0
```

Any `BI_FAIL > 0` indicates a real BI violation; `NON_DET > 0` requires investigating the test system rather than the operator; `ERROR` is usually an operator-plugin shape rejection.

## 6. Integration with the ATK Framework (To Be Discussed)

The current implementation is a standalone Python driver and does not depend on ATK. The ATK integration solution discussed in issue #2956 requires adding a `'bi'` key to the `standard` field and adding a `_run_bi_check` hook to the executor. During integration, three key points must be retained:

1. `n_warmup` and `n_trials` must be configurable (this is the core of the BI methodology, not boilerplate)
2. The three-state results `BI_PASS / BI_FAIL / NON_DETERMINISTIC` must be summarized independently and cannot be merged
3. `NON_DETERMINISTIC > 0` should not automatically mark the operator as failed, but it must be written into the test report to raise attention

The unified refactoring will be performed after the maintainers evaluate it. This PR prioritizes ensuring that the BI tests can run independently, serving as an executable specification of the methodology.
