# `all_gather_matmul_v2` Batch-Invariance (BI) Testing

English | [简体中文](./README.md)

Batch-invariance test suite for `aclnnAllGatherMatmulV2` / `npu_all_gather_quant_mm` (MXFP8, `block_size=32`). This suite is a companion implementation of the `matmul_reduce_scatter_v2` BI suite (**PR #6593**) and follows the same methodology. It covers the two `x2` orientations of this operator (**non-transposed** and `transB`), which is exactly the area changed by **PR #4705** ("Support `transB` and verify MX quantization BI consistency").

## 1. Why BI Testing Is Needed

Existing UT/ST covers `acc` (numerical precision tolerance) and `perf` (performance thresholds), but does not cover literature batch-invariance:

```text
y[i] = f(x[i], W)   — the i-th output row depends only on the i-th input row + the shared weights W, and is independent of other rows in the same batch.
```

An operator can satisfy `acc`, `perf`, and cross-rank determinism at the same time and still violate literature-BI (row-to-row interaction). For the definition, see: <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference>

## 2. Methodology (Three Steps, All Required, Consistent with README §2 of #6593)

- **Phase 0 HCCL warm-up**: `N_WARMUP=2` dummy calls stabilize the HCCL state (the first call is prone to cold-start errors such as `HcclAllocComResourceByTiling` / 561000, or selects a non-deterministic ring topology). Errors during warm-up are not counted.
- **Phase 1 multiple deterministic baselines**: run the same input `N_TRIALS=3` times; all SHAs must be identical; otherwise record `NON_DET`, and this cell does not enter Phase 2.
- **Phase 2 combined-BI (variant A vs B)**: the same pinned probe row with **different neighbor rows** (`seed_a` vs `seed_b`); the probe row output must be byte-identical. Identical → `BI_PASS`; different → `BI_FAIL`.

Three-state output: `BI_PASS` / `BI_FAIL` / `NON_DET` (in addition, `ERROR` = the operator rejects the call, and `SKIP` = the shape is not applicable).

## 3. Coverage

For each shape × `M ∈ {8,16,32}`, `(K,N)`:

| Group | (K, N) |
|---|---|
| #6593 range (small/medium) | (256,64) (256,128) (512,64) (512,128) (1024,64) (1024,128) |
| Large + non-square | (1024,1024) (2048,2048) (4096,4096) (4096,2048) (2048,4096) |

Orientation: `BI_ORIENT=both` (default, non-transposed + `transB`) / `notrans` / `trans`.
WS: `run_bi_test.sh` runs 2/4/8 by default.

## 4. Running

```bash
cd mc2/all_gather_matmul_v2/tests/bi/
bash run_bi_test.sh "2 4 8"            # all WS
bash run_bi_test.sh "2" 0,1            # single WS + specified cards
BI_ORIENT=notrans bash run_bi_test.sh  # non-transposed only
```

Each run prints:

```text
=== AllGatherMatmulV2 BI rigorous WS=2 N_WARMUP=2 N_TRIALS=3 DATA=random orient=both ===
  [NON-TRANS] SUMMARY BI_PASS=33 BI_FAIL=0 NON_DET=0 ERROR=0 SKIP=0
  [TRANS_B  ] SUMMARY ...
```

Any `BI_FAIL > 0` is a true BI violation (exit code 1); `NON_DET > 0` requires investigation of the test system or hardware; `ERROR` usually indicates that the `block_size` gate patch is not applied (EZ0002) or that the shape is rejected.

## 5. Prerequisites

- **`block_size` gate fix**: MXFP8 requires `block_size=32`; tiling without the patch rejects the call with `EZ0002 blockSize should be 0` (cann/ops-transformer issue #2778 / PR #6137). Run on a build or vendor tree that contains this fix.
- **Idle NPU cards**: `HcclAllocComResourceByTiling ret=4` indicates that a card is occupied by another HCCL job (port 16666). Switch to idle cards (`ASCEND_RT_VISIBLE_DEVICES`).
- **torchrun** (not manual multiprocessing); otherwise HCCL resource allocation for the MC2 operator fails.

## 6. Data Generation

- **Non-transposed**: `BI_DATA_MODE=random` (default) generates random `uint8` → `fp8_e4m3` / `e8m0`. It is self-contained, has no external dependencies, and has been fully verified.
- **`transB`**: the driver internally **integrates** `script/generate_mx_data.mx_quantize` (valid MX values, square K==N). `x2`/scale is passed as a **non-contiguous transposed view** (`x2.t()`, `torch.transpose(x2s,0,1)`, without `.contiguous()` — adding it raises `561002`). Random bytes trigger aicpu exceptions under `transB`, so `transB` **must** use valid MX data. It depends on `en_dtypes` / `ml_dtypes` (bundled in the ST tree), and runs directly in the ops-transformer ST tree.

## 7. Verified Status

- **Non-transposed**: WS=2/4/8 × 11 shapes × `M ∈ {8,16,32}` = **99/99 `BI_PASS`**.
- **`transB`**: WS=2/4 **`BI_PASS` + numerically correct** (combined-BI + batch-size-BI; the probe SHA is consistent across M/seed/WS; correctness versus the `fp8`-dequantized `fp32` reference gives median_rel 0.001). Verified on a vendor tree containing the merged version of #4705 (master `03b5c20`).

## 8. Integration with the ATK Framework (To Be Discussed)

Same as #6593: run as a standalone driver without depending on ATK. During integration, keep `n_warmup` / `n_trials` configurable, summarize the three-state results independently, and do not automatically mark `NON_DET` as a failure but write it into the report.
