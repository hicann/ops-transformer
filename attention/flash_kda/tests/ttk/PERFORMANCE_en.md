# FlashKDA A5/A6 performance measurement

[中文](PERFORMANCE.md)

Both architectures read the same 25 profiling cases from `cases/ttk_cases.csv`, preserving inputs, seeds, fusion/cache switches and training/inference semantics. There is no separate A6 CSV or generator. Metadata uses native architecture capacity; transport preparation is outside the measurement. Performance cases do not assert numerical accuracy.

## Two metrics

| Metric | Included | Excluded |
| --- | --- | --- |
| Main kernel | Profiler `Duration(us)` for the selected template's complete compute kernel | Metadata, auxiliary Cast/Transpose/Zero kernels, warmup and initial compilation |
| End to end | Wall time for regenerating metadata, forward conversions/allocations, the public three-output entry including FP32-to-BF16 chunk-state conversion, and NPU synchronization | Input generation/H2D, warmup, initial compilation and profiler overhead |

Sample the metrics separately. Do not estimate end-to-end time by adding a metadata measurement to main-kernel time, or replace operator validation functions. Run serially on one device/stream. Defaults are 10 warmups, 50 samples and three independent rounds; report the median of the three round means. Use exclusive devices and record SDK/wheel fingerprints. Old A6 SDK 0.8.2 measurements are not a current v3 baseline.

Compare the adapted interface against A5 with the same inputs, seed, output switches and timing boundaries. Convert old BNSD outputs to the same logical layout before numerical comparison; identify the end-to-end cost of newly returned training states. Retain before/after A6 data. Missing comparable A5 data means pending acceptance. `benchmark.py` records the reference commit as `interface_contract` in `environment.json`; the comparison rejects missing or different identities. Recapture old reports using matching versions rather than editing their identities.

## Recommended: one script for both metrics

Complete real compilation, installation and a successful public call first. Run from the repository root in each architecture's matching environment. `benchmark.py` defaults to `KDA_DUT=installed`; all three interfaces come from the dedicated wheel with Native mode `require`. Source measurements explicitly use `KDA_DUT=source` and are archived separately. Set `SDK_TAG` to the actual wheel/source fingerprint. The output directory must be outside the repository and must not already exist.

```bash
# A5 environment
export KDA_DUT=installed
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python attention/flash_kda/tests/ttk/benchmark.py \
  --arch arch35 --sdk-tag "$SDK_TAG" --templates auto staged persistent \
  --output /tmp/kda-a5-perf
python attention/flash_kda/tests/ttk/kda_runtime.py --check-native /tmp/kda-a5-perf/evidence

# A6 environment: matching DSL/OpKit 0.8.3 and v3 package
export CANNBOTDSL_NPU_ARCH=dav-9201
export CANNBOTDSL_CACHE_DIR="$(mktemp -d /tmp/kda-a6-cache.XXXXXX)"
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python attention/flash_kda/tests/ttk/benchmark.py \
  --arch arch92 --sdk-tag "$SDK_TAG" --templates auto staged persistent \
  --output /tmp/kda-a6-perf
python attention/flash_kda/tests/ttk/kda_runtime.py --check-native /tmp/kda-a6-perf/evidence
```

Use `--templates auto` for automatic routing only, or `--cases P22-001 P24-001` to focus on selected shapes. The forced-template order alternates by case/round; this test interface adds no operator parameter. Each profiler directory must contain exactly 50 main-kernel records for the selected template. Missing records, other templates or warmup records fail; there is no Host-time fallback. Output records include the schedule, active core count, DV, input hashes, source identity, kernel name and raw samples.

Copy both result directories to one machine and compare:

```bash
python attention/flash_kda/tests/ttk/compare_performance.py \
  /tmp/kda-a5-perf /tmp/kda-a6-perf > /tmp/kda-performance-comparison.json
```

The comparison rejects mismatched input hashes, attributes, cases/templates, rounds or sampling protocols. Both A6/A5 ratios must be at most 1 for every case; unmet criteria produce exit code 1. Native metadata, active core counts and actual automatic templates may differ by architecture; metadata hashes need not match.

## TTK entry point

Each architecture uses its own adapter, inputs and ABI validation, sharing `assets/performance.py` for timing. Configure the architecture, installed entry, PYTHONPATH and separate evidence directory, then select profiling IDs from the shared CSV:

```bash
KDA_PERF_IDS="$(python "$KDA_TTK/kda_cases.py" --suite profiling --ids)"
KDA_PERF_TIMING=kernel KDA_PERF_TEMPLATE=auto python -m ttk e2e \
  -i "$KDA_TTK/cases/ttk_cases.csv" -t "$KDA_PERF_IDS" \
  --plugin "$KDA_PLUGIN" --run 50 --warmup true -o "$KDA_OUT/kernel.csv"

KDA_PERF_TIMING=e2e KDA_PERF_TEMPLATE=auto python -m ttk e2e \
  -i "$KDA_TTK/cases/ttk_cases.csv" -t "$KDA_PERF_IDS" \
  --plugin "$KDA_PLUGIN" --run 50 --warmup true \
  --task-prof=false -o "$KDA_OUT/e2e.csv"
python "$KDA_TTK/kda_runtime.py" --check-native "$KDA_EVIDENCE_DIR"
```

`KDA_PERF_TEMPLATE` also accepts staged or persistent. Forced templates use the selected source's private metadata builder without adding a public operator parameter; auto uses the public metadata interface. Read complete end-to-end wall samples from evidence `e2e_us`, and main-kernel times from the profiler. Prefer the benchmark script above for three independent rounds and the final two-metric comparison. Check Native hits after execution. Performance results without numerical assertions do not certify training-cache accuracy.

## A6 data movement and regression checks

Shared Stage1 separates Q/K transfer into issue/finish: after current Q/K consumers and public cache writes finish, the original UB slots prefetch the next task, attempting to overlap the current Cube triangular inversion. Staged and persistent share this path. It adds no Q/K double buffer and does not change normalization, gating or matrix inversion.

The persistent cross-group UB state residency change in `0832a5119` was reverted. The user's A6 bisect passed at `06397b81a` and `347985d5a`, while `0832a5119` failed the staged/persistent final-state comparison with a reported maximum difference of about 0.0236. The computation was restored to `347985d5a`, retaining Q/K prefetch and common measurement. Persistent continues to use DV128, four-chunk groups and FP32 GM handoff between groups; cache keys and routing features were restored as well.

Inter-group `h` and its TensorSpec are FP32, and GM writeback directly copies `ub_state_fp32`. Both paths use the same BF16 conversion when publishing to L1. The bisect identifies a regressing commit, but does not prove that residency is more accurate or staged adds rounding. UB lifetime across rewind and actual compiled artifacts require device evidence; local control-flow simulation cannot validate them.

After updating, use fresh processes and independent caches to run public TTK correctness and all 13 redlines, followed by the 25 performance cases. Preserve the numerical criteria; one passing case does not certify the whole suite. Do not use `0832a5119` measurements as a performance baseline for the current implementation.
