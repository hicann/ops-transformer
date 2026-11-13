# Performance Test Scripts

This directory holds the scripts used to generate, run, collect and plot performance data for
flash_attn. All commands below are expected to be executed from
`attention/flash_attn/tests/pytests`.

## Script Overview

| Category | Script | Purpose |
|---|---|---|
| 1. Case generation | `gen_perf_base_cases.py` | Generate the base perf cases from `CASE_GROUPS` |
| 1. Case generation | `gen_perf_linearity_cases.py` | Generate the B/S1/S2 linearity cases |
| 2. Execution & collection | `run_perf_raw.py` | Run msprof directly, then export the per-run raw table and the final per-case table |
| 2. Execution & collection | `export_perf_raw.py` | Export the per-run raw table from an existing perf directory |
| 2. Execution & collection | `summarize_perf_raw.py` | Summarize the per-run raw table into one row per case |
| 3. Linearity test | `plot_perf_linearity.py` | Plot the B/S1/S2 linearity charts from the raw table |

## Prerequisites

1. The custom package has been built and installed, with both `flash_attn` and
   `flash_attn_metadata` compiled.
2. Compile with the full tiling-key set before running the full base/linearity case set,
   otherwise some kernels may be missing.
3. Source the CANN and custom-package environments yourself before running; the scripts no
   longer source them and do not hard-code any installation path.

---

## 1. Case Generation

### 1.1 Base Performance Cases

```bash
python3 tools/perf/gen_perf_base_cases.py
python3 tools/perf/gen_perf_base_cases.py --check
```

Outputs:

- `test_cases/perf/base_generated.py`
- `test_cases/perf/manifest.json`

Case name format (`QN`/`KVN` are the real head counts, `G = QN / KVN`):

```text
Perf_Base_D{D}_DV{DV}_M{mask}_QN{q_heads}_KVN{kv_heads}_{layout}_B{batch}_QS{s1}_KVS{s2}[_W{left}_{right}]_{dtype}
```

The case parameter keys are still `N1`/`N2` (framework contract); only the display name uses
`QN`/`KVN`, so that strings like `N116_N216` are not misread.

`win_left_right` only takes effect for `mask_mode=4`; cases without mask4 do not carry the
window fields.

### 1.2 Linearity Cases

```bash
python3 tools/perf/gen_perf_linearity_cases.py
```

Outputs:

- `test_cases/perf/linearity_generated.py`
- `test_cases/perf/linearity_manifest.json`

Both generators support `--dry-run`; `gen_perf_base_cases.py` additionally supports `--check`,
which validates the generated file and the manifest against the generator itself (case set,
per-case content, count and metadata).

## 2. Running Performance Cases and Collecting Results

### 2.1 Execution

Source the CANN and custom-package environments first, then run from
`attention/flash_attn/tests/pytests`:

```bash
python3 tools/perf/run_perf_raw.py --case-files perf.base_generated --runs 5 --out-dir /abs/path/run_base
```

Common options:

| Option | Description |
|---|---|
| `--case-files` | Case modules such as `perf.base_generated`; multiple modules can be comma-separated |
| `--runs` | Runs per case, default 5, warmup included |
| `--out-dir` | Absolute or relative path (relative to the current directory). The directory must not exist; when omitted, `perf_<timestamp>/` is created under `tools/perf/` |

### 2.2 Output Directory

```text
<out_dir>/
├── profiling_raw.csv        # per-run raw table, one row per run
├── profiling_summary.csv    # final performance table, one row per case
├── _perf_sweep.py           # the script measured by msprof for this run
├── _perf_log.jsonl          # per-case run status
├── run.log                  # condensed progress
├── run_full.log             # full msprof / subprocess output
└── msprof_output/           # raw profiling data
```

`run.log` keeps only `[batch]` and `[CASE_END]` lines, both using the same global index:

```text
start: 166 cases x 5 runs -> /path/to/out
  [batch] 1/166  Perf_Base_prefill_...
[CASE_END] 1/166  OK
  [batch] 2/166  Perf_Base_decode_...
[CASE_END] 2/166  OK
```

Both the terminal and `run.log` show the condensed progress only; see `run_full.log` for the
complete output.

### 2.3 Exporting an Existing Perf Directory

```bash
python3 tools/perf/export_perf_raw.py --perf-dir <perf_dir> --case-files perf.base_generated --runs 5
```

Writes `<perf_dir>/profiling_raw.csv` by default; use `--out` to override the path.
The top-level `op_summary.csv` is preferred; otherwise
`msprof_output/PROF_*/mindstudio_profiler_output/op_summary_*.csv` is used.

### 2.4 profiling_raw.csv

One row per run, so a case appears `--runs` times.

The first five columns are fixed:

```text
case, case_id, phase, mode, run
```

followed by the case parameters:

```text
B, KVN, G, D, DV, S1, S2, mask, dtype,
layout_q, layout_kv, layout_out,
max_seqlen_q, max_seqlen_kv,
cu_seqlens_q, cu_seqlens_kv,
seqused_q, seqused_kv
```

After that come the original `op_summary` columns, with `kernel_name` as the last column,
taken from `task_time_*.csv` (joined by Device / Stream / Task ID).
`cu_seqlens_*` and `seqused_*` are varlen fields and stay empty for ordinary cases.

### 2.5 profiling_summary.csv

One row per case. The column layout is similar to `profiling_raw.csv`, but the profiling part
only keeps the final performance data and the kernel name:

```text
case, case_id, phase, mode,
B, KVN, G, D, DV, S1, S2, mask, dtype,
layout_q, layout_kv, layout_out,
max_seqlen_q, max_seqlen_kv,
cu_seqlens_q, cu_seqlens_kv,
seqused_q, seqused_kv,
avg_us, theory_us, mfu_pct, kernel_name
```

`avg_us` is measured with the first run dropped for hot cases (`S1 > --cold-thr`, default 16)
and with min/max removed for cold cases; `kernel_name` is the first non-empty value of the case.

`theory_us` / `mfu_pct` are derived from the model FLOPs. The peak throughput must be passed
via `--peak-tflops` (default 0, meaning these two columns are left empty):

```text
theory_us = 2·B·N1·visible_pairs·(D+DV) / peak_flops      N1 = N2·G
mfu_pct   = theory_us / avg_us × 100
visible_pairs: mask=0 → S1·S2; mask=3 → right-aligned causal (row i sees min(S2, max(0, i+S2-S1+1))); mask=4 left empty
```

Generate it separately:

```bash
python3 tools/perf/summarize_perf_raw.py --raw <dir>/profiling_raw.csv --out <dir>/profiling_summary.csv
```

### 2.6 Chunking + Gaps (to keep long runs from affecting measurements)

```bash
python3 tools/perf/run_perf_raw.py --case-files perf.base_generated \
  --runs 5 --cold-thr 16 \
  --chunk-size 30 --gap-ms 20000 \
  --big-case-regex "KVS(32768|131072)" --big-gap-ms 30000 \
  --out-dir ./run_chunked
```

- `--chunk-size N`: split ordinary cases into chunks of N, sleeping `--gap-ms` between chunks;
- `--big-case-regex`: matching cases form their own chunk and run one by one, sleeping
  `--big-gap-ms` afterwards (0 means `--gap-ms`);
- each chunk gets its own `chunk_XX/` (with its own msprof output and raw/summary files), and the
  top-level `profiling_raw.csv` / `profiling_summary.csv` are merged automatically when done;
- `chunk_meta.csv` records the start/end time and the big-case flag of every chunk.

---

## 3. Linearity Test

### 3.1 Generate the Linearity Cases

See 1.2; generate `linearity_generated.py` first:

```bash
python3 tools/perf/gen_perf_linearity_cases.py
```

### 3.2 Run

```bash
python3 tools/perf/run_perf_raw.py --case-files perf.linearity_generated --runs 5 --out-dir /abs/path/run_linearity
```

The output directory also contains `profiling_raw.csv`, `profiling_summary.csv` and the raw
profiling data.

### 3.3 Plot

```bash
python3 tools/perf/plot_perf_linearity.py --raw <run_linearity>/profiling_raw.csv --out <run_linearity>
```

Outputs:

- `linearity_B.png`
- `linearity_S1.png`
- `linearity_S2.png`
- `linearity_all.png`
- `linearity_summary.csv`

The common fixed parameters (D, DV, mask, layout, G, KVN, dtype) are shown at the top, and each
subplot is annotated with the B/S1/S2 it fixes.

`--perf-dir` is the legacy entry point (reads `perf.csv` plus the per-case `op_summary`); new
workflows should use `--raw`.

---

## Custom Cases

Create a module under `test_cases/perf/`, following the existing `TestCases` format, then run:

```bash
python3 tools/perf/run_perf_raw.py --case-files perf.<module> --runs 5 --out-dir ./run_custom
```

The module must be importable as `test_cases.<module>`; `--case-files` accepts several
comma-separated modules. Only batch mode is supported; do not use
`test_flash_attn.py --one_by_one`, otherwise the export cannot map rows to cases.

## FAQ

| Symptom | Action |
|---|---|
| `out dir already exists` | Choose a new directory; reusing an existing one is not allowed |
| Row count differs from `runs × case count` | A case crashed or `--one_by_one` was used; check `run_full.log` and `_perf_log.jsonl` |
| `Cannot find tilingKey` | Rebuild and reinstall with the full tiling-key set |
| `kernel_name` is empty | Check that `task_time_*.csv` exists and the Device/Stream/Task ID match |
| Case module cannot be imported | Make sure the module lives under `test_cases/perf/` and pass `perf.<module>` to `--case-files` |

## Data Placement

Raw profiling data, `profiling_raw.csv` and `profiling_summary.csv` all live in the output
directory; for new runs, pass `--out-dir` explicitly to keep them in one place.
