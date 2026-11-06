# quant_flash_mla NPU Performance Test Tool

English | [简体中文](./README.md)

A minimal standalone set of performance test scripts extracted from ops-transformer-testkit (the non-quantized FA operator test framework), used to test the performance of **fully quantized MLA FlashAttention** (`QuantFlashMla`) on the NPU.

## Directory Structure

| File | Purpose |
|---|---|
| `run_perf.py` | CLI entry: one independent msprof process per case, summarizing into `perf.csv` |
| `perf_cases.py` | Case set: loaded at runtime from `../pytest/qfmla_paramset_func_rdv.CASES` (skipping `SKIP_CASES`) |
| `perf_codegen.py` | Generates a self-contained performance script per case (data construction + hot loop invocation); supports FP8/HIF8 and TND/BSND/BNSD |
| `perf_profiler.py` | `msprof --aic-mode=task-based` profiling + `msprof op --aic-metrics=PipeTimeline` pipe timeline collection |
| `perf_parser.py` | Filters by `OP Type=QuantFlashMla` and performs warm-start statistics (discards the first entry and averages the rest) |

## Quick Start

```bash
cd tests/perf

# 1. List all cases (same source as the pytest paramset, SKIP_CASES excluded automatically; currently 89 cases)
python3 run_perf.py --list

# 2. Single-case smoke test
python3 run_perf.py --runs 5 --case-filter _000000
python3 run_perf.py --runs 1 --case-filter _000000        # Run once only (including first-run overhead)
python3 run_perf.py --runs 5 --discard-first --case-filter _000000  # Discard warm-up and take the steady-state mean

# 3. Full run (recommended in the background)
nohup python3 run_perf.py --runs 5 > full_run.log 2>&1 &

# 4. Filter by substring (comma-separated; a case is selected if any substring matches)
python3 run_perf.py --runs 5 --case-filter Decode,NZ
python3 run_perf.py --runs 5 --case-filter PA_BnNBsD

# 5. Performance + PipeTimeline (additionally collects pipe timelines for all cases hit by --case-filter)
python3 run_perf.py --runs 5 --case-filter _000000 --pipe-timeline
python3 run_perf.py --runs 5 --case-filter Decode --pipe-timeline
```

## CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `--runs` | 5 | Number of iterations per case (≥ 1); averages over all entries by default |
| `--discard-first` | Disabled | Discards the first entry (warm-up) and takes the mean of the rest to obtain the steady-state value; requires `--runs >= 2` |
| `--case-filter` | None | Comma-separated substring filter on case names |
| `--output` | `results/<timestamp>` | Output directory |
| `--pipe-timeline` | Disabled | Boolean switch; for all cases hit by `--case-filter`, runs `msprof op --aic-metrics=PipeTimeline` once more after performance collection |
| `--list` | - | Lists case names only without running |

## Output Structure

```
results/<timestamp>/          (or the directory specified by --output)
├── perf.csv               # Summary: case_name, duration_us, mfu, samples, status, message, pipe_visualize_bin
└── <case_name>/
    ├── <case_name>.py     # Generated performance script (can be debugged standalone directly)
    ├── <case_name>.msprof.log
    ├── op_summary.csv     # Raw operator entries collected by msprof
    └── pipe_timeline/     # Present only when --pipe-timeline is enabled
        ├── <case_name>_pipe.py
        ├── <case_name>.msopprof.log
        └── OPPROF_*/      # PipeUtilization.csv + visualize_data.bin and more
```

The terminal also prints a summary:

```
[SUMMARY] 89 cases, 0 failed, results -> .../perf.csv
  [HOT] Decode_..._TND_PA_lSE_000000    117.83 us  MFU=12.34%
```

## MFU Calculation

After collecting the warm-start duration, the MFU of each case is estimated with the following formulas (per batch; multiple batches are accumulated one by one):

```
sp0:            compute_num = 2*kvs*qs*(qkd+vd)*qn
sp3 (qs<kvs):   compute_num = 2*(kvs-qs+1+kvs)*qs/2*(qkd+vd)*qn
sp3 (qs>=kvs):  compute_num = 2*kvs*kvs/2*(qkd+vd)*qn
fp8_peak_tflops = 2*16*16*32*1.6*core_num*0.001
mfu = (compute_num / (duration*1e-6)) * 1e-12 / fp8_peak_tflops
```

- `qs` = `seqused_q[b]`, `kvs` = `cache_seqlens[b]`, `qn` = `N_q`
- `qkd` = 576 (nope 512 + rope 64, constant `QK_HEAD_DIM`), `vd` = 512 (V head dim,
  constant `V_HEAD_DIM`, taken from `perf_codegen.D_V`); the head dims of QK and PV differ and must be counted separately
- `core_num` takes the device `cube_core_num` at runtime (falls back to 32 in a pure CPU environment); 32 cores correspond to `fp8_peak_tflops = 838.86`
- The result is written as a **percentage** to the `mfu` column of `perf.csv`, and the terminal also prints `MFU=xx.xx%`
- Cases with `mask_mode != 3` are computed with the sp0 formula

## Case Source and Fields

Cases are no longer maintained in the perf directory. Instead, `perf_cases.load_cases()` loads them at runtime from `CASES` in `../pytest/qfmla_paramset_func_rdv.py` and skips its `SKIP_CASES`, so performance cases and functional cases always share the same source. To add or remove cases, modify the pytest paramset.

`load_cases()` normalizes paramset entries into the following fields:

| Field | Description |
|---|---|
| `B / N_q / N_kv` | Number of batches / number of Q heads / number of KV heads |
| `seqused_q` | Actual Q sequence length per batch |
| `cache_seqlens` | KV cache sequence length per batch |
| `enable_pa` | Annotation only; the new interface requires `block_table`, which is constructed uniformly through PA order mapping, and the kernel data volume is semantically equivalent to noPA |
| `kv_cache_layout` | `BnNBsD` / `BnBsH` / `NZ` (mapped to `layout_kv`: `PA_BNBD` / `PA_BBND` / `PA_NZ`) |
| `sparse_mode` | Same as the `mask_mode` of the paramset: 0 = no mask, 3 = causal (mask passed in as int8) |
| `enable_lse` | Whether to return the softmax lse |
| `num_blocks` | Physical cache block reuse: 0 = no reuse (sequential allocation); when > 0, blocks are mapped randomly by block count to simulate production block reuse |
| `quant_mode` | 0 = hifp8 (uint8 carrying hif8 encoding, `32768/row_max`); 1 = fp8_e4m3 (`448/row_max`) |
| `layout` | Q input layout: `TND` / `BSND` / `BNSD`; also determines the `q_descale` shape and `layout_out` |
| `graph_path` | Annotation only; perf uniformly uses the single-operator invocation |

Construction differences between `quant_mode` and `layout`:

| Scenario | Q / q_descale | cu_seqlens_q |
|---|---|---|
| FP8 (1) | Quantized to `float8_e4m3fn`, `QK_MAX=448` | Per layout |
| HIF8 (0) | Reuses `pytest/common/generate_hifloat8_data.py` to encode as `uint8`, `QK_MAX=32768` | Per layout |
| TND | `q (T,N,576)` / `descale (T,N)` | Passed in |
| BSND | `q (B,S,N,576)` / `descale (B,S,N)` | `None` |
| BNSD | `q (B,N,S,576)` / `descale (B,N,S)` | `None` |

Note: the current paramset contains only `quant_mode=1 + TND`, so perf runs FP8/TND only by default. To add HIF8 or BSND/BNSD performance cases, first add the corresponding fields to the pytest paramset.

## Statistics Rules

- **Filtering**: only AI Core kernels with `OP Type=QuantFlashMla` are counted; interfering operators such as Metadata (AI_CPU) and ZerosLike are excluded automatically.
- **Statistics rule**: by default, the mean is taken over all samples. With `--discard-first`, the 1st call of each independent process (including tiling cache warm-up) is discarded and the mean is taken over the rest; `--runs` must be ≥ 2. The `samples` column records the number of entries actually included in the mean.
- **Metadata overhead**: `quant_flash_mla_metadata` is called only once outside the hot loop and is not counted in the kernel duration. To measure the end-to-end overhead including metadata, move the metadata call in the generated script into the loop (the script is at `<case_name>.py` in the output directory and can be modified manually and run separately).

## PipeTimeline

`--pipe-timeline` (a boolean switch) runs `msprof op --aic-metrics=PipeTimeline --launch-count=1 --kernel-name=QuantFlashMla` **once more separately** for **all** cases hit by `--case-filter` after performance collection completes. To narrow the pipe collection scope, narrow it directly with `--case-filter`; no second set of filter conditions is provided.

- It collects the Cube/Vector/MTE pipe timeline at the PMU hardware counter level (the same collection method as Phase 2 of the non-quantized framework `core/msopprof.py`).
- Artifacts are under `<case>/pipe_timeline/OPPROF_*/`:
  - `visualize_data.bin`: open it with **MindStudio Insight** to view the pipe timeline visualization;
  - `PipeUtilization.csv` and more: CSVs of the share of each pipe.
- `perf.csv` adds the `pipe_visualize_bin` column to record the bin path; it is empty when not collected.
- Reason for running it separately from performance collection: the PipeTimeline visualize_data.bin is large, and msprof collection modes are mutually exclusive.

## Known Notes

- **torch extension cache isolation**: `run_perf.py` automatically sets `TORCH_EXTENSIONS_DIR=perf/.torch_ext/`. The shared `~/.cache/torch_extensions/` may be polluted by an old `cann_ops_transformer` from another conda environment (the `.so` and the wrapper parameter signature are inconsistent, and metadata reports TypeError). The isolated cache directory ensures the `.so` is always compiled from the csrc of the current environment.
- Data construction is aligned with the NPU path of `tests/pytest/common/qfmla_golden.py`: Q per-token-head quantization (arranged into TND/BSND/BNSD by `layout`), K per-tensor quantization → PA cache; V is reused internally by the operator from the first 512 dimensions of K.
- HIF8 data is encoded into `uint8` on the CPU by `pytest/common/generate_hifloat8_data.py` and then passed directly to the operator (no dtype conversion), consistent with the data pipeline of the functional cases.
- The `attn_mask` of `mask_mode=3` must be passed in as **int8** (bool is rejected by operator validation).
- `results/`, `.torch_ext/`, and `*.log` are ignored in `.gitignore`; do not commit them.
