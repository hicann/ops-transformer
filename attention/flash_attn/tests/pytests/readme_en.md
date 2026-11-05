# FlashAttention Accuracy Test Framework

English | [简体中文](./readme.md)

## 1. Architecture Overview

```
test_flash_attn.py  (entry point)
        │
        ├── case_loader.load_case_modules()    ← loads test_cases/*.py + Cartesian product expansion
        ├── case_loader.resolve_case_ids()     ← --case_id filtering
        │
        ▼  loop over cases
        ├── case_loader.normalize_params()     ← completes default values, TND/PA special handling
        │
        ▼
        ├── orchestrator.run_case()
        │       │
        │       ├── data.flash_attn_inputs.generate()   ← declarative tensor generation (Q/K/V)
        │       ├── data.build_flash_attn_params()      ← constructs metadata/kernel kwargs
        │       │
        │       ├── primary.compute(inputs, params)     ← NPU/GPU backend
        │       ├── golden.compute(cpu_inputs, params)  ← CPU reference (tfoward)
        │       │
        │       └── compare.check_result()              ← accuracy comparison
        │
        └── reporter.record() → terminal table + CSV
```

**Core modules** (`core/`):

| Module | Responsibility |
|------|------|
| `backends/base.py` | Backend abstract base class (`is_available`, `compute`, `device`) |
| `backends/cpu.py` | CPU Golden — calls `backends/cpu_impl.tforward` to perform the reference computation |
| `backends/gpu.py` | GPU — wraps the `flash_attn` library |
| `backends/npu.py` | NPU — calls `flash_attn_metadata` + `flash_attn`, including `print_metadata` |
| `data.py` | `InputSpec` declarative tensor generation + `build_flash_attn_params` parameter group construction |
| `case_loader.py` | Case loading (`load_case_modules`), parameter normalization (`normalize_params`), ID resolution (`resolve_case_ids`) |
| `orchestrator.py` | Orchestrator — chains data generation → backend computation → accuracy comparison |
| `reporter.py` | `Reporter` — terminal table + incremental CSV output |

---

## 2. Directory Structure

```
pytests/
├── core/                           # Core modules
│   ├── backends/
│   │   ├── base.py                 # Backend abstract base class
│   │   ├── cpu.py                  # CPU Golden (calls tforward)
│   │   ├── gpu.py                  # GPU (flash_attn library)
│   │   └── npu.py                  # NPU (flash_attn + print_metadata)
│   ├── data.py                     # InputSpec tensor generation + build_flash_attn_params
│   ├── case_loader.py              # Case loading + parameter normalization
│   ├── orchestrator.py             # Execution orchestration
│   └── reporter.py                 # Result reporting (terminal + CSV)
├── backends/                       # Low-level implementations (called by core/backends)
│   ├── cpu_impl.py                 # tforward — CPU reference implementation
│   ├── gpu_impl.py                 # flash_attn_gpu — GPU implementation
│   └── npu_impl.py                 # NPU helper functions
├── test_cases/                     # Test case definitions
│   ├── functional_stc.py           # Functional STC cases
│   ├── performance_redline_train.py
│   ├── performance_redline_infer.py
│   ├── functional_redline_train.py
│   ├── functional_redline_train_tnd.py
│   ├── functional_redline_infer.py
│   ├── base.py                     # Base cases
│   └── fia_stc.py                  # FIA STC cases
├── utils/                          # Utility functions
│   ├── compare.py                  # Accuracy comparison + failure distribution analysis
│   ├── data.py                     # Data generation, mask, layout conversion
│   ├── io.py                       # Tensor file read/write
│   ├── perf_parser.py              # Performance data parsing
│   ├── perf_runner.py              # Performance test runner
│   └── precision_visual.py         # Accuracy heatmap visualization
├── tools/
│   └── xlsx_to_testcase.py         # Excel → test case conversion
├── test_flash_attn.py              # Main entry point
├── test_npu_metadata.py            # Standalone metadata operator test
└── readme.md
```

---

## 3. Environment Dependencies

### General

| Dependency | Description |
|------|------|
| Python 3.8+ | — |
| PyTorch 2.2+ | Foundation for CPU/GPU computation |
| `einops` | Layout conversion (`pip install einops`) |
| `numpy` | Numerical computation |

### NPU Mode

| Dependency | Description |
|------|------|
| `TorchNPU` | PyTorch NPU extension |
| `cann_ops_transformer` | Provides `flash_attn` and `flash_attn_metadata` |

### GPU Mode (Optional)

| Dependency | Description |
|------|------|
| `flash-attn` | FlashAttention library (`pip install flash-attn --no-build-isolation`) |
| CUDA 12.0+ | GPU environment |

### Environment Check

```bash
# NPU
python -c "import torch_npu, cann_ops_transformer; print('NPU OK')"

# GPU
python -c "import torch, flash_attn, einops; print('GPU OK')"

# CPU-only
python -c "import torch, einops, numpy; print('CPU OK')"
```

---

## 4. Running Methods

### 4.1 Accuracy Test (Default)

```bash
# Run all cases (NPU mode)
python test_flash_attn.py

# Specify cases
python test_flash_attn.py --case_id TND_05,TND_06

# GPU mode
python test_flash_attn.py --case_id BASE_01 --use_gpu

# Specify case files
python test_flash_attn.py --case_files functional_stc --case_id TND_05
```

### 4.2 Metadata Debugging

This mode calls only the AICPU metadata operator, prints the core partitioning result, and skips kernel execution:

```bash
python test_flash_attn.py --case_id TND_05 --meta_only
```

The output contains:

- **Header** — `sectionNum`, `isFD`, `mBaseSize`, `s2BaseSize`
- **FA Metadata** — core partitioning information of the 36 AIC cores for each section (start/end of BN2/M/S2)
- **FD Metadata** — reduction tasks of active AIV cores (only when M_NUM > 0)

```
==============================================================================
  [Metadata Header]
  sectionNum  = 1
  isFD        = 1
  mBaseSize   = 128
  s2BaseSize  = 64
------------------------------------------------------------------------------
  [Section 0] FA Metadata — AIC cores  (36 cores × 16 slots, 7 fields used)
------------------------------------------------------------------------------
  Core       BN2_START      M_START     S2_START      BN2_END        M_END       S2_END  FIRST_FD_WS_IDX
  AIC00              0            0            0              7          127           63                0
  ...
```

### 4.3 Performance Test

```bash
# Collect performance data in batch
python test_flash_attn.py --case_files performance_redline_train --perf_mode --perf_output ./perf_out

# Execute cases one by one
python test_flash_attn.py --perf_mode --one_by_one
```

### 4.4 Three-Way Comparison

```bash
# Real-time CPU vs GPU vs NPU comparison
python test_flash_attn.py --case_id BASE_01 --compare_mode --use_gpu

# Offline comparison (from dump files)
python test_flash_attn.py --case_id BASE_01 --compare_mode \
    --load_gpu_dump ./dump/BASE_01/gpu_out.txt \
    --load_npu_dump ./dump/BASE_01/npu_out.txt
```

---

## 5. Command-Line Parameters

### Basic Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `--case_files` | str | Automatic scan of `test_cases/` | Case files, separated by commas |
| `--case_id` | str | `all` | Case names, separated by commas, prefix matching supported |
| `--device_id` | int | `0` | NPU device ID |
| `--use_gpu` | flag | — | Use the GPU backend |
| `--gpu_device` | int | `0` | GPU device ID |

### Golden Persistence Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `--save_golden` | str | — | Runs the normal comparison process and saves inputs/golden to the specified directory (in .pt format) |
| `--load_golden` | str | — | Loads golden from the specified directory and runs only the device comparison (skips the CPU golden computation) |
| `--case_timeout` | int | `300` | Timeout for a single case in subprocess isolation mode (seconds) |

### Debugging Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `--meta_only` | flag | — | Calls only the metadata operator and prints the core partitioning result |
| `--verbose_diff` | flag | — | Prints all elements that exceed the threshold when accuracy does not meet the requirement (prints only the first 10 by default) |
| `--visualize` | flag | — | Generates a heatmap PNG when accuracy does not meet the requirement |
| `--viz_dir` | str | `./viz_output` | Directory for saving heatmaps/analysis reports |
| `--fail_analysis` | flag | — | Outputs a multi-dimensional failure distribution analysis report when accuracy does not meet the requirement |
| `--compare_mode` | flag | — | Multi-backend comparison |
| `--graph_mode` | flag | — | Executes in graph mode |

### Output Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `--result_csv` | str | `result.csv` | Path of the accuracy result CSV |
| `--report_interval` | int | `20` | Interval for printing intermediate statistics |
| `--skip_accuracy` | flag | — | Skips the accuracy comparison |

### Performance Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `--perf_mode` | flag | — | Performance test mode |
| `--perf_runs` | int | `5` | Number of performance collection rounds |
| `--perf_cold_thr` | int | `16` | Cold/hot case boundary (S1) |
| `--perf_output` | str | `./perf_output` | Performance output directory |
| `--one_by_one` | flag | — | Executes cases one by one |

### Offline Comparison Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `--load_gpu_dump` | str | None | Path of the GPU dump file |
| `--load_npu_dump` | str | None | Path of the NPU dump file |

---

## 6. Test Case Format

Test cases are defined in the `TestCases` dictionary in `test_cases/*.py`. Each field is a list, and the framework automatically performs Cartesian product expansion.

### 6.1 Field Description

```python
TestCases = {
    "MY_CASE": {
        # ── Required ──
        "B":        [2],          # batch size (can be omitted for TND)
        "N1":       [8],          # number of query heads
        "D":        [128],        # head dim
        "Dtype":    ["fp16"],     # fp16 / bf16

        # ── Layout ──
        "layout_q":   ["BNSD"],   # BNSD / BSND / TND
        "layout_kv":  ["BNSD"],   # BNSD / BSND / TND / PA_BBND / PA_BNBD
        "layout_out": ["BNSD"],   # BNSD / BSND / TND

        # ── Sequence Length ──
        "S1":       [64],         # query seq len (max_seqlen_q for TND)
        "S2":       [1024],       # KV seq len (max_seqlen_kv for TND)

        # ── Optional ──
        "N2":       [1],          # number of KV heads, default = N1
        "DV":       [128],        # value head dim, default = D; when DV != D, metadata passes head_dim_v=DV through (the (192,128) combination is supported)
        "scale":    [None],       # scale factor, default 1/sqrt(D)
        "mask_mode":[0],          # 0=no mask, 3=causal, 4=band+causal
        "win_left": [-1],
        "win_right":[-1],
        "return_softmax_lse": [False],

        # ── Q/K/V Value Range ──
        "q_range": [(-5.0, 5.0)],
        "k_range": [(-5.0, 5.0)],
        "v_range": [(-5.0, 5.0)],

        # ── TND-Specific ──
        "cu_seqlens_q":  [[None]],   # cumulative sequence lengths (B+1 elements)
        "cu_seqlens_kv": [[None]],
        "seqused_q":     [[None]],   # actual sequence length per batch
        "seqused_kv":    [[None]],
    },
}
```

### 6.2 TND Cases

```python
"TND_05": {**BASE,
    "layout_q": ["TND"], "layout_kv": ["TND"], "layout_out": ["TND"],
    "cu_seqlens_q":  [[0, 64, 128, 192, 256, 320, 384, 448, 512]],
    "cu_seqlens_kv": [[0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]]},
```

`normalize_params` automatically derives `seqused` (difference) and `B` (len-1) from `cu_seqlens`.

### 6.3 Paged Attention Cases

```python
"PA_07": {**BASE,
    "layout_kv": ["PA_BBND", "PA_BNBD"],
    "block_size": [128]},
```

`normalize_params` automatically generates `block_table` from `seqused_kv` + `block_size`.

---

## 7. Execution Flow Details

### 7.1 Case Loading

```
test_cases/functional_stc.py::TestCases
        │
        ▼ load_case_modules()
  Cartesian product expansion: each parameter combination → an independent case
  Naming: "test_cases.functional_stc/TND_05"
  Multiple combinations: "test_cases.functional_stc/GS1_14_0", "..._1", ...
        │
        ▼ resolve_case_ids()
  --case_id TND_05 → matches all cases containing "/TND_05"
```

### 7.2 Parameter Normalization

Auto-completion logic of `normalize_params()`:

| Condition | Auto-Completion |
|------|----------|
| `N2` is missing | `N2 = N1` |
| `S2` is missing | `S2 = S1` |
| `DV` is missing | `DV = D` |
| TND + `seqused_q` missing | Derived from the difference of `cu_seqlens_q` |
| TND + `cu_seqlens_kv` missing | Copied from `cu_seqlens_q` |
| PA + `block_table` missing | Automatically generated from `seqused_kv` + `block_size` |

### 7.3 Tensor Generation

`InputSpec` declaratively defines the shape, dtype, and generation method of Q/K/V:

```python
flash_attn_inputs = (
    InputSpec("flash_attn")
    .tensor("q", layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D", "TND": "total_s1,N1,D"})
    .tensor("k", layout={...}, layout_override="layout_kv")
    .tensor("v", layout={...}, layout_override="layout_kv")
)
```

Special tokens: `total_s1` = `cu_seqlens_q[-1]`, `total_s2` = `cu_seqlens_kv[-1]`.

### 7.4 Parameter Group Construction

`build_flash_attn_params()` returns `(meta_kwargs, kernel_kwargs, out_layout)`:

- **meta_kwargs** → passed to `flash_attn_metadata()`
  - `cu_seqlens_q/kv`, `seqused_q/kv`, `max_seqlen_q/kv`, `num_heads_q/kv`, `head_dim`, `batch_size`, `mask_mode`, `layout_*`
- **kernel_kwargs** → passed to `flash_attn()`
  - `softmax_scale`, `cu_seqlens_q/kv`, `seqused_q/kv`, `max_seqlen_q/kv` (actual values are passed for TND, otherwise -1), `mask_mode`, `layout_*`

### 7.5 Accuracy Comparison

Decision criteria of `check_result()`:

```
Per-element: diff ≤ max(|golden| × 0.5%, 0.000025)
Overall:     proportion of elements exceeding the threshold ≤ 0.5%
```

Output: `PASS` / `FAIL` + MaxAbsErr + MeanAbsErr + MaxRelErr + FailRatio.

---

## 8. Layout Support Matrix

| Layout | CPU | GPU | NPU | Description |
|--------|-----|-----|-----|------|
| **BNSD** | ✓ | ✓ | ✓ | (B, N, S, D) default |
| **BSND** | ✓ | ✓ | ✓ | (B, S, N, D) |
| **TND** | ✓ | ✓ | ✓ | (total_tokens, N, D) variable-length sequences |
| **PA_BBND** | ✗ | ✓ | ✓ | (num_blocks, block_size, N, D) Paged KV |
| **PA_BNBD** | ✗ | ✓ | ✓ | (num_blocks, N, block_size, D) Paged KV |

---

## 9. mask_mode

| Value | Name | Description |
|----|------|------|
| `0` | NO_MASK | Full attention |
| `3` | RIGHT_DOWN_CAUSAL | Right-down aligned causal mask |
| `4` | BAND_CAUSAL | BAND + CAUSAL combination, `win_left`/`win_right` take effect |

---

## 10. File Responsibility Quick Reference

| File | Change Frequency | Typical Changes |
|------|----------|----------|
| `test_cases/*.py` | **Frequent** | Add/modify test cases |
| `core/data.py` | Occasional | Parameter group construction, tensor generation rules |
| `core/backends/npu.py` | Occasional | NPU invocation, `print_metadata` |
| `test_flash_attn.py` | Occasional | CLI parameters, feature entry points |
| `core/case_loader.py` | Rare | Parameter normalization logic |
| `core/orchestrator.py` | Rare | Execution flow orchestration |
| `core/reporter.py` | Rare | Report format |
| `utils/compare.py` | Rare | Accuracy comparison, failure analysis |
| `utils/data.py` | Rare | Mask generation, layout conversion |
| `backends/cpu_impl.py` | Normally unchanged | CPU golden (tfoward) |
| `backends/gpu_impl.py` | Rare | GPU implementation |
| `utils/perf_*.py` | Rare | Performance collection and parsing |

---

## 11. Frequently Asked Questions

### Q1: `ModuleNotFoundError: No module named 'cann_ops_transformer'`

Install `cann_ops_transformer` for NPU mode, or use GPU mode instead:

```bash
python test_flash_attn.py --case_id BASE_01 --use_gpu
```

### Q2: Abnormal Accuracy for TND Cases

Check the format of `cu_seqlens_q` (it must include a leading 0, and the length = batch + 1):

```python
"cu_seqlens_q": [[0, 64, 128, 192]],  # 3 batches, 64 tokens each
```

### Q3: TND Case Kernel Hang

Confirm that `max_seqlen_q`/`max_seqlen_kv` are passed in correctly. The kernel requires the actual max values under the TND layout:

```bash
python test_flash_attn.py --case_id TND_05 --meta_only  # check metadata first
```

### Q4: PA Case `block_size` Error

Different backends have different constraints on `block_size`. Refer to the official documentation of the corresponding backend to confirm the specific restrictions:

- For GPU, refer to the `flash_attn` library documentation
- For NPU, refer to the `flash_attn` operator documentation

### Q5: How to View the Actual Parameters of a Case

```bash
python -c "
from core.case_loader import load_case_modules, normalize_params
cases = load_case_modules(['test_cases.functional_stc'])
for name, raw in cases.items():
    if 'TND_05' in name:
        print(normalize_params(raw))
"
```
