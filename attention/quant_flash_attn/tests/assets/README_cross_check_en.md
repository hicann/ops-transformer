# QuantFlashAttn: three-way cross_check comparison

Quantized attention kernels perform online softmax independently within each
core/section and quantize P to FP8. If the CPU golden computes with a global
running max, its P quantization noise does not match the NPU's, and the two-way
criterion mistakes this inherent noise for error, producing false failures.

The three-way comparison uses one high-precision CPU golden as the common
reference. The criterion is the ratio of "NPU error against the reference" to
"XPU sectioned reference error against the reference", i.e. the NPU error must
not exceed the competitor's noise upper bound by more than a fixed factor.

| Role | Source | TTK slot |
| --- | --- | --- |
| target | Output of the `quant_flash_attn` operator under test | `output` |
| bench | CPU golden under `--golden-mode Promote` (FP32 output) | `golden` (common reference) |
| third_party | XPU sectioned quantized reference (native precision, no promotion, main output cast to bf16) | `third_party` |

## Call chain

```text
customize_inputs → impl/inputs.py                    deterministic seed input generation
        │
        ├─ operator execution           → outputs = [atten(bf16), lse(fp32)?]
        │
golden  → spec._golden_dispatch()     → impl/golden.cpu_qfa_*()
        │                             (role switches to bench under golden_mode=Promote)
        │
xpu     → spec._xpu_reference()       → impl/golden.qfa_sectioned_reference()
        │                             (fixed-section quantization, native precision, no promotion)
        │
compare → spec.compare（_CompareHook）
             ├─ --compare cross_check: resolves to None, framework CrossCheckComparison runs three-way
             └─ default two-way: impl/compare.compare() judges each output by CSV precision thresholds
```

`spec.third_party = {"torch": _xpu_reference}` declares the XPU reference
implementation. The function body lazily imports `impl/golden.py` so the main
process does not pull up torch_npu early; for remote execution, TTK syncs the
spec file and the `impl/` directory to xpu_server.

third_party must be a **sectioned quantized** reference and must not reuse the
global exact golden: the denominator of cross_check is `|third_party - bench|`,
which tends to zero when both come from the same source, making the ratio blow
up. Only a sectioned reference with comparable P quantization noise provides a
valid noise floor.

## Reference implementation mapping

`impl/golden.py` selects the role through `_resolve_reference_mode()`:

- explicit `_reference` (the third_party entry injects `sectioned`);
- otherwise `--golden-mode Promote` or `--compare cross_check` → `bench`;
- otherwise → `golden` (two-way semantics).

| quant_mode | Mode | golden (two-way) | bench (three-way ground truth) | sectioned (XPU third_party) |
| --- | --- | --- | --- | --- |
| 1 / 2 | MXFP8 | `cpu_mxfp8_golden` | `cpu_mxfp8_bench` | `cpu_mxfp8_sectioned_golden` |
| 0 | HIF8 | `cpu_hif8_golden` (`cpu_hif8_scheduled_golden` for FD cases) | `cpu_hif8_bench(sectioned=False)` | `cpu_hif8_sectioned_golden` |
| 6 | GQA FP8 full quantization | `cpu_fp8_fullquant_golden` | `cpu_fp8_fullquant_bench(sectioned=False)` | `cpu_fp8_fullquant_bench(sectioned=True)` |
| 10 / 12 | HIFP4 / INT8 | two-way supported | three-way not supported | three-way not supported (`NotImplementedError`) |

## Generation and replay

### 0. Conventions

| Name | Value |
| --- | --- |
| ASSETS | `ops-transformer/attention/quant_flash_attn/tests/assets` |
| QFA_PYTEST | `ops-transformer/attention/quant_flash_attn/tests/pytest/qfa_mxfp8_test` |
| RUN | `qfa_run/` (working directory for generation and replay artifacts) |
| XPU port | `127.0.0.1:9100` |

`--plugin` takes the **ASSETS directory** (which contains `spec.py`), not a .py file.

### 1. Environment setup (do this in every new terminal)

Activate the Python environment with TTK and the PyTorch extensions installed,
and change to the parent directory of `ops-transformer`. Set `CANN_INSTALL_DIR`
to the local CANN installation directory and `QFA_OPP_DIR` to the vendor
directory of the installed QuantFlashAttn custom operator package:

```bash
source "${CANN_INSTALL_DIR}/set_env.sh"
export ASCEND_CUSTOM_OPP_PATH="${QFA_OPP_DIR}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

ASSETS="$PWD/ops-transformer/attention/quant_flash_attn/tests/assets"
QFA_PYTEST="$PWD/ops-transformer/attention/quant_flash_attn/tests/pytest/qfa_mxfp8_test"
RUN="$PWD/qfa_run"
mkdir -p "$RUN"
```

If `ninja` / `c++` are not on PATH, add them first (they are required to build
the JIT extension used by the CPU reference):

```bash
export PATH="<conda-env>/bin:$PATH"
```

### 2. Terminal A: start the XPU three-way service (keep it running)

```bash
python3 -m ttk.remote.server.xpu_server --port 9100
```

Verify:

```bash
ss -lntp | grep 9100      # should show LISTEN 127.0.0.1:9100
```

xpu is not another accelerator card; it is the sectioned quantized reference
running on the local CPU (`spec._xpu_reference` → `qfa_sectioned_reference`),
and TTK fetches data through this port.

### 3. Configuration file

`$RUN/ttk_remote_demo.yaml`:

```yaml
remote:
  endpoints:
    - host: 127.0.0.1
      port: 9100
```

`--config` is deep-merged with the `default.yaml` shipped in the TTK package;
this minimal yaml is equivalent to "default config + three-way endpoint". Do
not pass only `default.yaml`: its `endpoints` are commented out and will not
connect the third party.

### 4. Generate the case CSV

```bash
cd "$QFA_PYTEST"
python3 pytest_paramset_to_csv.py \
  --paramset quant_flash_attn_paramset_debug \
  --output "$RUN/qfa_mxfp8.csv"
```

The converter guarantees two preconditions (do not break them when changing the script):

- metadata slots are written as `(-1, -1)`: dynamic slots generated and filled
  back by `impl/npu_preprocess.py` before the main operator call;
- for TND cases, `cu_seqlens_q` is the packed prefix sum of `seqused_q`
  (`cu[-1] == sum(seqused_q)`), and the `batch_size` attribute is `None`
  (`npu_preprocess` forces `batch_size=None` for varlen TND/NTD, otherwise the
  metadata call fails).

Existing inventory CSVs (including redline / profiling converted cases) can also
be used directly as `-i` input.

### 5. Terminal B: generation phase (inputs + promoted golden + XPU in one pass)

```bash
cd "$RUN"
ttk e2e -i qfa_mxfp8.csv -o mxfp8_gen.csv \
  --plugin "$ASSETS" \
  --config "$RUN/ttk_remote_demo.yaml" \
  --no-prof --golden-mode Promote --dump in,golden,xpu \
  --manual-data-dirs "$RUN/manual_mxfp8"
```

You can also run the pieces separately:

- CPU promoted ground truth only: `--golden-mode Promote --dump in,golden`
- XPU sectioned reference only (native precision, no promotion): `--dump in,xpu`
- a single case: add `--ti=0` (0-based case index, not a device id)

Files under each case directory after a successful run:

| File | Meaning |
| --- | --- |
| `input_0..17_*.bin` | the operator's 18 input slots; `*_none.bin` are optional/dynamic slots (metadata, attn_mask, v_tail, etc.), 0 bytes when unused |
| `golden_0_float32_*.bin` | promoted CPU golden main output (FP32, three-way ground truth) |
| `golden_1_float32_*.bin` | LSE golden; absent when LSE is disabled (golden has only 1 output) |
| `xpu_torch_0_bfloat16_*.bin` | XPU sectioned reference main output (bf16) |
| `xpu_torch_1_float32_shape_0.bin` | empty placeholder from the third_party entry when LSE is disabled (0 bytes); fp32 LSE when enabled |

### 6. Terminal B: replay phase (run the NPU and perform three-way comparison)

```bash
cd "$RUN"
ttk e2e -i qfa_mxfp8.csv -o mxfp8_cross.csv \
  --plugin "$ASSETS" \
  --manual-data-dirs "$RUN/manual_mxfp8" \
  --compare cross_check
```

Note: **do not pass `--no-prof` during replay**; it conflicts with
`--manual-data-dirs` and reports
`--no-prof manual-data output requires --dump in`.

### 7. Reading results

- `precision_status`: overall PASS / FAIL verdict
- `precision_metrics`: per-output `standard` / `level` / `config` / `result` / `pass`
- `xpu_metrics`: three-way data source (should be `manual_binary` / `manual_xpu_binaries` / PASS)
- `*_compare_detail.csv`: per-output details (`testcase_name, run_mode, output_idx, output_dtype,
  standard, level, pass, mare, mere, rmse, small_err_cnt_target, small_err_cnt_third, ...`)

## Criterion

The framework `CrossCheckComparison` processes each output channel (atten bf16 + lse fp32):

1. all three are promoted to higher precision and shapes must match;
2. large-value region `|bench| >= small_value`:
   - `mare = max|target-bench| / max|third-bench|`
   - `mere = mean|target-bench| / mean|third-bench|`
   - `rmse = sqrt(mean((target-bench)^2)) / sqrt(mean((third-bench)^2))`
3. small-value region `|bench| < small_value`: count points with `|x-bench| > small_value_atol`,
   `count_target / max(count_third, 1) <= 2`;
4. NaN/Inf follow the framework's built-in three-way consistency rules;
5. division denominator protection and numerator/denominator lower bounds are internal to TTK.

## Thresholds

Standard `cross_check`, level **L2** (declared explicitly as `"level": "L2"` in
the `tolerance` of `spec.py`; TTK defaults to L1 when unspecified):

```
mare_ratio <= 2.0, mere_ratio <= 1.2, rmse_ratio <= 1.2
```

Small-region thresholds are looked up from the TTK default table by **output
dtype** (`ttk/core_modules/common/comparison/resolve.py`):

| Output dtype | small_value | small_value_atol | Small-region criterion |
| --- | --- | --- | --- |
| float16 | `2^-11` | `2^-16` | `err_target / max(err_third, 1) <= 2` |
| bfloat16 | `2^-8` | `2^-16` | same as above |
| float32 (LSE) | `2^-14` | `2^-30` | same as above |

The `tolerance` in `spec.py` also keeps the two-way `rtol/ptol/atol`; the
default two-way comparison still goes through `impl/compare.py` and is not
affected by `level`.

## Notes

- **Generation must use `--golden-mode Promote`**: it both selects the bench
  role and promotes golden outputs to FP32. Promote lifts e8m0 inputs to
  float32, and `_to_torch_e8m0` in `impl/golden.py` restores the e8m0 encoding
  via `round(log2(v)) + 127` (NaN keeps 0xFF); do not bypass it when changing
  the generation flow.
- **Do not pass `--no-prof` during replay**: see section 6.
- **`batch_size` for TND**: varlen TND/NTD must be `None`, handled uniformly by
  `npu_preprocess.py`; do not hard-code batch_size for TND in the CSV.
- **When LSE is disabled**: the operator returns only 1 output; the third_party
  entry pads an empty tensor slot so that `--dump xpu` and manual replay have
  the same number of slots.
- **dtype alignment**: bench/golden stay FP32; the third_party main output is
  cast to bf16 before comparison; LSE stays FP32. Do not promote the
  third_party, otherwise the noise floor disappears.
- **quant_mode 10/12** (HIFP4 / INT8) do not support three-way yet;
  `qfa_sectioned_reference` raises `NotImplementedError` directly.
