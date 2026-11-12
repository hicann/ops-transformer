# QuantFlashMla: Three-Way cross_check Comparison

Under full FP8 quantization, the P quantization paths of the CPU golden and the kernel differ (the kernel independently takes the running max and quantizes P per s2 segment derived from core partitioning). A direct two-way "NPU vs golden" criterion therefore treats the golden's own quantization noise as error and produces false failures.

The three-way comparison uses one high-precision CPU golden as the common baseline. The criterion is the ratio "NPU error against baseline / XPU reference error against baseline", i.e. the NPU error must not exceed a multiple of the reference noise.

| Role | Source | TTK slot |
| --- | --- | --- |
| target | Output of the quant_flash_mla operator under test | `output` |
| bench | CPU golden from `--golden-mode promote` (fp32 output) | `golden` (common baseline) |
| third_party | XPU (the **sectioned** CPU golden at native precision, not promoted) | `third_party` |

## Call Chain

```text
customize_inputs → inputs.generate_qfmla_inputs()   fixed-seed operator inputs
        │
        ├─ Operator execution               → outputs = [atten(bf16), lse(fp32)?]
        │
golden  → impl/golden.cpu_qfmla()            → CPU reference; promotion follows TTK golden_mode
        │
xpu     → spec._xpu_reference()              → remote xpu_server calls the sectioned
        │                                      CPU golden (fixed 256-column blocks,
        │                                      native precision, no promotion)
        │
compare → spec.compare()
             ├─ --compare cross_check: returns None; TTK CrossCheckComparison does the three-way check
             └─ default two-way: impl/compare.compare() applies CSV tolerances per output
```

`spec.third_party = {"torch": _xpu_reference}` declares the XPU reference (lazily importing
`impl/golden.py` inside the function so the main process never imports torch_npu early).
TTK syncs the spec file and the `impl/` directory to `xpu_server`.

third_party must be the **sectioned** reference (`fp8_fullquant_mla_sectioned_golden`,
fixed 256-column running max plus P quantization); it must not reuse the global exact
golden. The cross_check denominator is `|b-g|`; when the global reference and the promoted
golden share the same source that value collapses to zero and the ratio always blows up.
Only a sectioned reference carrying comparable P-quantization noise provides a meaningful
noise floor.

## Generation and Replay

### Prepare both terminals

Activate a Python environment with TTK and the PyTorch extension installed. In
each terminal, start from the parent directory of `ops-transformer`. Set
`CANN_INSTALL_DIR` to your CANN installation directory and `QFMLA_OPP_DIR` to the
vendor directory of your installed QFMLA custom operator package.

```bash
source "${CANN_INSTALL_DIR}/set_env.sh"
export ASCEND_CUSTOM_OPP_PATH="${QFMLA_OPP_DIR}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

ASSETS="$PWD/ops-transformer/attention/quant_flash_mla/tests/assets"
RUN="$PWD/qfmla_run"
mkdir -p "$RUN"
```

### Terminal A — Start the XPU service

```bash
python3 -m ttk.remote.server.xpu_server --port 9100
```

Keep this terminal open during data generation.

### Terminal B — Configure the endpoint and generate the case CSV

Save the following as `$RUN/ttk_remote_demo.yaml`. The bundled TTK defaults
leave `remote.endpoints` commented out. TTK merges the endpoint configuration
into its defaults when you pass this file with `--config`.

```yaml
remote:
  endpoints:
    - host: 127.0.0.1
      port: 9100
```

```bash
cd "$ASSETS/../pytest"
python3 pytest_paramset_to_csv.py --paramset qfmla_paramset_func_rdv \
  --output "$RUN/qfmla_func_rdv_dyn.csv"
```

Keep the generated metadata slot at `(-1, -1)` so the NPU preprocessing hook
can populate it before the operator runs. For packed TND inputs, the final
`cu_seqlens_q` value must equal the sum of `seqused_q` when the latter is supplied.

### Terminal B — Generate inputs and references

```bash
cd "$RUN"
ttk e2e -i qfmla_func_rdv_dyn.csv -o fp8_xpu_gen.csv \
  --plugin "$ASSETS" \
  --config "$RUN/ttk_remote_demo.yaml" \
  --no-prof --golden-mode Promote --dump in,golden,xpu \
  --manual-data-dirs "$RUN/manual_fp8"
```

Add `--ti=0` to select the first case. Each case directory stores
`input_*.bin`, promoted `golden_*.bin`, and `xpu_torch_*.bin` reference outputs.

### Terminal B — Replay on the NPU

```bash
cd "$RUN"
ttk e2e -i qfmla_func_rdv_dyn.csv -o fp8_cross.csv \
  --plugin "$ASSETS" \
  --manual-data-dirs "$RUN/manual_fp8" \
  --compare cross_check
```

Omit `--no-prof` during replay so TTK executes the NPU operator. The replay
reads the saved references directly from the manual-data directory.

### Check the results

- `precision_status` records the overall result.
- `precision_metrics` records the comparison standard, level, thresholds, and
  result for each output. Check for `cross_check` and `L2`.
- `xpu_metrics` records the reference source. For replay, look for
  `manual_xpu_binaries` with status `PASS`.

## Decision Method

TTK `CrossCheckComparison` handles each output channel separately (atten bf16 + lse fp32):

1. All three parts are upcast, and shapes must match;
2. Large value range `|bench| >= small_value`:
   - `mare = max|target-bench| / max|third-bench|`
   - `mere = mean|target-bench| / mean|third-bench|`
   - `rmse = sqrt(mean((target-bench)^2)) / sqrt(mean((third-bench)^2))`
3. Small value range `|bench| < small_value`: count points with `|x-bench| > small_value_atol`,
   require `count_target / count_third <= small_value_ratio`;
4. NaN/Inf follows the framework's built-in three-way consistency rules;
5. Denominator protection and lower bounds are implemented inside TTK.

## Thresholds

Standard `cross_check`, level **L2**, `mare/mere/rmse = 2.0/1.2/1.2` (declared as
`"level": "L2"` in `spec.py`'s `tolerance`):

| Output dtype | small_value | small_value_atol | small_value_ratio |
| --- | --- | --- | --- |
| float16 | `2^-10` | `2^-16` | 2 |
| bfloat16 | `2^-10` | `2^-16` | 2 |
| float32 (lse) | `2^-14` | `2^-30` | 2 |

The default two-way comparison still uses the CSV tolerances in `impl/compare.py` and is
unaffected by `level`.
