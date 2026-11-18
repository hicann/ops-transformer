# Mega TTK suites

The canonical CSVs use `mega_recurrent_kda` as `api_name`. Before running TTK, export a temporary execution CSV with `--for-ttk` to select plugin entrypoints by expected outcome. Default exports retain the operator name.

[简体中文](README.md)

TTK uses the CSV catalogues for each architecture. The default [arch35](arch35/) catalogue contains 1,241 distinct Mega cases.

| Suite | Cases |
|---|---:|
| Interception | 126 |
| Redline | 7 |
| Generalization | 960 |
| STC | 46 |
| Performance shapes | 102 |

The cases cover BSH and TH, S4/S8 and variable-length routing, paged storage and strides, stream quotas, invalid-input atomicity and graph capture. Performance shapes use BSH and 32 AIC for TP4, TP8 and TP16 and model batches 1 through 16 and 32.

Each row uses standard TTK tensor declarations for view shape, dtype, input range, backing
storage shape and view stride. Tensor slots follow the signature of `kda_ttk_ops.verify_csv`.
The `cu_seqlens` slot is `None` for BSH. The row's `attributes.case_config` stores the seed,
scalar values, PA mapping, core quota and design metadata, including legacy selection aliases.
Keep the tensor declarations and configuration in sync when changing a case.
TTK allocates the real numeric inputs, and the input hook populates them with the original seeded
data and checks their views. Interception and graph capture retain a standard control tensor and
build the corresponding contract scenario from the same row.
The `tensor_formats` column declares ND or FRACTAL_NZ. TTK casts the declared weights on-device.
`input_data_ranges` declares allowed numeric bounds. The input hook reads distribution parameters from `attributes.case_config.input_specs` and checks the generated values against those bounds.

### arch92 catalogue

Set `KDA_ARCH=arch92` to select the [arch92](arch92/) Mega-only cases. The catalogue includes 7 redline, 126 interception, 65 STC, 1,163 generalization and 102 performance shapes. arch92 uses one AIV per AIC, supports 8/16/24/32/64 AIC and routes S4 and S8 through the general path.

Build and install the custom operator package first, then build and install the Torch extension wheel.
Set `MKDA_TORCH_PACKAGE` to the installed package (for example `cann_ops_transformer_custom` for vendor custom), source the custom package's `bin/set_env.bash`, and require `CANNBOTDSL_NATIVE_BINARY_MODE=require`.
The test adapter uses the installed Torch dispatcher without a checkout-source fallback.

TTK selects the implementation using `KDA_ARCH` and defaults to `arch35`. Validate the selected catalogue and plugin with:

```bash
export PYTHONPATH="$PWD/experimental/attention/mega_recurrent_kda/tests:${PYTHONPATH:-}"
export MKDA_RUN_DIR=/tmp/mega_kda_ttk_run
export MKDA_TORCH_PACKAGE=cann_ops_transformer_custom
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --suite stc --for-ttk --output /tmp/mega_kda_stc_runtime.csv
python -B -m ttk e2e -i /tmp/mega_kda_stc_runtime.csv --plugin experimental/attention/mega_recurrent_kda/tests/kda_ttk_ops.py --validate
```

Numeric comparison retains exactly three implementations: the installed operator output, the CPU FP32 golden in `assets/golden.py`, and the independent NPU small-op reference in `assets/npu_peer.py`.
Configure the third-party service with an NPU backend and device. Its output-device assertions reject CPU substitution.
Keep the existing `cross_check/L2` thresholds for all five outputs. Ordinary FP32 accumulation, BF16 Fa/Fb rounding boundaries and replay dtypes remain intact; extra compensated dot products and Conv/SiLU residual correction are removed.
Use `--seed 3 --warmup=false --task-prof=false` for precision-only runs. The UT entry remains in `tests/ut/test_mega_recurrent_kda_ut.py`; all original `pytest_smoke` cases are retained and share the CPU FP32 golden.

Export existing cases by suite or ID:

```bash
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --suite interception --output /tmp/mega_kda_interception.csv
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --ids I-015 F-012 --output /tmp/mega_kda_selected.csv
```

Keep logs, results, input manifests and exported subsets outside the source checkout.

## Input distributions and backing storage

Each CSV stores the distribution, seed and bounds used by its input helper. `clipped_normal` draws normal samples with the declared mean and standard deviation, then clamps to the declared bounds; it is neither uniform sampling nor rejection-sampled truncation.

| Input | Standard deviation | Allowed range |
|---|---:|---|
| hidden_states, projection/Conv weights and Conv state | 0.02 | [-0.25, 0.25] |
| checkpoint_state | 0.002 | [-0.03125, 0.03125] |
| output_norm_weight | 0.2 | [-2, 2] |
| A_log and dt_bias | 0.1 | [-1, 1] |

Zero-input cases declare their constant bounds. Replay pools use records_fill, indices and accepted lengths use explicit case values, and TH cu_seqlens uses prefix sums of lengths. A control-tensor row retains range (0,0) for that control tensor; its payload generation bounds remain in input_specs. Invalid-input mutations are applied after baseline generation.

PERF-S4-TP4-MB01 has logical replay shapes [6,24,4,128] and backing shapes [12,24,4,128], [18,24,4,128] and [24,24,4,128]. Their axis-0 strides span 2, 3 and 4 contiguous slots. Both CPU and NPU allocate the complete declared backing and construct the strided views. Guards cover unselected blocks, inter-block gaps and the backing tail after the final logical block.

The CPU FP32 and torch_npu references select independent stage calculations using `KDA_ARCH`; neither consumes operator outputs or intermediates. Fa uses the complete K reduction without four-segment merging or precision compensation.

- arch35 retains its existing aligned path: QKV/Fa/Fb/Beta output tiles, padding, BF16 boundaries and explicit Conv/activation steps follow its schedule. These NPU projection tiles use full-K mm on 1 AIC/2 AIV; other small operators use the case quota.
- arch92 follows the variable-length stages used by its public entry. All projections use 128-by-128 output tiles, globally pad M to a multiple of 128 and pad Beta output columns to 128. Each tile accumulates the complete K dimension before its declared storage cast. Each NPU reference tile uses 1 AIC/1 AIV.
- arch92 Vector steps align the paired Conv multiply-adds, explicit SiLU/gates, two 64-element reduction segments, query scaling before reduction, state dot products after `decay*K`, state update multiply-adds and output RMSNorm order. Gate and output projections also use global M tiles rather than restarting GEMM per sequence.

This serial accuracy reference is not a parallel performance baseline. CPU and NPU retain their own reduction and transcendental implementations. Stage alignment does not guarantee bitwise equality or replace precision validation on the target architecture.

Numeric rows use input_data_ranges as the single source for bounds and input_specs for distribution parameters. Execution exports inject those bounds into case_config for the input hook. Control-tensor rows keep payload bounds in input_specs.
