# Mega TTK suites

The canonical CSVs use `mega_recurrent_kda` as `api_name`. Before running TTK, export a temporary execution CSV with `--for-ttk` to select plugin entrypoints by expected outcome. Default exports retain the operator name.

[简体中文](README.md)

Pytest and TTK share the CSV catalogues for each architecture. The default [arch35](arch35/) catalogue contains 1,235 distinct Mega cases.

| Suite | Cases |
|---|---:|
| Interception | 126 |
| Redline | 7 |
| Generalization | 960 |
| STC | 46 |
| Performance shapes | 96 |

The cases cover BSH and TH, S4/S8 and variable-length routing, paged storage and strides, stream quotas, invalid-input atomicity and graph capture. Performance shapes use BSH and 32 AIC for TP4, TP8 and TP16 and model batches 1 through 16.

Each row uses standard TTK tensor declarations for view shape, dtype, input range, backing
storage shape and view stride. Tensor slots follow the signature of `kda_ttk_ops.verify_csv`.
The `cu_seqlens` slot is `None` for BSH. The row's `attributes.case_config` stores the seed,
scalar values, PA mapping, core quota and design metadata, including legacy selection aliases.
Keep the tensor declarations and configuration in sync when changing a case.
TTK allocates the real numeric inputs, and the input hook populates them with the original seeded
data and checks their views. Interception and graph capture retain a standard control tensor and
build the corresponding contract scenario from the same row.
The `tensor_formats` column declares ND or FRACTAL_NZ. TTK casts the declared weights on-device.
A range of `(0, 0)` initializes the storage before the input hook fills it.

### arch92 catalogue

Set `KDA_ARCH=arch92` to select the [arch92](arch92/) Mega-only cases. The catalogue includes 7 redline, 126 interception, 65 STC, 1,163 generalization and 96 performance shapes. arch92 uses one AIV per AIC, supports 8/16/24/32/64 AIC and routes S4 and S8 through the general path.

Run from the repository root. Pytest selects 11 cases marked `pytest_smoke`.

```bash
python -B -m pytest -q experimental/attention/mega_recurrent_kda/tests/ut --basetemp=/tmp/mega_kda_pytest
```

TTK selects the implementation using `KDA_ARCH` and defaults to `arch35`. Validate the selected catalogue and plugin with:

```bash
export PYTHONPATH="$PWD/experimental/attention/mega_recurrent_kda/tests:${PYTHONPATH:-}"
export MKDA_RUN_DIR=/tmp/mega_kda_ttk_run
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --suite stc --for-ttk --output /tmp/mega_kda_stc_runtime.csv
python -B -m ttk e2e -i /tmp/mega_kda_stc_runtime.csv --plugin experimental/attention/mega_recurrent_kda/tests/kda_ttk_ops.py --validate
```

For NPU precision, configure a CPU third-party service. `assets/golden.py` provides both the FP64 reference and the original CPU implementation. Verify compares them with NPU results under `cross_check/L2`. Use `--seed 3 --warmup=false --task-prof=false` for precision-only runs. Pytest uses its own FP32 reference and tolerances.

Export existing cases by suite or ID:

```bash
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --suite interception --output /tmp/mega_kda_interception.csv
python -B experimental/attention/mega_recurrent_kda/tests/assets/inputs.py --ids I-015 F-012 --output /tmp/mega_kda_selected.csv
```

Keep logs, results, input manifests and exported subsets outside the source checkout.
