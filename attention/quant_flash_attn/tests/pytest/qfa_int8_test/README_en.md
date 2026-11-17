# QFA Full INT8 Quantization Tests

This suite follows the parameter sets, shared runner, functional and performance entry points, and cache stages of
`qfa_fp8_test`. INT8 uses mode 12. Coverage includes BNSD, BSND, TND, mask modes 0 and 3, metadata, sliding-window
attention (SWA), LSE, and boundary cases for head dimensions from 1 to 256.

```text
qfa_int8_test/
├── conftest.py
├── common/
│   ├── quant_flash_attn_int8_golden.py  # Reference implementation shared with tests/assets
│   └── test_runner.py                 # gen/cpu/npu/compare
├── quant_flash_attn_int8_paramset_common.py
├── quant_flash_attn_int8_paramset_func_rdv.py
├── quant_flash_attn_int8_paramset_perf_rdv.py
├── test_quant_flash_attn_int8_func_rdv.py
├── test_quant_flash_attn_int8_perf_rdv.py
└── test_quant_flash_attn_int8_contract.py  # API, metadata, graph replay, and TestSpec
```

Install the current operator and Torch extension. Set up CANN, PyTorch with torch-npu, pytest, and ATK.
The tests import `cann_ops_transformer.ops` by default. For a custom build, set
`QFA_TEST_OP_PACKAGE=cann_ops_transformer_custom`. Select a device with `QFA_TEST_NPU_DEVICE`.

Run these commands from this directory.

```bash
pytest -q -m ci                                      # Representative functional cases
pytest -q -m contract                                # API, metadata, graph replay, and TestSpec
pytest test_quant_flash_attn_int8_func_rdv.py --golden-mode=gen,cpu --cache-dir=/tmp/qfa_int8_cache
pytest test_quant_flash_attn_int8_func_rdv.py --golden-mode=npu,compare --cache-dir=/tmp/qfa_int8_cache
pytest test_quant_flash_attn_int8_perf_rdv.py --msprof --cache-dir=/tmp/qfa_int8_cache
pytest --parse-prof=/path/to/PROF_xxx --perf-baseline=/path/to/perf_report.log
```

The default `--golden-mode=all` runs gen, cpu, npu, and compare in order. Each case stores
`{name}_input.pt`, `{name}_cpu_output.pt`, and `{name}_npu_output.pt` and validates the parameters when loading the cache.
Cache stages apply to the functional and performance entry points. Contract tests run independently.

The functional parameter set includes representative regression cases for scales, layouts, tiling and padding, long
sequences, and empty batch members. The shared runner checks LSE toggle consistency, dtype, shape, and non-finite values.
Performance tests cover all model head-count combinations. Metadata is generated before timing. Each performance
case runs 5 warmups and 10 timed iterations. Long-sequence tests check the full output for non-finite values and
compare accuracy on selected query rows.

The performance report uses the FP8 parser to list the `QuantFlashAttn` kernel for warmup and timed runs.
End-to-end timing also includes kernels for Torch-side padding and other internal operations.

Accuracy follows the ATK double-reference standard, using an independent FP64 reference and an FP16 numerical
reference. For BF16 output, the maximum, average, and RMSE relative error ratio limits are 10, 2, and 2.
LSE is compared against independent FP64 logsumexp with `rtol=5e-4` and `atol=3e-3`.
Fully masked rows must return `-inf` for LSE.

`tests/assets/spec.py` also supports mode 12 and shares the same INT8 data generation and references.
Six contract cases cover input generation, accuracy comparison, metadata, optional LSE, NumPy BF16, and graph
binding without duplicating the functional accuracy matrix.

Host tiling, output inference, and AICPU unit tests cover validation and scheduling independently in `tests/ut`.
