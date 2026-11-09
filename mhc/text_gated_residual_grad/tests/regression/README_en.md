# TextGatedResidualGrad regression tests

Run the workspace arithmetic boundary test (no device allocation):

```bash
c++ -std=c++17 test_workspace_layout.cpp -o /tmp/tgrg-workspace
/tmp/tgrg-workspace
```

Run the host credit-protocol test from this directory (requires a C++17 compiler):

```bash
c++ -std=c++17 -DHC_GRAD_NPU_ARCH=HC_GRAD_ARCH_A5 -Istubs test_credit_sync.cpp -o /tmp/tgrg-credit-a5
/tmp/tgrg-credit-a5
c++ -std=c++17 -DHC_GRAD_NPU_ARCH=HC_GRAD_ARCH_A2 -Istubs test_credit_sync.cpp -o /tmp/tgrg-credit-a2
/tmp/tgrg-credit-a2
```

The test executes the production scheduler and credit-sync class, checks both AIV
subblocks, rejects waits without an earlier producer, checks reserved flag IDs,
and requires all credit counters to be zero at exit. It models event counts,
not hardware pipe timing. The A2 build checks the shared credit helper's mode-2
semantics on the default direct-read schedule; it does not validate hardware execution.

Run the host FP32 cancellation test (extracts the production methods, requires Python 3 and C++17):

```bash
python3 test_dgamma_host.py
# Compile the extracted production methods without running them:
python3 test_dgamma_host.py --compile-only
```

Build the package and its ACLNN example, activate that package's environment, then
run the exact BF16 cancellation regression on a supported Ascend device (requires numpy):

```bash
python3 test_dgamma_cancellation.py --runner /path/to/test_aclnn_text_gated_residual_grad --device 4
```

This case requires at least two AIV lanes per channel. The inputs put +2**24 and +1 in separate 1024-row tasks on lane 0,
with -2**24 on another lane. The original Kahan merge returns -1 for dgamma;
the corrected merge returns exactly +1. The runner is the
operator's existing `examples/test_aclnn_text_gated_residual_grad.cpp` binary.

The output-buffer lifetime regression uses an exact all-one/zero input. On the
56-AIV A5 tiling it exposes V6 clearing a V1 output before MTE3 finishes:

```bash
python3 test_pipeline_output_reuse.py --runner /path/to/test_aclnn_text_gated_residual_grad --device 4
```

The expected `dw_gate` is exactly 768. Reproducing the old race depends on tiling;
the expected output is valid regardless of core count. Both hardware regressions
are targeted diagnostics; use the project's independent benchmark for acceptance.

Run the ACLNN optional-pair and result-pointer checks after building the package.
Set `CUSTOM_OPP_VENDOR` to its `packages/vendors/custom_transformer` directory
and source the CANN environment first:

```bash
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_VENDOR"
export LD_LIBRARY_PATH="$CUSTOM_OPP_VENDOR/op_api/lib:$LD_LIBRARY_PATH"
c++ -std=c++17 test_optional_outputs.cpp \
  -I "$CUSTOM_OPP_VENDOR/op_api/include" -I "$ASCEND_HOME_PATH/include" \
  -L "$CUSTOM_OPP_VENDOR/op_api/lib" -L "$ASCEND_HOME_PATH/lib64" \
  -lcust_opapi -lascendcl -lnnopbase -o /tmp/tgrg-optional
/tmp/tgrg-optional 0
```

These 20 query-only checks cover all valid optional combinations, both directions
of the wGate/dwGate and gamma/dgamma pairs, and null workspaceSize/executor
pointers for FP16 and BF16. No kernel is launched.

The `rms_cancellation_cases.json` fixture preserves the failing ATK seed 2026110550
(BF16, N=1944, C=2, H=1280, R=160, gamma present, gate absent). With saturated
sigmoid inputs, the old RMS evaluation produced 6 small-domain errors in dx
against 2 in the independent benchmark, failing the unchanged count-ratio limit
of 2. It also includes seed 2026111050 (BF16, N=4308, C=4, H=2560, R=320,
gamma and gate present), which exposed remaining reduction error after normalizing
the RMS projection: 16 small-domain errors against 1 in the benchmark.
Run these cases using the [independent benchmark's ATK adapter](https://gitcode.com/azyuli/text_gated_residual_grad_bench)
and its CPU FP64 oracle; `--disable_id_seed` is required to preserve the fixture's
seed. Keep both the benchmark and golden comparisons enabled. This targeted
fixture does not replace the wider randomized accuracy suite.
