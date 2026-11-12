# FlashKDA TTK tests

[中文](README.md) · [Operator guide](../../docs/FlashKDA_en.md)

The suite uses the native [ops-test-kit](https://gitcode.com/cann/ops-test-kit) E2E CSV format and TestSpec. Each accuracy case calls both forward DSL entries on the NPU with the same inputs, metadata and mode. pytest has a separate [forward test suite](../ut/test_flash_kda.py).

## Files and entry points

| File | Role |
|---|---|
| `kda_ttk_ops.py` | TTK adapter, source loading, metadata preparation, training/inference context and exception checks |
| `assets/inputs.py` | Input generation, input mutations and independent ABI 4 metadata fixtures |
| `assets/golden.py` | Two CPU references and `FlashKdaTestSpec` |
| `assets/compare.py` | Return arity, extraction of valid regions, numerical and cache checks |

| CSV under `cases/` | Scope | Cases |
|---|---|---:|
| `ttk_cases_stc.csv` | Functional coverage including Meta outputs and NPU/Meta strides | 154 |
| `ttk_cases_invalid.csv` | Invalid inputs, valid counterexamples and recovery | 267 |
| `ttk_cases_generalized.csv` | Generalization | 70 |
| `ttk_cases_profiling.csv` | Performance | 25 |
| `ttk_cases_redline_source_draft.csv` | Additional redline training and inference specifications (draft) | 20 |

The first three CSVs comprise 491 ordinary regression cases. Profiling and the redline draft are run separately. The CSV `api_name` is `kda_ttk_ops.flash_kda`. The adapter loads the DSL implementation from `experimental/attention/flash_kda/op_dsl` in the source tree.

## Inputs, outputs and caches

The tensor order is `q, k, v, g, beta, metadata, initial_state, A_log, dt_bias, cu_seqlens, seqused`. Raw Q/K, g and beta logits are consumed with the three fused operations enabled by default. Fused gate activation requires `A_log`. A missing `lower_bound` selects softplus and a bound in `[-5,0)` selects bounded sigmoid. The default scale is `128 ** -0.5`. A scale of zero is retained.

The input generator takes shapes, dtypes and attributes from TTK. Redline cases use the CSV's `input_data_ranges`. The adapter derives the batch, head counts and sequence length from the input tensors and layout. It passes these sizes to `flash_kda_metadata` with `enable_grad=not inference`. The `seqlens` argument belongs to the metadata producer. The sixth shape in the CSV must match the actual one-dimensional metadata tensor for that case. `customize_inputs` checks each generated shape and dtype against the TTK allocation before copying it.

The adapter and both DSL forward entries annotate `metadata` as `Optional[torch.Tensor]` without a default. Valid calls supply the schedule tensor. Omitting the parameter from a direct DSL call raises `TypeError`, and passing `None` raises `ValueError`. TND uses cumulative sequence offsets. BNSD and BSND may use per-sequence lengths for padded inputs. Inference requires `torch.inference_mode()`. The suite also covers `torch.no_grad()` on its own.

CSV attributes including `valid_lengths`, `initial_state_mode`, `dt_bias_mode` and `seed` control generation. `omitted_attributes` removes a keyword from the DUT call to exercise its default. `test_config` supplies JSON recipes for mutations, schedule corruption, white-box probes and additional comparisons.

The adapter checks the return arity before selecting valid output regions. `flash_kda` always returns three slots, and `flash_kda_fwd` always returns sixteen. The first three are `out, final_state, chunk_states`. Final state follows `output_final_state`. Incoming chunk states are returned only in inference when requested. The remaining 13 slots are `None` during inference.

Training exposes the two matrices and any enabled normalization and beta caches through `flash_kda_fwd`. Setting `disable_recompute=True` also exposes `w/u/qg/kg/v_new/g_cumsum`; the default is `False`. Disabled slots retain their position. The adapter checks valid matrix rows/columns, tokens and chunks, exact agreement between the two entries on their first three valid results, and input immutability.

## Golden, peer and acceptance criteria

The references follow the chunk decomposition and three-stage recurrence used by the [pytest implementation](../ut/test_flash_kda.py). `cpu_chunk` computes the FP32 mathematical reference. `npu_chunk` simulates device rounding on the CPU and is registered as the `torch` peer. In the persistent route, the peer retains FP32 `U_pre` when adding the state product. In the staged route it rounds `U_pre` to BF16 before the addition. Both routes expose a BF16 `u` cache. The bounded-gate reference uses a common origin for each group of 16 rows. Unbounded or preprocessed gates use row-wise causal differences to avoid overflow in the unused upper triangle.

TTK applies its native `cross_check` at level `L2`. It compares DUT and peer errors relative to the golden reference, using MARE, MERE and RMSE ratio limits of 2.0, 1.2 and 1.2 and the framework's small-value rules. Enabled output slots retain their actual dtype. The adapter also checks shape, dtype, finite values and training-cache tolerances. Q/K normalization uses `atol=rtol=1e-3`. Reciprocal standard deviations and activated beta use `1e-6`. Inactive slots are recorded as `NOT_APPLICABLE`.

The adapter supplies TTK with the active regions of the full forward result and one boolean covering both entries' checks. Inference compares the first three slots. Training checks the enabled slots among all sixteen. Functional comparisons, segmented state handoff and recovery calls use the same L2 comparator. Exception cases check the expected error, the point at which it is raised and successful recovery with valid input. Legal view and stride cases inspect the named arguments of `FlashKDAFwd.run_flash_kda`.

## Profiling cases

The performance CSV contains 9 inference and 16 training cases. Both use fused Q/K normalization, gate and beta sigmoid, `lower_bound=-5.0`, the default head-dimension scale, `allow_neg_eigval=False`, `return_intermediate_states=False` and `disable_recompute=True`. Inference enables final state and uses BF16 g/beta. Training disables final state and uses FP32 g/beta. Both use a zero FP32 initial state and normally distributed FP32 `dt_bias` drawn from an independent CPU generator seeded with 101.

Only `flash_kda` is timed. The profiling CSV sets `golden_api=disable`, 10 DUT warmups and 50 samples. Metadata production is timed separately during TTK's excluded warmup. AICore task durations come from the profiler. Host validation, forward-call and synchronization times are recorded separately.

## Environment and execution

Use a Linux environment with CANN, matching PyTorch and torch_npu, CANNBotDSL, TTK and its CPU peer service. Starting from the ops-transformer repository root, create a separate working directory for the test results.

```bash
export KDA_SOURCE_ROOT="$(pwd)"
export KDA_TTK="$KDA_SOURCE_ROOT/experimental/attention/flash_kda/tests/ttk"
export PYTHONPATH="$KDA_TTK:${PYTHONPATH:-}"
export KDA_OUT="$(mktemp -d)"
export KDA_EVIDENCE_DIR="$KDA_OUT/evidence"
cd "$KDA_OUT"
```

Save the following worker configuration as `$KDA_OUT/cpu_reference.yaml`. Both the worker and the CPU peer need access to the same adapter and reference modules. In another terminal using the same environment and `PYTHONPATH`, start the CPU service on the configured port.

```yaml
remote:
  endpoints:
    - host: 127.0.0.1
      port: 9090
```

```bash
python -m xpu_server.xpu_server --devices cpu --bind 127.0.0.1 --port 9090
```

Run the following commands in the worker terminal. Validate the CSV and interface signature, then select one functional case by its exact `testcase_name` to run on the NPU.

```bash
python -m ttk e2e -i "$KDA_TTK/cases/ttk_cases_stc.csv" \
    --plugin "$KDA_TTK/assets/golden.py" --validate

python -m ttk e2e -i "$KDA_TTK/cases/ttk_cases_stc.csv" \
    --plugin "$KDA_TTK/assets/golden.py" --config "$KDA_OUT/cpu_reference.yaml" \
    -t F01-001 --pc 1 --device-whitelist 0 --run 1 --warmup false \
    --task-prof=false -o "$KDA_OUT/stc_single.csv"
```

Run each ordinary regression suite into its own output CSV.

```bash
for suite in stc invalid generalized; do
    python -m ttk e2e -i "$KDA_TTK/cases/ttk_cases_${suite}.csv" \
        --plugin "$KDA_TTK/assets/golden.py" --config "$KDA_OUT/cpu_reference.yaml" \
        --pc 1 --device-whitelist 0 --run 1 --warmup false --task-prof=false \
        -o "$KDA_OUT/${suite}_results.csv"
done
```

If the redline draft is present, select a training or inference case by name. For long sequences, allow sufficient time for the worker and the CPU peer's request, dispatch and execution deadlines.

```bash
python -m ttk e2e -i "$KDA_TTK/cases/ttk_cases_redline_source_draft.csv" \
    --plugin "$KDA_TTK/assets/golden.py" --config "$KDA_OUT/cpu_reference.yaml" \
    -t Prefill_KDA_6_38_38_1024_1024_128_L0_18_train \
    --pc 1 --device-whitelist 0 --run 1 --warmup false --task-prof=false \
    --proc-timeout 28800 -o "$KDA_OUT/redline_single.csv"
```

Run profiling separately.

```bash
python -m ttk e2e -i "$KDA_TTK/cases/ttk_cases_profiling.csv" \
    --plugin "$KDA_TTK/assets/golden.py" --pc 1 --device-whitelist 0 \
    --run 50 --warmup true -o "$KDA_OUT/performance.csv"
```

## Reading results and troubleshooting

- Find the row by `testcase_name`. For an accuracy run, check for normal process exit, `precision_status=PASS` and `torch.status=PASS` in `xpu_metrics`. See `precision_metrics` for numerical results.
- The per-case JSONL records under `KDA_EVIDENCE_DIR` identify the entry, output slot and metric. Inspect `l2_reference_check`, `numerical_contract_check` and `entry_consistency_checked`, then compare the active layout, lengths and switches with the CSV.
- A shape/dtype error in `customize_inputs` identifies a mismatch between the allocated and generated tensor. Check the sixth shape for metadata errors. For a missing CPU result, inspect the peer process, port, plugin version and timeout logs.
