# flash_kda_grad × TTK
English | [简体中文](./README.md)

This directory invokes the FlashKDA backward operator through the E2E path of TTK (ops-test-kit), providing interface adaptation,
a CPU reference implementation, acceptance configurations, CSV assets, and run scripts. Unit tests, metadata/scheduling tests, and the
CPU reference implementation self-check are all placed in [`../ut/test_flash_kda_grad_ut.py`](../ut/test_flash_kda_grad_ut.py).
Run in the repository root directory:

```bash
python -m pytest -m 'not whitebox' attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py
python -m pytest attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py
```

The `tests/pytest.ini` of this operator only registers markers and adds no collection hooks or default filtering. Use the native pytest marker grouping directly; `-m 'not whitebox'` selects regular cases, `-m whitebox` selects only whitebox cases, and without `-m`
all cases are collected. When you need combined conditions, write them in the same `-m` expression.

To run only the CPU tests, add `-m 'not npu and not ascendc_toolchain'`; to check only the TTK
reference implementation, you can add `-k test_ttk_reference`. You still need to install the PyTorch and
CANNBotDSL dependencies of this test module; this file contains both NPU and toolchain tests, filtered by the original markers.

The frontend and lowering validation of the CPU whitebox share one compilation; 8 representative cases are retained, covering the masked, full, mixed, and staged paths
in both the saved-cache and recompute modes, plus the three dtype pairings of beta inputs/
gradients and two gate dtypes; all cross combinations of them are no longer compiled one by one.
The parameters and ABI of all 72 exported variants are still checked by enumeration in lightweight tests; the full AOT compilation check is still
performed by the tests marked with `ascendc_toolchain`.

The reference formulas, cache adaptation, and accuracy checks of the original `attention/flash_kda_grad/tests/test_flash_kda_grad.py`
have been moved into the preceding UT module, and the original file has been deleted. The standalone CLI entries are migrated accordingly, for example:

```bash
python attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py --profile --help
python attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py --compare-bin --help
```

The `api.py` and `spec.py` of TTK still use `flash_kda_grad_ttk/golden.py` in this directory,
and do not import the UT module. Test logs, profile results, and bin comparison reports should be written outside the code repository.

## 1. Files and Entries

```text
attention/flash_kda_grad/tests/ttk/
├── flash_kda_grad_ttk/
│   ├── api.py                      # npu(...): NPU operator / CPU reference dispatch
│   ├── golden.py                   # Forward cache, token-by-token golden, staged backward peer
│   ├── spec.py                     # golden / peer and the TTK default cross_check L2
│   └── __init__.py                 # Exports npu
├── cases/
│   ├── flash_kda_grad_stc.csv      # 130 STC cases, including 30 new recompute-specific cases
│   └── flash_kda_grad_exception.csv # Exception cases, maintained separately
└── run_one.sh                      # The only launch entry, generating a temporary peer configuration on demand
```

The entry under test is `flash_kda_grad_ttk.npu`, which accepts PyTorch tensors, so the E2E path is used.
The NPU branch only loads the operator in the adjacent `op_dsl` directory; an import failure reports an error directly, avoiding accidentally testing other installed versions.
The CPU branch calls the reference implementation.

## 2. Inputs, Outputs, and Caches

The input order of [`api.py`](flash_kda_grad_ttk/api.py) is as follows; the BSND layout swaps the S/N axes, and TND uses `[T,N,D]`:

| Order | Tensor | Shape (BNSD) | dtype |
|---|---|---|---|
| 1-2 | `q`, `k` | `[B,Nq,S,128]` | BF16 |
| 3 | `v` | `[B,Nv,S,128]` | BF16 |
| 4 | `g` | `[B,Nv,S,128]` | BF16 / FP32 |
| 5 | `beta` | `[B,Nv,S]` | BF16 / FP32 |
| 6 | `A_log` | `[Nv]` | FP32 |
| 7 | `dt_bias` | `[Nv,128]` | FP32 |
| 8 | `initial_state` | `[B,Nv,128,128]` | FP32 |
| 9 | `do` | `[B,Nv,S,128]` | BF16 |
| 10 | `dht` | `[B,Nv,128,128]` | FP32 |

The attributes include `scale`, `gate_mode` (bounded/softplus), `lower_bound`,
`allow_neg_eigval`, `layout` (BNSD/BSND/TND), `cache_source` (ref/fwd), and the three
`use_*_in_kernel` switches.

The output order is fixed as `dq, dk, dv, dh0, dg, dbeta, dA, dBias`.
Among them, `dh0/dA/dBias` are FP32, and the other gradients follow the dtype of the corresponding inputs.

By default, `cache_source=ref`: the adapter constructs the forward cache on the CPU, uploads it to the NPU, then generates the AICPU
metadata and calls the backward operator. The peer uses the same cache construction rules and BF16 rounding points.
`cache_source=fwd` instead calls the real forward operator to generate the cache, which is used to check the complete forward-backward chain;
in this case, you need to set `KDA_DSL_ROOT` to point to a CANNBotDSL repository containing a compatible `samples/flash_kda_fwd`,
and the results also include the impact of the forward implementation. By default, all 130 STC cases use `cache_source=ref` and do not depend on this external forward.

The mode is selected case by case through the `disable_recompute` attribute: `False` enables recomputation, and `True` consumes the saved
cache. Explicit attributes take precedence over environment variables; only when omitted or set to `None` does it fall back to `KDA_GRAD_RECOMPUTE`
read when the adapter is imported; when this environment variable is not set, the saved cache is used by default.

There are 130 STC cases in total, uniformly placed in `flash_kda_grad_stc.csv`, numbered consecutively 001-130,
following the existing TTK format; exception cases are still placed separately in `flash_kda_grad_exception.csv`.

- The parameters, data ranges, and accuracy settings of the original 96 cases remain unchanged.
- `kdag_stc_022/065/066/067` replace the original configurations whose CPU autograd overhead was too high with boundary cases;
  the numbers are retained, and the parameters are shown in the following table.
- New cases `kdag_stc_101` to `kdag_stc_130` are added, all
  designed for the recompute mode. They cover 10 cases for each of the three layouts, BF16/FP32 gate and beta combinations, and
  lengths of 63/64/65, 127/128/129, 1025, empty sequences, variable lengths, GVA, N=128, initial states,
  optional bias, and all eight fusion switch combinations. The `remark` is updated according to the latest unified recompute implementation, recording the input
  dtype, the numbers of full blocks/tail blocks, and the constraint that tail blocks must use staged; these fields are used to describe the coverage,
  and no runtime scheduling assertions are executed. When there is no tail block, the metadata still selects the scheduling, and the number of device cores is not fixed.

| ID | Layout | Length and physical storage | Nq / Nv | Mode | Valid token-heads |
|---|---|---|---|---|---|
| 022 | TND | B=31, `[0,1,63,64,65]` repeated six times, plus 129; total T=1287 | 1 / 2 | Saved cache | 2574 |
| 065 | BNSD | B=2, physical S=64, valid lengths `[17,0]` | 127 / 127 | Saved cache | 2159 |
| 066 | BSND | B=1, physical S=33, valid length `[31]` | 1 / 128 | Recompute | 3968 |
| 067 | TND | B=3, lengths `[511,512,513]`; total T=1536 | 1 / 1 | Recompute | 1536 |

The inputs of the four replaced cases have changed; you must regenerate the golden and cannot reuse the old caches corresponding to these numbers.

The new and replaced cases keep `D=128`, `Chunk=64`, `1 <= Nq <= Nv <= 128`, and satisfy
`B*S <= 1M` (TND is counted by the total number of tokens). To control the token-by-token autograd cost, `valid tokens * Nv` of each new
recompute-specific case does not exceed 2860, and that of the four replaced cases does not exceed 3968;
they do not replace independent large-scale performance or capacity tests.

The recompute cases directly control NPU recomputation through `attributes.disable_recompute=False`;
currently `api.py:npu()` has integrated and passes through this parameter, so no additional environment variable needs to be set.
The CPU golden and peer still use the original implementation and cross_check L2; you cannot relax the thresholds to make the new paths pass.

After activating the environment, execute from any directory; all 130 STC cases are read by default:

```bash
bash "$KDA_ASSETS/run_one.sh"
bash "$KDA_ASSETS/run_one.sh" -t kdag_stc_109
```

Each case selects the saved cache or recomputation according to the `disable_recompute` attribute of the CSV; explicit attributes are not
overridden by global environment variables. Results and manual-data caches of different modes should be distinguishable to avoid confusion.
To run only one case, add `-t kdag_stc_022` and similar case names at the end of the command; without filtering, all 130 cases are run.

### Recompute Exception Contract

`flash_kda_grad_exception.csv` separately maintains a catalog of 60 exception cases; the original 001-050 are unchanged;
051-060 add recompute-mode errors for initial state types, the last dimension, sequence counts, head counts, and device errors, plus
missing Aqk, Akk, q_rtsd, beta activation caches, and a wrong k_rstd dtype. All explicitly set
`disable_recompute=False`, and are expected to throw `ValueError` before recompute cache allocation and the main computation.
`initial_state=None` is a legal zero initial state; the seven recomputable caches can be omitted, and none of them is treated as an exception.

Ordinary TTK E2E does not interpret `remark.mutation`, and caches such as Aqk/Akk are constructed inside the adapter.
The actual mutations of the exception catalog and the expected exception assertions are executed by the existing CSV contract tests; you cannot treat
an ordinary E2E run failure directly as successful exception interception. Run the CPU contract tests from the operator repository root directory:

```bash
python -m pytest -q -s -m 'not npu' \
  attention/flash_kda_grad/tests/ut/test_flash_kda_grad_ut.py \
  -k 'exception_csv_host_contract or exception_csv_special_contract'
```

In a process that has already selected the target NPU, filtering `exception_csv_npu_contract` instead verifies Host interception with real NPU
tensors; the tests have allocation/computation sentinels, and illegal inputs fail as soon as they enter recompute allocation or computation preparation.
This part verifies parameter contracts and does not modify golden, numerical thresholds, the SDK, or the TTK framework.

## 3. Golden, Peer, and Criteria

Currently [`spec.py`](flash_kda_grad_ttk/spec.py) uses the following implementations:

| Role | Implementation | Computation method |
|---|---|---|
| golden | `golden.token_golden_backward` | FP32 token-by-token recursion with autograd differentiation; BF16 intermediate storage is not simulated; the spec converts the output dtype on return |
| peer | `golden.bf16_peer` | BF16 forward cache and matmul operands, explicitly backpropagated by the chunk formula |
| Under test | `flash_kda_grad_ttk.npu` | Calls the actual backward operator on NPU and the staged reference implementation on CPU |

`golden.py` also retains reference interfaces such as `emulated_autograd_backward`, but they are not
the golden entries of the current spec. The CPU branch under test and the peer share the staged backward implementation, so a CPU run
can be used to check the integration chain, but no NPU accuracy conclusion can be drawn from it.

All outputs uniformly use the `cross_check L2` built into the TTK framework; only the standard name and level are declared:

```python
tolerance = {
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
}
```

The thresholds and small-value-range parameters are all provided by the L2 preset of TTK, and the project does not override these parameters.
The default run, `l2`, and `replay` of `run_one.sh` all use this configuration and do not override accuracy thresholds.

`cross_check` compares the errors of the implementation under test and the peer relative to golden respectively; pay attention to
`mare_ratio`, `mere_ratio`, and `rmse_ratio`. L2 uses the ratio thresholds 2.0 / 1.2 / 1.2.
At runtime, `spec.py` is loaded through `--plugin`, and `--compare` is not passed to override its criteria.

## 4. Environment Preparation

Use the Conda/CANN environment already activated on the current machine; PyTorch, torch_npu,
CANNBotDSL and the TTK dependencies need to be installed in Python. The script does not switch Conda environments and does not source a fixed CANN path.

```bash
export KDA_ROOT=/path/to/ops-transformer/attention/flash_kda_grad
export KDA_ASSETS="$KDA_ROOT/tests/ttk"
# Set this when using the TTK source code; it can be omitted when ttk is already installed in the current Python environment.
export TTK_REPO=/path/to/ops-test-kit
export KDA_DEVICES=0
export KDA_MANUAL_DATA=/path/to/your/input_and_golden
export KDA_OUT=/path/to/your/results.csv
```

`KDA_ROOT` / `KDA_ASSETS` are derived from the script location by default; Python uses `python3` of the current environment by default,
which can be overridden by `KDA_PY` or the compatible variable `TTK_PY`. When `TTK_REPO` is not set, the script first looks for
`ops-test-kit` at the same level as the repository; otherwise it uses the TTK installed in the environment. Both the worker and the CPU peer inherit
the `PYTHONPATH` of the assets directory. It is recommended that results and manual data be placed in self-created directories to avoid writing into code directories.

## 5. The Only Run Entry

```bash
bash "$KDA_ASSETS/run_one.sh" --help
bash "$KDA_ASSETS/run_one.sh" validate       # CSV/interface validation, does not execute the operator
bash "$KDA_ASSETS/run_one.sh" list           # List cases
bash "$KDA_ASSETS/run_one.sh" -t kdag_stc_109 # Default cross_check L2
bash "$KDA_ASSETS/run_one.sh" exec -t kdag_stc_109   # Execute without comparing golden
bash "$KDA_ASSETS/run_one.sh" dump -t kdag_stc_109   # Generate inputs/golden in a new directory
bash "$KDA_ASSETS/run_one.sh" replay -t kdag_stc_109 # Read existing inputs/golden
KDA_PROF=true bash "$KDA_ASSETS/run_one.sh" replay -t kdag_stc_109
bash "$KDA_ASSETS/run_one.sh" xpu            # Start the CPU peer separately
```

`run` and `l2` are aliases of the default mode. Extra arguments such as `-t` are passed through to TTK; `KDA_CASE` can also filter
a single case. `dump` only prepares data and cannot serve as evidence that NPU accuracy passes; use `replay` for existing caches.
The four replaced cases must use new data consistent with the current CSV; you cannot apply golden of the old definitions.

| Variable | Default value and purpose |
|---|---|
| `KDA_CASES` | `cases/flash_kda_grad_stc.csv` |
| `KDA_OUT` | `kda_result.csv` in the invoking directory |
| `KDA_MANUAL_DATA` | `kda_manual_data` in the invoking directory, used for dump/replay |
| `KDA_CASE` | Case filter, equivalent to `-t` |
| `KDA_DEVICES` | `0`, passed to the `--device-whitelist` of the TTK client |
| `KDA_SEED` | `20261027` |
| `KDA_PC` / `KDA_RUN` | Number of processes per card / number of executions, both `1` |
| `KDA_PROF` | `false`; set it to `true` to enable performance collection |
| `KDA_XPU_PORT` | `9090`, the port of the local CPU peer |
| `KDA_XPU_LOG` | `$TMPDIR/kda_xpu_<port>.log`; `/tmp` is used when TMPDIR is not set |

The default run, `l2`, and `replay` check the peer port and automatically start the CPU service when it is not started. The service is kept,
and the script neither restarts nor terminates an existing service; when switching code directories, use a separate `KDA_XPU_PORT` to avoid reusing
the peer of another copy of the code. The client temporary YAML is cleaned up at the end of the script; a fixed configuration file is no longer maintained.

When you need one continuous batch-run log, you can save the complete output at the call site:

```bash
set -o pipefail
bash "$KDA_ASSETS/run_one.sh" replay 2>&1 | tee /path/to/your/batch.log
```

A remote peer can be used through native TTK commands and a self-created configuration; for example, when `peer.yaml` already exists:

```bash
python3 -m ttk e2e -i "$KDA_ASSETS/cases/flash_kda_grad_stc.csv" \
  --plugin "$KDA_ASSETS/flash_kda_grad_ttk/spec.py" --config /path/to/peer.yaml \
  --device-whitelist "$KDA_DEVICES" --seed 20261027 --pc 1 \
  --proc-no-reuse --proc-timeout 3000 --warmup=false --run 1 \
  --task-prof false -o "$KDA_OUT"
```

## 6. Result Interpretation and Troubleshooting

The result CSV is specified with `-o` or `KDA_OUT`. The positions of the eight outputs in the terminal and accuracy details are, in order,
`dq, dk, dv, dh0, dg, dbeta, dA, dBias`. Common fields:

The `pre_compare` hook of [`spec.py`](flash_kda_grad_ttk/spec.py) prints the numerical table of the eight gradients to the terminal and `ttk.log`:
the first and last 10 elements each, the 10 elements with the largest absolute difference, and the largest finite
absolute error and the count of non-finite values. `Index` is the flattened element index,
`AbsDiff = abs(NpuOut - GoldenOut)`, and `RateDiff = AbsDiff / abs(GoldenOut)`; when golden
is zero, 0 is displayed if the two are equal, and inf is displayed otherwise (NaN remains displayed as NaN). Empty outputs or inconsistent shapes are
labeled separately. The table is only for locating problems; no separate PASS/FAIL is set for per-element errors; the final conclusion is still given by the native
`cross_check L2` three-way criterion, and golden, the three-way reference, and the thresholds are all unchanged.
The native comparison details remain in `ttk-compare.log`. Setting `KDA_COMPARE_DETAILS=0` disables the extra printing and scanning.

When changing the signature of `api.py`, remember: **Do not add `from __future__ import annotations`** — the
`_annotation_to_type` of TTK only recognizes real type objects, and string annotations all degrade to guessing types by parameter names
(`beta` would be guessed as `float`, and `gate_mode` would be guessed as `Tensor`).

### TND and Valid Sequence Length

The TTK adapter, golden, and CPU peer accept the same optional Tensor arguments, placed after `dht`:
`cu_seqlens=None, seqused=None`. Calls with the original ten tensors remain compatible.

- `layout="TND"`: pass a contiguous int32 `cu_seqlens` of `[B+1]`, with the first item 0 and the last item T; adjacent items are allowed to be equal (empty sequences); do not pass `seqused`.
- `layout="BNSD"` / `"BSND"`: you can pass a contiguous int32 `seqused` of `[B]`, with each item in `[0,S]`; omitting it means all S tokens are used. Do not pass `cu_seqlens`.
- On the NPU path, the length tensors must be on the same device as q. golden/peer use the same set of length values; input gradients in the padding region are zero.
- The batch dimension of `initial_state` / `dht` represents the number of logical sequences B, and the B of TND cannot be inferred from the token dimension of q.

The NPU entry keeps the actual layout of the request; the reference caches are generated by valid lengths and mapped to the padded chunk axis of the current operator or the reserved chunk slots of TND. `cache_source="fwd"` also passes the lengths to the forward metadata.
When running variable-length cases with the CSV, the length tensors need to be provided with exact values through TTK custom inputs or existing input files; random int32 data cannot be treated as a valid prefix sum.
