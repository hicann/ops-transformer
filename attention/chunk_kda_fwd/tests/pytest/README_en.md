# ChunkKdaFwd Operator Test Framework

English | [简体中文](./README.md)

## Function Description

Based on pytest and msopprof, this directory implements function, determinism, and performance verification for the ChunkKdaFwd operator:

- **Function verification**: Calls the operator through `cann_ops_transformer.ops.chunk_kda_fwd`, and verifies the four input layouts, optional output matrices, BFLOAT16 gate parameter promotion, variable-length sequence tail blocks, and `state_v_first`.
- **Consistency verification**: Compares the results of BNSD, TND, and NTD inputs with the results of BSND inputs across layouts.
- **Determinism verification**: Repeats execution with the same input, and verifies that the specified outputs are binary identical.
- **Performance verification**: Uses msopprof to collect the device-side time consumption of the key scenario, and does not use Python wall time as the performance conclusion.

## Current Implementation Scope

### Case Coverage

| Test Item | Covered Content |
| --- | --- |
| Input layouts | BSND, BNSD, TND, NTD |
| Data types | BFLOAT16 main inputs, FLOAT32 gate, and BFLOAT16 and FLOAT32 `A_log` and `dt_bias` |
| Optional outputs | Default return, `return_intermediate_states=True`, and `disable_recompute=True` |
| Variable-length sequences | NTD, `cu_seqlens=[0,64,65]`, and a sequence tail block of length 1 |
| State layout | `state_v_first=False` and `state_v_first=True` |
| Performance scenario | BFLOAT16, BSND, B=1, S=16384, H=HV=96, K=V=128, `chunk_size=64`, `safe_gate=True`, and `use_gate_in_kernel=True` |

### Environment Configuration

#### Prerequisites

1. Complete the installation and environment variable configuration of CANN, `torch_npu`, and ops-transformer. For details, refer to [ops-transformer](../../../../README_en.md).
2. The custom operator package that contains the `ChunkKdaFwd` operator and the `torch_extension` wheel that provides `cann_ops_transformer.ops.chunk_kda_fwd` are installed.
3. Before performance verification, confirm that the `msopprof` command is available.

## File Structure

```text
pytest/
├── README.md                    # Test framework description
├── common.py                    # Input construction, layout conversion, and common assertions
├── test_chunk_kda_fwd.py        # Function, consistency, and determinism cases
├── profile_a5_h96_t16k.py       # Ascend 950 key scenario performance and 50-run determinism entry
└── check_msopprof.py             # msopprof result parsing and performance gating
```

## Usage

Run the commands in the repository root directory of ops-transformer.

### Running Function Cases

```bash
python3 -m pytest -rA -s \
  attention/chunk_kda_fwd/tests/pytest/test_chunk_kda_fwd.py \
  -v
```

### Running 50-Round Binary Determinism Verification

```bash
python3 attention/chunk_kda_fwd/tests/pytest/profile_a5_h96_t16k.py \
  --check-determinism \
  --determinism-repeats 50
```

The determinism verification performs binary consistency comparison on `attn_out`, `final_state`, `Aqk`, and `Akk`.

### Running Ascend 950 Key Scenario Performance Verification

Create the performance data directory:

```bash
mkdir -m 700 -p output/chunk_kda_fwd_profiles
```

Collect the `ChunkKdaFwd` main kernel:

```bash
msopprof \
  --application="python3 attention/chunk_kda_fwd/tests/pytest/profile_a5_h96_t16k.py --repeats 8" \
  --output=output/chunk_kda_fwd_profiles/a5_h96_t16k \
  --aic-metrics=BasicInfo \
  --kernel-name=ChunkKdaFwd \
  --launch-count=5 \
  --warm-up=3 \
  --kill=off
```

Collect the BSND `beta` layout conversion kernel:

```bash
msopprof \
  --application="python3 attention/chunk_kda_fwd/tests/pytest/profile_a5_h96_t16k.py --repeats 8" \
  --output=output/chunk_kda_fwd_profiles/a5_h96_t16k_beta \
  --aic-metrics=BasicInfo \
  --kernel-name=Transpose_float16_int64_high_performance_10001 \
  --launch-count=5 \
  --warm-up=3 \
  --kill=off
```

Verify the end-to-end device-side time consumption:

```bash
python3 attention/chunk_kda_fwd/tests/pytest/check_msopprof.py \
  output/chunk_kda_fwd_profiles/a5_h96_t16k \
  --component Transpose_float16_int64_high_performance_10001 \
    output/chunk_kda_fwd_profiles/a5_h96_t16k_beta \
  --max-ms 11
```

`check_msopprof.py` takes the maximum of the valid samples of the main kernel and of the `beta` layout conversion kernel respectively, and then sums them. The performance gating check fails when the total time consumption exceeds 11 ms or when no valid sample is collected for either component.

## Output Description

- The pytest function cases directly output the PASS/FAIL result of each case.
- `profile_a5_h96_t16k.py` outputs `binary_determinism=PASS` when determinism passes.
- `check_msopprof.py` outputs the sample count, minimum/average/maximum time consumption, and end-to-end maximum time consumption of each component.
- `output/chunk_kda_fwd_profiles/` contains local profiling artifacts and must not be committed to the code repository.
