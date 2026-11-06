# TTK Test Assets for the Chunk_gated_delta_rule Operator

English | [简体中文](./README.md)

## Function Description

Based on the ops-test-kit (TTK) test framework, this directory implements E2E and ACLNN precision verification for the Chunk_gated_delta_rule operator:

- **E2E mode**: Obtains actual NPU data through direct calls to the TorchNPU Python API (`torch_npu.npu_chunk_gated_delta_rule`) or the torch.compile graph mode, and compares it with the CPU golden data
- **ACLNN mode**: Calls the operator through the aclnn C API (`aclnnChunkGatedDeltaRule`) and compares it with the CPU golden data
- **Precision comparison**: Three-way cross-check (NPU vs golden vs benchmark), supporting multiple execution modes including eager and aclgraph

## Current Implementation Scope

### Supported Test Modes

| Mode | API | Description |
|------|-----|------|
| E2E eager | `torch_npu.npu_chunk_gated_delta_rule` | Direct torch call, including profiling |
| E2E aclgraph | Same as above | torch.compile reduce-overhead mode (npugraph_ex backend) |
| ACLNN | `aclnnChunkGatedDeltaRule` | aclnn C API call |

> **Note**: The const graph (torchair GE backend) is currently not supported. The operator is not registered in the GE graph engine, and execution reports `Interface npu_chunk_gated_delta_rule is not supported in GE graph mode`. The aclgraph mode uses the `npugraph_ex` backend, captures graphs at the aclnn level, and can run properly.

### Precision Standard

- The tolerance declares the community standard `cross_check` (bfloat16) with level `L1`.
- Comparison determination conditions (`impl/compare.py`): three-way cross-check. Four ratios (NPU error / benchmark error) are computed for mare/mere/rmse/smra, with thresholds `CV_MAX_RE=5.0` / `CV_AVER_RE=1.5` / `CV_RMSE=1.5` / `CV_SMALL_VAL=2.0` respectively. The result is Pass only when all ratios are below the thresholds. The floor of the ratio denominator is `2^-8` (`_ERR_THRESHOLD`).

### Environment Configuration

#### Prerequisites

1. TorchNPU installation package download path (replace it with the latest version in time): [TorchNPU Installation Tutorial](https://atomgit.com/Ascend/pytorch)
2. Complete the environment installation and environment variable configuration. For details, refer to [ops-transformer](../../../../README_en.md).
3. ops-test-kit test framework (TTK): [ops-test-kit](https://atomgit.com/cann/ops-test-kit)

## File Structure

```
attention/chunk_gated_delta_rule/tests/assets/
├── convert_rdv_to_csv.py                        # Script for converting all RDV cases to TTK CSV (E2E + ACLNN)
├── spec.py                                      # TestSpec: registers API + golden/inputs/tolerance/compare (e2e + aclnn)
└── impl/
    ├── golden.py                                # golden adapter (e2e + aclnn), loads CPU golden + third-party benchmark
    ├── inputs.py                                # inputs adapter (e2e + aclnn), fills actual_seq_lengths + applies constraint clipping
    └── compare.py                               # Three-way cross-check precision comparison (output + state dual outputs)
```

### File Responsibilities

#### spec.py

- Defines the `ChunkGatedDeltaRuleSpec` class (e2e), which declares `golden`, `customize_inputs`, `tolerance`, and `compare`.
- Defines the `AclnnChunkGatedDeltaRuleSpec` class (aclnn), which declares `golden`, `customize_inputs`, `tolerance`, and `compare`.
- Registers through the `__spec__` dict:
    - `torch_npu.npu_chunk_gated_delta_rule` → `ChunkGatedDeltaRuleSpec`
    - `aclnnChunkGatedDeltaRule` → `AclnnChunkGatedDeltaRuleSpec`
- `compare` is a custom three-way comparison. The golden computation also computes the benchmark result and passes it to compare through `_GOLDEN_CONTEXT`.
- Dynamically loads the three modules under `impl/` to avoid hardcoded paths.

#### impl/golden.py

- Loads the CPU golden implementation and the third-party benchmark implementation through `importlib` without re-implementing them.
- **e2e adapter** `cpu_chunk_gated_delta_rule`: uses the `*args, **kwargs` signature, is compatible with positional and keyword arguments according to `_PARAM_ORDER`, and returns `[out, final_state]`.
- **aclnn adapter** `aclnn_chunk_gated_delta_rule_golden`: aligns the parameter order with the `aclnnChunkGatedDeltaRuleGetWorkspaceSize` function signature, returns `[out_golden, finalState_golden]`, and keeps the order consistent with `output_tensor_indexes`.
- The golden and benchmark results are computed in the same function `_compute_and_store`, and the benchmark result is stored in `_GOLDEN_CONTEXT` for compare to use.
- **prepare persistence** `_save_bench`: when golden data is dumped with `--dump in,golden`, the benchmark data is simultaneously saved to the sibling directory `<case_name>.bench/` in the manual-data directory (the TTK case directory schema is fixed and does not allow extra files); the format is consistent with the TTK golden bin: raw bytes, with dtype/shape encoded in the file name (`bench_<i>_<dtype>__shape_<DxN>.bin`, reusing `dump_to_file`/`load_numpy_data`).
- **replay loading** `resolve_bench`: during full bin replay, the golden function is skipped and `_GOLDEN_CONTEXT` is empty. In this case, the benchmark is loaded from the `.bench/` directory, consistent with the behavior of loading golden from the bin, and neither is recomputed. For legacy data without a `.bench` directory, the fallback is to recompute from inputs restored from the bin through `compare_context` (the replay-safe input context injected by TTK), with a warning in the log. Both paths are cached by `testcase_name` to prevent data mixing across cases.

#### impl/inputs.py

- Fills the int32 tensor that cannot be generated randomly and applies the operator numerical constraints:
    - `actual_seq_lengths`: evenly distributes T across B batches.
    - q/k: L2 normalization (the operator requires pre-normalized inputs).
    - `g`: clipped to `[-1, 0]`.
    - `beta`: clipped to `(0, 1)`.

#### impl/compare.py

- Three-way cross-check: NPU output vs golden output vs benchmark output, supporting dual outputs of output + finalState.
- Computes four ratios (NPU error / benchmark error) for mare/mere/rmse/smra. The thresholds and floor values are exactly the same as those of `compare_cv` on the operator side.
- Returns a dict structure containing `pass`/`precision`/`error_info`/`metrics` (including the four ratios and the raw error values).

### convert_rdv_to_csv.py

- Converts all RDV cases (38 redline cases + 34 STC cases, doubled after FP32 state and non-contiguous state extension) to TTK CSV.
- Covers combinations of contiguous/non-contiguous state and bf16/fp32 state dtypes. Non-contiguous state is described through `tensor_storage_shapes`/`tensor_view_strides`/`tensor_view_offsets`.
- Generates two CSV files: one for E2E mode (`chunk_gated_delta_rule_rdv.csv`) and one for ACLNN mode (`aclnn_chunk_gated_delta_rule_rdv.csv`).

```bash
cd attention/chunk_gated_delta_rule/tests/assets
python3 convert_rdv_to_csv.py
```

## Usage

Before use, prepare the CSV case files for E2E and ACLNN modes and define the path variables:

```bash
TTK_DIR=<ops-test-kit path>
ASSETS=attention/chunk_gated_delta_rule/tests/assets
CSV_E2E=<E2E CSV case path>
CSV_ACLNN=<ACLNN CSV case path>
```

### E2E Mode (eager + aclgraph)

```bash
cd $TTK_DIR
python3 -m ttk e2e \
  -i $CSV_E2E \
  --plugin $ASSETS \
  --aclgraph \
  -o <result output path>
```

### ACLNN Mode

```bash
cd $TTK_DIR
python3 -m ttk aclnn \
  -i $CSV_ACLNN \
  --plugin $ASSETS \
  -o <result output path>
```

### Offline Data Dump to Bin and Replay from Specified Bin (Two-Stage Execution)

Splits input generation/CPU golden generation and device execution into two steps: the prepare stage only generates and saves input/golden to bin files (without running the device), and the replay stage restores data from the bin files and then runs the device + compare. This applies to scenarios where the preparation machine and the execution machine are separate, or where fixed inputs need to be reused for repeated runs.

```bash
MANUAL_DIR=/tmp/cgdr_manual   # Root directory for storing bin data
OUT=<result output CSV path>
```

#### 1. prepare (dump bin)

Full data (input + golden):

```bash
# E2E
cd $TTK_DIR
python3 -m ttk e2e \
  -i $CSV_E2E --plugin $ASSETS \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT

# ACLNN
python3 -m ttk aclnn \
  -i $CSV_ACLNN --plugin $ASSETS \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT
```

Prepare input only (golden is recomputed during replay):

```bash
python3 -m ttk e2e \
  -i $CSV_E2E --plugin $ASSETS \
  --no-prof --dump in --dump-format bin \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT
```

#### 2. replay (run from specified bin)

```bash
# E2E
python3 -m ttk e2e \
  -i $CSV_E2E --plugin $ASSETS \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT

# ACLNN
python3 -m ttk aclnn \
  -i $CSV_ACLNN --plugin $ASSETS \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT
```

- The success status of prepare is `MANUAL_DATA_PREPARED`. When replay hits, random input generation is skipped, and input (and golden) are restored from the bin. The log shows `OnLoadManualData: loaded prepared input/scalar data from $MANUAL_DIR/...`.
- **The three-way benchmark is persisted together with golden and is not recomputed during replay**: during prepare with `--dump in,golden`, the benchmark is saved to `$MANUAL_DIR/<case_name>.bench/bench_<i>_<dtype>__shape_<DxN>.bin`; during replay, golden is loaded from the bin and the benchmark is loaded from that directory, with consistent behavior. For legacy data without a `.bench` directory, the fallback is recomputation from bin inputs (with a log warning); in input-only mode and direct mode, the benchmark is computed as usual during the golden stage.
- `--pc 1` runs serially in a single process. A single process is recommended for this operator.
- Both stages must use the same CSV and assets. After modifying the input shape/view or attributes, run prepare again.
- **`--plugin` is required**: both stages depend on spec.py to provide the golden/inputs/compare adapters.
- **`--no-prof` uses double dashes**: writing `-no-prof` reports `unrecognized arguments: -no-prof`.

### Key Parameters

| Parameter | Function | Value for This Operator |
|------|------|-----------|
| `-o FILE` | Output result CSV (including time consumption columns) | Recommended when profiling is enabled |
| `--aclgraph` | Tests the E2E aclgraph mode (reduce-overhead) | Recommended; const graph is not supported |
| `-c` | Tests the E2E const graph mode (GE backend) | Not supported; the operator is not registered in GE |
| `--pc N` | Number of concurrent processes | `--pc 1` is recommended for this operator |
| `--no-prof --dump in[,golden]` | Dumps bin data in the prepare stage without running the device | For details, refer to the "Offline Data Dump to Bin" section |
| `--manual-data-dirs DIR` | Restores bin data from DIR during replay | Keep DIR consistent with the one used in prepare |

### CSV Format Validation Only

```bash
# E2E
python3 -m ttk e2e -i $CSV_E2E --validate
# ACLNN
python3 -m ttk aclnn -i $CSV_ACLNN --plugin $ASSETS --validate
```
