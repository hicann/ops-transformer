# Recurrent_gated_delta_rule Operator TTK Test Assets

English | [简体中文](./README.md)

## Function Description

Based on the ops-test-kit (TTK) test framework, this directory implements E2E and ACLNN accuracy verification for the Recurrent_gated_delta_rule operator:

- **E2E mode**: Obtains the actual NPU data by directly calling the TorchNPU Python API (`torch_npu.npu_recurrent_gated_delta_rule`) or through the torch.compile graph mode, and compares it with the CPU golden
- **ACLNN mode**: Calls the operator through the aclnn C API (`aclnnRecurrentGatedDeltaRule`) and compares the result with the CPU golden
- **Accuracy comparison**: Supports multiple execution modes, such as eager and static graph (const graph), and compares the accuracy of the CPU and NPU results

## Current Implementation Scope

### Supported Test Modes

| Mode | API | Description |
|------|-----|------|
| E2E eager | `torch_npu.npu_recurrent_gated_delta_rule` | Direct torch call, with profiling |
| E2E const graph | Same as above | Compile and execute with static shapes through torch.compile |
| ACLNN | `aclnnRecurrentGatedDeltaRule` | Call through the aclnn C API |

### Accuracy Standards

- The tolerance declares the community standard `stat_rel_err` (bfloat16).
- The comparison criteria (`impl/compare.py`): `np.isclose(rtol=0.0078125, atol=0.0001)`. A case passes only when the pass rate is ≥ 99.5% and the maximum relative error is < 10.0.

### Environment Configuration

#### Prerequisites

1. TorchNPU installation package download path (replace it with the latest version in time): [TorchNPU Installation Guide](https://atomgit.com/Ascend/pytorch)
2. Complete the environment installation and environment variable configuration. For details, see [ops-transformer](../../../../README_en.md).
3. ops-test-kit test framework (TTK): [ops-test-kit](https://atomgit.com/cann/ops-test-kit)

## File Structure

```
attention/recurrent_gated_delta_rule/tests/
└── assets/
    ├── convert_rdv_to_csv.py                            # Script to convert all RDV cases to TTK CSV (E2E + ACLNN)
    ├── spec.py                                          # TestSpec: registers API + golden/inputs/tolerance/compare/torch_graph (e2e + aclnn)
    └── impl/
        ├── golden.py                                    # golden adapter (e2e + aclnn), reuses the CPU golden of pytest
        ├── inputs.py                                    # inputs adapter (e2e + aclnn), fills index/length tensors
        ├── compare.py                                   # Numeric precision comparison (dual output: output + state)
        └── graph.py                                     # Custom torch.nn.Module, dedicated to the torch.compile graph mode
```

### File Responsibilities

#### spec.py

- Defines the `RecurrentGatedDeltaRuleSpec` class (e2e), declaring `golden`, `customize_inputs`, `torch_graph`, `tolerance`, and `compare`.
- Defines the `AclnnRgdrSpec` class (aclnn), declaring `golden`, `customize_inputs`, `tolerance`, and `compare`.
- Registers through the `__spec__` dict:
    - `torch_npu.npu_recurrent_gated_delta_rule` → `RecurrentGatedDeltaRuleSpec`
    - `aclnnRecurrentGatedDeltaRule` → `AclnnRgdrSpec`
- `torch_graph` points to `RgdrGraphModule` in `impl/graph.py` for use in the torch.compile graph mode.
- Dynamically loads the four modules under `impl/` to avoid hardcoded paths.

#### impl/golden.py

- Loads the CPU golden in `tests/pytest/recurrent_gated_delta_rule_golden.py` through `importlib` without implementing it again.
- **e2e adapter** `cpu_recurrent_gated_delta_rule`: Uses the `*args, **kwargs` signature, is compatible with positional and keyword arguments according to `_PARAM_ORDER`, and returns `(output, state)`.
- **aclnn adapter** `aclnn_cpu_recurrent_gated_delta_rule`: Aligns the parameter order with the function signature of `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize`, returns the `[output, state_out]` list, and is consistent with the order of `output_tensor_indexes`.

#### impl/inputs.py

- Fills the int32 tensors that cannot be randomly generated:
    - `actual_seq_lengths`: Divides T evenly among B batches.
    - `ssm_state_indices`: `arange(T)`.
    - `num_accepted_tokens` (optional): One value within `[1, seq_len]` for each batch.

#### impl/compare.py

- Performs numeric precision comparison between the NPU output and the golden output, and supports dual output: output + state.

#### impl/graph.py

- `RgdrGraphModule`: Custom Module for the torch.compile graph mode. It explicitly returns `(output, state_out)` so that in-place state modifications can be traced.

### convert_rdv_to_csv.py

- Converts all RDV cases (148 cases) in `tests/pytest/test_recurrent_gated_delta_rule_paramset_rdv.py` to TTK CSV.
- Covers combinations of contiguous/non-contiguous state and bf16/fp32 state dtypes. A non-contiguous state is described by `tensor_storage_shapes`/`tensor_view_strides`/`tensor_view_offsets`.
- Generates two CSV files: E2E mode (`recurrent_gated_delta_rule_rdv.csv`) and ACLNN mode (`aclnn_recurrent_gated_delta_rule_rdv.csv`).

```bash
cd attention/recurrent_gated_delta_rule/tests/assets
python3 convert_rdv_to_csv.py
```

## Usage

Prepare the CSV case files for E2E and ACLNN, and define the path variables:

```bash
TTK_DIR=<ops-test-kit path>
ASSETS=attention/recurrent_gated_delta_rule/tests/assets
CSV_E2E=<E2E CSV case path>
CSV_ACLNN=<ACLNN CSV case path>
```

### E2E Direct torch Call (eager)

```bash
cd $TTK_DIR
python3 -m ttk e2e \
  -i $CSV_E2E \
  --plugin $ASSETS \
  --warmup false --run 1 \
  --pc 1
```

### E2E Static Graph (const graph)

```bash
cd $TTK_DIR
python3 -m ttk e2e \
  -i $CSV_E2E \
  --plugin $ASSETS \
  --warmup false --run 1 \
  -c \
  --pc 1 \
  -o /tmp/rgdr_perf.csv
```

### ACLNN Mode

```bash
cd $TTK_DIR
python3 -m ttk aclnn \
  -i $CSV_ACLNN \
  --plugin $ASSETS \
  --pc 1 \
  -o /tmp/rgdr_aclnn_result.csv
```

### Offline Data Storage as bin and bin Replay (Two-Phase Execution)

Splits input generation/CPU golden generation and device execution into two steps: the prepare phase only generates and saves input/golden to bin files (without running the device), and the replay phase restores data from the bin files and then runs the device + compare. This is applicable to scenarios where the preparation machine and execution machine are separated, or where fixed inputs need to be reused for rerunning.

```bash
MANUAL_DIR=/tmp/rgdr_manual   # Root directory for storing bin data
OUT=<result output CSV path>
```

#### 1. prepare (Save bin)

Complete data (input + golden):

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

Only prepare input (golden is computed again during replay):

```bash
python3 -m ttk e2e \
  -i $CSV_E2E --plugin $ASSETS \
  --no-prof --dump in --dump-format bin \
  --manual-data-dirs $MANUAL_DIR --pc 1 \
  -o $OUT
```

#### 2. replay (Run with the specified bin)

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

- The success status of prepare is `MANUAL_DATA_PREPARED`. After a replay hit, random input generation is skipped, and input (and golden) is restored from the bin files. The log shows `OnLoadManualData: loaded prepared input/scalar data from $MANUAL_DIR/...`.
- `--pc 1` runs serially in a single process. Single-process execution is recommended for in-place state cases.
- Both phases must use the same CSV and assets. After the input shape/view or attributes are modified, prepare must be performed again.
- **`--plugin` is required**: Both phases depend on spec.py to provide the golden/inputs adapters. If it is omitted, the error `cannot save golden[0] sentinel 'UNSUPPORTED'` is reported (`Golden Shapes: ()` is empty).
- **`--no-prof` uses a double dash**: Writing `-no-prof` reports the error `unrecognized arguments: -no-prof`.

### Key Parameters

| Parameter | Function | Value for This Operator |
|------|------|-----------|
| `--run N` | Number of executions for averaging | E2E must be 1 (in-place constraint); ACLNN default is 3 |
| `--warmup` | Warmup before profiling | E2E must be false (in-place constraint); ACLNN default is true |
| `-o FILE` | Output result CSV (with time-consumption columns) | Recommended during profiling |
| `-c` | Additionally measures the time consumption of the E2E static graph mode | Optional |
| `--pc N` | Number of concurrent processes | `--pc 1` is recommended for this operator (in-place state cases) |
| `--no-prof --dump in[,golden]` | Saves bin data in the prepare phase without running the device | For details, see the "Offline Data Storage as bin" section |
| `--manual-data-dirs DIR` | Restores bin data from DIR during replay | Keep it the same as the DIR of prepare |

### CSV Format Validation Only

```bash
# E2E
python3 -m ttk e2e -i $CSV_E2E --validate
# ACLNN
python3 -m ttk aclnn -i $CSV_ACLNN --plugin $ASSETS --validate
```
