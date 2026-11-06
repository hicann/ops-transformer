# Compressor Operator Test Framework

English | [简体中文](./README.md)

## Function Description

Based on the pytest test framework, this framework implements functional verification of the Compressor operator:

- **CPU side**: Reproduces the operator functionality to generate golden data
- **NPU side**: Directly calls the operator through torch_npu to obtain actual data
- **Accuracy comparison**: Compares the accuracy of CPU and NPU results to verify operator functionality
- **Dual-mode execution isolation**: Supports two batch modes: direct multi-process pytest execution and process isolation at the shell layer
- **Performance collection**: Supports mounting msprof to collect operator performance data and output a summary

## Current Implementation Scope

### Parameter Restrictions

- Supports D of 128/512.
- Supports H from 1K to 10K, aligned to 512.
- Supports cmp_ratio of 2, 4, 8, 16, 32, 64, and 128.
- When cmp_ratio is 4/128, the following three cases are supported:
    - C4A: D=512, coff=2, cmp_ratio=4;
    - C4Li: D=128, coff=2, cmp_ratio=4;
    - C128A: D=512, coff=1, cmp_ratio=128.

### Environment Configuration

#### Prerequisites

1. Ensure that torch_npu is the latest version  

2. Ensure that the compressor operator package and the torch extension have been built and installed (when the binary does not contain the compressor operator):

``` bash
# ① Build the compressor operator run package at the root directory of the ops-transformer repository (--vendor_name can be customized to avoid overwriting existing vendors packages)
# ${CANN_PATH}: Root installation path of CANNToolkit (for example, /usr/local/Ascend/ascend-toolkit)
# ${SOC}:       Target chip (for example, ascend910b)
source ${CANN_PATH}/set_env.sh
bash build.sh --pkg --soc=${SOC} --ops=compressor --vendor_name=cmpbit -j16
# Artifact: build_out/custom_opp_*.run. Install it to the CANN vendors directory (confirm interactively)
bash build_out/custom_opp_*.run --install-path=${CANN_PATH}/latest

# ② Build and install the torch extension wheel (full package or single operator)
bash build.sh --torch_extension --ops=compressor --vendor_name=cmpbit
pip3 install --force-reinstall --no-deps build_out/cann_ops_transformer_cmpbit-*.whl
```

3. Before running the tests, source the CANN environment variables and the operator vendor package environment (`${CANN_PATH}` is the root installation path of CANNToolkit):

``` bash
source ${CANN_PATH}/set_env.sh
# You also need to source the vendor package environment script that provides the Compressor operator:
source ${CANN_PATH}/latest/vendors/<vendor>_transformer/bin/set_env.bash
```

#### custom Package Invocation

The custom package invocation is supported.

## File Structure

### pytest File Structure Description

- test_run.sh                                  # Execution script. Supports three commands: single/bitlevel/batch (bitlevel supports only A2/A3)
- batch_isolated_run.sh                        # Batch isolated execution script (process isolation at the shell layer + msprof performance collection)
- compressor_golden.py                         # Golden implementation of the operator on the CPU side and accuracy comparison
- collect_perf_data.py                         # msprof performance data collection and summarization
- pytest.ini                                   # Creates test markers

Single test case:

- test_compressor_single.py                    # Main program for running a single test case
- compressor_operator_single.py                # CPU-side operator logic implementation for obtaining golden data and directly calling the NPU operator
- test_compressor_paramset.py                  # Input parameter configuration for a single test case

Bit-level consistency test:

- test_compressor_bitlevel.py                  # Main program for bit-level consistency tests (pair/position test cases)
- compressor_bitlevel_golden.py                # Deterministic data construction + direct NPU invocation + bit-level comparison support library
- test_compressor_bitlevel_paramset.py         # Parameter configuration for bit-level consistency test cases

Batch test:

- test_compressor_batch.py                     # Main program for batch test execution and Excel file generation for saving results (isolation mode supported)
- ./batch/compressor_pt_loadprocess.py         # Reads pt files and calls the operator to obtain NPU output
- ./batch/compressor_pt_save.py                # Reads Excel files and generates test case pt files in batches
- ./batch/replace_path.py                      # Placeholder replacement for test_compressor_batch.py

## Usage

Run the commands in the pytest directory:

### Running Test Cases

#### Single Test Case Debugging

1. Manually configure the ENABLED_PARAMS parameter in test_compressor_paramset.py

2. Run the command:

``` bash
bash test_run.sh single
```

#### Bit-Level Consistency Debugging

1. Manually configure the BITLEVEL_TEST_PARAMS parameter in test_compressor_bitlevel_paramset.py

2. Run the command:

``` bash
bash test_run.sh bitlevel
# Or directly
python3 -m pytest -rA -s test_compressor_bitlevel.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

3. Environment variables:

| Variable | Default Value | Description |
|---|---|---|
| `CMP_TEST_DEVICE_ID` | `0` | Specifies the ID of the NPU device to run on |
| `CMP_BITLEVEL_STRICT` | `1` | `1`: The test case fails if bit-level results are inconsistent; `0`: Downgrades failure to a warning plus threshold check (used to distinguish bit-level tail differences from functional errors) |

4. Test case description (see the comments at the head of test_compressor_bitlevel_paramset.py for details):

| Test Case | Purpose |
|---|---|
| B=32/33, S=4 | tokenSize=128/132, FULL_LOAD → PERF template switching, cross-template bit-level consistency |
| B=96/97, S=4 | PERF dynamic D partitioning changes under 24 AIC cores, coff=2, and D=512 (dBaseSize 32↔64, hIdxStart changes) |
| B=31/32 | tokenSize=124/128, both FULL_LOAD, control group with the same template |
| Same B, with the target sample moved to the first, middle, or last position | Batch position consistency |

Note: The `dyn` test cases in the `PerfDsplitCliff` group automatically derive the dynamic D partitioning cliff boundary pairs based on the number of AIC cores on the current device
(formula: `B_dyn = 128 * (AIC // (D/64)) / S`; B=96/97 on a 24-core device, and B=64/65 on the 910B3 device with 20 cores).
Duplicates with fixed test cases are automatically removed. Each test case performs both CPU golden accuracy verification and NPU result bit-level comparison.

5. Baseline results:

**(Problem record, L0 behavior)**: At the time of measurement, the operator did not yet include the batch consistency modification (deterministic level =0). This record documents the problem symptoms that this modification resolves.
Environment: Atlas 910B3, 20 AIC cores, CANN 9.1.0-beta.1.

| Test Case | Bit-Level Result | Description |
|---|---|---|
| `FullLoadCtrl-pair_31_32` | Consistent | Both use the FULL_LOAD template; the control group passes |
| `FullLoadToPerf-pair_32_33` | **Inconsistent** | cmp_kv: 7/16384 elements differ by 1 BF16 ULP; state update: about 0.001% of elements differ in FP32 mantissa (100% within the threshold) |
| `PerfDsplitCliffDyn-pair_64_65` | **Inconsistent** | The cliff mechanism changes hIdxStart: cmp_kv: 13/32768 elements differ by 1 BF16 ULP; state update: about 1.3% of elements differ in FP32 mantissa (100% within the threshold) |
| `PerfDsplitCliff-pair_96_97` | Consistent | Both sides use dBaseSize=64 under 20 cores, without crossing the cliff |
| `PositionConsistency-pos_B32/33/96/97` | Consistent | Results are bit-level consistent when the target sample is at the first, middle, or last position |

Conclusion (before the fix): **Bit-level inconsistencies exist both across templates (FULL_LOAD vs PERF) and across the PERF dynamic D partitioning cliff (dBaseSize 32↔64, which changes hIdxStart)**.
All differences are at the level of 1 BF16 ULP / FP32 accumulation tail difference (all CPU golden accuracy checks pass, with no functional errors);
batch position consistency is satisfied.

**Measured after the fix (level=3)**: `bash test_run.sh bitlevel` (deterministic level =3) with the default configuration (`CMP_BITLEVEL_STRICT=1`).
Environment: Atlas 910B3, 20 AIC cores, CANN 9.2.0; 7/7 test cases pass.

| Test Case | Bit-Level Result |
|---|---|
| `FullLoadCtrl-pair_31_32` | Consistent |
| `FullLoadToPerf-pair_32_33` | Consistent |
| `PerfDsplitCliff-pair_64_65` | Consistent |
| `PositionConsistency-pos_B32/33/96/97` | Consistent |

Conclusion: **At level=3, both sides are bit-level consistent across templates and across the PERF dynamic D partitioning cliff**; batch position consistency is maintained.
In addition, on the generalized batch splitting suite (13 groups of production shapes and 388 splitting policies), all cases at level=3 are bit-level consistent.
Setting `CMP_BITLEVEL_STRICT=0` downgrades bit-level failures to warnings to distinguish tail differences from functional errors (the default is 1, that is, strict bit-level assertion).

#### Batch Generation and Execution of Test Cases

##### Option A: Batch Execution Using test_run.sh

1. Place the test case Excel file in the excel directory

2. Set the test case Excel file path to read (PATH1) and the pt file storage path (PATH2) in test_run.sh

3. Run the command:

``` bash
bash test_run.sh batch
```

##### Option B: Manual Step-by-Step Execution

1. Generate pt files:

``` bash
python3 batch/compressor_pt_save.py "excel/*.xlsx" pt_path
```

2. Replace the test script path:

``` bash
python3 batch/replace_path.py test_compressor_batch.py pt_path
```

3. Run the test:

``` bash
python3 -m pytest -rA -s test_compressor_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

4. Restore the test script:

``` bash
cp test_compressor_batch.py.bak test_compressor_batch.py
```

##### Option C: Batch Isolated Execution (Recommended for Performance Collection)

This option starts a separate pytest process for each test case, achieving complete inter-process isolation and preventing a crashed test case from affecting other test cases.
Test case acquisition logic: The framework first reads the `Testcase_Name` column of the Excel file in the `./excel` directory and maps it to `.pt` file paths. If no Excel file exists or the `Testcase_Name` column is missing, the framework automatically falls back to scanning all `.pt` files in the test case directory.

``` bash
bash batch_isolated_run.sh ./pt_path 0         # Do not collect performance data
bash batch_isolated_run.sh ./pt_path 1         # Collect performance data (mount msprof)
```

## Excel Test Case Table Format

Sheet1 of `excel/test_cases.xlsx` must contain the following columns:

| Column Name | Type | Example | Description |
|---|---|---|---|
| Testcase_Name | str | `Prefill0` | Test case name, also used as the name of the generated .pt file |
| batch_size | int | `1` | |
| hidden_size | int | `4096` | |
| Seq_len | int | `8192` | |
| head_dim | int | `128` / `512` | |
| block_size | int | `128` | |
| cmp_ratio | int | `4` / `128` | |
| coff | int | `1` / `2` | |
| start_p | int | `0` / `8192` | |
| cache_mode | int | `1` / `2` | 1: continuous buffer, 2: circular buffer |
| layout_x | str | `TH` / `BSH` | |
| data_type | str | `BF16` / `FP16` | |
| cu_seqlens | None/str | `None` or `"[0,8192]"` | Required for the TH layout |
| seqused | None/str | `None` or `"[8192]"` | |
| start_pos | None/str | `None` or `"[0]"` | |
| x_datarange | str | `"[-10,10]"` | |
| wkv_datarange | str | `"[-10,10]"` | |
| wgate_datarange | str | `"[-10,10]"` | |
| ape_datarange | str | `"[-10,10]"` | |
| kv_state_datarange | str | `"[-10,10]"` | |
| score_state_datarange | str | `"[-10,10]"` | |

**Precautions**:

- When `layout_x=TH`, `cu_seqlens` is required (a list with a length of batch_size+1)

## Output Files

| File | Description |
|---|---|
| `result.xlsx` | Test results (accuracy, parameters, and so on) |
| `result_perf.xlsx` | Test results + performance data (msprof mode only) |
| `batch_summary.log` | Detailed log of batch isolated execution |
| `batch_fail_list.log` | List of failed test cases |
| `PROF_*/` | Directory of raw msprof performance data |
