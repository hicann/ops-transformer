# Recurrent_gated_delta_rule Operator Test Framework

English | [简体中文](./README.md)

## Function Description

Based on the pytest test framework, this directory implements functional verification for the Recurrent_gated_delta_rule operator:

- **CPU side**: Reproduces the operator function to generate golden data
- **NPU side**: Obtains the actual data by directly calling the operator through TorchNPU
- **Accuracy comparison**: Compares the accuracy of the CPU and NPU results to verify the operator function

## Current Implementation Scope

### Parameter Restrictions

- batch_size greater than 0 is supported.
- mtp from 1 to 8 is supported.
- NK and Nv less than or equal to 256 are supported; Nv must be greater than or equal to Nk and Nv must be divisible by Nk.
- Dk and Dv less than or equal to 512 are supported.
- The actual_seq_lengths input with a length of batch_size is supported.
    - The values in actual_seq_lengths must be greater than or equal to 0 and less than or equal to mtp.
    - If the input is not specified, an array with a length of batch_size and values of mtp is passed in by default.
    - T equals the sum of all elements in actual_seq_lengths.
- The ssm_state_indices input with a length of T is supported.
    - The values in ssm_state_indices must be less than block_num.
    - If the input is not specified, the default value is [0,1,...,T-1].
- block_num can be manually passed in and must be greater than or equal to T.
- data_type BF16 is supported.
- For query_datarange, the minimum left boundary is -1 and the maximum right boundary is 1.
- For key_datarange, the minimum left boundary is -1 and the maximum right boundary is 1.
- For gamma_datarange, the maximum right boundary is 0.
- For gamma_k_datarange, the maximum right boundary is 0.
- For beta_datarange, the minimum left boundary is 0 and the maximum right boundary is 1.

### Environment Configuration

#### Prerequisites

1. TorchNPU installation package download path (replace it with the latest version in time): [TorchNPU Installation Guide](https://atomgit.com/Ascend/pytorch)
2. Complete the environment installation and environment variable configuration. For details, see [ops-transformer](../../../../README_en.md).

#### custom Package Invocation

Custom package invocation is supported.

## File Structure

### pytest File Structure Description

- test_run.sh                               # Execution script
- conftest.py                               # pytest hook: records parameters/results/accuracy/seed for each case, and writes the CSV at the end of the session
- recurrent_gated_delta_rule_golden.py      # CPU-side operator golden implementation, and accuracy comparison between the CPU golden and NPU results
- pytest.ini                                # Creates test markers for the CI single-operator and graph modes

Single-case test:

- test_recurrent_gated_delta_rule_single.py                 # Main program for running a single test case
- recurrent_gated_delta_rule_operator_single.py             # CPU-side operator logic implementation to obtain the golden, and direct invocation of the NPU operator
- test_recurrent_gated_delta_rule_paramset.py               # Input parameter configuration for the single case
- test_recurrent_gated_delta_rule_paramset_rdv.py           # Input parameter configuration for the RDV test

## Usage

Run the commands in the pytest folder path:

### Running Test Cases

#### Single-Case Test

1. Manually configure the parameters in test_recurrent_gated_delta_rule_paramset.py

2. Run the command:

``` bash
bash test_run.sh single
```

#### RDV Test

1. Manually configure the parameters in test_recurrent_gated_delta_rule_paramset_rdv.py

2. Run the command:

``` bash
bash test_run.sh rdv
```

#### Random Case Test

Randomly generates N cases and runs them (with CPU golden accuracy comparison). The `RANDOM_SEED` environment variable can fix the random seed for reproduction (if it is not specified, it is automatically generated and recorded to the CSV):

``` bash
bash test_run.sh random 100
```

#### Random Case Test (NPU Only)

Randomly generates N cases. Set `SKIP_GOLDEN=1` to skip the CPU golden computation and accuracy comparison, and run only the NPU operator (input tensors are generated directly on the NPU, with low host memory usage), speeding up execution:

``` bash
bash test_run.sh random_npu 100
```

#### Random Case Generation Rules

The random/random_npu/mss modes use the same random parameter generator (`_generate_random_param_dict`), which generates values randomly from 0 within the operator constraints and does not depend on the single/rdv parameter pools. The input generation rules for each case are as follows (in the order of the operator API inputs):

| API Input | Random Rule | Restriction/Description |
|----------|---------|-----------|
| query (T, Nk, Dk) | T=B×mtp, Nk=randint(1,256), Dk=randint(1,min(512,budget)) | dtype is fixed to bf16 |
| key (T, Nk, Dk) | Reuses Nk/Dk | datarange is fixed to [-1,1] |
| value (T, Nv, Dv) | Nv=Nk×randint(1,256//Nk), Dv=randint(1,min(512,budget//Dk)) | datarange is randomly choice([-10,10], [-1,1]) |
| state (BlockNum, Nv, Dv, Dk) | BlockNum=B×mtp, reuses Nv/Dv/Dk | dtype is randomly choice(bf16,fp32), datarange [-10,10] |
| beta (T, Nv) | Reuses T/Nv | datarange is fixed to [0,1] |
| scale | 1/sqrt(Dk) | Automatically computed |
| actual_seq_lengths (B,) | Defaults to all mtp | Automatically generated when not specified |
| ssm_state_indices (T,) | Defaults to [0,1,...,T-1] | Automatically generated when not specified |
| num_accepted_tokens (B,) | randint(1, mtp) when has_num_accepted_tokens=True | Enabled with a 50% probability |
| g (T, Nv) | Generated when has_gamma=True, datarange choice(4 negative intervals) | Enabled with a 50% probability, dtype fp32 |
| gk (T, Nv, Dk) | Generated when has_gamma_k=True, datarange choice(4 negative intervals) | Enabled with a 50% probability, dtype fp32 |
| state_non_contiguous | choice([False, True]) | Non-contiguous with a 50% probability |

**Other fixed items**: data_type is fixed to bfloat16; query/key_datarange is fixed to [-1,1]; beta_datarange is fixed to [0,1]; state_datarange is fixed to [-10,10].

**Shape restrictions**: 0<Nk≤256, 0<Nv≤256, Nv≥Nk, Nv%Nk==0, 0<Dk≤512, 0<Dv≤512, mtp≤8, BlockNum≥T.

**Memory restriction**: Dk×Dv is constrained by the upper limit of state elements `_STATE_ELEM_CAP=2.0B` (`budget = STATE_ELEM_CAP // (BlockNum × Nv)`), preventing host OOM in a single process.

**Random seed mechanism**:
- `RANDOM_SEED` controls the shape/parameter sequence (one seed corresponds to a determined set of parameters for N cases)
- `TORCH_SEED` controls the tensor values (independent for each case; conftest automatically generates and records them to the CSV)
- If `RANDOM_SEED` is not set, it is automatically generated and written back to `os.environ`, and conftest writes the CSV
- Reproduction: `RANDOM_SEED=<seed> bash test_run.sh random N` (batch reproduction), or `TORCH_SEED=<tensor_seed>` combined with the CSV inputs (value-level reproduction of a single case)

**Batching mode** (dedicated to the mss mode, `MSS_BATCH=N`): Each batch restarts mssanitizer with the independent seed `seed+i` to avoid host memory accumulation. The CSV/log is appended to a single file.

**Duplication probability**: The parameter space is about 5×10¹⁰ (50 billion). The probability of collision for 10,000 cases is <0.1%, so duplication does not practically occur.

#### mssanitizer Detection

Randomly generates N cases, runs only on the NPU (without the golden), and runs under mssanitizer to detect device-side memory out-of-bounds, illegal address access, and other problems (for the detailed column semantics, see the "Result Output and Reproduction" section below). If the screen output contains `====== ERROR`, the result is FAIL (the script exit code is 1). By default, only the kernel of this operator is checked (`--kernel-name=RecurrentGatedDeltaRule`), skipping the detection overhead of bypass kernels such as ZerosLike/ViewCopy/TensorMove/rand:

``` bash
bash test_run.sh mss 10
```

**Prerequisites (mssanitizer installation)**:

- Official introduction: [mssanitizer Quick Start](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/optool/docs/zh/quick_start/mssanitizer_quick_start.md)
- No additional installation is required if the CANN toolkit already includes it (for example, `/home/developer/Ascend/cann-9.2.0/bin/mssanitizer`; ensure that its directory is in PATH).
- If it is not included, compile and install from the source code:

``` bash
git clone https://atomgit.com/Ascend/mssanitizer.git
cd mssanitizer
python3 build.py
# After compilation, install the run package
artifacts/mindstudio-sanitizer_26.0.0_x86_64.run --run --install-path=/home/developer/Ascend/
```

Environment variable customization is supported:

``` bash
MSS_TOOL=racecheck bash test_run.sh mss 10                     # Change the detection tool (memcheck/racecheck/initcheck/synccheck)
MSS_KERNEL='' bash test_run.sh mss 10                          # Disable kernel filtering and check all kernels
MSS_EXTRA_OPTS='--leak-check=yes --full-backtrace=yes' bash test_run.sh mss 10  # Append mssanitizer options
MSSANITIZER_BIN=/path/to/mssanitizer bash test_run.sh mss 10   # Specify the mssanitizer path
```

### Result Output and Reproduction

After execution, all modes output to the `output/` directory (gitignored):

- `run_<timestamp>.log` / `mss_<timestamp>.log`: Complete execution log (displayed on screen by tee)
- `result_<timestamp>.csv`: Per-case result table, with one case per row. The column meanings are as follows:

| Column | Description |
|----|------|
| random_seed | Random shape sequence seed (random modes; not present for single/rdv because they use fixed parameter sets) |
| tensor_seed | Tensor value seed of this case (recorded independently for each case) |
| test_mode | single/rdv/random |
| check_type | precision=with golden accuracy comparison / execution_only=NPU execution only / execution_only+mss_\<tool\>=mssanitizer detection |
| result | pytest execution result (PASSED/FAILED/SKIPPED) |
| mss_check | mssanitizer detection conclusion (PASS/FAIL(errors=N)/CRASH). Only in mss mode, the script parses the log and fills it back |
| out_pct_rlt / state_pct_rlt | Real PctRlt accuracy compliance values of output and state (for example, 99.999982%), recorded only in precision mode |
| batch_size...state_non_contiguous (24 columns) | All inputs of this case |
| errmsg | Failure details (truncated to 2000 characters) |

Reproducing failed cases (two-layer seeds combined with the CSV inputs):

``` bash
# Batch reproduction (same shape sequence): take random_seed from the CSV
RANDOM_SEED=<random_seed> bash test_run.sh random N

# Value-level reproduction of a single case (same shape + same tensor values): take the inputs and tensor_seed from the CSV
TORCH_SEED=<tensor_seed> bash test_run.sh random 1   # Combined with the inputs of the case (or locate the case by random_seed)
```

### Environment Variable Summary

| Variable | Function | Applicable Modes |
|------|------|---------|
| RANDOM_SEED | Fixes the random shape sequence seed (if not set, it is automatically generated and recorded to the CSV) | random/random_npu/mss |
| TORCH_SEED | Fixes the tensor value seed (if not set, it is automatically generated for each case and recorded to the CSV) | All |
| RANDOM_CASE_COUNT | Number of random cases (passed through by test_run.sh) | random modes |
| SKIP_GOLDEN | =1 skips the CPU golden and accuracy comparison, and runs only on the NPU | random_npu/mss |
| CSV_FILE | Specifies the CSV output path (set automatically by test_run.sh) | All |
| CSV_APPEND | =1 appends to the CSV (set automatically in batching mode; manage the header manually if used manually) | All |
| MSS_BATCH | mss batch size. Each batch restarts mssanitizer to avoid host memory accumulation (0=no batching) | mss |
| MSS_TOOL | mssanitizer detection tool (memcheck/racecheck/initcheck/synccheck) | mss |
| MSS_KERNEL | Kernel filter name (default RecurrentGatedDeltaRule; leave empty to check all) | mss |
| MSS_EXTRA_OPTS | Additional mssanitizer options (for example, --leak-check=yes) | mss |
| MSSANITIZER_BIN | Path of the mssanitizer executable | mss |

### Log and CSV Automatic Saving

All modes automatically save the screen output to the `output/` directory (gitignored) through `tee`:

| Mode | Log File | CSV File |
|------|---------|---------|
| single/rdv/random/random_npu | `output/run_<timestamp>.log` | `output/result_<timestamp>.csv` |
| mss (single process) | `output/mss_<timestamp>.log` | `output/result_<timestamp>.csv` |
| mss (batching, MSS_BATCH>0) | `output/mss_<timestamp>.log` (appended) | `output/result_<timestamp>.csv` (appended) |
