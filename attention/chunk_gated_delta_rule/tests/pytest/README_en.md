# Chunk_gated_delta_rule Operator Test Framework

English | [简体中文](./README.md)

## Function Description

Based on the pytest test framework, this directory implements functional verification for the Chunk_gated_delta_rule operator:

- **CPU side**: Reproduces the operator function to generate golden data
- **NPU side**: Obtains actual data through direct operator calls using TorchNPU
- **Precision comparison**: Compares the precision of CPU and NPU results to verify the operator function

## Current Implementation Scope

### Parameter Restrictions

- batch_size greater than 0 is supported.
- The seqlen sequence length is supported.
- NK and NV head counts are supported, and NV must be a multiple of NK.
- DK and DV no greater than 128 are supported.
- data_type BF16 is supported.

### Environment Configuration

#### Prerequisites

1. TorchNPU installation package download path (replace it with the latest version in time): [TorchNPU Installation Tutorial](https://atomgit.com/Ascend/pytorch)
2. Complete the environment installation and environment variable configuration. For details, refer to [ops-transformer](../../../../README_en.md).

#### custom Package Invocation

Custom package invocation is supported.

## File Structure

### pytest File Structure Description

- test_run.sh                               # Execution script
- conftest.py                               # pytest hook: records parameters/results/seed for each case, writes CSV at session end
- chunk_gated_delta_rule_golden.py          # CPU-side golden implementation of the operator
- chunk_gated_delta_rule_main.py            # Precision comparison between CPU golden and NPU results, and main test logic
- pytest.ini                                # Creates test markers for CI single-operator and graph modes

Single-case testing:

- test_chunk_gated_delta_rule_single.py     # Main program for testing single-case runs
- chunk_gated_delta_rule_operator_single.py # CPU-side operator logic implementation to obtain golden and direct NPU operator calls
- test_chunk_gated_delta_rule_paramset.py   # Single-case input parameter configuration
- test_chunk_gated_delta_rule_paramset_rdv.py   # RDV test input parameter configuration

## Usage

Run the commands in the pytest directory:

### Running Test Cases

#### Single-Case Debugging

1. Manually configure the parameters in test_chunk_gated_delta_rule_paramset.py.

2. Run the command:

``` bash
bash test_run.sh single
```

#### RDV Testing

1. Manually configure the parameters in test_chunk_gated_delta_rule_paramset_rdv.py.

2. Run the command:

``` bash
bash test_run.sh rdv
```

#### Random Case Testing

Randomly generates N cases and executes them (including CPU golden precision comparison). You can fix the random seed with the `RANDOM_SEED` environment variable for reproduction (if not specified, the seed is automatically generated and recorded in the CSV):

``` bash
bash test_run.sh random 100
```

#### Random Case Testing (NPU Only)

Randomly generates N cases. Setting `SKIP_GOLDEN=1` skips CPU golden computation and precision comparison and executes the NPU operator only, which speeds up execution:

``` bash
bash test_run.sh random_npu 100
```

#### Random Case Generation Rules

Both random/random_npu modes use the same random parameter generator (`_generate_random_param_dict`), which randomly generates from 0 within the operator constraints and does not depend on the single/rdv parameter pools. The generation rules for the input parameters of each case are as follows:

| API Input Parameter | Random Rule | Constraints/Description |
|----------|---------|-----------|
| B | choice([1,2,4,8,16,32,64,128]) | batch size |
| seqlen | choice([1,3,7,32,64,100,128,200,256,300,512,1000,1024,2048,4096,5000,8192,10000,16384,32768,65535]) | Sequence length; includes non-chunk-aligned values (such as 100, 300, 5000, 65535) to test the partial chunk path |
| seqlen (variable length) | With B>1, a list is generated with a 30% probability, and each batch has an independent random seqlen | Covers the actual_seq_lengths variable-length path |
| nk | randint(1,64) | Number of key heads |
| nv | nk × randint(1,64//nk) | Nv>=Nk and Nv%Nk==0 |
| dk | randint(1,min(128,budget)) | key dimension, constrained by the state element upper limit |
| dv | randint(1,min(128,budget//dk)) | value dimension, constrained by the state element upper limit |
| chunk_size | Fixed 64 | Hardcoded in operator tiling; golden must be consistent with NPU |
| data_type | Fixed bfloat16 | |
| state_data_type | choice([bfloat16,float32]) | State data type |
| has_g | choice([True,False]) | Gating is enabled with a 50% probability |
| is_contiguous | choice([True,False]) | Non-contiguous with a 50% probability |
| query_datarange | Fixed [-1,1] | q is L2-normalized |
| key_datarange | Fixed [-1,1] | k is L2-normalized |
| value_datarange | choice([-10,10], [-1,1]) | Random data range |
| gamma_datarange | choice([-1,0], [-0.5,0], [-0.1,0], [-1,-0.5]) | Document constraint [-1,0]; g decays through exp |
| beta_datarange | Fixed [0,1] | |
| state_datarange | Fixed [-10,10] | |

**Shape constraints**: B>0, seqlen>0, 0<Nk<=64, 0<Nv<=64 with Nv>=Nk and Nv%Nk==0, 0<Dk<=128, 0<Dv<=128.

**Memory constraints**:
- Dk×Dv is constrained by the state element count upper limit `_STATE_ELEM_CAP=2.0B` (`budget = STATE_ELEM_CAP // (B × Nv)`)
- T×max(Nk×Dk, Nv×Dv) is constrained by the QKV tensor element upper limit `_QKV_ELEM_CAP=1.0B` (`max_seqlen = QKV_ELEM_CAP // (B × qkv_per_token)`)
- The two constraints jointly prevent OOM in a single process

**Random seed mechanism**:
- `RANDOM_SEED` controls the shape/parameter sequence (one seed corresponds to a deterministic set of N case parameters)
- `TORCH_SEED` controls tensor values (independent for each case; conftest automatically generates and records it in the CSV)
- If `RANDOM_SEED` is not set, it is automatically generated, written back to `os.environ`, and written to the CSV by conftest
- Reproduction: `RANDOM_SEED=<seed> bash test_run.sh random N`

### Result Output and Reproduction

After execution, all modes output to the `output/` directory (gitignored):

- `run_<timestamp>.log`: complete execution log (displayed on screen through tee)
- `result_<timestamp>.csv`: per-case result table, with one case per row

| Column | Description |
|----|------|
| random_seed | Random shape sequence seed (random mode; single/rdv use a fixed parameter set and have no such value) |
| seed | Tensor value seed of this case (recorded independently for each case) |
| test_name | Case name |
| test_mode | single/rdv/random |
| check_type | precision=with golden precision comparison / execution_only=NPU execution only |
| model | Execution mode (direct torch call/aclgraph) |
| status | pytest execution result (PASSED/FAILED/SKIPPED) |
| B...is_continue | All input parameters of this case |
| errmsg | Failure details (truncated to 2000 characters) |
| durations | Operator time consumption (prof mode only) |

Reproducing failed cases:

``` bash
# Batch reproduction (same shape sequence): take random_seed from the CSV
RANDOM_SEED=<random_seed> bash test_run.sh random N

# Single-case value-level reproduction (same shape + same tensor values): take the input parameters and seed from the CSV
TORCH_SEED=<tensor_seed> bash test_run.sh random 1
```

### Environment Variable Summary

| Variable | Function | Applicable Modes |
|------|------|---------|
| RANDOM_SEED | Fixes the random shape sequence seed (if not set, it is automatically generated and recorded in the CSV) | random/random_npu |
| TORCH_SEED | Fixes the tensor value seed (if not set, it is automatically generated for each case and recorded in the CSV) | All |
| RANDOM_CASE_COUNT | Number of random cases (passed through by test_run.sh) | random series |
| SKIP_GOLDEN | =1 skips CPU golden and precision comparison, and executes NPU only | random_npu |
| CSV_FILE | Specifies the CSV output path (automatically set by test_run.sh) | All |
| CSV_APPEND | =1 appends to the CSV | All |
| USE_GRAPH | =true enables the aclgraph mode | All |
