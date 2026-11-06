# lightning_indexer_v2 Operator Test Framework

English | [简体中文](./README.md)

## Function Description

Based on the pytest test framework, this implements functional verification of the lightning_indexer_v2 operator:

- **CPU side**: Reproduce the operator function to generate golden data
- **NPU side**: Directly call the operator through TorchNPU to obtain actual data
- **Accuracy comparison**: Compare the accuracy of CPU and NPU results to verify the operator function
- **Dual-mode execution isolation**: Support two batch modes: direct pytest multi-process execution and shell-level process isolation
- **batch_exec mode**: Filter existing pt files based on the Excel table and directly execute NPU tests (skipping the pt generation step)
- **Performance collection**: Support mounting msprof to collect operator performance data and output a summary
- **Run mode switching**: Support two operator invocation modes: direct eager calls and graph (torch.compile + torchair)

## Current Implementation Scope

### Parameter Restrictions

- **Data formats**:
  - **query_layout**: BSND, TND
  - **key_layout**: BSND, TND, PA_BBND

- **Run modes**:
   - **eager**: Directly call `torch.ops.cann_ops_transformer.lightning_indexer`
   - **graph**: Compile and execute through the `torch.compile` + `torchair` backend (torchair support required)

### Environment Configuration

#### Prerequisites

1. Confirm that TorchNPU is the latest version
2. Refer to the Attention fusion operator Experimental usage instructions to activate the CANN package and custom operator package

#### custom Package Invocation

Invocation through the custom package is supported.

## File Structure

#### pytest File Structure Description

- test_run_sh                                  # Execution script, supporting five commands: single/candidate/candidate_reject/batch/batch_exec
- batch_isolated_run.sh                        # Batch isolated execution script (shell-level process isolation + msprof performance collection)
- lightning_indexer_v2_golden.py               # CPU-side golden implementation of the operator
- lightning_indexer_v2_acl_graph.py            # torchair backend implementation in graph mode
- result_compare_method.py                     # Accuracy comparison between CPU golden and NPU output
- collect_perf_data.py                         # msprof performance data collection and summary
- pytest.ini                                   # Create test markers

Single-case tests:

- test_lightning_indexer_v2_single.py       # Main program for running a single pytest test case
- test_lightning_indexer_v2_paramset.py     # Single-case input parameter configuration (A2/arch22 legacy function matrix, enabled only on ascend910b/910_93)
- test_lightning_indexer_v2_stc.py          # STC white-box case configuration (enabled only on Ascend950)
- test_lightning_indexer_v2_candidate.py    # candidate (two-level TopK candidate source) single-case function suite
- test_lightning_indexer_v2_candidate_reject.py  # candidate attribute host validation rejection cases

Batch tests:

- test_lightning_indexer_v2_batch.py        # Main program for batch case testing and generating the Excel file to save results
- ./batch/lightning_indexer_v2_pt_loadprocess.py    # Read pt files and call the operator to obtain NPU outputs
- ./batch/lightning_indexer_v2_pt_save.py           # Read the Excel table and batch generate case pt files
- ./batch/list_pt_from_excel.py                     # Extract Testcase_Name from Excel and match pt files by name (used in batch_exec mode)
- ./batch/replace_path.py                           # Placeholder replacement for test_lightning_indexer_v2_batch.py

## Architecture

- **single mode**: `liv2_output_acl_graph` calls `liv2_output_single(is_batch=True)` to generate data immediately → executed by `torch.compile` + `torchair`
- **batch mode**: Generate .pt files from Excel → read the pt files to execute NPU tests and accuracy comparison → generate result.xlsx
- **batch_exec mode**: Filter existing .pt files from Excel → skip pt generation → directly execute NPU tests and accuracy comparison → generate result.xlsx
- Both paths share `_liv2_prepare_tensors_and_metadata` and `_liv2_run_compiled_graph` and uniformly use `fullgraph=False`

## Usage

Execute in the pytest folder path:

### Running Test Cases

#### Single-Case Debugging

`test_lightning_indexer_v2_paramset.py` is the A2 (ascend910b/910_93) legacy function matrix parameter set
(29 groups: four dtype/layout combinations, S1/G, S2 boundaries, topk including over-2K (multiples of 1024), mask/cmp_ratio,
variable length, PA block boundaries, and return_value); it is automatically empty in the Ascend950 environment (the STC parameter set is used). The default parameter set is suitable
for temporary debugging; use `-C` to specify a single case first:

``` bash
bash test_run.sh single
bash test_run.sh single -C FP16_01,TND_04
```

The STC parameter set is in `test_lightning_indexer_v2_stc.py` and is used to execute white-box cases by code path (enabled only on
Ascend950):

``` bash
bash test_run.sh single --paramset stc
```

> Known A2 semantic divergence (disabled by default): when TND is combined with seqused_k override (seqused_k < the cu_seqlens difference),
> the arch22 kernel derives the valid k length from the cu_seqlens difference and ignores the seqused_k override (the BSND path responds correctly;
> the 950 kernel responds on both sides, and ACTUAL_31 in the 950 STC is a case of the same form). The corresponding case
> `ACTUAL_TND_SEQUSED_23` is not enabled by default. After the kernel is fixed, set `LIV2_ENABLE_TND_SEQUSED_K=1`
> to enable regression verification.

You can specify one or more cases by using `-C` or `--cases`, with multiple names separated by commas:

``` bash
# Specify a single-combination case
bash test_run.sh single --paramset stc -C META_TRAILING_K_ZERO_41

# Specify a multi-combination case and run all of its expanded nodes
bash test_run.sh single --paramset stc -C S2_16

# Specify one of the expanded nodes; _003 corresponds to k_seq=128
bash test_run.sh single --paramset stc -C S2_16_003

# Run multiple specified cases at a time
bash test_run.sh single --paramset stc -C S1_TOPK_14,META_TRAILING_Q_ZERO_42

# Specify graph mode; eager mode is used by default
bash test_run.sh single --paramset stc -C META_BSND_53_002 -M graph
```

#### Batch Generation and Testing of Cases

##### Method A: Batch Execution by Using test_run.sh

1. Place the case Excel table in the Excel path

2. Modify DEFAULT_EXCEL (Excel table path) and DEFAULT_PT_PATH (pt file storage path) in the configuration area of test_run.sh, or specify them through command line parameters

3. Execute the commands:

``` bash
bash test_run.sh batch                                    # Use the default values in the configuration area (eager mode)
bash test_run.sh batch -E ./excel/test_cases.xlsx        # Specify the Excel path
bash test_run.sh batch -S Sheet1 -P ./pt_path            # Specify the Sheet and pt directory
bash test_run.sh batch -M graph                          # graph mode
bash test_run.sh batch -E ./excel/my.xlsx -S Sheet2 -M graph  # Combined use
bash test_run.sh batch graph                             # Legacy usage: run_mode follows directly
```

4. Default values in the configuration area:

| Variable | Default Value | Command Line Parameter | Description |
|---|---|---|---|
| DEFAULT_EXCEL | `./excel/test_cases.xlsx` | `-E` | Excel case table path (**a specific file name must be specified**; wildcards such as `./excel/*` are not supported) |
| DEFAULT_PT_PATH | `./pt_path` | `-P` | Directory for storing pt files |
| (None) | `Sheet1` | `-S` | Excel Sheet page name |
| (None) | `eager` | `-M` | Run mode (eager/graph) |

#### Batch Execution of Existing pt Files Filtered by the Excel Table (batch_exec Mode)
>
> Only NPU tests and accuracy comparison are executed again; pt files are not regenerated. This mode applies to scenarios where pt files already exist and only accuracy results need to be updated.

1. Confirm that the corresponding .pt files already exist under pt_path (they can be generated in `batch` mode or manually)

2. Execute the commands:

``` bash
bash test_run.sh batch_exec                                    # Default: excel/test_cases.xlsx, Sheet1, pt_path
bash test_run.sh batch_exec -E ./excel/test_cases.xlsx         # Specify the Excel path
bash test_run.sh batch_exec -S Sheet1 -P ./pt_path             # Specify the Sheet and pt directory
bash test_run.sh batch_exec -E ./excel/test_cases.xlsx -M graph  # graph mode
```

3. Execution process:

- Read the `Testcase_Name` column from the Excel table
- Match the corresponding .pt files under pt_path by `<Testcase_Name>.pt`
- Execute NPU tests and accuracy comparison only for the matched .pt files
- Generate the `result.xlsx` test result table
- If a case in Excel has no corresponding .pt file, a warning is output and the case is skipped

4. Differences from the `batch` mode:

|  | `batch` | `batch_exec` |
|---|---|---|
| pt generation | Regenerated each time | Skipped |
| Execution speed | Slower (including pt generation) | Faster |
| Applicable scenarios | First run / parameter changes | Accuracy retest / NPU result update only |

##### Method B: Manual Step-by-Step Execution

1. Generate pt files:

``` bash
python3 batch/lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path
python3 batch/lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path --sheet Sheet1  # Specify the Sheet page
```

2. Execute the test (the pt directory reads `./pt_path` by default, and other directories can be specified through the `LIV2_TESTCASE_DIR` environment variable):

``` bash
python3 -m pytest -rA -s test_lightning_indexer_v2_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

##### Method C: Batch Isolated Execution (Recommended for Performance Collection)

A separate pytest process is started for each case, achieving complete inter-process isolation and preventing a single case crash from affecting other cases.

``` bash
bash batch_isolated_run.sh ./pt_path 0         # Do not collect performance data
bash batch_isolated_run.sh ./pt_path 1         # Collect performance data (mount msprof)
bash batch_isolated_run.sh ./pt_path 0 graph   # graph mode + no performance collection
bash batch_isolated_run.sh ./pt_path 1 graph   # graph mode + performance collection
```

## Excel Case Table Format

`excel/test_cases.xlsx` must contain the following columns (Sheet1):

| Column Name | Type | Example |
|---|---|---|
| Testcase_Name | str | `test_case_01` |
| batch_size | int | `8` |
| q_seq | int | `15` |
| k_seq | int | `111` |
| q_t_size | int | `8` |
| k_t_size | int | `15` |
| q_head_num | int | `64` |
| k_head_num | int | `1` |
| head_dim | int | `128` |
| block_size | int | `512` |
| block_num | int | `8` |
| qk_dtype | str | `FLOAT8_E4M3FN` / `INT8` / `HIFLOAT8` |
| cu_seqlens_q | None/str | `None` or `"[0, 1]"` |
| cu_seqlens_k | None/str | `None` or `"[0, 1]"` |
| seqused_q | None/str | `None` or `"[3,3,3,3,3,3,3,3]"` |
| seqused_k | str | `"[28,24,80,96,47,76,0,111]"` |
| cmp_residual_k | None/str | `None` or `"[0,0,0,0,0,0,0,0]"` (required when cmp_ratio>1)|
| max_seqlen_q | int | `-1` |
| layout_q | str | `BSND` / `TND` |
| layout_k | str | `PA_BBND` |
| topk | int | `512` |
| mask_mode | int | `0` / `3` |
| query_datarange | str | `"[-448,448]"` |
| key_datarange | str | `"[-20,20]"` |
| weights_datarange | str | `"[-123,123]"` |
| cmp_ratio | int | `1` / `4` |
| return_value | int | `0` / `1` |
| output_idx_offset | None/str | `None` or a list string |

**Precautions**:

- When `cmp_ratio > 1` and `sparse_mode != 0`, `cmp_residual_k` is required (a list with length = batch_size)

## Output Files

| File | Description |
|---|---|
| `result.xlsx` | Test results (accuracy, parameters, and so on) |
| `result_perf.xlsx` | Test results + performance data (msprof mode only) |
| `batch_summary.log` | Detailed batch execution log |
| `batch_fail_list.log` | List of failed cases |
| `PROF_*/` | Directory of raw msprof performance data |

## candidate (Two-Level TopK candidate source) Single-Case Function and Host Rejection Suites

In addition to the model scenario suite, the candidate mode (`candidateTopkBlocks != -1`) has a lightweight single-case matrix and a
host validation rejection suite, focusing on the candidate attribute surface and layout generalization (including TND):

### File Description

- `test_lightning_indexer_v2_candidate.py`  # candidate single-case function suite (26 cases)
- `test_lightning_indexer_v2_candidate_reject.py`  # candidate attribute host validation rejection cases (11 cases)

### Function Suite Coverage (`test_lightning_indexer_v2_candidate.py`)

| Dimension | Cases |
|---|---|
| candidate_block_size, all 6 tiers | 2 / 4 / 8 / 16 / 32 / 64 |
| candidate_topk_blocks boundaries | 64 (lower bound) / 2048 (upper bound) / degeneration to full coverage when numBlocks<candBlocks |
| layout generalization | BSND×BSND / TND×TND / BSND×PA_BBND / TND×PA_BBND (TND outputs [T,N2,K]) |
| mask / cmp_ratio | mode3+cmp2 (including residual) / cmp128 upper bound |
| dtype / variable length | BF16 / B>1 variable-length seqused_k (including 2049 misalignment) |
| topk boundaries | 1 (lower bound) / 2048 (upper bound when candidate is enabled) |
| return_value dimension (now unrestricted) | rv=1: BSND baseline / TND variant / mask3+cmp2 (values shape==indices, float32, isclose under the same-family criterion); rv=0 always-empty (0,) regression |
| rv consistency | Dual runs with the same input at rv=0/rv=1: sparse_indices and candidate_block_indices are bit-identical |
| Others | S2 tail block (513) / large S1 row count / off (-1) regression (output (0,) and sparse unchanged) |

The three per-case checks are the same as those of the model suite: sparse regression (the source must not change the position-level topk),
candidate_block_length always empty, and candidate_block_indices set comparison.

### Host Rejection Suite Coverage (`test_lightning_indexer_v2_candidate_reject.py`)

| Rule | Cases |
|---|---|
| C3' invalid candBlocks | 100 (not a multiple of 64) / 2112 (>2048) / -2 (negative value other than -1) |
| C4' invalid blockSize | 12 (not a power of 2) / 1 (<2) / 128 (>64) / also rejected when off (always validated) |
| V1 topk linkage | candidate enabled + topk=3072 (>2048) |
| V2 rv value range violation | candidate enabled + return_value=2 (TORCH_CHECK "Return_value only supports 0 or 1") |
| Valid reverse cases | candBlocks=64 + bs=2 + topk=64 must be accepted; candidate enabled + return_value=1 (now unrestricted) must be accepted |

### Running

```bash
# Full suite (Ascend910B / 910_93)
bash test_run.sh candidate
bash test_run.sh candidate_reject

# Specify cases
bash test_run.sh candidate -C LIV2_CAND_BS_2,LIV2_CAND_TND_TND
# Or use pytest directly
LIV2_CAND_CASE_NAMES=LIV2_CAND_OFF_REGRESSION python3 -m pytest -s test_lightning_indexer_v2_candidate.py -m ci
```

## candidate (Two-Level TopK candidate source) Model Scenario Suite

The two-level TopK candidate source (source) mode enabled when `candidateTopkBlocks != -1` has an independent model scenario test suite, which does not affect the preceding single/paramset/batch suites.

### File Description

- `liv2_candidate_golden.py`  # Pure CPU implementation of the candidate golden (`select_candidate_blocks_ref` with dual numpy/torch implementations, depending only on numpy+torch; it can be imported for self-check in an environment without NPU)
- `test_lightning_indexer_v2_model.py`  # candidate model scenario test suite (10 default cases + `LIV2_MODEL_LARGE=1` to append heavy boundary cases)
- `lightning_indexer_v2_golden.py` (candidate section at the end)  # candidate golden + NPU driver (`liv2_output_candidate_single` / `liv2_candidate_host_reject` 950 rejection probe / `resolve_candidate_op`)
- `result_compare_method.py` (candidate section at the end)  # `check_result_candidate` candidate comparator

### Model Scenario Specifications (BSND)

| Parameter | Value |
|---|---|
| layout | BSND×BSND (generalization: BSND×PA_BBND) |
| N1 (q_head_num) | 32 / 16 / 8 / 4 |
| D (head_dim) | 128 |
| topK | 512 |
| candidate_topk_blocks | 2048 |
| candidate_block_size | 8 |
| S1 / S2 | 4096 / 4096 (S2 is the post-compression K length; `LIV2_MODEL_LARGE=1` appends the 16K full-coverage boundary and 24K real filtering) |
| cmp_ratio | 2 |
| sparse_mode (mask_mode) | 0 / 3 |
| Other generalizations | dtype FP16/BF16, B>1 variable-length seqused_k, PA_BBND key |

Case list (10 cases by default; `LIV2_MODEL_LARGE=1` appends 2 heavy boundary cases):

| Case | Coverage Point |
|---|---|
| LIV2_MODEL_N1_32_MODE3 | Main specification: B=1, mask_mode=3 causal, cmp_residual_k=[1] |
| LIV2_MODEL_N1_16_MODE3 | N1=16, B=2 |
| LIV2_MODEL_N1_8_MODE3 | N1=8, B=2 |
| LIV2_MODEL_N1_4_MODE3 | N1=4, B=4 |
| LIV2_MODEL_N1_32_MODE0 | sparse_mode=0 (no mask), cmp_residual_k=None |
| LIV2_MODEL_N1_4_MODE0 | sparse_mode=0, N1=4, B=2 |
| LIV2_MODEL_N1_32_MODE3_BF16 | BF16 dtype |
| LIV2_MODEL_N1_8_MODE0_BF16 | BF16 + sparse_mode=0 |
| LIV2_MODEL_VARLEN_MODE3 | B=2 seqused_k=[4096, 2049] (S2 misaligned) |
| LIV2_MODEL_PA_MODE3 | BSND q × PA_BBND k (block_table) |
| LIV2_MODEL_S2_16K_N1_4_MODE3 | k_seq=16384 (numBlocks==candidate_topk_blocks full-coverage boundary); requires `LIV2_MODEL_LARGE=1` |
| LIV2_MODEL_S2_24K_N1_4_MODE3 | k_seq=24576 (numBlocks>candidate_topk_blocks block-level real filtering); requires `LIV2_MODEL_LARGE=1` |

Three per-case checks:

1. `sparse_indices` vs golden (`check_result`; the source must not change the position-level topk);
2. `candidate_block_length` is always empty (0,) (reserved interface);
3. `candidate_block_indices` vs golden (`check_result_candidate`; row-level block number set + hard check of the -1 slot count).

### candidate Comparison Contract (`check_result_candidate`)

The candidate output is an **unordered block number set** (the merge sort tie-break does not guarantee consistency with torch), and the comparison contract has five layers:

1. Multiset gate: the sets of valid (non -1) block numbers are equal → PASS (order-insensitive);
2. Boundary tolerance: the sets are not equal, but the golden blkScore of each swapped block pair is within the `thres` relative tolerance → PASS (bf16 tie arbitration);
3. -1 slot hard check: the number of -1 values in each row must be exactly equal; otherwise, FAIL;
4. Duplicate block hard check: duplicate block numbers appear in a row → FAIL;
5. Pin block hard check: the golden +inf (the block containing the latest reachable token) must appear in the NPU output row; otherwise, FAIL.

### Running

```bash
# Full suite (Ascend910B / 910_93; the candidate operator must be compiled)
python3 -m pytest -rA -s test_lightning_indexer_v2_model.py -v -m ci

# Specify cases
LIV2_CASE_NAMES=LIV2_MODEL_N1_32_MODE3 python3 -m pytest -s test_lightning_indexer_v2_model.py -m ci

# Append the 16K full-coverage boundary + 24K real filtering heavy cases
LIV2_MODEL_LARGE=1 python3 -m pytest -s test_lightning_indexer_v2_model.py -m ci
```

> Note: candidate source supports only arch22 (Ascend 910B / Ascend 910_93); on arch35 (Ascend 950),
> candidate-on is rejected by host validation (the tiling entry guard returns GRAPH_FAILED), and this suite is directly
> skipped in the 950 environment. It is automatically skipped when the candidate torch operator is not built/registered with the package (`candidate_ops_unavailable_reason`).
