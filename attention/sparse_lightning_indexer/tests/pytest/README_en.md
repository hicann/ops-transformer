# sparse_lightning_indexer Operator Test Framework

English | [简体中文](./README.md)

Based on the pytest test framework, this framework implements the functional verification of the sparse_lightning_indexer operator (candidate consumer / candidate_mode==2):

- **CPU side**: reproduces the operator function to generate golden data (leak-free two-tier mask → topk; when nvalid is less than topk, remaining slots are filled with -1; indices outside the candidates are never output)
- **NPU side**: obtains actual data by directly calling the operator through TorchNPU
- **Accuracy comparison**: reuses the `result_compare_method.check_result` of LIV2 (multiset gate →
  `compare_topk_valid` official rules → hard validation of -1 slots)

The golden infrastructure (GeneralizedLIV2 / test data generation) reuses the tests/pytest directory of the lightning_indexer_v2 operator at the same level (shared golden module). This directory implements only the consumer semantics extension.

## File Structure

- `sparse_lightning_indexer_golden.py`      # benchmark: consumer golden (leak-free mask + topk + rv=1 values) + NPU driver + candidate construction
- `test_sparse_lightning_indexer_single.py` # single-case functional test suite (A2 legacy functions + consumer semantics matrix + RV1 series)
- `test_sparse_lightning_indexer_paramset.py` # single-case parameter configuration (37 sets, enabled only on ascend910b/910_93)
- `test_sparse_lightning_indexer_reject.py` # host validation rejection test suite (C2-C8 / R18, 22 cases)
- `test_sparse_lightning_indexer_model.py`  # model scenario test suite (one-to-one paired with the LIV2 model suite)
- `test_run.sh`                             # execution script (three commands: single/reject/model)
- `pytest.ini`                              # test markers
- `sli_candidate_golden.py`                # CPU benchmark for consumer candidate block-level TopK (pure numpy/torch, no heavy dependencies)
- (Reused ../../lightning_indexer_v2/tests/pytest/: lightning_indexer_v2_golden.py,
   result_compare_method.py — LIV2 benchmark infrastructure and accuracy comparator)

## Single-Case Functional Suite (`test_sparse_lightning_indexer_single.py`)

The parameter sets are located in `test_sparse_lightning_indexer_paramset.py`, with 37 case sets (27 LIV2 base fields +
3 consumer extension fields cand_spec / cand_blocks / cand_block_size):

| Dimension | Cases |
|---|---|
| dtype (legacy) | FP16 / BF16 |
| Four layout combinations (legacy) | BSND×BSND / TND×TND / BSND×PA / TND×PA |
| mask / cmp_ratio (legacy) | mode3 / mode3+cmp2 (residual) / mode3+cmp128 (upper bound) |
| Variable length (legacy) | variable-length seqused_k / uneven TND cu_seqlens |
| topk boundary (C5) | 2048 upper bound / topk greater than the number of reachable positions (remaining slots filled with -1) |
| candidate_block_size, all 6 levels (new feature) | 2 / 4 / 8 / 16 / 32 / 64 |
| candBlocks boundary (C3) | 2048 upper bound / full-coverage degeneration when numBlocks<candBlocks |
| Four candidate source forms (new feature) | self / dirty (immunity to duplicate + out-of-range + -1 dirty data) / missing (missing blocks, nvalid<topk triggers -1 slot filling) / full (all candidates) |
| regime 2 leak-free (core semantics of the new feature) | self level (candidate coverage < topk < reachable) / missing level: all remaining slots are -1, and no index outside the candidates is output |
| return_value=1 special items (new feature, 8 RV1 series sets) | FP16 / BF16 / TND / PA / dirty / missing / mask3+cmp2 / regime2 (all rv=1, values=real scores in fp32) |
| mask3 × candidate interaction (new feature) | dirty candidates + causal |
| PA block boundaries (legacy) | block_size 16 / 1024 |
| G head groups (legacy) | N1=32 / 64 |
| TND variable length + candidates (new feature) | candidate consumption on the TND output layout [T,N2,topk] |

## Host Validation Rejection Suite (`test_sparse_lightning_indexer_reject.py`)

| Rule | Cases |
|---|---|
| C3 invalid candBlocks | 0 (below the lower bound) / 100 (not a multiple of 64) / 2112 (>2048) / explicit shape last dimension 4096 |
| C2 rank/dim validation | BSND q + rank3 candidates / TND q + rank4 candidates / wrong B dimension / wrong S1 dimension |
| C4 invalid blockSize | 12 (not a power of 2) / 1 (<2) / 128 (>64) |
| C6 return_value | rv=2 (∉{0,1}) rejected / rv=1 accepted (both positive and negative sides after rv=1 is opened) |
| C7 block_length | non-null (numel=3) rejected / empty tensor (numel=0) accepted |
| C5 topk | 2049 (>2048) |
| C8 cmpRatio | 3 (not a power of 2) / 256 (power of 2 but >128) |
| R18 arch22 does not consume inputs | max_seqlen_q=4 / seqused_q not null / output_idx_offset not null / metadata not null |

Note: C6 (return_value ∈ {0,1}) is triggered through torch kwargs passing: two cases, rv=2 rejected + rv=1 accepted
(after rv=1 is opened, sparse_values outputs real TopK scores).

## Model Scenarios (BSND, one-to-one paired with the LIV2 model suite)

| Parameter | Value |
|---|---|
| layout | BSND×BSND (generalization: BSND×PA_BBND) |
| N1 (q_head_num) | 32 / 16 / 8 / 4 |
| D (head_dim) | 128 |
| topK | 512 |
| candidate_topk_blocks (candBlocks) | 2048 |
| candidate_block_size | 8 |
| S1 / S2 | 4096 / 4096 (S2 is the compressed K length; generalization: 16K full-coverage boundary) |
| cmp_ratio | 2 |
| sparse_mode (mask_mode) | 0 / 3 |
| Candidate source | cand_spec="self" (constructed in the same way as the reference source row behavior, in the form of a real source→consumer link) |
| Other generalizations | dtype FP16/BF16, variable-length seqused_k with B>1, PA_BBND key |

Case list (10 cases by default; 16K boundary cases are appended when `SLI_MODEL_LARGE=1` is set):

| Case | Coverage Points |
|---|---|
| SLI_MODEL_N1_32_MODE3 | Main specification: B=1, mask_mode=3 causal, cmp_residual_k=[1] |
| SLI_MODEL_N1_16_MODE3 | N1=16, B=2 |
| SLI_MODEL_N1_8_MODE3 | N1=8, B=2 |
| SLI_MODEL_N1_4_MODE3 | N1=4, B=4 |
| SLI_MODEL_N1_32_MODE0 | sparse_mode=0 (no mask), cmp_residual_k=None |
| SLI_MODEL_N1_4_MODE0 | sparse_mode=0, N1=4, B=2 |
| SLI_MODEL_N1_32_MODE3_BF16 | BF16 dtype |
| SLI_MODEL_N1_8_MODE0_BF16 | BF16 + sparse_mode=0 |
| SLI_MODEL_VARLEN_MODE3 | B=2 seqused_k=[4096, 2049] (unaligned S2) |
| SLI_MODEL_PA_MODE3 | BSND q × PA_BBND k (block_table) |
| SLI_MODEL_S2_16K_N1_4_MODE3 | k_seq=16384 (numBlocks==candBlocks full-coverage boundary), requires `SLI_MODEL_LARGE=1` |

## Parameter Limits

- **candidate_block_indices**: REQUIRED, INT32; BSND `[B,S1,N2,candBlocks]`; candBlocks ∈ (0, 2048] and a multiple of 64
- **candidate_block_size**: [2,64] and a power of 2 (8 in this suite); must be consistent with the source side
- **topk**: (0, 2048] (no off fallback); **return_value**: ∈ {0,1}, default 0 (when the value is 1, sparse_values outputs real scores with the same shape as sparse_indices and dtype FLOAT32)
- **Platform**: only ascend910b / ascend910_93

## Environment Setup

1. Confirm that TorchNPU is the latest version
2. Refer to the Attention fused operator Experimental usage instructions to activate the CANN package and the custom operator package

## Running

```bash
# Single-case functional suite (Ascend910B / 910_93)
bash test_run.sh single
python3 -m pytest -rA -s test_sparse_lightning_indexer_single.py -v -m ci

# Specify cases
bash test_run.sh single -C SLI_FP16_01,SLI_CAND_DIRTY_19
SLI_CASE_NAMES=SLI_FP16_01 python3 -m pytest -s test_sparse_lightning_indexer_single.py -m ci

# Host validation rejection suite
bash test_run.sh reject
python3 -m pytest -rA -s test_sparse_lightning_indexer_reject.py -v -m ci

# Model scenario suite (full)
bash test_run.sh model
python3 -m pytest -rA -s test_sparse_lightning_indexer_model.py -v -m ci

# Specify model cases
SLI_CASE_NAMES=SLI_MODEL_N1_32_MODE3 python3 -m pytest -s test_sparse_lightning_indexer_model.py -m ci

# Append 16K full-coverage boundary cases
SLI_MODEL_LARGE=1 python3 -m pytest -s test_sparse_lightning_indexer_model.py -m ci
```

Cases are automatically skipped when the operator is not registered.
