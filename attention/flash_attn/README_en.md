# FlashAttn
English | [简体中文](./README.md)

## Function Description

- Operator function: implements self-attention computation based on the FlashAttention algorithm, unified for training and inference. Only non-quantization scenarios are supported.

- Computation formula:

  The forward computation formula of self-attention is as follows:

  $$
  Attention(Q,K,V)=Softmax(SoftmaxScale * (Q \cdot K^T))V
  $$

  where $Q \in (Q_S, Q_N, D)$, $K, V \in (KV_S, KV_N, D)$, and $D$ is the head dim. By default, $SoftmaxScale=1.0$. The standard implementation typically uses $1/\sqrt{D}$ as the scaling factor.

  The FlashAttention algorithm computes the $Q_S \times KV_S$ attention matrix block by block (tile), avoiding instantiation of the full attention matrix. The memory complexity decreases from $O(Q_S \cdot KV_S)$ to $O(tile_M \cdot tile_N)$.

- **head_dim support specifications**: the head_dim of QK and the head_dim of V can differ. They are expressed by the last dimension (D dimension) of each of the q/k/v tensors (the main operator has no separate parameter). Currently supported (QK D, V DV) combinations: **(64,64), (72,72), (128,128), (256,256), (192,128)** (192/128 is the MLA non-absorbed form, where Q/K use the 192 dimension of bmm1 and V and the output use the 128 dimension; 72 uses config 8/9 (the OriginalDV72 clone of the D=128 tier), see §3). For the semantics and contract of the `head_dim_v` parameter of the metadata operator, see the [torch interface documentation](docs/torchapi_flash_attn_en.md).

## Quick Start

### 1. Compiling and Installing the custom Package

Complete script (integrates the compilation and installation steps; `CANN_DIR` is the CANN installation root directory. Adjust it as needed):

```bash
# Prerequisite: load the CANN environment
source ${CANN_DIR}/cann/set_env.sh

# Clean up previous build artifacts (avoid residue that affects incremental compilation)
rm -rf ./build ./build_out
rm -rf ${CANN_DIR}/vendors

# Compile (flash_attn and flash_attn_metadata must be compiled together to generate kernel variants for all tiling keys)
bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata

# Install
cd build_out
./cann-ops-transformer-*.run --install-path=${CANN_DIR}
```

Compilation artifact: `build_out/cann-ops-transformer-custom_linux-x86_64.run`

**Optional parameter descriptions**:

- **`-j` (limits the number of parallel threads)**: by default, compilation runs in parallel on all machine cores. If the machine memory is insufficient, or the actual core count limited by cgroup is smaller than the value reported by `/proc/cpuinfo` (for example, the machine reports 128 cores but cgroup limits 32 cores), compilation may run into OOM or failures. In this case, explicitly specify a smaller value:

  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata -j16
  ```

- **`--op_debug_config dump_cce` (retains intermediate kernel compilation artifacts)**: kernel debugging (for example, checking assembly on the device) requires retaining `.cce` intermediate artifacts. Append the following to the compilation command:

  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata -j16 --op_debug_config dump_cce
  ```

- **`--tiling_key` (compiles only the specified tiling key variants)**: the default full compilation generates kernel variants for all tiling keys and takes a long time. When you only need to debug or verify specific layout and configuration combinations (for example, only BSND/BNSD/TND × no mask × D=128, which uses the DN template), compile only the corresponding variants to speed up the process:

  ```bash
  bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata --tiling_key="67239936;67239937;67239938"
  ```

  For the encoding meaning of each tiling key, see [Developer Guide §5](#5-tilingkey).

  **How to find the tiling key from runtime logs**: enable ASCEND DEBUG logs when running cases (`export ASCEND_SLOG_PRINT_TO_STDOUT=1` and `export ASCEND_GLOBAL_LOG_LEVEL=0`), and search the logs for the **last** occurrence of `kernel_name=FlashAttn_`. The format is `FlashAttn_<hash>_<tiling_key>`, and the number at the end is the tiling key of the current case. For example, the log contains:

  ```
  StreamLaunchKernelV2: ... kernel_name=FlashAttn_7a9b290f0be5028cb35fc6d409953d1d_262144 ...
  ```

  Then the tiling key of this case is `262144`. Fill it into `--tiling_key` to compile only this variant.

  You can also confirm it through tiling logs: search for `GenTilingKey`. The log directly prints the tiling key value, for example:

  ```
  [flash_attn_tiling.cpp:198] ... GenTilingKey ... The tilingkey is 262144.
  ```

### 2. Building and Installing the torch Extension Package

This step is mandatory before you use the torch interface. Build the torch extension whl in the repository root directory and install it:

```bash
# Prerequisite: load the CANN environment
source ${CANN_DIR}/cann/set_env.sh

# Clean the torch extension cache (~ is the current user home directory, which must match the user who installs and runs torch, to avoid loading stale build artifacts)
rm -rf ~/.cache/torch_extensions/*

# Build the torch extension whl (full package; the package name cann_ops_transformer stays unchanged; the whl is output to build_out/)
bash build.sh --torch_extension --soc=ascend950

# Install
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

**Verification after installation**:

```bash
python3 -c "from cann_ops_transformer.ops import flash_attn, flash_attn_metadata; print('ok')"
```

### 3. Interface Invocation

- **torch interface** (function prototypes, parameter descriptions, and return value descriptions of `flash_attn_metadata` + `flash_attn`): [torchapi_flash_attn.md](docs/torchapi_flash_attn_en.md)

Invocation consists of two steps: first use `flash_attn_metadata` to generate load balancing metadata, then call the `flash_attn` main operator. For complete invocation examples (with code), see the [Calling Examples](docs/torchapi_flash_attn_en.md#example) section in the interface documentation.

The import path is the same as the installation package name (the full package built in step 2):

```python
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata
```

### 4. Running the Operator Accuracy Test on the NPU

Prerequisite: load the CANN environment and the installed custom package environment:

```bash
source ${CANN_DIR}/cann/set_env.sh
source ${CANN_DIR}/vendors/custom_transformer/bin/set_env.bash
```

Run the pytests in the repository (cases are in `tests/pytests/`, covering the three backends npu/gpu/cpu):

```bash
cd attention/flash_attn/tests/pytests
python3 test_flash_attn.py --case_files functional_redline_infer --case_id Decode_Unquant_BF16_BF16_L0_1_5_1_1_2048_BNSD_BSND_000004
python3 test_flash_attn.py --device_id 0            # Run all functional cases
python3 test_flash_attn.py --device_id 0 --case_id <id>   # Run a single case
python3 test_flash_attn.py --device_id 0 --use_gpu --gpu_device 0   # Compare against the GPU flash_attn library
```

For the pytests framework, see [tests/pytests/readme.md](tests/pytests/readme_en.md) (functional/redline/perf case matrix and result CSV output).

### 5. Performance Test

Case files: `tests/pytests/test_cases/performance_redline_train.py` and `performance_redline_infer.py` (220+ performance cases each for train/infer).

```bash
cd attention/flash_attn/tests/pytests

# Collect performance data in batches (train performance case set, output to the specified directory)
python3 test_flash_attn.py --case_files performance_redline_train --perf_mode --perf_output ./perf_out

# Execute one by one (--one_by_one runs cases one by one for easy observation and interruption)
python3 test_flash_attn.py --perf_mode --one_by_one
```

Performance-related parameters: `--perf_runs` (number of collection rounds, default 5), `--perf_cold_thr` (S1 boundary between cold/hot cases, default 16), and `--perf_output` (output directory, default `./perf_output`). For result parsing and report generation, see `tests/pytests/utils/perf_runner.py` and `perf_parser.py`.

### 6. Running tiling Unit Tests

```bash
bash build.sh --ops=flash_attn,flash_attn_metadata --soc=ascend950
cd build && ctest -R flash_attn_tiling --output-on-failure
```

Case definitions: `tests/ut/op_host/arch35/test_flash_attn_tiling.csv`. Additional inferShape and inferDtype unit tests: `tests/ut/op_host/test_flash_attn_shape_infershape.cpp` and `test_flash_attn_dtype_infershape.cpp`.
## Developer Guide

### 1. Basic Concepts

#### 1.1 MHA and GQA

- **MHA (Multi-Head Attention)**: Q, K, and V each have a group of heads; Q_N = KV_N, and each Q head performs attention with its corresponding KV head.
- **GQA (Grouped Query Attention)**: Q_N is an integer multiple of KV_N, that is, g = Q_N / KV_N, and a group of g Q heads shares one copy of KV. GQA reduces the KV device memory and computation (when Q_N is much larger than KV_N, the KV cache shrinks significantly).
- This operator requires Q_N % KV_N == 0 (GQA scenario). In the kernel, `gSize = g`, and tasks are unrolled in a three-level loop over (bN2, gS1, s2), where bN2 is the flat index of batch × KV_N and gS1 is the flat index of the Q sequence × g.

#### 1.2 PagedAttention scenario and layout_kv values

- **PagedAttention**: stores the KV cache in pages by block (block_size tokens) and indexes physical blocks through `block_table`, avoiding excessively large contiguous device memory reserved for the KV cache, and supports dynamic allocation and reuse. It is typically used in inference scenarios (long-sequence decode where the KV cache grows).
- How to enable: set `layout_kv` to a PA layout and pass in `block_table` (INT32, shape (B, max_blocks_per_seq)) at the same time. block_size is inferred from the PA layout of KV; the value must be ∈ [16, 1024] and an integer multiple of 16.
- `layout_kv` values and the corresponding KV shapes:

| layout_kv | KV shape | Description |
|---|---|---|
| BSND / BNSD / TND | (B, S, N, D) / (B, N, S, D) / (T, N, D) | Continuous KV, non-paged |
| PA_BBND | (num_blocks, block_size, KV_N, D) | Arranged as BSND within a block |
| PA_BNBD | (num_blocks, KV_N, block_size, D) | Arranged as BNSD within a block |
| PA_NZ | (num_blocks, KV_N, D/16, block_size, 16) | NZ tiling format, requires head_dim to be a multiple of 16 (the fractal granularity of the D axis is 16 elements) |

- PA scenario constraints: in PA scenarios, `seqused_kv` is required; passing in `block_table` in non-PA scenarios directly triggers an error. When `layout_kv` is PA, `layout_q`/`layout_out` remain BSND/BNSD/TND unchanged.

#### 1.3 Variable-length sequences (TND)

- **TND layout**: concatenates the tokens of all sequences in a batch into one long sequence (total length Q_T / KV_T), with shapes (Q_T, Q_N, D) and (KV_T, KV_N, D), eliminating padding waste. It is used in scenarios where sequence lengths within a batch differ.
- How to enable: set `layout_q` (and `layout_kv`) to TND and pass in `cu_seqlens_q`/`cu_seqlens_kv` at the same time to describe the start position of each sequence.
- In TND scenarios, the `max_seqlen` attribute does not participate in the actual split (cu_seqlens prevails), and `layout_out` must be TND.
- "Packed Sequence" in the functional description refers to the TND scenario.

#### 1.4 cu_seqlens and seqused

- **cu_seqlens_q / cu_seqlens_kv**: INT32 1-D, shape (B+1,), monotonically non-decreasing, with the first element being 0; indicates the cumulative start/end of each sequence (the TND layout uses it to restore the boundaries of each sequence). Only the TND layout allows passing it in.
- **seqused_q / seqused_kv**: INT32 1-D, shape (B,), indicating the actual effective sequence length of each batch (truncating redundant padding). In PA scenarios, `seqused_kv` is required to determine the actual KV length of each batch.
- The meanings of these two fields: cu_seqlens locates sequence boundaries (TND), and seqused truncates lengths (PA/variable-length); when both are passed in, the actual shorter one takes effect.

#### 1.5 Mask (mask_mode and attn_mask)

- The **mask_mode** attribute selects the mask mode: 0 = full computation (no mask), 3 = causal (lower triangle), and 4 = window (band, with win_left/win_right tokens before and after).
- **attn_mask input**: an INT8 template matrix with the fixed shape (2048, 2048); the kernel extracts the required sub-blocks from it tile by tile, adapting to any S1/S2. When mask_mode=3/4, you must pass in attn_mask; when mask_mode=0, passing in attn_mask is prohibited and win_left/win_right must be -1.
- When mask_mode=4 (window), win_left/win_right define the window width, and the kernel skips invalid tiles accordingly.

#### 1.6 LSE (softmax_lse)

- **LSE (Log-Sum-Exp)**: when `return_softmax_lse=True`, outputs the log denominator of the softmax for each row: lse = log(Σ e^{s-m}) + m, where m is the max value within the row.
- Output shape: BSND/BNSD → (B, Q_N, Q_S); TND → (Q_T, Q_N); the dtype is FLOAT32.
- Typical uses: cross-batch KV cache concatenation (such as probability fusion in speculative decoding) and softmax state reconstruction when training resumes from a checkpoint.

#### 1.7 Row invalid (RowInvalid)

- **Concept**: when all tokens in a row (one row of Q tokens in a Q head) are excluded by the mask (outside the window, truncated by seqused, or the mask is all 1) and the row contains no valid elements participating in the computation, the row is an **invalid row**. The kernel uses whether the softmax max of the row remains `-inf` (softmaxMax == -inf, meaning no valid s value exists) as the determination criterion.
- **Output when a row is invalid**:
  - **attn_out**: invalid-row outputs are **cleared to zero** (written as 0). `RowInvalidUpdateVF` uses `Select` to write rows with max == -inf entirely as 0 (`vf_flashupdate_new.h`); in the FD stage, `InvalidRows`/`InvalidMaskRows` clear them to zero before writing out.
  - **softmax_lse**: invalid-row outputs are **+inf** (3e+99). The code does not rely on mathematical derivation (log(sum) + max would yield -inf) but handles it explicitly: `ComputeLseOutputVF` first computes `lse = log(sum) + max`, then uses `Select` to set rows with max == -inf to `infValue = 3e+99` (`vf_flashupdate_new.h:522-523`); in the FD stage, `ComputeScaleValue` handles it the same way (`vf_flash_decode_arch35.h:312-313`).

#### 1.8 Sink (learnable sink)

- **Sink**: adds an extra term `e^{sink - m}` to the softmax denominator (sink also participates in the max computation of m) to improve the numerical stability of some models (such as the Qwen/GLM series).
- The interface of this operator has reserved the `sinks` input (FLOAT32, shape (Q_N,)), but **the current version does not support it yet**: passing it in triggers an error intercepted by the checker ("sinks is currently not supported"); support will be opened in a later version.

#### 1.9 FlashDecode

- **FlashDecode**: an optimization for the decode stage when S1 is very small (such as 1) and S2 is very long—it splits along the KV direction into split-K segments computed in parallel by multiple AIV cores and then reduces across cores, avoiding serial scanning of the entire KV on a single core.
- It has two stages: in the FA stage, each split independently computes the local softmax max/sum and partial outputs (written to the workspace); in the FD stage, global max/sum are reduced across splits, scale is recomputed, and a weighted sum produces the final output.
- It is enabled automatically (determined by `enableFlashDecode` in the metadata) and requires no user configuration; `return_softmax_lse` and FlashDecode can be used together (LSE is produced directly from the FD reduction results).

### 2. Code layering

The operator consists of four parts from the call entry to the execution engine: the **torch API layer** (the user Python entry), the **AICPU side** (the accompanying `flash_attn_metadata` load balancing), the **Host side** (the interface/definition/tiling of the main operator), and the **Kernel side** (AIC+AIV heterogeneous execution), which share the common infrastructure of `attention/common/`.

#### 2.1 torch API layer (user entry)

| File | Key content |
|---|---|
| `torch_extension/flash_attn.py` | PyTorch operator schema (the `flash_attn_metadata` and `flash_attn` interfaces), the FakeTensor meta implementation (shape/dtype inference), and the NPU implementation of `PrivateUse1` dispatch; contains the metadata size formula `((36+72)*B*KV_N+1)*16` (4096B aligned) and batch inference |
| `torch_extension/graph_convert_flash_attn.py` | Dispatch conversion for the torchair graph mode |
| `torch_extension/csrc/` | C++ binding layer, hosting `OpBuilder` compilation and underlying calls |

For external function prototypes, parameter/return value descriptions, and call examples, see the interface document `torchapi_flash_attn.md` (not repeated here).

#### 2.2 AICPU side (flash_attn_metadata)

| Location | Key content |
|---|---|
| `attention/flash_attn_metadata/op_kernel_aicpu/` | AICPU operator entity: reads information such as cu_seqlens/seqused/mask and calls SectionStreamK to generate the task-split metadata |
| `attention/common/op_kernel/load_balance/section_stream_k/` | SectionStreamK algorithm library (shared with quant_flash_attn): splits sections based on the 96MB L2 budget, cost model `6*ceil(m/16)+10*ceil(s2/64)`, the FA segment (assigns bN2/gS1/s2 ranges by core) and the FD segment (cross-core split-K row assignment) |
| Inputs/outputs | The inputs are parameters such as shape/head count/mask; the output is the metadata tensor (header + FA segment + FD segment, consumed by the main operator kernel) |

The AICPU side and the Host side tiling share the level selection of `fa_adjust_sinner_souter_arch35.h`/`fa_adjust_sinner_souter_arch92.h`, ensuring a consistent split standard on both sides.

#### 2.3 Host side (3 layers)

**① Interface layer (op_api)**

| File | Key content |
|---|---|
| `op_api/aclnn_flash_attn.cpp` | The external two-stage `aclnnFlashAttn` interface (GetWorkspaceSize + execution), forwarding through a thin shell; contains placeholder tensor handling for sinks/softmax_lse and the extern declaration of the inner interface |

**② Operator definition and shape inference**

| File | Key content |
|---|---|
| `op_host/flash_attn_def.cpp` | Operator definition: 9 inputs (q/k/v/block_table/cu_seqlens/seqused/sinks/attn_mask/metadata) + 2 outputs (attn_out/softmax_lse) + 10 attributes, including dtype/format/default values and the ascend950 configuration |
| `op_host/flash_attn_infershape.cpp` | Output shape inference: attn_out follows the layout; softmax_lse follows (B, Q_N, Q_S) or (Q_T, Q_N) for TND; **does not check value ranges** (left to the checker in tiling) |

**③ Tiling layer** (key point: the three-layer call at the entry)

**Entry**: `TilingFlashAttn` (`op_host/flash_attn_tiling.cpp:40-66`, registered through `IMPL_OP_OPTILING(FlashAttn).Tiling(TilingFlashAttn)`). Its function body is exactly the **skeleton of the three-layer parser → check → doTiling calls**:

```cpp
ge::graphStatus TilingFlashAttn(gert::TilingContext *context)
{
    ...
    FaTilingInfo faInfo;
    FaInfoParser faInfoParser(context);
    if (faInfoParser.Parse(faInfo) != ge::GRAPH_SUCCESS) {      // Layer 1: parser
        return ge::GRAPH_FAILED;
    }
    FAChecker faChecker;
    faChecker.Init(faInfo);
    if (faChecker.Process(faInfo) != ge::GRAPH_SUCCESS) {       // Layer 2: check
        return ge::GRAPH_FAILED;
    }
    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &faInfo);  // Layer 3: doTiling
}
```

**Layer 1: parser (parameter parsing)** — `FaInfoParser::Parse` (`op_host/flash_attn_tiling_info_parser.cpp`)

```
GetOpParaInfo     Reads inputs/attributes (q/k/v/block_table/cu_seqlens/seqused/...)
Layout parsing    Determines BSND/BNSD/TND/PA and strides; a GraphStatus failure raises an error
GetKvStorageMode  KV storage mode (continuous/PA_BBND/PA_BNBD/PA_NZ)
ParseAxisInfo     GetN1/N2 → GetQueryTSize → GetQkHeadDim/ValueHeadDim
                  → GetBSize → GetS1Size → GetS2Size → GetGSize
ParseFeatureInfo  mask/sinks/lse/maxSeq
Output: faInfo (the unified information struct); subsequent tiling only reads it
```

**Layer 2: check (validity check)** — `FAChecker::Process` (`op_host/checkers/fa_checker.cpp:85-108`)

```
Runs 9 groups of validators on faInfo × four rounds: single parameter → existence → feature → combination
Any failed check returns an error and blocks subsequent tiling
(base/common/mask/metadata/paged_attention/seq_len/sinks/softmax_lse each handles one group of constraints)
```

**Layer 3: doTiling (split computation)** — `FiaTilingRegistry::DoTilingImpl` (common) → `FlashAttnTilingImplArch35` (arch35)

```
FiaTilingRegistry::DoTilingImpl (common/op_host/fia_tiling_templates_registry.h:94-126)
  Dispatches through a lookup table by NpuArch → FlashAttnTilingImplArch35::DoOpTiling (inherited from flash_attn_tiling_base.h, DAV_3510, priority 1)
    SetPlatMemoryInfo   Platform information: aiv/aic core counts, CV_RATIO=2, UB/L0/L1 capacities, sys workspace
    InitImplParam       Existence check on cu_seqlens/seqused
    SplitPolicy         Window handling + AdjustSinnerAndSouter + CalcNumBlocks (coreNum)
    FillTiling          ComputeTilingData: sizes required for the tiling data and metadata
    CalcWorkspaceSize   sys area + extra BMM2 staging for D>128 + PA alignment + FD area (see §10)
    GenTilingKey        UpdateTilingKeyInfo → GET_TPL_TILING_KEY
Output: tiling data / workspace size / tiling key / core count
```

**Output delivery** (at run time, passed to the kernel through TilingContext): `SetBlockDim` (core count), `SetTilingKey` (kernel variant selection), `SetWorkspaceSize`, and `SetTilingData` (raw struct).

The responsibility boundaries of the three layers: **the parser only performs parameter parsing** (it maps GE/Ascend parameters to the operator-internal `faInfo`), **the check layer only performs validity checks** (it produces no tiling results and is used to detect invalid input combinations), and **doTiling only performs split computation** (it trusts the results of the previous two layers). The three layers run in strict order and none can be omitted—after parsing and before computation, the check layer intercepts invalid inputs.

#### 2.4 Kernel side: entry + 4 layers

| Layer | Location | Key content |
|---|---|---|
| ① Entry layer | `op_kernel/flash_attn.cpp` | The only `__global__` entry: routes to ND, merged-axis DN, and non-merged-axis DN based on the `templateId` of the tiling key, and uses the `__DAV_CUBE__` macro to distinguish the dual AIC/AIV compilation (`KERNEL_TYPE_MIX_AIC_1_2`); the same source compiles into two variants, the Cube side and the Vector side |
| ② Scheduling framework layer | `op_kernel/flash_attn_kernel_dn.h` / `op_kernel/flash_attn_kernel_nd.h` / `op_kernel/flash_attn_kernel_dn_unmerged.h` | Task-level pipeline framework: `Process()` loops over metadata sections to run `FlashAttention` ((bN2, gS1, s2) triple loop + `CreateTask/ExecuteTask`, PRELOAD_N=2 prefetch) and `FlashDecode` (AIV performs cross-core reduction, with `SyncAll` at both ends) |
| ③ Compute block layer (AIC/AIV separation) | `arch35/flash_attn_block_cube_dn.h` / `arch35/flash_attn_block_cube_nd.h` (AIC side), `flash_attn_block_vec_dn.h` / `flash_attn_block_vec_nd.h` (AIV side), `flash_attn_block_vec_flashdecode.h` (FD reduction) | AIC: BMM1/BMM2 (multi-level L0A/L0B/L0C buffers, four-stage MTE2→MTE1→M→FIX pipeline); AIV: softmax VF (ProcessVec1) and output accumulation (ProcessVec2), plus the cross-split reduction for FD |
| ④ Common API layer | `attention/common/op_kernel/arch35/` (`flash_attention_score_common_regbase_arch35.h` and so on), `utils/` (`flash_attn_type.h`, `flash_attn_common_def.h`, `attenmask_m_gs1.h` (merged-axis GS/SG), `attenmask_m_s1.h` (non-merged-axis S1_ONLY)) | Instruction-level VF operator library (`ProcessVec1VfDn`/`FusedExpSub`/`FlashUpdateNew` and more) and type/layout/mask utilities; shared with operators such as flash_attention_score |

> For finer kernel-internal divisions, see the following chapters: kernel layer scheduling framework (§7), block cube layer (§8), block vector layer (§9), FD block (§10), and common layer (§11).

#### 2.5 Relationships between layers

```
torch API layer (flash_attn.py, user entry)
  → AICPU side (flash_attn_metadata: SectionStreamK generates the metadata)
  → Host side: interface layer (aclnn/L0)
      → def/infershape (definition and shape)
      → tiling layer (parser → check → doTiling)
          ↓ TilingContext delivers: tiling data / workspace size / tiling key / core count
      → Kernel side: flash_attn.cpp entry → flash_attn_kernel_dn/flash_attn_kernel_nd/flash_attn_kernel_dn_unmerged scheduling framework → flash_attn_block_cube_* + flash_attn_block_vec_*
          → attention/common VF instruction library (instruction level)
```

Key point: the kernel is unaware of global shapes and executes only according to **metadata task ranges**; at compilation, the **tiling key** (template parameters) determines the variant, and at run time, the **tiling data** determines the split. The two come from the compilation list and the split computation of doTiling respectively.

### 3. Basic Block

The core of FA computation splits the attention matrix into blocks and performs online computation block by block. The compilation-time config and the runtime shape jointly determine the basic block parameters.

**arch35 config mapping** (one of the compilation-time template parameters, see §5 TilingKey):

`D/DV` is the template width of `ConfigParams.d/dv`; `originalDV` is the real head dim of V/the output. The original dimensions of Q/K are given by the runtime shape and cannot be replaced by `originalDV`. For example, when (QK D, V DV)=(192,128), `originalDV=128`.

| config | sOuter | sInner | D (template) | DV (template) | originalDV (original V/output) |
|---|---|---|---|---|---|
| 0 | 64 | 128 | 64 | 64 | 64 |
| 1 | 32 | 256 | 64 | 64 | 64 |
| 2 | 64 | 128 | 128 | 128 | 128 |
| 3 | 32 | 256 | 128 | 128 | 128 |
| 4 | 64 | 128 | 256 | 256 | 256 |
| 5 | 32 | 256 | 256 | 256 | 256 |
| 6 | 64 | 128 | 192 | 128 | 128 |
| 7 | 32 | 256 | 192 | 128 | 128 |
| 8 | 64 | 128 | 128 | 128 | 72 |
| 9 | 32 | 256 | 128 | 128 | 72 |

- **sOuter**: number of block rows per core per iteration in the M direction (Q sequence).
- **sInner**: block size in the N direction (KV sequence), which equals s2BaseSize.

**Runtime basic block parameters**:

- **mBaseSize = sOuter × CV_RATIO** (CV_RATIO=2, AIC:AIV=1:2): one AIC core handles mBaseSize rows of Q, which corresponds to 2 AIV cores that each process mBaseSize/2 rows.
- **s2BaseSize = sInner**: block size in the KV sequence direction.
- The host-side `AdjustSinnerAndSouter` selects sOuter/sInner according to the head_dim of V (and the gSize/maxSeq/window conditions), and then maps the selection to a config (D=64→config0/1, D=72→config8/9, D=128→config2/3, D=256→config4/5: when `gSize × maxSeqQ ≥ 64`, take sOuter=64/sInner=128→config4, otherwise take sOuter=32/sInner=256→config5; QK D=192 and DV=128→config6/7). config8/9 have the same S1/S2/D/DV as config2/3, but `originalDV` changes from 128 to 72, and is passed to the kernel with a different tiling key for the output copy to determine row padding. Inside the kernel the L1→L0 transfer width aligns fractally to 16 elements (72→80, `MMParam.loadK/loadN`), Mmad accumulates precisely with the real D, and L1 clearing is unnecessary.

**Output row padding**: the entry parses `originalDV` from the configuration table and passes it through `FAType::originalDV` to each vector block and FlashDecode. `rowPad = (static_cast<uint32_t>(originalDV) % 16 != 0)`: true when the original DV is not aligned to 16, and false when it is aligned to 16. The currently supported DV values are 64/72/128/256, among which 72 is not aligned to 16 (the UB output row width is 128), so the padding path is used. After constructing `FaUbTensor<OUTPUT_T, rowPad>`, an ordinary call `copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord)` can derive the same `ROW_PAD` from the type of `ubTensor`, without explicitly specifying the template arguments of `operator()`. When there is no padding, the S1G and BSNGD/TNGD combinations can enable intermediate multi-row merged copy; D=72 retains the row-by-row cropped copy-out. The non-merged-axis DN uses `S1_ONLY`, whose copy function does not branch on `ROW_PAD` and handles padding by the actual column width and stride, so it keeps the default `FaUbTensor<OUTPUT_T>`. The shared FlashDecode also sets `rowPad` only in the merged-axis branch.

**Architecture differences**: the table above applies to `ConfigValue` (arch35). arch92 uses `ConfigValueArch92`, with different S1/S2 and buffer tiers; the host only dispatches config 1/2/3/4, whose `originalDV` equals the template DV. arch92 does not support real D=DV=72, and config 8/9 are only compilation-time placeholders for the shared SEL; except for `originalDV=72`, they reuse the fields of arch92 config 2/3 respectively.

**gS1 axis merging**:

- **Meaning**: under GQA, g = Q_N / KV_N Q heads share the same KV. To avoid traversing the two dimensions g and S1 separately inside the task loop, the implementation merges the two axes into one **gS1 axis**: `gS1Size = actSeqLensQ × gSize` (`flash_attn_kernel_dn.h:288`).
- **Layout difference** (the order of g and S1 inside the merged axis, which affects the row conversion of mask/RowInvalid; the non-merged-axis DN splits cores by N1/S1 and does not use this merged-axis encoding):
  - **S1G layout** (BSND/TND layout): S1 is outside and g is inside, `s1Idx = gS1Idx / gSize` (`flash_attn_kernel_dn.h:483`).
  - **GS1 layout** (BNSD layout): g is outside and S1 is inside, `s1Idx = gS1Idx % actSeqLensQ` (`flash_attn_kernel_dn.h:486`).
- **Benefit of the merge**: the bN2 × gS1 two-dimensional task space is tiled uniformly by mBaseSize (`gS1LoopTimes = ceil(gS1Size / mBaseSize)`); the g Q heads that share the same KV land in adjacent rows, which facilitates KV reuse and load-balanced splitting.

**Task loop structure** (triple loop inside the kernel, see §7 for details):

```
for bN2 in [bN2Start, bN2End):        # batch × KV head dimension
    for gS1 in [gS1Start, gS1End):    # Q sequence × GQA group dimension
        for s2 in [s2Start, s2End):   # KV sequence dimension
            CreateTask / ExecuteTask  # BMM1 + Softmax + BMM2
```

Each tile performs one pass of BMM1(Q×K^T) → Softmax → BMM2(attn×V), and the softmax and output accumulate online across the s2 chunks.

### 4. Load Balancing (SectionStreamK)

**Algorithm location**: `attention/common/op_kernel/load_balance/section_stream_k/`, called by `flash_attn_metadata` (AICPU), outputs the FA-segment + FD-segment task split metadata.

**SectionStreamK split key points**:

- **Section division**: accumulate cost batch by batch × head against the L2 budget (96MB token cost) and cut sections accordingly. This ensures that the KV data inside one section is not evicted from the L2 cache, which raises the L2 cache hit rate and shortens the copy time of KV data. No split happens when `maxGS1Size ≤ mBaseSize` or when the cost of one head ≤ budget / number of AICs, because in that case the KV data needs no repeated copies during the whole computation, so no L2 reuse exists.
- **Cost model** (not token weighted): block cost = `6*ceil(m/16) + 10*ceil(s2/64)`.
- **Core count per section**: heuristic lower bound `minCore = sqrt(blockCount)`, maxCore = min(36, blockCount).
- **FA task assignment**: three-level greedy — whole batch → whole row (gS1G) → s2 blocks within a row; each core receives a quota allocated by `cost / remainingCores`.
- **FD task assignment** (cross-core rows): when a row is split across multiple cores, the splitter records an FD task; `s2SplitNum` = number of split segments, spread across ≤72 AIVs by `Σ s2Split×m`.
- **FD enable decision** (`CheckChooseWithFd`): when sectionNum>1, FD is mandatory; otherwise, when the benefit of the no-FD scheme beats the cost of 10 blocks, FD is not used.

**Per-section core assignment (FA/FD segment layout in metadata)**:

Metadata records the two task segments independently for each section, and the kernel directly reads its own interval by core id for execution:

```
header (16 fields): sectionNum | isFd | mBaseSize | s2BaseSize | ...
FA segment  [section][aicIdx(36)][16 fields]: bN2Start/gS1Start/s2Start, bN2End/gS1End/s2End, firstFdDataWorkspaceIdx
FD segment  [section][aivIdx(72)][16 fields]: bN2Idx, mIdx(gS1Idx), workspaceIdx, s2SplitNum, mStart, mLen
Total size = 16 + sectionNum×(36+72)×16 uint32 entries
```

- **FA segment core assignment**: each AIC core (aicIdx) receives one interval of `(bN2, gS1, s2)` inside the section, and the kernel executes only the tasks inside its own interval; the three-level greedy assignment determines the interval boundaries (batch→row→block).
- **FD segment core assignment**: generated only when cross-core rows exist inside the section (the same gS1 row is split across multiple cores); each AIV core (aivIdx) receives one segment `mStart + mLen` of a row and maps to a workspace slot by `s2SplitNum`.
- **Relation between sections and core count**: the L2 budget determines sectionNum, and each section assigns cores independently — the same AIC core holds different task intervals in different sections, and `Process()` loops over sections (see §7).

**Metadata output**: `16 + sectionNum×(36 AIC + 72 AIV)×16` uint32 values (header + FA segment + FD segment); see `flash_attn_metadata.h:57-128` and `flash_attn_metadata_aicpu.cpp:63-68` for the size calculation.

**Impact on the NumBlock (core count) setting**:

- Because the host-side Tiling does not know the actual number of cores needed when it computes NumBlock (the metadata interface running on AICPU computes it), the implementation sets NumBlock to the total core count.

### 5. TilingKey

**Purpose**: decides at compilation time which kernel variants are generated, and selects the variant at runtime. The 5 template parameters are packed from bit0 into the tiling key with 8+8+1+8+4 bits (`flash_attn_template_tiling_key.h`):

| bits | Parameter | Value |
|---|---|---|
| 0-7 | InOutLayoutType | 0=BSND, 1=BNSD, 2=TND, 3=BNSD_BSND |
| 8-15 | KvLayoutType | 0=contiguous, 1=PA_BBND, 2=PA_BNBD, 3=PA_NZ |
| 16 | HasAttenMask | false/true |
| 17-24 | TemplateId | 0=ND template, 1=merged-axis DN template, 2=non-merged-axis DN template |
| 25-28 | Config | 0~9 (S1/S2/D/DV/originalDV and buffer tiers, see §3 Basic Block; arch35 config 6/7 is D=192/DV=128, and config 8/9 is template D=DV=128 with originalDV=72) |

**SEL instantiation range** (three segments; kernel variants are enumerated by combination at compilation time):

- ND (templateId=0): full combination 4×4×2×10=320 variants;
- Merged-axis DN (templateId=1): only no attenMask and config∈{0,2,6,8}, 4×4×1×4=64 variants;
- Non-merged-axis DN (templateId=2): full combination 4×4×2×10=320 variants.

The above is the compilation-time SEL enumeration range; at runtime only variants satisfying the host routing conditions are selected. arch35 currently enables the non-merged-axis DN by default, with 704 variants per data type across the three segments; arch92 still uses the shared SEL, but the host only selects the tiers it supports.

**Generation**: one step of host tiling, `GenTilingKey` — `UpdateTilingKeyInfo` (fills tilingKeyInfo by layout/mask/templateId/config) → `GET_TPL_TILING_KEY` (`arch35/flash_attn_tiling.cpp`). Inside it, `UpdateTilingKeyTemplateId` calls `GetFlashAttnTemplateId`, which is shared with the metadata, and selects the non-merged-axis DN, the merged-axis DN, and ND in turn; `dnFlag_` indicates only the merged-axis DN. See §6 for the specific conditions.

**Dispatch and selection**: during tiling, `SetTilingKey` writes into TilingContext; at runtime the kernel selects the variant by key for execution. At compilation time `--tiling_key` can specify generation of only a subset of variants (see Quick Start compilation).

**Example decoding**: tiling key `67239936` = 0x4020000 → InOutLayoutType=0(BSND), KvLayout=0(contiguous), no mask, templateId=1(DN), config=2(D=128, sOuter=64, sInner=128).

**Relation with template routing**: `config` distinguishes basic blocks and the original DV, and `templateId` distinguishes ND, the merged-axis DN, and the non-merged-axis DN; the two jointly determine the kernel variant. The host and the metadata share template selection; see §6 for details.

### 6. Kernel Entry and Template Range

The kernel compiles different variants according to the 5 template parameters of the tiling key; the runtime entry `op_kernel/flash_attn.cpp` compiles AIC and AIV together (`KERNEL_TYPE_MIX_AIC_1_2`, with `__DAV_CUBE__` distinguishing Cube compilation from Vector compilation), and each AIC pairs with 2 AIVs (CV_RATIO=2).

The arch35 routing is decided by `UpdateTilingKeyTemplateId` calling a shared function; the kernel dispatches by `templateId`:

```cpp
tilingKeyInfo_.templateId = fa_tiling_util::GetFlashAttnTemplateId(
    static_cast<uint32_t>(faInfo_->qkHeadDim), static_cast<uint32_t>(faInfo_->vHeadDim), faInfo_->maskMode,
    faInfo_->maxSeqQ, sOuterFactor_);
dnFlag_ = tilingKeyInfo_.templateId == FA_Template_DN;
```

| Path | Condition (checked in priority order) | Scheduling framework | softmax VF |
|---|---|---|---|
| Non-merged-axis DN (templateId=2) | `DN_UNMERGED_ROUTE_ENABLED=true` (the current default), real V head dim≤128, `maxSeqQ≥256`, and `maskMode∈{0,3}` | flash_attn_kernel_dn_unmerged.h | ProcessVec1VfDnLine / ProcessVec1VfDn |
| Merged-axis DN (templateId=1) | The non-merged-axis DN is not hit, and `maskMode=0`, `sOuter=64`; the supported dimensions correspond to config∈{0,2,6,8} | flash_attn_kernel_dn.h | ProcessVec1VfDn |
| ND (templateId=0) | Other supported scenarios | flash_attn_kernel_nd.h | ProcessVec1Vf |

For example, when real D=DV=72, `maxSeqQ=256`, and `maskMode=0`, config8 and templateId2 are used; when `maxSeqQ=128` and `sOuter=64`, config8 and templateId1 are used. The `originalDV` of config8/9 is 72, so changes in template routing do not lose the output padding information.

arch92 determines the template through an independent `GetConfigInfo`: when the real DV=128, `maxSeqQ≥128`, and `maskMode≠4`, the non-merged-axis DN is selected; other supported scenarios select ND, and the merged-axis DN is not dispatched. The 256 threshold of arch35 does not apply to arch92; both the configuration table and the routing conditions must be distinguished by architecture.

Change points for adding a new template combination (such as a new config or a new layout): `flash_attn_template_tiling_key.h` (parameter declarations and SEL instantiation range) → host `UpdateTilingKeyConfig`/`UpdateTilingKeyTemplateId` (mapping) → the routing in `flash_attn.cpp` and the kernel template instantiation sites.

### 7. Kernel Layer: Scheduling Framework (flash_attn_kernel_dn.h / flash_attn_kernel_nd.h)

**Overall flow** (`Process()`):

```text
Process():
  InitBuffers + InitCrossCoreSync + AllocEventID   # per-core initialization
  for sectionIdx in [0, sectionNum):
      FlashAttention(sectionIdx)                   # collaborative AIC+AIV computation (only cores with aicIdx < coreNum)
      FlashDecode(sectionIdx)                      # AIV only: wrapped by SyncAll on both ends
  FreeEventID + UnInitCrossCoreSync
```

**① InitOutput** (`ClearOutput` in `flash_attn_block_vec_dn.h` / `flash_attn_block_vec_nd.h`, executed when `needInitOutput=true`): computes the total size of attn_out and softmax_lse (TND uses T×g×N×DV), and `2×coreNum` AIVs write GM in parallel with 32KB POP buffers: attn_out is cleared to 0, and LSE is preset to `3e+99`; after the writes, `SyncAll` guarantees visibility for subsequent computation (`flash_attn_block_vec_dn.h:263-291`).

**② FlashAttention (FA) — core while loop** (`flash_attn_kernel_dn.h:213-265`):

```text
bN2Cur, gS1Cur, s2Cur ← metadata segment start points; createdTaskCount / executedTaskCount = 0
while (shouldDispatchTask || validTaskCount):
  ├─ shouldDispatchTask = ShouldDispatchTask(...)          # whether task space remains
  ├─ if dispatch:
  │     taskDealMode = GetTaskDealMode(...)                # decides how to handle this (bN2,gS1,s2)
  │     ├─ CREATE_TASK → CreateTask() → CalcParams() fills RunInfoX (bpIdx/n2Idx/gS1Idx/loop/mloop)
  │     │                → createdTaskCount++, validTaskCount++ → UpdateAxisInfo() advances the cursor
  │     └─ DEAL_ZERO / other (skip whole row / fast forward) → UpdateAxisInfo() and then continue
  └─ if validTaskCount > 0:
        ExecuteTask(executedTaskCount)                     # see below
        if executedTaskCount > PRELOAD_N: validTaskCount--
```

- `GetTaskDealMode`: computes `actSeqLensQ/Kv` from seqused/cu_seqlens; skips the whole row when the row length is 0 (DEAL_ZERO); in window mode it fast forwards when the s2 cursor has not reached `curS2Start` (NOT_START); when the mask is valid, `CalcCurS2StartEndWithSparse` decides to skip invalid tiles.
- **Essence of the while loop**: task creation (cursor advance) and task execution (pipeline consumption) are separated; `validTaskCount` tracks the number of unexecuted tasks, ensuring that creation stays at most 2 rounds (PRELOAD_N) behind execution to form the pipeline.

**③ Cross-core PRELOAD pipeline (ExecuteTask)** (`flash_attn_kernel_dn.h:422-444`):

```text
ExecuteTask(loop, taskRunInfo):
  runInfo0   = taskRunInfo[loop & 3]              # task of this round
  if isValid: AIC→ComputeMm1 (BMM1) | AIV→ComputeVec1 (softmax)
  if loop ≥ PRELOAD_N:
    runInfoNegN = taskRunInfo[(loop-PRELOAD_N) & 3]   # task of PRELOAD_N=2 rounds earlier
    if isValid: AIC→ComputeMm2 (BMM2) | AIV→ComputeVec2 (output accumulation)
                runInfoNegN.isValid = false
```

In other words, at the same moment BMM1 (this round) and BMM2 (2 rounds earlier) run in parallel on the AIC, and Vec1 (this round) and Vec2 (2 rounds earlier) run in parallel on the AIV; data passes through cross-core flag handoff (BMM1 result UB → Vec1 → L1 → BMM2), and `PRELOAD_TASK_CACHE_SIZE=4` uses `loop & 3` instead of a modulo operation.

**④ FlashDecode (FD scheduling)** (`flash_attn_kernel_dn.h:567-578`): reads the FD segment metadata (bN2Idx/mIdx/workspaceIdx/s2SplitNum/mStart) → `vecFdBlock_.InitBuffers()` → `ICachePreLoad` → `SyncAll()` (waits for all FA work to finish) → `vecFdBlock_.FlashDecode()` → `SyncAll()` (waits for FD to finish before entering the next section).

**⑤ FA CV computation flow (data flow with the CV ratio 1:2)**: under `KERNEL_TYPE_MIX_AIC_1_2`, 1 AIC pairs with 2 AIVs; the AIC handles the two BMMs and the AIVs handle softmax/output — the two AIVs synchronize independently of each other through flags + `AIV0_AIV1_OFFSET=16`, and each processes `mBaseSize/2` rows (`actVecMSize`) in the row direction. The fixpipe results of BMM1/BMM2 write into the **shared UB mmRes buffer** (see §8/§9), and the flag of the same buffer handshakes between the writer side (BMM1/BMM2) and the reader side (Vec1/Vec2):

```text
            AIC (Cube)                              AIV0 (first half of mBaseSize/2 rows)      AIV1 (second half of mBaseSize/2 rows)
─────────────────────────────────────────────────────────────────────────────────────────────────────
BMM1:      Q×K^T → L0C → Fixpipe → mmRes[bufId]
           (first WaitFlag the same flag ×2: wait until the previous reader of this buffer finishes reading and releases it)
           CrossCoreSetFlag(mmSyncIdx) ────────────────────► CrossCoreWaitFlag reads the first half of UB
           CrossCoreSetFlag(mmSyncIdx+16) ──────────────────────────────► CrossCoreWaitFlag reads the second half of UB
                                                      Vec1 softmax → cast → L1(P) ──► each AIV writes its own
           AIC CrossCoreWaitFlag waits for L1(P) readiness (CC_L1P, dual path) ◄─ CrossCoreSetFlag ×2
           Vec1 SetFlag(mmSyncIdx) after finishing reading ──► releases this buffer for the next writer to overwrite

BMM2:      P×V → L0C → Fixpipe → mmRes[bufId'] (bufId' advances by the mmResBufId_ counter, ≠bufId)
           (first WaitFlag the same flag ×2: wait until the previous reader of this buffer finishes reading and releases it)
           CrossCoreSetFlag(mmSyncIdx') ──────────────────► Vec2 first-half rows accumulate → write GM
           CrossCoreSetFlag(mmSyncIdx'+16) ───────────────────────────► Vec2 second-half rows accumulate → write GM
           Vec2 SetFlag(mmSyncIdx') after finishing reading ──► releases this buffer for the next writer to overwrite
```

- **Flag division of labor (read-write handoff with one flag per buffer)**: **the flag ID identifies the mmRes buffer (the resource), not the operation**. The results of BMM1/BMM2 share the same set of `ubMmResBuffers_`, and each buffer corresponds to one cross-core sync ID — the AIV side (block_vec) uses `CC_MM_0~3` and the AIC side (block_cube) uses `CROSSCORE_MM_0~3` (each side declares its own copy with identical values); the code converts through `mmSyncIdx = CC_MM_0 + mmResUbBufId` before passing to `CrossCoreSetFlag/WaitFlag` (4 flags for D≤128, 2 flags for D=256): the writer side (BMM1/BMM2) Sets after fixpipe completes (result ready), the reader side (Vec1/Vec2) Waits, reads, and Sets again after reading finishes (buffer released) — the same flag alternates along the "ready → released → ready" relay, replacing the earlier scheme in which BMM1/BMM2 each held an independent pair of flags (`CC_BMM1_0/1`/`CC_BMM2_0/1`); the `+16` (`AIV0_AIV1_OFFSET`) offset lets the two AIVs of the same AIC use independent flag bits without cross interference. Buffers are allocated by the monotonically increasing counter `mmResBufId_` (replacing the `loop%N` index — the latter would map BMM1(L) and BMM2(L-2) to the same buffer), which guarantees that BMM1(L) and BMM2(L-2), executed one after another inside the same ExecuteTask, land in different buffers and avoids write-write conflicts and circular waits.
- **L1(P) handoff**: after the two AIVs each write their Vec1 results to L1(P), they call CrossCoreSetFlag(CC_L1P) (3 buffers correspond to three flag paths), and the AIC waits for both paths to be ready before BMM2.
- **Row sharing**: the two AIVs each compute `mBaseSize/2` rows (the merged-axis DN splits by `align32(actM)>>1`, ND by `(actM+1)>>1`), which cooperates with the dual-destination write of the BMM1/BMM2 fixpipe (`dualDstCtl=1`, split in half along the M dimension into the UBs of the two AIVs) and guarantees that each AIV gets exactly its own half.


### 8. block cube layer

#### 8.1 buffer allocation
The AIC side allocates buffers according to the storage hierarchy, covering the data flow of the two Matmul levels BMM1 (Q×K^T) and BMM2 (P×V). The overall allocation in flash_attn_block_cube_nd.h is as follows:

| Storage level | Buffer | Count×Size | Purpose |
|---|---|---|---|
| L1 | `l1PBuffers_` | 3×(mBaseSize×s2BaseSize×sizeof(INPUT_T)) | softmax result (P matrix) for BMM2 to read |
| L1 | `l1QBuffers_` | 2×(mBaseSize×dBaseSize×sizeof(Q_T)) | Q matrix (BMM1 input), reused within a gS1 row |
| L1 | `l1KvBuffers_` | 4×64KB (when s2BaseSize=256 and D>128, that is config=5: 2×128KB large buffers) | K/V matrices; BMM1 uses K and BMM2 uses V, shared |
| L0A | `l0aBufferManager_` | 2×32KB (BufferManager initially 64KB) | K for BMM1 / P for BMM2 |
| L0B | `l0bBufferManager_` | 2×32KB (BufferManager initially 64KB) | Q for BMM1 / V for BMM2 |
| L0C | `l0CBuffers_` | 4×64KB | Matmul results of the two MMs (shared rotation) |
| UB | `ubMmResBuffers_` | (D≤128: 4; D=256: 2)×(mBaseSize/2×max(s2BaseSize, dVBaseSize)×sizeof(MM_T)) | BMM1/BMM2 fixpipe results time-division reused → AIV softmax / output accumulation |

**Explanation of the L1 buffer allocation strategy**:

- **3 buffers for P**: P is written by the AIV (softmax result) and read by the AIC (BMM2 input), so the read and write belong to different cores. Under PRELOAD_N=2, the AIV writing the P of round N runs in parallel with the AIC reading the P of round N-2. The rotation of 3 buffers exactly holds the three states "being written / pending read / released", preventing inter-core flag waits from blocking the pipeline.
- **2 buffers for Q**: Multiple s2 loops within the same gS1 row reuse the same Q, which is loaded only at `isFirstS2Loop` and released at `isLastS2Loop`. The 2 buffers alternate between the old and new Q when switching between consecutive gS1 rows, so MTE2 loading overlaps with computation.
- **KV 4×64KB**: K and V are consumed by BMM1/BMM2 respectively, and the two are staggered in time under the PRELOAD_N=2 pipeline; while K is being loaded, the V of the next round can be loaded. The 4 buffers let K/V load alternately, keeping MTE2 continuously working; the 64KB capacity covers a single matrix of mBaseSize×s2BaseSize×2B. When s2BaseSize=256 and D>128 (config=5), a single matrix (s2×d×2B=128KB) exceeds 64KB, so 2×128KB large buffers are used instead; config=4 (s2BaseSize=128) still uses 4×64KB.

**Explanation of the L0A/L0B allocation strategy**: L0A/L0B cache the A/B operands of the Matmul. The 2×32KB double buffer lets MTE1 copying (L1→L0) overlap with PIPE_M computation; while the data of round i+1 is being copied, round i is being computed. `BuffersPolicyDB` (LOCK_UNLOCK + external eventID) manages the lock dependencies.

**Explanation of the L0C allocation strategy**: L0C is the accumulator for Matmul results. The 4×64KB setup doubles the coverage of the overlap window of "Fixpipe read of the previous round + Matmul write of the current round". The two MMs share the same group of L0C buffers (reused with time staggering), and the 4 buffers keep the three-stage MTE1→M→FIX pipeline advancing continuously. When D=256 (dVBaseSize>128), BMM2 performs the outer loop with 128 columns as a tile (the `nLoops` loop of `IterateBmm2l0Split`, splitN), occupies the rotating 64KB buffers tile by tile, and performs Fixpipe independently; a single buffer does not need to hold the entire 128×256×4B result, and the L0C layout stays uniform for D=128/D=256.

**Explanation of the UB allocation strategy**: `ubMmResBuffers_` is the CV communication buffer from AIC to AIV (fixpipe writes, AIV reads). The BMM1/BMM2 results reuse the same group of buffers: the buffer size takes the larger of `s2BaseSize` and `dVBaseSize` (a slice of `mBaseSize/2` rows; the two AIVs each process half, `CV_RATIO=2`). The buffer count is tiered by D: 4 buffers for D≤128 and 2 buffers for D=256 (`UB_MM_RES_BUFCNT = (dBaseSize > 128) ? 2 : 4`). The multi-buffer rotation, combined with the inter-core synchronization IDs (AIC side `CROSSCORE_MM_0~3`, one flag per buffer), forms a pipeline: while the AIC fixpipe writes one buffer, the AIV is reading an earlier buffer; flag setting and waiting alternate, hiding the CV transfer latency. Buffers are allocated by the monotonically increasing counter `mmResBufId_`. The addresses and sizes must fully align with the corresponding UB buffers on the block_vec side.

### 9. block vector layer

#### 9.1 buffer allocation
**UB buffer allocation (taking ND as an example, the `UbLayout` struct definition, `flash_attn_block_vec_nd.h:203-216`)**: ND is the path with the most complete functionality (including mask). Its layout covers all DN buffers plus two extra items, mask/tmp. The layout is arranged from low to high addresses, and `static_assert(sizeof(UbLayout) <= 248KB)` limits the total usage. The total UB size is 256KB, but because 8K must be reserved for AscendC and VF, the business logic can only use 248K.

| Buffer | Count×Size | Purpose | Allocation strategy |
|---|---|---|---|
| `ubMmResBuffers_` | (D≤128: 4; D=256: 2)× (mBaseSize/2 × max(s2BaseSize, dVBaseSize) × sizeof(FP32)); at D=128 this is ≈ 4×32KB | CV communication BUF: the AIC BMM1/BMM2 fixpipe results are time-division reused (Vec1 softmax input / Vec2 accumulation input) | **Multi-buffer rotation**: AIC fixpipe writes run in parallel with AIV reads, one flag per buffer (AIV side `CC_MM_0~3` / AIC side `CROSSCORE_MM_0~3`), a read-write relay (see §7⑤). The buffer size takes the larger of s2 and dV, and the BMM1/BMM2 results are time-division reused. Buffers are allocated by the monotonically increasing counter `mmResBufId_` (replacing `loop%N`) |
| `ubMaskBuffers_` | 2 × (mBaseSize/2 × s2BaseSize × sizeof(uint8)) | Input BUF for copying the mask in from GM (ND only) | **Double-buffer prefetch**: while the current round `AttenMaskCopyIn` computes, MTE3 prefetches the next round mask, avoiding GM memory access latency. The actual requirement = mBaseSize/2 rows per AIV × s2BaseSize INT8 values (1B each), that is 2×(mBaseSize/2 × s2BaseSize × 1B) (the current code hardcodes 2×8KB, which only matches the requirement when s2BaseSize=128) |
| `ubVec2Res_` | 1 × (mBaseSize/2 × Align64(dVBaseSize) × sizeof(FP32)) | Vec2 intermediate result + attn_out output buffer | **Single buffer**: it works with the PRELOAD_N=2 pipeline (Vec2 executes at the same time as the Vec1 from 2 rounds earlier), and the first tile uses the `DataCopy` branch to skip dirty values (see §7), so double buffering is unnecessary. The size is computed dynamically from template parameters (`UB_VEC2_RES_BUF_BYTES = mBaseSize/CV_RATIO × Align64(dVBaseSize) × sizeof(T)`, previously hardcoded to 32768U) |
| `ubVec1ResBuffers_` | 2 × ((mBaseSize/2 + 1) × s2BaseSize × sizeof(BF16/FP16)) | Transit BUF for copying the cast softmax result to L1 (P) | **Double buffer**: the V pipe write runs in parallel with the MTE3 copy to L1, and the 2 buffers alternate the two transfers. The size is computed dynamically from template parameters (`UB_VEC1_RES_BUF_BYTES = (mBaseSize/CV_RATIO+1) × s2BaseSize × sizeof(INPUT_T)`, previously hardcoded to 33024U). **The row size is (mBaseSize/2 + 1) instead of mBaseSize/2**: each row reserves 1 extra element as an address offset, so the start addresses of adjacent rows are staggered (no longer congruent modulo the number of UB banks), preventing concurrent access of multiple rows from hitting the same bank and causing bank conflicts; ×2B is the element size when cast back to FP16 |
| `softmaxSumBuf_` / `softmaxMaxBuf_` | 3 × (mBaseSize/2 × sizeof(FP32)) each | sum/max accumulation state of the online softmax | **3 buffers, `mloop % 3`**: each gS1 row (mloop) has an independent set of state, and a rotation of 3 holds the mloop values active at the same time under PRELOAD_N=2 (the current row + 2 rows in flight). Note that due to VF processing constraints, the size of a single buffer must be 256B-aligned |
| `softmaxExpBuf_` | 3 × (mBaseSize/2 × sizeof(FP32)) | exp intermediate results | **3 buffers, `loop % 3`**: the rotation stays in sync with sum/max; note that due to VF processing constraints, the size of a single buffer must be 256B-aligned |
| `ubLseOutBuffers_` | 2 × (mBaseSize/2 × 32B) | Copying the FD intermediate results (sum/max) out to GM, or copying out the LSE result | **Double buffer**: the MTE3 copy-out overlaps with the V pipe computation, rotating `UB_OUT_LSE_OUT_EVENT0/1`; 32B is because the max and sum of a single row must be expanded to 32B before being output |
| `softmaxTmpBuf_` | 1 × 512B | Intermediate result cache of the general-purpose softmax VF (ND only) | **Permanent single buffer**: temporary quantities within a VF call, no cross-round dependencies; 512B is sufficient |

**L1 buffer allocation (ND is the same as DN)**: `l1PBuffers_` 3 × (mBaseSize×s2BaseSize×sizeof(INPUT_T)); the softmax result (P) is written to L1 for BMM2 to read. **The 3 buffers work with PRELOAD_N=2**: the AIV writes the P of round N and the AIC reads the P of round N-2, and there is always one buffer in a buffered state outside "being written / pending read / released", avoiding cross-core flag waits. The addresses and sizes must fully align with the corresponding L1 buffers on the block_cube side.

### 10. FD block: flash_attn_block_vec_flashdecode.h

**Mathematical principle** (online softmax expanded in reverse across the split-K reduction):

In the FA stage the KV direction is split into K parts (`fdS2SplitNum`), and each part independently computes local statistics: max $m_i$, sum $\ell_i = \sum e^{s-m_i}$, and partial output $o_i = \sum e^{s-m_i} v$. The FD stage reduces across the K parts:

$$
m_{global} = \max_i m_i, \qquad
\ell_{global} = \sum_i \ell_i \cdot e^{m_i - m_{global}}, \qquad
o_{final} = \frac{\sum_i o_i \cdot e^{m_i - m_{global}}}{\ell_{global}}
$$

- `ComputeScaleValue`: first a Max reduction obtains the global max; then for each part it performs `Sub(m_i, m_global) → Exp → Mul(l_i, ·) → Add accumulation` to obtain the global sum. The LSE output is `log(sum) + max`; when max==-inf (invalid row), it is set to +inf (`vf_flash_decode_arch35.h:299-317`).
- `ReduceFinalRes`: uses the scale $e^{m_i - m_{global}}$ to recompute the accumOut weight of each part, performs weighted accumulation, and then divides by $\ell_{global}$ (`ReduceFinalRes_VF`).

**Computation flow**:

```text
FlashDecode(fdParams):
  ├─ fdCoreEnable check; fdBalanceMBaseSize=8 splits mLen into small tasks grouped by 8 rows
  ├─ Per small task: combineTaskPrefixSum = fdWorkspaceIdx (prefix sum → workspace slot number)
  │     Load: accumOut, lseSum, lseMax are located in the workspace by slot number (splitKVIndex×mBaseSize×DV offset)
  ├─ ComputeScaleValue (softmax reduction across split-K, see above)
  ├─ ReduceFinalRes (output reduction across split-K, see above)
  └─ DealInvalidRows (zero out invalid rows) → Cast → Bmm2DataCopyOutTrans writes GM
```

**UB allocation** (`InitBuffers`, `flash_attn_block_vec_flashdecode.h:172-230`; **shares the same UB layout** with the FA block, managed manually by absolute byte offsets):

**BASE definition** (`flash_attn_block_vec_flashdecode.h:178-182`):

```cpp
// bmm1 and bmm2 share buffers
constexpr uint32_t mmSz = mBaseSize / 2U * (s2BaseSize > dVBaseSize ? s2BaseSize : dVBaseSize) * sizeof(T); // FA ubMmRes single buffer size
// Starting byte offset of the FD service area (skipping the bmm1/bmm2/mm2In region)
constexpr uint32_t BASE = mmSz * ((dVBaseSize > 128) ? 2U : 4U);
```

FD reuses the UB layout of FA (the same AIV), so it must start at an address **after** the buffers already occupied by the FA block. `BASE = mmSz × buffer count` (4 for D≤128 and 2 for D=256, consistent with the `UB_MM_RES_BUFCNT` convention on the FA side) skips exactly the **CV communication mmRes region** of FA (the shared result region of BMM1/BMM2 fixpipe), and the FD service area starts from this offset. The subsequent `SharedBuffer1/2/3` regions are used in the FA stage by `attenMaskBuf`/`stage1OutBuf`/`stage2OutBuf`; when FA ends and FD starts, those buffers are no longer needed, so the regions can be time-division reused (see the parallelism explanation below).

| Buffer | Count×Size | Purpose |
|---|---|---|
| `fdMm2ResBuf1_/2_` | 2 × 16KB | BMM2 partial results (the weighted output o_i of each split), double buffer with pipeline prefetch |
| `fdReduceBuf_` | 1 × 16KB | Weighted accumulation intermediate result (Σ o_i·scale) |
| `fdOutputBuf_` | 1 × 16KB | Final output buffer (for Cast + copy out to GM) |
| `fdSumBuf1_/2_` | 2 × 6KB | lseSum double buffer (local sum of each part) |
| `fdMaxBuf1_/2_` | 2 × 6KB | lseMax double buffer (local max of each part) |
| `fdLseExpBuf_` | 1 × 6KB | Staging of the rescale factor exp(m_i - m_global) |
| `fdLseMaxUbBuf1_/2_` | 2 × 256B | Small buffer for the reduced row max (invalid row determination) |
| `fdLseUbBuf_` | 1 × 256B | Staging for LSE computation |

> Note: the FD buffers and the FA `attenMaskBuf`/`stage1OutBuf`/`stage2OutBuf` **reuse the same physical UB region** (`InitBuffers` comment: the SharedBuffer1/2/3 regions are time-division shared between FA and FD), so no additional UB total is added.

**Parallelism between FD and block_cube** (`flash_attn_kernel_dn.h:613-637`):

```text
Process():       AIC (block_cube)                      AIV (block_vec / FD)
  FA(section)     BMM1/BMM2 fixpipe → UB(mm region)     softmax/accumulation (reads mm region)
  FD(section)     ×  not involved                       SyncAll → FD reduction (reads workspace) → SyncAll
  Next section    FA starts early (does not wait for FD)  Enters FA after FD completes
```

- **Synchronization division**: `FlashDecode` has one `SyncAll` (all-AIV synchronization) at each end. The first one ensures that the FA of all AIVs has completed (the workspace is fully written); the last one ensures that the FD of all AIVs has completed before entering the next section. **The AIC does not participate in SyncAll**.
- **BV parallel window**: after the AIC completes the BMM2 of the current section, it does not need to wait for FD and directly starts the FA of the next section early (the AIC branch in `Process()` only executes `FlashAttention`). The AIC writes the mm region `[0, BASE)` of the next section, and FD reads the reduction region `[BASE, ...)`; **the two UB regions do not overlap**. This is exactly the purpose of BASE separating the FD service area from the CV communication region: while FD executes, the AIC can continue copying/computing the BMM of the next section, and the two engines do not block each other.
- **Boundary**: the FA of the next section on an AIV must wait for its own FD to complete (SyncAll) before starting. Therefore the lead time of the AIC is limited because the mm1 results of the next section still need to be consumed by the AIV (flag wait). The actual overlap window is "AIC BMM computation/copying" running in parallel with "AIV FD reduction".

FD connects with FA through **workspace slots**: in the FA stage the partial result of each cross-core row is written to the `workspaceIdx` slot (a three-segment layout of accumOut + lseSum + lseMax), and in the FD stage the results are read back for reduction according to the `fdWorkspaceIdx` prefix sum (see §10 for the workspace slots and §4 for the intra-section core splitting).

### 11. common layer: shared facilities

The common layer is located in `attention/common/op_kernel/` and is divided by function into four categories of infrastructure:

**① Buffer management (`buffer.h`, `buffer_manager.h`, `buffers_policy.h`)**: infrastructure in the `fa_base_matmul` namespace related to buffer lifecycle/synchronization policies:
- `BufferManager` (`buffer_manager.h`): allocation and management of L0A/L0B/L0C buffers (initialization, size division).
- `BuffersPolicy*` (`buffers_policy.h`): buffer usage policies, including `BuffersPolicySingleBuffer` (single buffer), `BuffersPolicyDB` (double buffer, LOCK_UNLOCK/external eventID; the L0A/L0B of block_cube use this policy), `BuffersPolicy3buff` (three buffers), and `BuffersPolicy4buff` (four buffers). These determine how MTE1 copying overlaps with Matmul computation.
- Auxiliary tools: general functions in `buffer.h` related to buffer access/synchronization.

**② Matmul wrapper (`matmul.h`)**: the Matmul instruction wrapper in the `fa_base_matmul` namespace, called by the block_cube layer:
- Instruction family: `MatmulBase` (K dimension fixed at 128), `MatmulK` (128-block looping along the K direction), `MatmulN` (128 blocking along the N direction), `MatmulFull` (full computation in one pass).
- Supporting items: `ABLayout` (MK/KN layout enumeration), `LoadDataToL0A/L0B`, `MakeMMParam`, MM parameters (M/N/K, transpose flags), which work with `BuffersPolicyDB` to complete the L1→L0 copying and computation.

**③ Data movement (`memcopy/`, containing `copy_gm_to_l1.h`, `copy_gm_to_ub.h`, `copy_ub_to_gm.h`)**: copy wrappers between GM and L1/UB:
- `copy_gm_to_l1.h`: the `DataCopy` wrapper for GM→L1 (including the ND→NZ transformation, working with the GM coordinate positioning of `offset_calculator_v2.h`).
- `copy_gm_to_ub.h` / `copy_ub_to_gm.h`: GM↔UB copies.
- Supporting GM/UB tensor descriptions: `fa_gm_tensor.h`, `fa_l1_tensor.h`, `fa_ub_tensor.h`, `gm_layout.h`, `gm_coord.h`.

**④ VF instruction library (`arch35/` and `arch35/vf/`)**: a collection of instruction-level VF operators, called by the block vector layer:
- Entry points and general APIs: `flash_attention_score_common_regbase_arch35.h` (`FaVectorApi`: `ProcessVec1VfDn`, `FusedExpSub`, `FlashUpdateNew`, `ComputeLseOutputVF`, `RowInvalidUpdateVF`, and so on).
- VF operator families under `vf/`: the softmax/update series (`vf_basic_block_*`), flash_decode (`vf_flash_decode_arch35.h`), antiquant (`vf_antiquant_*`), and so on, split into multiple variants by D/alignment/update strategy.
- Shared with operators such as flash_attention_score; **modifying the VF affects all users**.

In addition:
- **Type/layout utilities** (`op_kernel/utils/` on the flash_attn side): `flash_attn_type.h` (FAType/layout mapping), `flash_attn_common_def.h` (constants/enums), `attenmask_m_gs1.h` (merged-axis GS/SG mask computation), `attenmask_m_s1.h` (non-merged-axis S1_ONLY mask computation).
- **Load balancing algorithm** (`common/op_kernel/load_balance/section_stream_k/`): the algorithm library for host-side metadata generation (see §4).

### 12. Suggestions for contributors

- **Where to start reading**: `flash_attn_def.cpp` → `checkers/` → `arch35/flash_attn_tiling.cpp` → `op_kernel/flash_attn.cpp` → `flash_attn_kernel_dn.h`/`flash_attn_kernel_nd.h`.
- **Changing the interface**: add or remove parameters in def.cpp → synchronize the checkers → infershape → the interface documentation (on the torch_extension side).
- **Changing the splitting strategy**: `fa_adjust_sinner_souter_arch35.h`/`fa_adjust_sinner_souter_arch92.h` (shared by the main operator and metadata; they must stay consistent) + `common/op_kernel/load_balance/section_stream_k/`.
- **Changing kernel computation**: first confirm the template scope in `flash_attn_template_tiling_key.h`. The AIC side focuses on `flash_attn_block_cube_*`, and the AIV side focuses on `flash_attn_block_vec_*`. The shared VF APIs are in `common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h`.
- **Adding tests**: add tiling cases to `tests/ut/op_host/arch35/test_flash_attn_tiling.csv`; add e2e cases to `tests/pytests/test_cases/`.
- **Precautions**: the kernel side has no independent unit tests (it relies on pytests for real-device verification). Some TND/BF16 positive cases in the tiling CSV are currently marked FAIL (to be updated based on actual results). When adding cases, express the expectation with `expectResult`.

## Reference resources

- Interface documentation: `../../attention/flash_attn/docs/torchapi_flash_attn.md`
- e2e test framework: `tests/pytests/readme.md`
- SectionStreamK algorithm: `../common/op_kernel/load_balance/section_stream_k/`
- Shared VF common APIs: `../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h`
