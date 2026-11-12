# ChunkKdaFwd Operator Design Introduction

## 1 Design Objectives

1. The top-level interface aligns with the non-CP FLA `chunk_kda_fwd`.
2. No public operator prototype is added; the A5 specialization reuses the existing
   `ChunkKdaFwd` L0 calling interface and the `chunk_kda_fwd` device kernel launch entry.
3. A2/A3/A5 use the same mathematical definition; A5 retains the regbase dual-issue specialization.
4. The input layout and the output layout are decoupled.
5. FwdH is maintained separately by KDA under arch22/arch35, and supports key-wise gate and `state_v_first`.

## 2 Computation Process

### 2.1 Interface Layering

```text
raw g -> ChunkKdaFwd[
    gate cumsum -> Prepare/Post-WU -> FwdH -> Finalize
] -> attn_out
```

The interface call chain is divided by layer:

```text
cann_ops_transformer.ops.chunk_kda_fwd
  -> PyTorch custom op registration and wrapper
  -> aclnnChunkKdaFwd
  -> ChunkKdaFwd L0 registration and launcher
  -> chunk_kda_fwd device kernel launch entry
  -> Gate / Prepare / Post-WU / FwdH / Finalize internal stages
```

`aclnnChunkKdaFwd` is responsible for contiguizing the public layouts, performing necessary view
transformations, and calling the registered `ChunkKdaFwd` L0 interface once. The L0 launcher continues to
use the `chunk_kda_fwd` device kernel launch entry. Gate, Prepare, Post-WU, FwdH, and Finalize are only
internal computation stages of this kernel; no `ChunkKdaFwdPrepare`, `ChunkKdaFwdPostWu`, or
`ChunkKdaFwdFinalize` L0 operators are registered. All information required for template selection is
derived from existing inputs, shapes, and attributes; no public attributes or interface fields are added.

### 2.2 Computation Stages

#### 2.2.1 Gate Cumsum

Converts the raw/activated gate into FP32 chunk-local log2 cumulative values:

```text
gamma = g                                                        use_gate_in_kernel=false
gamma = -exp(A_log) * softplus(g + dt_bias)                      use_gate_in_kernel=true, safe_gate=false
gamma = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))        use_gate_in_kernel=true, safe_gate=true
gk = chunk_local_cumsum(gamma) / ln(2)
```

#### 2.2.2 Prepare

Reads only `q/k/v/gk/beta` and variable-length metadata, and produces:

```text
Aqk, Akk, qg, w_seed, u_seed
```

```text
qg[i] = q[i] * exp2(gk[i])
w_seed[i] = beta[i] * k[i] * exp2(gk[i])
u_seed[i] = beta[i] * v[i]
```

The score dot product takes BF16/FP16 as input and accumulates in FP32. The `L` of `Akk=(I+L)^{-1}`, the
identity matrix, diagonal block inversion, block merging, Cube inputs/accumulation, and workspace are all
FP32, with HF32 explicitly disabled. The public `Akk` is written back as the q dtype only after the
complete FP32 solve finishes.

On A2/A3, score products keep the right matrix resident in one L1 B slot for `K<=128` and two slots
for `K>128`, so prefetching the next K tile cannot wait on the unconsumed current tile. A V pipe
barrier separates Aqk scaling from clamping and casting, including short tail chunks.

#### 2.2.3 Post-WU

Reads only `k/gk/w_seed/Akk/u_seed`, and produces:

```text
w, u, kg
w = Akk @ w_seed
u = Akk @ u_seed
kg[i] = k[i] * exp2(gk[last] - gk[i])
```

When `H_v>H_k`, the scenario is GVA (Grouped Value Attention): each group of `H_v/H_k` consecutive Value
heads shares one q/k head. When head numbering starts from 0, the q/k head corresponding to Value head
`h_v` is numbered `floor(h_v/(H_v/H_k))`. The head loop of `Akk` runs over `H_v`, and the GVA mapping is
converted only when q/k heads are read, avoiding duplication or omission based on `H_k`.
Whether Post-WU is executed as a separate internal stage is decided by tiling. When the A5 dense-aligned
fusion conditions are met, Prepare directly calls the Post-WU computation components; in other scenarios,
the internal Post-WU stage is executed after Prepare finishes and completes synchronization. Both modes
reuse the same `ChunkKdaFwd` L0 calling interface and device kernel launch entry.

#### 2.2.4 FwdH

Reads `kg/w/u/gk` and the optional `initial_state`, and computes the cross-chunk state update:

```text
v_new = u - w @ h_prev
h_next = exp2(gk_last) * h_prev + kg^T @ v_new
h[chunk] = h_prev
final_state = h_after_last_chunk
```

Both arch22 and arch35 use the KDA private FwdH implementation inside the `ChunkKdaFwd` directory and do
not depend on kernel source code of other operators. The cross-chunk `h_prev/h_next` is always stored in
the FP32 workspace or the FP32 `final_state`; the public `h` is written back as a snapshot for backward
only, in the q dtype, and cannot participate in the state update of the next chunk.

On A2/A3, Post-WU splits the output K dimension of the W product into at most 128-column tiles,
and FwdH splits `kg^T@v_new` into at most 128-row tiles to respect L1/L0C capacity. Tail Vector
Post-WU writes W into the existing FP32 scratch first. An AIV-to-AIC-to-AIV handshake confirms
that both AIVs have finished reading the seed before copying W back to its aliased GM storage.
The FwdH state-update fast path also requires the subblock matrix to fit its 32 KiB FP32 calculation
buffer; larger matrices use the existing 16-row loop to avoid overlapping adjacent UB buffers.

#### 2.2.5 Finalize

Reads only `qg/Aqk/v_new/h` and computes with FP32 accumulation:

```text
attn_out = scale * (qg @ h) + Aqk @ v_new
```

`scale` is applied after the first term completes the FP32 dot product; it is not pre-quantized into the
BF16 `qg`.

`attn_out` is written out directly in BSND/TND inside the kernel. Intermediate quantities for backward
remain in BNSD/NTD.

## 3 Numerical Design

- The `q`, `k`, `v`, and intermediate quantities stored in the input dtype in the formula serve as matmul
  inputs, and matmuls use FP32 accumulation.
- The $L$ of `Akk=(I+L)^{-1}`, the identity matrix, diagonal block inversion, block merging, Cube inputs,
  and accumulation remain FP32, and HF32 is not enabled.
- The computation of the cross-chunk state $h_c$ and `final_state` remains FP32; the low-precision snapshot
  of the public `h` is not used to continue state propagation to the next chunk.
- The two matmuls and the addition of `attn_out` use FP32 accumulation, and the result is converted to the
  output dtype during the final write-back.

## 4 Output Layout and Recomputation

### 4.1 State Layout

Internal states uniformly use `[...,K,V]`. When `state_v_first=true`, L2 transposes the initial state
before entering FwdH. The internal `hCompute` always remains head-major for Finalize to consume; the
public `hOut` is converted to sequence-major at the L2 export boundary, with the order of the last two
dimensions decided by `state_v_first`. `final_state` is arranged by sequence, consistent with the FLA
top-level output.

### 4.2 Recomputation Policy

L2 does not understand autograd recomputation policies. `final_state/gk/w/u/qg/kg/v_new/h` are mutually
independent `OPTIONAL_OUTPUT` items; a non-null pointer indicates export, and a null pointer indicates that
the result is not exposed. The arch35 fast path passes fixed ABI placeholders for hidden outputs, and
tiling receives the actual intermediate results in the kernel workspace; the generic path uses the same
rule. When a public output exists, it is used directly as the internal target.

The PyTorch API wrapper layer aligns with the fla-org `chunk_kda_fwd` commit
`0f0f0c97af39343855b43bbbaddcedfda5cb9d77`:

- `attn_out` is always returned.
- `final_state` is returned only when `output_final_state=true`.
- `gk` is returned when `use_gate_in_kernel=false` or `disable_recompute=true`.
- `Aqk` is always returned.
- `Akk` is always returned.
- `w` is returned when `disable_recompute=true`.
- `u` is returned when `disable_recompute=true`.
- `qg` is returned when `disable_recompute=true`.
- `kg` is returned when `disable_recompute=true`.
- `v_new` is returned when `disable_recompute=true`.
- `h` is returned when `disable_recompute=true` or `return_intermediate_states=true`.
- `initial_state` is passed through as the original object by the Python layer; it is not an aclnn output.

The internal `hCompute` and the public `hOut` have two lifecycles: `hCompute` is the required head-major
stage result from FwdH to Finalize, and the kernel workspace holds it when `hOut` is empty; when `hOut` is
not empty, L2 provides a head-major temporary output and converts it to sequence-major at the export
boundary. This rule aligns with the non-CP low-level 12-return-value interface; the 12th item
`initial_state` is passed through as the original object by the Python layer.

## 5 Tiling Design

A "template" in this document refers to a computation path selected by tiling or kernel conditions that
has an independent data split or pipeline organization. Ordinary C++ `template` helpers used only for
dtype, constant propagation, or code reuse are not listed separately as scenario templates.

### 5.1 Interface and Implementation Directory

`ChunkKdaFwd` retains one set of L0 registration, launcher, and the `op_kernel/chunk_kda_fwd.cpp` device
kernel launch entry. The A2/A3 generic implementations are located in `op_kernel/*.h`, the A5
specializations are located in `op_kernel/arch35/*.h`, and the A5 host derived options are located in
`op_host/arch35/chunk_kda_fwd_tiling_impl.h`. The `.h` files of Prepare, Post-WU, and Finalize are internal
computation components of the same kernel, not independent L0 operator interfaces.

### 5.2 Compile-Time Shape Templates

| tiling key | Compile-Time Constants | A2/A3 Implementation | A5 Implementation | Covered Scenarios |
| --- | --- | --- | --- | --- |
| `key=1` generic shape template | `COMPILE_BT/K/V=0/0/0` | Generic implementation in the root directory | Generic implementation in the root directory | Scenarios that do not satisfy `chunk=64,K=128,V=128`, including dense, tail, varlen, and GVA |
| `key=2` chunk64/K128/V128 template | `COMPILE_BT/K/V=64/128/128` | Generic implementation in the root directory, using compile-time constants | `arch35` specialized implementation | dense, tail, varlen, and different head counts with `chunk=64,K=V=128` |

`SetTilingKey` checks only chunk, K, and V, and does not check the SoC, layout, whether the input is
varlen, or whether a tail block exists. Therefore, the tiling key represents a family of shape templates,
not a platform, interface, or operator version. Platform differences are selected by the architecture
branches compiled under the same key.

### 5.3 A5 key2 Execution Combination Templates

`ConfigureChunkKdaFwdArch35` derives four internal options from the instantiated states of existing inputs
and outputs. The options are not public attributes and do not change the L0, aclnn L2, or Python
interfaces.

| Execution Combination | Selection Condition | Internal Stage Order |
| --- | --- | --- |
| standalone Gate | `computeGateInPrepare=false` | Gate executes fully and synchronizes, then Prepare starts |
| Prepare inline Gate | BF16 q/k/v, raw g in FP32, `A_log` present, `use_gate_in_kernel=true`, and `safe_gate=true` | The Prepare AIV computes chunk-local gk in the gate-product pipeline |
| A5 dense FwdH | BF16 and non-varlen, and `T % 64 == 0` | Uses the `ChunkKdaFwdFwdH` arch35 dense template |
| Intra-stage pipeline fusion of Prepare/Post-WU | A5 key2, BF16, `safe_gate=true`, dense-aligned, and `H_v % 2 == 0`, and the next-row template is not selected | After the Prepare AIC produces data by head pair, it directly calls the Post-WU components to consume data in batches |
| Intra-stage pipeline fusion of Post-WU/FwdH | The conditions of the previous row hold, Gate has already been computed inside Prepare, and `qg/v_new/h` are not exported | The dense FwdH directly completes the computation required for W/U and state propagation |
| Independent internal Post-WU stage | Neither of the two Post-WU fusions is selected | `RunChunkKdaPostWu` is called after Prepare finishes synchronization |
| Generic FwdH backend | FP16, varlen, non-aligned tail blocks, or conditions of dense FwdH not satisfied | Selects the KDA private `KdaFwdHTileShapes128/256` templates |

Therefore, "Prepare/Post-WU fusion" describes only the execution combinations that satisfy the conditions
in the table; it does not mean that all key2 scenarios remove the internal Post-WU stage. In particular,
varlen and non-aligned tail blocks use the independent internal Post-WU stage.

### 5.4 Gate and Prepare Internal Templates

| Template | Selection Condition | Main Implementation |
| --- | --- | --- |
| Gate generic chunk-cumsum | standalone Gate | Reuses `KdaGateCumsum::DispatchKdaGateCumsum`, supporting raw/activated gates and safe true/false |
| Gate/Prepare arch35 regbase | A5 key2, BF16, raw FP32, `A_log`, and safe gate | 16-row gate tiles and two sets of UB staging, with MTE2 and VEC overlapped; gk, qg, w seed, and kg factors are generated in batches |
| Prepare generic score/solve | key1, and A2/A3 key2 | Generic Aqk/Akk MMAD, causal mask, triangular solve, and seed write-back |
| Prepare arch35 key2 generic boundary | A5 key2 scenarios that miss batched fusion, including odd heads, safe false, tail, and varlen | Retains the same mathematical definition, using a bounded score queue, AIC/AIV handshaking, and a padded tail |
| Prepare arch35 safe-gate head-pair | A5 key2, safe gate, and `H_v % 2 == 0` | Two head lanes sharing two levels of score queues; 32-row score blocks; Akk solve overlapped with subsequent score movement |
| Prepare arch35 full-chunk fused-score | The previous row with `curT=64`, hitting the fused Post-WU | The two QK/Akk MMADs use a manual L1/L0 pipeline, and the Aqk/Akk masks and beta seeds are completed before write-back |

### 5.5 Post-WU Internal Templates

| Template | Selection Condition | Main Implementation |
| --- | --- | --- |
| Post-WU generic template | key1, and A2/A3 key2 | Generic Cube/Vector computes `w=Akk@w_seed` and `u=Akk@u_seed`, and generates kg |
| Post-WU arch35 full-chunk head-pair | A5 key2, `curT=64`, and fusion combination | At most 4 chunk tasks per batch and two head lanes per task; consumed directly after Prepare produces |
| Post-WU arch35 full-chunk standalone | A5 key2, `curT=64`, and not fused | W/U uses two sets of MTE2 staging with the MMAD/Fixpipe pipeline; Kg uses 16-row tiles and 32-row double buffering |
| Post-WU arch35 tail 16-63 | A5 key2, `16<=curT<64` | Padded Cube and write-back by valid rows, retaining the generic tail block synchronization protocol |
| Post-WU arch35 BF16 sub-16 | A5 key2, BF16, and `1<=curT<16` | `ComputeTailWuRegbaseArch35` fully stages the W/U seeds and Akk first; a single AIV accumulates in FP32 regbase and writes back, avoiding cross-AIV read/write contention when the W seed and W output reuse memory in place |
| Post-WU arch35 sub-16 fallback | A5 key2, non-BF16, and `1<=curT<16` | Uses the generic row-by-row Vector path; key1 and A2/A3 are covered by the Post-WU generic template |

The worst-case staging of the Post-WU sub-16 template is
`2*15*128*sizeof(BF16)+15*64*sizeof(BF16)=9,600 B`, and no workspace or public output is added.

### 5.6 FwdH Internal Templates

| Template | Selection Condition | Main Implementation |
| --- | --- | --- |
| `KdaFwdHTileShapes128` | Generic backend and `V<=128` | KDA private arch22/arch35 FwdH; on arch22, complete chunks and chunks of at least 16 rows compute `kg^T@v_new` with Cube MMAD using BF16 inputs and FP32 accumulation, and only sub-16 tail blocks use the bounded Vector fallback; cross-chunk states remain FP32; supports dense/varlen/tail |
| `KdaFwdHTileShapes256` | Generic backend and `128<V<=256` | Chunked with V=256; other semantics are the same as the 128 template |
| `ChunkKdaFwdFwdH` arch35 dense | A5 key2, BF16, non-varlen, and `T%64=0` | 16-token sub-chunks and 4-slot L1 staging, with AIC MMAD/Fixpipe cooperating with AIV gate/state updates |
| arch35 dense fused Post-WU/FwdH | `fusePostWuIntoFwdH=true` | FwdH directly forms W/U and v_new from the Prepare seeds and does not export the unrequested qg/v_new/h |

The state, W, and v_new L1 slots of the arch35 dense FwdH use ready/free credits to protect reuse. After
the last task completes, the AIV explicitly consumes the terminal free credit returned by the AIC before
exiting the kernel, preventing unconsumed credits from entering subsequent executions.

### 5.7 Finalize Internal Templates

| Template | Selection Condition | Main Implementation |
| --- | --- | --- |
| Finalize generic template | key1, and A2/A3 key2 | Generic Cube computes the two matmuls, and the AIV completes the FP32 addition, type conversion, and sequence-major write-back |
| Finalize arch35 dense-aligned pipeline | A5 key2, non-varlen, and `T%64=0` | `ProcessOutAicPipelinedArch35` uses dual L1 slots so that MTE2 of the next output tile overlaps with the current MTE1/MMAD/Fixpipe |
| Finalize arch35 full-chunk staged | A5 key2, `curT=64`, and dense-aligned pipeline not hit | Computes `scale*(qg@h)` and `Aqk@v_new` in stages, keeping FP32 intermediate results |
| Finalize arch35 tail 16-63 | A5 key2, `16<=curT<64` | Padded Cube computes the two results, and the AIV adds them by valid rows and writes back |
| Finalize arch35 BF16 sub-16 | A5 key2, BF16, and `1<=curT<16` | `ComputeTailOutputRegbaseRows` stages h, v_new, qg, and Aqk at one time, and after a single MTE2-to-V handoff directly accumulates the two terms in FP32 regbase, bypassing the tail block scalar EventID chain |
| Finalize arch35 sub-16 fallback | A5 key2, non-BF16, and `1<=curT<16` | Generic FP32 Vector row-by-row reduction; key1 and A2/A3 are covered by the Finalize generic template |

The worst-case staging of the Finalize sub-16 template is
`(128*128+2*15*128+15*16)*sizeof(BF16)=40,928 B`, which is smaller than the existing `40,960 B` of
`gateWritebackBuf_` under key2 BF16. Therefore, the threshold is fixed at 16 and does not depend on
runtime probing.

All templates keep the same formulas, public output order, optional output semantics, and layout contract.
Template selection must not change the interfaces of `ChunkKdaFwd` L0, `aclnnChunkKdaFwd`, or
`cann_ops_transformer.ops.chunk_kda_fwd`.

## 6 Pipeline Design

- The right matrix of Prepare resides in L1, avoiding repeated movement and transposition of K/K^T.
- The AIC organizes MTE2, MTE1, Cube, and Fixpipe with L1/L0 double buffering.
- The AIV uses input staging ping-pong so that MTE2 of the next tile overlaps with VEC of the current tile.
- The A5 VEC path uses regbase dual-issue; the main numerical computation remains FP32.
- Inter-sub-chunk merging uses an independent workspace region to avoid blocking the main tile pipeline.

Performance conclusions use only `msopprof`. Target regression cases are maintained by the test directory
of this operator. For the A5 H96/T16K, K=V=128, chunk=64, and BF16 key scenario, the target value is
10 ms; the gating check reads all valid samples of the `ChunkKdaFwd` main kernel and of the `beta` layout
conversion for BSND inputs, and conservatively accumulates the maximum of each of the two components. The
check fails if the total exceeds 11 ms or if no valid sample is collected for either component. The same
key scenario is executed 50 consecutive times with non-zero inputs, and `attn_out/final_state/Aqk/Akk`
must be binary identical.

## 7 Verification Matrix

- Platforms: A2/A3/A5.
- dtypes: FP16/BF16.
- layouts: BSND/BNSD/TND/NTD.
- gates: raw/activated, safe true/false.
- Shapes: K=128, V=128/256, chunk=64/128, dense/varlen/tail/GVA.
- Attributes: final state, recomputation policy, and `state_v_first`.
