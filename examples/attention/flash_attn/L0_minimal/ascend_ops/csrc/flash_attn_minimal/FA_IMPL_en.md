# flash_attn_minimal Operator Details

## 1. Operator Design

### 1.1 Function and Architecture

Based on the AscendC framework of Huawei Ascend NPUs, this operator adopts the **Host (CPU) launch + Device (NPU) kernel** architecture and implements the standard self-attention computation $Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V$. For interface parameters, see [README_en.md](./README_en.md).

Operator constraints: the Q/K/V data type is `bfloat16`, `D=128`, `S1=S2` and is divisible by 128, and `N2` divides `N1` (GQA is supported). It computes dense (full) attention. The input uses the `BSND` layout, and B/N1/N2/S1/S2 are all passed to the kernel as runtime parameters, with no separate Host Tiling stage.

Call chain: **Python** `torch.ops.ascend_ops.flash_attn_minimal(q,k,v,softmaxScale)` → **C++ entry** `FlashAttnNpu()` (`torch_interface.cpp`) → **Kernel dispatch** `FaKernel` (`fa_kernel_interface.h`) → dispatch by core type to `CubeFunc` (`fa_block_cube.h`) / `VectorFunc` (`fa_block_vec.h`).

### 1.2 Basic Block and Multi-Core Splitting Strategy

The kernel uses **M=128, N=128, D=128** as the basic tile (template parameters `M_BASE / N_BASE / D_SIZE` in `fa_kernel_interface.h`). Each AIC Cube core is paired with 2 AIV Vector cores (`KERNEL_TYPE_MIX_AIC_1_2`), and each AIV processes half of M (`M_BASE/2 = 64` rows). Therefore, the BMM result in UB is organized in units of a `64×128` half-block.

Cube and Vector both evenly divide the cores by **BN1 = B×N1** (batch × total number of query heads):

- AIC core (`CubeFunc`, `fa_block_cube.h`): `bn1PerCore = ceil(BN1/blockDim)`, processing `bn1 ∈ [aicIdx×bn1PerCore, +bn1PerCore)`.
- AIV core (`VectorFunc`, `fa_block_vec.h`): `aicIdx = aivIdx / 2`, `subBlockIdx = GetSubBlockIdx()` (0/1). The two AIVs are responsible for the upper and lower halves of the 64 rows of the tile respectively, and their processing ranges match those of the paired AICs.
- blockDim is computed by `platform_ascendc::CalcTschBlockDim(64, ...)` (`torch_interface.cpp`), and the entry uses `fixedAivNum=64`.

GQA mapping: in each BN1 iteration, Cube and Vector compute the corresponding KV head through `n2Head = n1Head * N2 / N1` (in `fa_block_cube.h` / `fa_block_vec.h`, implicit in the `bn1` decomposition), with no need to merge along the G axis.

Two-level main loop (`fa_block_cube.h` / `fa_block_vec.h`):

```
for bn1 in range(start, end):          ← head range assigned to the current core
    bIdx   = bn1 / N1
    n1Head = bn1 % N1
    n2Head = n1Head * N2 / N1          ← GQA: query head → KV head mapping
    for s1Block in range(numS1Blocks):  ← num S1 blocks = ceil(S1/M_BASE)
        for s2Block in range(numS2Blocks):  ← num S2 blocks = ceil(S2/N_BASE)
            C1(s1Block,s2Block) → V1 → C2 → V2   ← sequential execution, no software pipelining
```

For the typical shape `B=1, N1=32, N2=1, S1=8192, S2=8192`:

| Constant | Value | Meaning |
|------|-----|------|
| `BN1` | `1×32 = 32` | Total number of head groups (that is, the total number of query heads) |
| `numS1Blocks` | `8192/128 = 64` | Number of S1 blocks per head |
| `numS2Blocks` | `8192/128 = 64` | Number of KV (flash) blocks per row |

blockDim (from the AIC perspective) is usually 32 → `bn1PerCore = 32/32 = 1`, that is, **each AIC core processes 1 query head**, with 64×64 = 4096 tiles per head.

Each row attends to all of S2, and a row does not span cores, so the output of each core is the final result, with no cross-core reduction required.

### 1.3 Memory Partitioning

The storage hierarchy on the NPU, from largest to smallest, is GM → L1 → L0A/L0B/L0C → UB. All buffers are allocated with **explicit linear addresses** (`LocalTensor<T>(TPosition, addr, elems)`). Cube and Vector use **the same addresses for the shared regions (the BMM result in UB and the P matrix in L1)** to accomplish cross-core data exchange.

#### GM (Global Memory)

Q/K/V and the output all use the `BSND` format (`FaGmTensor<T, GmFormat::BSND>`, `offset_calculator.h`). The workspace is an empty tensor (`at::empty({0})`) and is not used by the kernel.

#### L1 (AIC-side staging, `fa_block_cube.h`)

| Buffer | Size | Description |
|--------|------|------|
| `l1P` (P matrix, `M×N` bf16) | 128×128×2 = 32KB | Written by the AIV (`CopySoftmaxToL1`), read by the AIC (BMM2) |
| `l1Q` (Q tile, `M×D` bf16) | 128×128×2 = 32KB | Reused for subsequent S2 iterations after the first-round S2 load of each S1 block |
| `l1K` (K tile, `N×D` bf16) | 128×128×2 = 32KB | Reloaded for each S2 block |
| `l1V` (V tile, `N×D` bf16) | 128×128×2 = 32KB | Reloaded for each S2 block |
| **L1 total** | **128KB** | Linearly laid out from address 0 |

#### L0 (dedicated to the AIC Cube, `fa_block_cube.h`)

| Space | Size | Description |
|------|------|------|
| **L0A** | `128×128` bf16 × 2 | Double-buffer, stores Q (BMM1) or P (BMM2) |
| **L0B** | `128×128` bf16 × 2 | Double-buffer, stores K (BMM1) or V (BMM2) |
| **L0C** | `128×128` fp32 × 1 | Single buffer, stores the Matmul result |

L0A/L0B use ping-pong double buffering (`l0aIdx`/`l0bIdx` rotate with `1-idx` inside `MatmulFull`), and L0C is a single buffer (before each BMM, `get_buf PIPE_M` → `Mmad` → `Fixpipe` → `rls_buf`).

#### UB (dedicated to the AIV Vector + written by the AIC through Fixpipe, `fa_block_vec.h`)

The AIC writes the matrix multiplication results into UB through Fixpipe (L0C→UB) for the AIV to consume. `ubBmm1`/`ubBmm2`/`l1P` have identical addresses in Cube and Vector (shared across cores); the rest are private to the AIV.

| Buffer | Size | Description |
|--------|------|------|
| `ubBmm1` | 64×128 fp32 = 32KB | BMM1 result, Cube→UB, consumed by Vec |
| `ubBmm2` | 64×128 fp32 = 32KB | BMM2 result, Cube→UB, consumed by Vec |
| `softmaxSum` | 64 fp32 = 256B | Softmax denominator sum (indexed by row) |
| `softmaxMax` | 64 fp32 = 256B | Softmax numerator max (indexed by row) |
| `softmaxExp` | 64 fp32 = 256B | Exponential scaling factor of Flash Update (indexed by tile) |
| `commonBuf` | 512B | General-purpose temporary space (used for sum/max updates) |
| `stage1Buf` | (64+2)×128 bf16 ≈ 16.5KB | Softmax result P (cast to bf16), moved to L1 |
| `stage2Buf` | 64×128 fp32 = 32KB | Flash Update accumulator |
| `castBuf` | **In-place reuse at the same address** as `stage2Buf` (64×128 bf16) | In-place downcast of the output from fp32 to bf16 |

> `stage2Buf` and `castBuf` share the same address at the tail of UB: `Cast` reads fp32 (4B/elem) and writes bf16 (2B/elem), so the write pointer advances more slowly than the read pointer, and the in-place conversion is safe, saving UB space.

### 1.4 Cross-Core Synchronization

The AIC and AIV are different physical cores. They run the same binary but take different branches (`#ifdef __DAV_C310_CUBE__`). Within a tile, execution proceeds sequentially as **C1 (BMM1) → V1 (Softmax) → C2 (BMM2) → V2 (FlashUpdate/output)**, and cross-core producer-consumer synchronization uses `CrossCoreSetFlag`/`CrossCoreWaitFlag`.

Cube and Vector **advance sequentially** in the inner S2 loop: after finishing BMM1, Cube sets `FLAG_BMM1_READY`; Vector waits for this flag → performs Softmax and copies L1P → sets `FLAG_BMM1_READY` (release) + `FLAG_L1P_READY` (notify Cube); Cube waits for `FLAG_L1P_READY` + `FLAG_BMM2_READY` → performs BMM2 → sets `FLAG_BMM2_READY`; Vector waits for `FLAG_BMM2_READY` → performs Rescale → releases `FLAG_BMM2_READY`. This sequential execution does not use PRELOAD software pipelining.

Cross-core flag definitions (`fa_block_cube.h` / `fa_block_vec.h`):

| Flag | ID | Meaning | Setter | Waiter |
|------|----|------|--------|--------|
| `FLAG_BMM1_READY` | 0 | BMM1 result in UB is ready | Cube | Vector |
| `FLAG_L1P_READY` | 1 | L1P is ready | Vector | Cube |
| `FLAG_BMM2_READY` | 2 | BMM2 result in UB is ready | Cube | Vector |

Each flag has an additional `+16` mirror corresponding to the second AIV sub-core (`fa_block_cube.h` and others). On the Vector side, initialization presets `FLAG_BMM1_READY` and `FLAG_BMM2_READY` (released state, `fa_block_vec.h`). After the Cube main loop finishes, it performs a barrier wait on these two flags and their `+16` mirrors, ensuring that the AIV has finished consuming the shared buffers before exit (`fa_block_cube.h`).

AIV-side local `get_buf/rls_buf` event IDs: `stage1Buf` / `ComputeSoftmax` uses `E_VEC1_ID = 0`; `stage2Buf` / `Cast` / output uses `E_VEC2_ID = 1`. The two do not conflict.

---

## 2. Code Structure

### 2.1 Overall Architecture Overview

```
torch_interface.cpp              ← PyTorch registration + entry (input validation, directly launches the kernel)
  └─ op_kernel/fa_kernel_interface.h   ← Device kernel entry (dispatches CubeFunc/VectorFunc by core type through #ifdef)
       ├─ kernel/fa_block_cube.h       ← AIC Cube: Q/K/V data movement + BMM1/BMM2 + Fixpipe
       └─ kernel/fa_block_vec.h        ← AIV Vector: Softmax + Flash Update + output
```

Auxiliary functional modules:

| Directory/File | Function |
|------|------|
| `op_kernel/fa_kernel_public.h` | Common header/namespace imports + utility functions (Max/Min/Clip) |
| `op_kernel/memcopy/offset_calculator.h` | GM multi-dimensional indexing (`FaGmTensor<BSND>`) + coordinate structs + address computation |
| `op_kernel/memcopy/memory_copy.h` | `CopyMatrixGmToL1` (ND→NZ) + `CopyAttnOutBN1UbToGm` (UB→GM) |
| `op_kernel/matmul/matmul.h` | `MatmulFull` (L1→L0A/L0B `LoadData` + `Mmad`) + `MMParam` |
| `op_kernel/vector/vf/vf_softmax.h` | Flash Softmax accelerated by VF instructions: `ProcessVec1Vf` (update / no_update, `EQ_128` path) + `UpdateExpSumAndExpMax` |
| `op_kernel/vector/vf/vf_softmax_impl/` | Concrete implementations in `vf_softmax_aligned128_{no_update,update}.h` + constants in `vf_softmax_const.h` |
| `op_kernel/vector/vf/vf_rescale.h` | `FlashUpdate` / `FlashUpdateLast` / `LastDiv` (Flash Update Rescale) |

### 2.2 Torch Interface: `torch_interface.cpp`

The single entry point `FlashAttnNpu()` (`torch_interface.cpp`):
1. `CheckInput` validates bf16, 4 dimensions, `D=128`, plus `S1=S2`, `S1%128=0`, `N1>=N2`, `N1%N2=0`, and consistent batch dimensions.
2. Obtains the platform information and the current NPU stream, and allocates an output of the same shape and an empty workspace.
3. Fixes `fixedAicNum=32 / fixedAivNum=64`, and sets `blockDim = CalcTschBlockDim(64, GetCoreNumAic(), GetCoreNumAiv())`.
4. Launches the kernel with `FaKernel<<<blockDim, nullptr, aclstream>>>(gq, gk, gv, go, gws, blockDim, scale, B, N1, N2, S1, S2)`, and then synchronizes with `aclrtSynchronizeStream`.

The operator registers the schema `flash_attn_minimal(Tensor q, Tensor k, Tensor v, float softmaxScale=0) -> Tensor` through `TORCH_LIBRARY(ascend_ops, ...)`, and binds the implementation to the NPU backend through `TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, ...)`. `PyInit__C` provides the Python module entry. The operator registration is fully self-contained in this file and does not require an external `npu_ops_def.cpp`.

### 2.3 Device Kernel: `op_kernel/`

#### 2.3.1 Entry and Dispatch (`fa_kernel_interface.h`)

After receiving the runtime shape parameters (`B, N1, N2, S1, S2`) and setting `KERNEL_TYPE_MIX_AIC_1_2` and `InitSocState()`, `FaKernel` dispatches by compilation macro: `__DAV_C310_CUBE__` → `CubeFunc<M,N,D>`, otherwise → `VectorFunc<M,N,D>`. The template defaults are `M=N=D=128`.

#### 2.3.2 Cube Side (`fa_block_cube.h`)

`CubeFunc()` creates the GM tensors (`BSND` format), allocates L1/L0/UB, and then executes sequentially in the three-level loop `bn1 → s1Block → s2Block`:

Core functions:

- **`ComputeMm1`** (BMM1: Q@Kᵀ, `fa_block_cube.h`):
  1. Waits for `FLAG_BMM1_READY` (ubBmm1 is free).
  2. When `isFirstS2Loop` holds, performs `CopyMatrixGmToL1` (Q GM→L1, ND→NZ), which is reused for subsequent S2 iterations of this S1 block.
  3. Performs `CopyMatrixGmToL1` (K GM→L1, ND→NZ) for each block.
  4. Performs `MatmulFull` (L1→L0A/L0B ping-pong → `Mmad` → L0C), with `isRightTranspose=true` implementing Kᵀ.
  5. Performs `FixpipeMm1` (L0C→UB, `dualDstCtl=1` splits it into two `64×128` fp32 copies for the 2 AIVs) → `CrossCoreSetFlag(FLAG_BMM1_READY)`.

- **`ComputeMm2`** (BMM2: P@V, `fa_block_cube.h`):
  1. Waits for `FLAG_L1P_READY` + `FLAG_BMM2_READY`.
  2. Performs `CopyMatrixGmToL1` (V GM→L1, ND→NZ).
  3. Performs `MatmulFull` (L1P@V, `realM=mActual`) → `FixpipeMm2` (L0C→UB dual-dest) → `CrossCoreSetFlag(FLAG_BMM2_READY)`.

#### 2.3.3 Vector Side (`fa_block_vec.h`)

`VectorFunc()` allocates the shared UB (`ubBmm*`, `l1P`) and the private UB, initializes the 2 cross-core flags to the released state, and then executes sequentially in the three-level loop `bn1 → s1Block → s2Block`:

- **`ComputeSoftmax`** (`fa_block_vec.h`): waits for `FLAG_BMM1_READY` → the first block takes `ProcessVec1Vf<...false>` (no_update), and non-first blocks take `<...true>` (update): row by row, `QKᵀ×scale → row_max → exp(x-max) → sum → cast bf16`; for non-first blocks, `UpdateExpSumAndExpMax` further merges the historical max/sum.
- **`CopySoftmaxToL1`** (`fa_block_vec.h`): `DataCopy` moves `stage1Buf` (P, bf16) to `l1P` (positioned to the upper/lower half by `subBlockIdx`), and `CrossCoreSetFlag` releases `FLAG_BMM1_READY` + sets `FLAG_L1P_READY`.
- **`ComputeRescale`** (`fa_block_vec.h`): waits for `FLAG_BMM2_READY` → when `isFirstS2Loop` holds, `DataCopy` initializes `stage2Buf`; intermediate blocks perform `FlashUpdate` (`stage2 = stage2·expMax + cur`); the last block performs `FlashUpdateLast` (accumulate and then divide by sum) → `CrossCoreSetFlag` releases `FLAG_BMM2_READY`.
- **`CopyAttnOutToGm`** (`fa_block_vec.h`): triggered only when `isLastS2Loop` holds: when `isFirstS2Loop` holds, `LastDiv` runs first (divide by sum), then `Cast` converts fp32→bf16 (written in place to `castBuf`), and `CopyAttnOutBN1UbToGm` writes back to GM (written in the `BSND` format at the position `s1BlockBase + subBlockIdx*halfM`).

---

## 3. Walkthrough of a Typical Case

Take `B=1, N1=32, N2=1, S1=8192, S2=8192, D=128` (GQA, G=32) as an example, running on 32 AICs + 64 AIVs.

### 3.1 Multi-Core Allocation

- `BN1 = B×N1 = 1×32 = 32`.
- blockDim (from the AIC perspective) = 32 → `bn1PerCore = ceil(32/32) = 1`. **Each AIC core processes 1 query head**, paired with 2 AIV cores that process the upper/lower halves of the 64 rows of that head.
- Within each core: `numS1Blocks=64` × `numS2Blocks=64` = 4096 tiles, executed sequentially.

### 3.2 Computation Flow of a Single Tile

One tile processes the S1 block of 1 query head (`n1Head`) (`s1Block × 128` rows) and attends to the KV head (`n2Head = n1Head × 1 / 32 = 0`).

**Step A — BMM1 (AIC Cube)**

1. **Copy Q** (only when `s2Block==0`): `CopyMatrixGmToL1` reads 128 rows from `Q[b=0, n1=n1Head, s1=s1BlockBase, d=0]` → L1 (NZ, 128×128 bf16 = 32KB). Subsequent S2 iterations of the same S1 block reuse this L1Q.
2. **Copy K**: `K[0, 0, s2Block×128, 0]` → L1 (NZ, 32KB).
3. **Matmul**: `Q(128×128) × Kᵀ(128×128)` → L0C.
4. **Fixpipe** L0C→UB: dual-dest splits it into 2×`64×128` fp32 copies, and the two AIVs are notified.

**Step B — Vec1 / Softmax (AIV Vector, 64 rows per sub-core)**

1. Waits for `FLAG_BMM1_READY` → `ProcessVec1Vf` (no_update for the first block / update for non-first blocks, `EQ_128`): `QKᵀ×scale → max → exp → sum → cast bf16`, written to `stage1Buf`.
2. For non-first blocks, `UpdateExpSumAndExpMax` updates `softmaxSum`/`softmaxMax`/`softmaxExp`.
3. `DataCopy` moves P (bf16) → `l1P`, and `CrossCoreSetFlag` releases BMM1 + notifies Cube (L1P ready).

**Step C — BMM2 (AIC Cube)**

1. Waits for `FLAG_L1P_READY` + `FLAG_BMM2_READY` → **Copy V**: `V[0, 0, s2Block×128, 0]` → L1 (NZ, 32KB).
2. `MatmulFull` `P(128×128) × V(128×128)` (`realM=mActual`) → Fixpipe L0C→UB dual-dest → notifies the AIV.

**Step D — Vec2 / Flash Update (AIV Vector)**

- First block of a row: `DataCopy` initializes `stage2Buf`;
- Intermediate blocks: `FlashUpdate` (`stage2 = stage2·expMax + cur`);
- Last block of a row: `FlashUpdateLast` (accumulate + divide by sum) → `LastDiv` (compensating division for all-zero rows) → `Cast` to bf16 (in place) → `CopyAttnOutBN1UbToGm` writes back to the corresponding output position.

In this way, the 64 S2 blocks of each row are accumulated into the final output through flash online-softmax; the 32 query heads are processed in parallel on 32 AIC cores, covering the entire output.

---

## 4. Performance Testing

The following scenarios were tested on the 950DT platform. Since the current implementation does not include Preload computation, all stages execute serially, and the MFU is low.

| B | Q_N | KV_N | Q_S | KV_S | D | Duration | MFU |
|---|-----|------|-----|------|---|----------|-----|
| 1 | 32 | 1 | 8192 | 8192 | 128 | 7853.29 | 32.40% |
