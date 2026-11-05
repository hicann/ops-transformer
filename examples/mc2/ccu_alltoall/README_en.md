# CCU AllToAll + MXFP8 MatMul Communication-Computation Fusion Sample (ccu_alltoall)

English | [简体中文](./README.md)

This is a sample of CCU direct invocation (`<<<>>>` kernel direct invocation) AllToAll collective communication + Blaze MXFP8 MatMul **dual-stream communication-computation fusion**:
a single `AllToAllMatmulCustom` invocation dispatches two kernels at the same time — the `__ccu_host__` communication
kernel on streamCcu and the AIC fused computation kernel on streamAicore, with tile-level overlap between communication and computation.

- Data semantics (K-parallel GEMM): `Y[M/R,N] = Σ_r recvData[r-th block,:] @ x2[r*K segment,:]` (L0C cross-source accumulation,
  the self term directly reads x1); `x1=[M,K]` FP8 E4M3FN (the token axis is evenly divided into R blocks for A2A),
  `workspace=[recvData|recvScale]`, `x2=[R*K,N]` complete weights without communication, and the output is BF16
- End-to-end ~230 us (4 ranks, M=8192 / tileM=512 / K=2048 / N=3584; see "Performance Test Method" for the criteria)
- Only Ascend 950 (dav-3510) + CANN ≥ 9.2.0 are supported; the number of ranks = the number of NPUs on the local machine (one thread per card in the process)

## Directory Structure

```
ccu_alltoall/
├── src/                                 # Single translation unit (main.cpp is the only compiled source, -xasc)
│   ├── main.cpp                         # Sample entry: read data → FillTiling → invoke → verify → release
│   ├── all_to_all_matmul_ccu.asc        # Anchors/advanced APIs (§1 host orchestration / §2 AIC kernel entry + Custom)
│   ├── ccu_all_to_all.asc               # CCU communication service (contract + engine + resources / tiling contract / host assembly / __ccu_host__ kernel)
│   ├── all_to_all_matmul_tiling_data.h  # Tiling aggregate struct + FillAllToAllMatmulTilingData (SWAT)
│   └── all_to_all_matmul_aic.h          # AIC computation logic (synchronization primitives / WaitPolicy + matmul orchestration)
├── scripts/
│   ├── gen_data.py                      # Data generation (bitwise closed form, bitwise identical to the original C++ generator)
│   ├── run.sh                           # Data (as needed) + build + run / --perf msprof collection
│   └── parse_prof.py                    # msprof op_summary parsing
├── CMakeLists.txt                       # Self-contained build entry (standalone cmake, see below)
├── README.md
└── .gitignore                           # build/ data/ output/ prof/
```

Include chain (single TU; the order is the dependency order):

```
main.cpp(-xasc) → all_to_all_matmul_ccu.asc (anchor, the only inclusion)
                → { ccu_all_to_all.asc → all_to_all_matmul_tiling_data.h → all_to_all_matmul_aic.h } → anchor body
```

Suffix rules: `.asc` = defines a kernel entry [included only once]; `.h` = pure logic.

The matmul kernel / tiling / utils are still referenced **in place** from the apace repository (not copied). During the build they are resolved with the `apace/...` prefix through
`-I <repo>/mc2/common/op_kernel`:

- `apace/kernel/matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel.h`
- `apace/utils/op_state_dump.h`
- `apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h` (shared by hcomm/urma)
- `apace/tiling/quant_matmul_tiling_swat.h`

## Mechanism Summary (quoted from the source file header)

- **Dual-stream architecture** (a single `AllToAllMatmulCustom` invocation dispatches two kernels):
  - streamCcu: `CcuAllToAllMesh1DKernel` (`__ccu_host__`) — after wake, a single class of publishing is performed once, looping
    for dataCommTurn rounds (round 0 = data block 0 + the entire scale block carried along), with PostSync landing confirmation at the end of each round +
    EventRecord to notify the AIC
  - streamAicore: `AllToAllQuantMatmulKernelCcu` — block 0 clears residual event values → commit (wake),
    all cores perform fused matmul and poll round-done tile by tile, achieving tile-level communication-computation overlap
- **Event contract** (AIC↔CCU synchronization): `ev[0] = wake` (AIC commit → CCU process, the only time before the loop);
  `ev[t+1] = data round t done` (CCU notify → AIC wait; round 0 includes the entire scale block carried with it)
- **Parameters directly reaching the stage**: each host invocation flushes the communication parameter block into `kernelArg->stage` (delivered by value with the CCU kernel
  launch, without chasing secondary pointers); the kernel reads it itself upon startup and derives it in closed form by round
  (`LoadRoundArgs`); the AIC does not handle communication parameters
- **Single TU and kernel name freezing**: the `__ccu_host__` kernel of `ccu_all_to_all.asc` can be
  included only once per binary (the anchor `all_to_all_matmul_ccu.asc` is the only inclusion point); the kernel name
  `CcuAllToAllMesh1DKernel` participates in the binaryCacheTag hash (tag = name hash + rankId; the correctness of coexisting multi-rank threads in a single process depends on it) and must never be renamed
- **Naming rules** (word root = functional domain, preventing evolution drift): `Ccu` = CCU transport primitives (touching only register/instance/channel
  semantics, independent of the specific collective communication algorithm); `AllToAll` = service functions (semantics bound to the AllToAll algorithm)

## Environment and Build

- Ascend 950 (dav-3510), CANN ≥ 9.2.0 (depends on the official asc-comm host launch header
  `ccu_host_launch.h`)
- Depends on ops-tensor (blaze / tensor_api, pin 43d554d): prefer reusing
  `mc2/common/op_kernel/apace/.wtdeps/ops-tensor` (use it if it exists); otherwise, automatically pull it through FetchContent to
  `build/third_party/` (network required)
- ⚠️ Overlay pitfall: if multiple cann-* directories under `~/Ascend` on the local machine overlay each other (`asc/include` mixed with an old-version
  tensor_api), explicitly run `export ASCEND_HOME_PATH=<cann-9.2.0 path>` before building/running, and do not rely on the latest symlink
- Build (the examples tree has no unified build entry; this sample uses standalone cmake):

```bash
export ASCEND_HOME_PATH=~/Ascend/cann-9.2.0/x86_64-linux        # Or apace/.wtdeps/cann
export PATH=~/Ascend/cann-9.2.0/x86_64-linux/ccec_compiler/bin:$PATH
cmake -S . -B build
cmake --build build --target ccu_alltoall --parallel 8
```

## Data Generation and Running

Data is generated by `scripts/gen_data.py` (pure stdlib; bitwise closed form identical to the original apace ST C++ generator).
At runtime, main reads `data/` (`--data <dir>` is optional):

```bash
python3 scripts/gen_data.py --M 8192 --K 2048 --N 3584 --ranks 4 --out data
export HCCL_OP_EXPANSION_MODE=CCU_SCHED        # CCU scheduling mode (required)
./build/ccu_alltoall precision 8192 512 2048 3584    # M tileM K N
./build/ccu_alltoall perf 8192 512 2048 3584 10      # + loop
./build/ccu_alltoall --data /path/to/data precision 512 128 256 256
```

Alternatively, use the orchestration script directly (data generated as needed + build + run):

```bash
bash scripts/run.sh                                       # Data (as needed) + build + precision, default 512 128 256 256
bash scripts/run.sh precision 8192 512 2048 3584          # Any case (meta mismatch triggers automatic regeneration)
bash scripts/run.sh perf 8192 512 2048 3584 10            # Direct perf run
bash scripts/run.sh --perf perf 8192 512 2048 3584 10     # perf + msprof collection (prof/)
bash scripts/run.sh --gen precision 8192 512 2048 3584    # Generate data only
bash scripts/run.sh --skip-build ...                      # Skip compilation
```

The number of ranks = the number of NPUs on the local machine (detected by `npu-smi info -l`, overridable by the `CCU_A2A_RANKS` environment variable); constraints:
`M` must be divisible by the number of ranks, `blockRows(=M/R)` must be divisible by `tileM` (dataCommTurn ≤ 16), and `K` must be a
multiple of 64 (MXFP8).

### Data Files

| File | Shape/Size | Description |
|---|---|---|
| `meta.txt` | One line `M K N R` | Consistency contract between the shape and the number of ranks (a mismatch reports an error and provides the regeneration command) |
| `x1_rank{r}.bin` | [M,K] FP8 E4M3FN | Input of rank r (row-constant encoding, row-level traceability) |
| `x1_scale_rank{r}.bin` | M×(K/64×2) B | E8M0 all 1.0 |
| `x2.bin` | [R×K,N] FP8 E4M3FN | DN column-major weights (shared by all ranks, no communication) |
| `x2_scale.bin` | R×(K/64×2)×N B | E8M0 all 1.0 |
| `y_expect_rank{r}.bin` | [M/R,N] BF16 | Expected output (compared in precision mode; skipped in perf mode) |

## Verification Semantics

- The expected value is a bitwise closed form: `Y[m] = K × Σ_{r=0}^{R-1} 2^((2i+r+m)%4 + r%2)` (each addend is a power of 2,
  exactly representable in float32; taking the high 16 bits yields the BF16 bit pattern) — **bit-level assertion with zero rounding noise**
- precision mode: the output is compared element by element against `y_expect_rank{r}.bin` in BF16 bit patterns, and PASS is returned only when all are identical
  (`=== ALL RANKS PASS ===`); regardless of success or failure, the output is written to `output/<rank>/npu_out.bin`
- Row-level traceable data can locate errors by category: source rank confusion / destination routing confusion / row-round misalignment / byte corruption

## Performance Test Method

### Test Commands

```bash
# Environment preparation (after each login)
export HCCL_OP_EXPANSION_MODE=CCU_SCHED

# Accuracy verification
./build/ccu_alltoall precision 8192 512 2048 3584

# Performance test (parameters: M tileM K N loop)
./build/ccu_alltoall perf 8192 512 2048 3584 20

# msprof collection (standard on-board criteria)
msprof --application="./build/ccu_alltoall perf 8192 512 2048 3584 20" \
       --output=./prof --ai-core=on
```

### Viewing Results

```bash
# Method 1: view the program output directly (event timing, end-to-end criteria)
# Focus on the "steady avg=XXXus/iter" line

# Method 2: msprof parsing (kernel duration criteria, standard metrics)
# Open prof/PROF_*/mindstudio_profiler_output/op_summary_*.csv
# Filter the rows whose Op Name contains "AllToAllQuantMatmulKernelCcu"
# Take the Task Duration (us) column, and take the median of each rank after removing the first frame
```

**Which metric to choose**:

- **Rank mean**: take the median of each of the 4 ranks and then average — used for comparison with historical baselines (240 us criteria)
- **Makespan** (the slowest of the 4 ranks per frame): the real time consumption of a frame — used for reporting throughput
- For large segments (≥8MB), the two criteria are basically the same; for small segments (≤4MB), skew dominates, so only makespan matters

### Why Heavy Operators Are Inserted

**Problem**: the host threads of the 4 cards submit kernels separately (Refresh → CCU launch → AIC launch), and the submission times
differ (~50-300 us deviation). The AIC kernel of the rank that submits first starts first, but it has to wait for the data of the slowest rank —
the waiting time is counted in its duration, resulting in **a difference of more than 50 us in time consumption across ranks**, making the data unreliable.

**Solution**: after each iteration ends, insert a 4×128 MB memset on streamAicore (~430 us device occupancy). While the device
is busy, all host threads complete the submission of the next frame. When the memset ends, the kernels of the 4 ranks start at the same time —
**the rank range converges from more than 50 us to 5~21 us**.

**Principle**:

```
Without a heavy operator:  rank0 submits (fast) → kernel starts → waits for data (~50 us) → computes
              rank3 submits (slow) → kernel starts → computes (data has already arrived)

With a heavy operator:  rank0 submits → memset (~430 us) ← rank3 also completes submission during this period
              rank3 submits → memset (~430 us)
              memset ends → the kernels of the 4 ranks start at the same time → compute at the same time
```

**Note**: when parsing msprof results, filter by kernel name (only AllToAllQuantMatmulKernelCcu is counted), and the memset
is automatically excluded from statistics. The heavy operator affects only small segments (≤4M, with a benefit of −17~−31%); for large segments (≥8M), communication/computation
dominates and there is no impact.

### Common Cases

| Purpose | Command Parameters (M tileM K N) |
|---|---|
| Target case (16 MB) | `8192 512 2048 3584` |
| Small case smoke test | `512 128 256 256` |
| Data volume sweep | `8192 512 {128\|256\|512\|1024\|2048\|4096\|8192} 3584` |

## Relationship with the apace Repository

- This sample is the **only copy** of the CCU variant (migrated from the three-layer apace structure: the four pieces of `core/ccu_comm` +
  the three CCU pieces of `kernel/fusions/all_to_all_quant_matmul` + `tests/st/all_to_all_quant_matmul/ccu`,
  merged into the five pieces of src/; the mechanism is unchanged, and only the include/comment paths are adjusted with the migration)
- The matmul kernel / tiling / utils reference apace in place (see the list in the "Directory Structure" section), and the build depends on the repository layout
- The ST of the hcomm / urma communication variants is still in `mc2/common/op_kernel/apace/tests/st/all_to_all_quant_matmul/`
