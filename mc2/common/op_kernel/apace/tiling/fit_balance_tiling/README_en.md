# fit-balance tiling Engine—Integration Guide

English | [简体中文](./README.md)

> Communication-computation pipeline balancing partitioning engine.
> This document is intended for operator developers who integrate this engine: interfaces, examples, and modeling formulas.

---

## 1. External Interfaces

All symbols are in the global namespace. Including this single header file makes them available:

```cpp
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"
#include "apace/tiling/core_occupancy_first_tiling.h"    // Fallback for calibration miss: core-utilization-first partitioning
```

### Main Entry

```cpp
CommCutResult FitBalanceCut(const FitBalanceTilingArgs &args, const FitBalanceCalibration &calib);
```

Single input parameter plus calibration. Internally, the function sequentially completes the four constructions (data flow shape flow / matmul input facts mmInput / matmul model parameters mmModel / communication model parameters commModel) → cut planner construction (MakeCutPlanner) → keyed solving (Schedule, with the four inputs).
After solving, the result is compared for timing on the same estimator against the core-utilization-first wave-geometry candidate (`DeriveCommCut`, see `core_occupancy_first_tiling.h` in the same directory), and the better one is returned.
Returns an M-axis partitioning scheme of "1 short + N long" (or the degenerate equal split/single block).
The communication engine (AICPU/CCU/AIV_URMA) is passed in through `args.commEngine`, which is parsed on the operator side by `ParseCommEngine(comm_mode)`.

### Dispatch Decision

```cpp
bool IsCommCalibrated(uint32_t rankDim, TilingCommEngine commEngine, const FitBalanceCalibration &calib);
```

Checks whether the communication calibration table matches the current shape (two-dimensional matching on communication engine + number of cards). Operators use it to dispatch the partitioning strategy:
**match → FitBalanceCut; miss → core-utilization-first partitioning as the fallback** (the common engine
`core_occupancy_first_tiling.h` in the same directory is purely geometric and has no calibration dependency; see the miss branch in the integration example);
operators can also replace the fallback when they have a more suitable strategy.

```cpp
TilingCommEngine ParseCommEngine(const char *commMode);   // "aiv_urma"→AIV_URMA, "ccu"→CCU, others (including "ai_cpu")→AICPU
```

---

## 2. Integration Example (from the Perspective of a New Operator)

A new operator directly uses the args/cut types of the engine, without any legacy type conversion:

```cpp
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"

// ── ① Calibration data: one copy per operator, the only editing surface for operator integration ─────────────────────────────────
// Fill in profile based on the pipeline structure of this operator; fill in rows with measured fitted values; fill in mmFit by dtype tier
static const FitBalanceCalibration CALIBRATION = {
    /* dataflow: Data flow shape, determined by the communication primitive and the kernel implementation */
    {
        CommComputeOrder::COMM_FIRST, // Communication first: communicate along the K axis; if communication is later, communicate along the N axis
        LocalMMPosition::FRONT,       // The local MM block is computed in advance → the engine derives the communication data volume multiplier as rankDim−1
        true,                         // mValue is a per-card share: the full volume aggregated from actual local computation (= mValue × rankDim); if false, mValue is already the full volume (×1)
    },
    /* commRows: Communication fitting rows—{communication engine, number of cards, k bandwidth slope (us/MB), b per-round startup overhead (us), c one-time fixed overhead (us)};
     * one row = one calibrated (engine, number of cards) shape; do not write rows for uncalibrated shapes (IsCommCalibrated returns false, and core-utilization partitioning is used).
     * Calibration assumption: the engine derives communication bytes and computation arithmetic tiers by dtype; the operator-side communication quantization mode (attribute switches such as pertile fp8/int8
     * compressed communication) is orthogonal to dtype—for operators whose mode is fixed to a single value, the byte difference is absorbed by the fitted slope k;
     * multi-mode operators need to split calibration rows by mode (the key has no mode dimension, so rows cannot coexist in the same table) */
    {
        { {TilingCommEngine::AICPU, 4, 26.5894, 33.17, 20.58}, // Measured on the 950 standard card with 4P (in-network row)
          // ↓ The following two rows demonstrate how to add new calibration entries (the values are for demonstration—replace them with measured values before release; delete the rows and leave them empty if not calibrated):
          {TilingCommEngine::AICPU, 8, 22.7301, 33.17, 20.58}, // New card tier: append a row with the same engine (follow this practice after 8P measurement)
          {TilingCommEngine::AIV_URMA, 4, 3.2, 4.45, 1.8} },   // New engine: the engine dimension is on the row, so just add a row to the same table
    },
    /* mmFit: matmul base block + platform specifications; computing power is left as zero by default */
    {
        {256, 256, 128},   // baseBlock
        // By default, the computing power tier is automatically derived by the formula total computing power / arithmetic bytes (950: fp16 4096 / fp8 8192 / fp4 16384);
        // When the measured effective computing power of a special scheduling shape deviates from the formula, fill in computesPerCycleOverride to overwrite it—the selection basis
        // is not limited by dtype; the operator side dispatches based on its own runtime attributes (see the "Operators with Multiple Data Flow Shapes" section)
        // Platform specifications are optional (default 950: total computing power 8192, frequency 1689.6 cycle/us); new chips overwrite them with their own specifications
    },
};

// ── ② Tiling main process: dispatch + one-line invocation ─────────────────────────────────────
ge::graphStatus MyOpTiling::DoTiling()
{
    const auto commEngine = ParseCommEngine(commMode_);   // "ai_cpu"/"ccu" → enum

    CommCutResult cut;
    if (IsCommCalibrated(rankDim_, commEngine, CALIBRATION)) {
        // Match: construct the engine input parameters and call the API (existing operators can also use MakeFitBalanceArgs to convert from legacy TilingArgs)
        FitBalanceTilingArgs args = {mValue_, nValue_, kValue_, rankDim_, aicCoreNum_,
                                     geAType_, inputADtypeSize_, outputDtypeSize_,
                                     ParseCommEngine(commMode_)};
        cut = FitBalanceCut(args, CALIBRATION);
    } else {
        // Miss fallback: core-utilization-first partitioning (purely geometric; communication follows the optimal matmul core partitioning, with no calibration dependency).
        // The SWAT shape is completed in one line; the result is converted by CutResultFromCof into the partitioning shape of this engine, and the output is unified
        QuantMatmulTilingData mmTiling;
        CommTilingData dataTiling, scaleTiling;
        GetCoreOccupancyFirstTiling(platformInfo_, mValue_, nValue_, kValue_, isFp4_, hasBias_,
                                    mmTiling, dataTiling, scaleTiling);
        cut = CutResultFromCof(dataTiling);
    }
    // When the engine consumes CommTilingData, you can use MapCutToHeadTail to fill in the partitioning result of CommCutResult
    UseCutResult(cut);
    return ge::GRAPH_SUCCESS;
}
```

There is only one integration element: **fill in CALIBRATION**. Both the structure declaration (the dataflow triple) and the measured data (fitting rows / computing power tiers) are maintained in this single place. On the calling side, construct `FitBalanceTilingArgs` (existing operators convert it through `MakeFitBalanceArgs`) and call the API with the single input parameter.
Operators with multiple shapes do not maintain the entire CALIBRATION. They split the tables by axis and assemble them in one line at the call site through `MakeFitBalanceCalibration` (see the
"Operators with Multiple Data Flow Shapes" section).

### Modeling Formulas—Computations Involving Calibration Data

**Communication model** (k/b/c of commRows):

```
Communication time        y = k·x + rounds×b + c
Communication data volume x = number of rows × communication axis length × bytes per element (MB)
```

k = bandwidth slope (us/MB), b = per-round startup overhead (us; paid once per round), c = one-time fixed overhead (us;
it does not scale with the number of rounds). Whether the communication axis length is K or N is determined by the order of computation and communication.

**Computation model** (computing power tiers of mmFit + platform specifications):

```
Time per row   gradient = n × k / (computesPerCycle × frequency × number of cores × util)
Block time     = total computation volume × gradient
Utilization    util = min(0.95, 0.75 × (K/1280)^0.7 × (harmonic mean/820)^0.7)
Harmonic mean  = (m × shares × n) / (m × shares + n)
```

- **Total computation volume**: The computation volume of a block converted into the number of rows (= block rows × multiplier). The multiplier is determined by the basis of input M—
  if M is a per-card share (perRankM = true) → ×rankDim; if M is already the full volume (perRankM = false) → ×1
  (that is, the total data volume multiplier `rowMul` in the derivation rules in the next section);
- **shares**: The data volume multiplier for the mutual dependency between communication and computation. See the next section for the derivation;
- **Exponent 0.7 / 0.6**: Use 0.7 for the main computation when the pipeline runs with normal overlap. Use 0.6 for the separate time estimation of the local blocks of FRONT/BACK (local window), which is more conservative.

computesPerCycle is the effective computing power (MACs/cycle; default = total computing power / arithmetic bytes, overwritten for special shapes); the frequency is the mmFit
platform specification; the number of cores is the runtime
input parameter aicCoreNum. The engine uses these two models for pipeline balancing (the communication duration of a long block covers the computation duration of adjacent blocks) and for partitioning benefit
determination (per-round startup overhead vs overlap benefit)—**distorted calibration data is directly reflected as distorted partitioning decisions**.

**Multi-chip operators**: Calibration is an ordinary input parameter, and the engine contains no chip branches—the operator side prepares one CALIBRATION for each chip
(the repository convention is that tiling maintains one copy in a separate directory per chip), and selects one after detecting the platform:

```cpp
static const FitBalanceCalibration CALIB_950 = {...};
static const FitBalanceCalibration CALIB_NEW = {...};    // Frequency/L2/effective computing power/communication rows all follow the new chip
// Inside DoTiling: PlatformAscendC platform(context_->GetPlatformInfo()); select the chip based on the result
const FitBalanceCalibration &calib = IsNewChip(platform) ? CALIB_NEW : CALIB_950;
if (!IsCommCalibrated(rankDim_, commEngine, calib)) { /* miss → core-utilization-first fallback (see above) */ }
cut = FitBalanceCut(args, calib);
```


### Data Flow Shape (the CALIBRATION.dataflow Triple)

The **structure declaration** fixed at operator integration time. The engine derives all remaining structural quantities based on it:

| Field | Question Answered | Meaning of Values |
|---|---|---|
| `order` | Which stage of the pipeline **goes first**? | `COMM_FIRST`: communication first (receive input before computing); `COMPUTE_FIRST`: computation first (compute before sending output) |
| `localMMPosition` | Has the kernel side implemented the "special scheduling of local computation blocks"? | `FRONT`: when communication is first, the local computation blocks that do not participate in communication do not depend on communication and start computing in advance (the kernel has implemented this scheduling); `BACK`: when communication is later, the blocks that communication depends on are computed first so that communication can start as early as possible, and the local blocks are computed later (the kernel has implemented this scheduling); `NONE`: the kernel has no such special implementation |
| `perRankM` | The basis of mValue: **per-card share** or **full volume of actual local computation**? | `true`: mValue is a per-card share—the total number of actually computed mm rows and the bytes of a single communication are both mValue × rankDim (the full volume aggregated from actual per-card computation, for communication aggregation/dispatch operators); `false`: mValue is already the full volume of actual local computation, and both are ×1 |

**Derivation rules** (internal to the engine; integrators do not need to fill in any further structural quantities):

| Derived Quantity | Rule |
|---|---|
| Communication bearer axis `axis` | Communication first → K axis (A-side data is transmitted, and the other dimension = kValue); communication later → N axis (C-side data is transmitted, and the other dimension = nValue) |
| Total data volume multiplier `rowMul` (= commVolumeMul) | mValue is a per-card share (perRankM = true) → rankDim; mValue is already the full volume (perRankM = false) → 1 |
| Data volume multiplier for the mutual dependency between communication and computation `shares` | Local blocks placed first (FRONT) → rankDim − 1 (local computation blocks do not participate in communication and are modeled separately as a single share); others → same as rowMul |

Consequences of a wrong choice: wrong order/localMM → misaligned balancing anchor, wrong short block position, and wrong total time formula shape (overall degradation of partitioning);
wrong perRankM → overall offset in the estimated communication volume and mm total volume.

Decision method (consider only the communication direction and kernel implementation of this operator, without considering other factors):
1. **order**: If the input of communication is the input of matmul → `COMM_FIRST`; if the input of communication is the output of matmul →
   `COMPUTE_FIRST`
2. **localMMPosition**: Bound to the communication-computation order and the kernel-side implementation—
   - Communication is first, local computation blocks that do not participate in communication exist and can start computing in advance without depending on communication, and the kernel side
     has implemented this scheduling → `FRONT`
   - Communication is later, local computation blocks not depended on by communication exist, the blocks that communication depends on are computed first so that communication can start as early as possible, the local
     blocks are computed later, and the kernel side has implemented this scheduling → `BACK`
   - The kernel side has no special scheduling implementation for local computation blocks → `NONE` (even if local blocks exist in the data flow)
3. **perRankM**: If the mValue passed to the engine is a per-card share (actual computation volume = mValue × rankDim) → `true`;
   if mValue itself is the full volume of actual local computation → `false`

### Operators with Multiple Data Flow Shapes / Multiple Computing Power Shapes (Per-Axis Calibration + Runtime Assembly)

The dataflow triple is fixed at operator integration time, but the `calib` parameter of `FitBalanceCut` is an **ordinary parameter passed in per call**
(not a global singleton). For operators with more than one shape (runtime attribute switching of localMMPosition, and so on) or whose calibration needs to be assembled at runtime,
maintain the calibration per axis with `MakeFitBalanceCalibration` and assemble it in one line at the call site:

```cpp
// Calibration header: one table per axis—the axes are decoupled, each table stores only the independent variants of its own axis, and combinations are not amplified on the table.
// Use const for the tables (do not use constexpr): unfilled items use NSDMI default values, and some host compilers do not support
// constexpr aggregate initialization of NSDMI; calibration is assembled at runtime, so the tables do not need to be constant expressions
const FitBalanceCalibration::DataflowShape DATAFLOWS[] = {
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::NONE,  true},   // Shape 0
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::FRONT, true},   // Shape 1
};
const FitBalanceCalibration::CommFitRow COMM_ROWS[] = {             // Communication does not vary with the axis: globally shared
    {TilingCommEngine::AICPU, 4, 26.5894, 33.17, 20.58},
};
const FitBalanceCalibration::MatmulFit MM_FITS[] = {                // Each element = the mmFit of one shape; unfilled items use default values
    {{256, 256, 128}, 0.0, 8192.0, 1.65 * BYTES_PER_KB},            // Fully explicit (regular shape; computing power 0 = use the formula)
    {{128, 128, 128}, 7000.0},                                      // Special scheduling: different base block/computing power, default platform specifications
};

// Scenarios exist where DATAFLOWS and MM_FITS are dynamically selected; choose the permutations and combinations based on the actual needs of the operator
const FitBalanceCalibration calib = MakeFitBalanceCalibration(
    DATAFLOWS[dataFlowIdx], COMM_ROWS, MM_FITS[mmFitsIdx]);
if (IsCommCalibrated(rankDim_, commEngine, calib)) {
    cut = FitBalanceCut(args, calib);
}
```
