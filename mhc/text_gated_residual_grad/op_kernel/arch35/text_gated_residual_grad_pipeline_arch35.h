/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file text_gated_residual_grad_pipeline_arch35.h
 * \brief TextGatedResidualGrad workspace layout and bounded task scheduler.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_PIPELINE_ARCH35_H
#define TEXT_GATED_RESIDUAL_GRAD_PIPELINE_ARCH35_H

#include "../text_gated_residual_grad_constants.h"

#if defined(__CCE_AICORE__)
#define HC_GRAD_HOST_DEVICE __aicore__ inline
#else
#define HC_GRAD_HOST_DEVICE inline
#endif

namespace TextGatedResidualGrad {

constexpr uint32_t HC_GRAD_FULL_BATCH_SCHEDULE = 2U;
constexpr uint32_t HC_GRAD_A5_FULL_K_SCHEDULE = 3U;
// The A2 tiler reports the usable UB after the 256-byte platform reserve.
constexpr uint32_t HC_GRAD_A2_NN_UB_BYTES = 192U * 1024U - 256U;
constexpr uint32_t HC_GRAD_C1A_UB_TILE_M = 192U;
constexpr uint32_t HC_GRAD_C1A_UB_TILE_N = 320U;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
// The sole Vector TBuf begins at hardware UB address zero. Keep the
// incoming C1a tile above the V3 pipelined down/gate staging and dgamma.
constexpr uint32_t HC_GRAD_C1A_UB_OFFSET = 96U * 1024U;
constexpr uint32_t HC_GRAD_C1A_UB_CORES = 27U;
constexpr uint32_t HC_GRAD_C1A_UB_LIMIT =
    HC_GRAD_C1A_UB_OFFSET + HC_GRAD_C1A_UB_TILE_M / 2U * HC_GRAD_C1A_UB_TILE_N * sizeof(float);
// These are the native Blaze FixpipeOpti mode-4 flags. The token-ring
// users of these IDs are disjoint from the retained chunk schedule.
constexpr uint16_t HC_GRAD_C1A_UB_FREE = 4U;
constexpr uint16_t HC_GRAD_C1A_UB_READY = 6U;
#endif

constexpr uint32_t HC_GRAD_NN_STAGE_C1A = 0U;
constexpr uint32_t HC_GRAD_NN_STAGE_C1B = 1U;
constexpr uint32_t HC_GRAD_NN_STAGE_G56 = 2U;
constexpr uint32_t HC_GRAD_NN_STAGE_DX = 3U;
constexpr uint32_t HC_GRAD_NN_STAGE_GATE = 4U;

// Model dimensions and A5 GEMM parameters shared by the host plan and fast-path checks.
constexpr uint32_t HC_GRAD_TUNED_CHANNELS = 4U;
constexpr uint32_t HC_GRAD_TUNED_HIDDEN_SIZE = 2560U;
constexpr uint32_t HC_GRAD_TUNED_PROJECTION_SIZE = 320U;
constexpr uint32_t HC_GRAD_TUNED_FEATURES = HC_GRAD_TUNED_CHANNELS * HC_GRAD_TUNED_HIDDEN_SIZE;
constexpr uint32_t HC_GRAD_AIV_PER_AIC = 2U;
constexpr uint32_t HC_GRAD_A5_TUNED_AIC_COUNT = 28U;
constexpr uint32_t HC_GRAD_A5_TUNED_AIV_COUNT = HC_GRAD_A5_TUNED_AIC_COUNT * HC_GRAD_AIV_PER_AIC;
constexpr uint64_t HC_GRAD_A5_BASIC_GEMM_KEY = 0x2U;
constexpr uint64_t HC_GRAD_A5_TRANSPOSE_A_KEY_BIT = 0x10U;
constexpr uint32_t HC_GRAD_A5_WEIGHT_TILE_M = 192U;
constexpr uint32_t HC_GRAD_A5_WEIGHT_TILE_N = 320U;
constexpr uint32_t HC_GRAD_A5_WEIGHT_TILE_K = 48U;
constexpr uint32_t HC_GRAD_A5_L1_K_TILES = 4U;
constexpr uint8_t HC_GRAD_A5_SINGLE_BUFFER = 1U;
constexpr uint8_t HC_GRAD_A5_DOUBLE_BUFFER = 2U;
constexpr uint32_t HC_GRAD_A5_UNSPLIT_TAIL = 1U;
constexpr uint32_t HC_GRAD_A5_G56_M_TAIL_SPLITS = 3U;

// Choose enough M/N output tiles for one wave of the cores actually launched.
// This is a fusion batch heuristic, not a universal optimum or an NN fixed task size.
HC_GRAD_HOST_DEVICE uint32_t ChooseHcGradTaskRows(uint32_t tokens, uint32_t cubeCores, uint32_t outputN, uint32_t tileM,
                                                  uint32_t tileN)
{
    if (cubeCores == 0U || outputN == 0U || tileM == 0U || tileN == 0U) {
        return 0U;
    }
    const uint32_t nTiles = (outputN + tileN - 1U) / tileN;
    const uint32_t mTiles = (cubeCores + nTiles - 1U) / nTiles;
    const uint32_t waveRows = mTiles * tileM;
    if (tokens != 0U && tokens < waveRows) {
        return (tokens + HC_GRAD_VECTOR_ALIGN - 1U) / HC_GRAD_VECTOR_ALIGN * HC_GRAD_VECTOR_ALIGN;
    }
    return waveRows;
}

enum class HcGradCubeOp : uint32_t {
    NONE = 0,
    C1 = 1,
    DX = 2,
    G56 = 3,
};

enum class HcGradVectorOp : uint32_t {
    NONE = 0,
    V1 = 1,
    V3 = 2,
    V6 = 3,
    TRANSPOSE = 4,
    REDUCE_WU = 5,
    REDUCE_WD = 6,
    DGAMMA = 7,
    REDUCE_DACT = 8,
    REDUCE_WU2 = 9,
    REDUCE_WD2 = 10,
    REDUCE_DGAMMA = 11,
    TRANSPOSE_DUP = 12,
};

HC_GRAD_HOST_DEVICE constexpr uint32_t HcGradAlign16(uint32_t v)
{
    return (v + HC_GRAD_GATE_ALIGN - 1U) / HC_GRAD_GATE_ALIGN * HC_GRAD_GATE_ALIGN;
}

HC_GRAD_HOST_DEVICE uint64_t HcGradAlign(uint64_t bytes)
{
    return (bytes + HC_GRAD_WS_ALIGN - 1U) / HC_GRAD_WS_ALIGN * HC_GRAD_WS_ALIGN;
}

struct HcGradWorkspaceLayout {
    uint64_t xn{0};
    uint64_t dup{0};
    uint64_t direct{0};
    uint64_t dx2{0};
    uint64_t wConcat{0};
    uint64_t act{0};
    uint64_t dact{0};
    uint64_t packed{0};
    uint64_t packedT{0};
    uint64_t denseDx{0}; // Dense row-major DX operand when its K differs from the padded G56 stride.
    // Full-K mode uses one merged G56 result [R+gateRows,D]; C1b writes dwUpAccum directly.
    // Legacy layout queries with weightRows=0 reserve one shared partial buffer per ring slot.
    uint64_t dwPartial{0};
    uint64_t dwUpPartial{0};    // Independent full-batch C1b result, allowing dW to overlap V6.
    uint64_t dwSlotElems{0};    // Element count per task slot.
    uint64_t dwGateRowElems{0}; // Element offset of dwGate within a slot: r*d.
    uint64_t dactPartial{0};
    uint64_t dgammaPartial{0};
    uint64_t dupT{0};
    // Full FP32 weight-gradient accumulators; convert to the weight dtype only after the final task.
    uint64_t dwDownAccum{0};
    uint64_t dwUpAccum{0};
    uint64_t dwGateAccum{0};
    uint32_t gateRows{0};
    uint32_t g56Rows{0};
    // Nonzero: retain dW operands for every token until the final full-K GEMMs.
    uint32_t weightRows{0};
    uint32_t taskRows{HC_GRAD_TASK_ROWS};
    bool fullBatch{false};
    bool recomputeDirect{false};
    bool duplicateV1Operands{false};
    uint8_t reserved{0}; // Initialize the alignment byte when this layout is copied.
    uint32_t cubeTailTask{0xffffffffU};
    uint64_t cubeTailDact{0};
    uint64_t cubeTailDx{0};
    uint64_t totalBytes{0};

    HC_GRAD_HOST_DEVICE uint32_t TaskBufferIndex(uint64_t offset, uint32_t task, uint32_t slots) const
    {
        const bool retained =
            (weightRows != 0U && (offset == xn || offset == dup || offset == act || offset == packed)) ||
            (fullBatch &&
             (offset == direct || offset == dx2 || offset == dact || offset == dactPartial || offset == denseDx));
        return retained ? task : task % slots;
    }

    HC_GRAD_HOST_DEVICE uint64_t TaskBufferOffset(uint64_t offset, uint32_t task, uint32_t slots, uint64_t slotBytes,
                                                  uint32_t chunk = 0U) const
    {
        if (task == cubeTailTask) {
            if (offset == dactPartial) {
                return cubeTailDact;
            }
            if (offset == dx2) {
                return cubeTailDx;
            }
        }
        const bool duplicated =
            duplicateV1Operands && (offset == xn || offset == dup || offset == act || offset == packed);
        const uint64_t chunkOffset =
            duplicated && taskRows != 0U ? static_cast<uint64_t>(chunk) * weightRows * slotBytes / taskRows : 0U;
        return offset + chunkOffset + static_cast<uint64_t>(TaskBufferIndex(offset, task, slots)) * slotBytes;
    }
};

HC_GRAD_HOST_DEVICE HcGradWorkspaceLayout BuildHcGradWorkspaceLayout(
    uint32_t workspaceSlots, uint32_t splitkFactor, uint32_t aivNum, uint32_t c, uint32_t h, uint32_t r, uint32_t d,
    uint32_t elementSize, bool hasGamma, bool materializeDupT = false, uint32_t taskRows = HC_GRAD_TASK_ROWS,
    uint32_t weightRows = 0U, bool fullBatch = false, uint32_t dxColumns = 0U, bool recomputeDirect = false,
    bool duplicateV1Operands = false, bool separateDx = true, bool directDwAccum = false)
{
    HcGradWorkspaceLayout layout;
    const uint64_t tokens = static_cast<uint64_t>(workspaceSlots) * taskRows;
    layout.weightRows = weightRows;
    layout.taskRows = taskRows;
    layout.fullBatch = fullBatch;
    layout.recomputeDirect = fullBatch && recomputeDirect;
    layout.duplicateV1Operands = fullBatch && duplicateV1Operands;
    const uint64_t weightTokens = weightRows != 0U ? weightRows : tokens;
    const uint64_t operandTokens = layout.duplicateV1Operands ? 2U * weightTokens : weightTokens;
    layout.gateRows = HcGradAlign16(c);
    layout.g56Rows = HcGradAlign16(r + layout.gateRows);
    const uint32_t g56Rows = layout.g56Rows;
    uint64_t offset = 0;
    layout.xn = offset;
    offset += HcGradAlign(operandTokens * d * elementSize);
    layout.dup = offset;
    offset += HcGradAlign(operandTokens * d * elementSize);
    layout.direct = offset;
    if (!layout.recomputeDirect) {
        offset += HcGradAlign((fullBatch ? weightTokens : tokens) * d * sizeof(float)); // FP32 V1 output for V6/dgamma.
    }
    if (fullBatch && !separateDx) {
        // Low-precision DX is written to the public output, leaving xn live for G56.
        layout.dx2 = layout.xn;
    } else {
        // G56 still reads xn after DX. Keep the FP32 Cube output live independently until V6.
        layout.dx2 = offset;
        offset += HcGradAlign((fullBatch ? weightTokens : tokens) * d * sizeof(float));
    }
    layout.wConcat = offset;
    offset += HcGradAlign(static_cast<uint64_t>(r + c) * d * elementSize); // Concatenate w_down and w_gate along K.
    layout.act = offset;
    offset += HcGradAlign(operandTokens * r * elementSize);
    layout.dact = offset;
    offset += HcGradAlign((fullBatch ? weightTokens : tokens) * r * sizeof(float));
    layout.packed = offset;
    offset += HcGradAlign(operandTokens * g56Rows * elementSize);
    layout.packedT = offset;
    if (fullBatch) {
        if (dxColumns != g56Rows) {
            layout.denseDx = offset;
            offset += HcGradAlign(weightTokens * dxColumns * elementSize);
        }
    } else {
        offset += HcGradAlign(tokens * g56Rows * elementSize);
    }
    // G56 combines dwDown and dwGate in one M=R+gateRows GEMM.
    // The retained-input path only needs one complete result buffer.
    layout.dwSlotElems = (static_cast<uint64_t>(r) + layout.gateRows) * d;
    layout.dwGateRowElems = static_cast<uint64_t>(r) * d;
    layout.dwPartial = offset;
    // Deferred G56 writes one complete [R+gateRows,D] result, not task partials.
    const uint32_t weightSlots = weightRows != 0U ? 1U : workspaceSlots;
    offset += HcGradAlign(weightSlots * static_cast<uint64_t>(layout.dwSlotElems) * sizeof(float));
    layout.dwUpPartial = offset;
    if (fullBatch) {
        offset += HcGradAlign(static_cast<uint64_t>(r) * d * sizeof(float));
    }
    layout.dactPartial = offset;
    offset += HcGradAlign((fullBatch ? weightTokens : tokens) * splitkFactor * r * sizeof(float));
    layout.dgammaPartial = offset;
    if (hasGamma) {
        // DGAMMA writes per-core partials; REDUCE_DGAMMA merges them in the next step.
        // Each core stores a Kahan accumulator and compensation; the second region is unused when compensation is
        // disabled.
        offset += HcGradAlign(static_cast<uint64_t>(HC_GRAD_DGAMMA_SLOTS) * aivNum * h * sizeof(float));
    }
    // Only the transpose fallback materializes dup^T as RowMajor [D,rows] with fixed stride
    // HC_GRAD_TASK_ROWS. A5 and A2/A3 normally read dup directly and allocate no dupT storage.
    layout.dupT = offset;
    if (materializeDupT) {
        offset += HcGradAlign(tokens * d * elementSize);
    }
    const uint64_t projectionBytes = workspaceSlots != 0U ? static_cast<uint64_t>(r) * d * sizeof(float) : 0U;
    const uint64_t gateBytes = workspaceSlots != 0U ? static_cast<uint64_t>(c) * d * sizeof(float) : 0U;
    if (fullBatch && directDwAccum) {
        // The first chunk overwrites the final FP32 result. Later Cube writes add to it in chunk order.
        // The G56 buffer already owns enough aligned rows for the padded gate.
        layout.dwDownAccum = layout.dwPartial;
        layout.dwUpAccum = layout.dwUpPartial;
        layout.dwGateAccum = layout.dwPartial + layout.dwGateRowElems * sizeof(float);
    } else {
        layout.dwDownAccum = offset;
        offset += HcGradAlign(projectionBytes);
        layout.dwUpAccum = offset;
        offset += HcGradAlign(projectionBytes);
        layout.dwGateAccum = offset;
        offset += HcGradAlign(gateBytes);
    }
    // The final short task can use a different NN M/N-to-core assignment.
    // Keep its Cube outputs separate from slots previously owned by full tasks.
    // All complete tasks still share the bounded ring; retained dW inputs are unchanged.
    if (!fullBatch && taskRows != 0U && weightRows % taskRows != 0U && weightRows / taskRows >= workspaceSlots &&
        workspaceSlots != 0U) {
        layout.cubeTailTask = weightRows / taskRows;
        const uint64_t tailRows = HcGradAlign16(weightRows % taskRows);
        layout.cubeTailDact = offset;
        offset += HcGradAlign(static_cast<uint64_t>(splitkFactor) * tailRows * r * sizeof(float));
        layout.cubeTailDx = offset;
        offset += HcGradAlign(tailRows * d * sizeof(float));
    }
    layout.totalBytes = offset;
    return layout;
}

struct HcGradBeat {
    HcGradCubeOp cubeOp{HcGradCubeOp::NONE};
    uint32_t cubeTask{0};
    // Optional second short Cube operation: only DX+G56 may pair to fill a long V1/V6 step.
    HcGradCubeOp cubeOp2{HcGradCubeOp::NONE};
    uint32_t cubeTask2{0};
    HcGradVectorOp vectorOp[HC_GRAD_VECTOR_OPS_PER_BEAT]{HcGradVectorOp::NONE, HcGradVectorOp::NONE,
                                                         HcGradVectorOp::NONE};
    uint32_t vectorTask[HC_GRAD_VECTOR_OPS_PER_BEAT]{HC_GRAD_NO_TASK, HC_GRAD_NO_TASK, HC_GRAD_NO_TASK};
};

template <bool USE_DUP_T = false, bool G56_PACKED_T = true, bool DEFER_WEIGHT_GRADS = false>
class HcGradScheduler {
    static_assert(!DEFER_WEIGHT_GRADS || (!USE_DUP_T && !G56_PACKED_T),
                  "The full-K schedule requires retained operands without a per-task transpose");

public:
    HC_GRAD_HOST_DEVICE HcGradScheduler(uint32_t taskCount, uint32_t workspaceSlots, bool hasGamma)
        : taskCount_(taskCount),
          workspaceSlots_(workspaceSlots),
          hasGamma_(hasGamma)
    {}

    HC_GRAD_HOST_DEVICE bool Done() const
    {
        return c1_ == taskCount_ && dx_ == taskCount_ && v1_ == taskCount_ && v3_ == taskCount_ &&
               (!G56_PACKED_T || vt_ == taskCount_) && v6_ == taskCount_ &&
               (DEFER_WEIGHT_GRADS || (g56_ == taskCount_ && v2_ == taskCount_ && v4_ == taskCount_)) &&
               rd_ == taskCount_ && (!USE_DUP_T || td_ == taskCount_) && (!hasGamma_ || dgammaReduced_);
    }

    HC_GRAD_HOST_DEVICE uint32_t MaxBeats() const
    {
        // The USE_DUP_T fallback adds V1 -> TRANSPOSE_DUP -> C1 dependencies.
        // An overestimate adds empty SyncAll steps; an underestimate could omit tasks.
        return (USE_DUP_T ? HC_GRAD_TRANSPOSE_BEATS_PER_TASK : HC_GRAD_DIRECT_BEATS_PER_TASK) * taskCount_ +
               HC_GRAD_DRAIN_BEATS;
    }

    HC_GRAD_HOST_DEVICE HcGradBeat Next()
    {
        HcGradBeat beat;
        const ReadyState ready{c1_, dx_, g56_, v1_, v3_, vt_, v6_, v2_, v4_, rd_, td_, dgammaDone_};
        SelectCube(ready, beat);
        uint32_t slot = 0;
        SelectVectorDependencies(ready, beat, slot);
        if constexpr (!DEFER_WEIGHT_GRADS) {
            SelectWeightReductions(ready, beat, slot);
        }
        SelectInputAndGamma(ready, beat, slot);
        return beat;
    }

private:
    struct ReadyState {
        uint32_t c1, dx, g56, v1, v3, vt, v6, v2, v4, rd, td;
        bool dgammaDone;
    };

    HC_GRAD_HOST_DEVICE bool HasReadyVector(const ReadyState& ready) const
    {
        return (ready.rd < ready.c1 && ready.rd < ready.v3 + workspaceSlots_) || (ready.v3 < ready.rd) ||
               (ready.v6 < ready.dx && ready.v6 < ready.v1) || (ready.v2 < ready.c1) || (ready.v4 < ready.g56) ||
               (ready.v1 < taskCount_ && ready.v1 < ready.v6 + workspaceSlots_ &&
                ready.v1 < ready.g56 + workspaceSlots_);
    }

    HC_GRAD_HOST_DEVICE void SelectCube(const ReadyState& ready, HcGradBeat& beat)
    {
        if constexpr (DEFER_WEIGHT_GRADS) {
            // Only dactPartial and dx2 require cross-core slot credits. The dW operands
            // are indexed by task and remain live until the full-K GEMMs run.
            if (ready.c1 < ready.v1 && ready.c1 - ready.rd < workspaceSlots_) {
                beat.cubeOp = HcGradCubeOp::C1;
                beat.cubeTask = c1_++;
            } else if (ready.dx < ready.v3 && ready.dx - ready.v6 < workspaceSlots_) {
                beat.cubeOp = HcGradCubeOp::DX;
                beat.cubeTask = dx_++;
            }
            return;
        }
        // Use vector work ready at step entry. Pair the short DX/G56 operations only when there is
        // long vector work to overlap. Cube readiness uses entry counters and does not consume newly produced inputs.
        const bool vectorReady = HasReadyVector(ready);
        const bool g56Ready = ready.g56 < (G56_PACKED_T ? ready.vt : ready.v3) && ready.g56 < ready.v2 &&
                              ready.g56 < ready.v4 + workspaceSlots_;

        const uint32_t c1Ready = USE_DUP_T ? ready.td : ready.v1;
        if (ready.c1 < c1Ready && ready.c1 < ready.v2 + workspaceSlots_ && ready.c1 < ready.rd + workspaceSlots_) {
            beat.cubeOp = HcGradCubeOp::C1;
            beat.cubeTask = c1_++;
        } else if (ready.dx < ready.v3 && ready.dx < ready.c1) {
            beat.cubeOp = HcGradCubeOp::DX;
            beat.cubeTask = dx_++;
        } else if (ready.g56 < (G56_PACKED_T ? ready.vt : ready.v3) && ready.g56 < ready.v2 &&
                   ready.g56 < ready.v4 + workspaceSlots_) {
            beat.cubeOp = HcGradCubeOp::G56;
            beat.cubeTask = g56_++;
        }
        // After short DX, pair G56 only if it was ready at step entry and the step contains vector work.
        // C1 is already long and does not pair with a second Cube operation.
        if (beat.cubeOp == HcGradCubeOp::DX && g56Ready && vectorReady) {
            beat.cubeOp2 = HcGradCubeOp::G56;
            beat.cubeTask2 = g56_++;
        }
    }

    HC_GRAD_HOST_DEVICE void SelectVectorDependencies(const ReadyState& ready, HcGradBeat& beat, uint32_t& slot)
    {
        if (ready.rd < ready.c1 && ready.rd < ready.v3 + workspaceSlots_) {
            beat.vectorOp[slot] = HcGradVectorOp::REDUCE_DACT;
            beat.vectorTask[slot++] = rd_++;
        }
        if (slot < HC_GRAD_VECTOR_OPS_PER_BEAT) {
            if (USE_DUP_T && ready.td < ready.v1 && ready.td < taskCount_) {
                beat.vectorOp[slot] = HcGradVectorOp::TRANSPOSE_DUP;
                beat.vectorTask[slot++] = td_++;
            } else if (ready.v3 < rd_) {
                // V3 may follow REDUCE_DACT in this beat; vector slots execute in order.
                beat.vectorOp[slot] = HcGradVectorOp::V3;
                beat.vectorTask[slot++] = v3_++;
            } else if (G56_PACKED_T && ready.vt < ready.v3) {
                beat.vectorOp[slot] = HcGradVectorOp::TRANSPOSE;
                beat.vectorTask[slot++] = vt_++;
            } else if (ready.v6 < ready.dx && ready.v6 < ready.v1) {
                beat.vectorOp[slot] = HcGradVectorOp::V6;
                beat.vectorTask[slot++] = v6_++;
            }
        }
    }

    HC_GRAD_HOST_DEVICE void SelectWeightReductions(const ReadyState& ready, HcGradBeat& beat, uint32_t& slot)
    {
        if (slot < HC_GRAD_VECTOR_OPS_PER_BEAT) {
            if (ready.v2 + 1U < ready.c1 && ready.v2 + 1U < taskCount_) {
                beat.vectorOp[slot] = HcGradVectorOp::REDUCE_WU2;
                beat.vectorTask[slot++] = ready.v2;
                v2_ += HC_GRAD_PAIRED_REDUCTION_TASKS;
            } else if (ready.v2 < ready.c1) {
                beat.vectorOp[slot] = HcGradVectorOp::REDUCE_WU;
                beat.vectorTask[slot++] = v2_++;
            }
        }
        if (slot < HC_GRAD_VECTOR_OPS_PER_BEAT) {
            if (ready.v4 + 1U < ready.g56 && ready.v4 + 1U < taskCount_) {
                beat.vectorOp[slot] = HcGradVectorOp::REDUCE_WD2;
                beat.vectorTask[slot++] = ready.v4;
                v4_ += HC_GRAD_PAIRED_REDUCTION_TASKS;
            } else if (ready.v4 < ready.g56) {
                beat.vectorOp[slot] = HcGradVectorOp::REDUCE_WD;
                beat.vectorTask[slot++] = v4_++;
            }
        }
    }

    HC_GRAD_HOST_DEVICE void SelectInputAndGamma(const ReadyState& ready, HcGradBeat& beat, uint32_t& slot)
    {
        if (slot < HC_GRAD_VECTOR_OPS_PER_BEAT && ready.v1 < taskCount_ && ready.v1 < ready.v6 + workspaceSlots_ &&
            (DEFER_WEIGHT_GRADS || ready.v1 < ready.g56 + workspaceSlots_)) {
            beat.vectorOp[slot] = HcGradVectorOp::V1;
            beat.vectorTask[slot++] = v1_++;
        }
        if (slot < HC_GRAD_VECTOR_OPS_PER_BEAT && hasGamma_ && ready.v6 == taskCount_ && !ready.dgammaDone) {
            beat.vectorOp[slot] = HcGradVectorOp::DGAMMA;
            beat.vectorTask[slot] = HC_GRAD_NO_TASK;
            dgammaDone_ = true;
        }
        // DGAMMA must finish writing all core partials in the preceding step before cross-core reduction.
        if (slot < HC_GRAD_VECTOR_OPS_PER_BEAT && hasGamma_ && ready.dgammaDone && !dgammaReduced_) {
            beat.vectorOp[slot] = HcGradVectorOp::REDUCE_DGAMMA;
            beat.vectorTask[slot] = HC_GRAD_NO_TASK;
            dgammaReduced_ = true;
        }
    }

    uint32_t taskCount_{0};
    uint32_t workspaceSlots_{0};
    bool hasGamma_{false};
    uint32_t v1_{0};
    uint32_t v3_{0};
    uint32_t vt_{0};
    uint32_t v6_{0};
    uint32_t c1_{0};
    uint32_t dx_{0};
    uint32_t g56_{0};
    uint32_t v2_{0};
    uint32_t v4_{0};
    uint32_t rd_{0};
    uint32_t td_{0};
    bool dgammaDone_{false};
    bool dgammaReduced_{false};
};

template <bool USE_DUP_T = false, bool G56_PACKED_T = true, bool DEFER_WEIGHT_GRADS = false>
HC_GRAD_HOST_DEVICE uint32_t CountHcGradBeats(uint32_t taskCount, uint32_t workspaceSlots, bool hasGamma)
{
    HcGradScheduler<USE_DUP_T, G56_PACKED_T, DEFER_WEIGHT_GRADS> scheduler(taskCount, workspaceSlots, hasGamma);
    uint32_t beats = 0;
    while (!scheduler.Done() && beats < scheduler.MaxBeats()) {
        (void)scheduler.Next();
        ++beats;
    }
    return scheduler.Done() ? beats : 0U;
}

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_PIPELINE_ARCH35_H
