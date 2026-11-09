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
    // dwUp (C1b) and dwDown/dwGate (G56) have disjoint lifetimes and share one buffer per slot.
    // Size each slot for the larger G56 result [R+gateRows,D], or (r+gateRows)*d elements.
    uint64_t dwPartial{0};
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
    uint64_t totalBytes{0};
};

HC_GRAD_HOST_DEVICE HcGradWorkspaceLayout BuildHcGradWorkspaceLayout(uint32_t workspaceSlots, uint32_t splitkFactor,
                                                                     uint32_t aivNum, uint32_t c, uint32_t h,
                                                                     uint32_t r, uint32_t d, uint32_t elementSize,
                                                                     bool hasGamma, bool materializeDupT = false)
{
    HcGradWorkspaceLayout layout;
    const uint64_t tokens = static_cast<uint64_t>(workspaceSlots) * HC_GRAD_TASK_ROWS;
    layout.gateRows = HcGradAlign16(c);
    layout.g56Rows = HcGradAlign16(r + layout.gateRows);
    const uint32_t g56Rows = layout.g56Rows;
    uint64_t offset = 0;
    layout.xn = offset;
    offset += HcGradAlign(tokens * d * elementSize);
    layout.dup = offset;
    offset += HcGradAlign(tokens * d * elementSize);
    layout.direct = offset;
    offset += HcGradAlign(tokens * d * sizeof(float)); // V1 direct remains FP32 for V6/dgamma.
    layout.dx2 = offset;
    offset += HcGradAlign(tokens * d * sizeof(float)); // FP32 result of ddown @ w_down + dlogits @ w_gate.
    layout.wConcat = offset;
    offset += HcGradAlign(static_cast<uint64_t>(r + c) * d * elementSize); // Concatenate w_down and w_gate along K.
    layout.act = offset;
    offset += HcGradAlign(tokens * r * elementSize);
    layout.dact = offset;
    offset += HcGradAlign(tokens * r * sizeof(float)); // Keep Cube output in FP32 for its Vector consumer.
    layout.packed = offset;
    offset += HcGradAlign(tokens * g56Rows * elementSize);
    layout.packedT = offset;
    offset += HcGradAlign(tokens * g56Rows * elementSize);
    // dwUp/dwDown/dwGate partials share storage: C1b writes [D,R], G56 writes [R+gateRows,D],
    // with disjoint lifetimes. Each slot holds (r+gateRows)*d elements. G56 combines dwDown and dwGate
    // in one M=R+gateRows GEMM, avoiding a full xn reread for the small gate GEMM.
    layout.dwSlotElems = (static_cast<uint64_t>(r) + layout.gateRows) * d;
    layout.dwGateRowElems = static_cast<uint64_t>(r) * d;
    layout.dwPartial = offset;
    offset += HcGradAlign(workspaceSlots * static_cast<uint64_t>(layout.dwSlotElems) * sizeof(float));
    layout.dactPartial = offset;
    offset += HcGradAlign(workspaceSlots * static_cast<uint64_t>(splitkFactor) * HC_GRAD_TASK_ROWS * r * sizeof(float));
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
    layout.dwDownAccum = offset;
    offset += HcGradAlign(projectionBytes);
    layout.dwUpAccum = offset;
    offset += HcGradAlign(projectionBytes);
    layout.dwGateAccum = offset;
    offset += HcGradAlign(gateBytes);
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

template <bool USE_DUP_T = false, bool G56_PACKED_T = true>
class HcGradScheduler {
public:
    HC_GRAD_HOST_DEVICE HcGradScheduler(uint32_t taskCount, uint32_t workspaceSlots, bool hasGamma)
        : taskCount_(taskCount),
          workspaceSlots_(workspaceSlots),
          hasGamma_(hasGamma)
    {}

    HC_GRAD_HOST_DEVICE bool Done() const
    {
        return c1_ == taskCount_ && dx_ == taskCount_ && g56_ == taskCount_ && v1_ == taskCount_ && v3_ == taskCount_ &&
               (!G56_PACKED_T || vt_ == taskCount_) && v6_ == taskCount_ && v2_ == taskCount_ && v4_ == taskCount_ &&
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
        SelectWeightReductions(ready, beat, slot);
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
            ready.v1 < ready.g56 + workspaceSlots_) {
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

template <bool USE_DUP_T = false, bool G56_PACKED_T = true>
HC_GRAD_HOST_DEVICE uint32_t CountHcGradBeats(uint32_t taskCount, uint32_t workspaceSlots, bool hasGamma)
{
    HcGradScheduler<USE_DUP_T, G56_PACKED_T> scheduler(taskCount, workspaceSlots, hasGamma);
    uint32_t beats = 0;
    while (!scheduler.Done() && beats < scheduler.MaxBeats()) {
        (void)scheduler.Next();
        ++beats;
    }
    return scheduler.Done() ? beats : 0U;
}

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_PIPELINE_ARCH35_H
