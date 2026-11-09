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
 * \file text_gated_residual_grad_kernel_common.h
 * \brief Shared TextGatedResidualGrad initialization and Cube/Vector execution.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_KERNEL_COMMON_H
#define TEXT_GATED_RESIDUAL_GRAD_KERNEL_COMMON_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "text_gated_residual_grad_tiling_data_arch35.h"
#include "text_gated_residual_grad_pipeline_arch35.h"
#include "text_gated_residual_grad_vector_arch35.h"
#include "text_gated_residual_grad_credit_sync.h"

namespace TextGatedResidualGrad {

// Ascend 950 and Atlas A2/A3 share Init/Process/RunCube. Each entry header supplies its Cube backend
// and tiles. USE_DUP_T/G56_PACKED_T remain available for transpose fallback paths.
template <typename D_T, int HAS_GAMMA, class GemmT, bool USE_DUP_T, bool G56_PACKED_T = true>
class KernelTextGatedResidualGrad {
public:
    __aicore__ inline void Init(GM_ADDR gradMixed, GM_ADDR x, GM_ADDR wDown, GM_ADDR wUp, GM_ADDR invRms,
                                GM_ADDR downPreAct, GM_ADDR up, GM_ADDR gradInject, GM_ADDR wGate, GM_ADDR gamma,
                                GM_ADDR inject, GM_ADDR dx, GM_ADDR dwDown, GM_ADDR dwUp, GM_ADDR dwGate,
                                GM_ADDR dgamma, GM_ADDR workspace,
                                const TextGatedResidualGradTilingDataArch35* tilingData, AscendC::TPipe* pipe)
    {
        n_ = static_cast<uint32_t>(tilingData->n);
        c_ = static_cast<uint32_t>(tilingData->c);
        r_ = static_cast<uint32_t>(tilingData->r);
        d_ = static_cast<uint32_t>(tilingData->d);
        taskCount_ = static_cast<uint32_t>(tilingData->taskCount);
        workspaceSlots_ = static_cast<uint32_t>(tilingData->workspaceSlots);
        splitkFactor_ = static_cast<uint32_t>(tilingData->splitkFactor);
        hasGate_ = static_cast<uint32_t>(tilingData->hasGate);
        scheduleBeatCount_ = static_cast<uint32_t>(tilingData->scheduleBeatCount);
        aicNum_ = static_cast<uint32_t>(tilingData->aicNum);
        workspace_ = workspace;
        producerConsumer_ = (tilingData->scheduleMode != 0) && !USE_DUP_T && !G56_PACKED_T;
        gmWUp_ = wUp;
        gmWDown_ = wDown;
        gmWGate_ = wGate;
        singleTask_ = taskCount_ == 1U;
        layout_ = BuildHcGradWorkspaceLayout(workspaceSlots_, splitkFactor_, static_cast<uint32_t>(tilingData->aivNum),
                                             c_, static_cast<uint32_t>(tilingData->h), r_, d_, sizeof(D_T),
                                             HAS_GAMMA != 0, USE_DUP_T);
        gmDwDown_ = Pointer(layout_.dwDownAccum);
        gmDwUp_ = Pointer(layout_.dwUpAccum);
        gmDwGate_ = Pointer(layout_.dwGateAccum);
        g56Rows_ = layout_.g56Rows;
        gateRows_ = layout_.gateRows;
        credit_.Init(workspace_, 0U, taskCount_, workspaceSlots_, aicNum_, static_cast<uint32_t>(tilingData->aivNum));
        if ASCEND_IS_AIC {
            gemm_.Init(aicNum_);
        }
        if ASCEND_IS_AIV {
            vector_.Init(gradMixed, x, wDown, wUp, invRms, downPreAct, up, gradInject, wGate, gamma, inject, dx, dwDown,
                         dwUp, dwGate, dgamma, workspace, layout_, tilingData, pipe);
        }
    }

    __aicore__ inline void Process()
    {
        // Empty token input still requires zeroing nonempty dw/dgamma outputs, but must bypass credit/ring scheduling.
        // With workspaceSlots=0, TaskPointer modulo operations or a DGAMMA step can hang the kernel.
        if (taskCount_ == 0U) {
            if ASCEND_IS_AIV {
                vector_.Prepare();
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            }
            AscendC::SyncAll<false>();
            return;
        }
        if ASCEND_IS_AIV {
            vector_.Prepare();
            vector_.PrepareWeights();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
        AscendC::SyncAll<false>();
        HcGradScheduler<USE_DUP_T, G56_PACKED_T> scheduler(taskCount_, workspaceSlots_, HAS_GAMMA != 0);
        for (uint32_t beatIndex = 0; beatIndex < scheduleBeatCount_ && !scheduler.Done(); ++beatIndex) {
            const HcGradBeat beat = scheduler.Next();
            if (!producerConsumer_) {
                ProcessSynchronizedBeat(beat);
                continue;
            }
            // Both sides share the beat sequence; credits provide cross-side readiness and slot reuse.
            ProcessCubeCredits(beat);
            ProcessVectorCredits(beat);
            SynchronizeDgamma(beat);
        }
        if (producerConsumer_) {
            AscendC::SyncAll<false>();
        }
        // All FP32 GEMM/reduction stores are complete before converting the public outputs.
        if ASCEND_IS_AIV {
            vector_.StoreWeightGradients();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
        AscendC::SyncAll<false>();
    }

private:
    __aicore__ inline void ProcessSynchronizedBeat(const HcGradBeat& beat)
    {
        if ASCEND_IS_AIC {
            RunCube(beat.cubeOp, beat.cubeTask);
            RunCube(beat.cubeOp2, beat.cubeTask2);
        }
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                if (beat.vectorOp[i] != HcGradVectorOp::NONE) {
                    vector_.Process(beat.vectorOp[i], beat.vectorTask[i]);
                }
            }
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
        AscendC::SyncAll<false>();
    }

    __aicore__ inline void ProcessCubeCredits(const HcGradBeat& beat)
    {
        if ASCEND_IS_AIC {
            const bool hasCube = (beat.cubeOp != HcGradCubeOp::NONE) || (beat.cubeOp2 != HcGradCubeOp::NONE);
            credit_.WaitCubeDeps(beat.cubeOp, beat.cubeTask);
            RunCube(beat.cubeOp, beat.cubeTask);
            credit_.WaitCubeDeps(beat.cubeOp2, beat.cubeTask2);
            RunCube(beat.cubeOp2, beat.cubeTask2);
            if (hasCube) {
                credit_.SameKindBarrier();
                credit_.MarkCube(beat.cubeOp, beat.cubeTask);
                credit_.MarkCube(beat.cubeOp2, beat.cubeTask2);
            }
        }
    }

    __aicore__ inline void ProcessVectorCredits(const HcGradBeat& beat)
    {
        if ASCEND_IS_AIV {
            bool hasVector = false;
            for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                if (beat.vectorOp[i] != HcGradVectorOp::NONE) {
                    credit_.WaitVectorDeps(beat.vectorOp[i], beat.vectorTask[i]);
                    vector_.Process(beat.vectorOp[i], beat.vectorTask[i]);
                    hasVector = true;
                }
            }
            if (hasVector) {
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                credit_.SameKindBarrier();
                for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                    credit_.MarkVector(beat.vectorOp[i], beat.vectorTask[i]);
                }
            }
        }
    }

    __aicore__ inline void SynchronizeDgamma(const HcGradBeat& beat)
    {
        // DGAMMA writes per-core partials; REDUCE_DGAMMA reads all cores in the next step.
        // Keep an explicit global barrier on this reduction edge; a same-core-type barrier alone is insufficient.
        for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
            if (beat.vectorOp[i] == HcGradVectorOp::DGAMMA) {
                AscendC::SyncAll<false>();
                break;
            }
        }
    }

    __aicore__ inline GM_ADDR TaskPointer(uint64_t offset, uint32_t task, uint32_t rowStride) const
    {
        const uint32_t slot = task % workspaceSlots_;
        return workspace_ + offset + static_cast<uint64_t>(slot) * HC_GRAD_TASK_ROWS * rowStride * sizeof(D_T);
    }

    __aicore__ inline GM_ADDR TaskPointerF32(uint64_t offset, uint32_t task, uint32_t rowStride) const
    {
        const uint32_t slot = task % workspaceSlots_;
        return workspace_ + offset + static_cast<uint64_t>(slot) * HC_GRAD_TASK_ROWS * rowStride * sizeof(float);
    }

    __aicore__ inline GM_ADDR SlotPointerF32(uint64_t offset, uint32_t task, uint64_t elements) const
    {
        const uint32_t slot = task % workspaceSlots_;
        return workspace_ + offset + static_cast<uint64_t>(slot) * elements * sizeof(float);
    }

    __aicore__ inline GM_ADDR Pointer(uint64_t offset) const
    {
        return workspace_ + offset;
    }

    __aicore__ inline uint32_t Rows(uint32_t task) const
    {
        const uint32_t rowBegin = task * HC_GRAD_TASK_ROWS;
        return n_ - rowBegin < HC_GRAD_TASK_ROWS ? n_ - rowBegin : HC_GRAD_TASK_ROWS;
    }

    __aicore__ inline void RunC1(uint32_t task, uint32_t rows)
    {
        GM_ADDR dactPartial =
            SlotPointerF32(layout_.dactPartial, task, static_cast<uint64_t>(splitkFactor_) * HC_GRAD_TASK_ROWS * r_);
        gemm_.RunC1aSplitK(TaskPointer(layout_.dup, task, d_), gmWUp_, dactPartial, rows, r_, d_, splitkFactor_);
        // Read dup directly as ColumnMajor by default; the fallback reads dupT materialized by Vector.
        const uint64_t c1bA = USE_DUP_T ? layout_.dupT : layout_.dup;
        if (singleTask_) {
            gemm_.RunC1bDirect(TaskPointer(c1bA, task, d_), TaskPointer(layout_.act, task, r_), gmDwUp_, rows, r_, d_);
        } else {
            gemm_.RunC1b(TaskPointer(c1bA, task, d_), TaskPointer(layout_.act, task, r_),
                         SlotPointerF32(layout_.dwPartial, task, layout_.dwSlotElems), rows, r_, d_);
        }
    }

    __aicore__ inline void RunCube(HcGradCubeOp op, uint32_t task)
    {
        const uint32_t rows = Rows(task);
        switch (op) {
            case HcGradCubeOp::C1: {
                RunC1(task, rows);
                break;
            }
            case HcGradCubeOp::G56: {
                GM_ADDR packedT = TaskPointer(layout_.packedT, task, g56Rows_);
                GM_ADDR packed = TaskPointer(layout_.packed, task, g56Rows_);
                GM_ADDR xn = TaskPointer(layout_.xn, task, d_);
                GM_ADDR dwPartial = SlotPointerF32(layout_.dwPartial, task, layout_.dwSlotElems);
                if constexpr (G56_PACKED_T) {
                    if (singleTask_) {
                        gemm_.RunG56Direct(packedT, xn, gmDwDown_, dwPartial + layout_.dwGateRowElems * sizeof(float),
                                           rows, r_, d_, gateRows_, hasGate_ != 0);
                    } else {
                        gemm_.RunG56Merged(packedT, xn, dwPartial, rows, r_, d_, gateRows_, hasGate_ != 0);
                    }
                } else {
                    // Reading packed as ColumnMajor computes packedT @ xn without TRANSPOSE or packedT GM transfers.
                    // The packed row stride remains g56Rows.
                    if (singleTask_) {
                        gemm_.RunG56DirectPacked(packed, xn, gmDwDown_,
                                                 dwPartial + layout_.dwGateRowElems * sizeof(float), rows, r_, d_,
                                                 g56Rows_, gateRows_, hasGate_ != 0);
                    } else {
                        gemm_.RunG56MergedPacked(packed, xn, dwPartial, rows, r_, d_, g56Rows_, gateRows_,
                                                 hasGate_ != 0);
                    }
                }
                break;
            }
            case HcGradCubeOp::DX:
                gemm_.RunDx(TaskPointer(layout_.packed, task, g56Rows_), gmWDown_, gmWGate_, Pointer(layout_.wConcat),
                            TaskPointerF32(layout_.dx2, task, d_), rows, d_, r_, c_, g56Rows_, hasGate_ != 0);
                break;
            default:
                break;
        }
    }

    uint32_t n_{0};
    uint32_t c_{0};
    uint32_t r_{0};
    uint32_t d_{0};
    uint32_t taskCount_{0};
    uint32_t workspaceSlots_{0};
    uint32_t splitkFactor_{0};
    uint32_t hasGate_{0};
    uint32_t g56Rows_{0};
    uint32_t gateRows_{0};
    uint32_t scheduleBeatCount_{0};
    uint32_t aicNum_{0};
    GM_ADDR workspace_{nullptr};
    GM_ADDR gmWUp_{nullptr};
    GM_ADDR gmWDown_{nullptr};
    GM_ADDR gmWGate_{nullptr};
    GM_ADDR gmDwUp_{nullptr};
    GM_ADDR gmDwDown_{nullptr};
    GM_ADDR gmDwGate_{nullptr};
    bool singleTask_{false};
    bool producerConsumer_{false};
    HcGradWorkspaceLayout layout_;
    HcGradCreditSync credit_;
    GemmT gemm_;
    HcGradVector<D_T, HAS_GAMMA> vector_;
};

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_KERNEL_COMMON_H
