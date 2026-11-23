/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file text_gated_residual_grad_credit_sync.h
 * \brief Architecture-specific credit synchronization for the shared gradient pipeline.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_CREDIT_SYNC_H
#define TEXT_GATED_RESIDUAL_GRAD_CREDIT_SYNC_H

#include "../text_gated_residual_grad_constants.h"

#include "kernel_operator.h"
#include "text_gated_residual_grad_pipeline_arch35.h"

namespace TextGatedResidualGrad {

// Bounded producer-consumer synchronization: mode 0 aggregates peers; A5 uses mode 4 and A2/A3 mode 2 across core
// types. Each dependency edge has one flag counted in task order. Ring-slot reuse also consumes credits.
class HcGradCreditSync {
    static constexpr uint8_t SAME_KIND_MODE = 0U;
    static constexpr uint8_t A2_EDGE_MODE = 2U;
    static constexpr uint8_t A5_EDGE_MODE = 4U;
    static constexpr uint16_t AIV_SUBBLOCK_FLAG_OFFSET = 16U;

    enum Edge : uint16_t {
        // AIV -> AIC
        EDGE_V1_C1 = 0,
        EDGE_V3_DX = 1,
        EDGE_V3_G56 = 2,
        EDGE_V6_DX = 3,
        EDGE_RD_C1 = 4,
        EDGE_WD_C1 = 5,
        EDGE_WU_G56 = 6,
        // AIC -> AIV: on 2201, avoid SyncAll IDs 11/12/13. Both directions may reuse this ID set.
        // Select the architecture with __NPU_ARCH__; __CCE_AICORE__ is not 2201 on A2/A3.
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        EDGE_C1_RD = 0,
        EDGE_C1_WU = 1,
        EDGE_DX_V6 = 2,
        EDGE_G56_WD = 3,
        EDGE_G56_V1 = 4,
        EDGE_G56_V3 = 5,
        EDGE_DX_V3 = 6,
#else
        // MIX 1:2 SyncAll<false> reserves mode-4 IDs 12/13 and AIV1 IDs 28/29.
        // The two directions use different receiving-core counters, so AIC -> AIV can independently use 6/7.
        EDGE_C1_RD = 8,
        EDGE_C1_WU = 9,
        EDGE_DX_V6 = 10,
        EDGE_G56_WD = 11,
        EDGE_G56_V1 = 6,
        EDGE_G56_V3 = 7,
        EDGE_DX_V3 = 14,
#endif
    };

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
    static constexpr uint16_t HC_GRAD_KIND_SYNC_FLAG = 10U;
    static constexpr uint8_t HC_GRAD_EDGE_MODE = A2_EDGE_MODE;
#else
    static constexpr uint16_t HC_GRAD_KIND_SYNC_FLAG = 15U;
    static constexpr uint8_t HC_GRAD_EDGE_MODE = A5_EDGE_MODE;
#endif

public:
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    // The chunk schedule and task credit ring are mutually exclusive.
    static constexpr uint16_t CHUNK_V1_READY = EDGE_V1_C1;
    static constexpr uint16_t CHUNK_V3_READY = EDGE_V3_DX;
    static constexpr uint16_t CHUNK_C1A_READY = EDGE_C1_RD;
    // The UB path delivers C1a tiles locally and uses this edge for G56 storage ownership.
    static constexpr uint16_t CHUNK_G56_READ_DONE = EDGE_C1_RD;
    static constexpr uint16_t CHUNK_DX_READY = EDGE_DX_V6;
    static constexpr uint16_t CHUNK_DW_READY = EDGE_G56_WD;

    HC_GRAD_HOST_DEVICE void PublishChunkInput(uint16_t flag)
    {
        // A Cube tile can consume entries from multiple Vector producers.
        constexpr uint16_t AIV_AGGREGATE_FLAG = 6U;
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::CrossCoreSetFlag<SAME_KIND_MODE, PIPE_MTE3>(AIV_AGGREGATE_FLAG);
        AscendC::CrossCoreWaitFlag<SAME_KIND_MODE, PIPE_MTE3>(AIV_AGGREGATE_FLAG);
        Send(flag);
    }

    template <uint16_t READY>
    HC_GRAD_HOST_DEVICE void PublishChunkOutput()
    {
        static_assert(READY == CHUNK_C1A_READY || READY == CHUNK_DX_READY || READY == CHUNK_DW_READY);
        // Use distinct aggregate counters because C1b and G56 can follow DX
        // before every Vector core has begun consuming the DX publication.
        constexpr uint16_t AIC_AGGREGATE_FLAG = READY == CHUNK_C1A_READY ? 3U : (READY == CHUNK_DX_READY ? 4U : 7U);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::CrossCoreSetFlag<SAME_KIND_MODE, PIPE_FIX>(AIC_AGGREGATE_FLAG);
        AscendC::CrossCoreWaitFlag<SAME_KIND_MODE, PIPE_FIX>(AIC_AGGREGATE_FLAG);
        Send(READY);
    }

    HC_GRAD_HOST_DEVICE void WaitChunkStage(uint16_t flag)
    {
        WaitEdge(flag);
    }

    HC_GRAD_HOST_DEVICE void JoinAivWritesForReads()
    {
        constexpr uint16_t AIV_AGGREGATE_FLAG = 8U;
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::CrossCoreSetFlag<SAME_KIND_MODE, PIPE_MTE3>(AIV_AGGREGATE_FLAG);
        // Peer producers write on MTE3 and consumers read on MTE2.
        AscendC::CrossCoreWaitFlag<SAME_KIND_MODE, PIPE_MTE2>(AIV_AGGREGATE_FLAG);
    }
#endif
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
    // The retained-chunk scheduler owns these edges separately from the task ring.
    static constexpr uint16_t A2_C1A_INPUT_READY = EDGE_V1_C1;
    static constexpr uint16_t A2_C1A_OUTPUT_READY = EDGE_C1_RD;
    static constexpr uint16_t A2_DX_INPUT_READY = EDGE_V3_DX;
    static constexpr uint16_t A2_DX_OUTPUT_READY = EDGE_DX_V6;
    static constexpr uint16_t A2_C1B_INPUT_READY = EDGE_RD_C1;
    static constexpr uint16_t A2_C1B_READ_DONE = EDGE_C1_WU;
    static constexpr uint16_t A2_G56_INPUT_READY = EDGE_V3_G56;
    static constexpr uint16_t A2_G56_READ_DONE = EDGE_G56_WD;

    HC_GRAD_HOST_DEVICE void PublishA2ChunkEdge(uint16_t edge)
    {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        SameKindBarrier();
        Send(edge);
    }

    HC_GRAD_HOST_DEVICE void WaitA2ChunkEdge(uint16_t edge)
    {
        WaitEdge(edge);
    }

    static constexpr uint16_t C1A_SLAB_READY = EDGE_V1_C1;
    HC_GRAD_HOST_DEVICE void MarkC1aSlabReady()
    {
        Send(C1A_SLAB_READY);
    }
#endif

    HC_GRAD_HOST_DEVICE void Init(GM_ADDR, uint64_t, uint32_t taskCount, uint32_t workspaceSlots, uint32_t, uint32_t)
    {
        workspaceSlots_ = workspaceSlots;
        taskCount_ = taskCount;
    }

    HC_GRAD_HOST_DEVICE void SameKindBarrier()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<SAME_KIND_MODE, PIPE_FIX>(HC_GRAD_KIND_SYNC_FLAG);
            if constexpr (HC_GRAD_EDGE_MODE == A2_EDGE_MODE) {
                AscendC::CrossCoreWaitFlag<SAME_KIND_MODE>(HC_GRAD_KIND_SYNC_FLAG);
            } else {
                AscendC::CrossCoreWaitFlag<SAME_KIND_MODE, PIPE_FIX>(HC_GRAD_KIND_SYNC_FLAG);
            }
        } else {
            AscendC::CrossCoreSetFlag<SAME_KIND_MODE, PIPE_MTE3>(HC_GRAD_KIND_SYNC_FLAG);
            if constexpr (HC_GRAD_EDGE_MODE == A2_EDGE_MODE) {
                AscendC::CrossCoreWaitFlag<SAME_KIND_MODE>(HC_GRAD_KIND_SYNC_FLAG);
            } else {
                AscendC::CrossCoreWaitFlag<SAME_KIND_MODE, PIPE_MTE3>(HC_GRAD_KIND_SYNC_FLAG);
            }
        }
    }

    template <bool DEFER_WEIGHT_GRADS = false>
    HC_GRAD_HOST_DEVICE void WaitCubeDeps(HcGradCubeOp op, uint32_t task)
    {
        switch (op) {
            case HcGradCubeOp::C1:
                WaitEdge(EDGE_V1_C1);
                if (task >= workspaceSlots_) {
                    WaitEdge(EDGE_RD_C1); // Credit for the dactPartial slot.
                    if constexpr (!DEFER_WEIGHT_GRADS) {
                        WaitEdge(EDGE_WD_C1); // Credit for the dwPartial slot.
                    }
                }
                break;
            case HcGradCubeOp::DX:
                WaitEdge(EDGE_V3_DX);
                // C1 and DX both run on AIC; the peer barrier already guarantees C1 completion.
                if (task >= workspaceSlots_) {
                    WaitEdge(EDGE_V6_DX); // Credit for the dx2 slot.
                }
                break;
            case HcGradCubeOp::G56:
                WaitEdge(EDGE_V3_G56); // All platforms normally read packed directly as ColumnMajor.
                WaitEdge(EDGE_WU_G56); // REDUCE_WU has consumed the dwUp partial.
                break;
            default:
                break;
        }
    }

    template <bool DEFER_WEIGHT_GRADS = false>
    HC_GRAD_HOST_DEVICE void WaitVectorDeps(HcGradVectorOp op, uint32_t task)
    {
        switch (op) {
            case HcGradVectorOp::V1:
                if constexpr (!DEFER_WEIGHT_GRADS) {
                    if (task >= workspaceSlots_) {
                        // The AIV peer barrier covers V6 -> V1; G56 requires a cross-core-type credit.
                        WaitEdge(EDGE_G56_V1);
                    }
                }
                break;
            case HcGradVectorOp::REDUCE_DACT:
                WaitEdge(EDGE_C1_RD);
                break;
            case HcGradVectorOp::V3:
                if constexpr (!DEFER_WEIGHT_GRADS) {
                    // Wait for the previous packed consumers before a ring slot can be overwritten.
                    if (task >= workspaceSlots_) {
                        WaitEdge(EDGE_DX_V3);
                        WaitEdge(EDGE_G56_V3);
                    }
                }
                break;
            case HcGradVectorOp::V6:
                WaitEdge(EDGE_DX_V6);
                break;
            case HcGradVectorOp::REDUCE_WU:
                WaitEdge(EDGE_C1_WU);
                break;
            case HcGradVectorOp::REDUCE_WD:
                WaitEdge(EDGE_G56_WD);
                break;
            case HcGradVectorOp::REDUCE_WU2:
                WaitEdge(EDGE_C1_WU);
                WaitEdge(EDGE_C1_WU);
                break;
            case HcGradVectorOp::REDUCE_WD2:
                WaitEdge(EDGE_G56_WD);
                WaitEdge(EDGE_G56_WD);
                break;
            default:
                // V3/RD, TRANSPOSE/V3, V6/V1 and DGAMMA/REDUCE_DGAMMA are same-core-type dependencies,
                // covered by per-step peer barriers and the established execution order.
                break;
        }
    }

    template <bool DEFER_WEIGHT_GRADS = false>
    HC_GRAD_HOST_DEVICE void MarkCube(HcGradCubeOp op, uint32_t task)
    {
        switch (op) {
            case HcGradCubeOp::C1:
                Send(EDGE_C1_RD);
                if constexpr (!DEFER_WEIGHT_GRADS) {
                    Send(EDGE_C1_WU);
                }
                break;
            case HcGradCubeOp::DX:
                Send(EDGE_DX_V6);
                if constexpr (!DEFER_WEIGHT_GRADS) {
                    SendSlotCredit(EDGE_DX_V3, task);
                }
                break;
            case HcGradCubeOp::G56:
                Send(EDGE_G56_WD);
                SendSlotCredit(EDGE_G56_V1, task);
                SendSlotCredit(EDGE_G56_V3, task);
                break;
            default:
                break;
        }
    }

    template <bool DEFER_WEIGHT_GRADS = false>
    HC_GRAD_HOST_DEVICE void MarkVector(HcGradVectorOp op, uint32_t task)
    {
        switch (op) {
            case HcGradVectorOp::V1:
                Send(EDGE_V1_C1);
                break;
            case HcGradVectorOp::REDUCE_DACT:
                SendSlotCredit(EDGE_RD_C1, task);
                break;
            case HcGradVectorOp::V3:
                Send(EDGE_V3_DX);
                if constexpr (!DEFER_WEIGHT_GRADS) {
                    Send(EDGE_V3_G56);
                }
                break;
            case HcGradVectorOp::V6:
                SendSlotCredit(EDGE_V6_DX, task);
                break;
            case HcGradVectorOp::REDUCE_WU:
                Send(EDGE_WU_G56);
                break;
            case HcGradVectorOp::REDUCE_WD:
                SendSlotCredit(EDGE_WD_C1, task);
                break;
            case HcGradVectorOp::REDUCE_WU2:
                Send(EDGE_WU_G56);
                Send(EDGE_WU_G56);
                break;
            case HcGradVectorOp::REDUCE_WD2:
                SendSlotCredit(EDGE_WD_C1, task);
                SendSlotCredit(EDGE_WD_C1, task + 1U);
                break;
            default:
                break;
        }
    }

private:
    HC_GRAD_HOST_DEVICE void WaitEdge(uint16_t flag)
    {
        if ASCEND_IS_AIC {
            if constexpr (HC_GRAD_EDGE_MODE == A2_EDGE_MODE) {
                AscendC::CrossCoreWaitFlag<A2_EDGE_MODE>(flag);
            } else {
                // NN prefetches GM inputs on MTE2. Waiting only on MTE1 permits
                // two L1 stages to read a producer's not-yet-ready GM buffer.
                AscendC::CrossCoreWaitFlag<HC_GRAD_EDGE_MODE, PIPE_MTE2>(flag);
            }
        } else {
            if constexpr (HC_GRAD_EDGE_MODE == A2_EDGE_MODE) {
                AscendC::CrossCoreWaitFlag<A2_EDGE_MODE>(flag);
            } else {
                AscendC::CrossCoreWaitFlag<HC_GRAD_EDGE_MODE, PIPE_MTE2>(flag);
            }
        }
    }

    HC_GRAD_HOST_DEVICE void Send(uint16_t flag)
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<HC_GRAD_EDGE_MODE, PIPE_FIX>(flag);
            if constexpr (HC_GRAD_EDGE_MODE == A5_EDGE_MODE) {
                AscendC::CrossCoreSetFlag<HC_GRAD_EDGE_MODE, PIPE_FIX>(flag + AIV_SUBBLOCK_FLAG_OFFSET);
            }
        } else if ((HC_GRAD_EDGE_MODE == A2_EDGE_MODE) || ((GetBlockIdx() & 1U) == 0U)) {
            // An A2/A3 AIC Wait needs two AIV Sets; A5 sends once after aggregating peers at the barrier.
            AscendC::CrossCoreSetFlag<HC_GRAD_EDGE_MODE, PIPE_MTE3>(flag);
        }
    }

    HC_GRAD_HOST_DEVICE void SendSlotCredit(uint16_t flag, uint32_t task)
    {
        // Send only if another ring iteration has a consumer; the final iteration must leave no unmatched credit.
        if (static_cast<uint64_t>(task) + workspaceSlots_ < taskCount_) {
            Send(flag);
        }
    }

    uint32_t taskCount_{0};
    uint32_t workspaceSlots_{0};
};

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_CREDIT_SYNC_H
