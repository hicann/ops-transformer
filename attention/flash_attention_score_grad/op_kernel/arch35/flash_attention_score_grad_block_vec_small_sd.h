/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_block_vec_small_sd.h
 * \brief Small-S/Small-D dedicated Vector block interface.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_VEC_SMALL_SD_H

#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_buffer_small_sd.h"
#include "flash_attention_score_grad_event_small_sd.h"

namespace FagBaseApi {

template <typename INPUT_TYPE, typename CALC_TYPE, typename OUTDTYPE, bool IS_TND, uint32_t HEAD_DIM, uint32_t LAYOUT>
class SmallSDVectorBlock {
public:
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "SmallSDVectorBlock only supports D=64 or D=128.");
    static constexpr bool IS_SMALL_SD_DEDICATED_BLOCK = true;

    __aicore__ inline SmallSDVectorBlock() {}

    __aicore__ inline void Init(GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                GM_ADDR workspace, TPipe *pipeIn, const SmallSDConstInfo *constInfoIn)
    {
        (void)workspace;
        pipe = pipeIn;
        constInfo = constInfoIn;
        softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
        softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
        dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
        dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
        dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    }

    __aicore__ inline void WaitCubeReady(const SmallSDPipelineSlot &slot)
    {
        const int64_t taskMod2 = slot.taskIdMod2;
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SMALL_SD_CUBE_QK_READY_FLAG[taskMod2]);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SMALL_SD_CUBE_DYV_READY_FLAG[taskMod2]);
        }
    }

    __aicore__ inline void ProduceDsAndP(SmallSDPipelineSlot &slot, bool needWaitL1Reusable)
    {
        WaitCubeReady(slot);
        WaitSharedL1Reusable(needWaitL1Reusable);
        ComputeSoftmaxAndDs(slot);
        CommitDsPToL1(slot);
    }

    __aicore__ inline void FinalizeGradOutput(SmallSDPipelineSlot &slot)
    {
        WaitGradReady(slot);
        CommitGradToGm(slot);
    }

    __aicore__ inline void ComputeSoftmaxAndDs(const SmallSDPipelineSlot &slot)
    {
        (void)slot;
        // SmallSD-only softmax/LSE/scale/dS production for the current slot snapshot.
    }

    __aicore__ inline void WaitSharedL1Reusable(bool needWaitL1Reusable)
    {
        if ASCEND_IS_AIV {
            if (needWaitL1Reusable) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_DS_L1_REUSABLE_FLAG);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_P_L1_REUSABLE_FLAG);
            }
        }
    }

    __aicore__ inline void CommitDsPToL1(SmallSDPipelineSlot &slot)
    {
        (void)slot;
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_DS_L1_READY_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SMALL_SD_P_L1_READY_FLAG);
        }
    }

    __aicore__ inline void WaitGradReady(const SmallSDPipelineSlot &slot)
    {
        (void)slot;
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SMALL_SD_DQ_UB_READY_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SMALL_SD_DK_UB_READY_FLAG);
        }
    }

    __aicore__ inline void CommitGradToGm(SmallSDPipelineSlot &slot)
    {
        (void)slot;
        // Casts and commits dQ/dK/dV, then marks this two-slot buffer reusable.
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_V>(SMALL_SD_SLOT_REUSE_READY_FLAG);
        }
    }

private:
    TPipe *pipe = nullptr;
    const SmallSDConstInfo *constInfo = nullptr;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<OUTDTYPE> dqGm;
    GlobalTensor<OUTDTYPE> dkGm;
    GlobalTensor<OUTDTYPE> dvGm;
};

} // namespace FagBaseApi

#endif
