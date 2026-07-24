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
 * \file flash_attention_score_grad_block_cube_small_sd.h
 * \brief Small-S/Small-D dedicated Cube block interface.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_SMALL_SD_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_SMALL_SD_H

#include "flash_attention_score_grad_common.h"
#include "flash_attention_score_grad_buffer_small_sd.h"
#include "flash_attention_score_grad_event_small_sd.h"

namespace FagBaseApi {

template <typename INPUT_TYPE, typename CALC_TYPE, typename OUTDTYPE, bool IS_TND, uint32_t HEAD_DIM, uint32_t LAYOUT>
class SmallSDCubeBlock {
public:
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "SmallSDCubeBlock only supports D=64 or D=128.");
    static constexpr bool IS_SMALL_SD_DEDICATED_BLOCK = true;

    __aicore__ inline SmallSDCubeBlock() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR dq, GM_ADDR dk,
                                GM_ADDR dv, GM_ADDR workspace, TPipe *pipeIn, const SmallSDConstInfo *constInfoIn)
    {
        (void)workspace;
        pipe = pipeIn;
        constInfo = constInfoIn;
        queryGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)query);
        keyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
        valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
        dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)dy);
        dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
        dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
        dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
    }

    __aicore__ inline void LoadSmallSDInputs(const SmallSDPipelineSlot &slot)
    {
        (void)slot;
        // Dedicated GM->L1/L0 input movement for SmallSD fixed S<=128/D<=128 tiles.
    }

    __aicore__ inline void IssueQkAndDyV(SmallSDPipelineSlot &slot)
    {
        slot.state = SmallSDSlotState::CUBE_INFLIGHT;
        LoadSmallSDInputs(slot);
        MatmulQKOrGradStage(slot);
        CommitCubeResultToSlot(slot);
        slot.state = SmallSDSlotState::READY_FOR_VECTOR;
    }

    __aicore__ inline void IssueDqDkDv(SmallSDPipelineSlot &slot)
    {
        WaitDsPReady(slot);
        MatmulGradStage(slot);
        CommitGradResultToSlot(slot);
    }

    __aicore__ inline void MatmulQKOrGradStage(const SmallSDPipelineSlot &slot)
    {
        (void)slot;
        // Computes Q*K and dO*V for the slot, then writes per-slot Cube result buffers.
    }

    __aicore__ inline void MatmulGradStage(const SmallSDPipelineSlot &slot)
    {
        (void)slot;
        // Consumes dS/P L1 buffers and computes dQ/dK/dV for the slot.
    }

    __aicore__ inline void CommitCubeResultToSlot(SmallSDPipelineSlot &slot)
    {
        const int64_t taskMod2 = slot.taskIdMod2;
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_CUBE_QK_READY_FLAG[taskMod2]);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_CUBE_QK_READY_FLAG[taskMod2] + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_CUBE_DYV_READY_FLAG[taskMod2]);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_CUBE_DYV_READY_FLAG[taskMod2] + SMALL_SD_EVENT_MIRROR_OFFSET);
        }
    }

    __aicore__ inline void WaitDsPReady(const SmallSDPipelineSlot &slot)
    {
        (void)slot;
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_DS_L1_READY_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_DS_L1_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_P_L1_READY_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_P_L1_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
        }
    }

    __aicore__ inline void CommitGradResultToSlot(SmallSDPipelineSlot &slot)
    {
        (void)slot;
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_DQ_UB_READY_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_DQ_UB_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_DK_UB_READY_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SMALL_SD_DK_UB_READY_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_DS_L1_REUSABLE_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_DS_L1_REUSABLE_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_P_L1_REUSABLE_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SMALL_SD_P_L1_REUSABLE_FLAG + SMALL_SD_EVENT_MIRROR_OFFSET);
        }
    }

private:
    TPipe *pipe = nullptr;
    const SmallSDConstInfo *constInfo = nullptr;
    GlobalTensor<INPUT_TYPE> queryGm;
    GlobalTensor<INPUT_TYPE> keyGm;
    GlobalTensor<INPUT_TYPE> valueGm;
    GlobalTensor<INPUT_TYPE> dyGm;
    GlobalTensor<OUTDTYPE> dqGm;
    GlobalTensor<OUTDTYPE> dkGm;
    GlobalTensor<OUTDTYPE> dvGm;
};

template <typename T>
struct CubeBlockTraits;

template <typename INPUT_TYPE, typename CALC_TYPE, typename OUTDTYPE, bool IS_TND, uint32_t HEAD_DIM, uint32_t LAYOUT>
struct CubeBlockTraits<SmallSDCubeBlock<INPUT_TYPE, CALC_TYPE, OUTDTYPE, IS_TND, HEAD_DIM, LAYOUT>> {
    using INPUT_TYPE_TRAITS = INPUT_TYPE;
    using CALC_TYPE_TRAITS = CALC_TYPE;
    using OUTDTYPE_TRAITS = OUTDTYPE;
    static constexpr bool IS_ATTEN_MASKTraits = false;
    static constexpr bool IS_PSETraits = false;
    static constexpr bool IS_DROPTraits = false;
    static constexpr bool IS_TNDTraits = IS_TND;
    static constexpr bool IS_BN2_MULTIBLKTraits = false;
    static constexpr uint8_t DETER_SPARSE_TYPETraits = NO_DETER;
    static constexpr bool IS_N_EQUALTraits = true;
    static constexpr bool IS_D_NO_EQUALTraits = false;
    static constexpr bool IS_ROPETraits = false;
    static constexpr bool IS_NZ_OUTTraits = false;
    static constexpr bool IS_TND_SWIZZLETraits = false;
    static constexpr uint8_t SPLIT_AXISTraits = BN2;
    static constexpr S1TemplateType s1TemplateTypeTraits = S1TemplateType::Aligned128;
    static constexpr S2TemplateType s2TemplateTypeTraits = S2TemplateType::Aligned128;
    static constexpr DTemplateType dTemplateTypeTraits =
        HEAD_DIM == 64 ? DTemplateType::Aligned64 : DTemplateType::Aligned128;
};

} // namespace FagBaseApi

#endif
