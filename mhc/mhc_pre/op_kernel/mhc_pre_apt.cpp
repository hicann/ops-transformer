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
 * \file mhc_pre_apt.cpp
 * \brief
 */
#if (defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
#include "arch35/mhc_pre_common.h"
#include "arch35/mhc_pre_split_bs.h"
#include "arch35/mhc_pre_m_k_split.h"
#if !defined(MHC_PRE_ENABLE_ASCEND950)
#include "arch35/mhc_pre_split_nd.h"
#endif
#include "arch35/mhc_pre_tiling_key.h"

using namespace AscendC;
using namespace MhcPre;

#if defined(MHC_PRE_ENABLE_ASCEND950)
// Select the schedule once. Buffer counts and the K/M traversal then become
// compile-time constants, avoiding repeated runtime branches in the kernel.
template <typename T, int8_t RESI_MODE, bool WEIGHT_REUSE>
__aicore__ inline void ProcessMk(InitParams initParams)
{
    MhcPreMKPart1<T, float32_t, RESI_MODE, WEIGHT_REUSE> part1;
    part1.Init(initParams);
    if constexpr (WEIGHT_REUSE && RESI_MODE == MHC_PRE_HAS_RESI && IsSameType<T, bfloat16_t>::value) {
        if (part1.IsWaveCase()) {
            part1.ProcessWave();
            return;
        }
    }
    part1.Process();
    // Part2 reuses UB after all Part1 consumers complete.
    initParams.tPipeIn->Destroy();
    TPipe stage2Pipe;
    initParams.tPipeIn = &stage2Pipe;
    MhcPreMKPart2<T, float32_t, RESI_MODE, WEIGHT_REUSE> part2;
    part2.Init(initParams);
    part2.Process();
}
#endif

template <int8_t TILING_MODE, int8_t RESI_MODE>
__global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma, GM_ADDR hin,
                                   GM_ADDR h_post, GM_ADDR h_res, GM_ADDR inv_rms, GM_ADDR h_mix, GM_ADDR h_pre,
                                   GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);

#if defined(MHC_PRE_ENABLE_ASCEND950)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    TPipe pipe;
    if constexpr (TILING_MODE == MHC_PRE_SPLIT_BS) {
        InitParams initParams{x,     phi,     alpha, bias,  gamma, hin,   h_post,
                              h_res, inv_rms, h_mix, h_pre, user,  &pipe, &tilingData};
#if defined(MHC_PRE_ENABLE_ASCEND950)
        if constexpr (IsSameType<DTYPE_X, bfloat16_t>::value && RESI_MODE == MHC_PRE_HAS_RESI) {
            if (tilingData.implMode == MHC_PRE_IMPL_MODE_FP32 && tilingData.N == 4U &&
                tilingData.D == MHC_PRE_BS_WAVE_D && tilingData.outFlag != 0U) {
                MhcPreSplitBSWave<DTYPE_X, float32_t, RESI_MODE> wave;
                wave.Init(initParams);
                wave.Process();
                return;
            }
        }
#endif
        MhcPreSplitBS<DTYPE_X, float32_t, RESI_MODE> op;
        op.Init(initParams);
        op.Process();
#if defined(MHC_PRE_ENABLE_ASCEND950)
    } else if constexpr (TILING_MODE == MHC_PRE_SPLIT_M_K || TILING_MODE == MHC_PRE_M_K_WEIGHT_REUSE) {
        InitParams initParams{x,     phi,     alpha, bias,  gamma, hin,   h_post,
                              h_res, inv_rms, h_mix, h_pre, user,  &pipe, &tilingData};
        ProcessMk<DTYPE_X, RESI_MODE, TILING_MODE == MHC_PRE_M_K_WEIGHT_REUSE>(initParams);
#else
    } else if constexpr (TILING_MODE == MHC_PRE_SPLIT_M_K) {
        InitParams initParams{x,     phi,     alpha, bias,  gamma, hin,   h_post,
                              h_res, inv_rms, h_mix, h_pre, user,  &pipe, &tilingData};
        MhcPreMKPart1<DTYPE_X, float32_t, RESI_MODE> part1;
        part1.Init(initParams);
        part1.Process();
        // Stage 1 and stage 2 use different UB layouts. Releasing stage-1 queues lets stage 2
        // reuse the full UB budget for reduction and postprocessing.
        pipe.Destroy();
        TPipe stage2Pipe;
        initParams.tPipeIn = &stage2Pipe;
        MhcPreMKPart2<DTYPE_X, float32_t, RESI_MODE> part2;
        part2.Init(initParams);
        part2.Process();
    } else if constexpr (TILING_MODE == MHC_PRE_SPLIT_ND) {
        InitParams initParams{x,     phi,     alpha, bias,  gamma, hin,   h_post,
                              h_res, inv_rms, h_mix, h_pre, user,  &pipe, &tilingData};
        MhcPreSplitND<DTYPE_X, float32_t, RESI_MODE> op;
        op.Init(initParams);
        op.Process();
#endif
    }
}
#endif
