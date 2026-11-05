/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310

#include "kernel_tiling/kernel_tiling.h"
#include "arch35/mega_mhc_common.h"
#include "arch35/mega_mhc_tiling_key.h"
#include "arch35/schedule_mk.h"
#include "arch35/schedule_bs.h"

using namespace AscendC;
using namespace MegaMhc;

/**
 * Total entry for the 950 no_mid/with_mid M-K and BS families.
 * NoMid: complete RMS0/Post rows -> overlapped m-k projection when wave tiling is selected.
 * WithMid: complete RMS0/Post/Mid -> barrier -> m-k projection -> barrier.
 * SKIP_RMS0 bypasses RMS0 in both families and passes x directly to Post.
 * MK supports arbitrary M/K grids, including one M lane; BS partitions tokens with full K.
 * All variants reuse the same ordered Pre/RMS1 compute routines.
 */
template <uint16_t TILING_MODE, bool WITH_MID, bool USE_HF32, bool SKIP_RMS0>
__global__ __aicore__ void mega_mhc(GM_ADDR x, GM_ADDR residual, GM_ADDR postMix, GM_ADDR fn, GM_ADDR mixScales,
                                    GM_ADDR mixBases, GM_ADDR rmsNorm1Weight, GM_ADDR combResMix,
                                    GM_ADDR shiftedPrevMix, GM_ADDR rmsNorm0Weight, GM_ADDR rmsNormMidWeight,
                                    GM_ADDR newResidual, GM_ADDR newPostMix, GM_ADDR newCombResMix, GM_ADDR newPrevMix,
                                    GM_ADDR y, GM_ADDR yQuant, GM_ADDR yScale, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using T = DTYPE_X;

    __gm__ uint8_t* userWorkspace = GetUserWorkspace(workspaceGM);
    Kernel::Tensors<T> tensors;
    tensors.hOut.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    tensors.residual.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(residual));
    tensors.hPost.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(postMix));
    tensors.hRes.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(combResMix));
    tensors.phi.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(fn));
    tensors.alpha.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mixScales));
    tensors.bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mixBases));
    tensors.gamma1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rmsNorm1Weight));
    if constexpr (!SKIP_RMS0) {
        tensors.gamma0.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rmsNorm0Weight));
    }
    tensors.hInPrime.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    tensors.hPostNew.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(newPostMix));
    tensors.hResNew.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(newCombResMix));
    tensors.x2Out.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(newResidual));
    tensors.partialMm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace + tilingData.partialMmOffset));
    tensors.partialRms.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace + tilingData.partialRmsOffset));
    tensors.stage.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(userWorkspace + tilingData.stageOffset));
    if constexpr (WITH_MID) {
        tensors.gamma2.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rmsNormMidWeight));
    }

    if constexpr (TILING_MODE == MEGA_MHC_SPLIT_M_K) {
        Schedule::RunMkSchedule<T, WITH_MID, USE_HF32, !SKIP_RMS0>(tensors, tilingData);
    } else if constexpr (TILING_MODE == MEGA_MHC_SPLIT_BS) {
        Schedule::RunBsSchedule<T, WITH_MID, USE_HF32, !SKIP_RMS0>(tensors, tilingData);
    }
}

#endif // defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
