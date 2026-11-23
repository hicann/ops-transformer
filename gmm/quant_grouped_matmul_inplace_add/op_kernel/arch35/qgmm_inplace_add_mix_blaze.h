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
 * \file qgmm_inplace_add_mix_blaze.h
 * \brief HiFloat8 per-tensor/per-channel grouped matmul with FP32 inplace add.
 */
#pragma once

#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_mix_fixpipe_quant.h"
#include "quant_grouped_matmul_inplace_add_tiling_data.h"

__aicore__ inline void QGmmInplaceAddMixTensorApi(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2, GM_ADDR groupList,
                                                  GM_ADDR scale1, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, quantGmmInplaceAddParams,
                           groupParams, tiling);
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, mmTilingData, mmTiling,
                           tiling);
    if (groupParams.groupNum == 0 || mmTiling.M == 0 || mmTiling.N == 0 || mmTiling.Ka == 0) {
        return;
    }
    using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::dn_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using Policy = Blaze::Gemm::MatmulWithScaleMix<Blaze::Gemm::NONE_FULL_LOAD_MODE, true,
                                                   Blaze::Gemm::KernelGroupedMmadWithScaleMixInplaceAdd>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, hifloat8_t, LayoutA, AscendC::Std::tuple<hifloat8_t, float>,
                                               LayoutB, float, LayoutB, float, LayoutB>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<float, float, float, float, float>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Epilogue, Scheduler>;
    Kernel::Params params{};
    params.groupListGmAddr = groupList;
    auto& gmmParams = params.gmmParams;
    gmmParams.groupNum = groupParams.groupNum;
    gmmParams.groupListType = groupParams.groupListType;
    gmmParams.m = mmTiling.M;
    gmmParams.n = mmTiling.N;
    gmmParams.k = mmTiling.Ka;
    gmmParams.baseM = mmTiling.baseM;
    gmmParams.baseN = mmTiling.baseN;
    gmmParams.baseK = mmTiling.baseK;

    auto& mmadParams = params.mmadParams;
    mmadParams.aGmAddr = x1;
    mmadParams.bGmAddr = x2;
    mmadParams.problemShape = Shape{gmmParams.m, gmmParams.n, gmmParams.k, 0};
    mmadParams.l0TileShape = Shape{gmmParams.baseM, gmmParams.baseN, gmmParams.baseK, 0};
    mmadParams.kAL1 = mmTiling.stepKa * mmTiling.baseK;
    mmadParams.kBL1 = mmTiling.stepKb * mmTiling.baseK;
    mmadParams.l1BufferNum = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    mmadParams.enableL0CPingPong = mmTiling.dbL0C == Blaze::Gemm::DOUBLE_BUFFER_COUNT;

    auto& epilogueParams = params.epilogueParams;
    epilogueParams.x1ScaleGmAddr = scale1;
    epilogueParams.x2ScaleGmAddr = scale2;
    epilogueParams.outGmAddr = y;
    epilogueParams.m = gmmParams.m;
    epilogueParams.n = gmmParams.n;
    epilogueParams.baseM = gmmParams.baseM;
    epilogueParams.baseN = gmmParams.baseN;
    epilogueParams.x1QuantMode = static_cast<uint32_t>(Epilogue::QuantMode::PERTENSOR_MODE);
    epilogueParams.x2QuantMode = static_cast<uint32_t>(Epilogue::QuantMode::PERCHANNEL_MODE);
    epilogueParams.disableOutL2Cache = true;

    Kernel op;
    op(params);
}
