/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "../grouped_matmul_tiling_data_apt.h"

namespace GROUPED_MATMUL {
template <class XType, class WType, class BiasType, bool TransB, bool HasOffset>
__aicore__ inline void GroupedMatmulWeightAntiquantTensorApi(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                             GM_ADDR offset, GM_ADDR groupList, GM_ADDR y,
                                                             GM_ADDR tiling)
{
    using Tiling = GroupedMatmulTilingData::GMMWeightQuantAswTilingData;
    GET_TILING_DATA_MEMBER(Tiling, gmmWeightQuantAswParams, gp, tiling);
    GET_TILING_DATA_MEMBER(Tiling, mmTilingData, mp, tiling);
    using Nd = asc::te::nd_ext_layout_ptn;
    // WQMM BlockMmad describes the physical weight in (N,K) coordinates.
    using WeightLayout = AscendC::Std::conditional_t<TransB, Nd, asc::te::dn_ext_layout_ptn>;
    constexpr uint64_t AIV_COUNT = 1;
    constexpr uint32_t UB_BUFFER_COUNT = 2;
    // Shared Policy compatibility argument; GMM UB geometry comes from GroupedWeightAntiquantConfig.
    constexpr uint32_t POLICY_UB_INNER_BYTES = 256;
    using Policy = Blaze::Gemm::MatmulWithWeightAntiquant<AIV_COUNT, POLICY_UB_INNER_BYTES, UB_BUFFER_COUNT,
                                                          Blaze::Gemm::QuantMode::PERCHANNEL_MODE, HasOffset,
                                                          Blaze::Gemm::KernelWqgmmMixAntiquant>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, XType, Nd, AscendC::Std::tuple<WType, XType>,
                                               AscendC::Std::tuple<WeightLayout, Nd>, XType, Nd, BiasType, Nd>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Scheduler::ProblemShape, Mmad,
                                                      Blaze::Epilogue::Block::BlockEpilogueEmpty, Scheduler>;
    typename Kernel::Params params;
    params.problemShape = {mp.M, mp.N, mp.Ka, 1};
    params.mmParams.kSize = mp.Ka;
    params.mmParams.hasBias = gp.hasBias != 0;
    const int64_t kL1 = static_cast<int64_t>(mp.baseK) * mp.stepKb;
    params.mmParams.l1TileShape = {mp.baseM, mp.baseN, kL1, kL1};
    params.mmParams.l0TileShape = {mp.baseM, mp.baseN, mp.baseK};
    params.schedulerParams = {mp.baseM, mp.baseN};
    params.x = x;
    params.weight = weight;
    params.bias = gp.hasBias ? bias : nullptr;
    params.scale = scale;
    params.offset = offset;
    params.groupList = groupList;
    params.y = y;
    params.groupNum = gp.groupNum;
    params.groupListType = gp.groupListType;
    Kernel{}(params);
}
} // namespace GROUPED_MATMUL
