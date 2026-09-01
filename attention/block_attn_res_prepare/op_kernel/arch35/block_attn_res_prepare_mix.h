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
 * \file block_attn_res_prepare_mix.h
 * \brief Blaze tensor API entry for BlockAttnResPrepare.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_MIX_H
#define BLOCK_ATTN_RES_PREPARE_MIX_H

#include "block_attn_res_prepare_tiling_data.h"
#include "kernel_operator.h"
#include "blaze/gemm/utils/common_utils.h"

using namespace AscendC;
using Blaze::Gemm::CeilAlign;

#include "blaze/attention/kernel/kernel_universal.h"

namespace BlockAttnResPrepare {

namespace BlockAttnResPrepareMixDetail {
using TensorApiKernel = Blaze::Attention::Kernel::KernelBlockAttnResPrepare;
using Params = typename TensorApiKernel::Params;
constexpr uint32_t MM1_L0_K_MAX = 64U;
constexpr uint32_t SINGLE_STAGE = 1U;

__aicore__ inline void AssembleProblemShape(Params &params, const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    params.problemShape = {static_cast<int64_t>(tiling.totalS), static_cast<int64_t>(tiling.totalN),
                           static_cast<int64_t>(tiling.totalD), static_cast<int64_t>(tiling.totalT)};
}

__aicore__ inline void AssembleMm1Params(Params &params, GM_ADDR blockRes, GM_ADDR pseudoQuery, GM_ADDR workspace,
                                         const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    auto &mm1 = params.mm1Params;
    mm1.aGmAddr = pseudoQuery;
    mm1.bGmAddr = blockRes;
    mm1.cGmAddr = workspace;
    mm1.workspaceGmAddr = workspace;
    mm1.mL1 = tiling.sAlign;
    mm1.nL1 = tiling.mm1NAlign;
    mm1.kL1 = tiling.baseDAlign;
    mm1.mL0 = tiling.sAlign;
    mm1.nL0 = tiling.mm1NAlign;
    mm1.kL0 = tiling.baseD < MM1_L0_K_MAX ? tiling.baseD : MM1_L0_K_MAX;
    mm1.l1Stages = tiling.qL1BufferNum < tiling.vL1BufferNum ? tiling.qL1BufferNum : tiling.vL1BufferNum;
    mm1.l0cStages = SINGLE_STAGE;
}

__aicore__ inline void AssembleMm2Params(Params &params, GM_ADDR blockRes, GM_ADDR numerator, GM_ADDR workspace,
                                         const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    auto &mm2 = params.mm2Params;
    mm2.aGmAddr = workspace;
    mm2.bGmAddr = blockRes;
    mm2.cGmAddr = numerator;
    mm2.workspaceGmAddr = workspace;
    mm2.mL1 = tiling.sAlign;
    mm2.nL1 = tiling.baseDAlign;
    mm2.kL1 = tiling.nAlign;
    mm2.mL0 = tiling.sAlign;
    mm2.nL0 = tiling.baseDAlign;
    mm2.kL0 = tiling.nAlign;
    mm2.l1Stages = SINGLE_STAGE;
    mm2.l0cStages = SINGLE_STAGE;
}

__aicore__ inline void AssembleEpilogueParams(Params &params, GM_ADDR validBlocks, GM_ADDR numerator, GM_ADDR logitMax,
                                              GM_ADDR expSum, GM_ADDR workspace,
                                              const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    auto &epilogue = params.epilogueParams;
    epilogue.validBlocksGmAddr = validBlocks;
    epilogue.softmaxMaxGmAddr = logitMax;
    epilogue.weightedOutputGmAddr = numerator;
    epilogue.softmaxSumGmAddr = expSum;
    epilogue.workspaceGmAddr = workspace;
    epilogue.totalD = tiling.totalD;
    epilogue.baseD = tiling.baseD;
    epilogue.baseDAlign = tiling.baseDAlign;
    epilogue.dTileNum = tiling.dTileNum;
    epilogue.sAlign = tiling.sAlign;
    epilogue.vUbBufferNum = tiling.vUbBufferNum;
    epilogue.eWorkspaceElems = tiling.eL1Elems;
    epilogue.vUbElems = tiling.vUbElems;
    epilogue.dotUbElems = tiling.dotUbElems;
    epilogue.reduceUbElems = tiling.reduceUbElems;
    epilogue.softmaxUbElems = tiling.softmaxUbElems;
    epilogue.workspacePerCoreElems = tiling.workspacePerCoreElems;
    epilogue.epsilon = tiling.eps;
}

__aicore__ inline void AssembleSchedulerParams(Params &params, const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    auto &scheduler = params.schedulerParams;
    scheduler.totalWorkUnits = tiling.totalWorkUnits;
    scheduler.usedCoreNum = tiling.usedCoreNum;
    scheduler.baseT = tiling.baseT;
    scheduler.baseS = tiling.baseS;
    scheduler.sTileNum = tiling.sTileNum;
    scheduler.mm1NAlign = tiling.mm1NAlign;
}
} // namespace BlockAttnResPrepareMixDetail

__aicore__ inline void BlockAttnResPrepareTensorApiBlazeKernel(GM_ADDR blockRes, GM_ADDR validBlocks,
                                                               GM_ADDR pseudoQuery, GM_ADDR numerator, GM_ADDR logitMax,
                                                               GM_ADDR expSum, GM_ADDR workspace,
                                                               const optiling::BlockAttnResPrepareMixTilingData &tiling)
{
    BlockAttnResPrepareMixDetail::Params params{};
    BlockAttnResPrepareMixDetail::AssembleProblemShape(params, tiling);
    BlockAttnResPrepareMixDetail::AssembleMm1Params(params, blockRes, pseudoQuery, workspace, tiling);
    BlockAttnResPrepareMixDetail::AssembleMm2Params(params, blockRes, numerator, workspace, tiling);
    BlockAttnResPrepareMixDetail::AssembleEpilogueParams(params, validBlocks, numerator, logitMax, expSum, workspace,
                                                         tiling);
    BlockAttnResPrepareMixDetail::AssembleSchedulerParams(params, tiling);

    BlockAttnResPrepareMixDetail::TensorApiKernel kernel;
    kernel(params);
}

} // namespace BlockAttnResPrepare

#endif // BLOCK_ATTN_RES_PREPARE_MIX_H
