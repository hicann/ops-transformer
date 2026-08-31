/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*! \file gmm_s8s4_perchannel_tensor_api.h
 *  \brief Thin GroupedMatmul adaptor for the ops-tensor S8S4 per-channel/per-group kernel.
 */
#pragma once

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "blaze/gemm/kernel/kernel_gmm_fixpipe_quant.h"
#include "../int4_weight_to_int8_preprocess.h"
#include "gmm_s8s4_rowsum_preprocess.h"
#include "../grouped_matmul_tiling_data_apt.h"
#include "../../grouped_matmul_utils.h"

namespace GROUPED_MATMUL::S8S4V5 {

template <typename YType, CubeFormat WFormat>
__aicore__ inline void InvokeGmmS8S4TensorApi(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset,
                                              GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                              AscendC::TPipe *pipe,
                                              const GroupedMatmulTilingData::GMMS8S4BasicApiTilingData *tiling)
{
    using LayoutB =
        std::conditional_t<WFormat == CubeFormat::NZ, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    using DispatchPolicy =
        Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false, Blaze::Gemm::KernelGroupedMmadWithScaleFixpipeQuant>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, int8_t, LayoutA, BTypeTuple, LayoutB, half, LayoutC,
                                                    int32_t, LayoutBias>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpiloguePerTokenScale<YType, half>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using ProblemShape = typename BlockMmad::ProblemShape;
    using TensorApiKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename TensorApiKernel::Params;
    const auto &mm = tiling->mmTilingData;
    const auto &s8s4 = tiling->s8s4Params;
    GM_ADDR expandedWeight = workspace + s8s4.expandedWeightOffsetBytes;
    GM_ADDR tileWorkspace = workspace + s8s4.tileWorkspaceOffsetBytes;
    GM_ADDR rowSumWorkspace = workspace + s8s4.rowSumOffsetBytes;
    GM_ADDR xGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<int8_t>(0, x));
    GM_ADDR weightGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<AscendC::int4b_t>(0, weight));

    // Operator-owned preprocessing phase. Both AIV producers finish their GM
    // writes before all cores enter the compute-only Tensor API kernel.
    {
        using WeightPreprocess = GROUPED_MATMUL::INT4_PREPROCESS::Int4WeightToInt8Preprocess<LayoutB>;
        typename WeightPreprocess::Params weightParams{};
        weightParams.weightGmAddr = weightGmAddr;
        weightParams.workspaceGmAddr = expandedWeight;
        weightParams.groupNum = tiling->gmmQuantParams.groupNum;
        weightParams.n = mm.n;
        weightParams.k = mm.k;
        weightParams.inputTransposedNz = WFormat == CubeFormat::NZ && s8s4.specialWeightFormat != 0U;
        weightParams.inputNzC032 = WFormat == CubeFormat::NZ;
        WeightPreprocess weightPreprocess;
        weightPreprocess(weightParams, pipe);

        S8S4RowSumPreprocess::Params rowSumParams{};
        rowSumParams.xGmAddr = xGmAddr;
        rowSumParams.rowSumGmAddr = rowSumWorkspace;
        rowSumParams.m = mm.m;
        rowSumParams.k = mm.k;
        rowSumParams.enabled = s8s4.dequantMode != 0U && s8s4.hasOffset != 0U;
        S8S4RowSumPreprocess rowSumPreprocess;
        rowSumPreprocess(rowSumParams, pipe);

        AscendC::SyncAll<false>();
        pipe->Reset();
    }

    Params params{};
    params.mmadParams.aGmAddr = xGmAddr;
    params.mmadParams.bGmAddr = expandedWeight;
    params.mmadParams.cGmAddr = tileWorkspace;
    params.mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<uint64_t>(0, scale));
    params.epilogueParams.workspaceGmAddr = tileWorkspace;
    params.epilogueParams.perTokenScaleGmAddr = perTokenScale;
    params.epilogueParams.offsetGmAddr =
        s8s4.hasOffset != 0U ? reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<float>(0, offset)) : nullptr;
    params.epilogueParams.xRowSumGmAddr = rowSumWorkspace;
    params.epilogueParams.outGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    params.epilogueParams.n = mm.n;
    params.epilogueParams.baseM = mm.baseM;
    params.epilogueParams.baseN = mm.baseN;
    params.epilogueParams.withOffset = s8s4.dequantMode != 0U && s8s4.hasOffset != 0U;
    params.groupListGmAddr = groupList;
    params.gmmParams.groupNum = tiling->gmmQuantParams.groupNum;
    params.gmmParams.m = mm.m;
    params.gmmParams.n = mm.n;
    params.gmmParams.k = mm.k;
    params.gmmParams.baseM = mm.baseM;
    params.gmmParams.baseN = mm.baseN;
    params.gmmParams.baseK = mm.baseK;
    params.gmmParams.quantGroupSize = s8s4.quantGroupSize;
    params.gmmParams.quantMode = s8s4.dequantMode == 0U ?
                                     static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERGROUP_MODE) :
                                     static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERCHANNEL_MODE);
    params.gmmParams.kAL1 = mm.kAL1;
    params.gmmParams.kBL1 = mm.kBL1;
    params.gmmParams.nBufferNum = 2U;
    params.gmmParams.dbL0C = mm.dbL0C;
    params.gmmParams.groupListType = tiling->gmmQuantParams.groupListType;
    {
        TensorApiKernel kernel;
        kernel(params);
    }
#if defined(BLAZE_S8S4_DEBUG)
    KERNEL_LOG(KERNEL_ERROR, "[S8S4][local kernel destroyed] block=%u ratio=%u", AscendC::GetBlockIdx(),
               AscendC::GetTaskRation());
#endif
}

} // namespace GROUPED_MATMUL::S8S4V5
