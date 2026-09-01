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
 * \file qgmm_s4s4_kernel.h
 * \brief
 */

#ifndef QGMM_S4S4_KERNEL_H
#define QGMM_S4S4_KERNEL_H

#include "../grouped_matmul_tiling_data_apt.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/kernel/kernel_gmm_fixpipe_quant.h"
#include "../int4_tensor_to_int8_preprocess.h"
#include "../int4_weight_to_int8_preprocess.h"
#include "../../grouped_matmul_utils.h"

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
          class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmS4S4Kernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR groupList,
                                     GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR user1,
                                     const GroupedMatmulTilingData::GMMBaseParamsS4S4 *gmmBaseParams,
                                     const TCubeTiling *mmTilingData, AscendC::TPipe *que)
{
    using AType = int8_t;
    using BType = int8_t;
    using CType = half;
    using BiasType = int32_t;
    using ScaleType = uint64_t;
    using YType = yType;
    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    using DispatchPolicy =
        Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false, Blaze::Gemm::KernelGroupedMmadWithScaleFixpipeQuant>;
    using QgmmBlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType,
                                                        LayoutC, BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpiloguePerTokenScale<YType, CType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using QgmmKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, QgmmBlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QgmmKernel::Params;
    using GMMTiling = typename QgmmKernel::GMMTiling;

    const uint32_t baseK = gmmBaseParams->baseK;
    const bool isPerGroup = gmmBaseParams->quantGroupNum > 1U;

    const uint32_t kAL1 = static_cast<uint32_t>(mmTilingData->stepKa) * baseK;
    const uint32_t kBL1 = static_cast<uint32_t>(mmTilingData->stepKb) * baseK;

    const uint32_t quantGroupSize =
        isPerGroup ? static_cast<uint32_t>(gmmBaseParams->k / gmmBaseParams->quantGroupNum) : 0U;
    const uint32_t quantMode = isPerGroup ? static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERGROUP_MODE) :
                                            static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERCHANNEL_MODE);

    GMMTiling gmmParams{};
    gmmParams.groupNum = gmmBaseParams->groupNum;
    gmmParams.m = static_cast<int64_t>(gmmBaseParams->m);
    gmmParams.n = static_cast<int64_t>(gmmBaseParams->n);
    gmmParams.k = static_cast<int64_t>(gmmBaseParams->k);
    gmmParams.baseM = gmmBaseParams->baseM;
    gmmParams.baseN = gmmBaseParams->baseN;
    gmmParams.baseK = baseK;
    gmmParams.quantGroupSize = quantGroupSize;
    gmmParams.quantMode = quantMode;
    gmmParams.kAL1 = kAL1;
    gmmParams.kBL1 = kBL1;
    gmmParams.nBufferNum = 2U;
    gmmParams.dbL0C = static_cast<uint8_t>(mmTilingData->dbL0C);
    gmmParams.groupListType = gmmBaseParams->groupListType;

    constexpr uint64_t S4S4_MMOUT_PIPELINE = 4UL;
    const uint64_t weightFmtFlags = gmmBaseParams->reserved;
    const bool weightIsNz = (weightFmtFlags & 1ULL) != 0;
    const bool transB = (weightFmtFlags & 2ULL) != 0;
    const bool weightNzC032 = (weightFmtFlags & 4ULL) != 0;
    constexpr uint64_t INT8_NZ_K_ALIGN = 16UL;
    constexpr uint64_t INT8_NZ_N_ALIGN = 32UL;
    const uint64_t int8WeightK = weightIsNz ? (static_cast<uint64_t>(gmmBaseParams->k) + INT8_NZ_K_ALIGN - 1UL) /
                                                  INT8_NZ_K_ALIGN * INT8_NZ_K_ALIGN :
                                              static_cast<uint64_t>(gmmBaseParams->k);
    const uint64_t int8WeightN = weightIsNz ? (static_cast<uint64_t>(gmmBaseParams->n) + INT8_NZ_N_ALIGN - 1UL) /
                                                  INT8_NZ_N_ALIGN * INT8_NZ_N_ALIGN :
                                              static_cast<uint64_t>(gmmBaseParams->n);
    const uint64_t int8WeightWs = static_cast<uint64_t>(gmmBaseParams->groupNum) * int8WeightK * int8WeightN;
    const uint64_t int8XWs = static_cast<uint64_t>(gmmBaseParams->m) * static_cast<uint64_t>(gmmBaseParams->k);
    const uint64_t mmOutWs =
        S4S4_MMOUT_PIPELINE * gmmBaseParams->baseM * gmmBaseParams->baseN * gmmBaseParams->coreNum * sizeof(uint16_t);
    GM_ADDR int8WeightWsAddr = user1;
    GM_ADDR int8XWsAddr = int8WeightWsAddr + int8WeightWs;
    GM_ADDR mmOutWsAddr = int8XWsAddr + int8XWs;
    GM_ADDR perTokenScaleFillWsAddr = mmOutWsAddr + mmOutWs;
    GM_ADDR xGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<int4b_t>(0, x));
    GM_ADDR weightGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<int4b_t>(0, weight));
    GM_ADDR yGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    GM_ADDR scaleGmAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleType>(0, scale));
    GM_ADDR perTokenScaleGmAddr = (gmmBaseParams->isPerTokenQuant == 0U) ? perTokenScaleFillWsAddr : perTokenScale;

    using WeightPreprocess = GROUPED_MATMUL::INT4_PREPROCESS::Int4WeightToInt8Preprocess<LayoutB>;
    typename WeightPreprocess::Params weightPreprocessParams{};
    weightPreprocessParams.weightGmAddr = weightGmAddr;
    weightPreprocessParams.workspaceGmAddr = int8WeightWsAddr;
    weightPreprocessParams.groupNum = gmmBaseParams->groupNum;
    weightPreprocessParams.n = gmmBaseParams->n;
    weightPreprocessParams.k = gmmBaseParams->k;
    weightPreprocessParams.inputTransposedNz = weightIsNz && transB;
    weightPreprocessParams.inputNzC032 = weightIsNz && weightNzC032;

    using XPreprocess = GROUPED_MATMUL::INT4_PREPROCESS::Int4TensorToInt8Preprocess<LayoutA>;
    typename XPreprocess::Params xPreprocessParams{};
    xPreprocessParams.srcInt4GmAddr = xGmAddr;
    xPreprocessParams.dstInt8GmAddr = int8XWsAddr;
    xPreprocessParams.groupNum = 1UL;
    xPreprocessParams.outerDim = gmmBaseParams->m;
    xPreprocessParams.innerDim = gmmBaseParams->k;
    xPreprocessParams.inputTransposedNz = false;

    {
        WeightPreprocess weightPreprocess;
        weightPreprocess(weightPreprocessParams, que);
        XPreprocess xPreprocess;
        xPreprocess(xPreprocessParams, que);
        AscendC::SyncAll<false>();
        que->Reset();
    }

    typename QgmmBlockMmad::Params mmadParams{};
    mmadParams.aGmAddr = int8XWsAddr;
    mmadParams.bGmAddr = int8WeightWsAddr;
    mmadParams.cGmAddr = mmOutWsAddr;
    mmadParams.biasGmAddr = nullptr;
    mmadParams.scaleAGmAddr = nullptr;
    mmadParams.scaleBGmAddr = scaleGmAddr;

    typename BlockEpilogue::Params epilogueParams{};
    epilogueParams.workspaceGmAddr = mmOutWsAddr;
    epilogueParams.perTokenScaleGmAddr = perTokenScaleGmAddr;
    epilogueParams.offsetGmAddr = nullptr;
    epilogueParams.xRowSumGmAddr = nullptr;
    epilogueParams.outGmAddr = yGmAddr;
    epilogueParams.n = gmmBaseParams->n;
    epilogueParams.baseM = gmmBaseParams->baseM;
    epilogueParams.baseN = gmmBaseParams->baseN;
    epilogueParams.withOffset = false;

    Params params{};
    params.mmadParams = mmadParams;
    params.epilogueParams = epilogueParams;
    params.groupListGmAddr = groupList;
    params.gmmParams = gmmParams;

    QgmmKernel gmm;
    gmm(params);
}

#endif
