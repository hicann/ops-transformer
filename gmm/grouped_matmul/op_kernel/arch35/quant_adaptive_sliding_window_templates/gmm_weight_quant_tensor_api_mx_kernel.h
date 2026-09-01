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
 * \file gmm_weight_quant_tensor_api_mx_kernel.h
 * \brief Adapter from GroupedMatmul operator inputs and tiling data to the MX A8W4 Tensor API kernel.
 */

#ifndef GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
#define GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H

#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

namespace GROUPED_MATMUL {

template <class XType_, class WeightType_, class ScaleBType_, class BiasType_, class YType_, bool IsSingleMultiSingle_>
__aicore__ inline void GmmWeightQuantTensorApiMxKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, const GroupedMatmulTilingData::GMMWeightQuantParam *gmmBaseParams, const TCubeTiling *mmTilingData)
{
    using XType = XType_;
    using WeightType = WeightType_;
    using ScaleBType = ScaleBType_;
    using BiasType = BiasType_;
    using YType = YType_;
    constexpr bool IS_SINGLE_MULTI_SINGLE = IsSingleMultiSingle_;
    using AType = XType;
    using BType = WeightType;
    using ScaleAType = ScaleBType;
    using CType = YType;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithWeightQuantMx;
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::ZNLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using LayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
    using LayoutScaleB = AscendC::Te::ScaleBDNLayoutPtn;

    using ProblemShape = decltype(AscendC::Te::MakeShape(0UL, 0UL, 0UL, 0UL));
    using BlockScheduler =
        Blaze::Gemm::Kernel::BlockSchedulerWqgmmNResplit<decltype(AscendC::Te::MakeShape(0UL, 0UL, 0UL))>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AscendC::Std::tuple<AType, ScaleAType>,
                                      AscendC::Std::tuple<LayoutA, LayoutScaleA>,
                                      AscendC::Std::tuple<BType, ScaleBType>,
                                      AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC, BiasType, LayoutBias>;
    using BlockEpilogue = void;
    using BlockPrologue = Blaze::Gemm::Kernel::GroupedWeightPrologueMx<AType, BType, BiasType>;
    using KernelImpl =
        Blaze::Gemm::Kernel::GmmWeightQuantMxKernel<ProblemShape, BlockMmad, BlockScheduler, BlockEpilogue,
                                                    BlockPrologue, IS_SINGLE_MULTI_SINGLE>;

    GM_ADDR xAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<XType>(0, x));
    GM_ADDR weightAddr = weight;
    GM_ADDR scaleBAddr = antiquantScale;
    GM_ADDR biasAddr = bias;
    GM_ADDR yAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    // SMS keeps W/ScaleB/Bias as tensor-list descriptors for per-group lookup.
    if constexpr (!IS_SINGLE_MULTI_SINGLE) {
        // The host builds this tiling-key bit as MX && singleX && !singleWeight && singleY. Therefore true is
        // SMS (singleWeight == 0), while a host-valid false key is SSS (singleWeight == 1); MX MMM is rejected by
        // the host NZ weight-dimension checks. Resolve each one-element list once for the SSS path.
        weightAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<WeightType>(0, weight));
        scaleBAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleBType>(0, antiquantScale));
        if (gmmBaseParams->hasBias != 0) {
            biasAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BiasType>(0, bias));
        }
    }

    typename BlockMmad::Params mmParams{xAddr, perTokenScale, scaleBAddr, biasAddr, yAddr};
    typename BlockScheduler::Params schedulerParams{static_cast<uint64_t>(gmmBaseParams->mainBlockCount),
                                                    static_cast<uint64_t>(gmmBaseParams->mainBlockSize),
                                                    static_cast<uint64_t>(gmmBaseParams->firstTailBlockCount),
                                                    static_cast<uint64_t>(gmmBaseParams->firstTailBlockSize),
                                                    static_cast<uint64_t>(gmmBaseParams->secondTailBlockCount),
                                                    static_cast<uint64_t>(gmmBaseParams->secondTailBlockSize),
                                                    static_cast<uint64_t>(gmmBaseParams->coreNum),
                                                    static_cast<uint64_t>(gmmBaseParams->cubeNumBlocksN),
                                                    static_cast<int32_t>(mmTilingData->baseM),
                                                    static_cast<uint64_t>(gmmBaseParams->nSize)};
    typename BlockPrologue::Params prologueParams{reinterpret_cast<__gm__ BType *>(weightAddr)};
    typename KernelImpl::Params params{
        {0UL, static_cast<uint64_t>(gmmBaseParams->kSize), static_cast<uint64_t>(gmmBaseParams->nSize),
         static_cast<uint64_t>(gmmBaseParams->groupNum)},
        mmParams,
        schedulerParams,
        prologueParams,
        groupList,
        static_cast<uint32_t>(gmmBaseParams->groupListType),
        static_cast<uint32_t>(gmmBaseParams->hasBias)};
    KernelImpl kernelImpl;
    kernelImpl(params);
}

} // namespace GROUPED_MATMUL

#endif // GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
