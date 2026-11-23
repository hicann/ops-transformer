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

#define GMM_MX_BUFFERED_INLINE __aicore__ inline
#include "gmm_weight_quant_mx_buffered_policy.h"
#undef GMM_MX_BUFFERED_INLINE
#include "../../mx_a8w4_pipeline_common.h"

namespace GROUPED_MATMUL {

// BEGIN_PRESERVED_ORIGINAL_MX_ADAPTER
template <class XType_, class WeightType_, class ScaleBType_, class BiasType_, class YType_, bool IsSingleMultiSingle_>
__aicore__ inline void GmmWeightQuantTensorApiMxKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, const GroupedMatmulTilingData::GMMWeightQuantParam* gmmBaseParams, const TCubeTiling* mmTilingData)
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
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::zn_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using LayoutScaleA = asc::te::scalea_nd_layout_ptn;
    using LayoutScaleB = asc::te::scaleb_dn_layout_ptn;

    using ProblemShape = decltype(asc::te::make_shape(0UL, 0UL, 0UL, 0UL));
    using BlockScheduler =
        Blaze::Gemm::Kernel::BlockSchedulerWqgmmNResplit<decltype(asc::te::make_shape(0UL, 0UL, 0UL))>;
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

    typename BlockMmad::Params mmParams{};
    mmParams.aGmAddr = xAddr;
    mmParams.scaleAGmAddr = perTokenScale;
    mmParams.scaleBGmAddr = scaleBAddr;
    mmParams.biasGmAddr = biasAddr;
    mmParams.cGmAddr = yAddr;
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
    typename BlockPrologue::Params prologueParams{reinterpret_cast<__gm__ BType*>(weightAddr)};
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

// END_PRESERVED_ORIGINAL_MX_ADAPTER

template <uint32_t Mode, bool ColumnFirst, class XType, class WeightType, class ScaleType, class BiasType, class YType,
          bool IsSingleMultiSingle>
struct MxKernelTypes {
    using DispatchPolicy = Blaze::Gemm::MxWeightPipelineConfig<Mode, XType, BiasType>;
    using BlockScheduler = Blaze::Gemm::Kernel::MxPipeline::BlockSchedulerWqgmmNResplit<
        decltype(asc::te::make_shape(0UL, 0UL, 0UL)), ColumnFirst,
        Mode != static_cast<uint32_t>(Blaze::Gemm::MxPipelineMode::ParameterDriven)>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AscendC::Std::tuple<XType, ScaleType>,
                                      AscendC::Std::tuple<asc::te::nd_ext_layout_ptn, asc::te::scalea_nd_layout_ptn>,
                                      AscendC::Std::tuple<WeightType, ScaleType>,
                                      AscendC::Std::tuple<asc::te::zn_layout_ptn, asc::te::scaleb_dn_layout_ptn>, YType,
                                      asc::te::nd_ext_layout_ptn, BiasType, asc::te::nd_ext_layout_ptn>;
    using BlockPrologue = Blaze::Gemm::Kernel::MxPipeline::GroupedWeightPrologueMx<XType, WeightType, BiasType, Mode>;
    using Kernel =
        Blaze::Gemm::Kernel::MxPipeline::GmmWeightQuantMxKernel<decltype(asc::te::make_shape(0UL, 0UL, 0UL, 0UL)),
                                                                BlockMmad, BlockScheduler, void, BlockPrologue,
                                                                IsSingleMultiSingle, Mode>;
    using Config = Blaze::Gemm::MxWeightPipelineConfig<Mode, XType, BiasType>;
};

// Keep list interpretation identical for all pipeline adapters; SMS descriptors stay unresolved.
template <class KernelImpl, bool IsSingleMultiSingle>
__aicore__ inline typename KernelImpl::Params MakeMxKernelParams(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, const GroupedMatmulTilingData::GMMWeightQuantParam* base, const TCubeTiling* mm)
{
    using XType = typename KernelImpl::AType;
    using WeightType = typename KernelImpl::BType;
    using ScaleBType = typename KernelImpl::ScaleBType;
    using BiasType = typename KernelImpl::BiasType;
    using YType = typename KernelImpl::CType;
    typename KernelImpl::Params params{};
    auto weightAddr = weight;
    auto scaleBAddr = antiquantScale;
    auto biasAddr = bias;
    // This key bit denotes SMS, not whether the arguments use TensorList ABI.
    // Host-valid false is SSS; resolve its one-element W/ScaleB lists only once.
    if constexpr (!IsSingleMultiSingle) {
        weightAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<WeightType>(0, weight));
        scaleBAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<ScaleBType>(0, antiquantScale));
        if (base->hasBias != 0) {
            biasAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<BiasType>(0, bias));
        }
    }
    params.problemShape =
        asc::te::make_shape(0UL, static_cast<uint64_t>(base->kSize), static_cast<uint64_t>(base->nSize),
                            static_cast<uint64_t>(base->groupNum));
    params.mmad.aGmAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<XType>(0, x));
    params.mmad.scaleAGmAddr = perTokenScale;
    params.mmad.scaleBGmAddr = scaleBAddr;
    params.mmad.biasGmAddr = biasAddr;
    params.mmad.cGmAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<YType>(0, y));
    params.scheduler = {static_cast<uint64_t>(base->mainBlockCount),
                        static_cast<uint64_t>(base->mainBlockSize),
                        static_cast<uint64_t>(base->firstTailBlockCount),
                        static_cast<uint64_t>(base->firstTailBlockSize),
                        static_cast<uint64_t>(base->secondTailBlockCount),
                        static_cast<uint64_t>(base->secondTailBlockSize),
                        static_cast<uint64_t>(base->coreNum),
                        static_cast<uint64_t>(base->cubeNumBlocksN),
                        static_cast<int32_t>(mm->baseM),
                        static_cast<uint64_t>(base->nSize)};
    params.prologue = {reinterpret_cast<__gm__ WeightType*>(weightAddr)};
    params.ptrGroupList = groupList;
    params.groupListType = static_cast<uint32_t>(base->groupListType);
    params.hasBias = static_cast<uint32_t>(base->hasBias);
    return params;
}

template <MxBuffered::Route Pipeline, class XType_, class WeightType_, class ScaleBType_, class BiasType_, class YType_,
          bool IsSingleMultiSingle_>
__aicore__ inline void RunMxBuffered(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList,
                                     GM_ADDR perTokenScale, GM_ADDR y,
                                     const GroupedMatmulTilingData::GMMWeightQuantParam* gmmBaseParams,
                                     const TCubeTiling* mmTilingData)
{
    static_assert(Pipeline == MxBuffered::Route::OperandTriple || Pipeline == MxBuffered::Route::WeightQuad,
                  "Buffered MX requires an operand-triple or weight-quad pipeline");
    constexpr bool IS_SINGLE_MULTI_SINGLE = IsSingleMultiSingle_;
    using Types = MxKernelTypes<static_cast<uint32_t>(Pipeline), true, XType_, WeightType_, ScaleBType_, BiasType_,
                                YType_, IS_SINGLE_MULTI_SINGLE>;
    using KernelImpl = typename Types::Kernel;

    auto params = MakeMxKernelParams<KernelImpl, IS_SINGLE_MULTI_SINGLE>(x, weight, antiquantScale, bias, groupList,
                                                                         perTokenScale, y, gmmBaseParams, mmTilingData);
    KernelImpl kernelImpl;
    kernelImpl(params);
}

__aicore__ inline MxBuffered::Route SelectMxBufferedRoute(const GroupedMatmulTilingData::GMMWeightQuantParam* base,
                                                          const TCubeTiling* mm, GM_ADDR groupList)
{
    const auto route = MxBuffered::Select(
        base->groupNum, base->kSize, base->nSize, base->coreNum, mm->baseM, base->hasBias != 0, base->groupType,
        base->groupListType, base->singleX != 0, base->singleWeight != 0, base->singleY != 0,
        base->mainBlockCount + base->firstTailBlockCount + base->secondTailBlockCount);
    const uint64_t maxN = Max(base->mainBlockCount ? base->mainBlockSize : 0UL,
                              Max(base->firstTailBlockCount ? base->firstTailBlockSize : 0UL,
                                  base->secondTailBlockCount ? base->secondTailBlockSize : 0UL));
    return route != MxBuffered::Route::Original &&
                   MxBuffered::MatchesNPartition(base->mainBlockCount, base->mainBlockSize, base->firstTailBlockCount,
                                                 base->firstTailBlockSize, base->secondTailBlockCount,
                                                 base->secondTailBlockSize, base->nSize) &&
                   MxBuffered::FitsBufferedGroups(reinterpret_cast<__gm__ int64_t*>(groupList), base->groupNum, route,
                                                  maxN) ?
               route :
               MxBuffered::Route::Original;
}

template <class XType, class WeightType, class ScaleBType, class BiasType, class YType, bool IsSingleMultiSingle>
__aicore__ inline bool TryRunMxBuffered(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias,
                                        GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                        const GroupedMatmulTilingData::GMMWeightQuantParam* base, const TCubeTiling* mm)
{
    if constexpr (!IsSingleMultiSingle && AscendC::IsSameType<XType, fp8_e4m3fn_t>::value &&
                  AscendC::IsSameType<WeightType, fp4x2_e2m1_t>::value &&
                  AscendC::IsSameType<ScaleBType, AscendC::fp8_e8m0_t>::value &&
                  AscendC::IsSameType<YType, bfloat16_t>::value) {
        const auto route = SelectMxBufferedRoute(base, mm, groupList);
        if (route == MxBuffered::Route::Original) {
            return false;
        }
        if (route == MxBuffered::Route::WeightQuad) {
            RunMxBuffered<MxBuffered::Route::WeightQuad, XType, WeightType, ScaleBType, BiasType, YType, false>(
                x, weight, antiquantScale, bias, groupList, perTokenScale, y, base, mm);
        } else {
            RunMxBuffered<MxBuffered::Route::OperandTriple, XType, WeightType, ScaleBType, BiasType, YType, false>(
                x, weight, antiquantScale, bias, groupList, perTokenScale, y, base, mm);
        }
        return true;
    }
    return false;
}

template <class XType_, class WeightType_, class ScaleBType_, class BiasType_, class YType_, bool IsSingleMultiSingle_>
__aicore__ inline void GmmWeightQuantTensorApiMxBufferedKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, const GroupedMatmulTilingData::GMMWeightQuantParam* gmmBaseParams, const TCubeTiling* mmTilingData)
{
    if (!TryRunMxBuffered<XType_, WeightType_, ScaleBType_, BiasType_, YType_, IsSingleMultiSingle_>(
            x, weight, antiquantScale, bias, groupList, perTokenScale, y, gmmBaseParams, mmTilingData)) {
        GmmWeightQuantTensorApiMxKernel<XType_, WeightType_, ScaleBType_, BiasType_, YType_, IsSingleMultiSingle_>(
            x, weight, antiquantScale, bias, groupList, perTokenScale, y, gmmBaseParams, mmTilingData);
    }
}

// Fixed Aligned needs only actual K/N/group count; other attributes are host guarantees.
__aicore__ inline GroupedMatmulTilingData::GMMWeightQuantParam MakeFixedMxGroupParams(const __gm__ void* address)
{
    const auto* source = reinterpret_cast<const __gm__ GroupedMatmulTilingData::GMMWeightQuantParam*>(address);
    GroupedMatmulTilingData::GMMWeightQuantParam params{};
    params.kSize = source->kSize;
    params.nSize = source->nSize;
    params.groupNum = source->groupNum;
    params.groupListType = 1;
    return params;
}

namespace MxPipeline {
namespace Aligned {
template <class XType_, class WeightType_, class ScaleBType_, class BiasType_, class YType_, bool IsSingleMultiSingle_,
          bool FixedTiles = false>
__aicore__ inline void GmmWeightQuantTensorApiMxKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, const GroupedMatmulTilingData::GMMWeightQuantParam* gmmBaseParams, const TCubeTiling* mmTilingData)
{
    // Aligned publishes weights only from AIV0; AIC never waits for AIV1.
    // Exit before descriptor loads and group traversal on the unused vector lane.
    if ASCEND_IS_AIV {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
    }
    constexpr bool IS_SINGLE_MULTI_SINGLE = IsSingleMultiSingle_;
    constexpr uint32_t MODE = static_cast<uint32_t>(Blaze::Gemm::MxPipelineMode::Aligned);
    using Types =
        MxKernelTypes<MODE, false, XType_, WeightType_, ScaleBType_, BiasType_, YType_, IS_SINGLE_MULTI_SINGLE>;
    using KernelImpl = typename Types::Kernel;
    using Config = typename Types::Config;

    auto params = MakeMxKernelParams<KernelImpl, IS_SINGLE_MULTI_SINGLE>(x, weight, antiquantScale, bias, groupList,
                                                                         perTokenScale, y, gmmBaseParams, mmTilingData);
    if constexpr (FixedTiles) {
        const uint64_t nb = gmmBaseParams->nSize / Config::Consumer::L1_N;
        params.scheduler = {nb,
                            Config::Consumer::L1_N,
                            0,
                            0,
                            0,
                            0,
                            ALIGNED_CORE_COUNT,
                            nb,
                            static_cast<int32_t>(Config::Consumer::L1_M),
                            gmmBaseParams->nSize};
    }
    // The fixed-tile host route guarantees count groups and an absent bias.
    params.groupListType = FixedTiles ? 1U : static_cast<uint32_t>(gmmBaseParams->groupListType);
    params.hasBias = FixedTiles ? 0U : static_cast<uint32_t>(gmmBaseParams->hasBias);
    KernelImpl kernelImpl;
    kernelImpl(params);
}

} // namespace Aligned
} // namespace MxPipeline

} // namespace GROUPED_MATMUL

#endif // GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
