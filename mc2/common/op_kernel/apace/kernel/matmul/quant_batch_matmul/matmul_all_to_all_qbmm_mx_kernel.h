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
 * \file matmul_all_to_all_qbmm_mx_kernel.h
 * \brief Blaze MXFP4/MXFP8 matmul used before MatmulAlltoAll communication.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "include/tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

using namespace AscendC;
using AscendC::Te::Get;

/**
 * The kernel computes [M, K] x [K, fullN] one destination rank at a time.
 * Each rank's [M, fullN/rankDim] result is written directly to a rank-major
 * Hcomm send buffer, removing the former AIV transpose stage.
 */
template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MatmulAllToAllQbmmMxKernel {
public:
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t CACHE_LINE_ALIGN_MASK = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;

    using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA =
        Std::conditional_t<transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<SCALE_C0>>,
                           Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB =
        Std::conditional_t<transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<SCALE_C0>>,
                           Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<SCALE_C0>>>;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
        uint32_t isBias;
    };

    // Describes the GM address stride between two adjacent communication slices.
    struct SliceParams {
        uint64_t aByteStride;
        uint64_t scaleAByteStride;
        uint64_t cByteStride;
        uint32_t sliceCount;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmParams;
        L1Params l1Params;
        BlockSchedulerParams schedulerParams;
        QBMMTiling qbmmParams;
        uint64_t rankStride;
        uint32_t rankDim;
        SliceParams sliceParams;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        Init(params);
        Run(params);
    }

    // Inject communication as a policy so QBMM owns the compute/sync/communication sequence.
    template <typename CommPolicy>
    __aicore__ inline void operator()(const Params& params, CommPolicy& commPolicy)
    {
        RunSliceGroup(params, commPolicy);
    }

private:
    static constexpr uint16_t AIC_BARRIER_FLAG_ID = 8U;

    BlockMmad mmOp_;

    __aicore__ inline void Init(const Params& params)
    {
        if (params.rankDim == 0) {
            return;
        }
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
        const bool isBias = params.qbmmParams.isBias != 0;
        mmOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias, params.qbmmParams.dbL0C > 1, 1U);
    }

    __aicore__ inline void Run(const Params& params);

    template <typename CommPolicy>
    __aicore__ inline void RunSliceGroup(const Params& groupParams, CommPolicy& commPolicy)
    {
        const auto& sliceParams = groupParams.sliceParams;
        if (groupParams.rankDim == 0 || sliceParams.sliceCount == 0) {
            return;
        }

        // All slices in a group share one BlockMmad resource lifetime.
        Init(groupParams);

        Params params = groupParams;
        for (uint32_t sliceIdx = 0; sliceIdx < sliceParams.sliceCount; ++sliceIdx) {
            Run(params);

            // Do not submit this slice until every AIC has completed its FixPipe write to GM.
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_BARRIER_FLAG_ID);
            AscendC::CrossCoreWaitFlag<0, PIPE_S>(AIC_BARRIER_FLAG_ID);
            commPolicy.IssueSlice(sliceIdx);

            if (sliceIdx + 1 < sliceParams.sliceCount) {
                params.mmParams.aGmAddr += sliceParams.aByteStride;
                params.mmParams.scaleAGmAddr += sliceParams.scaleAByteStride;
                params.mmParams.cGmAddr += sliceParams.cByteStride;
            }
        }
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(const ProblemShape& shape, uint64_t curBaseM, uint64_t scaleKL1, TensorB& gmB,
                                      TensorScaleB& gmScaleB);
};

template <class ProblemShape, class BlockMmad, class BlockScheduler>
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void MatmulAllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler>::SetL2Cache(
    const ProblemShape& shape, uint64_t curBaseM, uint64_t scaleKL1, TensorB& gmB, TensorScaleB& gmScaleB)
{
    const bool fullMTile = curBaseM >= Te::Get<MNK_M>(shape);
    if (!fullMTile) {
        return;
    }

    const int64_t scaleKRowBytes =
        Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(shape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    const int64_t scaleKL1RowBytes = Blaze::Gemm::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    const bool scaleAlign =
        (scaleKRowBytes & CACHE_LINE_ALIGN_MASK) == 0 && (scaleKL1RowBytes & CACHE_LINE_ALIGN_MASK) == 0;
    gmScaleB.SetL2CacheHint(scaleAlign ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);

    if constexpr (weightNz) {
        gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
    } else {
        const bool bAlign = (Te::Get<MNK_K>(shape) & CACHE_LINE_ALIGN_MASK) == 0;
        gmB.SetL2CacheHint(bAlign ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    }
}

template <class ProblemShape, class BlockMmad, class BlockScheduler>
__aicore__ inline void MatmulAllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler>::Run(const Params& params)
{
    const int64_t rankN = Te::Get<MNK_N>(params.problemShape);
    if (params.rankDim == 0) {
        return;
    }
    const int64_t fullN = rankN * static_cast<int64_t>(params.rankDim);

    auto layoutA = MakeLayoutA{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_K>(params.problemShape));
    auto layoutScaleA = MakeLayoutScaleA{}(
        Te::Get<MNK_M>(params.problemShape),
        Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE);
    auto layoutB = MakeLayoutB{}(Te::Get<MNK_K>(params.problemShape), fullN);
    auto layoutScaleB = MakeLayoutScaleB{}(
        Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE,
        fullN);
    auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), rankN);
    auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, fullN);

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ AType*>(params.mmParams.aGmAddr)), layoutA);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmParams.scaleAGmAddr)),
        layoutScaleA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BType*>(params.mmParams.bGmAddr)), layoutB);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmParams.scaleBGmAddr)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType*>(params.mmParams.biasGmAddr)), layoutBias);

    const int64_t scaleKLen =
        Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    SetL2Cache(params.problemShape, params.qbmmParams.baseM, params.l1Params.scaleKL1, gmB, gmScaleB);
    for (uint32_t rankIdx = 0; rankIdx < params.rankDim; ++rankIdx) {
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);
        if ((scheduler.GetEndBlockIdx() + 1) * params.schedulerParams.mTailTile * params.schedulerParams.nTailTile <=
            AscendC::GetBlockNum()) {
            scheduler.UpdateTailTile(params.schedulerParams.mTailTile, params.schedulerParams.nTailTile);
        }

        auto gmRankC =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ CType*>(params.mmParams.cGmAddr) +
                                                            rankIdx * params.rankStride),
                           layoutC);
        BlockCoord blockIdx;
        int64_t mPos = 0;
        int64_t nPos = 0;
        while (scheduler.GetTileIdx(blockIdx)) {
            BlockShape singleShape =
                scheduler.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(
                    blockIdx);
            if (Te::Get<IDX_M_TILEIDX>(singleShape) <= 0 || Te::Get<IDX_N_TILEIDX>(singleShape) <= 0) {
                return;
            }
            scheduler.GetTileCoord(blockIdx, mPos, nPos);
            const int64_t fullNPos = static_cast<int64_t>(rankIdx) * rankN + nPos;
            auto gmBlockA = gmA.Slice(Te::MakeCoord(mPos, 0L),
                                      Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
            auto gmBlockScaleA =
                gmScaleA.Slice(Te::MakeCoord(mPos, 0L), Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
            auto gmBlockB = gmB.Slice(Te::MakeCoord(0L, fullNPos),
                                      Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
            auto gmBlockScaleB =
                gmScaleB.Slice(Te::MakeCoord(0L, fullNPos), Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));
            auto gmBlockC = gmRankC.Slice(Te::MakeCoord(mPos, nPos),
                                          Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
            auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0L, fullNPos), Te::MakeShape(1L, Get<MNK_N>(singleShape)));
            mmOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape, 0);
        }
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
