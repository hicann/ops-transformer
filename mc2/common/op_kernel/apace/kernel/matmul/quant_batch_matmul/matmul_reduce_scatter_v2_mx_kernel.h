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
 * \file matmul_reduce_scatter_v2_mx_kernel.h
 * \brief MatmulReduceScatterV2 的 APACE MXFP8、MXFP4 和 HiFP4 计算实现。
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

// 将各目标 rank 在当前通信片中的行段按逻辑 M 统一调度。
// 计算块跨行段边界时分段计算，并映射回本卡输入的实际行号。
template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MatmulReduceScatterV2QbmmMxKernel {
public:
    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ != 2201 && __NPU_ARCH__ != 3510
    static constexpr bool IS_HIF4 = AscendC::IsSameType<AType, hifloat4x2_t>::value;
    using ScaleType = AscendC::Std::conditional_t<IS_HIF4, AscendC::hif4_scale, AscendC::fp8_e8m0_t>;
#else
    static constexpr bool IS_HIF4 = false;
    using ScaleType = AscendC::fp8_e8m0_t;
#endif
    static_assert(!IS_HIF4 || !transA, "HiFP4 MMRS only supports non-transposed x1");
    static constexpr int64_t C0_SIZE = (IS_HIF4 || IsFp4<AType>()) ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t CACHE_LINE_ALIGN_MASK = (IS_HIF4 || IsFp4<AType>()) ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = IS_HIF4 ? 1 : 2;

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    struct QBMMTiling {
        enum BiasMode : uint32_t {
            BIAS_DISABLED = 0,
            BIAS_ENABLED = 1
        };
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
        uint32_t isBias;
    };

    struct SliceParams {
        uint32_t rankDim;
        uint64_t perRankM;
        uint64_t sliceStart;
        uint64_t sliceM;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmParams;
        L1Params l1Params;
        BlockSchedulerParams schedulerParams;
        QBMMTiling qbmmParams;
        SliceParams sliceParams;
    };

    // 通信分片的循环和跨核同步由融合层统一管理。
    __aicore__ inline void InitGroup(const Params& params)
    {
        const bool isBias = !IS_HIF4 && params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED;
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
        mmOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias, params.qbmmParams.dbL0C > 1, 1);
    }

    __aicore__ inline void ProcessSlice(const Params& params)
    {
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);
        Process(params, scheduler);
    }

private:
    template <typename TensorScaleB>
    __aicore__ inline void SetScaleL2Cache(const ProblemShape& problemShape, uint64_t baseN, uint64_t scaleKL1,
                                           TensorScaleB& gmScaleB)
    {
        if (AscendC::Te::Get<MNK_B>(problemShape) != 1) {
            return;
        }
        if constexpr (transB) {
            const int64_t scaleKRowBytes =
                CeilDiv(AscendC::Te::Get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * SCALE_C0 *
                sizeof(ScaleType);
            const int64_t scaleKL1RowBytes =
                CeilDiv(scaleKL1, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * SCALE_C0 * sizeof(ScaleType);
            const bool aligned =
                (scaleKRowBytes & CACHE_LINE_ALIGN_MASK) == 0 && (scaleKL1RowBytes & CACHE_LINE_ALIGN_MASK) == 0;
            gmScaleB.SetL2CacheHint(aligned ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                              AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            const int64_t scaleNStrideBytes = AscendC::Te::Get<MNK_N>(problemShape) * SCALE_C0 * sizeof(ScaleType);
            const int64_t scaleBaseNStrideBytes = baseN * SCALE_C0 * sizeof(ScaleType);
            const bool aligned = (scaleNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0 &&
                                 (scaleBaseNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0;
            gmScaleB.SetL2CacheHint(aligned ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                              AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(const ProblemShape& problemShape, uint64_t curBaseM, uint64_t baseN,
                                      uint64_t scaleKL1, TensorB& gmB, TensorScaleB& gmScaleB)
    {
        if (curBaseM < static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(problemShape))) {
            return;
        }

        SetScaleL2Cache(problemShape, baseN, scaleKL1, gmScaleB);
        if constexpr (weightNz) {
            gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
        } else if constexpr (transB) {
            const bool aligned = (AscendC::Te::Get<MNK_K>(problemShape) & CACHE_LINE_ALIGN_MASK) == 0;
            gmB.SetL2CacheHint(aligned ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                         AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            const bool aligned = (AscendC::Te::Get<MNK_N>(problemShape) & CACHE_LINE_ALIGN_MASK) == 0 &&
                                 (baseN & CACHE_LINE_ALIGN_MASK) == 0;
            gmB.SetL2CacheHint(aligned ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                         AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }

    __aicore__ inline void Process(const Params& params, BlockScheduler& scheduler)
    {
        using AscendC::Te::Get;
        const auto& sp = params.sliceParams;
        const int64_t logicalM = Get<MNK_M>(params.problemShape);
        const int64_t n = Get<MNK_N>(params.problemShape);
        const int64_t k = Get<MNK_K>(params.problemShape);
        const int64_t totalInputM = static_cast<int64_t>(sp.perRankM * sp.rankDim);
        const int64_t scaleK = CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * SCALE_C0;

        auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                               reinterpret_cast<__gm__ AType*>(params.mmParams.aGmAddr)),
                                           MakeLayoutA{}(totalInputM, k));
        auto gmScaleA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                                    reinterpret_cast<__gm__ ScaleType*>(params.mmParams.scaleAGmAddr)),
                                                MakeLayoutScaleA{}(totalInputM, scaleK));
        auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                               reinterpret_cast<__gm__ BType*>(params.mmParams.bGmAddr)),
                                           MakeLayoutB{}(k, n));
        auto gmScaleB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                                    reinterpret_cast<__gm__ ScaleType*>(params.mmParams.scaleBGmAddr)),
                                                MakeLayoutScaleB{}(scaleK, n));
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                                  reinterpret_cast<__gm__ BiasType*>(params.mmParams.biasGmAddr)),
                                              AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, n));
        auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                               reinterpret_cast<__gm__ CType*>(params.mmParams.cGmAddr)),
                                           MakeLayoutC{}(logicalM, n));

        auto& mTailTile = params.schedulerParams.mTailTile;
        auto& nTailTile = params.schedulerParams.nTailTile;
        if ((scheduler.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
            scheduler.UpdateTailTile(mTailTile, nTailTile);
        }
        SetL2Cache(params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB,
                   gmScaleB);

        BlockCoord blockIdx;
        int64_t mPos = 0;
        int64_t nPos = 0;
        while (scheduler.GetTileIdx(blockIdx)) {
            BlockShape singleShape =
                scheduler.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(
                    blockIdx);
            const int64_t currentM = Get<IDX_M_TILEIDX>(singleShape);
            const int64_t currentN = Get<IDX_N_TILEIDX>(singleShape);
            if (currentM <= 0 || currentN <= 0) {
                return;
            }

            scheduler.GetTileCoord(blockIdx, mPos, nPos);
            // Slice 使用逻辑元素坐标，4 位数据的字节偏移由 Tensor API 换算。
            auto blockB = gmB.Slice(AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(k, currentN));
            int64_t scaleBPos = nPos;
            if constexpr (IS_HIF4 && !transB) {
                // 非转置 x2 的 scale 输入按 [K组, 低/高16位字段, N] 存放。
                // 每列字段占 2 字节，hif4_scale 指针步长为 4 字节，需换算列偏移。
                scaleBPos = nPos * sizeof(uint16_t) / sizeof(ScaleType);
            }
            auto blockScaleB =
                gmScaleB.Slice(AscendC::Te::MakeCoord(0L, scaleBPos), AscendC::Te::MakeShape(scaleK, currentN));
            auto blockBias = gmBias.Slice(AscendC::Te::MakeCoord(0L, nPos), AscendC::Te::MakeShape(1L, currentN));

            int64_t logicalMPos = mPos;
            int64_t remainingM = currentM;
            while (remainingM > 0) {
                const uint64_t rankIdx = static_cast<uint64_t>(logicalMPos) / sp.sliceM;
                const uint64_t rankSliceOffset = static_cast<uint64_t>(logicalMPos) % sp.sliceM;
                const int64_t currentRankM = static_cast<int64_t>(sp.sliceM - rankSliceOffset);
                const int64_t subM = remainingM < currentRankM ? remainingM : currentRankM;
                const int64_t inputMPos = static_cast<int64_t>(rankIdx * sp.perRankM + sp.sliceStart + rankSliceOffset);

                auto blockA = gmA.Slice(AscendC::Te::MakeCoord(inputMPos, 0L), AscendC::Te::MakeShape(subM, k));
                auto blockScaleA =
                    gmScaleA.Slice(AscendC::Te::MakeCoord(inputMPos, 0L), AscendC::Te::MakeShape(subM, scaleK));
                auto blockC =
                    gmC.Slice(AscendC::Te::MakeCoord(logicalMPos, nPos), AscendC::Te::MakeShape(subM, currentN));
                BlockShape subShape{subM, currentN, k, 0};
                mmOp_(blockA, blockB, blockScaleA, blockScaleB, blockBias, blockC, subShape, 0);

                logicalMPos += subM;
                remainingM -= subM;
            }
        }
    }

private:
    BlockMmad mmOp_;
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
