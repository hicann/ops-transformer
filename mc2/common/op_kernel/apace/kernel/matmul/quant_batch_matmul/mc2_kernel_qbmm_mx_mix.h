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
 * \file mc2_kernel_qbmm_mx_mix.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/sync.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class Mc2QbmmMxUniversal {
public:
    __aicore__ inline Mc2QbmmMxUniversal() {}
    __aicore__ inline ~Mc2QbmmMxUniversal() {}

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;
    using EpilogueParams = typename BlockEpilogue::Params;

    struct QBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t biasThreeDim;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
        uint32_t bMustHitL2 = 1U;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        EpilogueParams epilogueParams;
    };

    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_ATOMIC_ADD = BlockMmad::DispatchPolicy::IS_ATOMIC_ADD;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    // 同步旗语沿用 sync.h 约定：正向 AIC->AIV 用 MIX_AIC_SYNC_AIV_FLAG(=0)，
    // 反向 AIV->AIC ack 用 MIX_AIV_SYNC_AIC_FLAG(=1)；INTRA 模式下 id 与 id+16
    // 分别对应 vec0/vec1 两个子核域。

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::scalea_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<SCALE_C0>>,
        asc::te::frame_layout_format<asc::te::scaleb_nd_layout_ptn, AscendC::Std::Int<SCALE_C0>>>;

    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void ProcessSingleBatch(const Params& params, BlockScheduler& bs, uint64_t restBatch,
                                              bool isTailRound);

    __aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs);
    __aicore__ inline void AddBatchOffset(const Params& params, uint64_t aBatchElementStride,
                                          uint64_t bBatchElementStride, uint64_t cBatchStride,
                                          uint64_t scaleABatchStride, uint64_t scaleBBatchStride,
                                          uint64_t biasBatchStride);

    template <typename TensorB>
    __aicore__ inline void SetBL2Cache(const ProblemShape& problemShape, uint64_t currentBasicBlockM,
                                       uint64_t currentBasicBlockN, uint32_t bMustHitL2, TensorB& gmB);

    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;

    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // optional input
    __gm__ AscendC::fp8_e8m0_t* scaleAGmAddr_;
    __gm__ AscendC::fp8_e8m0_t* scaleBGmAddr_;

    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool isBiasThreeDim_{false};
    bool isBias_{false};
    bool isFirstBlock_{true};
    bool needUpdateTail_{false};
};

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void Mc2QbmmMxUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::Run(
    const Params& params)
{
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicAdd<float>();
    }
    const auto& problemShape = params.problemShape;
    const auto& qbmmParams = params.qbmmParams;
    Init(params);
    BlockScheduler bs(problemShape, params.schParams);

    if ASCEND_IS_AIC {
        const BlockShape l0BlockShape{qbmmParams.baseM, qbmmParams.baseN, qbmmParams.baseK, 0};
        mmadOp_.Init(problemShape, l0BlockShape, params.l1Params, isBias_, qbmmParams.dbL0C > 1);
    }

    if ASCEND_IS_AIV {
        epilogueOp_.Init(params.epilogueParams);
    }

    if (asc::te::get<MNK_B>(problemShape) == 1) {
        ProcessSingleBatch(params, bs, 0, true);
        if constexpr (IS_ATOMIC_ADD) {
            AscendC::SetAtomicNone();
        }
        return;
    }

    ProcessWithBatch(params, bs);
    if constexpr (IS_ATOMIC_ADD) {
        AscendC::SetAtomicNone();
    }
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
template <typename TensorB>
__aicore__ inline void Mc2QbmmMxUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::SetBL2Cache(
    const ProblemShape& problemShape, uint64_t currentBasicBlockM, uint64_t currentBasicBlockN, uint32_t bMustHitL2,
    TensorB& gmB)
{
    // 0xff: 256 cache line alignment for FP4 B matrix GM streaming
    // 0x7f: 128 cache line alignment for FP8 B matrix GM streaming
    constexpr uint64_t cacheLineAlignMask = IsFp4<BType>() ? 0xffUL : 0x7fUL;
    const bool isCurrentNAligned = TRANS_B || (currentBasicBlockN & cacheLineAlignMask) == 0UL;
    const bool disableWeightL2 =
        bMustHitL2 == 0U && currentBasicBlockM >= asc::te::get<MNK_M>(problemShape) && isCurrentNAligned;
    gmB.set_l2_cache_hint(disableWeightL2 ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void Mc2QbmmMxUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::Init(
    const Params& params)
{
    const auto& qbmmParams = params.qbmmParams;
    if (qbmmParams.isBias == 1) {
        if (qbmmParams.biasThreeDim == 1) {
            isBiasThreeDim_ = true;
        }
        isBias_ = true;
    }
    ResetGmAddr(params);
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void Mc2QbmmMxUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ResetGmAddr(
    const Params& params)
{
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void Mc2QbmmMxUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ProcessWithBatch(
    const Params& params, BlockScheduler& bs)
{
    const auto& qbmmParams = params.qbmmParams;
    const auto& problemShape = params.problemShape;
    const auto m = asc::te::get<MNK_M>(problemShape);
    const auto n = asc::te::get<MNK_N>(problemShape);
    const auto k = asc::te::get<MNK_K>(problemShape);
    const uint64_t aBatchElementStride = m * k;
    uint64_t bBatchElementStride = 0;
    if constexpr (WEIGHT_NZ) {
        if constexpr (TRANS_B) {
            const auto kBlockCnt = Blaze::Gemm::CeilDiv(k, C0_SIZE);
            const auto nBlockCnt = Blaze::Gemm::CeilDiv(n, static_cast<int64_t>(BLOCK_CUBE));
            bBatchElementStride = kBlockCnt * nBlockCnt * BLOCK_CUBE * C0_SIZE;
        } else {
            const auto nBlockCnt = Blaze::Gemm::CeilDiv(n, C0_SIZE);
            const auto kBlockCnt = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(BLOCK_CUBE));
            bBatchElementStride = nBlockCnt * kBlockCnt * BLOCK_CUBE * C0_SIZE;
        }
    } else {
        bBatchElementStride = n * k;
    }
    const uint64_t cBatchStride = m * n;
    const uint64_t biasBatchStride = isBiasThreeDim_ ? n : 0;
    const uint64_t scaleKLen = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    const uint64_t scaleABatchStride = m * scaleKLen;
    const uint64_t scaleBBatchStride = n * scaleKLen;
    const uint64_t batchC3C4 = static_cast<uint64_t>(qbmmParams.batchC3) * qbmmParams.batchC4;
    const uint64_t batchC2C3C4 = qbmmParams.batchC2 * batchC3C4;
    const uint64_t batchB3B4 = static_cast<uint64_t>(qbmmParams.batchB3) * qbmmParams.batchB4;
    const uint64_t batchB2B3B4 = qbmmParams.batchB2 * batchB3B4;
    const uint64_t batchA3A4 = static_cast<uint64_t>(qbmmParams.batchA3) * qbmmParams.batchA4;
    const uint64_t batchA2A3A4 = qbmmParams.batchA2 * batchA3A4;
    const uint32_t multiA1C1 = qbmmParams.batchA1 / qbmmParams.batchC1;
    const uint32_t multiA2C2 = qbmmParams.batchA2 / qbmmParams.batchC2;
    const uint32_t multiA3C3 = qbmmParams.batchA3 / qbmmParams.batchC3;
    const uint32_t multiA4C4 = qbmmParams.batchA4 / qbmmParams.batchC4;
    const uint32_t multiB1C1 = qbmmParams.batchB1 / qbmmParams.batchC1;
    const uint32_t multiB2C2 = qbmmParams.batchB2 / qbmmParams.batchC2;
    const uint32_t multiB3C3 = qbmmParams.batchB3 / qbmmParams.batchC3;
    const uint32_t multiB4C4 = qbmmParams.batchB4 / qbmmParams.batchC4;

    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t curBatchC = 1UL;
    const uint64_t singleBatchBlockCnt = bs.GetTotalCnt();
    const uint64_t batchCount = asc::te::get<MNK_B>(params.problemShape);
    const uint64_t tailRoundStart =
        (singleBatchBlockCnt * batchCount / AscendC::GetBlockNum()) * AscendC::GetBlockNum();
    for (uint64_t b1Index = 0; b1Index < qbmmParams.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < qbmmParams.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < qbmmParams.batchC3; ++b3Index) {
                batchCOffset_ = batchC3Offset;
                batchAOffset_ = batchA3Offset;
                batchBOffset_ = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < qbmmParams.batchC4; ++b4Index) {
                    const bool isTailRound = curBatchC * singleBatchBlockCnt > tailRoundStart;
                    AddBatchOffset(params, aBatchElementStride, bBatchElementStride, cBatchStride, scaleABatchStride,
                                   scaleBBatchStride, biasBatchStride);
                    ProcessSingleBatch(params, bs, batchCount - curBatchC, isTailRound);
                    curBatchC++;
                    batchCOffset_ += 1;
                    batchAOffset_ += multiA4C4;
                    batchBOffset_ += multiB4C4;
                }
                batchC3Offset += qbmmParams.batchC4;
                batchA3Offset += qbmmParams.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += qbmmParams.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void Mc2QbmmMxUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::AddBatchOffset(
    const Params& params, uint64_t aBatchElementStride, uint64_t bBatchElementStride, uint64_t cBatchStride,
    uint64_t scaleABatchStride, uint64_t scaleBBatchStride, uint64_t biasBatchStride)
{
    ResetGmAddr(params);
    constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
    aGmAddr_ += (batchAOffset_ * aBatchElementStride) >> sizeShift;
    bGmAddr_ += (batchBOffset_ * bBatchElementStride) >> sizeShift;
    if (isBiasThreeDim_) {
        biasGmAddr_ += batchCOffset_ * biasBatchStride;
    }
    scaleAGmAddr_ += batchAOffset_ * scaleABatchStride;
    scaleBGmAddr_ += batchBOffset_ * scaleBBatchStride;
}

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
__aicore__ inline void Mc2QbmmMxUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>::ProcessSingleBatch(
    const Params& params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
{
    const auto& problemShape = params.problemShape;
    const auto m = asc::te::get<MNK_M>(problemShape);
    const auto n = asc::te::get<MNK_N>(problemShape);
    const auto k = asc::te::get<MNK_K>(problemShape);
    const auto scaleKLen = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleKLen);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, n);
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, n);

    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_), layoutScaleA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

    const auto mTailTile = params.schParams.mTailTile;
    const auto nTailTile = params.schParams.nTailTile;
    // both tail of current batch and rest batch are tail round
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    BlockCoord blockCoord;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L; // K is not split, so the K coordinate is 0.
    while (bs.GetTileIdx(blockCoord)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, WEIGHT_NZ>(blockCoord);
        const auto baseM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        const auto baseN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            return;
        }
        SetBL2Cache(problemShape, baseM, baseN, params.qbmmParams.bMustHitL2, gmB);

        bs.GetTileCoord(blockCoord, mPos, nPos);
        if ASCEND_IS_AIC {
            // 反向同步（INTRA 分域）：等两个 AIV 子核各自 ack（flag 1 与 1+16），即上一 tile 的
            // UB FP32 槽已消费完（首 tile 除外）。挂 PIPE_FIX：与 L0C->UB fixpipe 同一 in-order
            // 队列，保证等待先于下一次 UB 写入。
            if (!isFirstBlock_) {
                Sync::WaitForVector(MIX_AIV_SYNC_AIC_FLAG, true);
            }

            auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, k));
            auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, scaleKLen));
            auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k, baseN));
            auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(scaleKLen, baseN));
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, baseN));
            // L0C->UB fixpipe 目的 tensor：DualDst SplitM 要求偶数行、列宽 32 元素对齐；
            // 硬件把 evenRows 行对半写入两个 AIV 子核的 UB 视图（offset 0，行距 ubOutCols）
            const int64_t ubOutRows = (baseM + 1) & ~1;
            const int64_t ubOutCols = (baseN + 31) & ~31;
            auto ubC =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, float>(0),
                                     asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(ubOutRows, ubOutCols));
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, ubC, singleShape);
            // 正向同步（INTRA 分域）：同时 arrive 两个 AIV 子核的域（flag 0 与 0+16）
            Sync::NotifyVector(MIX_AIC_SYNC_AIV_FLAG, true);
            isFirstBlock_ = false;
        }
        if ASCEND_IS_AIV {
            // 正向同步挂 PIPE_V（本子核域的 flag 0）：等待之后消费 UB 的是 V 管线上的 cast
            Sync::WaitForCube();
            epilogueOp_(mPos, nPos, singleShape);
        }
    }
    bs.UpdateNextBatchBlockRoundParams();
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
