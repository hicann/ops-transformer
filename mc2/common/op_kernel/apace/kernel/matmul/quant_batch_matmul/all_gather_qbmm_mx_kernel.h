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
 * \file all_gather_qbmm_mx_kernel.h
 * \brief AllGather Prefill 专用 QMM 内核 — 基于 FragmentTensor
 */

#pragma once

#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#define WINDOW_LEN 1L // 调度器窗口设置为1，非侵入式修改
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#undef WINDOW_LEN
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "include/tensor_api/tensor.h"
#include "apace/block/mmad/qmm_mx_block_mmad_fragment.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor_api.h"
#include "apace/tiling/quant_matmul_tiling_data.h"

namespace Apace {

using namespace AscendC;

using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::DNExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

enum RegionTag {
    HEAD,
    MAIN,
    TAIL
};

template <typename AType, typename BType, typename CType>
class AllGatherQbmmMxKernel {
public:
    __aicore__ inline AllGatherQbmmMxKernel() {}
    __aicore__ inline ~AllGatherQbmmMxKernel() {}

    // ---- Component types ----
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, float, LayoutC>;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    // ---- Compile-time constants ----
    static constexpr int64_t kC0Size = 32;               // C0_SIZE_B8 for non-fp4
    static constexpr int64_t kCacheLineAlignMask = 0x7f; // 128B cache line alignment for fp8
    static constexpr int32_t kScaleC0 = 2;

    // ---- Layout factories ----
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<kC0Size>>;
    using MakeLayoutScaleA =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<kScaleC0>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<kC0Size>>;
    using MakeLayoutScaleB =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<kScaleC0>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;

    // ---- Fragment MMAD types ----
    using BlockMmadFragC = Blaze::Gemm::Block::QmmMxBlockMmadFragment<0, false, AType, LayoutA, BType, LayoutB, CType,
                                                                      LayoutC, float, LayoutC>;
    static constexpr bool weightNz = BlockMmadFragC::weightNz;
    using FragL1Params = typename BlockMmadFragC::L1Params;
    using FragBlockShape = typename BlockMmadFragC::BlockShape;

    // ---- FragmentTensor types ----
    using FragTensorA = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutA, AType>;
    using FragScaleA =
        Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutScaleA, AscendC::fp8_e8m0_t>;
    using FragTensorC = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutC, CType>;

    /**
     * @brief Tiling 配置（对标 blaze QBMMTiling）
     */
    struct QBMMTiling {
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t baseK{0};
        uint32_t dbL0C{0};
    };

    /**
     * @brief Fragment 相关参数
     */
    struct FragmentParams {
        uint32_t tileCnt{0};
        uint32_t tileM{0};
        uint32_t tailCnt{0};
        uint32_t tailM{0};
        uint32_t paddedTailM{0};
        uint32_t commTurn{0};
        uint64_t headRows{0};
        uint32_t rankId{0};
        uint32_t rankSize{0};
        uint32_t mPerRank{0}; // M per rank
        uint64_t k{0};
        uint64_t n{0};
        uint64_t scaleKLen{0};
    };

    /**
     * @brief 顶层参数结构
     */
    struct Params {
        const QuantMatmulTilingData *mmTile{nullptr};
        QBMMTiling qbmmParams;
        FragmentParams fragParams;

        // GM pointers
        GM_ADDR aGM{nullptr};
        GM_ADDR aScaleGM{nullptr};
        GM_ADDR bGM{nullptr};
        GM_ADDR bScaleGM{nullptr};
        GM_ADDR cGM{nullptr};
        GM_ADDR biasGM{nullptr};
        bool isBias{false};

        // Win buffer base addresses (computed by communication layer)
        GM_ADDR winDataBase{nullptr};
        GM_ADDR winScaleBase{nullptr};

        // Computed dimensions
        uint64_t dataBytesPerMRow{0};
        uint64_t scaleBytesPerMRow{0};
        uint64_t cBytesPerM{0};
    };

    __aicore__ inline void Run(const Params &params);
    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    __aicore__ inline void Init(const Params &params);
    __aicore__ inline void Process(const Params &params, const ProblemShape &problemShape, BlockScheduler &bs,
                                   BlockMmadFragC &mmadFrag);

    // ---- L2 cache optimization ----
    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(const ProblemShape &problemShape, int64_t baseM, int64_t baseN, TensorB &gmB,
                                      TensorScaleB &gmScaleB);

    // ---- FragmentTensor builders ----
    __aicore__ inline void BuildFragmentTensors(const Params &params);
    __aicore__ inline void EnsureWinRankBasesReady(const Params &params);
    __aicore__ inline void BuildMainFragment(const Params &params, uint32_t roundIdx);
    __aicore__ inline void BuildTailFragment(const Params &params);
    __aicore__ inline void UpdateMainRoundAddrs(const Params &params, uint32_t roundIdx);
    __aicore__ inline Apace::Basic::FragmentParam<2> MakeFragParam(uint64_t fragSize, uint64_t realFragSize,
                                                                   uint32_t fragCnt, uint64_t shape1) const;

    // ---- Tile context resolution ----
    struct TileCtx {
        uint32_t dependTileIdx;
        uint32_t roundIdx;
        RegionTag region;
        int64_t regionMPos;
        const FragTensorA *fragA;
        const FragScaleA *fragScaleA;
        const FragTensorC *fragC;
        uint64_t rankCnt;
    };
    __aicore__ inline TileCtx ResolveTileCtx(int64_t mPos, int64_t headMainRows, int64_t mainRoundRows,
                                             int64_t mainSectionRows, uint32_t rankSize, uint32_t commTurn) const;

    // ---- GM addresses ----
    __gm__ AType *aGmAddr_{nullptr};
    __gm__ AscendC::fp8_e8m0_t *scaleAGmAddr_{nullptr};
    __gm__ BType *bGmAddr_{nullptr};
    __gm__ AscendC::fp8_e8m0_t *scaleBGmAddr_{nullptr};
    __gm__ CType *cGmAddr_{nullptr};
    __gm__ float *biasGmAddr_{nullptr};

    // ---- FragmentTensor state ----
    FragTensorA headFragA_{};
    FragScaleA headFragScaleA_{};
    FragTensorC headFragC_{};
    FragTensorA curMainA_{};
    FragScaleA curMainScaleA_{};
    FragTensorC curMainC_{};
    FragTensorA tailFragA_{};
    FragScaleA tailFragScaleA_{};
    FragTensorC tailFragC_{};

    static constexpr uint32_t MAX_RANK_SIZE = 16U; // 与 tiling rank_size 上限一致
    GM_ADDR cFragAddrs_[MAX_RANK_SIZE]{};
    GM_ADDR winDataRankBase_[MAX_RANK_SIZE]{};
    GM_ADDR winScaleRankBase_[MAX_RANK_SIZE]{};

    static constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;
    GM_ADDR headAddrListA_[MAX_FRAG]{};
    GM_ADDR headAddrListScale_[MAX_FRAG]{};
    GM_ADDR headAddrListC_[MAX_FRAG]{};
    GM_ADDR mainAddrListA_[MAX_FRAG]{};
    GM_ADDR mainAddrListScale_[MAX_FRAG]{};
    GM_ADDR mainAddrListC_[MAX_FRAG]{};
    GM_ADDR tailAddrListA_[MAX_FRAG]{};
    GM_ADDR tailAddrListScale_[MAX_FRAG]{};
    GM_ADDR tailAddrListC_[MAX_FRAG]{};

    uint32_t curMainRoundIdx_{0xFFFFFFFF};
    bool winBasesReady_{false};
    bool tailBuilt_{false};
    uint64_t curRoundDataOff_{};
    uint64_t curRoundScaleOff_{};
    uint64_t curRoundCOff_{};

    // ---- Tile strides ----
    uint64_t tileMDataStride_{};
    uint64_t tileMScaleStride_{};
    uint64_t tileMCStride_{};
};

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType>::Init(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType *>(params.aGM);
    bGmAddr_ = reinterpret_cast<__gm__ BType *>(params.bGM);
    cGmAddr_ = reinterpret_cast<__gm__ CType *>(params.cGM);
    scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(params.aScaleGM);
    scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(params.bScaleGM);
    biasGmAddr_ = reinterpret_cast<__gm__ float *>(params.biasGM);
}

template <typename AType, typename BType, typename CType>
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType>::SetL2Cache(const ProblemShape &problemShape,
                                                                              int64_t baseM, int64_t baseN,
                                                                              TensorB &gmB, TensorScaleB &gmScaleB)
{
    const bool fullMBlock = (baseM >= AscendC::Te::Get<Blaze::Gemm::MNK_M>(problemShape));

    // B (DN layout): K 轴为 leading dim，对齐 128B 时关闭 L2 cache 以 streaming
    if constexpr (weightNz) {
        gmB.SetL2CacheHint(fullMBlock ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                        AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        // DN: transB → fast dim = K
        const bool bAlignForL2Stream = (AscendC::Te::Get<Blaze::Gemm::MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
        gmB.SetL2CacheHint((fullMBlock && bAlignForL2Stream) ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                                               AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    }

    // ScaleB (DN layout): N * scaleC0 bytes 对齐
    const int64_t scaleNStrideBytes = AscendC::Te::Get<Blaze::Gemm::MNK_N>(problemShape) * kScaleC0;
    const int64_t scaleBaseNStrideBytes = baseN * kScaleC0;
    const bool scaleAlignForL2Stream =
        (scaleNStrideBytes & kCacheLineAlignMask) == 0 && (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint((fullMBlock && scaleAlignForL2Stream) ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                                                    AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType>::Run(const Params &params)
{
    Init(params);

    const auto &mmT = *params.mmTile;
    const auto &fp = params.fragParams;
    const int64_t Ki = static_cast<int64_t>(fp.k);
    const int64_t Ni = static_cast<int64_t>(fp.n);
    const int64_t scaleKLen = static_cast<int64_t>(fp.scaleKLen);

    // 总 logical M = head + main 段 + tail。
    bool hasTail = (fp.tailCnt > 0 && fp.tailM > 0);
    const int64_t headMainRows = static_cast<int64_t>(fp.headRows);
    const int64_t mainRoundRows = static_cast<int64_t>(fp.rankSize - 1) * static_cast<int64_t>(fp.tileM);
    const int64_t mainSectionRows = static_cast<int64_t>(fp.tileCnt) * mainRoundRows;
    const int64_t tailRows = hasTail ? static_cast<int64_t>(fp.rankSize) * static_cast<int64_t>(fp.paddedTailM) : 0;
    const int64_t totalLogicalM = headMainRows + mainSectionRows + tailRows;

    // 调度器参数
    BlockSchedulerParams schParams;
    schParams.baseM = mmT.baseM;
    schParams.baseN = mmT.baseN;
    schParams.mTailTile = mmT.mTailTile;
    schParams.nTailTile = mmT.nTailTile;
    schParams.mBaseTailSplitCnt = mmT.mBaseTailSplitCnt;
    schParams.nBaseTailSplitCnt = mmT.nBaseTailSplitCnt;
    schParams.mTailMain = mmT.mTailMain;
    schParams.nTailMain = mmT.nTailMain;

    FragBlockShape l0TileShape{static_cast<int64_t>(mmT.baseM), static_cast<int64_t>(mmT.baseN),
                               static_cast<int64_t>(mmT.baseK), 0L};
    FragL1Params fragL1{static_cast<uint64_t>(mmT.stepK) * mmT.baseK, mmT.scaleKL1, mmT.nBufferNum};

    ProblemShape problemShape{totalLogicalM, Ni, Ki, 1};
    BlockScheduler sch(problemShape, schParams);
    BlockMmadFragC mmadFrag;
    mmadFrag.Init(problemShape, l0TileShape, fragL1, params.isBias, mmT.dbL0c > 1);

    // 延迟构建 fragment tensor（head 先建，main/tail 在 Process 中按需建）
    BuildFragmentTensors(params);

    Process(params, problemShape, sch, mmadFrag);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType>::Process(const Params &params,
                                                                           const ProblemShape &problemShape,
                                                                           BlockScheduler &sch,
                                                                           BlockMmadFragC &mmadFrag)
{
    const auto &mmT = *params.mmTile;
    const auto &fp = params.fragParams;
    const int64_t Ki = static_cast<int64_t>(fp.k);
    const int64_t Ni = static_cast<int64_t>(fp.n);
    const int64_t scaleKLen = static_cast<int64_t>(fp.scaleKLen);

    const int64_t headMainRows = static_cast<int64_t>(fp.headRows);
    const int64_t mainRoundRows = static_cast<int64_t>(fp.rankSize - 1) * static_cast<int64_t>(fp.tileM);
    const int64_t mainSectionRows = static_cast<int64_t>(fp.tileCnt) * mainRoundRows;

    // B / scaleB / bias 全局共享 tensor
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), MakeLayoutB{}(Ki, Ni));
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_), MakeLayoutScaleB{}(scaleKLen, Ni));
    __gm__ float *biasNull = nullptr;
    __gm__ float *biasPtr = params.isBias ? biasGmAddr_ : biasNull;
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasPtr), Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Ni));

    const auto &mTailTile = mmT.mTailTile;
    const auto &nTailTile = mmT.nTailTile;
    // 尾块拆分：需核数(尾块数×M拆分×N拆分)≤总核数才拆分；mTailTile=nTailTile=1时等价no-op。
    if ((sch.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
        sch.UpdateTailTile(mTailTile, nTailTile);
    }
    uint32_t readyTileIdx = 0;
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(0); // dependTileIdx=0 由 AIV 预触发，无需等待。

    Te::Coord<int64_t, int64_t, int64_t, int64_t> blockIdx;
    int64_t mPos = 0L, nPos = 0L;

    while (sch.GetTileIdx(blockIdx)) {
        auto singleShape =
            sch.template GetBlockShape<Blaze::Gemm::QuantMode::MX_PERGROUP_MODE,
                                       Blaze::Gemm::QuantMode::MX_PERGROUP_MODE, BlockMmadFragC::weightNz>(blockIdx);
        int64_t curMtile = Te::Get<0>(singleShape);
        int64_t curNtile = Te::Get<1>(singleShape);
        if (curMtile <= 0 || curNtile <= 0) {
            break;
        }
        sch.GetTileCoord(blockIdx, mPos, nPos);

        auto ctx = ResolveTileCtx(mPos, headMainRows, mainRoundRows, mainSectionRows, fp.rankSize, fp.commTurn);
        while (readyTileIdx < ctx.dependTileIdx) {
            readyTileIdx++;
            CrossCoreWaitFlag<0x2, PIPE_MTE2>(readyTileIdx);
        }

        // Per-tile L2 cache hint
        SetL2Cache(problemShape, curMtile, curNtile, gmB, gmScaleB);

        auto gmBlockB = gmB.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(Ki, curNtile));
        auto gmBlockScaleB = gmScaleB.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(scaleKLen, curNtile));
        auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, curNtile));

        auto coordA = Te::MakeCoord(ctx.regionMPos, 0L);
        auto coordC = Te::MakeCoord(ctx.regionMPos, nPos);
        auto shapeA = Te::MakeShape(curMtile, Ki);
        auto shapeScaleA = Te::MakeShape(curMtile, scaleKLen);
        auto shapeC = Te::MakeShape(curMtile, curNtile);

        // MAIN/TAIL 的 fragment tensor 延迟到首次命中时构建/更新地址。
        if (ctx.region == MAIN) {
            if (curMainRoundIdx_ == 0xFFFFFFFF) {
                BuildMainFragment(params, ctx.roundIdx);
            } else if (curMainRoundIdx_ != ctx.roundIdx) {
                UpdateMainRoundAddrs(params, ctx.roundIdx);
            }
        } else if (ctx.region == TAIL && !tailBuilt_) {
            BuildTailFragment(params);
            tailBuilt_ = true;
        }

        auto blockA = ctx.fragA->Slice(coordA, shapeA);
        auto blockScaleA = ctx.fragScaleA->Slice(coordA, shapeScaleA);
        auto blockC = ctx.fragC->Slice(coordC, shapeC);
        mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, gmBlockBias, cFragAddrs_,
                 static_cast<uint64_t>(fp.mPerRank), static_cast<uint64_t>(fp.tileM), static_cast<uint64_t>(fp.tileCnt),
                 static_cast<uint64_t>(fp.tailM), ctx.rankCnt, Ni, singleShape, ctx.regionMPos, nPos, 0, blockC);
    }

    // 确保所有 dependTileIdx 均已 wait（收尾清理）。
    while (readyTileIdx < fp.commTurn) {
        readyTileIdx++;
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(readyTileIdx);
    }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline Apace::Basic::FragmentParam<2> AllGatherQbmmMxKernel<AType, BType, CType>::MakeFragParam(
    uint64_t fragSize, uint64_t realFragSize, uint32_t fragCnt, uint64_t shape1) const
{
    Apace::Basic::FragmentParam<2> param{};
    param.assembleAxis = 0;
    param.assembledShape[0] = fragSize * fragCnt;
    param.assembledShape[1] = shape1;
    param.fragmentSize = fragSize;
    param.realFragmentSize = realFragSize;
    param.fragmentCnt = fragCnt;
    return param;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType>::BuildFragmentTensors(const Params &params)
{
    const auto &fp = params.fragParams;

    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        cFragAddrs_[r] = params.cGM + static_cast<uint64_t>(r) * fp.mPerRank * params.cBytesPerM;
    }

    tileMDataStride_ = static_cast<uint64_t>(fp.tileM) * params.dataBytesPerMRow;
    tileMScaleStride_ = static_cast<uint64_t>(fp.tileM) * params.scaleBytesPerMRow;
    tileMCStride_ = static_cast<uint64_t>(fp.tileM) * params.cBytesPerM;

    // HEAD：自身数据，单 fragment tensor。
    headAddrListA_[0] = params.aGM;
    headAddrListScale_[0] = params.aScaleGM;
    headAddrListC_[0] = cFragAddrs_[fp.rankId];
    headFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fp.headRows, fp.headRows, 1, fp.k), headAddrListA_);
    headFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
        MakeFragParam(fp.headRows, fp.headRows, 1, fp.scaleKLen), headAddrListScale_);
    headFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fp.headRows, fp.headRows, 1, fp.n), headAddrListC_);
    // main / tail 的 fragment tensor 延迟到首次使用时构建，构建开销被 comm wait 掩盖。
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType>::EnsureWinRankBasesReady(const Params &params)
{
    if (winBasesReady_) {
        return;
    }
    const auto &fp = params.fragParams;
    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        winDataRankBase_[r] = params.winDataBase + static_cast<uint64_t>(r) * fp.mPerRank * params.dataBytesPerMRow;
        winScaleRankBase_[r] = params.winScaleBase + static_cast<uint64_t>(r) * fp.mPerRank * params.scaleBytesPerMRow;
    }
    winBasesReady_ = true;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType>::BuildMainFragment(const Params &params,
                                                                                     uint32_t roundIdx)
{
    EnsureWinRankBasesReady(params);
    const auto &fp = params.fragParams;

    curRoundDataOff_ = static_cast<uint64_t>(roundIdx) * tileMDataStride_;
    curRoundScaleOff_ = static_cast<uint64_t>(roundIdx) * tileMScaleStride_;
    curRoundCOff_ = static_cast<uint64_t>(roundIdx) * tileMCStride_;

    uint32_t fragIdx = 0;
    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        if (r == fp.rankId) {
            continue;
        }
        mainAddrListA_[fragIdx] = winDataRankBase_[r] + curRoundDataOff_;
        mainAddrListScale_[fragIdx] = winScaleRankBase_[r] + curRoundScaleOff_;
        mainAddrListC_[fragIdx] = cFragAddrs_[r] + curRoundCOff_;
        fragIdx++;
    }
    curMainA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fp.tileM, fp.tileM, fp.rankSize - 1, fp.k), mainAddrListA_);
    curMainScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
        MakeFragParam(fp.tileM, fp.tileM, fp.rankSize - 1, fp.scaleKLen), mainAddrListScale_);
    curMainC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fp.tileM, fp.tileM, fp.rankSize - 1, fp.n), mainAddrListC_);
    curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType>::BuildTailFragment(const Params &params)
{
    EnsureWinRankBasesReady(params);
    const auto &fp = params.fragParams;

    uint64_t tailRowOff = fp.headRows;
    uint64_t tailDataOff = tailRowOff * params.dataBytesPerMRow;
    uint64_t tailScaleOff = tailRowOff * params.scaleBytesPerMRow;
    uint64_t tailCOff = tailRowOff * params.cBytesPerM;

    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        if (r == fp.rankId) {
            tailAddrListA_[r] = params.aGM + tailDataOff;
            tailAddrListScale_[r] = params.aScaleGM + tailScaleOff;
        } else {
            tailAddrListA_[r] = winDataRankBase_[r] + tailDataOff;
            tailAddrListScale_[r] = winScaleRankBase_[r] + tailScaleOff;
        }
        tailAddrListC_[r] = cFragAddrs_[r] + tailCOff;
    }
    tailFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fp.paddedTailM, fp.tailM, fp.rankSize, fp.k), tailAddrListA_);
    tailFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
        MakeFragParam(fp.paddedTailM, fp.tailM, fp.rankSize, fp.scaleKLen), tailAddrListScale_);
    tailFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fp.paddedTailM, fp.tailM, fp.rankSize, fp.n), tailAddrListC_);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQbmmMxKernel<AType, BType, CType>::UpdateMainRoundAddrs(const Params &params,
                                                                                        uint32_t roundIdx)
{
    const auto &fp = params.fragParams;
    uint32_t delta = roundIdx - curMainRoundIdx_;
    curRoundDataOff_ += static_cast<uint64_t>(delta) * tileMDataStride_;
    curRoundScaleOff_ += static_cast<uint64_t>(delta) * tileMScaleStride_;
    curRoundCOff_ += static_cast<uint64_t>(delta) * tileMCStride_;

    uint32_t fragIdx = 0;
    for (uint32_t r = 0; r < fp.rankSize; ++r) {
        if (r == fp.rankId) {
            continue;
        }
        mainAddrListA_[fragIdx] = winDataRankBase_[r] + curRoundDataOff_;
        mainAddrListScale_[fragIdx] = winScaleRankBase_[r] + curRoundScaleOff_;
        mainAddrListC_[fragIdx] = cFragAddrs_[r] + curRoundCOff_;
        fragIdx++;
    }
    curMainA_.UpdateAddrList(mainAddrListA_);
    curMainScaleA_.UpdateAddrList(mainAddrListScale_);
    curMainC_.UpdateAddrList(mainAddrListC_);
    curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline typename AllGatherQbmmMxKernel<AType, BType, CType>::TileCtx
AllGatherQbmmMxKernel<AType, BType, CType>::ResolveTileCtx(int64_t mPos, int64_t headMainRows, int64_t mainRoundRows,
                                                           int64_t mainSectionRows, uint32_t rankSize,
                                                           uint32_t commTurn) const
{
    TileCtx ctx{};
    if (mPos < headMainRows) {
        ctx.dependTileIdx = 0;
        ctx.region = HEAD;
        ctx.regionMPos = mPos;
        ctx.fragA = &headFragA_;
        ctx.fragScaleA = &headFragScaleA_;
        ctx.fragC = &headFragC_;
        ctx.rankCnt = static_cast<uint64_t>(rankSize - 1);
    } else if (mPos < headMainRows + mainSectionRows) {
        int64_t relPos = mPos - headMainRows;
        ctx.roundIdx = static_cast<uint32_t>(relPos / mainRoundRows);
        ctx.dependTileIdx = ctx.roundIdx + 1;
        ctx.region = MAIN;
        ctx.regionMPos = relPos % mainRoundRows;
        ctx.fragA = &curMainA_;
        ctx.fragScaleA = &curMainScaleA_;
        ctx.fragC = &curMainC_;
        ctx.rankCnt = static_cast<uint64_t>(rankSize - 1);
    } else {
        ctx.dependTileIdx = commTurn;
        ctx.region = TAIL;
        ctx.regionMPos = mPos - headMainRows - mainSectionRows;
        ctx.fragA = &tailFragA_;
        ctx.fragScaleA = &tailFragScaleA_;
        ctx.fragC = &tailFragC_;
        ctx.rankCnt = static_cast<uint64_t>(rankSize);
    }
    return ctx;
}

} // namespace Apace
