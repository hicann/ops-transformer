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
 * \file qbmm_reduce_scatter_urma_kernel.h
 * \brief MatmulReduceScatter 的 QBMM MX 内核
 *
 * 按 rank 行段组织 A, ScaleA 和 C 的 FragmentTensor, 统一调度矩阵计算.
 * 布局, 矩阵乘和调度组件由融合层选择, 并通过模板参数注入.
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
#include "apace/block/mmad/qmm_mx_block_mmad_fragment.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor_api.h"

namespace Apace {

template <class BlockMmadFrag, class BlockScheduler>
class MatmulReduceScatterQbmmMxKernel {
public:
    __aicore__ inline MatmulReduceScatterQbmmMxKernel() {}
    __aicore__ inline ~MatmulReduceScatterQbmmMxKernel() {}

    // ---- Component types ----
    using AType = typename BlockMmadFrag::AType;
    using BType = typename BlockMmadFrag::BType;
    using CType = typename BlockMmadFrag::CType;
    using BiasType = typename BlockMmadFrag::BiasType;
    using LayoutA = typename BlockMmadFrag::LayoutA;
    using LayoutB = typename BlockMmadFrag::LayoutB;
    using LayoutC = typename BlockMmadFrag::LayoutC;
    using ProblemShape = typename BlockMmadFrag::ProblemShape;
    using FragL1Params = typename BlockMmadFrag::L1Params;
    using BlockMmadParams = typename BlockMmadFrag::Params;
    using BlockShape = typename BlockMmadFrag::BlockShape;
    using ScaleType = typename BlockMmadFrag::ScaleType;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    static constexpr bool transB = BlockMmadFrag::transB;

    // ---- Compile-time constants ----
    static constexpr bool weightNz = BlockMmadFrag::weightNz;
    static constexpr int64_t C0_SIZE = (Blaze::Gemm::IsFp4<AType>() || Blaze::Gemm::IsFp4<BType>()) ?
                                           Blaze::Gemm::C0_SIZE_B4 :
                                           Blaze::Gemm::C0_SIZE_B8;
    static constexpr int64_t kCacheLineAlignMask =
        (Blaze::Gemm::IsFp4<AType>() || Blaze::Gemm::IsFp4<BType>()) ? 0xffUL : 0x7fUL;
    static constexpr int32_t kScaleC0 = 2;
    static constexpr uint32_t DIMS_NUM = 2UL;

    static_assert((sizeof(AType) == 1 && sizeof(BType) == 1) ||
                      (Blaze::Gemm::IsFp4<AType>() && Blaze::Gemm::IsFp4<BType>()),
                  "MatmulReduceScatterQbmmMxKernel supports MXFP8 A/B (1-byte) or MXFP4 E2M1 packed A/B");
    static_assert(sizeof(CType) == 2, "MatmulReduceScatterQbmmMxKernel expects bfloat16 C (2-byte)");

    // ---- Layout factories ----
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<kScaleC0>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<kScaleC0>>,
        asc::te::frame_layout_format<asc::te::scaleb_nd_layout_ptn, AscendC::Std::Int<kScaleC0>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;

    // ---- FragmentTensor types ----
    using FragTensorA = Apace::Basic::FragmentTensor<DIMS_NUM, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutA, AType>;
    using FragScaleA =
        Apace::Basic::FragmentTensor<DIMS_NUM, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutScaleA, ScaleType>;
    using FragTensorC = Apace::Basic::FragmentTensor<DIMS_NUM, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutC, CType>;

    struct FragmentParams {
        uint32_t tileCnt{0};   // 每个 rank 行段沿 M 方向划分的 tile 轮数
        uint32_t tileM{0};     // 每个 rank fragment 的布局行数, 尾 tile 按 16 行对齐
        uint32_t realTileM{0}; // 每个 rank fragment 的有效行数; 为 0 时使用 tileM
        uint32_t rankSize{0};
        uint32_t mPerRank{0}; // 每个 rank 的总 M 行数, 用于定位 rank 段基址
        uint64_t k{0};
        uint64_t n{0};
        uint64_t scaleKLen{0};
    };

    struct QBMMTiling {
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t baseK{0};
        uint32_t dbL0C{0};
    };

    struct Params {
        FragmentParams fragParams;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        FragL1Params l1Params;
        QBMMTiling qbmmParams;

        // 以下三个步长的单位均为字节/行, 分别对应 A, ScaleA, C 的 rank 内行段
        uint64_t dataBytesPerMRow{0};
        uint64_t scaleBytesPerMRow{0};
        uint64_t cBytesPerM{0};
    };

    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Process(const Params& params, const ProblemShape& problemShape, BlockScheduler& sch,
                                   BlockMmadFrag& mmadFrag);

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(const ProblemShape& problemShape, int64_t baseM, int64_t baseN, TensorB& gmB,
                                      TensorScaleB& gmScaleB);

    __aicore__ inline void BuildFragmentTensors(const Params& params);
    __aicore__ inline void UpdateMainRoundAddrs(const Params& params, uint32_t roundIdx);
    // 填写各 rank 分段的 A, ScaleA 和 C 地址
    __aicore__ inline void BuildMainAddrLists(const Params& params, uint64_t dataOff, uint64_t scaleOff, uint64_t cOff);
    __aicore__ inline Apace::Basic::FragmentParam<DIMS_NUM> MakeFragParam(uint64_t fragSize, uint64_t realFragSize,
                                                                          uint32_t fragCnt, uint64_t shape1) const;

    // ---- Tile context resolution ----
    struct TileCtx {
        uint32_t roundIdx;
        int64_t fragMPos; // 当前块在本轮分段张量中的 M 偏移
    };
    __aicore__ inline TileCtx ResolveTileCtx(int64_t mPos, int64_t fragRoundRows) const;

    template <typename GmTensorB, typename GmTensorScaleB, typename GmTensorBias>
    __aicore__ inline void ProcessOneBlock(BlockMmadFrag& mmadFrag, const GmTensorB& gmB,
                                           const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias,
                                           const TileCtx& ctx, const BlockShape& singleShape, int64_t nPos,
                                           int64_t baseM, int64_t baseN, int64_t K, int64_t scaleKLen);

    // ---- GM addresses ----
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ ScaleType* scaleBGmAddr_{nullptr};

    // ---- FragmentTensor state ----
    // FragmentTensor 只保存地址列表指针, 地址数组必须与其保持相同生命周期
    static constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;
    struct FragStreams {
        FragTensorA a{};
        FragScaleA scaleA{};
        FragTensorC c{};
        GM_ADDR addrListA[MAX_FRAG]{};
        GM_ADDR addrListScaleA[MAX_FRAG]{};
        GM_ADDR addrListC[MAX_FRAG]{};
    };
    FragStreams curMainFrag_{};

    static constexpr uint32_t MAX_RANK_SIZE = 16U; // rank 数量上限

    GM_ADDR cFragAddrs_[MAX_RANK_SIZE]{};
    uint32_t curMainRoundIdx_{0}; // curMainFrag_ 当前装载的 round; BuildFragmentTensors 已建立 round 0
    uint64_t curRoundDataOff_{};
    uint64_t curRoundScaleOff_{};
    uint64_t curRoundCOff_{};

    // ---- Tile strides ----
    uint64_t tileMDataStride_{};
    uint64_t tileMScaleStride_{};
    uint64_t tileMCStride_{};
};

template <class BlockMmadFrag, class BlockScheduler>
__aicore__ inline void MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.scaleBGmAddr);
}

template <class BlockMmadFrag, class BlockScheduler>
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::SetL2Cache(
    const ProblemShape& problemShape, int64_t baseM, int64_t baseN, TensorB& gmB, TensorScaleB& gmScaleB)
{
    const bool fullMBlock = (baseM >= asc::te::get<Blaze::Gemm::MNK_M>(problemShape));

    // B 满足整块和缓存行对齐条件时关闭 L2 缓存
    if constexpr (weightNz) {
        gmB.set_l2_cache_hint(fullMBlock ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
    } else {
        if constexpr (transB) {
            // DN 布局的连续维度为 K
            const bool bAlignForL2Stream = (asc::te::get<Blaze::Gemm::MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint((fullMBlock && bAlignForL2Stream) ? asc::te::cache_mode::disable :
                                                                      asc::te::cache_mode::normal);
        } else {
            // ND 布局的连续维度为 N
            const bool bAlignForL2Stream =
                (asc::te::get<Blaze::Gemm::MNK_N>(problemShape) & kCacheLineAlignMask) == 0 &&
                (baseN & kCacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint((fullMBlock && bAlignForL2Stream) ? asc::te::cache_mode::disable :
                                                                      asc::te::cache_mode::normal);
        }
    }

    // 设置 ScaleB 的 L2 缓存策略
    if constexpr (transB) {
        // ScaleB DN 布局的连续维度为 scaleK
        const int64_t scaleKRowBytes = Blaze::Gemm::CeilDiv(asc::te::get<Blaze::Gemm::MNK_K>(problemShape),
                                                            static_cast<int64_t>(Blaze::Gemm::MXFP_DIVISOR_SIZE)) *
                                       kScaleC0 * sizeof(ScaleType);
        const bool scaleAlignForL2Stream = (scaleKRowBytes & kCacheLineAlignMask) == 0;
        gmScaleB.set_l2_cache_hint((fullMBlock && scaleAlignForL2Stream) ? asc::te::cache_mode::disable :
                                                                           asc::te::cache_mode::normal);
    } else {
        // ScaleB ND 布局的连续维度为 N
        const int64_t scaleNStrideBytes = asc::te::get<Blaze::Gemm::MNK_N>(problemShape) * kScaleC0 * sizeof(ScaleType);
        const int64_t scaleBaseNStrideBytes = baseN * kScaleC0 * sizeof(ScaleType);
        const bool scaleAlignForL2Stream =
            (scaleNStrideBytes & kCacheLineAlignMask) == 0 && (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
        gmScaleB.set_l2_cache_hint((fullMBlock && scaleAlignForL2Stream) ? asc::te::cache_mode::disable :
                                                                           asc::te::cache_mode::normal);
    }
}

template <class BlockMmadFrag, class BlockScheduler>
__aicore__ inline void MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::Run(const Params& params)
{
    Init(params);

    const auto& fragParams = params.fragParams;
    const auto& qbmmParams = params.qbmmParams;
    const int64_t K = static_cast<int64_t>(fragParams.k);
    const int64_t N = static_cast<int64_t>(fragParams.n);

    const int64_t fragRoundRows = static_cast<int64_t>(fragParams.rankSize) * static_cast<int64_t>(fragParams.tileM);
    const int64_t totalLogicalM = static_cast<int64_t>(fragParams.tileCnt) * fragRoundRows;

    const BlockShape l0TileShape{static_cast<int64_t>(qbmmParams.baseM), static_cast<int64_t>(qbmmParams.baseN),
                                 static_cast<int64_t>(qbmmParams.baseK), 0L};

    ProblemShape problemShape{totalLogicalM, N, K, 1};
    BlockScheduler sch(problemShape, params.schParams);
    BlockMmadFrag mmadFrag;
    mmadFrag.Init(problemShape, l0TileShape, params.l1Params, false, qbmmParams.dbL0C > 1);

    BuildFragmentTensors(params);

    Process(params, problemShape, sch, mmadFrag);
}

template <class BlockMmadFrag, class BlockScheduler>
__aicore__ inline void MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::Process(
    const Params& params, const ProblemShape& problemShape, BlockScheduler& sch, BlockMmadFrag& mmadFrag)
{
    const auto& fragParams = params.fragParams;
    const int64_t K = static_cast<int64_t>(fragParams.k);
    const int64_t N = static_cast<int64_t>(fragParams.n);
    const int64_t scaleKLen = static_cast<int64_t>(fragParams.scaleKLen);

    const int64_t fragRoundRows = static_cast<int64_t>(fragParams.rankSize) * static_cast<int64_t>(fragParams.tileM);

    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), MakeLayoutB{}(K, N));
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_),
                                         MakeLayoutScaleB{}(scaleKLen, N));
    __gm__ BiasType* biasNull = nullptr;
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasNull),
                                       asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, N));

    const auto& mTailTile = params.schParams.mTailTile;
    const auto& nTailTile = params.schParams.nTailTile;
    // 仅当尾块拆分后的任务数不超过 AIC 核数时启用拆分
    if ((sch.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
        sch.UpdateTailTile(mTailTile, nTailTile);
    }

    BlockCoord blockIdx;
    int64_t mPos = 0L;
    int64_t nPos = 0L;

    while (sch.GetTileIdx(blockIdx)) {
        auto singleShape = sch.template GetBlockShape<Blaze::Gemm::QuantMode::MX_PERGROUP_MODE,
                                                      Blaze::Gemm::QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        int64_t baseM = asc::te::get<0>(singleShape);
        int64_t baseN = asc::te::get<1>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            break;
        }
        sch.GetTileCoord(blockIdx, mPos, nPos);

        auto ctx = ResolveTileCtx(mPos, fragRoundRows);

        SetL2Cache(problemShape, baseM, baseN, gmB, gmScaleB);

        if (curMainRoundIdx_ != ctx.roundIdx) {
            UpdateMainRoundAddrs(params, ctx.roundIdx);
        }

        ProcessOneBlock(mmadFrag, gmB, gmScaleB, gmBias, ctx, singleShape, nPos, baseM, baseN, K, scaleKLen);
    }
}

template <class BlockMmadFrag, class BlockScheduler>
template <typename GmTensorB, typename GmTensorScaleB, typename GmTensorBias>
__aicore__ inline void MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::ProcessOneBlock(
    BlockMmadFrag& mmadFrag, const GmTensorB& gmB, const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias,
    const TileCtx& ctx, const BlockShape& singleShape, int64_t nPos, int64_t baseM, int64_t baseN, int64_t K,
    int64_t scaleKLen)
{
    auto gmBlockB = gmB.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(K, baseN));
    auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(scaleKLen, baseN));
    auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, baseN));

    auto coordA = asc::te::make_coord(ctx.fragMPos, 0L);
    auto coordC = asc::te::make_coord(ctx.fragMPos, nPos);
    auto blockA = curMainFrag_.a.slice(coordA, asc::te::make_shape(baseM, K));
    auto blockScaleA = curMainFrag_.scaleA.slice(coordA, asc::te::make_shape(baseM, scaleKLen));
    auto blockC = curMainFrag_.c.slice(coordC, asc::te::make_shape(baseM, baseN));

    mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, gmBlockBias, blockC, singleShape);
}

template <class BlockMmadFrag, class BlockScheduler>
__aicore__ inline Apace::Basic::FragmentParam<MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::DIMS_NUM>
MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::MakeFragParam(uint64_t fragSize, uint64_t realFragSize,
                                                                              uint32_t fragCnt, uint64_t shape1) const
{
    Apace::Basic::FragmentParam<DIMS_NUM> param{};
    param.assembleAxis = 0;
    param.assembledShape[0] = fragSize * fragCnt;
    param.assembledShape[1] = shape1;
    param.fragmentSize = fragSize;
    param.realFragmentSize = realFragSize;
    param.fragmentCnt = fragCnt;
    return param;
}

template <class BlockMmadFrag, class BlockScheduler>
__aicore__ inline void MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::BuildFragmentTensors(
    const Params& params)
{
    const auto& fragParams = params.fragParams;

    // 每个 rank 行段对应一个 fragment, 拷贝按段处理, 不跨 rank 解释连续地址
    for (uint32_t rankIdx = 0; rankIdx < fragParams.rankSize; ++rankIdx) {
        cFragAddrs_[rankIdx] =
            params.mmadParams.cGmAddr + static_cast<uint64_t>(rankIdx) * fragParams.mPerRank * params.cBytesPerM;
    }

    tileMDataStride_ = static_cast<uint64_t>(fragParams.tileM) * params.dataBytesPerMRow;
    tileMScaleStride_ = static_cast<uint64_t>(fragParams.tileM) * params.scaleBytesPerMRow;
    tileMCStride_ = static_cast<uint64_t>(fragParams.tileM) * params.cBytesPerM;

    const uint64_t realTileM = (fragParams.realTileM != 0) ? fragParams.realTileM : fragParams.tileM;
    BuildMainAddrLists(params, 0, 0, 0);
    curMainFrag_.a = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fragParams.tileM, realTileM, fragParams.rankSize, fragParams.k), curMainFrag_.addrListA);
    curMainFrag_.scaleA = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutScaleA, ScaleType>(
        MakeFragParam(fragParams.tileM, realTileM, fragParams.rankSize, fragParams.scaleKLen),
        curMainFrag_.addrListScaleA);
    curMainFrag_.c = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fragParams.tileM, realTileM, fragParams.rankSize, fragParams.n), curMainFrag_.addrListC);
    curMainRoundIdx_ = 0;
}

template <class BlockMmadFrag, class BlockScheduler>
__aicore__ inline void MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::BuildMainAddrLists(
    const Params& params, uint64_t dataOff, uint64_t scaleOff, uint64_t cOff)
{
    const auto& fragParams = params.fragParams;
    for (uint32_t rankIdx = 0; rankIdx < fragParams.rankSize; ++rankIdx) {
        curMainFrag_.addrListA[rankIdx] =
            params.mmadParams.aGmAddr + static_cast<uint64_t>(rankIdx) * fragParams.mPerRank * params.dataBytesPerMRow +
            dataOff;
        curMainFrag_.addrListScaleA[rankIdx] =
            params.mmadParams.scaleAGmAddr +
            static_cast<uint64_t>(rankIdx) * fragParams.mPerRank * params.scaleBytesPerMRow + scaleOff;
        curMainFrag_.addrListC[rankIdx] = cFragAddrs_[rankIdx] + cOff;
    }
}

template <class BlockMmadFrag, class BlockScheduler>
__aicore__ inline void MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::UpdateMainRoundAddrs(
    const Params& params, uint32_t roundIdx)
{
    // 调度块可跨 round 且访问顺序不保证单调, 因此按 roundIdx 绝对重算偏移, 不使用无符号 delta 累加
    curRoundDataOff_ = static_cast<uint64_t>(roundIdx) * tileMDataStride_;
    curRoundScaleOff_ = static_cast<uint64_t>(roundIdx) * tileMScaleStride_;
    curRoundCOff_ = static_cast<uint64_t>(roundIdx) * tileMCStride_;

    BuildMainAddrLists(params, curRoundDataOff_, curRoundScaleOff_, curRoundCOff_);
    curMainFrag_.a.UpdateAddrList(curMainFrag_.addrListA);
    curMainFrag_.scaleA.UpdateAddrList(curMainFrag_.addrListScaleA);
    curMainFrag_.c.UpdateAddrList(curMainFrag_.addrListC);
    curMainRoundIdx_ = roundIdx;
}

template <class BlockMmadFrag, class BlockScheduler>
__aicore__ inline typename MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::TileCtx
MatmulReduceScatterQbmmMxKernel<BlockMmadFrag, BlockScheduler>::ResolveTileCtx(int64_t mPos,
                                                                               int64_t fragRoundRows) const
{
    TileCtx ctx{};
    ctx.roundIdx = static_cast<uint32_t>(mPos / fragRoundRows);
    ctx.fragMPos = mPos % fragRoundRows;
    return ctx;
}

} // namespace Apace
