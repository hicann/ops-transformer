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
 * \file all_to_all_qbmm_mx_kernel_tpn.h
 * \brief MatmulAllToAll MX 场景 matmul 内核
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"

#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;
using asc::te::get;

/**
 * @brief MX 量化批量矩阵乘kernel
 *
 * 每次 Run(params) 处理一个 tile 的全部 batch 计算：
 *   - 遍历 batchCount 个 batch（每个 batch 对应一个 rank 的 B 段）
 *   - 每个 batch 内通过 BlockScheduler 调度 tile，调用 mmadOp_ 执行计算
 *   - 支持 skipSelfBatch 跳过本 rank batch（LocalDelay 场景）
 *   - 支持 cGmSelfAddr 将本 rank 结果直接写入 cGM
 */
QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
class QuantBatchMatmulMxKernel {
public:
    __aicore__ inline QuantBatchMatmulMxKernel() {}
    __aicore__ inline ~QuantBatchMatmulMxKernel() {}

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using TypeScaleA = ::fp8_e8m0_t;
    using TypeScaleB = ::fp8_e8m0_t;

    static constexpr bool WEIGHT_NZ = BlockMmad::WEIGHT_NZ;
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t CACHE_LINE_ALIGN_MASK = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;
    static constexpr uint64_t SIZE_SHIFT = IsFp4<AType>() ? 1 : 0; // FP4 两元素共 1 字节，元素数转字节数右移 1

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutScaleA = asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, Std::Int<SCALE_C0>>;
    using MakeLayoutScaleB = asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, Std::Int<SCALE_C0>>;

    struct QBMMTiling {
        enum BiasMode : uint32_t {
            BIAS_DISABLED = 0,
            BIAS_ENABLED = 1
        };
        uint32_t batchCount;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        BiasMode isBias;
        uint32_t dbL0c;
        bool disableCL2Cache;
    };

    struct LocalParams {
        uint64_t origM;        // Win 区单个 rank chunk 的 M 行数（主/尾块取较大者）
        uint32_t selfBatchIdx; // 本 rank 对应的 batch 序号
        GM_ADDR cGmSelfAddr;   // 融合模式下本 rank 结果直写 cGM 的目标地址
        bool skipSelfBatch;    // true 时 batch 循环跳过 selfBatchIdx（LocalDelay 的 remote 计算）
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        LocalParams localParams;
    };

    __aicore__ inline void Init(const Params &params);
    __aicore__ inline void Run(const Params &params);
    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ResetGmAddr(const Params &params);
    __aicore__ inline void ProcessSingleBatch(const Params &params, BlockScheduler &bs, uint64_t restBatch,
                                              bool isTailRound);

    template <typename TensorB, typename TensorScaleB, typename TensorC>
    __aicore__ inline void SetL2Cache(const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN,
                                      uint64_t scaleKL1, TensorB &gmB, TensorScaleB &gmScaleB, TensorC &gmC);

    template <typename TensorScaleB>
    __aicore__ inline void SetScaleL2Cache(const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1,
                                           TensorScaleB &gmScaleB);

    BlockMmad mmadOp_;

    __gm__ AType *aGmAddr_{};
    __gm__ BType *bGmAddr_{};
    __gm__ CType *cGmAddr_{};
    __gm__ CType *cGmSelfAddr_{};
    __gm__ BiasType *biasGmAddr_{};
    __gm__ TypeScaleA *scaleAGmAddr_{};
    __gm__ TypeScaleB *scaleBGmAddr_{};

    bool isBias_{false};
    bool needUpdateTail_{false};
    bool disableCL2Cache_{false};
};

// =================================================================================
// 公有方法实现
// =================================================================================

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::Init(const Params &params)
{
    isBias_ = (params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED);
    disableCL2Cache_ = params.qbmmParams.disableCL2Cache;
    needUpdateTail_ = false;
    ResetGmAddr(params);
}

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::Run(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    Init(params);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0c > 1);
    mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);

    BlockScheduler bs(params.problemShape, params.schParams);

    int64_t n = asc::te::get<MNK_N>(params.problemShape);
    int64_t k = asc::te::get<MNK_K>(params.problemShape);
    int64_t scaleKLen = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    uint32_t batchCnt = params.qbmmParams.batchCount;
    uint64_t singleBatchTileCnt = bs.GetTotalCnt();
    uint64_t tailRoundStart = (singleBatchTileCnt * batchCnt / AscendC::GetBlockNum()) * AscendC::GetBlockNum();

    for (uint32_t b = 0; b < batchCnt; ++b) {
        uint64_t curBatch = static_cast<uint64_t>(b) + 1;
        bool isTailRound = curBatch * singleBatchTileCnt > tailRoundStart;
        uint64_t restBatch = batchCnt - curBatch;

        // batch 序号到真实 rank 的映射：skipSelfBatch 时跳过本 rank，
        // 即 b < selfBatchIdx 时 rank=b，否则 rank=b+1
        uint32_t realRank = params.localParams.skipSelfBatch ? ((b < params.localParams.selfBatchIdx) ? b : b + 1) : b;

        ResetGmAddr(params);
        // B/scaleB/bias 按 rank 分段，寻址到起始地址
        bGmAddr_ += (static_cast<uint64_t>(realRank) * n * k) >> SIZE_SHIFT;
        scaleBGmAddr_ += static_cast<uint64_t>(realRank) * n * static_cast<uint64_t>(scaleKLen);
        if (isBias_) {
            biasGmAddr_ += static_cast<uint64_t>(realRank) * n;
        }

        // C 写入去向：本 rank batch 直写 cGmSelfAddr，其余写 Win 区
        if (!params.localParams.skipSelfBatch && b == params.localParams.selfBatchIdx) {
            cGmAddr_ = cGmSelfAddr_;
        } else {
            cGmAddr_ += static_cast<uint64_t>(realRank) * params.localParams.origM * n;
        }

        ProcessSingleBatch(params, bs, restBatch, isTailRound);
    }
}

// =================================================================================
// 私有方法实现
// =================================================================================

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::ResetGmAddr(const Params &params)
{
    aGmAddr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
    cGmSelfAddr_ = reinterpret_cast<__gm__ CType *>(params.localParams.cGmSelfAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ TypeScaleA *>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ TypeScaleB *>(params.mmadParams.scaleBGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType *>(params.mmadParams.biasGmAddr);
    }
}

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::ProcessSingleBatch(
    const Params &params, BlockScheduler &bs, uint64_t restBatch, bool isTailRound)
{
    const auto m = asc::te::get<MNK_M>(params.problemShape);
    const auto n = asc::te::get<MNK_N>(params.problemShape);
    const auto k = asc::te::get<MNK_K>(params.problemShape);
    const auto scaleKLen = Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleKLen);
    auto layoutB = MakeLayoutB{}(k, n);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, n);
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, n);
    auto layoutC = MakeLayoutC{}(m, n);

    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_), layoutScaleA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);

    auto &mTailTile = params.schParams.mTailTile;
    auto &nTailTile = params.schParams.nTailTile;
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    SetL2Cache(params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB,
               gmScaleB, gmC);

    BlockCoord blockIdx;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L;
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, WEIGHT_NZ>(blockIdx);
        const auto baseM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        const auto baseN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            return;
        }

        bs.GetTileCoord(blockIdx, mPos, nPos);
        auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, k));
        auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, scaleKLen));
        auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(k, baseN));
        auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(kPos, nPos), asc::te::make_shape(scaleKLen, baseN));
        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, baseN));
        auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos), asc::te::make_shape(baseM, baseN));
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
    }
    bs.UpdateNextBatchBlockRoundParams();
}

// =================================================================================
// L2 Cache 优化
// =================================================================================

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
template <typename TensorScaleB>
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::SetScaleL2Cache(
    const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB)
{
    if (asc::te::get<MNK_B>(problemShape) != 1) {
        return;
    }
    if constexpr (TRANS_B) {
        const int64_t scaleKRowBytes =
            Blaze::Gemm::CeilDiv(asc::te::get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        const int64_t scaleKL1RowBytes = Blaze::Gemm::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        const bool scaleAlignForL2Stream =
            (scaleKRowBytes & CACHE_LINE_ALIGN_MASK) == 0 && (scaleKL1RowBytes & CACHE_LINE_ALIGN_MASK) == 0;
        gmScaleB.set_l2_cache_hint(scaleAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
    } else {
        const int64_t scaleNStrideBytes = asc::te::get<MNK_N>(problemShape) * MXFP_MULTI_BASE_SIZE;
        const int64_t scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE;
        const bool scaleAlignForL2Stream =
            (scaleNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0 && (scaleBaseNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0;
        gmScaleB.set_l2_cache_hint(scaleAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
    }
}

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB, typename TensorC>
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::SetL2Cache(
    const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB &gmB,
    TensorScaleB &gmScaleB, TensorC &gmC)
{
    if (disableCL2Cache_) {
        gmC.set_l2_cache_hint(asc::te::cache_mode::disable);
    }

    const bool fullMTile = curBaseM >= asc::te::get<MNK_M>(problemShape);
    if (!fullMTile) {
        return;
    }

    SetScaleL2Cache(problemShape, baseN, scaleKL1, gmScaleB);

    if constexpr (WEIGHT_NZ) {
        gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
    } else {
        if constexpr (TRANS_B) {
            bool bAlignForL2Stream = (asc::te::get<MNK_K>(problemShape) & CACHE_LINE_ALIGN_MASK) == 0;
            gmB.set_l2_cache_hint(bAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        } else {
            bool bAlignForL2Stream = (asc::te::get<MNK_N>(problemShape) & CACHE_LINE_ALIGN_MASK) == 0 &&
                                     (baseN & CACHE_LINE_ALIGN_MASK) == 0;
            gmB.set_l2_cache_hint(bAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        }
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
