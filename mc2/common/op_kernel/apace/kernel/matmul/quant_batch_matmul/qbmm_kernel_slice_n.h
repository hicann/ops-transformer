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
 * \file qbmm_kernel_slice_n.h
 * \brief MatmulAllToAll MX/HiF4 场景 matmul 内核
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
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
 * @brief MX/HiF4 量化批量矩阵乘kernel
 *
 * 每次 Run(params) 处理一个 tile 的全部 batch 计算：
 *   - 遍历 batchCount 个 batch（每个 batch 对应一个 rank 的 B 段），按 realRank 坐标切片计算
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
    // Scale 元素类型由 BlockMmad 按 AType 键控导出(ops-tensor 公共 MxScaleType):
    // MX 量化 → fp8_e8m0_t, HiF4 量化(hifloat4x2_t) → hif4_scale，A/B 两侧 scale 元素类型对称。
    using ScaleType = typename BlockMmad::ScaleType;
    static_assert(Std::is_same_v<MxScaleType<AType>, MxScaleType<BType>>,
                  "A/B scale element types must be identical (MX/HiF4 symmetric quant)");

    static constexpr bool WEIGHT_NZ = BlockMmad::WEIGHT_NZ;
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t CACHE_LINE_ALIGN_MASK = IsFp4<AType>() ? 0xff : 0x7f;
    // Scale framing C0 由 ops-tensor 公共 GetScaleC0 按数据类型导出(MX/E8M0 → 2, HiF4 → 1)
    static constexpr int32_t SCALE_C0 = static_cast<int32_t>(GetScaleC0<AType>());

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutScaleA = asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, Std::Int<SCALE_C0>>;
    // ScaleB 布局: TRANS_B 为 DN(N, K组), 非转置为 ND(K组, N)，与 B 矩阵布局分支一致
    using MakeLayoutScaleB = asc::te::frame_layout_format<
        Std::conditional_t<TRANS_B, asc::te::scaleb_dn_layout_ptn, asc::te::scaleb_nd_layout_ptn>, Std::Int<SCALE_C0>>;

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
        uint32_t selfBatchIdx; // 需要单独输出到 cGmSelfAddr 地址的 batch 序号，通常是本卡对应的 batch 序号;无 self batch 场景传 batchCount 作哨兵。
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

    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void ProcessSingleBatch(const Params& params, BlockScheduler& bs, uint32_t realRank,
                                              bool isSelfBatch, uint64_t restBatch, bool isTailRound);

    template <typename TensorB, typename TensorScaleB, typename TensorC>
    __aicore__ inline void SetL2Cache(const ProblemShape& problemShape, uint64_t curBaseM, uint64_t baseN,
                                      uint64_t scaleKL1, TensorB& gmB, TensorScaleB& gmScaleB, TensorC& gmC);

    template <typename TensorScaleB>
    __aicore__ inline void SetScaleL2Cache(const ProblemShape& problemShape, uint64_t baseN, uint64_t scaleKL1,
                                           TensorScaleB& gmScaleB);

    BlockMmad mmadOp_;

    __gm__ AType* aGmAddr_{};
    __gm__ BType* bGmAddr_{};
    __gm__ CType* cGmAddr_{};
    __gm__ CType* cGmSelfAddr_{};
    __gm__ BiasType* biasGmAddr_{};
    __gm__ ScaleType* scaleAGmAddr_{};
    __gm__ ScaleType* scaleBGmAddr_{};

    bool isBias_{false};
    bool needUpdateTail_{false};
    bool disableCL2Cache_{false};
};

// =================================================================================
// 公有方法实现
// =================================================================================

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::Init(const Params& params)
{
    isBias_ = (params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED);
    disableCL2Cache_ = params.qbmmParams.disableCL2Cache;
    needUpdateTail_ = false;
    ResetGmAddr(params);
}

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::Run(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    Init(params);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0c > 1);
    mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);

    BlockScheduler bs(params.problemShape, params.schParams);

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
        // C 写入去向：本 rank batch 直写 cGmSelfAddr，其余写 Win 区
        // (selfBatchIdx 传 batchCount 哨兵时 b == selfBatchIdx 恒不成立, 恒走 Win 区分支)
        bool isSelfBatch = (!params.localParams.skipSelfBatch && b == params.localParams.selfBatchIdx);

        ProcessSingleBatch(params, bs, realRank, isSelfBatch, restBatch, isTailRound);
    }
}

// =================================================================================
// 私有方法实现
// =================================================================================

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    cGmSelfAddr_ = reinterpret_cast<__gm__ CType*>(params.localParams.cGmSelfAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.scaleBGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
    }
}

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::ProcessSingleBatch(
    const Params& params, BlockScheduler& bs, uint32_t realRank, bool isSelfBatch, uint64_t restBatch, bool isTailRound)
{
    const auto m = asc::te::get<MNK_M>(params.problemShape);
    const auto n = asc::te::get<MNK_N>(params.problemShape);
    const auto k = asc::te::get<MNK_K>(params.problemShape);
    const auto scaleKLen = static_cast<int64_t>(MxScaleKLen<AType>(static_cast<uint64_t>(k)));

    // 每次重建覆盖 batchCount×n 完整 N 区间的 B/scaleB/bias 视图
    const int64_t fullN =
        n * static_cast<int64_t>(params.qbmmParams.batchCount + (params.localParams.skipSelfBatch ? 1U : 0U));
    auto layoutA = MakeLayoutA{}(m, k);
    auto layoutScaleA = MakeLayoutScaleA{}(m, scaleKLen);
    auto layoutB = MakeLayoutB{}(k, fullN);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, fullN);
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, fullN);
    auto layoutC = MakeLayoutC{}(m, n);

    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmScaleA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleAGmAddr_), layoutScaleA);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
    auto gmCSelf = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmSelfAddr_), layoutC);

    const uint64_t cBatchStride = params.localParams.origM * static_cast<uint64_t>(n);
    auto gmC = isSelfBatch ? gmCSelf :
                             asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                                      cGmAddr_ + static_cast<uint64_t>(realRank) * cBatchStride),
                                                  layoutC);

    auto& mTailTile = params.schParams.mTailTile;
    auto& nTailTile = params.schParams.nTailTile;
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
        // rank 段在 B/scaleB/bias 全量视图中的列坐标
        const int64_t fullNPos = static_cast<int64_t>(realRank) * n + nPos;
        auto gmBlockA = gmA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, k));
        auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(mPos, kPos), asc::te::make_shape(baseM, scaleKLen));
        auto gmBlockB = gmB.slice(asc::te::make_coord(kPos, fullNPos), asc::te::make_shape(k, baseN));
        // HiF4+非转置: scale 沿 N 每列字段占 2B, hif4_scale 指针单位为 4B
        int64_t scaleBColPos = fullNPos;
        if constexpr (IsHif4<AType>() && !TRANS_B) {
            scaleBColPos = fullNPos >> 1;
        }
        auto gmBlockScaleB =
            gmScaleB.slice(asc::te::make_coord(kPos, scaleBColPos), asc::te::make_shape(scaleKLen, baseN));
        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, fullNPos), asc::te::make_shape(1L, baseN));
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
    const ProblemShape& problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB& gmScaleB)
{
    if (asc::te::get<MNK_B>(problemShape) != 1) {
        return;
    }
    // 每64元素的 scale 字节数: MX 为 2(E8M0), HiF4 为 4(hif4_scale), 单元数由公共 MxScaleKLen 按数据类型导出
    if constexpr (TRANS_B) {
        const int64_t scaleKRowBytes =
            static_cast<int64_t>(MxScaleKLen<BType>(static_cast<uint64_t>(asc::te::get<MNK_K>(problemShape)))) *
            static_cast<int64_t>(sizeof(ScaleType));
        const int64_t scaleKL1RowBytes =
            static_cast<int64_t>(MxScaleKLen<BType>(scaleKL1)) * static_cast<int64_t>(sizeof(ScaleType));
        const bool scaleAlignForL2Stream =
            (scaleKRowBytes & CACHE_LINE_ALIGN_MASK) == 0 && (scaleKL1RowBytes & CACHE_LINE_ALIGN_MASK) == 0;
        gmScaleB.set_l2_cache_hint(scaleAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
    } else {
        const int64_t scaleBytesPerDivisor =
            static_cast<int64_t>(MxScaleKLen<BType>(MXFP_DIVISOR_SIZE)) * static_cast<int64_t>(sizeof(ScaleType));
        const int64_t scaleNStrideBytes = asc::te::get<MNK_N>(problemShape) * scaleBytesPerDivisor;
        const int64_t scaleBaseNStrideBytes = static_cast<int64_t>(baseN) * scaleBytesPerDivisor;
        const bool scaleAlignForL2Stream =
            (scaleNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0 && (scaleBaseNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0;
        gmScaleB.set_l2_cache_hint(scaleAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
    }
}

QBMM_MX_KERNEL_TPN_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB, typename TensorC>
__aicore__ inline void QuantBatchMatmulMxKernel<QBMM_MX_KERNEL_TPN_FUNC_TEM_PARAMS>::SetL2Cache(
    const ProblemShape& problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
    TensorScaleB& gmScaleB, TensorC& gmC)
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
