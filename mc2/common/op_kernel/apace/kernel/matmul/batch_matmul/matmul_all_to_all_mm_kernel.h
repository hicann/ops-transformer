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
 * \file matmul_all_to_all_mm_kernel.h
 * \brief 非量化 MatmulAlltoAll Apace 计算内核。
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "include/tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

using namespace AscendC;
using asc::te::get;

/**
 * @brief 非量化 all-to-all MatMul 计算内核（先计算后通信方向）
 *
 * 核心机制：
 * 1. rank-major 直写：mmad 按目标 rank 的列段算出结果后直写发送缓冲第 rankIdx 个
 *    rank 区（偏移 rankIdx*rankStride），替代 APT 路径的 AIV 转置段；
 * 2. 无同步无通信：每算完一片即返回，核间屏障与发片由融合编排层持有；
 * 3. problemShape 的 N 是单 rank 列宽而非全宽：外层 for rankIdx 把同一个 [M, rankN]
 *    调度问题跑 rankDim 次，每次取全宽 B 的不同列段（rankIdx*rankN 起始），
 *    计算总量 = M*K*fullN。
 *
 * @tparam ProblemShape   问题规模类型 {M, N, K, batch}
 * @tparam BlockMmad      Blaze 非量化 mmad 块（bias 走 fixpipe）
 * @tparam BlockScheduler Blaze 非量化调度块
 */
template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MatmulAllToAllMmKernel {
public:
    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;

    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockShape = typename BlockScheduler::BlockShape; // {blkM, blkN, K, batch}
    using BlockCoord = typename BlockScheduler::BlockCoord; // {mOffset, nOffset, kOffset, batchIdx}

    // 转置/NZ 静态标志（调度器 GetBlockShape 的对齐分支依赖 transB）
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;
    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ_FORMAT;

    // 帧布局构造器：按 dtype 的 C0 对齐（c0_element = 32B/sizeof(T)）生成帧布局
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, Std::Int<asc::te::c0_element<AType>>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, Std::Int<asc::te::c0_element<CType>>>;

    // 融合编排层逐片推进地址用：x1/发送缓冲的片行首字节偏移与本批片数
    struct SliceParams {
        uint64_t aByteStride{0};
        uint64_t cByteStride{0};
        uint32_t sliceCount{0};
    };

    struct Params {
        ProblemShape problemShape{0, 0, 0, 0}; // {片行数, 单rank列宽, K, 1}
        BlockMmadParams mmParams{}; // GM 四地址 + L1/L0 块形 + 缓冲级数；aGmAddr/cGmAddr 被编排层逐片步进
        BlockSchedulerParams schedulerParams{};
        // rank 区大小（元素数）= 全量 rankM*rankNPerPeer；发送缓冲布局为 [rankDim][rankM][rankNPerPeer]，
        // 片行只在所属 rank 区内连续排布，跨片步进不会越过 rank 区
        uint64_t rankStride{0};
        uint32_t rankDim{0};
        SliceParams sliceParams{};
    };

    // 无通信自测入口：脱离通信编排做单批计算冒烟
    __aicore__ inline void operator()(const Params& params)
    {
        InitGroup(params);
        ProcessSlice(params);
    }

    /**
     * @brief 初始化 mmad 块（每批一次）
     *
     * BlockMmad::Init 只消费块形/缓冲级数等静态字段（不保存 GM 地址，地址经每次
     * 调用的张量传入），因此逐片仅地址变化时无需重新 Init。
     */
    __aicore__ inline void InitGroup(const Params& params)
    {
        if (params.rankDim == 0) {
            return;
        }
        mmOp_.Init(params.mmParams);
    }

    __aicore__ inline void ProcessSlice(const Params& params);

private:
    BlockMmad mmOp_;

    static constexpr int64_t kCacheLineAlignMask = 0x7f; // 128B cache line 对齐掩码

    // B 矩阵 L2 流式缓存提示：仅满 M 块设置（B 被所有 M 块共享读，满块才有驻留价值），128B 对齐时 disable 更快
    template <typename TensorB>
    __aicore__ inline void SetL2Cache(const ProblemShape& shape, uint64_t curBaseM, uint64_t curBaseN, TensorB& gmB)
    {
        const bool fullMTile = curBaseM >= asc::te::get<MNK_M>(shape);
        if (!fullMTile) {
            return;
        }
        if constexpr (transB) {
            // 转置 B（(N,K) 存储）：K 方向按 cache line 对齐判定
            const bool bAlign = (asc::te::get<MNK_K>(shape) & kCacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint(bAlign ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        } else {
            // 非转置 B（(K,N) 存储）：K 步长与块 N 都对齐才 disable
            const bool bAlign =
                (asc::te::get<MNK_N>(shape) & kCacheLineAlignMask) == 0 && (curBaseN & kCacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint(bAlign ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        }
    }
};

/**
 * @brief 单片计算：[taskM, K] x [K, fullN] 逐 rank 循环，结果直写 rank-major 发送缓冲
 *
 * 内存布局契约（地址由编排层 SetupParams 保证）：
 *   gmA      : [taskM, rankK]   本 rank 的 A 切片
 *   gmB      : [rankK, fullN]   全宽 B（转置时 [fullN, rankK]）
 *   gmBias   : [1, fullN]       全宽 bias（可空，mmad 内部判空跳过）
 *   发送缓冲 : [rankDim][rankM][rankNPerPeer]，本片写在 cGmAddr 起始的当前 rank 区内
 */
template <class ProblemShape, class BlockMmad, class BlockScheduler>
__aicore__ inline void MatmulAllToAllMmKernel<ProblemShape, BlockMmad, BlockScheduler>::ProcessSlice(
    const Params& params)
{
    const int64_t rankN = get<MNK_N>(params.problemShape); // 单 rank 列宽
    if (params.rankDim == 0) {
        return;
    }
    const int64_t fullN = rankN * static_cast<int64_t>(params.rankDim); // 全宽 = rankN * rankDim

    auto layoutA = MakeLayoutA{}(get<MNK_M>(params.problemShape), get<MNK_K>(params.problemShape));
    auto layoutB = MakeLayoutB{}(get<MNK_K>(params.problemShape), fullN);
    auto layoutC = MakeLayoutC{}(get<MNK_M>(params.problemShape), rankN);
    auto layoutBias = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, fullN);

    auto gmA = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(params.mmParams.aGmAddr)),
        layoutA);
    auto gmB = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BType*>(params.mmParams.bGmAddr)),
        layoutB);
    auto gmBias = asc::te::make_tensor(
        asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ BiasType*>(params.mmParams.biasGmAddr)),
        layoutBias);

    SetL2Cache(params.problemShape, params.schedulerParams.mL1, params.schedulerParams.nL1, gmB);

    for (uint32_t rankIdx = 0; rankIdx < params.rankDim; ++rankIdx) {
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);

        // 第 rankIdx 个 rank 区首地址（元素偏移 rankIdx * rankStride）
        auto gmRankC = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ CType*>(params.mmParams.cGmAddr) +
                                                         rankIdx * params.rankStride),
            layoutC);

        int64_t curBlockIdx = AscendC::GetBlockIdx();
        if (curBlockIdx >= scheduler.GetCoreNums()) {
            return;
        }
        int64_t coreNums = AscendC::GetBlockNum();

        for (int64_t blockIdx = curBlockIdx; blockIdx < scheduler.GetBlockNums(); blockIdx += coreNums) {
            // 块形 {blkM, blkN, K, batch}；transB/BType 供调度器内部的 N 对齐分支
            auto singleShape = scheduler.template GetBlockShape<transB, BType>(blockIdx);
            if (get<IDX_M_TILEIDX>(singleShape) <= 0 || get<IDX_N_TILEIDX>(singleShape) <= 0) {
                continue; // 尾轮静态切分的空子片
            }
            auto blockCoord = scheduler.GetBlockCoord(blockIdx);
            const int64_t mPos = get<MNK_M>(blockCoord);
            const int64_t nPos = get<MNK_N>(blockCoord);
            const int64_t fullNPos = static_cast<int64_t>(rankIdx) * rankN + nPos; // 全宽 B/bias 列起点

            auto gmBlockA =
                gmA.slice(asc::te::make_coord(mPos, 0L),
                          asc::te::make_shape(get<IDX_M_TILEIDX>(singleShape), get<MNK_K>(params.problemShape)));
            auto gmBlockB =
                gmB.slice(asc::te::make_coord(0L, fullNPos),
                          asc::te::make_shape(get<MNK_K>(params.problemShape), get<IDX_N_TILEIDX>(singleShape)));
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, fullNPos),
                                            asc::te::make_shape(1L, get<IDX_N_TILEIDX>(singleShape)));
            auto gmBlockC =
                gmRankC.slice(asc::te::make_coord(mPos, nPos),
                              asc::te::make_shape(get<IDX_M_TILEIDX>(singleShape), get<IDX_N_TILEIDX>(singleShape)));

            // K 整段入块（mmad 内部 kL1->kL0 两级循环），bias 在 fixpipe 阶段加入，
            // splitKIdx=0 表示单次调用完成全 K 累加并触发一次 fixpipe 搬出
            mmOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, singleShape, nullptr, 0);
        }
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
