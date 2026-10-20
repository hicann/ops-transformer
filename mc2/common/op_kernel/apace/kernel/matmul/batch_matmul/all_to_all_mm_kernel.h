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
 * \file all_to_all_mm_kernel.h
 * \brief 非量化场景 all-to-all MatMul 实现。
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#define WINDOW_LEN 1L // 调度器窗口设置为1，非侵入式修改
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#undef WINDOW_LEN

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define MM_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy, bool enableBufferReuse>
#define MM_KERNEL_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler, CommPolicy, enableBufferReuse

using namespace AscendC;
using asc::te::get;

/**
 * @brief 非量化 all-to-all MatMul 内核实现
 * 该类负责具体的矩阵乘块调度和计算，支持本地(LOCAL)和远程(REMOTE)两种切片模式。
 * 核心机制：
 * 1. 非量化 mmad(block_mmad_matmul_basic.h) 支持 splitKIdx 接口：同一个输出块由 rankSize 次
 *    mmad 调用在 L0C 上跨调用累加，splitKIdx==0 清零 L0C 并加 bias，splitKIdx==splitKNum-1 触发单次 fixpipe。
 * 2. 调度器使用非量化 BlockSchedulerMatmulBasic：按下标 for 循环遍历（blockIdx 步进 coreNums），
 *    GetBlockShape 返回 {m, n, k_, batch_}（K 即每 rank 的 K 段长度），尾块由 Params 静态切分，
 *    不参与尾块切分的核会收到空子片 {0,0,k_,batch_}，kernel 需 continue 跳过。
 * 3. 通信等待逻辑经 CommPolicy 策略类注入（组合模式）：调用点直接 commPolicy_.WaitTile(tileIdx)，
 *    编译期由模板参数绑定具体策略，无需继承与 static_cast。
 */

template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy, bool enableBufferReuse>
class AllToAllMmKernel {
public:
    __aicore__ inline AllToAllMmKernel() {}
    __aicore__ inline ~AllToAllMmKernel() {}

    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ_FORMAT;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;

    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    static constexpr int64_t kCacheLineAlignMask = 0x7f;

    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, Std::Int<asc::te::c0_element<AType>>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, Std::Int<asc::te::c0_element<CType>>>;
    /**
     * @brief 算子模式：REMOTE (远程同步数据后计算), LOCAL (仅本地计算),
     *                 DEFERRED_SYNC (per-tile 本地先算驻留 L0C → wait_flag → 远程累加 → 单次 fixpipe)
     */
    enum class MatmulMode : uint32_t {
        REMOTE = 0,
        LOCAL = 1,
        DEFERRED_SYNC = 2
    };
    /**
     * @brief Tiling 配置
     */
    struct MMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
    };

    /**
     * @brief 本地 Rank 相关参数
     */
    struct LocalParams {
        uint32_t rankId;
        uint32_t rankSize;
        uint64_t originalM;    // 单卡负责的总 M 行数
        GM_ADDR localAGmAddr;  // 本地输入基址
        uint32_t localMatmul;  // 0: remote, 1: DEFERRED_SYNC 语义, 2: GM 累加
        MatmulMode matmulMode; // LOCAL, REMOTE, DEFERRED_SYNC
        uint32_t headTileSize;
    };

    /**
     * @brief 顶层参数结构
     */
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        MMTiling mmTiling;
        LocalParams localParams;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

    __aicore__ inline CommPolicy& GetCommPolicy()
    {
        return commPolicy_;
    }

private:
    __aicore__ inline void ResetGmAddr(const Params& params);
    __aicore__ inline void ProcessLocalBatch(const Params& params, BlockScheduler& bs);
    __aicore__ inline void ProcessDeferredSyncBatch(const Params& params, BlockScheduler& bs);
    __aicore__ inline void ProcessRemoteBatch(const Params& params, BlockScheduler& bs);

    template <typename TensorALocal, typename TensorB, typename TensorBlockBias, typename TensorBlockC>
    __aicore__ inline void MmadSelfLocalRank(const Params& params, int64_t mPos, int64_t nPos, BlockShape& singleShape,
                                             TensorALocal& gmALocal, TensorB& gmB, TensorBlockBias& gmBlockBias,
                                             TensorBlockC& gmBlockC);

    template <typename TensorA, typename TensorB, typename TensorBlockBias, typename TensorBlockC>
    __aicore__ inline void MmadRemoteRanks(const Params& params, int64_t mPos, int64_t nPos, BlockShape& singleShape,
                                           uint64_t winPerTurn, TensorA& gmA, TensorB& gmB,
                                           TensorBlockBias& gmBlockBias, TensorBlockC& gmBlockC);

    __aicore__ inline void WaitDependTiles(const Params& params, int32_t dependTileIdx, int32_t& readyTileIdx);
    __aicore__ inline void DrainWaitTiles(const Params& params, uint32_t totalTiles, int32_t& readyTileIdx);

    __aicore__ inline int32_t CalcDependTileIdx(int64_t mPos, uint32_t headTileSize, uint32_t totalTiles) const;

    template <typename TensorB, typename TensorC>
    __aicore__ inline void SetL2Cache(const ProblemShape& problemShape, uint64_t curBaseM, uint64_t baseN, TensorB& gmB,
                                      TensorC& gmC);

private:
    BlockMmad mmadOp_;
    CommPolicy commPolicy_;

    __gm__ AType* aGmAddr_;      // 远程数据基址（通信缓冲区）
    __gm__ AType* localAGmAddr_; // 本地数据基址
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_; // 输出基址（已根据流水步偏移）
    __gm__ BiasType* biasGmAddr_ = nullptr;

    bool isAtomicAdd_{false}; // GM累加模式（模式3-REMOTE）需要原子累加到 C
};

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);

    if (isAtomicAdd_) {
        AscendC::SetAtomicAdd<CType>(); // 开启原子累加（模式3-REMOTE 单次 fixpipe 累加到 LOCAL 已写的 GM）
    }

    BlockScheduler bs(params.problemShape, params.schParams);

    BlockMmadParams mmadParams = params.mmadParams;

    // GM累加模式（模式3-REMOTE）：bias 已在 LOCAL 阶段写入 GM，必须跳过，否则重复累加导致精度偏差。
    // 把 mmadParams.biasGmAddr 置空，mmad Init 内部据此推导 isBias_=false（指针判断，不再传 isBias 参数）。
    if ((params.localParams.matmulMode == MatmulMode::REMOTE) && (params.localParams.localMatmul == 2)) {
        mmadParams.biasGmAddr = nullptr;
    }
    mmadOp_.Init(mmadParams);

    // 模式分发：判定式与原三分支一致
    //   LOCAL + localMatmul==2 → ProcessLocalBatch（独立 matmul）
    //   DEFERRED_SYNC          → ProcessDeferredSyncBatch（per-tile local 先算驻留 L0C → wait → 远程累加）
    //   其余                    → ProcessRemoteBatch（远程数据 L0C 累加）
    bool localFirst = ((params.localParams.matmulMode == MatmulMode::LOCAL) && (params.localParams.localMatmul == 2));
    bool deferredSync = (params.localParams.matmulMode == MatmulMode::DEFERRED_SYNC);
    if (localFirst) {
        ProcessLocalBatch(params, bs);
    } else if (deferredSync) {
        ProcessDeferredSyncBatch(params, bs);
    } else {
        ProcessRemoteBatch(params, bs);
    }

    if (isAtomicAdd_) {
        AscendC::SetAtomicNone();
    }
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }

    if ((params.localParams.matmulMode == MatmulMode::REMOTE) && (params.localParams.localMatmul == 2)) {
        isAtomicAdd_ = true; // GM累加模式（local已写GM）需要原子累加
        // bias 跳过：由 Run() 里把 mmadParams.biasGmAddr 置空，mmad 内部据此推导 isBias_=false
    }
    ResetGmAddr(params);
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::ResetGmAddr(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    localAGmAddr_ = reinterpret_cast<__gm__ AType*>(params.localParams.localAGmAddr);
    biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline int32_t AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::CalcDependTileIdx(int64_t mPos,
                                                                                         uint32_t headTileSize,
                                                                                         uint32_t totalTiles) const
{
    int32_t tileIdx = static_cast<int32_t>(mPos / headTileSize);
    if (tileIdx >= totalTiles) {
        tileIdx = totalTiles - 1;
    }
    return tileIdx;
}

MM_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorC>
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::SetL2Cache(const ProblemShape& problemShape,
                                                                               uint64_t curBaseM, uint64_t baseN,
                                                                               TensorB& gmB, TensorC& gmC)
{
    if (isAtomicAdd_) {
        gmC.set_l2_cache_hint(asc::te::cache_mode::disable);
    }

    const bool fullMTile = curBaseM >= asc::te::get<MNK_M>(problemShape);
    if (!fullMTile) {
        return;
    }

    if constexpr (weightNz) {
        gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
    } else {
        if constexpr (transB) {
            bool bAlignForL2Stream = (asc::te::get<MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint(bAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        } else {
            bool bAlignForL2Stream =
                (asc::te::get<MNK_N>(problemShape) & kCacheLineAlignMask) == 0 && (baseN & kCacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint(bAlignForL2Stream ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
        }
    }
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::ProcessLocalBatch(const Params& params,
                                                                                      BlockScheduler& bs)
{
    auto rankSize = params.localParams.rankSize;
    auto oriM = params.localParams.originalM;

    auto layoutALocal = MakeLayoutA{}(rankSize * oriM, asc::te::get<MNK_K>(params.problemShape));
    auto layoutB =
        MakeLayoutB{}(rankSize * asc::te::get<MNK_K>(params.problemShape), asc::te::get<MNK_N>(params.problemShape));
    auto layoutBias =
        asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, asc::te::get<MNK_N>(params.problemShape));
    auto layoutC = MakeLayoutC{}(asc::te::get<MNK_M>(params.problemShape), asc::te::get<MNK_N>(params.problemShape));

    auto gmALocal = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(localAGmAddr_), layoutALocal);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);

    SetL2Cache(params.problemShape, params.mmTiling.baseM, params.mmTiling.baseN, gmB, gmC);

    // 获取当前核编号，初始核编号 = 分配块编号，后续按总核数步进
    int64_t curBlockIdx = AscendC::GetBlockIdx();
    // bs.GetCoreNums()：调度器获取需要干活的核数。当前核编号超出实际所需，直接退出
    if (curBlockIdx >= bs.GetCoreNums()) {
        return;
    }
    // 总共启动的核数
    int64_t coreNums = AscendC::GetBlockNum();

    // 遍历当前核的调度任务（按下标步进，空子片 continue 跳过），bs.GetBlockNums()：调度器生成的任务块总数
    for (int64_t blockIdx = curBlockIdx; blockIdx < bs.GetBlockNums(); blockIdx += coreNums) {
        // 非量化调度器返回 {blkM, blkN, k_, batch_}，k_ 维即当前 rank 的 K 段长度 ka
        auto singleShape = bs.template GetBlockShape<transB, BType>(blockIdx);
        // 非量化尾块静态切分，不参与该尾块切分的核，跳过本次迭代
        if ((asc::te::get<IDX_M_TILEIDX>(singleShape) <= 0) || (asc::te::get<IDX_N_TILEIDX>(singleShape) <= 0)) {
            continue;
        }
        // GetBlockCoord 必须在 GetBlockShape 之后调用（mSplitOffset_ 由 GetBlockShape 计算）
        auto blockCoord = bs.GetBlockCoord(blockIdx);
        int64_t mPos = asc::te::get<MNK_M>(blockCoord);
        int64_t nPos = asc::te::get<MNK_N>(blockCoord);
        auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos),
                                  asc::te::make_shape(get<MNK_M>(singleShape), get<MNK_N>(singleShape)));
        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos),
                                        asc::te::make_shape(1L, asc::te::get<IDX_N_TILEIDX>(singleShape)));

        // LOCAL 模式：计算本 Rank 的 A 和本 Rank 的 B 对应部分，splitKIdx=0（独立 matmul）
        MmadSelfLocalRank(params, mPos, nPos, singleShape, gmALocal, gmB, gmBlockBias, gmBlockC);
    }
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::ProcessDeferredSyncBatch(const Params& params,
                                                                                             BlockScheduler& bs)
{
    auto rankSize = params.localParams.rankSize;
    auto oriM = params.localParams.originalM;

    const uint32_t headTileSize = params.localParams.headTileSize;
    const uint32_t totalTiles = (oriM + headTileSize - 1) / headTileSize;
    const uint64_t winPerTurn = static_cast<uint64_t>(rankSize) * headTileSize;
    // comm 通信窗口支持两种布局（由 enableBufferReuse 编译期选择）：
    //   enableBufferReuse=true  → [tile][rank] 布局：每轮窗口 = rankSize*headTileSize 行，
    //                             窗口总 M = totalTiles*rankSize*headTileSize；源 rank 步长 = headTileSize
    //   enableBufferReuse=false → [rank][tile] 布局：窗口总 M = rankSize*oriM；源 rank 步长 = oriM
    const uint64_t winM =
        enableBufferReuse ? static_cast<uint64_t>(totalTiles) * winPerTurn : static_cast<uint64_t>(rankSize) * oriM;

    auto layoutA = MakeLayoutA{}(winM, asc::te::get<MNK_K>(params.problemShape));
    auto layoutALocal = MakeLayoutA{}(rankSize * oriM, asc::te::get<MNK_K>(params.problemShape));
    auto layoutB =
        MakeLayoutB{}(rankSize * asc::te::get<MNK_K>(params.problemShape), asc::te::get<MNK_N>(params.problemShape));
    auto layoutBias =
        asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, asc::te::get<MNK_N>(params.problemShape));
    auto layoutC = MakeLayoutC{}(asc::te::get<MNK_M>(params.problemShape), asc::te::get<MNK_N>(params.problemShape));

    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmALocal = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(localAGmAddr_), layoutALocal);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);

    SetL2Cache(params.problemShape, params.mmTiling.baseM, params.mmTiling.baseN, gmB, gmC);

    int32_t readyTileIdx = -1;
    int64_t curBlockIdx = AscendC::GetBlockIdx();
    if (curBlockIdx >= bs.GetCoreNums()) {
        return;
    }
    int64_t coreNums = AscendC::GetBlockNum();

    for (int64_t blockIdx = curBlockIdx; blockIdx < bs.GetBlockNums(); blockIdx += coreNums) {
        auto singleShape = bs.template GetBlockShape<transB, BType>(blockIdx);
        if ((asc::te::get<IDX_M_TILEIDX>(singleShape) <= 0) || (asc::te::get<IDX_N_TILEIDX>(singleShape) <= 0)) {
            continue;
        }
        auto blockCoord = bs.GetBlockCoord(blockIdx);
        int64_t mPos = asc::te::get<MNK_M>(blockCoord);
        int64_t nPos = asc::te::get<MNK_N>(blockCoord);
        auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos),
                                  asc::te::make_shape(get<MNK_M>(singleShape), get<MNK_N>(singleShape)));
        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos),
                                        asc::te::make_shape(1L, asc::te::get<IDX_N_TILEIDX>(singleShape)));

        // Phase 1: 本 rank 的 local A × 本 rank 的 B 段 → L0C（reset，splitKIdx=0）
        //          此处读 GM 的 localAGmAddr_，不依赖通信，可与 AIV 的 UDMA put 并行。
        MmadSelfLocalRank(params, mPos, nPos, singleShape, gmALocal, gmB, gmBlockBias, gmBlockC);

        // Phase 2: 在 self rank mmad 之后 wait，阻塞后续 shmem 读（去重：同一 tile 只 wait 一次）
        int64_t blockM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        int32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, headTileSize, totalTiles);
        WaitDependTiles(params, dependTileIdx, readyTileIdx);

        // Phase 3: 遍历其它 rank，在 L0C 上累加（最后一个 rank 触发 fixpipe）
        MmadRemoteRanks(params, mPos, nPos, singleShape, winPerTurn, gmA, gmB, gmBlockBias, gmBlockC);
    }

    // 存在尾核没用满核，所以这里要等flag兜底
    DrainWaitTiles(params, totalTiles, readyTileIdx);
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::ProcessRemoteBatch(const Params& params,
                                                                                       BlockScheduler& bs)
{
    auto rankId = params.localParams.rankId;
    auto rankSize = params.localParams.rankSize;
    auto oriM = params.localParams.originalM;

    const uint32_t headTileSize = params.localParams.headTileSize;
    const uint32_t totalTiles = (oriM + headTileSize - 1) / headTileSize;
    const uint64_t winPerTurn = static_cast<uint64_t>(rankSize) * headTileSize;
    const uint64_t winM =
        enableBufferReuse ? static_cast<uint64_t>(totalTiles) * winPerTurn : static_cast<uint64_t>(rankSize) * oriM;
    constexpr int64_t kPos = 0L;

    auto layoutA = MakeLayoutA{}(winM, asc::te::get<MNK_K>(params.problemShape));
    auto layoutALocal = MakeLayoutA{}(rankSize * oriM, asc::te::get<MNK_K>(params.problemShape));
    auto layoutB =
        MakeLayoutB{}(rankSize * asc::te::get<MNK_K>(params.problemShape), asc::te::get<MNK_N>(params.problemShape));
    auto layoutBias =
        asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, asc::te::get<MNK_N>(params.problemShape));
    auto layoutC = MakeLayoutC{}(asc::te::get<MNK_M>(params.problemShape), asc::te::get<MNK_N>(params.problemShape));

    auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
    auto gmALocal = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(localAGmAddr_), layoutALocal);
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
    auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);

    SetL2Cache(params.problemShape, params.mmTiling.baseM, params.mmTiling.baseN, gmB, gmC);

    int32_t readyTileIdx = -1;
    int64_t curBlockIdx = AscendC::GetBlockIdx();
    if (curBlockIdx >= bs.GetCoreNums()) {
        return;
    }
    int64_t coreNums = AscendC::GetBlockNum();

    for (int64_t blockIdx = curBlockIdx; blockIdx < bs.GetBlockNums(); blockIdx += coreNums) {
        auto singleShape = bs.template GetBlockShape<transB, BType>(blockIdx);
        if ((asc::te::get<IDX_M_TILEIDX>(singleShape) <= 0) || (asc::te::get<IDX_N_TILEIDX>(singleShape) <= 0)) {
            continue;
        }
        auto blockCoord = bs.GetBlockCoord(blockIdx);
        int64_t mPos = asc::te::get<MNK_M>(blockCoord);
        int64_t nPos = asc::te::get<MNK_N>(blockCoord);
        auto gmBlockC = gmC.slice(asc::te::make_coord(mPos, nPos),
                                  asc::te::make_shape(get<MNK_M>(singleShape), get<MNK_N>(singleShape)));
        auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos),
                                        asc::te::make_shape(1L, asc::te::get<IDX_N_TILEIDX>(singleShape)));

        // 等待当前 block 依赖的通信 tile 完成（去重：同一 tile 只 wait 一次）
        int64_t blockM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        int32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, headTileSize, totalTiles);
        WaitDependTiles(params, dependTileIdx, readyTileIdx);

        // REMOTE 模式：遍历除本 Rank 外的所有其他卡发送过来的数据，L0C 累加，最后一个 rank 触发 fixpipe
        uint64_t rankMStep = enableBufferReuse ? static_cast<uint64_t>(headTileSize) : static_cast<uint64_t>(oriM);
        uint64_t mPosInLayout =
            enableBufferReuse ? (static_cast<uint64_t>(mPos) % headTileSize) : static_cast<uint64_t>(mPos);
        uint64_t tileBaseOffset = enableBufferReuse ? (static_cast<uint64_t>(mPos / headTileSize) * winPerTurn) : 0;
        uint32_t remoteRankCnt = 0;
        for (uint64_t rank = 0; rank < rankSize; rank++) {
            // 远程 A 按 comm 窗口寻址（布局由 enableBufferReuse 编译期选择）
            uint64_t remoteMPos = tileBaseOffset + rank * rankMStep + mPosInLayout;
            // 从通信buffer上切片
            auto gmBlockA = gmA.slice(asc::te::make_coord(remoteMPos, kPos),
                                      asc::te::make_shape(get<MNK_M>(singleShape), get<MNK_K>(params.problemShape)));

            if (rank == rankId) {
                if (params.localParams.localMatmul == 2) {
                    continue; // GM累加模式：self rank 已在 RunLocalMatmul 计算，REMOTE 阶段跳过
                } else {
                    // self 走本地 A（[rank][oriM] 布局，不经通信窗口）
                    auto selfMPos = rank * oriM + mPos;
                    gmBlockA =
                        gmALocal.slice(asc::te::make_coord(selfMPos, kPos),
                                       asc::te::make_shape(get<MNK_M>(singleShape), get<MNK_K>(params.problemShape)));
                }
            }

            auto gmBlockB = gmB.slice(asc::te::make_coord(rank * get<MNK_K>(params.problemShape), nPos),
                                      asc::te::make_shape(get<MNK_K>(params.problemShape), get<MNK_N>(singleShape)));

            // L0C上累加（splitKIdx==0 时 mmad 内部清零 L0C 并加 bias）
            mmadOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, singleShape, nullptr, remoteRankCnt);
            remoteRankCnt++;
        }
    }

    // 存在尾核没用满核，所以这里要等flag兜底
    DrainWaitTiles(params, totalTiles, readyTileIdx);
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::WaitDependTiles(const Params& params,
                                                                                    int32_t dependTileIdx,
                                                                                    int32_t& readyTileIdx)
{
    while (readyTileIdx < dependTileIdx) {
        readyTileIdx++;
        commPolicy_.WaitTile(readyTileIdx);
    }
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::DrainWaitTiles(const Params& params,
                                                                                   uint32_t totalTiles,
                                                                                   int32_t& readyTileIdx)
{
    while (readyTileIdx < static_cast<int32_t>(totalTiles) - 1) {
        readyTileIdx++;
        commPolicy_.WaitTile(readyTileIdx);
    }
}

MM_KERNEL_CLASS_TEM_PARAMS
template <typename TensorALocal, typename TensorB, typename TensorBlockBias, typename TensorBlockC>
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::MmadSelfLocalRank(
    const Params& params, int64_t mPos, int64_t nPos, BlockShape& singleShape, TensorALocal& gmALocal, TensorB& gmB,
    TensorBlockBias& gmBlockBias, TensorBlockC& gmBlockC)
{
    constexpr int64_t kPos = 0L;
    auto selfMPos = params.localParams.rankId * params.localParams.originalM + mPos;
    auto gmBlockA = gmALocal.slice(asc::te::make_coord(selfMPos, kPos),
                                   asc::te::make_shape(get<MNK_M>(singleShape), get<MNK_K>(params.problemShape)));
    auto gmBlockB = gmB.slice(asc::te::make_coord(params.localParams.rankId * get<MNK_K>(params.problemShape), nPos),
                              asc::te::make_shape(get<MNK_K>(params.problemShape), get<MNK_N>(singleShape)));
    mmadOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, singleShape, nullptr, 0);
}

MM_KERNEL_CLASS_TEM_PARAMS
template <typename TensorA, typename TensorB, typename TensorBlockBias, typename TensorBlockC>
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::MmadRemoteRanks(
    const Params& params, int64_t mPos, int64_t nPos, BlockShape& singleShape, uint64_t winPerTurn, TensorA& gmA,
    TensorB& gmB, TensorBlockBias& gmBlockBias, TensorBlockC& gmBlockC)
{
    auto rankId = params.localParams.rankId;
    auto rankSize = params.localParams.rankSize;
    const uint32_t headTileSize = params.localParams.headTileSize;
    const uint32_t oriM = params.localParams.originalM;
    constexpr int64_t kPos = 0L;

    uint32_t remoteRankCnt = 1;
    // 远程 A 按 comm 窗口寻址（布局由 enableBufferReuse 编译期选择）
    uint64_t rankMStep = enableBufferReuse ? static_cast<uint64_t>(headTileSize) : static_cast<uint64_t>(oriM);
    uint64_t mPosInLayout =
        enableBufferReuse ? (static_cast<uint64_t>(mPos) % headTileSize) : static_cast<uint64_t>(mPos);
    uint64_t tileBaseOffset = enableBufferReuse ? (static_cast<uint64_t>(mPos / headTileSize) * winPerTurn) : 0;
    for (uint64_t rank = 0; rank < rankSize; rank++) {
        if (rank == rankId) {
            continue;
        }
        uint64_t remoteMPos = tileBaseOffset + rank * rankMStep + mPosInLayout;
        auto gmBlockA_remote = gmA.slice(asc::te::make_coord(remoteMPos, kPos),
                                         asc::te::make_shape(get<MNK_M>(singleShape), get<MNK_K>(params.problemShape)));
        auto gmBlockB_r = gmB.slice(asc::te::make_coord(rank * get<MNK_K>(params.problemShape), nPos),
                                    asc::te::make_shape(get<MNK_K>(params.problemShape), get<MNK_N>(singleShape)));
        mmadOp_(gmBlockA_remote, gmBlockB_r, gmBlockBias, gmBlockC, singleShape, nullptr, remoteRankCnt);
        remoteRankCnt++;
    }
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
