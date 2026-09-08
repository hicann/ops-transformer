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
  * \file mx_quant_matmul_all_to_all_urma_impl.h
  * \brief MXQuantMatmul + AllToAll 算子 kernel 实现

  * 核心设计：
  * - AIC 负责计算：委托给 QuantBatchMatmulMxKernel，遍历所有 rank 的 Batch 做 QBMM
  * - AIV 负责通信：通过 AllToAll GET 从其他卡的 Win 区拉回本 rank 的计算结果到 cGM
  */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_all_to_all_tiling_data.h"
#include "adv_api/hcomm/hcomm.h"

#include "../../matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel_tpn.h"
#include "../../../core/aiv_comm/collective_comm_api.h"
#include "../../../core/aiv_comm/collective_comm_context.h"
#include "../../../tiling/comm_tiling_data.h"
#include "../../../core/aiv_comm/barrier/barrier_ubmem.h"

namespace Apace {

using namespace AscendC;
using namespace Blaze::Gemm;
using namespace Apace::AivComm;

template <typename AType, typename BType, typename CType, bool LocalDelay>
class MatmulAllToAllMxImpl {
public:
    __aicore__ inline MatmulAllToAllMxImpl() {}
    __aicore__ inline ~MatmulAllToAllMxImpl() {}

    /**
     * @brief 初始化算子状态和参数
     */
    __aicore__ inline void Init(__gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM,
                                GM_ADDR scaleBGM, GM_ADDR biasGM, GM_ADDR cGM,
                                const MatmulAllToAllTilingData *tilingData);

    /**
     * @brief 执行算子逻辑（包含 AIC/AIV 分离逻辑）
     */
    __aicore__ inline void Run();

    // Layout 定义
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // 组件定义
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using QuantMatmulKernelImpl =
        Blaze::Gemm::Kernel::QuantBatchMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    // ---------------- 通信相关组件与参数 ----------------
    __gm__ CommUbmemContext *barrierCtx_{nullptr};
    __gm__ CommUdmaContext *udmaCtx_{nullptr};
    CollectiveComm<CommCollectiveOp::AllToAll, CommMode::GET, CType, TeamBarrier> allToAll_;
    TeamBarrier teamBarrier_;

    struct BaseParams {
        GM_ADDR selfWinAddr{nullptr};
        GM_ADDR aGm{nullptr};
        GM_ADDR scaleAGm{nullptr};
        GM_ADDR bGm{nullptr};
        GM_ADDR scaleBGm{nullptr};
        GM_ADDR cGm{nullptr};
        GM_ADDR selfCBase{nullptr};
        GM_ADDR biasGm{nullptr};

        uint32_t rankId{0};
        uint32_t rankSize{0};
        uint32_t tileCnt{0};
        uint32_t tileM{0};
        uint32_t tailM{0};
        uint32_t tailCnt{0};
        uint32_t totalTileCnt{0};

        uint64_t axisM{0};
        uint64_t axisK{0};
        uint64_t axisNPerRank{0};
        uint64_t maxTileM{0};

        uint64_t aByteStrideK{0};
        uint64_t cByteStrideN{0};
        uint64_t scaleAByteStride{0};
        uint64_t scaleAElemCnt{0};

        uint64_t tileChunkBytes{0};
        uint64_t bufferSlotBytes{0};
    } baseParams_;

    static constexpr uint16_t MTE1_MTE2_EVENT_FLAG = 6;
    static constexpr uint64_t SIZE_SHIFT = Blaze::Gemm::IsFp4<AType>() ? 1 : 0;
    const MatmulAllToAllTilingData *tilingData_{nullptr};

    // ---------------- 私有方法 ----------------
    __aicore__ inline void InitBaseParams();
    __aicore__ inline void SetupParams(const QuantMatmulTilingData &mmTile, uint32_t batchSize,
                                       uint32_t batchRankOffset, bool isLocal, Params &params);
    __aicore__ inline void RunMatmul();
    __aicore__ inline void RunLocalMatmul();
    __aicore__ inline void RunAllToAll();
};

// =================================================================================
// 公有方法实现
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::Init(
    __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR biasGM,
    GM_ADDR cGM, const MatmulAllToAllTilingData *tilingData)
{
    tilingData_ = tilingData;

    barrierCtx_ = &(hcommCtx->ubmemCtx);
    udmaCtx_ = &(hcommCtx->udmaCtx);

    baseParams_.rankId = udmaCtx_->rankId;
    baseParams_.rankSize = udmaCtx_->rankSize;
    baseParams_.aGm = aGM;
    baseParams_.scaleAGm = scaleAGM;
    baseParams_.bGm = bGM;
    baseParams_.scaleBGm = scaleBGM;
    baseParams_.cGm = cGM;
    baseParams_.biasGm = biasGM;

    InitBaseParams();

    // selfCBase 为 all2all 中不参与通信的归属本 rank 计算结果输出地址；
    // selfWinAddr 为 本 rank win区地址
    baseParams_.selfCBase =
        baseParams_.cGm + static_cast<uint64_t>(baseParams_.rankId) * baseParams_.axisM * baseParams_.cByteStrideN;
    baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]);

    uint32_t ubOffset = 0;
    auto commPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierPtr.Get(), barrierCtx_, udmaCtx_->rankSize, static_cast<uint32_t>(GetBlockIdx()));
    allToAll_.Init(udmaCtx_, teamBarrier_, tilingData->commTilingData, cGM, commPtr.Get(),
                   static_cast<uint32_t>(udmaCtx_->rankSize), static_cast<uint32_t>(GetBlockIdx()));
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::Run()
{
    if ASCEND_IS_AIC {
        RunMatmul();
        if constexpr (LocalDelay) {
            RunLocalMatmul();
        }
    }
    if ASCEND_IS_AIV {
        RunAllToAll();
    }
}

// =================================================================================
// 参数推导与构造
// =================================================================================

// M 轴按通信切分：前 tileCnt 个 tile 尺寸为 tileM（主块），末尾 tailCnt 个为 tailM（尾块），
// axisM = tileCnt * tileM + tailCnt * tailM
template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::InitBaseParams()
{
    const CommTilingData commTilingData = tilingData_->commTilingData;
    baseParams_.axisK = tilingData_->tileQbmmTilingData.k;
    baseParams_.axisNPerRank = tilingData_->tileQbmmTilingData.n;
    baseParams_.tileCnt = static_cast<uint32_t>(commTilingData.splitAxisTileCnt);
    baseParams_.tileM = static_cast<uint32_t>(commTilingData.splitAxisTileSize);
    baseParams_.tailM = static_cast<uint32_t>(commTilingData.splitAxisTailSize);
    baseParams_.tailCnt = static_cast<uint32_t>(commTilingData.splitAxisTailCnt);
    baseParams_.totalTileCnt = baseParams_.tileCnt + baseParams_.tailCnt;
    baseParams_.axisM = static_cast<uint64_t>(baseParams_.tileCnt) * baseParams_.tileM +
                        static_cast<uint64_t>(baseParams_.tailCnt) * baseParams_.tailM;
    baseParams_.maxTileM = (baseParams_.tileM > baseParams_.tailM) ? baseParams_.tileM : baseParams_.tailM;
    baseParams_.cByteStrideN = baseParams_.axisNPerRank * sizeof(CType);
    uint32_t scaleKGroups = (baseParams_.axisK + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1) / Blaze::Gemm::MXFP_DIVISOR_SIZE;
    baseParams_.scaleAElemCnt = scaleKGroups * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    baseParams_.aByteStrideK = (baseParams_.axisK * sizeof(AType)) >> SIZE_SHIFT;
    baseParams_.scaleAByteStride = baseParams_.scaleAElemCnt * sizeof(AscendC::fp8_e8m0_t);
    // Win 区布局：每个 tile 占一个 slot，slot 内按 rank 依次存放各卡结果 chunk，
    // 即 [tile0: rank0..rankN-1][tile1: rank0..rankN-1]...，通信侧 AllToAll 按此布局寻址
    baseParams_.tileChunkBytes = baseParams_.maxTileM * baseParams_.cByteStrideN;
    baseParams_.bufferSlotBytes = baseParams_.rankSize * baseParams_.tileChunkBytes;
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::SetupParams(
    const QuantMatmulTilingData &mmTile, uint32_t batchSize, uint32_t batchRankOffset, bool isLocal, Params &params)
{
    // problemShape 语义为 [M, N, K, Batch]：M 为当前 tiling 的 tile 行数（主/尾/local块不同），
    // N/K 每卡固定，Batch 等于参与计算的 rank 数
    params.problemShape = ProblemShape{static_cast<int64_t>(mmTile.m), static_cast<int64_t>(baseParams_.axisNPerRank),
                                       static_cast<int64_t>(baseParams_.axisK), static_cast<int64_t>(batchSize)};

    GM_ADDR biasAddr = nullptr;
    // B/scaleB/bias 按 rank 分段存储，由batchRankOffset 定位到偏移基地址
    // （非local 场景恒为 0：由 kernel 内 realRank 逐 batch 推进；local 场景为 rankId）
    const bool biasEnabled = baseParams_.biasGm != nullptr;
    if (biasEnabled) {
        biasAddr =
            baseParams_.biasGm + static_cast<uint64_t>(batchRankOffset) * baseParams_.axisNPerRank * sizeof(BiasType);
    }
    params.mmadParams = BlockMmadParams{
        baseParams_.aGm,
        baseParams_.bGm +
            ((static_cast<uint64_t>(batchRankOffset) * baseParams_.axisNPerRank * baseParams_.axisK) >> SIZE_SHIFT),
        isLocal ? baseParams_.selfCBase : baseParams_.selfWinAddr,
        biasAddr,
        baseParams_.scaleAGm,
        baseParams_.scaleBGm +
            static_cast<uint64_t>(batchRankOffset) * baseParams_.axisNPerRank * baseParams_.scaleAElemCnt};
    params.l1Params = L1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    params.schParams = BlockSchedulerParams{mmTile.baseM,
                                            mmTile.baseN,
                                            mmTile.mTailTile,
                                            mmTile.nTailTile,
                                            mmTile.mBaseTailSplitCnt,
                                            mmTile.nBaseTailSplitCnt,
                                            mmTile.mTailMain,
                                            mmTile.nTailMain};
    params.qbmmParams = QBMMTiling{batchSize,
                                   mmTile.baseM,
                                   mmTile.baseN,
                                   mmTile.baseK,
                                   biasEnabled ? QBMMTiling::BIAS_ENABLED : QBMMTiling::BIAS_DISABLED,
                                   mmTile.dbL0c,
                                   LocalDelay && !isLocal};
    // skipSelfBatch 仅在"Local块后置计算 且 非local块"时为 true：
    // 本 rank 的 batch 延后到 RunLocalMatmul 单独算，非local块 遍历时跳过自身 batch
    params.localParams =
        LocalParams{baseParams_.maxTileM, baseParams_.rankId, baseParams_.selfCBase, LocalDelay && !isLocal};
}

// =================================================================================
// 计算逻辑实现 (AIC 侧)
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::RunMatmul()
{
    // LocalDelay=true 时跳过本 rank（batch 数 -1），自身结果延后由 RunLocalMatmul 直写 cGM
    uint32_t batchSize = LocalDelay ? baseParams_.rankSize - 1 : baseParams_.rankSize;
    uint32_t batchRankOffset = 0;
    Params param;
    SetupParams(tilingData_->tileQbmmTilingData, batchSize, batchRankOffset, false, param);

    GM_ADDR aCur = baseParams_.aGm;
    GM_ADDR aScaleCur = baseParams_.scaleAGm;
    GM_ADDR cSelfCur = baseParams_.selfCBase;
    bool switchFlag = false; // 主尾块转换标志
    for (uint32_t tid = 0; tid < baseParams_.totalTileCnt; ++tid) {
        bool isTail = (tid >= baseParams_.tileCnt);
        uint32_t curTileM = isTail ? baseParams_.tailM : baseParams_.tileM;
        if (isTail && !switchFlag) {
            // tiling切换前排空L1
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
            SetupParams(tilingData_->tailQbmmTilingData, batchSize, batchRankOffset, false, param);
            switchFlag = true;
        }

        GM_ADDR cCur = baseParams_.selfWinAddr + tid * baseParams_.bufferSlotBytes;
        param.mmadParams.aGmAddr = aCur;
        param.mmadParams.scaleAGmAddr = aScaleCur;
        param.mmadParams.cGmAddr = cCur;
        if (!LocalDelay) {
            // 本 rank batch 的结果直写 cGM
            param.localParams.cGmSelfAddr = cSelfCur;
        }
        quantMatmulKernelImpl_(param);

        // 通知 AIV：本 tile 全部 batch 已写完 Win 区
        CrossCoreSetFlag<0x2, PIPE_FIX>(tid);

        aCur += curTileM * baseParams_.aByteStrideK;
        aScaleCur += curTileM * baseParams_.scaleAByteStride;
        cSelfCur += curTileM * baseParams_.cByteStrideN;
    }
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::RunLocalMatmul()
{
    // tiling切换前排空L1
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
    Params localParam;
    SetupParams(tilingData_->localQbmmTilingData, 1, baseParams_.rankId, true, localParam);
    quantMatmulKernelImpl_(localParam);
}

// =================================================================================
// 通信实现 (AIV 侧)
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::RunAllToAll()
{
    for (uint32_t tid = 0; tid < baseParams_.totalTileCnt; ++tid) {
        // 与 AIC 的 RunMatmul 逐 tile 同步
        CrossCoreWaitFlag<0x2, PIPE_S>(tid);
        // 每个 AIV 负责拉取一部分远端 rank 的数据
        if (GetBlockIdx() < baseParams_.rankSize) {
            allToAll_.Commit();
            allToAll_.Wait(true);
        }
    }
    if (GetBlockIdx() < baseParams_.rankSize) {
        allToAll_.Finalize();
    }
}

} // namespace Apace
