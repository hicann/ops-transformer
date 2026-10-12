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
 * \file all_to_all_mx_quant_matmul_hcomm_impl.h
 * \brief AlltoAll MX Quant Matmul — 通信+计算融合实现（纯C核通信）
 *
 * 通信状态由 HcommCommState 容器存储，AllToAllMxQuantMatmulHcommImpl 直接持有；
 * 等待策略由 HcommCommWaitPolicy 承担，通过 state_ 指针引用 HcommCommState，
 * kernel 内逐 tile 通过 bufferSyncMgr_.Acquire() 等待通信完成。
 *
 * Init():
 *   AIC/AIV: 初始化纯 tiling 参数和 GM 地址；输出版本分离原始通信区与重排输出区。
 *   AIC: 在 commState_ 上直接执行 hccl_.InitV2()/SetCcTilingV2()，
 *        并批量下发 AlltoAll<true> 通信任务（scale + dataHead + dataTail），
 *        通过 GetBufferSyncMgr().state_ = &commState_ 将 WaitPolicy 绑定到通信状态。
 *
 * Run():
 *   AIC: local块前置 — 若 localMatmul != 0，先执行 MatmulProcess(LOCAL) 计算本 rank 数据，
 *        再执行 MatmulProcess(REMOTE) 计算通信收到的远端数据，以 local 计算掩盖通信延迟；
 *        最后所有核 Acquire 完成后再统一 Finalize。
 *        kernel 内逐 tile 通过 bufferSyncMgr_.Acquire() → state_->hccl_.Wait(handle) 等待通信完成，
 *        首次 wait 紧挨 data wait 之前执行 hccl->Wait(scaleHandle_)（每核仅一次，掩盖 matmul 头开销）。
 *
 * 输出版本：AIC 逐 tile 通知，AIV 每轮接收并全核同步、重排搬运，每个有界窗口回执一次。
 *
 * \note all2all_out为空时仅启动AIC；all2all_out非空时同时启动AIC和AIV，
 *       其中AIC负责HCCL通信和矩阵计算，AIV负责将通信结果重排并写入all2all_out。
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "basic_api/kernel_operator_block_sync_intf.h"
#include "../../../core/aiv_compute/all_to_all_permute.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "../../matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel.h"
#include "include/tensor_api/tensor.h"
#include "../../../utils/op_state_dump.h"

namespace Apace {
static constexpr AscendC::HcclServerConfig HCOMM_CONFIG = {AscendC::CoreType::ON_AIC, 0};
// Separate from SyncAll's reserved flags 11-14 and the matmul API's flags 0-7.
static constexpr uint16_t HCOMM_DATA_READY_FLAG = 9;
static constexpr uint16_t HCOMM_WINDOW_ACK_FLAG = 10;
static constexpr uint32_t HCOMM_NOTIFY_WINDOW = 8;

template <typename T>
constexpr uint64_t GetX1HcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_UINT8;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

template <AscendC::HcclServerType ServerType>
struct HcommCommState {
    AscendC::Hccl<ServerType, HCOMM_CONFIG> hccl_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHeadHandle_{0};
    AscendC::HcclHandle dataTailHandle_{0};
    uint32_t headTileCnt_{0};
    uint32_t totalTileCnt_{0};
    uint32_t nextNotifyTile_{0};
    uint32_t acknowledgedTiles_{0};
};

template <AscendC::HcclServerType ServerType, bool EnableHcommAllToAllOut>
struct HcommCommWaitPolicy {
    HcommCommState<ServerType>* state_{nullptr};
    uint32_t slotNum{0};

    __aicore__ inline void Release(uint32_t) {}

    __aicore__ inline void Acquire(uint32_t tileIdx)
    {
        if constexpr (!EnableHcommAllToAllOut) {
            // Keep legacy Wait behavior, including its original error handling.
            if (tileIdx == 0) {
                state_->hccl_.Wait(state_->scaleHandle_);
            }
            if (tileIdx < state_->headTileCnt_) {
                state_->hccl_.Wait(state_->dataHeadHandle_);
            } else {
                state_->hccl_.Wait(state_->dataTailHandle_);
            }
        } else {
            while (state_->nextNotifyTile_ <= tileIdx) {
                if (state_->nextNotifyTile_ - state_->acknowledgedTiles_ == HCOMM_NOTIFY_WINDOW) {
                    // A scalar wait prevents the sender loop from running ahead of the paired AIVs.
                    AscendC::CrossCoreWaitFlag(HCOMM_WINDOW_ACK_FLAG);
                    state_->acknowledgedTiles_ = state_->nextNotifyTile_;
                }
                const uint32_t nextTile = state_->nextNotifyTile_;
                if (nextTile == 0) {
                    state_->hccl_.Wait(state_->scaleHandle_);
                }
                const auto handle = nextTile < state_->headTileCnt_ ? state_->dataHeadHandle_ : state_->dataTailHandle_;
                state_->hccl_.Wait(handle);
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(HCOMM_DATA_READY_FLAG);
                ++state_->nextNotifyTile_;
            }
        }
    }

    __aicore__ inline void FinishNotifications()
    {
        if constexpr (EnableHcommAllToAllOut) {
            // Covers idle cores and an early return from the matmul scheduler. Never re-wait a completed tile.
            if (state_->nextNotifyTile_ < state_->totalTileCnt_) {
                Acquire(state_->totalTileCnt_ - 1);
            }
            if (state_->acknowledgedTiles_ < state_->nextNotifyTile_) {
                AscendC::CrossCoreWaitFlag(HCOMM_WINDOW_ACK_FLAG);
                state_->acknowledgedTiles_ = state_->nextNotifyTile_;
            }
        }
    }
};

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool EnableHcommAllToAllOut>
class AllToAllMxQuantMatmulHcommImpl {
public:
    explicit __aicore__ inline AllToAllMxQuantMatmulHcommImpl(AlltoAllMatmulTilingDataType* tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale,
                                GM_ADDR x2_scale, GM_ADDR workspaceGM, AscendC::TPipe* pipe);
    __aicore__ inline void Run();

    // Layout 定义
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::dn_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using BiasType = float;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    // 组件定义
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC,
                                                    BiasType, LayoutBias>;
    using QuantMatmulKernelImpl =
        Blaze::Gemm::Kernel::AllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler,
                                                  HcommCommWaitPolicy<ServerType, EnableHcommAllToAllOut>, false>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;
    HcommCommState<ServerType> commState_;

private:
    static constexpr uint8_t POS_COMM_BEFORE = 0U;
    static constexpr uint8_t POS_COMP_LOCAL_BEFORE = 1U;
    static constexpr uint8_t POS_COMP_REMOTE_BEFORE = 2U;
    static constexpr uint8_t POS_FINALIZE_BEFORE = 3U;
    AlltoAllMatmulTilingDataType* tilingData_;

    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR x1Scale_;
    GM_ADDR x2Scale_;
    GM_ADDR workspaceGM_;
    GM_ADDR commX1ScaleGM1_;
    GM_ADDR commOutGM_;
    GM_ADDR permuteOutGM_{nullptr};
    AscendC::TPipe* pipe_{nullptr};

    static constexpr uint64_t mxScaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;
    static constexpr uint64_t x1HcclDataType_ = GetX1HcclDataType<X1Type>();
    uint32_t rankId_{0};
    uint32_t rankDim_{0};
    uint64_t splitAxisSize_{0};
    bool isBias_{false};

    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;

    Mc2Kernel::OpStateDump opStateDump_;

    __aicore__ inline void SetupParams(Params& out, MatmulMode matmulMode);
    __aicore__ inline void MatmulProcess(MatmulMode matmulMode);
    __aicore__ inline void WaitOutputTiles();
};

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool EnableHcommAllToAllOut>
__aicore__ inline void
AllToAllMxQuantMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType, ServerType, IsMxFp4,
                               EnableHcommAllToAllOut>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
                                                             GM_ADDR all2all_out, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                                             GM_ADDR workspaceGM, AscendC::TPipe* pipe)
{
    pipe_ = pipe;
    auto&& commTiling = tilingData_->commTilingData;
    splitAxisSize_ = commTiling.splitAxisTileSize * commTiling.splitAxisTileCnt +
                     commTiling.splitAxisTailSize * commTiling.splitAxisTailCnt;
    rankDim_ = tilingData_->rankDim;
    if constexpr (EnableHcommAllToAllOut) {
        commState_.totalTileCnt_ = static_cast<uint32_t>(commTiling.splitAxisTileCnt + commTiling.splitAxisTailCnt);
        commState_.nextNotifyTile_ = 0;
        commState_.acknowledgedTiles_ = 0;
    }
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    isBias_ = (bias != nullptr);
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;
    // The generated arch35 entry already passes user workspace, after the system reservation.
    commX1ScaleGM1_ = workspaceGM;
    // Host validates per-rank K % 64 == 0, so grouping total K gives the same scale length
    // as summing per-rank scale counts. Keep the existing 512-byte-aligned layout.
    uint64_t x1ScaleLen = Blaze::Gemm::CeilDiv(commTiling.nonSplitAxisSize, MXFP_GROUP_SIZE) * MXFP_MULTI_BASE_SIZE *
                          static_cast<uint64_t>(splitAxisSize_) * sizeof(AscendC::fp8_e8m0_t);
    x1ScaleLen = Blaze::Gemm::CeilDiv(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM;
    if constexpr (EnableHcommAllToAllOut) {
        // AIC keeps reading raw communication data while AIV writes a separate output.
        commOutGM_ = workspaceGM + x1ScaleLen;
        permuteOutGM_ = all2all_out;
    } else {
        commOutGM_ = all2all_out;
        if (all2all_out == nullptr) {
            commOutGM_ = workspaceGM + x1ScaleLen;
        }
        permuteOutGM_ = nullptr;
    }
    if ASCEND_IS_AIC {
#if MC2_DFX_ENABLE
        opStateDump_.Init(workspaceGM_, &tilingData_->dumpInfo.workspaceLayout,
                          tilingData_->tileQbmmTilingData.usedCoreNum);
#endif
        quantMatmulKernelImpl_.GetBufferSyncMgr().state_ = &commState_;
        quantMatmulKernelImpl_.GetBufferSyncMgr().slotNum =
            static_cast<uint32_t>(commTiling.splitAxisTileCnt + commTiling.splitAxisTailCnt);
        commState_.hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
        commState_.hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(AlltoAllMatmulTilingDataType, mc2CcTiling)));
        rankId_ = commState_.hccl_.GetRankId();
        rankDim_ = commState_.hccl_.GetRankDim();

        uint64_t originalRankForComm = commTiling.nonSplitAxisSize / rankDim_;
        uint64_t rankForComm = originalRankForComm;
        if constexpr (IsMxFp4) {
            rankForComm = Blaze::Gemm::CeilDiv(rankForComm, MXFP_DATA_NUM_PER_BYTE);
        }
        uint32_t scaleKPerRank =
            static_cast<uint32_t>(Blaze::Gemm::CeilDiv(originalRankForComm, MXFP_GROUP_SIZE) * MXFP_MULTI_BASE_SIZE);

        uint64_t dataStrideCount = static_cast<uint64_t>(splitAxisSize_) * rankForComm;
        uint64_t scaleSendCount = static_cast<uint64_t>(splitAxisSize_) * scaleKPerRank;
        opStateDump_.DoDump(DUMP_FIELD_STEP, POS_COMM_BEFORE, static_cast<uint8_t>(Utils::RT_PHASE_COMM_PREPARE));
        // DoDump does not count the scale commit.
        commState_.scaleHandle_ = commState_.hccl_.template AlltoAll<true>(
            x1Scale_, commX1ScaleGM1_, scaleSendCount, static_cast<AscendC::HcclDataType>(mxScaleHcclDataType_),
            scaleSendCount, 1);

        if (commTiling.splitAxisTileCnt > 0) {
            uint64_t headSendCount = static_cast<uint64_t>(commTiling.splitAxisTileSize) * rankForComm;
            commState_.dataHeadHandle_ = commState_.hccl_.template AlltoAll<true>(
                x1_, commOutGM_, headSendCount, static_cast<AscendC::HcclDataType>(x1HcclDataType_), dataStrideCount,
                static_cast<uint8_t>(commTiling.splitAxisTileCnt));
            opStateDump_.DoDump(DUMP_FIELD_COMMIT, 0, 0, 0, commTiling.splitAxisTileCnt);
        }

        if (commTiling.splitAxisTailCnt > 0) {
            uint64_t headOffset = static_cast<uint64_t>(commTiling.splitAxisTileCnt) * commTiling.splitAxisTileSize *
                                  rankForComm * sizeof(X1Type);
            uint64_t tailSendCount = static_cast<uint64_t>(commTiling.splitAxisTailSize) * rankForComm;
            commState_.dataTailHandle_ = commState_.hccl_.template AlltoAll<true>(
                x1_ + headOffset, commOutGM_ + headOffset, tailSendCount,
                static_cast<AscendC::HcclDataType>(x1HcclDataType_), dataStrideCount,
                static_cast<uint8_t>(commTiling.splitAxisTailCnt));
            opStateDump_.DoDump(DUMP_FIELD_COMMIT, 0, 0, 0, commTiling.splitAxisTailCnt);
        }
        commState_.headTileCnt_ = static_cast<uint32_t>(commTiling.splitAxisTileCnt);
    }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool EnableHcommAllToAllOut>
__aicore__ inline void
AllToAllMxQuantMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType, ServerType, IsMxFp4,
                               EnableHcommAllToAllOut>::Run()
{
    if constexpr (EnableHcommAllToAllOut) {
        if ASCEND_IS_AIC {
            if (tilingData_->localMatmul != 0) {
                opStateDump_.DoDump(DUMP_FIELD_POSITION, POS_COMP_LOCAL_BEFORE);
                MatmulProcess(MatmulMode::LOCAL);
            }
            opStateDump_.DoDump(DUMP_FIELD_POSITION, POS_COMP_REMOTE_BEFORE);
            MatmulProcess(MatmulMode::REMOTE);
            quantMatmulKernelImpl_.GetBufferSyncMgr().FinishNotifications();
        }
        if ASCEND_IS_AIV {
            WaitOutputTiles();
        }
        // Both core types participate; AIV never enters HCCL Finalize.
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIC {
            commState_.hccl_.Finalize();
            opStateDump_.DoDump(DUMP_FIELD_PHASE, 0, static_cast<uint8_t>(Utils::RT_PHASE_COMM_FINALIZE));
        }
    } else {
        if (tilingData_->localMatmul != 0) {
            opStateDump_.DoDump(DUMP_FIELD_POSITION, POS_COMP_LOCAL_BEFORE);
            MatmulProcess(MatmulMode::LOCAL);
        }
        opStateDump_.DoDump(DUMP_FIELD_POSITION, POS_COMP_REMOTE_BEFORE);
        MatmulProcess(MatmulMode::REMOTE);
        AscendC::SyncAll();
        commState_.hccl_.Finalize();
        opStateDump_.DoDump(DUMP_FIELD_PHASE, 0, static_cast<uint8_t>(Utils::RT_PHASE_COMM_FINALIZE));
    }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool EnableHcommAllToAllOut>
__aicore__ inline void
AllToAllMxQuantMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType, ServerType, IsMxFp4,
                               EnableHcommAllToAllOut>::WaitOutputTiles()
{
    auto&& commTiling = tilingData_->commTilingData;
    const uint64_t elementsPerRank = commTiling.nonSplitAxisSize / rankDim_;
    const uint64_t bytesPerRank = IsMxFp4 ? elementsPerRank / MXFP_DATA_NUM_PER_BYTE : elementsPerRank;
    // MIX_AIC_1_2: AIV GetBlockIdx is already in [0, 2 * GetBlockNum()).
    // This method is called only by AIV in the output specialization; AIC allocates no UB here.
    AllToAllPermute permute;
    permute.Init(commOutGM_, permuteOutGM_, splitAxisSize_, rankDim_, bytesPerRank,
                 static_cast<uint32_t>(AscendC::GetBlockIdx()), static_cast<uint32_t>(2 * AscendC::GetBlockNum()),
                 pipe_);
    for (uint32_t tileIdx = 0; tileIdx < commState_.totalTileCnt_; ++tileIdx) {
        AscendC::CrossCoreWaitFlag(HCOMM_DATA_READY_FLAG);
        const bool isHead = tileIdx < commTiling.splitAxisTileCnt;
        const uint64_t rowCount = isHead ? commTiling.splitAxisTileSize : commTiling.splitAxisTailSize;
        const uint64_t rowStart =
            isHead ? static_cast<uint64_t>(tileIdx) * commTiling.splitAxisTileSize :
                     static_cast<uint64_t>(commTiling.splitAxisTileCnt) * commTiling.splitAxisTileSize +
                         (tileIdx - commTiling.splitAxisTileCnt) * commTiling.splitAxisTailSize;
        // AlltoAll's recv contract includes self rank. Keep matmul's original local-x1 path unchanged.
        permute.CopyTile(rowStart, rowCount);
        // Idle AIVs still participate in every notification and window ACK.
        if ((tileIdx + 1) % HCOMM_NOTIFY_WINDOW == 0 || tileIdx + 1 == commState_.totalTileCnt_) {
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(HCOMM_WINDOW_ACK_FLAG);
        }
    }
    permute.Finish();
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool EnableHcommAllToAllOut>
__aicore__ inline void
AllToAllMxQuantMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType, ServerType, IsMxFp4,
                               EnableHcommAllToAllOut>::SetupParams(Params& out, MatmulMode matmulMode)
{
    auto&& commTiling = tilingData_->commTilingData;
    const auto& mmTile = tilingData_->tileQbmmTilingData;

    ProblemShape problemShape{mmTile.m, mmTile.n, mmTile.k, 1UL};

    BlockMmadParams mmadParams;
    mmadParams.aGmAddr = commOutGM_;
    mmadParams.scaleAGmAddr = commX1ScaleGM1_;
    mmadParams.bGmAddr = x2_;
    mmadParams.scaleBGmAddr = x2Scale_;
    mmadParams.cGmAddr = y_;
    mmadParams.biasGmAddr = bias_;

    uint32_t splitKNum;
    if (matmulMode == MatmulMode::LOCAL) {
        splitKNum = 1;
    } else if (tilingData_->localMatmul != 0) {
        splitKNum = rankDim_ - 1;
    } else {
        splitKNum = rankDim_;
    }

    LocalParams localParams{rankId_,   rankDim_,   splitAxisSize_,
                            x1_,       x1Scale_,   tilingData_->localMatmul,
                            splitKNum, matmulMode, static_cast<uint32_t>(commTiling.splitAxisTileSize)};

    L1Params l1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};

    BlockSchedulerParams schedulerParams{mmTile.baseM,
                                         mmTile.baseN,
                                         mmTile.mTailTile,
                                         mmTile.nTailTile,
                                         mmTile.mBaseTailSplitCnt,
                                         mmTile.nBaseTailSplitCnt,
                                         mmTile.mTailMain,
                                         mmTile.nTailMain};

    QBMMTiling qbmmParams{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c, isBias_};

    out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool EnableHcommAllToAllOut>
__aicore__ inline void
AllToAllMxQuantMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType, ServerType, IsMxFp4,
                               EnableHcommAllToAllOut>::MatmulProcess(MatmulMode matmulMode)
{
    Params params;
    SetupParams(params, matmulMode);
    quantMatmulKernelImpl_(params, opStateDump_);
}

} // namespace Apace
