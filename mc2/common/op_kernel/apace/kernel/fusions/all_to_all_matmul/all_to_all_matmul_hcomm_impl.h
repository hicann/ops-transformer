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
 * \file all_to_all_matmul_hcomm_impl.h
 * \brief AlltoAll FP and MX/HiF4 Matmul — shared HCOMM communication and output permutation
 *
 * HcommCommState stores the shared communication handles and progress.
 * HcommCommWaitPolicy provides a common wait interface for both compute paths:
 *   - FP maps each matmul tile to the required communication tile through WaitTile();
 *   - MX/HiF4 waits for scale and data through BufferSyncMgr::Acquire().
 *
 * Init():
 *   AIC/AIV initialize tiling parameters and GM addresses.
 *   AIC initializes HCCL and submits data communication; MX/HiF4 additionally submits scale communication.
 *   The selected wait policy is then bound to commState_.
 *
 * Run():
 *   AIC executes the selected FP or MX/HiF4 matmul kernel and waits for all required communication.
 *   FP drains remaining notifications through FinishNotifications();
 *   MX/HiF4 without all2all_out drains them inside its matmul kernel.
 *
 * 输出版本：AIC 逐 tile 通知，AIV 每轮接收并全核同步、重排搬运，每个有界窗口回执一次。
 *
 * \note all2all_out非空时同时启动AIC和AIV；否则仅启动AIC，
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
#include "../../matmul/batch_matmul/all_to_all_mm_kernel.h"
#include "include/tensor_api/tensor.h"
#include "../../../utils/op_state_dump.h"

namespace Apace {
static constexpr AscendC::HcclServerConfig HCOMM_CONFIG = {AscendC::CoreType::ON_AIC, 0};
// Separate from SyncAll's reserved flags 11-14 and the matmul API's flags 0-7.
static constexpr uint16_t AIC_SYNC_EVENT_ID = 3;
static constexpr uint16_t HCOMM_DATA_READY_FLAG = 9;
static constexpr uint16_t HCOMM_WINDOW_ACK_FLAG = 10;
static constexpr uint32_t HCOMM_NOTIFY_WINDOW = 8;

template <typename T>
constexpr uint64_t GetX1HcclDataType()
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        return AscendC::HCCL_DATA_TYPE_FP16;
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::HCCL_DATA_TYPE_BFP16;
    } else if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_UINT8;
    } else if constexpr (AscendC::IsSameType<T, hifloat4x2_t>::value) {
        // HiF4 与 MXFP4 打包方式一致: 2 个 4-bit 元素打包为 1 字节, 按 UINT8 传输
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
    // FP kernel uses uniform virtual tiles; HCOMM may use unequal head/tail tiles.
    uint64_t originalM_{0};
    uint32_t headTileSize_{0};
    uint32_t tailTileSize_{0};
    uint32_t virtualTileM_{0};
};

template <AscendC::HcclServerType ServerType, bool EnableHcommAllToAllOut, bool HasScale = true>
struct HcommCommWaitPolicy {
    HcommCommState<ServerType>* state_{nullptr};
    uint32_t slotNum{0};

    __aicore__ inline void Release(uint32_t) {}

    __aicore__ inline void Acquire(uint32_t tileIdx)
    {
        if constexpr (!EnableHcommAllToAllOut && HasScale) {
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
            while (state_->nextNotifyTile_ <= tileIdx && state_->nextNotifyTile_ < state_->totalTileCnt_) {
                if (EnableHcommAllToAllOut &&
                    state_->nextNotifyTile_ - state_->acknowledgedTiles_ == HCOMM_NOTIFY_WINDOW) {
                    // A scalar wait prevents the sender loop from running ahead of the paired AIVs.
                    AscendC::CrossCoreWaitFlag(HCOMM_WINDOW_ACK_FLAG);
                    state_->acknowledgedTiles_ = state_->nextNotifyTile_;
                }
                const uint32_t nextTile = state_->nextNotifyTile_;
                if constexpr (HasScale) {
                    if (nextTile == 0) {
                        state_->hccl_.Wait(state_->scaleHandle_);
                    }
                }
                const auto handle = nextTile < state_->headTileCnt_ ? state_->dataHeadHandle_ : state_->dataTailHandle_;
                state_->hccl_.Wait(handle);
                if constexpr (EnableHcommAllToAllOut) {
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(HCOMM_DATA_READY_FLAG);
                }
                ++state_->nextNotifyTile_;
            }
        }
    }

    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        // Wait conservatively through the last row of the virtual tile. This
        // preserves the unchanged FP kernel contract even across a head/tail boundary.
        uint64_t rowEnd = (static_cast<uint64_t>(tileIdx) + 1) * state_->virtualTileM_;
        rowEnd = (rowEnd < state_->originalM_ ? rowEnd : state_->originalM_) - 1;
        const uint64_t headRows = static_cast<uint64_t>(state_->headTileCnt_) * state_->headTileSize_;
        const uint32_t commTile = rowEnd < headRows ?
                                      rowEnd / state_->headTileSize_ :
                                      state_->headTileCnt_ + (rowEnd - headRows) / state_->tailTileSize_;
        Acquire(commTile);
    }

    __aicore__ inline void FinishNotifications()
    {
        if constexpr (EnableHcommAllToAllOut || !HasScale) {
            // Covers idle cores and an early return from the matmul scheduler. Never re-wait a completed tile.
            if (state_->nextNotifyTile_ < state_->totalTileCnt_) {
                Acquire(state_->totalTileCnt_ - 1);
            }
            if (EnableHcommAllToAllOut && state_->acknowledgedTiles_ < state_->nextNotifyTile_) {
                AscendC::CrossCoreWaitFlag(HCOMM_WINDOW_ACK_FLAG);
                state_->acknowledgedTiles_ = state_->nextNotifyTile_;
            }
        }
    }
};

// Select only the compute component; communication and output handling stay shared.
template <typename X1, typename X2, typename Y, typename Bias, typename Shape, typename LayoutA, typename LayoutB,
          typename WaitPolicy, bool IsFp>
struct HcommMatmulComponents;

template <typename X1, typename X2, typename Y, typename Bias, typename Shape, typename LayoutA, typename LayoutB,
          typename WaitPolicy>
struct HcommMatmulComponents<X1, X2, Y, Bias, Shape, LayoutA, LayoutB, WaitPolicy, false> {
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<Shape, 0, LayoutA, LayoutB, X1>;
    using Dispatch = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Dispatch, X1, LayoutA, X2, LayoutB, Y, LayoutA, Bias, LayoutA>;
    using Kernel = Blaze::Gemm::Kernel::AllToAllQbmmMxKernel<Shape, Mmad, Scheduler, WaitPolicy, false>;
    using ScaleType = typename Kernel::ScaleType;
    static constexpr bool IS_HIF4 = Mmad::IS_HIF4;
};

template <typename X1, typename X2, typename Y, typename Bias, typename Shape, typename LayoutA, typename LayoutB,
          typename WaitPolicy>
struct HcommMatmulComponents<X1, X2, Y, Bias, Shape, LayoutA, LayoutB, WaitPolicy, true> {
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<Shape>;
    using Dispatch = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelMmadMultiBlockBasic, 0>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Dispatch, X1, LayoutA, X2, LayoutB, Y, LayoutA, Bias, LayoutA>;
    using Kernel = Blaze::Gemm::Kernel::AllToAllMmKernel<Shape, Mmad, Scheduler, WaitPolicy, false>;
    using ScaleType = uint8_t; // No scale storage or communication in FP mode.
    static constexpr bool IS_HIF4 = false;
};

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB,
          bool EnableHcommAllToAllOut, typename BiasType = float>
class AllToAllMatmulHcommImpl {
public:
    explicit __aicore__ inline AllToAllMatmulHcommImpl(AlltoAllMatmulTilingDataType* tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale,
                                GM_ADDR x2_scale, GM_ADDR workspaceGM, AscendC::TPipe* pipe);
    __aicore__ inline void Run();

    // Layout 定义
    using LayoutA = asc::te::nd_ext_layout_ptn;
    // B 矩阵布局: TransB=true 为转置 (N, K), false 为非转置 (K, N);
    // MX 场景恒为转置, HiF4 场景转置/非转置均支持
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    static constexpr bool IsFp =
        AscendC::IsSameType<X1Type, half>::value || AscendC::IsSameType<X1Type, bfloat16_t>::value;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using WaitPolicy = HcommCommWaitPolicy<ServerType, EnableHcommAllToAllOut, !IsFp>;
    using Components =
        HcommMatmulComponents<X1Type, X2Type, YType, BiasType, ProblemShape, LayoutA, LayoutB, WaitPolicy, IsFp>;
    using MatmulKernelImpl = typename Components::Kernel;

    // 参数类型
    using Params = typename MatmulKernelImpl::Params;
    using BlockMmadParams = typename MatmulKernelImpl::BlockMmadParams;
    using LocalParams = typename MatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename MatmulKernelImpl::BlockSchedulerParams;
    using MatmulMode = typename MatmulKernelImpl::MatmulMode;
    using ScaleType = typename Components::ScaleType;
    static constexpr bool isHiF4 = Components::IS_HIF4;

    MatmulKernelImpl matmulKernelImpl_;
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

    // Scale 通信数据类型: MX 为 E8M0; HiF4 的 hif4_scale 为完整 4 字节元数据(E6M2+E1_8+E1_16),
    // HCCL 未提供专用类型, E8M0和hif4_scale 均按 FP8E8M0 传输, 实际传输时乘数据类型
    static constexpr uint64_t scaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;
    static constexpr uint64_t x1HcclDataType_ = GetX1HcclDataType<X1Type>();
    // A common view adapts the existing FP and MX host tiling layouts.
    struct CommTilingView {
        uint64_t splitAxisTileSize;
        uint64_t splitAxisTileCnt;
        uint64_t splitAxisTailSize;
        uint64_t splitAxisTailCnt;
        uint64_t nonSplitAxisSize;
    };
    CommTilingView commTiling_{};
    uint32_t localMatmul_{0};
    uint32_t rankId_{0};
    uint32_t rankDim_{0};
    uint64_t splitAxisSize_{0};
    bool isBias_{false};

    static constexpr uint64_t PACKED_4BIT_ELEMENTS_PER_BYTE = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;

    Mc2Kernel::OpStateDump opStateDump_;

    __aicore__ inline WaitPolicy& GetWaitPolicy()
    {
        if constexpr (IsFp) {
            return matmulKernelImpl_.GetCommPolicy();
        } else {
            return matmulKernelImpl_.GetBufferSyncMgr();
        }
    }

    __aicore__ inline void SetupParams(Params& out, MatmulMode matmulMode);
    __aicore__ inline void MatmulProcess(MatmulMode matmulMode);
    __aicore__ inline void WaitOutputTiles();

    __aicore__ inline void SyncAicCores()
    {
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(AIC_SYNC_EVENT_ID);
    }
};

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB,
          bool EnableHcommAllToAllOut, typename BiasType>
__aicore__ inline void
AllToAllMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType, ServerType, IsMxFp4,
                        TransB, EnableHcommAllToAllOut, BiasType>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
                                                                        GM_ADDR all2all_out, GM_ADDR x1_scale,
                                                                        GM_ADDR x2_scale, GM_ADDR workspaceGM,
                                                                        AscendC::TPipe* pipe)
{
    pipe_ = pipe;
    if constexpr (IsFp) {
        const auto& info = tilingData_->alltoAllMatmulTilingInfo;
        commTiling_ = {info.tileM, info.tileCnt, info.tailM, info.tailCnt,
                       static_cast<uint64_t>(info.rankK) * info.rankDim};
        rankDim_ = info.rankDim;
        localMatmul_ = tilingData_->localMatmul;
    } else {
        const auto& info = tilingData_->commTilingData;
        commTiling_ = {info.splitAxisTileSize, info.splitAxisTileCnt, info.splitAxisTailSize, info.splitAxisTailCnt,
                       info.nonSplitAxisSize};
        rankDim_ = tilingData_->rankDim;
        localMatmul_ = tilingData_->localMatmul;
    }
    auto&& commTiling = commTiling_;
    splitAxisSize_ = commTiling.splitAxisTileSize * commTiling.splitAxisTileCnt +
                     commTiling.splitAxisTailSize * commTiling.splitAxisTailCnt;
    if constexpr (EnableHcommAllToAllOut || IsFp) {
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
    // scale 通信空间(字节): MX 为 组数*2(E8M0)*M 行; HiF4 为 组数*4(hif4_scale)*M 行
    // 每 64 元素组 scale 单元数与单元宽度统一由 MxScaleKLen 泛化
    uint64_t x1ScaleLen = 0;
    if constexpr (!IsFp) {
        x1ScaleLen = static_cast<uint64_t>(Blaze::Gemm::MxScaleKLen<X1Type>(commTiling.nonSplitAxisSize)) *
                     splitAxisSize_ * sizeof(ScaleType);
        x1ScaleLen = Blaze::Gemm::CeilDiv(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM;
    }
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
        if constexpr (!IsFp) {
            opStateDump_.Init(workspaceGM_, &tilingData_->dumpInfo.workspaceLayout,
                              tilingData_->tileQbmmTilingData.usedCoreNum);
        }
#endif
        GetWaitPolicy().state_ = &commState_;
        if constexpr (IsFp) {
            commState_.originalM_ = splitAxisSize_;
            commState_.headTileSize_ = commTiling.splitAxisTileSize;
            commState_.tailTileSize_ = commTiling.splitAxisTailSize;
            commState_.virtualTileM_ = tilingData_->apaceMmTilingData.baseM;
        }
        GetWaitPolicy().slotNum = static_cast<uint32_t>(commTiling.splitAxisTileCnt + commTiling.splitAxisTailCnt);
        commState_.hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
        commState_.hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(AlltoAllMatmulTilingDataType, mc2CcTiling)));
        rankId_ = commState_.hccl_.GetRankId();
        rankDim_ = commState_.hccl_.GetRankDim();

        uint64_t originalRankForComm = commTiling.nonSplitAxisSize / rankDim_;
        uint64_t rankForComm = originalRankForComm;
        if constexpr (IsMxFp4 || isHiF4) {
            // 4-bit 打包(MXFP4/HiF4): 2 个元素共享 1 字节, 通信字节数为元素数的一半
            rankForComm = Blaze::Gemm::CeilDiv(rankForComm, PACKED_4BIT_ELEMENTS_PER_BYTE);
        }
        // scale 每 rank 通信单元数(按 scale 元素计): MX 为 E8M0 单元(Ka/64*2), HiF4 为 UINT32 单元(Ka/64)
        uint64_t dataStrideCount = static_cast<uint64_t>(splitAxisSize_) * rankForComm;
        if constexpr (!IsFp) {
            uint32_t scaleKPerRank = static_cast<uint32_t>(Blaze::Gemm::MxScaleKLen<X1Type>(originalRankForComm));

            uint64_t scaleSendCount = static_cast<uint64_t>(splitAxisSize_) * scaleKPerRank;
            opStateDump_.DoDump(DUMP_FIELD_STEP, POS_COMM_BEFORE, static_cast<uint8_t>(Utils::RT_PHASE_COMM_PREPARE));
            // DoDump does not count the scale commit.
            commState_.scaleHandle_ = commState_.hccl_.template AlltoAll<true>(
                x1Scale_, commX1ScaleGM1_, scaleSendCount * static_cast<uint64_t>(sizeof(ScaleType)),
                static_cast<AscendC::HcclDataType>(scaleHcclDataType_),
                scaleSendCount * static_cast<uint64_t>(sizeof(ScaleType)), 1);
        }

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
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB,
          bool EnableHcommAllToAllOut, typename BiasType>
__aicore__ inline void AllToAllMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType,
                                               ServerType, IsMxFp4, TransB, EnableHcommAllToAllOut, BiasType>::Run()
{
    if ASCEND_IS_AIC {
        if constexpr (IsFp) {
            if (localMatmul_ == 2) {
                MatmulProcess(MatmulMode::LOCAL);
            }
            const MatmulMode mode = localMatmul_ == 1 ? MatmulMode::DEFERRED_SYNC : MatmulMode::REMOTE;
            MatmulProcess(mode);
        } else {
            if (localMatmul_ != 0) {
                opStateDump_.DoDump(DUMP_FIELD_POSITION, POS_COMP_LOCAL_BEFORE);
                MatmulProcess(MatmulMode::LOCAL);
            }
            opStateDump_.DoDump(DUMP_FIELD_POSITION, POS_COMP_REMOTE_BEFORE);
            MatmulProcess(MatmulMode::REMOTE);
        }
        GetWaitPolicy().FinishNotifications();
    }
    if constexpr (EnableHcommAllToAllOut) {
        if ASCEND_IS_AIV {
            WaitOutputTiles();
        }
    }
    if constexpr (EnableHcommAllToAllOut) {
        AscendC::SyncAll<false>();
    } else {
        SyncAicCores();
    }
    if ASCEND_IS_AIC {
        commState_.hccl_.Finalize();
        opStateDump_.DoDump(DUMP_FIELD_PHASE, 0, static_cast<uint8_t>(Utils::RT_PHASE_COMM_FINALIZE));
    }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB,
          bool EnableHcommAllToAllOut, typename BiasType>
__aicore__ inline void
AllToAllMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType, ServerType, IsMxFp4,
                        TransB, EnableHcommAllToAllOut, BiasType>::WaitOutputTiles()
{
    auto&& commTiling = commTiling_;
    const uint64_t elementsPerRank = commTiling.nonSplitAxisSize / rankDim_;
    // 4-bit 打包场景(MXFP4/HiF4): 2 个元素共享 1 字节, 通信字节数为元素数的一半,
    // 与 Init 中 rankForComm、tiling 中 commLen 的 (IsMxFp4 || isHiF4) 口径保持一致
    const uint64_t bytesPerRank =
        (IsMxFp4 || isHiF4) ? elementsPerRank / PACKED_4BIT_ELEMENTS_PER_BYTE : elementsPerRank * sizeof(X1Type);
    // This method is called only by AIV in the output specialization; AIC allocates no UB here.
    AllToAllPermute permute;
    // WaitOutputTiles() 在 Run() 中已被 ASCEND_IS_AIV 守护。 GetTaskRatio() 在950下配置CV核 MIX_AIC_1_2
    // 获得2，在960下配置CV核 MIX_AIC_1_1 获得1。
    const uint32_t aivWorkerCount =
        static_cast<uint32_t>(AscendC::GetBlockNum()) * static_cast<uint32_t>(AscendC::GetTaskRatio());
    permute.Init(commOutGM_, permuteOutGM_, splitAxisSize_, rankDim_, bytesPerRank,
                 static_cast<uint32_t>(AscendC::GetBlockIdx()), aivWorkerCount, pipe_);
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
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB,
          bool EnableHcommAllToAllOut, typename BiasType>
__aicore__ inline void
AllToAllMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType, ServerType, IsMxFp4,
                        TransB, EnableHcommAllToAllOut, BiasType>::SetupParams(Params& out, MatmulMode matmulMode)
{
    if constexpr (IsFp) {
        const auto& info = tilingData_->alltoAllMatmulTilingInfo;
        const auto& fpTile = tilingData_->apaceMmTilingData;
        out = {};
        out.problemShape = ProblemShape{static_cast<int64_t>(splitAxisSize_), info.rankN, info.rankK, 1L};
        auto& mm = out.mmadParams;
        mm.aGmAddr = commOutGM_;
        mm.bGmAddr = x2_;
        mm.cGmAddr = y_;
        // Match AddMM/MX split-K semantics: load bias on splitKIdx == 0 and accumulate all K fragments in L0C.
        mm.biasGmAddr = bias_;
        mm.mL1 = fpTile.mL1;
        mm.mL0 = fpTile.baseM;
        mm.nL1 = fpTile.nL1;
        mm.nL0 = fpTile.baseN;
        mm.kL0 = fpTile.baseK;
        mm.kL1 = fpTile.kL1;
        mm.l1Stages = fpTile.l1BufferNum;
        mm.l0cStages = fpTile.l0cDB;
        if (matmulMode == MatmulMode::LOCAL) {
            mm.splitKNum = 1;
        } else if (localMatmul_ == 2) {
            mm.splitKNum = rankDim_ - 1;
        } else {
            mm.splitKNum = rankDim_;
        }
        auto& sch = out.schParams;
        sch.mL1 = mm.mL1;
        sch.baseM = mm.mL0;
        sch.nL1 = mm.nL1;
        sch.baseN = mm.nL0;
        sch.kL1 = mm.kL1;
        sch.baseK = mm.kL0;
        sch.mTailCnt = fpTile.mTailCnt;
        sch.nTailCnt = fpTile.nTailCnt;
        sch.mBaseTailSplitCnt = fpTile.mBaseTailSplitCnt;
        sch.nBaseTailSplitCnt = fpTile.nBaseTailSplitCnt;
        sch.mTailMain = fpTile.mTailMain;
        sch.nTailMain = fpTile.nTailMain;
        sch.windowLen = 1;
        out.mmTiling = {mm.mL0, mm.nL0, mm.kL0, mm.l0cStages};
        // localMatmul is host-selected: 0 uses deterministic all-rank REMOTE, 1 uses DEFERRED_SYNC,
        // and 2 computes LOCAL first and accumulates the remaining ranks through REMOTE.
        out.localParams = {rankId_, rankDim_, splitAxisSize_, x1_, localMatmul_, matmulMode, fpTile.baseM};
    } else {
        using L1Params = typename MatmulKernelImpl::L1Params;
        using QBMMTiling = typename MatmulKernelImpl::QBMMTiling;
        auto&& commTiling = commTiling_;
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
        } else if (localMatmul_ != 0) {
            splitKNum = rankDim_ - 1;
        } else {
            splitKNum = rankDim_;
        }

        LocalParams localParams{rankId_,   rankDim_,   splitAxisSize_,
                                x1_,       x1Scale_,   localMatmul_,
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
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB,
          bool EnableHcommAllToAllOut, typename BiasType>
__aicore__ inline void
AllToAllMatmulHcommImpl<X1Type, X2Type, YType, CommDataTypeX1, AlltoAllMatmulTilingDataType, ServerType, IsMxFp4,
                        TransB, EnableHcommAllToAllOut, BiasType>::MatmulProcess(MatmulMode matmulMode)
{
    Params params;
    SetupParams(params, matmulMode);
    if constexpr (IsFp) {
        matmulKernelImpl_(params);
    } else {
        matmulKernelImpl_(params, opStateDump_);
    }
}

} // namespace Apace
