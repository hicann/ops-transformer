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
 * \file grouped_mat_mul_allto_allv_mte.h
 * \brief AIV-driven MTE communication for GroupedMatMulAlltoAllvV2 on Ascend 910B/910_93.
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H

#include "kernel_operator.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "grouped_mat_mul_allto_allv_mte_tiling.h"

namespace MC2KernelTemplate {

namespace GmmA2avMteDetail {
constexpr uint32_t DEFAULT_UB_MOVE_ELEMENTS = Gmma2avMteTiling::UB_MOVE_ELEMENTS;
constexpr uint32_t USED_UB_BYTES = 192U * 1024U;
constexpr uint32_t CONTROL_UB_OFFSET_BYTES = 64U * 1024U;
constexpr uint32_t DESTINATION_OFFSET_UB_OFFSET_BYTES = 72U * 1024U;
constexpr uint32_t SOURCE_OFFSET_UB_OFFSET_BYTES = 80U * 1024U;
constexpr uint32_t PONG_UB_OFFSET_BYTES = 96U * 1024U;
constexpr uint32_t CONTROL_MTE_BYTES = 32U;
constexpr uint32_t CONTROL_MTE_ELEMENTS = CONTROL_MTE_BYTES / sizeof(int32_t);
constexpr uint32_t SYNC_MTE_ELEMENTS = CONTROL_MTE_BYTES / sizeof(int64_t);
// The control scratch spans [CONTROL_UB_OFFSET_BYTES, DESTINATION_OFFSET_UB_
// OFFSET_BYTES) = 8 KiB. Expert-ready staging reuses it in a disjoint phase
// from the peer-count staging, so the full region is available; one batched
// flag transaction may cover up to 128 consecutive 64-byte slots.
constexpr uint32_t FLAG_STAGING_ELEMENTS =
    (DESTINATION_OFFSET_UB_OFFSET_BYTES - CONTROL_UB_OFFSET_BYTES) / sizeof(int64_t);
constexpr uint32_t SYNC_SLOT_ELEMENTS = Gmma2avMteTiling::SYNC_SLOT_BYTES / sizeof(int64_t);
static_assert(CONTROL_MTE_BYTES % sizeof(int64_t) == 0U,
              "The synchronization transaction must contain complete int64 elements");
constexpr uint32_t INVALID_TOKEN_OFFSET = 0xffffffffU;
static_assert(CONTROL_MTE_BYTES % sizeof(int32_t) == 0U,
              "The control transaction must contain complete int32 elements");
static_assert(Gmma2avMteTiling::SYNC_SLOT_BYTES >= CONTROL_MTE_BYTES,
              "Every synchronization slot must hold one control transaction");
static_assert(Gmma2avMteTiling::PAYLOAD_UB_REGION_BYTES <= CONTROL_UB_OFFSET_BYTES,
              "The payload ping buffer must fit before the control scratch");
static_assert(SOURCE_OFFSET_UB_OFFSET_BYTES % CONTROL_MTE_BYTES == 0U,
              "The source-offset buffer must be control-transaction aligned");
static_assert(CONTROL_UB_OFFSET_BYTES + Gmma2avMteTiling::MAX_COUNT_NUM * sizeof(int32_t) <=
                  DESTINATION_OFFSET_UB_OFFSET_BYTES,
              "The peer count table must fit before the destination-offset buffer");
static_assert(DESTINATION_OFFSET_UB_OFFSET_BYTES + Gmma2avMteTiling::MAX_COUNT_NUM * sizeof(uint32_t) <=
                  SOURCE_OFFSET_UB_OFFSET_BYTES,
              "The destination-offset table must fit before the source-offset buffer");
static_assert(SOURCE_OFFSET_UB_OFFSET_BYTES + Gmma2avMteTiling::MAX_COUNT_NUM * sizeof(uint32_t) <=
                  PONG_UB_OFFSET_BYTES,
              "The source-offset table must fit before the payload pong buffer");
static_assert(PONG_UB_OFFSET_BYTES + Gmma2avMteTiling::PAYLOAD_UB_REGION_BYTES <= USED_UB_BYTES,
              "The payload pong buffer must fit in the allocated UB");
} // namespace GmmA2avMteDetail

template <typename T>
class GmmA2avMteOp {
public:
    __aicore__ inline GmmA2avMteOp() {}

    __aicore__ inline void Init(const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer,
                                GM_ADDR recvBuffer, GM_ADDR cumsumBuffer, AscendC::TPipe *pipe,
                                const GmmA2avCoCTiling *cocTiling = nullptr, uint64_t commBufferSize = 0UL,
                                uint32_t isA3 = 0U)
    {
        taskTilingInfo_ = taskTilingInfo;
        sendBuffer_ = reinterpret_cast<__gm__ T *>(sendBuffer);
        recvBuffer_ = reinterpret_cast<__gm__ T *>(recvBuffer);
        // Kept in the common MTE Init signature for serialized-tiling ABI
        // compatibility. Expert-overlap offsets are owner-local in UB.
        (void)cumsumBuffer;
        if ASCEND_IS_AIC {
            return;
        }

        auto context = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        isA3_ = isA3 != 0U;
        if (isA3_) {
            // The established A3 package uses the full MC2 resource context.
            // Its peer windows are referenced by remoteRes[].
            a3WinContext_ = reinterpret_cast<__gm__ HcclOpResParam *>(context);
            rank_ = a3WinContext_->localUsrRankId;
            rankSize_ = a3WinContext_->rankSize;
            winSize_ = a3WinContext_->winSize == 0UL ? commBufferSize : a3WinContext_->winSize;
        } else {
            // Ascend 910B MultiPut exposes HcclCombineOpParam. For larger
            // communication domains HCCL supplies the dynamic data[] table.
            a2WinContext_ = reinterpret_cast<__gm__ AscendC::HcclCombineOpParam *>(context);
            rank_ = a2WinContext_->rankId;
            rankSize_ = a2WinContext_->rankNum;
            winSize_ = a2WinContext_->winSize == 0UL ? commBufferSize : a2WinContext_->winSize;
        }
        coreIdx_ = AscendC::GetBlockIdx();
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        logicalCoreIdx_ = coreIdx_ / AscendC::GetTaskRation();
        rankWorkerCount_ = static_cast<uint32_t>(taskTilingInfo_->aicCoreNum);
        AscendC::SetAtomicNone();
        if (cocTiling != nullptr && cocTiling->ubMoveNum > 0) {
            ubMoveElements_ = static_cast<uint32_t>(cocTiling->ubMoveNum);
        }

        // The first 64 KiB is the payload ping buffer. Count-table scratch is
        // staged at 64 KiB, destination offsets at 72 KiB, source offsets at
        // 80 KiB, and payload pong starts at 96 KiB.
        pipe->InitBuffer(ubBuffer_, GmmA2avMteDetail::USED_UB_BYTES);
    }

    __aicore__ inline void BeginExpertPipeline()
    {
        if ASCEND_IS_AIC {
            return;
        }
        BeginInvocation();

        // Counts do not depend on GMM output. Publish/acquire them while AIC
        // computes the first chunk. This rank phase protects the count table
        // independently of masked per-expert payload-ready flags. A local compute chunk can have no incoming
        // rows, so masked expert-ready waits cannot protect the count table.
        PublishLocalCountTable();
        PublishAndWaitRankPhase(Gmma2avMteTiling::COUNT_READY_BASE);
        BuildDestinationOffsets();
        LocalAivSync();
        localExpertBaseRows_ = 0UL;
    }

    __aicore__ inline void StageChunkPayload(uint32_t startExpertIdx, uint32_t endExpertIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }

        // Each AIV only acquires completion from its paired AIC, which
        // notifies once per chunk. The CATLASS scheduler rotates expert tiles
        // across AICs, while payload rows are redistributed independently
        // across AIVs; consequently the AIV that copies a row is not
        // necessarily paired with the AIC that produced it. Match mega_moe
        // A3's C2V hand-off: after every paired wait, all AIVs rendezvous
        // before any of them reads the shared GMM workspace.
        LocalAivSync();
        PublishLocalChunk(startExpertIdx, endExpertIdx);
    }

    __aicore__ inline void PublishChunkReady(uint32_t startExpertIdx, uint32_t endExpertIdx)
    {
        // Match mega_moe/combine A3's destination-local mailbox protocol. All
        // payload writers drain before the leading barrier, then source rank
        // S writes ExpertReady(D, S, E) into every destination D's local
        // window. A destination only polls its own window through the valid
        // L2-bypass alias; it never polls an encoded peer VA.
        LocalAivSync();
        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            for (uint32_t dstRank = logicalCoreIdx_; dstRank < rankSize_; dstRank += rankWorkerCount_) {
                if (PublishExpertReadyRange(dstRank, startExpertIdx, endExpertIdx, syncEpoch_)) {
                    continue;
                }
                PublishExpertReadyPerFlag(dstRank, startExpertIdx, endExpertIdx, syncEpoch_);
            }
        }
    }

    __aicore__ inline void WaitChunkReady(uint32_t startExpertIdx, uint32_t endExpertIdx)
    {
        // Finish every outbound publication before allowing any worker to
        // wait, preventing a rank-level publish/wait cycle.
        LocalAivSync();
        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
                if (WaitExpertReadyRange(srcRank, startExpertIdx, endExpertIdx, syncEpoch_)) {
                    continue;
                }
                WaitExpertReadyPerFlag(srcRank, startExpertIdx, endExpertIdx, syncEpoch_);
            }
        }
        LocalAivSync();
    }

    __aicore__ inline void PullExpertChunk(uint32_t startExpertIdx, uint32_t endExpertIdx)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (startExpertIdx == 0U) {
            // The same worker that acquired each source's ready flag loads
            // that source's count table and remains its payload-pull owner.
            BuildPeerSourceOffsets();
            LocalAivSync();
        }
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        AscendC::LocalTensor<uint32_t> destinationOffsets = GetDestinationOffsetTensor();
        AscendC::LocalTensor<uint32_t> sourceOffsets = GetSourceOffsetTensor();
        for (uint32_t srcRank = logicalCoreIdx_; subBlockIdx_ == 0U && rankWorkerCount_ != 0U && srcRank < rankSize_;
             srcRank += rankWorkerCount_) {
            // Source rows are expert-major/destination-major, whereas output
            // rows are source-major/expert-major. Other destinations can
            // leave gaps between this destination's expert slices. Merge
            // only when both sets of offsets prove the range contiguous.
            const uint64_t baseIndex = static_cast<uint64_t>(srcRank) * expertNum;
            const uint32_t firstSourceOffset = sourceOffsets.GetValue(baseIndex + startExpertIdx);
            const uint32_t firstDestinationOffset = destinationOffsets.GetValue(baseIndex + startExpertIdx);
            uint64_t chunkRows = 0UL;
            bool chunkContiguous = firstSourceOffset != GmmA2avMteDetail::INVALID_TOKEN_OFFSET &&
                                   firstDestinationOffset != GmmA2avMteDetail::INVALID_TOKEN_OFFSET;
            for (uint32_t expertIdx = startExpertIdx; chunkContiguous && expertIdx < endExpertIdx; ++expertIdx) {
                const int32_t recvCount = taskTilingInfo_->recvCnt[baseIndex + expertIdx];
                if (recvCount < 0 ||
                    sourceOffsets.GetValue(baseIndex + expertIdx) == GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
                    destinationOffsets.GetValue(baseIndex + expertIdx) == GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
                    static_cast<uint64_t>(sourceOffsets.GetValue(baseIndex + expertIdx)) !=
                        static_cast<uint64_t>(firstSourceOffset) + chunkRows ||
                    static_cast<uint64_t>(destinationOffsets.GetValue(baseIndex + expertIdx)) !=
                        static_cast<uint64_t>(firstDestinationOffset) + chunkRows) {
                    chunkContiguous = false;
                    break;
                }
                chunkRows += static_cast<uint64_t>(recvCount);
            }
            if (chunkContiguous) {
                if (chunkRows > 0UL) {
                    CopyGmToGm(GetOutputData(srcRank) + static_cast<uint64_t>(firstSourceOffset) * taskTilingInfo_->N1,
                               recvBuffer_ + static_cast<uint64_t>(firstDestinationOffset) * taskTilingInfo_->N1,
                               chunkRows * taskTilingInfo_->N1);
                }
                continue;
            }
            for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
                PullPeerExpertFromSource(srcRank, expertIdx);
            }
        }
        // Non-copying subblocks must join the same barrier before any AIV
        // advances to the next chunk or the invocation-completion protocol.
        LocalAivSync();
    }

    __aicore__ inline void Wait(uint32_t startExpertIdx)
    {
        (void)startExpertIdx;
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIV {
            // Every payload-pull worker has drained its MTE2/MTE3 pipeline
            // before the entry barrier below. Completion protects source
            // windows; acknowledgement protects all persistent phase slots
            // against reuse by the next invocation.
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            PublishAndWaitRankPhase(Gmma2avMteTiling::COMPLETION_BASE);
            PublishAndWaitRankPhase(Gmma2avMteTiling::ACK_BASE);
        }
        AscendC::SyncAll<false>();
    }

private:
    __aicore__ inline GM_ADDR GetWindow(uint32_t rank) const
    {
        if (isA3_) {
            if (rank == rank_) {
                return (GM_ADDR)(a3WinContext_->localWindowsIn);
            }
            auto relation =
                reinterpret_cast<__gm__ HcclRankRelationResV2 *>(a3WinContext_->remoteRes[rank].nextDevicePtr);
            return (GM_ADDR)(relation->windowsIn);
        }
        if (a2WinContext_->multiFlag == 0U) {
            return (GM_ADDR)(a2WinContext_->windowsIn[rank]);
        }
        return rank == rank_ ? (GM_ADDR)(a2WinContext_->data[rank].localInput.addr) :
                               (GM_ADDR)(a2WinContext_->data[rank].remoteInput.addr);
    }

    __aicore__ inline __gm__ int32_t *GetCountTable(uint32_t rank) const
    {
        return reinterpret_cast<__gm__ int32_t *>(GetWindow(rank) + winSize_ -
                                                  Gmma2avMteTiling::SEND_COUNT_STAGING_FROM_TAIL);
    }

    __aicore__ inline __gm__ T *GetOutputData(uint32_t rank) const
    {
        return reinterpret_cast<__gm__ T *>(GetWindow(rank));
    }

    __aicore__ inline void BeginInvocation()
    {
        // HCCL windows persist across cases. The control plane must provide a
        // zero-initialized fixed-layout region on first use and serialize all
        // invocations on this communicator. Under that contract the complete
        // acknowledgement vector is the ownership proof for advancing epoch.
        LocalAivSync();
        if (IsPrimaryAiv()) {
            const int64_t previousEpoch = AscendC::ReadGmByPassDCache(GetEpochSlot(rank_));
            for (uint32_t participant = 0U; participant < rankSize_; ++participant) {
                WaitSyncFlag(GetAckSlot(rank_, participant), previousEpoch);
            }
            // Admission guarantees 0 <= previousEpoch < MAX_SYNC_EPOCH. This
            // hot kernel intentionally has no unilateral wrap/recovery path;
            // the executor must quiesce and reinitialize before exhaustion.
            syncEpoch_ = previousEpoch == 0 ? 1 : previousEpoch + 1;
            StoreSyncFlag(GetEpochSlot(rank_), syncEpoch_);
        }
        LocalAivSync();
        if (!IsPrimaryAiv()) {
            syncEpoch_ = AscendC::ReadGmByPassDCache(GetEpochSlot(rank_));
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
    }

    __aicore__ inline void PublishLocalCountTable()
    {
        if (!IsPrimaryAiv()) {
            return;
        }
        const uint32_t countNum = rankSize_ * static_cast<uint32_t>(taskTilingInfo_->e);
        const uint32_t alignedCountNum = AlignControlElements(countNum);
        AscendC::LocalTensor<int32_t> local = GetControlTensor();
        for (uint32_t index = 0U; index < countNum; ++index) {
            local.SetValue(index, taskTilingInfo_->sendCnt[index]);
        }
        for (uint32_t index = countNum; index < alignedCountNum; ++index) {
            local.SetValue(index, 0);
        }
        StoreInt32Row(GetCountTable(rank_), local, alignedCountNum);
    }

    __aicore__ inline void PublishLocalChunk(uint32_t startExpertIdx, uint32_t endExpertIdx)
    {
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        uint64_t chunkRows = 0UL;
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
            for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                const int32_t count = taskTilingInfo_->sendCnt[static_cast<uint64_t>(dstRank) * expertNum + expertIdx];
                if (count > 0) {
                    chunkRows += static_cast<uint64_t>(count);
                }
            }
        }

        // A chunk's experts are contiguous in both the GMM output and the
        // send buffer, so the whole range is one copy per worker instead of
        // one small copy per expert.
        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            const uint64_t rowsPerWorker = (chunkRows + rankWorkerCount_ - 1UL) / rankWorkerCount_;
            const uint64_t workerRowStart = static_cast<uint64_t>(logicalCoreIdx_) * rowsPerWorker;
            if (workerRowStart < chunkRows) {
                const uint64_t rowCount =
                    chunkRows - workerRowStart > rowsPerWorker ? rowsPerWorker : chunkRows - workerRowStart;
                const uint64_t sourceRow = localExpertBaseRows_ + workerRowStart;
                CopyGmToGm(sendBuffer_ + sourceRow * taskTilingInfo_->N1,
                           GetOutputData(rank_) + sourceRow * taskTilingInfo_->N1, rowCount * taskTilingInfo_->N1);
            }
        }
        localExpertBaseRows_ += chunkRows;
    }

    __aicore__ inline void BuildDestinationOffsets()
    {
        if (subBlockIdx_ != 0U) {
            return;
        }
        AscendC::LocalTensor<uint32_t> offsets = GetDestinationOffsetTensor();
        const uint32_t countNum = rankSize_ * static_cast<uint32_t>(taskTilingInfo_->e);
        uint64_t runningOffset = 0UL;
        bool prefixValid = true;
        for (uint32_t index = 0U; index < countNum; ++index) {
            const int32_t count = taskTilingInfo_->recvCnt[index];
            const bool currentValid = prefixValid && count >= 0 &&
                                      runningOffset <= GmmA2avMteDetail::INVALID_TOKEN_OFFSET &&
                                      runningOffset <= taskTilingInfo_->BSK &&
                                      static_cast<uint64_t>(count) <= taskTilingInfo_->BSK - runningOffset;
            offsets.SetValue(
                index, currentValid ? static_cast<uint32_t>(runningOffset) : GmmA2avMteDetail::INVALID_TOKEN_OFFSET);
            if (!currentValid) {
                prefixValid = false;
            } else {
                runningOffset += static_cast<uint64_t>(count);
            }
        }
    }

    __aicore__ inline void BuildPeerSourceOffsets()
    {
        if (subBlockIdx_ != 0U || rankWorkerCount_ == 0U) {
            return;
        }
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        const uint32_t countNum = rankSize_ * expertNum;
        const uint64_t payloadCapacityBytes = winSize_ - Gmma2avMteTiling::CONTROL_REGION_BYTES;
        AscendC::LocalTensor<int32_t> peerCounts = GetControlTensor();
        AscendC::LocalTensor<uint32_t> destinationOffsets = GetDestinationOffsetTensor();
        AscendC::LocalTensor<uint32_t> sourceOffsets = GetSourceOffsetTensor();
        for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
            if (srcRank == rank_) {
                for (uint32_t index = 0U; index < countNum; ++index) {
                    peerCounts.SetValue(index, taskTilingInfo_->sendCnt[index]);
                }
            } else {
                LoadInt32Row(GetCountTable(srcRank), peerCounts, countNum);
            }
            uint64_t expertSourceBase = 0UL;
            bool sourcePrefixValid = true;
            for (uint32_t expert = 0U; expert < expertNum; ++expert) {
                uint64_t sourceOffsetInExpert = 0UL;
                uint64_t expertTotal = 0UL;
                bool expertLayoutValid = sourcePrefixValid;
                for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
                    const int32_t count = peerCounts.GetValue(static_cast<uint64_t>(dstRank) * expertNum + expert);
                    if (count < 0) {
                        expertLayoutValid = false;
                        continue;
                    }
                    uint64_t updated = 0UL;
                    if (dstRank < rank_ && !SafeAdd(sourceOffsetInExpert, static_cast<uint64_t>(count), updated)) {
                        expertLayoutValid = false;
                    } else if (dstRank < rank_) {
                        sourceOffsetInExpert = updated;
                    }
                    if (!SafeAdd(expertTotal, static_cast<uint64_t>(count), updated)) {
                        expertLayoutValid = false;
                    } else {
                        expertTotal = updated;
                    }
                }
                const uint64_t localIndex = static_cast<uint64_t>(srcRank) * expertNum + expert;
                const int32_t peerSendCount = peerCounts.GetValue(static_cast<uint64_t>(rank_) * expertNum + expert);
                const int32_t localRecvCount = taskTilingInfo_->recvCnt[localIndex];
                uint64_t sourceOffset = 0UL;
                bool expertValid = expertLayoutValid && SafeAdd(expertSourceBase, sourceOffsetInExpert, sourceOffset);
                uint64_t sourceEndRows = 0UL;
                uint64_t sourceEndElements = 0UL;
                if (peerSendCount < 0 || localRecvCount < 0 || peerSendCount != localRecvCount ||
                    sourceOffset >= GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
                    destinationOffsets.GetValue(localIndex) == GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
                    !SafeAdd(sourceOffset, static_cast<uint64_t>(peerSendCount), sourceEndRows) ||
                    !SafeMulU32(sourceEndRows, static_cast<uint64_t>(taskTilingInfo_->N1), sourceEndElements) ||
                    sourceEndElements > payloadCapacityBytes / sizeof(T)) {
                    expertValid = false;
                }
                // A source rank is built and consumed by the same AIV worker.
                // Keep its offsets owner-local instead of routing private
                // state through a reused shared-GM scratch whose cross-core
                // visibility and cache lifecycle provide no useful sharing.
                sourceOffsets.SetValue(localIndex, expertValid ? static_cast<uint32_t>(sourceOffset) :
                                                                 GmmA2avMteDetail::INVALID_TOKEN_OFFSET);
                if (!expertLayoutValid || !SafeAdd(expertSourceBase, expertTotal, expertSourceBase)) {
                    sourcePrefixValid = false;
                }
            }
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    __aicore__ inline void PullPeerExpertFromSource(uint32_t srcRank, uint32_t expertIdx)
    {
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        AscendC::LocalTensor<uint32_t> destinationOffsets = GetDestinationOffsetTensor();
        AscendC::LocalTensor<uint32_t> sourceOffsets = GetSourceOffsetTensor();
        const uint64_t index = static_cast<uint64_t>(srcRank) * expertNum + expertIdx;
        const int32_t recvCount = taskTilingInfo_->recvCnt[index];
        const uint32_t sourceOffset = sourceOffsets.GetValue(index);
        const uint32_t destinationOffset = destinationOffsets.GetValue(index);
        if (recvCount <= 0 || sourceOffset == GmmA2avMteDetail::INVALID_TOKEN_OFFSET ||
            destinationOffset == GmmA2avMteDetail::INVALID_TOKEN_OFFSET) {
            return;
        }
        CopyGmToGm(GetOutputData(srcRank) + static_cast<uint64_t>(sourceOffset) * taskTilingInfo_->N1,
                   recvBuffer_ + static_cast<uint64_t>(destinationOffset) * taskTilingInfo_->N1,
                   static_cast<uint64_t>(recvCount) * taskTilingInfo_->N1);
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> GetControlTensor()
    {
        return ubBuffer_.template Get<int32_t>()[GmmA2avMteDetail::CONTROL_UB_OFFSET_BYTES / sizeof(int32_t)];
    }

    __aicore__ inline AscendC::LocalTensor<int64_t> GetSyncTensor()
    {
        return ubBuffer_.template Get<int64_t>()[GmmA2avMteDetail::CONTROL_UB_OFFSET_BYTES / sizeof(int64_t)];
    }

    __aicore__ inline AscendC::LocalTensor<uint32_t> GetDestinationOffsetTensor()
    {
        return ubBuffer_
            .template Get<uint32_t>()[GmmA2avMteDetail::DESTINATION_OFFSET_UB_OFFSET_BYTES / sizeof(uint32_t)];
    }

    __aicore__ inline AscendC::LocalTensor<uint32_t> GetSourceOffsetTensor()
    {
        return ubBuffer_.template Get<uint32_t>()[GmmA2avMteDetail::SOURCE_OFFSET_UB_OFFSET_BYTES / sizeof(uint32_t)];
    }

    __aicore__ static inline bool SafeAdd(uint64_t lhs, uint64_t rhs, uint64_t &result)
    {
        constexpr uint64_t maxValue = ~static_cast<uint64_t>(0U);
        if (lhs > maxValue - rhs) {
            return false;
        }
        result = lhs + rhs;
        return true;
    }

    __aicore__ static inline bool SafeMulU32(uint64_t lhs, uint64_t rhs, uint64_t &result)
    {
        constexpr uint64_t maxU32 = 0xffffffffUL;
        if (lhs > maxU32 || rhs > maxU32) {
            return false;
        }
        // A sanitizer-instrumented 64 x 64 multiply lowers to compiler-rt's
        // __multi3, which is unavailable to arch22 AICore linking. Split the
        // bounded 32 x 32 product into native 16-bit partial products.
        const uint32_t lhs32 = static_cast<uint32_t>(lhs);
        const uint32_t rhs32 = static_cast<uint32_t>(rhs);
        const uint32_t lhsLow = lhs32 & 0xffffU;
        const uint32_t lhsHigh = lhs32 >> 16U;
        const uint32_t rhsLow = rhs32 & 0xffffU;
        const uint32_t rhsHigh = rhs32 >> 16U;
        const uint64_t low = static_cast<uint64_t>(lhsLow * rhsLow);
        const uint64_t middle = static_cast<uint64_t>(lhsLow * rhsHigh) + static_cast<uint64_t>(lhsHigh * rhsLow);
        const uint64_t high = static_cast<uint64_t>(lhsHigh * rhsHigh);
        result = low + (middle << 16U) + (high << 32U);
        return true;
    }

    __aicore__ inline void LoadInt32Row(__gm__ int32_t *source, const AscendC::LocalTensor<int32_t> &local,
                                        uint32_t elementCount)
    {
        const uint32_t alignedElementCount = AlignControlElements(elementCount);
        AscendC::GlobalTensor<int32_t> sourceGlobal;
        sourceGlobal.SetGlobalBuffer(source);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::DataCopy(local, sourceGlobal, alignedElementCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    __aicore__ inline void StoreInt32Row(__gm__ int32_t *destination, const AscendC::LocalTensor<int32_t> &local,
                                         uint32_t elementCount)
    {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::GlobalTensor<int32_t> destinationGlobal;
        destinationGlobal.SetGlobalBuffer(destination);
        AscendC::DataCopy(destinationGlobal, local, elementCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    __aicore__ inline void LocalAivSync()
    {
        // On arch22 SyncAll<true> drains local pipelines and performs an
        // AIV-only inter-core rendezvous. Every AIV, including non-primary
        // subblocks, enters every protocol phase in the same order.
        AscendC::SyncAll<true>();
    }

    __aicore__ inline void PublishAndWaitRankPhase(uint64_t phaseBase)
    {
        LocalAivSync();
        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            for (uint32_t dstRank = logicalCoreIdx_; dstRank < rankSize_; dstRank += rankWorkerCount_) {
                StoreSyncFlag(GetPhaseSlot(dstRank, phaseBase, rank_), syncEpoch_);
            }
        }
        // Do not allow a worker to wait until every outbound slot from this
        // rank has been published; otherwise ranks can form a wait cycle.
        LocalAivSync();
        if (subBlockIdx_ == 0U && rankWorkerCount_ != 0U) {
            for (uint32_t srcRank = logicalCoreIdx_; srcRank < rankSize_; srcRank += rankWorkerCount_) {
                WaitSyncFlag(GetPhaseSlot(rank_, phaseBase, srcRank), syncEpoch_);
            }
        }
        LocalAivSync();
    }

    __aicore__ inline __gm__ int64_t *GetFixedSyncSlot(uint32_t windowRank, uint64_t phaseBase, uint64_t index) const
    {
        const uint64_t syncOffset =
            winSize_ - Gmma2avMteTiling::SYNC_REGION_FROM_TAIL + phaseBase + index * Gmma2avMteTiling::SYNC_SLOT_BYTES;
        return reinterpret_cast<__gm__ int64_t *>(GetWindow(windowRank) + syncOffset);
    }

    __aicore__ inline __gm__ int64_t *GetEpochSlot(uint32_t windowRank) const
    {
        return GetFixedSyncSlot(windowRank, Gmma2avMteTiling::EPOCH_BASE, 0UL);
    }

    __aicore__ inline __gm__ int64_t *GetPhaseSlot(uint32_t windowRank, uint64_t phaseBase,
                                                   uint32_t participantRank) const
    {
        return GetFixedSyncSlot(windowRank, phaseBase, participantRank);
    }

    __aicore__ inline __gm__ int64_t *GetExpertReadySlot(uint32_t windowRank, uint32_t sourceRank,
                                                         uint32_t expertIdx) const
    {
        const uint64_t expertSlot = static_cast<uint64_t>(sourceRank) * taskTilingInfo_->e + expertIdx;
        return GetFixedSyncSlot(windowRank, Gmma2avMteTiling::EXPERT_READY_BASE, expertSlot);
    }

    __aicore__ inline __gm__ int64_t *GetAckSlot(uint32_t windowRank, uint32_t participantRank) const
    {
        return GetPhaseSlot(windowRank, Gmma2avMteTiling::ACK_BASE, participantRank);
    }

    // sendCnt/recvCnt come from the same tiling computation on both ends, so
    // a skipped publish slot is exactly the slot the peer skips waiting on:
    // publish is needed only when this rank actually sends expert e to D, and
    // waiting is needed only when this rank actually receives expert e from
    // S. Expert 0 is always kept unmasked on both sides: its flag is the
    // release/acquire handshake for this rank's count table, which
    // BuildPeerSourceOffsets loads even when no expert-0 rows are exchanged.
    // The self slot is always kept because the local wait doubles as the
    // all-AIV rendezvous for this expert.
    __aicore__ inline bool IsExpertSlotActive(const int32_t *countTable, uint32_t peerRank, uint32_t expertIdx,
                                              uint32_t expertNum) const
    {
        if (expertIdx == 0U || peerRank == rank_) {
            return true;
        }
        return countTable[static_cast<uint64_t>(peerRank) * expertNum + expertIdx] > 0;
    }

    // One batched MTE3 transaction publishes every expert slot of one chunk
    // for one destination rank, replacing one 32-byte store plus DDR barrier
    // per active expert. Inactive slots are written as zero; they are never
    // polled this epoch (the peer's wait mask is identical), and the next
    // epoch's publisher overwrites them before any waiter can observe them.
    __aicore__ inline bool PublishExpertReadyRange(uint32_t dstRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                                   int64_t flag)
    {
        const uint32_t slotCount = endExpertIdx - startExpertIdx;
        const uint32_t totalElements = slotCount * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS;
        if (totalElements > GmmA2avMteDetail::FLAG_STAGING_ELEMENTS) {
            return false;
        }
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        AscendC::LocalTensor<int64_t> staging = GetSyncTensor();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        // The batched copy transfers complete 64-byte slots, including padding
        // and inactive flags. Initialize every element read by that transfer.
        for (uint32_t index = 0U; index < totalElements; ++index) {
            staging.SetValue(index, 0);
        }
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
            if (!IsExpertSlotActive(taskTilingInfo_->sendCnt, dstRank, expertIdx, expertNum)) {
                continue;
            }
            staging.SetValue((expertIdx - startExpertIdx) * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS, flag);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::GlobalTensor<int64_t> global;
        global.SetGlobalBuffer(GetExpertReadySlot(dstRank, rank_, startExpertIdx));
        AscendC::DataCopy(global, staging, totalElements);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        return true;
    }

    __aicore__ inline void PublishExpertReadyPerFlag(uint32_t dstRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                                     int64_t flag)
    {
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
            if (!IsExpertSlotActive(taskTilingInfo_->sendCnt, dstRank, expertIdx, expertNum)) {
                continue;
            }
            StoreSyncFlag(GetExpertReadySlot(dstRank, rank_, expertIdx), flag);
        }
    }

    // One batched MTE2 transaction polls every expert slot of one chunk from
    // one source rank. A single DDR barrier after all active slots hold the
    // expected epoch replaces the per-flag barrier, so a ready chunk costs
    // one MTE round trip instead of one per active (expert, source) pair.
    __aicore__ inline bool WaitExpertReadyRange(uint32_t srcRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                                int64_t expected)
    {
        const uint32_t slotCount = endExpertIdx - startExpertIdx;
        const uint32_t totalElements = slotCount * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS;
        if (totalElements > GmmA2avMteDetail::FLAG_STAGING_ELEMENTS) {
            return false;
        }
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        AscendC::LocalTensor<int64_t> staging = GetSyncTensor();
        AscendC::GlobalTensor<int64_t> global;
        // Sync slots are always polled through this rank's local window. The
        // bypass alias is valid for local GM and prevents a stale cache line
        // after a peer updates the slot; never apply it to encoded peer VAs.
        global.SetGlobalBuffer(GetExpertReadySlot(rank_, srcRank, startExpertIdx));
        global.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (true) {
            AscendC::DataCopy(staging, global, totalElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            bool allReady = true;
            for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
                if (!IsExpertSlotActive(taskTilingInfo_->recvCnt, srcRank, expertIdx, expertNum)) {
                    continue;
                }
                if (staging.GetValue((expertIdx - startExpertIdx) * GmmA2avMteDetail::SYNC_SLOT_ELEMENTS) != expected) {
                    allReady = false;
                    break;
                }
            }
            if (allReady) {
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                return true;
            }
        }
    }

    __aicore__ inline void WaitExpertReadyPerFlag(uint32_t srcRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                                  int64_t expected)
    {
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
        for (uint32_t expertIdx = startExpertIdx; expertIdx < endExpertIdx; ++expertIdx) {
            if (!IsExpertSlotActive(taskTilingInfo_->recvCnt, srcRank, expertIdx, expertNum)) {
                continue;
            }
            WaitSyncFlag(GetExpertReadySlot(rank_, srcRank, expertIdx), expected);
        }
    }

    __aicore__ inline void StoreSyncFlag(__gm__ int64_t *destination, int64_t flag)
    {
        AscendC::LocalTensor<int64_t> local = GetSyncTensor();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        local.SetValue(0U, flag);
        for (uint32_t index = 1U; index < GmmA2avMteDetail::SYNC_MTE_ELEMENTS; ++index) {
            local.SetValue(index, 0);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
        AscendC::GlobalTensor<int64_t> global;
        global.SetGlobalBuffer(destination);
        // A3's established remote-state protocols publish one complete 32-byte
        // block.  Keeping the transfer naturally aligned also avoids issuing a
        // short MTE transaction against a peer mapping.
        AscendC::DataCopy(global, local, GmmA2avMteDetail::SYNC_MTE_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    __aicore__ inline bool IsPrimaryAiv() const
    {
        return logicalCoreIdx_ == 0U && subBlockIdx_ == 0U;
    }

    __aicore__ inline void WaitSyncFlag(__gm__ int64_t *source, int64_t expected)
    {
        AscendC::LocalTensor<int64_t> local = GetSyncTensor();
        AscendC::GlobalTensor<int64_t> global;
        global.SetGlobalBuffer(source);
        // Sync slots are always polled through this rank's local window.  The
        // bypass alias is valid for local GM and prevents a stale cache line
        // after a peer updates the slot; never apply it to encoded peer VAs.
        global.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (true) {
            AscendC::DataCopy(local, global, GmmA2avMteDetail::SYNC_MTE_ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
            if (local.GetValue(0U) == expected) {
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                break;
            }
        }
    }

    __aicore__ inline uint32_t AlignControlElements(uint32_t elementCount) const
    {
        constexpr uint32_t alignment = GmmA2avMteDetail::CONTROL_MTE_ELEMENTS;
        return (elementCount + alignment - 1U) / alignment * alignment;
    }

    __aicore__ inline void CopyGmToGm(__gm__ T *source, __gm__ T *destination, uint64_t elementCount)
    {
        uint32_t moveIdx = 0U;
        BeginGmToGmCopy();
        CopyGmToGmSegment(source, destination, elementCount, moveIdx);
        EndGmToGmCopy();
    }

    __aicore__ inline void BeginGmToGmCopy()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void CopyGmToGmSegment(__gm__ T *source, __gm__ T *destination, uint64_t elementCount,
                                             uint32_t &moveIdx)
    {
        AscendC::LocalTensor<T> local = ubBuffer_.template Get<T>();
        AscendC::LocalTensor<T> ping = local;
        AscendC::LocalTensor<T> pong = local[GmmA2avMteDetail::PONG_UB_OFFSET_BYTES / sizeof(T)];
        uint64_t copied = 0UL;
        while (copied < elementCount) {
            const uint32_t current = static_cast<uint32_t>(
                elementCount - copied > ubMoveElements_ ? ubMoveElements_ : elementCount - copied);
            const auto eventId = (moveIdx & 1U) != 0U ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> copyTensor = (moveIdx & 1U) != 0U ? ping : pong;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            AscendC::GlobalTensor<T> sourceGlobal;
            sourceGlobal.SetGlobalBuffer(source + copied);
            AscendC::DataCopyExtParams copyParams(1U, current * sizeof(T), 0U, 0U, 0U);
            AscendC::DataCopyPadExtParams<T> padParams;
            AscendC::DataCopyPad(copyTensor, sourceGlobal, copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            AscendC::GlobalTensor<T> destinationGlobal;
            destinationGlobal.SetGlobalBuffer(destination + copied);
            AscendC::DataCopyPad(destinationGlobal, copyTensor, copyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            copied += current;
            ++moveIdx;
        }
    }

    __aicore__ inline void EndGmToGmCopy()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

private:
    const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo_{nullptr};
    __gm__ HcclOpResParam *a3WinContext_{nullptr};
    __gm__ AscendC::HcclCombineOpParam *a2WinContext_{nullptr};
    __gm__ T *sendBuffer_{nullptr};
    __gm__ T *recvBuffer_{nullptr};
    uint64_t winSize_{0UL};
    uint64_t localExpertBaseRows_{0UL};
    uint32_t rank_{0U};
    uint32_t rankSize_{0U};
    bool isA3_{false};
    uint32_t coreIdx_{0U};
    uint32_t subBlockIdx_{0U};
    int64_t syncEpoch_{1};
    uint32_t logicalCoreIdx_{0U};
    uint32_t rankWorkerCount_{0U};
    uint32_t ubMoveElements_{GmmA2avMteDetail::DEFAULT_UB_MOVE_ELEMENTS};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuffer_;
};

template <typename CommOpType, typename ComputationOpType, typename SharedComputationOpType, bool IsNeedMM>
class GmmA2avMteScheduler {
private:
    // A3 CrossCore set-count allows at most 15 outstanding notifications.
    // Keep this explicit: the generic CATLASS default is 16 on this branch.
    using ExpertReadyFlag = Catlass::Arch::CrossCoreFlagWithReverse<15U>;

    struct AicExpertChunkReadyCallback {
        __aicore__ inline AicExpertChunkReadyCallback(ExpertReadyFlag &readyFlag,
                                                      const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo,
                                                      uint32_t expertChunkRows)
            : readyFlag_(readyFlag),
              taskTilingInfo_(taskTilingInfo),
              expertChunkRows_(expertChunkRows)
        {}

        // Returns true when this expert closes a chunk; the caller then drains
        // its FIX pipe and invokes NotifyChunkReady(). Boundary arithmetic
        // mirrors the AIV-side GetExpertChunkEnd exactly (same
        // positive-rows-only sendCnt prefix), and the final expert always
        // closes the last chunk, so the number of notifications equals the
        // number of AIV waits.
        __aicore__ inline bool operator()(uint32_t expertIdx)
        {
            chunkRows_ += GetExpertComputeRows(taskTilingInfo_, expertIdx);
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo_->e);
            if ((expertChunkRows_ == 0U || chunkRows_ < expertChunkRows_) && expertIdx + 1U != expertNum) {
                return false;
            }
            chunkRows_ = 0UL;
            return true;
        }

        // Every AIC first drains the same expert batch and rendezvouses with
        // all other AICs. Only then does each AIC notify its paired AIV.
        __aicore__ inline void NotifyChunkReady()
        {
            AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(AIC_ALL_READY_FLAG_IDX);
            AscendC::CrossCoreWaitFlag(AIC_ALL_READY_FLAG_IDX);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(readyFlag_);
        }

        ExpertReadyFlag &readyFlag_;
        const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo_;
        uint32_t expertChunkRows_;
        uint64_t chunkRows_{0UL};
    };

public:
    static __aicore__ inline void ProcessAic(ComputationOpType &computeOp, SharedComputationOpType &sharedComputeOp,
                                             const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo,
                                             uint32_t expertChunkRows)
    {
        if ASCEND_IS_AIC {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
            ExpertReadyFlag readyFlag(EXPERT_READY_FLAG_IDX, EXPERT_REVERSE_FLAG_IDX);
            AicExpertChunkReadyCallback callback(readyFlag, taskTilingInfo, expertChunkRows);
            computeOp.ProcessExperts(0U, expertNum, callback);

            if constexpr (IsNeedMM) {
                sharedComputeOp.Process(0U, 1U);
            }
            AscendC::SyncAll<false>();
        }
    }

    static __aicore__ inline void ProcessAiv(CommOpType &commOp,
                                             const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo,
                                             uint32_t expertChunkRows)
    {
        if ASCEND_IS_AIV {
            const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
            ExpertReadyFlag readyFlag(EXPERT_READY_FLAG_IDX, EXPERT_REVERSE_FLAG_IDX);
            commOp.BeginExpertPipeline();
            uint32_t expertIdx = 0U;
            while (expertIdx < expertNum) {
                // Consecutive experts are batched into one synchronization
                // stage until their cumulative local GMM rows reach the
                // tiling-provided threshold. Use sendCnt, as the AIC does
                // for each expert's M dimension. recvCnt instead describes
                // the output routing and can concentrate all rows on one
                // expert, delaying publication of unrelated local work.
                // Different ranks may still disagree, which is safe
                // because ExpertReady flags stay
                // expert-granular and per-rank chunks are monotone in
                // expert order.
                const uint32_t chunkEnd = GetExpertChunkEnd(taskTilingInfo, expertIdx, expertChunkRows);
                // The AIC notifies only at chunk boundaries, so a single
                // C2V wait (plus its reverse-credit release) covers every
                // expert in [expertIdx, chunkEnd).
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(readyFlag);
                commOp.StageChunkPayload(expertIdx, chunkEnd);
                commOp.PublishChunkReady(expertIdx, chunkEnd);
                commOp.WaitChunkReady(expertIdx, chunkEnd);
                commOp.PullExpertChunk(expertIdx, chunkEnd);
                expertIdx = chunkEnd;
            }

            commOp.End();
        }
    }

private:
    static __aicore__ inline uint64_t GetExpertComputeRows(
        const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo, uint32_t expertIdx)
    {
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
        const uint32_t rankSize = static_cast<uint32_t>(taskTilingInfo->epWorldSize);
        uint64_t rows = 0UL;
        for (uint32_t dstRank = 0U; dstRank < rankSize; ++dstRank) {
            const int32_t count = taskTilingInfo->sendCnt[static_cast<uint64_t>(dstRank) * expertNum + expertIdx];
            if (count > 0) {
                rows += static_cast<uint64_t>(count);
            }
        }
        return rows;
    }

    static __aicore__ inline uint32_t GetExpertChunkEnd(const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo,
                                                        uint32_t startExpertIdx, uint32_t expertChunkRows)
    {
        const uint32_t expertNum = static_cast<uint32_t>(taskTilingInfo->e);
        uint64_t chunkRows = 0UL;
        for (uint32_t expertIdx = startExpertIdx; expertIdx < expertNum; ++expertIdx) {
            chunkRows += GetExpertComputeRows(taskTilingInfo, expertIdx);
            // Zero selects one chunk including trailing empty experts.
            // A chunk always contains at least one expert; a threshold of 1
            // restores the legacy per-expert pipeline (modulo grouping runs
            // of experts with no local GMM rows, which carry no data).
            if (expertChunkRows != 0U && chunkRows >= expertChunkRows) {
                return expertIdx + 1U;
            }
        }
        return expertNum;
    }
    static constexpr Catlass::Arch::FlagID EXPERT_READY_FLAG_IDX = 0U;
    static constexpr Catlass::Arch::FlagID EXPERT_REVERSE_FLAG_IDX = 1U;
    static constexpr Catlass::Arch::FlagID AIC_ALL_READY_FLAG_IDX = 9U;
    static_assert(EXPERT_READY_FLAG_IDX <= 15U && EXPERT_REVERSE_FLAG_IDX <= 15U && AIC_ALL_READY_FLAG_IDX <= 15U,
                  "arch22 local CrossCore flag IDs must be in [0, 15]");
    static_assert(EXPERT_READY_FLAG_IDX != EXPERT_REVERSE_FLAG_IDX && EXPERT_READY_FLAG_IDX != AIC_ALL_READY_FLAG_IDX &&
                      EXPERT_REVERSE_FLAG_IDX != AIC_ALL_READY_FLAG_IDX,
                  "expert ready, reverse-credit and all-AIC barrier flags must be distinct");
};

} // namespace MC2KernelTemplate

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_MTE_H
