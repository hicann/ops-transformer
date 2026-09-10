/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_H

#include "../generic_block_sparse_attention_kernel_common.hpp"
#include "../generic_block_sparse_attention_metadata_kernel.h"
#include "../generic_block_sparse_attention_fd_utils.h"
#include "generic_block_sparse_attention_kernel_utils.hpp"
#include "generic_block_sparse_attention_fd_combine_arch22.h"

using namespace NpuArch;

namespace GsaKernelArch22 {

template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO>
class GsaRegularKernelArch22 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename EpilogueRescaleO::ElementOutput;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLse = layout::RowMajor;
    using LayoutUpdate = layout::RowMajor;

    static constexpr bool ANTIQUANT = !std::is_same<ElementQ, ElementO>::value;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = 131072;
    static constexpr uint32_t QUERY_READY_ID = 4;
    static constexpr uint32_t QK_READY_ID = 1;
    static constexpr uint32_t SOFTMAX_READY_ID = 2;
    static constexpr uint32_t PV_READY_ID = 3;
    // A depth-one W8A8 pipeline leaves stage 1 in flight while stage 0 is
    // consumed. Keep the in-flight stage on separate flags and do not reuse a
    // three-slot workspace until both AIV sub-blocks release it.
    static constexpr uint32_t WORKSPACE_READY_BASE_ID = 5;
    static constexpr uint32_t QK_READY_STAGE1_ID = 8;
    static constexpr uint32_t SOFTMAX_READY_STAGE1_ID = 9;
    static constexpr uint32_t PV_READY_STAGE1_ID = 10;

    __aicore__ inline GsaRegularKernelArch22() {}

    __aicore__ inline void operator()(GsaKernelParamsArch22 const &params)
    {
        __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData =
            reinterpret_cast<__gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *>(params.tiling);
        FetchTilingData(tilingData, params.metaData);
        InitAntiquant(params);
        __gm__ const GsaMetadata::Metadata *meta =
            reinterpret_cast<__gm__ const GsaMetadata::Metadata *>(params.metaData);
        const uint32_t physicalAicNum = AscendC::GetBlockNum();
        if (!GsaFd::ValidateMetadata(meta, tilingData, physicalAicNum)) {
            return;
        }
        const bool fdEnabled = tilingData->fdStaticEnabled != 0U &&
                               (static_cast<uint32_t>(meta->fdScheduleFlags) & GsaMetadata::FD_SCHEDULE_ENABLED) != 0U;
        // The metadata FD schedule owns base tasks and may split their KV
        // ranges across cores.  Keep its task ids unsplit; the regular path
        // alone uses the optional query-head expansion below.
        if (fdEnabled && headSplitFactor_ > 1U) {
            headSplitFactor_ = 1U;
            totalTaskNum_ = static_cast<uint32_t>(meta->saTotalTaskNum);
            groupSize_ = tilingData->groupSize;
        }

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx;
        gSparseBlockIdx.SetGlobalBuffer((__gm__ int32_t *)params.sparseBlockIdx);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)params.blockTable);
        AscendC::GlobalTensor<int32_t> gSparseBlockCount;
        gSparseBlockCount.SetGlobalBuffer((__gm__ int32_t *)params.sparseBlockCount);
        AscendC::GlobalTensor<int64_t> gCuSeqLengths;
        if (params.cuSeqLengths != nullptr) {
            gCuSeqLengths.SetGlobalBuffer((__gm__ int64_t *)params.cuSeqLengths);
        }
        AscendC::GlobalTensor<int64_t> gCuSeqLengthsKv;
        if (params.cuSeqLengthsKv != nullptr) {
            gCuSeqLengthsKv.SetGlobalBuffer((__gm__ int64_t *)params.cuSeqLengthsKv);
        }
        AscendC::GlobalTensor<int32_t> gSequsedQ;
        const bool hasSequsedQ = (params.sequsedQ != nullptr);
        if (hasSequsedQ) {
            gSequsedQ.SetGlobalBuffer((__gm__ int32_t *)params.sequsedQ);
        }
        AscendC::GlobalTensor<int32_t> gSequsedKv;
        const bool hasSequsedKv = (params.sequsedKv != nullptr);
        if (hasSequsedKv) {
            gSequsedKv.SetGlobalBuffer((__gm__ int32_t *)params.sequsedKv);
        }
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<float> gLse;
        gLse.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        AscendC::GlobalTensor<float> gPartialLse;
        gPartialLse.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialLseOffset));
        AscendC::GlobalTensor<float> gPartialO;
        gPartialO.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialOOffset));

        // Workspace: [queryPreProcessResGm(antiquant only)][S][P][OTmp][OUpdate][identity][amaxQ][amaxP].
        // Identity must not precede gS — Fixpipe into GM immediately after identity was flaky on arch22.
        uint64_t antiqQueryOffset = 0;
        if constexpr (ANTIQUANT) {
            antiqQueryOffset = queryPreProcessSize_;
            queryPreProcessResGm.SetGlobalBuffer((__gm__ int8_t *)params.workSpace);
        }
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace + antiqQueryOffset));
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + antiqQueryOffset + mm1OutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer(
            (__gm__ ElementOTmp *)(params.workSpace + antiqQueryOffset + mm1OutSize_ + smOnlineOutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer(
            (__gm__ ElementOTmp *)(params.workSpace + antiqQueryOffset + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_));
        AscendC::GlobalTensor<int32_t> gIdentityIdx;
        gIdentityIdx.SetGlobalBuffer((__gm__ int32_t *)(params.workSpace + antiqQueryOffset + mm1OutSize_ +
                                                        smOnlineOutSize_ + mm2OutSize_ + updateSize_));
        // amaxQOffset: base of Amax_Q region (per-core offset applied later in VEC section)
        uint64_t amaxQOffset = antiqQueryOffset + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_ +
                               static_cast<uint64_t>(topK_) * sizeof(int32_t);
        // amaxPOffset: base of Amax_P_v region (per-core offset applied later in VEC section)
        uint64_t amaxPOffset = amaxQOffset + amaxQSize_;
        (void)amaxQOffset;
        (void)amaxPOffset;

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        // Keep query and probability expansion on the same supported MSD
        // width.  MSD1 is not used: it saves traffic at the cost of a
        // material precision loss, especially on long-KV decode.
        const uint32_t pMsdIterNum = ANTIQUANT ? msdIterNum_ : 1U;

#ifdef __DAV_C220_CUBE__
        // Initialize Cube core hardware events
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ) +
            BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K * sizeof(ElementK) * 2;
        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
#endif

#ifdef __DAV_C220_VEC__
        // Initialize hardware events for vector core
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
        EpilogueRescaleO epilogueRescaleO(resource);

        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        // 伪量化：设置 gAmaxQ/gAmaxP（per-core offset）并传递 antiquant 参数到 epilogue
        if constexpr (ANTIQUANT) {
            // amaxQ per-core size: groupSize * 8 * sizeof(float)
            uint64_t amaxQPerCoreSize = static_cast<uint64_t>(groupSize_) * 8 * sizeof(float);
            gAmaxQ.SetGlobalBuffer(
                (__gm__ float *)(params.workSpace + amaxQOffset + static_cast<uint64_t>(coreIdx) * amaxQPerCoreSize));
            // amaxP per-core size: groupSize * 8 * sizeof(float) * MAX_CROSS_CORE_BUF_STAGES (3 stages)
            uint64_t amaxPPerCoreSize =
                static_cast<uint64_t>(groupSize_) * 8 * sizeof(float) * MAX_CROSS_CORE_BUF_STAGES;
            gAmaxP.SetGlobalBuffer(
                (__gm__ float *)(params.workSpace + amaxPOffset + static_cast<uint64_t>(coreIdx) * amaxPPerCoreSize));
            epilogueOnlineSoftmax.SetAntiquantParams(msdIterNum_, pMsdIterNum, groupSize_);
            epilogueOnlineSoftmax.SetAntiquantScaleGm(keyAntiqScaleGm, valueAntiqScaleGm, gAmaxQ, gAmaxP);
            epilogueRescaleO.SetAntiquantParams(pMsdIterNum, groupSize_);
            epilogueRescaleO.SetAntiquantScaleGm(gAmaxP);
        }
#endif

#ifdef __DAV_C220_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        // Only select slot 0 is ever consumed: the per-page protocol reads
        // gSelectIdx.GetValue(0) (nBlockOffset / y == 0 for a single-page
        // tile), and the stacked fast path takes physical ids from
        // validPhysicalIds instead.  Scalar GM stores serialise on the AIC
        // critical path, so the antiquant launch publishes slot 0 only.
        if constexpr (ANTIQUANT) {
            gIdentityIdx.SetValue(0, 0);
        } else {
            gIdentityIdx.SetValue(0, 0);
            for (uint32_t i = 1; i < topK_; i++) {
                gIdentityIdx.SetValue(i, 0);
            }
        }
        // Every CUBE block publishes the same identity sequence.  A local
        // pipe barrier is sufficient for its own subsequent QK reads and
        // avoids making AIV sub-blocks participate in a global rendezvous.
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
        // The exact Q8 W8A8 target has one split task per AIC.  IdentityIdx
        // is written with the same values by every CUBE, so the global
        // rendezvous is unnecessary on this balanced schedule.  Keep the
        // barrier for all generic and FD schedules.
        const bool exactQ8RegularShape = ANTIQUANT && !fdEnabled && batch_ == 1U && qBlockNum_ == 8U &&
                                         static_cast<uint32_t>(meta->saTotalTaskNum) == 8U &&
                                         totalTaskNum_ == coreNum && totalTaskNum_ == 8U && qHeads_ == 16U &&
                                         kvHeads_ == 1U && maxQSeqlen_ == 8U && groupSize_ == 16U && embed_ == 128U &&
                                         blockSize_ == 128U && topK_ == 16U;
        const bool exactQ1RegularShape = ANTIQUANT && !fdEnabled && batch_ == 1U && qBlockNum_ == 1U &&
                                         static_cast<uint32_t>(meta->saTotalTaskNum) == 1U && qHeads_ == 16U &&
                                         kvHeads_ == 1U && maxQSeqlen_ == 1U && embed_ == 128U && blockSize_ == 128U &&
                                         topK_ == 4U;
        const bool exactQ1TopK16Shape = ANTIQUANT && !fdEnabled && batch_ == 1U && qBlockNum_ == 1U &&
                                        static_cast<uint32_t>(meta->saTotalTaskNum) == 1U && qHeads_ == 16U &&
                                        kvHeads_ == 1U && maxQSeqlen_ == 1U && groupSize_ == 16U && embed_ == 128U &&
                                        blockSize_ == 128U && topK_ == 16U;
        const bool exactQStackShape = ANTIQUANT && !fdEnabled && batch_ == 1U && qBlockNum_ >= 1U &&
                                      qBlockNum_ <= 16U && static_cast<uint32_t>(meta->saTotalTaskNum) == qBlockNum_ &&
                                      qHeads_ == 16U && kvHeads_ == 1U && maxQSeqlen_ == qBlockNum_ &&
                                      groupSize_ == 16U && embed_ == 128U && blockSize_ == 128U && topK_ == 16U;
        // qSeq=1 GQA has one independent base task per KV head.  It can use
        // the same full-page stacked QK/PV schedule as MQA, while the generic
        // task decoder retains the per-task kvHeadIdx for K/V addressing.
        const bool exactQ1GqaStackShape = ANTIQUANT && !fdEnabled && batch_ == 1U && qBlockNum_ == 1U &&
                                          static_cast<uint32_t>(meta->saTotalTaskNum) == kvHeads_ && kvHeads_ >= 2U &&
                                          kvHeads_ <= 4U && qHeads_ == kvHeads_ * 16U && maxQSeqlen_ == 1U &&
                                          groupSize_ == 16U && embed_ == 128U && blockSize_ == 128U && topK_ == 16U;
        // The qHeads=8/groupSize=8 decode case has the same one-task,
        // full-page workload, but is not covered by the groupSize=16 path.
        const bool exactQ8Q1StackShape = ANTIQUANT && !fdEnabled && batch_ == 1U && qBlockNum_ == 1U &&
                                         static_cast<uint32_t>(meta->saTotalTaskNum) == 1U && qHeads_ == 8U &&
                                         kvHeads_ == 1U && maxQSeqlen_ == 1U && groupSize_ == 8U && embed_ == 128U &&
                                         blockSize_ == 128U && topK_ == 16U;
        // The qHeads=64/groupSize=16/qSeq=16 case has 64 base tasks and the
        // same 16 selected pages per task as the proven stacked schedule.
        // Keep this exact shape isolated from other multi-head schedules.
        const bool exactQ64Q16StackShape = ANTIQUANT && !fdEnabled && batch_ == 1U && qBlockNum_ == 16U &&
                                           static_cast<uint32_t>(meta->saTotalTaskNum) == 64U && totalTaskNum_ == 64U &&
                                           qHeads_ == 64U && kvHeads_ == 4U && maxQSeqlen_ == 16U &&
                                           groupSize_ == 16U && embed_ == 128U && blockSize_ == 128U && topK_ == 16U;
        // FD keeps the base-task ownership but splits the selected pages over
        // several cores.  The target shape has full pages in each split, so
        // it can use the same stacked-page QK/PV kernels without changing the
        // partial softmax semantics.  Causal/partial pages are rejected below.
        const bool exactQ8FdShape = ANTIQUANT && fdEnabled && batch_ == 1U && qBlockNum_ == 8U &&
                                    static_cast<uint32_t>(meta->saTotalTaskNum) == 8U && qHeads_ == 16U &&
                                    kvHeads_ == 1U && maxQSeqlen_ == 8U && groupSize_ == 8U && embed_ == 128U &&
                                    blockSize_ == 128U && topK_ == 16U;
        // Every antiquant Cube writes the same identity sequence and the
        // task-scoped Q handoff already orders each producer/consumer pair.
        // A kernel-wide rendezvous is therefore unnecessary for the
        // under-filled schedules (notably qSeq=1), where it otherwise adds a
        // fixed device-side wait before any useful work starts.
        const bool usePerTaskQueryHandshake =
            ANTIQUANT && (exactQ8RegularShape || exactQStackShape || exactQ1GqaStackShape || exactQ8Q1StackShape ||
                          exactQ64Q16StackShape || totalTaskNum_ != coreNum ||
                          (static_cast<uint32_t>(meta->saTotalTaskNum) == 1U && !fdEnabled) || fdEnabled);
        const bool skipInitialSync = exactQ8RegularShape || exactQStackShape || exactQ1GqaStackShape ||
                                     exactQ8Q1StackShape || exactQ64Q16StackShape ||
                                     (ANTIQUANT && !fdEnabled &&
                                      (totalTaskNum_ != coreNum || static_cast<uint32_t>(meta->saTotalTaskNum) == 1U));
        if (!skipInitialSync) {
            AscendC::SyncAll<false>();
        }

        uint32_t groupSize = groupSize_;
        int64_t strideQO = qHeads_ * embed_;
        int64_t strideKVRow = kvHeads_ * embed_;
        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t rowNumRound = RoundUp(groupSize, 16);
        // MSD expands the cube M dimension into one contiguous row block per
        // residual digit.  The vector epilogues continue to consume the
        // original group rows and combine the expanded segments themselves.
        uint32_t mmRowNum = ANTIQUANT ? groupSize * msdIterNum_ : groupSize;
        uint32_t mmRowNumRound = RoundUp(mmRowNum, 16);
        uint32_t pvMmRowNum = ANTIQUANT ? groupSize * pMsdIterNum : groupSize;
        uint32_t pvMmRowNumRound = RoundUp(pvMmRowNum, 16);

#ifdef __DAV_C220_VEC__
        // The scheduler omits storage padding from its packed task space.
        // Initialize those rows explicitly to avoid propagating stale GM data.
        if (hasSequsedQ) {
            uint32_t paddingTask = 0U;
            for (uint32_t batchIdx = 0U; batchIdx < batch_; ++batchIdx) {
                const uint32_t storageStart = static_cast<uint32_t>(gCuSeqLengths.GetValue(batchIdx));
                const uint32_t storageEnd = static_cast<uint32_t>(gCuSeqLengths.GetValue(batchIdx + 1U));
                const uint32_t storageLen = storageEnd - storageStart;
                uint32_t actualLen = static_cast<uint32_t>(gSequsedQ.GetValue(batchIdx));
                actualLen = actualLen < storageLen ? actualLen : storageLen;
                for (uint32_t token = actualLen; token < storageLen; ++token) {
                    for (uint32_t kvHeadIdx = 0U; kvHeadIdx < kvHeads_; ++kvHeadIdx) {
                        for (uint32_t splitIdx = 0U; splitIdx < headSplitFactor_; ++splitIdx, ++paddingTask) {
                            if (paddingTask % coreNum != coreIdx) {
                                continue;
                            }
                            const uint32_t qStorageToken = storageStart + token;
                            const uint32_t qHeadStart = kvHeadIdx * groupSize * headSplitFactor_ + splitIdx * groupSize;
                            const uint64_t gmOffsetO =
                                (static_cast<uint64_t>(qStorageToken) * qHeads_ + qHeadStart) * embed_;
                            const uint64_t gmOffsetLse = static_cast<uint64_t>(qStorageToken) * qHeads_ + qHeadStart;
                            epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
                        }
                    }
                }
            }
        }
#endif

        uint32_t taskLoopStart = coreIdx;
        uint32_t taskLoopEnd = totalTaskNum_;
        uint32_t taskLoopStep = coreNum;
        // AIV preprocessing and CUBE QK loading need an ordering edge.  With
        // one task per launched block, every participating block reaches a
        // single global barrier after preprocessing.  Once a block owns more
        // than one task (or FD assigns an irregular range), that barrier is no
        // longer balanced, so use the task-scoped producer/consumer token.
        // A single-head launch has one base task and one physical AIC block.
        // The fixed-count SyncAll path is fragile in that degenerate schedule:
        // the AIV producer and CUBE consumer can observe different stage
        // counts when the qN split leaves AIV0 idle.  Use the task-scoped
        // producer/consumer token for this case as well.
        uint32_t scheduleFirstBlock = 0U;
        uint32_t scheduleLastBlock = 0U;
        if (fdEnabled) {
            taskLoopStart = totalTaskNum_;
            taskLoopEnd = totalTaskNum_;
            taskLoopStep = 1U;
            if (coreIdx < static_cast<uint32_t>(meta->fdActiveCoreNum)) {
                const __gm__ GsaMetadata::DecodeSchedule &schedule = meta->decodeSchedules[coreIdx];
                taskLoopStart = static_cast<uint32_t>(schedule.baseTaskStart);
                taskLoopEnd = static_cast<uint32_t>(schedule.baseTaskEnd);
                scheduleFirstBlock = static_cast<uint32_t>(schedule.firstBlockStart);
                scheduleLastBlock = static_cast<uint32_t>(schedule.lastBlockEnd);
            }
        }

        for (uint32_t taskIdx = taskLoopStart; taskIdx < taskLoopEnd; taskIdx += taskLoopStep) {
            // Metadata task ids describe packed GQA base tasks.  A regular
            // W8A8 head split expands each base task into independent query
            // head owners while preserving the original KV-head/index map.
            const uint32_t baseTaskIdx = taskIdx / headSplitFactor_;
            const uint32_t splitIdx = taskIdx % headSplitFactor_;
            uint32_t rawBegin = 0U;
            uint32_t rawEnd = topK_;
            uint32_t fdPartialTaskId = 0U;
            uint32_t fdPartialCount = 0U;
            const bool isFdPartial =
                fdEnabled && GsaFd::FindPartialTask(meta, baseTaskIdx, coreIdx, fdPartialTaskId, fdPartialCount);
            if (fdEnabled) {
                rawBegin = baseTaskIdx == taskLoopStart ? scheduleFirstBlock : 0U;
                rawEnd = baseTaskIdx + 1U == taskLoopEnd ? scheduleLastBlock : topK_;
            }

            uint32_t qStorageToken = 0U;
            uint32_t qTokenInBatch = 0U;
            uint32_t batchIdx = 0U;
            uint32_t kvHeadIdx = 0U;
            bool taskStorageValid = true;
            if (exactQ8RegularShape || exactQ1RegularShape || exactQ1TopK16Shape) {
                // The exact TND shape has one token per q block and one KV
                // head.  Metadata task ids already map directly to the
                // packed token, so avoid the general batch-prefix decoder.
                qStorageToken = baseTaskIdx;
                qTokenInBatch = baseTaskIdx;
                batchIdx = 0U;
                kvHeadIdx = 0U;
            } else {
                taskStorageValid =
                    GsaFd::DecodeTaskStorage(baseTaskIdx, kvHeads_, batch_, gCuSeqLengths, gSequsedQ, hasSequsedQ,
                                             qStorageToken, qTokenInBatch, batchIdx, kvHeadIdx);
            }
            if (!taskStorageValid) {
                if constexpr (ANTIQUANT) {
                    if (!usePerTaskQueryHandshake && !fdEnabled) {
                        AscendC::SyncAll<false>();
                    }
                }
                continue;
            }
            uint32_t qHeadStart = kvHeadIdx * groupSize * headSplitFactor_ + splitIdx * groupSize;

            uint32_t kvStorageLen = static_cast<uint32_t>(gCuSeqLengthsKv.GetValue(static_cast<int64_t>(batchIdx + 1)) -
                                                          gCuSeqLengthsKv.GetValue(static_cast<int64_t>(batchIdx)));
            uint32_t qStorageLen = static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx + 1)) -
                                                         gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx)));
            uint32_t kvSeqlen = hasSequsedKv ?
                                    static_cast<uint32_t>(gSequsedKv.GetValue(static_cast<int64_t>(batchIdx))) :
                                    kvStorageLen;
            uint32_t qSeqlen =
                hasSequsedQ ? static_cast<uint32_t>(gSequsedQ.GetValue(static_cast<int64_t>(batchIdx))) : qStorageLen;
            int64_t gmOffsetQ =
                static_cast<int64_t>(qStorageToken) * strideQO + static_cast<int64_t>(qHeadStart) * embed_;
            int64_t gmOffsetO = gmOffsetQ;
            const int64_t gmOffsetLse = static_cast<int64_t>(qStorageToken) * qHeads_ + qHeadStart;

            if (qSeqlen == 0U || kvSeqlen == 0U) {
#ifdef __DAV_C220_VEC__
                if (isFdPartial) {
                    epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId, groupSize, embed_,
                                                         tilingData->fdLseSubStride);
                } else {
                    epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
                }
#endif
                if constexpr (ANTIQUANT) {
                    if (!usePerTaskQueryHandshake && !fdEnabled) {
                        AscendC::SyncAll<false>();
                    }
                }
                continue;
            }

            // TND + isPackedGQA=1: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
            // totalQBlocks spans storage (cu) blocks; align with metadata qStorageBlockStarts.
            uint32_t globalQBlock = 0U;
            if (exactQ8RegularShape || exactQ1RegularShape || exactQ1TopK16Shape) {
                globalQBlock = baseTaskIdx;
            } else {
                for (uint32_t b = 0; b < batchIdx; ++b) {
                    uint32_t qLen = static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(b + 1)) -
                                                          gCuSeqLengths.GetValue(static_cast<int64_t>(b)));
                    globalQBlock += (qLen + blockShapeX_ - 1) / blockShapeX_;
                }
                globalQBlock += qTokenInBatch / blockShapeX_;
            }
            int64_t sparseIdxBase =
                static_cast<int64_t>(kvHeadIdx) * qBlockNum_ * topK_ + static_cast<int64_t>(globalQBlock) * topK_;
            uint32_t validTopK = topK_;
            if (params.sparseBlockCount != nullptr) {
                // sparseBlockCount 2D: [N_kv, totalQBlocks]
                int64_t countOffset = static_cast<int64_t>(kvHeadIdx) * qBlockNum_ + static_cast<int64_t>(globalQBlock);
                validTopK = static_cast<uint32_t>(gSparseBlockCount.GetValue(countOffset));
            }
            // The stacked-pages fast path consumes a fixed topK_ of selected
            // blocks without consulting sparseBlockCount.  Only take it when
            // this Q block really selected a full topK_ set; otherwise the
            // stale/unselected sparseBlockIdx entries would compute garbage
            // for tokens whose select_num is smaller than topK_ (or zero).
            const bool exactQ8FullPages = exactQ8RegularShape && !hasSequsedQ && !hasSequsedKv &&
                                          qStorageLen == maxQSeqlen_ && kvStorageLen % blockShapeY_ == 0U &&
                                          kStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                          vStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                          validTopK == topK_;
            const bool exactQ1FullPages = exactQ1RegularShape && !hasSequsedQ && !hasSequsedKv &&
                                          qStorageLen == maxQSeqlen_ && kvStorageLen % blockShapeY_ == 0U &&
                                          kStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                          vStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                          validTopK == topK_;
            const bool exactQStackFullPages = exactQStackShape && !hasSequsedQ && !hasSequsedKv &&
                                              qStorageLen == maxQSeqlen_ && kvStorageLen % blockShapeY_ == 0U &&
                                              kStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                              vStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                              validTopK == topK_;
            const bool exactQ1GqaStackFullPages = exactQ1GqaStackShape && !hasSequsedQ && !hasSequsedKv &&
                                                  qStorageLen == maxQSeqlen_ && kvStorageLen % blockShapeY_ == 0U &&
                                                  kStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                                  vStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                                  validTopK == topK_;
            if (fdEnabled) {
                rawBegin = rawBegin < validTopK ? rawBegin : validTopK;
                rawEnd = rawEnd < validTopK ? rawEnd : validTopK;
            } else {
                rawEnd = validTopK;
            }

            uint32_t historyLen = kvSeqlen - qSeqlen;
            uint32_t lastBlockTileSize = (historyLen + qTokenInBatch) % blockShapeY_ + 1;

            uint32_t kvSLoopNum = rawEnd - rawBegin;
            int32_t validPhysicalIds[16];
            uint32_t validTileSize[16];
            uint32_t actualLoopNum = 0U;
            uint32_t lastLogicalBlockId = (historyLen + qTokenInBatch) / blockShapeY_;
            const bool exactQ8Q1StackFullPages = exactQ8Q1StackShape && !hasSequsedQ && !hasSequsedKv &&
                                                 qStorageLen == maxQSeqlen_ && kvStorageLen % blockShapeY_ == 0U &&
                                                 kStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                                 vStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                                 validTopK == topK_;
            const bool exactQ64Q16StackFullPages = exactQ64Q16StackShape && !hasSequsedQ && !hasSequsedKv &&
                                                   qStorageLen == maxQSeqlen_ && kvStorageLen % blockShapeY_ == 0U &&
                                                   kStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                                   vStride0_ == static_cast<uint64_t>(blockSize_) * kvHeads_ * embed_ &&
                                                   validTopK == topK_;
            if (exactQ8FullPages || exactQStackFullPages || exactQ1GqaStackFullPages || exactQ8Q1StackFullPages ||
                exactQ64Q16StackFullPages) {
                // The stacked-pages tile aggregates the per-page valid sizes
                // into one tile-level column count (kvSTileSizeAct), so only
                // trailing columns of the tile can be excluded.  Keep the
                // causally truncated page (the one holding the current
                // token's causal bound) at the END of the selection order so
                // its truncated tail lands in the tile tail, where the
                // softmax epilogue (columnNum = kvSTileSizeAct) and the BMM2
                // ZeroColumnPadding skip it.  Softmax and the PV matmul are
                // invariant to column order, so the reorder is math-neutral.
                uint32_t fillIdx = 0U;
                int32_t causalPhysicalId = -1;
                uint32_t causalTileSize = blockShapeY_;
                for (uint32_t i = 0U; i < topK_; ++i) {
                    int32_t logicalId = gSparseBlockIdx.GetValue(sparseIdxBase + i);
                    int64_t btOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_ + logicalId;
                    int32_t physicalId = gBlockTable.GetValue(btOffset);
                    if (static_cast<uint32_t>(logicalId) == lastLogicalBlockId) {
                        causalPhysicalId = physicalId;
                        causalTileSize = lastBlockTileSize;
                    } else {
                        validPhysicalIds[fillIdx] = physicalId;
                        validTileSize[fillIdx] = blockShapeY_;
                        fillIdx++;
                    }
                }
                if (causalPhysicalId >= 0) {
                    validPhysicalIds[fillIdx] = causalPhysicalId;
                    validTileSize[fillIdx] = causalTileSize;
                    fillIdx++;
                }
                actualLoopNum = fillIdx;
            } else {
                for (uint32_t i = rawBegin; i < rawEnd && i < topK_; i++) {
                    int32_t logicalId = gSparseBlockIdx.GetValue(sparseIdxBase + i);
                    if (logicalId < 0)
                        continue;
                    int64_t btOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_ + logicalId;
                    int32_t physicalId = gBlockTable.GetValue(btOffset);
                    validPhysicalIds[actualLoopNum] = physicalId;
                    validTileSize[actualLoopNum] =
                        (static_cast<uint32_t>(logicalId) == lastLogicalBlockId) ? lastBlockTileSize : blockShapeY_;
                    actualLoopNum++;
                }
            }
            kvSLoopNum = actualLoopNum;
            if (kvSLoopNum == 0U) {
#ifdef __DAV_C220_VEC__
                if (isFdPartial) {
                    epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId, groupSize, embed_,
                                                         tilingData->fdLseSubStride);
                } else {
                    epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
                }
#endif
                if constexpr (ANTIQUANT) {
                    if (!usePerTaskQueryHandshake && !fdEnabled) {
                        AscendC::SyncAll<false>();
                    }
                }
                continue;
            }
            // Disable prefetch when kv blocks <= PRE_LAUNCH (avoids empty CrossCore rounds).
            uint32_t preLaunch = (kvSLoopNum > PRE_LAUNCH) ? PRE_LAUNCH : 0;
            bool stackAllKvBlocks = false;
            if constexpr (ANTIQUANT) {
                // The validated short targets have complete selected pages.
                // Stack their pages into fixed-size tiles; all other shapes
                // retain the serialized generic protocol.
                stackAllKvBlocks =
                    (exactQ8FullPages && kvSLoopNum == 16U) || (exactQ1FullPages && kvSLoopNum == 4U) ||
                    (exactQStackFullPages && kvSLoopNum == 16U) || (exactQ1GqaStackFullPages && kvSLoopNum == 16U) ||
                    (exactQ8Q1StackFullPages && kvSLoopNum == 16U) || (exactQ64Q16StackFullPages && kvSLoopNum == 16U);
                if (exactQ8FdShape && kvSLoopNum > 1U) {
                    bool allPagesFull = true;
                    for (uint32_t i = 0U; i < kvSLoopNum; ++i) {
                        allPagesFull = allPagesFull && (validTileSize[i] == blockSize_);
                    }
                    // A causal page must remain a tail page in a stacked tile;
                    // leave such FD ranges on the proven generic path.
                    stackAllKvBlocks = allPagesFull;
                }
                // The generic AIC/AIV mapping does not provide a stable
                // per-stage producer/consumer pair across successive sparse
                // KV blocks. Keep W8A8 tiles serialized until that mapping is
                // made explicit; otherwise a topK larger than the workspace
                // stage ring can wait on a token from another tile.
                preLaunch = 0U;
            }
            constexpr uint32_t stackBlockNum = 8U;
            const uint32_t kvTileLoopNum =
                stackAllKvBlocks ? (kvSLoopNum + stackBlockNum - 1U) / stackBlockNum : kvSLoopNum;
            // The stacked two-tile schedule has an independent query
            // producer/consumer handoff, so the first tile can be prefetched
            // for every MSD width.  Restricting this to MSD2 needlessly
            // serializes the lower-cost MSD1 path and hides its query-side
            // savings behind the fixed first-tile latency.
            // A single stacked tile has no later tile to overlap with.  Keep
            // the prelaunch slot only for the multi-tile schedule; otherwise
            // qSeq=1/topK<=8 pays an unnecessary producer/consumer round.
            const uint32_t taskPreLaunch = stackAllKvBlocks ? (kvTileLoopNum > 1U ? 1U : 0U) : preLaunch;
            uint32_t rowNum = groupSize;
            int64_t blockTOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_;
            // qS tile along TND S-axis; equals tokens per Q-block (tiling currently requires ==1).
            uint32_t qSBlockSize = blockShapeX_;

#ifdef __DAV_C220_VEC__
            if constexpr (ANTIQUANT) {
                // Query preprocessing and the Amax_Q result are vector-local,
                // so refresh them immediately before this task's QK stage.
                // For an oversubscribed/FD schedule, hand the task to CUBE
                // with a task-scoped CrossCore token.  For one task per core,
                // the matching SyncAll is issued by both AIV and CUBE below.
                if (kvSLoopNum > 0U) {
                    // Split exactly as the softmax path does.  For a single
                    // query head only AIV1 owns the row; letting both AIV
                    // sub-blocks preprocess it races on Query/Amax workspace.
                    const uint32_t firstSubBlockRows = groupSize / 2U;
                    const bool isSecondSubBlock = AscendC::GetSubBlockIdx() == 1;
                    const uint32_t queryRowStart = isSecondSubBlock ? firstSubBlockRows : 0U;
                    const uint32_t queryRowCount = isSecondSubBlock ? groupSize - firstSubBlockRows : firstSubBlockRows;
                    if (queryRowCount > 0U) {
                        QueryPreProcess(qStorageToken, qHeadStart, batchIdx, groupSize, embed_, queryRowStart,
                                        queryRowCount,
                                        (exactQStackShape || exactQ1GqaStackShape || exactQ8Q1StackShape ||
                                         exactQ64Q16StackShape) &&
                                            msdIterNum_ == 2U);
                    }
                    if (usePerTaskQueryHandshake) {
                        NpuArch::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                        NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(queryReady_);
                    }
                }
            }
#endif
            if constexpr (ANTIQUANT) {
                if (!usePerTaskQueryHandshake && !fdEnabled) {
                    AscendC::SyncAll<false>();
                }
            }

#ifdef __DAV_C220_CUBE__
            // Load Q into L1 once per task
            LayoutQ gmQLayout(ANTIQUANT ? mmRowNum : rowNum, ANTIQUANT ? embedRound : embed_);
            if constexpr (ANTIQUANT) {
                if (usePerTaskQueryHandshake) {
                    NpuArch::Arch::CrossCoreWaitFlag<0x2>(queryReady_);
                }
                // In antiquant mode, load preprocessed int8 Query from workspace
                const int64_t queryWorkspaceStride = static_cast<int64_t>(qHeads_) * msdIterNum_ * RoundUp(embed_, 16);
                const int64_t queryWorkspaceOffset =
                    static_cast<int64_t>(qStorageToken) * queryWorkspaceStride +
                    static_cast<int64_t>(qHeadStart) * msdIterNum_ * RoundUp(embed_, 16);
                blockMmadQK.loadQGM(queryPreProcessResGm[queryWorkspaceOffset], gmQLayout, mmRowNum, mmRowNum,
                                    mmRowNum * embedRound);
            } else {
                blockMmadQK.loadQGM(gQ[gmOffsetQ], gmQLayout, rowNum, groupSize, embed_);
            }
#endif

#ifdef __DAV_C220_VEC__
            // Rescale LSE treats layout*.shape(0) as qS (must match qSBlockSize arg).
            LayoutO gmOLayout(qSBlockSize, strideQO);
            LayoutLse gmLseLayout(qSBlockSize, qHeads_);
#endif
            for (uint32_t kvBlockIdx = 0; kvBlockIdx < kvTileLoopNum + taskPreLaunch; kvBlockIdx++) {
                // === Stage 1+2: QK Matmul & Online Softmax ===
                if (kvBlockIdx < kvTileLoopNum) {
                    const uint32_t kvBlockStart = stackAllKvBlocks ? kvBlockIdx * stackBlockNum : kvBlockIdx;
                    const uint32_t blockStackNum =
                        stackAllKvBlocks ? min(stackBlockNum, kvSLoopNum - kvBlockStart) : 1U;
                    uint32_t kvSTileSizeAct = 0U;
                    for (uint32_t blockIdx = 0U; blockIdx < blockStackNum; ++blockIdx) {
                        kvSTileSizeAct += validTileSize[kvBlockStart + blockIdx];
                    }
                    int32_t physicalBlockId = validPhysicalIds[kvBlockStart];
                    // PA_NZ page packs [kvHead, D/C0, blockSize, C0], so the per-head
                    // in-page stride is embed*blockSize; BBND rows keep embed per head.
                    const int64_t kvHeadPageStride =
                        ANTIQUANT ? static_cast<int64_t>(embed_) * static_cast<int64_t>(blockSize_) :
                                    static_cast<int64_t>(embed_);
                    // PA_NZ gather consumes physical page ids directly.  The
                    // generic path used to pass the first selected page as
                    // the GM base and then feed logical ids through
                    // gIdentityIdx, which is only initialized at slot 0 for
                    // W8A8.  That silently repeated the first page whenever
                    // a task selected more than one page (and was especially
                    // visible on the second KV head).  Keep the GM base at
                    // the KV-head page origin and let the gather use the
                    // physical-id list for every antiquant tile.
                    const bool usePhysicalBlockIds = ANTIQUANT || stackAllKvBlocks;
                    int64_t gmOffsetK = usePhysicalBlockIds ?
                                            static_cast<int64_t>(kvHeadIdx) * kvHeadPageStride :
                                            static_cast<int64_t>(physicalBlockId) * static_cast<int64_t>(kStride0_) +
                                                static_cast<int64_t>(kvHeadIdx) * kvHeadPageStride;

                    uint32_t stageId = kvBlockIdx % MAX_CROSS_CORE_BUF_STAGES;
                    uint64_t gmOffsetS =
                        static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;

#ifdef __DAV_C220_CUBE__
                    // Stage 1: QK Matmul
                    if constexpr (ANTIQUANT) {
                        if (taskPreLaunch > 0U && kvTileLoopNum > MAX_CROSS_CORE_BUF_STAGES &&
                            kvBlockIdx >= MAX_CROSS_CORE_BUF_STAGES) {
                            Arch::CrossCoreFlag workspaceStageReady(WORKSPACE_READY_BASE_ID + stageId);
                            NpuArch::Arch::CrossCoreWaitFlag<0x2>(workspaceStageReady);
                        }
                    }
                    LayoutK gmKLayout(strideKVRow, blockSize_);
                    LayoutS ubSLayout(ANTIQUANT ? mmRowNumRound : rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    GemmCoord actualBlockShapeQK{ANTIQUANT ? mmRowNum : rowNum, kvSTileSizeAct, embed_};

                    blockMmadQK(gQ[gmOffsetQ], gK[gmOffsetK], gS[gmOffsetS], gBlockTable[blockTOffset], gIdentityIdx,
                                gmQLayout, gmKLayout, ubSLayout, actualBlockShapeQK,
                                // NZ gather steps whole pages via currentYIdx (page stride);
                                // BBND token rows use the kvHeads*embed row stride.
                                0, 0, blockSize_, ANTIQUANT ? kStride0_ : static_cast<uint64_t>(strideKVRow),
                                blockSize_, blockStackNum, 1, kvSTileSizeAct, validPhysicalIds + kvBlockStart,
                                usePhysicalBlockIds);

                    Arch::CrossCoreFlag qkStageReady(taskPreLaunch > 0U && stageId == 1U ? QK_READY_STAGE1_ID :
                                                                                           QK_READY_ID);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkStageReady);
#endif

#ifdef __DAV_C220_VEC__
                    // Stage 2: Online Softmax
                    uint64_t gmOffsetP = gmOffsetS;
                    LayoutS ubSLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    uint32_t pTileStride =
                        ANTIQUANT ? RoundUp<2 * C0_NUM_PER_FRACTAL>(kvSTileSizeAct) : RoundUp<16>(kvSTileSizeAct);
                    LayoutP ubPLayout(rowNumRound, pTileStride);
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};

                    // 伪量化：设置当前 KV block 的 scale offset（per-token scale 按 physicalBlockId 索引）
                    if constexpr (ANTIQUANT) {
                        if (stackAllKvBlocks) {
                            epilogueOnlineSoftmax.SetAntiquantScalePages(validPhysicalIds + kvBlockStart, blockStackNum,
                                                                         blockSize_);
                        } else {
                            epilogueOnlineSoftmax.SetAntiquantScaleOffset(static_cast<uint64_t>(physicalBlockId) *
                                                                          blockSize_);
                        }
                    }

                    Arch::CrossCoreFlag qkStageReady(taskPreLaunch > 0U && stageId == 1U ? QK_READY_STAGE1_ID :
                                                                                           QK_READY_ID);
                    NpuArch::Arch::CrossCoreWaitFlag<0x2>(qkStageReady);

                    Arch::CrossCoreFlag softmaxStageReady(
                        taskPreLaunch > 0U && stageId == 1U ? SOFTMAX_READY_STAGE1_ID : SOFTMAX_READY_ID);

                    epilogueOnlineSoftmax(gP[gmOffsetP], gS[gmOffsetS], ubPLayout, ubSLayout, actualBlockShapeQK,
                                          (kvBlockIdx == 0), (kvBlockIdx == kvTileLoopNum - 1), qSBlockSize, groupSize,
                                          stageId, softmaxStageReady);
#endif
                }

                // === Stage 3+4: PV Matmul & RescaleO ===
                if (kvBlockIdx >= taskPreLaunch) {
                    uint32_t kvBlockIdxDe = kvBlockIdx - taskPreLaunch;
                    const uint32_t kvBlockStart = stackAllKvBlocks ? kvBlockIdxDe * stackBlockNum : kvBlockIdxDe;
                    const uint32_t blockStackNum =
                        stackAllKvBlocks ? min(stackBlockNum, kvSLoopNum - kvBlockStart) : 1U;
                    uint32_t kvSTileSizeAct = 0U;
                    for (uint32_t blockIdx = 0U; blockIdx < blockStackNum; ++blockIdx) {
                        kvSTileSizeAct += validTileSize[kvBlockStart + blockIdx];
                    }
                    int32_t physicalBlockIdV = validPhysicalIds[kvBlockStart];
                    // PA_NZ per-head in-page stride is embed*blockSize (same as K).
                    const int64_t vHeadPageStride =
                        ANTIQUANT ? static_cast<int64_t>(embed_) * static_cast<int64_t>(blockSize_) :
                                    static_cast<int64_t>(embed_);
                    const bool usePhysicalBlockIdsV = ANTIQUANT || stackAllKvBlocks;
                    int64_t gmOffsetV = usePhysicalBlockIdsV ?
                                            static_cast<int64_t>(kvHeadIdx) * vHeadPageStride :
                                            static_cast<int64_t>(physicalBlockIdV) * static_cast<int64_t>(vStride0_) +
                                                static_cast<int64_t>(kvHeadIdx) * vHeadPageStride;

                    uint32_t stageId = kvBlockIdxDe % MAX_CROSS_CORE_BUF_STAGES;
                    uint64_t gmOffsetP =
                        static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;
                    uint64_t gmOffsetOTmp =
                        static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;

#ifdef __DAV_C220_CUBE__
                    // Stage 3: PV Matmul
                    uint32_t pvKSize = ANTIQUANT ? RoundUp<2 * C0_NUM_PER_FRACTAL>(kvSTileSizeAct) : kvSTileSizeAct;
                    uint32_t pTileStride = ANTIQUANT ? pvKSize : RoundUp<16>(kvSTileSizeAct);
                    LayoutP ubPLayout(pvMmRowNumRound, pTileStride);
                    LayoutV gmVLayout(blockSize_, strideKVRow);
                    LayoutOTmp ubOTmpLayout(pvMmRowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{pvMmRowNum, embed_, pvKSize};

                    Arch::CrossCoreFlag softmaxStageReady(
                        taskPreLaunch > 0U && stageId == 1U ? SOFTMAX_READY_STAGE1_ID : SOFTMAX_READY_ID);
                    blockMmadPV(
                        gP[gmOffsetP], gV[gmOffsetV], gOTmp[gmOffsetOTmp], gBlockTable[blockTOffset], gIdentityIdx,
                        ubPLayout, gmVLayout, ubOTmpLayout, actualBlockShapePV, 0, 0, blockSize_, kvSTileSizeAct,
                        // NZ gather steps whole pages via currentYIdx (page stride).
                        ANTIQUANT ? vStride0_ : static_cast<uint64_t>(strideKVRow), blockStackNum, softmaxStageReady,
                        blockSize_, blockStackNum, 1, validPhysicalIds + kvBlockStart, usePhysicalBlockIdsV);
                    Arch::CrossCoreFlag pvStageReady(taskPreLaunch > 0U && stageId == 1U ? PV_READY_STAGE1_ID :
                                                                                           PV_READY_ID);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvStageReady);
#endif

#ifdef __DAV_C220_VEC__
                    // Stage 4: RescaleO
                    uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB;
                    LayoutOTmp ubOTmpLayout(rowNumRound, embedRound);
                    LayoutUpdate ubUpdateLayout(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};

                    Arch::CrossCoreFlag pvStageReady(taskPreLaunch > 0U && stageId == 1U ? PV_READY_STAGE1_ID :
                                                                                           PV_READY_ID);
                    NpuArch::Arch::CrossCoreWaitFlag<0x2>(pvStageReady);

                    // LSE GM must use storage token index (same as O), not packed task index.
                    if (isFdPartial) {
                        epilogueRescaleO.ProcessPartial(
                            gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[gmOffsetLse], gmOLayout,
                            ubOTmpLayout, ubUpdateLayout, gmLseLayout, actualBlockShapePV, qSBlockSize, groupSize,
                            (kvBlockIdxDe == 0), (kvBlockIdxDe == kvTileLoopNum - 1), stageId, gPartialO, gPartialLse,
                            fdPartialTaskId, tilingData->fdLseSubStride);
                    } else {
                        epilogueRescaleO(gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
                                         gLse[gmOffsetLse], gmOLayout, ubOTmpLayout, ubUpdateLayout, gmLseLayout,
                                         actualBlockShapePV, qSBlockSize, groupSize, (kvBlockIdxDe == 0),
                                         (kvBlockIdxDe == kvTileLoopNum - 1), stageId);
                    }
                    if constexpr (ANTIQUANT) {
                        if (taskPreLaunch > 0U && kvTileLoopNum > MAX_CROSS_CORE_BUF_STAGES) {
                            Arch::CrossCoreFlag workspaceStageReady(WORKSPACE_READY_BASE_ID + stageId);
                            NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(workspaceStageReady);
                        }
                    }
#endif
                }
            }

#ifdef __DAV_C220_CUBE__
            if constexpr (ANTIQUANT && (MAX_CROSS_CORE_BUF_STAGES > 0U)) {
                // The prefetch can leave the final three stages unconsumed at
                // task end. Drain them before a later task reuses their flags.
                if (taskPreLaunch > 0U && kvTileLoopNum > MAX_CROSS_CORE_BUF_STAGES) {
                    for (uint32_t stage = 0; stage < MAX_CROSS_CORE_BUF_STAGES; ++stage) {
                        Arch::CrossCoreFlag workspaceStageReady(WORKSPACE_READY_BASE_ID + stage);
                        NpuArch::Arch::CrossCoreWaitFlag<0x2>(workspaceStageReady);
                    }
                }
            }
#endif
        }

#ifdef __DAV_C220_CUBE__
        // Wait for all Cube core events
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif

#ifdef __DAV_C220_VEC__
        // Wait for all VECTOR core events
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
        if (fdEnabled) {
            AscendC::SyncAll<false>();
#ifdef __DAV_C220_VEC__
            GenericBlockSparseAttentionFdCombineArch22<ElementO, Arch::Resource<ArchTag>> combine(resource);
            combine(meta, tilingData, gPartialLse, gPartialO, gO, gCuSeqLengths, gSequsedQ, hasSequsedQ);
#endif
        }
    }

private:
    __aicore__ inline void FetchTilingData(
        __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData, GM_ADDR metaData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        embed_ = tilingData->embeddingSize;
        blockShapeY_ = tilingData->blockShapeY;
        blockShapeX_ = tilingData->blockShapeX;
        blockSize_ = tilingData->blockSize;
        qBlockNum_ = tilingData->qBlockNum;
        topK_ = tilingData->topK;
        maxBlocksPerBatch_ = tilingData->maxBlocksPerBatch;
        // Full AICPU metadata protocol overlay (no tiling fallback for task schedule).
        __gm__ GsaMetadata::Metadata *meta = reinterpret_cast<__gm__ GsaMetadata::Metadata *>(metaData);
        headSplitFactor_ = tilingData->headSplitFactor == 0U ? 1U : tilingData->headSplitFactor;
        totalTaskNum_ = static_cast<uint32_t>(meta->saTotalTaskNum) * headSplitFactor_;
        scaleValue_ = tilingData->scaleValue;
        maxQSeqlen_ = tilingData->maxQSeqlen;
        groupSize_ = tilingData->groupSize / headSplitFactor_;
        qBaseTile_ = tilingData->qBaseTile;
        kvBaseTile_ = tilingData->kvBaseTile;
        mm1OutSize_ = tilingData->mm1OutSize;
        smOnlineOutSize_ = tilingData->smOnlineOutSize;
        mm2OutSize_ = tilingData->mm2OutSize;
        updateSize_ = tilingData->updateSize;
        kStride0_ = tilingData->kStride0;
        vStride0_ = tilingData->vStride0;
        antiquantFlag_ = tilingData->antiquantFlag;
        msdIterNum_ = tilingData->msdIterNum;
        queryPreProcessSize_ = tilingData->queryPreProcessSize;
        amaxQSize_ = tilingData->amaxQSize;
        amaxPSize_ = tilingData->amaxPSize;
    }

    __aicore__ inline void InitAntiquant(GsaKernelParamsArch22 const &params)
    {
        if constexpr (ANTIQUANT) {
            // Set up original Query GlobalTensor (half/bf16)
            gQOrig.SetGlobalBuffer((__gm__ ElementO *)params.q);

            // per-token scale: shape is [batch, seq_kv]
            // In paged attention, total KV tokens = batch * maxBlocksPerBatch * blockSize
            int64_t antiValueOffset = static_cast<int64_t>(batch_) * maxBlocksPerBatch_ * blockSize_;

            if (params.kDequantScale != nullptr) {
                // Independent K/V scale mode
                keyAntiqScaleGm.SetGlobalBuffer((__gm__ float *)params.kDequantScale);
                valueAntiqScaleGm.SetGlobalBuffer((__gm__ float *)params.vDequantScale);
            } else {
                // Combined mode: qDequantScale contains K scale (first half) and V scale (second half)
                keyAntiqScaleGm.SetGlobalBuffer((__gm__ float *)params.qDequantScale);
                valueAntiqScaleGm.SetGlobalBuffer((__gm__ float *)params.qDequantScale + antiValueOffset);
            }
        }
    }

    __aicore__ inline void QueryPreProcess(uint32_t qStorageToken, uint32_t qHeadStart, uint32_t batchIdx,
                                           uint32_t groupSize, uint32_t embed, uint32_t startRow, uint32_t dealRowCount,
                                           bool enableMte3DoubleBuffer)
    {
        if constexpr (!ANTIQUANT) {
            return;
        }
        // 伪量化 Query 预处理：half/bf16 → float → int8（含 Amax_Q 计算）
        // 参考 incre_flash_attention 的 QueryPreProcess 实现
        constexpr uint32_t UB_BLOCK_SIZE = 16384;
        constexpr uint32_t UB_VEC_SIZE = 1024;

        uint32_t columnCount = RoundUp(embed, 16); // padded column count
        uint32_t actualColumnCount = embed;        // actual embedding size

        // Query offset in GM
        int64_t strideQO = qHeads_ * embed_;
        int64_t qOffset =
            static_cast<int64_t>(qStorageToken) * strideQO + static_cast<int64_t>(qHeadStart + startRow) * embed_;

        // Query workspace is padded to 16 elements per row by the host tiler.
        // Keep the GM row stride identical to the padded layout used by the
        // vector pre-process and the cube loader.
        int64_t qPreProcessOffset = static_cast<int64_t>(qStorageToken) * qHeads_ * msdIterNum_ * RoundUp(embed_, 16) +
                                    static_cast<int64_t>(qHeadStart) * msdIterNum_ * RoundUp(embed_, 16);

        // UB buffer allocation (reuse epilogue antiquant regions — safe before KV loop)
        constexpr uint32_t QUERY_FLOAT_OFFSET = 4 * UB_BLOCK_SIZE;                     // queryCastUb (float)
        constexpr uint32_t QUERY_INPUT_OFFSET = 4 * UB_BLOCK_SIZE + 4 * UB_VEC_SIZE;   // inputUb (half)
        constexpr uint32_t QUERY_TMPFLOOR_OFFSET = 6 * UB_BLOCK_SIZE;                  // tmpAFloorUb (float)
        constexpr uint32_t QUERY_RESHALF_OFFSET = 6 * UB_BLOCK_SIZE + 4 * UB_VEC_SIZE; // aResOutUb (half)
        // Two slots let MSD digit 0 drain to GM while digit 1 is prepared.
        constexpr uint32_t QUERY_RESHALF_SLOT_BYTES = 8 * UB_VEC_SIZE;
        constexpr uint32_t QUERY_RESHALF_ALT_OFFSET = QUERY_RESHALF_OFFSET + QUERY_RESHALF_SLOT_BYTES;
        // This must match ANTIQ_AMAXQ_OFFSET in BlockEpilogue.  The former
        // +23*UB_VEC_SIZE location is the V-scale staging buffer, so using it
        // silently overwrites Amax_Q before OnlineSoftmax consumes the scale.
        constexpr uint32_t QUERY_AMAX_OFFSET = 10 * UB_BLOCK_SIZE + 27 * UB_VEC_SIZE;   // aMaxResUb (float)
        constexpr uint32_t QUERY_ROWMAX_OFFSET = 10 * UB_BLOCK_SIZE + 15 * UB_VEC_SIZE; // tmpRowMaxUb (float)

        AscendC::LocalTensor<float> queryCastUb = resource.ubBuf.template GetBufferByByte<float>(QUERY_FLOAT_OFFSET);
        AscendC::LocalTensor<ElementO> inputUb = resource.ubBuf.template GetBufferByByte<ElementO>(QUERY_INPUT_OFFSET);
        AscendC::LocalTensor<float> tmpAFloorUb = resource.ubBuf.template GetBufferByByte<float>(QUERY_TMPFLOOR_OFFSET);
        AscendC::LocalTensor<half> aResOutUb = resource.ubBuf.template GetBufferByByte<half>(QUERY_RESHALF_OFFSET);
        AscendC::LocalTensor<int8_t> aResOutUbI8 =
            resource.ubBuf.template GetBufferByByte<int8_t>(QUERY_RESHALF_OFFSET);
        AscendC::LocalTensor<half> aResOutUbAlt =
            resource.ubBuf.template GetBufferByByte<half>(QUERY_RESHALF_ALT_OFFSET);
        AscendC::LocalTensor<int8_t> aResOutUbI8Alt =
            resource.ubBuf.template GetBufferByByte<int8_t>(QUERY_RESHALF_ALT_OFFSET);
        AscendC::LocalTensor<float> aMaxResUb = resource.ubBuf.template GetBufferByByte<float>(QUERY_AMAX_OFFSET);
        AscendC::LocalTensor<float> tmpRowMaxUb = resource.ubBuf.template GetBufferByByte<float>(QUERY_ROWMAX_OFFSET);

        // Step 1: Copy Query from GM and cast to float (MTE2 + V)
        CopyAntiqQuery(queryCastUb, inputUb, gQOrig, qOffset, dealRowCount, columnCount, actualColumnCount);
        AscendC::PipeBarrier<PIPE_V>();

        // Step 2: AntiquantMatmulPreProcess — AbsRowMax + scale(127/Amax) + MSD expand → int8 to GM
        // 写入 queryPreProcessResGm[qPreProcessOffset]，与 CUBE loadQGM 读取位置一致
        // aMaxResUb[0..groupSize*8-1] contains per-row Amax_Q (Brcb format).
        AntiquantMatmulPreProcess(queryPreProcessResGm[qPreProcessOffset], aMaxResUb, queryCastUb, tmpAFloorUb,
                                  tmpRowMaxUb, aResOutUb, aResOutUbI8, aResOutUbAlt, aResOutUbI8Alt, startRow,
                                  dealRowCount, columnCount, actualColumnCount, groupSize, msdIterNum_,
                                  enableMte3DoubleBuffer);

        // The double-buffered target has already drained every MTE3->V
        // event before returning from AntiquantMatmulPreProcess.  A full
        // pipeline rendezvous here only delays the task-scoped handshake;
        // keep the conservative barrier for generic schedules.
        if (enableMte3DoubleBuffer) {
            AscendC::PipeBarrier<PIPE_MTE3>();
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        (void)batchIdx;
    }

    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag queryReady_{QUERY_READY_ID};
    Arch::CrossCoreFlag qkReady_{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady_{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady_{PV_READY_ID};
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t blockShapeY_;
    uint32_t blockShapeX_;
    uint32_t blockSize_;
    uint32_t qBlockNum_;
    uint32_t topK_;
    uint32_t maxBlocksPerBatch_;
    uint32_t totalTaskNum_;
    uint32_t headSplitFactor_;
    float scaleValue_;
    uint32_t maxQSeqlen_;
    uint32_t groupSize_;
    // PAGED_BBND page base strides (elements); may exceed blockSize*Nkv*D when dim0 is strided.
    uint64_t kStride0_;
    uint64_t vStride0_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // workspace partition sizes
    uint64_t mm1OutSize_;
    uint64_t smOnlineOutSize_;
    uint64_t mm2OutSize_;
    uint64_t updateSize_;
    // W8A8 pseudo-quantization state
    uint32_t antiquantFlag_;
    uint32_t msdIterNum_;
    uint64_t queryPreProcessSize_;
    uint64_t amaxQSize_;
    uint64_t amaxPSize_;
    AscendC::GlobalTensor<float> keyAntiqScaleGm;
    AscendC::GlobalTensor<float> valueAntiqScaleGm;
    AscendC::GlobalTensor<int8_t> queryPreProcessResGm;
    AscendC::GlobalTensor<ElementO> gQOrig;
    AscendC::GlobalTensor<float> gAmaxQ;
    AscendC::GlobalTensor<float> gAmaxP;
};

} // namespace GsaKernelArch22

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_H
