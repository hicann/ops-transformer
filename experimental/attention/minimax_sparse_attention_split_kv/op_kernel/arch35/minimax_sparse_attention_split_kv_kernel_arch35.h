/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MINIMAX_SPARSE_ATTENTION_SPLIT_KV_KERNEL_ARCH35_H
#define MINIMAX_SPARSE_ATTENTION_SPLIT_KV_KERNEL_ARCH35_H

#include <type_traits>
#include "kernel_utils.hpp"
#include "../kernel_common.hpp"

using namespace NpuArch;
using namespace tla;

namespace MinimaxSaSplitKvKernelArch35 {

template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO>
class MinimaxSparseAttentionSplitKvKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementQ;
    using ElementK = typename BlockMmadQK::ElementK;
    using ElementS = typename BlockMmadQK::ElementS;
    using ElementP = typename BlockMmadPV::ElementP;
    using ElementV = typename BlockMmadPV::ElementV;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using ElementOTmp = float;
    // O_partial (workspace accumOut) dtype follows the PV block's ElementO (= REDtype):
    // float for the fp32 path, bfloat16_t for the innerPrecise==1 bf16 path.
    using ElementWorkspaceO = typename BlockMmadPV::ElementO;

    using LayoutK = layout::ColumnMajor;
    using LayoutV = layout::RowMajor;
    using LayoutQ = layout::RowMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutO = layout::RowMajor;

    static constexpr uint32_t PRE_LAUNCH = 2U;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1U;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2U;
    static constexpr uint32_t P_L1_BUF_NUM = PRE_LAUNCH + 1U;

    static constexpr uint32_t FP32_ONE_BLOCK_SIZE = 8U;
    // Workspace softmaxMax/softmaxSum init: -inf for rowMax, 0 for rowSum so that
    // slots not written by Phase1 are treated as invalid by Phase2 combine.
    static constexpr float WS_ROWMAX_INIT = -3.4028235e38f;
    static constexpr float WS_ROWSUM_INIT = 0.0f;
    static constexpr uint32_t WS_INIT_EVT = 5U;      // free V_MTE3/MTE3_V id on VEC
    static constexpr uint32_t WS_INIT_CHUNK = 4096U; // fp32 elems per chunk (multiple of 8)

    // gmUb/glUb are 2-deep by ubSBufId inside the prefill softmax epilogue (bf16:
    // reg_low_prec_bf16.hpp; fp32: reg_high_prec.hpp). Both ctors keep stats at
    // 7*32KB so this offset stays valid. CopyPartialStatsToGm (MTE3) reads
    // gmUb[ubSBufId]/glUb[ubSBufId] right after each softmax.
    static constexpr uint32_t SM_UB_BLOCK = 32768U;                                         // UB_UINT8_BLOCK_SIZE
    static constexpr uint32_t SM_UB_GM_OFFSET = 7U * SM_UB_BLOCK;                           // 229376 rowMax base
    static constexpr uint32_t SM_UB_GL_OFFSET = SM_UB_GM_OFFSET + 2U * 64U * sizeof(float); // 229888 rowSum base
    static constexpr uint32_t SM_UB_STAGE_BYTES = 64U * sizeof(float);                      // 256, per-stage stride

    static constexpr uint32_t COPY_GRANULARITY = 2;

    __aicore__ inline uint64_t TaskLinearIdx(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx;
    }

    __aicore__ inline uint64_t SlotStatOffset(uint32_t qToken, uint32_t kvHeadIdx, uint32_t slotK) const
    {
        return TaskLinearIdx(qToken, kvHeadIdx) * topK_ * slotStatElems_ +
               static_cast<uint64_t>(slotK) * slotStatElems_;
    }

    __aicore__ inline void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(2);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(18);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(3);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(19);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(4);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(20);
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(0);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(1);
#endif
    }

    __aicore__ inline void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(0);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(1);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(16);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(17);
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(2);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(3);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(4);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline static uint32_t VecRowSplit(uint32_t groupSize)
    {
        uint32_t half = RoundUp(groupSize, 8U) / 2U;
        return (half > groupSize) ? groupSize : half;
    }

    __aicore__ inline static uint32_t VecNumRows(uint32_t groupSize)
    {
        if (groupSize <= 1U) {
            return (AscendC::GetSubBlockIdx() == 0U) ? groupSize : 0U;
        }
        uint32_t split = VecRowSplit(groupSize);
        return (AscendC::GetSubBlockIdx() == 0U) ? split : (groupSize - split);
    }

    __aicore__ inline static uint32_t VecGlobalRowOffset(uint32_t groupSize)
    {
        return (AscendC::GetSubBlockIdx() == 0U) ? 0U : VecRowSplit(groupSize);
    }

    __aicore__ inline MinimaxSparseAttentionSplitKvKernelArch35() {}

    __aicore__ inline void operator()(MinimaxSaSplitKvKernelParamsArch35 const &params)
    {
        __gm__ MinimaxSaSplitKv::MinimaxSparseAttentionSplitKvTilingData *tilingData =
            reinterpret_cast<__gm__ MinimaxSaSplitKv::MinimaxSparseAttentionSplitKvTilingData *>(params.tiling);
        FetchBaseShapeInfo(tilingData);

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<int32_t> gK2qRowPtr;
        gK2qRowPtr.SetGlobalBuffer((__gm__ int32_t *)params.k2qRowPtr);
        AscendC::GlobalTensor<int32_t> gK2qQIndices;
        gK2qQIndices.SetGlobalBuffer((__gm__ int32_t *)params.k2qQIndices);
        AscendC::GlobalTensor<int32_t> gK2qSlotIndices;
        gK2qSlotIndices.SetGlobalBuffer((__gm__ int32_t *)params.k2qSlotIndices);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        if (isPageAttention_ == 1U) {
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)params.blockTable);
        }
        gActualQseqlen_.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);
        gActualKvseqlen_.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<float> gSoftmaxLse;
        if (softmaxLseFlag_ == 1U) {
            gSoftmaxLse.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        }

        uint64_t wsOffset = 0U;
        AscendC::GlobalTensor<ElementWorkspaceO> gAccumOut;
        gAccumOut.SetGlobalBuffer((__gm__ ElementWorkspaceO *)(params.workSpace + wsOffset));
        wsOffset += accumOutSize_ * sizeof(ElementWorkspaceO);
        AscendC::GlobalTensor<float> gSoftmaxMax;
        gSoftmaxMax.SetGlobalBuffer((__gm__ float *)(params.workSpace + wsOffset));
        wsOffset += lseStatSize_ * sizeof(float);
        AscendC::GlobalTensor<float> gSoftmaxSum;
        gSoftmaxSum.SetGlobalBuffer((__gm__ float *)(params.workSpace + wsOffset));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        InitSyncFlags();
        // VEC-side workspace init: prefill softmaxMax<-(-inf), softmaxSum<-0 so
        // unwritten (invalid) slots are detected by Phase2. Placed before
        // SyncAll<false> so that barrier doubles as the init-completion sync
        // (no extra SyncAll needed). CUBE cores idle here.
        InitWorkspaceStats(gSoftmaxMax, gSoftmaxSum);
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
        Phase1KvCentricCompute(coreIdx, coreNum, gQ, gK, gV, gK2qRowPtr, gK2qQIndices, gK2qSlotIndices, gBlockTable,
                               gAccumOut, gSoftmaxMax, gSoftmaxSum);
        AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
        Phase2CombineScale(gO, gAccumOut, gSoftmaxMax, gSoftmaxSum, gSoftmaxLse);
#endif
        ReleaseSyncFlags();
    }

private:
    __aicore__ inline void FetchBaseShapeInfo(
        __gm__ MinimaxSaSplitKv::MinimaxSparseAttentionSplitKvTilingData *tilingData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        groupSize_ = tilingData->groupSize;
        embed_ = tilingData->embeddingSize;
        blockSize_ = tilingData->blockSize;
        topK_ = tilingData->topK;
        totalKvRows_ = tilingData->numKvBlocks;
        maxBlocksPerBatch_ = tilingData->maxBlocksPerBatch;
        k2qNnzUpperBound_ = tilingData->k2qNnzUpperBound;
        totalTaskNumP1_ = tilingData->totalTaskNumP1;
        totalTaskNumP2_ = tilingData->totalTaskNumP2;
        scaleValue_ = tilingData->scaleValue;
        accumOutSize_ = tilingData->accumOutSize;
        lseStatSize_ = tilingData->lseStatSize;
        isPageAttention_ = tilingData->isPageAttention;
        softmaxLseFlag_ = tilingData->softmaxLseFlag;
        layoutType_ = tilingData->layoutType;
        qSeqLen_ = tilingData->qSeqLen;
        kvSeqLen_ = tilingData->kvSeqLen;
        slotOElems_ = static_cast<uint64_t>(groupSize_) * embed_;
        slotStatElems_ = static_cast<uint64_t>(groupSize_);
    }

    // VEC-only: prefill contiguous softmaxMax<-(-inf) and softmaxSum<-0 in
    // WS_INIT_CHUNK blocks (tail chunk uses remainder length). UB filled once
    // per buffer; one loop writes both GM regions per chunk. Work split across
    // VEC sub-blocks by chunk index.
    __aicore__ inline void InitWorkspaceStats(AscendC::GlobalTensor<float> &gSoftmaxMax,
                                              AscendC::GlobalTensor<float> &gSoftmaxSum)
    {
        if (lseStatSize_ == 0U) {
            return;
        }
#ifdef __DAV_VEC__
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t totalSubBlocks = AscendC::GetBlockNum() * subBlockNum;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t numFullChunks = static_cast<uint32_t>(lseStatSize_ / WS_INIT_CHUNK);
        uint32_t tailElems = static_cast<uint32_t>(lseStatSize_ % WS_INIT_CHUNK);
        uint32_t totalChunks = numFullChunks + (tailElems > 0U ? 1U : 0U);

        AscendC::LocalTensor<float> ubMax = resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::LocalTensor<float> ubSum =
            resource.ubBuf.template GetBufferByByte<float>(WS_INIT_CHUNK * sizeof(float));
        AscendC::Duplicate(ubMax, WS_ROWMAX_INIT, WS_INIT_CHUNK);
        AscendC::Duplicate(ubSum, WS_ROWSUM_INIT, WS_INIT_CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(WS_INIT_EVT);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(WS_INIT_EVT);

        for (uint32_t chunkIdx = blockIdx; chunkIdx < totalChunks; chunkIdx += totalSubBlocks) {
            uint64_t off = static_cast<uint64_t>(chunkIdx) * WS_INIT_CHUNK;
            uint32_t cnt = (chunkIdx < numFullChunks) ? WS_INIT_CHUNK : tailElems;
            uint16_t blockLen = static_cast<uint16_t>(cnt * sizeof(float));
            AscendC::DataCopyPad(gSoftmaxMax[off], ubMax, {1, blockLen, 0, 0});
            AscendC::DataCopyPad(gSoftmaxSum[off], ubSum, {1, blockLen, 0, 0});
        }
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }

    // packedRow uses the MSA INTERLEAVED (round-robin) packing: it enumerates
    // (kvBlockIdx outer, batchIdx inner), skipping batches whose row count <= kvBlockIdx.
    // i.e. packedRow 0=(b0,blk0), 1=(b1,blk0), 2=(b2,blk0), 3=(b0,blk1), ... -- NOT
    // "all of b0's blocks then all of b1's". Matches host _build_packed_row_map /
    // decode_packed_row (golden). For batch_=1 this collapses to packedRow=(0, packedRow).
    // Padding requests pass q_len=kv_len=0 -> KvRowsPerBatch=0, so they occupy no packed
    // rows. InitPackedRowCoord skips leading empty batches (packedRow 0 is the first
    // non-empty batch at blk 0, not unconditionally (0,0)). Advance skips holes in the
    // middle the same way. totalKvRows_==0 (all padding) never enters the decode loop.
    __aicore__ inline uint32_t KvRowsPerBatch(uint32_t batchIdx)
    {
        return CeilDiv(static_cast<uint32_t>(gActualKvseqlen_.GetValue(batchIdx)), blockSize_);
    }

    // Advance past batches with no row at this kvBlockIdx (kv_len=0 => 0 rows).
    __aicore__ inline void SkipEmptyAtCurrentBlk(uint32_t &batchIdx, uint32_t kvBlockIdx)
    {
        while (batchIdx < batch_ && kvBlockIdx >= KvRowsPerBatch(batchIdx)) {
            batchIdx++;
        }
    }

    // packedRow 0 coordinate. Does not wrap to blk 1: that row only exists if some
    // batch has kv_len>0 (then it sits at blk 0). All-empty is totalKvRows_==0.
    __aicore__ inline void InitPackedRowCoord(uint32_t &batchIdx, uint32_t &kvBlockIdx)
    {
        batchIdx = 0;
        kvBlockIdx = 0;
        SkipEmptyAtCurrentBlk(batchIdx, kvBlockIdx);
    }

    __aicore__ inline void AdvancePackedRowCoord(uint32_t &batchIdx, uint32_t &kvBlockIdx)
    {
        batchIdx++;
        SkipEmptyAtCurrentBlk(batchIdx, kvBlockIdx);
        if (batchIdx >= batch_) {
            kvBlockIdx++;
            batchIdx = 0;
            SkipEmptyAtCurrentBlk(batchIdx, kvBlockIdx);
        }
    }

    // After the state machine resolves (batchIdx, kvBlockIdx) for this packedRow, fetch
    // the per-batch scalars the cvl functor needs: kvSeqlenBatch/numBlocksB for validSize,
    // and cumQStart (= cumQ[batchIdx] = sum qseqlen[0..batchIdx-1]) + qSeqlenBatch for
    // the Q side. Q and KV are same-batch (no cross-batch attention) -> qBatch == batchIdx,
    // so the cvl loop (CalcCausalValidLen) does 0 GM GetValue. Cost: 1 kvseqlen GetValue +
    // (batchIdx+1) qseqlen GetValues per task (1 each for batch_=1). Replaces the old
    // FindBatchAndLocalQ scan (redundant: Q/KV same batch) + the per-cvl kv/qseqlen
    // re-fetches.
    __aicore__ inline void ResolveBatchQSide(uint32_t batchIdx, uint32_t &kvSeqlenBatch, uint32_t &numBlocksB,
                                             uint32_t &cumQStart, uint32_t &qSeqlenBatch, uint32_t &cumKvStart)
    {
        kvSeqlenBatch = static_cast<uint32_t>(gActualKvseqlen_.GetValue(batchIdx));
        numBlocksB = CeilDiv(kvSeqlenBatch, blockSize_);
        cumQStart = 0;
        cumKvStart = 0;
        for (uint32_t b = 0; b <= batchIdx; ++b) {
            uint32_t qB = static_cast<uint32_t>(gActualQseqlen_.GetValue(b));
            if (b < batchIdx) {
                cumQStart += qB;
                cumKvStart += static_cast<uint32_t>(gActualKvseqlen_.GetValue(b));
            } else {
                qSeqlenBatch = qB;
            }
        }
    }

    // kvSeqlenBatch + numBlocksB are pre-fetched by ResolveBatchQSide for the batch
    // resolved by the state machine (avoids a redundant gActualKvseqlen_ GetValue here).
    __aicore__ inline uint32_t CalcKvBlockValidSize(uint32_t kvSeqlenBatch, uint32_t numBlocksB, uint32_t localBlockIdx)
    {
        // kv_len=0 => numBlocksB=0; unsigned (numBlocksB-1) would wrap and look like a tail block.
        if (numBlocksB == 0U || localBlockIdx >= numBlocksB) {
            return 0U;
        }
        if (localBlockIdx == numBlocksB - 1) {
            return kvSeqlenBatch - localBlockIdx * blockSize_;
        }
        return blockSize_;
    }

    // qBatch == batchIdx (Q/KV same batch). cumQStart/qSeqlenBatch/kvSeqlenBatch are all
    // pre-fetched by ResolveBatchQSide (state machine resolved batchIdx) -> pure compute,
    // 0 GM GetValue in the cvl loop.
    // BNSD/BSND CSR qToken is padded flatten b * S + t; TND qToken is packed flatten.
    __aicore__ inline uint32_t CalcCausalValidLen(uint32_t qToken, uint32_t validSize, uint32_t kvStartPos,
                                                  uint32_t cumQStart, uint32_t qSeqlenBatch, uint32_t kvSeqlenBatch)
    {
        // Padding request: q_len=kv_len=0 (or either side empty) has no causal overlap.
        if (qSeqlenBatch == 0U || kvSeqlenBatch == 0U) {
            return 0U;
        }
        uint32_t localQIdx;
        if ((layoutType_ == MinimaxSaSplitKv::LAYOUT_BNSD || layoutType_ == MinimaxSaSplitKv::LAYOUT_BSND) &&
            qSeqLen_ > 0U) {
            localQIdx = qToken % qSeqLen_;
            if (localQIdx >= qSeqlenBatch) {
                return 0U;
            }
        } else {
            localQIdx = qToken - cumQStart;
        }
        uint32_t qPosition = kvSeqlenBatch - qSeqlenBatch + localQIdx;
        if (qPosition < kvStartPos) {
            return 0U;
        }
        uint32_t maxLen = qPosition - kvStartPos + 1;
        return (maxLen < validSize) ? maxLen : validSize;
    }

    // BNSD/BSND keep padded S in the tensor; a dummy request is q_len=kv_len=0 (and
    // tokens t>=q_len are intra-seq padding). TND is packed, so q_len=0 batches add
    // no qTokens; q_len>0 && kv_len=0 still has packed qTokens and must be skipped in
    // Phase2 so O/LSE stay zero (ComputeScaleValue would otherwise write -inf LSE).
    __aicore__ inline bool IsPaddingQToken(uint32_t qToken) const
    {
        if (layoutType_ == MinimaxSaSplitKv::LAYOUT_BNSD || layoutType_ == MinimaxSaSplitKv::LAYOUT_BSND) {
            if (qSeqLen_ == 0U) {
                return false;
            }
            uint32_t b = qToken / qSeqLen_;
            uint32_t t = qToken - b * qSeqLen_;
            uint32_t qLen = static_cast<uint32_t>(gActualQseqlen_.GetValue(b));
            uint32_t kvLen = static_cast<uint32_t>(gActualKvseqlen_.GetValue(b));
            return (qLen == 0U) || (kvLen == 0U) || (t >= qLen);
        }
        uint32_t cumQ = 0U;
        for (uint32_t b = 0U; b < batch_; ++b) {
            uint32_t qLen = static_cast<uint32_t>(gActualQseqlen_.GetValue(b));
            if (qToken < cumQ + qLen) {
                uint32_t kvLen = static_cast<uint32_t>(gActualKvseqlen_.GetValue(b));
                return kvLen == 0U;
            }
            cumQ += qLen;
        }
        return true;
    }

    // GM offset of query[qToken, qHeadStart, 0:D).
    // TND/BSND: token-major, consecutive heads of one token are D apart.
    // BNSD: head-major S, consecutive heads of one token are S*D apart.
    __aicore__ inline uint64_t QTokenHeadOffset(uint32_t qToken, uint32_t qHeadStart) const
    {
        if (layoutType_ == MinimaxSaSplitKv::LAYOUT_BNSD && qSeqLen_ > 0U) {
            uint32_t b = qToken / qSeqLen_;
            uint32_t t = qToken - b * qSeqLen_;
            return (static_cast<uint64_t>(b) * qHeads_ + qHeadStart) * qSeqLen_ * embed_ +
                   static_cast<uint64_t>(t) * embed_;
        }
        return static_cast<uint64_t>(qToken) * qHeads_ * embed_ + static_cast<uint64_t>(qHeadStart) * embed_;
    }

    // Copies rowMax + rowSum for 1 or 2 groups (PAIR) from UB to GM in one DataCopyPad each
    // (blockCount = ndNum). Mirrors the Q-gather/O-partial ndNum=2 pairing: the caller
    // computes the per-pair GM dst gap (dstStride, 32B-block units = end-of-group0 to
    // start-of-group1) and UB src gap (srcStride, 32B-block units) from the two groups'
    // actual (qToken, slotK) lseOff -- no uniform-stride assumption (ndNum=2 needs only
    // the one inter-pair gap, computed per pair). Falls back to blockCount=1 (single group)
    // at the caller when the pair's gaps aren't 32B-aligned or exceed the uint16 stride.
    __aicore__ inline void CopyPartialStatsToGm(AscendC::GlobalTensor<float> &gSoftmaxMax,
                                                AscendC::GlobalTensor<float> &gSoftmaxSum,
                                                const AscendC::LocalTensor<float> &rowMaxLocal,
                                                const AscendC::LocalTensor<float> &rowSumLocal, uint64_t lseOffset,
                                                uint32_t rowCount, uint16_t blockCount, uint32_t srcStride,
                                                uint32_t dstStride)
    {
        if (rowCount == 0U) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::DataCopyPad(gSoftmaxMax[lseOffset], rowMaxLocal,
                             AscendC::DataCopyExtParams{blockCount, static_cast<uint16_t>(rowCount * sizeof(float)),
                                                        srcStride, dstStride, 0});
        AscendC::DataCopyPad(gSoftmaxSum[lseOffset], rowSumLocal,
                             AscendC::DataCopyExtParams{blockCount, static_cast<uint16_t>(rowCount * sizeof(float)),
                                                        srcStride, dstStride, 0});
    }

    // Batched stats scatter: the softmax wrote this AIV's rowMax/rowSum into gmUb/glUb
    // (ubSBufId stage). Cores split by GROUP (not M/2): each AIV owns whole groups [gLo, gHi),
    // no group straddles the two AIVs; every owned group has rowCount = groupRows (never 0).
    // lseOff is computed on demand from the batch's qToken/slotK (== SlotStatOffset(qToken,
    // kvHeadIdx, slotK,0)), so no lseOff array is marshalled. gmUbAlias/glUbAlias are bound
    // ONCE before the loop to the stage base (loop-invariant ubSBufId); the per-group stride
    // (localOff) is applied via [localOff] in the loop. The rowMax/rowSum group base uses an
    // 8-float-aligned stride (matches the softmax epilogue's statsOff): CopyPartialStatsToGm's
    // DataCopyPad needs a 32B-aligned source; groupRows<8 (local groupSize=4) would otherwise
    // misalign it. No-op for groupRows>=8 (production groupSize=16).
    __aicore__ inline void ScatterBatchStats(AscendC::GlobalTensor<float> &gSoftmaxMax,
                                             AscendC::GlobalTensor<float> &gSoftmaxSum, uint32_t ubSBufId,
                                             uint32_t batchM, uint32_t groupRows, const uint32_t *qTokens,
                                             const uint32_t *slotKs, uint32_t kvHeadIdx, uint32_t groupCount)
    {
        // AIV0-only for fp32 S (innerPrecise=0): matches QK NoQuant Fixpipe and high-prec
        // softmax. AIV1 returns immediately (no stats to scatter).
        uint32_t m;
        uint32_t startRow;
        uint32_t gLo;
        uint32_t gHi;
        if constexpr (std::is_same<ElementS, float>::value) {
            m = (AscendC::GetSubBlockIdx() == 0U) ? batchM : 0U;
            startRow = 0U;
            gLo = 0U;
            gHi = groupCount;
        } else {
            uint32_t gSplit = CeilDiv(groupCount, 2U);
            uint32_t mCopyOffset = gSplit * groupRows;
            uint32_t mHalf = mCopyOffset < batchM ? mCopyOffset : batchM;
            m = (AscendC::GetSubBlockIdx() == 0U) ? mHalf : (batchM - mHalf);
            startRow = AscendC::GetSubBlockIdx() * mCopyOffset;
            gLo = (m == 0U) ? 0U : startRow / groupRows;
            gHi = gLo + ((m == 0U) ? 0U : m / groupRows);
        }
        if (m == 0U) {
            return;
        }
        (void)startRow;
        // rowMax/rowSum group base stride = 8-float aligned (matches softmax epilogue's statsOff;
        // DataCopyPad source must be 32B-aligned). No-op for groupRows>=8 (production 16).
        uint32_t grpStride = RoundUp(groupRows, 8U);
        // Bind once to the stage base (loop-invariant); per-group offset via [localOff].
        AscendC::LocalTensor<float> gmUbAlias =
            resource.ubBuf.template GetBufferByByte<float>(SM_UB_GM_OFFSET + ubSBufId * SM_UB_STAGE_BYTES);
        AscendC::LocalTensor<float> glUbAlias =
            resource.ubBuf.template GetBufferByByte<float>(SM_UB_GL_OFFSET + ubSBufId * SM_UB_STAGE_BYTES);

        for (uint32_t g = gLo; g < gHi; g += COPY_GRANULARITY) {
            uint32_t localOff0 = (g - gLo) * grpStride;
            uint64_t lseOff0 = SlotStatOffset(qTokens[g], kvHeadIdx, slotKs[g]);
            uint16_t blockCount = 1U;
            uint32_t srcStride = 0U;
            uint32_t dstStride = 0U;
            if ((g + 1U) < gHi) {
                uint64_t lseOff1 = SlotStatOffset(qTokens[g + 1U], kvHeadIdx, slotKs[g + 1U]);
                if (lseOff0 < lseOff1) {
                    dstStride = (lseOff1 - lseOff0 - groupRows) * sizeof(float);
                } else {
                    dstStride = (lseOff0 - lseOff1 - groupRows) * sizeof(float);
                    lseOff0 = lseOff1;
                }
                blockCount = COPY_GRANULARITY;
            }
            CopyPartialStatsToGm(gSoftmaxMax, gSoftmaxSum, gmUbAlias[localOff0], glUbAlias[localOff0], lseOff0,
                                 groupRows, blockCount, srcStride, dstStride);
        }
    }

    // Per-core nnz-balanced partition boundary. Replaces the size-33
    // coreRowStart/coreQOffset array the old init built on EVERY core: the kernel
    // is parallel, so each core needs ONLY its own [rowLo,offLo,rowHi,offHi).
    // This walks row_ptr once and stops as soon as BOTH boundaries are found ->
    // early cores terminate after a fraction of the walk (core 0 stops at row 0;
    // only the last core walks the full prefix). Boundary semantics are identical
    // to the old full-array build: target_c = c*totalNnz/activeCores; the boundary
    // is the row CONTAINING target_c and the batchGroupsMax-aligned intra-row
    // q-offset where core c STARTS (target==rowEndG defers to the next row, as
    // before). Core 0 starts at (0,0); a core whose target is past totalNnz
    // (degenerate totalNnz<activeCores) keeps rowHi=totalTasks -> empty slice,
    // matching the old trailing-core fill. Deterministic -> every core derives
    // the same two values it would have read from the array -> no cross-core sync.
    __aicore__ inline void ComputeCoreBounds(uint32_t coreIdx, uint32_t activeCores, uint32_t batchGroupsMax,
                                             AscendC::GlobalTensor<int32_t> const &gK2qRowPtr, uint32_t &rowLo,
                                             uint32_t &offLo, uint32_t &rowHi, uint32_t &offHi)
    {
        uint32_t totalTasks = totalKvRows_ * kvHeads_;
        uint32_t headStride = totalKvRows_ + 1U;

        // totalNnz = sum of each head's last-column (per-head cumsum terminal).
        uint64_t totalNnz = 0U;
        for (uint32_t h = 0U; h < kvHeads_; h++) {
            totalNnz +=
                static_cast<uint32_t>(gK2qRowPtr.GetValue(static_cast<uint64_t>(h) * headStride + totalKvRows_));
        }

        uint64_t targetLo = static_cast<uint64_t>(coreIdx) * totalNnz / activeCores;
        uint64_t targetHi = static_cast<uint64_t>(coreIdx + 1U) * totalNnz / activeCores;

        // Defaults: core 0 starts at task 0 offset 0; an unfound hi boundary
        // (target past totalNnz) leaves rowHi=totalTasks -> past-the-end guard,
        // the last core iterates [rowLo, totalTasks) with offHi=0 -> localEnd=csrEnd.
        rowLo = 0U;
        offLo = 0U;
        rowHi = totalTasks;
        offHi = 0U;
        bool loFound = (coreIdx == 0U);
        bool hiFound = false;

        uint64_t cum = 0U; // cumulative nnz before the current (h,r)
        for (uint32_t h = 0U; h < kvHeads_ && !(loFound && hiFound); h++) {
            for (uint32_t r = 0U; r < totalKvRows_ && !(loFound && hiFound); r++) {
                uint32_t rs = static_cast<uint32_t>(gK2qRowPtr.GetValue(static_cast<uint64_t>(h) * headStride + r));
                uint32_t re =
                    static_cast<uint32_t>(gK2qRowPtr.GetValue(static_cast<uint64_t>(h) * headStride + r + 1U));
                uint32_t rowNnz = (re > rs) ? (re - rs) : 0U;
                uint64_t rowEndG = cum + rowNnz;
                if (!loFound && targetLo < rowEndG) {
                    uint64_t intra = targetLo - cum; // targetLo >= cum (head-outer monotone)
                    uint32_t intra8 = RoundUp(static_cast<uint32_t>(intra), batchGroupsMax);
                    if (intra8 > rowNnz) {
                        intra8 = rowNnz; // row too small to split at batchGroupsMax -> no split
                    }
                    rowLo = h * totalKvRows_ + r;
                    offLo = intra8;
                    loFound = true;
                }
                if (!hiFound && targetHi < rowEndG) {
                    uint64_t intra = targetHi - cum;
                    uint32_t intra8 = RoundUp(static_cast<uint32_t>(intra), batchGroupsMax);
                    if (intra8 > rowNnz) {
                        intra8 = rowNnz;
                    }
                    rowHi = h * totalKvRows_ + r;
                    offHi = intra8;
                    hiFound = true;
                }
                cum = rowEndG;
            }
        }
    }

    // Phase 1: KV-centric partial compute (Cube QK/PV + Vector softmax).
    __aicore__ inline void Phase1KvCentricCompute(
        uint32_t coreIdx, uint32_t coreNum, AscendC::GlobalTensor<ElementQ> const &gQ,
        AscendC::GlobalTensor<ElementK> const &gK, AscendC::GlobalTensor<ElementV> const &gV,
        AscendC::GlobalTensor<int32_t> const &gK2qRowPtr, AscendC::GlobalTensor<int32_t> const &gK2qQIndices,
        AscendC::GlobalTensor<int32_t> const &gK2qSlotIndices, AscendC::GlobalTensor<int32_t> const &gBlockTable,
        AscendC::GlobalTensor<ElementWorkspaceO> &gAccumOut, AscendC::GlobalTensor<float> &gSoftmaxMax,
        AscendC::GlobalTensor<float> &gSoftmaxSum)
    {
        // Batched QK/softmax/PV: BATCH_GROUPS qTokens per batch -> M = groupCount*groupSize_
        // (up to BlockMmadQK::L0_TILE_M=128). MAX_BATCH_GROUPS caps per-batch arrays; groupSize>=16
        // fills the tile, smaller groupSize underfills but stays correct.
        constexpr uint32_t MAX_BATCH_GROUPS = 8U;
        uint32_t batchGroupsMax = BlockMmadQK::L0_TILE_M / groupSize_;
        if (batchGroupsMax == 0U || batchGroupsMax > MAX_BATCH_GROUPS) {
            batchGroupsMax = MAX_BATCH_GROUPS;
        }
        if constexpr (std::is_same<ElementS, float>::value) {
            // AIV0-only softmax: full S tile must fit tmp (32KB=8192 fp32) and stats (64).
            constexpr uint32_t HIGH_PREC_TMP_FLOATS = 8192U;
            uint32_t nRound = RoundUp(blockSize_, 16U);
            uint32_t grpStride = RoundUp(groupSize_, 8U);
            while (batchGroupsMax > 0U) {
                uint32_t batchMCap = batchGroupsMax * groupSize_;
                if (batchMCap > BlockMmadQK::L0_TILE_M) {
                    batchMCap = BlockMmadQK::L0_TILE_M;
                }
                uint64_t sElem = static_cast<uint64_t>(RoundUp(batchMCap, 16U)) * nRound;
                uint32_t statsElem = batchGroupsMax * grpStride;
                if (sElem <= HIGH_PREC_TMP_FLOATS && statsElem <= 64U) {
                    break;
                }
                batchGroupsMax--;
            }
            if (batchGroupsMax == 0U) {
                batchGroupsMax = 1U;
            }
        }

#ifdef __DAV_CUBE__
        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;

        blockMmadQK.Init(resource, blockSize_, embed_);
        uint32_t qkL1Used = blockSize_ * embed_ * sizeof(ElementK) + BlockMmadQK::L0_TILE_M * embed_ * sizeof(ElementQ);
        // PV P L1 stage must hold the batched P [L0_TILE_M, blockSize_] (max batchM, K=validSize<=blockSize).
        blockMmadPV.Init(resource, qkL1Used, blockSize_, embed_, BlockMmadQK::L0_TILE_M);

        // CUBE Fixpipe dst aliases softmax lsUbTensor (offset 0,
        // MAX_UB_S_ELEM_NUM * sizeof(ElementS) per stage), shared cross-core with VEC softmax.
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(EpilogueOnlineSoftmax::MAX_UB_S_ELEM_NUM *
                                                                             sizeof(ElementS) * i);
        }

        int64_t strideKVRow = (layoutType_ == MinimaxSaSplitKv::LAYOUT_BNSD) ? static_cast<int64_t>(embed_) :
                                                                               static_cast<int64_t>(kvHeads_) * embed_;
        uint32_t qGmRowStride =
            (layoutType_ == MinimaxSaSplitKv::LAYOUT_BNSD && qSeqLen_ > 0U) ? (qSeqLen_ * embed_) : embed_;
#endif
#ifdef __DAV_VEC__
        // Softmax self-allocates UB (lsUb/lpUb/gmUb/glUb; high-prec also tmpUb).
        // Stats offsets must match SM_UB_GM_OFFSET / SM_UB_GL_OFFSET above.
        EpilogueOnlineSoftmax epilogueSoftmax(resource, scaleValue_);

        uint32_t qkL1Used = blockSize_ * embed_ * sizeof(ElementK) + BlockMmadQK::L0_TILE_M * embed_ * sizeof(ElementQ);
        uint32_t vBufBytes = blockSize_ * embed_ * sizeof(ElementV);
        uint32_t l1PStageBytes = RoundUp(BlockMmadQK::L0_TILE_M, 16U) *
                                 RoundUp(blockSize_, BYTE_PER_C0 / sizeof(ElementP)) * sizeof(ElementP);
        AscendC::LocalTensor<ElementP> l1PBuf[P_L1_BUF_NUM];
        for (uint32_t i = 0; i < P_L1_BUF_NUM; i++) {
            l1PBuf[i] = resource.l1Buf.template GetBufferByByte<ElementP>(qkL1Used + vBufBytes + l1PStageBytes * i);
        }
#endif

        // Kernel-side nnz-balanced partition. row_ptr stays a plain runtime tensor
        // (NO host value-read, NO .tolist/ConvertToTensor). Head-outer task index
        // (task = h*totalKvRows + r) so a hot row's kvHeads heads land in DIFFERENT
        // contiguous task ranges -> different cores (a row-outer contiguous range
        // would pile a hot row's 4 heads on one core = the 4x worse
        // "all-heads-per-core" trap). A boundary row may split at a batchGroupsMax-
        // aligned (8) q-token offset so a hot row is shared by <=2 cores -> ~1.000x
        // balance. AIV wall = Phase1-Vec + Phase2; Phase1-Vec is on the critical
        // path (~62%) and partition-coupled, so this balance moves op wall, not just
        // AIC. The kernel is parallel -> each core derives ONLY its own
        // [rowLo,offLo,rowHi,offHi) via ComputeCoreBounds (walks row_ptr once,
        // stops once both boundaries found) instead of materializing the full
        // 33-boundary array on every core. Deterministic -> no cross-core sync.
        uint32_t totalTasks = totalKvRows_ * kvHeads_;
        if (totalKvRows_ == 0U) {
            return; // all requests padded (q_len=kv_len=0): no packed rows
        }
        uint32_t headStride = totalKvRows_ + 1U;
        uint32_t activeCores = coreNum;
        uint32_t rowLo;
        uint32_t offLo;
        uint32_t rowHi;
        uint32_t offHi;
        ComputeCoreBounds(coreIdx, activeCores, batchGroupsMax, gK2qRowPtr, rowLo, offLo, rowHi, offHi);

        // Walker seed state (incremental round-robin packedRow -> (batchIdx, kvBlockIdx)).
        // Head-outer: packedRow = taskIdx % totalKvRows is per-head row r, which RESETS to 0 at
        // each head boundary, so the walker re-seeds at every head crossing. The (batchIdx,
        // kvBlockIdx) decode is head-INDEPENDENT (global packedRow r -> same coord for all heads),
        // so InitPackedRowCoord (skip leading kv_len=0 batches) + forward walk to r is enough.
        // prevHead init = kvHeads_ (invalid) forces the first task to re-seed.
        int32_t curPackedRow = 0;
        uint32_t batchIdx = 0;
        uint32_t kvBlockIdx = 0;
        uint32_t prevHead = kvHeads_;

        for (uint32_t taskIdx = rowLo; taskIdx <= rowHi && taskIdx < totalTasks; ++taskIdx) {
            uint32_t kvHeadIdx = taskIdx / totalKvRows_;
            uint32_t packedRow = taskIdx % totalKvRows_;
            if (kvHeadIdx != prevHead) {
                curPackedRow = 0;
                InitPackedRowCoord(batchIdx, kvBlockIdx);
                prevHead = kvHeadIdx;
            }
            // Incremental round-robin decode of packedRow -> (batchIdx, kvBlockIdx).
            // State persists within a head; Advance steps forward only.
            while (curPackedRow < static_cast<int32_t>(packedRow)) {
                AdvancePackedRowCoord(batchIdx, kvBlockIdx);
                curPackedRow++;
            }

            uint32_t kvSeqlenBatch = 0;
            uint32_t numBlocksB = 0;
            uint32_t cumQStart = 0;
            uint32_t qSeqlenBatch = 0;
            uint32_t cumKvStart = 0;
            if (batchIdx >= batch_) {
                continue;
            }
            // Fetch per-batch scalars for the resolved batchIdx (Q/KV same batch ->
            // qBatch==batchIdx). cvl loop reuses these -> 0 GM GetValue per (bi,g).
            ResolveBatchQSide(batchIdx, kvSeqlenBatch, numBlocksB, cumQStart, qSeqlenBatch, cumKvStart);
            if (numBlocksB == 0U) {
                continue;
            }
            uint32_t localBlockIdx = kvBlockIdx;

            uint32_t validSize = CalcKvBlockValidSize(kvSeqlenBatch, numBlocksB, localBlockIdx);
            (void)cumKvStart;
            (void)gBlockTable;

            uint64_t rowPtrBase = static_cast<uint64_t>(kvHeadIdx) * headStride;
            uint32_t csrStart = static_cast<uint32_t>(gK2qRowPtr.GetValue(rowPtrBase + packedRow));
            uint32_t csrEnd = static_cast<uint32_t>(gK2qRowPtr.GetValue(rowPtrBase + packedRow + 1U));
            // Per-row q-token clip at the core's [offLo, offHi) boundary (v2 split-row
            // semantics): first row (taskIdx==rowLo) starts at csrStart+offLo, last row
            // (taskIdx==rowHi) ends at csrStart+offHi, middle rows full [csrStart, csrEnd).
            // A split row is visited by BOTH sharing cores, each taking its [localStart,
            // localEnd) q-range -> distinct (q,slot) accumOut/stat slots, no overlap, no sync.
            // myNumQ/myCsrStart feed the bf16/buffer body below byte-for-byte unchanged.
            uint32_t localStart = (taskIdx == rowLo) ? (csrStart + offLo) : csrStart;
            uint32_t localEnd = (taskIdx == rowHi) ? (csrStart + offHi) : csrEnd;
            uint32_t myNumQ = (localEnd > localStart) ? (localEnd - localStart) : 0U;
            if (validSize == 0U || myNumQ == 0U) {
                continue; // empty KV block / no q-tokens in this core's slice -> skip
            }
            uint32_t myCsrStart = localStart;

            uint64_t csrDataBase = static_cast<uint64_t>(kvHeadIdx) * k2qNnzUpperBound_;
            uint32_t kvStartPos = localBlockIdx * blockSize_;

#ifdef __DAV_CUBE__
            // PA: [numPhysicalBlocks, blockSize, kvHeads, D], indexed by block_table.
            // TND contiguous: [T_kv, kvHeads, D], tokens packed by actual_seq_lengths_kv.
            // BNSD contiguous: [B, kvHeads, S, D], tokens of one head are contiguous (stride D).
            // BSND contiguous: [B, S, kvHeads, D], same intra-block stride as TND (kvHeads*D).
            uint64_t kvBlockBase;
            if (isPageAttention_ == 1U) {
                int64_t btOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_ + localBlockIdx;
                int32_t physicalBlockId = gBlockTable.GetValue(btOffset);
                kvBlockBase = static_cast<uint64_t>(physicalBlockId) * blockSize_ * kvHeads_ * embed_ +
                              static_cast<uint64_t>(kvHeadIdx) * embed_;
            } else if (layoutType_ == MinimaxSaSplitKv::LAYOUT_BNSD) {
                kvBlockBase = (static_cast<uint64_t>(batchIdx) * kvHeads_ + kvHeadIdx) * kvSeqLen_ * embed_ +
                              static_cast<uint64_t>(localBlockIdx) * blockSize_ * embed_;
            } else if (layoutType_ == MinimaxSaSplitKv::LAYOUT_BSND) {
                kvBlockBase =
                    (static_cast<uint64_t>(batchIdx) * kvSeqLen_ + static_cast<uint64_t>(localBlockIdx) * blockSize_) *
                        kvHeads_ * embed_ +
                    static_cast<uint64_t>(kvHeadIdx) * embed_;
            } else {
                kvBlockBase = (static_cast<uint64_t>(cumKvStart) + static_cast<uint64_t>(localBlockIdx) * blockSize_) *
                                  kvHeads_ * embed_ +
                              static_cast<uint64_t>(kvHeadIdx) * embed_;
            }

            auto gmKLayout = tla::MakeLayout<ElementK, LayoutK>(strideKVRow, blockSize_);
            auto gmKTensor = tla::MakeTensor(gK[kvBlockBase], gmKLayout, Arch::PositionGM{});
            blockMmadQK.LoadKResident(gmKTensor, validSize, embed_);

            auto gmVLayout = tla::MakeLayout<ElementV, LayoutV>(blockSize_, strideKVRow);
            auto gmVTensor = tla::MakeTensor(gV[kvBlockBase], gmVLayout, Arch::PositionGM{});
            blockMmadPV.LoadVResident(gmVTensor, validSize, embed_);
#endif

            // --- Batched QK/softmax/PV: group myNumQ (the core's clipped q-token count for this
            // row) into batches of batchGroupsMax ---
            // Input contract: every qToken listed for this kvBlock has causal overlap
            // (causalValidLen >= 1), so myNumQ itself is the valid count -- no cvl==0
            // pre-scan / skip needed. (myNumQ==0 / no-overlap already continued above.)
            uint32_t numBatches = CeilDiv(myNumQ, batchGroupsMax);

            // qHeadStart is shared by the CUBE QK qBase computation below.
            uint32_t qHeadStart = kvHeadIdx * groupSize_;
            // PV wsOOff arithmetic precomputes (== SlotOOffset terms), constant within this
            // kvHead: perQTokenStride = kvHeads*topK*slotOElems, kvHeadBase = kvHeadIdx*topK*slotOElems.
            uint64_t perQTokenStride = static_cast<uint64_t>(kvHeads_) * topK_ * slotOElems_;
            uint64_t kvHeadBase = static_cast<uint64_t>(kvHeadIdx) * topK_ * slotOElems_;

            // qToken/slotK ring: batch bi's identity is filled at bi and consumed by PV at
            // bi+PRE_LAUNCH (reading the ring row, no GM re-read). PRE_LAUNCH+1 stages keep the
            // row live across the 2-deep gap (slot s refilled at bi=s+3, PV reads it last at
            // bi=s+2). Derived offsets (qBase/cvl/lseOff/wsOOff) are computed at their single use
            // site from these values; only qToken/slotK -- used multiple times -- are stashed.
            constexpr uint32_t BATCH_RING_STAGES = PRE_LAUNCH + 1U;
            uint32_t batchQToken[BATCH_RING_STAGES][MAX_BATCH_GROUPS] = {};
            uint32_t batchSlotK[BATCH_RING_STAGES][MAX_BATCH_GROUPS] = {};
#ifdef __DAV_CUBE__
            uint64_t batchQBase[MAX_BATCH_GROUPS] = {};
#endif

            for (uint32_t bi = 0U; bi < numBatches + PRE_LAUNCH; bi++) {
                uint32_t stage = bi % BATCH_RING_STAGES;

                // QK + softmax for batch bi (collect + compute). Input guarantees every qToken
                // has cvl>=1; batchGroupCount = batchGroupsMax except the tail batch (remainder).
                if (bi < numBatches) {
                    uint32_t rem = myNumQ - bi * batchGroupsMax;
                    uint32_t batchGroupCount = (rem < batchGroupsMax) ? rem : batchGroupsMax;
                    uint32_t batchM = batchGroupCount * groupSize_;
#ifdef __DAV_VEC__
                    // cvl[g] is only used by the VEC softmax mask; compute it here alongside the
                    // qToken/slotK gather. Pure compute -- CalcCausalValidLen takes pre-fetched
                    // cumQStart/qSeqlenBatch/kvSeqlenBatch, 0 GM GetValue.
                    uint32_t cvlArr[MAX_BATCH_GROUPS];
#endif
                    for (uint32_t g = 0U; g < batchGroupCount; g++) {
                        uint32_t csrIdx = myCsrStart + bi * batchGroupsMax + g;
                        batchQToken[stage][g] = static_cast<uint32_t>(gK2qQIndices.GetValue(csrDataBase + csrIdx));
                        batchSlotK[stage][g] = static_cast<uint32_t>(gK2qSlotIndices.GetValue(csrDataBase + csrIdx));
#ifdef __DAV_CUBE__
                        batchQBase[g] = QTokenHeadOffset(batchQToken[stage][g], qHeadStart);
#endif
#ifdef __DAV_VEC__
                        cvlArr[g] = CalcCausalValidLen(batchQToken[stage][g], validSize, kvStartPos, cumQStart,
                                                       qSeqlenBatch, kvSeqlenBatch);
#endif
                    }

                    uint32_t ubSBufId = bi % UB_S_OTMP_BUF_STAGES;
                    uint32_t l1PBufId = bi % P_L1_BUF_NUM;
                    uint32_t mm1ToSmFlagId = ubSBufId;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag mm1ToSmFlag(mm1ToSmFlagId);
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);

#ifdef __DAV_CUBE__
                    // per-block S stride = nRound = RoundUp(validSize,16). fp32 S is AIV0-only
                    // (full batchM); bf16 still uses the per-AIV half-tile layout.
                    uint32_t mLayoutB;
                    if constexpr (std::is_same<ElementS, float>::value) {
                        mLayoutB = RoundUp(batchM, 16u);
                    } else {
                        mLayoutB = RoundUp(VecRowSplit(batchM), 16u);
                    }
                    auto ubSLayout = tla::MakeLayout<ElementS, LayoutS>(mLayoutB, RoundUp(validSize, 16U));
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayout, Arch::PositionUB{});
                    blockMmadQK(gQ, batchQBase, batchGroupCount, groupSize_, ubSTensorTla, validSize, embed_,
                                qGmRowStride, numBatches, bi, mm1ToSmFlag);
#endif
#ifdef __DAV_VEC__
                    // P L1 layout extent = validSize (uniform batch K). softmax writes the
                    // full [0, validSize) row per group (real [0, cvl[g]) + Select-zeroed
                    // [cvl[g], validSize)); PV reduces over uniform K=validSize (= residentValidSize_).
                    auto l1PLayout = tla::MakeLayout<ElementP, layout::zN>(batchM, validSize);
                    auto l1PTensor = tla::MakeTensor(l1PBuf[l1PBufId], l1PLayout, Arch::PositionL1{});
                    // cvlArr filled above alongside the qToken/slotK gather.
                    epilogueSoftmax(l1PTensor, GemmCoord{batchM, validSize, embed_}, ubSBufId, l1PBufId, mm1ToSmFlag,
                                    smToMm2Flag, cvlArr, batchGroupCount, groupSize_);
                    // softmax wrote this AIV's rowMax/rowSum into 2-deep gmUb/glUb[ubSBufId];
                    // scatter to per-qToken GM stat slots (per-AIV group ownership may straddle).
                    // lseOff computed on demand inside ScatterBatchStats from qToken/slotK.
                    ScatterBatchStats(gSoftmaxMax, gSoftmaxSum, ubSBufId, batchM, groupSize_, batchQToken[stage],
                                      batchSlotK[stage], kvHeadIdx, batchGroupCount);
#endif
                }

                // PV for batch bi-PRE_LAUNCH. wsOOff computed on demand inside blockMmadPV
                // from the batch's qToken/slotK, read from the ring row filled at bi=bDe (no GM
                // re-read). The P-L1-buffer / smToMm2Flag pipeline (3-buffer, 2-deep) is
                // independent of this metadata and stays as-is.
                if (bi >= PRE_LAUNCH) {
                    uint32_t bDe = bi - PRE_LAUNCH;
                    if (bDe < numBatches) {
                        uint32_t l1PBufIdDe = bDe % P_L1_BUF_NUM;
                        uint32_t smToMm2FlagId = l1PBufIdDe + UB_S_OTMP_BUF_STAGES;
                        Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                        uint32_t stageDe = bDe % BATCH_RING_STAGES;
#ifdef __DAV_CUBE__
                        uint32_t pvRem = myNumQ - bDe * batchGroupsMax;
                        uint32_t pvGrpCnt = (pvRem < batchGroupsMax) ? pvRem : batchGroupsMax;
                        blockMmadPV.SetL1PBuf(l1PBufIdDe);
                        blockMmadPV(gAccumOut, batchQToken[stageDe], batchSlotK[stageDe], pvGrpCnt, groupSize_, embed_,
                                    perQTokenStride, kvHeadBase, slotOElems_, numBatches, bDe, smToMm2Flag);
#endif
                    }
                }
            }
#ifdef __DAV_CUBE__
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
#endif
        }
    }

    // Phase 2: FlashDecode-style combine (IFA FlashDecodeCompute / CombineSplitKVRes).
    __aicore__ inline void Phase2CombineScale(AscendC::GlobalTensor<ElementO> &gO,
                                              AscendC::GlobalTensor<ElementWorkspaceO> &gAccumOut,
                                              AscendC::GlobalTensor<float> &gSoftmaxMax,
                                              AscendC::GlobalTensor<float> &gSoftmaxSum,
                                              AscendC::GlobalTensor<float> &gSoftmaxLse)
    {
#ifdef __DAV_VEC__
        AscendC::LocalTensor<float> ubBase = resource.ubBuf.template GetBufferByByte<float>(0);

        EpilogueRescaleO epilogueRescaleO;
        epilogueRescaleO.InitFDBuffers(ubBase, embed_, groupSize_, topK_, kvHeads_, softmaxLseFlag_, layoutType_,
                                       qSeqLen_, qHeads_);
        // VEC: GetBlockIdx() is the linear AIV id in [0, GetBlockNum()*GetSubBlockNum());
        // GetBlockNum() is the AIC block count (half of AIV count). Stride must cover
        // all AIV sub-blocks, not blockNum alone.
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        if (coreNum == 0U) {
            coreNum = 1U;
        }
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNumP2_; taskIdx += coreNum) {
            uint32_t qToken = taskIdx / kvHeads_;
            if (IsPaddingQToken(qToken)) {
                continue; // dummy request (q_len=kv_len=0) or t>=q_len: leave O/LSE unwritten
            }
            epilogueRescaleO.FlashDecodeCompute(taskIdx, totalTaskNumP2_, gO, gAccumOut, gSoftmaxMax, gSoftmaxSum,
                                                gSoftmaxLse, qHeads_, kvHeads_);
        }
#endif
    }

private:
    Arch::Resource<ArchTag> resource;

    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t groupSize_;
    uint32_t embed_;
    uint32_t blockSize_;
    uint32_t topK_;
    uint32_t totalKvRows_;
    uint32_t maxBlocksPerBatch_;
    uint32_t k2qNnzUpperBound_;
    uint32_t totalTaskNumP1_;
    uint32_t totalTaskNumP2_;
    float scaleValue_;
    uint64_t accumOutSize_;
    uint64_t lseStatSize_;
    uint64_t slotOElems_;
    uint64_t slotStatElems_;
    uint32_t isPageAttention_;
    uint32_t softmaxLseFlag_;
    uint32_t layoutType_;
    uint32_t qSeqLen_;
    uint32_t kvSeqLen_;

    AscendC::GlobalTensor<int32_t> gActualQseqlen_;
    AscendC::GlobalTensor<int32_t> gActualKvseqlen_;
};

} // namespace MinimaxSaSplitKvKernelArch35

#endif // MINIMAX_SPARSE_ATTENTION_SPLIT_KV_KERNEL_ARCH35_H
