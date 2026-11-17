/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_grad_ksplit_det.h
 * \brief Deterministic K-split. One query row's topK is split across compute AICs.
 * AIC r: mm12(r), ACK(r-3), mm345(r-1), DONE(r-1).
 * AIV r: softmax(r-1), wait DONE(r-2), wait barrier(r-3),
 * scatter(r-2) or dq-reduce(r-3), publish barrier(r-2), ACK(r-2), gather(r+1).
 * Row 0 is gathered before the loop. gather(r+1) then overlaps mm345(r-1).
 * Empty slices still retire every flag. A seed barrier precedes the first scatter.
 */

#pragma once
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "../../basic_modules/sparse_flash_attention_grad_cube_op.h"
#include "../../basic_modules/sparse_flash_attention_grad_vec_op.h"
#include "../../basic_modules/sparse_flash_attention_grad_common_header.h"
#include "../../basic_modules/ksplit_valid_prefix.h"
#include "sparse_flash_attention_grad_post.h"

namespace SFAG_BASIC {

template <typename SFAGT>
class SelectedAttentionGradKsplitDet : public VecOp<SFAGT> {
    using TILING_CLASS = typename SFAGT::tiling_class;
    using T1 = typename SFAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = SFAGT::atten_enable;
    static constexpr bool HAS_ROPE = SFAGT::has_rope;
    static constexpr bool IS_BSND = SFAGT::is_bsnd;
    static constexpr bool KV_MERGE = SFAGT::kv_merge;

    constexpr static uint32_t KSPLIT_MM3_DONE_PING = 9;
    constexpr static uint32_t KSPLIT_MM3_DONE_PONG = 8;
    constexpr static uint32_t KSPLIT_ROW_BARRIER = 10;
    constexpr static uint32_t KSPLIT_MM3_ACK_FLAG = 6;
    constexpr static uint32_t KSPLIT_MM3_ACK_FLAG_PONG = 7;
    constexpr static uint32_t CUBE_WAIT_VEC_PING = 0;
    constexpr static uint32_t CUBE_WAIT_VEC_PONG = 1;
    constexpr static uint32_t VEC_WAIT_CUBE_PING = 2;
    constexpr static uint32_t VEC_WAIT_CUBE_PONG = 3;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PING = 4;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PONG = 5;
    constexpr static int64_t REDUCE_CHUNK_ROWS = 16;
    constexpr static int64_t SCATTER_LAG_ROWS = 2;
    constexpr static uint32_t KSPLIT_SCATTER_SLOTS = 3;
    constexpr static uint32_t KSPLIT_PARTIAL_SLOTS = 5;

public:
    __aicore__ inline SelectedAttentionGradKsplitDet(){};
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                   GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                   GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen,
                                   GM_ADDR query_rope, GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                   GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
                                   const TILING_CLASS* __restrict tilingData);

private:
    __aicore__ inline void InitKsplit(const TILING_CLASS* __restrict tilingData);
    __aicore__ inline void AicProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                      GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                      GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen,
                                      GM_ADDR query_rope, GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                      GM_ADDR workspace, const TILING_CLASS* __restrict tilingData);
    __aicore__ inline void AivProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                      GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                      GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen,
                                      GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope,
                                      GM_ADDR dk_rope, GM_ADDR workspace, const TILING_CLASS* __restrict tilingData);
    __aicore__ inline void GetTndSeqLenK(const GM_ADDR cur_seq_qlen_addr, const GM_ADDR cur_seq_kvlen_addr,
                                         const int64_t t1Idx);
    __aicore__ inline int32_t GetRowActualSelCount(const int64_t t1Idx);
    __aicore__ inline int32_t SliceCountOf(const uint32_t coreIdx, const int32_t rowActual) const;
    __aicore__ inline void UpdateGmOffsetKsplit(const int64_t task, const int32_t sliceCount, const uint32_t slot,
                                                const uint32_t parity);
    __aicore__ inline void PrefetchGather(const int64_t row, const GM_ADDR cur_seq_qlen, const GM_ADDR cur_seq_kvlen);
    __aicore__ inline void GatherKsplit128(const RunInfo& ri);
    __aicore__ inline void DqReduceRow(const int64_t rowBase, const int32_t rowActual, const uint32_t slot);

    uint32_t aicBlockIdx{0};
    uint32_t kSplitComputeCoreNum{0};
    uint32_t kSplitChunkPerCore{0};
    uint32_t kSplitLastChunk{0};
    uint32_t kSplitReduceAivNum{0};
    int64_t validT1{0};
    int64_t ksN1{0};
    int64_t dimDTotal{0};
    int64_t t1Offset{0};
    int64_t t2Offset{0};
    int64_t bIndex{0};
    int64_t s1Index{0};
    int64_t sliceStart{0};
    int64_t sliceChunk{0};
    int64_t ksMm12SlotLen{0};
    int64_t ksMm345SlotLen{0};
    int64_t ksSelKWspOffset{0};
    uint32_t selectdKPPPidx{0};
    uint32_t scatterTaskId{0};
    event_t processMte2WaitV{};
    event_t reduceVMte2{};
    RunInfo runInfo[KSPLIT_SCATTER_SLOTS];
    int32_t scatterCntRing[KSPLIT_SCATTER_SLOTS]{0, 0, 0};
    int32_t reduceActualRing[KSPLIT_PARTIAL_SLOTS]{};
    int64_t reduceRowBaseRing[KSPLIT_PARTIAL_SLOTS]{};
    GlobalTensor<int32_t> topkIndicesGmK;
    GlobalTensor<float> dqPartialGm;
    GlobalTensor<T1> dqOutGm;
    GlobalTensor<T1> dqRopeOutGm;
};

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::InitKsplit(const TILING_CLASS* __restrict tilingData)
{
    this->dimB = tilingData->opInfo.B;
    this->dimS1 = tilingData->opInfo.S1;
    this->dimS2 = tilingData->opInfo.S2;
    this->dimG = tilingData->opInfo.G;
    this->dimN2 = tilingData->opInfo.N2;
    this->dimD = tilingData->opInfo.D;
    this->dimD2 = tilingData->opInfo.D2;
    this->dimDqk = tilingData->opInfo.D;
    this->dimDv = tilingData->opInfo.D2;
    this->dimRope = tilingData->opInfo.ropeD;
    this->selectedBlockCount = tilingData->opInfo.selectedBlockCount;
    this->selectedBlockSize = tilingData->opInfo.selectedBlockSize;
    ksN1 = this->dimG * this->dimN2;
    dimDTotal = HAS_ROPE ? this->dimDqk + this->dimRope : this->dimDqk;
    ksMm12SlotLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(float);
    ksMm345SlotLen = tilingData->opInfo.mm12WorkspaceLen / 2 / sizeof(T1);
    ksSelKWspOffset = tilingData->opInfo.selectedKWorkspaceLen / sizeof(T1) / 4;
    kSplitComputeCoreNum = tilingData->opInfo.kSplitComputeCoreNum;
    kSplitChunkPerCore = tilingData->opInfo.kSplitChunkPerCore;
    kSplitLastChunk = tilingData->opInfo.kSplitLastChunk;
    kSplitReduceAivNum = tilingData->opInfo.kSplitReduceAivNum;

    if ASCEND_IS_AIC {
        aicBlockIdx = GetBlockIdx();
    }
    if ASCEND_IS_AIV {
        aicBlockIdx = GetBlockIdx() / 2;
    }
    if (aicBlockIdx + 1 < kSplitComputeCoreNum) {
        sliceStart = aicBlockIdx * kSplitChunkPerCore;
        sliceChunk = kSplitChunkPerCore;
    } else if (aicBlockIdx + 1 == kSplitComputeCoreNum) {
        sliceStart = aicBlockIdx * kSplitChunkPerCore;
        sliceChunk = kSplitLastChunk;
    } else {
        sliceStart = 0;
        sliceChunk = 0;
    }
    if constexpr (IS_BSND) {
        this->curS1 = this->dimS1;
        this->curS2 = this->dimS2;
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::Process(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen, GM_ADDR query_rope,
    GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
    const TILING_CLASS* __restrict tilingData)
{
    InitKsplit(tilingData);
    topkIndicesGmK.SetGlobalBuffer((__gm__ int32_t*)topk_indices);
    if constexpr (IS_BSND) {
        validT1 = this->dimB * this->dimS1;
    } else {
        validT1 = ((__gm__ int32_t*)cur_seq_qlen)[this->dimB - 1];
    }

    if ASCEND_IS_AIC {
        AicProcess(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                   cur_seq_qlen, cur_seq_kvlen, query_rope, key_rope, dq, dk, dv, workspace, tilingData);
    }
    if ASCEND_IS_AIV {
        AivProcess(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                   cur_seq_qlen, cur_seq_kvlen, key_rope, dq, dk, dv, dq_rope, dk_rope, workspace, tilingData);
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::GetTndSeqLenK(const GM_ADDR cur_seq_qlen_addr,
                                                                            const GM_ADDR cur_seq_kvlen_addr,
                                                                            const int64_t t1Idx)
{
    if constexpr (IS_BSND == false) {
        int64_t curT1 = ((__gm__ int32_t*)cur_seq_qlen_addr)[bIndex];
        while (t1Idx >= curT1) {
            curT1 = ((__gm__ int32_t*)cur_seq_qlen_addr)[++bIndex];
        }
        if (unlikely(bIndex == 0)) {
            t1Offset = 0;
            t2Offset = 0;
            this->curS1 = ((__gm__ int32_t*)cur_seq_qlen_addr)[bIndex];
            this->curS2 = ((__gm__ int32_t*)cur_seq_kvlen_addr)[bIndex];
        } else {
            t1Offset = ((__gm__ int32_t*)cur_seq_qlen_addr)[bIndex - 1];
            t2Offset = ((__gm__ int32_t*)cur_seq_kvlen_addr)[bIndex - 1];
            this->curS1 =
                ((__gm__ int32_t*)cur_seq_qlen_addr)[bIndex] - ((__gm__ int32_t*)cur_seq_qlen_addr)[bIndex - 1];
            this->curS2 =
                ((__gm__ int32_t*)cur_seq_kvlen_addr)[bIndex] - ((__gm__ int32_t*)cur_seq_kvlen_addr)[bIndex - 1];
        }
        s1Index = t1Idx - t1Offset;
    } else {
        t1Offset = t1Idx;
        bIndex = t1Idx / this->dimS1;
        s1Index = t1Idx % this->dimS1;
        this->curS1 = this->dimS1;
        this->curS2 = this->dimS2;
        t2Offset = bIndex * this->curS2;
    }
}

template <typename SFAGT>
__aicore__ inline int32_t SelectedAttentionGradKsplitDet<SFAGT>::GetRowActualSelCount(const int64_t t1Idx)
{
    int64_t maxS2Blk = (this->curS2 + this->selectedBlockSize - 1) / this->selectedBlockSize;
    if constexpr (ATTEN_ENABLE) {
        int64_t newMaxS2 = this->curS2 - this->curS1 + s1Index + 1;
        if (newMaxS2 < 0) {
            newMaxS2 = 0;
        }
        maxS2Blk = (newMaxS2 + this->selectedBlockSize - 1) / this->selectedBlockSize;
    }
    int32_t upper = static_cast<int32_t>(this->selectedBlockCount);
    if (maxS2Blk < upper) {
        upper = static_cast<int32_t>(maxS2Blk);
    }
    int64_t topkBase = 0;
    if constexpr (IS_BSND) {
        topkBase = t1Offset * this->dimN2 * this->selectedBlockCount;
    } else {
        topkBase = (t1Offset + s1Index) * this->dimN2 * this->selectedBlockCount;
    }
    return KsplitValidPrefixCount(topkIndicesGmK, topkBase, upper);
}

template <typename SFAGT>
__aicore__ inline int32_t SelectedAttentionGradKsplitDet<SFAGT>::SliceCountOf(const uint32_t coreIdx,
                                                                              const int32_t rowActual) const
{
    if (coreIdx >= kSplitComputeCoreNum) {
        return 0;
    }
    int64_t start = static_cast<int64_t>(coreIdx) * kSplitChunkPerCore;
    int64_t chunk = (coreIdx + 1 == kSplitComputeCoreNum) ? kSplitLastChunk : kSplitChunkPerCore;
    int64_t cnt = rowActual - start;
    if (cnt < 0) {
        cnt = 0;
    }
    if (cnt > chunk) {
        cnt = chunk;
    }
    return static_cast<int32_t>(cnt);
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::UpdateGmOffsetKsplit(const int64_t task,
                                                                                   const int32_t sliceCount,
                                                                                   const uint32_t slot,
                                                                                   const uint32_t parity)
{
    RunInfo& ri = runInfo[slot];
    ri.valid = true;
    int64_t queryGmOffset = 0;
    int64_t queryRopeGmOffset = 0;
    int64_t dyGmOffset = 0;
    int64_t indicesGmOffset = 0;
    int64_t sumGmOffset = 0;
    if constexpr (IS_BSND) {
        queryGmOffset = t1Offset * (ksN1 * this->dimDqk);
        queryRopeGmOffset = t1Offset * (ksN1 * this->dimRope);
        dyGmOffset = t1Offset * (ksN1 * this->dimDv);
        indicesGmOffset = t1Offset * (this->dimN2 * this->selectedBlockCount) + sliceStart;
        sumGmOffset = bIndex * this->dimN2 * this->dimS1 * this->dimG + s1Index * this->dimG;
    } else {
        int64_t rowT1 = t1Offset + s1Index;
        queryGmOffset = rowT1 * (ksN1 * this->dimDqk);
        queryRopeGmOffset = rowT1 * (ksN1 * this->dimRope);
        dyGmOffset = rowT1 * (ksN1 * this->dimDv);
        indicesGmOffset = rowT1 * (this->dimN2 * this->selectedBlockCount) + sliceStart;
        sumGmOffset = rowT1 * this->dimG;
    }
    int64_t keyGmOffset = t2Offset * (this->dimN2 * this->dimDqk);
    int64_t keyRopeGmOffset = t2Offset * (this->dimN2 * this->dimRope);
    int64_t valueGmOffset = t2Offset * (this->dimN2 * this->dimDv);

    ri.isSmallS2 = false;
    ri.noReload = false;
    ri.task = task;
    ri.sumGmOffset = sumGmOffset;
    ri.blkCntOffset = 0;
    ri.queryGmOffset = queryGmOffset;
    ri.queryRopeGmOffset = queryRopeGmOffset;
    ri.keyGmOffset = keyGmOffset;
    ri.keyRopeGmOffset = keyRopeGmOffset;
    ri.dyGmOffset = dyGmOffset;
    ri.valueGmOffset = valueGmOffset;
    ri.indicesGmOffset = indicesGmOffset;
    ri.mm12GmOffset = parity * ksMm12SlotLen;
    ri.mm345GmOffset = parity * ksMm345SlotLen;
    ri.mm3OutGmOffset = queryGmOffset + queryRopeGmOffset;
    ri.mm4OutGmOffset = keyGmOffset + keyRopeGmOffset;
    ri.mm5OutGmOffset = valueGmOffset;
    ri.actualSelCntOffset = sliceCount;
    ri.lastBlockSize = this->selectedBlockSize;
    ri.isLastBasicBlock = true;
    ri.scatterTaskId = scatterTaskId;
    ri.s1Index = s1Index;
    ri.s1Begin = 0;
    ri.s1End = 0;
    ri.actualSelectedBlockCount = sliceCount;
    ri.curS1 = this->curS1;
    ri.curS2 = this->curS2;
    ri.selectedKGmOffset = selectdKPPPidx * ksSelKWspOffset;
    ri.processMte2WaitV = processMte2WaitV;
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::AicProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen, GM_ADDR query_rope,
    GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS* __restrict tilingData)
{
    if (aicBlockIdx >= kSplitComputeCoreNum) {
        return;
    }

    TPipe pipeCube;
    CubeOp<SFAGT> cubeOp;
    cubeOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                cur_seq_qlen, cur_seq_kvlen, query_rope, key_rope, dq, dk, dv, workspace, tilingData, &pipeCube);
    AllocEventID();
    dqPartialGm.SetGlobalBuffer((__gm__ float*)workspace + tilingData->opInfo.dqPartialWorkspaceOffset / sizeof(float));

    int32_t prevSliceCount = 0;
    for (int64_t r = 0; r < validT1 + 1; r++) {
        int32_t sliceCount = 0;
        if (r < validT1) {
            GetTndSeqLenK(cur_seq_qlen, cur_seq_kvlen, r);
            int32_t rowActual = GetRowActualSelCount(r);
            sliceCount = SliceCountOf(aicBlockIdx, rowActual);
            uint32_t parity = static_cast<uint32_t>(r & 1);
            uint32_t slot = static_cast<uint32_t>(r % KSPLIT_SCATTER_SLOTS);
            if (sliceCount > 0) {
                scatterTaskId = slot;
                UpdateGmOffsetKsplit(r, sliceCount, slot, parity);
                RunInfo& ri = runInfo[slot];
                WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS);
                WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DY_EVENTS);
                cubeOp.preloadQueryAndDy(ri);
                CrossCoreWaitFlag<2, PIPE_MTE2>(parity == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);
                cubeOp.cube12Process(ri, 0, parity);
                CrossCoreSetFlag<2, PIPE_FIX>(parity == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
                // mm4/mm5 reload Q and Dy from GM, so the preloaded L1 can be reused by the next row.
                SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS);
                SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DY_EVENTS);
                selectdKPPPidx = (selectdKPPPidx + 1) % 4;
            }
        }
        if (r >= SCATTER_LAG_ROWS + 1) {
            uint32_t ackRow = static_cast<uint32_t>((r - SCATTER_LAG_ROWS - 1) & 1);
            CrossCoreWaitFlag<2, PIPE_MTE2>(ackRow ? KSPLIT_MM3_ACK_FLAG_PONG : KSPLIT_MM3_ACK_FLAG);
        }
        if (prevSliceCount > 0) {
            uint32_t prevParity = static_cast<uint32_t>((r - 1) & 1);
            RunInfo& prevRi = runInfo[(r - 1) % KSPLIT_SCATTER_SLOTS];
            CrossCoreWaitFlag<2, PIPE_MTE2>(prevParity == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
            int64_t partialOff =
                (static_cast<int64_t>(aicBlockIdx) * KSPLIT_PARTIAL_SLOTS + (r - 1) % KSPLIT_PARTIAL_SLOTS) *
                this->dimG * dimDTotal;
            cubeOp.cube345ProcessKsplit(prevRi, 0, prevParity, dqPartialGm[partialOff]);
        }
        if (r >= 1) {
            CrossCoreSetFlag<2, PIPE_FIX>(((r - 1) & 1) ? KSPLIT_MM3_DONE_PONG : KSPLIT_MM3_DONE_PING);
        }
        prevSliceCount = sliceCount;
    }
    FreeEventID();
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::AivProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen, GM_ADDR key_rope,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
    const TILING_CLASS* __restrict tilingData)
{
    TPipe pipeVec;
    this->Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
               cur_seq_qlen, cur_seq_kvlen, key_rope, dq, dk, dv, workspace, tilingData, &pipeVec);
    if (tilingData->opInfo.rowSumRows > 0) {
        this->PrecomputeRowSum();
    }
    SyncAll();
    CrossCoreSetFlag<0, PIPE_MTE3>(KSPLIT_ROW_BARRIER);

    processMte2WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    SET_FLAG(V, MTE2, processMte2WaitV);
    reduceVMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    SET_FLAG(V, MTE2, reduceVMte2);

    const bool isReduceAiv = (aicBlockIdx >= kSplitComputeCoreNum);
    if (isReduceAiv) {
        dqPartialGm.SetGlobalBuffer((__gm__ float*)workspace +
                                    tilingData->opInfo.dqPartialWorkspaceOffset / sizeof(float));
        dqOutGm.SetGlobalBuffer((__gm__ T1*)dq);
        dqRopeOutGm.SetGlobalBuffer((__gm__ T1*)dq_rope);
        SET_FLAG(MTE3, V, this->vWaitMte3);
    }

    // Cube waits for this flag at mm12(0). Later rows are gathered after scatter.
    PrefetchGather(0, cur_seq_qlen, cur_seq_kvlen);

    int32_t prevSliceCount = 0;
    for (int64_t r = 0; r < validT1 + SCATTER_LAG_ROWS + 1; r++) {
        int32_t curSliceCount = 0;
        if (r < validT1) {
            GetTndSeqLenK(cur_seq_qlen, cur_seq_kvlen, r);
            const int32_t rowActual = GetRowActualSelCount(r);
            curSliceCount = SliceCountOf(aicBlockIdx, rowActual);
            const uint32_t slot = static_cast<uint32_t>(r % KSPLIT_SCATTER_SLOTS);
            scatterCntRing[slot] = curSliceCount;
            int64_t rowToken = IS_BSND ? t1Offset : (t1Offset + s1Index);
            reduceActualRing[r % KSPLIT_PARTIAL_SLOTS] = rowActual;
            reduceRowBaseRing[r % KSPLIT_PARTIAL_SLOTS] = rowToken * this->dimG * dimDTotal;
        }

        if (prevSliceCount > 0) {
            const uint32_t prevParity = static_cast<uint32_t>((r - 1) & 1);
            RunInfo& prevRi = runInfo[(r - 1) % KSPLIT_SCATTER_SLOTS];
            CrossCoreWaitFlag(prevParity == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
            this->VecOp<SFAGT>::Process(prevRi);
            CrossCoreSetFlag<2, PIPE_MTE3>(prevParity == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
        }

        if (r >= SCATTER_LAG_ROWS) {
            const int64_t row = r - SCATTER_LAG_ROWS;
            const uint32_t slot = static_cast<uint32_t>(row % KSPLIT_SCATTER_SLOTS);
            const uint32_t parity = static_cast<uint32_t>(row & 1);
            if (row < validT1 && !isReduceAiv) {
                CrossCoreWaitFlag(parity == 0 ? KSPLIT_MM3_DONE_PING : KSPLIT_MM3_DONE_PONG);
            }
            CrossCoreWaitFlag<0, PIPE_MTE3>(KSPLIT_ROW_BARRIER);
            if (isReduceAiv && row > 0) {
                const uint32_t reduceSlot = static_cast<uint32_t>((row - 1) % KSPLIT_PARTIAL_SLOTS);
                DqReduceRow(reduceRowBaseRing[reduceSlot], reduceActualRing[reduceSlot], reduceSlot);
            }
            if (row < validT1) {
                if (!isReduceAiv && scatterCntRing[slot] > 0) {
                    PIPE_BARRIER(PIPE_ALL);
                    this->ScatterAddUnDeter(runInfo[slot]);
                }
                CrossCoreSetFlag<0, PIPE_MTE3>(KSPLIT_ROW_BARRIER);
                if (!isReduceAiv && row + 2 < validT1) {
                    CrossCoreSetFlag<2, PIPE_MTE3>(parity == 0 ? KSPLIT_MM3_ACK_FLAG : KSPLIT_MM3_ACK_FLAG_PONG);
                }
            }
        }
        // gather(r+1) reuses the runInfo slot scatter(r-2) just released, and overlaps mm345(r-1).
        if (!isReduceAiv && prevSliceCount > 0) {
            PIPE_BARRIER(PIPE_ALL);
        }
        PrefetchGather(r + 1, cur_seq_qlen, cur_seq_kvlen);
        prevSliceCount = curSliceCount;
    }

    WAIT_FLAG(V, MTE2, processMte2WaitV);
    WAIT_FLAG(V, MTE2, reduceVMte2);
    if (isReduceAiv) {
        WAIT_FLAG(MTE3, V, this->vWaitMte3);
    }
    SyncAll();
    pipeVec.Destroy();

    if (validT1 > 0) {
        TPipe pipeCast;
        if constexpr (KV_MERGE) {
            SparseFlashAttentionGradPost<T1, TILING_CLASS, false, 3, 0, HAS_ROPE> opCast;
            opCast.skipDq = true;
            opCast.Init(dq, dk, dv, cur_seq_qlen, cur_seq_kvlen, dq_rope, dk_rope, workspace, tilingData, &pipeCast);
            opCast.Process();
        } else {
            SparseFlashAttentionGradPost<T1, TILING_CLASS, true, 3, 0, HAS_ROPE> opCast;
            opCast.skipDq = true;
            opCast.Init(dq, dk, dv, cur_seq_qlen, cur_seq_kvlen, dq_rope, dk_rope, workspace, tilingData, &pipeCast);
            opCast.Process();
        }
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::PrefetchGather(const int64_t row,
                                                                             const GM_ADDR cur_seq_qlen,
                                                                             const GM_ADDR cur_seq_kvlen)
{
    if (row < 0 || row >= validT1 || aicBlockIdx >= kSplitComputeCoreNum) {
        return;
    }
    GetTndSeqLenK(cur_seq_qlen, cur_seq_kvlen, row);
    const int32_t rowActual = GetRowActualSelCount(row);
    const int32_t sliceCount = SliceCountOf(aicBlockIdx, rowActual);
    if (sliceCount <= 0) {
        return;
    }
    const uint32_t parity = static_cast<uint32_t>(row & 1);
    const uint32_t slot = static_cast<uint32_t>(row % KSPLIT_SCATTER_SLOTS);
    scatterTaskId = slot;
    UpdateGmOffsetKsplit(row, sliceCount, slot, parity);
    if (kSplitComputeCoreNum == 16 && kSplitChunkPerCore == 128 && kSplitLastChunk == 128 &&
        !this->enableOptimizedScatter && this->selectedBlockSize == 1 && this->dimDqk == 512 && this->dimN2 == 1) {
        GatherKsplit128(runInfo[slot]);
    } else {
        RunInfo gatherInfo = runInfo[slot];
        gatherInfo.blkCntOffset = sliceStart;
        this->GatherKV(0, t1Offset, gatherInfo);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(parity == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);
    selectdKPPPidx = (selectdKPPPidx + 1) % 4;
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::GatherKsplit128(const RunInfo& ri)
{
    constexpr uint32_t SLICE_ROWS = 128;
    constexpr uint32_t BATCH_ROWS = 16;
    const uint32_t split = (ri.actualSelCntOffset + 1) / 2;
    const uint32_t first = this->subBlockIdx == 0 ? 0 : split;
    const uint32_t end = this->subBlockIdx == 0 ? split : ri.actualSelCntOffset;
    if (first == end) {
        return;
    }

    // Stage the indices in the unused tail of the 64-row key ping buffer.
    LocalTensor<int32_t> indices =
        this->gatherTensorPing[BATCH_ROWS * this->dimDqk].template ReinterpretCast<int32_t>();
    SET_FLAG(MTE3, MTE2, this->mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, this->mte2WaitMte3);
    DataCopy(indices, this->topkIndicesGm[ri.indicesGmOffset], SLICE_ROWS);
    SET_FLAG(MTE2, S, this->sWaitMte2);
    WAIT_FLAG(MTE2, S, this->sWaitMte2);
    SET_FLAG(MTE3, MTE2, this->mte2WaitMte3);
    SET_FLAG(MTE3, MTE2, this->mte2WaitMte3Pong);

    const uint32_t keyBytes = this->dimDqk * sizeof(T1);
    const uint32_t ropeBytes = this->dimRope * sizeof(T1);
    const int64_t ropePlane = PER_LOOP_BLOCK_SIZE * this->dimDqk;
    bool pong = false;
    for (uint32_t row = first; row < end; row += BATCH_ROWS) {
        const uint32_t count = end - row < BATCH_ROWS ? end - row : BATCH_ROWS;
        const event_t freeEvent = pong ? this->mte2WaitMte3Pong : this->mte2WaitMte3;
        const event_t readyEvent = pong ? this->mte3WaitMte2Pong : this->mte3WaitMte2;
        LocalTensor<T1>& key = pong ? this->gatherTensorPong : this->gatherTensorPing;
        LocalTensor<T1>& rope = pong ? this->gatherRopePong : this->gatherRopePing;
        WAIT_FLAG(MTE3, MTE2, freeEvent);
        for (uint32_t j = 0; j < count;) {
            const int64_t key0 = indices.GetValue(row + j);
            const bool hasPair = j + 1 < count;
            const int64_t key1 = hasPair ? indices.GetValue(row + j + 1) : key0;
            const uint32_t gap = key1 > key0 ? static_cast<uint32_t>(key1 - key0 - 1) : 0;
            const bool strided = key1 > key0 && gap <= UINT32_MAX / keyBytes;
            const uint16_t blocks = strided ? 2 : 1;
            DataCopyExtParams keyIn{blocks, keyBytes, strided ? gap * keyBytes : 0, 0, 0};
            DataCopyPad(key[j * this->dimDqk], this->keyGm[ri.keyGmOffset + key0 * this->dimDqk], keyIn,
                        this->padParams);
            if constexpr (HAS_ROPE) {
                DataCopyExtParams ropeIn{blocks, ropeBytes, strided ? gap * ropeBytes : 0, 0, 0};
                DataCopyPad(rope[j * this->dimRope], this->keyRopeGm[ri.keyRopeGmOffset + key0 * this->dimRope], ropeIn,
                            this->padParams);
            }
            j += blocks;
        }
        SET_FLAG(MTE2, MTE3, readyEvent);
        WAIT_FLAG(MTE2, MTE3, readyEvent);
        DataCopyExtParams keyOut{static_cast<uint16_t>(count), keyBytes, 0, 0, 0};
        DataCopyPad(this->selectedKWorkspaceGm[ri.selectedKGmOffset + row * this->dimDqk], key, keyOut);
        if constexpr (HAS_ROPE) {
            DataCopyExtParams ropeOut{static_cast<uint16_t>(count), ropeBytes, 0, 0, 0};
            DataCopyPad(this->selectedKWorkspaceGm[ri.selectedKGmOffset + ropePlane + row * this->dimRope], rope,
                        ropeOut);
        }
        SET_FLAG(MTE3, MTE2, freeEvent);
        pong = !pong;
    }
    WAIT_FLAG(MTE3, MTE2, this->mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, this->mte2WaitMte3Pong);
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::DqReduceRow(const int64_t rowBase,
                                                                          const int32_t rowActual, const uint32_t slot)
{
    uint32_t reduceIdx = this->vecBlockIdx - kSplitComputeCoreNum * 2;
    int64_t gPerAiv = this->dimG / kSplitReduceAivNum;
    int64_t g0 = static_cast<int64_t>(reduceIdx) * gPerAiv;
    int64_t halfFloats = static_cast<int64_t>(this->NON_OPT_UB_ROW_SIZE) * this->dimDAlign;
    LocalTensor<float> accTensor = this->scatterAddTensorK;
    LocalTensor<float> bufTensor = this->scatterAddTensorK[halfFloats];
    LocalTensor<T1> castOut = accTensor.template ReinterpretCast<T1>()[halfFloats * (sizeof(float) / sizeof(T1))];

    for (int64_t gChunk = 0; gChunk < gPerAiv; gChunk += REDUCE_CHUNK_ROWS) {
        int64_t chunkRows = gPerAiv - gChunk;
        if (chunkRows > REDUCE_CHUNK_ROWS) {
            chunkRows = REDUCE_CHUNK_ROWS;
        }
        int64_t chunkLen = chunkRows * dimDTotal;
        bool firstContrib = true;
        for (uint32_t c = 0; c < kSplitComputeCoreNum; c++) {
            if (SliceCountOf(c, rowActual) == 0) {
                continue;
            }
            int64_t srcOff = (static_cast<int64_t>(c) * KSPLIT_PARTIAL_SLOTS + slot) * this->dimG * dimDTotal +
                             (g0 + gChunk) * dimDTotal;
            WAIT_FLAG(V, MTE2, reduceVMte2);
            DataCopy(bufTensor, dqPartialGm[srcOff], static_cast<uint32_t>(chunkLen));
            SET_FLAG(MTE2, V, this->vWaitMte2);
            WAIT_FLAG(MTE2, V, this->vWaitMte2);
            if (firstContrib) {
                WAIT_FLAG(MTE3, V, this->vWaitMte3);
                Adds(accTensor, bufTensor, 0.0f, static_cast<uint32_t>(chunkLen));
                firstContrib = false;
            } else {
                Add(accTensor, accTensor, bufTensor, static_cast<uint32_t>(chunkLen));
            }
            SET_FLAG(V, MTE2, reduceVMte2);
        }
        if (firstContrib) {
            WAIT_FLAG(MTE3, V, this->vWaitMte3);
            WAIT_FLAG(V, MTE2, reduceVMte2);
            Duplicate(accTensor, 0.0f, static_cast<uint32_t>(chunkLen));
            SET_FLAG(V, MTE2, reduceVMte2);
        }
        PIPE_BARRIER(PIPE_V);
        Muls(accTensor, accTensor, static_cast<float>(this->tilingData->postTilingData.scaleValue),
             static_cast<uint32_t>(chunkLen));
        PIPE_BARRIER(PIPE_V);
        Cast(castOut, accTensor, RoundMode::CAST_ROUND, static_cast<uint32_t>(chunkLen));
        PIPE_BARRIER(PIPE_V);
        SET_FLAG(V, MTE3, this->mte3WaitV);
        WAIT_FLAG(V, MTE3, this->mte3WaitV);
        int64_t headIdx = rowBase / dimDTotal + g0 + gChunk;
        DataCopyParams dqRepeatParams;
        dqRepeatParams.blockCount = static_cast<uint16_t>(chunkRows);
        dqRepeatParams.blockLen = static_cast<uint16_t>(this->dimDqk * sizeof(T1) / 32);
        dqRepeatParams.srcStride = static_cast<uint16_t>(this->dimRope * sizeof(T1) / 32);
        dqRepeatParams.dstStride = 0;
        DataCopy(dqOutGm[headIdx * this->dimDqk], castOut, dqRepeatParams);
        if constexpr (HAS_ROPE) {
            dqRepeatParams.blockLen = static_cast<uint16_t>(this->dimRope * sizeof(T1) / 32);
            dqRepeatParams.srcStride = static_cast<uint16_t>(this->dimDqk * sizeof(T1) / 32);
            DataCopy(dqRopeOutGm[headIdx * this->dimRope], castOut[this->dimDqk], dqRepeatParams);
        }
        SET_FLAG(MTE3, V, this->vWaitMte3);
    }
}

} // namespace SFAG_BASIC
