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
 * \file sparse_flash_attention_grad_ksplit_det.h
 * \brief TND K-split: 21 AICs x 96 keys + 1 AIC x 32 keys; four AIVs reduce dq.
 * Indices contain a unique valid prefix followed by -1 padding. All stages use the
 * prefix length; empty slices skip computation but still retire every query row.
 * Causal rows selecting every visible KV use dense loads and contiguous scatter;
 * partial selections and rows with more than 2048 visible keys use sparse gather.
 * AIC iteration r: mm12(r), ACK(r-3), mm345(r-1), DONE(r-1).
 * AIV window r: gather(r), softmax(r-1), DONE(r-2), wait for barrier(r-3),
 * then scatter(r-2) or dq reduce(r-3), publish barrier(r-2), and ACK(r-2).
 * The last three windows drain the pipeline, including the final barrier token.
 * A seed token precedes the first scatter; subsequent tokens order scatter
 * across rows and publish partials to the reducers in the following window.
 * Partial data and metadata use five slots. ACK backpressure protects their
 * reuse until all four reduce AIVs finish reading. Row/flag parity is never
 * compressed when a slice is empty. v=k semantics are preserved (dv is merged into dk).
 */
#pragma once
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "./basic_modules/cube_op.h"
#include "./basic_modules/vec_op.h"
#include "./basic_modules/common_header.h"
#include "./basic_modules/ksplit_valid_prefix.h"
#include "sparse_flash_attention_grad_post.h"

namespace SFAG_BASIC {

template <typename SFAGT>
class SelectedAttentionGradKsplitDet : public VecOp<SFAGT> {
    using TILING_CLASS = typename SFAGT::tiling_class;
    using T1 = typename SFAGT::t1;
    static constexpr uint32_t ATTEN_ENABLE = SFAGT::atten_enable;
    static constexpr bool HAS_ROPE = SFAGT::has_rope;
    static constexpr bool IS_BSND = SFAGT::is_bsnd;

    static_assert(!IS_BSND, "K-split requires TND layout");
    // 21 * 96 + 32 = 2048; AICs 22/23 are unused, their four AIVs reduce dq.
    constexpr static uint32_t KSPLIT_COMPUTE_CORES = 22;
    constexpr static uint32_t KSPLIT_REDUCE_AIV_BASE = KSPLIT_COMPUTE_CORES * 2; // 核 22/23 的 4 个 AIV
    // Keep user flags in 0..10 to avoid SyncAll/runtime flags. Use matching modes
    // and scalar waits explicitly; A2/A3 do not implement a pipe-specific wait.
    constexpr static uint32_t KSPLIT_MM3_DONE_PING = 9; // AIC→本组 AIV，mm3 partial 落定（偶数行，mode 2）
    constexpr static uint32_t KSPLIT_MM3_DONE_PONG = 8; // AIC→本组 AIV，mm3 partial 落定（奇数行，mode 2）
    // All 48 AIVs: one seed plus N row-completion SETs, matched by N+1 WAITs.
    constexpr static uint32_t KSPLIT_ROW_BARRIER = 10;
    // max(N-2, 0) ACKs: rows [0, N-2) match AIC iterations [3, N].
    constexpr static uint32_t KSPLIT_MM3_ACK_FLAG = 6;
    constexpr static uint32_t KSPLIT_MM3_ACK_FLAG_PONG = 7;
    // 核内 AIC↔AIV handshake，复用既有 0~5（per core pair，语义与既有路径一致）
    constexpr static uint32_t CUBE_WAIT_VEC_PING = 0;
    constexpr static uint32_t CUBE_WAIT_VEC_PONG = 1;
    constexpr static uint32_t VEC_WAIT_CUBE_PING = 2;
    constexpr static uint32_t VEC_WAIT_CUBE_PONG = 3;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PING = 4;
    constexpr static uint32_t CUBE_WAIT_VEC_GATHER_PONG = 5;
    // dq reduce 单次处理的行数（16 行 × 576 × 4B = 36KB，UB 预算内）
    constexpr static int64_t REDUCE_CHUNK_ROWS = 16;
    // Scatter retires row r in window r+2; dq reduction follows in window r+3.
    // Workspace slot counts still match the host allocation.
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
                                   const TILING_CLASS *__restrict tilingData);

private:
    __aicore__ inline void InitKsplit(const TILING_CLASS *__restrict tilingData);
    __aicore__ inline void AicProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                      GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                      GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen,
                                      GM_ADDR query_rope, GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                      GM_ADDR workspace, const TILING_CLASS *__restrict tilingData);
    __aicore__ inline void AivProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out,
                                      GM_ADDR attention_out_grad, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                      GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen,
                                      GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope,
                                      GM_ADDR dk_rope, GM_ADDR workspace, const TILING_CLASS *__restrict tilingData);
    __aicore__ inline void GetTndSeqLenK(const GM_ADDR cur_seq_qlen_addr, const GM_ADDR cur_seq_kvlen_addr,
                                         const int64_t t1Idx);
    __aicore__ inline int32_t GetRowActualSelCount(const int64_t t1Idx);
    __aicore__ inline int32_t SliceCountOf(const uint32_t coreIdx, const int32_t rowActual) const;
    __aicore__ inline void UpdateGmOffsetKsplit(const int64_t task, const int32_t rowActual, const int32_t sliceCount,
                                                const uint32_t slot, const uint32_t parity);
    __aicore__ inline void DqReduceRow(const int64_t rowBase, const int32_t rowActual, const uint32_t slot);

    // core / row state
    uint32_t aicBlockIdx{0};
    int64_t validT1{0};
    int64_t ksN1{0};
    int64_t dimDTotal{0};
    int64_t t1Offset{0};
    int64_t t2Offset{0};
    int64_t bIndex{0};
    int64_t s1Index{0};
    int64_t sliceStart{0}; // 本核 slice 在行内的起始 block
    int64_t sliceChunk{0}; // 本核 slice 最大 block 数（96/32，空转核 0）
    uint32_t kSplitChunkPerCore{96};
    uint32_t kSplitLastChunk{32};
    int64_t ksMm12SlotLen{0};  // 每 ping/pong slot 的 mm12 元素数（fp32）
    int64_t ksMm345SlotLen{0}; // 每 ping/pong slot 的 mm345 元素数（T1）
    int64_t ksSelKWspOffset{0};
    uint32_t selectdKPPPidx{0};
    uint32_t scatterTaskId{0};
    event_t processMte2WaitV{};
    event_t reduceVMte2{};
    RunInfo runInfo[KSPLIT_SCATTER_SLOTS]; // 3 槽环：r 在算、r-1 待 scatter、r-2 scatter 中
    // Scatter metadata follows the three scatter slots. Reduction is one window
    // later, so its row base/count must survive beyond the three-slot ring.
    int32_t scatterCntRing[KSPLIT_SCATTER_SLOTS]{0, 0, 0};
    int32_t reduceActualRing[KSPLIT_PARTIAL_SLOTS]{};
    int64_t reduceRowBaseRing[KSPLIT_PARTIAL_SLOTS]{};
    GlobalTensor<int32_t> topkIndicesGmK;
    GlobalTensor<float> dqPartialGm;
    // dq reduce 直接乘 scale + cast 输出（不再经 dqWorkspace 交给 post）：dq（Dqk 段）与 dq_rope（rope 段）
    GlobalTensor<T1> dqOutGm;
    GlobalTensor<T1> dqRopeOutGm;
    // gather 走 GatherKVOptimized 的专用事件（仅 AIV 分配/使用；初始各 SET 1 次供首次 WAIT 消费）
    event_t gatherMte2WaitMte3K;
    event_t gatherMte2WaitMte3PongK;
};

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::InitKsplit(const TILING_CLASS *__restrict tilingData)
{
    // 与 VecOp::InitParams 同源的字段在此先填（AIC 侧不走 VecOp::Init），AIV 侧 VecOp::Init 会重填同值
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
    kSplitChunkPerCore = tilingData->opInfo.kSplitChunkPerCore;
    kSplitLastChunk = tilingData->opInfo.kSplitLastChunk;
    // ksplit 路径 GatherKV 分发到 GatherKVOptimized（32 行粒度冲刷，布局与 NonOptimized 一致）
    this->ksplitGatherOpt = true;

    if ASCEND_IS_AIC {
        aicBlockIdx = GetBlockIdx();
    }
    if ASCEND_IS_AIV {
        aicBlockIdx = GetBlockIdx() / 2;
    }
    if (aicBlockIdx < KSPLIT_COMPUTE_CORES - 1) {
        sliceStart = aicBlockIdx * kSplitChunkPerCore;
        sliceChunk = kSplitChunkPerCore;
    } else if (aicBlockIdx == KSPLIT_COMPUTE_CORES - 1) {
        sliceStart = (KSPLIT_COMPUTE_CORES - 1) * kSplitChunkPerCore;
        sliceChunk = kSplitLastChunk;
    } else {
        sliceStart = 0;
        sliceChunk = 0;
    }
}

template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::Process(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen, GM_ADDR query_rope,
    GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
    const TILING_CLASS *__restrict tilingData)
{
    InitKsplit(tilingData);
    topkIndicesGmK.SetGlobalBuffer((__gm__ int32_t *)topk_indices);
    // cur_seq_qlen 为 int64 cumsum（长度 B+1），末元素即有效 T1 总行数；全核读同一 GM 值，循环上界一致
    validT1 = ((__gm__ int64_t *)cur_seq_qlen)[this->dimB];

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
        // cur_seq 为 int64 cumsum（长度 B+1），curS1/curS2 由相邻元素差分得到；行号单调递增，bIndex 不回退
        int64_t curT1 = ((__gm__ int64_t *)cur_seq_qlen_addr)[bIndex + 1];
        while (t1Idx >= curT1) {
            curT1 = ((__gm__ int64_t *)cur_seq_qlen_addr)[++bIndex + 1];
        }
        t1Offset = ((__gm__ int64_t *)cur_seq_qlen_addr)[bIndex];
        t2Offset = ((__gm__ int64_t *)cur_seq_kvlen_addr)[bIndex];
        this->curS1 = ((__gm__ int64_t *)cur_seq_qlen_addr)[bIndex + 1] - ((__gm__ int64_t *)cur_seq_qlen_addr)[bIndex];
        this->curS2 =
            ((__gm__ int64_t *)cur_seq_kvlen_addr)[bIndex + 1] - ((__gm__ int64_t *)cur_seq_kvlen_addr)[bIndex];
        s1Index = t1Idx - t1Offset;
    } else {
        t1Offset = t1Idx;
        s1Index = t1Idx % this->dimS1;
        this->curS1 = this->dimS1;
        this->curS2 = this->dimS2;
        t2Offset = (t1Idx / this->dimS1) * this->dimS2;
    }
}

template <typename SFAGT>
__aicore__ inline int32_t SelectedAttentionGradKsplitDet<SFAGT>::GetRowActualSelCount(const int64_t t1Idx)
{
    int64_t maxS2Blk = (this->curS2 + this->selectedBlockSize - 1) / this->selectedBlockSize;
    if constexpr (ATTEN_ENABLE) {
        int64_t newMaxS2 = Max(this->curS2 - this->curS1 + s1Index + 1, 0);
        maxS2Blk = (newMaxS2 + this->selectedBlockSize - 1) / this->selectedBlockSize;
    }
    const int32_t upper = Min((int64_t)this->selectedBlockCount, maxS2Blk);
    const int64_t topkBase = t1Idx * this->dimN2 * this->selectedBlockCount;
    return KsplitValidPrefixCount(topkIndicesGmK, topkBase, upper);
}

template <typename SFAGT>
__aicore__ inline int32_t SelectedAttentionGradKsplitDet<SFAGT>::SliceCountOf(const uint32_t coreIdx,
                                                                              const int32_t rowActual) const
{
    if (coreIdx >= KSPLIT_COMPUTE_CORES) {
        return 0;
    }
    int64_t start = coreIdx * kSplitChunkPerCore;
    int64_t chunk = (coreIdx == KSPLIT_COMPUTE_CORES - 1) ? kSplitLastChunk : kSplitChunkPerCore;
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
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::UpdateGmOffsetKsplit(
    const int64_t task, const int32_t rowActual, const int32_t sliceCount, const uint32_t slot, const uint32_t parity)
{
    /*
     *  query:    T1 N2 G D   (TND)
     *  dy/out:   T1 N2 G D2
     *  indices:  T1 N2 SELECTED_BLOCK_COUNT
     *  sum/max:  N2 T1 G
     *  key:      T2 N2 D
     *  value:    T2 N2 D2
     * N2=1（门控保证）；blkCntOffset 恒 0，slice 起点折进 indicesGmOffset（sparse gather 单独传 sliceStart）。
     * slot（r%3）：runInfo / mm4/mm5 scatter 源槽位；parity（r%2）：mm12/mm345 握手槽位（仍为 2 份）。
     */
    RunInfo &ri = runInfo[slot];
    ri.valid = true;
    int64_t rowT1 = t1Offset + s1Index;

    int64_t queryGmOffset = rowT1 * (ksN1 * this->dimDqk);
    int64_t queryRopeGmOffset = rowT1 * (ksN1 * this->dimRope);
    int64_t dyGmOffset = rowT1 * (ksN1 * this->dimDv);
    int64_t indicesGmOffset = rowT1 * (this->dimN2 * this->selectedBlockCount) + sliceStart;
    int64_t sumGmOffset = rowT1 * this->dimG;
    int64_t keyGmOffset = t2Offset * (this->dimN2 * this->dimDqk);
    int64_t keyRopeGmOffset = t2Offset * (this->dimN2 * this->dimRope);
    int64_t valueGmOffset = t2Offset * (this->dimN2 * this->dimDv);
    int64_t mm12GmOffset = parity * ksMm12SlotLen;
    int64_t mm345GmOffset = parity * ksMm345SlotLen;
    int64_t selectedKGmOffset = selectdKPPPidx * ksSelKWspOffset;

    ri.isSmallS2 = false;
    if constexpr (ATTEN_ENABLE) {
        const int64_t visibleKeys = Max(this->curS2 - this->curS1 + s1Index + 1, 0);
        ri.isSmallS2 = KsplitIsFullCausalSelection(visibleKeys, rowActual, this->selectedBlockCount);
    }
    if (ri.isSmallS2) {
        // Dense MM1/2/3 and scatter use the slice's first KV directly. Sparse
        // gather/scatter instead resolve row-local indices from the batch base.
        keyGmOffset += sliceStart * this->dimN2 * this->dimDqk;
        keyRopeGmOffset += sliceStart * this->dimN2 * this->dimRope;
        valueGmOffset += sliceStart * this->dimN2 * this->dimDv;
    }

    ri.mm3OutGmOffset = queryGmOffset + queryRopeGmOffset; // dq 行起点（行级，不含 slice）

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
    ri.mm12GmOffset = mm12GmOffset;
    ri.mm345GmOffset = mm345GmOffset;
    ri.mm4OutGmOffset = keyGmOffset + keyRopeGmOffset;
    ri.mm5OutGmOffset = valueGmOffset;
    ri.actualSelCntOffset = sliceCount;
    ri.lastBlockSize = this->selectedBlockSize; // Host requires blockSize=1.
    ri.isLastBasicBlock = true;                 // 每核每行仅 1 个 chunk
    ri.scatterTaskId = scatterTaskId;
    ri.s1Index = s1Index;
    ri.s1Begin = 0;
    ri.s1End = 0;
    ri.actualSelectedBlockCount = sliceCount;
    ri.curS1 = this->curS1;
    ri.curS2 = this->curS2;
    ri.selectedKGmOffset = selectedKGmOffset;
    ri.processMte2WaitV = processMte2WaitV;
    if ASCEND_IS_AIV {
        // GatherKVOptimized 专用事件（AIV 侧已分配；AIC 不 gather，不使用这两个字段）
        ri.gatherMte2WaitMte3 = gatherMte2WaitMte3K;
        ri.gatherMte2WaitMte3Pong = gatherMte2WaitMte3PongK;
    }
}

// AIC iteration r: mm12(r), ACK(r-3), mm345(r-1), DONE(r-1).
template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::AicProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen, GM_ADDR query_rope,
    GM_ADDR key_rope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS *__restrict tilingData)
{
    if (aicBlockIdx >= KSPLIT_COMPUTE_CORES) {
        return; // Only this pair's AIVs participate, as dq reducers.
    }

    TPipe pipeCube;
    CubeOp<SFAGT> cubeOp;
    cubeOp.Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
                cur_seq_qlen, cur_seq_kvlen, query_rope, key_rope, dq, dk, dv, workspace, tilingData, &pipeCube);
    // scatter 区 3 槽化（host 已按 3 槽分配）：mm5Res 基址按 3 槽重定位（CubeOp 内默认按 2 槽布局）
    cubeOp.RepointMm5ResBase(KSPLIT_SCATTER_SLOTS);
    AllocEventID();
    dqPartialGm.SetGlobalBuffer((__gm__ float *)workspace +
                                tilingData->opInfo.dqPartialWorkspaceOffset / sizeof(float));

    int32_t prevSliceCount = 0; // 行 r-1 的本核 slice 计数（决定是否补做 mm345(r-1)）
    for (int64_t r = 0; r < validT1 + 1; r++) {
        int32_t sliceCount = 0;
        if (r < validT1) {
            GetTndSeqLenK(cur_seq_qlen, cur_seq_kvlen, r);
            int32_t rowActual = GetRowActualSelCount(r);
            sliceCount = SliceCountOf(aicBlockIdx, rowActual);
            uint32_t parity = r & 1;
            uint32_t slot = r % KSPLIT_SCATTER_SLOTS;
            if (sliceCount > 0) {
                scatterTaskId = slot; // mm4/mm5 写槽 r%3（scatter(r) 在窗口 r+2 读取）
                UpdateGmOffsetKsplit(r, rowActual, sliceCount, slot, parity);
                RunInfo &ri = runInfo[slot];
                // Q/Dy preload 提前：不等 gather，L1 装载与 AIV gather(r) 重叠
                WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS);
                WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_DY_EVENTS);
                cubeOp.preloadQueryAndDy(ri);
                // 等本组 AIV gather 完成
                CrossCoreWaitFlag<2>(parity == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);
                cubeOp.cube12Process(ri, 0, parity);
                CrossCoreSetFlag<2, PIPE_FIX>(parity == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
                // L1 Q/Dy 仅 mm1/2 消费（ksplit mm4/5 恒 reloadQuery/reloadDy=true 走 GM reload）：
                // mm12 发完即放行下一轮 preload，不必等 mm345
                SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS);
                SetFlag<HardEvent::MTE1_MTE2>(MM_L1_DY_EVENTS);
                selectdKPPPidx = (selectdKPPPidx + 1) % 4;
            }
        }
        // Let mm12(r) overlap vector work before bounding mm345(r-1).
        // ACK(r-3) follows both local scatters and the preceding global barrier,
        // protecting slot reuse. Empty slices must consume the same ACKs.
        if (r >= SCATTER_LAG_ROWS + 1) {
            CrossCoreWaitFlag<2>(((r - SCATTER_LAG_ROWS - 1) & 1) ? KSPLIT_MM3_ACK_FLAG_PONG : KSPLIT_MM3_ACK_FLAG);
        }
        // 行 r-1 的 mm345（AIV softmax(r-1) 已落 p/ds；与 mm12(r) 在 cube 流水上首尾衔接）
        if (prevSliceCount > 0) {
            uint32_t prevParity = (r - 1) & 1;
            RunInfo &prevRi = runInfo[(r - 1) % KSPLIT_SCATTER_SLOTS];
            CrossCoreWaitFlag<2>(prevParity == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
            cubeOp.cube345ProcessKsplit(
                prevRi, 0, prevParity,
                dqPartialGm[(aicBlockIdx * KSPLIT_PARTIAL_SLOTS + (r - 1) % KSPLIT_PARTIAL_SLOTS) * this->dimG *
                            dimDTotal]);
        }
        // flag9(r-1)：每行恰 1 次，无条件；PIPE_FIX 管序保证 mm3 Fixpipe 落定后 flag 才发出
        if (r >= 1) {
            CrossCoreSetFlag<2, PIPE_FIX>(((r - 1) & 1) ? KSPLIT_MM3_DONE_PONG : KSPLIT_MM3_DONE_PING);
        }
        prevSliceCount = sliceCount;
    }
    FreeEventID();
}

// AIV window r: gather(r), softmax(r-1), then wait for row r-3 before scattering
// row r-2 or reducing dq(r-3). The extra final window consumes the last token.
template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::AivProcess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attention_out, GM_ADDR attention_out_grad, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR topk_indices, GM_ADDR cur_seq_qlen, GM_ADDR cur_seq_kvlen, GM_ADDR key_rope,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dq_rope, GM_ADDR dk_rope, GM_ADDR workspace,
    const TILING_CLASS *__restrict tilingData)
{
    TPipe pipeVec;
    this->Init(query, key, value, attention_out, attention_out_grad, softmax_max, softmax_sum, topk_indices,
               cur_seq_qlen, cur_seq_kvlen, key_rope, dq, dk, dv, workspace, tilingData, &pipeVec);
    // VecOp initializes all three scatter slots before any AIC is released
    // by a gather signal. All AIVs, including the reducers, join initialization.
    SyncAll();
    // Seed the first scatter. Even an empty launch consumes this token in drain.
    CrossCoreSetFlag<0, PIPE_MTE3>(KSPLIT_ROW_BARRIER);

    processMte2WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    SET_FLAG(V, MTE2, processMte2WaitV);
    reduceVMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    SET_FLAG(V, MTE2, reduceVMte2);
    // GatherKVOptimized 专用事件：初始各 SET 1 次（供首个 WAIT 消费，与 bs1_basic 同款不变量）
    gatherMte2WaitMte3K = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    gatherMte2WaitMte3PongK = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    SET_FLAG(MTE3, MTE2, gatherMte2WaitMte3K);
    SET_FLAG(MTE3, MTE2, gatherMte2WaitMte3PongK);

    const bool isReduceAiv = (aicBlockIdx >= KSPLIT_COMPUTE_CORES);
    if (isReduceAiv) {
        dqPartialGm.SetGlobalBuffer((__gm__ float *)workspace +
                                    tilingData->opInfo.dqPartialWorkspaceOffset / sizeof(float));
        // dq reduce 直接写出到 dq / dq_rope 输出（*scale + cast），不再走 dqWorkspace
        dqOutGm.SetGlobalBuffer((__gm__ T1 *)dq);
        dqRopeOutGm.SetGlobalBuffer((__gm__ T1 *)dq_rope);
        // reduce 事件不变量：每行结束 vWaitMte3 / reduceVMte2 各留 1 个未消费 SET，行首复用
        SET_FLAG(MTE3, V, this->vWaitMte3);
    }

    int32_t prevSliceCount = 0;
    for (int64_t r = 0; r < validT1 + SCATTER_LAG_ROWS + 1; r++) {
        int32_t curSliceCount = 0;
        if (r < validT1) {
            GetTndSeqLenK(cur_seq_qlen, cur_seq_kvlen, r);
            const int32_t rowActual = GetRowActualSelCount(r);
            curSliceCount = SliceCountOf(aicBlockIdx, rowActual);
            const uint32_t parity = r & 1;
            const uint32_t slot = r % KSPLIT_SCATTER_SLOTS;
            scatterCntRing[slot] = curSliceCount;
            reduceActualRing[r % KSPLIT_PARTIAL_SLOTS] = rowActual;
            reduceRowBaseRing[r % KSPLIT_PARTIAL_SLOTS] = (t1Offset + s1Index) * this->dimG * dimDTotal;
            if (curSliceCount > 0) {
                scatterTaskId = slot;
                UpdateGmOffsetKsplit(r, rowActual, curSliceCount, slot, parity);
                RunInfo gatherInfo = runInfo[slot];
                gatherInfo.blkCntOffset = sliceStart;
                // The preceding window's softmax/scatter may still read this
                // aliased UB. Drain those reads before gather reuses the buffer.
                SET_FLAG(MTE3, MTE2, this->mte2WaitMte3);
                WAIT_FLAG(MTE3, MTE2, this->mte2WaitMte3);
                this->GatherKV(0, t1Offset, gatherInfo);
                // Both AIVs must signal, even if one sub-core gathered no keys.
                CrossCoreSetFlag<2, PIPE_MTE3>(parity == 0 ? CUBE_WAIT_VEC_GATHER_PING : CUBE_WAIT_VEC_GATHER_PONG);
                // Gather aliases the softmax/scatter UB; drain its writes before
                // the next stage overwrites that UB.
                SET_FLAG(MTE3, MTE2, this->mte2WaitMte3);
                WAIT_FLAG(MTE3, MTE2, this->mte2WaitMte3);
                selectdKPPPidx = (selectdKPPPidx + 1) % 4;
            }
        }

        // This uses the PREVIOUS row's activity, not the current slice count.
        if (prevSliceCount > 0) {
            const uint32_t prevParity = (r - 1) & 1;
            RunInfo &prevRi = runInfo[(r - 1) % KSPLIT_SCATTER_SLOTS];
            CrossCoreWaitFlag<2>(prevParity == 0 ? VEC_WAIT_CUBE_PING : VEC_WAIT_CUBE_PONG);
            this->VecOp<SFAGT>::Process(prevRi);
            CrossCoreSetFlag<2, PIPE_MTE3>(prevParity == 0 ? CUBE_WAIT_VEC_PING : CUBE_WAIT_VEC_PONG);
        }

        if (r >= SCATTER_LAG_ROWS) {
            const int64_t row = r - SCATTER_LAG_ROWS;
            const uint32_t slot = row % KSPLIT_SCATTER_SLOTS;
            const uint32_t parity = row & 1;
            if (row < validT1 && !isReduceAiv) {
                // Real rows consume DONE even for empty slices; final drain has none.
                CrossCoreWaitFlag<2>(parity == 0 ? KSPLIT_MM3_DONE_PING : KSPLIT_MM3_DONE_PONG);
            }
            // Consume the preceding row's completion before the next scatter.
            // The extra final window consumes the last completion token.
            CrossCoreWaitFlag<0>(KSPLIT_ROW_BARRIER);
            if (isReduceAiv && row > 0) {
                const uint32_t reduceSlot = (row - 1) % KSPLIT_PARTIAL_SLOTS;
                DqReduceRow(reduceRowBaseRing[reduceSlot], reduceActualRing[reduceSlot], reduceSlot);
            }
            if (row < validT1) {
                if (!isReduceAiv && scatterCntRing[slot] > 0) {
                    this->ScatterAddUnDeter(runInfo[slot]);
                }
                CrossCoreSetFlag<0, PIPE_MTE3>(KSPLIT_ROW_BARRIER);
                // Match AIC iterations 3..N without leaving an unused final ACK.
                if (!isReduceAiv && row + 2 < validT1) {
                    CrossCoreSetFlag<2, PIPE_MTE3>(parity == 0 ? KSPLIT_MM3_ACK_FLAG : KSPLIT_MM3_ACK_FLAG_PONG);
                }
            }
        }
        prevSliceCount = curSliceCount;
    }

    WAIT_FLAG(V, MTE2, processMte2WaitV);
    WAIT_FLAG(V, MTE2, reduceVMte2);
    // GatherKVOptimized 专用事件收尾：吸收 init（空切片行从未 gather）与末次 flush
    // 遗留的未消费 SET，保证 kernel 退出时事件静默——否则脏 flag 跨 launch 残留，
    // 下一次 launch 的首个 gather WAIT 误通过（WAR 精度损坏）或 init 双 SET 计数溢出（挂死）。
    // 对齐 bs1_basic 主循环后的 gatherMte2WaitMte3(/Pong) 收尾 WAIT。
    WAIT_FLAG(MTE3, MTE2, gatherMte2WaitMte3K);
    WAIT_FLAG(MTE3, MTE2, gatherMte2WaitMte3PongK);
    if (isReduceAiv) {
        WAIT_FLAG(MTE3, V, this->vWaitMte3);
    }
    SyncAll();
    pipeVec.Destroy();

    // task==0（全 batch 无有效行）时全核一致跳过后处理；validT1>0 时 48 AIV 全部参与 post
    if (validT1 > 0) {
        TPipe pipeCast;
        SparseFlashAttentionGradPost<T1, TILING_CLASS, 3, 0, HAS_ROPE, true> opCast;
        opCast.Init(dq, dk, dv, cur_seq_qlen, cur_seq_kvlen, dq_rope, dk_rope, workspace, tilingData, &pipeCast);
        opCast.Process();
    }
}

// Reduce only the active partials in fixed core order. An empty valid prefix
// writes zero without reading stale partial slots. This row's completion token precedes
// every call, and the next row's barrier cannot finish until this reduce completes.
template <typename SFAGT>
__aicore__ inline void SelectedAttentionGradKsplitDet<SFAGT>::DqReduceRow(const int64_t rowBase,
                                                                          const int32_t rowActual, const uint32_t slot)
{
    uint32_t reduceIdx = this->vecBlockIdx - KSPLIT_REDUCE_AIV_BASE; // 0..3
    int64_t gQuarter = this->dimG / 4;                               // 门控保证 G 16 对齐
    int64_t g0 = reduceIdx * gQuarter;

    LocalTensor<float> accTensor = this->scatterAddTensorK; // 32*dimDAlign fp32，用前 16*Dtotal
    LocalTensor<float> bufTensor = this->scatterAddTensorV; // 32*dimD2Align fp32，用前 16*Dtotal

    for (int64_t gChunk = 0; gChunk < gQuarter; gChunk += REDUCE_CHUNK_ROWS) {
        int64_t chunkRows = Min(REDUCE_CHUNK_ROWS, gQuarter - gChunk);
        int64_t chunkLen = chunkRows * dimDTotal;
        bool firstContrib = true;
        for (uint32_t c = 0; c < KSPLIT_COMPUTE_CORES; c++) {
            if (SliceCountOf(c, rowActual) == 0) {
                continue;
            }
            int64_t srcOff =
                ((int64_t)c * KSPLIT_PARTIAL_SLOTS + slot) * this->dimG * dimDTotal + (g0 + gChunk) * dimDTotal;
            WAIT_FLAG(V, MTE2, reduceVMte2);
            DataCopy(bufTensor, dqPartialGm[srcOff], chunkLen);
            SET_FLAG(MTE2, V, this->vWaitMte2);
            WAIT_FLAG(MTE2, V, this->vWaitMte2);
            if (firstContrib) {
                WAIT_FLAG(MTE3, V, this->vWaitMte3);        // acc 上一 chunk 的写出已完成
                Adds(accTensor, bufTensor, 0.0f, chunkLen); // 首贡献核：定序累加起点（x+0 逐位一致）
                firstContrib = false;
            } else {
                Add(accTensor, accTensor, bufTensor, chunkLen);
            }
            SET_FLAG(V, MTE2, reduceVMte2);
        }
        if (firstContrib) {
            // 无任何贡献核（rowActual==0）：dq 行置 0
            WAIT_FLAG(MTE3, V, this->vWaitMte3);
            WAIT_FLAG(V, MTE2, reduceVMte2);
            Duplicate(accTensor, 0.0f, chunkLen);
            SET_FLAG(V, MTE2, reduceVMte2);
        }
        PIPE_BARRIER(PIPE_V);
        // *scale + cast 到 T1，直接搬出到 dq / dq_rope 输出（不再写 dqWorkspace 交给 post）
        Muls(accTensor, accTensor, (float)this->tilingData->postTilingData.scaleValue, chunkLen);
        PIPE_BARRIER(PIPE_V);
        // castOut 用 attentionGradT1Tensor（reduce AIV 不跑 softmax，该 T1 缓冲空闲，避免与 bufTensor 复用竞态）
        LocalTensor<T1> castOut = this->attentionGradT1Tensor;
        Cast(castOut, accTensor, RoundMode::CAST_ROUND, chunkLen);
        PIPE_BARRIER(PIPE_V);
        SET_FLAG(V, MTE3, this->mte3WaitV);
        WAIT_FLAG(V, MTE3, this->mte3WaitV);
        int64_t headIdx = rowBase / dimDTotal + g0 + gChunk;
        DataCopyParams dqRepeatParams;
        dqRepeatParams.blockCount = chunkRows;
        dqRepeatParams.blockLen = this->dimDqk * sizeof(T1) / 32;
        dqRepeatParams.srcStride = this->dimRope * sizeof(T1) / 32;
        dqRepeatParams.dstStride = 0;
        DataCopy(dqOutGm[headIdx * this->dimDqk], castOut, dqRepeatParams);
        if constexpr (HAS_ROPE) {
            dqRepeatParams.blockLen = this->dimRope * sizeof(T1) / 32;
            dqRepeatParams.srcStride = this->dimDqk * sizeof(T1) / 32;
            DataCopy(dqRopeOutGm[headIdx * this->dimRope], castOut[this->dimDqk], dqRepeatParams);
        }
        SET_FLAG(MTE3, V, this->vWaitMte3);
    }
}

} // namespace SFAG_BASIC
