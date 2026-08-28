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
 * \file bsa_radix_topk_service_arch35.h
 * \brief
 */
#ifndef BSA_RADIX_TOPK_SERVICE_ARCH35_H
#define BSA_RADIX_TOPK_SERVICE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include <limits.h>
#include "bsa_select_block_mask_common_arch35.h"
#include "bsa_select_block_mask_tiling_data_arch35.h"

template <typename BSAT>
class BSARadixTopKService {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using SCORE_T = half;
    using OUT_T = typename BSAT::outputT;
    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV;
    static constexpr uint64_t COMPARES_ALIGN_LEN = REPEAT_SIZE / sizeof(SCORE_T);

    __aicore__ inline BSARadixTopKService(){};
    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGM(GlobalTensor<SCORE_T> &attnScoreGm, GlobalTensor<int32_t> &topkWorkspaceGm,
                                  GlobalTensor<uint8_t> &maskOutGmU8);
    __aicore__ inline void UpdateRuntimeParams(uint32_t validXBlocks, uint32_t validYBlocks);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    __aicore__ inline void ProcessRadixTopKAndWriteMask(uint32_t batchIdx, uint32_t headIdx);

private:
    __aicore__ inline void ConfigureTopKParams(uint64_t sortLen, uint64_t kValue);
    __aicore__ inline void BindWorkspaceViews();
    __aicore__ inline void NegateDataForLargest(TBuf<TPosition::VECIN> &xBuf, uint64_t curTileLen);
    __aicore__ inline void TwiddleInB16(TBuf<TPosition::VECIN> &xBuf, uint64_t curTileLen, int32_t roundId,
                                        int32_t tileId);
    __aicore__ inline void DoAndMask(TBuf<TPosition::VECIN> &xBuf, uint64_t curTileLen);
    __aicore__ inline void CalcCumsumHistogram(TBuf<TPosition::VECIN> &xBuf, int32_t roundId, int32_t tileId,
                                               uint64_t curTileLen);
    __aicore__ inline void CopyIn(TBuf<TPosition::VECIN> &xBuf, uint64_t dataNum, uint64_t xOffset);
    __aicore__ inline void ClearTileTopKInWs();
    __aicore__ inline void ClearTileHistInWs(uint32_t dstOffset, uint32_t dataNum);
    __aicore__ inline void CopyTileHistWs2Ub(LocalTensor<int32_t> &dstTensor, uint32_t srcOffset, uint32_t dataNum);
    __aicore__ inline void CopyTileTopKWs2Ub(LocalTensor<int32_t> &dstTensor, uint32_t srcOffset, uint32_t dataNum);
    __aicore__ inline void CopyTileTopKUb2Ws(LocalTensor<int32_t> &srcTensor, uint32_t dstOffset, uint32_t dataNum);
    __aicore__ inline void CopyTileHistInWs(uint32_t dstOffset, uint32_t srcOffset);
    __aicore__ inline void SubTileHistWs2Ub(LocalTensor<int32_t> &dstTensor, LocalTensor<int32_t> &srcTensor,
                                            uint32_t srcOffset0, uint32_t srcOffset1, uint32_t dataNum);
    __aicore__ inline void SubTileHistInWs(uint32_t dstOffset, uint32_t srcOffset0, uint32_t srcOffset1,
                                           uint32_t dataNum);
    __aicore__ inline void SubTileHistInWsAll(uint32_t dstOffset, uint32_t srcOffset0, uint32_t srcOffset1);
    __aicore__ inline void AddTileHist2TileTopKInWs(uint64_t tileHistOffset);
    __aicore__ inline void ClearHistInWs(int32_t roundId);
    template <bool isInit>
    __aicore__ inline void CopyOutHistToWs(uint64_t maskLen, uint64_t gmOffset);
    __aicore__ inline bool Update(int32_t roundId);
    __aicore__ inline void CopyInGlobalHist(LocalTensor<int32_t> &globalHist);
    __aicore__ inline void ReduceGlobalHist(LocalTensor<int32_t> &globalHist);
    __aicore__ inline void FindBoundaryBin(const LocalTensor<int32_t> &globalHist, int32_t roundId);
    __aicore__ inline void AddTileHistToTileTopK(LocalTensor<int32_t> &globalHist);
    __aicore__ inline void HandleLastRoundBoundary(LocalTensor<int32_t> &resTensor, LocalTensor<float> &tileHistFp32);
    __aicore__ inline void WriteCoreTopKFromWs(LocalTensor<int32_t> &resTensor, LocalTensor<float> &tileHistFp32);
    __aicore__ inline void CopyOutBoundaryBinCumSum(LocalTensor<int32_t> &resTensor);
    __aicore__ inline void ComputeCumSumPrev(LocalTensor<int32_t> &resTensor, uint64_t &cumSumValuePrev);
    __aicore__ inline void CopyOutCoreTopK(LocalTensor<int32_t> &resTensor, int32_t totalTileTopKInCore);
    __aicore__ inline void SubTopKAndWriteMaskGT(LocalTensor<half> &xLocal, half boundaryHalf, uint64_t curTileLen,
                                                 uint64_t outputGmOffset, int32_t &curTileK);
    __aicore__ inline void SubTopKAndWriteMaskEQ(LocalTensor<half> &xLocal, half boundaryHalf, uint64_t curTileLen,
                                                 uint64_t outputGmOffset, int32_t &curTileK);
    __aicore__ inline void SubTopKAndWriteMaskGTTnd(LocalTensor<half> &xLocal, half boundaryHalf, uint64_t curTileLen,
                                                    uint64_t compactTileOffset, uint64_t outputHeadOffset,
                                                    int32_t &curTileK);
    __aicore__ inline void SubTopKAndWriteMaskEQTnd(LocalTensor<half> &xLocal, half boundaryHalf, uint64_t curTileLen,
                                                    uint64_t compactTileOffset, uint64_t outputHeadOffset,
                                                    int32_t &curTileK);
    __aicore__ inline void ProcessSingleElementMask(uint32_t batchIdx, uint32_t headIdx);
    __aicore__ inline void TileTopK(uint32_t batchIdx, uint32_t headIdx);
    __aicore__ inline void VToSSync();
    __aicore__ inline void SToVSync();
    __aicore__ inline void MTE2ToMTE3Sync();
    __aicore__ inline void MTE3ToMTE2Sync();
    __aicore__ inline void MTE3ToVSync();
    __aicore__ inline void VToMTE3Sync();
    __aicore__ inline void MTE2ToVSync();
    __aicore__ inline void MTE2ToSSync();
    __aicore__ inline void MTE3ToSSync();
    __aicore__ inline void SToMTE3Sync();

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    GlobalTensor<SCORE_T> attnScoreGmLocal;
    GlobalTensor<int32_t> topkWorkspaceGmLocal;
    GlobalTensor<uint8_t> maskOutGmU8Local;

    GlobalTensor<int32_t> globalHistGm_;
    GlobalTensor<int32_t> boundaryBinCumSumGm_;
    GlobalTensor<int32_t> coreTopKGm_;
    GlobalTensor<int32_t> tileTopKGm_;
    GlobalTensor<int32_t> tileHistGm_;

    TBuf<TPosition::VECIN> xBufPing_;
    TBuf<TPosition::VECIN> xBufPong_;
    TBuf<TPosition::VECOUT> outValueBuf_;
    TBuf<TPosition::VECOUT> outIndexBuf_;
    TBuf<TPosition::VECOUT> maskLocalBuf_;
    TBuf<TPosition::VECOUT> globalHistBuf_;
    TBuf<TPosition::VECCALC> tileHistAndTopKBuf_;
    TBuf<TPosition::VECCALC> tempBuf_;

    int16_t involvedMask16_;
    int16_t andMask16_;
    int32_t boundaryBin;
    int32_t boundaryBinPrev;
    uint64_t globalHistBoundaryNum_;
    uint64_t totalDefinitelyInTopK_;
    uint64_t remainK_;
    uint64_t blockOffset_;
    uint64_t headScoreOffset_;
    uint64_t maskOutRowStride_;
    uint64_t globalHistSum_[RADIX_NUM_BINS];

    uint64_t tileLen_;
    uint64_t tileLenScoreAlign_;
    uint64_t tileStartId_;
    uint64_t tileNum_;
    uint64_t tileNumAlign_;
    uint64_t sortLen_;
    uint64_t kValue_;
    uint64_t maxTileLen_;
    uint64_t maxTileLenScoreAlign_;
    bool useCompactTopKMapping_;
    uint32_t validXBlocks_;
    uint32_t validYBlocks_;
    uint32_t round_;
    uint32_t numValue_;
    uint32_t bitsPerRound_;

    uint64_t totalTileNum_;
    uint64_t formerCoreNum_;
    uint64_t tailCoreNum_;
    uint64_t formerTileNum_;
    uint64_t tailTileNum_;
    uint64_t formerTileLen_;
    uint64_t tailTileLen_;
    uint32_t tileNumRepeatTimes_;
    uint32_t tileNumRemain_;
    uint32_t tileNumBy2RepeatTimes_;
    uint32_t tileNumBy2Remain_;
    uint32_t tileNumBy3RepeatTimes_;
    uint32_t tileNumBy3Remain_;

    event_t eventIDMTE2ToVForX_;
    bool pingPongFlag;
};

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitParams(
    const BSAConstInfo &constInfo, const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;

    this->bitsPerRound_ = RADIX_BITS_PER_ROUND;
    this->numValue_ = RADIX_NUM_BINS;
    this->round_ = RADIX_ROUNDS_FP16;
    if (constInfo.usePostBlockShape) {
        this->validXBlocks_ = constInfo.postXBlocks;
        this->validYBlocks_ = constInfo.postYBlocks;
        // post 模式直接输出粗粒度 mask：行跨距用 postYBlocks（输出布局 [B,N,postX,postY]）
        this->maskOutRowStride_ = constInfo.postYBlocks;
    } else {
        this->validXBlocks_ = constInfo.xBlocks;
        this->validYBlocks_ = constInfo.yBlocks;
        this->maskOutRowStride_ = constInfo.yBlocks;
    }
    this->useCompactTopKMapping_ = tilingData->baseParams.useActualSeqLenQ && tilingData->baseParams.useActualSeqLenK;

    uint64_t maxSortLen = constInfo.usePostBlockShape ?
                              static_cast<uint64_t>(constInfo.postXBlocks) * constInfo.postYBlocks :
                              static_cast<uint64_t>(constInfo.xBlocks) * constInfo.yBlocks;
    ConfigureTopKParams(maxSortLen, constInfo.topKValue);
    this->maxTileLen_ = tileLen_;
    this->maxTileLenScoreAlign_ = tileLenScoreAlign_;
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ConfigureTopKParams(uint64_t sortLen, uint64_t kValue)
{
    this->sortLen_ = sortLen;
    this->kValue_ = kValue;
    this->tileLen_ = BSAMin(static_cast<uint64_t>(7168), BSAMax(static_cast<uint64_t>(128), sortLen_));
    this->tileLenScoreAlign_ = BSAAlignTo(tileLen_, COMPARES_ALIGN_LEN);
    this->totalTileNum_ = (sortLen_ + tileLen_ - 1) / tileLen_;
    this->tileNum_ = totalTileNum_;
    this->tileNumAlign_ = BSACeilDiv(tileNum_, static_cast<uint64_t>(8)) * 8;

    uint32_t coreNum = constInfo.aivNum;
    uint64_t baseTilesPerCore = totalTileNum_ / coreNum;
    uint64_t extraCores = totalTileNum_ % coreNum;
    this->formerCoreNum_ = extraCores;
    this->tailCoreNum_ = coreNum - extraCores;
    this->formerTileNum_ = baseTilesPerCore + (extraCores > 0 ? 1 : 0);
    this->tailTileNum_ = baseTilesPerCore;

    uint64_t myTileNum = (GetBlockIdx() < formerCoreNum_) ? formerTileNum_ : tailTileNum_;
    this->tileStartId_ = (GetBlockIdx() < formerCoreNum_) ?
                             GetBlockIdx() * formerTileNum_ :
                             formerCoreNum_ * formerTileNum_ + (GetBlockIdx() - formerCoreNum_) * tailTileNum_;
    this->tileNum_ = myTileNum;
    this->tileNumAlign_ = BSACeilDiv(myTileNum, static_cast<uint64_t>(8)) * 8;
    this->formerTileLen_ = tileLen_;
    uint64_t lastTileLen = sortLen_ - (totalTileNum_ - 1) * tileLen_;
    if (lastTileLen == 0) {
        lastTileLen = tileLen_;
    }
    this->tailTileLen_ = (myTileNum > 0 && tileStartId_ + myTileNum == totalTileNum_) ? lastTileLen : tileLen_;

    this->tileNumRepeatTimes_ = static_cast<uint32_t>(myTileNum / MAX_TILE_NUM_IN_UB);
    this->tileNumRemain_ = static_cast<uint32_t>(myTileNum % MAX_TILE_NUM_IN_UB);
    this->tileNumBy2RepeatTimes_ = static_cast<uint32_t>(myTileNum / MAX_TILE_NUM_IN_UB_BY2);
    this->tileNumBy2Remain_ = static_cast<uint32_t>(myTileNum % MAX_TILE_NUM_IN_UB_BY2);
    this->tileNumBy3RepeatTimes_ = static_cast<uint32_t>(myTileNum / MAX_TILE_NUM_IN_UB_BY3);
    this->tileNumBy3Remain_ = static_cast<uint32_t>(myTileNum % MAX_TILE_NUM_IN_UB_BY3);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::UpdateRuntimeParams(uint32_t validXBlocks, uint32_t validYBlocks)
{
    uint64_t sortLen;
    if (constInfo.usePostBlockShape) {
        uint32_t validPostXBlocks = BSACeilDiv(validXBlocks, static_cast<uint32_t>(constInfo.postBlockShapeX));
        uint32_t validPostYBlocks = BSACeilDiv(validYBlocks, static_cast<uint32_t>(constInfo.postBlockShapeY));
        this->validXBlocks_ = validPostXBlocks;
        this->validYBlocks_ = validPostYBlocks;
        sortLen = static_cast<uint64_t>(validPostXBlocks) * validPostYBlocks;
    } else {
        this->validXBlocks_ = validXBlocks;
        this->validYBlocks_ = validYBlocks;
        sortLen = static_cast<uint64_t>(validXBlocks) * validYBlocks;
    }
    float ratio = (constInfo.sparsityMode == BSASparseMode::TopK) ? constInfo.sparsity : (1.0f - constInfo.sparsity);
    int64_t signedSortLen = static_cast<int64_t>(sortLen);
    float selectedCount = ratio * static_cast<float>(signedSortLen);
    int64_t signedKValue = static_cast<int64_t>(selectedCount + 0.5f);
    uint64_t kValue = static_cast<uint64_t>(signedKValue);
    kValue = BSAMax(kValue, static_cast<uint64_t>(1));
    ConfigureTopKParams(sortLen, kValue);
    BindWorkspaceViews();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(xBufPing_, maxTileLenScoreAlign_ * sizeof(float));
    pipe->InitBuffer(xBufPong_, maxTileLenScoreAlign_ * sizeof(float));
    pipe->InitBuffer(outValueBuf_, maxTileLenScoreAlign_ * sizeof(float));
    pipe->InitBuffer(outIndexBuf_, maxTileLenScoreAlign_ * sizeof(float));
    pipe->InitBuffer(tileHistAndTopKBuf_, MAX_TILE_NUM_IN_UB * sizeof(int32_t));

    uint32_t cmpMaskSize = BSAAlignTo(maxTileLen_, static_cast<uint64_t>(VEC_ALIGN_SIZE)) * sizeof(uint8_t) / BYTE_SIZE;
    pipe->InitBuffer(tempBuf_, cmpMaskSize);

    pipe->InitBuffer(globalHistBuf_, numValue_ * sizeof(int32_t));
    pipe->InitBuffer(maskLocalBuf_,
                     maxTileLen_ * sizeof(int32_t) + maxTileLen_ * sizeof(int16_t) + maxTileLen_ * sizeof(uint8_t));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitGM(GlobalTensor<SCORE_T> &attnScoreGm,
                                                         GlobalTensor<int32_t> &topkWorkspaceGm,
                                                         GlobalTensor<uint8_t> &maskOutGmU8)
{
    this->attnScoreGmLocal = attnScoreGm;
    this->topkWorkspaceGmLocal = topkWorkspaceGm;
    this->maskOutGmU8Local = maskOutGmU8;

    BindWorkspaceViews();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::BindWorkspaceViews()
{
    uint32_t coreNum = constInfo.aivNum;
    uint32_t headerSize = (numValue_ * coreNum + coreNum + coreNum);
    this->globalHistGm_.SetGlobalBuffer((__gm__ int32_t *)topkWorkspaceGmLocal.GetPhyAddr());
    this->boundaryBinCumSumGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGmLocal.GetPhyAddr() + numValue_ * coreNum));
    this->coreTopKGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGmLocal.GetPhyAddr() + numValue_ * coreNum + coreNum));

    uint64_t tileTopKOffset = tileStartId_;
    uint64_t tileHistOffset = numValue_ * tileTopKOffset;

    this->tileTopKGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGmLocal.GetPhyAddr() + headerSize + tileTopKOffset));
    this->tileHistGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGmLocal.GetPhyAddr() + headerSize + totalTileNum_ + tileHistOffset));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::AllocEventID()
{
    eventIDMTE2ToVForX_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::FreeEventID()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ProcessRadixTopKAndWriteMask(uint32_t batchIdx, uint32_t headIdx)
{
    if (sortLen_ == 1) {
        ProcessSingleElementMask(batchIdx, headIdx);
        return;
    }
    // 1. Calculate blockOffset
    if (constInfo.usePostBlockShape) {
        // pooledScore 按最大粗粒度网格 (postXBlocks * postYBlocks) 逐 head 布局（与写入侧一致）
        this->headScoreOffset_ = (static_cast<uint64_t>(batchIdx) * constInfo.numHeads + headIdx) *
                                 constInfo.postXBlocks * constInfo.postYBlocks;
    } else {
        this->headScoreOffset_ = 0;
    }
    this->blockOffset_ = tileStartId_ * tileLen_;

    // 3. ClearTileTopKInWs() — batch zero tileTopK in WS
    ClearTileTopKInWs();

    // 4. Reset state variables
    involvedMask16_ = 0;
    andMask16_ = static_cast<int16_t>(((1 << bitsPerRound_) - 1) << (16 - bitsPerRound_));
    totalDefinitelyInTopK_ = 0;
    boundaryBin = 0;
    boundaryBinPrev = 1;
    globalHistBoundaryNum_ = (tileNum_ > 0) ? (tileNum_ - 1) * tileLen_ + tailTileLen_ : 0;
    remainK_ = kValue_;
    for (uint32_t binMask = 0; binMask < RADIX_NUM_BINS; binMask++) {
        globalHistSum_[binMask] = 0;
    }

    // 4.5. Clear globalHist for all AIVs (including those with tileNum_=0)
    // This ensures ReduceGlobalHist doesn't sum garbage data from idle AIVs
    {
        LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();
        SToVSync();
        Duplicate<int32_t>(globalHist, 0, numValue_);
        VToMTE3Sync();
        DataCopyExtParams params{1, static_cast<uint32_t>(numValue_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(globalHistGm_[GetBlockIdx() * numValue_], globalHist, params);
    }

    // 5. RADIX LOOP
    bool earlyExit = false;
    for (int32_t roundId = round_ - 1; roundId >= 0; roundId--) {
        // a. Prefetch first tile (double buffering)
        if (tileNum_ > 0) {
            uint64_t firstTileLen = tileNum_ > 1 ? tileLen_ : tailTileLen_;
            CopyIn(xBufPing_, firstTileLen, blockOffset_);
        }
        pingPongFlag = true;

        // b. CopyOutHistToWs<true>
        CopyOutHistToWs<true>(numValue_, GetBlockIdx() * numValue_);

        // c. ClearHistInWs — batch init/reset tileHist in WS
        ClearHistInWs(roundId);

        // d. Signal first tile ready
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

        // e. Batched tile loop (each batch MAX_TILE_NUM_IN_UB_BY3 = 2048)
        int32_t repeatTimes = static_cast<int32_t>(tileNumBy3RepeatTimes_ + (tileNumBy3Remain_ > 0 ? 1 : 0));
        for (int32_t repeatId = 0; repeatId < repeatTimes; repeatId++) {
            int32_t startTileId = repeatId * MAX_TILE_NUM_IN_UB_BY3;
            int32_t endTileId = BSAMin(startTileId + MAX_TILE_NUM_IN_UB_BY3, static_cast<int32_t>(tileNum_));

            for (int32_t tileId = startTileId; tileId < endTileId; tileId++) {
                uint64_t curTileLen = (tileId == static_cast<int32_t>(tileNum_) - 1) ? tailTileLen_ : tileLen_;
                TBuf<TPosition::VECIN> &curBuf = pingPongFlag ? xBufPing_ : xBufPong_;
                TBuf<TPosition::VECIN> &nextBuf = pingPongFlag ? xBufPong_ : xBufPing_;

                // Prefetch next tile (double buffering)
                if (tileId < static_cast<int32_t>(tileNum_) - 1) {
                    uint64_t nextOffset = blockOffset_ + (tileId + 1) * tileLen_;
                    uint64_t nextTileLen = (tileId == static_cast<int32_t>(tileNum_) - 2) ? tailTileLen_ : tileLen_;
                    CopyIn(nextBuf, nextTileLen, nextOffset);
                }

                // Wait current tile ready
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

                // Core processing
                NegateDataForLargest(curBuf, curTileLen);
                TwiddleInB16(curBuf, curTileLen, roundId, tileId);
                PipeBarrier<PIPE_V>();
                DoAndMask(curBuf, curTileLen);
                PipeBarrier<PIPE_V>();
                VToSSync();
                CalcCumsumHistogram(curBuf, roundId, tileId - startTileId, curTileLen);

                // Signal next tile ready
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

                pingPongFlag = !pingPongFlag; // Switch double buffering
            }

            // Write this batch's tileHist to WS
            SToMTE3Sync();
            LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
            if (endTileId > startTileId) {
                for (int32_t binMask = numValue_ - 1; binMask > 0; binMask--) {
                    uint32_t tileHistOffset = binMask * tileNum_ + repeatId * MAX_TILE_NUM_IN_UB_BY3;
                    DataCopyPad(tileHistGm_[tileHistOffset], tileHist[(binMask - 1) * MAX_TILE_NUM_IN_UB_BY3],
                                DataCopyParams(1, sizeof(int32_t) * (endTileId - startTileId), 0, 0));
                }
            }
        }

        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

        // f. CopyOutHistToWs<false>
        CopyOutHistToWs<false>(numValue_, GetBlockIdx() * numValue_);

        // g. SyncAll
        PipeBarrier<PIPE_ALL>();
        SyncAll();

        // h. Update
        earlyExit = Update(roundId);
        PipeBarrier<PIPE_ALL>();
        SyncAll();

        // i. Shift mask
        andMask16_ >>= RADIX_BITS_PER_ROUND;

        // j. Early exit
        if (earlyExit)
            break;
    }

    // 6. SyncAll before final selection
    SyncAll();

    // 7. TileTopK (batched processing)
    TileTopK(batchIdx, headIdx);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::NegateDataForLargest(TBuf<TPosition::VECIN> &xBuf,
                                                                       uint64_t curTileLen)
{
    if (constInfo.sparsityMode == BSASparseMode::BottomK) {
        LocalTensor<SCORE_T> xLocal = xBuf.Get<SCORE_T>();
        int32_t ubOffset = (sizeof(SCORE_T) == 2) ? tileLenScoreAlign_ : 0;
        uint64_t alignLen = BSAAlignTo(curTileLen, COMPARES_ALIGN_LEN);
        if constexpr (IsSameType<SCORE_T, bfloat16_t>::value) {
            Muls(xLocal[ubOffset].template ReinterpretCast<half>(), xLocal[ubOffset].template ReinterpretCast<half>(),
                 (half)-1, alignLen);
        } else {
            Muls(xLocal[ubOffset], xLocal[ubOffset], static_cast<SCORE_T>(-1), alignLen);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TwiddleInB16(TBuf<TPosition::VECIN> &xBuf, uint64_t curTileLen,
                                                               int32_t roundId, int32_t tileId)
{
    (void)roundId;
    (void)tileId;
    LocalTensor<SCORE_T> xLocalB16 = xBuf.Get<SCORE_T>();
    LocalTensor<int16_t> xLocalInt16 = xLocalB16[tileLenScoreAlign_].template ReinterpretCast<int16_t>();

    LocalTensor<int16_t> signMaskTensor = outValueBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensor = outIndexBuf_.Get<int16_t>();

    uint64_t curTileLenAlign = BSAAlignTo(curTileLen, COMPARES_ALIGN_LEN);
    uint64_t calcLen = curTileLenAlign * (sizeof(SCORE_T) / sizeof(int16_t));
    LocalTensor<int16_t> temp1 = tempTensor[curTileLenAlign];

    ShiftRight(tempTensor, xLocalInt16, (int16_t)15, curTileLenAlign);
    Duplicate<int16_t>(signMaskTensor, XOR_OP_VALUE_16, curTileLenAlign);
    PipeBarrier<PIPE_V>();
    Or(tempTensor, tempTensor, signMaskTensor, calcLen);
    PipeBarrier<PIPE_V>();

    And(temp1, xLocalInt16, tempTensor, calcLen);
    Or(xLocalInt16, xLocalInt16, tempTensor, calcLen);
    PipeBarrier<PIPE_V>();
    Not(temp1, temp1, calcLen);
    PipeBarrier<PIPE_V>();
    And(xLocalInt16, xLocalInt16, temp1, calcLen);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::DoAndMask(TBuf<TPosition::VECIN> &xBuf, uint64_t curTileLen)
{
    LocalTensor<int16_t> xLocalInt16 = xBuf.Get<int16_t>();
    uint64_t curTileLenAlign = BSAAlignTo(curTileLen, COMPARES_ALIGN_LEN);
    uint64_t calcLen = curTileLenAlign * (sizeof(SCORE_T) / sizeof(int16_t));
    LocalTensor<int32_t> maskTensor = outValueBuf_.Get<int32_t>();

    Duplicate<int16_t>(maskTensor.ReinterpretCast<int16_t>(), andMask16_, curTileLenAlign);
    PipeBarrier<PIPE_V>();
    And(xLocalInt16[tileLenScoreAlign_], xLocalInt16[tileLenScoreAlign_], maskTensor.ReinterpretCast<int16_t>(),
        calcLen);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CalcCumsumHistogram(TBuf<TPosition::VECIN> &xBuf, int32_t roundId,
                                                                      int32_t tileId, uint64_t curTileLen)
{
    LocalTensor<int16_t> xLocalInt16 = xBuf.Get<int16_t>();
    LocalTensor<int32_t> xLocalInt32 = xLocalInt16.ReinterpretCast<int32_t>();
    LocalTensor<float> xLocalFp32 = xLocalInt16.ReinterpretCast<float>();

    Cast(xLocalFp32, xLocalInt16[tileLenScoreAlign_], RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();
    Cast(xLocalInt32, xLocalFp32, RoundMode::CAST_RINT, curTileLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<half> tempHalf = outValueBuf_.Get<half>();
    LocalTensor<uint8_t> cmpMaskTensor = outIndexBuf_.Get<uint8_t>();
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();

    const int32_t tileHistBinStride = MAX_TILE_NUM_IN_UB_BY3;
    const int32_t tileHistBinOffset = MAX_TILE_NUM_IN_UB_BY3;

    int32_t cumSumOne = 0;
    for (int16_t binMask = numValue_ - 1; binMask > 0; binMask--) {
        int32_t involvedMaskTemp =
            static_cast<int32_t>(involvedMask16_ | static_cast<int16_t>(binMask << (roundId * bitsPerRound_)));

        uint64_t curTileLenAlign = BSAAlignTo(curTileLen, static_cast<uint64_t>(REPEAT_SIZE / sizeof(int32_t)));
        Compares<int32_t, uint8_t>(cmpMaskTensor, xLocalInt32, involvedMaskTemp, AscendC::CMPMODE::EQ, curTileLenAlign);
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMask(tempHalf, tempHalf, cmpMaskTensor.ReinterpretCast<uint16_t>(), true, curTileLen, {1, 1, 0, 0},
                   rsvdCnt);
        VToSSync();

        cumSumOne += rsvdCnt;

        tileHist[binMask * tileHistBinStride - tileHistBinOffset].SetValue(tileId, cumSumOne);

        globalHist.SetValue(binMask, globalHist.GetValue(binMask) + cumSumOne);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::VToSSync()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SToVSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE2ToMTE3Sync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventID);
    WaitFlag<HardEvent::MTE2_MTE3>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE3ToMTE2Sync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE3ToVSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventID);
    WaitFlag<HardEvent::MTE3_V>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::VToMTE3Sync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE2ToVSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID);
    WaitFlag<HardEvent::MTE2_V>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE2ToSSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventID);
    WaitFlag<HardEvent::MTE2_S>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE3ToSSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventID);
    WaitFlag<HardEvent::MTE3_S>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SToMTE3Sync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventID);
    WaitFlag<HardEvent::S_MTE3>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyIn(TBuf<TPosition::VECIN> &xBuf, uint64_t dataNum,
                                                         uint64_t xOffset)
{
    LocalTensor<SCORE_T> xLocal = xBuf.Get<SCORE_T>();
    int32_t ubOffset = 0;
    if constexpr (sizeof(SCORE_T) == 2) {
        ubOffset = tileLenScoreAlign_;
    }
    uint64_t xAlign = UB_BLOCK_SIZE / sizeof(SCORE_T);
    uint8_t xPadNum = static_cast<uint8_t>((xAlign - dataNum % xAlign) % xAlign);
    DataCopyExtParams xCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(SCORE_T) * dataNum), 0, 0, 0};
    DataCopyPadExtParams<SCORE_T> xPadParams{true, 0, xPadNum, 0};
    GlobalTensor<SCORE_T> attnScoreTileGm;
    attnScoreTileGm.SetGlobalBuffer((__gm__ SCORE_T *)(attnScoreGmLocal.GetPhyAddr() + headScoreOffset_ + xOffset));
    DataCopyPad(xLocal[ubOffset], attnScoreTileGm[0], xCopyParams, xPadParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyTileHistWs2Ub(LocalTensor<int32_t> &dstTensor, uint32_t srcOffset,
                                                                    uint32_t dataNum)
{
    DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(dstTensor, tileHistGm_[srcOffset], params, padParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyTileTopKWs2Ub(LocalTensor<int32_t> &dstTensor, uint32_t srcOffset,
                                                                    uint32_t dataNum)
{
    DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(dstTensor, tileTopKGm_[srcOffset], params, padParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyTileTopKUb2Ws(LocalTensor<int32_t> &srcTensor, uint32_t dstOffset,
                                                                    uint32_t dataNum)
{
    DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPad(tileTopKGm_[dstOffset], srcTensor, params);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ClearTileTopKInWs()
{
    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    for (int i = 0; i < static_cast<int>(tileNumRepeatTimes_); i++) {
        Duplicate<int32_t>(tileTopK, 0, MAX_TILE_NUM_IN_UB);
        VToMTE3Sync();
        CopyTileTopKUb2Ws(tileTopK, i * MAX_TILE_NUM_IN_UB, MAX_TILE_NUM_IN_UB);
    }
    if (tileNumRemain_ > 0) {
        Duplicate<int32_t>(tileTopK, 0, tileNumRemain_);
        VToMTE3Sync();
        CopyTileTopKUb2Ws(tileTopK, tileNumRepeatTimes_ * MAX_TILE_NUM_IN_UB, tileNumRemain_);
    }
    MTE3ToVSync();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ClearTileHistInWs(uint32_t dstOffset, uint32_t dataNum)
{
    if (dataNum == 0)
        return;
    uint32_t repeatTimes = dataNum / MAX_TILE_NUM_IN_UB;
    uint32_t remain = dataNum % MAX_TILE_NUM_IN_UB;
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    for (uint32_t i = 0; i < repeatTimes; i++) {
        Duplicate<int32_t>(tileHist, 0, MAX_TILE_NUM_IN_UB);
        VToMTE3Sync();
        DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * MAX_TILE_NUM_IN_UB),
                                 0, 0, 0};
        DataCopyPad(tileHistGm_[dstOffset + i * MAX_TILE_NUM_IN_UB], tileHist, params);
    }
    if (remain > 0) {
        Duplicate<int32_t>(tileHist, 0, remain);
        VToMTE3Sync();
        DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * remain), 0, 0, 0};
        DataCopyPad(tileHistGm_[dstOffset + repeatTimes * MAX_TILE_NUM_IN_UB], tileHist, params);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyTileHistInWs(uint32_t dstOffset, uint32_t srcOffset)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    for (uint32_t i = 0; i <= tileNumRepeatTimes_; i++) {
        uint32_t dataNum = (i == tileNumRepeatTimes_) ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0)
            continue;
        CopyTileHistWs2Ub(tileHist, srcOffset + i * MAX_TILE_NUM_IN_UB, dataNum);
        MTE2ToMTE3Sync();
        DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
        DataCopyPad(tileHistGm_[dstOffset + i * MAX_TILE_NUM_IN_UB], tileHist, params);
        MTE3ToMTE2Sync();
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTileHistWs2Ub(LocalTensor<int32_t> &dstTensor,
                                                                   LocalTensor<int32_t> &srcTensor, uint32_t srcOffset0,
                                                                   uint32_t srcOffset1, uint32_t dataNum)
{
    DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(srcTensor, tileHistGm_[srcOffset0], params, padParams);
    DataCopyPad(srcTensor[MAX_TILE_NUM_IN_UB_BY2], tileHistGm_[srcOffset1], params, padParams);
    MTE2ToVSync();
    Sub(dstTensor, srcTensor, srcTensor[MAX_TILE_NUM_IN_UB_BY2], dataNum);
    PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTileHistInWs(uint32_t dstOffset, uint32_t srcOffset0,
                                                                  uint32_t srcOffset1, uint32_t dataNum)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    SubTileHistWs2Ub(tileHist, tileHist, srcOffset0, srcOffset1, dataNum);
    VToMTE3Sync();
    DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPad(tileHistGm_[dstOffset], tileHist, params);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTileHistInWsAll(uint32_t dstOffset, uint32_t srcOffset0,
                                                                     uint32_t srcOffset1)
{
    for (uint32_t i = 0; i <= tileNumBy2RepeatTimes_; i++) {
        uint32_t dataNum = (i == tileNumBy2RepeatTimes_) ? tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0)
            continue;
        SubTileHistInWs(dstOffset + i * MAX_TILE_NUM_IN_UB_BY2, srcOffset0 + i * MAX_TILE_NUM_IN_UB_BY2,
                        srcOffset1 + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        MTE3ToMTE2Sync();
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::AddTileHist2TileTopKInWs(uint64_t tileHistOffset)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    LocalTensor<int32_t> tileTopK = tileHist[MAX_TILE_NUM_IN_UB_BY2];
    for (uint32_t i = 0; i <= tileNumBy2RepeatTimes_; i++) {
        uint32_t dataNum = (i == tileNumBy2RepeatTimes_) ? tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0)
            continue;
        CopyTileHistWs2Ub(tileHist, tileHistOffset + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        CopyTileTopKWs2Ub(tileTopK, i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        MTE2ToVSync();
        Add(tileTopK, tileTopK, tileHist, dataNum);
        VToMTE3Sync();
        CopyTileTopKUb2Ws(tileTopK, i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        MTE3ToMTE2Sync();
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ClearHistInWs(int32_t roundId)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();

    if (roundId == static_cast<int32_t>(round_) - 1) {
        for (int i = 0; i < static_cast<int>(tileNumRepeatTimes_); i++) {
            Duplicate<int32_t>(tileHist, static_cast<int32_t>(tileLen_), MAX_TILE_NUM_IN_UB);
            if (tileNumRemain_ == 0 && tailTileLen_ != tileLen_) {
                VToSSync();
                tileHist.SetValue(MAX_TILE_NUM_IN_UB - 1, static_cast<int32_t>(tailTileLen_));
                SToMTE3Sync();
            } else {
                VToMTE3Sync();
            }
            DataCopyExtParams params{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(sizeof(int32_t) * MAX_TILE_NUM_IN_UB), 0, 0, 0};
            DataCopyPad(tileHistGm_[i * MAX_TILE_NUM_IN_UB], tileHist, params);
        }
        if (tileNumRemain_ > 0) {
            Duplicate<int32_t>(tileHist, static_cast<int32_t>(tileLen_), tileNumRemain_);
            if (tailTileLen_ != tileLen_) {
                VToSSync();
                tileHist.SetValue(tileNumRemain_ - 1, static_cast<int32_t>(tailTileLen_));
                SToMTE3Sync();
            } else {
                VToMTE3Sync();
            }
            DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * tileNumRemain_),
                                     0, 0, 0};
            DataCopyPad(tileHistGm_[tileNumRepeatTimes_ * MAX_TILE_NUM_IN_UB], tileHist, params);
        }
        globalHistBoundaryNum_ = (tileNum_ > 0) ? (tileNum_ - 1) * tileLen_ + tailTileLen_ : 0;
    } else {
        if (boundaryBin < static_cast<int32_t>(numValue_) - 1) {
            SubTileHistInWsAll(0, boundaryBin * static_cast<uint32_t>(tileNum_),
                               boundaryBinPrev * static_cast<uint32_t>(tileNum_));
        } else {
            CopyTileHistInWs(0, boundaryBin * static_cast<uint32_t>(tileNum_));
        }
    }

    SToVSync();
    globalHist.SetValue(0, static_cast<int32_t>(globalHistBoundaryNum_));
    MTE3ToVSync();
    ClearTileHistInWs(static_cast<uint32_t>(tileNum_), (numValue_ - 1) * static_cast<uint32_t>(tileNum_));
    MTE3ToMTE2Sync();
}

template <typename BSAT>
template <bool isInit>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyOutHistToWs(uint64_t maskLen, uint64_t gmOffset)
{
    LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();
    if constexpr (isInit) {
        globalHistBoundaryNum_ = globalHist.GetValue(boundaryBin);
        if (boundaryBin < static_cast<int32_t>(numValue_) - 1) {
            globalHistBoundaryNum_ -= globalHist.GetValue(boundaryBinPrev);
        }
        SToVSync();
        Duplicate<int32_t>(globalHist, 0, maskLen);
        VToMTE3Sync();
    } else {
        SToMTE3Sync();
    }
    DataCopyExtParams globalHistParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * maskLen), 0, 0,
                                       0};
    DataCopyPad(globalHistGm_[gmOffset], globalHist, globalHistParams);
    MTE3ToSSync();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyInGlobalHist(LocalTensor<int32_t> &globalHist)
{
    DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(sizeof(int32_t) * numValue_ * constInfo.aivNum), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(globalHist, globalHistGm_, copyParams, padParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ReduceGlobalHist(LocalTensor<int32_t> &globalHist)
{
    if (sortLen_ <= static_cast<uint64_t>(INT32_MAX)) {
        int32_t coresInBlock = UB_BLOCK_SIZE / sizeof(int32_t) / numValue_;
        int32_t splitPoint = 32;
        while (splitPoint >= static_cast<int32_t>(constInfo.aivNum) && splitPoint > coresInBlock) {
            splitPoint /= 2;
        }
        int32_t remain = static_cast<int32_t>(constInfo.aivNum);
        while (remain > coresInBlock) {
            Add(globalHist, globalHist, globalHist[splitPoint * numValue_], (remain - splitPoint) * numValue_);
            PipeBarrier<PIPE_V>();
            remain = splitPoint;
            splitPoint /= 2;
        }
        VToSSync();
        for (int32_t binMask = 0; binMask < static_cast<int32_t>(numValue_); binMask++) {
            int32_t sum = 0;
            for (int32_t i = 0; i < BSAMin(remain, coresInBlock); i++) {
                sum += globalHist.GetValue(binMask + i * numValue_);
            }
            globalHist.SetValue(binMask, sum);
        }
        return;
    }

    for (int32_t binMask = 0; binMask < static_cast<int32_t>(numValue_); binMask++) {
        uint64_t sum = 0;
        for (int32_t i = 0; i < static_cast<int32_t>(constInfo.aivNum); i++) {
            sum += static_cast<uint32_t>(globalHist.GetValue(binMask + i * numValue_));
        }
        globalHistSum_[binMask] = sum;
        int32_t sumInTensor = (sum > static_cast<uint64_t>(INT32_MAX)) ? INT32_MAX : static_cast<int32_t>(sum);
        globalHist.SetValue(binMask, sumInTensor);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::FindBoundaryBin(const LocalTensor<int32_t> &globalHist,
                                                                  int32_t roundId)
{
    boundaryBin = -1;
    boundaryBinPrev = -1;
    for (int32_t binMask = static_cast<int32_t>(numValue_) - 1; binMask >= 0; binMask--) {
        bool isBoundary = (sortLen_ <= static_cast<uint64_t>(INT32_MAX)) ?
                              (globalHist.GetValue(binMask) >= static_cast<int32_t>(remainK_)) :
                              (globalHistSum_[binMask] >= remainK_);
        if (isBoundary) {
            boundaryBin = binMask;
            boundaryBinPrev = binMask + 1;
            break;
        }
    }
    involvedMask16_ = involvedMask16_ | (static_cast<int16_t>(boundaryBin) << (roundId * bitsPerRound_));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::AddTileHistToTileTopK(LocalTensor<int32_t> &globalHist)
{
    if (sortLen_ <= static_cast<uint64_t>(INT32_MAX)) {
        if (globalHist.GetValue(boundaryBin) == static_cast<int32_t>(remainK_)) {
            AddTileHist2TileTopKInWs(boundaryBin * tileNum_);
            totalDefinitelyInTopK_ += static_cast<uint32_t>(globalHist.GetValue(boundaryBin));
        } else if (boundaryBinPrev < static_cast<int32_t>(numValue_)) {
            AddTileHist2TileTopKInWs(boundaryBinPrev * tileNum_);
            totalDefinitelyInTopK_ += static_cast<uint32_t>(globalHist.GetValue(boundaryBinPrev));
        }
        return;
    }

    if (globalHistSum_[boundaryBin] == remainK_) {
        AddTileHist2TileTopKInWs(boundaryBin * tileNum_);
        totalDefinitelyInTopK_ += globalHistSum_[boundaryBin];
    } else if (boundaryBinPrev < static_cast<int32_t>(numValue_)) {
        AddTileHist2TileTopKInWs(boundaryBinPrev * tileNum_);
        totalDefinitelyInTopK_ += globalHistSum_[boundaryBinPrev];
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyOutBoundaryBinCumSum(LocalTensor<int32_t> &resTensor)
{
    DataCopyExtParams boundaryBinSumCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * 1), 0,
                                               0, 0};
    DataCopyPad(boundaryBinCumSumGm_[GetBlockIdx()], resTensor, boundaryBinSumCopyParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ComputeCumSumPrev(LocalTensor<int32_t> &resTensor,
                                                                    uint64_t &cumSumValuePrev)
{
    DataCopyExtParams copyInParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * constInfo.aivNum),
                                   0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(resTensor, boundaryBinCumSumGm_, copyInParams, padParams);
    MTE2ToSSync();
    cumSumValuePrev = 0;
    for (int32_t coreId = 0; coreId < static_cast<int32_t>(GetBlockIdx()); coreId++) {
        cumSumValuePrev += static_cast<uint32_t>(resTensor.GetValue(coreId));
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyOutCoreTopK(LocalTensor<int32_t> &resTensor,
                                                                  int32_t totalTileTopKInCore)
{
    resTensor.SetValue(0, totalTileTopKInCore);
    SToMTE3Sync();
    DataCopyExtParams coreTopKParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * 1), 0, 0, 0};
    DataCopyPad(coreTopKGm_[GetBlockIdx()], resTensor, coreTopKParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::HandleLastRoundBoundary(LocalTensor<int32_t> &resTensor,
                                                                          LocalTensor<float> &tileHistFp32)
{
    uint32_t reduceSumValue = 0;
    for (int32_t i = 0; i <= static_cast<int32_t>(tileNumBy2RepeatTimes_); i++) {
        uint32_t dataNum =
            (i == static_cast<int32_t>(tileNumBy2RepeatTimes_)) ? tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0)
            continue;

        if (boundaryBinPrev < static_cast<int32_t>(numValue_)) {
            SubTileHistWs2Ub(resTensor, resTensor,
                             boundaryBin * static_cast<uint32_t>(tileNum_) + i * MAX_TILE_NUM_IN_UB_BY2,
                             boundaryBinPrev * static_cast<uint32_t>(tileNum_) + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
            MTE2ToVSync();
            Cast(tileHistFp32, resTensor, RoundMode::CAST_NONE, dataNum);
            PipeBarrier<PIPE_V>();
        } else {
            CopyTileHistWs2Ub(resTensor, boundaryBin * static_cast<uint32_t>(tileNum_) + i * MAX_TILE_NUM_IN_UB_BY2,
                              dataNum);
            MTE2ToVSync();
            Cast(tileHistFp32, resTensor, RoundMode::CAST_NONE, dataNum);
            PipeBarrier<PIPE_V>();
        }

        int32_t int32Align = UB_BLOCK_SIZE / sizeof(int32_t);
        int32_t safeSumTileNum = FLOAT32_SAFE_INT / static_cast<int32_t>(tileLen_);
        safeSumTileNum = (safeSumTileNum / int32Align) * int32Align;
        // Ensure safeSumTileNum is at least int32Align to avoid segSize = 0
        if (safeSumTileNum == 0) {
            safeSumTileNum = int32Align;
        }
        int32_t segSize =
            static_cast<int32_t>(dataNum) < safeSumTileNum ? static_cast<int32_t>(dataNum) : safeSumTileNum;
        for (int32_t segOff = 0; segOff < static_cast<int32_t>(dataNum); segOff += segSize) {
            int32_t curSegLen = BSAMin(segSize, static_cast<int32_t>(dataNum) - segOff);
            ReduceSum(tileHistFp32, tileHistFp32[segOff], tileHistFp32[segOff], curSegLen);
            VToSSync();
            reduceSumValue += static_cast<uint32_t>(static_cast<int32_t>(tileHistFp32.GetValue(0)));
        }
    }

    resTensor.SetValue(0, static_cast<int32_t>(reduceSumValue));
    SToMTE3Sync();
    CopyOutBoundaryBinCumSum(resTensor);
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    uint64_t cumSumValuePrev = 0;
    ComputeCumSumPrev(resTensor, cumSumValuePrev);
    uint64_t remainCoreBoundaryNum = (cumSumValuePrev < remainK_) ? (remainK_ - cumSumValuePrev) : 0;

    if (remainCoreBoundaryNum > 0) {
        LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
        LocalTensor<int32_t> tileHistBoundary = tileTopK[MAX_TILE_NUM_IN_UB_BY3];
        LocalTensor<int32_t> tileHistBoundaryPrev = tileTopK[2 * MAX_TILE_NUM_IN_UB_BY3];
        for (int32_t i = 0; i <= static_cast<int32_t>(tileNumBy3RepeatTimes_); i++) {
            uint32_t dataNum =
                (i == static_cast<int32_t>(tileNumBy3RepeatTimes_)) ? tileNumBy3Remain_ : MAX_TILE_NUM_IN_UB_BY3;
            if (dataNum == 0)
                continue;
            uint32_t startTileId = i * MAX_TILE_NUM_IN_UB_BY3;
            uint32_t endTileId = startTileId + dataNum;

            CopyTileHistWs2Ub(tileHistBoundary, boundaryBin * static_cast<uint32_t>(tileNum_) + startTileId, dataNum);
            CopyTileHistWs2Ub(tileHistBoundaryPrev, boundaryBinPrev * static_cast<uint32_t>(tileNum_) + startTileId,
                              dataNum);
            CopyTileTopKWs2Ub(tileTopK, startTileId, dataNum);
            MTE2ToSSync();

            for (int32_t tileId = static_cast<int32_t>(startTileId); tileId < static_cast<int32_t>(endTileId);
                 tileId++) {
                int32_t curTileBoundaryNum = tileHistBoundary.GetValue(tileId - static_cast<int32_t>(startTileId));
                if (boundaryBinPrev < static_cast<int32_t>(numValue_)) {
                    curTileBoundaryNum -= tileHistBoundaryPrev.GetValue(tileId - static_cast<int32_t>(startTileId));
                }

                if (static_cast<uint64_t>(curTileBoundaryNum) < remainCoreBoundaryNum) {
                    remainCoreBoundaryNum -= static_cast<uint32_t>(curTileBoundaryNum);
                    tileTopK.SetValue(
                        tileId - static_cast<int32_t>(startTileId),
                        tileTopK.GetValue(tileId - static_cast<int32_t>(startTileId)) + curTileBoundaryNum);
                } else {
                    int32_t remainTileBoundaryNum = static_cast<int32_t>(remainCoreBoundaryNum);
                    tileTopK.SetValue(
                        tileId - static_cast<int32_t>(startTileId),
                        tileTopK.GetValue(tileId - static_cast<int32_t>(startTileId)) + remainTileBoundaryNum);
                    remainCoreBoundaryNum = 0;
                    break;
                }
            }
            SToMTE3Sync();
            CopyTileTopKUb2Ws(tileTopK, startTileId, dataNum);
            MTE3ToMTE2Sync();
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::WriteCoreTopKFromWs(LocalTensor<int32_t> &resTensor,
                                                                      LocalTensor<float> &tileHistFp32)
{
    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    int32_t totalTileTopKInCore = 0;
    int32_t int32Align = UB_BLOCK_SIZE / sizeof(int32_t);
    int32_t safeSumTileNum = FLOAT32_SAFE_INT / static_cast<int32_t>(tileLen_);
    safeSumTileNum = (safeSumTileNum / int32Align) * int32Align;
    if (safeSumTileNum == 0) {
        safeSumTileNum = int32Align;
    }
    for (int32_t i = 0; i <= static_cast<int32_t>(tileNumRepeatTimes_); i++) {
        uint32_t dataNum = (i == static_cast<int32_t>(tileNumRepeatTimes_)) ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0)
            continue;
        CopyTileTopKWs2Ub(tileTopK, i * MAX_TILE_NUM_IN_UB, dataNum);
        MTE2ToVSync();
        int32_t segSize =
            static_cast<int32_t>(dataNum) < safeSumTileNum ? static_cast<int32_t>(dataNum) : safeSumTileNum;
        for (int32_t segOff = 0; segOff < static_cast<int32_t>(dataNum); segOff += segSize) {
            int32_t curSegLen = BSAMin(segSize, static_cast<int32_t>(dataNum) - segOff);
            SToVSync();
            Cast(tileHistFp32, tileTopK[segOff], RoundMode::CAST_NONE, curSegLen);
            PipeBarrier<PIPE_V>();
            ReduceSum(tileHistFp32, tileHistFp32, tileHistFp32, curSegLen);
            VToSSync();
            totalTileTopKInCore += static_cast<int32_t>(tileHistFp32.GetValue(0));
        }
    }

    CopyOutCoreTopK(resTensor, totalTileTopKInCore);
}

template <typename BSAT>
__aicore__ inline bool BSARadixTopKService<BSAT>::Update(int32_t roundId)
{
    LocalTensor<int32_t> resTensor = outIndexBuf_.Get<int32_t>();
    LocalTensor<int32_t> globalHist = resTensor;

    CopyInGlobalHist(globalHist);
    if (sortLen_ <= static_cast<uint64_t>(INT32_MAX)) {
        MTE2ToVSync();
    } else {
        MTE2ToSSync();
    }
    ReduceGlobalHist(globalHist);
    FindBoundaryBin(globalHist, roundId);

    AddTileHistToTileTopK(globalHist);

    remainK_ = kValue_ - totalDefinitelyInTopK_;
    LocalTensor<float> tileHistFp32 = resTensor.ReinterpretCast<float>();

    if (roundId == 0 && remainK_ > 0) {
        HandleLastRoundBoundary(resTensor, tileHistFp32);
    }

    bool isLastRound = (remainK_ == 0 || roundId == 0);
    if (isLastRound) {
        WriteCoreTopKFromWs(resTensor, tileHistFp32);
    }

    return isLastRound;
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTopKAndWriteMaskGT(LocalTensor<half> &xLocal, half boundaryHalf,
                                                                        uint64_t curTileLen, uint64_t outputGmOffset,
                                                                        int32_t &curTileK)
{
    uint64_t alignLen = BSAAlignTo(curTileLen, static_cast<uint64_t>(VEC_ALIGN_SIZE));

    // The count overload of Compares only processes complete 256-byte blocks.
    if (sortLen_ <= 64 || alignLen % COMPARES_ALIGN_LEN != 0) {
        LocalTensor<uint8_t> maskLocal = maskLocalBuf_.Get<uint8_t>();
        uint8_t defaultValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 0 : 1;
        uint8_t writeValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 1 : 0;
        uint32_t writeCnt = 0;
        for (uint32_t idx = 0; idx < static_cast<uint32_t>(curTileLen); ++idx) {
            maskLocal.SetValue(idx, defaultValue);
        }
        for (uint32_t idx = 0; idx < static_cast<uint32_t>(curTileLen) && writeCnt < static_cast<uint32_t>(curTileK);
             ++idx) {
            float value = static_cast<float>(xLocal.GetValue(idx));
            float boundary = static_cast<float>(boundaryHalf);
            bool selected =
                (constInfo.sparsityMode == BSASparseMode::BottomK) ? (value < boundary) : (value > boundary);
            if (selected) {
                maskLocal.SetValue(idx, writeValue);
                ++writeCnt;
            }
        }

        GlobalTensor<uint8_t> maskOutTileGm;
        maskOutTileGm.SetGlobalBuffer((__gm__ uint8_t *)(maskOutGmU8Local.GetPhyAddr() + outputGmOffset));

        SToMTE3Sync();
        DataCopyPad(maskOutTileGm[0], maskLocal,
                    DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(uint8_t)), 0, 0));
        MTE3ToMTE2Sync();

        curTileK -= static_cast<int32_t>(writeCnt);
        return;
    }

    LocalTensor<uint8_t> cmpMask = tempBuf_.Get<uint8_t>();
    AscendC::CMPMODE cmpMode =
        (constInfo.sparsityMode == BSASparseMode::BottomK) ? AscendC::CMPMODE::LT : AscendC::CMPMODE::GT;
    Compares<half, uint8_t>(cmpMask, xLocal, boundaryHalf, cmpMode, alignLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<half> tempHalf = outIndexBuf_.Get<half>();
    uint64_t rsvdCnt = 0;
    GatherMask(tempHalf, tempHalf, cmpMask.template ReinterpretCast<uint16_t>(), true, curTileLen, {1, 1, 0, 0},
               rsvdCnt);
    PipeBarrier<PIPE_V>();

    LocalTensor<half> maskHalf = outValueBuf_.Get<half>();
    LocalTensor<half> onesHalf = outIndexBuf_.Get<half>();

    if (constInfo.sparsityMode == BSASparseMode::TopK) {
        Duplicate<half>(maskHalf, (half)0, alignLen);
        Duplicate<half>(onesHalf, (half)1, alignLen);
        PipeBarrier<PIPE_V>();
        Select<half>(maskHalf, cmpMask, onesHalf, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignLen);
    } else {
        Duplicate<half>(maskHalf, (half)1, alignLen);
        Duplicate<half>(onesHalf, (half)0, alignLen);
        PipeBarrier<PIPE_V>();
        Select<half>(maskHalf, cmpMask, onesHalf, (half)1, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignLen);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> maskU8 = outIndexBuf_.Get<uint8_t>();
    Cast<uint8_t, half>(maskU8, maskHalf, RoundMode::CAST_NONE, alignLen);
    PipeBarrier<PIPE_V>();

    GlobalTensor<uint8_t> maskOutTileGm;
    maskOutTileGm.SetGlobalBuffer((__gm__ uint8_t *)(maskOutGmU8Local.GetPhyAddr() + outputGmOffset));

    VToMTE3Sync();
    DataCopyPad(maskOutTileGm[0], maskU8, DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(uint8_t)), 0, 0));
    MTE3ToSSync();
    MTE3ToMTE2Sync();

    VToSSync();

    curTileK -= static_cast<int32_t>(rsvdCnt);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTopKAndWriteMaskEQ(LocalTensor<half> &xLocal, half boundaryHalf,
                                                                        uint64_t curTileLen, uint64_t outputGmOffset,
                                                                        int32_t &curTileK)
{
    uint64_t alignLen = BSAAlignTo(curTileLen, static_cast<uint64_t>(VEC_ALIGN_SIZE));

    // The count overload of Compares only processes complete 256-byte blocks.
    if (sortLen_ <= 64 || alignLen % COMPARES_ALIGN_LEN != 0) {
        LocalTensor<uint8_t> maskLocal = maskLocalBuf_.Get<uint8_t>();
        GlobalTensor<uint8_t> maskOutTileGm;
        maskOutTileGm.SetGlobalBuffer((__gm__ uint8_t *)(maskOutGmU8Local.GetPhyAddr() + outputGmOffset));
        DataCopyPad(maskLocal, maskOutTileGm[0],
                    DataCopyExtParams(1, static_cast<uint32_t>(curTileLen * sizeof(uint8_t)), 0, 0, 0),
                    DataCopyPadExtParams<uint8_t>(false, 0, static_cast<uint8_t>(alignLen - curTileLen), 0));
        MTE2ToSSync();

        uint8_t writeValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 1 : 0;
        uint32_t writeCnt = 0;
        for (uint32_t idx = 0; idx < static_cast<uint32_t>(curTileLen) && writeCnt < static_cast<uint32_t>(curTileK);
             ++idx) {
            if (static_cast<float>(xLocal.GetValue(idx)) == static_cast<float>(boundaryHalf)) {
                maskLocal.SetValue(idx, writeValue);
                ++writeCnt;
            }
        }

        SToMTE3Sync();
        DataCopyPad(maskOutTileGm[0], maskLocal,
                    DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(uint8_t)), 0, 0));
        MTE3ToMTE2Sync();

        curTileK -= static_cast<int32_t>(writeCnt);
        return;
    }

    LocalTensor<uint8_t> cmpMask = tempBuf_.Get<uint8_t>();
    Compares<half, uint8_t>(cmpMask, xLocal, boundaryHalf, AscendC::CMPMODE::EQ, alignLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> vecIndex = outIndexBuf_.Get<int32_t>();
    for (uint64_t i = 0; i < curTileLen; i++) {
        vecIndex.SetValue(i, static_cast<int32_t>(i));
    }
    SToVSync();

    uint64_t eqRsvdCnt = 0;
    GatherMask(vecIndex, vecIndex, cmpMask.template ReinterpretCast<uint32_t>(), true, curTileLen, {1, 1, 0, 0},
               eqRsvdCnt);
    VToSSync();

    uint32_t writeCnt = BSAMin(static_cast<uint32_t>(curTileK), static_cast<uint32_t>(eqRsvdCnt));

    LocalTensor<uint8_t> maskLocal = maskLocalBuf_.Get<uint8_t>();
    GlobalTensor<uint8_t> maskOutTileGm;
    maskOutTileGm.SetGlobalBuffer((__gm__ uint8_t *)(maskOutGmU8Local.GetPhyAddr() + outputGmOffset));
    DataCopyPad(maskLocal, maskOutTileGm[0],
                DataCopyExtParams(1, static_cast<uint32_t>(curTileLen * sizeof(uint8_t)), 0, 0, 0),
                DataCopyPadExtParams<uint8_t>(false, 0, static_cast<uint8_t>(alignLen - curTileLen), 0));
    MTE2ToSSync();

    uint8_t writeValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 1 : 0;
    for (uint32_t i = 0; i < writeCnt; i++) {
        int32_t idx = vecIndex.GetValue(i);
        maskLocal.SetValue(idx, writeValue);
    }

    SToMTE3Sync();
    DataCopyPad(maskOutTileGm[0], maskLocal,
                DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(uint8_t)), 0, 0));
    MTE3ToMTE2Sync();

    curTileK -= static_cast<int32_t>(writeCnt);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTopKAndWriteMaskGTTnd(LocalTensor<half> &xLocal, half boundaryHalf,
                                                                           uint64_t curTileLen,
                                                                           uint64_t compactTileOffset,
                                                                           uint64_t outputHeadOffset, int32_t &curTileK)
{
    LocalTensor<uint8_t> maskLocal = maskLocalBuf_.Get<uint8_t>();
    uint8_t defaultValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 0 : 1;
    uint8_t writeValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 1 : 0;
    float boundary = static_cast<float>(boundaryHalf);
    uint64_t processed = 0;

    while (processed < curTileLen) {
        uint64_t compactIndex = compactTileOffset + processed;
        uint32_t row = static_cast<uint32_t>(compactIndex / validYBlocks_);
        uint32_t col = static_cast<uint32_t>(compactIndex % validYBlocks_);
        uint32_t segmentLen =
            static_cast<uint32_t>(BSAMin(curTileLen - processed, static_cast<uint64_t>(validYBlocks_ - col)));
        uint32_t writeCnt = 0;

        for (uint32_t idx = 0; idx < segmentLen; ++idx) {
            maskLocal.SetValue(idx, defaultValue);
        }
        for (uint32_t idx = 0; idx < segmentLen && writeCnt < static_cast<uint32_t>(curTileK); ++idx) {
            float value = static_cast<float>(xLocal.GetValue(processed + idx));
            bool selected =
                (constInfo.sparsityMode == BSASparseMode::BottomK) ? (value < boundary) : (value > boundary);
            if (selected) {
                maskLocal.SetValue(idx, writeValue);
                ++writeCnt;
            }
        }

        uint64_t outputOffset = outputHeadOffset + static_cast<uint64_t>(row) * maskOutRowStride_ + col;
        GlobalTensor<uint8_t> maskOutSegmentGm;
        maskOutSegmentGm.SetGlobalBuffer((__gm__ uint8_t *)(maskOutGmU8Local.GetPhyAddr() + outputOffset));
        SToMTE3Sync();
        DataCopyPad(maskOutSegmentGm[0], maskLocal, DataCopyExtParams(1, segmentLen * sizeof(uint8_t), 0, 0, 0));
        MTE3ToSSync();
        MTE3ToMTE2Sync();

        curTileK -= static_cast<int32_t>(writeCnt);
        processed += segmentLen;
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTopKAndWriteMaskEQTnd(LocalTensor<half> &xLocal, half boundaryHalf,
                                                                           uint64_t curTileLen,
                                                                           uint64_t compactTileOffset,
                                                                           uint64_t outputHeadOffset, int32_t &curTileK)
{
    LocalTensor<uint8_t> maskLocal = maskLocalBuf_.Get<uint8_t>();
    uint8_t writeValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 1 : 0;
    float boundary = static_cast<float>(boundaryHalf);
    uint64_t processed = 0;

    while (processed < curTileLen && curTileK > 0) {
        uint64_t compactIndex = compactTileOffset + processed;
        uint32_t row = static_cast<uint32_t>(compactIndex / validYBlocks_);
        uint32_t col = static_cast<uint32_t>(compactIndex % validYBlocks_);
        uint32_t segmentLen =
            static_cast<uint32_t>(BSAMin(curTileLen - processed, static_cast<uint64_t>(validYBlocks_ - col)));
        uint64_t outputOffset = outputHeadOffset + static_cast<uint64_t>(row) * maskOutRowStride_ + col;
        GlobalTensor<uint8_t> maskOutSegmentGm;
        maskOutSegmentGm.SetGlobalBuffer((__gm__ uint8_t *)(maskOutGmU8Local.GetPhyAddr() + outputOffset));
        DataCopyPad(maskLocal, maskOutSegmentGm[0], DataCopyExtParams(1, segmentLen * sizeof(uint8_t), 0, 0, 0),
                    DataCopyPadExtParams<uint8_t>(false, 0, 0, 0));
        MTE2ToSSync();

        uint32_t writeCnt = 0;
        for (uint32_t idx = 0; idx < segmentLen && writeCnt < static_cast<uint32_t>(curTileK); ++idx) {
            if (static_cast<float>(xLocal.GetValue(processed + idx)) == boundary) {
                maskLocal.SetValue(idx, writeValue);
                ++writeCnt;
            }
        }

        SToMTE3Sync();
        DataCopyPad(maskOutSegmentGm[0], maskLocal, DataCopyExtParams(1, segmentLen * sizeof(uint8_t), 0, 0, 0));
        MTE3ToMTE2Sync();

        curTileK -= static_cast<int32_t>(writeCnt);
        processed += segmentLen;
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ProcessSingleElementMask(uint32_t batchIdx, uint32_t headIdx)
{
    if (GetBlockIdx() != 0) {
        return;
    }
    uint64_t maxSortLen = constInfo.usePostBlockShape ?
                              static_cast<uint64_t>(constInfo.postXBlocks) * constInfo.postYBlocks :
                              static_cast<uint64_t>(constInfo.xBlocks) * constInfo.yBlocks;
    uint64_t outputOffset = (static_cast<uint64_t>(batchIdx) * constInfo.numHeads + headIdx) * maxSortLen;
    LocalTensor<uint8_t> maskLocal = maskLocalBuf_.Get<uint8_t>();
    maskLocal.SetValue(0, 1);
    GlobalTensor<uint8_t> maskOutSingleGm;
    maskOutSingleGm.SetGlobalBuffer((__gm__ uint8_t *)(maskOutGmU8Local.GetPhyAddr() + outputOffset));
    SToMTE3Sync();
    DataCopyPad(maskOutSingleGm[0], maskLocal, DataCopyExtParams(1, 1, 0, 0, 0));
    MTE3ToSSync();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TileTopK(uint32_t batchIdx, uint32_t headIdx)
{
    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    uint64_t outputStride;
    if (constInfo.usePostBlockShape) {
        outputStride = static_cast<uint64_t>(constInfo.postXBlocks) * constInfo.postYBlocks;
    } else {
        outputStride = static_cast<uint64_t>(constInfo.xBlocks) * constInfo.yBlocks;
    }
    uint64_t outputOffset = (static_cast<uint64_t>(batchIdx) * constInfo.numHeads + headIdx) * outputStride;

    // Restore boundary value (twiddle inverse transform)
    int16_t boundaryInt16 = involvedMask16_ ^ ((~(involvedMask16_ >> 15)) | 0x8000);
    half boundaryHalf = *reinterpret_cast<half *>(&boundaryInt16);
    if (constInfo.sparsityMode == BSASparseMode::BottomK) {
        boundaryHalf = static_cast<half>(-static_cast<float>(boundaryHalf));
    }

    // WS version: batch read tileTopK, use ping-pong double buffering
    for (uint32_t repeatId = 0; repeatId <= tileNumRepeatTimes_; repeatId++) {
        uint32_t startTileId = repeatId * MAX_TILE_NUM_IN_UB;
        uint32_t dataNum = (repeatId == tileNumRepeatTimes_) ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0)
            continue;

        CopyTileTopKWs2Ub(tileTopK, startTileId, dataNum);
        MTE2ToSSync();

        // Prefetch first tile
        uint32_t firstTileId = startTileId;
        uint64_t firstTileLen = (firstTileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;
        uint64_t firstOffset = blockOffset_ + firstTileId * tileLen_;
        CopyIn(xBufPing_, firstTileLen, firstOffset);
        pingPongFlag = true;
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

        for (uint32_t i = 0; i < dataNum; i++) {
            uint64_t tileId = startTileId + i;
            int32_t curTileK = tileTopK.GetValue(i);
            // v2.3: no longer skip curTileK==0, GT phase naturally handles all tiles

            uint64_t curTileLen = (tileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;
            uint64_t tileGmOffset = blockOffset_ + tileId * tileLen_;
            uint64_t outputGmOffset = outputOffset + tileGmOffset;

            TBuf<TPosition::VECIN> &curBuf = pingPongFlag ? xBufPing_ : xBufPong_;
            TBuf<TPosition::VECIN> &nextBuf = pingPongFlag ? xBufPong_ : xBufPing_;

            // Prefetch next tile
            if (i < dataNum - 1) {
                uint64_t nextTileId = tileId + 1;
                uint64_t nextTileLen = (nextTileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;
                uint64_t nextOffset = blockOffset_ + nextTileId * tileLen_;
                CopyIn(nextBuf, nextTileLen, nextOffset);
            }

            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);
            MTE2ToSSync();

            LocalTensor<half> xLocal = curBuf.template Get<half>()[tileLenScoreAlign_].template ReinterpretCast<half>();

            if (useCompactTopKMapping_) {
                SubTopKAndWriteMaskGTTnd(xLocal, boundaryHalf, curTileLen, tileGmOffset, outputOffset, curTileK);
            } else {
                SubTopKAndWriteMaskGT(xLocal, boundaryHalf, curTileLen, outputGmOffset, curTileK);
            }

            if (curTileK > 0) {
                if (useCompactTopKMapping_) {
                    SubTopKAndWriteMaskEQTnd(xLocal, boundaryHalf, curTileLen, tileGmOffset, outputOffset, curTileK);
                } else {
                    SubTopKAndWriteMaskEQ(xLocal, boundaryHalf, curTileLen, outputGmOffset, curTileK);
                }
            }

            if (i < dataNum - 1) {
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);
            }
            pingPongFlag = !pingPongFlag;
        }
    }
}

#endif // BSA_RADIX_TOPK_SERVICE_ARCH35_H
