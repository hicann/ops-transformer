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
 * \file compressor_block_vec_full_load.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_FULL_LOAD_H
#define COMPRESSOR_BLOCK_VEC_FULL_LOAD_H

#include "compressor_block_vec.h"
#include "compressor_tools.h"
#include "vf/vf_softmax.h"
#include "vf/vf_add.h"
#include "vf/vf_mul.h"
#include "limits"

using namespace AscendC;

namespace Compressor {

template <typename COMP>
class CompressorBlockVectorFullLoad : public CompressorBlockVector<COMP> {
public:
    using Base = CompressorBlockVector<COMP>;
    using T = typename Base::T;
    using X_T = typename Base::X_T;

    __aicore__ inline CompressorBlockVectorFullLoad(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void UpdateMRange(uint32_t mStart, uint32_t mEnd);
    __aicore__ inline void ComputeVec1(uint32_t c1v1DbIdx);

private:
    __aicore__ inline void CopyInApe(uint32_t dStartIdx, uint32_t dDealSize);
    template <bool IS_FULLLOAD>
    __aicore__ inline void AddApe(const LocalTensor<T> &scoreLocal, uint32_t dealRowCount, uint32_t dealColCount,
                                  uint32_t scoreSingleRowCount, uint32_t apeSingleRowCount, uint64_t scoreOffset,
                                  uint64_t apeOffset);
    __aicore__ inline void AddApeToScore(const LocalTensor<T> &scoreLocal, const Vec1SliceInfo &sliceInfo,
                                         uint32_t dDealSize, uint32_t dBaseSize, uint32_t dStartIdx,
                                         bool isApeFullLoad);
    template <bool IS_SCORE>
    __aicore__ inline void OverLap(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                   const GlobalTensor<T> &srcGm, const GlobalTensor<T> &stateGm,
                                   const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
                                   const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
                                   uint32_t dBaseOffset, uint32_t globalSeqIdx, uint32_t dDealSize, uint32_t dBaseSize);
    __aicore__ inline void OverLapScoreKv(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal,
                                          const LoopInfo &loopInfo, const StatisticInfo &statisticInfo,
                                          const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx,
                                          uint32_t dBaseOffset, uint32_t dDealSize, uint32_t dBaseSize,
                                          uint32_t dealSeqStartIdx, uint32_t needDealTcSize);
    __aicore__ inline void DealVec1BaseBlock(CompressorVec1SliceIterator<COMP> &sliceIterator, const LoopInfo &loopInfo,
                                             uint32_t dStartIdx, uint32_t dBaseOffset, uint32_t dDealSize,
                                             uint32_t dBaseSize, uint32_t dealSeqStartIdx);
    __aicore__ inline void CalcGroupInfo(Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(Vec1SplitInfo &splitInfo);
    __aicore__ inline Vec1SplitInfo SplitCoreV1();

    uint32_t c1v1DbIdx_ = 0;
    uint64_t mm1ResMOffset_ = 0;
    uint32_t totalCompressedCnt_ = 0;
    uint32_t kvStateIdx_ = 0;
    uint32_t scoreStateIdx_ = 1;
    LocalTensor<T> scoreUb;
    LocalTensor<T> kvUb;
    TQue<QuePosition::VECIN, 1> inputQueApe;
};

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(this->inputQue1, 1, BUFFER_SIZE_BYTE_64K);
    pipe->InitBuffer(this->inputQue2, 1, BUFFER_SIZE_BYTE_64K);
    pipe->InitBuffer(this->outputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(inputQueApe, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->outputQue2, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->tmpBuff2, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->apeBuf, BUFFER_SIZE_BYTE_32K);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::UpdateMRange(uint32_t mStart, uint32_t mEnd)
{
    this->constInfo_.mStart = mStart;
    this->constInfo_.mEnd = mEnd;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::CopyInApe(uint32_t dStartIdx, uint32_t dDealSize)
{
    this->apeUb = this->apeBuf.template Get<T>();

    uint32_t copyRowCount = this->coff_ * this->cmpRatio_;
    uint32_t copyColCount = dDealSize;
    uint32_t dstSingleRowCount = dDealSize;
    uint32_t srcSingleRowCount = this->constInfo_.headDim;

    uint64_t gmOffset = dStartIdx;

    this->DataCopyWithInputQue(this->apeUb, this->apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount,
                               dstSingleRowCount);
    PipeBarrier<PIPE_V>();
}
template <typename COMP>
template <bool IS_FULLLOAD>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::AddApe(const LocalTensor<T> &scoreLocal,
                                                                   uint32_t dealRowCount, uint32_t dealColCount,
                                                                   uint32_t scoreSingleRowCount,
                                                                   uint32_t apeSingleRowCount, uint64_t scoreOffset,
                                                                   uint64_t apeOffset)
{
    if constexpr (IS_FULLLOAD) {
        AddVF(scoreLocal[scoreOffset], this->apeUb[apeOffset], this->coff_ * dealRowCount, dealColCount,
              scoreSingleRowCount, apeSingleRowCount);
    } else {
        this->apeUb = inputQueApe.AllocTensor<T>();
        this->DataCopyAlignGmToUb(this->apeUb, this->apeGm_[apeOffset], this->coff_ * dealRowCount, dealColCount,
                                  this->constInfo_.headDim, apeSingleRowCount);
        inputQueApe.EnQue(this->apeUb);
        inputQueApe.DeQue<T>();
        AddVF(scoreLocal[scoreOffset], this->apeUb, this->coff_ * dealRowCount, dealColCount, scoreSingleRowCount,
              apeSingleRowCount);
        inputQueApe.FreeTensor(this->apeUb);
    }
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::AddApeToScore(const LocalTensor<T> &scoreLocal,
                                                                          const Vec1SliceInfo &sliceInfo,
                                                                          uint32_t dDealSize, uint32_t dBaseSize,
                                                                          uint32_t dStartIdx, bool isApeFullLoad)
{
    uint32_t singleUbRowElemNum = dBaseSize * this->coff_;
    uint32_t singleApeRowElemNum = isApeFullLoad ? singleUbRowElemNum : this->constInfo_.headDim * this->coff_;
    uint64_t scoreOffset = sliceInfo.dealedSeqCnt * singleUbRowElemNum;

    uint32_t tcDealSize = sliceInfo.dealTcSize;
    if (sliceInfo.headHolderSeqCnt > 0) {
        uint32_t row = tcDealSize == 1 ? sliceInfo.validSeqCnt : (this->cmpRatio_ - sliceInfo.headHolderSeqCnt);

        if (isApeFullLoad) {
            uint64_t apeOffset = sliceInfo.headHolderSeqCnt * singleApeRowElemNum;
            AddApe<true>(scoreLocal, row, dDealSize, dBaseSize, dBaseSize, scoreOffset, apeOffset);

        } else {
            uint64_t apeOffset = sliceInfo.headHolderSeqCnt * singleApeRowElemNum + dStartIdx;
            AddApe<false>(scoreLocal, row, dDealSize, dBaseSize, dDealSize, scoreOffset, apeOffset);
        }
        scoreOffset += row * singleUbRowElemNum;
        tcDealSize -= 1;
    }
    if (tcDealSize == 0) {
        return;
    }
    if (sliceInfo.tailHolderSeqCnt > 0) {
        tcDealSize -= 1;
        uint32_t row = this->cmpRatio_ - sliceInfo.tailHolderSeqCnt;
        uint32_t tailScoreOffset = scoreOffset + tcDealSize * this->cmpRatio_ * singleUbRowElemNum;
        if (isApeFullLoad) {
            uint64_t apeOffset = 0;
            AddApe<true>(scoreLocal, row, dDealSize, dBaseSize, dBaseSize, tailScoreOffset, apeOffset);

        } else {
            uint64_t apeOffset = dStartIdx;
            AddApe<false>(scoreLocal, row, dDealSize, dBaseSize, dDealSize, tailScoreOffset, apeOffset);
        }
    }
    if (tcDealSize == 0) {
        return;
    }

    if (isApeFullLoad) {
        uint32_t row = this->cmpRatio_;
        for (uint32_t r = 0; r < tcDealSize; r++) {
            uint64_t curScoreOffset = scoreOffset + r * row * singleUbRowElemNum;
            AddApe<true>(scoreLocal, row, dDealSize, dBaseSize, dDealSize, curScoreOffset, 0U);
        }
    }
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::OverLap(
    const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const GlobalTensor<T> &srcGm,
    const GlobalTensor<T> &stateGm, const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
    const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx, uint32_t dBaseOffset,
    uint32_t globalSeqIdx, uint32_t dDealSize, uint32_t dBaseSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    this->template ReadState<IS_SCORE>(dstLocal, stateGm, blockTableGm, sliceInfo, dStartIdx + dBaseOffset, dDealSize,
                                       static_cast<uint32_t>(IS_SCORE));

    if (sliceInfo.compressTcSize > 0) {
        this->PadAlign(dstLocal, srcLocal, sliceInfo, dBaseOffset, dDealSize, dBaseSize);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            GlobalTensor<T> curCacheTcGm = cacheTcGm;
            this->LoadFromWorkSpace(dstLocal, curCacheTcGm, srcGm, srcLocal, sliceInfo, loopInfo, dStartIdx,
                                    globalSeqIdx, dDealSize);
        }
    }
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const LoopInfo &loopInfo,
    const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dBaseOffset,
    uint32_t dDealSize, uint32_t dBaseSize, uint32_t dealSeqStartIdx, uint32_t needDealTcSize)
{
    CompressorVec1SliceIterator overLapSliceIterator(this->tools_);
    overLapSliceIterator.SetMaxBatchSize(this->constInfo_.batchSize);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();

    GlobalTensor<T> scoreDBMm1ResGm = this->scoreMm1ResGm_[c1v1DbIdx_ * this->constInfo_.dbSize + mm1ResMOffset_];
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, originSliceInfo.dealedSeqCnt, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);

    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<true>(scoreLocal, scoreUb, scoreDBMm1ResGm, this->stateCacheGm_, this->stateBlockTableGm_,
                      this->scoreCacheTcGm_, overLapSliceInfo, loopInfo, dStartIdx, dBaseOffset,
                      originSliceInfo.dealedSeqCnt + dealSeqStartIdx, dDealSize, dBaseSize);
        overLapSliceIterator.IteratorSlice();
    }
    if constexpr (COMP::coff == COFF::OVERLAP) {
        if (originSliceInfo.sIdx != 0 && originSliceInfo.compressTcSize > 0 &&
            (!loopInfo.isCoreRowFirst || !loopInfo.isCoreLoopFirst)) {
            PipeBarrier<PIPE_V>();
            uint32_t singleRowElemNum = dDealSize * this->coff_;
            uint32_t dealRowCount = min(originSliceInfo.sIdx, this->cmpRatio_);
            uint64_t scoreOffset = (this->cmpRatio_ - dealRowCount) * singleRowElemNum +
                                   originSliceInfo.compressoredScCnt * this->cmpRatio_ * singleRowElemNum;
            uint64_t apeOffset = (this->cmpRatio_ - dealRowCount) * dDealSize;
            AddVF(scoreLocal[scoreOffset], this->apeUb[apeOffset], dealRowCount, dDealSize, singleRowElemNum,
                  dDealSize);
        }
    }
    GlobalTensor<T> kvDBMm1ResGm = this->kvMm1ResGm_[c1v1DbIdx_ * this->constInfo_.dbSize + mm1ResMOffset_];
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, originSliceInfo.dealedSeqCnt, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);

    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<false>(kvLocal, kvUb, kvDBMm1ResGm, this->stateCacheGm_, this->stateBlockTableGm_, this->kvCacheTcGm_,
                       overLapSliceInfo, loopInfo, dStartIdx, dBaseOffset,
                       originSliceInfo.dealedSeqCnt + dealSeqStartIdx, dDealSize, dBaseSize);
        overLapSliceIterator.IteratorSlice();
    }
    PipeBarrier<PIPE_V>();
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::DealVec1BaseBlock(
    CompressorVec1SliceIterator<COMP> &sliceIterator, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t dBaseOffset, uint32_t dDealSize, uint32_t dBaseSize, uint32_t dealSeqStartIdx)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }
    LocalTensor<T> scoreLocal = this->tmpBuff1.template Get<T>();
    LocalTensor<T> kvLocal = this->tmpBuff2.template Get<T>();
    OverLapScoreKv(scoreLocal, kvLocal, loopInfo, statisticInfo, originSliceInfo, dStartIdx, dBaseOffset, dDealSize,
                   dBaseSize, dealSeqStartIdx, needDealTcSize);
    if (statisticInfo.compressorScCnt > 0) {
        this->SoftmaxDN(scoreLocal, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        if constexpr (COMP::gradEnabled == GRAD_ENABLED::ENABLE) {
            this->CopyOutMidResToOutput(kvLocal, scoreLocal, originSliceInfo, statisticInfo.compressorScCnt,
                                        dStartIdx + dBaseOffset, dDealSize, this->totalCompressedCnt_);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<T> comperssoredUb = this->outputQue2.template AllocTensor<T>();
        PipeBarrier<PIPE_V>();
        this->KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, statisticInfo.compressorScCnt, dDealSize);
        this->outputQue2.EnQue(comperssoredUb);
        this->outputQue2.template DeQue<T>();
        this->CopyOutVec1ResToOutput(comperssoredUb, originSliceInfo, statisticInfo.compressorScCnt,
                                     dStartIdx + dBaseOffset, dDealSize);
        this->outputQue2.FreeTensor(comperssoredUb);
    }
    this->compressedCnt_ += statisticInfo.compressorScCnt;
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::CalcGroupInfo(Vec1SplitInfo &splitInfo)
{
    uint32_t aiCoreNum = this->constInfo_.usedCoreNum * this->cvRatio_;
    uint32_t mStartBatch = this->constInfo_.mStart / this->constInfo_.sSize;
    uint32_t mEndBatch = (this->constInfo_.mEnd + this->constInfo_.sSize - 1) / this->constInfo_.sSize;
    uint32_t effectiveBatchSize =
        min(mEndBatch, this->constInfo_.batchSize) - min(mStartBatch, this->constInfo_.batchSize);
    effectiveBatchSize = max(effectiveBatchSize, 1U);
    uint32_t vecCoresPerMBlock = this->constInfo_.kBaseNum * this->constInfo_.dBasicBlockNum;
    splitInfo.dBaseSize = this->constInfo_.headDim /
                          min(FloorPow2(vecCoresPerMBlock), CeilPow2(CeilDivT(vecCoresPerMBlock, effectiveBatchSize)));
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_64K / (this->cmpRatio_ * this->coff_ * sizeof(T));
    splitInfo.dBaseSize = min(splitInfo.dBaseSize, FloorPow2(Trunc(maxDealColNum, BlockElementNum<T>())));
    if (this->constInfo_.kBaseNum > 1) {
        splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_REPEAT_ELEMENT_NUM);
    }
    splitInfo.dBaseSize = max(splitInfo.dBaseSize, BlockElementNum<X_T>());
    splitInfo.vec1GroupSize = this->constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum =
        min(static_cast<uint32_t>(vecCoresPerMBlock / splitInfo.vec1GroupSize), effectiveBatchSize);
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::CalcTaskDistribution(Vec1SplitInfo &splitInfo)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t groupSize = splitInfo.vec1GroupSize;
    uint32_t groupNum = splitInfo.vec1GroupNum;
    uint32_t mStartBatch = this->constInfo_.mStart / this->constInfo_.sSize;
    uint32_t mEndBatch = (this->constInfo_.mEnd + this->constInfo_.sSize - 1) / this->constInfo_.sSize;
    uint32_t totalDealBatchNum =
        min(mEndBatch, this->constInfo_.batchSize) - min(mStartBatch, this->constInfo_.batchSize);

    uint32_t effectiveMIdx = (this->constInfo_.mLoopNum == 1 && this->constInfo_.mBaseSize != 0) ?
                                 this->constInfo_.mStart / this->constInfo_.mBaseSize :
                                 0;
    uint32_t localBlockIdx = blockIdx - effectiveMIdx * this->constInfo_.kBaseNum * this->constInfo_.dBasicBlockNum;

    if (localBlockIdx < groupSize * (totalDealBatchNum % groupNum)) {
        splitInfo.dealBatchNum = totalDealBatchNum / groupNum + 1;
        splitInfo.preDealBatchNum = splitInfo.dealBatchNum * (localBlockIdx / groupSize);
    } else if (localBlockIdx < groupSize * groupNum) {
        splitInfo.dealBatchNum = totalDealBatchNum / groupNum;
        splitInfo.preDealBatchNum = splitInfo.dealBatchNum * (localBlockIdx / groupSize) + totalDealBatchNum % groupNum;
    } else {
        splitInfo.dealBatchNum = 0;
        splitInfo.preDealBatchNum = totalDealBatchNum;
    }
    splitInfo.preDealBatchNum += min(mStartBatch, this->constInfo_.batchSize);
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::UpdateIteratorState(Vec1SplitInfo &splitInfo)
{
    splitInfo.preCompressedCnt = 0;
    uint32_t mStartBatch = this->constInfo_.mStart / this->constInfo_.sSize;
    splitInfo.dealSeqStartIdx = (splitInfo.preDealBatchNum - mStartBatch) * this->constInfo_.sSize;
    splitInfo.curBStart = splitInfo.preDealBatchNum;
    splitInfo.dealSeqCnt = splitInfo.dealBatchNum * this->constInfo_.sSize;
    splitInfo.curSStart = 0;
    totalCompressedCnt_ = 0;
    for (uint32_t curB = splitInfo.curBStart; curB < splitInfo.curBStart + splitInfo.dealBatchNum; curB++) {
        uint32_t startPos = this->GetStartPos(curB);
        uint32_t seqLength = this->GetSeqLength(curB);
        totalCompressedCnt_ += (startPos + seqLength) / this->cmpRatio_ - startPos / this->cmpRatio_;
    }
    totalCompressedCnt_ += splitInfo.preCompressedCnt;
}
template <typename COMP>
__aicore__ inline Vec1SplitInfo CompressorBlockVectorFullLoad<COMP>::SplitCoreV1()
{
    Vec1SplitInfo splitInfo;

    // 1. 计算基础分组和分片大小
    CalcGroupInfo(splitInfo);

    // 2. 根据当前的 BlockIdx 计算任务分配（负载均衡）
    CalcTaskDistribution(splitInfo);

    // 3. 刷新迭代器并获取当前核的起始位置状态
    UpdateIteratorState(splitInfo);

    if (splitInfo.dealBatchNum == 0) {
        return splitInfo;
    }

    // 4. 计算具体在内存中的切块（Tiling）逻辑
    this->CalcTilingStrategy(splitInfo);

    return splitInfo;
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorFullLoad<COMP>::ComputeVec1(uint32_t c1v1DbIdx)
{
    c1v1DbIdx_ = c1v1DbIdx;
    mm1ResMOffset_ = (this->constInfo_.mLoopNum == 1 && this->constInfo_.mBaseSize != 0) ?
                         (uint64_t)(this->constInfo_.mStart / this->constInfo_.mBaseSize) * this->constInfo_.kBaseNum *
                             this->constInfo_.mm1KvResSize :
                         0;
    GlobalTensor<T> scoreDbGm = this->scoreMm1ResGm_[(uint64_t)c1v1DbIdx * this->constInfo_.dbSize + mm1ResMOffset_];
    GlobalTensor<T> kvDbGm = this->kvMm1ResGm_[(uint64_t)c1v1DbIdx * this->constInfo_.dbSize + mm1ResMOffset_];
    Vec1SplitInfo splitInfo = SplitCoreV1();
    // 计算当前VecCore的任务量
    if (splitInfo.dealBatchNum == 0) {
        return;
    }

    LoopInfo loopInfo;
    loopInfo.groupSize = splitInfo.vec1GroupSize;
    loopInfo.groupNum = splitInfo.vec1GroupNum;
    uint32_t effectiveMIdxL = (this->constInfo_.mLoopNum == 1 && this->constInfo_.mBaseSize != 0) ?
                                  this->constInfo_.mStart / this->constInfo_.mBaseSize :
                                  0;
    uint32_t localBlockIdxL =
        GetBlockIdx() - effectiveMIdxL * this->constInfo_.kBaseNum * this->constInfo_.dBasicBlockNum;
    loopInfo.coreRowIdx = localBlockIdxL / splitInfo.vec1GroupSize;
    loopInfo.coreColIdx = localBlockIdxL % splitInfo.vec1GroupSize;
    loopInfo.isCoreRowLast = loopInfo.coreRowIdx == splitInfo.vec1GroupNum - 1;
    loopInfo.isCoreRowFirst = loopInfo.coreRowIdx == 0;

    CompressorVec1SliceIterator sliceIterator(this->tools_);
    sliceIterator.SetMaxBatchSize(this->constInfo_.batchSize);
    // 切块循环
    uint64_t baseOffset = loopInfo.coreColIdx * splitInfo.dBaseSize;

    uint32_t cnt = this->constInfo_.sSize * splitInfo.dBaseSize * this->coff_;
    uint32_t singleLoopBatchNum = BUFFER_SIZE_BYTE_32K / (cnt * sizeof(T));
    uint32_t loopTimes = CeilDivT(splitInfo.dealBatchNum, singleLoopBatchNum);
    bool isApeFullLoad = this->coff_ * this->cmpRatio_ * splitInfo.dBaseSize * sizeof(T) <= BUFFER_SIZE_BYTE_32K;
    if (isApeFullLoad) {
        CopyInApe(baseOffset, splitInfo.dBaseSize);
    }
    for (uint32_t idx = 0; idx < loopTimes; idx++) {
        uint32_t curLoopBatchNum = min(singleLoopBatchNum, splitInfo.dealBatchNum - singleLoopBatchNum * idx);
        scoreUb = this->inputQue1.template AllocTensor<T>();
        kvUb = scoreUb[BUFFER_SIZE_BYTE_32K / sizeof(T)];
        this->FromWokrSpaceToUb(scoreUb, scoreDbGm, splitInfo.dealSeqStartIdx, curLoopBatchNum * this->constInfo_.sSize,
                                baseOffset, splitInfo.dBaseSize);
        this->FromWokrSpaceToUb(kvUb, kvDbGm, splitInfo.dealSeqStartIdx, curLoopBatchNum * this->constInfo_.sSize,
                                baseOffset, splitInfo.dBaseSize);
        this->inputQue1.template EnQue(scoreUb);
        this->inputQue1.template DeQue<T>();
        splitInfo.dealTcNum = 0;
        uint32_t curLoopCompressedCnt = 0;
        for (uint32_t curB = splitInfo.curBStart; curB < splitInfo.curBStart + curLoopBatchNum; curB++) {
            uint32_t startPos = this->GetStartPos(curB);
            uint32_t seqLength = this->GetSeqLength(curB);
            uint32_t seqUsed = this->GetSeqUsed(curB);
            splitInfo.dealTcNum += CeilDivT(startPos + seqLength, this->cmpRatio_) - (startPos / this->cmpRatio_);
            curLoopCompressedCnt += (startPos + seqUsed) / this->cmpRatio_ - startPos / this->cmpRatio_;
        }
        sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, 0U, 0U);
        sliceIterator.SetNeedDealTcSize(splitInfo.dealTcNum);
        sliceIterator.SetDealedTcCnt(0U);
        Vec1SliceInfo &sliceInfo = sliceIterator.GetSlice();
        while (!sliceIterator.IsEnd()) {
            sliceIterator.GetSlice();
            this->SaveState(kvUb, this->stateCacheGm_, this->stateBlockTableGm_, sliceInfo, baseOffset,
                            splitInfo.dBaseSize, splitInfo.dBaseSize, kvStateIdx_);

            AddApeToScore(scoreUb, sliceInfo, splitInfo.dBaseSize, splitInfo.dBaseSize, baseOffset, isApeFullLoad);
            this->SaveState(scoreUb, this->stateCacheGm_, this->stateBlockTableGm_, sliceInfo, baseOffset,
                            splitInfo.dBaseSize, splitInfo.dBaseSize, scoreStateIdx_);
            sliceIterator.IteratorSlice();
        }

        if (curLoopCompressedCnt == 0) {
            this->inputQue1.template FreeTensor(scoreUb);
            continue;
        }
        for (uint32_t dLoopIdx = 0; dLoopIdx < splitInfo.dLoopCount; dLoopIdx++) {
            uint64_t dBaseOffset = baseOffset + dLoopIdx * splitInfo.dSplitSize;
            loopInfo.dLoopIdx = dLoopIdx;

            sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, 0U, 0U);
            this->compressedCnt_ = splitInfo.preCompressedCnt;
            for (uint32_t tcIdx = 0; tcIdx < splitInfo.dealTcNum; tcIdx += splitInfo.tcSplitSize) {
                uint32_t actDealTcSize = min(splitInfo.tcSplitSize, splitInfo.dealTcNum - tcIdx);

                loopInfo.isCoreLoopFirst = tcIdx == 0;
                loopInfo.isCoreLoopLast = tcIdx + splitInfo.tcSplitSize >= splitInfo.dealTcNum;
                // 处理单个切块
                sliceIterator.SetNeedDealTcSize(actDealTcSize);
                sliceIterator.SetDealedTcCnt(0U);
                DealVec1BaseBlock(sliceIterator, loopInfo, baseOffset, dLoopIdx * splitInfo.dSplitSize,
                                  splitInfo.dSplitSize, splitInfo.dBaseSize, splitInfo.dealSeqStartIdx);
            }
        }
        this->inputQue1.template FreeTensor(scoreUb);
        splitInfo.curBStart += curLoopBatchNum;
        splitInfo.dealSeqStartIdx += curLoopBatchNum * this->constInfo_.sSize;
        splitInfo.preCompressedCnt += curLoopCompressedCnt;
    }
}
} // namespace Compressor

#endif // COMPRESSOR_BLOCK_VEC_FULL_LOAD_H
