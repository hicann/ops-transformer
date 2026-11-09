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
 * \file compressor_block_vec_normal.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_NORMAL_H
#define COMPRESSOR_BLOCK_VEC_NORMAL_H

#include "compressor_block_vec_arch92.h"
#include "compressor_tiling_data_arch92.h"
#include "compressor_tools_arch92.h"
#include "vf/vf_softmax_arch92.h"
#include "vf/vf_add_arch92.h"
#include "vf/vf_mul_arch92.h"
#include "limits"

using namespace AscendC;

namespace Compressor {

template <typename COMP>
class CompressorBlockVectorNormal : public CompressorBlockVector<COMP> {
public:
    using Base = CompressorBlockVector<COMP>;
    using T = typename Base::T;
    using X_T = typename Base::X_T;

    __aicore__ inline CompressorBlockVectorNormal(){};
    __aicore__ inline void ComputeVec1(const Vec1RunInfo& info);

private:
    __aicore__ inline void CopyInApe(const LocalTensor<T>& apeUb, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void AddApeToScore(const LocalTensor<T>& scoreLocal, const LocalTensor<T>& apeUb,
                                         const Vec1SliceInfo& sliceInfo, uint32_t dDealSize);
    __aicore__ inline void AddApe(const LocalTensor<T>& scoreLocal, const LocalTensor<T>& apeUb,
                                  const Vec1SliceInfo& sliceInfo, uint32_t dDealSize);
    template <bool IS_SCORE>
    __aicore__ inline void OverLap(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                   const GlobalTensor<T>& srcGm, const GlobalTensor<T>& stateGm,
                                   const GlobalTensor<int32_t>& blockTableGm, const GlobalTensor<T>& cacheTcGm,
                                   const Vec1RunInfo& info, const Vec1SliceInfo& sliceInfo, const LoopInfo& loopInfo,
                                   uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize);
    __aicore__ inline void OverLapScoreKv(const LocalTensor<T>& scoreLocal, const LocalTensor<T>& kvLocal,
                                          const Vec1RunInfo& info, const LoopInfo& loopInfo,
                                          const StatisticInfo& statisticInfo, const Vec1SliceInfo& originSliceInfo,
                                          uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize,
                                          uint32_t needDealTcSize);
    __aicore__ inline void CalcGroupInfo(const Vec1RunInfo& info, Vec1SplitInfo& splitInfo);
    __aicore__ inline void CalcTaskDistribution(const Vec1RunInfo& info, Vec1SplitInfo& splitInfo);
    __aicore__ inline void UpdateIteratorState(const Vec1RunInfo& info, Vec1SplitInfo& splitInfo);
    __aicore__ inline void DealVec1BaseBlock(const Vec1RunInfo& info, CompressorVec1SliceIterator<COMP>& sliceIterator,
                                             const LoopInfo& loopInfo, uint32_t dStartIdx, uint32_t dDealSize,
                                             uint32_t dBaseSize);
    __aicore__ inline Vec1SplitInfo SplitCoreV1(const Vec1RunInfo& info);
    __aicore__ inline void ComputeBatchTiles(const Vec1RunInfo& info, const Vec1SplitInfo& splitInfo,
                                             LoopInfo& loopInfo, uint32_t dStartIdx, uint32_t dDealSize);
    // s<=4 时 vec 按 batch 分核（同 FULL_LOAD），否则按 TC 基本组分核
    __aicore__ inline bool IsVecSplitByBatch() const;

    uint32_t prevApeDStartIdx_ = 0;
    uint32_t prevApeDDealSize_ = 0;
    bool apeIsLoad_ = false;
    uint32_t totalBatchNum_ = 0; // 本轮内待处理的 batch 总数，仅 batch 分核模式使用
    MSplitInfo mSplitInfo = {};
    bool rawBatchLoaded_ = false;
    uint32_t rawSeqStart_ = 0;
};

template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::CopyInApe(const LocalTensor<T>& apeUb, uint32_t dStartIdx,
                                                                    uint32_t dDealSize)
{
    if (apeIsLoad_ && prevApeDStartIdx_ == dStartIdx && prevApeDDealSize_ == dDealSize) {
        return;
    }

    uint32_t copyRowCount = this->coff_ * this->cmpRatio_;
    uint32_t copyColCount = dDealSize;
    uint32_t dstSingleRowCount = dDealSize;
    uint32_t srcSingleRowCount = this->constInfo_.headDim;

    uint64_t gmOffset = dStartIdx;

    this->CopyInWithSync(this->apeUb, this->apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount,
                         dstSingleRowCount, COMP_MTX_IN_APE);

    prevApeDStartIdx_ = dStartIdx;
    prevApeDDealSize_ = dDealSize;
    apeIsLoad_ = true;
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::AddApeToScore(const LocalTensor<T>& scoreLocal,
                                                                        const LocalTensor<T>& apeUb,
                                                                        const Vec1SliceInfo& sliceInfo,
                                                                        uint32_t dDealSize)
{
    uint32_t singleRowElemNum = dDealSize * this->coff_;
    uint64_t scoreOffset = sliceInfo.dealedSeqCnt * singleRowElemNum;

    uint32_t tcDealSize = sliceInfo.dealTcSize;
    if (sliceInfo.headHolderSeqCnt > 0) {
        uint64_t apeOffset = sliceInfo.headHolderSeqCnt * singleRowElemNum;
        uint32_t row = tcDealSize == 1 ? sliceInfo.validSeqCnt : (this->cmpRatio_ - sliceInfo.headHolderSeqCnt);
        AddVF(scoreLocal[scoreOffset], this->apeUb[apeOffset], this->coff_ * row, dDealSize, dDealSize);
        scoreOffset += row * singleRowElemNum;
        tcDealSize -= 1;
    }
    if (tcDealSize == 0) {
        return;
    }
    if (sliceInfo.tailHolderSeqCnt > 0) {
        tcDealSize -= 1;
        uint64_t apeOffset = 0;
        uint32_t row = this->cmpRatio_ - sliceInfo.tailHolderSeqCnt;
        uint32_t tailScoreOffset = scoreOffset + tcDealSize * this->cmpRatio_ * singleRowElemNum;
        AddVF(scoreLocal[tailScoreOffset], this->apeUb[apeOffset], this->coff_ * row, dDealSize, dDealSize);
    }
    if (tcDealSize == 0) {
        return;
    }
    uint32_t row = this->cmpRatio_;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        AddVF(scoreLocal[scoreOffset + r * row * singleRowElemNum], this->apeUb, this->coff_ * row, dDealSize,
              dDealSize);
    }
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::AddApe(const LocalTensor<T>& scoreLocal,
                                                                 const LocalTensor<T>& apeUb,
                                                                 const Vec1SliceInfo& sliceInfo, uint32_t dDealSize)
{
    uint32_t singleRowElemNum = dDealSize * this->coff_;
    uint32_t dealRowCount = min(sliceInfo.sIdx, this->cmpRatio_);
    uint64_t scoreOffset = (this->cmpRatio_ - dealRowCount) * singleRowElemNum;
    uint64_t apeOffset = (this->cmpRatio_ - dealRowCount) * singleRowElemNum;
    AddVF(scoreLocal[scoreOffset], this->apeUb[apeOffset], dealRowCount, dDealSize, singleRowElemNum);
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::OverLap(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const GlobalTensor<T>& srcGm,
    const GlobalTensor<T>& stateGm, const GlobalTensor<int32_t>& blockTableGm, const GlobalTensor<T>& cacheTcGm,
    const Vec1RunInfo& info, const Vec1SliceInfo& sliceInfo, const LoopInfo& loopInfo, uint32_t dStartIdx,
    uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    if constexpr (IS_SCORE) {
        AddApeToScore(srcLocal, this->apeUb, sliceInfo, dDealSize);
        PipeBarrier<PIPE_V>();
    }
    this->SaveState(srcLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize, dDealSize,
                    static_cast<uint32_t>(IS_SCORE));
    this->template ReadState<IS_SCORE>(dstLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize,
                                       static_cast<uint32_t>(IS_SCORE));

    if constexpr (COMP::coff == COFF::OVERLAP) {
        uint32_t nextC1V1DbIdx = (info.c1v1DbIdx + 1) % this->constInfo_.dbWorkspaceRatio;
        GlobalTensor<T> nextCacheTcGm = cacheTcGm[nextC1V1DbIdx * this->cmpRatio_ * this->constInfo_.headDim];
        this->SaveToWorkSpace(srcLocal, nextCacheTcGm, sliceInfo, loopInfo, dStartIdx, dDealSize,
                              IS_SCORE ? COMP_MTX_CACHE_SCORE : COMP_MTX_CACHE_KV);
    }
    if (sliceInfo.compressTcSize > 0) {
        this->PadAlign(dstLocal, srcLocal, sliceInfo, 0, dDealSize, dDealSize);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            GlobalTensor<T> curCacheTcGm = cacheTcGm[info.c1v1DbIdx * this->cmpRatio_ * this->constInfo_.headDim];
            this->LoadFromWorkSpace(dstLocal, curCacheTcGm, srcGm, srcLocal, sliceInfo, loopInfo, dStartIdx,
                                    globalSeqIdx, dDealSize, IS_SCORE ? COMP_MTX_CACHE_SCORE : COMP_MTX_CACHE_KV);
        }
    }
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::OverLapScoreKv(
    const LocalTensor<T>& scoreLocal, const LocalTensor<T>& kvLocal, const Vec1RunInfo& info, const LoopInfo& loopInfo,
    const StatisticInfo& statisticInfo, const Vec1SliceInfo& originSliceInfo, uint32_t dStartIdx, uint32_t dDealSize,
    uint32_t dBaseSize, uint32_t needDealTcSize)
{
    CompressorVec1SliceIterator overLapSliceIterator(this->tools_);
    overLapSliceIterator.SetMaxBatchSize(this->constInfo_.batchSize);
    Vec1SliceInfo& overLapSliceInfo = overLapSliceIterator.GetSlice();

    GlobalTensor<T> scoreDBMm1ResGm = this->scoreMm1ResGm_[info.c1v1DbIdx * this->constInfo_.dbSize];
    // 普通路径 score/KV 顺序复用完整 64K；批量预载路径使用各 32K 的驻留原始数据。
    uint32_t rawOffset = rawBatchLoaded_ ? (originSliceInfo.dealedSeqCnt - rawSeqStart_) * this->coff_ * dDealSize : 0U;
    LocalTensor<T> scoreUb = rawBatchLoaded_ ? this->scoreUb[rawOffset] : this->mm1ResUb;
    if (!rawBatchLoaded_) {
        Mutex::Lock<PIPE_MTE2>(COMP_MTX_MM1);
        this->LoadRawFromWorkspace(scoreUb, scoreDBMm1ResGm, originSliceInfo.dealedSeqCnt, statisticInfo.dealSeqCnt,
                                   dStartIdx, dDealSize);
        Mutex::Unlock<PIPE_MTE2>(COMP_MTX_MM1);
        Mutex::Lock<PIPE_V>(COMP_MTX_MM1);
    }
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<true>(scoreLocal, scoreUb, scoreDBMm1ResGm, this->stateCacheGm_, this->stateBlockTableGm_,
                      this->scoreCacheTcGm_, info, overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt,
                      dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    if (!rawBatchLoaded_) {
        Mutex::Unlock<PIPE_V>(COMP_MTX_MM1);
    }

    if constexpr (COMP::coff == COFF::OVERLAP) {
        if (originSliceInfo.sIdx != 0 && originSliceInfo.compressTcSize > 0 &&
            (!loopInfo.isCoreRowFirst || !loopInfo.isCoreLoopFirst)) {
            PipeBarrier<PIPE_V>();
            AddApe(scoreLocal, this->apeUb, originSliceInfo, dDealSize);
        }
    }

    GlobalTensor<T> kvDBMm1ResGm = this->kvMm1ResGm_[info.c1v1DbIdx * this->constInfo_.dbSize];
    LocalTensor<T> kvUb = rawBatchLoaded_ ? this->kvUb[rawOffset] : this->mm1ResUb;
    if (!rawBatchLoaded_) {
        Mutex::Lock<PIPE_MTE2>(COMP_MTX_MM1);
        this->LoadRawFromWorkspace(kvUb, kvDBMm1ResGm, originSliceInfo.dealedSeqCnt, statisticInfo.dealSeqCnt,
                                   dStartIdx, dDealSize);
        Mutex::Unlock<PIPE_MTE2>(COMP_MTX_MM1);
        Mutex::Lock<PIPE_V>(COMP_MTX_MM1);
    }
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);

    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<false>(kvLocal, kvUb, kvDBMm1ResGm, this->stateCacheGm_, this->stateBlockTableGm_, this->kvCacheTcGm_,
                       info, overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    if (!rawBatchLoaded_) {
        Mutex::Unlock<PIPE_V>(COMP_MTX_MM1);
    }

    PipeBarrier<PIPE_V>();
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::DealVec1BaseBlock(
    const Vec1RunInfo& info, CompressorVec1SliceIterator<COMP>& sliceIterator, const LoopInfo& loopInfo,
    uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo& statisticInfo = sliceIterator.CollectSliceStats();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }
    LocalTensor<T> scoreLocal = this->scoreLocalUb;
    LocalTensor<T> kvLocal = this->kvLocalUb;

    OverLapScoreKv(scoreLocal, kvLocal, info, loopInfo, statisticInfo, originSliceInfo, dStartIdx, dDealSize, dBaseSize,
                   needDealTcSize);

    if (statisticInfo.compressorScCnt > 0) {
        if constexpr (COMP::gradEnabled == GRAD_ENABLED::DISABLE) {
            if (this->coff_ * this->cmpRatio_ == 8 && dDealSize >= baseD64 && dDealSize % baseD64 == 0) {
                // 快路径是否命中只由 shape 决定，与 batch 分核/循环位置无关：
                //   窗口 = 一个压缩窗覆盖的行数 coff*cmpRatio，必须恰为 8（VF 一次吃满 8 行，
                //          在寄存器内完成 softmax 与加权归约）；
                //   列宽 = 本轮 D 轴处理列数 dDealSize，必须 >= 64 且是 baseD64(64) 的整数倍，
                //          因为 VF 按 baseD64 分 chunk 处理。
                // 仅无中间输出场景做这个融合，省掉 softmax 概率写回 UB 再读出的往返。
                SoftmaxMulReduce8VF(kvLocal, scoreLocal, this->vec1ResUb, dDealSize, statisticInfo.compressorScCnt,
                                    this->SOFTMAX_MIN_NUM);
                PipeBarrier<PIPE_V>();
                this->CopyOutVec1ResToOutput(this->vec1ResUb, originSliceInfo, statisticInfo.compressorScCnt, dStartIdx,
                                             dDealSize);
                this->compressedCnt_ += statisticInfo.compressorScCnt;
                return;
            }
        }
        this->SoftmaxDN(scoreLocal, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        if constexpr (COMP::gradEnabled == GRAD_ENABLED::ENABLE) {
            this->CopyOutMidResToOutput(kvLocal, scoreLocal, originSliceInfo, statisticInfo.compressorScCnt, dStartIdx,
                                        dDealSize, this->compressedCnt_);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<T> comperssoredUb = this->vec1ResUb;
        this->KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        this->CopyOutVec1ResToOutput(comperssoredUb, originSliceInfo, statisticInfo.compressorScCnt, dStartIdx,
                                     dDealSize);
    }
    this->compressedCnt_ += statisticInfo.compressorScCnt;
}
template <typename COMP>
__aicore__ inline bool CompressorBlockVectorNormal<COMP>::IsVecSplitByBatch() const
{
    // 与 kernel 的 CompressorKernel::IsVecSplitByBatch 保持一致：
    // BSH 布局且 seqSize<=4 时轮次按 batch 收尾，vec 按整 batch 分核；
    // vec 对核组 M 段划分透明（核组之间可以切 batch）
    return COMP::xLayout == X_LAYOUT::BSH && this->constInfo_.sSize > 0 &&
           this->constInfo_.sSize <= VEC_BATCH_SPLIT_MAX_SEQ_SIZE;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::CalcGroupInfo(const Vec1RunInfo& info,
                                                                        Vec1SplitInfo& splitInfo)
{
    uint32_t aiCoreNum = this->constInfo_.usedCoreNum * this->cvRatio_;
    // 分核参照量：batch 分核时为轮次内 batch 总数，否则为轮次内 TC 块总数
    uint32_t splitRefNum = info.dealTcNum;
    if (IsVecSplitByBatch()) {
        totalBatchNum_ = info.windowSeqCnt / this->constInfo_.sSize;
        splitRefNum = totalBatchNum_;
    }
    splitRefNum = max(splitRefNum, 1U); // 防空除
    splitInfo.dBaseSize =
        this->constInfo_.headDim / min(FloorPow2(aiCoreNum), CeilPow2(CeilDivT(aiCoreNum, splitRefNum)));
    // 32B(8个FP32)对齐的UB列窗口上限。dBaseSize 超过它时 this->CalcTilingStrategy 会走
    // dSplitSize = dBaseSize/dLoopCount 整数除法拆分，切出的列窗口非32B对齐
    // （DataCopy 的 blockLen/srcGap 整数除法错位 → 行间源偏移错误 → 数据错乱）。
    // clamp 到“不超过 maxDealColNum 的最大2的幂”后，dSplitSize 恒等于 dBaseSize
    // （2的幂，天然8元素对齐），不再进入拆分分支。
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_64K / (this->cmpRatio_ * this->coff_ * sizeof(T));
    splitInfo.dBaseSize = min(splitInfo.dBaseSize, FloorPow2(Trunc(maxDealColNum, BlockElementNum<T>())));
    if (this->constInfo_.kBaseNum > 1) {
        splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_REPEAT_ELEMENT_NUM);
    }
    // 结果输出到GM前必须转换成X_T，dBaseSize * sizeof(X_T)需32B对齐
    splitInfo.dBaseSize = max(splitInfo.dBaseSize, BlockElementNum<X_T>());
    splitInfo.vec1GroupSize = this->constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum = min(static_cast<uint32_t>(aiCoreNum / splitInfo.vec1GroupSize), splitRefNum);
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::CalcTaskDistribution(const Vec1RunInfo& info,
                                                                               Vec1SplitInfo& splitInfo)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t groupSize = splitInfo.vec1GroupSize;
    uint32_t groupNum = splitInfo.vec1GroupNum;

    if (IsVecSplitByBatch()) {
        // 按 batch 均分（同 FULL_LOAD）：行组 blockIdx/groupSize 负责轮次内从轮次起点数起的
        // 第 [preDealBatchNum, preDealBatchNum + dealBatchNum) 个 batch，batch 不会被切开
        uint32_t totalDealBatchNum = totalBatchNum_;
        if (blockIdx < groupSize * (totalDealBatchNum % groupNum)) {
            splitInfo.dealBatchNum = totalDealBatchNum / groupNum + 1;
            splitInfo.preDealBatchNum = splitInfo.dealBatchNum * (blockIdx / groupSize);
        } else if (blockIdx < groupSize * groupNum) {
            splitInfo.dealBatchNum = totalDealBatchNum / groupNum;
            splitInfo.preDealBatchNum = splitInfo.dealBatchNum * (blockIdx / groupSize) + totalDealBatchNum % groupNum;
        } else {
            splitInfo.dealBatchNum = 0;
            splitInfo.preDealBatchNum = totalDealBatchNum;
        }
        return;
    }

    uint32_t dealTcNum = info.dealTcNum;

    if (blockIdx < groupSize * (dealTcNum % groupNum)) {
        splitInfo.dealTcSize = dealTcNum / groupNum + 1;
        splitInfo.preDealTcSize = splitInfo.dealTcSize * (blockIdx / groupSize);
    } else if (blockIdx < groupSize * groupNum) {
        splitInfo.dealTcSize = dealTcNum / groupNum;
        splitInfo.preDealTcSize = splitInfo.dealTcSize * (blockIdx / groupSize) + dealTcNum % groupNum;
    } else {
        splitInfo.dealTcSize = 0;
        splitInfo.preDealTcSize = dealTcNum;
    }
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::UpdateIteratorState(const Vec1RunInfo& info,
                                                                              Vec1SplitInfo& splitInfo)
{
    if (IsVecSplitByBatch()) {
        // 按 batch 分核：轮次起点/终点均为 batch 起点（cube 侧轮次按 batch 收尾），
        // workspace 行偏移按已跳过 batch 的行数推进
        splitInfo.curBStart = info.bStart + splitInfo.preDealBatchNum;
        splitInfo.curSStart = splitInfo.preDealBatchNum == 0U ? info.sStart : 0U;
        splitInfo.dealSeqStartIdx = splitInfo.preDealBatchNum * this->constInfo_.sSize;
        splitInfo.preDealTcSize = 0U;
        splitInfo.curCompressedCnt = 0U;
        uint32_t dealTcSize = 0U;
        uint32_t compressedCnt = 0U;
        uint32_t dealBEnd = splitInfo.curBStart + splitInfo.dealBatchNum;
        for (uint32_t bIdx = splitInfo.curBStart; bIdx < dealBEnd && bIdx < this->constInfo_.batchSize; ++bIdx) {
            // 基类是依赖基类（依赖 COMP），基类成员必须显式 this-> 限定
            uint32_t bStartPos = this->GetStartPos(bIdx);
            uint32_t seqLength = this->GetSeqLength(bIdx);
            uint32_t seqUsed = this->GetSeqUsed(bIdx);
            // 与 cube 侧 tcNum 口径一致：本 batch 参与的压缩窗数
            dealTcSize +=
                static_cast<uint32_t>(CeilDivT(bStartPos + seqLength, this->cmpRatio_)) - bStartPos / this->cmpRatio_;
            compressedCnt += (bStartPos + seqUsed) / this->cmpRatio_ - bStartPos / this->cmpRatio_;
        }
        splitInfo.dealTcSize = dealTcSize;
        splitInfo.totalCompressedCnt = compressedCnt;
        return;
    }

    CompressorVec1SliceIterator sliceIterator(this->tools_);
    sliceIterator.SetMaxBatchSize(this->constInfo_.batchSize);
    sliceIterator.Reset(info.bStart, info.sStart, 0U, 0U);
    Vec1SliceInfo& sliceInfo = sliceIterator.GetSlice();

    // 处理前序任务量，更新起始索引
    if (splitInfo.preDealTcSize > 0) {
        sliceIterator.SetNeedDealTcSize(splitInfo.preDealTcSize);
        StatisticInfo& statisticInfo = sliceIterator.CollectSliceStats();
        splitInfo.curCompressedCnt = statisticInfo.compressorScCnt;
        splitInfo.dealSeqStartIdx = sliceInfo.dealedSeqCnt;
        splitInfo.curBStart = sliceInfo.bIdx;
        splitInfo.curSStart = sliceInfo.sIdx;
    } else {
        splitInfo.curCompressedCnt = 0;
        splitInfo.dealSeqStartIdx = 0;
        splitInfo.curBStart = info.bStart;
        splitInfo.curSStart = info.sStart;
    }

    // 处理当前核实际要跑的任务量
    sliceIterator.SetNeedDealTcSize(info.dealTcNum - splitInfo.preDealTcSize);
    StatisticInfo& statisticInfo = sliceIterator.CollectSliceStats();
    splitInfo.totalCompressedCnt = splitInfo.curCompressedCnt + statisticInfo.compressorScCnt;
}
template <typename COMP>
__aicore__ inline Vec1SplitInfo CompressorBlockVectorNormal<COMP>::SplitCoreV1(const Vec1RunInfo& info)
{
    Vec1SplitInfo splitInfo;

    // 1. 计算基础分组和分片大小
    CalcGroupInfo(info, splitInfo);

    // 2. 根据当前的 BlockIdx 计算任务分配（负载均衡）
    if (splitInfo.vec1GroupNum == 0U) {
        return splitInfo;
    }
    CalcTaskDistribution(info, splitInfo);

    // 3. 刷新迭代器并获取当前核的起始位置状态
    UpdateIteratorState(info, splitInfo);

    if (splitInfo.dealTcSize == 0) {
        return splitInfo;
    }

    // 4. 计算具体在内存中的切块（Tiling）逻辑
    this->CalcTilingStrategy(splitInfo);

    return splitInfo;
}
template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::ComputeBatchTiles(const Vec1RunInfo& info,
                                                                            const Vec1SplitInfo& splitInfo,
                                                                            LoopInfo& loopInfo, uint32_t dStartIdx,
                                                                            uint32_t dDealSize)
{
    // 原始数据按 S 切批量，历史/压缩窗仍按 R 切块，避免 R 大时逐 batch 搬入很小的原始结果。
    const uint32_t batchCapacity =
        BUFFER_SIZE_BYTE_32K / (this->constInfo_.sSize * this->coff_ * dDealSize * sizeof(T));
    GlobalTensor<T> scoreGm = this->scoreMm1ResGm_[info.c1v1DbIdx * this->constInfo_.dbSize];
    GlobalTensor<T> kvGm = this->kvMm1ResGm_[info.c1v1DbIdx * this->constInfo_.dbSize];
    CompressorVec1SliceIterator<COMP> sliceIterator(this->tools_);
    for (uint32_t batchOffset = 0; batchOffset < splitInfo.dealBatchNum; batchOffset += batchCapacity) {
        const uint32_t batchCount = min(batchCapacity, splitInfo.dealBatchNum - batchOffset);
        const uint32_t bStart = splitInfo.curBStart + batchOffset;
        const uint32_t bEnd = bStart + batchCount;
        rawSeqStart_ = splitInfo.dealSeqStartIdx + batchOffset * this->constInfo_.sSize;
        Mutex::Lock<PIPE_MTE2>(COMP_MTX_MM1);
        this->LoadRawFromWorkspace(this->scoreUb, scoreGm, rawSeqStart_, batchCount * this->constInfo_.sSize, dStartIdx,
                                   dDealSize);
        this->LoadRawFromWorkspace(this->kvUb, kvGm, rawSeqStart_, batchCount * this->constInfo_.sSize, dStartIdx,
                                   dDealSize);
        Mutex::Unlock<PIPE_MTE2>(COMP_MTX_MM1);
        Mutex::Lock<PIPE_V>(COMP_MTX_MM1);
        rawBatchLoaded_ = true;

        uint32_t tcCount = 0;
        for (uint32_t bIdx = bStart; bIdx < bEnd; ++bIdx) {
            uint64_t sp = this->GetStartPos(bIdx);
            tcCount +=
                CeilDivT(sp + this->constInfo_.sSize, static_cast<uint64_t>(this->cmpRatio_)) - sp / this->cmpRatio_;
        }
        sliceIterator.SetMaxBatchSize(bEnd);
        sliceIterator.Reset(bStart, 0U, rawSeqStart_, 0U);
        for (uint32_t tcIdx = 0; tcIdx < tcCount; tcIdx += splitInfo.tcSplitSize) {
            loopInfo.isCoreLoopFirst = batchOffset == 0 && tcIdx == 0;
            loopInfo.isCoreLoopLast =
                batchOffset + batchCount == splitInfo.dealBatchNum && tcIdx + splitInfo.tcSplitSize >= tcCount;
            sliceIterator.SetNeedDealTcSize(min(splitInfo.tcSplitSize, tcCount - tcIdx));
            DealVec1BaseBlock(info, sliceIterator, loopInfo, dStartIdx, dDealSize, splitInfo.dBaseSize);
        }
        rawBatchLoaded_ = false;
        Mutex::Unlock<PIPE_V>(COMP_MTX_MM1);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorNormal<COMP>::ComputeVec1(const Vec1RunInfo& info)
{
    if (IsVecSplitByBatch() ? (info.windowSeqCnt == 0U) : (info.dealTcNum == 0U)) {
        return;
    }
    uint32_t preCompressedCnt = this->compressedCnt_;
    Vec1SplitInfo splitInfo = SplitCoreV1(info);
    // 计算当前VecCore的任务量
    if (splitInfo.dealTcSize == 0) {
        this->compressedCnt_ += splitInfo.totalCompressedCnt;
        return;
    }

    LoopInfo loopInfo = this->GetLoopInfo(splitInfo);

    CompressorVec1SliceIterator sliceIterator(this->tools_);
    sliceIterator.SetMaxBatchSize(this->constInfo_.batchSize);
    // 切块循环
    uint64_t baseOffset = loopInfo.coreColIdx * splitInfo.dBaseSize;
    for (uint32_t dLoopIdx = 0; dLoopIdx < splitInfo.dLoopCount; dLoopIdx++) {
        uint64_t dBaseOffset = baseOffset + dLoopIdx * splitInfo.dSplitSize;

        CopyInApe(this->apeUb, dBaseOffset, splitInfo.dSplitSize);

        this->compressedCnt_ = preCompressedCnt + splitInfo.curCompressedCnt;
        if (IsVecSplitByBatch()) {
            ComputeBatchTiles(info, splitInfo, loopInfo, dBaseOffset, splitInfo.dSplitSize);
            continue;
        }
        sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, splitInfo.dealSeqStartIdx, 0U);
        for (uint32_t tcIdx = 0; tcIdx < splitInfo.dealTcSize; tcIdx += splitInfo.tcSplitSize) {
            uint32_t actDealTcSize = min(splitInfo.tcSplitSize, splitInfo.dealTcSize - tcIdx);

            loopInfo.isCoreLoopFirst = tcIdx == 0;
            loopInfo.isCoreLoopLast = tcIdx + splitInfo.tcSplitSize >= splitInfo.dealTcSize;
            // 处理单个切块
            sliceIterator.SetNeedDealTcSize(actDealTcSize);
            DealVec1BaseBlock(info, sliceIterator, loopInfo, dBaseOffset, splitInfo.dSplitSize, splitInfo.dBaseSize);
        }
    }
    this->compressedCnt_ = preCompressedCnt + splitInfo.totalCompressedCnt;
}
} // namespace Compressor

#endif // COMPRESSOR_BLOCK_VEC_NORMAL_H
