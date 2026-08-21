/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_finalize_routing_v2_grad_with_scale_cut_h.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_GRAD_REGBASE_CUT_H_H
#define MOE_FINALIZE_ROUTING_V2_GRAD_REGBASE_CUT_H_H

#include "moe_finalize_routing_v2_grad_regbase.h"

namespace MoeFinalizeRoutingV2Grad {
using namespace AscendC;

template <typename T1, typename T2, typename T3, bool IsBiasExist = true>
class MoeFinalizeRoutingV2GradRegBaseCutH : MoeFinalizeRoutingV2GradRegbase<T1, T2, T3, IsBiasExist> {
public:
    __aicore__ inline MoeFinalizeRoutingV2GradRegBaseCutH(){};
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR scales,
                                GM_ADDR expertIdx, GM_ADDR bias, GM_ADDR gradExpandedX, GM_ADDR gradScales,
                                GM_ADDR workspace, const MoeFinalizeRoutingV2GradSplitHTilingData *tilingData,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOutputGm();
    __aicore__ inline void PrepareComputeParams();
    __aicore__ inline void Compute();
    __aicore__ inline void OverLapPartProcess(const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache);
    __aicore__ inline void FoldTailProcess(const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache);
    __aicore__ inline void NoOverLapPartProcess(const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache);
    __aicore__ inline void TailBlockNumOneProcess(const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache);
    __aicore__ inline void TailBlockNumTwoProcess(const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache);
    __aicore__ inline void ProcessGradExpandedX(const LocalTensor<T1> &gradYUb, int64_t mainLen, int64_t foldLen,
                                                int64_t mainOutOffset, int64_t foldOutOffset);
    __aicore__ inline bool IsNeedUpdateLevel1Cache(const int64_t cacheIdx);
    __aicore__ inline bool IsNeedUpdateLevel2Cache(const int64_t cacheIdx);
    __aicore__ inline void UpdateCache(const int64_t cacheIdx, const LocalTensor<float> &binaryAddCache);
    __aicore__ inline void CacheReduceSum(const LocalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor,
                                          const uint32_t idx);
    __aicore__ inline void ProcessGradScales(int64_t batchIdx);

private:
    const MoeFinalizeRoutingV2GradSplitHTilingData *tilingData_;
    TBuf<TPosition::VECCALC> bianryAddCacheBuf_;
    const uint32_t CACHE_BUFF_SIZE = 256;
    const uint32_t MAX_CACHE_DEPTH = 3;
    const uint32_t CACHE_LEVEL0_IDX = 0;
    const uint32_t CACHE_LEVEL1_IDX = 256;
    const uint32_t CACHE_LEVEL2_IDX = 512;
    int64_t mainFactor_;
    int64_t foldFactor_;
    int64_t typeAlignFactor_;
    int64_t cacheIdx_ = 0;
    int64_t levelIdx_ = 0;
    int64_t h_ = 0;
    T2 rowIdx_ = 0;
    T2 expertIdx_ = 0;
    T3 scale_ = 0.0;
};

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::Init(
    GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR bias,
    GM_ADDR gradExpandedX, GM_ADDR gradScales, GM_ADDR workspace,
    const MoeFinalizeRoutingV2GradSplitHTilingData *tilingData, TPipe *pipe)
{
    this->InitCommon(pipe);
    this->tilingData_ = tilingData;
    this->mainFactor_ = tilingData->mainFactor;
    this->foldFactor_ = tilingData->foldFactor;
    this->h_ = tilingData_->baseParams.hidden;
    this->typeAlignFactor_ = BLOCK_BYTE_SIZE / sizeof(T1);
    this->gradYGm_.SetGlobalBuffer((__gm__ T1 *)gradY);
    this->expandedRowIdxGm_.SetGlobalBuffer((__gm__ T2 *)expandedRowIdx);
    this->expandedXGm_.SetGlobalBuffer((__gm__ T1 *)expandedX);
    this->scalesGm_.SetGlobalBuffer((__gm__ T3 *)scales);
    this->expertIdxGm_.SetGlobalBuffer((__gm__ T2 *)expertIdx);
    this->biasGm_.SetGlobalBuffer((__gm__ T1 *)bias);
    this->gradExpandedXGm_.SetGlobalBuffer((__gm__ T1 *)gradExpandedX);
    this->gradScalesGm_.SetGlobalBuffer((__gm__ T3 *)gradScales);
    if (tilingData_->baseParams.dropPadMode == 1) {
        this->InitOutputGm();
        SyncAll();
    }
    this->pipe_->InitBuffer(this->gradYInQueue_, DOUBLE_BUFFER, mainFactor_ * sizeof(T1));
    this->pipe_->InitBuffer(this->expandedXInQueue_, DOUBLE_BUFFER, mainFactor_ * sizeof(T1));
    this->pipe_->InitBuffer(this->gradExpandedXOutQueue_, DOUBLE_BUFFER, mainFactor_ * sizeof(T1));
    this->pipe_->InitBuffer(this->gradScalesOutQueue_, 1, BLOCK_BYTE_SIZE);
    this->pipe_->InitBuffer(this->bianryAddCacheBuf_, this->MAX_CACHE_DEPTH * this->CACHE_BUFF_SIZE * sizeof(float));
    this->pipe_->InitBuffer(this->binaryAddSumBuf_, this->CACHE_BUFF_SIZE * sizeof(float));
    if constexpr (IsBiasExist) {
        this->pipe_->InitBuffer(this->biasInQueue_, DOUBLE_BUFFER, mainFactor_ * sizeof(T1));
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::Process()
{
    PrepareComputeParams();
    if (unlikely(this->endBatchIdx_ == 0 || this->startBatchIdx_ == this->endBatchIdx_)) {
        return;
    }
    Compute();
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::InitOutputGm()
{
    if (this->blockIdx_ < tilingData_->baseParams.initOutModCoreNum) {
        this->startBatchIdx_ = this->blockIdx_ * (tilingData_->baseParams.initOutEachCoreBatchNum + 1);
        this->endBatchIdx_ = this->startBatchIdx_ + tilingData_->baseParams.initOutEachCoreBatchNum + 1;
    } else if (this->blockIdx_ < tilingData_->baseParams.initOutNeedCoreNum) {
        this->startBatchIdx_ = this->blockIdx_ * tilingData_->baseParams.initOutEachCoreBatchNum +
                               tilingData_->baseParams.initOutModCoreNum;
        this->endBatchIdx_ = this->startBatchIdx_ + tilingData_->baseParams.initOutEachCoreBatchNum;
    } else {
        this->startBatchIdx_ = 0;
        this->endBatchIdx_ = 0;
    }
    if (this->endBatchIdx_ != 0 && this->startBatchIdx_ != this->endBatchIdx_) {
        GlobalTensor<T1> initGm = this->gradExpandedXGm_[this->startBatchIdx_ * this->h_];
        InitOutput<T1>(initGm, this->h_ * (this->endBatchIdx_ - this->startBatchIdx_), 0);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::PrepareComputeParams()
{
    if (this->blockIdx_ < tilingData_->baseParams.computeModCoreNum) {
        this->startBatchIdx_ = this->blockIdx_ * (tilingData_->baseParams.computeEachCoreBatchNum + 1);
        this->endBatchIdx_ = this->startBatchIdx_ + tilingData_->baseParams.computeEachCoreBatchNum + 1;
    } else if (this->blockIdx_ < tilingData_->baseParams.computeNeedCoreNum) {
        this->startBatchIdx_ = this->blockIdx_ * tilingData_->baseParams.computeEachCoreBatchNum +
                               tilingData_->baseParams.computeModCoreNum;
        this->endBatchIdx_ = this->startBatchIdx_ + tilingData_->baseParams.computeEachCoreBatchNum;
    } else {
        this->startBatchIdx_ = 0;
        this->endBatchIdx_ = 0;
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::Compute()
{
    LocalTensor<float> gradScalesUb;
    for (uint32_t batchIdx = this->startBatchIdx_; batchIdx < this->endBatchIdx_; batchIdx++) {
        LocalTensor<float> binaryAddCache = this->bianryAddCacheBuf_.template Get<float>();
        Duplicate<float>(binaryAddCache, 0.0f, this->CACHE_BUFF_SIZE * this->MAX_CACHE_DEPTH);
        this->rowIdx_ = this->expandedRowIdxGm_.GetValue(batchIdx);
        this->scale_ = this->scalesGm_.GetValue(batchIdx);
        this->cacheIdx_ = 0;
        if constexpr (IsBiasExist) {
            this->expertIdx_ = this->expertIdxGm_.GetValue(batchIdx);
        }
        gradScalesUb = this->gradScalesOutQueue_.template AllocTensor<float>();
        OverLapPartProcess(batchIdx, binaryAddCache);
        FoldTailProcess(batchIdx, binaryAddCache);
        NoOverLapPartProcess(batchIdx, binaryAddCache);

        int64_t totalBlocks = this->cacheIdx_;
        int64_t level1Count = totalBlocks / CACHE_BUFF_SIZE;

        if (level1Count == 0) {
            CacheReduceSum(gradScalesUb, binaryAddCache[CACHE_LEVEL0_IDX], 0);
        } else if (level1Count < CACHE_BUFF_SIZE) {
            LocalTensor<float> level1Temp = binaryAddCache[CACHE_LEVEL2_IDX];
            CacheReduceSum(level1Temp, binaryAddCache[CACHE_LEVEL1_IDX], 0);
            CacheReduceSum(gradScalesUb, binaryAddCache[CACHE_LEVEL0_IDX], 0);
            __ubuf__ float *dstAddr = (__ubuf__ float *)gradScalesUb.GetPhyAddr();
            __ubuf__ float *srcAddr = (__ubuf__ float *)level1Temp.GetPhyAddr();
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<float> vregLevel1;
                MicroAPI::RegTensor<float> vregLevel0;
                MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
                LoadAlign(vregLevel1, srcAddr);
                LoadAlign(vregLevel0, dstAddr);
                Add(vregLevel0, vregLevel0, vregLevel1, preg);
                StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, vregLevel0, preg);
            }
        } else {
            int64_t level2Count = level1Count / CACHE_BUFF_SIZE;
            LocalTensor<float> level2Temp = binaryAddCache[CACHE_LEVEL2_IDX];
            LocalTensor<float> level1Temp = binaryAddCache[CACHE_LEVEL2_IDX + 1];
            CacheReduceSum(level2Temp, binaryAddCache[CACHE_LEVEL2_IDX], 0);
            CacheReduceSum(level1Temp, binaryAddCache[CACHE_LEVEL1_IDX], 0);
            CacheReduceSum(gradScalesUb, binaryAddCache[CACHE_LEVEL0_IDX], 0);
            __ubuf__ float *dstAddr = (__ubuf__ float *)gradScalesUb.GetPhyAddr();
            __ubuf__ float *level1Addr = (__ubuf__ float *)level1Temp.GetPhyAddr();
            __ubuf__ float *level2Addr = (__ubuf__ float *)level2Temp.GetPhyAddr();
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<float> vregLevel2;
                MicroAPI::RegTensor<float> vregLevel1;
                MicroAPI::RegTensor<float> vregLevel0;
                MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
                LoadAlign(vregLevel2, level2Addr);
                LoadAlign(vregLevel1, level1Addr);
                LoadAlign(vregLevel0, dstAddr);
                Add(vregLevel0, vregLevel0, vregLevel2, preg);
                Add(vregLevel0, vregLevel0, vregLevel1, preg);
                StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr, vregLevel0, preg);
            }
        }

        this->gradScalesOutQueue_.template EnQue(gradScalesUb);
        this->ProcessGradScales(batchIdx);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::OverLapPartProcess(
    const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache)
{
    LocalTensor<T1> gradYUb;
    LocalTensor<T1> biasUb;
    BinAddParams binAddParams = {tilingData_->mainBinAddParams.binaryAddQuotient,
                                 tilingData_->mainBinAddParams.binaryAddk,
                                 tilingData_->mainBinAddParams.binaryAddLastNum};
    int64_t gmOffset = batchIdx / tilingData_->baseParams.topK * tilingData_->baseParams.hidden;
    for (uint32_t foldIdx = 0; foldIdx < tilingData_->foldBlockNum; foldIdx++) {
        int64_t mainOffset = foldIdx * foldFactor_;
        int64_t foldOffset = mainFactor_ * tilingData_->mainBlockNum + foldIdx * foldFactor_;
        gradYUb = this->gradYInQueue_.template AllocTensor<T1>();
        DataCopy(gradYUb, this->gradYGm_[gmOffset + mainOffset], foldFactor_);
        DataCopy(gradYUb[foldFactor_], this->gradYGm_[gmOffset + foldOffset], foldFactor_);
        this->gradYInQueue_.template EnQue<T1>(gradYUb);
        gradYUb = this->gradYInQueue_.template DeQue<T1>();
        /*
            gradScale = reduceSum((expandedX + bias) * gradY)
                expandedXDim0 > rowIdx >= 0 是，expandedX 值才有效
                bias, expandedX都为0时，gradScales为0
        */
        if constexpr (IsBiasExist) {
            int64_t biasMainOffset = this->expertIdx_ * this->h_ + mainOffset;
            int64_t biasFoldOffset = this->expertIdx_ * this->h_ + foldOffset;
            biasUb = this->biasInQueue_.template AllocTensor<T1>();
            DataCopy(biasUb, this->biasGm_[biasMainOffset], foldFactor_);
            DataCopy(biasUb[foldFactor_], this->biasGm_[biasFoldOffset], foldFactor_);
        }
        if (this->rowIdx_ >= 0 && this->rowIdx_ < tilingData_->baseParams.expandedXDim0) {
            int64_t xMianOffset = this->rowIdx_ * this->h_ + mainOffset;
            int64_t xFoldOffset = this->rowIdx_ * this->h_ + foldOffset;
            ProcessGradExpandedX(gradYUb, foldFactor_, foldFactor_, xMianOffset, xFoldOffset);
            LocalTensor<T1> expandedXUb = this->expandedXInQueue_.template AllocTensor<T1>();
            DataCopy(expandedXUb, this->expandedXGm_[xMianOffset], foldFactor_);
            DataCopy(expandedXUb[foldFactor_], this->expandedXGm_[xFoldOffset], foldFactor_);
            this->expandedXInQueue_.template EnQue<T1>(expandedXUb);
            expandedXUb = this->expandedXInQueue_.template DeQue<T1>();
            if constexpr (IsBiasExist) {
                this->biasInQueue_.template EnQue<T1>(biasUb);
                biasUb = this->biasInQueue_.template DeQue<T1>();
                Add(expandedXUb, expandedXUb, biasUb, mainFactor_);
            }
            this->VfCalGradScale(expandedXUb, gradYUb, binAddParams, mainFactor_, binaryAddCache,
                                 this->cacheIdx_ % CACHE_BUFF_SIZE);
            this->expandedXInQueue_.FreeTensor(expandedXUb);
        } else {
            if constexpr (IsBiasExist) {
                this->biasInQueue_.template EnQue<T1>(biasUb);
                biasUb = this->biasInQueue_.template DeQue<T1>();
                this->VfCalGradScale(biasUb, gradYUb, binAddParams, mainFactor_, binaryAddCache,
                                     this->cacheIdx_ % CACHE_BUFF_SIZE);
            }
        }
        if constexpr (IsBiasExist) {
            this->biasInQueue_.FreeTensor(biasUb);
        }
        this->cacheIdx_ = this->cacheIdx_ + 1;
        UpdateCache(this->cacheIdx_, binaryAddCache);
        this->gradYInQueue_.FreeTensor(gradYUb);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::FoldTailProcess(
    const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache)
{
    // tailBlockNum只有一块的时候，主块和折叠块累加时，主块只有一半，折叠部分为空
    if (tilingData_->foldTailBlockNum == 2) {
        TailBlockNumTwoProcess(batchIdx, binaryAddCache);
    } else if (tilingData_->foldTailBlockNum == 1) {
        TailBlockNumOneProcess(batchIdx, binaryAddCache);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::TailBlockNumTwoProcess(
    const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache)
{
    LocalTensor<T1> gradYUb;
    LocalTensor<T1> biasUb;
    BinAddParams binAddParams = {tilingData_->mainBinAddParams.binaryAddQuotient,
                                 tilingData_->mainBinAddParams.binaryAddk,
                                 tilingData_->mainBinAddParams.binaryAddLastNum};
    int64_t gmOffset = batchIdx / tilingData_->baseParams.topK * tilingData_->baseParams.hidden;
    for (uint32_t foldIdx = 0; foldIdx < tilingData_->foldTailBlockNum; foldIdx++) {
        int64_t mainOffset = (tilingData_->foldBlockNum + foldIdx) * foldFactor_;
        int64_t foldOffset =
            mainFactor_ * tilingData_->mainBlockNum + (tilingData_->foldBlockNum + foldIdx) * foldFactor_;
        int64_t foldTailFactor = foldIdx == 0 ? foldFactor_ : tilingData_->foldTailLen - foldFactor_;
        gradYUb = this->gradYInQueue_.template AllocTensor<T1>();
        DataCopy(gradYUb, this->gradYGm_[gmOffset + mainOffset], foldFactor_);
        DataCopy(gradYUb[foldFactor_], this->gradYGm_[gmOffset + foldOffset],
                 Ops::Base::CeilAlign(foldTailFactor, typeAlignFactor_));
        this->gradYInQueue_.template EnQue<T1>(gradYUb);
        gradYUb = this->gradYInQueue_.template DeQue<T1>();
        if (foldFactor_ + foldTailFactor < tilingData_->mainBinAddParams.binaryAddQuotient) {
            binAddParams = {tilingData_->foldBinAddParams.binaryAddQuotient, tilingData_->foldBinAddParams.binaryAddk,
                            tilingData_->foldBinAddParams.binaryAddLastNum};
        }
        if constexpr (IsBiasExist) {
            int64_t biasMainOffset = this->expertIdx_ * this->h_ + mainOffset;
            int64_t biasFoldOffset = this->expertIdx_ * this->h_ + foldOffset;
            biasUb = this->biasInQueue_.template AllocTensor<T1>();
            DataCopy(biasUb, this->biasGm_[biasMainOffset], foldFactor_);
            DataCopy(biasUb[foldFactor_], this->biasGm_[biasFoldOffset],
                     Ops::Base::CeilAlign(foldTailFactor, typeAlignFactor_));
        }
        if (this->rowIdx_ >= 0 && this->rowIdx_ < tilingData_->baseParams.expandedXDim0) {
            int64_t xMainOffset = this->rowIdx_ * this->h_ + mainOffset;
            int64_t xFoldOffset = this->rowIdx_ * this->h_ + foldOffset;
            ProcessGradExpandedX(gradYUb, foldFactor_, foldTailFactor, xMainOffset, xFoldOffset);
            LocalTensor<T1> expandedXUb = this->expandedXInQueue_.template AllocTensor<T1>();
            DataCopy(expandedXUb, this->expandedXGm_[xMainOffset], foldFactor_);
            DataCopy(expandedXUb[foldFactor_], this->expandedXGm_[xFoldOffset],
                     Ops::Base::CeilAlign(foldTailFactor, typeAlignFactor_));
            this->expandedXInQueue_.template EnQue<T1>(expandedXUb);
            expandedXUb = this->expandedXInQueue_.template DeQue<T1>();
            if constexpr (IsBiasExist) {
                this->biasInQueue_.template EnQue<T1>(biasUb);
                biasUb = this->biasInQueue_.template DeQue<T1>();
                Add(expandedXUb, expandedXUb, biasUb, foldFactor_ + foldTailFactor);
            }
            this->VfCalGradScale(expandedXUb, gradYUb, binAddParams, foldFactor_ + foldTailFactor, binaryAddCache,
                                 this->cacheIdx_ % CACHE_BUFF_SIZE);
            this->expandedXInQueue_.FreeTensor(expandedXUb);
        } else {
            if constexpr (IsBiasExist) {
                this->biasInQueue_.template EnQue<T1>(biasUb);
                biasUb = this->biasInQueue_.template DeQue<T1>();
                this->VfCalGradScale(biasUb, gradYUb, binAddParams, foldFactor_ + foldTailFactor, binaryAddCache,
                                     this->cacheIdx_ % CACHE_BUFF_SIZE);
            }
        }
        if constexpr (IsBiasExist) {
            this->biasInQueue_.FreeTensor(biasUb);
        }
        this->cacheIdx_ = this->cacheIdx_ + 1;
        UpdateCache(this->cacheIdx_, binaryAddCache);
        this->gradYInQueue_.FreeTensor(gradYUb);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::TailBlockNumOneProcess(
    const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache)
{
    LocalTensor<T1> biasUb;
    int64_t gmOffset = batchIdx / tilingData_->baseParams.topK * tilingData_->baseParams.hidden;
    int64_t mainOffset = tilingData_->foldBlockNum * foldFactor_;
    int64_t foldOffset = mainFactor_ * tilingData_->mainBlockNum + tilingData_->foldBlockNum * foldFactor_;
    int64_t foldTailFactor = tilingData_->foldTailLen;
    LocalTensor<T1> gradYUb = this->gradYInQueue_.template AllocTensor<T1>();
    DataCopy(gradYUb, this->gradYGm_[gmOffset + mainOffset], foldFactor_);
    DataCopy(gradYUb[foldFactor_], this->gradYGm_[gmOffset + foldOffset],
             Ops::Base::CeilAlign(foldTailFactor, typeAlignFactor_));
    this->gradYInQueue_.template EnQue<T1>(gradYUb);
    gradYUb = this->gradYInQueue_.template DeQue<T1>();
    if constexpr (IsBiasExist) {
        int64_t biasMainOffset = this->expertIdx_ * this->h_ + mainOffset;
        int64_t biasFoldOffset = this->expertIdx_ * this->h_ + foldOffset;
        biasUb = this->biasInQueue_.template AllocTensor<T1>();
        DataCopy(biasUb, this->biasGm_[biasMainOffset], foldFactor_);
        DataCopy(biasUb[foldFactor_], this->biasGm_[biasFoldOffset],
                 Ops::Base::CeilAlign(foldTailFactor, typeAlignFactor_));
    }
    BinAddParams binAddParams = {tilingData_->mainBinAddParams.binaryAddQuotient,
                                 tilingData_->mainBinAddParams.binaryAddk,
                                 tilingData_->mainBinAddParams.binaryAddLastNum};
    if (foldFactor_ + foldTailFactor < tilingData_->mainBinAddParams.binaryAddQuotient) {
        binAddParams = {tilingData_->foldBinAddParams.binaryAddQuotient, tilingData_->foldBinAddParams.binaryAddk,
                        tilingData_->foldBinAddParams.binaryAddLastNum};
    }
    if (this->rowIdx_ >= 0 && this->rowIdx_ < tilingData_->baseParams.expandedXDim0) {
        int64_t xMainOffset = this->rowIdx_ * this->h_ + mainOffset;
        int64_t xFoldOffset = this->rowIdx_ * this->h_ + foldOffset;
        ProcessGradExpandedX(gradYUb, foldFactor_, foldTailFactor, xMainOffset, xFoldOffset);
        LocalTensor<T1> expandedXUb = this->expandedXInQueue_.template AllocTensor<T1>();
        DataCopy(expandedXUb, this->expandedXGm_[xMainOffset], foldFactor_);
        DataCopy(expandedXUb[foldFactor_], this->expandedXGm_[xFoldOffset],
                 Ops::Base::CeilAlign(foldTailFactor, typeAlignFactor_));
        this->expandedXInQueue_.template EnQue<T1>(expandedXUb);
        expandedXUb = this->expandedXInQueue_.template DeQue<T1>();
        if constexpr (IsBiasExist) {
            this->biasInQueue_.template EnQue<T1>(biasUb);
            biasUb = this->biasInQueue_.template DeQue<T1>();
            Add(expandedXUb, expandedXUb, biasUb, foldFactor_ + foldTailFactor);
        }
        this->VfCalGradScale(expandedXUb, gradYUb, binAddParams, foldFactor_ + foldTailFactor, binaryAddCache,
                             this->cacheIdx_ % CACHE_BUFF_SIZE);
        this->expandedXInQueue_.FreeTensor(expandedXUb);
    } else {
        if constexpr (IsBiasExist) {
            this->biasInQueue_.template EnQue<T1>(biasUb);
            biasUb = this->biasInQueue_.template DeQue<T1>();
            this->VfCalGradScale(biasUb, gradYUb, binAddParams, foldFactor_ + foldTailFactor, binaryAddCache,
                                 this->cacheIdx_ % CACHE_BUFF_SIZE);
        }
    }
    if constexpr (IsBiasExist) {
        this->biasInQueue_.FreeTensor(biasUb);
    }
    this->cacheIdx_ = this->cacheIdx_ + 1;
    UpdateCache(this->cacheIdx_, binaryAddCache);
    this->gradYInQueue_.FreeTensor(gradYUb);

    // 这部分只有main部分，没有折叠部分
    mainOffset = (tilingData_->foldBlockNum + tilingData_->foldTailBlockNum) * foldFactor_;
    gradYUb = this->gradYInQueue_.template AllocTensor<T1>();
    DataCopy(gradYUb, this->gradYGm_[gmOffset + mainOffset], foldFactor_);
    this->gradYInQueue_.template EnQue<T1>(gradYUb);
    gradYUb = this->gradYInQueue_.template DeQue<T1>();
    if constexpr (IsBiasExist) {
        int64_t biasMainOffset = this->expertIdx_ * this->h_ + mainOffset;
        biasUb = this->biasInQueue_.template AllocTensor<T1>();
        DataCopy(biasUb, this->biasGm_[biasMainOffset], foldFactor_);
    }
    binAddParams = {tilingData_->foldBinAddParams.binaryAddQuotient, tilingData_->foldBinAddParams.binaryAddk,
                    tilingData_->foldBinAddParams.binaryAddLastNum};
    if (this->rowIdx_ >= 0 && this->rowIdx_ < tilingData_->baseParams.expandedXDim0) {
        int64_t xMainOffset = this->rowIdx_ * this->h_ + mainOffset;
        ProcessGradExpandedX(gradYUb, foldFactor_, 0, xMainOffset, 0);
        LocalTensor<T1> expandedXUb = this->expandedXInQueue_.template AllocTensor<T1>();
        DataCopy(expandedXUb, this->expandedXGm_[xMainOffset], foldFactor_);
        this->expandedXInQueue_.template EnQue<T1>(expandedXUb);
        expandedXUb = this->expandedXInQueue_.template DeQue<T1>();
        if constexpr (IsBiasExist) {
            this->biasInQueue_.template EnQue<T1>(biasUb);
            biasUb = this->biasInQueue_.template DeQue<T1>();
            Add(expandedXUb, expandedXUb, biasUb, foldFactor_);
        }
        this->VfCalGradScale(expandedXUb, gradYUb, binAddParams, foldFactor_, binaryAddCache,
                             this->cacheIdx_ % CACHE_BUFF_SIZE);
        this->expandedXInQueue_.FreeTensor(expandedXUb);
    } else {
        if constexpr (IsBiasExist) {
            this->biasInQueue_.template EnQue<T1>(biasUb);
            biasUb = this->biasInQueue_.template DeQue<T1>();
            this->VfCalGradScale(biasUb, gradYUb, binAddParams, foldFactor_, binaryAddCache,
                                 this->cacheIdx_ % CACHE_BUFF_SIZE);
        }
    }
    if constexpr (IsBiasExist) {
        this->biasInQueue_.FreeTensor(biasUb);
    }
    this->cacheIdx_ = this->cacheIdx_ + 1;
    UpdateCache(this->cacheIdx_, binaryAddCache);
    this->gradYInQueue_.FreeTensor(gradYUb);
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::NoOverLapPartProcess(
    const uint32_t batchIdx, const LocalTensor<float> &binaryAddCache)
{
    LocalTensor<T1> gradYUb;
    LocalTensor<T1> biasUb;
    BinAddParams binAddParams = {tilingData_->mainBinAddParams.binaryAddQuotient,
                                 tilingData_->mainBinAddParams.binaryAddk,
                                 tilingData_->mainBinAddParams.binaryAddLastNum};
    int64_t gmOffset = batchIdx / tilingData_->baseParams.topK * tilingData_->baseParams.hidden;
    int64_t mainIdxStart = tilingData_->foldBlockNum / 2 + (tilingData_->foldTailBlockNum > 0 ? 1 : 0);
    for (uint32_t mainIdx = mainIdxStart; mainIdx < tilingData_->mainBlockNum; mainIdx++) {
        int64_t mainOffset = mainFactor_ * mainIdx;
        gradYUb = this->gradYInQueue_.template AllocTensor<T1>();
        DataCopy(gradYUb, this->gradYGm_[gmOffset + mainOffset], mainFactor_);
        this->gradYInQueue_.template EnQue<T1>(gradYUb);
        gradYUb = this->gradYInQueue_.template DeQue<T1>();
        if constexpr (IsBiasExist) {
            int64_t biasMainOffset = this->expertIdx_ * this->h_ + mainOffset;
            biasUb = this->biasInQueue_.template AllocTensor<T1>();
            DataCopy(biasUb, this->biasGm_[biasMainOffset], mainFactor_);
        }
        if (this->rowIdx_ >= 0 && this->rowIdx_ < tilingData_->baseParams.expandedXDim0) {
            int64_t xMainOffset = this->rowIdx_ * this->h_ + mainOffset;
            ProcessGradExpandedX(gradYUb, mainFactor_, 0, xMainOffset, 0);
            LocalTensor<T1> expandedXUb = this->expandedXInQueue_.template AllocTensor<T1>();
            DataCopy(expandedXUb, this->expandedXGm_[xMainOffset], mainFactor_);
            this->expandedXInQueue_.template EnQue<T1>(expandedXUb);
            expandedXUb = this->expandedXInQueue_.template DeQue<T1>();
            if constexpr (IsBiasExist) {
                this->biasInQueue_.template EnQue<T1>(biasUb);
                biasUb = this->biasInQueue_.template DeQue<T1>();
                Add(expandedXUb, expandedXUb, biasUb, mainFactor_);
            }
            this->VfCalGradScale(expandedXUb, gradYUb, binAddParams, mainFactor_, binaryAddCache,
                                 this->cacheIdx_ % CACHE_BUFF_SIZE);
            this->expandedXInQueue_.FreeTensor(expandedXUb);
        } else {
            if constexpr (IsBiasExist) {
                this->biasInQueue_.template EnQue<T1>(biasUb);
                biasUb = this->biasInQueue_.template DeQue<T1>();
                this->VfCalGradScale(biasUb, gradYUb, binAddParams, mainFactor_, binaryAddCache,
                                     this->cacheIdx_ % CACHE_BUFF_SIZE);
            }
        }
        if constexpr (IsBiasExist) {
            this->biasInQueue_.FreeTensor(biasUb);
        }
        this->cacheIdx_ = this->cacheIdx_ + 1;
        UpdateCache(this->cacheIdx_, binaryAddCache);
        this->gradYInQueue_.FreeTensor(gradYUb);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::ProcessGradExpandedX(
    const LocalTensor<T1> &gradYUb, int64_t mainLen, int64_t foldLen, int64_t mainOutOffset, int64_t foldOutOffset)
{
    LocalTensor<T1> gradExpandedXUb = this->gradExpandedXOutQueue_.template AllocTensor<T1>();
    __ubuf__ T1 *gradExpandedXMem = (__ubuf__ T1 *)gradExpandedXUb.GetPhyAddr();
    __ubuf__ T1 *gradYMem = (__ubuf__ T1 *)gradYUb.GetPhyAddr();
    uint32_t count = mainLen + foldLen;
    uint16_t loopCount = (count + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> gradYReg;
        MicroAPI::MaskReg pregLoop;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = MicroAPI::UpdateMask<float>(count);
            // 拷贝到RegBase内
            ops::LoadOneTensorForDtypeT<T1>(gradYMem, gradYReg, pregLoop, i * VL_FLOAT32_SIZE);
            // 计算
            if constexpr (IsSameType<T3, bfloat16_t>::value) {
                Muls(gradYReg, gradYReg, ToFloat(this->scale_), pregLoop);
            } else {
                Muls(gradYReg, gradYReg, (float)this->scale_, pregLoop);
            }
            // 拷贝回UB
            ops::StoreOneTensorForDtypeT<T1>(gradExpandedXMem, gradYReg, pregLoop, i * VL_FLOAT32_SIZE);
        }
    }
    this->gradExpandedXOutQueue_.template EnQue<T1>(gradExpandedXUb);
    gradExpandedXUb = this->gradExpandedXOutQueue_.template DeQue<T1>();
    DataCopyExtParams copyExtParams{1, static_cast<uint16_t>(mainLen * sizeof(T1)), 0, 0, 0};
    DataCopyPad(this->gradExpandedXGm_[mainOutOffset], gradExpandedXUb, copyExtParams);
    if (foldLen != 0) {
        DataCopyExtParams foldCopyExtParams{1, static_cast<uint16_t>(foldLen * sizeof(T1)), 0, 0, 0};
        DataCopyPad(this->gradExpandedXGm_[foldOutOffset], gradExpandedXUb[foldFactor_], foldCopyExtParams);
    }
    this->gradExpandedXOutQueue_.FreeTensor(gradExpandedXUb);
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline bool MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::IsNeedUpdateLevel1Cache(
    const int64_t cacheIdx)
{
    return (cacheIdx % CACHE_BUFF_SIZE == 0) && (cacheIdx > 0);
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline bool MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::IsNeedUpdateLevel2Cache(
    const int64_t cacheIdx)
{
    int64_t level1Filled = cacheIdx / CACHE_BUFF_SIZE;
    return (level1Filled % CACHE_BUFF_SIZE == 0) && (level1Filled > 0);
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::UpdateCache(
    const int64_t cacheIdx, const LocalTensor<float> &binaryAddCache)
{
    if (IsNeedUpdateLevel1Cache(cacheIdx)) {
        int64_t level1Idx = (cacheIdx / CACHE_BUFF_SIZE) - 1;
        CacheReduceSum(binaryAddCache[CACHE_LEVEL1_IDX], binaryAddCache[CACHE_LEVEL0_IDX], level1Idx);
        Duplicate(binaryAddCache[CACHE_LEVEL0_IDX], 0.0f, CACHE_BUFF_SIZE);
    }
    if (IsNeedUpdateLevel2Cache(cacheIdx)) {
        int64_t level1Filled = cacheIdx / CACHE_BUFF_SIZE;
        int64_t level2Idx = (level1Filled / CACHE_BUFF_SIZE) - 1;
        CacheReduceSum(binaryAddCache[CACHE_LEVEL2_IDX], binaryAddCache[CACHE_LEVEL1_IDX], level2Idx);
        Duplicate(binaryAddCache[CACHE_LEVEL0_IDX], 0.0f, CACHE_BUFF_SIZE);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::CacheReduceSum(
    const LocalTensor<float> &dstTensor, const LocalTensor<float> &srcTensor, const uint32_t idx)
{
    __ubuf__ float *dst = (__ubuf__ float *)dstTensor.GetPhyAddr();
    __ubuf__ float *src = (__ubuf__ float *)srcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> x1;
        MicroAPI::RegTensor<float> x2;
        MicroAPI::RegTensor<float> x3;
        MicroAPI::RegTensor<float> x4;
        MicroAPI::RegTensor<float> sum1;
        MicroAPI::RegTensor<float> sum2;
        MicroAPI::RegTensor<float> sum12;
        MicroAPI::RegTensor<float> vlSum;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        LoadAlign(x1, src + 0 * VL_FLOAT32_SIZE);
        LoadAlign(x2, src + 1 * VL_FLOAT32_SIZE);
        LoadAlign(x3, src + 2 * VL_FLOAT32_SIZE);
        LoadAlign(x4, src + 3 * VL_FLOAT32_SIZE);
        Add(sum1, x1, x3, pregAll);
        Add(sum2, x2, x4, pregAll);
        Add(sum12, sum1, sum2, pregAll);
        Reg::Reduce<Reg::ReduceType::SUM>(vlSum, sum12, pregAll);
        MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dst + idx, vlSum, pregMerge);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegBaseCutH<T1, T2, T3, IsBiasExist>::ProcessGradScales(int64_t batchIdx)
{
    LocalTensor<float> gradScalesUb = this->gradScalesOutQueue_.template DeQue<float>();
    LocalTensor<T3> tempBuf = this->binaryAddSumBuf_.template Get<T3>(); // 复用binaryAddSumBuf_ 临时空间
    __ubuf__ float *srcAddr = (__ubuf__ float *)gradScalesUb.GetPhyAddr();
    __ubuf__ T3 *dstAddr = (__ubuf__ T3 *)tempBuf.GetPhyAddr();

    __VEC_SCOPE__
    {
        uint32_t sregMask = (uint32_t)1;
        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(sregMask);
        ;
        if constexpr (!IsSameType<T3, float>::value) {
            MicroAPI::RegTensor<T3> vregB16;
            MicroAPI::RegTensor<float> vregF32;
            LoadAlign(vregF32, srcAddr);
            Cast<T3, float, castTraitB322B16>(vregB16, vregF32, preg);
            StoreAlign<T3, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr, vregB16, preg);
        }
    }
    DataCopyExtParams copyExtParams{1, 1, 0, 0, 0};
    copyExtParams.blockLen = sizeof(T3);
    if constexpr (IsSameType<T3, float>::value) {
        DataCopyPad(this->gradScalesGm_[batchIdx], gradScalesUb, copyExtParams);
    } else {
        DataCopyPad(this->gradScalesGm_[batchIdx], tempBuf, copyExtParams);
    }
    this->gradScalesOutQueue_.FreeTensor(gradScalesUb);
}

} // namespace MoeFinalizeRoutingV2Grad
#endif // MOE_FINALIZE_ROUTING_V2_GRAD_REGBASE_CUT_H_H
