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
 * \file moe_finalize_routing_v2_grad_regbase_not_cut_h.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_GRAD_REGBASE_NOT_CUT_H_H
#define MOE_FINALIZE_ROUTING_V2_GRAD_REGBASE_NOT_CUT_H_H

#include "kernel_operator.h"
#include "moe_finalize_routing_v2_grad_regbase.h"

namespace MoeFinalizeRoutingV2Grad {
using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename T1, typename T2, typename T3, bool IsBiasExist = true>
class MoeFinalizeRoutingV2GradRegbaseNotCutH : MoeFinalizeRoutingV2GradRegbase<T1, T2, T3, IsBiasExist> {
public:
    __aicore__ inline MoeFinalizeRoutingV2GradRegbaseNotCutH(){};
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR scales,
                                GM_ADDR expertIdx, GM_ADDR bias, GM_ADDR gradExpandedX, GM_ADDR gradScales,
                                GM_ADDR workspace, const MoeFinalizeRoutingV2GradNotSplitHTilingData *tilingData,
                                TPipe *pipe);
    __aicore__ inline void Process();

private:
    // Init过程使用的内部函数
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR scales,
                                               GM_ADDR expertIdx, GM_ADDR bias, GM_ADDR gradExpandedX,
                                               GM_ADDR gradScales);
    __aicore__ inline void InitAllBuffer();
    // 计算预处理时使用的内部函数
    __aicore__ inline void InitOutputGm();
    __aicore__ inline void ComputeLoopParam();
    // 计算过程使用的内部函数
    __aicore__ inline void CopyIn(int64_t batchIdx, T2 rowIdx, int64_t hidden);
    __aicore__ inline void CopyInWithInvalidRowIdx(int64_t batchIdx,
                                                   int64_t hidden); // 对应行被drop、被activeNum截断的情况
    __aicore__ inline void ComputeOneRow(int64_t batchIdx, T2 rowIdx, int64_t hidden);
    __aicore__ inline void ComputeOneRowWithInvalidRowIdx(int64_t batchIdx, int64_t hidden);
    __aicore__ inline void CopyOutGradExpandedX(T2 rowIdx, int64_t hidden);
    // 计算过程使用的VF函数（Todo：替换为基类函数）
    __aicore__ inline void CalcGradExpandedXVF(__ubuf__ T1 *gradExpandedXUb, __ubuf__ T1 *gradYUb, T3 scale,
                                               uint32_t count);
    __aicore__ inline void CalcGradScaleVF(__ubuf__ T3 *gradScaleUb, __ubuf__ T1 *gradYUb, __ubuf__ T1 *expandedXUb,
                                           __ubuf__ T1 *biasUb, uint32_t count);
    __aicore__ inline void CalcGradScaleWithoutExpandedXVF(__ubuf__ T3 *gradScaleUb, __ubuf__ T1 *gradYUb,
                                                           __ubuf__ T1 *biasUb,
                                                           uint32_t count); // 针对expandedRowIdx无效的情况
    __aicore__ inline void CalcGradScaleForSmallHVF(__ubuf__ T3 *gradScaleUb, __ubuf__ T1 *gradYUb,
                                                    __ubuf__ T1 *expandedXUb, __ubuf__ T1 *biasUb, uint32_t count);
    __aicore__ inline void CalcGradScaleForSmallHWithoutExpandedXVF(__ubuf__ T3 *gradScaleUb, __ubuf__ T1 *gradYUb,
                                                                    __ubuf__ T1 *biasUb,
                                                                    uint32_t count); // 针对expandedRowIdx无效的情况
    __aicore__ inline void BinaryAddVF(__ubuf__ T3 *gradScaleUb, __ubuf__ float *binaryAddTmpAddr, RegTensor<float> &x1,
                                       RegTensor<float> &x2, uint16_t binaryAddKLoop, uint16_t binaryAddInnerLoop,
                                       MaskReg &pregAll, MaskReg &pregLastLoop, MaskReg &pregOne);

private:
    const MoeFinalizeRoutingV2GradBaseTilingData *baseParams_;
    const MoeFinalizeRoutingV2GradBinaryAddTilingData *binAddParams_;
    int64_t hAlign_ = 0;
    uint64_t binaryAddBufSize_;
};

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::Init(
    GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR bias,
    GM_ADDR gradExpandedX, GM_ADDR gradScales, GM_ADDR workspace,
    const MoeFinalizeRoutingV2GradNotSplitHTilingData *tilingData, TPipe *pipe)
{
    this->baseParams_ = &(tilingData->baseParams);
    this->binAddParams_ = &(tilingData->binAddParams);
    this->hAlign_ = tilingData->hAlign;
    this->InitCommon(pipe);
    this->InitAllGlobalBuffer(gradY, expandedRowIdx, expandedX, scales, expertIdx, bias, gradExpandedX, gradScales);
    if (this->baseParams_->dropPadMode == 1) {
        this->InitOutputGm();
        SyncAll();
    }
    this->InitAllBuffer();
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::Process()
{
    this->ComputeLoopParam();
    if (unlikely(this->endBatchIdx_ == 0 || this->startBatchIdx_ >= this->endBatchIdx_)) {
        // 该核无需处理计算，直接返回
        return;
    }
    for (int64_t batchIdx = this->startBatchIdx_; batchIdx < this->endBatchIdx_; batchIdx++) {
        T2 expandedRowIdx = this->expandedRowIdxGm_.GetValue(batchIdx);
        if (expandedRowIdx >= 0 && expandedRowIdx < baseParams_->expandedXDim0) {
            CopyIn(batchIdx, expandedRowIdx, this->baseParams_->hidden);
            ComputeOneRow(batchIdx, expandedRowIdx, this->baseParams_->hidden);
            CopyOutGradExpandedX(expandedRowIdx, this->baseParams_->hidden);
            this->CopyOutGradScales(batchIdx);
        } else {
            CopyInWithInvalidRowIdx(batchIdx, this->baseParams_->hidden);
            ComputeOneRowWithInvalidRowIdx(batchIdx, this->baseParams_->hidden);
            this->CopyOutGradScales(batchIdx);
        }
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::InitAllGlobalBuffer(
    GM_ADDR gradY, GM_ADDR expandedRowIdx, GM_ADDR expandedX, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR bias,
    GM_ADDR gradExpandedX, GM_ADDR gradScales)
{
    this->gradYGm_.SetGlobalBuffer((__gm__ T1 *)gradY);
    this->expandedRowIdxGm_.SetGlobalBuffer((__gm__ T2 *)expandedRowIdx);
    this->expandedXGm_.SetGlobalBuffer((__gm__ T1 *)expandedX);
    this->scalesGm_.SetGlobalBuffer((__gm__ T3 *)scales);
    this->expertIdxGm_.SetGlobalBuffer((__gm__ T2 *)expertIdx);
    this->biasGm_.SetGlobalBuffer((__gm__ T1 *)bias);
    this->gradExpandedXGm_.SetGlobalBuffer((__gm__ T1 *)gradExpandedX);
    this->gradScalesGm_.SetGlobalBuffer((__gm__ T3 *)gradScales);
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::InitAllBuffer()
{
    this->pipe_->InitBuffer(this->gradYInQueue_, DOUBLE_BUFFER, this->hAlign_ * sizeof(T1));
    this->pipe_->InitBuffer(this->expandedXInQueue_, DOUBLE_BUFFER, this->hAlign_ * sizeof(T1));
    if constexpr (IsBiasExist) {
        this->pipe_->InitBuffer(this->biasInQueue_, DOUBLE_BUFFER, this->hAlign_ * sizeof(T1));
    }
    this->pipe_->InitBuffer(this->gradExpandedXOutQueue_, DOUBLE_BUFFER, this->hAlign_ * sizeof(T1));
    this->pipe_->InitBuffer(this->gradScalesOutQueue_, DOUBLE_BUFFER, BLOCK_BYTE_SIZE);
    binaryAddBufSize_ = binAddParams_->binaryAddQuotient / VL_FLOAT32_SIZE;
    if (binaryAddBufSize_ > 0) {
        this->pipe_->InitBuffer(this->binaryAddSumBuf_, (binaryAddBufSize_ * sizeof(float) + BLOCK_BYTE_SIZE - 1) /
                                                            BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::InitOutputGm()
{
    if (this->blockIdx_ < baseParams_->initOutModCoreNum) {
        this->startBatchIdx_ = this->blockIdx_ * (baseParams_->initOutEachCoreBatchNum + 1);
        this->endBatchIdx_ = this->startBatchIdx_ + baseParams_->initOutEachCoreBatchNum + 1;
    } else if (this->blockIdx_ < baseParams_->initOutNeedCoreNum) {
        this->startBatchIdx_ = this->blockIdx_ * baseParams_->initOutEachCoreBatchNum + baseParams_->initOutModCoreNum;
        this->endBatchIdx_ = this->startBatchIdx_ + baseParams_->initOutEachCoreBatchNum;
    } else {
        return;
    }
    GlobalTensor<T1> initGm;
    for (int64_t batchIdx = this->startBatchIdx_; batchIdx < this->endBatchIdx_; batchIdx++) {
        initGm = this->gradExpandedXGm_[batchIdx * baseParams_->hidden];
        InitOutput<T1>(initGm, baseParams_->hidden, 0);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::ComputeLoopParam()
{
    if (this->blockIdx_ < baseParams_->computeModCoreNum) {
        this->startBatchIdx_ = this->blockIdx_ * (baseParams_->computeEachCoreBatchNum + 1);
        this->endBatchIdx_ = this->startBatchIdx_ + baseParams_->computeEachCoreBatchNum + 1;
    } else if (this->blockIdx_ < baseParams_->computeNeedCoreNum) {
        this->startBatchIdx_ = this->blockIdx_ * baseParams_->computeEachCoreBatchNum + baseParams_->computeModCoreNum;
        this->endBatchIdx_ = this->startBatchIdx_ + baseParams_->computeEachCoreBatchNum;
    } else {
        this->startBatchIdx_ = 0;
        this->endBatchIdx_ = 0;
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::CopyIn(int64_t batchIdx,
                                                                                               T2 rowIdx,
                                                                                               int64_t hidden)
{
    LocalTensor<T1> gradYUb = this->gradYInQueue_.template AllocTensor<T1>();
    DataCopyExtParams copyGradYExtParams{1, 1, 0, 0, 0};
    copyGradYExtParams.blockLen = hidden * sizeof(T1);
    DataCopyPadExtParams<T1> copyPadGradYExtparams{false, 0, 0, 0};
    DataCopyPad(gradYUb, this->gradYGm_[batchIdx / baseParams_->topK * baseParams_->hidden], copyGradYExtParams,
                copyPadGradYExtparams);
    this->gradYInQueue_.template EnQue(gradYUb);

    LocalTensor<T1> expandedXUb = this->expandedXInQueue_.template AllocTensor<T1>();
    DataCopyExtParams copyExpandedXExtParams{1, static_cast<uint32_t>(hidden * sizeof(T1)), 0, 0, 0};
    DataCopyPadExtParams<T1> copyPadExpandedXExtParams{false, 0, 0, 0};
    DataCopyPad(expandedXUb, this->expandedXGm_[rowIdx * hidden], copyExpandedXExtParams, copyPadExpandedXExtParams);
    this->expandedXInQueue_.template EnQue(expandedXUb);

    if constexpr (IsBiasExist) {
        LocalTensor<T1> biasUb = this->biasInQueue_.template AllocTensor<T1>();
        T2 expertIdx = this->expertIdxGm_.GetValue(batchIdx);
        DataCopyPad(biasUb, this->biasGm_[expertIdx * baseParams_->hidden], copyGradYExtParams, copyPadGradYExtparams);
        this->biasInQueue_.template EnQue(biasUb);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::CopyInWithInvalidRowIdx(
    int64_t batchIdx, int64_t hidden)
{
    if constexpr (IsBiasExist) {
        LocalTensor<T1> gradYUb = this->gradYInQueue_.template AllocTensor<T1>();
        DataCopyExtParams copyGradYExtParams{1, 1, 0, 0, 0};
        copyGradYExtParams.blockLen = hidden * sizeof(T1);
        DataCopyPadExtParams<T1> copyPadGradYExtparams{false, 0, 0, 0};
        DataCopyPad(gradYUb, this->gradYGm_[batchIdx / baseParams_->topK * baseParams_->hidden], copyGradYExtParams,
                    copyPadGradYExtparams);
        this->gradYInQueue_.template EnQue(gradYUb);
        LocalTensor<T1> biasUb = this->biasInQueue_.template AllocTensor<T1>();
        T2 expertIdx = this->expertIdxGm_.GetValue(batchIdx);
        DataCopyPad(biasUb, this->biasGm_[expertIdx * baseParams_->hidden], copyGradYExtParams, copyPadGradYExtparams);
        this->biasInQueue_.template EnQue(biasUb);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::ComputeOneRow(int64_t batchIdx,
                                                                                                      T2 rowIdx,
                                                                                                      int64_t hidden)
{
    LocalTensor<T1> gradYUb = this->gradYInQueue_.template DeQue<T1>();
    LocalTensor<T1> expandedXUb = this->expandedXInQueue_.template DeQue<T1>();
    LocalTensor<T1> biasUb;
    if constexpr (IsBiasExist) {
        biasUb = this->biasInQueue_.template DeQue<T1>();
    }
    // 计算gradExpandedX
    T3 scale = this->scalesGm_.GetValue(batchIdx);
    LocalTensor<T1> gradExpandedXUb = this->gradExpandedXOutQueue_.template AllocTensor<T1>();
    CalcGradExpandedXVF((__ubuf__ T1 *)gradExpandedXUb.GetPhyAddr(), (__ubuf__ T1 *)gradYUb.GetPhyAddr(), scale,
                        (uint32_t)hidden);
    this->gradExpandedXOutQueue_.template EnQue(gradExpandedXUb);
    // 计算gradScales
    LocalTensor<T3> gradScalesUb = this->gradScalesOutQueue_.template AllocTensor<T3>();
    if constexpr (IsBiasExist) {
        CalcGradScaleVF((__ubuf__ T3 *)gradScalesUb.GetPhyAddr(), (__ubuf__ T1 *)gradYUb.GetPhyAddr(),
                        (__ubuf__ T1 *)expandedXUb.GetPhyAddr(), (__ubuf__ T1 *)biasUb.GetPhyAddr(), (uint32_t)hidden);
    } else {
        CalcGradScaleVF((__ubuf__ T3 *)gradScalesUb.GetPhyAddr(), (__ubuf__ T1 *)gradYUb.GetPhyAddr(),
                        (__ubuf__ T1 *)expandedXUb.GetPhyAddr(), nullptr, (uint32_t)hidden);
    }
    this->gradScalesOutQueue_.template EnQue(gradScalesUb);
    // 释放资源
    this->expandedXInQueue_.FreeTensor(expandedXUb);
    this->gradYInQueue_.FreeTensor(gradYUb);
    if constexpr (IsBiasExist) {
        this->biasInQueue_.FreeTensor(biasUb);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::ComputeOneRowWithInvalidRowIdx(
    int64_t batchIdx, int64_t hidden)
{
    // 计算gradScales
    LocalTensor<T3> gradScalesUb = this->gradScalesOutQueue_.template AllocTensor<T3>();
    LocalTensor<T1> gradYUb;
    LocalTensor<T1> biasUb;
    if constexpr (IsBiasExist) {
        gradYUb = this->gradYInQueue_.template DeQue<T1>();
        biasUb = this->biasInQueue_.template DeQue<T1>();
        CalcGradScaleWithoutExpandedXVF((__ubuf__ T3 *)gradScalesUb.GetPhyAddr(), (__ubuf__ T1 *)gradYUb.GetPhyAddr(),
                                        (__ubuf__ T1 *)biasUb.GetPhyAddr(), (uint32_t)hidden);
        this->gradYInQueue_.FreeTensor(gradYUb);
        this->biasInQueue_.FreeTensor(biasUb);
    } else {
        Duplicate(gradScalesUb, static_cast<T3>(0.0f), 1);
    }
    this->gradScalesOutQueue_.template EnQue(gradScalesUb);
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::CopyOutGradExpandedX(
    T2 rowIdx, int64_t hidden)
{
    LocalTensor<T1> gradExpandedXUb = this->gradExpandedXOutQueue_.template DeQue<T1>();
    DataCopyExtParams copyExtParams{1, 1, 0, 0, 0};
    copyExtParams.blockLen = hidden * sizeof(T1);
    DataCopyPad(this->gradExpandedXGm_[rowIdx * baseParams_->hidden], gradExpandedXUb, copyExtParams);
    this->gradExpandedXOutQueue_.FreeTensor(gradExpandedXUb);
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::CalcGradExpandedXVF(
    __ubuf__ T1 *gradExpandedXUb, __ubuf__ T1 *gradYUb, T3 scale, uint32_t count)
{
    uint16_t loopCount = (count + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    __VEC_SCOPE__
    {
        RegTensor<float> gradYReg;
        MaskReg pregLoop;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(count);
            // 拷贝到RegBase内
            ops::LoadOneTensorForDtypeT<T1>(gradYUb, gradYReg, pregLoop, i * VL_FLOAT32_SIZE);
            // 计算
            if constexpr (IsSameType<T3, bfloat16_t>::value) {
                Muls(gradYReg, gradYReg, ToFloat(scale), pregLoop);
            } else {
                Muls(gradYReg, gradYReg, (float)scale, pregLoop);
            }
            // 拷贝回UB
            ops::StoreOneTensorForDtypeT<T1>(gradExpandedXUb, gradYReg, pregLoop, i * VL_FLOAT32_SIZE);
        }
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::CalcGradScaleVF(
    __ubuf__ T3 *gradScaleUb, __ubuf__ T1 *gradYUb, __ubuf__ T1 *expandedXUb, __ubuf__ T1 *biasUb, uint32_t count)
{
    uint16_t loopCount = (count + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    if (loopCount == 1) {
        CalcGradScaleForSmallHVF(gradScaleUb, gradYUb, expandedXUb, biasUb, count);
        return;
    }
    __ubuf__ float *binaryAddUb = (__ubuf__ float *)this->binaryAddSumBuf_.template Get<float>().GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> gradYReg;
        RegTensor<float> expandedXReg;
        RegTensor<float> biasReg;
        RegTensor<float> gradYTailReg;
        RegTensor<float> expandedXTailReg;
        RegTensor<float> biasTailReg;
        RegTensor<float> sumReg;

        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();

        uint32_t mainCount = (uint32_t)binAddParams_->binaryAddQuotient;
        uint32_t tailCount = count - mainCount;
        MaskReg pregLoop;
        MaskReg pregLoopTail;

        uint16_t tailLoopNum = loopCount - binaryAddBufSize_;
        // 前tailLoopNum次循环，同时处理当前块和对应尾块，然后累加再拷出
        for (uint16_t i = 0; i < tailLoopNum; i++) {
            pregLoop = UpdateMask<float>(mainCount);
            pregLoopTail = UpdateMask<float>(tailCount);
            // 拷贝输入到RegBase内
            ops::LoadOneTensorForDtypeT<T1>(gradYUb, gradYReg, pregLoop, i * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<T1>(expandedXUb, expandedXReg, pregLoop, i * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<T1>(gradYUb, gradYTailReg, pregLoopTail,
                                            (i + binaryAddBufSize_) * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<T1>(expandedXUb, expandedXTailReg, pregLoopTail,
                                            (i + binaryAddBufSize_) * VL_FLOAT32_SIZE);
            if constexpr (IsBiasExist) {
                ops::LoadOneTensorForDtypeT<T1>(biasUb, biasReg, pregLoop, i * VL_FLOAT32_SIZE);
                Add(expandedXReg, biasReg, expandedXReg, pregLoop);
                ops::LoadOneTensorForDtypeT<T1>(biasUb, biasTailReg, pregLoopTail,
                                                (i + binaryAddBufSize_) * VL_FLOAT32_SIZE);
                Add(expandedXTailReg, biasTailReg, expandedXTailReg, pregLoopTail);
            }
            // 计算
            Mul(gradYReg, expandedXReg, gradYReg, pregLoop);
            Mul(gradYTailReg, expandedXTailReg, gradYTailReg, pregLoopTail);
            Add(gradYReg, gradYReg, gradYTailReg, pregAll);
            Reg::Reduce<Reg::ReduceType::SUM>(sumReg, gradYReg, pregAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(binaryAddUb + i, sumReg, pregOne);
        }
        // 处理剩余循环
        for (uint16_t i = tailLoopNum; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(mainCount);
            // 拷贝输入到RegBase内
            ops::LoadOneTensorForDtypeT<T1>(gradYUb, gradYReg, pregLoop, i * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<T1>(expandedXUb, expandedXReg, pregLoop, i * VL_FLOAT32_SIZE);
            if constexpr (IsBiasExist) {
                ops::LoadOneTensorForDtypeT<T1>(biasUb, biasReg, pregLoop, i * VL_FLOAT32_SIZE);
                Add(expandedXReg, biasReg, expandedXReg, pregLoop);
            }
            // 计算
            Mul(gradYReg, expandedXReg, gradYReg, pregLoop);
            Reg::Reduce<Reg::ReduceType::SUM>(sumReg, gradYReg, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(binaryAddUb + i, sumReg, pregOne);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        // 计算最终结果并拷贝
        count = (uint32_t)binAddParams_->binaryAddLastNum;
        pregLoop = UpdateMask<float>(count);
        BinaryAddVF(gradScaleUb, binaryAddUb, sumReg, gradYReg, binAddParams_->binaryAddk,
                    binaryAddBufSize_ / VL_FLOAT32_SIZE, pregAll, pregLoop, pregOne);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::CalcGradScaleWithoutExpandedXVF(
    __ubuf__ T3 *gradScaleUb, __ubuf__ T1 *gradYUb, __ubuf__ T1 *biasUb, uint32_t count)
{
    uint16_t loopCount = (count + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    if (loopCount == 1) {
        CalcGradScaleForSmallHWithoutExpandedXVF(gradScaleUb, gradYUb, biasUb, count);
        return;
    }
    __ubuf__ float *binaryAddUb = (__ubuf__ float *)this->binaryAddSumBuf_.template Get<float>().GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> gradYReg;
        RegTensor<float> biasReg;
        RegTensor<float> gradYTailReg;
        RegTensor<float> biasTailReg;
        RegTensor<float> sumReg;

        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();

        uint32_t mainCount = (uint32_t)binAddParams_->binaryAddQuotient;
        uint32_t tailCount = count - mainCount;
        MaskReg pregLoop;
        MaskReg pregLoopTail;

        uint16_t tailLoopNum = loopCount - binaryAddBufSize_;
        // 前tailLoopNum次循环，同时处理当前块和对应尾块，然后累加再拷出
        for (uint16_t i = 0; i < tailLoopNum; i++) {
            pregLoop = UpdateMask<float>(mainCount);
            pregLoopTail = UpdateMask<float>(tailCount);
            // 拷贝输入到RegBase内
            ops::LoadOneTensorForDtypeT<T1>(gradYUb, gradYReg, pregLoop, i * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<T1>(gradYUb, gradYTailReg, pregLoopTail,
                                            (i + binaryAddBufSize_) * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<T1>(biasUb, biasReg, pregLoop, i * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<T1>(biasUb, biasTailReg, pregLoopTail,
                                            (i + binaryAddBufSize_) * VL_FLOAT32_SIZE);
            // 计算
            Mul(gradYReg, biasReg, gradYReg, pregLoop);
            Mul(gradYTailReg, biasTailReg, gradYTailReg, pregLoopTail);
            Add(gradYReg, gradYReg, gradYTailReg, pregAll);
            Reg::Reduce<Reg::ReduceType::SUM>(sumReg, gradYReg, pregAll);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(binaryAddUb + i, sumReg, pregOne);
        }
        // 处理剩余循环
        for (uint16_t i = tailLoopNum; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(mainCount);
            // 拷贝输入到RegBase内
            ops::LoadOneTensorForDtypeT<T1>(gradYUb, gradYReg, pregLoop, i * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<T1>(biasUb, biasReg, pregLoop, i * VL_FLOAT32_SIZE);
            // 计算
            Mul(gradYReg, biasReg, gradYReg, pregLoop);
            Reg::Reduce<Reg::ReduceType::SUM>(sumReg, gradYReg, pregLoop);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(binaryAddUb + i, sumReg, pregOne);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        // 计算最终结果并拷贝
        count = (uint32_t)binAddParams_->binaryAddLastNum;
        pregLoop = UpdateMask<float>(count);
        BinaryAddVF(gradScaleUb, binaryAddUb, sumReg, gradYReg, binAddParams_->binaryAddk,
                    binaryAddBufSize_ / VL_FLOAT32_SIZE, pregAll, pregLoop, pregOne);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::CalcGradScaleForSmallHVF(
    __ubuf__ T3 *gradScaleUb, __ubuf__ T1 *gradYUb, __ubuf__ T1 *expandedXUb, __ubuf__ T1 *biasUb, uint32_t count)
{
    __VEC_SCOPE__
    {
        RegTensor<float> gradYReg;
        RegTensor<float> expandedXReg;
        RegTensor<float> biasReg;
        RegTensor<float> sumReg;
        MaskReg pregLoop;
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        {
            pregLoop = UpdateMask<float>(count);
            // 拷贝到RegBase内
            ops::LoadOneTensorForDtypeT<T1>(gradYUb, gradYReg, pregLoop, 0);
            ops::LoadOneTensorForDtypeT<T1>(expandedXUb, expandedXReg, pregLoop, 0);
            if constexpr (IsBiasExist) {
                ops::LoadOneTensorForDtypeT<T1>(biasUb, biasReg, pregLoop, 0);
                Add(expandedXReg, biasReg, expandedXReg, pregLoop);
            }
            // 计算
            Mul(gradYReg, expandedXReg, gradYReg, pregLoop);
            Reg::Reduce<Reg::ReduceType::SUM>(sumReg, gradYReg, pregLoop);
            ops::StoreOneTensorForDtypeT<T3>(gradScaleUb, sumReg, pregOne, 0);
        }
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void
MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::CalcGradScaleForSmallHWithoutExpandedXVF(
    __ubuf__ T3 *gradScaleUb, __ubuf__ T1 *gradYUb, __ubuf__ T1 *biasUb, uint32_t count)
{
    __VEC_SCOPE__
    {
        RegTensor<float> gradYReg;
        RegTensor<float> biasReg;
        RegTensor<float> sumReg;
        MaskReg pregLoop;
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        pregLoop = UpdateMask<float>(count);
        // 拷贝到RegBase内
        ops::LoadOneTensorForDtypeT<T1>(gradYUb, gradYReg, pregLoop, 0);
        ops::LoadOneTensorForDtypeT<T1>(biasUb, biasReg, pregLoop, 0);
        // 计算
        Mul(gradYReg, biasReg, gradYReg, pregLoop);
        Reg::Reduce<Reg::ReduceType::SUM>(sumReg, gradYReg, pregLoop);
        // 拷贝出结果
        ops::StoreOneTensorForDtypeT<T3>(gradScaleUb, sumReg, pregOne, 0);
    }
}

template <typename T1, typename T2, typename T3, bool IsBiasExist>
__aicore__ inline void MoeFinalizeRoutingV2GradRegbaseNotCutH<T1, T2, T3, IsBiasExist>::BinaryAddVF(
    __ubuf__ T3 *gradScaleUb, __ubuf__ float *binaryAddTmpAddr, RegTensor<float> &x1, RegTensor<float> &x2,
    uint16_t binaryAddKLoop, uint16_t binaryAddInnerLoop, MaskReg &pregAll, MaskReg &pregLastLoop, MaskReg &pregOne)
{
    uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
    for (uint16_t i = 0; i < binaryAddKLoop; i++) {
        curBinaryAddInnerLoop = curBinaryAddInnerLoop / BINARY_ADD_COF;
        for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
            ops::LoadOneTensorForDtypeT<float>(binaryAddTmpAddr, x1, pregAll, j * VL_FLOAT32_SIZE);
            ops::LoadOneTensorForDtypeT<float>(binaryAddTmpAddr, x2, pregAll,
                                               (j + curBinaryAddInnerLoop) * VL_FLOAT32_SIZE);
            Add(x1, x1, x2, pregAll);
            ops::StoreOneTensorForDtypeT<float>(binaryAddTmpAddr, x1, pregAll, j * VL_FLOAT32_SIZE);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    ops::LoadOneTensorForDtypeT<float>(binaryAddTmpAddr, x1, pregLastLoop, 0);
    Reg::Reduce<Reg::ReduceType::SUM>(x2, x1, pregLastLoop);
    ops::StoreOneTensorForDtypeT<T3>(gradScaleUb, x2, pregOne, 0);
}
} // namespace MoeFinalizeRoutingV2Grad

#endif // MOE_FINALIZE_ROUTING_V2_GRAD_REGBASE_NOT_CUT_H_H
