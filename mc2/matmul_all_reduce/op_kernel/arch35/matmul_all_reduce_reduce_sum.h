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
 * \file matmul_all_reduce_reduce_sum.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_REDUCE_SUM_H
#define MATMUL_ALL_REDUCE_REDUCE_SUM_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace MatmulAllReduceReduceSumImpl {
using namespace AscendC;

constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t UB_DATABLOCK = 32;
constexpr uint32_t SUM_TWO = 2;

template <class T>
class MatmulAllReduceReduceSum {
public:
    uint64_t xUbSize_ = 0;
    uint64_t zUbSize_ = 0;
    uint64_t maxCnt_ = 0;
    uint64_t tailDataCnt_ = 0;
    uint64_t totalNeedCore_ = 0;
    uint64_t tailNeedCore_ = 0;
    uint64_t loopCnt_ = 0;
    uint64_t totalDataCnt_ = 0;
    uint64_t rankDataCnt_ = 0;
    uint64_t coreNum_ = 0;
    uint64_t rankNum_;
    TPipe *pipe_;
    GM_ADDR dequantOut_;
    GM_ADDR output_;
    GlobalTensor<float> inputGM_;
    GlobalTensor<float> outputGM_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueZ_;

    __aicore__ inline MatmulAllReduceReduceSum() {}

    __aicore__ inline void Init(GM_ADDR dequantOut, GM_ADDR output, uint64_t dataCnt, uint64_t rankNum,
                                uint64_t coreNum, TPipe *tPipe)
    {
        this->coreNum_ = coreNum == 0 ? 1 : coreNum;
        this->rankNum_ = rankNum == 0 ? 1 : rankNum;
        uint64_t curUbSize = static_cast<uint64_t>(TOTAL_UB_SIZE);
        uint64_t ubDenom = (this->rankNum_ + 1) * DOUBLE_BUFFER;
        this->xUbSize_ = (curUbSize / ubDenom) * this->rankNum_;
        this->zUbSize_ = curUbSize / ubDenom;
        this->zUbSize_ = (this->zUbSize_ / UB_DATABLOCK) * UB_DATABLOCK;
        this->xUbSize_ = this->zUbSize_ * this->rankNum_;
        this->maxCnt_ = this->zUbSize_ / sizeof(float);
        this->rankDataCnt_ = dataCnt / this->rankNum_;
        this->totalNeedCore_ = Ceil(this->rankDataCnt_, this->maxCnt_);
        this->tailDataCnt_ = this->rankDataCnt_ % this->maxCnt_;
        this->loopCnt_ = Ceil(this->totalNeedCore_, this->coreNum_);
        this->tailNeedCore_ = this->totalNeedCore_ % this->coreNum_;
        this->totalDataCnt_ = dataCnt;
        this->dequantOut_ = dequantOut;
        this->output_ = output;
        this->pipe_ = tPipe;
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < this->loopCnt_; i++) {
            InnerProcess(i);
        }
    }

    __aicore__ inline void InnerProcess(uint64_t progress)
    {
        uint64_t calCnt = this->maxCnt_;
        if ((this->tailNeedCore_ != 0) && (progress == (this->loopCnt_ - 1))) {
            if (GetBlockIdx() >= this->tailNeedCore_) {
                return;
            }
            if ((this->tailDataCnt_ != 0) && (GetBlockIdx() == (this->tailNeedCore_ - 1))) {
                calCnt = this->tailDataCnt_;
            }
        } else if ((this->tailNeedCore_ == 0) && (progress == (this->loopCnt_ - 1))) {
            if ((this->tailDataCnt_ != 0) && (GetBlockIdx() == (this->coreNum_ - 1))) {
                calCnt = this->tailDataCnt_;
            }
        }
        pipe_->Reset();
        inputGM_.SetGlobalBuffer((__gm__ float *)dequantOut_, this->totalDataCnt_);
        outputGM_.SetGlobalBuffer((__gm__ float *)output_, this->rankDataCnt_);
        pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER, this->xUbSize_);
        pipe_->InitBuffer(outQueueZ_, DOUBLE_BUFFER, this->zUbSize_);
        ProcessSum(progress, calCnt);
    }

    __aicore__ inline void ProcessSum(uint64_t progress, uint64_t calCnt)
    {
        uint64_t gmOffset = GetBlockIdx() * this->maxCnt_ + progress * this->coreNum_ * this->maxCnt_;
        DataCopyPadExtParams<float> padParams = {false, 0, 0, 0};
        LocalTensor<float> inputLocal = inQueueX_.AllocTensor<float>();
        Duplicate(inputLocal, 0.0f, this->maxCnt_ * this->rankNum_);
        for (uint64_t i = 0; i < this->rankNum_; i++) {
            uint64_t localOffset = i * this->maxCnt_;
            uint64_t globalOffset = i * this->rankDataCnt_ + gmOffset;
            DataCopyExtParams GtLCopyParams = {1, static_cast<uint32_t>(calCnt * sizeof(float)), 0, 0, 0};
            DataCopyPad<float>(inputLocal[localOffset], inputGM_[globalOffset], GtLCopyParams, padParams);
        }
        inQueueX_.EnQue(inputLocal);
        Calculate(calCnt, gmOffset);
        inQueueX_.FreeTensor(inputLocal);
    }

    __aicore__ inline void Calculate(uint64_t calCnt, uint64_t gmOffset)
    {
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ_.AllocTensor<float>();
        uint64_t curTargets = this->rankNum_;
        for (uint64_t step = 1; step < this->rankNum_; step *= SUM_TWO) {
            if (((curTargets % SUM_TWO) != 0) && (curTargets > SUM_TWO)) {
                uint32_t leftOffset = static_cast<uint32_t>((curTargets - 2) * this->maxCnt_);
                uint32_t rightOffset = static_cast<uint32_t>((curTargets - 1) * this->maxCnt_);
                Add(xLocal[leftOffset], xLocal[leftOffset], xLocal[rightOffset], calCnt);
                curTargets -= 1;
            }
            uint64_t j = 0;
            for (uint64_t i = 0; i < curTargets; i += SUM_TWO) {
                uint32_t leftOffset = static_cast<uint32_t>(i * this->maxCnt_);
                uint32_t resultOffset = static_cast<uint32_t>(j * this->maxCnt_);
                uint32_t rightOffset = static_cast<uint32_t>((i + 1) * this->maxCnt_);
                if (curTargets == SUM_TWO) {
                    Add(zLocal, xLocal[leftOffset], xLocal[rightOffset], calCnt);
                } else {
                    Add(xLocal[resultOffset], xLocal[leftOffset], xLocal[rightOffset], calCnt);
                }
                j++;
            }
            curTargets = curTargets / SUM_TWO;
        }
        outQueueZ_.EnQue(zLocal);
        LocalTensor<float> outLocal = outQueueZ_.DeQue<float>();
        DataCopyExtParams LtGCopyParams = {1, static_cast<uint32_t>(calCnt * sizeof(float)), 0, 0, 0};
        DataCopyPad<float>(outputGM_[gmOffset], outLocal, LtGCopyParams);
        outQueueZ_.FreeTensor(zLocal);
    }
};

template <class T>
__aicore__ inline void MatmulAllReduceReduceSumKernel(GM_ADDR dequantOut, GM_ADDR output, uint32_t dataCnt,
                                                      uint32_t rankNum, TPipe *tPipe)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint64_t coreNum = GetBlockNum() * GetTaskRation();
    if (GetBlockIdx() >= coreNum) {
        return;
    }
    MatmulAllReduceReduceSum<T> op;
    op.Init(dequantOut, output, static_cast<uint64_t>(dataCnt), static_cast<uint64_t>(rankNum),
            static_cast<uint64_t>(coreNum), tPipe);
    op.Process();
}
} // namespace MatmulAllReduceReduceSumImpl
#endif //  MATMUL_ALL_REDUCE_REDUCE_SUM_H
