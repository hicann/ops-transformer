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
 * \file matmul_all_reduce_element_wise_add.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_ELEMENT_WISE_ADD_H
#define MATMUL_ALL_REDUCE_ELEMENT_WISE_ADD_H

namespace AscendC {

template <typename xDtype, typename yDtype>
class MatmulAllReduceElementWiseAdd {
public:
    __aicore__ inline MatmulAllReduceElementWiseAdd() {}

    __aicore__ inline void Init(GM_ADDR left, GM_ADDR right, uint64_t totalCnt, uint64_t tileCnt, TPipe *tPipe,
                                uint32_t coreNum)
    {
        pipe_ = tPipe;
        this->blockCnt_ = totalCnt / coreNum;
        uint64_t blockAddr = this->blockCnt_ * GetBlockIdx();
        if ((coreNum - 1) == GetBlockIdx()) {
            this->blockCnt_ = totalCnt - this->blockCnt_ * GetBlockIdx();
        }
        this->tileNum_ = Ceil(this->blockCnt_, tileCnt);
        this->tileCnt_ = tileCnt;
        mmOutGm_.SetGlobalBuffer((__gm__ xDtype *)left + blockAddr, this->blockCnt_);
        addGm_.SetGlobalBuffer((__gm__ yDtype *)right + blockAddr, this->blockCnt_);
        pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER, tileCnt * sizeof(xDtype));
        pipe_->InitBuffer(inQueueY_, DOUBLE_BUFFER, tileCnt * sizeof(yDtype));
        pipe_->InitBuffer(outQueueZ_, DOUBLE_BUFFER, tileCnt * sizeof(xDtype));
        if constexpr (X_NEED_INCREASE_ACCURACY) {
            pipe_->InitBuffer(tempQueOutFp32_, tileCnt * sizeof(float));
        }
        if constexpr (Y_NEED_INCREASE_ACCURACY) {
            pipe_->InitBuffer(tempQueAddFp32_, tileCnt * sizeof(float));
        }
    }

    __aicore__ inline void Process(uint64_t progress)
    {
        if (this->blockCnt_ == 0) {
            return;
        }
        uint64_t calcCnt =
            (progress == (this->tileNum_ - 1)) ? (this->blockCnt_ - progress * this->tileCnt_) : this->tileCnt_;
        DataCopyParams copyParamsX = {1, static_cast<uint16_t>(calcCnt * sizeof(xDtype)), 0, 0};
        DataCopyParams copyParamsY = {1, static_cast<uint16_t>(calcCnt * sizeof(yDtype)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, 0};
        LocalTensor<xDtype> mmOutLocal = inQueueX_.AllocTensor<xDtype>();
        DataCopyPad(mmOutLocal, mmOutGm_[progress * this->tileCnt_], copyParamsX, padParams);
        inQueueX_.EnQue(mmOutLocal);
        LocalTensor<yDtype> addLocal = inQueueY_.AllocTensor<yDtype>();
        DataCopyPad(addLocal, addGm_[progress * this->tileCnt_], copyParamsY, padParams);
        inQueueY_.EnQue(addLocal);
        LocalTensor<xDtype> xLocal = inQueueX_.DeQue<xDtype>();
        LocalTensor<yDtype> yLocal = inQueueY_.DeQue<yDtype>();
        LocalTensor<xDtype> zLocal = outQueueZ_.AllocTensor<xDtype>();
        if constexpr (NEED_INCREASE_ACCURACY) {
            auto outFp32LocalTmp =
                X_NEED_INCREASE_ACCURACY ? tempQueOutFp32_.Get<float>() : xLocal.template ReinterpretCast<float>();
            auto addFp32LocalTmp =
                Y_NEED_INCREASE_ACCURACY ? tempQueAddFp32_.Get<float>() : yLocal.template ReinterpretCast<float>();
            if constexpr (X_NEED_INCREASE_ACCURACY) {
                Cast(outFp32LocalTmp, xLocal, RoundMode::CAST_NONE, calcCnt);
            }
            if constexpr (Y_NEED_INCREASE_ACCURACY) {
                Cast(addFp32LocalTmp, yLocal, RoundMode::CAST_NONE, calcCnt);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (X_NEED_INCREASE_ACCURACY) {
                Add(outFp32LocalTmp, outFp32LocalTmp, addFp32LocalTmp, calcCnt);
                Cast(zLocal, outFp32LocalTmp, RoundMode::CAST_RINT, calcCnt);
            } else {
                Add(zLocal, outFp32LocalTmp.template ReinterpretCast<xDtype>(),
                    addFp32LocalTmp.template ReinterpretCast<xDtype>(), calcCnt);
            }
        } else if (std::is_same<xDtype, yDtype>::value) {
            Add(zLocal, xLocal, yLocal.template ReinterpretCast<xDtype>(), calcCnt);
        }
        outQueueZ_.EnQue<xDtype>(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        LocalTensor<xDtype> outLocal = outQueueZ_.DeQue<xDtype>();
        DataCopyPad(mmOutGm_[progress * this->tileCnt_], outLocal, copyParamsX);
        outQueueZ_.FreeTensor(outLocal);
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // 如果add的两路精度不一致，需要升精度累加后cast回去
    static constexpr bool NEED_INCREASE_ACCURACY = !std::is_same<xDtype, yDtype>::value;
    static constexpr bool X_NEED_INCREASE_ACCURACY = NEED_INCREASE_ACCURACY && (!std::is_same<xDtype, float>::value);
    static constexpr bool Y_NEED_INCREASE_ACCURACY = NEED_INCREASE_ACCURACY && (!std::is_same<yDtype, float>::value);
#else
    // arch22芯片为未支持bfp16直接add，需要升精度运算后cast回去
    static constexpr bool NEED_INCREASE_ACCURACY = std::is_same<xDtype, bfloat16_t>::value;
    static constexpr bool X_NEED_INCREASE_ACCURACY = NEED_INCREASE_ACCURACY;
    static constexpr bool Y_NEED_INCREASE_ACCURACY = NEED_INCREASE_ACCURACY && (!std::is_same<yDtype, float>::value);
#endif

    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueY_;
    TQue<QuePosition::VECOUT, 1> outQueueZ_;
    TBuf<TPosition::VECCALC> tempQueOutFp32_;
    TBuf<TPosition::VECCALC> tempQueAddFp32_;
    GlobalTensor<xDtype> mmOutGm_;
    GlobalTensor<yDtype> addGm_;
    uint64_t blockCnt_;
    uint64_t tileNum_;
    uint64_t tileCnt_;
};

template <typename xDtype, typename yDtype = xDtype>
__aicore__ inline void MatmulAllReduceElementWiseAddKernel(GM_ADDR x, GM_ADDR y, uint64_t totalCnt, uint64_t tileCnt,
                                                           TPipe *tPipe)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t coreNum = GetBlockNum() * GetTaskRation();
    if (GetBlockIdx() >= coreNum) {
        return;
    }
    tPipe->Reset();
    MatmulAllReduceElementWiseAdd<xDtype, yDtype> op;
    op.Init(x, y, totalCnt, tileCnt, tPipe, coreNum);
    for (uint64_t i = 0; i < op.tileNum_; i++) {
        op.Process(i);
    }
}
} // namespace AscendC
#endif // MATMUL_ALL_REDUCE_ELEMENT_WISE_ADD_H
