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
 * \file moe_finalize_routing_fp.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_FP
#define MOE_FINALIZE_ROUTING_FP

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRouting {

using namespace AscendC;

template <typename T>
class MoeFinalizeRoutingFP {
public:
    __aicore__ inline MoeFinalizeRoutingFP(){};
    __aicore__ inline void Init(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expandedSrcToDstRow, GM_ADDR expertForSourceRow, GM_ADDR out,
                                GM_ADDR workspace, const MoeFinalizeRoutingTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const MoeFinalizeRoutingTilingData *tilingData);
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t Int32AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessInt32(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> skip1Queue_;
    TQue<QuePosition::VECIN, 1> skip2Queue_;
    TQue<QuePosition::VECIN, 1> scalesQueue_;
    TQue<QuePosition::VECIN, 1> expertForSourceRowQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> expandedPermutedRowsBuf_;
    TBuf<QuePosition::VECCALC> biasBuf_;

    GlobalTensor<T> gmExpandedPermutedRows_;
    GlobalTensor<T> gmSkip1_;
    GlobalTensor<T> gmSkip2_;
    GlobalTensor<T> gmBias_;
    GlobalTensor<T> gmScales_;
    GlobalTensor<int32_t> gmExpandedSrcToDstRow_;
    GlobalTensor<int32_t> gmExpertForSourceRow_;
    GlobalTensor<T> gmOut_;

    // base param
    int64_t BS_{0};
    int64_t H_{0};
    int64_t K_{0};
    int64_t E_{0};

    // Judge skip2 if null
    int64_t skip2IsNull_{1};

    // normal core
    int64_t normalCoreHandleNum_{0};
    int64_t normalCoreLoopNum_{0};
    int64_t normalCoreHandleNumPerLoop_{0};
    int64_t normalCoreHandleNumTailLoop_{0};

    // tail core
    int64_t tailCoreHandleNum_{0};
    int64_t tailCoreLoopNum_{0};
    int64_t tailCoreHandleNumPerLoop_{0};
    int64_t tailCoreHandleNumTailLoop_{0};

    // used core
    int64_t totalCoreNum_{0};
    int64_t usedCoreNum_{0};
    int64_t workspace_{0};

    // input number
    int64_t inputSkipIdx_{0};
    int64_t inputScalesAndExpertIdx_{0};
    int64_t inputExpandedSrcToDstRowIdx_{0};
    int64_t outputIdx_{0};
    int64_t curCoreHandleNumPerLoop_{0};

    const int64_t ONE_BLK_SIZE{32};
    const int64_t ONCE_ALGN_NUM_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadH_ = false;
    bool isPadK_ = false;
    bool isPadKInt32_ = false;
    uint16_t rightPaddingH_ = 0;
    uint16_t rightPaddingK_ = 0;
    uint16_t rightPaddingKInt32_ = 0;

    const int64_t ONCE_ALGN_NUM_INT32{8};
    const int64_t INT32_BYTES{4};
    const int64_t BUFFER_NUM{1};
};

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFP<T>::AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_ - 1) / ONCE_ALGN_NUM_ * ONCE_ALGN_NUM_;
}

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFP<T>::PadProcessT(int64_t param)
{
    return ONCE_ALGN_NUM_ - param % ONCE_ALGN_NUM_;
}

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFP<T>::PadProcessInt32(int64_t param)
{
    return ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32;
}

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFP<T>::Int32AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT32 - 1) / ONCE_ALGN_NUM_INT32 * ONCE_ALGN_NUM_INT32;
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFP<T>::ParseTilingData(const MoeFinalizeRoutingTilingData *tilingData)
{
    skip2IsNull_ = tilingData->skip2IsNull;
    BS_ = tilingData->totalRowNum;
    H_ = tilingData->H;
    K_ = tilingData->K;
    E_ = tilingData->biasRowNum;

    // 非尾核
    normalCoreHandleNum_ = tilingData->normalCoreHandleNum;
    normalCoreLoopNum_ = tilingData->normalCoreLoopNum;
    normalCoreHandleNumPerLoop_ = tilingData->normalCoreHandleNumPerLoop;
    normalCoreHandleNumTailLoop_ = tilingData->normalCoreHandleNumTailLoop;

    // 尾核
    tailCoreHandleNum_ = tilingData->tailCoreHandleNum;
    tailCoreLoopNum_ = tilingData->tailCoreLoopNum;
    tailCoreHandleNumPerLoop_ = tilingData->tailCoreHandleNumPerLoop;
    tailCoreHandleNumTailLoop_ = tilingData->tailCoreHandleNumTailLoop;

    // 使用核数信息
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFP<T>::Init(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2,
                                                     GM_ADDR bias, GM_ADDR scales, GM_ADDR expandedSrcToDstRow,
                                                     GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace,
                                                     const MoeFinalizeRoutingTilingData *tilingData)
{
    // init tiling data
    ParseTilingData(tilingData);

    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreHandleNumPerLoop_ = tailCoreHandleNumPerLoop_;
    } else {
        curCoreHandleNumPerLoop_ = normalCoreHandleNumPerLoop_;
    }

    if (H_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadH_ = true;
        rightPaddingH_ = PadProcessT(H_);
    }

    if (K_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadK_ = true;
        rightPaddingK_ = PadProcessT(K_);
    }

    if (K_ * INT32_BYTES % ONE_BLK_SIZE) {
        isPadKInt32_ = true;
        rightPaddingKInt32_ = PadProcessInt32(K_);
    }

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = GetBlockIdx() * normalCoreHandleNum_ * H_;
    gmSkip1_.SetGlobalBuffer((__gm__ T *)skip1 + inputSkipIdx_, normalCoreHandleNum_ * H_);
    pipe.InitBuffer(skip1Queue_, 1, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(T));
    if (skip2IsNull_ == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T *)skip2 + inputSkipIdx_, normalCoreHandleNum_ * H_);
        pipe.InitBuffer(skip2Queue_, 1, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(T));
    }

    inputScalesAndExpertIdx_ = GetBlockIdx() * normalCoreHandleNum_ * K_;
    gmScales_.SetGlobalBuffer((__gm__ T *)scales + inputScalesAndExpertIdx_, normalCoreHandleNum_ * K_);
    gmExpertForSourceRow_.SetGlobalBuffer((__gm__ int32_t *)expertForSourceRow + inputScalesAndExpertIdx_,
                                          normalCoreHandleNum_ * K_);

    gmExpandedSrcToDstRow_.SetGlobalBuffer((__gm__ int32_t *)expandedSrcToDstRow, BS_ * K_);

    outputIdx_ = GetBlockIdx() * normalCoreHandleNum_ * H_;
    gmOut_.SetGlobalBuffer((__gm__ T *)out + outputIdx_, normalCoreHandleNum_ * H_);

    gmExpandedPermutedRows_.SetGlobalBuffer((__gm__ T *)expandedPermutedRows, BS_ * K_ * H_);
    gmBias_.SetGlobalBuffer((__gm__ T *)bias, E_ * H_);

    // 申请 buffer 空间
    pipe.InitBuffer(scalesQueue_, BUFFER_NUM, normalCoreHandleNumPerLoop_ * AlignmentProcess(K_) * sizeof(T));
    pipe.InitBuffer(expertForSourceRowQueue_, BUFFER_NUM,
                    normalCoreHandleNumPerLoop_ * Int32AlignmentProcess(K_) * sizeof(int32_t));
    pipe.InitBuffer(outQueue_, BUFFER_NUM, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(T));

    pipe.InitBuffer(expandedPermutedRowsBuf_, AlignmentProcess(H_) * sizeof(T));
    pipe.InitBuffer(biasBuf_, AlignmentProcess(H_) * sizeof(T));
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFP<T>::CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<T> scalesLocal = scalesQueue_.AllocTensor<T>();
    LocalTensor<int32_t> expertForSourceRowLocal = expertForSourceRowQueue_.AllocTensor<int32_t>();

    LocalTensor<T> skip1Local = skip1Queue_.AllocTensor<T>();
    LocalTensor<T> skip2Local;
    if (skip2IsNull_ == 0) {
        skip2Local = skip2Queue_.AllocTensor<T>();
    }

    // ---------------------------- [Skip] -------------------------------
    DataCopyParams copyParamsNormalSkip{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(H_ * sizeof(T)), 0,
                                        0};
    DataCopyPadParams padParamsNormalSkip{isPadH_, 0, static_cast<uint8_t>(rightPaddingH_), 0};

    DataCopyPad(skip1Local, gmSkip1_[nLoopIdx * curCoreHandleNumPerLoop_ * H_], copyParamsNormalSkip,
                padParamsNormalSkip);
    if (skip2IsNull_ == 0) {
        DataCopyPad(skip2Local, gmSkip2_[nLoopIdx * curCoreHandleNumPerLoop_ * H_], copyParamsNormalSkip,
                    padParamsNormalSkip);
    }

    // ---------------------------- [Scales] -------------------------------
    DataCopyParams copyParamsNormalScales{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(K_ * sizeof(T)),
                                          0, 0};
    DataCopyPadParams padParamsNormalScales{isPadK_, 0, static_cast<uint8_t>(rightPaddingK_), 0};
    DataCopyPad(scalesLocal, gmScales_[nLoopIdx * curCoreHandleNumPerLoop_ * K_], copyParamsNormalScales,
                padParamsNormalScales);

    // ---------------------------- [Expert] -------------------------------
    DataCopyParams copyParamsNormalExpert{static_cast<uint16_t>(curRepeatTimes),
                                          static_cast<uint16_t>(K_ * sizeof(int32_t)), 0, 0};
    DataCopyPadParams padParamsNormalExpert{isPadKInt32_, 0, static_cast<uint8_t>(rightPaddingKInt32_), 0};
    DataCopyPad(expertForSourceRowLocal, gmExpertForSourceRow_[nLoopIdx * curCoreHandleNumPerLoop_ * K_],
                copyParamsNormalExpert, padParamsNormalExpert);

    if (skip2IsNull_ == 0) {
        skip2Queue_.EnQue(skip2Local);
    }
    skip1Queue_.EnQue(skip1Local);
    scalesQueue_.EnQue(scalesLocal);
    expertForSourceRowQueue_.EnQue(expertForSourceRowLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFP<T>::Compute(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<int32_t> expertForSourceRowLocal = expertForSourceRowQueue_.DeQue<int32_t>();
    LocalTensor<T> scalesLocal = scalesQueue_.DeQue<T>();
    LocalTensor<T> skip1Local = skip1Queue_.DeQue<T>();
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    LocalTensor<T> skip2Local;
    if (skip2IsNull_ == 0) {
        skip2Local = skip2Queue_.DeQue<T>();
        Add(outLocal, skip1Local, skip2Local, curRepeatTimes * AlignmentProcess(H_));
    } else {
        Adds(outLocal, skip1Local, (T)0, curRepeatTimes * AlignmentProcess(H_));
    }

    LocalTensor<T> expandedPermutedTmpUb = expandedPermutedRowsBuf_.Get<T>();
    LocalTensor<T> biasTmpUb = biasBuf_.Get<T>();

    for (int64_t i = 0; i < curRepeatTimes; i++) {
        int64_t outRowIndex = i * AlignmentProcess(H_);
        for (int64_t j = 0; j < K_; j++) {
            int64_t expandedSrcToDstRowIndex =
                nLoopIdx * curCoreHandleNumPerLoop_ + i + j * BS_ + GetBlockIdx() * normalCoreHandleNum_;
            int64_t expandedPermutedRowsIndex = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndex);

            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            int64_t biasIndex = expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(K_) + j);

            SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
            T scalesVal = scalesLocal.GetValue(i * AlignmentProcess(K_) + j);

            // expandedPermutedRows的row拷贝
            DataCopyParams copyParams{1, static_cast<uint16_t>(H_ * sizeof(T)), 0, 0};
            DataCopyPadParams padParams{isPadH_, 0, static_cast<uint8_t>(rightPaddingH_), 0};
            DataCopyPad(expandedPermutedTmpUb, gmExpandedPermutedRows_[expandedPermutedRowsIndex * H_], copyParams,
                        padParams);

            DataCopyPad(biasTmpUb, gmBias_[biasIndex * H_], copyParams, padParams);

            PipeBarrier<PIPE_ALL>();
            Add(expandedPermutedTmpUb, expandedPermutedTmpUb, biasTmpUb, H_);

            PipeBarrier<PIPE_ALL>();
            Muls(expandedPermutedTmpUb, expandedPermutedTmpUb, scalesVal, H_);

            PipeBarrier<PIPE_ALL>();
            Add(outLocal[outRowIndex], outLocal[outRowIndex], expandedPermutedTmpUb, H_);
        }
    }
    outQueue_.EnQue(outLocal);

    expertForSourceRowQueue_.FreeTensor(expertForSourceRowLocal);
    scalesQueue_.FreeTensor(scalesLocal);
    skip1Queue_.FreeTensor(skip1Local);
    if (skip2IsNull_ == 0) {
        skip2Queue_.FreeTensor(skip2Local);
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFP<T>::CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParamsOut{(uint16_t)curRepeatTimes, (uint16_t)(H_ * sizeof(T)), (uint16_t)0, (uint16_t)0};
    DataCopyPad(gmOut_[nLoopIdx * H_ * curCoreHandleNumPerLoop_], outLocal, copyParamsOut);
    outQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFP<T>::Process()
{
    if (GetBlockIdx() >= usedCoreNum_) {
        return;
    }
    int64_t loopCount = normalCoreLoopNum_;
    int64_t tailLoopBlock = normalCoreHandleNumTailLoop_;
    if ((GetBlockIdx() + 1) == usedCoreNum_) {
        loopCount = tailCoreLoopNum_;
        tailLoopBlock = tailCoreHandleNumTailLoop_;
    }

    for (int64_t n = 0; n < loopCount - 1; n++) {
        CopyIn(n, curCoreHandleNumPerLoop_);
        Compute(n, curCoreHandleNumPerLoop_);
        CopyOut(n, curCoreHandleNumPerLoop_);
    }
    // tail loop
    CopyIn(loopCount - 1, tailLoopBlock);
    Compute(loopCount - 1, tailLoopBlock);
    CopyOut(loopCount - 1, tailLoopBlock);
}

} // namespace MoeFinalizeRouting
#endif // MOE_FINALIZE_ROUTING_FP
