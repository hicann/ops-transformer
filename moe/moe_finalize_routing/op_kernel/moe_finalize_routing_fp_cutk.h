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
 * \file moe_finalize_routing_fp_cutk.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_FP_CUTK
#define MOE_FINALIZE_ROUTING_FP_CUTK

#include "moe_finalize_routing_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRouting {

using namespace AscendC;

template <typename T>
class MoeFinalizeRoutingFpCutK {
public:
    __aicore__ inline MoeFinalizeRoutingFpCutK(){};
    __aicore__ inline void Init(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expandedSrcToDstRow, GM_ADDR expertForSourceRow, GM_ADDR out,
                                GM_ADDR workspace, const MoeFinalizeRoutingTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const MoeFinalizeRoutingTilingData* tilingData);
    __aicore__ inline void CkechColAlignment();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> skip1Queue_;
    TQue<QuePosition::VECIN, 1> skip2Queue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> expertForSourceRowBuf_;
    TBuf<QuePosition::VECCALC> scalesBuf_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb0_;
    TBuf<QuePosition::VECCALC> biasBufDb0_;

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
    int64_t normalH_{0};
    int64_t unnormalH_{0};
    int64_t cutNumH_{0};
    int64_t normalK_{0};
    int64_t unnormalK_{0};
    int64_t cutNumK_{0};

    // Judge skip2 if null
    int64_t skip2IsNull_{true};

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
    int64_t outputIdx_{0};
    int64_t curCoreHandleNumPerLoop_{0};
    int64_t biasInCore_{0};

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadNormalH_ = false;
    bool isPadUnnormalH_ = false;
    bool isPadNormalK_ = false;
    bool isPadUnnormalK_ = false;
    bool isPadNormalKInt32_ = false;
    bool isPadUnnormalKInt32_ = false;
    uint16_t rightPaddingNormalH_ = 0;
    uint16_t rightPaddingUnnormalH_ = 0;
    uint16_t rightPaddingNormalK_ = 0;
    uint16_t rightPaddingUnnormalK_ = 0;
    uint16_t rightPaddingNormalKInt32_ = 0;
    uint16_t rightPaddingUnnormalKInt32_ = 0;
};

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFpCutK<T>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFpCutK<T>::PadProcessT(int64_t param)
{
    return onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCutK<T>::ParseTilingData(const MoeFinalizeRoutingTilingData* tilingData)
{
    skip2IsNull_ = tilingData->skip2IsNull;
    BS_ = tilingData->totalRowNum;
    H_ = tilingData->H;
    K_ = tilingData->K;
    E_ = tilingData->biasRowNum;
    normalH_ = tilingData->normalH;
    unnormalH_ = tilingData->unnormalH;
    cutNumH_ = tilingData->hSliceNum;
    normalK_ = tilingData->normalK;
    unnormalK_ = tilingData->unnormalK;
    cutNumK_ = tilingData->kSliceNum;

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
__aicore__ inline void MoeFinalizeRoutingFpCutK<T>::CkechColAlignment()
{
    if (normalH_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalH_ = true;
        rightPaddingNormalH_ = PadProcessT(normalH_);
    }

    if (unnormalH_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalH_ = true;
        rightPaddingUnnormalH_ = PadProcessT(unnormalH_);
    }

    if (normalK_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalK_ = true;
        rightPaddingNormalK_ = PadProcessT(normalK_);
    }

    if (unnormalK_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalK_ = true;
        rightPaddingUnnormalK_ = PadProcessT(unnormalK_);
    }

    if (normalK_ * INT32_BYTES % ONE_BLK_SIZE) {
        isPadNormalKInt32_ = true;
        rightPaddingNormalKInt32_ = PadProcessInt32(normalK_);
    }

    if (unnormalK_ * INT32_BYTES % ONE_BLK_SIZE) {
        isPadUnnormalKInt32_ = true;
        rightPaddingUnnormalKInt32_ = PadProcessInt32(unnormalK_);
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCutK<T>::Init(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2,
                                                         GM_ADDR bias, GM_ADDR scales, GM_ADDR expandedSrcToDstRow,
                                                         GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace,
                                                         const MoeFinalizeRoutingTilingData* tilingData)
{
    // init tiling data
    ParseTilingData(tilingData);

    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreHandleNumPerLoop_ = tailCoreHandleNumPerLoop_;
    } else {
        curCoreHandleNumPerLoop_ = normalCoreHandleNumPerLoop_;
    }
    biasInCore_ = GetBlockIdx() * normalCoreHandleNum_;

    // 检查要处理的列数是否对齐以及应该如何对齐
    CkechColAlignment();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = biasInCore_ * H_;
    gmSkip1_.SetGlobalBuffer((__gm__ T*)skip1 + inputSkipIdx_, normalCoreHandleNum_ * H_);
    pipe.InitBuffer(skip1Queue_, 1, normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_) * sizeof(T));
    if (skip2IsNull_ == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T*)skip2 + inputSkipIdx_, normalCoreHandleNum_ * H_);
        pipe.InitBuffer(skip2Queue_, 1, normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_) * sizeof(T));
    }

    inputScalesAndExpertIdx_ = biasInCore_ * K_;
    gmScales_.SetGlobalBuffer((__gm__ T*)scales + inputScalesAndExpertIdx_, normalCoreHandleNum_ * K_);
    gmExpertForSourceRow_.SetGlobalBuffer((__gm__ int32_t*)expertForSourceRow + inputScalesAndExpertIdx_,
                                          normalCoreHandleNum_ * K_);

    gmExpandedSrcToDstRow_.SetGlobalBuffer((__gm__ int32_t*)expandedSrcToDstRow, BS_ * K_);

    outputIdx_ = biasInCore_ * H_;
    gmOut_.SetGlobalBuffer((__gm__ T*)out + outputIdx_, normalCoreHandleNum_ * H_);

    gmExpandedPermutedRows_.SetGlobalBuffer((__gm__ T*)expandedPermutedRows, BS_ * K_ * H_);
    gmBias_.SetGlobalBuffer((__gm__ T*)bias, E_ * H_);

    // 申请 buffer 空间
    pipe.InitBuffer(outQueue_, BUFFER_NUM, normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_) * sizeof(T));
    pipe.InitBuffer(scalesBuf_, AlignmentProcess(normalK_) * sizeof(T));
    pipe.InitBuffer(expertForSourceRowBuf_, Int32AlignmentProcess(normalK_) * sizeof(int32_t));

    pipe.InitBuffer(expandedPermutedRowsBufDb0_, AlignmentProcess(normalH_) * sizeof(T));
    pipe.InitBuffer(biasBufDb0_, AlignmentProcess(normalH_) * sizeof(T));
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCutK<T>::CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    bool isNormalH = (nLoopIdx + 1) % (cutNumH_ + 1) != 0;
    int64_t bias = isNormalH ? (nLoopIdx % cutNumH_) * normalH_ : cutNumH_ * normalH_;
    int64_t dataLen = isNormalH ? normalH_ : unnormalH_;
    int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
    bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;
    LocalTensor<T> skip1Local = skip1Queue_.AllocTensor<T>();

    DataCopyParams copyParams{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    if (cutNumH_ == 0) {
        bias = 0;
    }
    DataCopyPad(skip1Local, gmSkip1_[nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_ * H_ + bias], copyParams,
                padParams);
    LocalTensor<T> skip2Local;
    if (skip2IsNull_ == 0) {
        skip2Local = skip2Queue_.AllocTensor<T>();
        DataCopyPad(skip2Local, gmSkip2_[nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_ * H_ + bias], copyParams,
                    padParams);
        skip2Queue_.EnQue(skip2Local);
    }

    skip1Queue_.EnQue(skip1Local);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCutK<T>::Compute(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    bool isNormalH = (nLoopIdx + 1) % (cutNumH_ + 1) != 0;
    int64_t bias = isNormalH ? (nLoopIdx % cutNumH_) * normalH_ : cutNumH_ * normalH_;
    if (cutNumH_ == 0) {
        bias = 0;
    }
    int64_t dataLen = isNormalH ? normalH_ : unnormalH_;
    int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
    bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;
    int64_t biasInRow = nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_;
    LocalTensor<T> skip1Local = skip1Queue_.DeQue<T>();
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    LocalTensor<T> skip2Local;
    if (skip2IsNull_ == 0) {
        skip2Local = skip2Queue_.DeQue<T>();
        Add(outLocal, skip1Local, skip2Local, curRepeatTimes * AlignmentProcess(dataLen));
    } else {
        Adds(outLocal, skip1Local, (T)0, curRepeatTimes * AlignmentProcess(dataLen));
    }

    LocalTensor<T> expandedPermutedTmpUb = expandedPermutedRowsBufDb0_.Get<T>();
    LocalTensor<T> biasTmpUb = biasBufDb0_.Get<T>();

    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};
    for (int64_t i = 0; i < curRepeatTimes; i++) {
        int64_t outRowIndex = i * AlignmentProcess(dataLen);
        int64_t len = normalK_;
        bool isPadK = isPadNormalK_;
        bool isPadKInt32 = isPadNormalKInt32_;
        uint16_t rightPaddingK = rightPaddingNormalK_;
        uint16_t rightPaddingKInt32 = rightPaddingNormalKInt32_;
        for (int64_t n = 0; n < cutNumK_; n++) {
            if (n == cutNumK_ - 1) {
                len = unnormalK_;
                isPadK = isPadUnnormalK_;
                isPadKInt32 = isPadUnnormalKInt32_;
                rightPaddingK = rightPaddingUnnormalK_;
                rightPaddingKInt32 = rightPaddingUnnormalKInt32_;
            }
            int64_t biasOfK = n * normalK_;
            LocalTensor<int32_t> expertForSourceRowLocal = expertForSourceRowBuf_.Get<int32_t>();
            LocalTensor<T> scalesLocal = scalesBuf_.Get<T>();
            // ---------------------------- [Scales] -------------------------------
            DataCopyParams copyParamsScales{1, static_cast<uint16_t>(len * sizeof(T)), 0, 0};
            DataCopyPadParams padParamsScales{isPadK, 0, static_cast<uint8_t>(rightPaddingK), 0};
            DataCopyPad(scalesLocal, gmScales_[biasInRow * K_ + i * K_ + biasOfK], copyParamsScales, padParamsScales);

            // ---------------------------- [Expert] -------------------------------
            DataCopyParams copyParamsExpert{1, static_cast<uint16_t>(len * sizeof(int32_t)), 0, 0};
            DataCopyPadParams padParamsExpert{isPadKInt32, 0, static_cast<uint8_t>(rightPaddingKInt32), 0};
            DataCopyPad(expertForSourceRowLocal, gmExpertForSourceRow_[biasInRow * K_ + i * K_ + biasOfK],
                        copyParamsExpert, padParamsExpert);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            for (int64_t j = 0; j < len; j++) {
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                int64_t expandedSrcToDstRowIndex = biasInRow + i + (j + biasOfK) * BS_ + biasInCore_;
                int64_t expandedPermutedRowsIndex = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndex);
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

                WaitFlag<HardEvent::V_S>(EVENT_ID1);
                int64_t biasIndexDb0 = expertForSourceRowLocal.GetValue(j);
                T scalesValDb0 = scalesLocal.GetValue(j);
                SetFlag<HardEvent::S_MTE2>(EVENT_ID1);

                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
                DataCopyPad(expandedPermutedTmpUb, gmExpandedPermutedRows_[expandedPermutedRowsIndex * H_ + bias],
                            copyParams, padParams);
                DataCopyPad(biasTmpUb, gmBias_[biasIndexDb0 * H_ + bias], copyParams, padParams);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Add(skip1Local, expandedPermutedTmpUb, biasTmpUb, dataLen);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                PipeBarrier<PIPE_V>();
                Muls(skip1Local, skip1Local, scalesValDb0, dataLen);
                SetFlag<HardEvent::V_S>(EVENT_ID1);
                PipeBarrier<PIPE_V>();
                Add(outLocal[outRowIndex], outLocal[outRowIndex], skip1Local, dataLen);
            }
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
        }
    }
    outQueue_.EnQue(outLocal);

    skip1Queue_.FreeTensor(skip1Local);
    if (skip2IsNull_ == 0) {
        skip2Queue_.FreeTensor(skip2Local);
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCutK<T>::CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    bool isNormalH = (nLoopIdx + 1) % (cutNumH_ + 1) != 0;
    int64_t bias = isNormalH ? (nLoopIdx % cutNumH_) * normalH_ : cutNumH_ * normalH_;
    if (cutNumH_ == 0) {
        bias = 0;
    }
    int64_t dataLen = isNormalH ? normalH_ : unnormalH_;
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPad(gmOut_[nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_ * H_ + bias], outLocal, copyParams);
    outQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCutK<T>::Process()
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
    {
        CopyIn(loopCount - 1, tailLoopBlock);
        Compute(loopCount - 1, tailLoopBlock);
        CopyOut(loopCount - 1, tailLoopBlock);
    }
}

} // namespace MoeFinalizeRouting
#endif // MOE_FINALIZE_ROUTING_FP_CUTK
