/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_finalize_routing_fp_db.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_FP_DB
#define MOE_FINALIZE_ROUTING_FP_DB

#include "moe_finalize_routing_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRouting {

using namespace AscendC;

template <typename T>
class MoeFinalizeRoutingFpDb {
public:
    __aicore__ inline MoeFinalizeRoutingFpDb(){};
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
    __aicore__ inline int64_t PadProcessT(int64_t param);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> skip1Queue_;
    TQue<QuePosition::VECIN, 1> skip2Queue_;
    TQue<QuePosition::VECIN, 1> scalesQueue_;
    TQue<QuePosition::VECIN, 1> expertForSourceRowQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb0_;
    TBuf<QuePosition::VECCALC> biasBufDb0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb1_;
    TBuf<QuePosition::VECCALC> biasBufDb1_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

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
    int64_t outputIdx_{0};
    int64_t curCoreHandleNumPerLoop_{0};

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadH_ = false;
    bool isPadK_ = false;
    bool isPadKInt32_ = false;
    uint16_t rightPaddingH_ = 0;
    uint16_t rightPaddingK_ = 0;
    uint16_t rightPaddingKInt32_ = 0;
};

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFpDb<T>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFpDb<T>::PadProcessT(int64_t param)
{
    return onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpDb<T>::ParseTilingData(const MoeFinalizeRoutingTilingData *tilingData)
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
__aicore__ inline void MoeFinalizeRoutingFpDb<T>::Init(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2,
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
#ifndef __CCE_KT_TEST__
    pipe.InitBuffer(skip1Queue_, 1, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(T));
#else 
    pipe.InitBuffer(skip1Queue_, 1, ONE_BLK_SIZE);
#endif
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
#ifndef __CCE_KT_TEST__
    pipe.InitBuffer(scalesQueue_, BUFFER_NUM, normalCoreHandleNumPerLoop_ * AlignmentProcess(K_) * sizeof(T));
    pipe.InitBuffer(expertForSourceRowQueue_, BUFFER_NUM,
                    normalCoreHandleNumPerLoop_ * Int32AlignmentProcess(K_) * sizeof(int32_t));
    pipe.InitBuffer(outQueue_, BUFFER_NUM, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(T));
#else 
    pipe.InitBuffer(scalesQueue_, BUFFER_NUM, ONE_BLK_SIZE);
    pipe.InitBuffer(expertForSourceRowQueue_, BUFFER_NUM, ONE_BLK_SIZE);
    pipe.InitBuffer(outQueue_, BUFFER_NUM, ONE_BLK_SIZE);
#endif
    pipe.InitBuffer(expandedPermutedRowsBufDb0_, AlignmentProcess(H_) * sizeof(T));
    pipe.InitBuffer(biasBufDb0_, AlignmentProcess(H_) * sizeof(T));

    pipe.InitBuffer(expandedPermutedRowsBufDb1_, AlignmentProcess(H_) * sizeof(T));
    pipe.InitBuffer(biasBufDb1_, AlignmentProcess(H_) * sizeof(T));
    pipe.InitBuffer(tmpBuf_, AlignmentProcess(H_) * sizeof(T));
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpDb<T>::CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<T> scalesLocal = scalesQueue_.AllocTensor<T>();
    LocalTensor<int32_t> expertForSourceRowLocal = expertForSourceRowQueue_.AllocTensor<int32_t>();

    LocalTensor<T> skip1Local = skip1Queue_.AllocTensor<T>();
    LocalTensor<T> skip2Local;
    if (skip2IsNull_ == 0) {
        skip2Local = skip2Queue_.AllocTensor<T>();
    }

    // ---------------------------- [Skip] -------------------------------
    DataCopyParams copyParamsSkip{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(H_ * sizeof(T)), 0, 0};
    DataCopyPadParams padParamsSkip{isPadH_, 0, static_cast<uint8_t>(rightPaddingH_), 0};

    DataCopyPad(skip1Local, gmSkip1_[nLoopIdx * curCoreHandleNumPerLoop_ * H_], copyParamsSkip, padParamsSkip);
    if (skip2IsNull_ == 0) {
        DataCopyPad(skip2Local, gmSkip2_[nLoopIdx * curCoreHandleNumPerLoop_ * H_], copyParamsSkip, padParamsSkip);
    }

    // ---------------------------- [Scales] -------------------------------
    DataCopyParams copyParamsScales{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(K_ * sizeof(T)), 0, 0};
    DataCopyPadParams padParamsScales{isPadK_, 0, static_cast<uint8_t>(rightPaddingK_), 0};
    DataCopyPad(scalesLocal, gmScales_[nLoopIdx * curCoreHandleNumPerLoop_ * K_], copyParamsScales, padParamsScales);

    // ---------------------------- [Expert] -------------------------------
    DataCopyParams copyParamsExpert{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(K_ * sizeof(int32_t)),
                                    0, 0};
    DataCopyPadParams padParamsExpert{isPadKInt32_, 0, static_cast<uint8_t>(rightPaddingKInt32_), 0};
    DataCopyPad(expertForSourceRowLocal, gmExpertForSourceRow_[nLoopIdx * curCoreHandleNumPerLoop_ * K_],
                copyParamsExpert, padParamsExpert);

    if (skip2IsNull_ == 0) {
        skip2Queue_.EnQue(skip2Local);
    }
    skip1Queue_.EnQue(skip1Local);
    scalesQueue_.EnQue(scalesLocal);
    expertForSourceRowQueue_.EnQue(expertForSourceRowLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpDb<T>::Compute(int64_t nLoopIdx, int64_t curRepeatTimes)
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
    PipeBarrier<PIPE_ALL>();
    LocalTensor<T> expandedPermutedTmpUbDb0 = expandedPermutedRowsBufDb0_.Get<T>();
    LocalTensor<T> biasTmpUbDb0 = biasBufDb0_.Get<T>();

    LocalTensor<T> expandedPermutedTmpUbDb1 = expandedPermutedRowsBufDb1_.Get<T>();
    LocalTensor<T> biasTmpUbDb1 = biasBufDb1_.Get<T>();
    LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();

    DataCopyParams copyParams{1, static_cast<uint16_t>(H_ * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH_, 0, static_cast<uint8_t>(rightPaddingH_), 0};

    SetFlag<HardEvent::V_S>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    SetFlag<HardEvent::V_S>(EVENT_ID3);
    for (int64_t i = 0; i < curRepeatTimes; i++) {
        int64_t outRowIndex = i * AlignmentProcess(H_);
        for (int64_t j = 0; j < K_ / PARALLEL_NUM; j++) {
            /*******************************乒***********************************************/
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            int64_t expandedSrcToDstRowIndexDb0 =
                nLoopIdx * curCoreHandleNumPerLoop_ + i + PARALLEL_NUM * j * BS_ + GetBlockIdx() * normalCoreHandleNum_;
            int64_t expandedPermutedRowsIndexDb0 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            int64_t biasIndexDb0 = expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(K_) + PARALLEL_NUM * j);
            T scalesValDb0 = scalesLocal.GetValue(i * AlignmentProcess(K_) + PARALLEL_NUM * j);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID2);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
            int64_t expandedSrcToDstRowIndexDb1 = nLoopIdx * curCoreHandleNumPerLoop_ + i +
                                                  (PARALLEL_NUM * j + 1) * BS_ + GetBlockIdx() * normalCoreHandleNum_;
            int64_t expandedPermutedRowsIndexDb1 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb1);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID1);

            WaitFlag<HardEvent::V_S>(EVENT_ID3);
            int64_t biasIndexDb1 =
                expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(K_) + PARALLEL_NUM * j + 1);
            T scalesValDb1 = scalesLocal.GetValue(i * AlignmentProcess(K_) + PARALLEL_NUM * j + 1);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID3);

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
            DataCopyPad(expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * H_],
                        copyParams, padParams);
            DataCopyPad(biasTmpUbDb0, gmBias_[biasIndexDb0 * H_], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID3);
            DataCopyPad(expandedPermutedTmpUbDb1, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb1 * H_],
                        copyParams, padParams);
            DataCopyPad(biasTmpUbDb1, gmBias_[biasIndexDb1 * H_], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Add(skip1Local, expandedPermutedTmpUbDb0, biasTmpUbDb0, H_);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            PipeBarrier<PIPE_V>();
            Muls(skip1Local, skip1Local, scalesValDb0, H_);
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            PipeBarrier<PIPE_V>();
            Add(outLocal[outRowIndex], outLocal[outRowIndex], skip1Local, H_);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Add(tmpLocal, expandedPermutedTmpUbDb1, biasTmpUbDb1, H_);
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            PipeBarrier<PIPE_V>();
            Muls(tmpLocal, tmpLocal, scalesValDb1, H_);
            SetFlag<HardEvent::V_S>(EVENT_ID3);
            PipeBarrier<PIPE_V>();
            Add(outLocal[outRowIndex], outLocal[outRowIndex], tmpLocal, H_);
        }
        if (K_ % PARALLEL_NUM != 0) {
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            int64_t expandedSrcToDstRowIndexDb0 =
                nLoopIdx * curCoreHandleNumPerLoop_ + i + (K_ - 1) * BS_ + GetBlockIdx() * normalCoreHandleNum_;
            int64_t expandedPermutedRowsIndexDb0 = gmExpandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            int64_t biasIndexDb0 = expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(K_) + K_ - 1);
            T scalesValDb0 = scalesLocal.GetValue(i * AlignmentProcess(K_) + K_ - 1);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID2);

            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
            DataCopyPad(expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * H_],
                        copyParams, padParams);
            DataCopyPad(biasTmpUbDb0, gmBias_[biasIndexDb0 * H_], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Add(skip1Local, expandedPermutedTmpUbDb0, biasTmpUbDb0, H_);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            PipeBarrier<PIPE_V>();
            Muls(skip1Local, skip1Local, scalesValDb0, H_);
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            PipeBarrier<PIPE_V>();
            Add(outLocal[outRowIndex], outLocal[outRowIndex], skip1Local, H_);
        }
    }
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID3);
    outQueue_.EnQue(outLocal);

    expertForSourceRowQueue_.FreeTensor(expertForSourceRowLocal);
    scalesQueue_.FreeTensor(scalesLocal);
    skip1Queue_.FreeTensor(skip1Local);
    if (skip2IsNull_ == 0) {
        skip2Queue_.FreeTensor(skip2Local);
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpDb<T>::CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(H_ * sizeof(T)), 0, 0};
    DataCopyPad(gmOut_[nLoopIdx * H_ * curCoreHandleNumPerLoop_], outLocal, copyParams);
    outQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpDb<T>::Process()
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
#endif // MOE_FINALIZE_ROUTING_FP_DB
