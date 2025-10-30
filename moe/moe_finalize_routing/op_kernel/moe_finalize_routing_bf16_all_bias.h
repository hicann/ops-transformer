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
 * \file moe_finalize_routing_bf16_all_bias.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_BF16_ALL_BIAS
#define MOE_FINALIZE_ROUTING_BF16_ALL_BIAS

#include "moe_finalize_routing_common.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MoeFinalizeRouting {

using namespace AscendC;

template <typename T>
class MoeFinalizeRoutingBF16AllBias {
public:
    __aicore__ inline MoeFinalizeRoutingBF16AllBias(){};
    __aicore__ inline void Init(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expandedSrcToDstRow, GM_ADDR expertForSourceRow, GM_ADDR out,
                                GM_ADDR workspace, const MoeFinalizeRoutingTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const MoeFinalizeRoutingTilingData *tilingData);
    __aicore__ inline void CkechColAlignment();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcess(int64_t param);
    __aicore__ inline void PrepareData();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> skip1Queue_;
    TQue<QuePosition::VECIN, 1> skip2Queue_;
    TQue<QuePosition::VECIN, 1> scalesQueue_;
    TQue<QuePosition::VECIN, 1> expertForSourceRowQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> expandedPermutedRowsBuf0_;
    TBuf<QuePosition::VECCALC> biasBuf0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBuf1_;
    TBuf<QuePosition::VECCALC> biasBuf1_;

    TBuf<QuePosition::VECCALC> expandedSrcToDstRowBuff_;

    TBuf<QuePosition::VECCALC> skip1CastBuf_;
    TBuf<QuePosition::VECCALC> skip2CastBuf_;
    TBuf<QuePosition::VECCALC> biasCastBuf0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf0_;
    TBuf<QuePosition::VECCALC> biasCastBuf1_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsCastBuf1_;

    GlobalTensor<T> gmExpandedPermutedRows_;
    GlobalTensor<T> gmSkip1_;
    GlobalTensor<T> gmSkip2_;
    GlobalTensor<T> gmBias_;
    GlobalTensor<T> gmScales_;
    GlobalTensor<int32_t> gmExpandedSrcToDstRow_;
    GlobalTensor<int32_t> gmExpertForSourceRow_;
    GlobalTensor<T> gmOut_;

    LocalTensor<int32_t> expandedSrcToDstRow_;

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
    int64_t biasInCore_{0};

    // tiling params
    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadH = false;
    bool isPadK = false;
    bool isPadKInt32 = false;
    uint16_t rightPaddingH = 0;
    uint16_t rightPaddingK = 0;
    uint16_t rightPaddingKInt32 = 0;

    bool isPadSourceToDstRow_ = false;
    uint16_t rightPaddingSourceToDstRow_ = 0;
    int64_t curCoreHandleNum_ = 0;
};

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingBF16AllBias<T>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingBF16AllBias<T>::PadProcess(int64_t param)
{
    return onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingBF16AllBias<T>::ParseTilingData(const MoeFinalizeRoutingTilingData *tilingData)
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
__aicore__ inline void MoeFinalizeRoutingBF16AllBias<T>::CkechColAlignment()
{
    if (H_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadH = true;
        rightPaddingH = PadProcess(H_);
    }

    if (K_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadK = true;
        rightPaddingK = PadProcess(K_);
    }

    if (K_ * INT32_BYTES % ONE_BLK_SIZE) {
        isPadKInt32 = true;
        rightPaddingKInt32 = PadProcessInt32(K_);
    }

    if (curCoreHandleNum_ * INT32_BYTES % ONE_BLK_SIZE) {
        isPadSourceToDstRow_ = true;
        rightPaddingSourceToDstRow_ = PadProcessInt32(curCoreHandleNum_);
    }
}

template <typename T>
__aicore__ inline void
MoeFinalizeRoutingBF16AllBias<T>::Init(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
                                       GM_ADDR scales, GM_ADDR expandedSrcToDstRow, GM_ADDR expertForSourceRow,
                                       GM_ADDR out, GM_ADDR workspace, const MoeFinalizeRoutingTilingData *tilingData)
{
    // init tiling data
    ParseTilingData(tilingData);

    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreHandleNumPerLoop_ = tailCoreHandleNumPerLoop_;
        curCoreHandleNum_ = tailCoreHandleNum_;
    } else {
        curCoreHandleNumPerLoop_ = normalCoreHandleNumPerLoop_;
        curCoreHandleNum_ = normalCoreHandleNum_;
    }
    biasInCore_ = GetBlockIdx() * normalCoreHandleNum_;

    CkechColAlignment();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = biasInCore_ * H_;
    gmSkip1_.SetGlobalBuffer((__gm__ T *)skip1 + inputSkipIdx_, normalCoreHandleNum_ * H_);
    pipe.InitBuffer(skip1Queue_, BUFFER_NUM, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(T));
    if (skip2IsNull_ == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T *)skip2 + inputSkipIdx_, normalCoreHandleNum_ * H_);
        pipe.InitBuffer(skip2Queue_, BUFFER_NUM, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(T));
        pipe.InitBuffer(skip2CastBuf_, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(float));
    }

    inputScalesAndExpertIdx_ = biasInCore_ * K_;
    gmScales_.SetGlobalBuffer((__gm__ T *)scales + inputScalesAndExpertIdx_, normalCoreHandleNum_ * K_);
    gmExpertForSourceRow_.SetGlobalBuffer((__gm__ int32_t *)expertForSourceRow + inputScalesAndExpertIdx_,
                                          normalCoreHandleNum_ * K_);

    gmExpandedSrcToDstRow_.SetGlobalBuffer((__gm__ int32_t *)expandedSrcToDstRow, BS_ * K_);

    outputIdx_ = biasInCore_ * H_;
    gmOut_.SetGlobalBuffer((__gm__ T *)out + outputIdx_, normalCoreHandleNum_ * H_);

    gmExpandedPermutedRows_.SetGlobalBuffer((__gm__ T *)expandedPermutedRows, BS_ * K_ * H_);
    gmBias_.SetGlobalBuffer((__gm__ T *)bias, E_ * H_);

    // 申请 buffer 空间
    pipe.InitBuffer(scalesQueue_, BUFFER_NUM, normalCoreHandleNumPerLoop_ * AlignmentProcess(K_) * sizeof(T));
    pipe.InitBuffer(expertForSourceRowQueue_, BUFFER_NUM,
                    normalCoreHandleNumPerLoop_ * Int32AlignmentProcess(K_) * sizeof(int32_t));
    pipe.InitBuffer(outQueue_, BUFFER_NUM, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(T));

    pipe.InitBuffer(expandedPermutedRowsBuf0_, AlignmentProcess(H_) * sizeof(T));
    pipe.InitBuffer(biasBuf0_, AlignmentProcess(H_) * sizeof(T));
    pipe.InitBuffer(expandedPermutedRowsBuf1_, AlignmentProcess(H_) * sizeof(T));
    pipe.InitBuffer(biasBuf1_, AlignmentProcess(H_) * sizeof(T));

    pipe.InitBuffer(skip1CastBuf_, normalCoreHandleNumPerLoop_ * AlignmentProcess(H_) * sizeof(float));
    pipe.InitBuffer(biasCastBuf0_, AlignmentProcess(H_) * sizeof(float));
    pipe.InitBuffer(expandedPermutedRowsCastBuf0_, AlignmentProcess(H_) * sizeof(float));
    pipe.InitBuffer(biasCastBuf1_, AlignmentProcess(H_) * sizeof(float));
    pipe.InitBuffer(expandedPermutedRowsCastBuf1_, AlignmentProcess(H_) * sizeof(float));

    pipe.InitBuffer(expandedSrcToDstRowBuff_, Int32AlignmentProcess(curCoreHandleNum_) * K_ * sizeof(int32_t));
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingBF16AllBias<T>::PrepareData()
{
    expandedSrcToDstRow_ = expandedSrcToDstRowBuff_.Get<int32_t>();

    DataCopyExtParams copyParamsSrcToDstRow{static_cast<uint16_t>(K_),
                                            static_cast<uint32_t>(curCoreHandleNum_ * sizeof(int32_t)),
                                            static_cast<uint32_t>((BS_ - curCoreHandleNum_) * sizeof(int32_t)), 0, 0};
    DataCopyPadExtParams<int32_t> padParamsSrcToDstRow{isPadSourceToDstRow_, 0,
                                                       static_cast<uint8_t>(rightPaddingSourceToDstRow_), 0};

    DataCopyPad(expandedSrcToDstRow_, gmExpandedSrcToDstRow_[GetBlockIdx() * normalCoreHandleNum_],
                copyParamsSrcToDstRow, padParamsSrcToDstRow);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingBF16AllBias<T>::CopyIn(int64_t nLoopIdx, int64_t curRepeatTimes)
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
    DataCopyPadParams padParamsSkip{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    DataCopyPad(skip1Local, gmSkip1_[nLoopIdx * curCoreHandleNumPerLoop_ * H_], copyParamsSkip, padParamsSkip);
    if (skip2IsNull_ == 0) {
        DataCopyPad(skip2Local, gmSkip2_[nLoopIdx * curCoreHandleNumPerLoop_ * H_], copyParamsSkip, padParamsSkip);
    }

    // ---------------------------- [Scales] -------------------------------
    DataCopyParams copyParamsScales{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(K_ * sizeof(T)), 0, 0};
    DataCopyPadParams padParamsScales{isPadK, 0, static_cast<uint8_t>(rightPaddingK), 0};
    DataCopyPad(scalesLocal, gmScales_[nLoopIdx * curCoreHandleNumPerLoop_ * K_], copyParamsScales, padParamsScales);

    // ---------------------------- [Expert] -------------------------------
    DataCopyParams copyParamsExpert{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(K_ * sizeof(int32_t)),
                                    0, 0};
    DataCopyPadParams padParamsExpert{isPadKInt32, 0, static_cast<uint8_t>(rightPaddingKInt32), 0};
    DataCopyPad(expertForSourceRowLocal, gmExpertForSourceRow_[nLoopIdx * curCoreHandleNumPerLoop_ * K_],
                copyParamsExpert, padParamsExpert);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);

    if (skip2IsNull_ == 0) {
        skip2Queue_.EnQue(skip2Local);
    }
    skip1Queue_.EnQue(skip1Local);
    scalesQueue_.EnQue(scalesLocal);
    expertForSourceRowQueue_.EnQue(expertForSourceRowLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingBF16AllBias<T>::Compute(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<int32_t> expertForSourceRowLocal = expertForSourceRowQueue_.DeQue<int32_t>();
    LocalTensor<T> scalesLocal = scalesQueue_.DeQue<T>();
    LocalTensor<T> skip1Local = skip1Queue_.DeQue<T>();
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    LocalTensor<float> skip1CastUb = skip1CastBuf_.Get<float>();
    Cast(skip1CastUb, skip1Local, RoundMode::CAST_NONE, curRepeatTimes * AlignmentProcess(H_));

    LocalTensor<T> skip2Local;
    LocalTensor<float> skip2CastUb;
    if (skip2IsNull_ == 0) {
        skip2Local = skip2Queue_.DeQue<T>();
        skip2CastUb = skip2CastBuf_.Get<float>();
        Cast(skip2CastUb, skip2Local, RoundMode::CAST_NONE, curRepeatTimes * AlignmentProcess(H_));
        PipeBarrier<PIPE_V>();
        Add(skip1CastUb, skip1CastUb, skip2CastUb, curRepeatTimes * AlignmentProcess(H_));
    }

    LocalTensor<T> expandedPermutedTmpUbDb0 = expandedPermutedRowsBuf0_.Get<T>();
    LocalTensor<T> biasTmpUbDb0 = biasBuf0_.Get<T>();
    LocalTensor<T> expandedPermutedTmpUbDb1 = expandedPermutedRowsBuf1_.Get<T>();
    LocalTensor<T> biasTmpUbDb1 = biasBuf1_.Get<T>();

    LocalTensor<float> expandedPermutedRowsCastUb0 = expandedPermutedRowsCastBuf0_.Get<float>();
    LocalTensor<float> biasCastUb0 = biasCastBuf0_.Get<float>();

    LocalTensor<float> expandedPermutedRowsCastUb1 = expandedPermutedRowsCastBuf1_.Get<float>();
    LocalTensor<float> biasCastUb1 = biasCastBuf1_.Get<float>();

    DataCopyParams copyParams{1, static_cast<uint16_t>(H_ * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    int64_t biasInLoop = nLoopIdx * curCoreHandleNumPerLoop_;

    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    SetFlag<HardEvent::V_S>(EVENT_ID3);
    for (int64_t i = 0; i < curRepeatTimes; i++) {
        int64_t outRowIndex = i * AlignmentProcess(H_);
        for (int64_t j = 0; j < K_ / PARALLEL_NUM; j++) {
            /*******************************乒***********************************************/
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            int64_t expandedSrcToDstRowIndexDb0 =
                biasInLoop + i + PARALLEL_NUM * j * Int32AlignmentProcess(curCoreHandleNum_);
            int64_t expandedPermutedRowsIndexDb0 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            int64_t biasIndexDb0 = expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(K_) + PARALLEL_NUM * j);
            float scalesValDb0 = ToFloat(scalesLocal.GetValue(i * AlignmentProcess(K_) + PARALLEL_NUM * j));
            SetFlag<HardEvent::S_MTE2>(EVENT_ID2);

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
            DataCopyPad(expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * H_],
                        copyParams, padParams);
            DataCopyPad(biasTmpUbDb0, gmBias_[biasIndexDb0 * H_], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            int64_t expandedSrcToDstRowIndexDb1 =
                biasInLoop + i + (PARALLEL_NUM * j + 1) * Int32AlignmentProcess(curCoreHandleNum_);
            int64_t expandedPermutedRowsIndexDb1 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb1);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID1);

            WaitFlag<HardEvent::V_S>(EVENT_ID3);
            int64_t biasIndexDb1 =
                expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(K_) + PARALLEL_NUM * j + 1);
            float scalesValDb1 = ToFloat(scalesLocal.GetValue(i * AlignmentProcess(K_) + PARALLEL_NUM * j + 1));
            SetFlag<HardEvent::S_MTE2>(EVENT_ID3);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID3);
            DataCopyPad(expandedPermutedTmpUbDb1, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb1 * H_],
                        copyParams, padParams);
            DataCopyPad(biasTmpUbDb1, gmBias_[biasIndexDb1 * H_], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(expandedPermutedRowsCastUb0, expandedPermutedTmpUbDb0, RoundMode::CAST_NONE, H_);
            Cast(biasCastUb0, biasTmpUbDb0, RoundMode::CAST_NONE, H_);
            PipeBarrier<PIPE_V>();
            Add(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, biasCastUb0, H_);
            PipeBarrier<PIPE_V>();
            Muls(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, scalesValDb0, H_);
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            PipeBarrier<PIPE_V>();
            Add(skip1CastUb[outRowIndex], skip1CastUb[outRowIndex], expandedPermutedRowsCastUb0, H_);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Cast(expandedPermutedRowsCastUb1, expandedPermutedTmpUbDb1, RoundMode::CAST_NONE, H_);
            Cast(biasCastUb1, biasTmpUbDb1, RoundMode::CAST_NONE, H_);
            PipeBarrier<PIPE_V>();
            Add(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, biasCastUb1, H_);
            PipeBarrier<PIPE_V>();
            Muls(expandedPermutedRowsCastUb1, expandedPermutedRowsCastUb1, scalesValDb1, H_);
            SetFlag<HardEvent::V_S>(EVENT_ID3);
            PipeBarrier<PIPE_V>();
            Add(skip1CastUb[outRowIndex], skip1CastUb[outRowIndex], expandedPermutedRowsCastUb1, H_);
        }
        if (K_ % PARALLEL_NUM != 0) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            int64_t expandedSrcToDstRowIndexDb0 = biasInLoop + i + (K_ - 1) * Int32AlignmentProcess(curCoreHandleNum_);
            int64_t expandedPermutedRowsIndexDb0 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            int64_t biasIndexDb0 = expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(K_) + K_ - 1);
            float scalesValDb0 = ToFloat(scalesLocal.GetValue(i * AlignmentProcess(K_) + K_ - 1));
            SetFlag<HardEvent::S_MTE2>(EVENT_ID2);

            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
            DataCopyPad(expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * H_],
                        copyParams, padParams);
            DataCopyPad(biasTmpUbDb0, gmBias_[biasIndexDb0 * H_], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);

            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(expandedPermutedRowsCastUb0, expandedPermutedTmpUbDb0, RoundMode::CAST_NONE, H_);
            Cast(biasCastUb0, biasTmpUbDb0, RoundMode::CAST_NONE, H_);
            PipeBarrier<PIPE_V>();
            Add(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, biasCastUb0, H_);
            PipeBarrier<PIPE_V>();
            Muls(expandedPermutedRowsCastUb0, expandedPermutedRowsCastUb0, scalesValDb0, H_);
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            PipeBarrier<PIPE_V>();
            Add(skip1CastUb[outRowIndex], skip1CastUb[outRowIndex], expandedPermutedRowsCastUb0, H_);
        }
    }
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID3);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, skip1CastUb, RoundMode::CAST_ROUND, curRepeatTimes * AlignmentProcess(H_));
#ifndef __CCE_KT_TEST__
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
#endif
    outQueue_.EnQue(outLocal);

    expertForSourceRowQueue_.FreeTensor(expertForSourceRowLocal);
    scalesQueue_.FreeTensor(scalesLocal);
    skip1Queue_.FreeTensor(skip1Local);
    if (skip2IsNull_ == 0) {
        skip2Queue_.FreeTensor(skip2Local);
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingBF16AllBias<T>::CopyOut(int64_t nLoopIdx, int64_t curRepeatTimes)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{static_cast<uint16_t>(curRepeatTimes), static_cast<uint16_t>(H_ * sizeof(T)), 0, 0};
#ifndef __CCE_KT_TEST__
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
#endif
    DataCopyPad(gmOut_[nLoopIdx * H_ * curCoreHandleNumPerLoop_], outLocal, copyParams);
    outQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingBF16AllBias<T>::Process()
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
    PrepareData();
    for (int64_t n = 0; n < loopCount - 1; n++) {
        CopyIn(n, curCoreHandleNumPerLoop_);
        Compute(n, curCoreHandleNumPerLoop_);
        CopyOut(n, curCoreHandleNumPerLoop_);
    }

    CopyIn(loopCount - 1, tailLoopBlock);
    Compute(loopCount - 1, tailLoopBlock);
    CopyOut(loopCount - 1, tailLoopBlock);
}

} // namespace MoeFinalizeRouting
#endif // MOE_FINALIZE_ROUTING_BF16_ALL_BIAS
