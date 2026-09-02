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
 * \file moe_finalize_routing_fp_cuth_k2_optimized.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_FP_CUTH_K_TWO_OPTIMIZED
#define MOE_FINALIZE_ROUTING_FP_CUTH_K_TWO_OPTIMIZED

#include "moe_finalize_routing_common.h"
#include "kernel_tiling/kernel_tiling.h"

constexpr int64_t BUFFER_NUM_OPTIMIZED = 2;
namespace MoeFinalizeRouting {

using namespace AscendC;
template <typename T>
class MoeFinalizeRoutingFpCuthK2Optimized {
public:
    __aicore__ inline MoeFinalizeRoutingFpCuthK2Optimized(){};
    __aicore__ inline void Init(GM_ADDR expandedPermutedRows, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expandedSrcToDstRow, GM_ADDR expertForSourceRow, GM_ADDR out,
                                GM_ADDR workspace, const MoeFinalizeRoutingTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const MoeFinalizeRoutingTilingData *tilingData);
    __aicore__ inline void CkechColAlignment();
    __aicore__ inline void CopyIn(int64_t nLoopIdx, int64_t lineNumInCurrentLoop, int64_t bias, int64_t dataLen,
                                  bool isPadH, int64_t rightPaddingH);
    __aicore__ inline void Compute(int64_t nLoopIdx, int64_t lineNumInCurrentLoop, int64_t bias, int64_t dataLen,
                                   bool isPadH, int64_t rightPaddingH);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t lineNumInCurrentLoop, int64_t bias, int64_t dataLen);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);
    __aicore__ inline void PrepareData();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM_OPTIMIZED> skip1Skip2ScalesQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_OPTIMIZED> expertForSourceRowQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_OPTIMIZED> outQueue_;

    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb0_;
    TBuf<QuePosition::VECCALC> biasBufDb0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb1_;
    TBuf<QuePosition::VECCALC> biasBufDb1_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> expandedSrcToDstRowBuff_;

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
    int64_t normalH_{0};
    int64_t unnormalH_{0};
    int64_t cutNumH_{0};

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

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadNormalH_ = false;
    bool isPadUnnormalH_ = false;
    bool isPadK_ = false;
    bool isPadKInt32_ = false;
    uint16_t rightPaddingNormalH_ = 0;
    uint16_t rightPaddingUnnormalH_ = 0;
    uint16_t rightPaddingK_ = 0;
    uint16_t rightPaddingKInt32_ = 0;

    int64_t curHandleNumTailLoop_{0};
    int64_t curNormLoopCount_{0};
    int64_t curTotalLoopCount_{0};

    bool isPadSourceToDstRow_ = false;
    uint16_t rightPaddingSourceToDstRow_ = 0;
    int64_t curCoreHandleNum_ = 0;
};

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFpCuthK2Optimized<T>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingFpCuthK2Optimized<T>::PadProcessT(int64_t param)
{
    return onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCuthK2Optimized<T>::ParseTilingData(
    const MoeFinalizeRoutingTilingData *tilingData)
{
    skip2IsNull_ = tilingData->skip2IsNull;
    BS_ = tilingData->totalRowNum;
    H_ = tilingData->H;
    K_ = tilingData->K;
    E_ = tilingData->biasRowNum;
    normalH_ = tilingData->normalH;
    unnormalH_ = tilingData->unnormalH;
    cutNumH_ = tilingData->hSliceNum;

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
__aicore__ inline void MoeFinalizeRoutingFpCuthK2Optimized<T>::CkechColAlignment()
{
    if (normalH_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalH_ = true;
        rightPaddingNormalH_ = PadProcessT(normalH_);
    }

    if (unnormalH_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalH_ = true;
        rightPaddingUnnormalH_ = PadProcessT(unnormalH_);
    }

    if (K_ * sizeof(T) % ONE_BLK_SIZE) {
        isPadK_ = true;
        rightPaddingK_ = PadProcessT(K_);
    }

    if (K_ * INT32_BYTES % ONE_BLK_SIZE) {
        isPadKInt32_ = true;
        rightPaddingKInt32_ = PadProcessInt32(K_);
    }

    if (curCoreHandleNum_ * INT32_BYTES % ONE_BLK_SIZE) {
        isPadSourceToDstRow_ = true;
        rightPaddingSourceToDstRow_ = PadProcessInt32(curCoreHandleNum_);
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCuthK2Optimized<T>::Init(GM_ADDR expandedPermutedRows, GM_ADDR skip1,
                                                                    GM_ADDR skip2, GM_ADDR bias, GM_ADDR scales,
                                                                    GM_ADDR expandedSrcToDstRow,
                                                                    GM_ADDR expertForSourceRow, GM_ADDR out,
                                                                    GM_ADDR workspace,
                                                                    const MoeFinalizeRoutingTilingData *tilingData)
{
    // init tiling data
    ParseTilingData(tilingData);

    if (GetBlockIdx() + 1 == usedCoreNum_) {
        curCoreHandleNumPerLoop_ = tailCoreHandleNumPerLoop_;
        curHandleNumTailLoop_ = tailCoreHandleNumTailLoop_;
        curNormLoopCount_ = (tailCoreLoopNum_ - 1) * (cutNumH_ + 1);
        curTotalLoopCount_ = tailCoreLoopNum_ * (cutNumH_ + 1);
        curCoreHandleNum_ = tailCoreHandleNum_;
    } else {
        curCoreHandleNumPerLoop_ = normalCoreHandleNumPerLoop_;
        curHandleNumTailLoop_ = normalCoreHandleNumTailLoop_;
        curNormLoopCount_ = (normalCoreLoopNum_ - 1) * (cutNumH_ + 1);
        curTotalLoopCount_ = normalCoreLoopNum_ * (cutNumH_ + 1);
        curCoreHandleNum_ = normalCoreHandleNum_;
    }

    // 检查要处理的列数是否对齐以及应该如何对齐
    CkechColAlignment();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = GetBlockIdx() * normalCoreHandleNum_ * H_;
    gmSkip1_.SetGlobalBuffer((__gm__ T *)skip1 + inputSkipIdx_, normalCoreHandleNum_ * H_);

    if (skip2IsNull_ == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T *)skip2 + inputSkipIdx_, normalCoreHandleNum_ * H_);
        pipe.InitBuffer(skip1Skip2ScalesQueue_, BUFFER_NUM_OPTIMIZED,
                        normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_) * sizeof(T) * 2 +
                            normalCoreHandleNumPerLoop_ * AlignmentProcess(K_) * sizeof(T));
    } else {
        pipe.InitBuffer(skip1Skip2ScalesQueue_, BUFFER_NUM_OPTIMIZED,
                        normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_) * sizeof(T) +
                            normalCoreHandleNumPerLoop_ * AlignmentProcess(K_) * sizeof(T));
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
    pipe.InitBuffer(expertForSourceRowQueue_, BUFFER_NUM_OPTIMIZED,
                    normalCoreHandleNumPerLoop_ * Int32AlignmentProcess(K_) * sizeof(int32_t));
    pipe.InitBuffer(outQueue_, BUFFER_NUM_OPTIMIZED,
                    normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_) * sizeof(T));

    pipe.InitBuffer(expandedPermutedRowsBufDb0_, AlignmentProcess(normalH_) * sizeof(T));
    pipe.InitBuffer(biasBufDb0_, AlignmentProcess(normalH_) * sizeof(T));

    pipe.InitBuffer(expandedPermutedRowsBufDb1_, AlignmentProcess(normalH_) * sizeof(T));
    pipe.InitBuffer(biasBufDb1_, AlignmentProcess(normalH_) * sizeof(T));
    pipe.InitBuffer(tmpBuf_, AlignmentProcess(normalH_) * sizeof(T));

    pipe.InitBuffer(expandedSrcToDstRowBuff_, Int32AlignmentProcess(curCoreHandleNum_) * K_ * sizeof(int32_t));
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCuthK2Optimized<T>::PrepareData()
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
__aicore__ inline void MoeFinalizeRoutingFpCuthK2Optimized<T>::CopyIn(int64_t nLoopIdx, int64_t lineNumInCurrentLoop,
                                                                      int64_t bias, int64_t dataLen, bool isPadH,
                                                                      int64_t rightPaddingH)
{
    LocalTensor<int32_t> expertForSourceRowLocal = expertForSourceRowQueue_.AllocTensor<int32_t>();

    LocalTensor<T> skip1Skip2ScalesLocal = skip1Skip2ScalesQueue_.AllocTensor<T>();

    // ---------------------------- [Expert] -------------------------------
    DataCopyParams copyParamsExpert{static_cast<uint16_t>(lineNumInCurrentLoop),
                                    static_cast<uint16_t>(K_ * sizeof(int32_t)), 0, 0};
    DataCopyPadParams padParamsExpert{isPadKInt32_, 0, static_cast<uint8_t>(rightPaddingKInt32_), 0};
    DataCopyPad(expertForSourceRowLocal,
                gmExpertForSourceRow_[nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_ * K_], copyParamsExpert,
                padParamsExpert);

    // ---------------------------- [Scales] -------------------------------
    DataCopyParams copyParamsScales{static_cast<uint16_t>(lineNumInCurrentLoop), static_cast<uint16_t>(K_ * sizeof(T)),
                                    0, 0};
    DataCopyPadParams padParamsScales{isPadK_, 0, static_cast<uint8_t>(rightPaddingK_), 0};
    DataCopyPad(skip1Skip2ScalesLocal[normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_)],
                gmScales_[nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_ * K_], copyParamsScales,
                padParamsScales);

    expertForSourceRowQueue_.EnQue(expertForSourceRowLocal);

    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

    // ---------------------------- [Skip] -------------------------------
    DataCopyParams copyParamsSkip{static_cast<uint16_t>(lineNumInCurrentLoop),
                                  static_cast<uint16_t>(dataLen * sizeof(T)),
                                  static_cast<uint16_t>((H_ - dataLen) * sizeof(T)), 0};
    DataCopyPadParams padParamsSkip{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    DataCopyPad(skip1Skip2ScalesLocal[0], gmSkip1_[nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_ * H_ + bias],
                copyParamsSkip, padParamsSkip);

    if (skip2IsNull_ == 0) {
        DataCopyPad(skip1Skip2ScalesLocal[normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_) +
                                          normalCoreHandleNumPerLoop_ * AlignmentProcess(K_)],
                    gmSkip2_[nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_ * H_ + bias], copyParamsSkip,
                    padParamsSkip);
    }

    skip1Skip2ScalesQueue_.EnQue(skip1Skip2ScalesLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCuthK2Optimized<T>::Compute(int64_t nLoopIdx, int64_t lineNumInCurrentLoop,
                                                                       int64_t bias, int64_t dataLen, bool isPadH,
                                                                       int64_t rightPaddingH)
{
    LocalTensor<int32_t> expertForSourceRowLocal = expertForSourceRowQueue_.DeQue<int32_t>();
    LocalTensor<T> skip1Skip2ScalesLocal = skip1Skip2ScalesQueue_.DeQue<T>();
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    LocalTensor<T> expandedPermutedTmpUbDb0 = expandedPermutedRowsBufDb0_.Get<T>();
    LocalTensor<T> biasTmpUbDb0 = biasBufDb0_.Get<T>();

    LocalTensor<T> expandedPermutedTmpUbDb1 = expandedPermutedRowsBufDb1_.Get<T>();
    LocalTensor<T> biasTmpUbDb1 = biasBufDb1_.Get<T>();

    LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
    int64_t baseRowLine = nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_;

    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

    for (int64_t i = 0; i < lineNumInCurrentLoop; i++) {
        int64_t outRowIndex = i * AlignmentProcess(normalH_);
        for (int64_t j = 0; j < K_ / PARALLEL_NUM; j++) {
            /*******************************乒***********************************************/
            if (!((i == 0) && (j == 0))) {
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
            }
            int64_t expandedSrcToDstRowIndexDb0 =
                baseRowLine + i + PARALLEL_NUM * j * Int32AlignmentProcess(curCoreHandleNum_);

            int64_t expandedPermutedRowsIndexDb0 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);

            int64_t biasIndexDb0 = expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(K_) + PARALLEL_NUM * j);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            DataCopyPad(expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * H_ + bias],
                        copyParams, padParams);
            DataCopyPad(biasTmpUbDb0, gmBias_[biasIndexDb0 * H_ + bias], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            /*******************************乓***********************************************/
            if (!((i == 0) && (j == 0))) {
                WaitFlag<HardEvent::V_S>(EVENT_ID1);
            }
            int64_t expandedSrcToDstRowIndexDb1 =
                baseRowLine + i + (PARALLEL_NUM * j + 1) * Int32AlignmentProcess(curCoreHandleNum_);
            int64_t expandedPermutedRowsIndexDb1 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb1);
            int64_t biasIndexDb1 =
                expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(K_) + PARALLEL_NUM * j + 1);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID1);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
            DataCopyPad(expandedPermutedTmpUbDb1, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb1 * H_ + bias],
                        copyParams, padParams);
            DataCopyPad(biasTmpUbDb1, gmBias_[biasIndexDb1 * H_ + bias], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

            if (!((i == 0) && (j == 0))) {
                WaitFlag<HardEvent::V_S>(EVENT_ID2);
            }
            T scalesValDb0 = skip1Skip2ScalesLocal[normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_)].GetValue(
                i * AlignmentProcess(K_) + PARALLEL_NUM * j);
            SetFlag<HardEvent::S_V>(EVENT_ID2);

            if (!((i == 0) && (j == 0))) {
                WaitFlag<HardEvent::V_S>(EVENT_ID3);
            }
            T scalesValDb1 = skip1Skip2ScalesLocal[normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_)].GetValue(
                i * AlignmentProcess(K_) + PARALLEL_NUM * j + 1);
            SetFlag<HardEvent::S_V>(EVENT_ID3);

            if ((i == 0) && (j == 0)) {
                if (skip2IsNull_ == 0) {
                    Add(outLocal, skip1Skip2ScalesLocal,
                        skip1Skip2ScalesLocal[normalCoreHandleNumPerLoop_ * AlignmentProcess(normalH_) +
                                              normalCoreHandleNumPerLoop_ * AlignmentProcess(K_)],
                        lineNumInCurrentLoop * AlignmentProcess(dataLen));
                } else {
                    Adds(outLocal, skip1Skip2ScalesLocal, (T)0, lineNumInCurrentLoop * AlignmentProcess(dataLen));
                }
            }

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Add(skip1Skip2ScalesLocal, expandedPermutedTmpUbDb0, biasTmpUbDb0, dataLen);
            if (!((i == (lineNumInCurrentLoop - 1)) && (j == (K_ / PARALLEL_NUM - 1)))) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
            }
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::S_V>(EVENT_ID2);
            Muls(skip1Skip2ScalesLocal, skip1Skip2ScalesLocal, scalesValDb0, dataLen);
            if (!((i == (lineNumInCurrentLoop - 1)) && (j == (K_ / PARALLEL_NUM - 1)))) {
                SetFlag<HardEvent::V_S>(EVENT_ID2);
            }
            PipeBarrier<PIPE_V>();
            Add(outLocal[outRowIndex], outLocal[outRowIndex], skip1Skip2ScalesLocal, dataLen);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Add(tmpLocal, expandedPermutedTmpUbDb1, biasTmpUbDb1, dataLen);
            if (!((i == (lineNumInCurrentLoop - 1)) && (j == (K_ / PARALLEL_NUM - 1)))) {
                SetFlag<HardEvent::V_S>(EVENT_ID1);
            }
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::S_V>(EVENT_ID3);
            Muls(tmpLocal, tmpLocal, scalesValDb1, dataLen);
            if (!((i == (lineNumInCurrentLoop - 1)) && (j == (K_ / PARALLEL_NUM - 1)))) {
                SetFlag<HardEvent::V_S>(EVENT_ID3);
            }

            PipeBarrier<PIPE_V>();
            Add(outLocal[outRowIndex], outLocal[outRowIndex], tmpLocal, dataLen);
        }
    }

    expertForSourceRowQueue_.FreeTensor(expertForSourceRowLocal);
    skip1Skip2ScalesQueue_.FreeTensor(skip1Skip2ScalesLocal);
    outQueue_.EnQue(outLocal);
}
template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCuthK2Optimized<T>::CopyOut(int64_t nLoopIdx, int64_t lineNumInCurrentLoop,
                                                                       int64_t bias, int64_t dataLen)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{static_cast<uint16_t>(lineNumInCurrentLoop), static_cast<uint16_t>(dataLen * sizeof(T)),
                              0, static_cast<uint16_t>((H_ - dataLen) * sizeof(T))};
    DataCopyPad(gmOut_[nLoopIdx / (cutNumH_ + 1) * curCoreHandleNumPerLoop_ * H_ + bias], outLocal, copyParams);
    outQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingFpCuthK2Optimized<T>::Process()
{
    if (GetBlockIdx() >= usedCoreNum_) {
        return;
    }
    PrepareData();
    for (int64_t n = 0; n < curNormLoopCount_; n++) {
        bool isNormalH = (n + 1) % (cutNumH_ + 1) != 0;
        int64_t bias = isNormalH ? (n % (cutNumH_ + 1)) * normalH_ : cutNumH_ * normalH_;
        int64_t dataLen = isNormalH ? normalH_ : unnormalH_;
        int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
        bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;
        CopyIn(n, curCoreHandleNumPerLoop_, bias, dataLen, isPadH, rightPaddingH);
        Compute(n, curCoreHandleNumPerLoop_, bias, dataLen, isPadH, rightPaddingH);
        CopyOut(n, curCoreHandleNumPerLoop_, bias, dataLen);
    }
    for (int64_t n = curNormLoopCount_; n < curTotalLoopCount_; n++) {
        bool isNormalH = (n + 1) % (cutNumH_ + 1) != 0;
        int64_t bias = isNormalH ? (n % (cutNumH_ + 1)) * normalH_ : cutNumH_ * normalH_;
        int64_t dataLen = isNormalH ? normalH_ : unnormalH_;
        int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
        bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;

        CopyIn(n, curHandleNumTailLoop_, bias, dataLen, isPadH, rightPaddingH);
        Compute(n, curHandleNumTailLoop_, bias, dataLen, isPadH, rightPaddingH);
        CopyOut(n, curHandleNumTailLoop_, bias, dataLen);
    }
}
} // namespace MoeFinalizeRouting
#endif // MOE_FINALIZE_ROUTING_FP_CUTH_K_TWO_OPTIMIZED
