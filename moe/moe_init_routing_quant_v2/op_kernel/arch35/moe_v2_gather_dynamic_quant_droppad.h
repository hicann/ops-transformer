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
 * \file moe_v2_gather_dynamic_quant_droppad.h
 * \brief
 */
#ifndef MOE_V2_GATHER_DYNAMIC_QUANT_DROPPAD_H
#define MOE_V2_GATHER_DYNAMIC_QUANT_DROPPAD_H

#include "moe_init_routing_quant_v2_arch35_common.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;

template <typename T>
class MoeV2GatherDynamicQuantDroppad {
public:
    __aicore__ inline MoeV2GatherDynamicQuantDroppad(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR dynamicQuantScale, GM_ADDR workspace, const TilingData* tilingData,
                                TPipe* tPipe);
    __aicore__ inline void Process();

private:
    template <typename TilingData>
    __aicore__ inline void InitTilingData(const TilingData* tilingData);
    __aicore__ inline void CopyInExpandedRowIdx(int64_t progress);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyOutXQuant1H(int64_t progress);
    __aicore__ inline void CopyOutXQuantEH(int64_t progress);
    __aicore__ inline void Compute(LocalTensor<float>& smoothLocal);
    __aicore__ inline void CopyOutPartialXQuantEH(int64_t progress);
    __aicore__ inline void CopyOutPartialXQuant1H(int64_t progress);
    __aicore__ inline float ComputeMax(LocalTensor<float>& inLocal, LocalTensor<float>& tempLocal,
                                       LocalTensor<float>& dynamicQuantLocal, int32_t srcIdx, int32_t expertIdx,
                                       int64_t j);
    __aicore__ inline void ComputeScale(LocalTensor<float>& inLocal, LocalTensor<float>& tempLocal, float scaleTemp,
                                        int64_t dstIndex, int64_t j);
    __aicore__ inline void PreClearOutput();

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputXInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> smoothInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> calcQueue_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> quantSmoothGm_;
    GlobalTensor<float> dynamicQuantScaleGm_;
    GlobalTensor<float> quantSrcGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> oriExpertIdGm_;

    MoeV2Arch35GatherOutComputeTilingData gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t totalLength_;
    int64_t activateRows_;
    int64_t coreRows_;
    int64_t currentLoopRows_;
    int64_t currentLoopRowsAlign_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t colsTileLength_;
    int64_t colLoops_;
    int64_t perLoopCols_;
    int64_t perLoopColsAlign_;
    int64_t lastLoopCols_;
    int64_t dropPadMode_;
    int64_t smoothType_;
    int64_t expertNum_;
    int64_t indicesOffset_;
};

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::CopyInExpandedRowIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams); // 第一轮sort的结果
    DataCopyPad(indicesLocal[currentLoopRowsAlign_], oriExpertIdGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::Compute(LocalTensor<float>& smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();
    LocalTensor<float> tempLocal = calcQueue_.AllocTensor<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    LocalTensor<float> dynamicQuantLocal = scaleOutQueue_.AllocTensor<float>();
    if constexpr (!IsSameType<T, float>::value) {
        Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign_], RoundMode::CAST_NONE, cols_);
    }
    if (smoothType_ != 0) {
        Mul(inLocal, inLocal, smoothLocal, cols_);
    }
    Abs(tempLocal, inLocal, cols_);
    ReduceMax(dynamicQuantLocal, tempLocal, tempLocal, cols_);
    float maxValue = dynamicQuantLocal.GetValue(0) / 127.0f;
    Duplicate<float>(dynamicQuantLocal, maxValue, 8);
    Duplicate<float>(tempLocal, maxValue, cols_);
    AscendC::PipeBarrier<PIPE_V>();
    Div(tempLocal, inLocal, tempLocal, cols_);
    Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, cols_);
    Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, cols_);
    calcQueue_.FreeTensor(tempLocal);
    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(dynamicQuantLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::CopyOutXQuant1H(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();

    int64_t initialRow = gatherOutTilingData_.perCoreRows * blockIdx_ + perLoopRows_ * progress;
    int64_t curLoopRow = 0;
    int64_t currentLoopStartRow = initialRow / k_;
    int64_t currentLoopLastRow = (initialRow + currentLoopRows_ - 1) / k_;
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(cols_ * sizeof(int8_t)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(cols_ * sizeof(float)), 0, 0, 0};

    LocalTensor<float> smoothLocal;
    if (smoothType_ == 1) {
        smoothLocal = smoothInQueue_.AllocTensor<float>();
        DataCopyPad(smoothLocal, quantSmoothGm_, smoothParams, {false, 0, 0, 0});
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();
    }

    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, inputXGm_[row * cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[perLoopColsAlign_], inputXGm_[row * cols_], copyInParams, {false, 0, 0, 0});
        }

        inputXInQueue_.EnQue<T>(inLocal);

        // 计算quant
        Compute(smoothLocal);
        LocalTensor<float> quantScaleLocal = scaleOutQueue_.DeQue<float>();
        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        while (curLoopRow < currentLoopRows_ && initialRow / k_ == row) {
            int32_t outIndex = indicesLocal.GetValue(curLoopRow);
            curLoopRow++;
            initialRow++;
            if (outIndex == -1) {
                continue;
            }
            DataCopyPad(expandedXGm_[outIndex * cols_], outLocal, copyOutParams);
            DataCopyPad(dynamicQuantScaleGm_[outIndex], quantScaleLocal, {1, 4, 0, 0, 0});
        }
        inputXInQueue_.FreeTensor(inLocal);
        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(quantScaleLocal);
    }
    if (smoothType_ == 1) {
        smoothInQueue_.FreeTensor(smoothLocal);
    }
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::CopyOutXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    MoeInitRoutingQuantV2::SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(int8_t)), 0, 0, 0};

    int64_t rowOffset = gatherOutTilingData_.perCoreRows * blockIdx_ + perLoopRows_ * progress;
    for (int64_t i = 0; i < currentLoopRows_; i++) {
        int32_t outIndex = indicesLocal.GetValue(i);
        if (outIndex < 0) {
            continue;
        }
        int32_t xRowIndex = (rowOffset + i) / k_;
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i);
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, inputXGm_[xRowIndex * cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[perLoopColsAlign_], inputXGm_[xRowIndex * cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);
        DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * cols_], smoothParams, {false, 0, 0, 0});
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();

        Compute(smoothLocal);
        LocalTensor<float> quantScaleLocal = scaleOutQueue_.DeQue<float>();
        DataCopyPad(dynamicQuantScaleGm_[outIndex], quantScaleLocal, {1, 4, 0, 0, 0});
        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        DataCopyPad(expandedXGm_[outIndex * cols_], outLocal, copyOutParams);
        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(quantScaleLocal);
        inputXInQueue_.FreeTensor(inLocal);
        smoothInQueue_.FreeTensor(smoothLocal);
    }

    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline float MoeV2GatherDynamicQuantDroppad<T>::ComputeMax(LocalTensor<float>& inLocal,
                                                                      LocalTensor<float>& tempLocal,
                                                                      LocalTensor<float>& dynamicQuantLocal,
                                                                      int32_t srcIdx, int32_t expertIdx, int64_t j)
{
    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    DataCopyExtParams intriParamsT{1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams intriParamsFp32{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    if constexpr (!IsSameType<T, float>::value) {
        DataCopyPad(inLocal.ReinterpretCast<T>()[perLoopColsAlign_], inputXGm_[srcIdx * cols_ + j * perLoopCols_],
                    intriParamsT, {false, 0, 0, 0});
    } else {
        DataCopyPad(inLocal, inputXGm_[srcIdx * cols_ + j * perLoopCols_], intriParamsT, {false, 0, 0, 0});
    }

    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();
    if (smoothType_ != 0) {
        DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * cols_ + j * perLoopCols_], intriParamsFp32,
                    {false, 0, 0, 0});
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();
    }
    if constexpr (!IsSameType<T, float>::value) {
        Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign_], RoundMode::CAST_NONE, colsTileLength_);
    }
    if (smoothType_ != 0) {
        Mul(inLocal, inLocal, smoothLocal, colsTileLength_);
    }
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    Abs(tempLocal, inLocal, colsTileLength_);
    ReduceMax(dynamicQuantLocal[8], tempLocal, tempLocal, colsTileLength_);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(quantSrcGm_[j * perLoopCols_], inLocal, intriParamsFp32);
    smoothInQueue_.FreeTensor(smoothLocal);
    MoeInitRoutingQuantV2::SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    return dynamicQuantLocal.GetValue(8);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::ComputeScale(LocalTensor<float>& inLocal,
                                                                       LocalTensor<float>& tempLocal, float scaleTemp,
                                                                       int64_t dstIndex, int64_t j)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(int8_t)), 0, 0, 0};

    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();

    DataCopyPad(inLocal, quantSrcGm_[j * perLoopCols_], copyInParams, {false, 0, 0, 0});
    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    Duplicate<float>(tempLocal, scaleTemp, colsTileLength_);
    Div(tempLocal, inLocal, tempLocal, colsTileLength_);
    Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, colsTileLength_);
    Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, colsTileLength_);

    inputXOutQueue_.EnQue(outLocal);
    outLocal = inputXOutQueue_.DeQue<int8_t>();
    DataCopyPad(expandedXGm_[dstIndex * cols_ + j * perLoopCols_], outLocal, copyOutParams);

    inputXOutQueue_.FreeTensor(outLocal);
    MoeInitRoutingQuantV2::SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::CopyOutPartialXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    MoeInitRoutingQuantV2::SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    int64_t rowOffset = gatherOutTilingData_.perCoreRows * blockIdx_ + perLoopRows_ * progress;
    for (int64_t i = 0; i < currentLoopRows_; i++) {
        int32_t xRowIndex = (rowOffset + i) / k_;
        int32_t outIndex = indicesLocal.GetValue(i);

        if (outIndex < 0) {
            continue;
        }

        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i);
        LocalTensor<float> inLocal = inputXInQueue_.AllocTensor<float>();
        LocalTensor<float> tempLocal = calcQueue_.AllocTensor<float>();
        LocalTensor<float> quantScaleLocal = scaleOutQueue_.AllocTensor<float>();

        uint32_t tmp = INF;
        float reduceMax = *((float*)&tmp);
        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            float tileMax = ComputeMax(inLocal, tempLocal, quantScaleLocal, xRowIndex, expertIdx, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / 127.0f;
        Duplicate<float>(quantScaleLocal, scaleTemp, 8);
        scaleOutQueue_.EnQue(quantScaleLocal);
        quantScaleLocal = scaleOutQueue_.DeQue<float>();

        DataCopyPad(dynamicQuantScaleGm_[outIndex], quantScaleLocal, {1, 4, 0, 0, 0});

        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }

            ComputeScale(inLocal, tempLocal, scaleTemp, outIndex, j);
        }

        inputXInQueue_.FreeTensor(inLocal);
        calcQueue_.FreeTensor(tempLocal);
        scaleOutQueue_.FreeTensor(quantScaleLocal);
    }

    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::CopyOutPartialXQuant1H(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();

    int64_t initialRow = gatherOutTilingData_.perCoreRows * blockIdx_ + perLoopRows_ * progress;
    int64_t curLoopRow = 0;

    int64_t currentLoopStartRow = initialRow / k_;
    int64_t currentLoopLastRow = (initialRow + currentLoopRows_ - 1) / k_;

    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
        LocalTensor<float> inLocal = inputXInQueue_.AllocTensor<float>();
        LocalTensor<float> tempLocal = calcQueue_.AllocTensor<float>();
        LocalTensor<float> quantScaleLocal = scaleOutQueue_.AllocTensor<float>();

        uint32_t tmp = INF;
        float reduceMax = *((float*)&tmp);
        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }

            float tileMax = ComputeMax(inLocal, tempLocal, quantScaleLocal, row, 0, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / 127.0f;
        Duplicate<float>(quantScaleLocal, scaleTemp, 8);
        scaleOutQueue_.EnQue(quantScaleLocal);
        quantScaleLocal = scaleOutQueue_.DeQue<float>();

        while (curLoopRow < currentLoopRows_ && initialRow / k_ == row) {
            int32_t outIndex = indicesLocal.GetValue(curLoopRow);
            curLoopRow++;
            initialRow++;
            if (outIndex == -1) {
                continue;
            }
            DataCopyPad(dynamicQuantScaleGm_[outIndex], quantScaleLocal, {1, 4, 0, 0, 0});
            for (int64_t j = 0; j < colLoops_; j++) {
                colsTileLength_ = perLoopCols_;
                if (j == colLoops_ - 1) {
                    colsTileLength_ = lastLoopCols_;
                }
                ComputeScale(inLocal, tempLocal, scaleTemp, outIndex, j);
            }
        }
        inputXInQueue_.FreeTensor(inLocal);
        calcQueue_.FreeTensor(tempLocal);
        scaleOutQueue_.FreeTensor(quantScaleLocal);
    }

    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
template <typename TilingData>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::InitTilingData(const TilingData* tilingData)
{
    gatherOutTilingData_ = tilingData->gatherOutComputeParamsOp;

    needCoreNum_ = gatherOutTilingData_.needCoreNum;
    activateRows_ = gatherOutTilingData_.activateRows;
    cols_ = tilingData->cols;
    n_ = tilingData->n;
    k_ = tilingData->k;
    totalLength_ = tilingData->n * tilingData->k;
    dropPadMode_ = tilingData->dropPadMode;
    smoothType_ = tilingData->smoothType;
    expertNum_ = tilingData->expertNum;
    totalLength_ = tilingData->n * tilingData->k;

    activateRows_ = tilingData->expertCapacity * tilingData->expertNum;

    if (blockIdx_ == gatherOutTilingData_.needCoreNum - 1) {
        coreRows_ = gatherOutTilingData_.lastCoreRows;
        perLoopRows_ = gatherOutTilingData_.lastCorePerLoopRows;
        lastLoopRows_ = gatherOutTilingData_.lastCoreLastLoopRows;
        rowLoops_ = gatherOutTilingData_.lastCoreLoops;
    } else {
        coreRows_ = gatherOutTilingData_.perCoreRows;
        perLoopRows_ = gatherOutTilingData_.perCorePerLoopRows;
        lastLoopRows_ = gatherOutTilingData_.perCoreLastLoopRows;
        rowLoops_ = gatherOutTilingData_.perCoreLoops;
    }
    perLoopCols_ = gatherOutTilingData_.perLoopCols;
    lastLoopCols_ = gatherOutTilingData_.lastLoopCols;
    colLoops_ = gatherOutTilingData_.colLoops;
    perLoopColsAlign_ = MoeInitRoutingQuantV2::Align(perLoopCols_, sizeof(T));
}

template <typename T>
template <typename TilingData>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::Init(GM_ADDR inputX, GM_ADDR quantSmooth,
                                                               GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                               GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                                               const TilingData* tilingData, TPipe* tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitTilingData(tilingData);

    inputXGm_.SetGlobalBuffer((__gm__ T*)inputX);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t*)expandedX);

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx + blockIdx_ * gatherOutTilingData_.perCoreRows,
                                      MoeInitRoutingQuantV2::Align(coreRows_, sizeof(int32_t)));

    quantSmoothGm_.SetGlobalBuffer((__gm__ float*)quantSmooth);
    dynamicQuantScaleGm_.SetGlobalBuffer((__gm__ float*)dynamicQuantScale);

    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + blockIdx_ * gatherOutTilingData_.perCoreRows,
                                         MoeInitRoutingQuantV2::Align(coreRows_, sizeof(int32_t)));
    sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)workspace +
                                        MoeInitRoutingQuantV2::Align(totalLength_, sizeof(int32_t)) +
                                        blockIdx_ * gatherOutTilingData_.perCoreRows,
                                    MoeInitRoutingQuantV2::Align(coreRows_, sizeof(int32_t)));
    if (cols_ > 1) {
        quantSrcGm_.SetGlobalBuffer((__gm__ float*)workspace +
                                        MoeInitRoutingQuantV2::Align(totalLength_, sizeof(int32_t)) * 4 + expertNum_ +
                                        blockIdx_ * cols_,
                                    cols_ * sizeof(float));
    }

    oriExpertIdGm_.SetGlobalBuffer((__gm__ int32_t*)workspace +
                                       MoeInitRoutingQuantV2::Align(totalLength_, sizeof(int32_t)) * 3 + expertNum_ +
                                       blockIdx_ * gatherOutTilingData_.perCoreRows,
                                   cols_ * sizeof(float));

    currentLoopRowsAlign_ = MoeInitRoutingQuantV2::Align(perLoopRows_, sizeof(int32_t));

    int64_t perLoopColsAlignBytes = MoeInitRoutingQuantV2::AlignBytes(perLoopCols_, sizeof(T));
    perLoopColsAlignBytes = MoeInitRoutingQuantV2::Max(int64_t(perLoopColsAlignBytes * sizeof(float) / sizeof(T)),
                                                       int64_t(BLOCK_BYTES + BLOCK_BYTES));

    pipe_->InitBuffer(expandRowIdxInQueue_, 1, 2 * MoeInitRoutingQuantV2::AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(inputXInQueue_, BUFFER_NUM, perLoopColsAlignBytes);
    pipe_->InitBuffer(smoothInQueue_, BUFFER_NUM, MoeInitRoutingQuantV2::AlignBytes(perLoopCols_, sizeof(float)));
    pipe_->InitBuffer(calcQueue_, 1, MoeInitRoutingQuantV2::AlignBytes(perLoopCols_, sizeof(float)));
    pipe_->InitBuffer(inputXOutQueue_, 1, MoeInitRoutingQuantV2::AlignBytes(perLoopCols_, sizeof(int8_t)));
    pipe_->InitBuffer(scaleOutQueue_, 1, BLOCK_BYTES + BLOCK_BYTES);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::PreClearOutput()
{
    // expendedX: activateRows_ * cols_ (int8) 分片清零
    int64_t totalX = activateRows_ * cols_;
    int64_t chunkX = (totalX + needCoreNum_ - 1) / needCoreNum_;
    int64_t begX = blockIdx_ * chunkX;
    int64_t lenX = totalX - begX;
    if (lenX > chunkX) {
        lenX = chunkX;
    }
    if (lenX > 0) {
        GlobalTensor<uint8_t> xGm;
        xGm.SetGlobalBuffer((__gm__ uint8_t*)expandedXGm_.GetPhyAddr() + begX, static_cast<uint64_t>(lenX));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        InitGlobalMemory(xGm, static_cast<uint64_t>(lenX), static_cast<uint8_t>(0));
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
    // dynamicQuantScale: activateRows_个 float分片清零
    int64_t totalS = activateRows_;
    int64_t chunkS = (totalS + needCoreNum_ - 1) / needCoreNum_;
    int64_t begS = blockIdx_ * chunkS;
    int64_t lenS = totalS - begS;
    if (lenS > chunkS) {
        lenS = chunkS;
    }
    if (lenS > 0) {
        GlobalTensor<float> sGm;
        sGm.SetGlobalBuffer((__gm__ float*)dynamicQuantScaleGm_.GetPhyAddr() + begS, static_cast<uint64_t>(lenS));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        InitGlobalMemory(sGm, static_cast<uint64_t>(lenS), 0.0f);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDroppad<T>::Process()
{
    // 1) 先按核分片一次性清零 expandedX 与dynamicQuantScale(pad/空槽必须为 0)
    if (blockIdx_ < needCoreNum_) {
        PreClearOutput();
    }
    // 2) 全核屏障：所有核清零完成后再写有效槽，避免跨loop/跨核覆盖有效值
    SyncAll();
    // 3) 逐 loop 只写有效槽
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        if (colLoops_ > 1) { // 一行无法全载，需要workspace
            if (smoothType_ == 2) {
                for (int64_t loop = 0; loop < rowLoops_; loop++) {
                    if (loop == rowLoops_ - 1) {
                        currentLoopRows_ = lastLoopRows_;
                    }
                    CopyInExpandedExpertIdx(loop);
                    CopyOutPartialXQuantEH(loop);
                }
            } else {
                for (int64_t loop = 0; loop < rowLoops_; loop++) {
                    if (loop == rowLoops_ - 1) {
                        currentLoopRows_ = lastLoopRows_;
                    }
                    CopyInExpandedRowIdx(loop);
                    CopyOutPartialXQuant1H(loop);
                }
            }
        } else { // 一行可以全载
            if (smoothType_ == 2) {
                for (int64_t loop = 0; loop < rowLoops_; loop++) {
                    if (loop == rowLoops_ - 1) {
                        currentLoopRows_ = lastLoopRows_;
                    }
                    CopyInExpandedExpertIdx(loop);
                    CopyOutXQuantEH(loop);
                }
            } else {
                for (int64_t loop = 0; loop < rowLoops_; loop++) {
                    if (loop == rowLoops_ - 1) {
                        currentLoopRows_ = lastLoopRows_;
                    }
                    CopyInExpandedRowIdx(loop);
                    CopyOutXQuant1H(loop);
                }
            }
        }
    }
}
} // namespace MoeInitRoutingQuantV2
#endif // MOE_V2_GATHER_DYNAMIC_QUANT_DROPPAD_H
