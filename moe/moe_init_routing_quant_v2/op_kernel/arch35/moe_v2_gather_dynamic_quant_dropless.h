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
 * \file moe_v2_gather_dynamic_quant_dropless.h
 * \brief
 */
#ifndef MOE_V2_GATHER_DYNAMIC_QUANT_DROPLESS_H
#define MOE_V2_GATHER_DYNAMIC_QUANT_DROPLESS_H

#include "moe_init_routing_quant_v2_arch35_common.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;

template <typename T>
class MoeV2GatherDynamicQuantDropless {
public:
    __aicore__ inline MoeV2GatherDynamicQuantDropless(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                const MoeInitRoutingQuantV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTilingData(const MoeInitRoutingQuantV2TilingData *tilingData);
    __aicore__ inline void CopyInExpandedRowIdx(int64_t progress);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyOutXQuantEH(int64_t progress);
    __aicore__ inline void CopyOutXQuant1H(int64_t progress);
    __aicore__ inline void CopyOutPartialXQuantEH(int64_t progress);
    __aicore__ inline void CopyOutPartialXQuant1H(int64_t progress);
    __aicore__ inline float ComputeMax(LocalTensor<float> &inLocal, LocalTensor<float> &tempLocal,
                                       LocalTensor<float> &dynamicQuantLocal, int32_t srcIdx, int32_t expertIdx,
                                       int64_t j);
    __aicore__ inline void ComputeScale(LocalTensor<float> &inLocal, LocalTensor<float> &tempLocal, float scaleTemp,
                                        int64_t dstIndex, int64_t j);
    __aicore__ inline void Compute(LocalTensor<float> &smoothLocal);
    __aicore__ inline void CopyInZeroIndices(int64_t progress);
    __aicore__ inline void CopyOutZero(int64_t progress);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputXInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> smoothInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> calcQueue_;
    TQue<QuePosition::VECOUT, 1> inputXOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;
    TQue<QuePosition::VECIN, 1> expandedRowIdxIndexCopyInQueue_;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<int8_t> expandedXGm_;
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<float> quantSmoothGm_;
    GlobalTensor<float> dynamicQuantScaleGm_;
    GlobalTensor<float> quantSrcGm_;
    GlobalTensor<int32_t> expandedExpertIdxGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expandedRowIdxIndexGm_;

    const InnerMoeV2GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t n_;
    int64_t k_;
    int64_t totalLength_;
    int64_t activateRows_;
    int64_t coreRows_;
    int64_t rowLoops_;
    int64_t currentLoopRows_;
    int64_t currentLoopRowsAlign_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t colsTileLength_;
    int64_t perLoopCols_;
    int64_t perLoopColsAlign_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t dropPadMode_;
    int64_t smoothType_;
    int64_t expertNum_;
    int64_t indicesOffset_;
};

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::CopyInExpandedRowIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, expandedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    DataCopyPad(indicesLocal[currentLoopRowsAlign_], expandedExpertIdxGm_[indicesOffset_], dataCopyParams,
                dataCopyPadParams);
    expandRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::CopyInZeroIndices(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> expandedRowIdxIndexLocal = expandedRowIdxIndexCopyInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>((currentLoopRows_ + 1) * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(expandedRowIdxIndexLocal, expandedRowIdxIndexGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    expandedRowIdxIndexCopyInQueue_.EnQue<int32_t>(expandedRowIdxIndexLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::CopyOutZero(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandedRowIdxIndexCopyInQueue_.DeQue<int32_t>();
    if (blockIdx_ == 0) {
        int32_t count = indicesLocal.GetValue(0);
        if (count > 0) {
            InitOutput(expandedXGm_[0], count * cols_, static_cast<int8_t>(0));
            InitOutput(dynamicQuantScaleGm_[0], count, static_cast<float>(0));
        }
    }
    for (int rowIdx = 0; rowIdx < currentLoopRows_; rowIdx++) {
        int32_t curIndex = indicesLocal.GetValue(rowIdx) + 1;
        if (activateRows_ <= curIndex) {
            break;
        }
        int32_t nextIndex = 0;
        if (blockIdx_ == gatherOutTilingData_->needCoreNum - 1 && progress == rowLoops_ - 1 &&
            rowIdx == currentLoopRows_ - 1) {
            nextIndex = activateRows_;
        } else {
            nextIndex = indicesLocal.GetValue(rowIdx + 1);
        }
        int32_t count = nextIndex - curIndex;
        if (count > 0) {
            InitOutput(expandedXGm_[curIndex * cols_], count * cols_, static_cast<int8_t>(0));
            InitOutput(dynamicQuantScaleGm_[curIndex], count, static_cast<float>(0));
        }
    }
    expandedRowIdxIndexCopyInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::Compute(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue_.DeQue<float>();

    LocalTensor<float> tempLocal = calcQueue_.AllocTensor<float>();
    LocalTensor<float> dynamicQuantLocal = scaleOutQueue_.AllocTensor<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();

    if constexpr (!IsSameType<T, float>::value) {
        Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign_], RoundMode::CAST_NONE, cols_);
    }

    if (smoothType_ != 0) {
        Mul(inLocal, inLocal, smoothLocal, cols_);
    }

    Abs(tempLocal, inLocal, cols_);
    ReduceMax(dynamicQuantLocal, tempLocal, tempLocal, cols_);

    float maxValue = dynamicQuantLocal.GetValue(0) / MAX_INT8;

    Duplicate<float>(dynamicQuantLocal, maxValue, 8);
    Duplicate<float>(tempLocal, maxValue, cols_);
    Div(tempLocal, inLocal, tempLocal, cols_);
    Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, cols_);
    Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, cols_);

    calcQueue_.FreeTensor(tempLocal);
    inputXOutQueue_.EnQue(outLocal);
    scaleOutQueue_.EnQue(dynamicQuantLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::CopyOutXQuant1H(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();

    int64_t initialRow = gatherOutTilingData_->perCoreRows * blockIdx_ + perLoopRows_ * progress;
    int64_t curLoopRow = 0;
    int64_t currentLoopStartRow = initialRow / k_;
    int64_t currentLoopLastRow = (initialRow + currentLoopRows_ - 1) / k_;
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(cols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(cols_ * sizeof(int8_t)), 0, 0, 0};
    DataCopyPadExtParams<float> smoothPadParams{false, 0, 0, 0};
    DataCopyPadExtParams<T> inputXPadParams{false, 0, 0, 0};

    LocalTensor<float> smoothLocal;
    if (smoothType_ == 1) {
        smoothLocal = smoothInQueue_.AllocTensor<float>();
        DataCopyPad(smoothLocal, quantSmoothGm_, smoothParams, smoothPadParams);
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();
    }

    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, inputXGm_[row * cols_], copyInParams, inputXPadParams);
        } else {
            DataCopyPad(inLocal[perLoopColsAlign_], inputXGm_[row * cols_], copyInParams, inputXPadParams);
        }

        inputXInQueue_.EnQue<T>(inLocal);

        // 计算quant
        Compute(smoothLocal);

        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        LocalTensor<float> quantScaleLocal = scaleOutQueue_.DeQue<float>();

        while (curLoopRow < currentLoopRows_ && initialRow / k_ == row) {
            int32_t outIndex = indicesLocal.GetValue(curLoopRow);
            curLoopRow++;
            initialRow++;
            if (outIndex == -1 || outIndex >= activateRows_) {
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
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::CopyOutXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    MoeInitRoutingQuantV2::SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(perLoopCols_ * sizeof(int8_t)), 0, 0, 0};

    int32_t lastExpertIdx = -1;
    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    int64_t rowOffset = gatherOutTilingData_->perCoreRows * blockIdx_ + perLoopRows_ * progress;
    for (int64_t i = 0; i < currentLoopRows_; i++) {
        if (rowOffset + i >= activateRows_) {
            break;
        }
        LocalTensor<T> inLocal = inputXInQueue_.AllocTensor<T>();
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i);

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[perLoopColsAlign_], inputXGm_[srcIdx / k_ * cols_], copyInParams, {false, 0, 0, 0});
        }
        inputXInQueue_.EnQue<T>(inLocal);

        if (expertIdx != lastExpertIdx) {
            DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * cols_], smoothParams, {false, 0, 0, 0});
            smoothInQueue_.EnQue(smoothLocal);
            smoothLocal = smoothInQueue_.DeQue<float>();
            lastExpertIdx = expertIdx;
        }

        Compute(smoothLocal);
        LocalTensor<float> quantScaleLocal = scaleOutQueue_.DeQue<float>();
        DataCopyPad(dynamicQuantScaleGm_[(rowOffset + i)], quantScaleLocal, {1, 4, 0, 0, 0});
        LocalTensor<int8_t> outLocal = inputXOutQueue_.DeQue<int8_t>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(expandedXGm_[(rowOffset + i) * cols_], outLocal, copyOutParams);

        inputXInQueue_.FreeTensor(inLocal);
        inputXOutQueue_.FreeTensor(outLocal);
        scaleOutQueue_.FreeTensor(quantScaleLocal);
    }

    smoothInQueue_.FreeTensor(smoothLocal);
    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline float MoeV2GatherDynamicQuantDropless<T>::ComputeMax(LocalTensor<float> &inLocal,
                                                                       LocalTensor<float> &tempLocal,
                                                                       LocalTensor<float> &dynamicQuantLocal,
                                                                       int32_t srcIdx, int32_t expertIdx, int64_t j)
{
    LocalTensor<float> smoothLocal = smoothInQueue_.AllocTensor<float>();
    DataCopyExtParams intriParamsT{1, static_cast<uint32_t>(colsTileLength_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<float> smoothPadParams{false, 0, 0, 0};
    DataCopyPadExtParams<T> inputXPadParams{false, 0, 0, 0};

    if constexpr (!IsSameType<T, float>::value) {
        DataCopyPad(inLocal.ReinterpretCast<T>()[perLoopColsAlign_], inputXGm_[srcIdx * cols_ + j * perLoopCols_],
                    intriParamsT, inputXPadParams);
    } else {
        DataCopyPad(inLocal, inputXGm_[srcIdx * cols_ + j * perLoopCols_], intriParamsT, inputXPadParams);
    }

    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();

    DataCopyExtParams intriParamsFp32{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};
    if (smoothType_ != 0) {
        DataCopyPad(smoothLocal, quantSmoothGm_[expertIdx * cols_ + j * perLoopCols_], intriParamsFp32,
                    smoothPadParams);
        smoothInQueue_.EnQue(smoothLocal);
        smoothLocal = smoothInQueue_.DeQue<float>();
    }

    if constexpr (!IsSameType<T, float>::value) {
        Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign_], RoundMode::CAST_NONE, colsTileLength_);
    }

    if (smoothType_ != 0) {
        Mul(inLocal, inLocal, smoothLocal, colsTileLength_);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
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
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::ComputeScale(LocalTensor<float> &inLocal,
                                                                        LocalTensor<float> &tempLocal, float scaleTemp,
                                                                        int64_t dstIndex, int64_t j)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> copyInPadParams{false, 0, 0, 0};

    LocalTensor<int8_t> outLocal = inputXOutQueue_.AllocTensor<int8_t>();
    DataCopyPad(inLocal, quantSrcGm_[j * perLoopCols_], copyInParams, copyInPadParams);
    inputXInQueue_.EnQue<float>(inLocal);
    inLocal = inputXInQueue_.DeQue<float>();
    Duplicate<float>(tempLocal, scaleTemp, colsTileLength_);
    Div(tempLocal, inLocal, tempLocal, colsTileLength_);
    Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, colsTileLength_);
    Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, colsTileLength_);

    inputXOutQueue_.EnQue(outLocal);
    outLocal = inputXOutQueue_.DeQue<int8_t>();
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLength_ * sizeof(int8_t)), 0, 0, 0};
    DataCopyPad(expandedXGm_[dstIndex * cols_ + j * perLoopCols_], outLocal, copyOutParams);

    inputXOutQueue_.FreeTensor(outLocal);
    MoeInitRoutingQuantV2::SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::CopyOutPartialXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();
    MoeInitRoutingQuantV2::SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    int64_t rowOffset = gatherOutTilingData_->perCoreRows * blockIdx_ + perLoopRows_ * progress;
    for (int64_t i = 0; i < currentLoopRows_; i++) {
        if (rowOffset + i >= activateRows_) {
            break;
        }
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign_ + i);

        uint32_t tmp = INF;
        LocalTensor<float> inLocal = inputXInQueue_.AllocTensor<float>();
        LocalTensor<float> tempLocal = calcQueue_.AllocTensor<float>();
        LocalTensor<float> quantScaleLocal = scaleOutQueue_.AllocTensor<float>();

        float reduceMax = *((float *)&tmp);
        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            float tileMax = ComputeMax(inLocal, tempLocal, quantScaleLocal, srcIdx / k_, expertIdx, j);

            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / MAX_INT8;
        Duplicate<float>(quantScaleLocal, scaleTemp, 8);
        scaleOutQueue_.EnQue(quantScaleLocal);
        quantScaleLocal = scaleOutQueue_.DeQue<float>();
        DataCopyPad(dynamicQuantScaleGm_[(rowOffset + i)], quantScaleLocal, {1, 4, 0, 0, 0});
        colsTileLength_ = perLoopCols_;
        for (int64_t j = 0; j < colLoops_; j++) {
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            ComputeScale(inLocal, tempLocal, scaleTemp, rowOffset + i, j);
        }

        inputXInQueue_.FreeTensor(inLocal);
        calcQueue_.FreeTensor(tempLocal);
        scaleOutQueue_.FreeTensor(quantScaleLocal);
    }

    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::CopyOutPartialXQuant1H(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue_.DeQue<int32_t>();

    int64_t initialRow = gatherOutTilingData_->perCoreRows * blockIdx_ + perLoopRows_ * progress;
    int64_t curLoopRow = 0;

    int64_t currentLoopStartRow = initialRow / k_;
    int64_t currentLoopLastRow = (initialRow + currentLoopRows_ - 1) / k_;

    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
        LocalTensor<float> inLocal = inputXInQueue_.AllocTensor<float>();
        LocalTensor<float> tempLocal = calcQueue_.AllocTensor<float>();
        LocalTensor<float> quantScaleLocal = scaleOutQueue_.AllocTensor<float>();

        uint32_t tmp = INF;
        float reduceMax = *((float *)&tmp);
        for (int64_t j = 0; j < colLoops_; j++) {
            colsTileLength_ = perLoopCols_;
            if (j == colLoops_ - 1) {
                colsTileLength_ = lastLoopCols_;
            }
            float tileMax = ComputeMax(inLocal, tempLocal, quantScaleLocal, row, 0, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / MAX_INT8;
        Duplicate<float>(quantScaleLocal, scaleTemp, 8);
        scaleOutQueue_.EnQue(quantScaleLocal);
        quantScaleLocal = scaleOutQueue_.DeQue<float>();

        while (curLoopRow < currentLoopRows_ && initialRow / k_ == row) {
            int32_t outIndex = indicesLocal.GetValue(curLoopRow);
            curLoopRow++;
            initialRow++;
            if (outIndex == -1 || outIndex >= activateRows_) {
                continue;
            }
            DataCopyPad(dynamicQuantScaleGm_[outIndex], quantScaleLocal, {1, 4, 0, 0, 0});
            colsTileLength_ = perLoopCols_;
            for (int64_t loop = 0; loop < colLoops_; loop++) {
                if (loop == colLoops_ - 1) {
                    colsTileLength_ = lastLoopCols_;
                }
                ComputeScale(inLocal, tempLocal, scaleTemp, outIndex, loop);
            }
        }
        inputXInQueue_.FreeTensor(inLocal);
        calcQueue_.FreeTensor(tempLocal);
        scaleOutQueue_.FreeTensor(quantScaleLocal);
    }

    expandRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::InitTilingData(
    const MoeInitRoutingQuantV2TilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    needCoreNum_ = gatherOutTilingData_->needCoreNum;
    activateRows_ = gatherOutTilingData_->activateRows;
    n_ = tilingData->n;
    k_ = tilingData->k;
    cols_ = tilingData->cols;
    totalLength_ = tilingData->n * tilingData->k;
    dropPadMode_ = tilingData->dropPadMode;
    smoothType_ = tilingData->smoothType;
    expertNum_ = tilingData->expertNum;
    totalLength_ = tilingData->n * tilingData->k;

    if (blockIdx_ == gatherOutTilingData_->needCoreNum - 1) {
        coreRows_ = gatherOutTilingData_->lastCoreRows;
        rowLoops_ = gatherOutTilingData_->lastCoreLoops;
        perLoopRows_ = gatherOutTilingData_->lastCorePerLoopRows;
        lastLoopRows_ = gatherOutTilingData_->lastCoreLastLoopRows;
    } else {
        coreRows_ = gatherOutTilingData_->perCoreRows;
        rowLoops_ = gatherOutTilingData_->perCoreLoops;
        perLoopRows_ = gatherOutTilingData_->perCorePerLoopRows;
        lastLoopRows_ = gatherOutTilingData_->perCoreLastLoopRows;
    }
    colLoops_ = gatherOutTilingData_->colLoops;
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    perLoopColsAlign_ = MoeInitRoutingQuantV2::Align(perLoopCols_, sizeof(T));
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::Init(GM_ADDR inputX, GM_ADDR quantSmooth,
                                                                GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                                GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                                                const MoeInitRoutingQuantV2TilingData *tilingData,
                                                                TPipe *tPipe)
{
    pipe_ = tPipe;
    InitTilingData(tilingData);

    inputXGm_.SetGlobalBuffer((__gm__ T *)inputX);
    expandedXGm_.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx + blockIdx_ * gatherOutTilingData_->perCoreRows,
                                      MoeInitRoutingQuantV2::Align(coreRows_, sizeof(int32_t)));

    quantSmoothGm_.SetGlobalBuffer((__gm__ float *)quantSmooth);
    dynamicQuantScaleGm_.SetGlobalBuffer((__gm__ float *)dynamicQuantScale);

    expandedExpertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + blockIdx_ * gatherOutTilingData_->perCoreRows,
                                         MoeInitRoutingQuantV2::Align(coreRows_, sizeof(int32_t)));
    sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)workspace +
                                        MoeInitRoutingQuantV2::Align(totalLength_, sizeof(int32_t)) +
                                        blockIdx_ * gatherOutTilingData_->perCoreRows,
                                    MoeInitRoutingQuantV2::Align(coreRows_, sizeof(int32_t)));
    if (cols_ > 1) {
        quantSrcGm_.SetGlobalBuffer((__gm__ float *)workspace +
                                        MoeInitRoutingQuantV2::Align(totalLength_, sizeof(int32_t)) * 3 + expertNum_ +
                                        blockIdx_ * cols_,
                                    cols_ * sizeof(float));
    }

    expandedRowIdxIndexGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(totalLength_, sizeof(int32_t)) * 2 +
                                               expertNum_ + blockIdx_ * gatherOutTilingData_->perCoreRows,
                                           0);

    currentLoopRowsAlign_ = MoeInitRoutingQuantV2::Align(perLoopRows_, sizeof(int32_t));

    int64_t perLoopColsAlignBytes = MoeInitRoutingQuantV2::AlignBytes(perLoopCols_, sizeof(T));
    perLoopColsAlignBytes = MoeInitRoutingQuantV2::Max(int64_t(perLoopColsAlignBytes * sizeof(float) / sizeof(T)),
                                                       int64_t(BLOCK_BYTES + BLOCK_BYTES));

    pipe_->InitBuffer(expandRowIdxInQueue_, BUFFER_NUM,
                      2 * MoeInitRoutingQuantV2::AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(inputXInQueue_, BUFFER_NUM, perLoopColsAlignBytes);
    pipe_->InitBuffer(smoothInQueue_, BUFFER_NUM, MoeInitRoutingQuantV2::AlignBytes(perLoopCols_, sizeof(float)));
    pipe_->InitBuffer(calcQueue_, 1, MoeInitRoutingQuantV2::AlignBytes(perLoopCols_, sizeof(float)));
    pipe_->InitBuffer(inputXOutQueue_, 1, MoeInitRoutingQuantV2::AlignBytes(perLoopCols_, sizeof(int8_t)));
    pipe_->InitBuffer(scaleOutQueue_, 1, BLOCK_BYTES + BLOCK_BYTES);
    pipe_->InitBuffer(expandedRowIdxIndexCopyInQueue_, 1,
                      MoeInitRoutingQuantV2::AlignBytes(perLoopRows_ + 1, sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void MoeV2GatherDynamicQuantDropless<T>::Process()
{
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
#endif // MOE_V2_GATHER_DYNAMIC_QUANT_DROPLESS_H
