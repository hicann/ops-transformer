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
 * \file moe_v2_gather_dynamic_quant.h
 * \brief
 */
#ifndef MOE_V2_GATHER_DYNAMIC_QUANT_H
#define MOE_V2_GATHER_DYNAMIC_QUANT_H

#include "inner/moe_init_routing_quant_v2_inner_common.h"

namespace MoeInitRoutingQuantV2 {
using namespace AscendC;
constexpr int32_t ELEM_PER_REP_FP32 = 64; // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr int32_t ONE_REPEAT_NUM = 256;   // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr int32_t ONE_BYTE_INT4_NUM = 2;  // int 8 / init4
constexpr int32_t BIT_OFFSET_REP = 6;     // ELEM_PER_REP_FP32 bit offset
constexpr uint64_t REM_OFFSET = 0x3f;     // ELEM_PER_REP_FP32 bit offset

template <typename T>
struct quantTypeValue {
    static constexpr int outQueDepth = 1;
};

// preload
template <>
struct quantTypeValue<int4b_t> {
    static constexpr int outQueDepth = 2;
};

__aicore__ inline void MulsCust(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                const LocalTensor<float> &src1, uint32_t count)
{
    uint64_t repsFp32 = count >> BIT_OFFSET_REP;       // 6 is cound / ELEM_PER_REP_FP32
    uint64_t offsetsFp32 = repsFp32 << BIT_OFFSET_REP; // 6 is repsFp32 * ELEM_PER_REP_FP32
    uint64_t remsFp32 = count & REM_OFFSET;            // 0x3f 63, count % ELEM_PER_REP_FP32
    PipeBarrier<PIPE_V>();
    Mul(dst, src0, src1, ELEM_PER_REP_FP32, repsFp32, {1, 1, 0, INT32_ONE_BLOCK_NUM, INT32_ONE_BLOCK_NUM, 0});
    PipeBarrier<PIPE_V>();
    Mul(dst[offsetsFp32], src0[offsetsFp32], src1, remsFp32, 1, {1, 1, 0, INT32_ONE_BLOCK_NUM, INT32_ONE_BLOCK_NUM, 0});
}
__aicore__ inline void ReduceMaxCust(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
                                     uint32_t count)
{
    uint64_t repsFp32 = count >> BIT_OFFSET_REP;       // 6 is cound / ELEM_PER_REP_FP32
    uint64_t offsetsFp32 = repsFp32 << BIT_OFFSET_REP; // 6 is repsFp32 * ELEM_PER_REP_FP32
    uint64_t remsFp32 = count & REM_OFFSET;            // 0x3f 63, count % ELEM_PER_REP_FP32

    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        PipeBarrier<PIPE_V>();
        Max(src_local, src_local[ELEM_PER_REP_FP32], src_local, ELEM_PER_REP_FP32, repsFp32 - 1,
            {1, 1, 1, 0, INT32_ONE_BLOCK_NUM, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        PipeBarrier<PIPE_V>();
        Max(src_local, src_local[offsetsFp32], src_local, remsFp32, 1, {1, 1, 1, 0, INT32_ONE_BLOCK_NUM, 0});
        PipeBarrier<PIPE_V>();
    }
    // 8 is rep stride
    uint32_t mask = repsFp32 > 0 ? ELEM_PER_REP_FP32 : count;
    PipeBarrier<PIPE_V>();
    WholeReduceMax(dst_local, src_local, mask, 1, INT32_ONE_BLOCK_NUM, 1, INT32_ONE_BLOCK_NUM);
}

template <typename T, typename quantType>
class MoeV2GatherDynamicQuant {
public:
    __aicore__ inline MoeV2GatherDynamicQuant(){};
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR quantSmooth, GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                const MoeInitRoutingQuantV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInExpandedRowIdx(int64_t progress);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyOutXQuant1H(int64_t progress);
    __aicore__ inline void CopyOutXQuantEH(int64_t progress);
    __aicore__ inline void OnceCopyOut(LocalTensor<int32_t> indicesLocal, int64_t row, int64_t &curLoopRow,
                                       int64_t &initialRow, DataCopyExtParams copyOutParams,
                                       DataCopyExtParams quantScaleParams);
    __aicore__ inline void OnceCopyIn(int64_t row, DataCopyExtParams copyInParams);

    __aicore__ inline void Compute(LocalTensor<float> &smoothLocal);
    __aicore__ inline void CopyOutPartialXQuantEH(int64_t progress);
    __aicore__ inline void CopyOutPartialXQuant1H(int64_t progress);
    __aicore__ inline float ComputeMax(LocalTensor<float> &inLocal, LocalTensor<float> &tempLocal,
                                       LocalTensor<float> &dynamicQuantLocal, int32_t srcIdx, int32_t expertIdx,
                                       int64_t j);
    __aicore__ inline void ComputeScale(LocalTensor<float> &inLocal, LocalTensor<float> &tempLocal, float scaleTemp,
                                        int64_t dstIndex, int64_t j);

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputXInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> smoothInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> expandRowIdxInQueue;
    TQue<QuePosition::VECOUT, 1> calcQueue;
    TQue<QuePosition::VECOUT, quantTypeValue<quantType>::outQueDepth> inputXOutQueue;
    TQue<QuePosition::VECOUT, quantTypeValue<quantType>::outQueDepth> scaleOutQueue;
    TBuf<AscendC::TPosition::VECCALC> tempScaleBuf, maxValueBuf, mulBuf;
    LocalTensor<float> constScaleTensor, maxValueTensor, mulTensor;

    GlobalTensor<T> inputXGm;
    GlobalTensor<int8_t> expandedXGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<float> quantSmoothGm;
    GlobalTensor<float> dynamicQuantScaleGm;
    GlobalTensor<float> quantSrcGm;
    GlobalTensor<int32_t> expandedExpertIdxGm;
    GlobalTensor<int32_t> sortedRowIdxGm;

    const InnerMoeV2GatherOutComputeTilingData *gatherOutTilingData;

    int64_t needCoreNum;
    int64_t blockIdx;
    int64_t cols;
    int64_t colsAsInt8 = 0;
    int64_t n;
    int64_t k;
    int64_t totalLength;
    int64_t activateRows;
    int64_t currentLoopRows;
    int64_t currentLoopRowsAlign;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t rowLoops;
    int64_t colsTileLength;
    int64_t perLoopCols;
    int64_t perLoopcolsAsInt8 = 0;
    int64_t perLoopColsAlign;
    int64_t lastLoopCols;
    int64_t colLoops;
    int64_t dropPadMode;
    int64_t smoothType;

    int64_t indicesOffset;
    int64_t inputOffset;
    int64_t outOffset;
};

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::CopyInExpandedRowIdx(int64_t progress)
{
    this->indicesOffset = progress * this->perLoopRows;
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopyPad(indicesLocal, expandedRowIdxGm[indicesOffset], dataCopyParams, dataCopyPadParams);
    expandRowIdxInQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::CopyInExpandedExpertIdx(int64_t progress)
{
    this->indicesOffset = progress * this->perLoopRows;
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(this->currentLoopRows * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopyPad(indicesLocal, sortedRowIdxGm[indicesOffset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(indicesLocal[currentLoopRowsAlign], expandedExpertIdxGm[indicesOffset], dataCopyParams,
                dataCopyPadParams);
    expandRowIdxInQueue.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::Compute(LocalTensor<float> &smoothLocal)
{
    LocalTensor<float> inLocal = inputXInQueue.DeQue<float>();

    LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
    calcQueue.FreeTensor(tempLocal);

    if constexpr (!IsSameType<T, float>::value) {
        PipeBarrier<PIPE_V>();
        Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign], RoundMode::CAST_NONE, this->cols);
        PipeBarrier<PIPE_V>();
    }

    if (smoothType != 0) {
        PipeBarrier<PIPE_V>();
        Mul(inLocal, inLocal, smoothLocal, this->cols);
        PipeBarrier<PIPE_V>();
    }

    PipeBarrier<PIPE_V>();
    Abs(tempLocal, inLocal, this->cols);
    PipeBarrier<PIPE_V>();

    if constexpr (IsSameType<quantType, int4b_t>::value) {
        ReduceMaxCust(tempLocal, tempLocal, this->cols);

        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> tempInt32 = inLocal.template ReinterpretCast<int32_t>();
        LocalTensor<half> tempHalf = tempLocal.template ReinterpretCast<half>();
        LocalTensor<float> dynamicQuantLocal = scaleOutQueue.template AllocTensor<float>();

        PipeBarrier<PIPE_V>();
        Div(dynamicQuantLocal[MAX_VALUE_NUM], constScaleTensor, tempLocal, MAX_VALUE_NUM);
        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
        Div(dynamicQuantLocal, maxValueTensor, dynamicQuantLocal[MAX_VALUE_NUM],
            MAX_VALUE_NUM); // maxValueTensor 中值为1

        scaleOutQueue.template EnQue(dynamicQuantLocal);

        PipeBarrier<PIPE_V>();
        Brcb(mulTensor, dynamicQuantLocal[MAX_VALUE_NUM], 1, {1, 8});
        PipeBarrier<PIPE_V>();

        MulsCust(tempLocal, inLocal, mulTensor, this->cols);

        PipeBarrier<PIPE_V>();
        Cast(tempInt32, tempLocal, RoundMode::CAST_RINT, this->cols);

        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0));
        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
        Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, this->cols);
        inputXInQueue.FreeTensor(inLocal);

        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> outLocal = inputXOutQueue.template AllocTensor<int8_t>();
        auto outLocalInt4 = outLocal.template ReinterpretCast<int4b_t>();

        PipeBarrier<PIPE_V>();
        Cast(outLocalInt4, tempHalf, RoundMode::CAST_TRUNC, this->cols);
        inputXOutQueue.template EnQue(outLocal);
    } else {
        LocalTensor<int8_t> outLocal = inputXOutQueue.template AllocTensor<int8_t>();
        LocalTensor<float> dynamicQuantLocal = scaleOutQueue.template AllocTensor<float>();
        PipeBarrier<PIPE_V>();
        ReduceMax(dynamicQuantLocal, tempLocal, tempLocal, this->cols);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float maxValue = dynamicQuantLocal.GetValue(0) / 127.0f;

        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Duplicate<float>(dynamicQuantLocal, maxValue, 8);
        PipeBarrier<PIPE_V>();
        Duplicate<float>(tempLocal, maxValue, this->cols);
        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
        Div(tempLocal, inLocal, tempLocal, this->cols);
        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
        Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, this->cols);
        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
        Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, this->cols);

        inputXOutQueue.template EnQue(outLocal);
        scaleOutQueue.template EnQue(dynamicQuantLocal);
    }
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::OnceCopyOut(LocalTensor<int32_t> indicesLocal,
                                                                          int64_t row, int64_t &curLoopRow,
                                                                          int64_t &initialRow,
                                                                          DataCopyExtParams copyOutParams,
                                                                          DataCopyExtParams quantScaleParams)
{
    LocalTensor<float> quantScaleLocal = scaleOutQueue.template DeQue<float>();
    LocalTensor<int8_t> outLocal = inputXOutQueue.template DeQue<int8_t>();

    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    while (curLoopRow < this->currentLoopRows && initialRow / this->k == row) {
        int32_t outIndex = indicesLocal.GetValue(curLoopRow);
        curLoopRow++;
        initialRow++;
        if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
            continue;
        }
        DataCopyPad(expandedXGm[outIndex * this->colsAsInt8], outLocal, copyOutParams);
        DataCopyPad(dynamicQuantScaleGm[outIndex], quantScaleLocal, quantScaleParams);
    }
    inputXOutQueue.template FreeTensor(outLocal);
    scaleOutQueue.template FreeTensor(quantScaleLocal);
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::OnceCopyIn(int64_t row, DataCopyExtParams copyInParams)
{
    LocalTensor<T> inLocal = inputXInQueue.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(inLocal, inputXGm[(row) * this->cols], copyInParams, {false, 0, 0, 0});
    } else {
        DataCopyPad(inLocal[perLoopColsAlign], inputXGm[(row) * this->cols], copyInParams, {false, 0, 0, 0});
    }
    inputXInQueue.EnQue<T>(inLocal);
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::CopyOutXQuant1H(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue.DeQue<int32_t>();

    int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    int64_t curLoopRow = 0;
    int64_t currentLoopStartRow = initialRow / this->k;
    int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(this->cols * sizeof(int8_t)), 0, 0, 0};
    if constexpr (IsSameType<quantType, int4b_t>::value) {
        copyOutParams.blockLen = static_cast<uint32_t>(this->colsAsInt8 * sizeof(int8_t));
    }
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(this->cols * sizeof(float)), 0, 0, 0};
    DataCopyExtParams quantScaleParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    LocalTensor<float> smoothLocal;
    if (smoothType == 1) {
        smoothLocal = smoothInQueue.AllocTensor<float>();
        DataCopyPad(smoothLocal, quantSmoothGm, smoothParams, {false, 0, 0, 0});
        smoothInQueue.EnQue(smoothLocal);
        smoothLocal = smoothInQueue.DeQue<float>();
    }

    PipeBarrier<PIPE_V>();

    if constexpr (IsSameType<quantType, int4b_t>::value) {
        OnceCopyIn(currentLoopStartRow, copyInParams);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow - 1; row++) {
            OnceCopyIn(row + 1, copyInParams);
            Compute(smoothLocal);
            if (likely(row != currentLoopStartRow)) {
                OnceCopyOut(indicesLocal, row - 1, curLoopRow, initialRow, copyOutParams, quantScaleParams);
            }
        }
        if (currentLoopLastRow - currentLoopStartRow >= 1) {
            OnceCopyOut(indicesLocal, currentLoopLastRow - 1, curLoopRow, initialRow, copyOutParams, quantScaleParams);
        }
        if (currentLoopLastRow - currentLoopStartRow >= 0) {
            Compute(smoothLocal);
            OnceCopyOut(indicesLocal, currentLoopLastRow, curLoopRow, initialRow, copyOutParams, quantScaleParams);
        }
    } else {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
            LocalTensor<T> inLocal = inputXInQueue.AllocTensor<T>();
            if constexpr (IsSameType<T, float>::value) {
                DataCopyPad(inLocal, inputXGm[row * this->cols], copyInParams, {false, 0, 0, 0});
            } else {
                DataCopyPad(inLocal[perLoopColsAlign], inputXGm[row * this->cols], copyInParams, {false, 0, 0, 0});
            }

            inputXInQueue.EnQue<T>(inLocal);
            Compute(smoothLocal);
            inputXInQueue.FreeTensor(inLocal);
            OnceCopyOut(indicesLocal, row, curLoopRow, initialRow, copyOutParams, quantScaleParams);
        }
    }

    if (smoothType == 1) {
        smoothInQueue.FreeTensor(smoothLocal);
    }

    expandRowIdxInQueue.FreeTensor(indicesLocal);
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::CopyOutXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(this->perLoopCols * sizeof(T)), 0, 0, 0};
    DataCopyExtParams smoothParams{1, static_cast<uint32_t>(this->perLoopCols * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(this->perLoopCols * sizeof(int8_t)), 0, 0, 0};
    DataCopyExtParams quantScaleParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    int32_t lastExpertIdx = -1;
    LocalTensor<T> inLocal = inputXInQueue.AllocTensor<T>();
    LocalTensor<float> smoothLocal = smoothInQueue.AllocTensor<float>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (int64_t i = 0; i < this->currentLoopRows; i++) {
        int64_t rowOffset = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
        if (this->dropPadMode == DROPLESS_MODE && rowOffset + i >= this->activateRows) {
            break;
        }
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign + i);

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(inLocal, inputXGm[srcIdx / this->k * this->cols], copyInParams, {false, 0, 0, 0});
        } else {
            DataCopyPad(inLocal[perLoopColsAlign], inputXGm[srcIdx / this->k * this->cols], copyInParams,
                        {false, 0, 0, 0});
        }
        inputXInQueue.EnQue<T>(inLocal);

        if (expertIdx != lastExpertIdx) {
            DataCopyPad(smoothLocal, quantSmoothGm[expertIdx * this->cols], smoothParams, {false, 0, 0, 0});
            smoothInQueue.EnQue(smoothLocal);
            smoothLocal = smoothInQueue.DeQue<float>();
            lastExpertIdx = expertIdx;
        }

        Compute(smoothLocal);

        LocalTensor<float> quantScaleLocal = scaleOutQueue.template DeQue<float>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(dynamicQuantScaleGm[(rowOffset + i)], quantScaleLocal, quantScaleParams);

        LocalTensor<int8_t> outLocal = inputXOutQueue.template DeQue<int8_t>();
        DataCopyPad(expandedXGm[(rowOffset + i) * this->cols], outLocal, copyOutParams);

        inputXOutQueue.template FreeTensor(outLocal);
        scaleOutQueue.template FreeTensor(quantScaleLocal);
    }

    inputXInQueue.FreeTensor(inLocal);
    smoothInQueue.FreeTensor(smoothLocal);
    expandRowIdxInQueue.FreeTensor(indicesLocal);
}

template <typename T, typename quantType>
__aicore__ inline float MoeV2GatherDynamicQuant<T, quantType>::ComputeMax(LocalTensor<float> &inLocal,
                                                                          LocalTensor<float> &tempLocal,
                                                                          LocalTensor<float> &dynamicQuantLocal,
                                                                          int32_t srcIdx, int32_t expertIdx, int64_t j)
{
    LocalTensor<float> smoothLocal = smoothInQueue.AllocTensor<float>();

    DataCopyExtParams intriParamsT{1, static_cast<uint32_t>(colsTileLength * sizeof(T)), 0, 0, 0};
    DataCopyExtParams intriParamsFp32{1, static_cast<uint32_t>(colsTileLength * sizeof(float)), 0, 0, 0};

    if constexpr (!IsSameType<T, float>::value) {
        DataCopyPad(inLocal.ReinterpretCast<T>()[perLoopColsAlign],
                    inputXGm[srcIdx * this->cols + j * this->perLoopCols], intriParamsT, {false, 0, 0, 0});
    } else {
        DataCopyPad(inLocal, inputXGm[srcIdx * this->cols + j * this->perLoopCols], intriParamsT, {false, 0, 0, 0});
    }

    inputXInQueue.EnQue<float>(inLocal);
    inLocal = inputXInQueue.DeQue<float>();

    if (smoothType != 0) {
        DataCopyPad(smoothLocal, quantSmoothGm[expertIdx * this->cols + j * this->perLoopCols], intriParamsFp32,
                    {false, 0, 0, 0});
        smoothInQueue.EnQue(smoothLocal);
        smoothLocal = smoothInQueue.DeQue<float>();
    }

    if constexpr (!IsSameType<T, float>::value) {
        PipeBarrier<PIPE_V>();
        Cast(inLocal, inLocal.ReinterpretCast<T>()[perLoopColsAlign], RoundMode::CAST_NONE, colsTileLength);
        PipeBarrier<PIPE_V>();
    }

    if (smoothType != 0) {
        PipeBarrier<PIPE_V>();
        Mul(inLocal, inLocal, smoothLocal, colsTileLength);
        PipeBarrier<PIPE_V>();
    }

    PipeBarrier<PIPE_V>();
    Abs(tempLocal, inLocal, colsTileLength);
    PipeBarrier<PIPE_V>();

    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    ReduceMax(dynamicQuantLocal[8], tempLocal, tempLocal, colsTileLength);

    DataCopyPad(quantSrcGm[j * this->perLoopCols], inLocal, intriParamsFp32);
    smoothInQueue.FreeTensor(smoothLocal);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return dynamicQuantLocal.GetValue(8);
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::ComputeScale(LocalTensor<float> &inLocal,
                                                                           LocalTensor<float> &tempLocal,
                                                                           float scaleTemp, int64_t dstIndex, int64_t j)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(colsTileLength * sizeof(float)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(colsTileLength * sizeof(int8_t)), 0, 0, 0};

    LocalTensor<int8_t> outLocal = inputXOutQueue.template AllocTensor<int8_t>();
    DataCopyPad(inLocal, quantSrcGm[j * this->perLoopCols], copyInParams, {false, 0, 0, 0});
    inputXInQueue.EnQue<float>(inLocal);
    inLocal = inputXInQueue.DeQue<float>();
    if constexpr (IsSameType<quantType, int4b_t>::value) {
        copyOutParams.blockLen = static_cast<uint32_t>(this->colsTileLength / 2);
        LocalTensor<int32_t> tempInt32 = tempLocal.template ReinterpretCast<int32_t>();
        LocalTensor<half> tempHalf = inLocal.template ReinterpretCast<half>();
        auto outLocalInt4 = outLocal.template ReinterpretCast<int4b_t>();
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

        PipeBarrier<PIPE_V>();
        Muls(inLocal, inLocal, scaleTemp, colsTileLength);

        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
        Cast(tempInt32, inLocal, RoundMode::CAST_RINT, colsTileLength);

        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0));
        PipeBarrier<PIPE_V>();
        Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, colsTileLength);

        PipeBarrier<PIPE_V>();
        Cast(outLocalInt4, tempHalf, RoundMode::CAST_TRUNC, colsTileLength);
    } else {
        PipeBarrier<PIPE_V>();
        Duplicate<float>(tempLocal, scaleTemp, colsTileLength);
        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
        Div(tempLocal, inLocal, tempLocal, colsTileLength);
        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
        Cast(tempLocal.ReinterpretCast<half>(), tempLocal, RoundMode::CAST_TRUNC, colsTileLength);
        PipeBarrier<PIPE_V>();

        PipeBarrier<PIPE_V>();
        Cast(outLocal, tempLocal.ReinterpretCast<half>(), RoundMode::CAST_ROUND, colsTileLength);
    }

    inputXOutQueue.template EnQue(outLocal);
    outLocal = inputXOutQueue.template DeQue<int8_t>();
    if constexpr (IsSameType<quantType, int4b_t>::value) {
        DataCopyPad(expandedXGm[dstIndex * this->colsAsInt8 + j * this->perLoopcolsAsInt8], outLocal, copyOutParams);
    } else {
        DataCopyPad(expandedXGm[dstIndex * this->cols + j * this->perLoopCols], outLocal, copyOutParams);
    }

    inputXOutQueue.template FreeTensor(outLocal);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::CopyOutPartialXQuantEH(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    DataCopyExtParams quantScaleParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (int64_t i = 0; i < this->currentLoopRows; i++) {
        int64_t rowOffset = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
        if (this->dropPadMode == DROPLESS_MODE && rowOffset + i >= this->activateRows) {
            break;
        }
        int32_t srcIdx = indicesLocal.GetValue(i);
        int32_t expertIdx = indicesLocal.GetValue(currentLoopRowsAlign + i);

        LocalTensor<float> inLocal = inputXInQueue.AllocTensor<float>();
        LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
        LocalTensor<float> quantScaleLocal = scaleOutQueue.template AllocTensor<float>();

        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *)&tmp);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int64_t j = 0; j < this->colLoops; j++) {
            colsTileLength = this->perLoopCols;
            if (j == this->colLoops - 1) {
                colsTileLength = this->lastLoopCols;
            }
            float tileMax = ComputeMax(inLocal, tempLocal, quantScaleLocal, srcIdx / this->k, expertIdx, j);
            reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
        }

        float scaleTemp = reduceMax / 127.0f;
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Duplicate<float>(quantScaleLocal, scaleTemp, 8);
        scaleOutQueue.template EnQue(quantScaleLocal);
        quantScaleLocal = scaleOutQueue.template DeQue<float>();

        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(dynamicQuantScaleGm[(rowOffset + i)], quantScaleLocal, quantScaleParams);

        for (int64_t j = 0; j < this->colLoops; j++) {
            colsTileLength = this->perLoopCols;
            if (j == this->colLoops - 1) {
                colsTileLength = this->lastLoopCols;
            }

            ComputeScale(inLocal, tempLocal, scaleTemp, rowOffset + i, j);
        }

        inputXInQueue.FreeTensor(inLocal);
        calcQueue.FreeTensor(tempLocal);
        scaleOutQueue.template FreeTensor(quantScaleLocal);
    }

    expandRowIdxInQueue.FreeTensor(indicesLocal);
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::CopyOutPartialXQuant1H(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = expandRowIdxInQueue.DeQue<int32_t>();

    int64_t initialRow = this->gatherOutTilingData->perCoreRows * this->blockIdx + this->perLoopRows * progress;
    int64_t curLoopRow = 0;

    int64_t currentLoopStartRow = initialRow / this->k;
    int64_t currentLoopLastRow = (initialRow + this->currentLoopRows - 1) / this->k;
    DataCopyExtParams quantScaleParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};

    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (int64_t row = currentLoopStartRow; row <= currentLoopLastRow; row++) {
        LocalTensor<float> inLocal = inputXInQueue.AllocTensor<float>();
        LocalTensor<float> tempLocal = calcQueue.AllocTensor<float>();
        LocalTensor<float> quantScaleLocal = scaleOutQueue.template AllocTensor<float>();

        uint32_t tmp = 0xFF7FFFFF;
        float reduceMax = *((float *)&tmp);
        if constexpr (IsSameType<quantType, int4b_t>::value) {
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Duplicate<float>(maxValueTensor, -1.0, MAX_VALUE_NUM);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for (int64_t j = 0; j < this->colLoops; j++) {
            colsTileLength = this->perLoopCols;
            if (j == this->colLoops - 1) {
                colsTileLength = this->lastLoopCols;
            }

            float tileMax = ComputeMax(inLocal, tempLocal, quantScaleLocal, row, 0, j);
            if constexpr (IsSameType<quantType, int4b_t>::value) {
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Max(maxValueTensor, maxValueTensor, quantScaleLocal[MAX_VALUE_NUM], MAX_VALUE_NUM);
                PipeBarrier<PIPE_V>();
            } else {
                reduceMax = (reduceMax > tileMax) ? reduceMax : tileMax;
            }
        }
        float scaleTemp;
        if constexpr (IsSameType<quantType, int4b_t>::value) {
            PipeBarrier<PIPE_V>();
            Div(maxValueTensor, constScaleTensor, maxValueTensor, MAX_VALUE_NUM);
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            scaleTemp = maxValueTensor.GetValue(0);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            quantScaleLocal.SetValue(0, 1 / scaleTemp);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        } else {
            scaleTemp = reduceMax / 127.0f;
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Duplicate<float>(quantScaleLocal, scaleTemp, 8);
        }
        scaleOutQueue.template EnQue(quantScaleLocal);
        quantScaleLocal = scaleOutQueue.template DeQue<float>();

        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        while (curLoopRow < this->currentLoopRows && initialRow / this->k == row) {
            int32_t outIndex = indicesLocal.GetValue(curLoopRow);
            curLoopRow++;
            initialRow++;
            if (outIndex == -1 || (this->dropPadMode == DROPLESS_MODE && outIndex >= this->activateRows)) {
                continue;
            }
            DataCopyPad(dynamicQuantScaleGm[outIndex], quantScaleLocal, quantScaleParams);
            for (int64_t j = 0; j < this->colLoops; j++) {
                colsTileLength = this->perLoopCols;
                if (j == this->colLoops - 1) {
                    colsTileLength = this->lastLoopCols;
                }

                ComputeScale(inLocal, tempLocal, scaleTemp, outIndex, j);
            }
        }
        inputXInQueue.FreeTensor(inLocal);
        calcQueue.FreeTensor(tempLocal);
        scaleOutQueue.template FreeTensor(quantScaleLocal);
    }

    expandRowIdxInQueue.FreeTensor(indicesLocal);
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::Init(GM_ADDR inputX, GM_ADDR quantSmooth,
                                                                   GM_ADDR expandedRowIdx, GM_ADDR expandedX,
                                                                   GM_ADDR dynamicQuantScale, GM_ADDR workspace,
                                                                   const MoeInitRoutingQuantV2TilingData *tilingData,
                                                                   TPipe *tPipe)
{
    this->pipe = tPipe;
    this->blockIdx = GetBlockIdx();
    this->gatherOutTilingData = &(tilingData->gatherOutComputeParamsOp);

    this->needCoreNum = this->gatherOutTilingData->needCoreNum;
    this->activateRows = this->gatherOutTilingData->activateRows;
    this->cols = tilingData->cols;
    this->n = tilingData->n;
    this->k = tilingData->k;
    this->totalLength = tilingData->n * tilingData->k;
    this->dropPadMode = tilingData->dropPadMode;
    this->smoothType = tilingData->smoothType;

    if (this->blockIdx == this->gatherOutTilingData->needCoreNum - 1) {
        this->coreRows = this->gatherOutTilingData->lastCoreRows;
        this->perLoopRows = this->gatherOutTilingData->lastCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->lastCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->lastCoreLoops;
    } else {
        this->coreRows = this->gatherOutTilingData->perCoreRows;
        this->perLoopRows = this->gatherOutTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->gatherOutTilingData->perCoreLastLoopRows;
        this->rowLoops = this->gatherOutTilingData->perCoreLoops;
    }
    this->perLoopCols = this->gatherOutTilingData->perLoopCols;
    this->lastLoopCols = this->gatherOutTilingData->lastLoopCols;
    this->colLoops = this->gatherOutTilingData->colLoops;
    this->perLoopColsAlign = Align(this->perLoopCols, sizeof(T));

    inputXGm.SetGlobalBuffer((__gm__ T *)inputX);
    expandedXGm.SetGlobalBuffer((__gm__ int8_t *)expandedX);

    expandedRowIdxGm.SetGlobalBuffer(
        (__gm__ int32_t *)expandedRowIdx + this->blockIdx * this->gatherOutTilingData->perCoreRows,
        Align(this->coreRows, sizeof(int32_t)));

    quantSmoothGm.SetGlobalBuffer((__gm__ float *)quantSmooth);
    dynamicQuantScaleGm.SetGlobalBuffer((__gm__ float *)dynamicQuantScale);

    expandedExpertIdxGm.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + this->blockIdx * this->gatherOutTilingData->perCoreRows,
        Align(this->coreRows, sizeof(int32_t)));
    sortedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)workspace + Align(this->totalLength, sizeof(int32_t)) +
                                       this->blockIdx * this->gatherOutTilingData->perCoreRows,
                                   Align(this->coreRows, sizeof(int32_t)));
    if (this->cols > 1) {
        quantSrcGm.SetGlobalBuffer(
            (__gm__ float *)workspace + Align(this->totalLength, sizeof(int32_t)) * 2 + this->blockIdx * this->cols,
            this->cols * sizeof(float));
    }

    this->currentLoopRowsAlign = Align(this->perLoopRows, sizeof(int32_t));

    int64_t perLoopColsAlignBytes = AlignBytes(this->perLoopCols, sizeof(T));
    perLoopColsAlignBytes =
        Max(int64_t(perLoopColsAlignBytes * sizeof(float) / sizeof(T)), int64_t(BLOCK_BYTES + BLOCK_BYTES));

    pipe->InitBuffer(expandRowIdxInQueue, BUFFER_NUM, 2 * AlignBytes(this->perLoopRows, sizeof(int32_t)));
    pipe->InitBuffer(inputXInQueue, BUFFER_NUM, perLoopColsAlignBytes);
    if constexpr (IsSameType<quantType, int4b_t>::value) {
        this->colsAsInt8 = this->cols / ONE_BYTE_INT4_NUM;
        this->perLoopcolsAsInt8 = this->perLoopCols / ONE_BYTE_INT4_NUM;
        pipe->InitBuffer(smoothInQueue, 1, AlignBytes(this->perLoopCols, sizeof(float)));
        pipe->InitBuffer(calcQueue, 1, AlignBytes(this->perLoopCols, sizeof(float)));
        pipe->InitBuffer(inputXOutQueue, BUFFER_NUM, AlignBytes(perLoopcolsAsInt8, sizeof(int8_t)));
        pipe->InitBuffer(scaleOutQueue, BUFFER_NUM, BLOCK_BYTES + BLOCK_BYTES);

        pipe->InitBuffer(tempScaleBuf, BLOCK_BYTES);
        pipe->InitBuffer(maxValueBuf, BLOCK_BYTES);
        pipe->InitBuffer(mulBuf, ONE_REPEAT_NUM);
        constScaleTensor = tempScaleBuf.Get<float>();
        maxValueTensor = maxValueBuf.Get<float>();
        mulTensor = mulBuf.Get<float>();

        PipeBarrier<PIPE_V>();
        Duplicate<float>(constScaleTensor, DYNAMIC_QUANT_INT4_SYM_SCALE, MAX_VALUE_NUM);
        PipeBarrier<PIPE_V>();
        Duplicate<float>(maxValueTensor, 1.0, MAX_VALUE_NUM);
    } else {
        this->colsAsInt8 = this->cols;
        this->perLoopcolsAsInt8 = this->perLoopCols;
        pipe->InitBuffer(smoothInQueue, BUFFER_NUM, AlignBytes(this->perLoopCols, sizeof(float)));
        pipe->InitBuffer(calcQueue, 1, AlignBytes(this->perLoopCols, sizeof(float)));
        pipe->InitBuffer(inputXOutQueue, 1, AlignBytes(this->perLoopCols, sizeof(int8_t)));
        pipe->InitBuffer(scaleOutQueue, 1, BLOCK_BYTES + BLOCK_BYTES);
    }
}

template <typename T, typename quantType>
__aicore__ inline void MoeV2GatherDynamicQuant<T, quantType>::Process()
{
    if (this->cols == 0) { // 空tensor场景提前对dynamicQuantScale赋值
        LocalTensor<float> dynamicQuantLocal = scaleOutQueue.template AllocTensor<float>();
        PipeBarrier<PIPE_V>();
        Duplicate<float>(dynamicQuantLocal, 0.0, MAX_VALUE_NUM);
        scaleOutQueue.FreeTensor(dynamicQuantLocal);
    }
    if (this->blockIdx < this->needCoreNum) {
        currentLoopRows = perLoopRows;
        if (colLoops > 1) { // 一行无法全载，需要workspace
            if (smoothType == 2) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedExpertIdx(loop);
                    CopyOutPartialXQuantEH(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedExpertIdx(this->rowLoops - 1);
                CopyOutPartialXQuantEH(this->rowLoops - 1);
            } else {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedRowIdx(loop);
                    CopyOutPartialXQuant1H(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedRowIdx(this->rowLoops - 1);
                CopyOutPartialXQuant1H(this->rowLoops - 1);
            }
        } else { // 一行可以全载
            if (smoothType == 2) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedExpertIdx(loop);
                    CopyOutXQuantEH(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedExpertIdx(this->rowLoops - 1);
                CopyOutXQuantEH(this->rowLoops - 1);
            } else {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (int64_t loop = 0; loop < this->rowLoops - 1; loop++) {
                    CopyInExpandedRowIdx(loop);
                    CopyOutXQuant1H(loop);
                }
                currentLoopRows = lastLoopRows;
                CopyInExpandedRowIdx(this->rowLoops - 1);
                CopyOutXQuant1H(this->rowLoops - 1);
            }
        }
    }
}
} // namespace MoeInitRoutingQuantV2
#endif // MOE_V2_GATHER_DYNAMIC_QUANT_H
