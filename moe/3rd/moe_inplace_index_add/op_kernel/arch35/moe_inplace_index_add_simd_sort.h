/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_inplace_index_add_simd_sort.h
 * \brief moe_inplace_index_add_simd_sort
 */
#ifndef ASCENDC_MOE_INPLACE_INDEX_ADD_SIMD_SORT_H_
#define ASCENDC_MOE_INPLACE_INDEX_ADD_SIMD_SORT_H_

#include "moe_inplace_index_add_common.h"

constexpr int64_t INDICES_SORT_THRESHOLD = 128;

namespace MoeInplaceIndexAdd {
using namespace AscendC;

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
class MoeInplaceIndexAddSimdSort : MoeInplaceIndexAddBase<VAR_T, IDX_T, CAST_MODE> {
public:
    using CAST_T = typename MoeInplaceIndexAddBase<VAR_T, IDX_T, CAST_MODE>::CAST_T;
    __aicore__ inline MoeInplaceIndexAddSimdSort(const MoeInplaceIndexAddSimdSortTilingData &tilingData, TPipe &pipe)
        : tilingData_(tilingData),
          pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha, GM_ADDR workspace);
    __aicore__ inline void HandleAlpha(LocalTensor<VAR_T> updatesLocal, VAR_T alphaValue, int64_t dataCount);

    __aicore__ inline void ProcessPreSmallIndices(int64_t preOfset, int64_t colIdx, int64_t preLen, int64_t colLen);
    __aicore__ inline void ComputeSumAndCopyOutSlitPre(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyInAndCopyOutSlitPre(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessPre();

    __aicore__ inline void CopyIndiceIn(int64_t rowIdx, int64_t rowLen);
    __aicore__ inline void ComputeSumAndCopyOutSlitAfter(int64_t rowIdx, int64_t colIdx, int64_t rowLen,
                                                         int64_t colLen);
    __aicore__ inline void CopyInAndCopyOutSlitAfter(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessAfter();

    __aicore__ inline void ComputeSumAndCopyOutSlitIndices(int64_t rowIdx, int64_t colIdx, int64_t rowLen,
                                                           int64_t colLen);
    __aicore__ inline void CopyInAndCopyOutSlitIndices(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessIndices();
    __aicore__ inline void Process();

private:
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;
    GlobalTensor<VAR_T> alpha_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> updatesCastQue_;
    TBuf<QuePosition::VECCALC> maxScoreBuf_;

    TPipe &pipe_;
    const MoeInplaceIndexAddSimdSortTilingData &tilingData_;

    int64_t curPreAxisCount_{0};
    VAR_T alphaValue_{0};
    float maxScore_ = static_cast<float>(0);
};

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::Init(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha, GM_ADDR workspace)
{
    var_.SetGlobalBuffer((__gm__ VAR_T *)(var));
    indices_.SetGlobalBuffer((__gm__ IDX_T *)(indices));
    updates_.SetGlobalBuffer((__gm__ VAR_T *)(updates));
    alpha_.SetGlobalBuffer((__gm__ VAR_T *)(alpha));

    pipe_.InitBuffer(maxScoreBuf_, 128 * sizeof(float));
    this->indicesFactor_ = tilingData_.ubIndexFactor;
    this->afterAxis_ = tilingData_.afterAxis;
    this->afterAxisFactor_ = tilingData_.afterAxisFactor;
    this->eachCoreAfterAxisCount_ = tilingData_.eachCoreAfterAxisCount;
    this->varInAxis_ = tilingData_.varInAxis;
    this->updatesInAxis_ = tilingData_.updatesInAxis;
    this->InitBaseBuffer(pipe_, var, indices, updates);

    if (tilingData_.isWithAlpha) {
        alphaValue_ = alpha_(0);
    }
    if (tilingData_.isSplitPreAxis == 1) {
        curPreAxisCount_ = (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCorePreAxisCount :
                                                                                   tilingData_.tailCorePreAxisCount);
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::HandleAlpha(
    LocalTensor<VAR_T> updatesLocal, VAR_T alphaValue, int64_t dataCount)
{
    if (tilingData_.isWithAlpha) {
        if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, bool>::value) {
            ComputeMulWithIntCast<VAR_T>(updatesLocal, updatesLocal, alphaValue_, dataCount);
        } else if (IsSameType<VAR_T, half>::value || IsSameType<VAR_T, bfloat16_t>::value) {
            AscendC::Muls(updatesLocal, updatesLocal, alphaValue, dataCount);
        } else {
            AscendC::Muls(updatesLocal, updatesLocal, alphaValue, dataCount);
        }
    }
    return;
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::CopyIndiceIn(int64_t rowIdx,
                                                                                                        int64_t rowLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    LocalTensor<float> dstLocal = maxScoreBuf_.Get<float>();
    /* 不在indice上分核时核间偏移为0 */
    int64_t indicesOfset = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.ubIndexFactor;
    if constexpr (IS_CONTIGUOUS) {
        CopyIn<IDX_T>(indicesLocal, indices_[indicesOfset], rowLen);
    } else {
        CopyInNoContiguous<IDX_T>(indicesLocal, indices_[indicesOfset * tilingData_.indicesStride], rowLen, 1,
                                  tilingData_.indicesStride - 1);
    }
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if constexpr (IsSameType<IDX_T, int32_t>::value) {
        IndexStatisticInt32(indicesLocal, dstLocal, maxScore_, rowLen, tilingData_.afterAxis);
    } else {
        IndexStatisticInt64(indicesLocal, dstLocal, maxScore_, rowLen, tilingData_.afterAxis);
    }
    indicesQue_.EnQue(indicesLocal);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::ProcessPreSmallIndices(
    int64_t preOfset, int64_t colIdx, int64_t preLen, int64_t colLen)
{
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();

    int64_t rowLen = preLen * tilingData_.updatesInAxis;
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);

    if constexpr (IS_CONTIGUOUS) {
        CopyIn<IDX_T>(indicesLocal, indices_, tilingData_.updatesInAxis);
    } else {
        CopyInNoContiguous<IDX_T>(indicesLocal, indices_, tilingData_.updatesInAxis, 1, tilingData_.indicesStride - 1);
    }
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    int64_t startPreAxis = GetBlockIdx() * tilingData_.eachCorePreAxisCount + preOfset;
    int64_t rowOfset = (startPreAxis * tilingData_.updatesInAxis) * tilingData_.afterAxis;
    int64_t updatesOfset = rowOfset + colIdx * tilingData_.afterAxisFactor;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};

    DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    HandleAlpha(updatesLocal, alphaValue_, rowLen * colLenAlignSize);

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    this->CopyOutSplitPre(indicesLocal, updatesLocal, startPreAxis, preLen, colLen, colIdx);

    updatesQue_.FreeTensor(updatesLocal);
    indicesQue_.FreeTensor(indicesLocal);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::ComputeSumAndCopyOutSlitPre(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
    LocalTensor<VAR_T> updateSumLocal = updatesCastQue_.AllocTensor<VAR_T>();
    LocalTensor<IDX_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<IDX_T>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};

    int64_t indicesOfset = rowIdx * tilingData_.ubIndexFactor;
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t startPreAxis = GetBlockIdx() * tilingData_.eachCorePreAxisCount;
    for (int64_t preAxisIdx = startPreAxis; preAxisIdx < curPreAxisCount_ + startPreAxis; preAxisIdx++) {
        int64_t rowOfset = (indicesOfset + preAxisIdx * tilingData_.updatesInAxis) * tilingData_.afterAxis;
        int64_t updatesOfset = colIdx * tilingData_.afterAxisFactor + rowOfset;
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        HandleAlpha(updatesLocal, alphaValue_, rowLen * colLenAlignSize);

        if constexpr (IsSameType<VAR_T, bfloat16_t>::value || IsSameType<VAR_T, half>::value) {
            this->ComputeSumWithCast(updatesLocal, updateSumLocal, this->uniqueIdNum_, colLen);
        } else {
            this->ComputeSumWithOutCast(updatesLocal, updateSumLocal, this->uniqueIdNum_, colLen);
        }
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        this->CopyOutSplitIndices(updateSumIdxLocal, updateSumLocal, preAxisIdx, this->uniqueIdNum_, colLen, colIdx);
    }
    updatesQue_.FreeTensor(updatesLocal);
    updatesCastQue_.FreeTensor(updateSumLocal);
    this->updateSumIdxQue_.template EnQue(updateSumIdxLocal);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::CopyInAndCopyOutSlitPre(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
    LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
    LocalTensor<VAR_T> updateOutLocal = updatesCastQue_.AllocTensor<VAR_T>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};

    int64_t indicesOfset = rowIdx * tilingData_.ubIndexFactor;
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t startPreAxis = GetBlockIdx() * tilingData_.eachCorePreAxisCount;
    for (int64_t preAxisIdx = startPreAxis; preAxisIdx < startPreAxis + curPreAxisCount_; preAxisIdx++) {
        int64_t rowOfset = (preAxisIdx * tilingData_.updatesInAxis + indicesOfset) * tilingData_.afterAxis;
        int64_t updatesOfset = rowOfset + colIdx * tilingData_.afterAxisFactor;
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        HandleAlpha(updatesLocal, alphaValue_, rowLen * colLenAlignSize);

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        this->CopyOutSplitIndices(indicesLocal, updatesLocal, preAxisIdx, this->uniqueIdNum_, colLen, colIdx);
    }
    updatesQue_.FreeTensor(updatesLocal);
    updatesCastQue_.FreeTensor(updateOutLocal);
    indicesQue_.EnQue(indicesLocal);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::ProcessPre()
{
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER, (tilingData_.ubIndexFactor) * sizeof(IDX_T));
    pipe_.InitBuffer(updatesQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));
    pipe_.InitBuffer(updatesCastQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));

    int64_t rowTailDataLen = tilingData_.indiceAxisTailNum;
    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowLoopNum = tilingData_.indicesLoopSize;

    int64_t colTailDataLen = tilingData_.updateTailNum;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colLoopNum = tilingData_.updateLoopSize;
    /* indices可以一次搬入的情况, 对pre分loop处理 */
    if (rowLoopNum == 1) {
        int64_t ubPreFactor = tilingData_.ubIndexFactor / tilingData_.updatesInAxis;
        int64_t preLoopNum = Ops::Base::CeilDiv(curPreAxisCount_, ubPreFactor);
        int64_t preMainDataLen = ubPreFactor;
        int64_t preTailDataLen = curPreAxisCount_ - (preLoopNum - 1) * preMainDataLen;

        for (int64_t preIdx = 0; preIdx < preLoopNum; preIdx++) {
            int64_t preDataLen = (preIdx == preLoopNum - 1) ? preTailDataLen : preMainDataLen;
            int64_t preOfset = preMainDataLen * preIdx;
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                ProcessPreSmallIndices(preOfset, colIdx, preDataLen, colDataLen);
            }
        }
        return;
    }

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        CopyIndiceIn(rowIdx, rowDataLen);
        if (maxScore_ > SORT_HIST_THRESHOLD) {
            LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
            if constexpr (CAST_MODE == CAST_0) {
                this->SortIndices(indicesLocal, rowDataLen);
            } else {
                LocalTensor<CAST_T> indicesCastLocal = this->indicesCastQue_.template AllocTensor<CAST_T>();
                this->IndicesSortCast(indicesLocal, indicesCastLocal, rowDataLen);
                this->SortIndices(indicesCastLocal, rowDataLen);
                this->indicesCastQue_.template FreeTensor(indicesCastLocal);
            }
            indicesQue_.FreeTensor(indicesLocal);
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                ComputeSumAndCopyOutSlitPre(rowIdx, colIdx, rowDataLen, colDataLen);
            }
            LocalTensor<IDX_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<IDX_T>();
            LocalTensor<uint32_t> updatesOriginIdexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
            LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();
            this->updateSumIdxQue_.template FreeTensor(updateSumIdxLocal);
            this->updatesOriginIdexQue_.template FreeTensor(updatesOriginIdexLocal);
            this->uniqueIdCountQue_.template FreeTensor(uniqueIdCountLocal);
        } else {
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                CopyInAndCopyOutSlitPre(rowIdx, colIdx, rowDataLen, colDataLen);
            }
            LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
            indicesQue_.FreeTensor(indicesLocal);
        }
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void
MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::ComputeSumAndCopyOutSlitAfter(int64_t rowIdx,
                                                                                                  int64_t colIdx,
                                                                                                  int64_t rowLen,
                                                                                                  int64_t colLen)
{
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
    LocalTensor<VAR_T> updateSumLocal = updatesCastQue_.AllocTensor<VAR_T>();
    LocalTensor<IDX_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<IDX_T>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t indicesOfset = rowIdx * tilingData_.ubIndexFactor;
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    for (int64_t preAxisIdx = 0; preAxisIdx < tilingData_.preAxis; preAxisIdx++) {
        int64_t rowOfset = (tilingData_.updatesInAxis * preAxisIdx + indicesOfset) * tilingData_.afterAxis;
        int64_t updatesOfset =
            rowOfset + GetBlockIdx() * tilingData_.eachCoreAfterAxisCount + tilingData_.afterAxisFactor * colIdx;

        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        HandleAlpha(updatesLocal, alphaValue_, rowLen * colLenAlignSize);

        if constexpr (IsSameType<VAR_T, half>::value || IsSameType<VAR_T, bfloat16_t>::value) {
            this->ComputeSumWithCast(updatesLocal, updateSumLocal, this->uniqueIdNum_, colLen);
        } else {
            this->ComputeSumWithOutCast(updatesLocal, updateSumLocal, this->uniqueIdNum_, colLen);
        }
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        this->CopyOutSplitAfter(updateSumIdxLocal, updateSumLocal, preAxisIdx, this->uniqueIdNum_, colLen, colIdx);
    }
    updatesQue_.FreeTensor(updatesLocal);
    updatesCastQue_.FreeTensor(updateSumLocal);
    this->updateSumIdxQue_.template EnQue(updateSumIdxLocal);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::CopyInAndCopyOutSlitAfter(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};

    int64_t indicesOfset = rowIdx * tilingData_.ubIndexFactor;
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    for (int64_t preAxisIdx = 0; preAxisIdx < tilingData_.preAxis; preAxisIdx++) {
        int64_t rowOfset = tilingData_.afterAxis * (preAxisIdx * tilingData_.updatesInAxis + indicesOfset);
        int64_t updatesOfset =
            rowOfset + GetBlockIdx() * tilingData_.eachCoreAfterAxisCount + colIdx * tilingData_.afterAxisFactor;

        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        HandleAlpha(updatesLocal, alphaValue_, rowLen * colLenAlignSize);

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        this->CopyOutSplitAfter(indicesLocal, updatesLocal, preAxisIdx, rowLen, colLen, colIdx);
    }
    updatesQue_.FreeTensor(updatesLocal);
    indicesQue_.EnQue(indicesLocal);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::ProcessAfter()
{
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER, (tilingData_.ubIndexFactor) * sizeof(IDX_T));
    pipe_.InitBuffer(updatesQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));
    pipe_.InitBuffer(updatesCastQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(float));

    int64_t rowLoopNum = tilingData_.indicesLoopSize;
    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowTailDataLen = tilingData_.indiceAxisTailNum;

    int64_t colLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateLoopSize :
                                                                                tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateAxisNum :
                                                                                    tilingData_.updateTailNum;

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        CopyIndiceIn(rowIdx, rowDataLen);
        if (maxScore_ > SORT_HIST_THRESHOLD) {
            LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
            if constexpr (CAST_MODE == CAST_0) {
                this->SortIndices(indicesLocal, rowDataLen);
            } else {
                LocalTensor<CAST_T> indicesCastLocal = this->indicesCastQue_.template AllocTensor<CAST_T>();
                this->IndicesSortCast(indicesLocal, indicesCastLocal, rowDataLen);
                this->SortIndices(indicesCastLocal, rowDataLen);
                this->indicesCastQue_.template FreeTensor(indicesCastLocal);
            }
            indicesQue_.FreeTensor(indicesLocal);
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                ComputeSumAndCopyOutSlitAfter(rowIdx, colIdx, rowDataLen, colDataLen);
            }
            LocalTensor<IDX_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<IDX_T>();
            LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();
            LocalTensor<uint32_t> updatesOriginIdexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
            this->updateSumIdxQue_.template FreeTensor(updateSumIdxLocal);
            this->uniqueIdCountQue_.template FreeTensor(uniqueIdCountLocal);
            this->updatesOriginIdexQue_.template FreeTensor(updatesOriginIdexLocal);
        } else {
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                CopyInAndCopyOutSlitAfter(rowIdx, colIdx, rowDataLen, colDataLen);
            }
            LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
            indicesQue_.FreeTensor(indicesLocal);
        }
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void
MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::ComputeSumAndCopyOutSlitIndices(int64_t rowIdx,
                                                                                                    int64_t colIdx,
                                                                                                    int64_t rowLen,
                                                                                                    int64_t colLen)
{
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
    LocalTensor<VAR_T> updateSumLocal = updatesCastQue_.AllocTensor<VAR_T>();
    LocalTensor<IDX_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<IDX_T>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t indicesOfset = tilingData_.eachCoreIndexCount * GetBlockIdx() + rowIdx * tilingData_.ubIndexFactor;
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    for (int64_t preAxisIdx = 0; preAxisIdx < tilingData_.preAxis; preAxisIdx++) {
        int64_t rowOfset = (tilingData_.updatesInAxis * preAxisIdx + indicesOfset) * tilingData_.afterAxis;
        int64_t updatesOfset =
            rowOfset + GetBlockIdx() * tilingData_.eachCoreAfterAxisCount + tilingData_.afterAxisFactor * colIdx;
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        HandleAlpha(updatesLocal, alphaValue_, rowLen * colLenAlignSize);

        if constexpr (IsSameType<VAR_T, half>::value || IsSameType<VAR_T, bfloat16_t>::value) {
            this->ComputeSumWithCast(updatesLocal, updateSumLocal, this->uniqueIdNum_, colLen);
        } else {
            this->ComputeSumWithOutCast(updatesLocal, updateSumLocal, this->uniqueIdNum_, colLen);
        }
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        this->CopyOutSplitIndices(updateSumIdxLocal, updateSumLocal, preAxisIdx, this->uniqueIdNum_, colLen, colIdx);
    }
    updatesQue_.FreeTensor(updatesLocal);
    updatesCastQue_.FreeTensor(updateSumLocal);
    this->updateSumIdxQue_.template EnQue(updateSumIdxLocal);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::CopyInAndCopyOutSlitIndices(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};

    int64_t indicesOfset = tilingData_.eachCoreIndexCount * GetBlockIdx() + rowIdx * tilingData_.ubIndexFactor;
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    for (int64_t preAxisIdx = 0; preAxisIdx < tilingData_.preAxis; preAxisIdx++) {
        int64_t rowOfset = tilingData_.afterAxis * (preAxisIdx * tilingData_.updatesInAxis + indicesOfset);
        int64_t updatesOfset = rowOfset + colIdx * tilingData_.afterAxisFactor;

        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        HandleAlpha(updatesLocal, alphaValue_, rowLen * colLenAlignSize);

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        this->CopyOutSplitIndices(indicesLocal, updatesLocal, preAxisIdx, rowLen, colLen, colIdx);
    }
    updatesQue_.FreeTensor(updatesLocal);
    indicesQue_.EnQue(indicesLocal);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::ProcessIndices()
{
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER, (tilingData_.ubIndexFactor) * sizeof(IDX_T));
    pipe_.InitBuffer(updatesQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));
    pipe_.InitBuffer(updatesCastQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(float));

    int64_t rowLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailCoreIndicesLoop :
                                                                                tilingData_.mainCoreIndicesLoop;
    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailCoreTailIndices :
                                                                                    tilingData_.mainCoreTailIndices;

    int64_t colLoopNum = tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = tilingData_.updateTailNum;

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        CopyIndiceIn(rowIdx, rowDataLen);
        if (maxScore_ > SORT_HIST_THRESHOLD && rowDataLen > INDICES_SORT_THRESHOLD) {
            LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
            if constexpr (CAST_MODE == CAST_0) {
                this->SortIndices(indicesLocal, rowDataLen);
            } else {
                LocalTensor<CAST_T> indicesCastLocal = this->indicesCastQue_.template AllocTensor<CAST_T>();
                this->IndicesSortCast(indicesLocal, indicesCastLocal, rowDataLen);
                this->SortIndices(indicesCastLocal, rowDataLen);
                this->indicesCastQue_.template FreeTensor(indicesCastLocal);
            }
            indicesQue_.FreeTensor(indicesLocal);
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                ComputeSumAndCopyOutSlitIndices(rowIdx, colIdx, rowDataLen, colDataLen);
            }
            LocalTensor<IDX_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<IDX_T>();
            LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();
            LocalTensor<uint32_t> updatesOriginIdexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
            this->updateSumIdxQue_.template FreeTensor(updateSumIdxLocal);
            this->uniqueIdCountQue_.template FreeTensor(uniqueIdCountLocal);
            this->updatesOriginIdexQue_.template FreeTensor(updatesOriginIdexLocal);
        } else {
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                CopyInAndCopyOutSlitIndices(rowIdx, colIdx, rowDataLen, colDataLen);
            }
            LocalTensor<IDX_T> indicesLocal = indicesQue_.DeQue<IDX_T>();
            indicesQue_.FreeTensor(indicesLocal);
        }
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS, uint32_t CAST_MODE>
__aicore__ inline void MoeInplaceIndexAddSimdSort<VAR_T, IDX_T, IS_CONTIGUOUS, CAST_MODE>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    if (tilingData_.isSplitPreAxis == 1) {
        ProcessPre();
    } else if (tilingData_.isSplitAfterAxis == 1) {
        ProcessAfter();
    } else if (tilingData_.isSplitIndicesAxis == 1) {
        ProcessIndices();
    }
}
} // namespace MoeInplaceIndexAdd

#endif
