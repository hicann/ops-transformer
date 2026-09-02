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
 * \file moe_inplace_index_add_simd.h
 * \brief moe_inplace_index_add_simd
 */
#ifndef ASCENDC_MOE_INPLACE_INDEX_ADD_SIMD_H_
#define ASCENDC_MOE_INPLACE_INDEX_ADD_SIMD_H_

#include "moe_inplace_index_add_common.h"
namespace MoeInplaceIndexAdd {
using namespace AscendC;

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
class MoeInplaceIndexAddSimd {
public:
    __aicore__ inline MoeInplaceIndexAddSimd(const MoeInplaceIndexAddSimdTilingData &tilingData, TPipe &pipe)
        : tilingData_(tilingData),
          pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha, GM_ADDR workspace);
    __aicore__ inline void ProcessPreSingleLoop(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessPreSmallIndices(int64_t preOfset, int64_t colIdx, int64_t preLen, int64_t colLen);
    __aicore__ inline void ProcessAfterSingleLoop(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ComputeMulWithCast(LocalTensor<VAR_T> updatesLocal, LocalTensor<VAR_T> updateMulLocal,
                                              int16_t alphaValue, int64_t colLen);
    __aicore__ inline void ProcessPre();
    __aicore__ inline void ProcessAfter();
    __aicore__ inline void ProcessIndicesSingleLoop(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessIndices();
    __aicore__ inline void Process();

private:
    using mulsSelType = typename MulsSelType<VAR_T>::type;
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;
    GlobalTensor<VAR_T> alpha_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> varQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> varCastQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesQue_;

    TPipe &pipe_;
    const MoeInplaceIndexAddSimdTilingData &tilingData_;
    int64_t blockIdx_;
    int64_t blockNum_;
    int64_t normBlockData_{0};
    int64_t usedCoreNum_{0};
    int64_t loopNum_{0};
    int64_t tailLoopLength_{0};
    mulsSelType alphaValue_{0};
    int64_t curPreAxisCount_{0};
};

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::Init(GM_ADDR var, GM_ADDR indices,
                                                                                         GM_ADDR updates, GM_ADDR alpha,
                                                                                         GM_ADDR workspace)
{
    updates_.SetGlobalBuffer((__gm__ VAR_T *)(updates));
    indices_.SetGlobalBuffer((__gm__ IDX_T *)(indices));
    alpha_.SetGlobalBuffer((__gm__ VAR_T *)(alpha));
    var_.SetGlobalBuffer((__gm__ VAR_T *)(var));
    if (tilingData_.isWithAlpha) {
        alphaValue_ = alpha_(0);
    }
    if (tilingData_.isSplitPreAxis == 1) {
        curPreAxisCount_ = (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCorePreAxisCount :
                                                                                   tilingData_.tailCorePreAxisCount);
        return;
    }
}

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::ComputeMulWithCast(
    LocalTensor<VAR_T> updatesLocal, LocalTensor<VAR_T> updateMulLocal, int16_t alphaValue, int64_t colLen)
{
    __local_mem__ VAR_T *updatesAddr = (__local_mem__ VAR_T *)updatesLocal.GetPhyAddr();
    __local_mem__ VAR_T *updateMulAddr = (__local_mem__ VAR_T *)updateMulLocal.GetPhyAddr();

    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int16_t);
    int32_t loopSize = Ops::Base::CeilDiv(static_cast<uint32_t>(colLen), vfLen);
    int32_t idLocation = 0;
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t colLenAlignIn16 = Ops::Base::CeilAlign(colLen * sizeof(int16_t), UB_AGLIN_VALUE) / sizeof(int16_t);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int16_t> sumReg;
        AscendC::MicroAPI::RegTensor<int16_t> castReg;
        AscendC::MicroAPI::RegTensor<int16_t> alphaReg;
        AscendC::MicroAPI::MaskReg maskReg;
        uint32_t maskLen = static_cast<uint32_t>(colLen);
        for (uint16_t j = 0; j < static_cast<uint16_t>(loopSize); j++) {
            maskReg = AscendC::MicroAPI::UpdateMask<int16_t>(maskLen);
            AscendC::MicroAPI::Duplicate(alphaReg, alphaValue, maskReg);
            auto updatesOffet = j * vfLen;
            LoadOneTensorForDtypeInt<VAR_T>(updatesAddr, castReg, maskReg, updatesOffet);
            AscendC::MicroAPI::Mul(sumReg, castReg, alphaReg, maskReg);
            StoreOneTensorForDtypeInt<VAR_T>(updateMulAddr, sumReg, maskReg, updatesOffet);
        }
    }
}

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::ProcessPreSmallIndices(
    int64_t preOfset, int64_t colIdx, int64_t preLen, int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();

    int64_t colLenAlignedSize = Ops::Base::CeilAlign(sizeof(VAR_T) * colLen, UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t rowLen = tilingData_.updatesInAxis * preLen;
    if constexpr (IS_CONTIGUOUS) {
        CopyIn<IDX_T>(indicesLocal, indices_, tilingData_.updatesInAxis);
    } else {
        CopyInNoContiguous<IDX_T>(indicesLocal, indices_, tilingData_.updatesInAxis, 1, tilingData_.indicesStride - 1);
    }
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    int64_t colOfset = colIdx * tilingData_.afterAxisFactor;
    int64_t startPreAxis = GetBlockIdx() * tilingData_.eachCorePreAxisCount + preOfset;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};
    int64_t rowOfset = (startPreAxis * tilingData_.updatesInAxis) * tilingData_.afterAxis;
    int64_t updatesOfset = rowOfset + colOfset;
    DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (tilingData_.isWithAlpha) {
        if constexpr (IsSameType<VAR_T, bool>::value || IsSameType<VAR_T, int8_t>::value) {
            ComputeMulWithCast(updatesLocal, updatesLocal, alphaValue_, rowLen * colLenAlignedSize);
        } else {
            AscendC::Muls(updatesLocal, updatesLocal, alphaValue_, rowLen * colLenAlignedSize);
        }
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    for (int64_t preAxisIdx = 0; preAxisIdx < preLen; preAxisIdx++) {
        int64_t rowLocalOfset = preAxisIdx * tilingData_.updatesInAxis;
        int64_t rowPreOfset = (startPreAxis + preAxisIdx) * tilingData_.varInAxis * tilingData_.afterAxis;
        for (int64_t i = 0; i < tilingData_.updatesInAxis; i++) {
            int64_t rowOutOfset = rowPreOfset + indicesLocal(i) * tilingData_.afterAxis;
            int64_t outOfset = rowOutOfset + colOfset;
            int64_t localOfset = (rowLocalOfset + i) * colLenAlignedSize;
            if constexpr (IsSameType<VAR_T, bool>::value) {
                SetAtomicMax<int8_t>();
                CopyOut<VAR_T>(var_[outOfset], updatesLocal[localOfset], colLen);
                SetAtomicNone();
            } else {
                SetAtomicAdd<VAR_T>();
                CopyOut<VAR_T>(var_[outOfset], updatesLocal[localOfset], colLen);
                SetAtomicNone();
            }
        }
    }

    updatesQue_.FreeTensor(updatesLocal);
    indicesQue_.FreeTensor(indicesLocal);
}

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::ProcessPreSingleLoop(int64_t rowIdx,
                                                                                                         int64_t colIdx,
                                                                                                         int64_t rowLen,
                                                                                                         int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();

    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t indicesOfset = tilingData_.ubIndexFactor * rowIdx;
    if constexpr (IS_CONTIGUOUS) {
        CopyIn<IDX_T>(indicesLocal, indices_[indicesOfset], rowLen);
    } else {
        CopyInNoContiguous<IDX_T>(indicesLocal, indices_[indicesOfset * tilingData_.indicesStride], rowLen, 1,
                                  tilingData_.indicesStride - 1);
    }
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    int64_t colOfset = colIdx * tilingData_.afterAxisFactor;
    int64_t startPreAxis = GetBlockIdx() * tilingData_.eachCorePreAxisCount;
    int64_t BlockCount = GetBlockIdx() * tilingData_.eachCoreAfterAxisCount;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};

    for (int64_t preAxisIdx = startPreAxis; preAxisIdx < startPreAxis + curPreAxisCount_; preAxisIdx++) {
        int64_t rowOfset = (preAxisIdx * tilingData_.updatesInAxis + indicesOfset) * tilingData_.afterAxis;
        int64_t updatesOfset = rowOfset + colOfset;
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if (tilingData_.isWithAlpha) {
            if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, bool>::value) {
                ComputeMulWithCast(updatesLocal, updatesLocal, alphaValue_, colLenAlignSize * rowLen);
            } else {
                AscendC::Muls(updatesLocal, updatesLocal, alphaValue_, colLenAlignSize * rowLen);
            }
        }

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        int64_t AxisOffset = preAxisIdx * tilingData_.varInAxis * tilingData_.afterAxis;
        for (int64_t i = 0; i < rowLen; i++) {
            rowOfset = AxisOffset + indicesLocal(i) * tilingData_.afterAxis;
            int64_t outOfset = rowOfset + BlockCount + colOfset;
            if constexpr (IsSameType<VAR_T, bool>::value) {
                SetAtomicMax<int8_t>();
                CopyOut<VAR_T>(var_[outOfset], updatesLocal[colLenAlignSize * i], colLen);
                SetAtomicNone();
            } else {
                SetAtomicAdd<VAR_T>();
                CopyOut<VAR_T>(var_[outOfset], updatesLocal[colLenAlignSize * i], colLen);
                SetAtomicNone();
            }
        }
    }
    updatesQue_.FreeTensor(updatesLocal);
    indicesQue_.FreeTensor(indicesLocal);
}

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::ProcessPre()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER, (tilingData_.ubIndexFactor) * sizeof(IDX_T));
    pipe_.InitBuffer(updatesQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));

    int64_t colLoopNum = tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = tilingData_.updateTailNum;

    int64_t rowLoopNum = tilingData_.indicesLoopSize;
    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowTailDataLen = tilingData_.indiceAxisTailNum;

    /* indices可以一次搬入的情况, 对pre分loop处理 */
    if (rowLoopNum == 1) {
        int64_t ubPreFactor = tilingData_.ubIndexFactor / tilingData_.updatesInAxis;
        int64_t preLoopNum = Ops::Base::CeilDiv(curPreAxisCount_, ubPreFactor);
        int64_t preMainDataLen = ubPreFactor;
        int64_t preTailDataLen = curPreAxisCount_ - (preLoopNum - 1) * preMainDataLen;

        for (int64_t preIdx = 0; preIdx < preLoopNum; preIdx++) {
            int64_t preDataLen = (preIdx == preLoopNum - 1) ? preTailDataLen : preMainDataLen;
            int64_t preOfset = preIdx * preMainDataLen;
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                ProcessPreSmallIndices(preOfset, colIdx, preDataLen, colDataLen);
            }
        }
        return;
    }

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            ProcessPreSingleLoop(rowIdx, colIdx, rowDataLen, colDataLen);
        }
    }
}

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::ProcessAfterSingleLoop(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();

    int64_t colLenAlignSize = Ops::Base::CeilAlign(sizeof(VAR_T) * colLen, UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t indicesOfset = rowIdx * tilingData_.ubIndexFactor;
    if constexpr (IS_CONTIGUOUS) {
        CopyIn<IDX_T>(indicesLocal, indices_[indicesOfset], rowLen);
    } else {
        CopyInNoContiguous<IDX_T>(indicesLocal, indices_[indicesOfset * tilingData_.indicesStride], rowLen, 1,
                                  tilingData_.indicesStride - 1);
    }
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    int64_t colOfset = tilingData_.afterAxisFactor * colIdx;
    int64_t BlockCount = GetBlockIdx() * tilingData_.eachCoreAfterAxisCount;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};
    for (int64_t preAxisIdx = 0; preAxisIdx < tilingData_.preAxis; preAxisIdx++) {
        int64_t rowOfset = (preAxisIdx * tilingData_.updatesInAxis + indicesOfset) * tilingData_.afterAxis;
        int64_t updatesOfset = rowOfset + BlockCount + colOfset;

        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if (tilingData_.isWithAlpha) {
            if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, bool>::value) {
                ComputeMulWithCast(updatesLocal, updatesLocal, alphaValue_, rowLen * colLenAlignSize);
            } else {
                AscendC::Muls(updatesLocal, updatesLocal, alphaValue_, rowLen * colLenAlignSize);
            }
        }

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        int64_t AxisOfset = preAxisIdx * tilingData_.varInAxis * tilingData_.afterAxis;
        for (int64_t i = 0; i < rowLen; i++) {
            rowOfset = AxisOfset + indicesLocal(i) * tilingData_.afterAxis;
            int64_t outOfset = rowOfset + BlockCount + colOfset;
            if constexpr (IsSameType<VAR_T, bool>::value) {
                SetAtomicMax<int8_t>();
                CopyOut<VAR_T>(var_[outOfset], updatesLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            } else {
                SetAtomicAdd<VAR_T>();
                CopyOut<VAR_T>(var_[outOfset], updatesLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            }
        }
    }
    updatesQue_.FreeTensor(updatesLocal);
    indicesQue_.FreeTensor(indicesLocal);
}

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::ProcessAfter()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    pipe_.InitBuffer(updatesQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER, (tilingData_.ubIndexFactor) * sizeof(IDX_T));

    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowLoopNum = tilingData_.indicesLoopSize;
    int64_t rowTailDataLen = tilingData_.indiceAxisTailNum;

    int64_t colLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateLoopSize :
                                                                                tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateAxisNum :
                                                                                    tilingData_.updateTailNum;

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            ProcessAfterSingleLoop(rowIdx, colIdx, rowDataLen, colDataLen);
        }
    }
}

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::ProcessIndicesSingleLoop(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();

    int64_t colLenAlignSize = Ops::Base::CeilAlign(sizeof(VAR_T) * colLen, UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t indicesOfset = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.ubIndexFactor;
    if constexpr (IS_CONTIGUOUS) {
        CopyIn<IDX_T>(indicesLocal, indices_[indicesOfset], rowLen);
    } else {
        CopyInNoContiguous<IDX_T>(indicesLocal, indices_[indicesOfset * tilingData_.indicesStride], rowLen, 1,
                                  tilingData_.indicesStride - 1);
    }
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    int64_t colOfset = tilingData_.afterAxisFactor * colIdx;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};
    for (int64_t preAxisIdx = 0; preAxisIdx < tilingData_.preAxis; preAxisIdx++) {
        int64_t rowOfset = (preAxisIdx * tilingData_.updatesInAxis + indicesOfset) * tilingData_.afterAxis;
        int64_t updatesOfset = rowOfset + colOfset;

        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if (tilingData_.isWithAlpha) {
            if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, bool>::value) {
                ComputeMulWithCast(updatesLocal, updatesLocal, alphaValue_, rowLen * colLenAlignSize);
            } else {
                AscendC::Muls(updatesLocal, updatesLocal, alphaValue_, rowLen * colLenAlignSize);
            }
        }

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        int64_t varOfset = preAxisIdx * tilingData_.varInAxis;
        for (int64_t i = 0; i < rowLen; i++) {
            rowOfset = (varOfset + indicesLocal(i)) * tilingData_.afterAxis;
            int64_t outOfset = rowOfset + colOfset;
            if constexpr (IsSameType<VAR_T, bool>::value) {
                SetAtomicMax<int8_t>();
                CopyOut<VAR_T>(var_[outOfset], updatesLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            } else {
                SetAtomicAdd<VAR_T>();
                CopyOut<VAR_T>(var_[outOfset], updatesLocal[i * colLenAlignSize], colLen);
                SetAtomicNone();
            }
        }
    }
    updatesQue_.FreeTensor(updatesLocal);
    indicesQue_.FreeTensor(indicesLocal);
}

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::ProcessIndices()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER, (tilingData_.ubIndexFactor) * sizeof(IDX_T));
    pipe_.InitBuffer(updatesQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));

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
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            ProcessIndicesSingleLoop(rowIdx, colIdx, rowDataLen, colDataLen);
        }
    }
}

template <typename VAR_T, typename IDX_T, typename CAST_T, bool IS_CONTIGUOUS>
__aicore__ inline void MoeInplaceIndexAddSimd<VAR_T, IDX_T, CAST_T, IS_CONTIGUOUS>::Process()
{
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
