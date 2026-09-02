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
 * \file moe_inplace_index_add_determinstic_notquant.h
 * \brief moe_inplace_index_add_determinstic_notquant
 */
#ifndef ASCENDC_MOE_INPLACE_INDEX_ADD_DETERMINSTIC_NOTQUANT_H_
#define ASCENDC_MOE_INPLACE_INDEX_ADD_DETERMINSTIC_NOTQUANT_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "moe_inplace_index_add_common.h"

namespace MoeInplaceIndexAdd {
using namespace AscendC;

constexpr uint64_t ALIGN_SIZE = 32;
constexpr int64_t MAXSCORE_SORT_THRESHOLD = 128;

template <typename VAR_T, typename IDX_T>
class MoeInplaceIndexAddDeterminsticNotQuant : public MoeInplaceIndexAddBase<VAR_T, IDX_T, CAST_0> {
public:
    __aicore__ inline MoeInplaceIndexAddDeterminsticNotQuant(const MoeInplaceIndexAddDeterminsticTilingData &tilingData,
                                                             TPipe &pipe)
        : tilingData_(tilingData),
          pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha, GM_ADDR workspace);
    __aicore__ inline void CopyXToOut(__local_mem__ int8_t *inAddr, __local_mem__ int8_t *outAddr, int64_t dataCount);
    __aicore__ inline void ProcessPreSingleLoop(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessPreSmallIndices(int64_t preOfset, int64_t colIdx, int64_t preLen, int64_t colLen);
    __aicore__ inline void ProcessAfterSingleLoop(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessDeterminsticPre();
    __aicore__ inline void ProcessDeterminsticAfter();
    __aicore__ inline void Process();

private:
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;
    GlobalTensor<VAR_T> alpha_;

    TQue<QuePosition::VECIN, 1> indicesQue_;
    TQue<QuePosition::VECIN, 1> updatesQue_;
    TQue<QuePosition::VECOUT, 1> updatesCastQue_;
    TBuf<QuePosition::VECCALC> maxScoreBuf_;

    TPipe &pipe_;
    const MoeInplaceIndexAddDeterminsticTilingData &tilingData_;

    VAR_T alphaValue_{0};
    int64_t curPreAxisCount_{0};
    int64_t uniqueIdNumDuplicateIdx_ = 0;
};

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminsticNotQuant<VAR_T, IDX_T>::Init(GM_ADDR var, GM_ADDR indices,
                                                                                  GM_ADDR updates, GM_ADDR alpha,
                                                                                  GM_ADDR workspace)
{
    var_.SetGlobalBuffer((__gm__ VAR_T *)(var));
    indices_.SetGlobalBuffer((__gm__ IDX_T *)(indices));
    updates_.SetGlobalBuffer((__gm__ VAR_T *)(updates));
    alpha_.SetGlobalBuffer((__gm__ VAR_T *)(alpha));

    pipe_.InitBuffer(maxScoreBuf_, MAXSCORE_SORT_THRESHOLD * sizeof(float));
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
        return;
    }
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminsticNotQuant<VAR_T, IDX_T>::CopyXToOut(__local_mem__ int8_t *inAddr,
                                                                                        __local_mem__ int8_t *outAddr,
                                                                                        int64_t dataCount)
{
    uint32_t totalBytes = dataCount * sizeof(VAR_T);
    uint16_t stride = Ops::Base::GetVRegSize();
    uint16_t size = (totalBytes + stride - 1) / stride;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int8_t> inputRegTensor;
        uint32_t sreg = totalBytes;
        AscendC::MicroAPI::MaskReg preg;

        for (uint16_t i = 0; i < size; i++) {
            preg = AscendC::MicroAPI::UpdateMask<int8_t>(sreg);
            AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<int8_t>(i, stride);
            AscendC::MicroAPI::DataCopy(inputRegTensor, inAddr, offset);
            AscendC::MicroAPI::DataCopy(outAddr, inputRegTensor, offset, preg);
        }
    }
    return;
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminsticNotQuant<VAR_T, IDX_T>::ProcessPreSmallIndices(int64_t preOfset,
                                                                                                    int64_t colIdx,
                                                                                                    int64_t preLen,
                                                                                                    int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
    LocalTensor<VAR_T> updateOutLocal = updatesCastQue_.AllocTensor<VAR_T>();

    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t rowLen = preLen * tilingData_.updatesInAxis;
    CopyIn<IDX_T>(indicesLocal, indices_, tilingData_.updatesInAxis);

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};

    int64_t startPreAxis = GetBlockIdx() * tilingData_.eachCorePreAxisCount + preOfset;
    int64_t rowOfset = (startPreAxis * tilingData_.updatesInAxis) * tilingData_.afterAxis;
    int64_t updatesOfset = rowOfset + colIdx * tilingData_.afterAxisFactor;
    DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);
    updatesQue_.EnQue(updatesLocal);

    updatesLocal = updatesQue_.DeQue<VAR_T>();
    if (tilingData_.isWithAlpha) {
        AscendC::Muls(updateOutLocal, updatesLocal, alphaValue_, rowLen * colLenAlignSize);
    } else {
        auto inAddr = reinterpret_cast<__local_mem__ int8_t *>(updatesLocal.GetPhyAddr());
        auto outAddr = reinterpret_cast<__local_mem__ int8_t *>(updateOutLocal.GetPhyAddr());
        CopyXToOut(inAddr, outAddr, rowLen * colLenAlignSize);
    }
    updatesCastQue_.EnQue(updateOutLocal);

    updateOutLocal = updatesCastQue_.DeQue<VAR_T>();
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    SetAtomicAdd<VAR_T>();
    for (int64_t preAxisIdx = 0; preAxisIdx < preLen; preAxisIdx++) {
        int64_t rowLocalOfset = preAxisIdx * tilingData_.updatesInAxis;
        for (int64_t i = 0; i < tilingData_.updatesInAxis; i++) {
            int64_t rowOutOfset =
                ((startPreAxis + preAxisIdx) * tilingData_.varInAxis + indicesLocal(i)) * tilingData_.afterAxis;
            int64_t outOfset = rowOutOfset + colIdx * tilingData_.afterAxisFactor;
            int64_t localOfset = (rowLocalOfset + i) * colLenAlignSize;
            CopyOut<VAR_T>(var_[outOfset], updateOutLocal[localOfset], colLen);
            PipeBarrier<PIPE_MTE3>();
        }
    }
    SetAtomicNone();
    updatesQue_.FreeTensor(updatesLocal);
    updatesCastQue_.FreeTensor(updateOutLocal);
    indicesQue_.FreeTensor(indicesLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminsticNotQuant<VAR_T, IDX_T>::ProcessPreSingleLoop(int64_t rowIdx,
                                                                                                  int64_t colIdx,
                                                                                                  int64_t rowLen,
                                                                                                  int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t indicesOfset = rowIdx * tilingData_.ubIndexFactor;
    CopyIn<IDX_T>(indicesLocal, indices_[indicesOfset], rowLen);

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    int64_t startPreAxis = GetBlockIdx() * tilingData_.eachCorePreAxisCount;
    for (int64_t preAxisIdx = startPreAxis; preAxisIdx < startPreAxis + curPreAxisCount_; preAxisIdx++) {
        int64_t rowOfset = (preAxisIdx * tilingData_.updatesInAxis + indicesOfset) * tilingData_.afterAxis;
        int64_t updatesOfset = rowOfset + colIdx * tilingData_.afterAxisFactor;
        LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);
        updatesQue_.EnQue(updatesLocal);

        updatesLocal = updatesQue_.DeQue<VAR_T>();
        LocalTensor<VAR_T> updateOutLocal = updatesCastQue_.AllocTensor<VAR_T>();
        if (tilingData_.isWithAlpha) {
            AscendC::Muls(updateOutLocal, updatesLocal, alphaValue_, rowLen * colLenAlignSize);
        } else {
            auto inAddr = reinterpret_cast<__local_mem__ int8_t *>(updatesLocal.GetPhyAddr());
            auto outAddr = reinterpret_cast<__local_mem__ int8_t *>(updateOutLocal.GetPhyAddr());
            CopyXToOut(inAddr, outAddr, rowLen * colLenAlignSize);
        }
        updatesCastQue_.EnQue(updateOutLocal);
        updatesQue_.FreeTensor(updatesLocal);

        updateOutLocal = updatesCastQue_.DeQue<VAR_T>();
        SetAtomicAdd<VAR_T>();
        for (int64_t i = 0; i < rowLen; i++) {
            rowOfset = (preAxisIdx * tilingData_.varInAxis + indicesLocal(i)) * tilingData_.afterAxis;
            int64_t outOfset = rowOfset + colIdx * tilingData_.afterAxisFactor;
            CopyOut<VAR_T>(var_[outOfset], updateOutLocal[i * colLenAlignSize], colLen);
            PipeBarrier<PIPE_MTE3>();
        }
        SetAtomicNone();
        updatesCastQue_.FreeTensor(updateOutLocal);
    }
    indicesQue_.FreeTensor(indicesLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminsticNotQuant<VAR_T, IDX_T>::ProcessDeterminsticPre()
{
    pipe_.InitBuffer(indicesQue_, 1, (tilingData_.ubIndexFactor) * sizeof(IDX_T));
    pipe_.InitBuffer(updatesQue_, 1, tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));
    pipe_.InitBuffer(updatesCastQue_, 1, tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));

    int64_t rowLoopNum = tilingData_.indicesLoopSize;
    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowTailDataLen = tilingData_.indiceAxisTailNum;

    int64_t colLoopNum = tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = tilingData_.updateTailNum;

    /* indices可以一次搬入的情况, 对pre分loop处理 */
    if (rowLoopNum == 1) {
        int64_t ubPreFactor = tilingData_.ubIndexFactor / tilingData_.updatesInAxis;
        int64_t preLoopNum = Ops::Base::CeilDiv(curPreAxisCount_, ubPreFactor);
        int64_t preMainDataLen = ubPreFactor;
        int64_t preTailDataLen = curPreAxisCount_ - preMainDataLen * (preLoopNum - 1);

        for (int64_t preIdx = 0; preIdx < preLoopNum - 1; preIdx++) {
            int64_t preOfset = preIdx * preMainDataLen;
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                ProcessPreSmallIndices(preOfset, colIdx, preMainDataLen, colDataLen);
            }
        }
        int64_t preOfset = (preLoopNum - 1) * preMainDataLen;
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            ProcessPreSmallIndices(preOfset, colIdx, preTailDataLen, colDataLen);
        }
        return;
    }

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            ProcessPreSingleLoop(rowIdx, colIdx, rowMainDataLen, colDataLen);
        }
    }
    for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
        int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
        ProcessPreSingleLoop(rowLoopNum - 1, colIdx, rowTailDataLen, colDataLen);
    }
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminsticNotQuant<VAR_T, IDX_T>::ProcessAfterSingleLoop(int64_t rowIdx,
                                                                                                    int64_t colIdx,
                                                                                                    int64_t rowLen,
                                                                                                    int64_t colLen)
{
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t indicesOfset = rowIdx * tilingData_.ubIndexFactor;
    CopyIn<IDX_T>(indicesLocal, indices_[indicesOfset], rowLen);

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};
    indicesQue_.EnQue(indicesLocal);

    indicesLocal = indicesQue_.DeQue<IDX_T>();
    this->ComputeIdxCount(indicesLocal, rowLen, uniqueIdNumDuplicateIdx_);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    for (int64_t preAxisIdx = 0; preAxisIdx < tilingData_.preAxis; preAxisIdx++) {
        int64_t rowOfset = (preAxisIdx * tilingData_.updatesInAxis + indicesOfset) * tilingData_.afterAxis;
        int64_t updatesOfset =
            rowOfset + GetBlockIdx() * tilingData_.eachCoreAfterAxisCount + colIdx * tilingData_.afterAxisFactor;
        LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
        DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);
        updatesQue_.EnQue(updatesLocal);

        updatesLocal = updatesQue_.DeQue<VAR_T>();
        LocalTensor<VAR_T> updateOutLocal = updatesCastQue_.AllocTensor<VAR_T>();
        if (tilingData_.isWithAlpha) {
            AscendC::Muls(updateOutLocal, updatesLocal, alphaValue_, rowLen * colLenAlignSize);
        } else {
            auto inAddr = reinterpret_cast<__local_mem__ int8_t *>(updatesLocal.GetPhyAddr());
            auto outAddr = reinterpret_cast<__local_mem__ int8_t *>(updateOutLocal.GetPhyAddr());
            CopyXToOut(inAddr, outAddr, rowLen * colLenAlignSize);
        }
        updatesCastQue_.EnQue(updateOutLocal);
        updatesQue_.FreeTensor(updatesLocal);

        updateOutLocal = updatesCastQue_.DeQue<VAR_T>();
        SetAtomicAdd<VAR_T>();
        for (int64_t i = 0; i < rowLen; i++) {
            rowOfset = (preAxisIdx * tilingData_.varInAxis + indicesLocal(i)) * tilingData_.afterAxis;
            int64_t outOfset =
                rowOfset + GetBlockIdx() * tilingData_.eachCoreAfterAxisCount + colIdx * tilingData_.afterAxisFactor;

            if (uniqueIdNumDuplicateIdx_ != rowLen) {
                PipeBarrier<PIPE_MTE3>();
            }
            CopyOut<VAR_T>(var_[outOfset], updateOutLocal[i * colLenAlignSize], colLen);
        }
        SetAtomicNone();
        updatesCastQue_.FreeTensor(updateOutLocal);
    }
    indicesQue_.FreeTensor(indicesLocal);
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminsticNotQuant<VAR_T, IDX_T>::ProcessDeterminsticAfter()
{
    pipe_.InitBuffer(indicesQue_, 1, Ops::Base::CeilAlign(tilingData_.ubIndexFactor * sizeof(IDX_T), ALIGN_SIZE));
    pipe_.InitBuffer(updatesQue_, 1, tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));
    pipe_.InitBuffer(updatesCastQue_, 1, tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));

    int64_t rowLoopNum = tilingData_.indicesLoopSize;
    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowTailDataLen = tilingData_.indiceAxisTailNum;

    int64_t colLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateLoopSize :
                                                                                tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateAxisNum :
                                                                                    tilingData_.updateTailNum;

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            ProcessAfterSingleLoop(rowIdx, colIdx, rowMainDataLen, colDataLen);
        }
    }
    for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
        int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
        ProcessAfterSingleLoop(rowLoopNum - 1, colIdx, rowTailDataLen, colDataLen);
    }
}

template <typename VAR_T, typename IDX_T>
__aicore__ inline void MoeInplaceIndexAddDeterminsticNotQuant<VAR_T, IDX_T>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    if (tilingData_.isSplitPreAxis == 1) {
        ProcessDeterminsticPre();
    } else if (tilingData_.isSplitAfterAxis == 1) {
        ProcessDeterminsticAfter();
    }
}
} // namespace MoeInplaceIndexAdd

#endif
