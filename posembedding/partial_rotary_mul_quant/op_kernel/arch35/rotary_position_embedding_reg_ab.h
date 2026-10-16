/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotary_position_embedding_reg_ab.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_REG_AB_H
#define ROTARY_POSITION_EMBEDDING_REG_AB_H

#include "partial_rotary_mul_quant_hif4_common.h"

namespace PartialRotaryMulQuantOps {
using namespace AscendC::MicroAPI;
using namespace AscendC;

template <typename T>
class RotaryPositionEmbeddingAB {
public:
    __aicore__ inline RotaryPositionEmbeddingAB(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR y_scale, GM_ADDR workspace,
                                const QuantRopeRegbaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessLoop(int64_t xGmOffset, LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer,
                                       int64_t ubIdx, int64_t bsCount, int64_t nCount);
    __aicore__ inline void CopyIn(GlobalTensor<T>& source, int64_t xGmOffset, int64_t ubIdx, int64_t totalCount);
    __aicore__ inline void ComputeAndQuantize(LocalTensor<T>& inUb, LocalTensor<T>& cosBuffer,
                                              LocalTensor<T>& sinBuffer, LocalTensor<int32_t>& yQuantLocal,
                                              LocalTensor<uint16_t>& scaleLocal, int64_t bsCount, int64_t nCount,
                                              int64_t sliceLen);
    __aicore__ inline void CopyOut(int64_t xGmOffset, int64_t ubIdx, int64_t totalCount, int64_t dCount);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> cosInQueue_;
    TQue<QuePosition::VECIN, 1> sinInQueue_;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECOUT, 1> scaleQueue_;

    TBuf<TPosition::VECCALC> ropeBuf_;
    TBuf<TPosition::VECCALC> eaBuf_;
    TBuf<TPosition::VECCALC> ebBuf_;
    TBuf<TPosition::VECCALC> ecBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<float> scaleGm_;
    const QuantRopeRegbaseTilingData* tilingData_;
    DataCopyPadExtParams<T> padParams_ = {false, 0, 0, static_cast<T>(0)};
    uint8_t DB_FLAG = 2;
    uint32_t dSplitSize_ = 0;
    int64_t bsBlockCount_ = 0;
    int64_t nBlockCount_ = 0;
    int64_t dAlign_ = 0;
    int64_t dAlignH8_ = 0;
    int64_t sliceAlign_ = 0;
};

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                          GM_ADDR y_scale, GM_ADDR workspace,
                                                          const QuantRopeRegbaseTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    tilingData_ = tilingData;
    dSplitSize_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(T);
    int64_t blockDimBS = GetBlockIdx() / tilingData_->blockNumN;
    int64_t blockDimN = GetBlockIdx() % tilingData_->blockNumN;
    bsBlockCount_ = (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    nBlockCount_ = (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN : tilingData_->blockFactorN;

    int64_t cosOffset = blockDimBS * tilingData_->blockFactorBS * tilingData_->sliceLength;
    int64_t offset = blockDimBS * tilingData_->blockFactorBS * tilingData_->D;
    int64_t xOffset = offset * tilingData_->N + blockDimN * tilingData_->blockFactorN * tilingData_->D;
    this->cosGm_.SetGlobalBuffer((__gm__ T*)cos + cosOffset);
    this->sinGm_.SetGlobalBuffer((__gm__ T*)sin + cosOffset);
    this->xGm_.SetGlobalBuffer((__gm__ T*)x + xOffset);
    this->yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xOffset / 2);
    this->scaleGm_.SetGlobalBuffer((__gm__ float*)y_scale + xOffset / 64);

    dAlign_ = ops::CeilAlign<int64_t>(tilingData_->D, GetUbBlockSize() / sizeof(T));
    dAlignH8_ = ops::CeilAlign<int64_t>(tilingData_->sliceLength, GetUbBlockSize() / sizeof(uint8_t));
    sliceAlign_ =
        ops::CeilAlign<int64_t>(tilingData_->sliceLength / tilingData_->dSplitCoef, GetUbBlockSize() / sizeof(T)) *
        tilingData_->dSplitCoef;
    pipe_->InitBuffer(inQueue_, DB_FLAG, dAlign_ * sizeof(T) * tilingData_->ubFactorBS * tilingData_->ubFactorN);
    if constexpr (IsSameType<T, float>::value) {
        this->pipe_->InitBuffer(this->ropeBuf_,
                                dAlign_ * sizeof(bfloat16_t) * tilingData_->ubFactorBS * tilingData_->ubFactorN);
    }
    pipe_->InitBuffer(cosInQueue_, DB_FLAG, sliceAlign_ * sizeof(T) * tilingData_->ubFactorBS);
    pipe_->InitBuffer(sinInQueue_, DB_FLAG, sliceAlign_ * sizeof(T) * tilingData_->ubFactorBS);
    pipe_->InitBuffer(eaBuf_, tilingData_->ubFactorBS * tilingData_->ubFactorN * dAlign_ / 64);
    pipe_->InitBuffer(ebBuf_, tilingData_->ubFactorBS * tilingData_->ubFactorN * dAlign_ / 64);
    pipe_->InitBuffer(ecBuf_, tilingData_->ubFactorBS * tilingData_->ubFactorN * dAlign_ / 32);
    pipe_->InitBuffer(outQueue_, 1, dAlign_ * tilingData_->ubFactorBS * tilingData_->ubFactorN / 2); // hifloat4_t
    pipe_->InitBuffer(scaleQueue_, 1,
                      dAlign_ * tilingData_->ubFactorBS * tilingData_->ubFactorN / 64 * 4); // hifloat4_scale
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::Process()
{
    uint32_t bsLoopCnt = ops::CeilDiv(bsBlockCount_, tilingData_->ubFactorBS);
    uint32_t nLoopCnt = ops::CeilDiv(nBlockCount_, tilingData_->ubFactorN);
    for (uint32_t bsLoopIdx = 0; bsLoopIdx < bsLoopCnt; bsLoopIdx++) {
        int64_t xGmOffset = bsLoopIdx * tilingData_->ubFactorBS * tilingData_->N * tilingData_->D;
        uint32_t currBSNum = (bsLoopIdx != bsLoopCnt - 1) ? tilingData_->ubFactorBS :
                                                            bsBlockCount_ - (bsLoopIdx * tilingData_->ubFactorBS);

        DataCopyExtParams cosParams = {
            static_cast<uint16_t>(currBSNum * tilingData_->dSplitCoef), dSplitSize_, 0,
            static_cast<uint32_t>((sliceAlign_ - tilingData_->sliceLength) * sizeof(T) / tilingData_->dSplitCoef) /
                ops::BLOCK_BYTES,
            0};

        LocalTensor<T> cosBuffer = cosInQueue_.AllocTensor<T>();
        LocalTensor<T> sinBuffer = sinInQueue_.AllocTensor<T>();
        DataCopyPad(cosBuffer, cosGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->sliceLength], cosParams,
                    padParams_);
        cosInQueue_.EnQue(cosBuffer);
        cosBuffer = cosInQueue_.DeQue<T>();
        DataCopyPad(sinBuffer, sinGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->sliceLength], cosParams,
                    padParams_);
        sinInQueue_.EnQue(sinBuffer);
        sinBuffer = sinInQueue_.DeQue<T>();

        for (int64_t nLoopIdx = 0; nLoopIdx < nLoopCnt; nLoopIdx++) {
            int64_t currNNum = (nLoopIdx != nLoopCnt - 1) ? tilingData_->ubFactorN :
                                                            nBlockCount_ - (nLoopIdx * tilingData_->ubFactorN);
            ProcessLoop(xGmOffset, cosBuffer, sinBuffer, nLoopIdx, currBSNum, currNNum);
        }

        cosInQueue_.FreeTensor(cosBuffer);
        sinInQueue_.FreeTensor(sinBuffer);
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::ProcessLoop(int64_t xGmOffset, LocalTensor<T> cosBuffer,
                                                                 LocalTensor<T> sinBuffer, int64_t ubIdx,
                                                                 int64_t bsCount, int64_t nCount)
{
    int64_t totalCount = bsCount * nCount;
    int64_t sliceLen = tilingData_->sliceLength;

    CopyIn(xGm_, xGmOffset, ubIdx, totalCount);
    LocalTensor<T> inUb = inQueue_.DeQue<T>();

    LocalTensor<int32_t> yQuantLocal = outQueue_.AllocTensor<int32_t>();
    LocalTensor<uint16_t> scaleLocal = scaleQueue_.AllocTensor<uint16_t>();

    ComputeAndQuantize(inUb, cosBuffer, sinBuffer, yQuantLocal, scaleLocal, bsCount, nCount, sliceLen);
    CopyOut(xGmOffset, ubIdx, totalCount, tilingData_->D);

    inQueue_.FreeTensor(inUb);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::CopyIn(GlobalTensor<T>& source, int64_t xGmOffset, int64_t ubIdx,
                                                            int64_t totalCount)
{
    LocalTensor<T> target = inQueue_.AllocTensor<T>();
    DataCopyExtParams inParams = {static_cast<uint16_t>(totalCount), static_cast<uint32_t>(tilingData_->D * sizeof(T)),
                                  0, 0, 0};
    DataCopyPad(target, source[xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D], inParams, padParams_);
    inQueue_.EnQue(target);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::ComputeAndQuantize(
    LocalTensor<T>& inUb, LocalTensor<T>& cosBuffer, LocalTensor<T>& sinBuffer, LocalTensor<int32_t>& yQuantLocal,
    LocalTensor<uint16_t>& scaleLocal, int64_t bsCount, int64_t nCount, int64_t sliceLen)
{
    LocalTensor<float> outUb = inUb.template ReinterpretCast<float>();
    if constexpr (IsSameType<T, float>::value) {
        outUb = ropeBuf_.Get<float>();
    }
    __local_mem__ T* inAddr = (__local_mem__ T*)inUb.GetPhyAddr();
    __local_mem__ T* sinAddr = (__local_mem__ T*)sinBuffer.GetPhyAddr();
    __local_mem__ T* cosAddr = (__local_mem__ T*)cosBuffer.GetPhyAddr();
    __local_mem__ bfloat16_t* outAddr = (__local_mem__ bfloat16_t*)outUb.GetPhyAddr();
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
        HalfAlignVF<T>(sinAddr, cosAddr, inAddr, outAddr, tilingData_->sliceLength, dAlign_, sliceAlign_, dAlign_,
                       bsCount, nCount, tilingData_->sliceStart);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
        InterleaveModeVF<T>(sinAddr, cosAddr, inAddr, outAddr, tilingData_->sliceLength, dAlign_, sliceAlign_, dAlign_,
                            bsCount, nCount, tilingData_->sliceStart);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
    } else {
    }

    LocalTensor<uint8_t> eaLocal = eaBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> ebLocal = ebBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> ecLocal = ecBuf_.Get<uint8_t>();

    VFProcessSwigluHiF4InvScaleAndQuant(yQuantLocal, outUb, eaLocal, ebLocal.ReinterpretCast<uint32_t>(), ecLocal,
                                        bsCount * nCount * tilingData_->D);
    VFProcessSwigluGroupQuantScale(scaleLocal, eaLocal, ebLocal, ecLocal.ReinterpretCast<uint16_t>(),
                                   bsCount * nCount * tilingData_->D);

    outQueue_.template EnQue(yQuantLocal);
    scaleQueue_.template EnQue(scaleLocal);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::CopyOut(int64_t xGmOffset, int64_t ubIdx, int64_t totalCount,
                                                             int64_t dCount)
{
    LocalTensor<uint8_t> yQuantLocal = outQueue_.template DeQue<uint8_t>();
    LocalTensor<float> scaleLocal = scaleQueue_.template DeQue<float>();
    DataCopyExtParams outParams = {1, static_cast<uint32_t>(totalCount * dCount * sizeof(uint8_t) / 2), 0, 0, 0};
    DataCopyExtParams scaleParams = {1, static_cast<uint32_t>(totalCount * dCount * sizeof(float) / 64), 0, 0, 0};
    DataCopyPad(yGm_[(xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D) / 2], yQuantLocal, outParams);
    DataCopyPad(scaleGm_[(xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D) / 64], scaleLocal, scaleParams);

    outQueue_.FreeTensor(yQuantLocal);
    scaleQueue_.FreeTensor(scaleLocal);
}

} // namespace PartialRotaryMulQuantOps

#endif // ROTARY_POSITION_EMBEDDING_REG_AB_H
