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
 * \file nsa_compress_grad_bash.h
 * \brief nsa_compress_grad_bash head file
 */

#ifndef NSA_COMPRESS_GRAD_BASE_H
#define NSA_COMPRESS_GRAD_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace NsaCompressGrad {

using namespace AscendC;

template <typename T>
class NsaCompressGradND {
public:
    __aicore__ inline NsaCompressGradND(){};
    __aicore__ inline void Init(GM_ADDR outputGrad, GM_ADDR input, GM_ADDR weight, GM_ADDR actSeqLen, GM_ADDR inputGrad,
                                GM_ADDR weightGrad, GM_ADDR workspace, const NsaCompressGradTilingData* tilingData,
                                TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void MoveResult();

private:
    __aicore__ inline void InitTilingData(const NsaCompressGradTilingData* tilingData);
    __aicore__ inline void InitGlobalTensor(GM_ADDR outputGrad, GM_ADDR input, GM_ADDR weight, GM_ADDR actSeqLen,
                                            GM_ADDR inputGrad, GM_ADDR weightGrad);
    __aicore__ inline void InitWorkspace(GM_ADDR workspace);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void calculateProcessInfo();
    __aicore__ inline void SetWsToZero();
    __aicore__ inline void CopyInCmpGrad(uint32_t offsetCmpGrad, uint32_t size);
    __aicore__ inline void CopyInInput(uint32_t offset, uint32_t size);
    __aicore__ inline void ComputeWtGrad(uint32_t size);
    __aicore__ inline void ComputeInputGrad(uint32_t size);
    __aicore__ inline void CopyOutWtGrad(uint32_t offsetWt, uint32_t size);
    __aicore__ inline void CopyInWt(uint32_t offset, uint32_t size);
    __aicore__ inline void CopyOutInputGrad(uint32_t offsetWt, uint32_t size);
    __aicore__ inline void CastByCondition(LocalTensor<T> dstTensor, LocalTensor<float> srcTensor, uint32_t castSize,
                                           bool condition);
    __aicore__ inline void MoveInputGrad();
    __aicore__ inline void MoveWtGrad();
    __aicore__ inline void PresetFlag();
    __aicore__ inline void ClearPresetFlag();
    __aicore__ inline void DeterministicComputeSumWtGrad();
    __aicore__ inline uint32_t GetSeqLenTarBch(uint32_t batchIdx);

private:
    TPipe* pipe_;

    // Global tensors
    GlobalTensor<T> outputGradGm_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> weightGm_;
    GlobalTensor<int64_t> actSeqLenListGm_;
    GlobalTensor<T> inputGradGm_;
    GlobalTensor<T> weightGradGm_;
    GlobalTensor<float> wtGradAtomicWs_;
    GlobalTensor<float> inputGradAtomicWs_;

    // UB Buffers
    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<float> cmpGrad;
    LocalTensor<float> inputFp32Ub_;
    LocalTensor<float> tmpx;
    LocalTensor<float> weightFp32;
    LocalTensor<float> weightBrdc;
    LocalTensor<T> cmpGradHf;
    LocalTensor<T> inputHf;
    LocalTensor<T> weightHf;
    LocalTensor<uint8_t> tmpBuffer;
    LocalTensor<float> transInputGradPing_;
    LocalTensor<float> transInputGradPong_;
    LocalTensor<T> halfTransInputGradPing_;
    LocalTensor<T> halfTransInputGradPong_;
    LocalTensor<float> transDeterministicWtGrad_;
    LocalTensor<T> halfTransDeterministicWtGrad_;
    LocalTensor<float> transWtGrad_;
    LocalTensor<T> halfTransWtGrad_;

    // UB输出
    LocalTensor<T> inputGradHf;
    LocalTensor<T> wtGradHf;

    uint32_t batchSize_;
    uint32_t headDim_;
    uint32_t headNum_;
    uint32_t blockSize_;
    uint32_t blockStride_;
    uint32_t totalBlockNum_;
    uint32_t seqLenType_;
    uint32_t headToProcess_;
    uint32_t headRemainder_;
    uint32_t headOffsetCurCore_;
    uint32_t nBlockCurCore_;
    uint32_t startBatchIdx_ = 0;
    uint32_t startBlcIdx_g = 0;
    uint32_t blcIdxOfStartBatch_ = 0;
    uint32_t batchOffsetCurCore_ = 0;
    uint32_t coreNum_;
    uint32_t coreId_;

    uint32_t ubCalcSize_;
    uint32_t ubMaxSize_;
    uint32_t maxHeadNumInUb_ = 1;
    uint32_t nRowsOnce_ = 1;
    uint32_t nHeadOnce_ = 1;
    bool isOverLap_ = true;
    bool deterministicFlag_ = true;
};

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::CastByCondition(LocalTensor<T> dstTensor, LocalTensor<float> srcTensor,
                                                             uint32_t castSize, bool condition)
{
    if (!condition) {
        return;
    }
    if (std::is_same_v<T, half>) {
        Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, castSize);
    } else if (std::is_same_v<T, bfloat16_t>) {
        Cast(dstTensor, srcTensor, RoundMode::CAST_RINT, castSize);
    }
}

template <typename T>
__aicore__ inline uint32_t NsaCompressGradND<T>::GetSeqLenTarBch(uint32_t batchIdx)
{
    if (seqLenType_ == 1) {
        return actSeqLenListGm_.GetValue(batchIdx);
    }

    if (batchIdx == 0) {
        return actSeqLenListGm_.GetValue(batchIdx);
    } else {
        return actSeqLenListGm_.GetValue(batchIdx) - actSeqLenListGm_.GetValue(batchIdx - 1);
    }
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::PresetFlag()
{
    // weight的搬运
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    // inputKV的搬运
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    // cmpGrad的搬运
    SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::ClearPresetFlag()
{
    // weight的搬运
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    // inputKV的搬运
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    // cmpGrad的搬运
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
}
} // namespace NsaCompressGrad

#endif
