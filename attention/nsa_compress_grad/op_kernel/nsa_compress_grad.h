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
 * \file compress_backward.h
 * \brief compress_backward head file
 */

#ifndef NSA_COMPRESS_GRAD_H
#define NSA_COMPRESS_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "nsa_compress_grad_common.h"
#include "nsa_compress_grad_base.h"

namespace NsaCompressGrad {

using namespace AscendC;

constexpr uint32_t BLOCK_16 = 16;
constexpr uint32_t ONE_KILO = 1024;
constexpr float ZERO_FLOAT = 0.0;
constexpr uint32_t UB_PRESERVE = 24;

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::Init(GM_ADDR outputGrad, GM_ADDR input, GM_ADDR weight, GM_ADDR actSeqLen,
                                                  GM_ADDR inputGrad, GM_ADDR weightGrad, GM_ADDR workspace,
                                                  const NsaCompressGradTilingData* tilingData, TPipe* tPipe)
{
    pipe_ = tPipe;
    InitTilingData(tilingData);
    InitGlobalTensor(outputGrad, input, weight, actSeqLen, inputGrad, weightGrad);
    InitWorkspace(workspace);
    InitBuffers();
    calculateProcessInfo();
    SetWsToZero();
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::InitTilingData(const NsaCompressGradTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    blockSize_ = tilingData->blockSize;
    blockStride_ = tilingData->blockStride;
    headDim_ = tilingData->dimOfHead;
    headNum_ = tilingData->numOfHead;
    totalBlockNum_ = tilingData->numOfBlock;
    seqLenType_ = tilingData->seqLenType;
    headToProcess_ = tilingData->headToProcess;
    headRemainder_ = tilingData->headRemainder;
    ubMaxSize_ = tilingData->ubMaxSize;
    coreNum_ = GetBlockNum();
    coreId_ = GetBlockIdx();

    ubCalcSize_ = ubMaxSize_ - UB_PRESERVE * ONE_KILO;
    maxHeadNumInUb_ = ubCalcSize_ / headDim_ / static_cast<uint32_t>(sizeof(float)) / 3 / BLOCK_16 * BLOCK_16;
    // 预留，当前都按1行来处理
    nRowsOnce_ = maxHeadNumInUb_ >= headNum_ ? maxHeadNumInUb_ / headNum_ : 1;
    if (nRowsOnce_ > blockSize_) {
        nRowsOnce_ = blockSize_;
    }
    nHeadOnce_ = maxHeadNumInUb_ >= headNum_ ? headNum_ : (headNum_ / CeilDiv(headNum_, maxHeadNumInUb_));
    // 有overlap时计算inputGrad需要累加，用fp32的workspace累加完再拷贝到inputGradGm
    isOverLap_ = blockSize_ > blockStride_;

    // 最后搬运inputGrad的分核处理信息
    headOffsetCurCore_ = coreId_ * headToProcess_;
    if (coreId_ < headRemainder_) {
        headToProcess_ += 1;
        headOffsetCurCore_ += coreId_;
    } else {
        headOffsetCurCore_ += headRemainder_;
    }
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::InitGlobalTensor(GM_ADDR outputGrad, GM_ADDR input, GM_ADDR weight,
                                                              GM_ADDR actSeqLen, GM_ADDR inputGrad, GM_ADDR weightGrad)
{
    outputGradGm_.SetGlobalBuffer((__gm__ T*)outputGrad);
    inputGm_.SetGlobalBuffer((__gm__ T*)input);
    weightGm_.SetGlobalBuffer((__gm__ T*)weight);
    actSeqLenListGm_.SetGlobalBuffer((__gm__ int64_t*)actSeqLen);
    inputGradGm_.SetGlobalBuffer((__gm__ T*)inputGrad);
    weightGradGm_.SetGlobalBuffer((__gm__ T*)weightGrad);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::InitWorkspace(GM_ADDR workspace)
{
    wtGradAtomicWs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));

    if (isOverLap_) {
        // overLap场景需要使用workspace累加
        uint64_t inputGradOffset = static_cast<uint64_t>(blockSize_) * headNum_ * coreNum_ * sizeof(float);
        inputGradAtomicWs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + inputGradOffset));
    }
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::InitBuffers()
{
    uint32_t maxInputNum = maxHeadNumInUb_ * headDim_;

    pipe_->InitBuffer(ubTBuf, ubMaxSize_);
    cmpGrad = ubTBuf.GetWithOffset<float>(maxInputNum, 0);
    inputFp32Ub_ = ubTBuf.GetWithOffset<float>(maxInputNum, maxInputNum * sizeof(float));
    weightBrdc = ubTBuf.GetWithOffset<float>(maxInputNum, maxInputNum * sizeof(float) * 2);
    tmpBuffer = ubTBuf.GetWithOffset<uint8_t>((ONE_KILO * 16) / sizeof(uint8_t), ubCalcSize_ + 8 * 1024);
    tmpx = ubTBuf.GetWithOffset<float>(ONE_KILO * 8 / sizeof(float), ubCalcSize_);

    // cmpGradHalf地址，在cmpGradFp32的后一半
    cmpGradHf = ubTBuf.GetWithOffset<T>(maxInputNum, maxInputNum * sizeof(half));
    // inputHalf地址，在inputFp32后一半
    inputHf = ubTBuf.GetWithOffset<T>(maxInputNum, maxInputNum * sizeof(float) + maxInputNum * sizeof(half));

    // weightHalf地址，在weightFp32的后一半，即最后
    weightHf = ubTBuf.GetWithOffset<T>(maxHeadNumInUb_, ubCalcSize_ - maxHeadNumInUb_ * sizeof(half));
    // weightFp32地址，在weightBrdc最后
    weightFp32 = ubTBuf.GetWithOffset<float>(maxHeadNumInUb_, ubCalcSize_ - maxHeadNumInUb_ * sizeof(float));

    // 计算weightGrad，原地更新，起始位置与input一样
    wtGradHf = ubTBuf.GetWithOffset<T>(maxHeadNumInUb_, maxInputNum * sizeof(float));
    // 计算inputGrad，原地更新，位置与weightBrdc一样
    inputGradHf = ubTBuf.GetWithOffset<T>(maxInputNum, maxInputNum * sizeof(float) * 2);

    // 搬运inputGrad的UB空间分配
    transInputGradPing_ = ubTBuf.GetWithOffset<float>(maxInputNum, maxInputNum * sizeof(float));
    transInputGradPong_ = ubTBuf.GetWithOffset<float>(maxInputNum, maxInputNum * sizeof(float) * 2);
    halfTransInputGradPing_ = ubTBuf.GetWithOffset<T>(maxInputNum, maxInputNum * sizeof(float));
    halfTransInputGradPong_ = ubTBuf.GetWithOffset<T>(maxInputNum, maxInputNum * sizeof(float) * 2);

    if (deterministicFlag_) {
        transDeterministicWtGrad_ = ubTBuf.GetWithOffset<float>(headNum_ * coreNum_ * (blockSize_ / coreNum_ + 1), 0);
        halfTransDeterministicWtGrad_ = ubTBuf.GetWithOffset<T>(headNum_ * coreNum_ * (blockSize_ / coreNum_ + 1), 0);
    } else {
        transWtGrad_ = ubTBuf.GetWithOffset<float>(headNum_ * blockSize_, 0);
        halfTransWtGrad_ = ubTBuf.GetWithOffset<T>(headNum_ * blockSize_, 0);
    }
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::calculateProcessInfo()
{
    auto nBlcBase = totalBlockNum_ / coreNum_;
    auto nRemainder = totalBlockNum_ % coreNum_;
    // 当前core从第几个block开始计算
    auto blkPlus = coreId_ < nRemainder ? coreId_ : nRemainder;
    startBlcIdx_g = nBlcBase * coreId_ + blkPlus;
    // 当前core需要处理几个block
    blkPlus = coreId_ < nRemainder ? 1 : 0;
    nBlockCurCore_ = nBlcBase + blkPlus;

    auto cumBlcs = 0;
    auto cumSeqLen = 0;
    for (auto bidx = 0; bidx < batchSize_; bidx++) {
        // 当前batch的inputKV的seqLen
        uint32_t seqLenCurBch = GetSeqLenTarBch(bidx);
        // 当前batch压缩后的blkNum
        uint32_t nBlcCurBch = CompressBlkNum(seqLenCurBch, blockSize_, blockStride_);
        if (startBlcIdx_g < cumBlcs + nBlcCurBch) {
            // 当前核从哪个batch开始处理
            startBatchIdx_ = bidx;
            // 这个batch从第几个block开始处理
            blcIdxOfStartBatch_ = cumBlcs;
            batchOffsetCurCore_ = cumSeqLen;
            break;
        }
        cumBlcs += nBlcCurBch;
        cumSeqLen += seqLenCurBch;
    }
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::SetWsToZero()
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (coreId_ >= coreNum_) {
        return;
    }

    // 清零wtGradAtomicWs_
    uint64_t wtGradNum = static_cast<uint64_t>(blockSize_) * headNum_;
    Duplicate<float>(cmpGrad, ZERO_FLOAT, static_cast<uint32_t>(wtGradNum));

    SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(wtGradNum * sizeof(float)), 0, 0, 0};
    DataCopyPad(wtGradAtomicWs_[coreId_ * wtGradNum], cmpGrad, dataCopyParams);

    // 清零inputGradAtomicWs_
    if (isOverLap_) {
        Duplicate<float>(transInputGradPing_, ZERO_FLOAT, maxHeadNumInUb_ * headDim_);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);

        uint32_t copyTimes = CeilDiv(headToProcess_, maxHeadNumInUb_);

        for (uint32_t copyIdx = 0; copyIdx < copyTimes; copyIdx++) {
            uint32_t nHeadCur =
                (copyIdx == copyTimes - 1) ? (headToProcess_ - copyIdx * maxHeadNumInUb_) : maxHeadNumInUb_;
            uint32_t processNumCur = nHeadCur * headDim_;
            uint32_t inputGradOffset = (headOffsetCurCore_ + copyIdx * maxHeadNumInUb_) * headDim_;

            DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(processNumCur * sizeof(float)), 0, 0, 0};
            DataCopyPad(inputGradAtomicWs_[inputGradOffset], transInputGradPing_, dataCopyParams);
        }
    } else {
        AscendC::InitOutput<T>(inputGradGm_[headOffsetCurCore_ * headDim_], headToProcess_ * headDim_, 0);
    }

#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::Process()
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (coreId_ >= coreNum_) {
        return;
    }

    PresetFlag();

    uint32_t curBIdx = startBatchIdx_;
    uint64_t offsetPerRow = static_cast<uint64_t>(headDim_) * headNum_;
    uint64_t offsetPerBlc = offsetPerRow * blockStride_;
    // inptKv当前处理的第一个batch的偏移
    uint64_t batchOffset = static_cast<uint64_t>(batchOffsetCurCore_) * offsetPerRow;

    uint32_t blcStartIdxCurBch = blcIdxOfStartBatch_;
    uint32_t seqLenCurBch = GetSeqLenTarBch(curBIdx);
    // 当前batch要处理的block数量
    uint32_t nBlcCurBch = CompressBlkNum(seqLenCurBch, blockSize_, blockStride_);

    // iterate blocks
    for (auto blcId = startBlcIdx_g; blcId < startBlcIdx_g + nBlockCurCore_; blcId++) {
        // 判断是否进入下一个batch
        if (blcId >= blcStartIdxCurBch + nBlcCurBch) {
            // 更新循环参数
            curBIdx += 1;
            blcStartIdxCurBch += nBlcCurBch;
            batchOffset += seqLenCurBch * offsetPerRow;
            seqLenCurBch = GetSeqLenTarBch(curBIdx);
            nBlcCurBch = CompressBlkNum(seqLenCurBch, blockSize_, blockStride_);
        }
        // inputKV当前处理的block的偏移
        uint64_t blcOffset = batchOffset + static_cast<uint64_t>(blcId - blcStartIdxCurBch) * offsetPerBlc;

        // deal with current block
        for (auto headId = 0; headId < headNum_; headId += nHeadOnce_) {
            // Block内按行处理
            uint32_t headToProcess = (headId + nHeadOnce_ > headNum_) ? (headNum_ - headId) : nHeadOnce_;
            uint64_t cmpOffset = static_cast<uint64_t>(blcId) * offsetPerRow + static_cast<uint64_t>(headId) * headDim_;

            WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
            CopyInCmpGrad(cmpOffset, headToProcess * headDim_);

            for (auto rowId = 0; rowId < blockSize_; rowId += nRowsOnce_) {
                uint64_t inputOffset =
                    blcOffset + static_cast<uint64_t>(rowId) * offsetPerRow + static_cast<uint64_t>(headId) * headDim_;
                uint64_t weightOffset = static_cast<uint64_t>(rowId) * headNum_ + headId;
                uint32_t rowsToProcess = (rowId + nRowsOnce_ > blockSize_) ? (blockSize_ - rowId) : nRowsOnce_;

                // 拷贝inputKV
                CopyInInput(inputOffset, headToProcess * headDim_ * rowsToProcess);
                // 拷贝weight
                CopyInWt(weightOffset, headToProcess * rowsToProcess);
                // 计算WtGrad
                ComputeWtGrad(headToProcess * headDim_ * rowsToProcess);
                // 拷贝wtGrad计算结果
                CopyOutWtGrad(weightOffset, headToProcess * rowsToProcess);
                // 计算kvGrad
                ComputeInputGrad(headToProcess * headDim_ * rowsToProcess);
                // 拷贝kvGrad计算结果
                CopyOutInputGrad(inputOffset, headToProcess * headDim_ * rowsToProcess);
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
        }
    }
    ClearPresetFlag();
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::CopyInCmpGrad(uint32_t offsetCmpGrad, uint32_t size)
{
    DataCopyExtParams cdataCopyParams{1, static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(cmpGradHf, outputGradGm_[offsetCmpGrad], cdataCopyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    Cast(cmpGrad, cmpGradHf, RoundMode::CAST_NONE, size);

    if (nRowsOnce_ > 1) {
        PipeBarrier<PIPE_V>();
        uint32_t srcShape[2] = {1, size};
        uint32_t dstShape[2] = {nRowsOnce_, size};
        AscendC::BroadCast<float, 2, 0>(cmpGrad, cmpGrad, dstShape, srcShape, tmpBuffer);
    }
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::CopyInInput(uint32_t offset, uint32_t size)
{
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputHf, inputGm_[offset], dataCopyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::ComputeWtGrad(uint32_t size)
{
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    Cast(inputFp32Ub_, inputHf, RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();
    Mul(inputFp32Ub_, inputFp32Ub_, cmpGrad, size);
    PipeBarrier<PIPE_V>();
    int32_t m = headDim_ / 64;
    int32_t n = headDim_ % 64;
    for (int32_t a = 0; a < m; a++) {
        BlockReduceSum(tmpx[8 * a], inputFp32Ub_[64 * a], size / headDim_, 64, m + 1, 1, headDim_ / 8);
        PipeBarrier<PIPE_V>();
    }
    if (n != 0) {
        BlockReduceSum(tmpx[8 * m], inputFp32Ub_[64 * m], size / headDim_, n, m + 1, 1, headDim_ / 8);
        PipeBarrier<PIPE_V>();
    }
    WholeReduceSum(tmpx, tmpx, headDim_ / 8, size / headDim_, 1, 1, m + 1);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::ComputeInputGrad(uint32_t size)
{
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);

    // cast to fp32
    Cast(weightFp32, weightHf, RoundMode::CAST_NONE, size / headDim_);
    PipeBarrier<PIPE_V>();
    // broadcast to headDim
    uint32_t srcShape[2] = {size / headDim_, 1};
    uint32_t dstShape[2] = {size / headDim_, headDim_};
    AscendC::BroadCast<float, 2, 1>(weightBrdc, weightFp32, dstShape, srcShape, tmpBuffer);
    PipeBarrier<PIPE_V>();
    // vmul
    Mul(weightBrdc, weightBrdc, cmpGrad, size);
    PipeBarrier<PIPE_V>();
    // cast by condition
    CastByCondition(inputGradHf, weightBrdc, size, !isOverLap_);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::CopyOutWtGrad(uint32_t offsetWt, uint32_t size)
{
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);

    if (deterministicFlag_) {
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(size * sizeof(float)), 0, 0, 0};
        AscendC::SetAtomicAdd<float>();
        DataCopyPad(wtGradAtomicWs_[offsetWt + blockSize_ * headNum_ * coreId_], tmpx, dataCopyParams);
        AscendC::SetAtomicNone();

    } else {
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(size * sizeof(float)), 0, 0, 0};
        AscendC::SetAtomicAdd<float>();
        DataCopyPad(wtGradAtomicWs_[offsetWt], tmpx, dataCopyParams);
        AscendC::SetAtomicNone();
    }
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::CopyInWt(uint32_t offset, uint32_t size)
{
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(weightHf, weightGm_[offset], dataCopyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::CopyOutInputGrad(uint32_t offsetInput, uint32_t size)
{
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

    if (isOverLap_) {
        AscendC::SetAtomicAdd<float>();
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(size * sizeof(float)), 0, 0, 0};
        DataCopyPad(inputGradAtomicWs_[offsetInput], weightBrdc, dataCopyParams);
        AscendC::SetAtomicNone();
    } else {
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(size * sizeof(T)), 0, 0, 0};
        DataCopyPad(inputGradGm_[offsetInput], inputGradHf, dataCopyParams);
    }

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::MoveResult()
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif

    if (deterministicFlag_) {
        DeterministicComputeSumWtGrad();
    } else if (coreId_ == 0) {
        // wtGrad数量很少，一个核搬运即可
        MoveWtGrad();
    }

    if (isOverLap_) {
        MoveInputGrad();
    }
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::DeterministicComputeSumWtGrad()
{
    if (coreId_ >= coreNum_) {
        return;
    }

    uint64_t totalWeightLength = static_cast<uint64_t>(blockSize_) * headNum_;
    uint64_t minOffset = 16;
    uint64_t nLengthBase = (static_cast<uint64_t>(blockSize_) * headNum_ / minOffset) / coreNum_;
    uint64_t nLengthReminder = (static_cast<uint64_t>(blockSize_) * headNum_ / minOffset) % coreNum_;
    // 当前core从第几个数据开始算
    auto startLengthIdx = (nLengthBase * coreId_ + (coreId_ < nLengthReminder ? coreId_ : nLengthReminder)) * minOffset;
    // 当前core需要处理几个数据
    auto nLengthCurCore = (nLengthBase + (coreId_ < nLengthReminder ? 1 : 0)) * minOffset;

    auto linesNum = coreNum_;

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(coreNum_),
                                     static_cast<uint32_t>(nLengthCurCore * sizeof(float)),
                                     static_cast<uint32_t>((totalWeightLength - nLengthCurCore) * sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(transDeterministicWtGrad_, wtGradAtomicWs_[startLengthIdx], dataCopyParams, padParams);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);

    // 记录额外行
    auto extraLine = coreNum_ + 1;
    while (linesNum > 1 || (extraLine > 1 && extraLine < (coreNum_ + 1))) {
        PipeBarrier<PIPE_V>();
        if (linesNum % 2 == 0) {
            // 偶数，前一半加后一半
            Add(transDeterministicWtGrad_, transDeterministicWtGrad_,
                transDeterministicWtGrad_[(linesNum / 2) * nLengthCurCore], linesNum / 2 * nLengthCurCore);
        } else {
            // 奇数，总行数减一后前一半加后一半
            // 判断行数减一后是否大于1，如果是则相加
            if (linesNum > 1) {
                Add(transDeterministicWtGrad_, transDeterministicWtGrad_,
                    transDeterministicWtGrad_[(linesNum / 2) * nLengthCurCore], linesNum / 2 * nLengthCurCore);
            }
            // 如果之前存在额外行
            if (extraLine <= coreNum_) {
                PipeBarrier<PIPE_V>();
                Add(transDeterministicWtGrad_[(linesNum / 2) * nLengthCurCore],
                    transDeterministicWtGrad_[(linesNum - 1) * nLengthCurCore],
                    transDeterministicWtGrad_[(extraLine - 1) * nLengthCurCore], nLengthCurCore);
                extraLine = linesNum / 2 + 1;
            } else {
                // 否则记录额外行
                extraLine = linesNum;
            }
        }
        linesNum = linesNum / 2;
    }

    PipeBarrier<PIPE_V>();

    // 每个核有自己的计算结果，原地cast，每份数据大小为nLengthCurCore
    CastByCondition(halfTransDeterministicWtGrad_, transDeterministicWtGrad_, nLengthCurCore, true);

    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
    // copyout ,核把自己的处理好的数据拷出到对应位置，每份数据大小为nLengthCurCore
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(nLengthCurCore * sizeof(T)), 0, 0, 0};
    DataCopyPad(weightGradGm_[startLengthIdx], halfTransDeterministicWtGrad_, copyOutParams);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::MoveWtGrad()
{
    uint64_t wtNum = static_cast<uint64_t>(blockSize_) * headNum_;
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(wtNum * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(transWtGrad_, wtGradAtomicWs_, dataCopyParams, padParams);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);

    CastByCondition(halfTransWtGrad_, transWtGrad_, static_cast<uint32_t>(wtNum), true);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);

    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(wtNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(weightGradGm_, halfTransWtGrad_, copyOutParams);
}

template <typename T>
__aicore__ inline void NsaCompressGradND<T>::MoveInputGrad()
{
    if (coreId_ >= coreNum_) {
        return;
    }

    uint32_t copyTimes = CeilDiv(headToProcess_, maxHeadNumInUb_);
    uint32_t pingPongFlag = 0;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    for (uint32_t copyIdx = 0; copyIdx < copyTimes; copyIdx++) {
        uint32_t nHeadCur = (copyIdx == copyTimes - 1) ? (headToProcess_ - copyIdx * maxHeadNumInUb_) : maxHeadNumInUb_;
        uint32_t processNumCur = nHeadCur * headDim_;
        uint32_t inputGradOffset = (headOffsetCurCore_ + copyIdx * maxHeadNumInUb_) * headDim_;
        LocalTensor<float> transUbCur = pingPongFlag ? transInputGradPong_ : transInputGradPing_;
        LocalTensor<T> halfTransUbCur = pingPongFlag ? halfTransInputGradPong_ : halfTransInputGradPing_;

        WaitFlag<HardEvent::MTE3_MTE2>(pingPongFlag);
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(processNumCur * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(transUbCur, inputGradAtomicWs_[inputGradOffset], dataCopyParams, padParams);

        SetFlag<HardEvent::MTE2_V>(pingPongFlag);
        WaitFlag<HardEvent::MTE2_V>(pingPongFlag);

        CastByCondition(halfTransUbCur, transUbCur, processNumCur, isOverLap_);

        SetFlag<HardEvent::V_MTE3>(pingPongFlag);
        WaitFlag<HardEvent::V_MTE3>(pingPongFlag);

        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(processNumCur * sizeof(T)), 0, 0, 0};
        DataCopyPad(inputGradGm_[inputGradOffset], halfTransUbCur, copyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(pingPongFlag);

        pingPongFlag = !pingPongFlag;
    }

    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

} // namespace NsaCompressGrad
#endif
