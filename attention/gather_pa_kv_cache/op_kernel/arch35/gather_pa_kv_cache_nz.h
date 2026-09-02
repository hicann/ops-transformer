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
 * \file gather_pa_kv_cache_nd.h
 * \brief
 */

#ifndef GATHER_PA_KV_CACHE_NZ_H_
#define GATHER_PA_KV_CACHE_NZ_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace GatherPaKvCacheV35 {
using namespace AscendC;

/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T, typename T_INDEX, bool isSeqLensCumsum, bool hasSeqOffset>
class GatherPaKvCacheNz {
    static constexpr int32_t BLOCK_SIZE = 32;
    static constexpr uint32_t DOUBLE_BUFFER = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;
    static constexpr uint8_t CONST_ONE = 1;
    static constexpr uint8_t DIM_TWO = 2;
    static constexpr uint8_t DIM_THREE = 3;
    static constexpr int32_t MAX_BLOCK_CNT = 4095;
    static constexpr uint32_t VL_NUMS = GetVRegSize() / sizeof(T);

public:
    __aicore__ inline GatherPaKvCacheNz(TPipe *pipe, const GatherPaKvCacheTilingDataV35 *__restrict tiling)
        : pipe_(pipe),
          tl_(tiling){};

    __aicore__ inline void Init(GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables, GM_ADDR seq_lens,
                                GM_ADDR key_in, GM_ADDR value_in, GM_ADDR seq_offset, GM_ADDR key_out,
                                GM_ADDR value_out)
    {
        InitParams();

        keyCacheGm_.SetGlobalBuffer((__gm__ T *)(key_cache));
        valueCacheGm_.SetGlobalBuffer((__gm__ T *)(value_cache));
        blockTablesGm_.SetGlobalBuffer((__gm__ T_INDEX *)block_tables);
        seqLensGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_lens));
        seqOffsetGm_.SetGlobalBuffer((__gm__ T_INDEX *)(seq_offset));

        outKeyGm_.SetGlobalBuffer((__gm__ T *)(key_out));
        outValueGm_.SetGlobalBuffer((__gm__ T *)(value_out));

        pipe_->InitBuffer(cacheQueue_, DOUBLE_BUFFER, (maxUbHiddenSize_) * sizeof(T));
        pipe_->InitBuffer(seqLensQueue_, DOUBLE_BUFFER, (seqLenAccumSize_) * sizeof(T_INDEX));

        pipe_->InitBuffer(sumBuffer, sizeof(T_INDEX));
    }

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_DEPTH> cacheQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> keyCacheQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> valueCacheQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> blockTablesQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> seqLensQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> seqOffsetQueue_;

    TBuf<TPosition::VECCALC> sumBuffer;

    const GatherPaKvCacheTilingDataV35 *tl_;

    GlobalTensor<T> keyCacheGm_;
    GlobalTensor<T> valueCacheGm_;
    GlobalTensor<T_INDEX> blockTablesGm_;
    GlobalTensor<T_INDEX> seqLensGm_;
    GlobalTensor<T_INDEX> seqOffsetGm_;

    GlobalTensor<T> outKeyGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<T> scrGm;
    GlobalTensor<T> dstGm;

    uint32_t eleNumPerBlk = BLOCK_SIZE / sizeof(T);
    uint32_t blockSize_;
    uint32_t batchPerCore_;
    uint32_t needCoreNum_;
    uint64_t batchCount_;
    int64_t blockTableWidth_;
    uint64_t maxUbHiddenSizeK_;
    uint64_t maxUbHiddenSizeV_;
    uint64_t maxUbHiddenSize_;
    uint64_t hiddenSizeK_;
    uint64_t hiddenSizeV_;
    int64_t numTokens_;
    int64_t numBlocks_;
    uint32_t seqLenAccumSize_;
    bool isSmallShape;

    // 非连续支持
    uint32_t nonContiguousFlag_;
    int64_t kCacheStride0_;
    int64_t vCacheStride0_;
    int64_t keyOutStride0_;
    int64_t valueOutStride0_;

private:
    __aicore__ inline void InitParams()
    {
        blockSize_ = tl_->kvCacheBlockSize;
        batchPerCore_ = tl_->batchPerCore;
        needCoreNum_ = tl_->needCoreNum;
        batchCount_ = tl_->batchCount;
        blockTableWidth_ = tl_->blockTableWidth;
        maxUbHiddenSizeK_ = tl_->maxUbHiddenSizeK * sizeof(DTYPE_KEY);
        maxUbHiddenSizeV_ = tl_->maxUbHiddenSizeV * sizeof(DTYPE_VALUE);
        maxUbHiddenSize_ = tl_->maxUbHiddenSize * sizeof(DTYPE_KEY);
        hiddenSizeK_ = tl_->hiddenSizeK * sizeof(DTYPE_KEY);
        hiddenSizeV_ = tl_->hiddenSizeV * sizeof(DTYPE_VALUE);
        numTokens_ = tl_->numTokens;
        numBlocks_ = tl_->numBlocks;
        seqLenAccumSize_ = tl_->seqLenAccumSize;

        // 非连续支持
        nonContiguousFlag_ = tl_->nonContiguousFlag;
        kCacheStride0_ = tl_->kCacheStride0;
        vCacheStride0_ = tl_->vCacheStride0;
        keyOutStride0_ = tl_->keyOutStride0;
        valueOutStride0_ = tl_->valueOutStride0;

        isSmallShape =
            (maxUbHiddenSizeK_ >= blockSize_ * hiddenSizeK_) && (maxUbHiddenSizeV_ >= blockSize_ * hiddenSizeV_) &&
            // 输出非连续(ref dim0非连续)时不走small-shape的DataCopyFromeCache(它整block连续写, 无法按token散写),
            // 改走RestoreFromCache逐token循环按keyOutStride0散写。bit2:keyOut非连续, bit3:valueOut非连续。
            ((nonContiguousFlag_ & ((1U << 2) | (1U << 3))) == 0);
    }

public:
    __aicore__ inline void Process()
    {
        uint32_t batchStart = GetBlockIdx() * batchPerCore_;
        uint32_t batchEnd = batchStart + batchPerCore_;
        if (GetBlockIdx() == needCoreNum_ - 1) {
            batchEnd = batchCount_;
        }

        int64_t coreOffset_;
        if constexpr (isSeqLensCumsum) {
            coreOffset_ = seqLensGm_.GetValue(batchStart);
        } else {
            coreOffset_ = CalcKvCoreOffset(batchStart); // 计算累加和
        }

        // 偏移量用64位: 大hiddenSize且token多时, 字节偏移可超过uint32上限(4GB)导致回绕写错地址。
        uint64_t keyCacheOffset = 0;
        uint64_t valueCacheOffset = 0;
        for (uint32_t i = batchStart; i < batchEnd; i++) {
            T_INDEX seqLen;
            uint32_t batchOffset, accumSeqLen;
            if constexpr (isSeqLensCumsum) {
                // 当前batch对应的seqLen
                seqLen = seqLensGm_.GetValue(i + 1) - seqLensGm_.GetValue(i);
                batchOffset = seqLensGm_.GetValue(i);
                accumSeqLen = seqLensGm_.GetValue(i + 1);
            } else {
                seqLen = seqLensGm_.GetValue(i);
                if (i == batchStart) {
                    batchOffset = coreOffset_;
                    accumSeqLen = coreOffset_ + seqLensGm_.GetValue(i);
                } else {
                    batchOffset += seqLensGm_.GetValue(i - 1);
                    accumSeqLen += seqLensGm_.GetValue(i);
                }
            }

            if (batchOffset >= numTokens_) {
                break;
            }

            if (numTokens_ > batchOffset && numTokens_ <= accumSeqLen) {
                seqLen = numTokens_ - batchOffset;
            }

            T_INDEX seqOffset = 0;
            if constexpr (hasSeqOffset) {
                seqOffset = seqOffsetGm_.GetValue(i) / blockSize_;
            }
            T_INDEX blockStart = 0;

            uint32_t blockCount = CeilDivision(seqLen, blockSize_);

            // 搬运key
            uint64_t keyOffset = (nonContiguousFlag_ & (1 << 2)) ? (batchOffset * keyOutStride0_ * sizeof(DTYPE_KEY)) :
                                                                   (batchOffset * hiddenSizeK_);
            for (int j = 0; j < blockCount; j++) {
                uint32_t curLen = blockSize_; // 4
                if (j == blockCount - 1) {
                    curLen = seqLen - (blockCount - 1) * blockSize_;
                }

                uint64_t blockTableOffset = j + seqOffset;

                bool isFulledWithZero;
                T_INDEX blockId;
                if (blockTableOffset >= blockTableWidth_ || blockTableOffset < 0) {
                    isFulledWithZero = true;
                    blockId = 0;
                } else {
                    isFulledWithZero = false;
                    blockId = blockTablesGm_.GetValue(blockTableWidth_ * i + seqOffset + j);
                    // blockId数值越界时填零, 与golden对齐(非连续view场景numBlocks_为view dim0)。
                    if (blockId >= numBlocks_ || blockId < 0) {
                        isFulledWithZero = true;
                        blockId = 0;
                    }
                }
                keyCacheOffset = (nonContiguousFlag_ & 1) ? (blockId * kCacheStride0_ * sizeof(DTYPE_KEY)) :
                                                            (blockId * blockSize_ * hiddenSizeK_);

                if (isSmallShape) {
                    DataCopyFromeCache(curLen, keyCacheOffset, keyOffset, isFulledWithZero, true);
                } else {
                    // ref dim0非连续(bit2)时token间按keyOutStride0散写; 连续时token紧密排列(=hiddenSizeK_)
                    RestoreFromCache(
                        outKeyGm_[keyOffset], keyCacheGm_[keyCacheOffset], hiddenSizeK_, curLen, isFulledWithZero,
                        (nonContiguousFlag_ & (1 << 2)) ? (keyOutStride0_ * sizeof(DTYPE_KEY)) : hiddenSizeK_);
                }

                keyOffset += (nonContiguousFlag_ & (1 << 2)) ? (curLen * keyOutStride0_ * sizeof(DTYPE_KEY)) :
                                                               (curLen * hiddenSizeK_);
            }

            // 搬运value
            uint64_t valueOffset = (nonContiguousFlag_ & (1 << 3)) ?
                                       (batchOffset * valueOutStride0_ * sizeof(DTYPE_VALUE)) :
                                       (batchOffset * hiddenSizeV_);

            for (int j = 0; j < blockCount; j++) {
                uint32_t curLen = blockSize_;
                if (j == blockCount - 1) {
                    curLen = seqLen - (blockCount - 1) * blockSize_;
                }

                bool isFulledWithZero;
                uint64_t blockTableOffset = j + seqOffset;
                T_INDEX blockId;
                if (blockTableOffset >= blockTableWidth_ || blockTableOffset < 0) {
                    isFulledWithZero = true;
                    blockId = 0;
                } else {
                    isFulledWithZero = false;
                    blockId = blockTablesGm_.GetValue(blockTableWidth_ * i + seqOffset + j);
                    // blockId数值越界时填零, 与golden对齐(非连续view场景numBlocks_为view dim0)。
                    if (blockId >= numBlocks_ || blockId < 0) {
                        isFulledWithZero = true;
                        blockId = 0;
                    }
                }
                valueCacheOffset = (nonContiguousFlag_ & 2) ? (blockId * vCacheStride0_ * sizeof(DTYPE_VALUE)) :
                                                              (blockId * blockSize_ * hiddenSizeV_);

                if (isSmallShape) {
                    DataCopyFromeCache(curLen, valueCacheOffset, valueOffset, isFulledWithZero, false);
                } else {
                    // ref dim0非连续(bit3)时token间按valueOutStride0散写; 连续时token紧密排列(=hiddenSizeV_)
                    RestoreFromCache(
                        outValueGm_[valueOffset], valueCacheGm_[valueCacheOffset], hiddenSizeV_, curLen,
                        isFulledWithZero,
                        (nonContiguousFlag_ & (1 << 3)) ? (valueOutStride0_ * sizeof(DTYPE_VALUE)) : hiddenSizeV_);
                }

                valueOffset += (nonContiguousFlag_ & (1 << 3)) ? (curLen * valueOutStride0_ * sizeof(DTYPE_VALUE)) :
                                                                 (curLen * hiddenSizeV_);
            }
        }
    }

private:
    __aicore__ inline T_INDEX CalcKvCoreOffset(uint32_t batchStart)
    {
        LocalTensor<T_INDEX> sumLocal = sumBuffer.Get<T_INDEX>();
        uint32_t reduceLoops = CeilDivision(batchStart, seqLenAccumSize_);
        DataCopyPadExtParams<T_INDEX> padParams = {false, 0, 0, 0};
        DataCopyExtParams seqLensCopyParams;
        seqLensCopyParams.blockCount = 1;
        seqLensCopyParams.srcStride = 0;
        seqLensCopyParams.dstStride = 0;
        uint32_t seqOffset, seqLength;
        uint32_t curLen = batchStart;

        T_INDEX coreOffset = 0;
        for (uint32_t i = 0; i < reduceLoops; i++) {
            seqOffset = i * seqLenAccumSize_;
            seqLength = seqLenAccumSize_;
            if (i == reduceLoops - 1) {
                seqLength = batchStart - (reduceLoops - 1) * seqLenAccumSize_;
            }
            seqLensCopyParams.blockLen = static_cast<uint32_t>(seqLength * sizeof(T_INDEX));
            LocalTensor<T_INDEX> accumSeqLenLocal = seqLensQueue_.AllocTensor<T_INDEX>();
            DataCopyPad(accumSeqLenLocal, seqLensGm_[seqOffset], seqLensCopyParams, padParams);
            seqLensQueue_.EnQue<T_INDEX>(accumSeqLenLocal);
            accumSeqLenLocal = seqLensQueue_.DeQue<T_INDEX>();

            uint32_t srcShape[2] = {uint32_t(1), uint32_t(seqLength)};
            AscendC::ReduceSum<T_INDEX, Pattern::Reduce::AR, true>(sumLocal, accumSeqLenLocal, srcShape, false);
            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
            seqLensQueue_.FreeTensor<T_INDEX>(accumSeqLenLocal);
            coreOffset += sumLocal.GetValue(0);
        }
        return coreOffset;
    }

    __aicore__ inline void DataCopyFromeCache(uint32_t curLen, uint64_t scrOffset, uint64_t dstOffset,
                                              bool isFulledWithZero, bool isKey)
    {
        // 小shape情况
        uint64_t hiddenSize;
        if (isKey) {
            hiddenSize = hiddenSizeK_;
            scrGm = keyCacheGm_;
            dstGm = outKeyGm_;
        } else {
            hiddenSize = hiddenSizeV_;
            scrGm = valueCacheGm_;
            dstGm = outValueGm_;
        }
        static constexpr NdDmaConfig config = {false};
        AscendC::NdDmaLoopInfo<DIM_THREE> copyLoopInfo;
        copyLoopInfo.loopSrcStride[0] = blockSize_ * eleNumPerBlk;
        copyLoopInfo.loopSrcStride[1] = eleNumPerBlk;
        copyLoopInfo.loopSrcStride[DIM_TWO] = CONST_ONE;
        copyLoopInfo.loopDstStride[0] = eleNumPerBlk;
        copyLoopInfo.loopDstStride[1] = hiddenSize;
        copyLoopInfo.loopDstStride[DIM_TWO] = CONST_ONE;
        copyLoopInfo.loopSize[0] = hiddenSize / eleNumPerBlk;
        copyLoopInfo.loopSize[1] = blockSize_;
        copyLoopInfo.loopSize[DIM_TWO] = eleNumPerBlk;
        AscendC::NdDmaParams<T, DIM_THREE> params = {copyLoopInfo, 0};

        LocalTensor<T> cacheLocal = cacheQueue_.AllocTensor<T>();
        if (isFulledWithZero) {
            AscendC::TEventID eventIdMTE3ToVec = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
            Duplicate<T>(cacheLocal, 0, curLen * hiddenSize);
            AscendC::TEventID eventIdVecToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
        } else {
            DataCopy<T, DIM_THREE, config>(cacheLocal, scrGm[scrOffset], params);
            cacheQueue_.EnQue<T>(cacheLocal);
            cacheLocal = cacheQueue_.DeQue<T>();
        }

        DataCopyExtParams x1DataCopyParams;
        x1DataCopyParams.blockCount = 1;
        x1DataCopyParams.blockLen = curLen * hiddenSize * sizeof(T);
        x1DataCopyParams.srcStride = 0;
        x1DataCopyParams.dstStride = 0;
        DataCopyPad<T, PaddingMode::Normal>(dstGm[dstOffset], cacheLocal, x1DataCopyParams);
        event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        cacheQueue_.FreeTensor(cacheLocal);
    }

    // dstTokenStrideBytes: 输出端相邻token间步长(字节)。输出连续时=hiddenSize; ref dim0非连续时=keyOutStride0*元素宽,
    // 使block内每个token按stride散写到原ref内存(否则按hiddenSize稠密写会错位)。
    __aicore__ inline void RestoreFromCache(GlobalTensor<T> dstGm, GlobalTensor<T> scrGm, uint64_t hiddenSize,
                                            uint32_t curblockSize, bool isFulledWithZero, uint64_t dstTokenStrideBytes)
    {
        uint32_t fracBlockSize = hiddenSize / eleNumPerBlk;
        // 每轮最多搬运的32B块数: 既受UB buffer容量(maxUbHiddenSize_/BLOCK_SIZE)约束, 也受硬件
        // blockCount字段上限(MAX_BLOCK_CNT)约束。大hiddenSize时若仅按MAX_BLOCK_CNT切, 单次
        // DataCopy(inBlkCnt*32B)会超过UB buffer导致越界, 故取二者较小值。
        uint32_t maxBlkPerLoop = static_cast<uint32_t>(maxUbHiddenSize_ / BLOCK_SIZE);
        if (maxBlkPerLoop > MAX_BLOCK_CNT) {
            maxBlkPerLoop = MAX_BLOCK_CNT;
        }
        uint16_t inBlkCnt = 0;
        uint16_t inSrcStride = blockSize_ - 1;
        DataCopyParams seqLensCopyParams = {inBlkCnt, 1, inSrcStride, 0};
        DataCopyParams lensCopyParams = {1, inBlkCnt, 0, 0};

        for (int i = 0; i < curblockSize; i++) {
            uint64_t scroffset = i * eleNumPerBlk;
            uint64_t dstoffset = i * dstTokenStrideBytes;

            uint32_t ubLoops = CeilDivision(fracBlockSize, maxBlkPerLoop);
            for (int j = 0; j < ubLoops; j++) {
                inBlkCnt = maxBlkPerLoop;
                if (j == (ubLoops - 1)) {
                    inBlkCnt = fracBlockSize - maxBlkPerLoop * j;
                }

                seqLensCopyParams.blockCount = inBlkCnt;
                LocalTensor<T> cacheLocal = cacheQueue_.AllocTensor<T>();
                if (isFulledWithZero) {
                    AscendC::TEventID eventIdMTE3ToVec = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
                    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
                    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToVec);
                    Duplicate<T>(cacheLocal, 0, inBlkCnt * eleNumPerBlk);
                    AscendC::TEventID eventIdVecToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                    SetFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMTE3);
                } else {
                    DataCopy(cacheLocal, scrGm[scroffset], seqLensCopyParams);
                    cacheQueue_.EnQue<T>(cacheLocal);
                    cacheLocal = cacheQueue_.DeQue<T>();
                }

                lensCopyParams.blockLen = inBlkCnt;
                DataCopy(dstGm[dstoffset], cacheLocal, lensCopyParams);
                event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
                cacheQueue_.FreeTensor(cacheLocal);

                scroffset += inBlkCnt * blockSize_ * eleNumPerBlk;
                dstoffset += inBlkCnt * eleNumPerBlk;
            }
        }
    }
};

} // namespace GatherPaKvCacheV35
#endif
