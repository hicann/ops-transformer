/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_VECTOR_H
#define BLOCK_ATTN_RES_PREPARE_VECTOR_H

#include "block_attn_res_prepare_tiling_data.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "vf/block_attn_res_prepare_vf.h"

namespace BlockAttnResPrepare {

using namespace AscendC;

class BlockAttnResPrepareVector {
public:
    __aicore__ inline explicit BlockAttnResPrepareVector(
        const optiling::BlockAttnResPrepareTilingData *__restrict tilingData)
        : tilingData_(tilingData)
    {
        for (uint32_t eventIndex = 0; eventIndex < MAX_PIPE_ID; ++eventIndex) {
            set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(eventIndex));
        }
        for (uint32_t eventIndex = 0; eventIndex < MTE3_MAX_PIPE_ID; ++eventIndex) {
            set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(eventIndex));
        }
    }

    __aicore__ inline ~BlockAttnResPrepareVector()
    {
        for (uint32_t eventIndex = 0; eventIndex < MAX_PIPE_ID; ++eventIndex) {
            wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(eventIndex));
        }
        for (uint32_t eventIndex = 0; eventIndex < MTE3_MAX_PIPE_ID; ++eventIndex) {
            wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(eventIndex));
        }
    }

    __aicore__ inline void Init(GM_ADDR blockRes, GM_ADDR validBlocks, GM_ADDR pseudoQuery, GM_ADDR numerator,
                                GM_ADDR logitMax, GM_ADDR expSum)
    {
        initialized_ = false;
        validBlocksAddr_ = reinterpret_cast<__gm__ uint64_t *>(validBlocks);
        validN_ = ReadValidBlocks();
        const uint64_t dAlign =
            ((static_cast<uint64_t>(tilingData_->totalD) + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS) * FP32_REG_ELEMS;
        const uint64_t dTileNum =
            (static_cast<uint64_t>(tilingData_->totalD) + tilingData_->baseD - 1U) / tilingData_->baseD;
        dAlign_ = static_cast<uint32_t>(dAlign);
        dTileNum_ = static_cast<uint32_t>(dTileNum);
        // LoadAlign reads a complete FP32 vector register. A 64-element pitch
        // keeps a masked tail load inside the current Q/V/O logical UB row.
        const uint64_t baseDAlign =
            ((static_cast<uint64_t>(tilingData_->baseD) + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS) * FP32_REG_ELEMS;
        baseDAlign_ = static_cast<uint32_t>(baseDAlign);
        qBufferNum_ = tilingData_->qBufferNum;
        vBufferNum_ = tilingData_->vBufferNum;

        blockResAddr_ = reinterpret_cast<__gm__ float *>(blockRes);
        pseudoQueryAddr_ = reinterpret_cast<__gm__ float *>(pseudoQuery);
        numeratorAddr_ = reinterpret_cast<__gm__ float *>(numerator);
        logitMaxAddr_ = reinterpret_cast<__gm__ float *>(logitMax);
        expSumAddr_ = reinterpret_cast<__gm__ float *>(expSum);

        InitUbOffset();
        initialized_ = true;
    }

    __aicore__ inline void Process()
    {
        if (!initialized_) {
            return;
        }
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= tilingData_->usedCoreNum) {
            return;
        }
        const uint64_t workStart = static_cast<uint64_t>(blockIdx) * tilingData_->blockFactor +
                                   ((blockIdx < tilingData_->bigCoreNum) ? blockIdx : tilingData_->bigCoreNum);
        const uint32_t workCount = tilingData_->blockFactor + ((blockIdx < tilingData_->bigCoreNum) ? 1U : 0U);
        uint64_t validNValue = validN_;
        if (validNValue == 0U) {
            // Zero valid blocks produces the identity state of online softmax.
            for (uint32_t workOffset = 0; workOffset < workCount; ++workOffset) {
                ProcessEmptyWork(workStart + workOffset);
            }
            return;
        } else if (validNValue > tilingData_->totalN) {
            validNValue = static_cast<uint64_t>(tilingData_->totalN);
        }
        const uint32_t validN = static_cast<uint32_t>(validNValue);
        // validN comes from valid_blocks at runtime. The host only provides
        // the capacity; choose cache/reload here for the concrete N.
        const bool cacheAllV =
            validN <= tilingData_->vCacheRows && baseDAlign_ >= dAlign_ && validN <= (MAX_PIPE_ID - qBufferNum_);
        for (uint32_t workOffset = 0; workOffset < workCount; ++workOffset) {
            ProcessWork(workStart + workOffset, validN, cacheAllV);
        }
    }

private:
    static constexpr uint64_t FP32_BYTES = sizeof(float);
    static constexpr uint32_t FP32_32B_ALIGN_ELEMS = 8U;
    static constexpr uint32_t FP32_REG_ELEMS = BlockAttnResPrepareVF::FP32_REG_ELEMS;
    static constexpr uint32_t MAX_PIPE_ID = 8U;
    static constexpr uint32_t MTE3_MAX_PIPE_ID = 4U;
    static constexpr uint32_t STAT_BUFFER_NUM = 2U;
    __aicore__ inline static void WaitMte2ToVector(event_t eventId)
    {
        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
    }

    __aicore__ inline static void WaitVectorToMte3(event_t eventId)
    {
        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
    }

    __aicore__ inline void InitUbOffset()
    {
        qUbOffset_ = 0U;
        oUbOffset_ = qUbOffset_ + static_cast<uint64_t>(tilingData_->qBufferNum) * baseDAlign_ * FP32_BYTES;
        statUbOffset_ = oUbOffset_ + static_cast<uint64_t>(tilingData_->oBufferNum) * baseDAlign_ * FP32_BYTES;
        vCacheUbOffset_ =
            statUbOffset_ + static_cast<uint64_t>(STAT_BUFFER_NUM) * tilingData_->statUbElems * FP32_BYTES;
        vUbOffset_ = vCacheUbOffset_;
    }

    template <typename T>
    __aicore__ inline static __ubuf__ T *GetUbAddr(uint64_t byteOffset)
    {
        return AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, T>(byteOffset).Get();
    }

    template <typename T>
    __aicore__ inline static auto MakeGmTensor(__gm__ T *address, int64_t rows, int64_t columns)
    {
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(address),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<T>>(rows,
                                                                                                          columns));
    }

    template <typename T, typename GmTensor>
    __aicore__ inline static void CopyGmToUb(uint64_t dstByteOffset, const GmTensor &srcTensor, int64_t rowIndex,
                                             int64_t columnIndex, int64_t validElements, int64_t ubPitch)
    {
        auto ubStorageTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, T>(dstByteOffset),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<T>>(1, ubPitch));
        auto ubTensor = ubStorageTensor.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                              AscendC::Te::MakeShape(static_cast<int64_t>(1), validElements));
        auto gmTensor = srcTensor.Slice(AscendC::Te::MakeCoord(rowIndex, columnIndex),
                                        AscendC::Te::MakeShape(static_cast<int64_t>(1), validElements));
        auto copyGmToUb = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGmToUb, ubTensor, gmTensor);
    }

    template <typename T, typename GmTensor>
    __aicore__ inline static void CopyUbToGm(GmTensor &dstTensor, int64_t rowIndex, int64_t columnIndex,
                                             uint64_t srcByteOffset, int64_t validElements, int64_t ubPitch)
    {
        auto gmTensor = dstTensor.Slice(AscendC::Te::MakeCoord(rowIndex, columnIndex),
                                        AscendC::Te::MakeShape(static_cast<int64_t>(1), validElements));
        auto ubStorageTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, T>(srcByteOffset),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<T>>(1, ubPitch));
        auto ubTensor = ubStorageTensor.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                              AscendC::Te::MakeShape(static_cast<int64_t>(1), validElements));
        auto copyUbToGm = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
        AscendC::Te::Copy(copyUbToGm, gmTensor, ubTensor);
    }

    __aicore__ inline uint64_t ReadValidBlocks() const
    {
        auto validBlocksTensor = MakeGmTensor<uint64_t>(validBlocksAddr_, 1, 1);
        return validBlocksTensor[AscendC::Te::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0))];
    }

    __aicore__ inline __ubuf__ float *CopyInQ(uint64_t sIndex, uint32_t dOffset, uint32_t validD, uint32_t qBufferIndex)
    {
        const uint64_t byteOffset = qUbOffset_ + static_cast<uint64_t>(qBufferIndex) * baseDAlign_ * FP32_BYTES;
        auto pseudoQueryTensor = MakeGmTensor<float>(pseudoQueryAddr_, tilingData_->totalS, tilingData_->totalD);
        CopyGmToUb<float>(byteOffset, pseudoQueryTensor, static_cast<int64_t>(sIndex), dOffset, validD, baseDAlign_);
        WaitMte2ToVector(static_cast<event_t>(qBufferIndex));
        return GetUbAddr<float>(byteOffset);
    }

    __aicore__ inline __ubuf__ float *CopyInV(uint64_t tIndex, uint32_t nIndex, uint32_t dOffset, uint32_t validD,
                                              uint64_t vLoop)
    {
        const uint32_t bufferIndex = static_cast<uint32_t>(vLoop % vBufferNum_);
        const uint64_t byteOffset = vUbOffset_ + static_cast<uint64_t>(bufferIndex) * baseDAlign_ * FP32_BYTES;
        const int64_t rowIndex = static_cast<int64_t>(tIndex * tilingData_->totalN + nIndex);
        auto blockResTensor = MakeGmTensor<float>(
            blockResAddr_, static_cast<int64_t>(tilingData_->totalT) * tilingData_->totalN, tilingData_->totalD);
        CopyGmToUb<float>(byteOffset, blockResTensor, rowIndex, dOffset, validD, baseDAlign_);
        WaitMte2ToVector(static_cast<event_t>(bufferIndex));
        return GetUbAddr<float>(byteOffset);
    }

    __aicore__ inline __ubuf__ float *CopyVToCache(uint64_t tIndex, uint32_t nIndex, uint32_t dOffset,
                                                   uint32_t cacheOffset, uint32_t validD)
    {
        const uint64_t byteOffset = vCacheUbOffset_ + static_cast<uint64_t>(cacheOffset) * FP32_BYTES;
        const int64_t rowIndex = static_cast<int64_t>(tIndex * tilingData_->totalN + nIndex);
        auto blockResTensor = MakeGmTensor<float>(
            blockResAddr_, static_cast<int64_t>(tilingData_->totalT) * tilingData_->totalN, tilingData_->totalD);
        CopyGmToUb<float>(byteOffset, blockResTensor, rowIndex, dOffset, validD, dAlign_);
        WaitMte2ToVector(static_cast<event_t>(0));
        return GetUbAddr<float>(byteOffset);
    }

    __aicore__ inline void CopyOutStats(uint64_t sIndex, uint64_t tIndex, uint64_t statByteOffset, event_t eventId)
    {
        WaitVectorToMte3(eventId);
        auto logitMaxTensor = MakeGmTensor<float>(logitMaxAddr_, tilingData_->totalS, tilingData_->totalT);
        auto expSumTensor = MakeGmTensor<float>(expSumAddr_, tilingData_->totalS, tilingData_->totalT);
        CopyUbToGm<float>(logitMaxTensor, static_cast<int64_t>(sIndex), static_cast<int64_t>(tIndex),
                          statByteOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::MAX_OFFSET) * FP32_BYTES, 1,
                          FP32_32B_ALIGN_ELEMS);
        CopyUbToGm<float>(expSumTensor, static_cast<int64_t>(sIndex), static_cast<int64_t>(tIndex),
                          statByteOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::SUM_OFFSET) * FP32_BYTES, 1,
                          FP32_32B_ALIGN_ELEMS);
    }

    __aicore__ inline void CopyOutO(uint64_t sIndex, uint64_t tIndex, uint32_t dOffset, uint32_t validD, uint32_t dLoop)
    {
        const uint32_t bufferIndex = dLoop % tilingData_->oBufferNum;
        const uint64_t byteOffset = oUbOffset_ + static_cast<uint64_t>(bufferIndex) * baseDAlign_ * FP32_BYTES;
        const event_t eventId = static_cast<event_t>(bufferIndex);
        WaitVectorToMte3(eventId);
        const int64_t rowIndex = static_cast<int64_t>(sIndex * tilingData_->totalT + tIndex);
        auto numeratorTensor = MakeGmTensor<float>(
            numeratorAddr_, static_cast<int64_t>(tilingData_->totalS) * tilingData_->totalT, tilingData_->totalD);
        CopyUbToGm<float>(numeratorTensor, rowIndex, dOffset, byteOffset, validD, baseDAlign_);
    }

    __aicore__ inline void AccumulateStats(__ubuf__ float *qAddr, __ubuf__ float *vAddr, __ubuf__ float *statAddr,
                                           uint32_t dLoop, uint32_t validD, uint32_t nIndex)
    {
        if (dLoop == 0U) {
            BlockAttnResPrepareVF::AccumulateSquareDot<true>(qAddr, vAddr, statAddr, validD, nIndex);
        } else {
            BlockAttnResPrepareVF::AccumulateSquareDot<false>(qAddr, vAddr, statAddr, validD, nIndex);
        }
    }

    __aicore__ inline void AccumulateOutput(__ubuf__ float *vAddr, __ubuf__ float *weightAddr,
                                            __ubuf__ float *outputAddr, uint32_t nIndex, uint32_t validN,
                                            uint32_t validD)
    {
        if (validN == 1U) {
            BlockAttnResPrepareVF::CopySingleBlock(vAddr, outputAddr, validD);
        } else if (nIndex == 0U) {
            BlockAttnResPrepareVF::WeightedAccumulate<true>(vAddr, weightAddr, outputAddr, validD);
        } else {
            BlockAttnResPrepareVF::WeightedAccumulate<false>(vAddr, weightAddr, outputAddr, validD);
        }
    }

    __aicore__ inline void FinalizeStats(__ubuf__ float *statAddr, uint32_t validN)
    {
        const float reciprocalD = 1.0F / static_cast<float>(tilingData_->totalD);
        if (validN == 1U) {
            BlockAttnResPrepareVF::FinalizeSingleBlock(statAddr, reciprocalD, tilingData_->eps);
        } else {
            BlockAttnResPrepareVF::FinalizeSoftmax(statAddr, validN, reciprocalD, tilingData_->eps);
        }
    }

    __aicore__ inline void ProcessEmptyWork(uint64_t workId)
    {
        const uint64_t tIndex = workId / tilingData_->totalS;
        const uint64_t sIndex = workId - tIndex * tilingData_->totalS;
        const uint32_t statBufferIndex = statLoop_ % STAT_BUFFER_NUM;
        const event_t statEvent = static_cast<event_t>(tilingData_->oBufferNum + statBufferIndex);
        const uint64_t statByteOffset =
            statUbOffset_ + static_cast<uint64_t>(statBufferIndex) * tilingData_->statUbElems * FP32_BYTES;
        wait_flag(PIPE_MTE3, PIPE_V, statEvent);
        __ubuf__ float *statAddr = GetUbAddr<float>(statByteOffset);
        BlockAttnResPrepareVF::InitializeEmptyOnlineSoftmax(statAddr);
        CopyOutStats(sIndex, tIndex, statByteOffset, statEvent);
        set_flag(PIPE_MTE3, PIPE_V, statEvent);

        for (uint32_t dLoop = 0; dLoop < dTileNum_; ++dLoop) {
            const uint32_t dOffset = dLoop * tilingData_->baseD;
            const uint64_t remainingD = tilingData_->totalD - dOffset;
            const uint32_t validD =
                static_cast<uint32_t>(remainingD < tilingData_->baseD ? remainingD : tilingData_->baseD);
            const uint32_t outputBufferIndex = oLoop_ % tilingData_->oBufferNum;
            const uint64_t outputByteOffset =
                oUbOffset_ + static_cast<uint64_t>(outputBufferIndex) * baseDAlign_ * FP32_BYTES;
            __ubuf__ float *outputAddr = GetUbAddr<float>(outputByteOffset);
            const event_t outputEvent = static_cast<event_t>(outputBufferIndex);
            wait_flag(PIPE_MTE3, PIPE_V, outputEvent);
            BlockAttnResPrepareVF::FillZero(outputAddr, validD);

            CopyOutO(sIndex, tIndex, dOffset, validD, oLoop_);
            set_flag(PIPE_MTE3, PIPE_V, outputEvent);
            ++oLoop_;
        }
        ++statLoop_;
    }

    __aicore__ inline void ProcessWork(uint64_t workId, uint32_t validN, bool cacheAllV)
    {
        const uint64_t tIndex = workId / tilingData_->totalS;
        const uint64_t sIndex = workId - tIndex * tilingData_->totalS;
        const uint32_t statBufferIndex = statLoop_ % STAT_BUFFER_NUM;
        const event_t statEvent = static_cast<event_t>(tilingData_->oBufferNum + statBufferIndex);
        const uint64_t statByteOffset =
            statUbOffset_ + static_cast<uint64_t>(statBufferIndex) * tilingData_->statUbElems * FP32_BYTES;
        __ubuf__ float *statAddr = GetUbAddr<float>(statByteOffset);

        wait_flag(PIPE_MTE3, PIPE_V, statEvent);
        for (uint32_t dLoop = 0; dLoop < dTileNum_; ++dLoop) {
            const uint32_t dOffset = dLoop * tilingData_->baseD;
            const uint64_t remainingD = tilingData_->totalD - dOffset;
            const uint32_t validD =
                static_cast<uint32_t>(remainingD < tilingData_->baseD ? remainingD : tilingData_->baseD);
            const uint32_t qBufferIndex = qLoop_ % tilingData_->qBufferNum;
            const event_t qEvent = static_cast<event_t>(qBufferIndex);
            wait_flag(PIPE_V, PIPE_MTE2, qEvent);
            __ubuf__ float *qAddr = CopyInQ(sIndex, dOffset, validD, qBufferIndex);

            for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
                __ubuf__ float *vAddr;
                if (cacheAllV) {
                    uint32_t vBufferId = validN > 1 ? nIndex : (vLoop_ & 1);
                    wait_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + static_cast<event_t>(vBufferId));
                    const uint32_t cacheOffset = vBufferId * dAlign_ + dOffset;
                    vAddr = CopyVToCache(tIndex, nIndex, dOffset, cacheOffset, validD);
                } else {
                    wait_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + (vLoop_ % vBufferNum_));
                    vAddr = CopyInV(tIndex, nIndex, dOffset, validD, vLoop_);
                }
                AccumulateStats(qAddr, vAddr, statAddr, dLoop, validD, nIndex);
                if (!cacheAllV) {
                    set_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + (vLoop_ % vBufferNum_));
                    vLoop_++;
                }
            }
            set_flag(PIPE_V, PIPE_MTE2, qEvent);
            qLoop_++;
        }

        FinalizeStats(statAddr, validN);
        CopyOutStats(sIndex, tIndex, statByteOffset, statEvent);
        set_flag(PIPE_MTE3, PIPE_V, statEvent);

        __ubuf__ float *weightAddr = statAddr + BlockAttnResPrepareVF::DOT_OFFSET;
        for (uint32_t dLoop = 0; dLoop < dTileNum_; ++dLoop) {
            const uint32_t dOffset = dLoop * tilingData_->baseD;
            const uint64_t remainingD = tilingData_->totalD - dOffset;
            const uint32_t validD =
                static_cast<uint32_t>(remainingD < tilingData_->baseD ? remainingD : tilingData_->baseD);
            const uint32_t outputBufferIndex = oLoop_ % tilingData_->oBufferNum;
            const uint64_t outputByteOffset =
                oUbOffset_ + static_cast<uint64_t>(outputBufferIndex) * baseDAlign_ * FP32_BYTES;
            __ubuf__ float *outputAddr = GetUbAddr<float>(outputByteOffset);
            const event_t outputEvent = static_cast<event_t>(outputBufferIndex);
            wait_flag(PIPE_MTE3, PIPE_V, outputEvent);
            for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
                __ubuf__ float *vAddr;
                if (cacheAllV) {
                    uint32_t vBufferId = validN > 1 ? nIndex : (vLoop_ & 1);
                    const uint32_t cacheOffset = vBufferId * dAlign_ + dOffset;
                    vAddr = GetUbAddr<float>(vCacheUbOffset_ + static_cast<uint64_t>(cacheOffset) * FP32_BYTES);
                } else {
                    wait_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + (vLoop_ % vBufferNum_));
                    vAddr = CopyInV(tIndex, nIndex, dOffset, validD, vLoop_);
                }
                AccumulateOutput(vAddr, weightAddr + nIndex, outputAddr, nIndex, validN, validD);
                if (cacheAllV) {
                    uint32_t vBufferId = validN > 1 ? nIndex : (vLoop_ & 1);
                    set_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + vBufferId);
                } else {
                    set_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + (vLoop_ % vBufferNum_));
                }
                vLoop_++;
            }

            CopyOutO(sIndex, tIndex, dOffset, validD, oLoop_);
            set_flag(PIPE_MTE3, PIPE_V, outputEvent);
            ++oLoop_;
        }
        PipeBarrier<PIPE_V>();
        ++statLoop_;
    }

    const optiling::BlockAttnResPrepareTilingData *__restrict tilingData_ = nullptr;
    bool initialized_ = false;

    uint64_t qUbOffset_ = 0U;
    uint64_t vUbOffset_ = 0U;
    uint64_t oUbOffset_ = 0U;
    uint64_t statUbOffset_ = 0U;
    uint64_t vCacheUbOffset_ = 0U;
    uint32_t baseDAlign_ = 0U;
    uint32_t dAlign_ = 0U;
    uint32_t dTileNum_ = 0U;
    uint32_t qLoop_ = 0;
    uint32_t vLoop_ = 0;
    uint32_t oLoop_ = 0;
    uint32_t statLoop_ = 0;
    uint64_t validN_ = 0U;
    uint8_t qBufferNum_ = 1;
    uint8_t vBufferNum_ = 1;

    __gm__ float *blockResAddr_ = nullptr;
    __gm__ uint64_t *validBlocksAddr_ = nullptr;
    __gm__ float *pseudoQueryAddr_ = nullptr;
    __gm__ float *numeratorAddr_ = nullptr;
    __gm__ float *logitMaxAddr_ = nullptr;
    __gm__ float *expSumAddr_ = nullptr;
};

} // namespace BlockAttnResPrepare

#endif // BLOCK_ATTN_RES_PREPARE_VECTOR_H
