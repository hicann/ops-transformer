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

#include <cstdint>

#include "block_attn_res_prepare_tiling_data.h"
#include "c_api/scalar_compute/scalar_load.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "vf/block_attn_res_prepare_vf.h"

namespace BlockAttnResPrepare {

using namespace AscendC;

// SINGLE_S_CACHE specializes single-S tasks; V may persist across queries.
template <typename BLOCK_RES_DTYPE, bool SINGLE_S_CACHE = false>
class BlockAttnResPrepareVector {
public:
    __aicore__ inline BlockAttnResPrepareVector()
    {
#pragma unroll
        for (uint32_t eventIndex = 0; eventIndex < V_MTE2_EVENT_COUNT; ++eventIndex) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(eventIndex));
        }
#pragma unroll
        for (uint32_t eventIndex = 0; eventIndex < MTE3_V_EVENT_COUNT; ++eventIndex) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(eventIndex));
        }
    }

    __aicore__ inline ~BlockAttnResPrepareVector()
    {
#pragma unroll
        for (uint32_t eventIndex = 0; eventIndex < V_MTE2_EVENT_COUNT; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(eventIndex));
        }
#pragma unroll
        for (uint32_t eventIndex = 0; eventIndex < MTE3_V_EVENT_COUNT; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(eventIndex));
        }
    }

    __aicore__ inline void operator()(
        GM_ADDR blockRes, GM_ADDR validBlocks, GM_ADDR pseudoQuery, GM_ADDR numerator, GM_ADDR logitMax, GM_ADDR expSum,
        __tiling_data_ptr__ optiling::BlockAttnResPrepareTilingData* __restrict tilingData)
    {
        Init(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, tilingData);
        Process();
    }

private:
    __aicore__ inline void Init(GM_ADDR blockRes, GM_ADDR validBlocks, GM_ADDR pseudoQuery, GM_ADDR numerator,
                                GM_ADDR logitMax, GM_ADDR expSum,
                                __tiling_data_ptr__ optiling::BlockAttnResPrepareTilingData* __restrict tilingData)
    {
        tilingData_ = tilingData;
        const uint64_t rawValidN = asc_load_dev(reinterpret_cast<__gm__ uint64_t*>(validBlocks));
        validN_ = static_cast<uint32_t>(rawValidN <= tilingData_->totalN ? rawValidN : tilingData_->totalN);
        // Tail loads may read adjacent rows; VF masks exclude lanes beyond D.
        dAlign_ = tilingData_->dAlign;
        sBlockCount_ = tilingData_->sBlockCount;
        if constexpr (SINGLE_S_CACHE) {
            qBufferNum_ = tilingData_->qBufferNum;
            oBufferNum_ = qBufferNum_;
            // Per-query V caching uses one event per V row. Cache all token V
            // under one event when Q leaves too few events for that layout.
            cacheVPerToken_ = validN_ > V_MTE2_EVENT_COUNT - qBufferNum_;
        } else {
            // Packed-S tasks keep all token V rows resident across their S span.
            cacheVPerToken_ = tilingData_->sBlockSize > 1U;
            qBufferNum_ = tilingData_->qBufferNum;
            oBufferNum_ = qBufferNum_;
            if (cacheVPerToken_ && validN_ != 0U) {
                InitSGroupBuffers();
            }
        }

        blockResAddr_ = reinterpret_cast<__gm__ BLOCK_RES_DTYPE*>(blockRes);
        pseudoQueryAddr_ = reinterpret_cast<__gm__ float*>(pseudoQuery);
        numeratorAddr_ = reinterpret_cast<__gm__ float*>(numerator);
        logitMaxAddr_ = reinterpret_cast<__gm__ float*>(logitMax);
        expSumAddr_ = reinterpret_cast<__gm__ float*>(expSum);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t usedCoreNum = tilingData_->usedCoreNum;
        if (blockIdx >= usedCoreNum) {
            return;
        }
        if constexpr (std::is_same<BLOCK_RES_DTYPE, bfloat16_t>::value) {
            if (validN_ == 1U && tilingData_->qCacheCoresPerGroup != 0U) {
                ProcessTokensWithCachedQ(blockResAddr_, pseudoQueryAddr_, numeratorAddr_, logitMaxAddr_, expSumAddr_,
                                         tilingData_);
                return;
            }
        }
        SetUbOffsets();
        // The host supplies balanced per-core work and the remainder.
        const uint32_t workPerCore = tilingData_->workPerCore;
        const uint32_t remainder = tilingData_->remainder;
        const uint32_t workStart = blockIdx * workPerCore + ((blockIdx < remainder) ? blockIdx : remainder);
        const uint32_t workCount = workPerCore + ((blockIdx < remainder) ? 1U : 0U);
        if (validN_ == 0U) {
            ProcessEmptyRange(workStart, workCount);
            return;
        }
        if (!cacheVPerToken_) {
            ProcessQuery(workStart, workCount);
            return;
        }
        ProcessWorkByToken(workStart, workCount);
    }

    __aicore__ inline static void CopyCachedQueryV(__gm__ bfloat16_t* blockRes, uint32_t t, uint32_t n, uint32_t d,
                                                   uint64_t vOffset, uint32_t slot)
    {
        const event_t event = static_cast<event_t>(QUERY_CACHE_V_EVENT_BASE + slot);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        auto layout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<bfloat16_t>>(
            1L, static_cast<int64_t>(d));
        auto gm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(blockRes + static_cast<uint64_t>(t) * n * d), layout);
        auto ub = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(
                                           vOffset + static_cast<uint64_t>(slot) * d * sizeof(bfloat16_t)),
                                       layout);
        asc::te::copy(ub, gm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    // Keep three S rows of Q resident across a balanced token range. Each output
    // is a converted copy of its single valid V row. Two V slots and three O/stat
    // slots let input loads and output writeback rotate independently.
    // The constructor initialized V_MTE2[0..7] and MTE3_V[0..4].
    __aicore__ inline static void ProcessTokensWithCachedQ(
        __gm__ bfloat16_t* blockRes, __gm__ float* query, __gm__ float* numerator, __gm__ float* logitMax,
        __gm__ float* expSum, __tiling_data_ptr__ optiling::BlockAttnResPrepareTilingData* __restrict td)
    {
        const uint32_t coresPerSGroup = td->qCacheCoresPerGroup;
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t group = core / coresPerSGroup;
        const uint32_t local = core - group * coresPerSGroup;
        const uint32_t baseTokens = td->totalT / coresPerSGroup;
        const uint32_t extraTokens = td->totalT - baseTokens * coresPerSGroup;
        const uint32_t tokens = baseTokens + (local < extraTokens ? 1U : 0U);
        const uint32_t tStart = local * baseTokens + (local < extraTokens ? local : extraTokens);
        const uint32_t sStart = group * THREE_S_ROWS;
        const uint32_t d = td->totalD;
        // This path requires D % FP32_REG_ELEMS == 0; the tail contains zero or one VL.
        const uint16_t tailLoops = static_cast<uint16_t>(d / BlockAttnResPrepareVF::FP32_REG_ELEMS %
                                                         BlockAttnResPrepareVF::BF16_STATS_D_UNROLL_FACTOR);
        const uint64_t rowBytes = static_cast<uint64_t>(d) * sizeof(float);
        const uint64_t vRowBytes = static_cast<uint64_t>(d) * sizeof(bfloat16_t);
        const uint64_t oOffset = THREE_S_ROWS * rowBytes;
        const uint64_t vOffset = oOffset + QUERY_CACHE_OUTPUT_BUFFER_NUM * rowBytes;
        const uint64_t statOffset = vOffset + DOUBLE_BUFFER_NUM * vRowBytes;
        auto qMem = asc::te::make_mem_ptr<asc::te::location::ub, float>(0UL);
        auto qLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(THREE_S_ROWS),
                                                                              static_cast<int64_t>(d));
        auto qGm = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(query + static_cast<uint64_t>(sStart) * d), qLayout);
        auto qUb = asc::te::make_tensor(qMem, qLayout);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        asc::te::copy(qUb, qGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        uint32_t slot = 0U;
        uint32_t outputSlot = 0U;
        auto oLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, static_cast<int64_t>(d));
        auto statUbLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
            static_cast<int64_t>(THREE_S_ROWS),
            static_cast<int64_t>(BlockAttnResPrepareVF::CACHED_QUERY_STAT_ROW_ELEMS));
        auto statGmLayout = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(static_cast<int64_t>(THREE_S_ROWS),
                                                                                   static_cast<int64_t>(td->totalT));
        for (uint32_t i = 0U; i < tokens; ++i) {
            const uint32_t t = tStart + i;
            CopyCachedQueryV(blockRes, t, td->totalN, d, vOffset, slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(outputSlot));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(QUERY_CACHE_V_EVENT_BASE + slot));
            auto vMem = asc::te::make_mem_ptr<asc::te::location::ub, bfloat16_t>(vOffset + slot * vRowBytes);
            auto oMem = asc::te::make_mem_ptr<asc::te::location::ub, float>(oOffset + outputSlot * rowBytes);
            auto statMem =
                asc::te::make_mem_ptr<asc::te::location::ub, float>(statOffset + outputSlot * QUERY_CACHE_STAT_BYTES);
            asc_vf_call<BlockAttnResPrepareVF::PrepareThreeCachedQueries>(
                qMem.get(), vMem.get(), oMem.get(), statMem.get(), d, td->reciprocalD, td->eps, tailLoops);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(QUERY_CACHE_V_EVENT_BASE + slot));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(outputSlot));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(outputSlot));
            auto outUb = asc::te::make_tensor(oMem, oLayout);
#pragma unroll
            for (uint32_t row = 0U; row < THREE_S_ROWS; ++row) {
                auto outGm =
                    asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
                                             numerator + (static_cast<uint64_t>(sStart + row) * td->totalT + t) * d),
                                         oLayout);
                asc::te::copy(outGm, outUb);
            }
            auto statUb = asc::te::make_tensor(statMem, statUbLayout);
            auto statShape = asc::te::make_shape(static_cast<int64_t>(THREE_S_ROWS), 1L);
            auto mSrc = statUb.slice(asc::te::make_coord(0L, 0L), statShape);
            auto lSrc = statUb.slice(
                asc::te::make_coord(0L, static_cast<int64_t>(BlockAttnResPrepareVF::SCALAR_BLOCK_ELEMS)), statShape);
            const uint64_t gmOffset = static_cast<uint64_t>(sStart) * td->totalT + t;
            auto mGm =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(logitMax + gmOffset), statGmLayout);
            auto lGm =
                asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(expSum + gmOffset), statGmLayout);
            auto mDst = mGm.slice(asc::te::make_coord(0L, 0L), statShape);
            auto lDst = lGm.slice(asc::te::make_coord(0L, 0L), statShape);
            asc::te::copy(mDst, mSrc);
            asc::te::copy(lDst, lSrc);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(outputSlot));
            slot ^= 1U;
            outputSlot = NextBuffer(outputSlot, QUERY_CACHE_OUTPUT_BUFFER_NUM);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    }

    // Coalesce adjacent tasks for the same token (or token pair) into one S span.
    // Load V once per span; single-S tasks can reuse V across queries here too.
    __aicore__ inline void ProcessWorkByToken(uint32_t workStart, uint32_t workCount)
    {
        const uint32_t tokenCount = SINGLE_S_CACHE ? 1U : (useTokenPair_ ? TOKEN_PAIR_SIZE : 1U);
        const uint32_t sBlocksPerTokenGroup = SINGLE_S_CACHE ? tilingData_->totalS : tokenCount * sBlockCount_;
        const uint32_t sPerBlock = SINGLE_S_CACHE ? 1U : tilingData_->sBlockSize / tokenCount;
        const uint32_t tokenGroupIndex = workStart / sBlocksPerTokenGroup;
        uint32_t tIndex = tokenGroupIndex * tokenCount;
        uint32_t sBlockIndex = workStart - tokenGroupIndex * sBlocksPerTokenGroup;
        for (uint32_t remainingWork = workCount; remainingWork != 0U;) {
            const uint32_t availableWork = sBlocksPerTokenGroup - sBlockIndex;
            const uint32_t currentWork = remainingWork < availableWork ? remainingWork : availableWork;
            const uint32_t sStart = sBlockIndex * sPerBlock;
            const uint32_t sBlockEnd = sBlockIndex + currentWork;
            uint32_t sEnd = sBlockEnd;
            if constexpr (!SINGLE_S_CACHE) {
                // The padded last block ends at S; avoid overflowing its nominal uint32 extent.
                sEnd = sBlockEnd == sBlocksPerTokenGroup ? tilingData_->totalS : sBlockEnd * sPerBlock;
            }
            if constexpr (SINGLE_S_CACHE) {
                ProcessSRangeWithCachedV(tIndex, sStart, sEnd, validN_);
            } else if (useTokenPair_) {
                ProcessSGroupsWithCachedV<TWO_S_ROWS, TOKEN_PAIR_SIZE>(tIndex, sStart, sEnd, validN_);
            } else if (validN_ > 1U && (sEnd - sStart) % sRowsPerGroup_ == 0U) {
                if (sRowsPerGroup_ == THREE_S_ROWS) {
                    ProcessSGroupsWithCachedV<THREE_S_ROWS>(tIndex, sStart, sEnd, validN_);
                } else {
                    ProcessSGroupsWithCachedV<TWO_S_ROWS>(tIndex, sStart, sEnd, validN_);
                }
            } else {
                ProcessSRangeWithCachedV(tIndex, sStart, sEnd, validN_);
            }
            remainingWork -= currentWork;
            sBlockIndex = 0U;
            tIndex += tokenCount;
        }
    }

    static constexpr uint64_t FP32_BYTES = sizeof(float);
    static constexpr uint64_t V_ELEM_BYTES = sizeof(BLOCK_RES_DTYPE);
    static constexpr uint32_t FP32_32B_ALIGN_ELEMS = 8U;
    // [RegBase-native] Event IDs 0..7 are reserved for V->MTE2 flags and
    // 0..4 for MTE3->V flags: four output buffers and one statistics buffer.
    static constexpr uint32_t V_MTE2_EVENT_COUNT = 8U;
    static constexpr event_t TOKEN_V_EVENT_ID = static_cast<event_t>(V_MTE2_EVENT_COUNT - 1U);
    static constexpr uint32_t MTE3_V_EVENT_COUNT = 5U;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2U;
    static constexpr uint32_t QUERY_CACHE_OUTPUT_BUFFER_NUM = 3U;
    static constexpr uint32_t STAT_UB_ELEMS = BlockAttnResPrepareVF::STAT_UB_ELEMS;
    static constexpr uint32_t GROUP_STAT_BUFFER_NUM = 4U;
    static constexpr uint32_t TWO_S_ROWS = 2U;
    static constexpr uint32_t THREE_S_ROWS = 3U;
    static constexpr uint32_t TOKEN_PAIR_SIZE = 2U;
    static constexpr uint32_t ROW_STAT_BUFFER_NUM = 2U;
    static constexpr uint32_t THREE_N_BLOCKS = BlockAttnResPrepareVF::THREE_N_BLOCKS;
    static constexpr uint32_t FOUR_N_BLOCKS = BlockAttnResPrepareVF::FOUR_N_BLOCKS;
    static constexpr uint32_t FIVE_N_BLOCKS = BlockAttnResPrepareVF::FIVE_N_BLOCKS;
    static constexpr uint32_t FOUR_OUTPUT_BUFFERS = 4U;
    static constexpr uint32_t THIRD_OUTPUT_BUFFER_INDEX = 2U;
    // Each extra FP32 output buffer consumes two rows of the spare BF16 cache.
    static constexpr uint32_t MIN_SPARE_V_ROWS_FOR_THREE_OUTPUTS = 2U;
    static constexpr uint32_t MIN_SPARE_V_ROWS_FOR_FOUR_OUTPUTS = 4U;
    static constexpr uint32_t TWO_N_BLOCKS = 2U;
    static constexpr uint32_t FOUR_N_BLOCKS_MIN_T = 64U;
    static constexpr uint32_t Q_GROUP_EVENT_ID = 0U;
    static constexpr uint32_t QUERY_CACHE_V_EVENT_BASE = V_MTE2_EVENT_COUNT - DOUBLE_BUFFER_NUM;
    static constexpr uint64_t QUERY_CACHE_STAT_BYTES =
        THREE_S_ROWS * BlockAttnResPrepareVF::CACHED_QUERY_STAT_ROW_ELEMS * FP32_BYTES;
    __aicore__ inline static void SyncMte2ToVector(event_t eventId)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline static void SyncVectorToMte3(event_t eventId)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline static uint32_t NextBuffer(uint32_t bufferIndex, uint32_t bufferCount)
    {
        ++bufferIndex;
        return bufferIndex == bufferCount ? 0U : bufferIndex;
    }

    __aicore__ inline void InitSGroupBuffers()
    {
        const bool hasFullSBlocks =
            static_cast<uint64_t>(sBlockCount_) * tilingData_->sBlockSize == tilingData_->totalS;
        // Pair tokens only for full-width S blocks; each pair keeps the original work count.
        useTokenPair_ = validN_ == TWO_N_BLOCKS && tilingData_->totalT % TOKEN_PAIR_SIZE == 0U &&
                        static_cast<uint64_t>(tilingData_->totalN) >= TOKEN_PAIR_SIZE * validN_ && hasFullSBlocks &&
                        tilingData_->sBlockSize % (TOKEN_PAIR_SIZE * TWO_S_ROWS) == 0U;
        const bool useThreeSRows = validN_ >= THREE_N_BLOCKS && validN_ <= tilingData_->threeSMaxN && hasFullSBlocks &&
                                   tilingData_->sBlockSize % THREE_S_ROWS == 0U;
        sRowsPerGroup_ = useThreeSRows ? THREE_S_ROWS : TWO_S_ROWS;
        const uint64_t cachedVRows = useTokenPair_ ? TOKEN_PAIR_SIZE * validN_ : validN_;
        const uint64_t spareVRows = tilingData_->totalN - cachedVRows;
        const bool useGroupedS =
            validN_ > 1U && hasFullSBlocks && (useThreeSRows || tilingData_->sBlockSize % TWO_S_ROWS == 0U);
        // Reuse spare BF16 cache storage for extra FP32 Q/output rows.
        oBufferNum_ =
            !useGroupedS ?
                DOUBLE_BUFFER_NUM :
                (useThreeSRows ?
                     THREE_S_ROWS :
                     (spareVRows >= MIN_SPARE_V_ROWS_FOR_FOUR_OUTPUTS ?
                          FOUR_OUTPUT_BUFFERS :
                          (spareVRows >= MIN_SPARE_V_ROWS_FOR_THREE_OUTPUTS ? THREE_S_ROWS : DOUBLE_BUFFER_NUM)));
        qBufferNum_ = useThreeSRows ? THREE_S_ROWS : DOUBLE_BUFFER_NUM;
    }

    __aicore__ inline void SetUbOffsets()
    {
        qUbOffset_ = 0U;
        oUbOffset_ = qUbOffset_ + static_cast<uint64_t>(qBufferNum_) * dAlign_ * FP32_BYTES;
        statUbOffset_ = oUbOffset_ + static_cast<uint64_t>(oBufferNum_) * dAlign_ * FP32_BYTES;
        const uint32_t statBufferNum = cacheVPerToken_ ? GROUP_STAT_BUFFER_NUM : ROW_STAT_BUFFER_NUM;
        vCacheUbOffset_ = statUbOffset_ + static_cast<uint64_t>(statBufferNum) * STAT_UB_ELEMS * FP32_BYTES;
    }

    template <typename T>
    __aicore__ inline static __ubuf__ T* GetUbAddr(uint64_t byteOffset)
    {
        return asc::te::make_mem_ptr<asc::te::location::ub, T>(byteOffset).get();
    }

    template <typename T>
    __aicore__ inline static auto MakeGmTensor(__gm__ T* address, int64_t rows, int64_t columns)
    {
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(address),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(rows, columns));
    }

    template <typename T, typename GmTensor>
    __aicore__ inline static void CopyGmToUb(uint64_t dstByteOffset, const GmTensor& srcTensor, int64_t rowIndex,
                                             int64_t columnIndex, int64_t validElements, int64_t ubRowStride)
    {
        auto ubStorageTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, T>(dstByteOffset),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(1, ubRowStride));
        auto ubTensor = ubStorageTensor.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                              asc::te::make_shape(static_cast<int64_t>(1), validElements));
        auto gmTensor = srcTensor.slice(asc::te::make_coord(rowIndex, columnIndex),
                                        asc::te::make_shape(static_cast<int64_t>(1), validElements));
        asc::te::copy(ubTensor, gmTensor);
    }

    template <typename T, typename GmTensor>
    __aicore__ inline static void CopyUbToGm(GmTensor& dstTensor, int64_t rowIndex, int64_t columnIndex,
                                             uint64_t srcByteOffset, int64_t validElements, int64_t ubRowStride)
    {
        auto gmTensor = dstTensor.slice(asc::te::make_coord(rowIndex, columnIndex),
                                        asc::te::make_shape(static_cast<int64_t>(1), validElements));
        auto ubStorageTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, T>(srcByteOffset),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(1, ubRowStride));
        auto ubTensor = ubStorageTensor.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                              asc::te::make_shape(static_cast<int64_t>(1), validElements));
        asc::te::copy(gmTensor, ubTensor);
    }

    // Issue the copy here; the caller waits for MTE2 before consuming Q.
    __aicore__ inline __ubuf__ float* CopyInQuery(uint64_t sIndex, uint32_t qBufferIndex)
    {
        const uint64_t byteOffset = qUbOffset_ + static_cast<uint64_t>(qBufferIndex) * dAlign_ * FP32_BYTES;
        auto pseudoQueryTensor = MakeGmTensor<float>(pseudoQueryAddr_, tilingData_->totalS, tilingData_->totalD);
        CopyGmToUb<float>(byteOffset, pseudoQueryTensor, static_cast<int64_t>(sIndex), 0U, tilingData_->totalD,
                          dAlign_);
        return GetUbAddr<float>(byteOffset);
    }

    // Cache and streaming callers supply their own UB region and wait before consuming V.
    __aicore__ inline __ubuf__ BLOCK_RES_DTYPE* CopyInV(uint64_t tIndex, uint32_t nIndex, uint64_t byteOffset)
    {
        const int64_t rowIndex = static_cast<int64_t>(tIndex * tilingData_->totalN + nIndex);
        auto blockResTensor = MakeGmTensor<BLOCK_RES_DTYPE>(
            blockResAddr_, static_cast<int64_t>(tilingData_->totalT) * tilingData_->totalN, tilingData_->totalD);
        CopyGmToUb<BLOCK_RES_DTYPE>(byteOffset, blockResTensor, rowIndex, 0U, tilingData_->totalD, dAlign_);
        return GetUbAddr<BLOCK_RES_DTYPE>(byteOffset);
    }

    // Load all validN V rows per token once and retain them until this S span finishes.
    __aicore__ inline void LoadTokenV(uint64_t tIndex, uint32_t validN, uint32_t tokenCount = 1U)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(TOKEN_V_EVENT_ID);
        for (uint32_t token = 0; token < tokenCount; ++token) {
            for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
                const uint64_t vRowIndex = static_cast<uint64_t>(token) * validN + nIndex;
                const uint64_t byteOffset = vCacheUbOffset_ + vRowIndex * dAlign_ * V_ELEM_BYTES;
                CopyInV(tIndex + token, nIndex, byteOffset);
            }
        }
    }

    __aicore__ inline void CopyOutStats(uint64_t sIndex, uint64_t tIndex, uint64_t statByteOffset, event_t eventId)
    {
        SyncVectorToMte3(eventId);
        auto logitMaxTensor = MakeGmTensor<float>(logitMaxAddr_, tilingData_->totalS, tilingData_->totalT);
        auto expSumTensor = MakeGmTensor<float>(expSumAddr_, tilingData_->totalS, tilingData_->totalT);
        CopyUbToGm<float>(logitMaxTensor, static_cast<int64_t>(sIndex), static_cast<int64_t>(tIndex),
                          statByteOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::LOGIT_MAX_OFFSET) * FP32_BYTES,
                          1, FP32_32B_ALIGN_ELEMS);
        CopyUbToGm<float>(expSumTensor, static_cast<int64_t>(sIndex), static_cast<int64_t>(tIndex),
                          statByteOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::EXP_SUM_OFFSET) * FP32_BYTES, 1,
                          FP32_32B_ALIGN_ELEMS);
    }

    template <bool SYNC_BEFORE_COPY = true>
    __aicore__ inline void CopyOutNumerator(uint64_t sIndex, uint64_t tIndex, uint32_t bufferIndex)
    {
        const uint64_t byteOffset = oUbOffset_ + static_cast<uint64_t>(bufferIndex) * dAlign_ * FP32_BYTES;
        const event_t eventId = static_cast<event_t>(bufferIndex);
        if constexpr (SYNC_BEFORE_COPY) {
            SyncVectorToMte3(eventId);
        }
        const int64_t rowIndex = static_cast<int64_t>(sIndex * tilingData_->totalT + tIndex);
        auto numeratorTensor = MakeGmTensor<float>(
            numeratorAddr_, static_cast<int64_t>(tilingData_->totalS) * tilingData_->totalT, tilingData_->totalD);
        CopyUbToGm<float>(numeratorTensor, rowIndex, 0U, byteOffset, tilingData_->totalD, dAlign_);
    }

    __aicore__ inline void AccumulateOutput(__ubuf__ BLOCK_RES_DTYPE* vAddr, __ubuf__ float* weightAddr,
                                            __ubuf__ float* outputAddr, uint32_t nIndex, uint32_t validN,
                                            uint32_t validD)
    {
        if (validN == 1U) {
            asc_vf_call<BlockAttnResPrepareVF::CopySingleBlock<BLOCK_RES_DTYPE>>(vAddr, outputAddr, validD);
        } else if (nIndex == 0U) {
            asc_vf_call<BlockAttnResPrepareVF::WeightedAccumulate<BLOCK_RES_DTYPE, true>>(vAddr, weightAddr, outputAddr,
                                                                                          validD);
        } else {
            asc_vf_call<BlockAttnResPrepareVF::WeightedAccumulate<BLOCK_RES_DTYPE, false>>(vAddr, weightAddr,
                                                                                           outputAddr, validD);
        }
    }

    __aicore__ inline void FinalizeStats(__ubuf__ float* statAddr, uint32_t validN)
    {
        const float reciprocalD = tilingData_->reciprocalD;
        if (validN == 1U) {
            asc_vf_call<BlockAttnResPrepareVF::FinalizeSingleBlock>(statAddr, reciprocalD, tilingData_->eps);
        } else {
            asc_vf_call<BlockAttnResPrepareVF::FinalizeSoftmax>(statAddr, validN, reciprocalD, tilingData_->eps);
        }
    }

    __aicore__ inline void ProcessEmptyRange(uint32_t workStart, uint32_t workCount)
    {
        // Empty inputs only write the online-softmax identity; no Q/V prefetch or reduction.
        uint32_t tIndex = workStart / sBlockCount_;
        uint32_t sBlockIndex = workStart - tIndex * sBlockCount_;
        for (uint32_t workOffset = 0; workOffset < workCount; ++workOffset) {
            const uint32_t sStart = sBlockIndex * tilingData_->sBlockSize;
            const uint32_t sRows = tilingData_->totalS - sStart < tilingData_->sBlockSize ?
                                       tilingData_->totalS - sStart :
                                       tilingData_->sBlockSize;
            for (uint32_t sOffset = 0; sOffset < sRows; ++sOffset) {
                const uint64_t sIndex = static_cast<uint64_t>(sStart) + sOffset;
                const uint32_t statBufferIndex = cacheVPerToken_ ? 0U : statBufferCounter_ % ROW_STAT_BUFFER_NUM;
                const event_t statEvent = static_cast<event_t>(oBufferNum_ + statBufferIndex);
                const uint64_t statByteOffset =
                    statUbOffset_ + static_cast<uint64_t>(statBufferIndex) * STAT_UB_ELEMS * FP32_BYTES;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(statEvent);
                __ubuf__ float* statAddr = GetUbAddr<float>(statByteOffset);
                asc_vf_call<BlockAttnResPrepareVF::InitializeEmptyOnlineSoftmax>(statAddr);
                CopyOutStats(sIndex, tIndex, statByteOffset, statEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(statEvent);

                const uint32_t outputBufferIndex = oBufferIndex_;
                const uint64_t outputByteOffset =
                    oUbOffset_ + static_cast<uint64_t>(outputBufferIndex) * dAlign_ * FP32_BYTES;
                __ubuf__ float* outputAddr = GetUbAddr<float>(outputByteOffset);
                const event_t outputEvent = static_cast<event_t>(outputBufferIndex);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
                asc_vf_call<BlockAttnResPrepareVF::FillZero>(outputAddr, tilingData_->totalD);

                CopyOutNumerator(sIndex, tIndex, oBufferIndex_);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
                oBufferIndex_ = NextBuffer(oBufferIndex_, oBufferNum_);
                ++statBufferCounter_;
            }
            if (++sBlockIndex == sBlockCount_) {
                sBlockIndex = 0U;
                ++tIndex;
            }
        }
    }

    // Process a work range, optionally retaining V between each query's dot and output phases.
    __aicore__ inline void ProcessQuery(uint32_t workStart, uint32_t workCount)
    {
        const uint32_t validN = validN_;
        const bool reuseVInQuery = validN <= tilingData_->vCacheRows && validN <= (V_MTE2_EVENT_COUNT - qBufferNum_);
        for (uint32_t workOffset = 0; workOffset < workCount; ++workOffset) {
            const uint32_t workId = workStart + workOffset;
            const uint32_t tIndex = workId / tilingData_->totalS;
            const uint32_t sIndex = workId - tIndex * tilingData_->totalS;
            const uint32_t statBufferIndex = statBufferCounter_ % ROW_STAT_BUFFER_NUM;
            const event_t statEvent = static_cast<event_t>(oBufferNum_ + statBufferIndex);
            const uint64_t statByteOffset =
                statUbOffset_ + static_cast<uint64_t>(statBufferIndex) * STAT_UB_ELEMS * FP32_BYTES;
            __ubuf__ float* statAddr = GetUbAddr<float>(statByteOffset);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(statEvent);
            const uint32_t qBufferIndex = qBufferIndex_;
            const event_t qEvent = static_cast<event_t>(qBufferIndex);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qEvent);
            __ubuf__ float* qAddr = CopyInQuery(sIndex, qBufferIndex);

            for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
                __ubuf__ BLOCK_RES_DTYPE* vAddr;
                if (reuseVInQuery) {
                    uint32_t vBufferIndex = validN > 1 ? nIndex : (vBufferCounter_ & 1);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qBufferNum_ + static_cast<event_t>(vBufferIndex));
                    const uint32_t cacheOffset = vBufferIndex * dAlign_;
                    const uint64_t byteOffset = vCacheUbOffset_ + static_cast<uint64_t>(cacheOffset) * V_ELEM_BYTES;
                    vAddr = CopyInV(tIndex, nIndex, byteOffset);
                    SyncMte2ToVector(static_cast<event_t>(0));
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qBufferNum_ + (vBufferCounter_ % DOUBLE_BUFFER_NUM));
                    const uint32_t bufferIndex = static_cast<uint32_t>(vBufferCounter_ % DOUBLE_BUFFER_NUM);
                    const uint64_t byteOffset =
                        vCacheUbOffset_ + static_cast<uint64_t>(bufferIndex) * dAlign_ * V_ELEM_BYTES;
                    vAddr = CopyInV(tIndex, nIndex, byteOffset);
                    SyncMte2ToVector(static_cast<event_t>(bufferIndex));
                }
                asc_vf_call<BlockAttnResPrepareVF::AccumulateSquareSumAndDot<BLOCK_RES_DTYPE>>(
                    qAddr, vAddr, statAddr, tilingData_->totalD, nIndex);
                if (!reuseVInQuery) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qBufferNum_ + (vBufferCounter_ % DOUBLE_BUFFER_NUM));
                    vBufferCounter_++;
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qEvent);
            qBufferIndex_ = NextBuffer(qBufferIndex_, qBufferNum_);

            FinalizeStats(statAddr, validN);
            CopyOutStats(sIndex, tIndex, statByteOffset, statEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(statEvent);

            __ubuf__ float* weightAddr = statAddr + BlockAttnResPrepareVF::DOT_OR_WEIGHT_OFFSET;
            const uint32_t outputBufferIndex = oBufferIndex_;
            const uint64_t outputByteOffset =
                oUbOffset_ + static_cast<uint64_t>(outputBufferIndex) * dAlign_ * FP32_BYTES;
            __ubuf__ float* outputAddr = GetUbAddr<float>(outputByteOffset);
            const event_t outputEvent = static_cast<event_t>(outputBufferIndex);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
            for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
                __ubuf__ BLOCK_RES_DTYPE* vAddr;
                if (reuseVInQuery) {
                    uint32_t vBufferIndex = validN > 1 ? nIndex : (vBufferCounter_ & 1);
                    const uint32_t cacheOffset = vBufferIndex * dAlign_;
                    vAddr =
                        GetUbAddr<BLOCK_RES_DTYPE>(vCacheUbOffset_ + static_cast<uint64_t>(cacheOffset) * V_ELEM_BYTES);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qBufferNum_ + (vBufferCounter_ % DOUBLE_BUFFER_NUM));
                    const uint32_t bufferIndex = static_cast<uint32_t>(vBufferCounter_ % DOUBLE_BUFFER_NUM);
                    const uint64_t byteOffset =
                        vCacheUbOffset_ + static_cast<uint64_t>(bufferIndex) * dAlign_ * V_ELEM_BYTES;
                    vAddr = CopyInV(tIndex, nIndex, byteOffset);
                    SyncMte2ToVector(static_cast<event_t>(bufferIndex));
                }
                AccumulateOutput(vAddr, weightAddr + nIndex, outputAddr, nIndex, validN, tilingData_->totalD);
                if (reuseVInQuery) {
                    uint32_t vBufferIndex = validN > 1 ? nIndex : (vBufferCounter_ & 1);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qBufferNum_ + vBufferIndex);
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qBufferNum_ + (vBufferCounter_ % DOUBLE_BUFFER_NUM));
                }
                vBufferCounter_++;
            }

            CopyOutNumerator(sIndex, tIndex, oBufferIndex_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
            oBufferIndex_ = NextBuffer(oBufferIndex_, oBufferNum_);
            ++statBufferCounter_;
        }
    }

    // Keep one token's valid V rows and square sums in UB across [sStart, sEnd).
    // Both single-S tasks and packed-S tasks use this loop to compute queries
    // one at a time. The first query's stat row retains V square sums for the whole span.
    // Load each Q into its reusable slot inside the S loop.
    __aicore__ inline void ProcessSRangeWithCachedV(uint64_t tIndex, uint32_t sStart, uint32_t sEnd, uint32_t validN)
    {
        LoadTokenV(tIndex, validN);
        // m/l copies do not touch this row's square sums, even while the stat slots rotate.
        vSquareSumUbOffset_ = statUbOffset_ + static_cast<uint64_t>(statBufferCounter_ % ROW_STAT_BUFFER_NUM) *
                                                  STAT_UB_ELEMS * FP32_BYTES;
        for (uint32_t sIndex = sStart; sIndex < sEnd; ++sIndex) {
            const uint32_t qBufferIndex = qBufferIndex_;
            const event_t qEvent = static_cast<event_t>(qBufferIndex);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qEvent);
            __ubuf__ float* qAddr = CopyInQuery(sIndex, qBufferIndex);
            SyncMte2ToVector(qEvent);
            ProcessQueryWithCachedV(tIndex, sIndex, sIndex == sStart, validN, qAddr, qEvent);
            qBufferIndex_ = NextBuffer(qBufferIndex, qBufferNum_);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(TOKEN_V_EVENT_ID);
    }

    __aicore__ inline void ProcessQueryWithCachedV(uint64_t tIndex, uint64_t sIndex, bool isFirstQuery, uint32_t validN,
                                                   __ubuf__ float* qAddr, event_t qEvent)
    {
        const uint32_t statBufferIndex = statBufferCounter_ % ROW_STAT_BUFFER_NUM;
        const event_t statEvent = static_cast<event_t>(oBufferNum_ + statBufferIndex);
        const uint64_t statByteOffset =
            statUbOffset_ + static_cast<uint64_t>(statBufferIndex) * STAT_UB_ELEMS * FP32_BYTES;
        __ubuf__ float* statAddr = GetUbAddr<float>(statByteOffset);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(statEvent);
        for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
            __ubuf__ BLOCK_RES_DTYPE* vAddr =
                GetUbAddr<BLOCK_RES_DTYPE>(vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * V_ELEM_BYTES);
            if (isFirstQuery) {
                asc_vf_call<BlockAttnResPrepareVF::AccumulateSquareSumAndDot<BLOCK_RES_DTYPE>>(
                    qAddr, vAddr, statAddr, tilingData_->totalD, nIndex);
            } else {
                asc_vf_call<BlockAttnResPrepareVF::AccumulateDotWithCachedSquareSum<BLOCK_RES_DTYPE>>(
                    qAddr, vAddr, statAddr, GetUbAddr<float>(vSquareSumUbOffset_), tilingData_->totalD, nIndex);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qEvent);

        FinalizeStats(statAddr, validN);
        CopyOutStats(sIndex, tIndex, statByteOffset, statEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(statEvent);

        __ubuf__ float* weightAddr = statAddr + BlockAttnResPrepareVF::DOT_OR_WEIGHT_OFFSET;
        const uint32_t outputBufferIndex = oBufferIndex_;
        const uint64_t outputByteOffset = oUbOffset_ + static_cast<uint64_t>(outputBufferIndex) * dAlign_ * FP32_BYTES;
        __ubuf__ float* outputAddr = GetUbAddr<float>(outputByteOffset);
        const event_t outputEvent = static_cast<event_t>(outputBufferIndex);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
        // A single V produces the same numerator for every S in this token span.
        // Keep the single-S template unchanged; packed-S tasks retain the converted output.
        if (SINGLE_S_CACHE || validN != 1U || isFirstQuery) {
            for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
                __ubuf__ BLOCK_RES_DTYPE* vAddr = GetUbAddr<BLOCK_RES_DTYPE>(
                    vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * V_ELEM_BYTES);
                AccumulateOutput(vAddr, weightAddr + nIndex, outputAddr, nIndex, validN, tilingData_->totalD);
            }
        }

        CopyOutNumerator(sIndex, tIndex, oBufferIndex_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
        if (SINGLE_S_CACHE || validN != 1U) {
            oBufferIndex_ = NextBuffer(oBufferIndex_, oBufferNum_);
        }
        ++statBufferCounter_;
    }

    template <uint32_t S_ROWS, uint32_t TOKEN_COUNT>
    __aicore__ inline void CopyOutGroupedStats(uint64_t tIndex, uint32_t sGroupStart)
    {
        SyncVectorToMte3(static_cast<event_t>(oBufferNum_));
        auto logitMaxTensor = MakeGmTensor<float>(logitMaxAddr_, tilingData_->totalS, tilingData_->totalT);
        auto expSumTensor = MakeGmTensor<float>(expSumAddr_, tilingData_->totalS, tilingData_->totalT);
        uint32_t statRowIndex = 0U;
        for (uint32_t token = 0; token < TOKEN_COUNT; ++token) {
            for (uint32_t row = 0; row < S_ROWS; ++row, ++statRowIndex) {
                const uint64_t statOffset =
                    statUbOffset_ + static_cast<uint64_t>(statRowIndex) * STAT_UB_ELEMS * FP32_BYTES;
                CopyUbToGm<float>(
                    logitMaxTensor, sGroupStart + row, tIndex + token,
                    statOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::LOGIT_MAX_OFFSET) * FP32_BYTES, 1,
                    FP32_32B_ALIGN_ELEMS);
                CopyUbToGm<float>(
                    expSumTensor, sGroupStart + row, tIndex + token,
                    statOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::EXP_SUM_OFFSET) * FP32_BYTES, 1,
                    FP32_32B_ALIGN_ELEMS);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(oBufferNum_));
    }

    __aicore__ inline void ProcessTwoSRowOutputs(uint64_t tIndex, uint32_t sIndex, uint32_t statRowIndex,
                                                 uint32_t validN, uint32_t vRowOffset)
    {
        __ubuf__ float* weight0Addr =
            GetUbAddr<float>(statUbOffset_ + static_cast<uint64_t>(statRowIndex) * STAT_UB_ELEMS * FP32_BYTES) +
            BlockAttnResPrepareVF::DOT_OR_WEIGHT_OFFSET;
        __ubuf__ float* weight1Addr =
            GetUbAddr<float>(statUbOffset_ + static_cast<uint64_t>(statRowIndex + 1U) * STAT_UB_ELEMS * FP32_BYTES) +
            BlockAttnResPrepareVF::DOT_OR_WEIGHT_OFFSET;
        const uint32_t outputBufferIndex0 = oBufferIndex_;
        const uint32_t outputBufferIndex1 = NextBuffer(oBufferIndex_, oBufferNum_);
        const event_t outputEvent0 = static_cast<event_t>(outputBufferIndex0);
        const event_t outputEvent1 = static_cast<event_t>(outputBufferIndex1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent1);
        __ubuf__ float* output0Addr =
            GetUbAddr<float>(oUbOffset_ + static_cast<uint64_t>(outputBufferIndex0) * dAlign_ * FP32_BYTES);
        __ubuf__ float* output1Addr =
            GetUbAddr<float>(oUbOffset_ + static_cast<uint64_t>(outputBufferIndex1) * dAlign_ * FP32_BYTES);
        uint32_t nIndex = 0U;
        if (tilingData_->totalT >= FOUR_N_BLOCKS_MIN_T) {
            for (; nIndex + FOUR_N_BLOCKS - 1U < validN; nIndex += FOUR_N_BLOCKS) {
                __ubuf__ BLOCK_RES_DTYPE* vAddr = GetUbAddr<BLOCK_RES_DTYPE>(
                    vCacheUbOffset_ + static_cast<uint64_t>(vRowOffset + nIndex) * dAlign_ * V_ELEM_BYTES);
                if (nIndex == 0U) {
                    asc_vf_call<
                        BlockAttnResPrepareVF::WeightedAccumulateTwoSRows<BLOCK_RES_DTYPE, FOUR_N_BLOCKS, true>>(
                        vAddr, dAlign_, weight0Addr + nIndex, weight1Addr + nIndex, output0Addr, output1Addr,
                        tilingData_->totalD);
                } else {
                    asc_vf_call<
                        BlockAttnResPrepareVF::WeightedAccumulateTwoSRows<BLOCK_RES_DTYPE, FOUR_N_BLOCKS, false>>(
                        vAddr, dAlign_, weight0Addr + nIndex, weight1Addr + nIndex, output0Addr, output1Addr,
                        tilingData_->totalD);
                }
            }
        }
        for (; nIndex + TWO_N_BLOCKS - 1U < validN; nIndex += TWO_N_BLOCKS) {
            __ubuf__ BLOCK_RES_DTYPE* v0Addr = GetUbAddr<BLOCK_RES_DTYPE>(
                vCacheUbOffset_ + static_cast<uint64_t>(vRowOffset + nIndex) * dAlign_ * V_ELEM_BYTES);
            if (nIndex == 0U) {
                asc_vf_call<BlockAttnResPrepareVF::WeightedAccumulateTwoSRows<BLOCK_RES_DTYPE, TWO_N_BLOCKS, true>>(
                    v0Addr, dAlign_, weight0Addr + nIndex, weight1Addr + nIndex, output0Addr, output1Addr,
                    tilingData_->totalD);
            } else {
                asc_vf_call<BlockAttnResPrepareVF::WeightedAccumulateTwoSRows<BLOCK_RES_DTYPE, TWO_N_BLOCKS, false>>(
                    v0Addr, dAlign_, weight0Addr + nIndex, weight1Addr + nIndex, output0Addr, output1Addr,
                    tilingData_->totalD);
            }
        }
        if (nIndex < validN) {
            // Grouped callers require validN >= 2; the paired loop has already initialized both outputs.
            __ubuf__ BLOCK_RES_DTYPE* vAddr = GetUbAddr<BLOCK_RES_DTYPE>(
                vCacheUbOffset_ + static_cast<uint64_t>(vRowOffset + nIndex) * dAlign_ * V_ELEM_BYTES);
            asc_vf_call<BlockAttnResPrepareVF::WeightedAccumulateTwoSRows<BLOCK_RES_DTYPE, 1U, false>>(
                vAddr, dAlign_, weight0Addr + nIndex, weight1Addr + nIndex, output0Addr, output1Addr,
                tilingData_->totalD);
        }
        CopyOutNumerator(sIndex, tIndex, oBufferIndex_);
        CopyOutNumerator<false>(sIndex + 1U, tIndex, outputBufferIndex1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent1);
        oBufferIndex_ = NextBuffer(outputBufferIndex1, oBufferNum_);
    }

    template <uint32_t S_ROWS, uint32_t TOKEN_COUNT, bool FIRST_S_GROUP>
    __aicore__ inline void AccumulateSGroupStats(uint32_t validN)
    {
        __ubuf__ float* qAddr = GetUbAddr<float>(qUbOffset_);
        for (uint32_t token = 0; token < TOKEN_COUNT; ++token) {
            __ubuf__ float* statAddr =
                GetUbAddr<float>(statUbOffset_ + static_cast<uint64_t>(token) * S_ROWS * STAT_UB_ELEMS * FP32_BYTES);
            // Each token reuses its first stat row's square sums across S groups.
            __ubuf__ float* vSquareSumAddr = statAddr;
            uint32_t nIndex = 0U;
            for (; nIndex + TWO_N_BLOCKS - 1U < validN; nIndex += TWO_N_BLOCKS) {
                __ubuf__ BLOCK_RES_DTYPE* v0Addr = GetUbAddr<BLOCK_RES_DTYPE>(
                    vCacheUbOffset_ + static_cast<uint64_t>(token * validN + nIndex) * dAlign_ * V_ELEM_BYTES);
                __ubuf__ BLOCK_RES_DTYPE* v1Addr = v0Addr + dAlign_;
                asc_vf_call<BlockAttnResPrepareVF::AccumulateVPairStats<BLOCK_RES_DTYPE, S_ROWS, FIRST_S_GROUP>>(
                    qAddr, v0Addr, v1Addr, statAddr, vSquareSumAddr, tilingData_->totalD, nIndex, dAlign_,
                    STAT_UB_ELEMS);
            }
            if (nIndex < validN) {
                __ubuf__ BLOCK_RES_DTYPE* vAddr = GetUbAddr<BLOCK_RES_DTYPE>(
                    vCacheUbOffset_ + static_cast<uint64_t>(token * validN + nIndex) * dAlign_ * V_ELEM_BYTES);
                asc_vf_call<BlockAttnResPrepareVF::AccumulateSGroupStats<BLOCK_RES_DTYPE, S_ROWS, FIRST_S_GROUP>>(
                    qAddr, vAddr, statAddr, vSquareSumAddr, tilingData_->totalD, nIndex, dAlign_, STAT_UB_ELEMS);
            }
        }
    }

    __aicore__ inline void ProcessThreeSRowOutputs(uint64_t tIndex, uint32_t sGroupStart, uint32_t validN)
    {
        for (uint32_t slot = 0; slot < THREE_S_ROWS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(slot));
        }
        __ubuf__ float* weightAddr = GetUbAddr<float>(statUbOffset_) + BlockAttnResPrepareVF::DOT_OR_WEIGHT_OFFSET;
        __ubuf__ BLOCK_RES_DTYPE* vAddr = GetUbAddr<BLOCK_RES_DTYPE>(vCacheUbOffset_);
        __ubuf__ float* output0Addr = GetUbAddr<float>(oUbOffset_);
        __ubuf__ float* output1Addr = GetUbAddr<float>(oUbOffset_ + static_cast<uint64_t>(dAlign_) * FP32_BYTES);
        __ubuf__ float* output2Addr =
            GetUbAddr<float>(oUbOffset_ + THIRD_OUTPUT_BUFFER_INDEX * static_cast<uint64_t>(dAlign_) * FP32_BYTES);
        if (validN == THREE_N_BLOCKS) {
            asc_vf_call<BlockAttnResPrepareVF::WeightedAccumulateThreeSRows<BLOCK_RES_DTYPE, THREE_N_BLOCKS>>(
                vAddr, dAlign_, weightAddr, STAT_UB_ELEMS, output0Addr, output1Addr, output2Addr, tilingData_->totalD);
        } else if (validN == FOUR_N_BLOCKS) {
            asc_vf_call<BlockAttnResPrepareVF::WeightedAccumulateThreeSRows<BLOCK_RES_DTYPE, FOUR_N_BLOCKS>>(
                vAddr, dAlign_, weightAddr, STAT_UB_ELEMS, output0Addr, output1Addr, output2Addr, tilingData_->totalD);
        } else {
            asc_vf_call<BlockAttnResPrepareVF::WeightedAccumulateThreeSRows<BLOCK_RES_DTYPE, FIVE_N_BLOCKS>>(
                vAddr, dAlign_, weightAddr, STAT_UB_ELEMS, output0Addr, output1Addr, output2Addr, tilingData_->totalD);
        }
        SyncVectorToMte3(static_cast<event_t>(oBufferIndex_));
        for (uint32_t slot = 0; slot < THREE_S_ROWS; ++slot) {
            CopyOutNumerator<false>(sGroupStart + slot, tIndex, slot);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(slot));
        }
    }

    // Reuse all validN full-D V rows per token across [sStart, sEnd), processing
    // sRowsPerGroup_ S rows at a time. Requires validN > 1 and a span divisible by sRowsPerGroup_.
    // Dot/output VFs fuse two or three S rows to share V loads and BF16 casts.
    // Each token's V is loaded once for this S span; the first S group caches its square sums.
    // tokenCount == 2 is the validN == 2 paired-token mode: both T values share
    // the loaded Q group, with separate V, square sums, stats and outputs.
    // Each loop loads its current Q group after all tokens' previous dots release
    // these slots. Output slots become reusable only after their MTE3 completion events.
    template <uint32_t S_ROWS, uint32_t TOKEN_COUNT = 1U>
    __aicore__ inline void ProcessSGroupsWithCachedV(uint64_t tIndex, uint32_t sStart, uint32_t sEnd, uint32_t validN)
    {
        LoadTokenV(tIndex, validN, TOKEN_COUNT);
        for (uint32_t sGroupStart = sStart; sGroupStart < sEnd; sGroupStart += S_ROWS) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(Q_GROUP_EVENT_ID));
            for (uint32_t row = 0; row < S_ROWS; ++row) {
                CopyInQuery(sGroupStart + row, row);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<event_t>(oBufferNum_));
            SyncMte2ToVector(static_cast<event_t>(Q_GROUP_EVENT_ID));
            if (sGroupStart == sStart) {
                AccumulateSGroupStats<S_ROWS, TOKEN_COUNT, true>(validN);
            } else {
                AccumulateSGroupStats<S_ROWS, TOKEN_COUNT, false>(validN);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<event_t>(Q_GROUP_EVENT_ID));
            for (uint32_t row = 0; row < TOKEN_COUNT * S_ROWS; ++row) {
                FinalizeStats(GetUbAddr<float>(statUbOffset_ + static_cast<uint64_t>(row) * STAT_UB_ELEMS * FP32_BYTES),
                              validN);
            }
            CopyOutGroupedStats<S_ROWS, TOKEN_COUNT>(tIndex, sGroupStart);
            if constexpr (S_ROWS == THREE_S_ROWS) {
                ProcessThreeSRowOutputs(tIndex, sGroupStart, validN);
            } else {
                for (uint32_t token = 0; token < TOKEN_COUNT; ++token) {
                    ProcessTwoSRowOutputs(tIndex + token, sGroupStart, token * S_ROWS, validN, token * validN);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(TOKEN_V_EVENT_ID);
    }

    __tiling_data_ptr__ optiling::BlockAttnResPrepareTilingData* __restrict tilingData_ = nullptr;
    __gm__ BLOCK_RES_DTYPE* blockResAddr_ = nullptr;
    __gm__ float* pseudoQueryAddr_ = nullptr;
    __gm__ float* numeratorAddr_ = nullptr;
    __gm__ float* logitMaxAddr_ = nullptr;
    __gm__ float* expSumAddr_ = nullptr;

    uint64_t qUbOffset_ = 0U;
    uint64_t oUbOffset_ = 0U;
    uint64_t statUbOffset_ = 0U;
    uint64_t vSquareSumUbOffset_ = 0U;
    uint64_t vCacheUbOffset_ = 0U;
    uint32_t validN_ = 0U;

    uint32_t sBlockCount_ = 1U;
    uint32_t dAlign_ = 0U;
    uint32_t qBufferIndex_ = 0U;
    uint32_t vBufferCounter_ = 0;
    uint32_t oBufferIndex_ = 0U;
    uint32_t statBufferCounter_ = 0;
    uint32_t sRowsPerGroup_ = TWO_S_ROWS;

    uint8_t qBufferNum_ = 1;
    uint8_t oBufferNum_ = 2;
    bool useTokenPair_ = false;
    bool cacheVPerToken_ = false;
};

} // namespace BlockAttnResPrepare

#endif // BLOCK_ATTN_RES_PREPARE_VECTOR_H
